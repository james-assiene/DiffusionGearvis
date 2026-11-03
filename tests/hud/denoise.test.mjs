// T-0128c group B — the real denoising view in the Thinking pane: the pure cell model
// (hud/src/denoise/model.js) and the panel that draws it (hud/src/denoise/index.js).
//
// docs/tasks/T-0128c.md criteria B2, B4, B5, B6, B7, B8 and B9 (the node half); *Tests to write*
// "tests/hud/denoise.test.mjs". The browser half (the grid's geometry, the token colours in both
// themes, the theme-switch redraw, devicePixelRatio, reduced motion in real Chrome, aria-label on a
// real canvas, the parity rows) lives in tests/integration/test_hud_denoise.py and
// tests/integration/test_hud_denoise_parity.py.
//
// The failure mode this file is aimed at is the one CLAUDE.md names: "the denoising view shows the
// real per-step canvases from the plugin tap, never a fake trajectory presented as real". So:
//   * a page that interpolates between two frames (frames at steps 4 and 10 only must show exactly
//     the labels {4, 10}, and draws must equal frames — B5, adversarial);
//   * a page that draws something with no frame (zero frames -> zero draws and no <canvas> at all;
//     an astra.canvas.synthetic draws nothing and says it refused — B5, adversarial);
//   * a cell model that reads the stable mask LSB-first, lets "kept" outrank "stable", or marks a
//     cell revised that was never stable (B2, hand-computed arrays);
//   * a focus rule that keeps an aborted prefill on screen while the generate request denoises,
//     or lets an older request take the view back (B6 as ruled in O-C5: the request whose first
//     frame arrived most recently takes the view at once; the 1500 ms silence window is gone).
//
// API this pins (the task names the files and the hooks; these are the names):
//
//   hud/src/denoise/model.js — pure: no DOM, no clock, importable in node
//     COLUMNS                         32 (B3)
//     cellStates(frame, previous)     -> string[] of length frame.canvas_length, each one of
//                                        "noise" | "kept" | "stable" | "revised" | "committed" |
//                                        "empty" (B2). `frame` is a decoded astra.canvas.step;
//                                        `previous` is the frame received before it (or null). It
//                                        is only consulted when it is a denoise frame of the same
//                                        (request_id, canvas_idx); anything else is as if null.
//     label(frame)                    -> "canvas {canvas_idx} · step {step}/{max_steps}" (denoise)
//                                        or "canvas {canvas_idx} · committed at step
//                                        {step}/{max_steps}" (commit), verbatim (B4).
//     digest(states)                  -> FNV-1a (32-bit, 8 lowercase hex digits) of the state
//                                        array (B9).
//
//   hud/src/denoise/index.js — mount(host), host = { document, subscribe } as hud/src/main.js
//   passes it. It finds, by attribute, in host.document:
//     [data-denoise-root]   the mount point (T-0128 criterion 5); no element child until the first
//                           decoded astra.canvas.step, then exactly one <canvas> (B3, B5)
//     [data-denoise-label]  the honest label (B4), a sibling element in the Thinking heading row,
//                           declared empty in hud/index.html (pinned in tests/hud/shell.test.mjs)
//     [data-denoise-empty]  the empty line (B5 copy; denoise_no_tap and synthetic messages)
//   and writes on the root: data-denoise-frames (astra.canvas.step messages received),
//   data-denoise-draws (times the grid was painted), data-denoise-digest (digest() of the state
//   array last painted) (B9). The canvas carries aria-label = the label plus the four counts (B3).
//   Time and frames come only from host.document.defaultView: performance.now() (the page clock of
//   B7/B8), requestAnimationFrame / cancelAnimationFrame, setTimeout / clearTimeout, matchMedia (B8),
//   devicePixelRatio and getComputedStyle (the tokens). In a browser defaultView is window, so this
//   costs the implementation nothing; here it is a virtual clock. requestAnimationFrame,
//   cancelAnimationFrame, matchMedia, getComputedStyle, devicePixelRatio and MutationObserver are
//   also installed on globalThis as the same fakes, so either spelling works.
//
// Strict-xfail gate (as T-0154's node rows): node tests cannot carry pytest's `pending` marker.
// Every row below that needs the implementation runs through `pendingT0128cB`, which reads
// docs/tasks/T-0128c.md's **Status:** line: until it records group B's activation
// ("group B: tests <a>..<b>, activated <sha>", written by the test-author when /implement T-0128c B
// starts) each row must FAIL, and a row that passes throws XPASS(strict). Group A's or C's
// activation phrase does not switch these rows on. `ASTRA_RUN_PENDING=1 node --test
// tests/hud/denoise.test.mjs` runs them for real now (the node twin of `pytest --pending`).
//
// Every expected array, label and count below is computed by hand from the B2/B4 rules and written
// out literally; nothing is read back from the module under test except where a row checks that
// two parts of it agree (the panel's digest is model.digest of the hand-written array).
//
// ---- T-0159 group B (docs/tasks/T-0159.md B-1 .. B-10): the grid of token cells ----------------
//
// T-0159 replaces the 2D-canvas bitmap with a DOM grid of token cells, one <details> section per
// canvas (James's reference image: each cell shows the token at its position and changes at every
// step). The rows that pinned the bitmap (a <canvas> created at the first frame, its aria-label)
// are rewritten below to the DOM form and gated on T-0159 group B (`pendingT0159B`, the same
// Status-phrase gate as T-0128c's, reading docs/tasks/T-0159.md). The T-0128c rows that still hold
// (the cell model, labels, digest, focus, reduced motion) are unchanged.
//
// API these rows pin (T-0159 Implementation notes: "a DOM grid of cells, updating textContent and
// a state class only where they changed"):
//
//   [data-denoise-root]   no element child until the first drawn astra.canvas.step (T-0128c B5);
//                         then its element children are exactly the sections: one native
//                         <details> per canvas_idx of the shown request, in arrival order (B-7).
//   <details>             first element child a <summary> whose text contains the section's label
//                         in the B4 form (model.label of that canvas's last drawn frame), B-7.
//   [data-denoise-cell]   the cells of a section, descendants of its <details>, one per position,
//                         in position order (B-6). A cell's state is its `data-state` attribute or,
//                         failing that, its one class token equal to a state name ("stable", "kept",
//                         "noise", "revised", "committed", "empty"). Its text (textContent) is the
//                         display form of its piece; its `title` attribute is the full piece with
//                         whitespace made visible (" " -> "␣", "\n" -> "↵", "\t" -> "⇥"); no piece,
//                         no title. A special piece (the whole piece is of the `<…>` form) is marked
//                         by `data-special` or a class token "special" on the cell or a descendant.
//   [data-denoise-root]'s data-denoise-frames / -draws / -digest and [data-denoise-label] keep their
//   T-0128c meanings (the digest is of the newest drawn frame's states, B-5).
//
// The display forms (B-6) are written out by hand per piece in the rows below. The fake DOM logs
// every write to an element (attributes, class, text, title, open, style, children) so the B-8 row
// "only changed cells are updated" can see a cell that was rewritten with the value it already had.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);

// --- the group B strict-xfail gate -----------------------------------------------------------

const TASK = fileURLToPath(new URL("docs/tasks/T-0128c.md", REPO));
const STATUS = existsSync(TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(TASK, "utf8")) || [])[1] || "").trim()
  : "";
/** "group B: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6" — the phrase the B test-author writes. */
const B_ACTIVATED = /\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const B_PENDING = !B_ACTIVATED.test(STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0128cB(name, fn) {
  test(name + (B_PENDING ? " [pending T-0128c B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0128c group B row passed while docs/tasks/T-0128c.md's status does " +
          "not record group B's activation (" + JSON.stringify(STATUS.slice(0, 80)) + "). Either " +
          "group B has landed (record 'group B: tests …, activated <sha>' in the status) or the " +
          "row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0128c B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0128c B gate: the status line is readable and only group B's own phrase activates the rows", () => {
  assert.ok(STATUS.length > 0, "docs/tasks/T-0128c.md must carry a **Status:** line");
  // Hand-written samples: the other groups' activation wording must not switch these rows on.
  assert.equal(B_ACTIVATED.test("planned 2026-10-26 (James: build it before the v0.1 release)"), false);
  assert.equal(B_ACTIVATED.test("in progress (group A: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6; group B planned)"), false);
  assert.equal(B_ACTIVATED.test("in progress (group A: …, activated 9a8b7c6; group B: tests a1b2c3d..e4f5a6b, activated c3d4e5f)"), true);
  assert.equal(B_ACTIVATED.test("in progress (group B: tests a1b2c3d..e4f5a6b, not yet activated)"), false, "activation names its commit");
  assert.equal(B_ACTIVATED.test("in progress (group B tests written; group C: tests …, activated 1234abc)"), false, "group C's activation is not group B's");
  // The planned status this file was written against carries "B medium, one reviewer" in its
  // blast-radius clause on the same line; that must not read as an activation either.
  assert.equal(B_ACTIVATED.test("planned · **Blast radius:** medium overall (A medium; B medium, one reviewer; C low)"), false);
});

// --- the T-0159 group B strict-xfail gate --------------------------------------------------------
//
// The same mechanism for T-0159 group B: docs/tasks/T-0159.md's **Status:** must record
// "group B: tests <a>..<b>, activated <sha>" (written by the test-author when /implement T-0159 B
// starts). Group A's activation does not switch these rows on. ASTRA_RUN_PENDING=1 runs them now.

const TASK_T0159 = fileURLToPath(new URL("docs/tasks/T-0159.md", REPO));
const STATUS_T0159 = existsSync(TASK_T0159)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(TASK_T0159, "utf8")) || [])[1] || "").trim()
  : "";
const T0159_B_PENDING = !B_ACTIVATED.test(STATUS_T0159) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0159B(name, fn) {
  test(name + (T0159_B_PENDING ? " [pending T-0159 B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!T0159_B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0159 group B row passed while docs/tasks/T-0159.md's status does " +
          "not record group B's activation (" + JSON.stringify(STATUS_T0159.slice(0, 80)) + "). Either " +
          "group B has landed (record 'group B: tests …, activated <sha>' in the status) or the " +
          "row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0159 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0159 B gate: docs/tasks/T-0159.md has a status line, and group A's activation does not switch B on", () => {
  assert.ok(STATUS_T0159.length > 0, "docs/tasks/T-0159.md must carry a **Status:** line");
  assert.equal(
    B_ACTIVATED.test("in progress 2026-10-26 (groups A and B start in parallel; B merges after A)"),
    false,
    "the plan's own status does not activate group B",
  );
  assert.equal(B_ACTIVATED.test("in progress (group A: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6)"), false);
  assert.equal(B_ACTIVATED.test("in progress (group A: …, activated 9a8b7c6; group B: tests a1b2c3d..e4f5a6b, activated c3d4e5f)"), true);
});

// --- the T-0159 group B2 strict-xfail gate -------------------------------------------------------
//
// Group B2 (the trailing end-of-sequence canvas, docs/tasks/T-0159.md B2-1 .. B2-6) lands after B
// is activated, so its rows need their own phrase: the Status line must record
// "group B2: tests <a>..<b>, activated <sha>" (written by the test-author when /implement T-0159 B2
// starts). Group B's phrase ("group B: …") does not switch these rows on, and B2's does not switch
// B's on. ASTRA_RUN_PENDING=1 runs them now.

/** "group B2: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6" — the phrase the B2 test-author writes. */
const B2_ACTIVATED = /\bgroup B2:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const T0159_B2_PENDING = !B2_ACTIVATED.test(STATUS_T0159) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0159B2(name, fn) {
  test(name + (T0159_B2_PENDING ? " [pending T-0159 B2: strict xfail until group B2 is activated]" : ""), async (t) => {
    if (!T0159_B2_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0159 group B2 row passed while docs/tasks/T-0159.md's status does " +
          "not record group B2's activation (" + JSON.stringify(STATUS_T0159.slice(0, 80)) + "). Either " +
          "group B2 has landed (record 'group B2: tests …, activated <sha>' in the status) or the " +
          "row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0159 B2 pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0159 B2 gate: only group B2's own phrase activates the B2 rows, and it does not activate B's", () => {
  // Hand-written samples, including the status as it reads while B2 is in progress.
  assert.equal(B2_ACTIVATED.test("group B: tests 13122b5, activated 68cf1377, merged 72942a89; open: group B2 in progress"), false);
  assert.equal(B2_ACTIVATED.test("in progress (group B2: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6)"), true);
  assert.equal(B_ACTIVATED.test("in progress (group B2: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6)"), false);
});

// --- module loading (the ESM shim of tests/hud/panels.test.mjs) ------------------------------

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

// Origin proof (.claude/rules/tests.md): where the modules under test were imported from.
console.log(`# denoise origin: repo=${fileURLToPath(REPO)} modules=${srcRoot().href}`);

async function load(name) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch (err) {
    assert.fail(
      `T-0128c B adds hud/src/${name}, importable in node as an ES module with no DOM at module ` +
        `top level. node could not import it: ${err.message}`,
    );
  }
}

async function model() {
  const mod = await load("denoise/model.js");
  for (const name of ["cellStates", "label", "digest"])
    assert.equal(typeof mod[name], "function", `T-0128c B2/B4/B9: hud/src/denoise/model.js exports ${name}()`);
  return mod;
}

async function panel() {
  const mod = await load("denoise/index.js");
  assert.equal(typeof mod.mount, "function", "T-0128c B: hud/src/denoise/index.js exports mount(host)");
  return mod;
}

// --- hand-built frames (canvas length 8 unless stated) ----------------------------------------
//
// The stable mask is hex, one bit per position, MSB first, whole bytes (contracts/protocol.py
// AstraCanvasStep; plugin tap.py). For CL 8 it is one byte: "d0" = 1101 0000 = positions 0, 1, 3.

const step = (over = {}) => ({
  type: "astra.canvas.step",
  request_id: "chatcmpl-aaaa-11111111",
  canvas_idx: 0,
  step: 4,
  max_steps: 48,
  canvas_length: 8,
  phase: "denoise",
  ...over,
});

const commit = (over = {}) => ({
  type: "astra.canvas.step",
  request_id: "chatcmpl-aaaa-11111111",
  canvas_idx: 0,
  step: 20,
  max_steps: 48,
  canvas_length: 8,
  phase: "commit",
  committed_tokens: [1, 2, 3, 4, 5],
  ...over,
});

// F1: p0 stable (canvas == argmax too); p1 stable although canvas != argmax (the real trajectory
// has many such positions: at step 4 of canvas 0, 150 stable bits but only 54 kept positions);
// p2 noise; p3 stable; p4 noise; p5 kept; p6 noise; p7 kept.
const F1 = step({
  step: 4,
  canvas: [5, 60, 7, 8, 9, 10, 11, 12],
  argmax: [5, 6, 70, 8, 90, 10, 110, 12],
  stable: "d0",
});
const EXPECT_F1 = ["stable", "stable", "noise", "stable", "noise", "kept", "noise", "kept"];

// F2, the next frame of the same (request, canvas): stable "60" = 0110 0000 = p1, p2.
//   p0: no stable bit, canvas 5 != argmax 50 -> noise; but p0 was stable in F1 and its argmax moved
//       5 -> 50 -> shown revised.
//   p1: stable.  p2: stable.
//   p3: was stable in F1, no bit now, argmax unchanged (8) and canvas == argmax -> kept, NOT revised.
//   p4: noise.  p5: kept.  p6: noise.
//   p7: argmax moved 12 -> 13 but p7 was only kept in F1, never stable -> noise, NOT revised.
const F2 = step({
  step: 5,
  canvas: [5, 6, 71, 8, 9, 10, 11, 12],
  argmax: [50, 6, 71, 8, 90, 10, 110, 13],
  stable: "60",
});
const EXPECT_F2 = ["revised", "stable", "stable", "kept", "noise", "kept", "noise", "noise"];
// The same F2 read with no (or an unrelated) previous frame: p0 is plain noise.
const EXPECT_F2_ALONE = ["noise", "stable", "stable", "kept", "noise", "kept", "noise", "noise"];

const EXPECT_COMMIT_5_OF_8 = ["committed", "committed", "committed", "committed", "committed", "empty", "empty", "empty"];

/** A CL 8 denoise frame with no stable bit; positions below `kept` have canvas == argmax. Its
 *  states are, by B2 and by construction, `kept` x kept then `noise` — no revised is possible. */
function plain(stepNo, over = {}) {
  const kept = Math.min(8, stepNo % 9);
  return step({
    step: stepNo,
    canvas: [0, 1, 2, 3, 4, 5, 6, 7].map((p) => 100 + p),
    argmax: [0, 1, 2, 3, 4, 5, 6, 7].map((p) => (p < kept ? 100 + p : 200 + p)),
    stable: "00",
    ...over,
  });
}

// --- B2: cell states from the frame only (model.js) ------------------------------------------

pendingT0128cB("T-0128c B2: a denoise frame — stable outranks kept, kept is canvas == argmax, the rest is noise", async () => {
  const m = await model();
  assert.deepEqual(m.cellStates(F1, null), EXPECT_F1);
});

pendingT0128cB("T-0128c B2 adversarial: the stable mask is MSB first — 'd0' is positions 0, 1 and 3, not 4, 6 and 7", async () => {
  const m = await model();
  // Read LSB first, "d0" would set p4, p6, p7 and leave p0/p1/p3 to the canvas == argmax rule:
  // [kept, noise, noise, kept, stable, kept, stable, stable]. The row tells the two apart.
  const got = m.cellStates(F1, null);
  assert.deepEqual(
    [got[0], got[1], got[3], got[4], got[6], got[7]],
    ["stable", "stable", "stable", "noise", "noise", "kept"],
    "B2: MSB first, one bit per position (AstraCanvasStep.stable)",
  );
});

pendingT0128cB("T-0128c B2 adversarial: across whole bytes — CL 40, mask 0000000001 is position 39 alone", async () => {
  const m = await model();
  // 40 bits = 5 bytes = 10 hex digits. Byte 4 is 0x01: MSB first that is bit 7 of byte 4 =
  // position 4*8 + 7 = 39. Read LSB first within the byte it would be position 32.
  const ids = Array.from({ length: 40 }, (_, p) => p);
  const f = step({
    canvas_length: 40,
    canvas: ids,
    argmax: ids.map((p) => p + 1000), // canvas != argmax everywhere: noise but for the stable bit
    stable: "0000000001",
  });
  const want = Array.from({ length: 40 }, (_, p) => (p === 39 ? "stable" : "noise"));
  assert.deepEqual(m.cellStates(f, null), want);
});

pendingT0128cB("T-0128c B2: padding bits past canvas_length are ignored — one state per position, no more", async () => {
  const m = await model();
  // CL 4 in one whole byte: "ff" sets 8 bits, only the first 4 name positions.
  const f = step({ canvas_length: 4, canvas: [1, 2, 3, 4], argmax: [9, 9, 9, 9], stable: "ff" });
  assert.deepEqual(m.cellStates(f, null), ["stable", "stable", "stable", "stable"]);
  const g = step({ canvas_length: 4, canvas: [1, 2, 3, 4], argmax: [1, 9, 3, 9], stable: "0f" });
  // "0f" = 0000 1111: none of p0..p3 is stable; the padding bits are set but name nothing.
  assert.deepEqual(m.cellStates(g, null), ["kept", "noise", "kept", "noise"]);
});

pendingT0128cB("T-0128c B2: revised — argmax moved where the previous frame of the same request and canvas had p stable", async () => {
  const m = await model();
  assert.deepEqual(m.cellStates(F2, F1), EXPECT_F2);
});

pendingT0128cB("T-0128c B2 (guard both ways): revised needs the same request and canvas, and is never invented without a previous frame", async () => {
  const m = await model();
  assert.deepEqual(m.cellStates(F2, null), EXPECT_F2_ALONE, "no previous frame: nothing to have been revised");
  assert.deepEqual(
    m.cellStates({ ...F2, canvas_idx: 1 }, F1),
    EXPECT_F2_ALONE,
    "F1 is canvas 0; a canvas-1 frame was never stable at those positions",
  );
  assert.deepEqual(
    m.cellStates({ ...F2, request_id: "chatcmpl-bbbb-22222222" }, F1),
    EXPECT_F2_ALONE,
    "F1 belongs to another request (the idle captioner, say)",
  );
  // ... and the guard is not so wide that it never fires (the same pair, same key, revises p0):
  assert.equal(m.cellStates(F2, F1)[0], "revised");
});

pendingT0128cB("T-0128c B2: a commit frame — positions below len(committed_tokens) are committed, the rest empty", async () => {
  const m = await model();
  assert.deepEqual(m.cellStates(commit(), null), EXPECT_COMMIT_5_OF_8);
  assert.deepEqual(m.cellStates(commit(), F1), EXPECT_COMMIT_5_OF_8, "a commit frame reads only its own ids");
  assert.deepEqual(
    m.cellStates(commit({ committed_tokens: [7, 7, 7, 7, 7, 7, 7, 7] }), null),
    Array(8).fill("committed"),
  );
  assert.deepEqual(
    m.cellStates(commit({ committed_tokens: [7], step: 0 }), null),
    ["committed", "empty", "empty", "empty", "empty", "empty", "empty", "empty"],
    "a step-0 commit (T-0069 ruling O6) is still a commit",
  );
});

pendingT0128cB("T-0128c B2: the model is pure — it does not mutate the frame it is given", async () => {
  const m = await model();
  const a = structuredClone(F2);
  const b = structuredClone(F1);
  m.cellStates(a, b);
  assert.deepEqual(a, F2);
  assert.deepEqual(b, F1);
  assert.equal(m.COLUMNS, 32, "B3: 32 columns, exported for the panel");
});

// --- B4: the honest label ---------------------------------------------------------------------

pendingT0128cB("T-0128c B4: the label strings, verbatim from the frame (0-based canvas_idx, H-T128c-5)", async () => {
  const m = await model();
  assert.equal(m.label(F1), "canvas 0 · step 4/48");
  assert.equal(m.label(step({ canvas_idx: 3, step: 48, max_steps: 48 })), "canvas 3 · step 48/48");
  assert.equal(m.label(step({ canvas_idx: 2, step: 0, max_steps: 12 })), "canvas 2 · step 0/12");
  assert.equal(m.label(commit()), "canvas 0 · committed at step 20/48");
  assert.equal(m.label(commit({ canvas_idx: 4, step: 0 })), "canvas 4 · committed at step 0/48");
});

// --- B9: the digest -----------------------------------------------------------------------------

pendingT0128cB("T-0128c B9: digest is a deterministic 32-bit FNV-1a of the state array", async () => {
  const m = await model();
  // FNV-1a 32-bit starts from the offset basis 0x811c9dc5; an empty array hashes no byte, so it
  // is the basis itself whatever the per-state encoding is.
  assert.equal(m.digest([]), "811c9dc5", "B9: FNV-1a (32-bit offset basis for the empty array)");
  for (const states of [EXPECT_F1, EXPECT_F2, EXPECT_COMMIT_5_OF_8]) {
    assert.match(m.digest(states), /^[0-9a-f]{8}$/);
    assert.equal(m.digest([...states]), m.digest(states), "same states, same digest");
  }
  // Distinct arrays, distinct digests: one cell changed, two cells swapped, one cell longer.
  const variants = [
    EXPECT_F1,
    ["revised", ...EXPECT_F1.slice(1)],
    [EXPECT_F1[1], EXPECT_F1[0], EXPECT_F1[3], EXPECT_F1[2], ...EXPECT_F1.slice(4)].map((s, i) => (i === 2 ? "kept" : s)),
    [...EXPECT_F1, "noise"],
    EXPECT_F2,
    EXPECT_F2_ALONE,
    EXPECT_COMMIT_5_OF_8,
    Array(8).fill("empty"),
    Array(8).fill("noise"),
  ];
  const seen = new Set(variants.map((v) => m.digest(v)));
  assert.equal(seen.size, variants.length, "B9: the digest tells these hand-picked state arrays apart");
  assert.notEqual(m.digest(["kept", "noise"]), m.digest(["noise", "kept"]), "the digest depends on order");
});

// --- a DOM double for the panel ----------------------------------------------------------------
//
// Closed by delegation where it matters (.claude/rules/tests.md): the 2D context records every
// call and every property write through a Proxy, so the implementation may use any drawing
// primitive. Markup sinks throw (the panel draws; it never parses HTML).

const kebab = (key) => "data-" + key.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase());
const camel = (attr) => attr.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());

function fakeContext(canvas) {
  const calls = [];
  const state = {};
  return {
    calls,
    ctx: new Proxy(state, {
      get(target, prop) {
        if (prop === "canvas") return canvas;
        if (prop in target) return target[prop];
        if (typeof prop === "symbol") return undefined;
        return (...args) => {
          calls.push([String(prop), args]);
          if (prop === "measureText") return { width: 0 };
          if (prop === "createLinearGradient" || prop === "createRadialGradient") return { addColorStop() {} };
          if (prop === "getImageData") return { data: new Uint8ClampedArray(4) };
          return undefined;
        };
      },
      set(target, prop, value) {
        target[prop] = value;
        calls.push(["=" + String(prop), [value]]);
        return true;
      },
    }),
  };
}

/** T-0159 B-8: every write to an element is logged on the document as [element, kind]. */
function logWrite(doc, el, kind) {
  if (doc && doc.writes && el) doc.writes.push([el, kind]);
}

/** A text node whose data writes are logged against its parent element. */
function makeText(doc, data) {
  let value = String(data);
  const node = {
    nodeType: 3,
    parentNode: null,
    ownerDocument: doc,
    get data() {
      return value;
    },
    set data(v) {
      value = String(v);
      logWrite(doc, node.parentNode, "text");
    },
    get nodeValue() {
      return value;
    },
    set nodeValue(v) {
      node.data = v;
    },
    get textContent() {
      return value;
    },
    set textContent(v) {
      node.data = v;
    },
    remove() {
      if (node.parentNode) node.parentNode.removeChild(node);
    },
  };
  return node;
}

function makeElement(doc, tag) {
  const attrs = new Map();
  const styleProps = {};
  const styleTarget = {
    setProperty(k, v) {
      styleProps[k] = String(v);
      logWrite(doc, el, "style");
    },
    getPropertyValue(k) {
      return styleProps[k] || "";
    },
    removeProperty(k) {
      delete styleProps[k];
      logWrite(doc, el, "style");
    },
  };
  const el = {
    nodeType: 1,
    tagName: tag.toUpperCase(),
    localName: tag,
    ownerDocument: doc,
    parentNode: null,
    childNodes: [],
    listeners: {},
    style: new Proxy(styleTarget, {
      set(target, prop, value) {
        target[prop] = value;
        logWrite(doc, el, "style");
        return true;
      },
    }),
    get firstChild() {
      return this.childNodes[0] || null;
    },
    get lastChild() {
      return this.childNodes[this.childNodes.length - 1] || null;
    },
    get lastElementChild() {
      const c = this.children;
      return c[c.length - 1] || null;
    },
    get nextSibling() {
      const s = this.parentNode ? this.parentNode.childNodes : [];
      return s[s.indexOf(this) + 1] || null;
    },
    get nextElementSibling() {
      const s = this.parentNode ? this.parentNode.children : [];
      return s[s.indexOf(this) + 1] || null;
    },
    get previousSibling() {
      const s = this.parentNode ? this.parentNode.childNodes : [];
      const i = s.indexOf(this);
      return i > 0 ? s[i - 1] : null;
    },
    get isConnected() {
      for (let n = this; n; n = n.parentNode) if (n.localName === "html") return true;
      return false;
    },
    get title() {
      return attrs.get("title") || "";
    },
    set title(v) {
      attrs.set("title", String(v));
      logWrite(doc, el, "title");
    },
    get open() {
      return attrs.has("open");
    },
    set open(v) {
      if (v) attrs.set("open", "");
      else attrs.delete("open");
      logWrite(doc, el, "open");
    },
    tabIndex: tag === "summary" || tag === "button" || tag === "a" ? 0 : -1,
    focus() {
      doc.activeElement = el;
    },
    blur() {
      if (doc.activeElement === el) doc.activeElement = null;
    },
    dispatchEvent(ev) {
      for (const fn of this.listeners[ev && ev.type] || []) fn.call(this, ev);
      return true;
    },
    get children() {
      return this.childNodes.filter((n) => n.nodeType === 1);
    },
    get childElementCount() {
      return this.children.length;
    },
    get firstElementChild() {
      return this.children[0] || null;
    },
    get parentElement() {
      return this.parentNode && this.parentNode.nodeType === 1 ? this.parentNode : null;
    },
    get id() {
      return attrs.get("id") || "";
    },
    set id(v) {
      attrs.set("id", String(v));
    },
    get className() {
      return attrs.get("class") || "";
    },
    set className(v) {
      attrs.set("class", String(v));
      logWrite(doc, el, "class");
    },
    get hidden() {
      return attrs.has("hidden");
    },
    set hidden(v) {
      if (v) attrs.set("hidden", "");
      else attrs.delete("hidden");
    },
    get textContent() {
      return this.childNodes.map((n) => (n.nodeType === 3 ? n.data : n.textContent)).join("");
    },
    set textContent(v) {
      for (const c of this.childNodes) c.parentNode = null;
      if (v === "" || v == null) this.childNodes = [];
      else {
        const t = makeText(doc, v);
        t.parentNode = this;
        this.childNodes = [t];
      }
      logWrite(doc, el, "text");
    },
    get innerText() {
      return this.textContent;
    },
    set innerText(v) {
      this.textContent = v;
    },
    set innerHTML(_v) {
      throw new Error("the denoising view writes no markup (innerHTML)");
    },
    get innerHTML() {
      return "";
    },
    insertAdjacentHTML() {
      throw new Error("the denoising view writes no markup (insertAdjacentHTML)");
    },
    clientWidth: 342,
    clientHeight: 0,
    offsetWidth: 342,
    getBoundingClientRect() {
      return { x: 0, y: 0, left: 0, top: 0, width: 342, height: 0, right: 342, bottom: 0 };
    },
    getAttribute(k) {
      return attrs.has(k) ? attrs.get(k) : null;
    },
    setAttribute(k, v) {
      attrs.set(k, String(v));
      logWrite(doc, el, "attr:" + k);
    },
    hasAttribute(k) {
      return attrs.has(k);
    },
    removeAttribute(k) {
      attrs.delete(k);
      logWrite(doc, el, "attr:" + k);
    },
    toggleAttribute(k, force) {
      const on = force === undefined ? !attrs.has(k) : !!force;
      if (on) attrs.set(k, "");
      else attrs.delete(k);
      logWrite(doc, el, "attr:" + k);
      return on;
    },
    get attributes() {
      return [...attrs].map(([name, value]) => ({ name, value }));
    },
    appendChild(child) {
      if (child.nodeType === 11) {
        for (const c of [...child.childNodes]) this.appendChild(c);
        return child;
      }
      if (child.parentNode) child.parentNode.removeChild(child);
      child.parentNode = this;
      this.childNodes.push(child);
      logWrite(doc, el, "children");
      return child;
    },
    append(...nodes) {
      for (const n of nodes) this.appendChild(typeof n === "string" ? doc.createTextNode(n) : n);
    },
    prepend(...nodes) {
      for (const n of nodes.reverse()) this.insertBefore(typeof n === "string" ? doc.createTextNode(n) : n, this.childNodes[0] || null);
    },
    insertBefore(child, ref) {
      if (child.nodeType === 11) {
        for (const c of [...child.childNodes]) this.insertBefore(c, ref);
        return child;
      }
      logWrite(doc, el, "children");
      if (child.parentNode) child.parentNode.removeChild(child);
      child.parentNode = this;
      const i = ref ? this.childNodes.indexOf(ref) : -1;
      if (i < 0) this.childNodes.push(child);
      else this.childNodes.splice(i, 0, child);
      return child;
    },
    removeChild(child) {
      const i = this.childNodes.indexOf(child);
      if (i >= 0) this.childNodes.splice(i, 1);
      child.parentNode = null;
      logWrite(doc, el, "children");
      return child;
    },
    replaceChildren(...nodes) {
      for (const c of this.childNodes) c.parentNode = null;
      this.childNodes = [];
      this.append(...nodes);
    },
    replaceWith(...nodes) {
      const p = this.parentNode;
      if (!p) return;
      for (const n of nodes) p.insertBefore(n, this);
      p.removeChild(this);
    },
    remove() {
      if (this.parentNode) this.parentNode.removeChild(this);
    },
    contains(other) {
      for (let n = other; n; n = n.parentNode) if (n === this) return true;
      return false;
    },
    closest(sel) {
      for (let n = this; n && n.nodeType === 1; n = n.parentNode) if (matches(n, sel)) return n;
      return null;
    },
    matches(sel) {
      return matches(this, sel);
    },
    querySelector(sel) {
      return queryAll(this, sel)[0] || null;
    },
    querySelectorAll(sel) {
      return queryAll(this, sel);
    },
    addEventListener(type, fn) {
      (this.listeners[type] ||= []).push(fn);
    },
    removeEventListener(type, fn) {
      this.listeners[type] = (this.listeners[type] || []).filter((f) => f !== fn);
    },
  };
  el.classList = {
    add: (...c) => el.setAttribute("class", [...new Set([...el.className.split(/\s+/).filter(Boolean), ...c])].join(" ")),
    remove: (...c) => el.setAttribute("class", el.className.split(/\s+/).filter((x) => x && !c.includes(x)).join(" ")),
    contains: (c) => el.className.split(/\s+/).includes(c),
    toggle: (c, force) => {
      const on = force === undefined ? !el.classList.contains(c) : !!force;
      if (on) el.classList.add(c);
      else el.classList.remove(c);
      return on;
    },
  };
  el.dataset = new Proxy(
    {},
    {
      get: (_t, k) => (typeof k === "string" && attrs.has(kebab(k)) ? attrs.get(kebab(k)) : undefined),
      set: (_t, k, v) => (attrs.set(kebab(String(k)), String(v)), logWrite(doc, el, "data"), true),
      deleteProperty: (_t, k) => (attrs.delete(kebab(String(k))), logWrite(doc, el, "data"), true),
      has: (_t, k) => attrs.has(kebab(String(k))),
      ownKeys: () => [...attrs.keys()].filter((a) => a.startsWith("data-")).map(camel),
      getOwnPropertyDescriptor: (_t, k) =>
        attrs.has(kebab(String(k))) ? { enumerable: true, configurable: true, value: attrs.get(kebab(String(k))) } : undefined,
    },
  );
  if (tag === "canvas") {
    let ctx2d = null;
    el.width = 300;
    el.height = 150;
    el.getContext = (kind) => {
      if (kind !== "2d") return null;
      if (!ctx2d) {
        ctx2d = fakeContext(el);
        doc.contexts.push(ctx2d);
      }
      return ctx2d.ctx;
    };
  }
  return el;
}

// A small selector engine: comma lists, descendant (" ") and child (">") combinators, and compound
// simple selectors of tag, #id, .class, [attr] and [attr="value"].
function parseCompound(s) {
  const out = { tag: null, id: null, classes: [], attrs: [] };
  const re = /([a-zA-Z][\w-]*)|#([\w-]+)|\.([\w-]+)|\[([\w-]+)(?:([~|^$*]?=)["']?([^"'\]]*)["']?)?\]|(\*)/g;
  let m;
  while ((m = re.exec(s))) {
    if (m[1]) out.tag = m[1].toLowerCase();
    else if (m[2]) out.id = m[2];
    else if (m[3]) out.classes.push(m[3]);
    else if (m[4]) out.attrs.push([m[4], m[5] ? m[6] : null]);
  }
  return out;
}
function matchCompound(el, c) {
  if (!el || el.nodeType !== 1) return false;
  if (c.tag && el.localName !== c.tag) return false;
  if (c.id && el.id !== c.id) return false;
  const cls = el.className.split(/\s+/);
  if (c.classes.some((x) => !cls.includes(x))) return false;
  return c.attrs.every(([k, v]) => el.hasAttribute(k) && (v === null || el.getAttribute(k) === v));
}
function matches(el, sel) {
  return sel.split(",").some((one) => {
    const parts = one.trim().replace(/\s*>\s*/g, " > ").split(/\s+/);
    let i = parts.length - 1;
    if (!matchCompound(el, parseCompound(parts[i]))) return false;
    let node = el;
    i -= 1;
    while (i >= 0) {
      if (parts[i] === ">") {
        node = node.parentNode;
        if (!matchCompound(node, parseCompound(parts[i - 1]))) return false;
        i -= 2;
      } else {
        const c = parseCompound(parts[i]);
        let p = node.parentNode;
        while (p && p.nodeType === 1 && !matchCompound(p, c)) p = p.parentNode;
        if (!p || p.nodeType !== 1) return false;
        node = p;
        i -= 1;
      }
    }
    return true;
  });
}
function queryAll(root, sel) {
  const out = [];
  const walk = (n) => {
    for (const c of n.childNodes || []) {
      if (c.nodeType !== 1) continue;
      if (matches(c, sel)) out.push(c);
      walk(c);
    }
  };
  walk(root);
  return out;
}

/** The page's clock and frame loop: a virtual clock the test advances by hand. */
function fakeView({ reduce = false, dpr = 2, tokens = {} } = {}) {
  const view = {
    now: 1000,
    rafQueue: [],
    rafCalls: 0,
    timers: [],
    timerId: 0,
    reduce,
    devicePixelRatio: dpr,
  };
  view.performance = { now: () => view.now };
  view.requestAnimationFrame = (fn) => {
    view.rafCalls += 1;
    view.timerId += 1;
    view.rafQueue.push({ id: view.timerId, fn });
    return view.timerId;
  };
  view.cancelAnimationFrame = (id) => {
    view.rafQueue = view.rafQueue.filter((r) => r.id !== id);
  };
  view.setTimeout = (fn, ms = 0, ...args) => {
    view.timerId += 1;
    view.timers.push({ id: view.timerId, at: view.now + Math.max(0, Number(ms) || 0), fn: () => fn(...args) });
    return view.timerId;
  };
  view.clearTimeout = (id) => {
    view.timers = view.timers.filter((t) => t.id !== id);
  };
  view.setInterval = () => {
    throw new Error("the denoising view runs no interval: no frame, no animation (B5)");
  };
  // O-B3 F2: every listener and observer the panel registers is recorded (never fired unless a row
  // fires it), so a row can play a theme switch or a resize through whichever seam the panel uses.
  view.mediaListeners = [];
  view.listeners = [];
  view.mutationObservers = [];
  view.resizeObservers = [];
  view.addEventListener = (type, fn) => view.listeners.push({ type, fn });
  view.removeEventListener = (type, fn) => {
    view.listeners = view.listeners.filter((l) => !(l.type === type && l.fn === fn));
  };
  view.matchMedia = (query) => {
    const mql = {
      media: query,
      matches: /prefers-reduced-motion\s*:\s*reduce/.test(query) ? view.reduce : false,
      addEventListener(type, fn) {
        if (type === "change") view.mediaListeners.push({ query, fn, mql });
      },
      removeEventListener(_type, fn) {
        view.mediaListeners = view.mediaListeners.filter((l) => l.fn !== fn);
      },
      addListener(fn) {
        view.mediaListeners.push({ query, fn, mql });
      },
      removeListener(fn) {
        view.mediaListeners = view.mediaListeners.filter((l) => l.fn !== fn);
      },
    };
    return mql;
  };
  const TOKENS = { "--accent": "#8ea2ff", "--accent-2": "#c9a7ff", "--surface": "#1b1e26", "--muted": "#9aa1b2", ...tokens };
  view.getComputedStyle = () => ({ getPropertyValue: (name) => TOKENS[name] || "" });
  view.MutationObserver = class {
    constructor(fn) {
      this.fn = fn;
      this.targets = [];
      view.mutationObservers.push(this);
    }
    observe(target, options) {
      this.targets.push({ target, options });
    }
    disconnect() {
      this.targets = [];
    }
    takeRecords() {
      return [];
    }
  };
  view.ResizeObserver = class {
    constructor(fn) {
      this.fn = fn;
      this.targets = [];
      view.resizeObservers.push(this);
    }
    observe(target) {
      this.targets.push(target);
    }
    unobserve(target) {
      this.targets = this.targets.filter((t) => t !== target);
    }
    disconnect() {
      this.targets = [];
    }
  };
  /** Run the page for `ms` of virtual time, one 16 ms vsync at a time: due timers fire, then the
   *  animation-frame callbacks queued before this vsync run (callbacks they queue wait for the next). */
  view.advance = (ms) => {
    const end = view.now + ms;
    while (view.now < end) {
      view.now = Math.min(end, view.now + 16);
      for (;;) {
        const due = view.timers.filter((t) => t.at <= view.now).sort((a, b) => a.at - b.at)[0];
        if (!due) break;
        view.timers = view.timers.filter((t) => t !== due);
        due.fn();
      }
      const batch = view.rafQueue;
      view.rafQueue = [];
      for (const r of batch) r.fn(view.now);
    }
  };
  return view;
}

/** The Thinking pane as hud/index.html declares it (T-0128 criterion 5 + B4's label sibling). */
function fakePage(opts = {}) {
  const view = fakeView(opts);
  const doc = { nodeType: 9, contexts: [], defaultView: view, writes: [], activeElement: null };
  doc.createElement = (tag) => makeElement(doc, String(tag).toLowerCase());
  doc.createElementNS = (_ns, tag) => makeElement(doc, String(tag).toLowerCase());
  doc.createTextNode = (data) => makeText(doc, data);
  doc.createDocumentFragment = () => {
    const f = makeElement(doc, "#document-fragment");
    f.nodeType = 11;
    return f;
  };
  const html = makeElement(doc, "html");
  html.setAttribute("data-theme", "dark");
  const body = makeElement(doc, "body");
  html.appendChild(body);
  const main = makeElement(doc, "main");
  body.appendChild(main);
  const pane = makeElement(doc, "section");
  pane.className = "pane pane-thinking";
  main.appendChild(pane);
  const h2 = makeElement(doc, "h2");
  h2.textContent = "Thinking";
  pane.appendChild(h2);
  const label = makeElement(doc, "span");
  label.setAttribute("data-denoise-label", "");
  pane.appendChild(label);
  const meter = makeElement(doc, "div");
  meter.className = "meter";
  pane.appendChild(meter);
  const root = makeElement(doc, "div");
  root.className = "denoise";
  root.setAttribute("data-denoise-root", "");
  pane.appendChild(root);
  const empty = makeElement(doc, "p");
  empty.className = "empty-line denoise-empty";
  empty.setAttribute("data-denoise-empty", "");
  empty.textContent = "no denoising data yet — the model's real canvases appear here while it thinks";
  pane.appendChild(empty);
  doc.documentElement = html;
  doc.body = body;
  doc.head = makeElement(doc, "head");
  doc.querySelector = (sel) => (matches(html, sel) ? html : html.querySelector(sel));
  doc.querySelectorAll = (sel) => html.querySelectorAll(sel);
  doc.getElementById = (id) => html.querySelector("#" + id);
  doc.addEventListener = () => {};
  doc.removeEventListener = () => {};
  // The globals a browser page would also have, as the same fakes (so either spelling works).
  for (const k of ["requestAnimationFrame", "cancelAnimationFrame", "matchMedia", "getComputedStyle", "devicePixelRatio", "MutationObserver", "ResizeObserver"])
    globalThis[k] = view[k];
  globalThis.document = doc;

  const subscribers = [];
  const host = {
    document: doc,
    subscribe: (fn) => (subscribers.push(fn), () => subscribers.splice(subscribers.indexOf(fn), 1)),
    send: () => {
      throw new Error("the denoising view sends nothing (client->server types are the page's whitelist)");
    },
    log: () => {},
  };
  const deliver = (msg) => {
    for (const fn of [...subscribers]) fn(structuredClone(msg));
  };
  return { doc, view, root, label, empty, pane, host, deliver };
}

async function mounted(opts) {
  const { mount } = await panel();
  const page = fakePage(opts);
  await mount(page.host);
  return page;
}

const draws = (page) => Number(page.root.getAttribute("data-denoise-draws") || 0);
const frames = (page) => Number(page.root.getAttribute("data-denoise-frames") || 0);
const canvases = (page) => page.root.querySelectorAll("canvas");

// T-0159 B: the sections and cells (the API in the header).
const STATE_NAMES = ["stable", "kept", "noise", "revised", "committed", "empty"];
/** The root's <details> children: the sections, in DOM order (the first-frame row also checks
 *  that they are all of the root's element children). */
const sections = (page) => page.root.children.filter((el) => el.localName === "details");
const cellsOf = (section) => (section ? section.querySelectorAll("[data-denoise-cell]") : []);
/** A cell's state: data-state, else its one class token naming a state, else null. */
function stateOf(cell) {
  const d = cell.getAttribute("data-state");
  if (d !== null) return d;
  const hits = cell.className.split(/\s+/).filter((c) => STATE_NAMES.includes(c));
  return hits.length === 1 ? hits[0] : null;
}
/** True when the cell, or an element inside it, is marked special (data-special or class "special"). */
function isSpecial(cell) {
  const marked = (el) => el.hasAttribute("data-special") || el.className.split(/\s+/).includes("special");
  return marked(cell) || cell.querySelectorAll("*").some(marked);
}
const summaryText = (section) => (section && section.querySelector("summary") ? section.querySelector("summary").textContent : null);
const texts = (section) => [...cellsOf(section)].map((c) => c.textContent);
const titles = (section) => [...cellsOf(section)].map((c) => c.getAttribute("title") || "");
const statesIn = (section) => [...cellsOf(section)].map(stateOf);
/** The cells a run of writes touched: each logged element mapped to its enclosing cell. */
function touchedCells(page, writes) {
  const out = new Set();
  for (const [el] of writes) {
    for (let n = el; n && n.nodeType === 1; n = n.parentNode) {
      if (n.hasAttribute("data-denoise-cell")) {
        out.add(n);
        break;
      }
    }
  }
  return out;
}

/** Deliver a frame and let the page run 50 ms (three vsyncs) — enough for any rAF-scheduled draw. */
function send(page, msg, ms = 50) {
  page.deliver(msg);
  page.view.advance(ms);
}

// --- B5: never fabricated -------------------------------------------------------------------

pendingT0159B("T-0128c B5 / T-0159 B-2 adversarial: zero frames — no section, no cell, zero draws, no animation loop", async () => {
  const page = await mounted();
  // A whole turn's worth of other traffic, none of it a canvas frame.
  for (const msg of [
    { type: "astra.turn.state", state: "LISTENING", turn_id: "t1" },
    { type: "astra.transcript", turn_id: "t1", text: "what is 17 times 23?", final: true },
    { type: "astra.turn.state", state: "THINKING", turn_id: "t1", thinking_ms: 0 },
    { type: "astra.turn.state", state: "THINKING", turn_id: "t1", thinking_ms: 900 },
    { type: "astra.spoken", turn_id: "t1", utterance_id: "u1", text: "391." },
    { type: "astra.turn.state", state: "IDLE", turn_id: "t1" },
  ])
    send(page, msg);
  page.view.advance(3000);
  assert.equal(page.root.childElementCount, 0, "B5: [data-denoise-root] has no element child before the first frame");
  assert.equal(canvases(page).length, 0);
  assert.equal(page.doc.querySelectorAll("[data-denoise-cell]").length, 0, "T-0159 B-2: no frame, no cell anywhere");
  assert.equal(draws(page), 0, "B5: no frame -> no draw");
  assert.equal(page.label.textContent, "", "B5: no frame -> no label");
  const before = page.view.rafCalls;
  page.view.advance(2000);
  assert.equal(page.view.rafCalls, before, "B5: no requestAnimationFrame loop without a pending frame");
  assert.equal(page.view.timers.length, 0, "B5: no timer left running without a pending frame");
  assert.match(page.empty.textContent, /no denoising data yet — the model's real canvases appear here while it thinks/);
  // The premise the row stands on: the panel is mounted and listening (a later frame does draw).
  send(page, F1);
  assert.equal(sections(page).length, 1, "premise: a real frame does create its canvas's section");
  assert.equal(cellsOf(sections(page)[0]).length, 8, "premise: ... with one cell per position");
});

pendingT0159B("T-0128c B5 / T-0159 B-6: the section is created at the first frame, once, and its cells are updated in place, never replaced", async () => {
  const page = await mounted();
  assert.equal(page.root.childElementCount, 0, "B5: nothing before the first frame");
  send(page, F1);
  assert.equal(page.root.childElementCount, 1, "B-7: one section (canvas 0) in [data-denoise-root]");
  const [sec] = sections(page);
  assert.ok(sec, "B-7: the root's child is a native <details> section; root children: " + page.root.children.map((e) => e.localName));
  assert.equal(sec.firstElementChild && sec.firstElementChild.localName, "summary", "B-7: its first child is a <summary>");
  assert.equal(canvases(page).length, 0, "B-6: no 2D-canvas bitmap any more");
  const cells = [...cellsOf(sec)];
  assert.equal(cells.length, 8, "B-6: one cell per canvas position (CL 8)");
  send(page, F2);
  send(page, commit({ step: 6 }));
  assert.equal(page.root.childElementCount, 1, "still one section after three frames of canvas 0");
  assert.equal(sections(page)[0], sec, "the same <details> is updated, never replaced");
  assert.deepEqual([...cellsOf(sec)], cells, "B-8: the same cell elements are updated in place");
  assert.deepEqual([...cellsOf(sec)].map(stateOf), EXPECT_COMMIT_5_OF_8, "B-5: the commit's states, cell by cell");
});

pendingT0159B("T-0159 B-6: a CL 40 canvas has 40 cells, in position order", async () => {
  const page = await mounted();
  const ids = Array.from({ length: 40 }, (_, p) => p);
  const pieces = ids.map((p) => `t${p}`);
  send(page, step({ canvas_length: 40, canvas: ids, argmax: ids.map((p) => p + 1000), stable: "0000000001", argmax_pieces: pieces }));
  const cells = [...cellsOf(sections(page)[0])];
  assert.equal(cells.length, 40);
  assert.deepEqual(cells.map((c) => c.textContent), pieces, "B-6: cell i shows argmax_pieces[i]");
  assert.deepEqual(cells.map(stateOf), Array.from({ length: 40 }, (_, p) => (p === 39 ? "stable" : "noise")));
});

pendingT0128cB("T-0128c B5 adversarial: frames at steps 4 and 10 only -> labels shown exactly {4, 10}, draws = frames", async () => {
  const page = await mounted();
  const shown = new Set();
  const watch = () => {
    // Sample the label at every vsync: an interpolating page would show 5..9 in between.
    for (let i = 0; i < 40; i += 1) {
      page.view.advance(16);
      if (page.label.textContent) shown.add(page.label.textContent);
    }
  };
  page.deliver(plain(4));
  watch();
  page.deliver(plain(10));
  watch();
  page.view.advance(3000);
  if (page.label.textContent) shown.add(page.label.textContent);
  assert.deepEqual([...shown].sort(), ["canvas 0 · step 10/48", "canvas 0 · step 4/48"], "B5: nothing interpolated");
  assert.equal(frames(page), 2, "B9: two frames received");
  assert.equal(draws(page), 2, "B5: draws = frames — no tween, no extra paint");
});

pendingT0128cB("T-0128c B5 adversarial: astra.canvas.synthetic draws nothing and the line says it was refused", async () => {
  const page = await mounted();
  send(page, { type: "astra.canvas.synthetic", request_id: "chatcmpl-aaaa-11111111", reason: "plugin not loaded" });
  page.view.advance(2000);
  assert.equal(page.root.childElementCount, 0, "B5: a synthetic fallback creates no canvas");
  assert.equal(draws(page), 0, "B5: on astra.canvas.synthetic nothing is drawn");
  assert.equal(page.label.textContent, "", "no label for a synthetic canvas");
  assert.match(page.empty.textContent, /synthetic canvas refused: plugin not loaded/, "B5: the line names the reason verbatim");
});

pendingT0128cB("T-0128c B5: astra.error denoise_no_tap shows its message on the empty line; another error code does not", async () => {
  const page = await mounted();
  const before = page.empty.textContent;
  send(page, { type: "astra.error", code: "llm_timeout", message: "the model did not answer in time" });
  assert.equal(page.empty.textContent, before, "B5 (guard too wide): only denoise_no_tap speaks on this line");
  const message = "no denoising frames from the engine's tap (tcp://127.0.0.1:8002) this session";
  send(page, { type: "astra.error", code: "denoise_no_tap", message });
  assert.ok(
    page.empty.textContent.includes(message),
    "B5: on astra.error denoise_no_tap the empty line shows that message; it reads " + JSON.stringify(page.empty.textContent),
  );
  assert.equal(page.root.childElementCount, 0, "and still nothing is drawn");
  assert.equal(draws(page), 0);
});

// --- B9: counters ---------------------------------------------------------------------------

pendingT0128cB("T-0128c B9: data-denoise-frames / -draws / -digest on the root, the digest of the drawn state array", async () => {
  const m = await model();
  const page = await mounted();
  send(page, F1);
  assert.equal(frames(page), 1);
  assert.equal(draws(page), 1);
  assert.equal(page.root.getAttribute("data-denoise-digest"), m.digest(EXPECT_F1), "B9: digest of F1's hand-computed states");
  send(page, F2);
  assert.equal(frames(page), 2);
  assert.equal(draws(page), 2);
  assert.equal(
    page.root.getAttribute("data-denoise-digest"),
    m.digest(EXPECT_F2),
    "B2/B9: the panel passes the previous frame of the same request and canvas — p0 is revised",
  );
  send(page, commit());
  assert.equal(page.root.getAttribute("data-denoise-digest"), m.digest(EXPECT_COMMIT_5_OF_8));
  assert.equal(frames(page), 3);
  assert.equal(draws(page), 3);
});

pendingT0128cB("T-0128c B9: the digest is deterministic — two pages fed the same frames agree, a different frame differs", async () => {
  const a = await mounted();
  const b = await mounted();
  for (const f of [F1, F2]) {
    send(a, f);
    send(b, f);
  }
  assert.equal(a.root.getAttribute("data-denoise-digest"), b.root.getAttribute("data-denoise-digest"));
  const c = await mounted();
  send(c, F1);
  send(c, { ...F2, canvas_idx: 1 }); // no revised: another canvas
  assert.notEqual(c.root.getAttribute("data-denoise-digest"), a.root.getAttribute("data-denoise-digest"));
});

// --- B4: the label in the page ----------------------------------------------------------------

pendingT0128cB("T-0128c B4: the label element shows the drawn frame's label, denoise and commit", async () => {
  const page = await mounted();
  send(page, F1);
  assert.equal(page.label.textContent, "canvas 0 · step 4/48");
  send(page, F2);
  assert.equal(page.label.textContent, "canvas 0 · step 5/48");
  send(page, commit());
  assert.equal(page.label.textContent, "canvas 0 · committed at step 20/48");
  send(page, plain(3, { canvas_idx: 1 }));
  assert.equal(page.label.textContent, "canvas 1 · step 3/48");
});

pendingT0159B("T-0128c B3 / T-0159 (reading): the newest section's grid carries an aria-label stating the label and the four counts", async () => {
  const page = await mounted();
  // F2 first, then F1 as the next frame of the same (request, canvas). F2 had p1 and p2 stable;
  // F1's argmax is 6 at p1 (F2's was 6: not revised) and 70 at p2 (F2's was 71: revised). F1's
  // states are then [stable, stable, revised, stable, noise, kept, noise, kept]:
  // stable 3, kept 2, noise 2, revised 1 — four distinct counts, so a swapped pair is caught.
  // T-0159 reading: the bitmap's aria-label (T-0128c B3, O-B (3)) moves to the section: the
  // <details> or an element inside it carries it (T-0159 does not restate it; kept, not dropped).
  send(page, F2);
  send(page, F1);
  const [sec] = sections(page);
  const holders = sec ? [sec, ...sec.querySelectorAll("[aria-label]")].filter((e) => e.hasAttribute("aria-label")) : [];
  const aria = holders.map((e) => e.getAttribute("aria-label")).find((a) => a.includes("canvas 0 · step 4/48")) || null;
  assert.ok(aria, "B3: the section (or its grid) carries an aria-label with the label; found " + JSON.stringify(holders.map((e) => e.getAttribute("aria-label"))));
  assert.ok(aria.includes("canvas 0 · step 4/48"), "B3: the aria-label states the label: " + aria);
  for (const [word, n] of [["stable", 3], ["kept", 2], ["noise", 2], ["revised", 1]]) {
    const re = new RegExp(`\\b${n}\\s+${word}\\b|\\b${word}\\W+${n}\\b`, "i");
    assert.match(aria, re, `B3: the aria-label states ${n} ${word}: ${aria}`);
  }
});

// --- B6: which request -------------------------------------------------------------------------

// B6 as ruled in O-C5 (docs/tasks/T-0128c.md; supersedes O-B (1) and O-B3 F5): the request whose
// first frame arrived most recently takes the view AT ONCE; frames of any other request are not
// drawn, but are still counted in data-denoise-frames (O-B (6)); an older request can never take
// the view back; FOCUS_MS (the 1500 ms silence window) is gone. The live defect behind it (second
// /launch at ffba98c): every turn opens with the anticipatory prefill (admitted in LISTENING),
// which denoises 2-5 steps and is aborted ~150 ms later when the generate request starts; the
// generate's whole canvas (11-18 steps) lands within 0.4-1 s, so the window never let it in and
// the page showed only the prefill.
//
// Mutants these rows are aimed at: the old silence window (any window: 1500 ms fails the 50 ms and
// 1400 ms gaps, a short one fails the 0 ms gap); "older request takes back" (latest frame's request
// wins; or takes back after the shown one's silence; or a new canvas_idx of an old request counts
// as a new first frame); "only the first takeover" (the third request).

const OTHER = "chatcmpl-bbbb-22222222";
const PREFILL = "chatcmpl-1111111111111111-0000000a";
const GENERATE = "chatcmpl-2222222222222222-0000000b";
const THIRD = "chatcmpl-3333333333333333-0000000c";

/** plain(s)'s states by its construction (no stable bit, so no revised): positions below
 *  min(8, s % 9) kept, the rest noise. */
const plainStates = (s) => [0, 1, 2, 3, 4, 5, 6, 7].map((p) => (p < Math.min(8, s % 9) ? "kept" : "noise"));

/** Deliver `msg`, run the page `ms` (25 frames/s by default), and say what it shows then. */
function shownAfter(page, msg, ms = 40) {
  send(page, msg, ms);
  return {
    label: page.label.textContent,
    digest: page.root.getAttribute("data-denoise-digest"),
    draws: draws(page),
    frames: frames(page),
  };
}

pendingT0128cB("T-0128c B6/O-C5: prefill 3 frames, then the generate request 50 ms later — all 10 of its frames are drawn and the label ends on its commit", async () => {
  const m = await model();
  // 50 ms is the ruling's case; 0 ms (same tick) kills a short silence window; 1400 ms is the old
  // row's "keeps the view" gap, now a takeover too.
  for (const gap of [50, 0, 1400]) {
    const page = await mounted();
    for (let s = 1; s <= 3; s += 1) send(page, plain(s, { request_id: PREFILL }), s < 3 || gap >= 40 ? 40 : 0);
    if (gap >= 40) {
      assert.equal(draws(page), 3, "O-C5: the prefill's own frames are real canvases, drawn while it is the newest request");
      assert.equal(page.label.textContent, "canvas 0 · step 3/48", "premise: the prefill is shown");
      page.view.advance(gap - 40);
    }
    // the generate request: canvas 0, steps 1..9 denoising, then its commit at step 10 (5 of 8)
    const gen = [...Array.from({ length: 9 }, (_, i) => plain(i + 1, { request_id: GENERATE })), commit({ request_id: GENERATE, step: 10 })];
    const got = gen.map((f) => shownAfter(page, f));
    assert.deepEqual(
      got.map((g) => g.label),
      [...Array.from({ length: 9 }, (_, i) => `canvas 0 · step ${i + 1}/48`), "canvas 0 · committed at step 10/48"],
      `B6/O-C5 (gap ${gap} ms): every generate frame is shown as it arrives, ending on its commit`,
    );
    assert.deepEqual(
      got.map((g) => g.digest),
      [...Array.from({ length: 9 }, (_, i) => m.digest(plainStates(i + 1))), m.digest(EXPECT_COMMIT_5_OF_8)],
      `B6/O-C5 (gap ${gap} ms): the drawn state is the generate frame's own, frame by frame`,
    );
    if (gap >= 40)
      assert.deepEqual(got.map((g) => g.draws), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13], "B6/O-C5: all 10 generate frames are drawn (draws 3 -> 13)");
    assert.equal(frames(page), 13, "B9: 3 prefill + 10 generate frames received");
  }
});

pendingT0128cB("T-0128c B6/O-C5: after the takeover an older request's frames are never drawn — at once, after a long silence, on a new canvas — but are counted", async () => {
  const m = await model();
  const page = await mounted();
  for (let s = 1; s <= 3; s += 1) send(page, plain(s, { request_id: PREFILL }), 40);
  send(page, plain(1, { request_id: GENERATE }), 40);
  send(page, plain(2, { request_id: GENERATE }), 40);
  const held = { label: "canvas 0 · step 2/48", digest: m.digest(plainStates(2)) };
  assert.equal(page.label.textContent, held.label, "premise: the generate request took the view");
  assert.equal(page.root.getAttribute("data-denoise-digest"), held.digest);
  assert.equal(draws(page), 5, "premise: 3 prefill + 2 generate frames drawn");

  // the aborted prefill's late frame, 40 ms after the takeover ("latest frame wins" draws it)
  let g = shownAfter(page, plain(4, { request_id: PREFILL }));
  assert.equal(g.label, held.label, "B6/O-C5: an older request's frame after the takeover does not change the label");
  assert.equal(g.digest, held.digest, "B6/O-C5: ... nor the drawn state");
  assert.equal(g.draws, 5, "B6/O-C5: ... and is not drawn");
  assert.equal(g.frames, 6, "O-B (6): ... but is counted in data-denoise-frames");

  // the generate request goes on and is drawn
  g = shownAfter(page, plain(5, { request_id: GENERATE }));
  assert.equal(g.label, "canvas 0 · step 5/48", "the shown request's next frame draws");
  assert.equal(g.draws, 6);
  const held2 = { label: g.label, digest: m.digest(plainStates(5)) };
  assert.equal(g.digest, held2.digest);

  // the shown request falls silent for 5 s; the older request can still never take the view back,
  // with a denoise frame, a frame of a canvas it has not sent before, or a commit
  page.view.advance(5000);
  for (const [what, f] of [
    ["a denoise frame after 5 s of silence", plain(6, { request_id: PREFILL })],
    ["its first frame of a new canvas", plain(7, { request_id: PREFILL, canvas_idx: 1 })],
    ["a commit", commit({ request_id: PREFILL, step: 8 })],
  ]) {
    g = shownAfter(page, f);
    assert.equal(g.label, held2.label, `B6/O-C5: an older request never takes the view back (${what})`);
    assert.equal(g.digest, held2.digest, `B6/O-C5: the drawn state stays (${what})`);
    assert.equal(g.draws, 6, `B6/O-C5: nothing drawn (${what})`);
  }
  assert.equal(frames(page), 10, "O-B (6): every received frame is counted, drawn or not");

  // and the guard is not so wide that nothing takes over: a new request's first frame does
  g = shownAfter(page, plain(2, { request_id: THIRD, canvas_idx: 2 }));
  assert.equal(g.label, "canvas 2 · step 2/48", "a newer request's first frame still takes the view");
  assert.equal(g.draws, 7);
});

pendingT0128cB("T-0128c B6/O-C5: a third request takes over from the second while the second is still denoising", async () => {
  const m = await model();
  const page = await mounted();
  for (let s = 1; s <= 3; s += 1) send(page, plain(s, { request_id: PREFILL }), 40);
  for (let s = 1; s <= 5; s += 1) send(page, plain(s, { request_id: GENERATE }), 40);
  assert.equal(page.label.textContent, "canvas 0 · step 5/48", "premise: the second request is shown");
  assert.equal(draws(page), 8);
  // 40 ms after the second request's last frame (it is mid-canvas): the third request's first frame
  let g = shownAfter(page, plain(1, { request_id: THIRD, canvas_idx: 3 }));
  assert.equal(g.label, "canvas 3 · step 1/48", "B6/O-C5: the newest request takes the view at once");
  assert.equal(g.digest, m.digest(plainStates(1)));
  assert.equal(g.draws, 9);
  // now the second request is the older one, and so is the first
  for (const [who, f] of [
    ["the second request", plain(6, { request_id: GENERATE })],
    ["the first request", plain(4, { request_id: PREFILL })],
    ["the second request's commit", commit({ request_id: GENERATE, step: 7 })],
  ]) {
    g = shownAfter(page, f);
    assert.equal(g.label, "canvas 3 · step 1/48", `B6/O-C5: ${who} no longer draws after the third request's first frame`);
    assert.equal(g.draws, 9, `B6/O-C5: ${who} is not drawn`);
  }
  g = shownAfter(page, plain(2, { request_id: THIRD, canvas_idx: 3 }));
  assert.equal(g.label, "canvas 3 · step 2/48", "the third request goes on drawing");
  assert.equal(g.digest, m.digest(plainStates(2)));
  g = shownAfter(page, commit({ request_id: THIRD, canvas_idx: 3, step: 9 }));
  assert.equal(g.label, "canvas 3 · committed at step 9/48");
  assert.equal(g.digest, m.digest(EXPECT_COMMIT_5_OF_8));
  assert.equal(g.draws, 11);
  assert.equal(g.frames, 14, "O-B (6): 3 + 5 + 1 + 3 + 2 frames received");
});

// --- B7: between turns --------------------------------------------------------------------------

pendingT0159B("T-0128c B7 / T-0159 B-7: the last frame, its cells and label stay through IDLE and silence until another request's first frame", async () => {
  const m = await model();
  const page = await mounted();
  send(page, F1);
  send(page, commit({ committed_pieces: ["The", " cat", " sat", " on", " mat"] }));
  const [sec] = sections(page);
  const digest = page.root.getAttribute("data-denoise-digest");
  const n = draws(page);
  send(page, { type: "astra.turn.state", state: "SPEAKING", turn_id: "t1" });
  send(page, { type: "astra.turn.state", state: "IDLE", turn_id: "t1" });
  page.view.advance(30000);
  assert.ok(sec, "premise: the frames drew a section");
  assert.equal(sections(page)[0], sec, "B7: the drawn section stays");
  assert.deepEqual(texts(sec), ["The", "cat", "sat", "on", "mat", "", "", ""], "B7: its tokens stay");
  assert.equal(page.label.textContent, "canvas 0 · committed at step 20/48", "B7: the label stays");
  assert.equal(page.root.getAttribute("data-denoise-digest"), digest, "B7: the drawn state stays");
  assert.equal(draws(page), n, "B7: nothing is redrawn while nothing arrives");
  send(page, plain(1, { request_id: OTHER, canvas_idx: 0 }));
  assert.equal(page.label.textContent, "canvas 0 · step 1/48", "B7: the next request's first frame replaces it");
  assert.equal(page.root.getAttribute("data-denoise-digest"), m.digest(["kept", "noise", "noise", "noise", "noise", "noise", "noise", "noise"]));
  assert.equal(sections(page).length, 1, "B-7: a new request starts with an empty view: one section, its own");
  assert.notEqual(sections(page)[0], sec, "B-7: the previous request's section is gone");
  assert.ok(!page.root.contains(sec), "B-7: ... removed from the root, not hidden in it");
  assert.deepEqual(texts(sections(page)[0]), Array(8).fill(""), "B-2: the new frame carries no pieces, so no text — nothing carried over");
});

pendingT0159B("T-0128c B7 / T-0159: a fresh page (a reload) starts empty — frames are not kept anywhere the next page reads", async () => {
  const first = await mounted();
  send(first, F1);
  assert.equal(sections(first).length, 1, "premise: the first page drew a section");
  const second = await mounted();
  second.view.advance(1000);
  assert.equal(second.root.childElementCount, 0, "B7: a reload shows the empty state until the next frame");
  assert.equal(draws(second), 0);
  assert.equal(second.label.textContent, "");
});

// --- B8: reduced motion --------------------------------------------------------------------------

/** 48 frames at 25 frames/s (40 ms apart): canvas 0, steps 1..48, no stable bits (no revised). */
function stream(page, { every = 40 } = {}) {
  const labels = [];
  for (let s = 1; s <= 48; s += 1) {
    page.deliver(plain(s));
    page.view.advance(every);
    labels.push(page.label.textContent);
  }
  return labels;
}

pendingT0128cB("T-0128c B8: reduced motion — at most two redraws per second, latest wins, the label updates every frame", async () => {
  const m = await model();
  const page = await mounted({ reduce: true });
  const labels = stream(page); // 48 frames over 1920 ms
  // <= 2 per second over 1.92 s: at most 4, plus the first frame drawn at once = 5 (generous).
  assert.ok(draws(page) <= 5, `B8: ${draws(page)} redraws in 1.92 s at 25 frames/s under reduced motion (<= 2/s)`);
  assert.ok(draws(page) >= 1, "B8: reduced motion still draws");
  assert.deepEqual(
    labels,
    Array.from({ length: 48 }, (_, i) => `canvas 0 · step ${i + 1}/48`),
    "B8: the label still updates every frame",
  );
  page.view.advance(1000);
  // Latest wins: once the stream stops, the last frame is what is drawn. plain(48): 48 % 9 = 3
  // positions kept, the rest noise.
  assert.equal(
    page.root.getAttribute("data-denoise-digest"),
    m.digest(["kept", "kept", "kept", "noise", "noise", "noise", "noise", "noise"]),
    "B8: latest wins — the last frame is drawn within half a second of the stream stopping",
  );
  assert.equal(frames(page), 48, "B9: every frame is counted even when it is not drawn");
});

pendingT0128cB("T-0128c B8 (the other side): without reduced motion every frame 40 ms apart is drawn", async () => {
  const page = await mounted({ reduce: false });
  stream(page);
  assert.equal(draws(page), 48, "B8 applies only under prefers-reduced-motion: reduce");
});

// --- O-B3: the group B review findings (F1-F5) ------------------------------------------------
//
// docs/tasks/T-0128c.md ruling O-B3. F1's Chrome twin is in tests/integration/test_hud_denoise.py.

/** Play a theme switch through every seam a page could watch: the attribute itself, every
 *  MutationObserver observing <html>, every prefers-color-scheme media listener. */
function switchTheme(page, theme) {
  const { doc, view } = page;
  const html = doc.documentElement;
  const old = html.getAttribute("data-theme");
  html.setAttribute("data-theme", theme);
  for (const obs of view.mutationObservers)
    for (const { target } of obs.targets)
      if (target === html || target.contains?.(html))
        obs.fn([{ type: "attributes", attributeName: "data-theme", target: html, oldValue: old }], obs);
  for (const l of view.mediaListeners)
    if (/prefers-color-scheme/.test(l.query)) l.fn.call(l.mql, { matches: theme === "dark", media: l.query });
}

/** Play a resize of the rail to `width` CSS px: the root's box changes, then every ResizeObserver
 *  watching it and every window "resize" listener is told. */
function resizeRoot(page, width) {
  const { root, view } = page;
  root.clientWidth = width;
  root.offsetWidth = width;
  root.getBoundingClientRect = () => ({ x: 0, y: 0, left: 0, top: 0, width, height: 0, right: width, bottom: 0 });
  for (const obs of view.resizeObservers)
    if (obs.targets.some((t) => t === root || t.contains?.(root)))
      obs.fn([{ target: root, contentRect: { width, height: 0 }, contentBoxSize: [{ inlineSize: width, blockSize: 0 }], borderBoxSize: [{ inlineSize: width, blockSize: 0 }] }], obs);
  for (const l of view.listeners) if (l.type === "resize") l.fn({ type: "resize" });
}

/** No animation loop is left: over 3 s of virtual time after `settleMs`, no new
 *  requestAnimationFrame call, and nothing queued or pending at the end. */
function assertQuiet(page, when, settleMs = 1000) {
  page.view.advance(settleMs);
  const before = page.view.rafCalls;
  page.view.advance(3000);
  assert.equal(page.view.rafCalls, before, `O-B3 F2 (${when}): requestAnimationFrame is still being called with nothing to draw`);
  assert.equal(page.view.rafQueue.length, 0, `O-B3 F2 (${when}): an animation frame is still queued`);
  assert.equal(page.view.timers.length, 0, `O-B3 F2 (${when}): a timer is still pending`);
}

pendingT0159B("T-0128c O-B3 F2 / T-0159 B-8: no rAF or timer keeps running after a stream stops, after a theme switch, after a resize", async () => {
  for (const reduce of [false, true]) {
    const page = await mounted({ reduce });
    stream(page);
    assertQuiet(page, `stream stopped, reduce=${reduce}`);
    const n = draws(page);
    assert.ok(n >= 1, "premise: the stream was drawn");
    switchTheme(page, "light");
    assertQuiet(page, `after a theme switch, reduce=${reduce}`);
    resizeRoot(page, 906);
    assertQuiet(page, `after a resize, reduce=${reduce}`);
    // T-0159: a DOM grid takes its colours from CSS and its columns from layout, so it need not
    // watch the theme or its size at all (the T-0128c premises that it did are dropped; a repaint
    // of the same frame stays allowed, O-B3). What stays: the grid is still the last frame's.
    assert.deepEqual(
      statesIn(sections(page)[0]),
      ["kept", "kept", "kept", "noise", "noise", "noise", "noise", "noise"],
      "plain(48): 48 % 9 = 3 kept positions, still drawn after the theme switch and the resize",
    );
  }
});

pendingT0128cB("T-0128c O-B3 F3: a position stable in the previous frame AND now whose argmax moved is revised", async () => {
  const m = await model();
  // CL 8. A: p0, p1 stable ("c0" = 1100 0000), argmax 5 and 6; the rest noise.
  const A = step({ step: 4, canvas: [5, 6, 0, 0, 0, 0, 0, 0], argmax: [5, 6, 1, 1, 1, 1, 1, 1], stable: "c0" });
  // B: p0, p1 still stable ("c0"); p0's argmax moved 5 -> 50 (canvas follows it), p1's did not.
  const B = step({ step: 5, canvas: [50, 6, 0, 0, 0, 0, 0, 0], argmax: [50, 6, 1, 1, 1, 1, 1, 1], stable: "c0" });
  const want = ["revised", "stable", "noise", "noise", "noise", "noise", "noise", "noise"];
  // "revised only when now unstable" reads p0 stable; "stable outranks revised" reads p0 stable.
  assert.deepEqual(m.cellStates(B, A), want, "O-B (4): revised whatever its current stable bit");
  assert.deepEqual(m.cellStates(B, null), ["stable", "stable", "noise", "noise", "noise", "noise", "noise", "noise"], "no previous frame: p0 is plain stable");
  // and in the panel: the digest drawn for B is that of `want`.
  const page = await mounted();
  send(page, A);
  send(page, B);
  assert.equal(page.root.getAttribute("data-denoise-digest"), m.digest(want));
});

pendingT0128cB("T-0128c O-B3 F4: the digest is FNV-1a 32 of the state names joined by ',' — pinned vectors", async () => {
  const m = await model();
  // Computed from the FNV-1a definition (offset basis 0x811c9dc5, prime 0x01000193, xor then
  // multiply, per byte) over the ASCII text; the same procedure gives the published vectors
  // FNV-1a("a") = e40c292c and FNV-1a("foobar") = bf9cf968. Kill table (same text):
  //   ';' separator -> 54fb7286; multiply-then-xor (FNV-1) -> 9831d4d6.
  assert.equal(m.digest(["stable", "kept", "noise"]), "914acc9c", 'FNV-1a("stable,kept,noise")');
  assert.equal(m.digest(["committed", "empty"]), "cf21350c", 'FNV-1a("committed,empty")');
  assert.equal(m.digest(["revised"]), "991aded7", 'FNV-1a("revised") — one state, no separator');
  assert.equal(m.digest(["noise", "kept"]), "57fe1537", 'FNV-1a("noise,kept")');
});

// O-B3 F5 (an older request's frame during the shown one's silence does not take over) is
// superseded by O-C5; its "older never takes the view back" half lives on in the B6/O-C5 rows.

/** CL 8 frame whose kept set is the bit pattern of `s` (canvas == argmax at p iff bit p of s):
 *  every s in 1..255 has its own state array, so its own digest. */
function patterned(s) {
  return step({
    step: s,
    max_steps: 64,
    canvas: [0, 1, 2, 3, 4, 5, 6, 7].map((p) => 100 + p),
    argmax: [0, 1, 2, 3, 4, 5, 6, 7].map((p) => ((s >> p) & 1 ? 100 + p : 200 + p)),
    stable: "00",
  });
}

pendingT0128cB("T-0128c O-B3 F1: reduced motion with the rail width switching every frame — at most 5 new-frame draws for 50 frames", async () => {
  // "New-frame draw" = a write of data-denoise-digest that changes its value. Every frame here has
  // a distinct state array (patterned), so a repaint of the SAME frame at a new size keeps the
  // digest (allowed, O-B3), and drawing a NEWER frame early changes it (counted).
  const page = await mounted({ reduce: true });
  const writes = [];
  const set = page.root.setAttribute.bind(page.root);
  page.root.setAttribute = (k, v) => {
    if (k === "data-denoise-digest") writes.push(String(v));
    set(k, v);
  };
  const labels = [];
  for (let s = 1; s <= 50; s += 1) {
    page.deliver(patterned(s));
    resizeRoot(page, s % 2 ? 906 : 342); // the rail at 1000 px, then at 1440 px
    page.view.advance(40);
    labels.push(page.label.textContent);
  }
  page.view.advance(1000);
  const changes = writes.filter((v, i) => i === 0 || v !== writes[i - 1]).length;
  assert.ok(changes >= 2, `premise: reduced motion still draws new frames (${changes})`);
  assert.ok(changes <= 5, `O-B3 F1: ${changes} new-frame draws for 50 frames over 2 s under reduced motion (<= 5)`);
  assert.deepEqual(labels, Array.from({ length: 50 }, (_, i) => `canvas 0 · step ${i + 1}/64`), "O-B (2): the label takes every frame");
  const m = await model();
  const last = [0, 1, 2, 3, 4, 5, 6, 7].map((p) => ((50 >> p) & 1 ? "kept" : "noise"));
  assert.equal(page.root.getAttribute("data-denoise-digest"), m.digest(last), "latest wins once the stream stops");
});

// =================================================================================================
// T-0159 group B — the token cells (docs/tasks/T-0159.md B-1 .. B-10, *Tests to write* "B, node").
//
// The pieces below are hand-written; their display forms and titles are worked out by hand from
// B-6's rules (trim leading/trailing spaces; only spaces -> "␣"; "\n" -> "↵"; "\t" -> "⇥"; a `<…>`
// piece is special) and written out literally.
// =================================================================================================

// F1 and F2 (above) with the A1 field `argmax_pieces` (length canvas_length). F2's argmax differs
// from F1's at p0 (5 -> 50), p2 (70 -> 71) and p7 (12 -> 13); its pieces differ at exactly those.
const F1P = { ...F1, argmax_pieces: [" The", " cat", "<|channel>", "\n", " ", "\t", "</script><b>x", "s"] };
const SHOW_F1P = ["The", "cat", "<|channel>", "↵", "␣", "⇥", "</script><b>x", "s"];
const TITLE_F1P = ["␣The", "␣cat", "<|channel>", "↵", "␣", "⇥", "</script><b>x", "s"];
const SPECIAL_F1P = [false, false, true, false, false, false, false, false];
const F2P = { ...F2, argmax_pieces: [" A", " cat", " sat", "\n", " ", "\t", "</script><b>x", "<eos>"] };
const SHOW_F2P = ["A", "cat", "sat", "↵", "␣", "⇥", "</script><b>x", "<eos>"];
// A commit of 5 of 8 with the A1 field `committed_pieces` (length len(committed_tokens)).
const CP = commit({ committed_pieces: ["The", " cat", " sat", " on", " mat"] });
const SHOW_CP = ["The", "cat", "sat", "on", "mat", "", "", ""];
const TITLE_CP = ["The", "␣cat", "␣sat", "␣on", "␣mat", "", "", ""];

pendingT0159B("T-0159 B-6: each cell shows its argmax piece in display form; the text follows the argmax at every drawn frame", async () => {
  const page = await mounted();
  send(page, F1P);
  const [sec] = sections(page);
  assert.ok(sec, "B-7: the first frame draws its canvas's section" + " (root children: " + page.root.childElementCount + ")");
  assert.deepEqual(texts(sec), SHOW_F1P, "B-6: cell i shows the display form of argmax_pieces[i]");
  assert.deepEqual(statesIn(sec), EXPECT_F1, "B-5: the cells carry F1's hand-computed states");
  assert.deepEqual(titles(sec), TITLE_F1P, "B-6: the title is the full piece with whitespace made visible");
  send(page, F2P);
  assert.deepEqual(texts(sec), SHOW_F2P, "B-6: a cell whose argmax changed shows the new piece (p0, p2, p7)");
  assert.deepEqual(statesIn(sec), EXPECT_F2, "B-5: F2's states (p0 revised) — the pieces change no state");
  // Noise cells show their current guess (dimmed by CSS, checked in Chrome), never a blank.
  assert.equal(texts(sec)[4], "␣", "B-6: a noise cell still shows its guess");
  assert.equal(texts(sec)[6], "</script><b>x", "B-6/B-3: ... literally");
});

pendingT0159B("T-0159 B-6: a commit frame shows committed_pieces below len(committed_tokens); the tail cells are empty with no text, whatever was there before", async () => {
  const page = await mounted();
  send(page, F1P);
  send(page, CP);
  const [sec] = sections(page);
  assert.ok(sec, "a section was drawn");
  assert.deepEqual(texts(sec), SHOW_CP, "B-6: committed pieces, then no text in the empty tail (F1P's p5..p7 text is gone)");
  assert.deepEqual(titles(sec), TITLE_CP, "B-6: the tail cells carry no title either");
  assert.deepEqual(statesIn(sec), EXPECT_COMMIT_5_OF_8, "B-5: committed x5, empty x3");
});

pendingT0159B("T-0159 B-2 adversarial: no pieces means no text — a frame without pieces clears the text a previous frame left", async () => {
  const page = await mounted();
  send(page, F1);
  const [sec] = sections(page);
  assert.ok(sec, "premise: a frame without pieces still draws its cells (the states come from the ids)");
  assert.deepEqual(texts(sec), Array(8).fill(""), "B-2: no pieces, no text");
  assert.deepEqual(titles(sec), Array(8).fill(""), "B-2: no pieces, no title");
  assert.deepEqual(statesIn(sec), EXPECT_F1, "B-2: the cells are drawn as today: states from the ids");
  send(page, F2P);
  assert.deepEqual(texts(sec), SHOW_F2P, "premise: pieces do show");
  send(page, plain(9)); // the next frame of the same request and canvas, without pieces
  assert.deepEqual(texts(sec), Array(8).fill(""), "B-2: a cell shows text only from the frame it draws — never the last frame's");
  assert.deepEqual(titles(sec), Array(8).fill(""));
  send(page, commit());
  assert.deepEqual(texts(sec), Array(8).fill(""), "B-2: a commit without committed_pieces shows no text");
});

pendingT0159B("T-0159 B-1 adversarial: the page reads the pieces it is given — one id with eight pieces shows eight texts, eight ids with one piece show one", async () => {
  const page = await mounted();
  const sameId = step({ canvas: Array(8).fill(7), argmax: Array(8).fill(7), stable: "00", argmax_pieces: ["one", "two", "three", "four", "five", "six", "seven", "eight"] });
  send(page, sameId);
  assert.deepEqual(texts(sections(page)[0]), ["one", "two", "three", "four", "five", "six", "seven", "eight"], "B-1: text comes from argmax_pieces, not from an id table");
  const manyIds = step({ step: 5, canvas: [1, 2, 3, 4, 5, 6, 7, 8], argmax: [11, 12, 13, 14, 15, 16, 17, 18], stable: "00", argmax_pieces: Array(8).fill("x") });
  send(page, manyIds);
  assert.deepEqual(texts(sections(page)[0]), Array(8).fill("x"), "B-1: eight ids shown as the one piece given for each");
});

pendingT0159B("T-0159 B-6: display forms — spaces, newlines, tabs, specials (and `<` inside a piece is not special)", async () => {
  const page = await mounted();
  const pieces = ["  ", "a b", "\n\n", "<0xE2>", "<eos>", "a<b>", " x ", "<channel|>"];
  send(page, step({ canvas: [1, 2, 3, 4, 5, 6, 7, 8], argmax: [1, 2, 3, 4, 5, 6, 7, 8], stable: "ff", argmax_pieces: pieces }));
  const sec = sections(page)[0];
  assert.ok(sec, "a section was drawn");
  assert.deepEqual(texts(sec), ["␣", "a b", "↵↵", "<0xE2>", "<eos>", "a<b>", "x", "<channel|>"], "B-6 display forms");
  assert.deepEqual(titles(sec), ["␣␣", "a␣b", "↵↵", "<0xE2>", "<eos>", "a<b>", "␣x␣", "<channel|>"], "B-6: titles show every whitespace");
  assert.deepEqual([...cellsOf(sec)].map(isSpecial), [false, false, false, true, true, false, false, true], "B-6: `<…>` pieces (bytes included) are marked; `a<b>` is not");
  send(page, F1P);
  assert.deepEqual([...cellsOf(sec)].map(isSpecial), SPECIAL_F1P, "B-6: the mark follows the piece — set on <|channel>, cleared where the previous frame had a special");
});

pendingT0159B("T-0159 B-6: a piece longer than the cell keeps the cell's width — cut with … (CSS or text), the full piece in its title", async () => {
  const page = await mounted();
  const long = "x".repeat(40) + "END";
  send(page, step({ canvas: [1, 2, 3, 4, 5, 6, 7, 8], argmax: [1, 2, 3, 4, 5, 6, 7, 8], stable: "ff", argmax_pieces: [long, "a", "b", "c", "d", "e", "f", "g"] }));
  const [cell] = [...cellsOf(sections(page)[0])];
  assert.ok(cell, "a cell was drawn");
  const t = cell.textContent;
  const cutInText = t.endsWith("…") && t.length > 1 && long.startsWith(t.slice(0, -1));
  assert.ok(t === long || cutInText, "B-6: the cell shows the piece (cut by CSS) or a prefix ending in …; got " + JSON.stringify(t));
  assert.equal(cell.getAttribute("title"), long, "B-6: the full piece is in the title");
});

pendingT0159B("T-0159 B-3 adversarial: pieces are text, never markup — no element is made from a piece", async () => {
  const page = await mounted();
  // The fake DOM throws on innerHTML / insertAdjacentHTML; here also no element named by a piece.
  const pieces = ["</script><b>x", "<img src=x onerror=alert(1)>", "<b>", "&amp;", "<script>", "</details>", "<summary>", "x"];
  send(page, step({ canvas: [1, 2, 3, 4, 5, 6, 7, 8], argmax: [1, 2, 3, 4, 5, 6, 7, 8], stable: "ff", argmax_pieces: pieces }));
  const sec = sections(page)[0];
  assert.ok(sec, "a section was drawn");
  assert.deepEqual(texts(sec), pieces, "B-3: each piece shows literally");
  assert.equal(page.root.querySelectorAll("b, img, script").length, 0, "B-3: no <b>, <img> or <script> was created");
  assert.equal(page.root.querySelectorAll("summary").length, 1, "B-3: the only <summary> is the section's own");
  assert.equal(page.root.querySelectorAll("details").length, 1, "B-3: the only <details> is the section itself");
});

/** Every element of the page outside [data-denoise-root] whose own text or an attribute holds `s`. */
function leaksOutsideRoot(page, s) {
  const out = [];
  const walk = (el) => {
    if (el === page.root) return;
    for (const c of el.childNodes) {
      if (c.nodeType === 3 && String(c.data).includes(s)) out.push(`<${el.localName}> text`);
      if (c.nodeType === 1) walk(c);
    }
    for (const a of el.attributes) if (String(a.value).includes(s)) out.push(`<${el.localName} ${a.name}>`);
  };
  walk(page.doc.documentElement);
  if (String(page.doc.title || "").includes(s)) out.push("document.title");
  return out;
}

pendingT0159B("T-0159 B-4: thought text from a canvas frame appears only inside [data-denoise-root]", async () => {
  const page = await mounted();
  const MARK = "QZXthoughtQZX";
  send(page, step({ canvas: [1, 2, 3, 4, 5, 6, 7, 8], argmax: [1, 2, 3, 4, 5, 6, 7, 8], stable: "f0", argmax_pieces: Array.from({ length: 8 }, (_, i) => `${MARK}${i}`) }));
  send(page, commit({ committed_pieces: Array.from({ length: 5 }, (_, i) => ` ${MARK}c${i}`) }));
  send(page, plain(2, { canvas_idx: 1, argmax_pieces: Array(8).fill(`<|channel>${MARK}`) }));
  assert.ok(page.root.textContent.includes(MARK), "premise: the pieces are shown in the root");
  assert.deepEqual(leaksOutsideRoot(page, MARK), [], "B-4: no element outside the denoising view gains model text");
  assert.equal(page.label.textContent, "canvas 1 · step 2/48", "the pane label carries the label, not text");
});

pendingT0159B("T-0159 B-5: the pieces never change a state or the digest — the same ids with and without pieces draw the same", async () => {
  const m = await model();
  const bare = await mounted();
  const pieced = await mounted();
  const other = await mounted();
  const seq = [
    [F1, F1P, { ...F1, argmax_pieces: Array(8).fill("<eos>") }, EXPECT_F1],
    [F2, F2P, { ...F2, argmax_pieces: Array(8).fill(" ") }, EXPECT_F2],
    [commit(), CP, commit({ committed_pieces: Array(5).fill("zz") }), EXPECT_COMMIT_5_OF_8],
  ];
  for (const [a, b, c, want] of seq) {
    send(bare, a);
    send(pieced, b);
    send(other, c);
    for (const page of [bare, pieced, other]) {
      assert.equal(page.root.getAttribute("data-denoise-digest"), m.digest(want), "B-5: data-denoise-digest exactly as at BASE");
      assert.deepEqual(statesIn(sections(page)[0]), want, "B-5: the cell states from ids and masks only");
    }
  }
});

pendingT0159B("T-0159 B-7: a new canvas appends an open section and closes the previous newest; a closed section keeps its last drawn tokens", async () => {
  const page = await mounted();
  send(page, F1P);
  send(page, CP);
  const c1 = plain(3, { canvas_idx: 1, argmax_pieces: ["I", " am", " the", " next", " canvas", ".", "\n", "<eos>"] });
  send(page, c1);
  const [s0, s1] = sections(page);
  assert.equal(sections(page).length, 2, "B-7: one section per canvas_idx of the shown request");
  assert.ok(s0 && s1 && s0.localName === "details" && s1.localName === "details", "B-7: native <details>");
  assert.ok(summaryText(s0).includes("canvas 0 · committed at step 20/48"), "B-7: summary in the B4 form: " + summaryText(s0));
  assert.ok(summaryText(s1).includes("canvas 1 · step 3/48"), "B-7: summary in the B4 form: " + summaryText(s1));
  assert.equal(s1.hasAttribute("open"), true, "B-7: the new canvas's section is open");
  assert.equal(s0.hasAttribute("open"), false, "B-7: the previous newest closes");
  assert.equal(page.label.textContent, "canvas 1 · step 3/48", "B-7: the pane label is the newest drawn frame's");
  assert.deepEqual(texts(s0), SHOW_CP, "B-7: the closed section keeps canvas 0's last drawn tokens");
  assert.deepEqual(statesIn(s0), EXPECT_COMMIT_5_OF_8);
  assert.deepEqual(texts(s1), ["I", "am", "the", "next", "canvas", ".", "↵", "<eos>"]);
  // canvas 1 goes on; canvas 0's section is not touched at all
  page.doc.writes.length = 0;
  send(page, plain(4, { canvas_idx: 1, argmax_pieces: ["I", " am", " the", " last", " canvas", ".", "\n", "<eos>"] }));
  assert.ok(summaryText(s1).includes("canvas 1 · step 4/48"), "B-7: the summary follows its canvas's drawn frame");
  assert.ok(summaryText(s0).includes("canvas 0 · committed at step 20/48"), "B-7: the other summary stays");
  assert.equal(texts(s1)[3], "last");
  const touchedS0 = page.doc.writes.filter(([el]) => s0.contains(el));
  assert.deepEqual(touchedS0.map(([, k]) => k), [], "B-8: a frame of canvas 1 writes nothing into canvas 0's section");
  // opening the closed section shows canvas 0's last drawn tokens (they are there, not re-made)
  s0.setAttribute("open", "");
  page.view.advance(50);
  assert.deepEqual(texts(s0), SHOW_CP, "B-7: a closed section opened shows that canvas's last drawn tokens");
});

pendingT0159B("T-0159 B-7: a section the viewer opened stays open; a newest section the viewer closed is not reopened by its own next frame", async () => {
  const page = await mounted();
  send(page, plain(1, { canvas_idx: 0 }));
  send(page, plain(2, { canvas_idx: 1 }));
  const [s0, s1] = sections(page);
  assert.ok(s0 && s1, "premise: two sections");
  assert.equal(s0.hasAttribute("open"), false, "premise: canvas 0 closed when canvas 1 arrived");
  s0.open = true; // the viewer opens the older section
  send(page, plain(3, { canvas_idx: 2 }));
  const s2 = sections(page)[2];
  assert.ok(s2, "B-7: canvas 2's section is appended");
  assert.equal(sections(page)[0], s0, "B-7: sections stay in arrival order");
  assert.equal(s0.hasAttribute("open"), true, "B-7: older sections keep the open state the viewer left (opened)");
  assert.equal(s1.hasAttribute("open"), false, "B-7: the previous newest closes");
  assert.equal(s2.hasAttribute("open"), true, "B-7: the new canvas's section is open");
  // Reading (T-0159 does not say it; stated in the report): only a NEW canvas opens a section, so
  // the viewer's choice on the newest section survives that canvas's next frames.
  s2.open = false;
  send(page, plain(4, { canvas_idx: 2 }));
  send(page, commit({ canvas_idx: 2, step: 5 }));
  assert.equal(s2.hasAttribute("open"), false, "B-7 (reading): a frame of the same canvas does not reopen it");
  assert.equal(s0.hasAttribute("open"), true, "B-7: the viewer's opened section stays open through it");
  assert.ok(summaryText(s2).includes("canvas 2 · committed at step 5/48"));
});

pendingT0159B("T-0159 B-7 / B6: a new request starts with an empty view; an older request's frames add no section", async () => {
  const page = await mounted();
  send(page, plain(1, { canvas_idx: 0, argmax_pieces: Array(8).fill("old") }));
  send(page, plain(2, { canvas_idx: 1, argmax_pieces: Array(8).fill("old") }));
  const old = [...sections(page)];
  assert.equal(old.length, 2, "premise: two sections of the first request");
  send(page, plain(1, { request_id: OTHER, canvas_idx: 0, argmax_pieces: Array(8).fill("new") }));
  assert.equal(sections(page).length, 1, "B-7: the previous request's sections go");
  for (const s of old) assert.ok(!page.root.contains(s), "B-7: an old section is removed");
  assert.deepEqual(texts(sections(page)[0]), Array(8).fill("new"));
  assert.ok(!page.root.textContent.includes("old"), "B-7: no text of the previous request is left in the view");
  // the first request is now the older one (B6/O-C5): its next canvas adds nothing
  send(page, plain(3, { canvas_idx: 2, argmax_pieces: Array(8).fill("old") }));
  assert.equal(sections(page).length, 1, "B6: an older request's new canvas adds no section");
  assert.ok(!page.root.textContent.includes("old"));
  // and a newer request's canvas 1 is appended after its canvas 0
  send(page, plain(2, { request_id: OTHER, canvas_idx: 1, argmax_pieces: Array(8).fill("new1") }));
  assert.equal(sections(page).length, 2);
  assert.ok(summaryText(sections(page)[1]).includes("canvas 1 · step 2/48"));
});

pendingT0159B("T-0159 B-8 adversarial: only changed cells are updated — F1 -> F2 touches cells 0, 2, 3 and 7 and no other", async () => {
  const page = await mounted();
  send(page, F1P);
  const sec = sections(page)[0];
  assert.ok(sec, "a section was drawn");
  const cells = [...cellsOf(sec)];
  page.doc.writes.length = 0;
  send(page, F2P);
  assert.deepEqual([...cellsOf(sec)], cells, "B-8: the cells are the same elements");
  assert.deepEqual(texts(sec), SHOW_F2P, "premise: F2's text is shown");
  // By hand: F1 -> F2 states [stable,stable,noise,stable,noise,kept,noise,kept] ->
  // [revised,stable,stable,kept,noise,kept,noise,noise]; text changes at p0, p2, p7.
  // Changed (state or text): p0, p2, p3, p7. Unchanged: p1, p4, p5, p6.
  const touched = touchedCells(page, page.doc.writes);
  const touchedIdx = cells.map((c, i) => (touched.has(c) ? i : -1)).filter((i) => i >= 0);
  assert.deepEqual(touchedIdx, [0, 2, 3, 7], "B-8: writes reached exactly the changed cells; log: " +
    JSON.stringify(page.doc.writes.filter(([el]) => sec.contains(el)).slice(0, 12).map(([el, k]) => [el.localName, k])));
});

pendingT0159B("T-0159 B-8: at most one draw per animation frame — three frames in one tick draw once, the latest", async () => {
  const m = await model();
  const page = await mounted();
  page.deliver(F1P);
  page.deliver(F2P);
  page.deliver(CP);
  page.view.advance(50);
  assert.equal(frames(page), 3);
  assert.equal(draws(page), 1, "B-8: one draw for three frames that arrived within one animation frame");
  assert.equal(page.label.textContent, "canvas 0 · committed at step 20/48");
  assert.deepEqual(texts(sections(page)[0]), SHOW_CP, "latest wins");
  assert.equal(page.root.getAttribute("data-denoise-digest"), m.digest(EXPECT_COMMIT_5_OF_8));
});

pendingT0159B("T-0159 B-10: astra.error denoise_no_pieces is stated on the pane's line like denoise_no_tap, while the cells draw; other codes are not", async () => {
  const page = await mounted();
  send(page, F1);
  const before = page.empty.textContent;
  send(page, { type: "astra.error", code: "llm_timeout", message: "the model did not answer in time" });
  assert.equal(page.empty.textContent, before, "B-10 (guard too wide): another code leaves the line alone");
  const message = "denoising text off: the tokenizer could not load (tokenizer.json sha mismatch)";
  send(page, { type: "astra.error", code: "denoise_no_pieces", message });
  send(page, F2);
  assert.ok(page.empty.textContent.includes(message), "B-10: the line states the message; it reads " + JSON.stringify(page.empty.textContent));
  for (let n = page.empty; n && n.nodeType === 1; n = n.parentNode)
    assert.ok(!n.hasAttribute("hidden"), "B-10: the line is not hidden (<" + n.localName + " hidden>)");
  assert.equal(sections(page).length, 1, "B-10: the cells still draw (states from the ids)");
  assert.deepEqual(texts(sections(page)[0]), Array(8).fill(""), "B-2: and still show no text");
});

test("T-0159 B-1 (guard): hud/ holds no tokenizer, no vocabulary table and no SentencePiece space mark", () => {
  // A negative criterion: true at BASE and must stay true, so it is not gated. Prose can trip the
  // '▁' check (a comment quoting the mark): write "U+2581" in prose instead.
  const hud = fileURLToPath(new URL("hud/", REPO));
  const offenders = [];
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      const p = join(dir, name);
      const st = statSync(p);
      if (st.isDirectory()) walk(p);
      else {
        if (/tokeni[sz]er|vocab/i.test(name)) offenders.push(`${p}: name`);
        if (st.size > 1024 * 1024) offenders.push(`${p}: ${st.size} bytes (a vocabulary-sized file)`);
        if (/\.(js|mjs|json|html|css)$/.test(name) && readFileSync(p, "utf8").includes("▁")) offenders.push(`${p}: U+2581`);
      }
    }
  };
  walk(hud);
  assert.deepEqual(offenders, [], "T-0159 B-1: the page reads pieces, never decodes ids");
});

// --- T-0159 B review (T1, T2) ---------------------------------------------------------------------
//
// T1: latest-wins coalescing (B-8) is per drawn frame, but B-7 keeps a section per canvas showing
// that canvas's last tokens. A commit of canvas k and the first frame of canvas k+1 arriving in one
// reduced-motion slot (or one animation frame) must leave both sections: k showing its committed
// pieces under a "committed at" summary. The reviewer's probe: 4 canvases x (9 frames + commit) at
// 30 Hz under reduce drew only sections 0, 1 and 3 and lost every commit but the last.

/** Canvas k of the shown request: denoise frames at steps 1..n with pieces "k:s", then its commit
 *  at step n + 1 with 5 committed pieces " c{k}w{i}" (display "c{k}w{i}"). */
function canvasRun(k, n = 9) {
  const frames = [];
  for (let s = 1; s <= n; s += 1) frames.push(plain(s, { canvas_idx: k, argmax_pieces: Array(8).fill(`${k}:${s}`) }));
  frames.push(commit({ canvas_idx: k, step: n + 1, committed_pieces: [0, 1, 2, 3, 4].map((i) => ` c${k}w${i}`) }));
  return frames;
}
const committedShow = (k) => [...[0, 1, 2, 3, 4].map((i) => `c${k}w${i}`), "", "", ""];

pendingT0159B("T-0159 B review T1: reduced motion, 4 canvases x (9 frames + commit) at 30 Hz — every section exists and shows its commit", async () => {
  const page = await mounted({ reduce: true });
  for (let k = 0; k < 4; k += 1)
    for (const f of canvasRun(k)) {
      page.deliver(f);
      page.view.advance(33);
    }
  page.view.advance(2000);
  const secs = sections(page);
  assert.equal(secs.length, 4, "B-7: one section per canvas, none lost to coalescing; got " + secs.length);
  for (let k = 0; k < 4; k += 1) {
    assert.ok(summaryText(secs[k]).includes(`canvas ${k} · committed at step 10/48`), `B-7: section ${k}'s summary: ${summaryText(secs[k])}`);
    assert.deepEqual(texts(secs[k]), committedShow(k), `B-6/B-7: section ${k} shows its committed pieces`);
    assert.deepEqual(statesIn(secs[k]), EXPECT_COMMIT_5_OF_8, `B-5: section ${k} shows its commit's states`);
  }
  assert.equal(page.label.textContent, "canvas 3 · committed at step 10/48", "the pane label is the newest frame's");
});

pendingT0159B("T-0159 B review T1: a commit of canvas 0 and canvas 1's first frame in one slot (reduce) or one animation frame — both sections kept", async () => {
  for (const reduce of [true, false]) {
    const page = await mounted({ reduce });
    send(page, F1P, 600);
    // the commit of canvas 0 and the first frame of canvas 1 in the same tick
    page.deliver(CP);
    page.deliver(plain(1, { canvas_idx: 1, argmax_pieces: Array(8).fill("next") }));
    page.view.advance(1200);
    const secs = sections(page);
    assert.equal(secs.length, 2, `reduce=${reduce}: two sections; got ${secs.length}`);
    assert.ok(summaryText(secs[0]).includes("canvas 0 · committed at step 20/48"), `reduce=${reduce}: section 0's summary: ${summaryText(secs[0])}`);
    assert.deepEqual(texts(secs[0]), SHOW_CP, `reduce=${reduce}: section 0 shows the committed pieces`);
    assert.deepEqual(statesIn(secs[0]), EXPECT_COMMIT_5_OF_8);
    assert.deepEqual(texts(secs[1]), Array(8).fill("next"), `reduce=${reduce}: section 1 shows its frame`);
    assert.equal(secs[1].hasAttribute("open"), true);
    assert.equal(secs[0].hasAttribute("open"), false);
  }
});

/** True when the notice line shows `message`: its text holds it and neither it nor an ancestor is
 *  `hidden` (node cannot see CSS; the Chrome twin checks visibility). */
const noticeShows = (page, message) => {
  if (!page.empty.textContent.includes(message)) return false;
  for (let n = page.empty; n && n.nodeType === 1; n = n.parentNode) if (n.hasAttribute("hidden")) return false;
  return true;
};

pendingT0159B("T-0159 B review T2: after denoise_no_pieces, frames without pieces keep the notice; a frame with pieces clears or hides it", async () => {
  const page = await mounted();
  const message = "denoising text off: the tokenizer could not load";
  send(page, F1);
  send(page, { type: "astra.error", code: "denoise_no_pieces", message });
  send(page, F2);
  assert.ok(noticeShows(page, message), "B-10: frames without pieces keep the notice");
  send(page, plain(9, { argmax_pieces: Array(8).fill("back") }));
  assert.deepEqual(texts(sections(page)[0]), Array(8).fill("back"), "premise: the pieces are drawn");
  assert.ok(!noticeShows(page, message), "T2: a frame with pieces clears (or hides) the denoise_no_pieces notice — the text is back");
});

// =================================================================================================
// T-0159 group B2 (docs/tasks/T-0159.md B2-1 .. B2-6): the trailing end-of-sequence canvas.
//
// After the answer commits, the engine starts one more canvas whose argmax is `<eos>` at every
// position, then the request ends (live check at 80856fd4). B2 folds that section (closed, summary
// "canvas k · end of sequence") and opens the newest non-end-of-sequence section of the request, once
// per transition; a later frame of that canvas that is not all `<eos>` makes it a normal section
// again (B2-4). The failure modes aimed at: folding too much (a partly-`<eos>` frame, a frame
// without pieces, a commit frame), sticking (B2-4), fighting the viewer (B2-3 "once"), and hiding or
// dropping a frame (B2-5: frames, draws, digest, label and every cell computed exactly as before).
//
// Every row is pending on `pendingT0159B2`; each one contains at least one all-`<eos>` step, so it
// fails on the B page (which opens the new canvas and closes the previous newest, B-7).
// =================================================================================================

const EOS8 = Array(8).fill("<eos>");
/** Canvas 1's all-`<eos>` denoise frame at step s (its states are plainStates(s)). */
const eosFrame = (s, over = {}) => plain(s, { canvas_idx: 1, argmax_pieces: EOS8, ...over });
const EOS_SUMMARY = (k) => `canvas ${k} · end of sequence`;
/** The summary reads the end-of-sequence text *instead of* the B4 label (B2-2). */
function assertEosSummary(section, k, why) {
  const s = summaryText(section);
  assert.ok(s !== null && s.includes(EOS_SUMMARY(k)), `${why}: summary ${JSON.stringify(s)} should read "${EOS_SUMMARY(k)}"`);
  assert.ok(!/\bstep\b/.test(s), `${why}: the B4 label is replaced, not appended: ${JSON.stringify(s)}`);
}
const isOpen = (s) => s.hasAttribute("open");

pendingT0159B2("T-0159 B2-2/B2-3: commit c0, then an all-<eos> c1 frame — c1 is closed with the end-of-sequence summary, c0 is open showing its committed pieces", async () => {
  const page = await mounted();
  send(page, F1P);
  send(page, CP);
  send(page, eosFrame(1));
  const secs = sections(page);
  assert.equal(secs.length, 2, "B2-5: canvas 1's section exists — nothing is dropped");
  const [s0, s1] = secs;
  assertEosSummary(s1, 1, "B2-2");
  assert.equal(isOpen(s1), false, "B2-2: an end-of-sequence section is closed");
  assert.equal(isOpen(s0), true, "B2-3: the newest non-end-of-sequence section (the committed answer) is opened");
  assert.ok(summaryText(s0).includes("canvas 0 · committed at step 20/48"), "B-7: c0 keeps its B4 summary: " + summaryText(s0));
  assert.deepEqual(texts(s0), SHOW_CP, "B2-3: the open section shows c0's committed pieces");
  assert.deepEqual(statesIn(s0), EXPECT_COMMIT_5_OF_8);
  // B2-2: it keeps its cells — the real <eos> cells, with their id-derived states (plain(1): p0 kept).
  assert.deepEqual(texts(s1), EOS8, "B2-2: opening the folded section still shows the real <eos> cells");
  assert.deepEqual(statesIn(s1), ["kept", "noise", "noise", "noise", "noise", "noise", "noise", "noise"], "B2-5: drawn into its own cells");
  assert.equal(page.label.textContent, "canvas 1 · step 1/48", "B2-2: the pane label is still the newest drawn frame's B4 label");
  assert.equal(frames(page), 3, "B2-2/B2-5: every frame is counted");
});

pendingT0159B2("T-0159 B2-1: a partly-<eos> frame is a normal open section; only an all-<eos> frame folds", async () => {
  const page = await mounted();
  send(page, CP);
  // Seven <eos> and one real piece; then near-misses that are not the piece "<eos>".
  for (const [s, pieces] of [
    [1, [...Array(7).fill("<eos>"), "."]],
    [2, [...Array(7).fill("<eos>"), " <eos>"]],
    [3, Array(8).fill("<pad>")],
  ]) {
    send(page, plain(s, { canvas_idx: 1, argmax_pieces: pieces }));
    const [s0, s1] = sections(page);
    assert.ok(summaryText(s1).includes(`canvas 1 · step ${s}/48`), `B2-1: ${JSON.stringify(pieces)} is a normal section (B4 summary): ${summaryText(s1)}`);
    assert.equal(isOpen(s1), true, `B2-1: ${JSON.stringify(pieces)}: the newest section stays open`);
    assert.equal(isOpen(s0), false, `B-7: ${JSON.stringify(pieces)}: the previous newest stays closed`);
  }
  // Contrast: the same canvas, now all <eos>, folds.
  send(page, eosFrame(4));
  const [s0, s1] = sections(page);
  assertEosSummary(s1, 1, "B2-1 (contrast)");
  assert.equal(isOpen(s1), false, "B2-2 (contrast): all <eos> closes it");
  assert.equal(isOpen(s0), true, "B2-3 (contrast): and c0 opens");
});

pendingT0159B2("T-0159 B2-1: a frame without pieces, or a commit frame, is never end of sequence", async () => {
  const page = await mounted();
  send(page, CP);
  send(page, plain(1, { canvas_idx: 1 })); // no pieces
  let [s0, s1] = sections(page);
  assert.ok(summaryText(s1).includes("canvas 1 · step 1/48"), "B2-1: no pieces — a normal section: " + summaryText(s1));
  assert.equal(isOpen(s1), true);
  assert.equal(isOpen(s0), false);
  send(page, eosFrame(2));
  assertEosSummary(sections(page)[1], 1, "premise");
  assert.equal(isOpen(sections(page)[0]), true, "premise: c0 opened by the transition");
  // The newest drawn frame decides: a frame without pieces after the <eos> one is not end of sequence.
  send(page, plain(3, { canvas_idx: 1 }));
  [s0, s1] = sections(page);
  assert.ok(summaryText(s1).includes("canvas 1 · step 3/48"), "B2-1: the newest drawn frame has no pieces — normal again: " + summaryText(s1));
  assert.equal(isOpen(s1), true, "B2-4: open as the newest section");
  assert.equal(isOpen(s0), false, "B2-4: the section opened in B2-3 closes again");
  // A commit frame whose committed pieces are all <eos> is not a denoise frame: normal.
  send(page, commit({ canvas_idx: 2, step: 6, committed_tokens: [1, 1, 1, 1, 1], committed_pieces: Array(5).fill("<eos>") }));
  const s2 = sections(page)[2];
  assert.ok(s2, "a third section for canvas 2");
  assert.ok(summaryText(s2).includes("canvas 2 · committed at step 6/48"), "B2-1: a commit frame is never end of sequence: " + summaryText(s2));
  assert.equal(isOpen(s2), true, "B-7: a normal new canvas is open");
  assert.equal(isOpen(sections(page)[1]), false, "B-7: the previous newest closes");
});

pendingT0159B2("T-0159 B2-4: a later non-<eos> frame of the folded canvas reopens it with the B4 summary and closes c0 again", async () => {
  const page = await mounted();
  send(page, CP);
  send(page, eosFrame(1));
  const [s0, s1] = sections(page);
  assertEosSummary(s1, 1, "premise");
  assert.equal(isOpen(s0), true, "premise: c0 opened");
  send(page, plain(2, { canvas_idx: 1, argmax_pieces: ["Not", " done", " yet", ".", "<eos>", "<eos>", "<eos>", "<eos>"] }));
  assert.ok(summaryText(s1).includes("canvas 1 · step 2/48"), "B2-4: the B4 summary is back: " + summaryText(s1));
  assert.ok(!summaryText(s1).includes("end of sequence"), "B2-4: not sticky");
  assert.equal(isOpen(s1), true, "B2-4: open as the newest section");
  assert.equal(isOpen(s0), false, "B2-4: the section opened in B2-3 is closed again");
  assert.deepEqual(texts(s1), ["Not", "done", "yet", ".", "<eos>", "<eos>", "<eos>", "<eos>"], "B2-5: drawn into its own cells");
  assert.equal(sections(page).length, 2, "B2-4: the same section, no new one");
  // A second transition to all <eos> is a new transition: c0 opens again.
  send(page, eosFrame(3));
  assertEosSummary(s1, 1, "B2-3 (second transition)");
  assert.equal(isOpen(s1), false);
  assert.equal(isOpen(s0), true, "B2-3: once per end-of-sequence transition — this is another one");
});

pendingT0159B2("T-0159 B2-3: a viewer's choice after the transition is kept; older sections keep their state; the newest non-EOS section is the one opened", async () => {
  const page = await mounted();
  send(page, commit({ canvas_idx: 0, step: 10 }));
  send(page, commit({ canvas_idx: 1, step: 20 }));
  const [s0] = sections(page);
  assert.equal(isOpen(s0), false, "premise: c0 closed when c1 arrived");
  send(page, eosFrame(1, { canvas_idx: 2 }));
  let [, s1, s2] = sections(page);
  assertEosSummary(s2, 2, "B2-2");
  assert.equal(isOpen(s1), true, "B2-3: the newest non-end-of-sequence section (c1, not c0) opens");
  assert.equal(isOpen(s0), false, "B2-3: older sections keep their state (closed)");
  assert.equal(isOpen(s2), false);
  // The viewer closes the answer and opens the end-of-sequence section; more <eos> frames follow.
  s1.open = false;
  s2.open = true;
  send(page, eosFrame(2, { canvas_idx: 2 }));
  send(page, eosFrame(3, { canvas_idx: 2 }));
  [, s1, s2] = sections(page);
  assert.equal(isOpen(s1), false, "B2-3: once per transition — the viewer's closing of the answer is kept");
  assert.equal(isOpen(s2), true, "B2-3: the viewer's opening of the end-of-sequence section is kept");
  assertEosSummary(s2, 2, "B2-2: still end of sequence");
  assert.equal(isOpen(s0), false, "older sections untouched");
  // An older section the viewer opened stays open through a transition.
  const page2 = await mounted();
  send(page2, commit({ canvas_idx: 0, step: 10 }));
  send(page2, commit({ canvas_idx: 1, step: 20 }));
  sections(page2)[0].open = true;
  send(page2, eosFrame(1, { canvas_idx: 2 }));
  const [t0, t1, t2] = sections(page2);
  assert.equal(isOpen(t0), true, "B2-3: an older section the viewer opened keeps its state");
  assert.equal(isOpen(t1), true, "B2-3: c1 opened");
  assert.equal(isOpen(t2), false, "B2-2: folded");
});

pendingT0159B2("T-0159 B2-2/B2-5: frames, draws, digest and label equal those of the same frames with ordinary pieces — only open state and one summary differ", async () => {
  const m = await model();
  const run = (pieces) => [F1P, CP, plain(1, { canvas_idx: 1, argmax_pieces: pieces }), plain(2, { canvas_idx: 1, argmax_pieces: pieces })];
  const eos = await mounted();
  for (const f of run(EOS8)) send(eos, f);
  const twin = await mounted();
  for (const f of run(Array(8).fill("word"))) send(twin, f);
  const attrs = (p) => ["data-denoise-frames", "data-denoise-draws", "data-denoise-digest"].map((a) => p.root.getAttribute(a));
  assert.deepEqual(attrs(eos), attrs(twin), "B2-2: frames/draws/digest computed from the frames exactly as before");
  assert.deepEqual(attrs(eos), ["4", "4", m.digest(["kept", "kept", "noise", "noise", "noise", "noise", "noise", "noise"])], "hand count: 4 frames, 4 draws, plain(2)'s states");
  assert.equal(eos.label.textContent, "canvas 1 · step 2/48", "B2-2: the pane label is the newest frame's B4 label");
  assert.equal(eos.label.textContent, twin.label.textContent);
  const [e0, e1] = sections(eos);
  const [w0, w1] = sections(twin);
  assert.deepEqual(statesIn(e0), statesIn(w0), "B2-5: same cells drawn");
  assert.deepEqual(statesIn(e1), statesIn(w1), "B2-5: same cells drawn");
  assert.deepEqual(texts(e0), texts(w0));
  assert.deepEqual(texts(e1), EOS8);
  // The difference B2 makes:
  assertEosSummary(e1, 1, "B2-2");
  assert.deepEqual([isOpen(e0), isOpen(e1)], [true, false], "B2-2/B2-3 in the <eos> run");
  assert.deepEqual([isOpen(w0), isOpen(w1)], [false, true], "B-7 in the ordinary run");
});

// =================================================================================================
// T-0162 group D — the denoising view clears on astra.session.cleared (D-2, clear rows only)
// =================================================================================================
//
// docs/tasks/T-0162.md D-2: "On astra.session.cleared (live or replayed), every panel clears once per
// new resets value: … the denoising view: its sections, its label (back to its empty line) and the
// thinking meter" (the meter is main.js's, from state.js: tests/hud/state.test.mjs). R1: "A stale
// aborted-request canvas … repaints the page after the clear." The view is mounted through the
// panels.js seam, so it hears of the reset the only way it hears of anything: the message on its
// own subscription.
//
// Two adversarial shapes of R1 at the page: a frame received before the clear but still waiting for
// its animation frame (painted after the clear unless the clear drops it), and a late frame of a
// request from before the reset. The server drops the second (group A ruling (a)); the page must not
// repaint it either. The guard's other side: a new request after the clear draws as usual, and a
// repeated `resets` value (the hello's replay) clears nothing.
//
// Strict xfail until docs/tasks/T-0162.md's status records "group D: tests …, activated <sha>".

const T0162_TASK = fileURLToPath(new URL("docs/tasks/T-0162.md", REPO));
const T0162_STATUS = existsSync(T0162_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0162_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0162_D_PENDING =
  !/\bgroup D:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0162_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0162D(name, fn) {
  test(name + (T0162_D_PENDING ? " [pending T-0162 D: strict xfail until group D is activated]" : ""), async (t) => {
    if (!T0162_D_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail("XPASS(strict): this T-0162 group D row passed while docs/tasks/T-0162.md's status does not record group D's activation.");
    }
    t.diagnostic("expected failure (T-0162 D pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

const EMPTY_COPY = "no denoising data yet — the model's real canvases appear here while it thinks";
const sessionCleared = (resets, extra = {}) => ({ type: "astra.session.cleared", resets, ended_turn_id: null, memory_forgotten: 0, replay: false, ...extra });
const NEW_REQUEST = "chatcmpl-4444444444444444-0000000d";

/** The view as it stands after a pre-reset turn: two canvases of one request drawn. */
async function drawnBeforeReset() {
  const page = await mounted();
  send(page, F1);
  send(page, plain(3, { canvas_idx: 1 }));
  assert.equal(sections(page).length, 2, "premise: two canvases drawn before the reset");
  assert.equal(page.label.textContent, "canvas 1 · step 3/48", "premise: the label names the newest frame");
  return page;
}

pendingT0162D("T-0162 D-2: the cleared message empties the view — no section, the label empty, the empty line's own copy", async () => {
  const page = await drawnBeforeReset();
  const old = sections(page);
  send(page, sessionCleared(1, { ended_turn_id: "t4" }));
  assert.equal(page.root.childElementCount, 0, `D-2: the sections are removed from the root; ${page.root.childElementCount} remain`);
  assert.ok(old.every((s) => !page.root.contains(s)), "…removed, not hidden in it");
  assert.equal(page.label.textContent, "", "D-2: the label is back to its empty state (as on a fresh page, T-0128c B7)");
  assert.equal(page.empty.textContent, EMPTY_COPY, "D-2: the empty line reads its own copy again");
  page.view.advance(30000);
  assert.equal(page.root.childElementCount, 0, "…and nothing is drawn back while nothing arrives");
});

pendingT0162D("T-0162 D-2: a notice on the empty line (a refused synthetic canvas) is cleared back to the empty line's copy", async () => {
  const page = await drawnBeforeReset();
  send(page, { type: "astra.canvas.synthetic", reason: "no_tap" });
  assert.notEqual(page.empty.textContent, EMPTY_COPY, "premise: the notice replaced the empty line's copy");
  send(page, sessionCleared(1));
  assert.equal(page.empty.textContent, EMPTY_COPY, "D-2: the line is back to its empty copy");
  assert.equal(page.empty.hasAttribute("data-denoise-notice"), false, "…and no longer marked as a notice");
});

pendingT0162D("T-0162 D-2 / R1 (adversarial): a frame received before the clear but not yet painted is never painted after it", async () => {
  const page = await drawnBeforeReset();
  const n = draws(page);
  page.deliver(F2); // same request, canvas 0: waits for the next animation frame
  page.deliver(plain(5, { canvas_idx: 2 })); // a new canvas of the same request, also waiting
  page.deliver(sessionCleared(1, { ended_turn_id: "t4" })); // the reset's echo arrives before the paint
  page.view.advance(1000);
  assert.equal(page.root.childElementCount, 0, `R1: a frame of the aborted request was painted after the clear (${page.root.childElementCount} sections)`);
  assert.equal(page.label.textContent, "", "R1: the label does not come back with it");
  assert.equal(draws(page), n, "R1: no draw happens after the clear");
});

pendingT0162D("T-0162 D-2 / R1 (adversarial): a late frame of a pre-reset request is not drawn after the clear", async () => {
  const page = await drawnBeforeReset();
  send(page, sessionCleared(1, { ended_turn_id: "t4" }));
  send(page, plain(9, { canvas_idx: 1 })); // the aborted request (the default request_id), one more step
  send(page, plain(10, { canvas_idx: 3 })); // …and a canvas of it the page never drew
  assert.equal(page.root.childElementCount, 0, "R1: the aborted request's frames never reappear");
  assert.equal(page.label.textContent, "", "R1: …nor its label");
});

pendingT0162D("T-0162 D-2 (guard, too wide): after the clear a new request draws as usual, and the hello's replayed cleared keeps it", async () => {
  const page = await drawnBeforeReset();
  send(page, sessionCleared(1));
  send(page, plain(1, { request_id: NEW_REQUEST, canvas_idx: 0 }));
  assert.equal(sections(page).length, 1, "the new session's first canvas is drawn: one section");
  assert.equal(page.label.textContent, "canvas 0 · step 1/48", "…with its own label");
  const sec = sections(page)[0];
  send(page, sessionCleared(1, { replay: true })); // a reconnect's hello: the same reset, replayed
  send(page, sessionCleared(1)); // and a duplicate live copy
  assert.equal(sections(page)[0], sec, "D-2: a repeated resets value clears nothing — the same section stays");
  assert.equal(page.label.textContent, "canvas 0 · step 1/48");
  send(page, plain(2, { request_id: NEW_REQUEST, canvas_idx: 0 }));
  assert.equal(page.label.textContent, "canvas 0 · step 2/48", "…and the new request keeps drawing");
  // the other side again: a second Restart (resets 2) clears the second session's view
  send(page, sessionCleared(2));
  assert.equal(page.root.childElementCount, 0, "D-2: resets 2 is new, so the view clears again");
});
