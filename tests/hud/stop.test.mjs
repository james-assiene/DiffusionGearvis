// T-0193 group B — the HUD's "Interrupt Gearvis" button, the node half (docs/tasks/T-0193.md B-1..B-5,
// rulings H-T193-3: no keyboard shortcut; H-T193-4: the label "Interrupt Gearvis", the title "Interrupt
// Gearvis: it stops talking and stops what it is doing in the browser", the id stays #stop-button, placed
// right after #turn-chip). The Chrome half (real clicks and keys, overflow, contrast, loopback) is
// tests/integration/test_t0193_hud_stop.py.
//
// Every expected string and every expected message below is written by hand from the task's criteria and
// H-T193-4, never read from an implementation: hud/src/stop.js does not exist when this file is written.
//
// Gate. Every row runs through `pendingT0193B`, the Status-phrase strict-xfail gate of
// tests/hud/mic.test.mjs, reading docs/tasks/T-0193.md. While its **Status:** line does not record
// "group B: tests <a>..<b>, activated <sha>", each row must FAIL (an XPASS fails the row: a stub that
// satisfies it is vacuous). ASTRA_RUN_PENDING=1 runs them for real now. Group A0's and A's wording
// ("Group A tests activated at e4b5e83a", "Group B (the "Interrupt Gearvis" button) next") does not
// switch these rows on.
//
// API these rows pin (the task names the file, `hud/src/stop.js`, "DOM-free (the mic.js shape)",
// "accepts astra.turn.state and astra.session.cleared; view(), click()"; the JS names and the reconnect
// seam are the test-author's):
//
//   hud/src/stop.js — importable in node, no DOM at module top level.
//     export const STOP_CLIENT_TYPES = ["astra.turn.stop"]   (the MIC_CLIENT_TYPES convention, B-5)
//     export function createStop() -> {
//       accept(msg)        every decoded astra.* message, in wire order; never throws. Returns true for an
//                          astra.turn.state it took as the echo (re-render the button); false for any
//                          message of another type than astra.turn.state / astra.session.cleared.
//       connection(open)   main.js reports the socket on every change (the host.onConnection signal the
//                          Restart button already uses). open === false forgets the echo: the button is
//                          disabled, and the next socket's hello reply is the only source of its turn_id
//                          (reviewer A's note on group A). Return value unpinned.
//       view()             null before the first astra.turn.state (B-1: hidden); else
//                          { text, title, disabled }
//                            text      "Interrupt Gearvis" in every rendered state (H-T193-4)
//                            title     the H-T193-4 title in every rendered state
//                            disabled  true unless the last echo is THINKING, SPEAKING or PAUSED and no
//                                      click has been sent since that echo
//       click()            { type: "astra.turn.stop", turn_id: <the last echoed turn_id> } while enabled,
//                          and then null until the next accepted astra.turn.state (B-2); null while
//                          disabled or before a state.
//     }
//     It has no keyboard entry point at all (B-4, H-T193-3).
//
// Readings the task leaves open (stated, not guessed silently):
//   R1 "stale" (B-3) is state.js's rule: an astra.turn.state whose turn_id this page has already taken and
//      that is not the turn of the last echo moves nothing, live or replayed. A replay for the current
//      turn is an echo like any other (the hello reply's closing copy of the current state).
//   R2 After a click, the next accepted astra.turn.state re-enables the button if its state is THINKING,
//      SPEAKING or PAUSED — also for the same turn (the server's echo of a stop that did not act, or a
//      PAUSED→SPEAKING resume). B-2: "disabled until the next astra.turn.state".
//   R3 astra.session.cleared (live or replayed) ends the turn on screen: the button stays rendered and is
//      disabled until a turn.state of a turn other than the reset's (ids never repeat after a reset,
//      T-0162); a state for the ended turn after the reset is stale. Before any turn.state a cleared shows
//      nothing.
//   R4 A server restart reuses turn ids (a fresh MachineState starts at "t0"), so the stale rule's memory
//      is per socket: after connection(false), the new server's "t1" is a new turn even if the old
//      server's "t1" was seen; until its first astra.turn.state the button is disabled and click() is null.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);

// --- the T-0193 group B strict-xfail gate --------------------------------------------------------

const TASK = fileURLToPath(new URL("docs/tasks/T-0193.md", REPO));
const STATUS = existsSync(TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(TASK, "utf8")) || [])[1] || "").trim()
  : "";
/** "group B: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6" — the phrase the orchestrator writes. */
const B_ACTIVATED = /\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const B_PENDING = !B_ACTIVATED.test(STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0193B(name, fn) {
  test(name + (B_PENDING ? " [pending T-0193 B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0193 group B row passed while docs/tasks/T-0193.md's status does " +
          "not record group B's activation (" + JSON.stringify(STATUS.slice(0, 80)) + "). Either " +
          "group B has landed (record 'group B: tests …, activated <sha>' in the status) or the " +
          "row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0193 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0193 B gate: the status line is readable and only group B's own phrase activates the rows", () => {
  assert.ok(STATUS.length > 0, "docs/tasks/T-0193.md must carry a **Status:** line");
  // Hand-written samples: the first three are this task's wording for groups A0 and A and for B's plan.
  assert.equal(B_ACTIVATED.test("Group A tests activated at e4b5e83a (`tests/contract/test_t0193_stop_wire.py` 23 rows"), false);
  assert.equal(B_ACTIVATED.test("Earlier: group A0 tests activated at cfb29d19 (`tests/unit/test_t0193_stop_row.py`"), false);
  assert.equal(B_ACTIVATED.test('Group B (the "Interrupt Gearvis" button) next. Follow-up (reviewer B F2'), false);
  assert.equal(B_ACTIVATED.test("in progress (group A: tests 1a2b3c4, activated 9a8b7c6; group B: tests planned)"), false);
  assert.equal(B_ACTIVATED.test("in progress (group B: tests a1b2c3d..e4f5a6b, activated c3d4e5f)"), true);
  assert.equal(B_ACTIVATED.test("in progress. group B: tests a1b2c3d, activated c3d4e5f; nothing else"), true);
});

// --- loading the modules --------------------------------------------------------------------------

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
console.log(`# stop origin: repo=${fileURLToPath(REPO)} modules=${srcRoot().href}`);

async function load(name) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch (err) {
    assert.fail(
      `T-0193 B: hud/src/${name} must be importable in node as an ES module with no DOM at module ` +
        `top level. node could not import it: ${err.message}`,
    );
  }
}

/** The module's code with comments removed (a source guard must not match prose; strings are kept). */
function codeOf(rel) {
  const src = readFileSync(fileURLToPath(new URL(rel, REPO)), "utf8");
  return src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:\\])\/\/.*$/gm, "$1");
}

/** A wire message through the page's real decoder (hud/src/protocol.js), as main.js hands it on. */
async function wire(obj) {
  const { decode } = await load("protocol.js");
  const got = decode(JSON.stringify(obj));
  assert.equal(got.kind, "astra", `premise: ${JSON.stringify(obj)} decodes as a typed message; got ${JSON.stringify(got)}`);
  return got.msg;
}

async function stopper() {
  const mod = await load("stop.js");
  assert.equal(typeof mod.createStop, "function", "hud/src/stop.js exports createStop()");
  return mod.createStop();
}

// --- the expected copy and states, by hand from H-T193-4 and B-1 ---------------------------------

const TEXT = "Interrupt Gearvis";
const TITLE = "Interrupt Gearvis: it stops talking and stops what it is doing in the browser";
const ALL_STATES = ["IDLE", "LISTENING", "ENDPOINTING", "THINKING", "SPEAKING", "PAUSED", "INTERRUPTED"];
/** B-1: enabled iff the last echoed state is one of these (H-T193-5: not while James is speaking). */
const ENABLED = new Set(["THINKING", "SPEAKING", "PAUSED"]);

const ts = (state, turn_id, extra = {}) => ({
  type: "astra.turn.state",
  state,
  turn_id,
  thinking_ms: state === "THINKING" ? 120 : null,
  ...extra,
});
const cleared = (resets, ended_turn_id, extra = {}) => ({
  type: "astra.session.cleared",
  resets,
  ended_turn_id,
  memory_forgotten: 0,
  ...extra,
});
const stopFor = (turn_id) => ({ type: "astra.turn.stop", turn_id });

async function feed(s, ...objs) {
  for (const obj of objs) s.accept(await wire(obj));
}

function assertRendered(view, disabled, why) {
  assert.ok(view, `${why}: the button is rendered (view() is not null) once an astra.turn.state arrived`);
  assert.equal(view.text, TEXT, `${why}: H-T193-4 label`);
  assert.equal(view.title, TITLE, `${why}: H-T193-4 title`);
  assert.equal(view.disabled, disabled, `${why}: disabled is ${disabled}`);
}

// --- B-5: the module ------------------------------------------------------------------------------

pendingT0193B("T-0193 B-5: stop.js imports in node with no DOM and names its one client type, astra.turn.stop", async () => {
  assert.equal(typeof globalThis.document, "undefined", "premise: node has no document");
  const mod = await load("stop.js");
  assert.deepEqual(mod.STOP_CLIENT_TYPES, ["astra.turn.stop"], "B-5: the only new message the page sends is astra.turn.stop");
  assert.equal(typeof mod.createStop, "function", "createStop() is exported");
});

pendingT0193B("T-0193 B-5: the click's message passes the page's own astra.turn.stop field table (group A's mirror)", async () => {
  const { decode } = await load("protocol.js");
  const s = await stopper();
  await feed(s, ts("SPEAKING", "t4"));
  const msg = s.click();
  assert.deepEqual(msg, stopFor("t4"), "premise: the click's message");
  const got = decode(JSON.stringify(msg));
  assert.equal(got.kind, "astra", `B-5/A-1: the sent message is a well-formed astra.turn.stop (no extra field): ${JSON.stringify(got)}`);
  assert.equal(got.msg.type, "astra.turn.stop");
});

// --- B-1: visibility --------------------------------------------------------------------------------

pendingT0193B("T-0193 B-1: hidden before the first astra.turn.state, whatever else arrives; a click sends nothing", async () => {
  const s = await stopper();
  assert.equal(s.view(), null, "B-1: hidden before any astra.turn.state");
  assert.equal(s.click(), null, "B-2: nothing to stop before a state");
  const others = [
    { type: "astra.mic.state", muted: false, available: true },
    { type: "astra.transcript", turn_id: "t1", text: "how many", final: false },
    { type: "astra.spoken", turn_id: "t1", utterance_id: "t1-u0", text: "Three." },
    cleared(1, "t1"),
    cleared(2, null, { replay: true }),
    { type: "astra.error", code: "x", message: "y" },
  ];
  for (const raw of others) {
    const msg = await wire(raw);
    assert.doesNotThrow(() => s.accept(msg), `accept(${raw.type}) must not throw`);
    assert.equal(s.view(), null, `B-1: still hidden after ${raw.type}`);
    assert.equal(s.click(), null, `B-2: still nothing to send after ${raw.type}`);
  }
  assert.equal(s.accept(await wire(others[0])), false, "accept() is false for a message that is not the button's");
  assert.equal(s.accept(await wire(ts("LISTENING", "t1"))), true, "accept() is true for the first astra.turn.state");
  assertRendered(s.view(), true, "B-1: rendered after the first state (LISTENING: disabled)");
});

for (const state of ALL_STATES) {
  pendingT0193B(`T-0193 B-1: as the first state, ${state} renders the button ${ENABLED.has(state) ? "enabled" : "disabled"}, with the H-T193-4 label and title`, async () => {
    const s = await stopper();
    await feed(s, ts(state, "t3"));
    assertRendered(s.view(), !ENABLED.has(state), `first state ${state}`);
  });
}

pendingT0193B("T-0193 B-1: enabled iff the LAST echo is THINKING, SPEAKING or PAUSED, over every ordered pair of the seven states", async () => {
  // Every (from, to) pair within one turn, and across a turn boundary: the view follows the second echo.
  for (const from of ALL_STATES) {
    for (const to of ALL_STATES) {
      for (const sameTurn of [true, false]) {
        const s = await stopper();
        await feed(s, ts(from, "t5"), ts(to, sameTurn ? "t5" : "t6"));
        const why = `${from} (t5) then ${to} (${sameTurn ? "t5" : "t6"})`;
        assertRendered(s.view(), !ENABLED.has(to), why);
        if (ENABLED.has(to)) {
          assert.deepEqual(s.click(), stopFor(sameTurn ? "t5" : "t6"), `${why}: the click names the echoed turn`);
        } else {
          assert.equal(s.click(), null, `${why}: a disabled button sends nothing`);
        }
      }
    }
  }
});

// --- B-2: the click -----------------------------------------------------------------------------------

for (const state of ["THINKING", "SPEAKING", "PAUSED"]) {
  pendingT0193B(`T-0193 B-2: in ${state}, one click sends exactly one stop for the echoed turn, then nothing until the next echo`, async () => {
    const s = await stopper();
    await feed(s, ts("LISTENING", "t2"), ts(state, "t2"));
    assertRendered(s.view(), false, `premise: ${state} is enabled`);
    assert.deepEqual(s.click(), stopFor("t2"), `B-2: the click sends {type: astra.turn.stop, turn_id: "t2"}`);
    assertRendered(s.view(), true, "B-2: after the click the button is disabled (the echo is withheld)");
    // Adversarial: the stub withholds the echo; further clicks (a double click, an impatient user) send nothing.
    for (let i = 0; i < 5; i += 1) {
      assert.equal(s.click(), null, `B-2: click ${i + 2} with the echo withheld sends nothing`);
    }
    assertRendered(s.view(), true, "B-2: still disabled with no echo");
  });
}

pendingT0193B("T-0193 B-2: the next echo moves it, including the same turn's echo (R2), and the click then names that echo's turn", async () => {
  const s = await stopper();
  await feed(s, ts("SPEAKING", "t2"));
  assert.deepEqual(s.click(), stopFor("t2"));
  // The server's echo of the stop: INTERRUPTED then IDLE; neither enables.
  await feed(s, ts("INTERRUPTED", "t2"));
  assertRendered(s.view(), true, "INTERRUPTED echo");
  assert.equal(s.click(), null);
  await feed(s, ts("IDLE", "t2"));
  assertRendered(s.view(), true, "IDLE echo");
  // A new turn starts and speaks: enabled again, and the click names the new turn.
  await feed(s, ts("LISTENING", "t3"), ts("ENDPOINTING", "t3"));
  assertRendered(s.view(), true, "James is still speaking (H-T193-5)");
  await feed(s, ts("THINKING", "t3"));
  assertRendered(s.view(), false, "THINKING t3");
  assert.deepEqual(s.click(), stopFor("t3"), "B-2: the click names the last echoed turn, t3");
  // R2: a stop the server did not act on still re-arms the button on its next echo of an enabled state.
  await feed(s, ts("SPEAKING", "t3"));
  assertRendered(s.view(), false, "R2: the next astra.turn.state (same turn, SPEAKING) re-enables it");
  assert.deepEqual(s.click(), stopFor("t3"));
  await feed(s, ts("PAUSED", "t3"));
  assertRendered(s.view(), false, "R2: PAUSED after a click re-enables it");
  assert.deepEqual(s.click(), stopFor("t3"));
});

pendingT0193B("T-0193 B-2: a click while disabled sends nothing and changes nothing (the later echo still arms it)", async () => {
  for (const state of ["IDLE", "LISTENING", "ENDPOINTING", "INTERRUPTED"]) {
    const s = await stopper();
    await feed(s, ts(state, "t1"));
    for (let i = 0; i < 3; i += 1) assert.equal(s.click(), null, `${state}: a click on a disabled button sends nothing`);
    await feed(s, ts("THINKING", "t1"));
    assertRendered(s.view(), false, `${state} → THINKING: the disabled clicks left nothing behind`);
    assert.deepEqual(s.click(), stopFor("t1"));
  }
});

pendingT0193B("T-0193 B-2: the turn_id is the echo's, verbatim (any non-empty string the server sends, A-R3)", async () => {
  for (const id of ["t1", "t12345", " ", "turn-α/β \"q\""]) {
    const s = await stopper();
    await feed(s, ts("SPEAKING", id));
    assert.deepEqual(s.click(), stopFor(id), `the click carries turn_id ${JSON.stringify(id)} as echoed`);
  }
});

// --- B-3: only the echo moves it; stale and replayed states -----------------------------------------

pendingT0193B("T-0193 B-3: a stale state for an earlier turn (live) moves nothing: no re-enable, no turn_id change", async () => {
  const s = await stopper();
  await feed(s, ts("SPEAKING", "t1"), ts("IDLE", "t1"), ts("THINKING", "t2"));
  assert.deepEqual(s.click(), stopFor("t2"), "premise: the button is armed for t2 and fires");
  // Adversarial: a late echo of the earlier turn arrives while the t2 click is unanswered.
  for (const state of ["SPEAKING", "THINKING", "PAUSED"]) {
    await feed(s, ts(state, "t1"));
    assertRendered(s.view(), true, `B-3: a late ${state} for t1 does not re-enable the button`);
    assert.equal(s.click(), null, `B-3: … and a click sends no stop (not for t1, not for t2)`);
  }
  await feed(s, ts("SPEAKING", "t2"));
  assertRendered(s.view(), false, "the current turn's echo re-arms it");
  assert.deepEqual(s.click(), stopFor("t2"), "B-3: the turn_id is still t2, never the stale t1");
});

pendingT0193B("T-0193 B-3: a stale state for an earlier turn does not disable or retarget an armed button either", async () => {
  const s = await stopper();
  await feed(s, ts("SPEAKING", "t1"), ts("IDLE", "t1"), ts("SPEAKING", "t2"));
  await feed(s, ts("IDLE", "t1"), ts("INTERRUPTED", "t1"));
  assertRendered(s.view(), false, "B-3: the button follows t2's SPEAKING, not a late t1 IDLE");
  assert.deepEqual(s.click(), stopFor("t2"));
});

pendingT0193B("T-0193 B-3: a replayed astra.turn.state for an earlier turn re-enables nothing", async () => {
  const s = await stopper();
  await feed(s, ts("SPEAKING", "t1"), ts("IDLE", "t1"), ts("IDLE", "t2"));
  await feed(s, ts("SPEAKING", "t1", { replay: true }), ts("THINKING", "t1", { replay: true }));
  assertRendered(s.view(), true, "B-3: replays for t1 leave the t2 IDLE echo in force");
  assert.equal(s.click(), null, "B-3: no stop for the replayed earlier turn");
});

pendingT0193B("T-0193 B-3: a fresh page's hello reply arms the button from the open turn (live state, then its replays)", async () => {
  // hud_server.hello_reply order: [cleared replay], assist, turn.state (live), log, mic, journal…, the
  // INTERRUPTED copy if any, turn.state again (replay: true).
  const s = await stopper();
  await feed(
    s,
    cleared(1, "t3", { replay: true }),
    { type: "astra.assist.state", flags: {}, badge: "pixels" },
    ts("SPEAKING", "t4"),
    { type: "astra.mic.state", muted: false, available: true },
    { type: "astra.transcript", turn_id: "t4", text: "what's on screen", final: true },
    ts("SPEAKING", "t4", { replay: true }),
  );
  assertRendered(s.view(), false, "the hello reply of a speaking turn arms the button");
  assert.deepEqual(s.click(), stopFor("t4"));
  // The hello reply after a stop: live IDLE, the INTERRUPTED copy, the closing IDLE copy.
  const after = await stopper();
  await feed(after, ts("IDLE", "t4"), ts("INTERRUPTED", "t4", { replay: true }), ts("IDLE", "t4", { replay: true }));
  assertRendered(after.view(), true, "the hello reply of a stopped turn leaves it disabled");
  assert.equal(after.click(), null);
});

// --- reconnect (reviewer A's note on group A, R4) -----------------------------------------------------

pendingT0193B("T-0193 B-3 (reconnect): a dropped socket disarms the button; only the new hello's state re-arms it", async () => {
  const s = await stopper();
  await feed(s, ts("SPEAKING", "t1"), ts("IDLE", "t1"), ts("THINKING", "t2"), ts("IDLE", "t2"), ts("SPEAKING", "t3"));
  assertRendered(s.view(), false, "premise: armed for t3");
  s.connection(false);
  assertRendered(s.view(), true, "R4: with the socket closed the button stays rendered and is disabled");
  assert.equal(s.click(), null, "R4: no stop for the old server's t3");
  s.connection(true);
  assertRendered(s.view(), true, "R4: a new socket alone does not re-arm it (no hello reply yet)");
  assert.equal(s.click(), null, "R4 (adversarial): a click between the reconnect and the hello reply sends no stale t3");
  // A restarted server reuses ids: its "t1" is a new turn, though the old server's t1 was seen.
  await feed(s, ts("SPEAKING", "t1"));
  assertRendered(s.view(), false, "R4: the new server's live SPEAKING t1 arms the button");
  assert.deepEqual(s.click(), stopFor("t1"), "R4: the turn_id is the new server's, from its hello reply");
});

pendingT0193B("T-0193 B-3 (reconnect): a new server that is idle leaves the button disabled", async () => {
  const s = await stopper();
  await feed(s, ts("SPEAKING", "t3"));
  s.connection(false);
  s.connection(true);
  await feed(s, ts("IDLE", "t0"), { type: "astra.mic.state", muted: false, available: true });
  assertRendered(s.view(), true, "the new server's IDLE");
  assert.equal(s.click(), null);
  await feed(s, ts("THINKING", "t1"));
  assert.deepEqual(s.click(), stopFor("t1"), "the new server's next turn arms it");
});

pendingT0193B("T-0193 B-1: a dropped socket before any state shows nothing", async () => {
  const s = await stopper();
  s.connection(false);
  s.connection(true);
  assert.equal(s.view(), null, "B-1: still hidden: no astra.turn.state yet");
  assert.equal(s.click(), null);
});

// --- astra.session.cleared (R3) ----------------------------------------------------------------------

pendingT0193B("T-0193 B (R3): a live astra.session.cleared disarms the button; a state for the ended turn is stale; a new turn re-arms it", async () => {
  const s = await stopper();
  await feed(s, ts("SPEAKING", "t2"));
  await feed(s, cleared(1, "t2"));
  assertRendered(s.view(), true, "R3: the reset ended t2: rendered, disabled");
  assert.equal(s.click(), null, "R3: no stop for the turn the reset cut");
  await feed(s, ts("SPEAKING", "t2"));
  assertRendered(s.view(), true, "R3: a late SPEAKING for the reset's ended turn re-arms nothing");
  assert.equal(s.click(), null);
  await feed(s, ts("LISTENING", "t3"), ts("THINKING", "t3"));
  assertRendered(s.view(), false, "R3: the next turn's THINKING arms it");
  assert.deepEqual(s.click(), stopFor("t3"));
});

pendingT0193B("T-0193 B (R3): a cleared after a click does not re-arm the button", async () => {
  const s = await stopper();
  await feed(s, ts("THINKING", "t5"));
  assert.deepEqual(s.click(), stopFor("t5"));
  await feed(s, cleared(1, "t5"));
  assert.equal(s.click(), null, "R3: still nothing to send");
  assertRendered(s.view(), true, "R3: disabled");
});

pendingT0193B("T-0193 B (R3): a replayed cleared at the head of a hello reply is followed by the live state, which wins", async () => {
  const s = await stopper();
  await feed(s, ts("SPEAKING", "t2"));
  s.connection(false);
  s.connection(true);
  await feed(s, cleared(1, "t2", { replay: true }), ts("THINKING", "t3"), ts("THINKING", "t3", { replay: true }));
  assertRendered(s.view(), false, "R3: the hello reply's live THINKING t3 arms it");
  assert.deepEqual(s.click(), stopFor("t3"));
});

pendingT0193B("T-0193 B: accept() never throws on anything the page may hand it", async () => {
  const s = await stopper();
  for (const junk of [undefined, null, 0, "astra.turn.state", {}, { type: 7 }, [], { type: "astra.turn.state" }]) {
    assert.doesNotThrow(() => s.accept(junk), `accept(${JSON.stringify(junk)}) must not throw`);
  }
});

// --- B-4: no shortcut (H-T193-3, H-T181-4) ------------------------------------------------------------

pendingT0193B("T-0193 B-4: stop.js has no keyboard entry point (no key event, no event.key, nothing shortcut-shaped)", async () => {
  const path = "hud/src/stop.js";
  assert.ok(existsSync(fileURLToPath(new URL(path, REPO))), `B: ${path} exists (Owned files: new, DOM-free)`);
  const code = codeOf(path);
  assert.ok(code.length > 200, "anti-vacuity: stop.js was read");
  assert.ok(!/\bkey(down|up|press)\b/i.test(code), "B-4: stop.js handles no key event; only the button stops a turn");
  assert.ok(!/\.key\b|\bkeyCode\b|\.code\b|Escape/.test(code), "B-4: stop.js reads no event.key / keyCode / code and names no Escape");
  assert.ok(!/aria-keyshortcuts|accesskey/i.test(code), "B-4: no announced shortcut");
  const mod = await load("stop.js");
  const named = Object.keys(mod).filter((k) => /key|shortcut/i.test(k));
  assert.deepEqual(named, [], "B-4: stop.js exports nothing key- or shortcut-shaped");
  const s = mod.createStop();
  const methods = Object.keys(s).filter((k) => /key|shortcut|escape/i.test(k));
  assert.deepEqual(methods, [], "B-4: createStop() offers no key entry point");
});

// --- the shell: the button right after the turn chip (H-T193-4) ---------------------------------------

pendingT0193B("T-0193 B-1 (H-T193-4): <button id=\"stop-button\" type=\"button\" hidden> sits in the header right after #turn-chip", () => {
  const html = readFileSync(fileURLToPath(new URL("hud/index.html", REPO)), "utf8");
  const header = (/<header\b[\s\S]*?<\/header>/.exec(html) || [""])[0];
  assert.ok(header.includes('id="turn-chip"'), "premise: the header holds #turn-chip");
  const ids = [...html.matchAll(/\sid="stop-button"/g)];
  assert.equal(ids.length, 1, "B-1: exactly one #stop-button in the shell");
  // The next element after the chip's closing tag (comments between them are allowed) is the button.
  const after = /<span\b[^>]*\bid="turn-chip"[^>]*>[^<]*<\/span>((?:\s|<!--[\s\S]*?-->)*)<([a-z]+)\b([^>]*)>/.exec(header);
  assert.ok(after, "premise: the chip is a <span> followed by an element");
  assert.equal(after[2], "button", `H-T193-4: the element right after #turn-chip is a <button>; got <${after[2]}>`);
  const attrs = after[3];
  assert.match(attrs, /\bid="stop-button"/, "H-T193-4: the element right after #turn-chip is #stop-button");
  assert.match(attrs, /\btype="button"/, "a type=button (never a form submit)");
  assert.match(attrs, /(^|\s)hidden(\s|=|$)/, "B-1: shown only after an astra.turn.state arrives, so the shell ships it hidden");
  assert.doesNotMatch(attrs, /aria-keyshortcuts|accesskey/i, "B-4: no announced shortcut");
});
