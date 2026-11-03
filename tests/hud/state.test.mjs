// T-0008 criteria 8, 9 and 11 — hud/src/state.js, the reducer behind the two channels, the turn
// chip and the thinking meter. Pure: no DOM, no clock of its own (the page clock is an argument).
//
// API this pins:
//   initialState() -> state
//   reduce(state, message, nowMs) -> state      (message = a decoded astra.* object)
//   thinkingMs(state, nowMs) -> number | null   (null unless the turn state is THINKING)
//   spokenGroups(state) -> [{ turn_id, utterances: [{ utterance_id, text }] }]
//   transcriptLines(state) -> [{ text, final, ... }]   (T-0037 criterion 11 adds turn_id)
//   state.turnState, state.turnId, state.interrupted (array of turn ids)
//   THINKING_MARKS_MS = [3000, 4000]            (docs/01 D11: budget mark and interim-update mark)
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";
import { join } from "node:path";

const REPO = new URL("../../", import.meta.url);

// node 18 treats a bare .js file as CommonJS unless a package.json in scope says
// {"type":"module"}, and T-0008 creates no package.json under hud/ (T-0010 owns hud/package.json,
// per the split in docs/tasks/T-0008.md). In the browser the MIME type decides and the very same
// files are plain ES modules; so to import them here they are copied once into a temp directory
// that does carry {"type":"module"}. If hud/ ever gains that package.json this shim stays correct.
let ESM_ROOT = null;
function srcRoot() {
  if (ESM_ROOT) return ESM_ROOT;
  const real = fileURLToPath(new URL("../../hud/src", import.meta.url));
  const dir = mkdtempSync(join(tmpdir(), "astra-hud-esm-"));
  writeFileSync(join(dir, "package.json"), '{"type":"module","private":true}');
  if (existsSync(real)) cpSync(real, join(dir, "src"), { recursive: true });
  ESM_ROOT = pathToFileURL(join(dir, "src") + "/");
  return ESM_ROOT;
}

async function load(name, criterion) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch (err) {
    assert.fail(
      `T-0008 ${criterion} adds hud/src/${name}, importable in node as an ES module with no DOM ` +
        `at module top level (docs/tasks/T-0008.md implementation notes: "keep DOM out of ` +
        `state.js, protocol.js, frames.js, phash.js, ws.js"). node could not import it: ` +
        `${err.message}`,
    );
  }
}

const turn = (state, turn_id, thinking_ms = null) => ({
  type: "astra.turn.state",
  state,
  turn_id,
  ...(thinking_ms === null ? {} : { thinking_ms }),
});
const spoken = (turn_id, utterance_id, text) => ({
  type: "astra.spoken",
  turn_id,
  utterance_id,
  text,
});
const transcript = (turn_id, text, final = false) => ({
  type: "astra.transcript",
  turn_id,
  text,
  final,
});

// --- criterion 8: the spoken channel and the user transcript ---------------------------------

test("criterion 8: spoken text groups by turn_id in arrival order, keyed by utterance_id", async () => {
  const { initialState, reduce, spokenGroups } = await load("state.js", "criterion 8");
  let s = initialState();
  s = reduce(s, spoken("t1", "u1", "It is a stack"), 1000);
  s = reduce(s, spoken("t2", "u3", "Four."), 2000);
  s = reduce(s, spoken("t1", "u2", "The error is on line four."), 3000);
  s = reduce(s, spoken("t1", "u1", "It is a stack trace."), 4000); // same utterance: in place
  assert.deepEqual(spokenGroups(s), [
    {
      turn_id: "t1",
      utterances: [
        { utterance_id: "u1", text: "It is a stack trace." },
        { utterance_id: "u2", text: "The error is on line four." },
      ],
    },
    { turn_id: "t2", utterances: [{ utterance_id: "u3", text: "Four." }] },
  ]);
});

test("criterion 8: a partial transcript is replaced; a final one freezes and the next starts a line", async () => {
  const { initialState, reduce, transcriptLines } = await load("state.js", "criterion 8");
  // The assertion is over { text, final } only: T-0037 criterion 11 makes each line carry its
  // turn_id (that is what lets a *replayed* line replace the line of its own turn instead of
  // appending a duplicate final line after a reload). The behaviour this row pins — a partial is
  // replaced, a final one freezes, a new turn starts a line — is unchanged, and the turn_id is
  // pinned in tests/hud/replay.test.mjs.
  const lines = () => transcriptLines(s).map(({ text, final }) => ({ text, final }));
  let s = initialState();
  s = reduce(s, transcript("t1", "what is", false), 1000);
  s = reduce(s, transcript("t1", "what is on my", false), 1200);
  assert.deepEqual(lines(), [{ text: "what is on my", final: false }]);
  s = reduce(s, transcript("t1", "what is on my screen", true), 1500);
  assert.deepEqual(lines(), [{ text: "what is on my screen", final: true }]);
  s = reduce(s, transcript("t2", "and", false), 5000);
  assert.deepEqual(lines(), [
    { text: "what is on my screen", final: true },
    { text: "and", final: false },
  ]);
});

test("reduce is pure: the state handed in is never mutated", async () => {
  const { initialState, reduce } = await load("state.js", "criteria 8-11");
  const s0 = initialState();
  const before = JSON.stringify(s0);
  reduce(s0, spoken("t1", "u1", "hello"), 1000);
  reduce(s0, turn("THINKING", "t1", 500), 1000);
  assert.equal(JSON.stringify(s0), before, "reduce(state, event) returns a new state (docs/02 §4)");
});

// --- criterion 9: the thinking meter ----------------------------------------------------------

test("criterion 9: the chip equals the server's turn state, for every TurnState", async () => {
  const { initialState, reduce } = await load("state.js", "criterion 9");
  // contracts/turn.py TurnState
  for (const st of [
    "IDLE",
    "LISTENING",
    "ENDPOINTING",
    "THINKING",
    "SPEAKING",
    "PAUSED",
    "INTERRUPTED",
  ]) {
    const s = reduce(initialState(), turn(st, "t1"), 1000);
    assert.equal(s.turnState, st, `criterion 9: the state chip equals state (${st})`);
    assert.equal(s.turnId, "t1");
  }
});

test("criterion 9: the meter advances with the page clock between messages", async () => {
  const { initialState, reduce, thinkingMs } = await load("state.js", "criterion 9");
  const s = reduce(initialState(), turn("THINKING", "t1", 500), 10_000);
  assert.equal(thinkingMs(s, 10_000), 500);
  assert.equal(thinkingMs(s, 11_200), 1_700, "500 ms from the server + 1200 ms of page clock");
});

test("criterion 9 adversarial: a SMALLER server value wins over the local extrapolation", async () => {
  const { initialState, reduce, thinkingMs } = await load("state.js", "criterion 9");
  let s = reduce(initialState(), turn("THINKING", "t1", 2000), 10_000);
  assert.equal(thinkingMs(s, 11_500), 3_500, "premise: the meter extrapolates to 3500 locally");
  s = reduce(s, turn("THINKING", "t1", 900), 11_500);
  assert.equal(
    thinkingMs(s, 11_500),
    900,
    "criterion 9: EVERY message snaps the meter to the server value, including when the server " +
      "value is smaller than the local extrapolation (3500 would be the page inventing time)",
  );
  assert.equal(thinkingMs(s, 11_900), 1_300, "and it keeps advancing from the snapped value");
});

test("criterion 9 adversarial: a stale THINKING for an older turn does not restart the meter", async () => {
  const { initialState, reduce, thinkingMs } = await load("state.js", "criterion 9");
  let s = reduce(initialState(), turn("THINKING", "t1", 1000), 10_000);
  s = reduce(s, turn("SPEAKING", "t2"), 11_000);
  s = reduce(s, turn("THINKING", "t1", 4000), 11_100); // late delivery for the finished turn
  assert.equal(s.turnState, "SPEAKING", "criterion 9: the chip does not go backwards to an old turn");
  assert.equal(s.turnId, "t2");
  assert.equal(
    thinkingMs(s, 12_000),
    null,
    "criterion 9 negative: no elapsed number is shown outside THINKING",
  );
});

test("criterion 9 negative: a non-THINKING state never starts the meter, even with thinking_ms", async () => {
  const { initialState, reduce, thinkingMs } = await load("state.js", "criterion 9");
  for (const st of ["LISTENING", "SPEAKING", "PAUSED", "INTERRUPTED", "IDLE", "ENDPOINTING"]) {
    const s = reduce(initialState(), turn(st, "t1", 2500), 10_000);
    assert.equal(
      thinkingMs(s, 12_000),
      null,
      `criterion 9 negative: state ${st} carrying thinking_ms must not start or advance the meter`,
    );
  }
});

test("criterion 9: the meter marks are 3000 ms and 4000 ms (docs/01 D11)", async () => {
  const mod = await load("state.js", "criterion 9");
  const marks = mod.THINKING_MARKS_MS;
  assert.deepEqual(
    [...(marks || [])],
    [3000, 4000],
    "criterion 9: marks at 3000 ms (the ~2000-token thinking budget) and 4000 ms (the " +
      "interim-update point); export them as THINKING_MARKS_MS so the meter and this test agree",
  );
});

// --- criterion 11: interrupted, and no invented played/unplayed state --------------------------

test("criterion 11: INTERRUPTED marks that turn in the spoken pane", async () => {
  const { initialState, reduce } = await load("state.js", "criterion 11");
  let s = initialState();
  s = reduce(s, spoken("t1", "u1", "It is a stack trace."), 1000);
  s = reduce(s, turn("INTERRUPTED", "t1"), 2000);
  assert.deepEqual([...s.interrupted], ["t1"], "criterion 11: the interrupted turn is marked");
});

test("criterion 11 negative: the page invents no played/unplayed distinction", async () => {
  const { initialState, reduce } = await load("state.js", "criterion 11");
  let s = initialState();
  s = reduce(s, spoken("t1", "u1", "It is a stack trace."), 1000);
  s = reduce(s, turn("INTERRUPTED", "t1"), 2000);
  const keys = [];
  const walk = (v, path) => {
    if (!v || typeof v !== "object") return;
    for (const [k, child] of Object.entries(v)) {
      keys.push(`${path}.${k}`);
      walk(child, `${path}.${k}`);
    }
  };
  walk(s, "state");
  const invented = keys.filter((k) => /played|unplayed|struck|strike/i.test(k));
  assert.deepEqual(
    invented,
    [],
    "criterion 11: no wire message carries played text (MachineState.played_text is server-side " +
      "only) — this is a named spec gap for T-0007 and must not be faked from timing: " +
      invented.join(", "),
  );
});

// T-0030 ruling A7: the T-0008-era pin above ("nothing under hud/ strikes text through") was
// written to be retired the moment the protocol carried the played prefix, which is what
// astra.spoken.played does. Its positive twin keeps the property that mattered — the page dims
// what the *server measured* and nothing else — by allowing exactly one strike-through rule, in
// one selector, and forbidding every clock in the code path that reaches it. The companion row
// above (no played message -> no played/unplayed distinction in the state) stays unchanged.
const STRIKE = /line-through|<s>|<strike|text-decoration\s*:\s*[^;]*line-through/i;
const UNCOMMENTED = (body) =>
  body
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/(^|[^:])\/\/[^\n]*/g, "$1 ")
    .replace(/<!--[\s\S]*?-->/g, " ");

test("T-0030 criteria 19-22: hud/ strikes text through in exactly one rule, and never from timing", async () => {
  const files = [];
  const walk = (dir) => {
    for (const e of readdirSync(dir)) {
      const p = join(dir, e);
      if (statSync(p).isDirectory()) walk(p);
      else if (/\.(js|mjs|css|html)$/.test(p)) files.push(p);
    }
  };
  const hud = fileURLToPath(new URL("hud", REPO));
  walk(hud);
  assert.ok(files.length > 0, "T-0008 adds hud/index.html, hud/styles.css and hud/src/*.js");

  // (a) Outside hud/styles.css nothing strikes anything: no inline style, no <s>, no class that
  // builds the decoration in JS. The rendering is one [data-unplayed] element and a stylesheet.
  const styles = join(hud, "styles.css");
  const offenders = files.filter((f) => f !== styles && STRIKE.test(UNCOMMENTED(readFileSync(f, "utf8"))));
  assert.deepEqual(
    offenders,
    [],
    "criterion 19: the tail is marked with [data-unplayed] and struck by the stylesheet; a " +
      "second place that draws a line through text is a second, unmeasured, way to claim the " +
      "user did not hear something",
  );

  // (b) In the stylesheet it is the [data-unplayed] rule and no other selector. Rule blocks are
  // matched crudely (`selector { declarations }`), which is all this stylesheet needs.
  const css = UNCOMMENTED(readFileSync(styles, "utf8"));
  const striking = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)]
    .filter((m) => STRIKE.test(m[2]))
    .map((m) => m[1].trim().replace(/\s+/g, " "));
  assert.deepEqual(
    striking,
    ["[data-unplayed]"],
    "criterion 20: exactly one rule strikes text through, and its selector is the element " +
      "spoken.js puts the unheard words in — so a struck word on the page is always a word the " +
      "server said was never rendered",
  );

  // (c) ... and nothing on the path that produces that element reads a clock. state.js takes the
  // page clock as the `nowMs` *argument* (criterion 17) and spoken.js takes none at all; a
  // setTimeout, a Date.now or a performance.now in either is how "dim it after a while" gets
  // written (criterion 22: no timing-derived strike-through).
  for (const name of ["src/state.js", "src/panels/spoken.js"]) {
    const body = UNCOMMENTED(readFileSync(join(hud, name), "utf8"));
    const clocks = ["setTimeout", "setInterval", "requestAnimationFrame", "Date.now", "performance.now"]
      .filter((token) => body.includes(token));
    assert.deepEqual(
      clocks,
      [],
      `criterion 22: hud/${name} reads a clock (${clocks.join(", ")}); the played/unplayed cut ` +
        "comes from astra.spoken.played and from nothing else",
    );
  }
});

// =================================================================================================
// T-0162 group D — the reducer handles astra.session.cleared (D-2, clear rows only)
// =================================================================================================
//
// docs/tasks/T-0162.md D-2: "On astra.session.cleared (live or replayed), every panel clears once
// per new resets value: … the thinking meter; the transcript and the spoken and played state. A
// repeated resets value clears nothing." The reducer is the state behind the meter, the transcript
// and the spoken/played channel, so the clear is pinned here through the existing accessors
// (spokenGroups, transcriptLines, thinkingMs) — no new export is required.
//
// The guard is two-sided (.claude/rules/tests.md): a reducer that clears on *every* cleared message
// wipes the new session's first turn at the next reconnect (the hello replays the same `resets`),
// and one that clears only once ever keeps the second session's turns after a second Restart.
// Rows are in the wire order the server publishes: the reset's echo, then the new turn's content,
// then the replayed echo at a reconnect's hello (hud_server.hello_reply puts it first).
//
// Strict xfail until docs/tasks/T-0162.md's status records "group D: tests …, activated <sha>"
// (the pendingT0162D mechanism of tests/hud/session.test.mjs).

const T0162_TASK = fileURLToPath(new URL("../../docs/tasks/T-0162.md", import.meta.url));
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

const sessionCleared = (resets, extra = {}) => ({
  type: "astra.session.cleared",
  resets,
  ended_turn_id: null,
  memory_forgotten: 0,
  replay: false,
  ...extra,
});
const playedReport = (turn_id, utterance_id, played_words, complete) => ({
  type: "astra.spoken.played",
  turn_id,
  played_ms: 900,
  utterances: [{ utterance_id, played_words, complete, played_text: "It is" }],
});

/** Two pre-reset turns, the second still THINKING: every piece of state D-2 names is non-empty. */
async function twoTurns() {
  const mod = await load("state.js", "T-0162 D-2");
  let s = mod.initialState();
  s = mod.reduce(s, turn("LISTENING", "t1"), 1000);
  s = mod.reduce(s, transcript("t1", "what is this", true), 1100);
  s = mod.reduce(s, spoken("t1", "u1", "It is a stack trace."), 1500);
  s = mod.reduce(s, playedReport("t1", "u1", 2, false), 1900);
  s = mod.reduce(s, transcript("t2", "and this one", true), 2100);
  s = mod.reduce(s, turn("THINKING", "t2", 1200), 2200);
  // premises, from the hand-written messages above
  assert.equal(mod.spokenGroups(s).length, 1, "premise: one spoken group");
  assert.equal(mod.spokenGroups(s)[0].utterances[0].played_words, 2, "premise: the played cut is recorded");
  assert.equal(mod.transcriptLines(s).length, 2, "premise: two transcript lines");
  assert.equal(mod.thinkingMs(s, 2300), 1300, "premise: the meter runs (1200 ms reported + 100 ms since)");
  return { mod, s };
}

pendingT0162D("T-0162 D-2: astra.session.cleared empties the transcript, the spoken and played state and the thinking meter", async () => {
  const { mod, s: before } = await twoTurns();
  const s = mod.reduce(before, sessionCleared(1, { ended_turn_id: "t2", memory_forgotten: 3 }), 2400);
  assert.deepEqual(mod.spokenGroups(s), [], "D-2: no utterance (and so no played cut) survives the reset");
  assert.deepEqual(mod.transcriptLines(s), [], "D-2: the transcript is empty");
  assert.equal(mod.thinkingMs(s, 2500), null, "D-2: the thinking meter shows nothing (no elapsed number from the cut turn)");
  // A late played report for a pre-reset turn invents nothing (T-0030: nothing is struck that was not shown).
  const late = mod.reduce(s, playedReport("t1", "u1", 4, true), 2600);
  assert.deepEqual(mod.spokenGroups(late), [], "a played report for a cleared turn creates no line");
});

pendingT0162D("T-0162 D-2: the replayed cleared is a clear too — a page that missed the live echo clears at its hello", async () => {
  const { mod, s: before } = await twoTurns();
  const s = mod.reduce(before, sessionCleared(1, { replay: true }), 2400);
  assert.deepEqual(mod.spokenGroups(s), [], "D-2 (live or replayed): a first-seen resets value clears, replay or not");
  assert.deepEqual(mod.transcriptLines(s), []);
  assert.equal(mod.thinkingMs(s, 2500), null);
});

pendingT0162D("T-0162 D-2 (guard, too wide): a repeated resets value clears nothing — the new session's turn survives the hello's replay", async () => {
  const { mod, s: before } = await twoTurns();
  let s = mod.reduce(before, sessionCleared(1), 2400);
  // the new session's first turn, in the order the server publishes it
  s = mod.reduce(s, turn("LISTENING", "t3"), 3000);
  s = mod.reduce(s, transcript("t3", "a new question", true), 3100);
  s = mod.reduce(s, turn("THINKING", "t3", 400), 3200);
  s = mod.reduce(s, spoken("t3", "u7", "A new answer."), 3600);
  // a reconnect: the hello reply begins with the same reset, replayed
  s = mod.reduce(s, sessionCleared(1, { replay: true }), 4000);
  s = mod.reduce(s, sessionCleared(1), 4001); // and a duplicate live copy changes nothing either
  assert.deepEqual(
    mod.spokenGroups(s),
    [{ turn_id: "t3", utterances: [{ utterance_id: "u7", text: "A new answer." }] }],
    "D-2: a repeated resets value clears nothing (the spoken line of the new session stays)",
  );
  assert.deepEqual(mod.transcriptLines(s), [{ text: "a new question", final: true, turn_id: "t3" }], "…and its transcript line stays");
});

pendingT0162D("T-0162 D-2 (guard, too narrow): a second Restart (resets 2) clears the second session in turn", async () => {
  const { mod, s: before } = await twoTurns();
  let s = mod.reduce(before, sessionCleared(1), 2400);
  s = mod.reduce(s, transcript("t3", "second session", true), 3100);
  s = mod.reduce(s, spoken("t3", "u7", "Second answer."), 3600);
  s = mod.reduce(s, sessionCleared(2, { ended_turn_id: null }), 5000);
  assert.deepEqual(mod.spokenGroups(s), [], "D-2: a new resets value (2) clears again");
  assert.deepEqual(mod.transcriptLines(s), []);
});

pendingT0162D("T-0162 D-2 (reviewer A T-4): a restarted CLI's resets 1 after 2 is new — `!==`, not `>`", async () => {
  const { mod, s: before } = await twoTurns();
  let s = mod.reduce(before, sessionCleared(1), 2400);
  s = mod.reduce(s, sessionCleared(2), 2500);
  s = mod.reduce(s, transcript("t5", "after two resets", true), 2600);
  s = mod.reduce(s, spoken("t5", "u5", "Still here."), 2700);
  assert.equal(mod.spokenGroups(s).length, 1, "premise: the session after reset 2 has content");
  s = mod.reduce(s, sessionCleared(1, { replay: true }), 2800); // a new CLI process, first reset
  assert.deepEqual(mod.spokenGroups(s), [], "D-2: 1 after 2 is a different value, so it clears");
  assert.deepEqual(mod.transcriptLines(s), []);
});

pendingT0162D("T-0162 D-2: a fresh page first told of reset 3 at its hello clears once, then keeps what follows", async () => {
  const mod = await load("state.js", "T-0162 D-2");
  let s = mod.initialState();
  s = mod.reduce(s, sessionCleared(3, { replay: true }), 100); // hello_reply's first message
  s = mod.reduce(s, transcript("t9", "hello again", true), 200);
  s = mod.reduce(s, spoken("t9", "u1", "Hi."), 300);
  s = mod.reduce(s, sessionCleared(3, { replay: true }), 400); // the next reconnect's hello
  assert.deepEqual(mod.spokenGroups(s), [{ turn_id: "t9", utterances: [{ utterance_id: "u1", text: "Hi." }] }], "D-2: resets 3 was already seen");
  assert.deepEqual(mod.transcriptLines(s).map((l) => l.turn_id), ["t9"]);
  // …and the next new value (a Restart from this page) clears it: the page counts from 3, not from 0.
  s = mod.reduce(s, sessionCleared(4), 500);
  assert.deepEqual(mod.spokenGroups(s), [], "D-2: resets 4 is new, so it clears");
  assert.deepEqual(mod.transcriptLines(s), []);
});

pendingT0162D("T-0162 D-2: other message types clear nothing (the reducer keys on the type)", async () => {
  const { mod, s: before } = await twoTurns();
  let s = mod.reduce(before, { type: "astra.session.reset" }, 2400); // the page's own outbound type, echoed by mistake
  s = mod.reduce(s, { type: "astra.log.state", on: false, path: null, records: 0, dropped: 0, bytes: 0, error: null }, 2401);
  assert.equal(mod.spokenGroups(s).length, 1, "only astra.session.cleared clears");
  assert.equal(mod.transcriptLines(s).length, 2);
  // the anti-vacuity half: the very same state is cleared by the real message
  s = mod.reduce(s, sessionCleared(1), 2402);
  assert.deepEqual(mod.spokenGroups(s), []);
});
