// T-0227 group B — the page follows the capture interval (docs/tasks/T-0227.md B-1 .. B-3), the node half.
//
// Every row is derived from the acceptance criteria and "Tests to write" (Node, B), never from an
// implementation (none exists when this file is written). The Chrome half (the panel, the echo, the measured
// cadence, the next request's image count, no egress, the B-4 frozen files) is
// tests/integration/test_hud_frame_settings.py.
//
// Gate. The new-code rows run through `pendingT0227B`, the Status-phrase strict-xfail gate of
// tests/hud/tools.test.mjs (`pendingT0162B`), reading docs/tasks/T-0227.md. While its **Status:** line does not
// record "group B: tests <a>..<b>, activated <sha>", each gated row must FAIL (an XPASS fails the row: a stub
// that satisfies it is vacuous). ASTRA_RUN_PENDING=1 runs them for real now. Group A's activation does not switch
// these rows on. The ungated rows are B-1's / B-4's "exactly as today" negatives and hold at BASE by design.
//
// API these rows pin (the task names `setIntervalMs` "e.g."; the other two names are the test-author's, so the
// state reader and the page's tick are testable without a DOM — hud/src/main.js is not importable in node):
//
//   hud/src/frames.js
//     FRAME_INTERVALS_MS                    the allowed intervals, ascending, a mirror of
//                                           config.FRAME_INTERVALS_MS (250 .. 10000) (B-1)
//     createFramePipeline(...).setIntervalMs(ms)
//                                           ms an integer in FRAME_INTERVALS_MS: from the next accepted frame
//                                           on, at most one frame per source per ms; anything else is ignored
//                                           (the gap in force is kept) and logged through the pipeline's `log`
//                                           (B-1). Without a call the gap is 1000 / MAX_FPS.
//     frameIntervalFromState(message) -> number | null
//                                           the `value` (never `pending`) of the `frame_interval_ms` item of an
//                                           astra.settings.state, replayed or not, iff it is an integer in
//                                           FRAME_INTERVALS_MS; null for anything else (B-2). main.js calls
//                                           pipeline.setIntervalMs with a non-null result.
//     captureTickMs(intervalMs) -> number   the page's offer tick at that interval (main.js's setInterval
//                                           period); 250 at 1000 ms (B-3).
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);

// --- the T-0227 group B strict-xfail gate ----------------------------------------------------------------------

const TASK = fileURLToPath(new URL("docs/tasks/T-0227.md", REPO));
const STATUS = existsSync(TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(TASK, "utf8")) || [])[1] || "").trim()
  : "";
/** "group B: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6" — the phrase the orchestrator writes at activation. */
const B_ACTIVATED = /\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const B_PENDING = !B_ACTIVATED.test(STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0227B(name, fn) {
  test(name + (B_PENDING ? " [pending T-0227 B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0227 group B row passed while docs/tasks/T-0227.md's status does not record " +
          "group B's activation (" + JSON.stringify(STATUS.slice(0, 80)) + "). Either group B has landed " +
          "(record 'group B: tests …, activated <sha>' in the status) or the row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0227 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0227 B gate: the status line is readable and only group B's own phrase activates the rows", () => {
  assert.ok(STATUS.length > 0, "docs/tasks/T-0227.md must carry a **Status:** line");
  assert.equal(B_ACTIVATED.test("planned 2026-11-01 (James, 2026-11-01: \"I hope ...\")"), false);
  assert.equal(B_ACTIVATED.test("in progress (group A: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6)"), false);
  assert.equal(
    B_ACTIVATED.test("in progress (group A: …, activated 9a8b7c6; group B: tests a1b2c3d..e4f5a6b, activated c3d4e5f)"),
    true,
  );
  assert.equal(B_ACTIVATED.test("in progress (group B: tests a1b2c3d..e4f5a6b, not yet activated)"), false);
  assert.equal(B_ACTIVATED.test("in progress (group B tests written; group A: tests …, activated 1234abc)"), false);
});

// --- loading the module ----------------------------------------------------------------------------------------

// node 18 treats a bare .js as CommonJS without a {"type":"module"} package.json in scope (see
// tests/hud/throttle.test.mjs): the page's sources are copied once into a temp directory that carries one.
let ESM_ROOT = null;
function srcRoot() {
  if (ESM_ROOT) return ESM_ROOT;
  const real = fileURLToPath(new URL("hud/src", REPO));
  const dir = mkdtempSync(join(tmpdir(), "astra-hud-esm-t0227-"));
  writeFileSync(join(dir, "package.json"), '{"type":"module","private":true}');
  if (existsSync(real)) cpSync(real, join(dir, "src"), { recursive: true });
  ESM_ROOT = pathToFileURL(join(dir, "src") + "/");
  return ESM_ROOT;
}

// Origin proof (.claude/rules/tests.md): where the module under test was imported from.
console.log(`# frame interval origin: repo=${fileURLToPath(REPO)} modules=${srcRoot().href}`);

async function frames() {
  try {
    return await import(new URL("frames.js", srcRoot()).href);
  } catch (err) {
    assert.fail(`hud/src/frames.js must import in node with no DOM (T-0008): ${err.message}`);
  }
}

async function pipelineWithSetter(opts) {
  const mod = await frames();
  const pipe = mod.createFramePipeline(opts);
  assert.equal(
    typeof pipe.setIntervalMs,
    "function",
    "T-0227 B-1: createFramePipeline gains an interval setter, setIntervalMs(ms)",
  );
  return pipe;
}

async function stateReader() {
  const mod = await frames();
  assert.equal(
    typeof mod.frameIntervalFromState,
    "function",
    "T-0227 B-2: frames.js exports frameIntervalFromState(message) -> number | null (the DOM-free state reader " +
      "main.js applies; API named in this file's header)",
  );
  return mod.frameIntervalFromState;
}

async function tickOf() {
  const mod = await frames();
  assert.equal(
    typeof mod.captureTickMs,
    "function",
    "T-0227 B-3: frames.js exports captureTickMs(intervalMs), the page's offer tick at that interval (API " +
      "named in this file's header), so the tick can be driven from node",
  );
  return mod.captureTickMs;
}

// --- the harness -----------------------------------------------------------------------------------------------

/** Let every queued microtask and the pipeline's own async steps run. */
const flush = () => new Promise((resolve) => setImmediate(resolve));

const ENCODED = { width: 1568, height: 980, jpeg_b64: "/9j/AAAA", phash: "0123456789abcdef" };

/** The allowed intervals, typed from A-1 / B-1 (never read from frames.js or config.py). */
const SPEC_INTERVALS = [250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000];

function harness({ encode } = {}) {
  const h = { t: 0, sent: [], logs: [], inFlight: 0, maxInFlight: 0 };
  h.now = () => h.t;
  h.log = (line) => h.logs.push(String(line));
  h.encode = async (...args) => {
    h.inFlight += 1;
    h.maxInFlight = Math.max(h.maxInFlight, h.inFlight);
    try {
      return encode ? await encode(...args) : ENCODED;
    } finally {
      h.inFlight -= 1;
    }
  };
  h.send = async (msg) => {
    h.sent.push(msg);
  };
  return h;
}

function opts(h) {
  return { now: h.now, encode: h.encode, send: h.send, log: h.log };
}

/** Offer every `stepMs` from `from` to `to` (inclusive), flushing after each; returns the sent ts_ms in the range. */
async function drive(h, pipe, { from, to, stepMs, source = "screen" }) {
  const before = h.sent.length;
  for (let t = from; t <= to; t += stepMs) {
    h.t = t;
    pipe.offer(source);
    await flush();
  }
  return h.sent.slice(before).map((m) => m.ts_ms);
}

const gaps = (ts) => ts.slice(1).map((v, i) => v - ts[i]);
function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

/** A deterministic ±2 ms jitter (an LCG, seed fixed): each value in {-2, -1, 0, 1, 2}. */
function jitter(seed) {
  let x = seed >>> 0;
  return () => {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    return (x % 5) - 2;
  };
}

function settingsState(items, { replay = false } = {}) {
  return {
    type: "astra.settings.state",
    rev: 3,
    items,
    system_sha12: "0123456789ab",
    instructions: "",
    instructions_max: 1000,
    refused: [],
    replay,
  };
}

function intervalItem(value, pending = null) {
  return { key: "frame_interval_ms", value, source: "ui", kind: "next_turn", changeable: true, pending };
}

// --- B-1: the default, exactly as today (ungated: holds at BASE by design) -------------------------------------

test("B-1 negative: without a call the gap is 1000 / MAX_FPS = 1000 ms, and MAX_FPS stays 1.0", async () => {
  const { createFramePipeline, MAX_FPS } = await frames();
  assert.equal(MAX_FPS, 1.0, "B-1: MAX_FPS stays exported and equal to 1.0");
  const h = harness();
  const pipe = createFramePipeline(opts(h));
  const ts = await drive(h, pipe, { from: 0, to: 9990, stepMs: 10 });
  // Hand-computed: offers at 0, 10, ..., 9990 and a 1000 ms gap send at 0, 1000, ..., 9000.
  assert.deepEqual(ts, [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]);
});

// --- B-1: the interval setter ----------------------------------------------------------------------------------

pendingT0227B("B-1: FRAME_INTERVALS_MS mirrors the task's list (and config.FRAME_INTERVALS_MS once A has it)", async () => {
  const { FRAME_INTERVALS_MS } = await frames();
  assert.ok(FRAME_INTERVALS_MS, "B-1: frames.js exports FRAME_INTERVALS_MS");
  assert.deepEqual([...FRAME_INTERVALS_MS], SPEC_INTERVALS, "A-1 / B-1: the nine presets, ascending, in ms");
  // Parity with the Python constant (A-1). The paired Python row is A's (test_hud_constants_parity.py). Here the
  // literal is read from config.py when group A has written it; until then the task's list is the oracle.
  const config = readFileSync(fileURLToPath(new URL("local_astra/config.py", REPO)), "utf8");
  const m = /^FRAME_INTERVALS_MS\b[^=\n]*=\s*\(([^)]*)\)/m.exec(config);
  if (m) {
    const py = m[1]
      .split(",")
      .map((s) => s.trim().replace(/_/g, ""))
      .filter(Boolean)
      .map(Number);
    assert.deepEqual([...FRAME_INTERVALS_MS], py, "B-1: frames.js FRAME_INTERVALS_MS == config.FRAME_INTERVALS_MS");
  }
});

pendingT0227B("B-1: after setIntervalMs(250), exactly one frame per 250 ms under offers every 10 ms", async () => {
  const h = harness();
  const pipe = await pipelineWithSetter(opts(h));
  pipe.setIntervalMs(250);
  const ts = await drive(h, pipe, { from: 0, to: 9990, stepMs: 10 });
  // Hand-computed: 0, 250, ..., 9750 — forty frames, every gap 250.
  const want = Array.from({ length: 40 }, (_, k) => k * 250);
  assert.deepEqual(ts, want, `B-1: one frame per 250 ms; got ${ts.length} frames, gaps ${JSON.stringify(gaps(ts))}`);
});

pendingT0227B("B-1: every allowed interval gives exactly that gap under offers every 10 ms", async () => {
  const { FRAME_INTERVALS_MS } = await frames();
  for (const ms of SPEC_INTERVALS) {
    const h = harness();
    const pipe = await pipelineWithSetter(opts(h));
    pipe.setIntervalMs(ms);
    const ts = await drive(h, pipe, { from: 0, to: 4 * ms, stepMs: 10 });
    assert.deepEqual(ts, [0, ms, 2 * ms, 3 * ms, 4 * ms], `B-1: interval ${ms} ms`);
  }
  assert.ok(FRAME_INTERVALS_MS, "premise: the list exists");
});

pendingT0227B("B-1 (T-0008 3-5): single flight with a slow encoder at 250 ms: one in flight, one pending, older dropped", async () => {
  const releases = [];
  const encode = () => new Promise((resolve) => releases.push(() => resolve(ENCODED)));
  const h = harness({ encode });
  const pipe = await pipelineWithSetter(opts(h));
  pipe.setIntervalMs(250);
  const outcomes = [];
  h.t = 0;
  outcomes.push(pipe.offer("screen")); // starts a 600 ms encode
  await flush();
  for (let t = 10; t <= 590; t += 10) {
    h.t = t;
    outcomes.push(pipe.offer("screen"));
    await flush();
    assert.ok(h.maxInFlight <= 1, `T-0008 4: one encode in flight per source, saw ${h.maxInFlight}`);
    assert.ok(pipe.stats().pending <= 1, `T-0008 3: at most one pending, saw ${pipe.stats().pending}`);
  }
  // 59 offers arrived behind the stuck encode; all but the newest pending one must already be dropped.
  const early = await Promise.race([Promise.all(outcomes.slice(1, -1)), flush().then(() => "unsettled")]);
  assert.notEqual(early, "unsettled", "T-0008 3: the older offers behind a stuck encode resolve at once (dropped)");
  assert.ok(early.every((o) => o === "dropped"), `T-0008 3: older offers are dropped, got ${JSON.stringify(early)}`);
  h.t = 600;
  releases.shift()();
  await flush();
  await flush();
  // The pending one (offered at 590) runs at now() = 600, 600 - 0 >= 250: sent; its encode is the second.
  assert.equal(releases.length, 1, "the retained pending frame starts the next encode, alone");
  assert.ok(h.maxInFlight <= 1, `still one in flight, saw ${h.maxInFlight}`);
  h.t = 800;
  releases.shift()();
  const all = await Promise.all(outcomes);
  assert.equal(all[0], "sent");
  assert.equal(all.at(-1), "sent", "the newest pending offer is the one kept");
  assert.equal(h.sent.length, 2, `two frames out of 60 offers; got ${h.sent.length}`);
  assert.deepEqual(gaps(h.sent.map((m) => m.ts_ms)), [600]);
  const s = pipe.stats();
  assert.equal(s.sent + s.dropped, 60, `conservation: every offer is sent or dropped (${JSON.stringify(s)})`);
});

pendingT0227B("B-1: a change mid-flight takes effect from the next accepted frame (1000 -> 250)", async () => {
  let release = null;
  let first = true;
  const encode = () => {
    if (!first) return ENCODED;
    first = false;
    return new Promise((resolve) => (release = () => resolve(ENCODED)));
  };
  const h = harness({ encode });
  const pipe = await pipelineWithSetter(opts(h));
  h.t = 0;
  const inflight = pipe.offer("screen");
  await flush();
  pipe.setIntervalMs(250); // while the 1000 ms-era frame is in flight
  h.t = 100;
  release();
  assert.equal(await inflight, "sent", "the frame in flight is not cancelled by the change");
  const ts = await drive(h, pipe, { from: 110, to: 600, stepMs: 10 });
  assert.deepEqual(ts, [250, 500], `the next accepted frame is judged by 250 ms (from the in-flight one at 0); got ${ts}`);
});

pendingT0227B("B-1: a change mid-flight takes effect from the next accepted frame (250 -> 2000)", async () => {
  let release = null;
  let calls = 0;
  const encode = () => {
    calls += 1;
    if (calls !== 2) return ENCODED;
    return new Promise((resolve) => (release = () => resolve(ENCODED)));
  };
  const h = harness({ encode });
  const pipe = await pipelineWithSetter(opts(h));
  pipe.setIntervalMs(250);
  h.t = 0;
  await pipe.offer("screen");
  h.t = 250;
  const inflight = pipe.offer("screen"); // the second encode, held
  await flush();
  pipe.setIntervalMs(2000);
  h.t = 300;
  release();
  assert.equal(await inflight, "sent");
  const ts = await drive(h, pipe, { from: 310, to: 4500, stepMs: 10 });
  assert.deepEqual(ts, [2250, 4250], `from the in-flight frame at 250, the next ones are 2000 ms apart; got ${ts}`);
});

pendingT0227B("B-1: setIntervalMs refuses 249, 251, \"500\", NaN, null, 1e9 (and other non-members), keeps the gap, logs", async () => {
  const bad = [249, 251, "500", NaN, null, 1e9, undefined, true, 250.5, -250, Infinity, 0];
  // (a) from the default: the gap stays 1000.
  {
    const h = harness();
    const pipe = await pipelineWithSetter(opts(h));
    for (const v of bad) pipe.setIntervalMs(v);
    const ts = await drive(h, pipe, { from: 0, to: 3990, stepMs: 10 });
    assert.deepEqual(ts, [0, 1000, 2000, 3000], `B-1: refused values leave the default; got ${ts}`);
    assert.ok(h.logs.length >= 1, "B-1: a refused value is logged");
    assert.ok(h.logs.length <= bad.length, `B-1: logged once per refusal at most, got ${h.logs.length} lines`);
  }
  // (b) from 500: a refusal keeps the interval in force, not the default.
  {
    const h = harness();
    const pipe = await pipelineWithSetter(opts(h));
    pipe.setIntervalMs(500);
    const logsBefore = h.logs.length;
    for (const v of bad) pipe.setIntervalMs(v);
    const ts = await drive(h, pipe, { from: 0, to: 1990, stepMs: 10 });
    assert.deepEqual(ts, [0, 500, 1000, 1500], `B-1: refused values keep 500; got ${ts}`);
    assert.ok(h.logs.length > logsBefore, "B-1: the refusals are logged");
  }
  // (c) guard too wide: every allowed value is accepted (a setter that refuses all keeps 1000 and fails here).
  for (const ms of [250, 750, 1500, 10000]) {
    const h = harness();
    const pipe = await pipelineWithSetter(opts(h));
    pipe.setIntervalMs(ms);
    const ts = await drive(h, pipe, { from: 0, to: ms, stepMs: 10 });
    assert.deepEqual(ts, [0, ms], `B-1: ${ms} is in FRAME_INTERVALS_MS and is accepted`);
  }
});

// --- B-3: the offer tick never quantises the interval ----------------------------------------------------------

/** An independent model of today's landmine (not the code under test): a strict throttle fed by a jittered tick. */
function strictThrottleModel(intervalMs, tickMs, seconds, seed) {
  const j = jitter(seed);
  const sent = [];
  let last = -Infinity;
  for (let k = 0; k * tickMs <= seconds * 1000; k++) {
    const t = Math.max(0, k * tickMs + j());
    if (t - last >= intervalMs) {
      sent.push(t);
      last = t;
    }
  }
  return sent;
}

// Why the B-3 row checks the mean as well as the task's median. With iid ±2 ms jitter a strict throttle at a 250 ms
// tick drops a frame whenever a tick fires earlier (relative to its predecessor) than the last accepted one did,
// about 2 ticks in 5: the gaps are then ~250 (60 %) and ~500 (40 %). The median stays at ~251, so the task's median
// bound alone cannot see the landmine under this model; the mean gap (~350 ms, a 30 % rate cut) does. A mean bound
// of 1.3x is the same threshold applied to the rate, which is the failure mode B-3 names ("halves the rate").
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

test("B-3 premise: the ±2 ms jitter model bites a 250 ms tick at a 250 ms strict interval (the landmine)", () => {
  for (const seed of [1, 0x7227, 0xbeef]) {
    const g = gaps(strictThrottleModel(250, 250, 60, seed));
    assert.ok(mean(g) > 325, `premise (seed ${seed}): the model cuts the rate, mean gap ${mean(g)} > 325`);
    assert.ok(Math.max(...g) >= 496, `premise (seed ${seed}): some ticks are dropped (a ~500 ms gap)`);
    const ok = gaps(strictThrottleModel(1000, 250, 60, seed));
    assert.ok(mean(ok) <= 1300 && median(ok) <= 1300, "premise: at 1000 ms a 250 ms tick is fine, as today");
  }
});

pendingT0227B("B-3: the tick is 250 ms at the default 1000 ms interval", async () => {
  const captureTickMs = await tickOf();
  assert.equal(captureTickMs(1000), 250, "B-3: the tick is 250 ms when the interval is 1000 ms (as today)");
  for (const ms of SPEC_INTERVALS) {
    const tick = captureTickMs(ms);
    assert.ok(Number.isFinite(tick) && tick > 0, `captureTickMs(${ms}) is a positive period, got ${tick}`);
  }
});

pendingT0227B("B-3 (adversarial) tick-quantisation: at every interval, ±2 ms tick jitter keeps gaps >= interval, median and mean <= 1.3x", async () => {
  const captureTickMs = await tickOf();
  const problems = [];
  for (const ms of SPEC_INTERVALS) {
    for (const seed of [1, 0x7227, 0xbeef]) {
      const h = harness();
      const pipe = await pipelineWithSetter(opts(h));
      pipe.setIntervalMs(ms);
      const tick = captureTickMs(ms);
      const j = jitter(seed);
      const horizon = Math.max(60_000, 30 * ms);
      let prev = -1;
      for (let k = 0; k * tick <= horizon; k++) {
        h.t = Math.max(prev + 1, k * tick + j()); // a page clock never runs backwards
        prev = h.t;
        pipe.offer("screen");
        await flush();
      }
      const g = gaps(h.sent.map((m) => m.ts_ms));
      if (g.length < 5) {
        problems.push(`${ms} ms seed ${seed}: only ${g.length + 1} frames in ${horizon} ms`);
        continue;
      }
      const min = Math.min(...g);
      const med = median(g);
      if (min < ms) problems.push(`${ms} ms seed ${seed}: a gap of ${min} ms < the interval (tick ${tick})`);
      if (med > 1.3 * ms) problems.push(`${ms} ms seed ${seed}: median gap ${med} ms > ${1.3 * ms} (tick ${tick})`);
      const avg = mean(g);
      if (avg > 1.3 * ms) problems.push(`${ms} ms seed ${seed}: mean gap ${avg.toFixed(1)} ms > ${1.3 * ms}, the rate is cut (tick ${tick})`);
    }
  }
  assert.deepEqual(problems, [], "B-3:\n  " + problems.join("\n  "));
});

// --- B-2: the page reads the value in force --------------------------------------------------------------------

pendingT0227B("B-2: an item with pending 500 and value 1000 keeps 1000; a later state with value 500 applies it", async () => {
  const read = await stateReader();
  assert.equal(read(settingsState([intervalItem(1000, 500)])), 1000, "B-2: the value in force, never pending");
  assert.equal(read(settingsState([intervalItem(500)])), 500, "B-2: the applied value");
  // the same through the pipeline, as main.js wires it
  const h = harness();
  const pipe = await pipelineWithSetter(opts(h));
  for (const m of [settingsState([intervalItem(1000, 500)]), settingsState([intervalItem(500)])]) {
    const ms = read(m);
    if (ms !== null) pipe.setIntervalMs(ms);
  }
  const ts = await drive(h, pipe, { from: 0, to: 1990, stepMs: 10 });
  assert.deepEqual(ts, [0, 500, 1000, 1500]);
});

pendingT0227B("B-2: a replayed state (the hello) applies", async () => {
  const read = await stateReader();
  const other = { key: "visual_tokens", value: 560, source: "flag", kind: "next_turn_cold", changeable: true };
  assert.equal(read(settingsState([other, intervalItem(750)], { replay: true })), 750);
  assert.equal(read(settingsState([intervalItem(2000), other], { replay: true })), 2000);
});

pendingT0227B("B-2: absent, null, wrongly typed or out-of-set values leave the current interval (null)", async () => {
  const read = await stateReader();
  const cases = [
    ["no item", settingsState([{ key: "visual_tokens", value: 560, source: "flag", kind: "next_turn_cold", changeable: true }])],
    ["no items", settingsState([])],
    ["value null", settingsState([intervalItem(null)])],
    ["value string", settingsState([intervalItem("500")])],
    ["value bool", settingsState([intervalItem(true)])],
    ["value 499", settingsState([intervalItem(499)])],
    ["value 500.5", settingsState([intervalItem(500.5)])],
    ["value 1e9", settingsState([intervalItem(1e9)])],
    ["only pending", settingsState([{ ...intervalItem(null), pending: 500 }])],
    ["another type", { ...settingsState([intervalItem(500)]), type: "astra.assist.state" }],
    ["items not a list", { ...settingsState([]), items: { frame_interval_ms: 500 } }],
    ["null message", null],
  ];
  const bad = [];
  for (const [what, m] of cases) {
    let got;
    try {
      got = read(m);
    } catch (err) {
      got = `threw ${err && err.message}`;
    }
    if (got !== null) bad.push(`${what}: ${JSON.stringify(got)}`);
  }
  assert.deepEqual(bad, [], "B-2: each leaves the interval in force (null), never throws");
  // guard too wide: a valid value is read (a reader returning null always fails here)
  assert.equal(read(settingsState([intervalItem(250)])), 250);
});

pendingT0227B("B-2 wiring: main.js applies the reader's value to the pipeline and derives its tick", async () => {
  // A source grep, scoped to code (comments stripped): main.js is not importable in node. The Chrome module is the
  // behavioural check of the same wiring; this row only catches a frames.js API that main.js never calls.
  const src = readFileSync(fileURLToPath(new URL("hud/src/main.js", REPO)), "utf8")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/.*$/gm, "$1");
  for (const name of ["frameIntervalFromState(", ".setIntervalMs(", "captureTickMs("]) {
    assert.ok(src.includes(name), `B-2/B-3: hud/src/main.js calls ${name}…)`);
  }
});

/** `function <name>(...) { <body> }` declarations in `src`, bodies found by brace matching (strings ignored). */
function functionBodies(src) {
  const out = new Map();
  const re = /function\s+([A-Za-z_$][\w$]*)\s*\([^)]*\)\s*\{/g;
  let m;
  while ((m = re.exec(src))) {
    let depth = 1;
    let i = re.lastIndex;
    let quote = null;
    for (; i < src.length && depth > 0; i++) {
      const c = src[i];
      if (quote) {
        if (c === "\\") i++;
        else if (c === quote) quote = null;
      } else if (c === '"' || c === "'" || c === "`") quote = c;
      else if (c === "{") depth++;
      else if (c === "}") depth--;
    }
    out.set(m[1], src.slice(re.lastIndex, i - 1));
  }
  return out;
}

pendingT0227B("B-3 wiring: the function that applies a new interval also restarts the offer tick from captureTickMs", async () => {
  // Review finding (T-0227 B): the startup call alone satisfies the row above, so a page that changes the pipeline's
  // gap but keeps the startup tick was unpinned. Code only (comments stripped); a source grep, as stated above. The
  // restart may be inline (setInterval + captureTickMs in the same body) or one call into a function holding both.
  const src = readFileSync(fileURLToPath(new URL("hud/src/main.js", REPO)), "utf8")
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/(^|[^:])\/\/.*$/gm, "$1");
  const bodies = functionBodies(src);
  const restarts = (body) => /\bsetInterval\s*\(/.test(body) && /\bcaptureTickMs\s*\(/.test(body);
  const appliers = [...bodies].filter(([, body]) => /\.setIntervalMs\s*\(/.test(body));
  assert.ok(appliers.length >= 1, "B-2: a named function in main.js calls pipeline.setIntervalMs(…)");
  const loops = new Set([...bodies].filter(([, body]) => restarts(body)).map(([name]) => name));
  const bad = appliers
    .filter(([, body]) => !restarts(body) && ![...loops].some((n) => new RegExp(`\\b${n}\\s*\\(`).test(body)))
    .map(([name]) => name);
  assert.deepEqual(
    bad,
    [],
    `B-3: ${bad.join(", ")} sets a new interval without restarting the offer tick (setInterval with ` +
      `captureTickMs(...), inline or through ${[...loops].join(", ") || "no such function"})`,
  );
});
