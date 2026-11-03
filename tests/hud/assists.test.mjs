// T-0025 — the assist panel's DOM-free half: hud/src/assists/gate.js and its extractor registry.
//
// Criteria 1, 4, 5, 6, 7, 8, 9, 10 and 16 of docs/tasks/T-0025.md. The rows, the badge element
// and the counters are DOM and live in tests/integration/test_hud_assists.py (headless Chrome);
// everything here runs on injected `send`, `now` and extractors, which is what makes the egress
// gate testable without a browser (T-0025 implementation notes: "Keep the pure part DOM-free").
//
// This file *skips itself* while hud/src/assists/gate.js is absent: node test files cannot carry
// a pytest `pending` marker and tests/unit/test_hud_js.py asserts `node --test tests/hud` exits 0.
// The moment the implementer creates gate.js every case below runs for real.
//
// API this pins (the task leaves the JS names to the test-author; these are them):
//
//   hud/src/assists/gate.js — no DOM at module top level, importable in node
//     ASSIST_CLIENT_TYPES  ["astra.assist.toggle", "astra.frame.text"]   (criterion 10)
//     ASSIST_KINDS         ["page_text", "pdf_text", "cells", "at_spi"]  (criterion 2, the order
//                          is AssistFlags declaration order and is pinned against the Python
//                          contract by tests/unit/test_hud_constants_parity.py)
//     MAX_TEXT_BYTES       32768 — UTF-8 bytes, marker included (criterion 9)
//     TRUNCATION_MARKER    the in-band marker a truncated text ends with
//     truncateUtf8(text, maxBytes = MAX_TEXT_BYTES) -> { text, truncated }
//     createAssistGate({ send, now, extractors, reasons, maxFps, log }) -> gate
//       extractors: { [kind]: ({ source, ts_ms }) -> string | null | Promise<string|null> }
//       reasons:    { [kind]: string } — why a kind has no extractor here (criterion 8)
//       gate.onState(msg)      feed a decoded astra.assist.state; nothing else confirms a flag
//       gate.flags()           the last server-confirmed flag set ({kind: bool}, all four keys)
//       gate.badge()           the server's badge string, verbatim ("" before any state)
//       gate.pending()         kinds whose toggle has not been echoed yet (criterion 3)
//       gate.available()       { [kind]: { available: bool, reason: string } }
//       gate.toggle(kind, on)  -> bool: sent a toggle? (whole flag set, never a per-kind merge)
//       gate.offer(source)     -> Promise<{ sent: [kind], dropped: [kind] }>
//       gate.stats()           { sent, dropped, truncated, inFlight }
//
//   hud/src/assists/extract.js — the v1 registry (criterion 8)
//     registryFor({ source, synthetic, doc }) -> { extractors, reasons }
//
//   hud/src/assists/index.js — the DOM half
//     mount(host) where host is { shown, document, send, subscribe, log } (T-0025 criterion 13)
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);
const ASSISTS_DIR = fileURLToPath(new URL("hud/src/assists", REPO));
const GATE_JS = join(ASSISTS_DIR, "gate.js");
const INSTALLED = existsSync(GATE_JS);
const SKIP = {
  skip:
    "T-0025 is not implemented yet: hud/src/assists/gate.js does not exist. This file is the " +
    "acceptance suite of docs/tasks/T-0025.md (the HUD assist panel: checkboxes, badge, egress " +
    "gate); it runs the moment that module lands. Node tests cannot carry pytest's `pending` " +
    "marker and tests/unit/test_hud_js.py requires `node --test tests/hud` to exit 0.",
};
const opts = INSTALLED ? {} : SKIP;

// node 18 treats a bare .js file as CommonJS unless a package.json in scope says
// {"type":"module"}; hud/package.json (T-0010's file) says so today, but T-0008's tests do not
// depend on that and neither does this one: the sources are copied once into a temp directory
// that carries the flag itself. Same shim as tests/hud/protocol.test.mjs.
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
      `T-0025 adds hud/src/${name}, importable in node as an ES module with no DOM at module ` +
        `top level (implementation notes: "keep the pure part DOM-free"). node said: ${err.message}`,
    );
  }
}

const gateModule = () => load("assists/gate.js");
const KINDS = ["page_text", "pdf_text", "cells", "at_spi"];

// --- helpers: a fake clock, a recording send, and the state messages the server would send -----

function clock(start = 1_700_000_000_000) {
  let t = start;
  return { now: () => t, advance: (ms) => (t += ms) };
}

function recorder() {
  const sent = [];
  return {
    sent,
    send: (message) => {
      sent.push(message);
      return Promise.resolve();
    },
    ofType: (type) => sent.filter((m) => m && m.type === type),
  };
}

/** A decoded astra.assist.state exactly as hud/src/protocol.js hands it over. */
function stateMessage(on = {}, badge = "pixels") {
  const flags = {};
  for (const kind of KINDS) flags[kind] = Boolean(on[kind]);
  return { type: "astra.assist.state", flags, badge };
}

const utf8 = (s) => new TextEncoder().encode(s).length;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Bounded wait: a gate that *queues* instead of dropping would otherwise hang this file. */
async function within(promise, ms, what) {
  const timeout = Symbol("timeout");
  const got = await Promise.race([promise, sleep(ms).then(() => timeout)]);
  assert.notEqual(got, timeout, `${what} did not settle within ${ms} ms`);
  return got;
}

function newGate(mod, { extractors = {}, reasons = {}, clk = clock(), rec = recorder() } = {}) {
  const gate = mod.createAssistGate({
    send: rec.send,
    now: clk.now,
    extractors,
    reasons,
    log: () => {},
  });
  return { gate, clk, rec };
}

function jsFilesUnder(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) out.push(...jsFilesUnder(p));
    else if (/\.(js|mjs)$/.test(p)) out.push(p);
  }
  return out;
}

function stripComments(text) {
  return text.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/(^|[^:])\/\/[^\n]*/g, "$1 ");
}

// --- criterion 5: nothing goes out before the server confirms a flag ---------------------------

test("gate: nothing goes out before the server confirms a flag", opts, async () => {
  const mod = await gateModule();
  const extractors = { page_text: () => "THE SENTINEL" };

  // (a) no astra.assist.state has ever arrived.
  {
    const { gate, clk, rec } = newGate(mod, { extractors });
    for (let i = 0; i < 20; i += 1) {
      clk.advance(1100);
      await within(gate.offer("screen"), 2000, "offer()");
    }
    assert.deepEqual(
      rec.sent,
      [],
      "criterion 5: with no astra.assist.state received the page sends no assist text at all — " +
        "not buffered, not queued, not 'just once at startup' (docs/01 D12a)",
    );
    assert.deepEqual(gate.flags(), Object.fromEntries(KINDS.map((k) => [k, false])));
  }

  // (b) a state message whose flags are all false.
  {
    const { gate, clk, rec } = newGate(mod, { extractors });
    gate.onState(stateMessage({}, "pixels"));
    for (let i = 0; i < 20; i += 1) {
      clk.advance(1100);
      await within(gate.offer("screen"), 2000, "offer()");
    }
    assert.deepEqual(
      rec.sent,
      [],
      "criterion 5: an astra.assist.state with every flag false is the all-off regime; the " +
        "extractor may return whatever it likes and none of it reaches send()",
    );
  }
});

test("gate adversarial: a click with no echo sends the toggle and never any text", opts, async () => {
  // The most likely defect in this task: the gate keys on optimistic local state (the checkbox
  // the user just ticked) instead of on the flags the server confirmed. Criterion 3/5.
  const mod = await gateModule();
  const extractors = { page_text: () => "THE SENTINEL" };
  const { gate, clk, rec } = newGate(mod, { extractors });

  assert.equal(
    gate.toggle("page_text", true),
    true,
    "criterion 3: a click on an available kind sends an astra.assist.toggle",
  );
  for (let i = 0; i < 20; i += 1) {
    clk.advance(1100);
    await within(gate.offer("screen"), 2000, "offer()");
  }

  assert.deepEqual(
    rec.ofType("astra.frame.text"),
    [],
    "criterion 5: on-ness comes from astra.assist.state, never from the click — with no echo " +
      "the kind is still off and no assist text may leave the page",
  );
  assert.equal(
    rec.ofType("astra.assist.toggle").length,
    1,
    `criterion 3: exactly one toggle goes out for one click, got ${JSON.stringify(rec.sent)}`,
  );
  assert.equal(gate.flags().page_text, false, "criterion 3: unechoed means off");
  assert.deepEqual(
    [...gate.pending()],
    ["page_text"],
    "criterion 3: the row is marked pending until the echo arrives",
  );

  gate.onState(stateMessage({ page_text: true }, "pixels+dom"));
  assert.equal(gate.flags().page_text, true);
  assert.deepEqual([...gate.pending()], [], "criterion 3: the echo clears pending");
});

// --- criteria 5 and 10: only confirmed kinds, and the message is the contract's ----------------

test("gate: only confirmed kinds are sent, and the message is the contract's", opts, async () => {
  const mod = await gateModule();
  const extractors = {
    page_text: () => "page text sentinel",
    cells: () => "A1,B2 — must never go out",
  };
  const { gate, clk, rec } = newGate(mod, { extractors });
  gate.onState(stateMessage({ page_text: true }, "pixels+dom"));
  clk.advance(5);
  const at = clk.now();
  await within(gate.offer("screen"), 2000, "offer()");

  const texts = rec.ofType("astra.frame.text");
  assert.equal(texts.length, 1, `criterion 5: only the confirmed kind, got ${JSON.stringify(rec.sent)}`);
  const msg = texts[0];
  assert.equal(msg.kind, "page_text", "criterion 5: cells was never confirmed");
  assert.equal(msg.source, "screen", "criterion 10: source is the live capture's source");
  assert.equal(msg.ts_ms, at, "criterion 10: ts_ms is the extraction time in ms (the injected now)");
  assert.equal(msg.text, "page text sentinel");
  const keys = Object.keys(msg).sort();
  assert.deepEqual(
    keys.filter((k) => k !== "url"),
    ["kind", "source", "text", "ts_ms", "type"],
    "criterion 10: contracts/base.Contract is extra='forbid' — an extra field on " +
      "astra.frame.text is a Malformed on the server and an astra.error back, not a warning; " +
      `the only optional field is url. Got ${keys.join(", ")}`,
  );

  const toggles = rec.ofType("astra.assist.toggle");
  assert.deepEqual(toggles, [], "no click happened, so no toggle");
  gate.toggle("page_text", false);
  const toggle = rec.ofType("astra.assist.toggle")[0];
  assert.deepEqual(
    Object.keys(toggle).sort(),
    ["flags", "type"],
    "criterion 10: AstraAssistToggle is { type, flags } and nothing else",
  );
  assert.deepEqual(
    Object.keys(toggle.flags).sort(),
    [...KINDS].sort(),
    "criterion 10: the toggle carries all four booleans explicitly — a whole flag set, never a " +
      "per-kind merge (T-0012 implementation notes)",
  );
  for (const [kind, value] of Object.entries(toggle.flags)) {
    assert.equal(typeof value, "boolean", `criterion 10: flags.${kind} is a boolean`);
  }
});

test("gate: the declared whitelist is exactly the two types the panel may emit", opts, async () => {
  const mod = await gateModule();
  assert.deepEqual(
    [...mod.ASSIST_CLIENT_TYPES].sort(),
    ["astra.assist.toggle", "astra.frame.text"],
    "criterion 10: the panel declares its own whitelist; protocol.js CLIENT_TYPES stays T-0008's " +
      "two types and is not widened (criterion 12)",
  );
  const { rec, gate, clk } = newGate(mod, { extractors: { page_text: () => "x" } });
  gate.onState(stateMessage({ page_text: true }));
  gate.toggle("page_text", false);
  clk.advance(1100);
  gate.onState(stateMessage({ page_text: true }));
  await within(gate.offer("screen"), 2000, "offer()");
  const emitted = [...new Set(rec.sent.map((m) => m.type))].sort();
  assert.ok(
    emitted.every((t) => mod.ASSIST_CLIENT_TYPES.includes(t)),
    `criterion 10: the panel emitted ${emitted.join(", ")}, which is not its own whitelist`,
  );
});

// --- criterion 6: unticking purges immediately, on the page too --------------------------------

test("gate adversarial: unticking drops text already extracted and held in flight", opts, async () => {
  const mod = await gateModule();
  let release = null;
  const extractors = {
    page_text: () =>
      new Promise((resolve) => {
        release = () => resolve("text extracted while the box was still ticked");
      }),
  };
  const { gate, clk, rec } = newGate(mod, { extractors });
  gate.onState(stateMessage({ page_text: true }, "pixels+dom"));

  const inFlight = gate.offer("screen"); // extraction starts while the kind is on
  await sleep(10);
  assert.ok(release, "the extractor must have been called while the kind was confirmed on");

  clk.advance(50);
  assert.equal(gate.toggle("page_text", false), true, "criterion 6: unticking sends the toggle");
  release();
  await within(inFlight, 2000, "the held offer");
  await sleep(20);

  assert.deepEqual(
    rec.ofType("astra.frame.text"),
    [],
    "criterion 6: text extracted while a kind was on must not reach send() after the untick, " +
      "even if the send was already scheduled (page-side mirror of T-0012 ruling A11)",
  );
  const toggle = rec.ofType("astra.assist.toggle").at(-1);
  assert.deepEqual(
    toggle.flags,
    { page_text: false, pdf_text: false, cells: false, at_spi: false },
    "criterion 6: the untick toggle sets page_text false and leaves the other three unchanged",
  );
  assert.equal(
    gate.flags().page_text,
    false,
    "criterion 6: extraction stops for that kind immediately, without waiting for the echo",
  );

  clk.advance(5000);
  await within(gate.offer("screen"), 2000, "offer() after the untick");
  assert.deepEqual(rec.ofType("astra.frame.text"), [], "criterion 6: and it stays stopped");
});

test("gate: unticking one kind leaves the others' confirmed flags alone", opts, async () => {
  const mod = await gateModule();
  const { gate, rec } = newGate(mod, {
    extractors: { page_text: () => "a", cells: () => "b" },
  });
  gate.onState(stateMessage({ page_text: true, cells: true }, "pixels+dom+cells"));
  gate.toggle("cells", false);
  assert.deepEqual(
    rec.ofType("astra.assist.toggle")[0].flags,
    { page_text: true, pdf_text: false, cells: false, at_spi: false },
    "criterion 10: the toggle is the whole flag set — the kinds the user did not touch keep " +
      "the value the server last confirmed",
  );
});

// --- criterion 7: rate-limited alongside frames, single flight, drop-not-queue -----------------

test("gate: at most one message per kind per 1000/MAX_FPS ms", opts, async () => {
  const mod = await gateModule();
  const frames = await load("frames.js");
  // Hand-computed oracle, stated for MAX_FPS = 1: offers at t = 0, 167, 334 ... 9853 ms (60 of
  // them over 10 s). A send is due when now - last >= 1000 ms, so they land at i = 0, 6, 12, 18,
  // 24, 30, 36, 42, 48, 54 -> exactly 10. i = 60 would be 10020 ms and there is no i = 60.
  assert.equal(
    frames.MAX_FPS,
    1,
    "this oracle was computed by hand for policy.MAX_FPS = 1; if the policy changes, recompute " +
      "the expected send count here rather than deriving it from the code under test",
  );
  const { gate, clk, rec } = newGate(mod, { extractors: { page_text: () => "same text" } });
  gate.onState(stateMessage({ page_text: true }));
  for (let i = 0; i < 60; i += 1) {
    if (i > 0) clk.advance(167);
    await within(gate.offer("screen"), 2000, `offer ${i}`);
  }
  assert.equal(
    rec.ofType("astra.frame.text").length,
    10,
    "criterion 7: at most one astra.frame.text per kind per 1000/MAX_FPS ms, where MAX_FPS is " +
      "imported from hud/src/frames.js — 60 offers over 10 s of simulated time is 10 sends",
  );
  assert.equal(gate.stats().sent, 10);
  assert.equal(
    gate.stats().sent + gate.stats().dropped,
    60,
    "criterion 7: every offer has an outcome and the two counters are displayed " +
      "(data-assist-sent / data-assist-dropped), exactly as the frame pipeline counts a frame " +
      "refused by the rate limit as dropped (hud/src/frames.js)",
  );
});

test("gate: a slow extractor drops, never queues, and is never doubled up", opts, async () => {
  const mod = await gateModule();
  let calls = 0;
  let release = null;
  let maxInFlight = 0;
  const extractors = {
    page_text: () => {
      calls += 1;
      maxInFlight = Math.max(maxInFlight, gate.stats().inFlight);
      if (calls === 1) return new Promise((resolve) => (release = () => resolve("slow text")));
      return Promise.resolve(`fast text ${calls}`);
    },
  };
  const clk = clock();
  const rec = recorder();
  const gate = mod.createAssistGate({
    send: rec.send,
    now: clk.now,
    extractors,
    reasons: {},
    log: () => {},
  });
  gate.onState(stateMessage({ page_text: true }));

  const first = gate.offer("screen");
  await sleep(10);
  const during = [];
  for (let i = 0; i < 5; i += 1) {
    clk.advance(600); // 3 s of extraction time in total, well past the rate window
    during.push(gate.offer("screen"));
  }
  await sleep(10);
  assert.equal(
    calls,
    1,
    "criterion 7: at no point are two extractions in flight for one kind; the extractor was " +
      `called ${calls} times while the first had not returned`,
  );
  assert.ok(gate.stats().inFlight <= 1, "criterion 7: single flight");
  release();
  await within(first, 2000, "the slow offer");
  await within(Promise.all(during), 2000, "the offers made while the extractor was slow");
  await sleep(20);

  assert.equal(
    calls,
    1,
    "criterion 7: a slow extractor causes a drop, never a queue — the five offers made while it " +
      `was working must not be replayed afterwards (the extractor ran ${calls} times)`,
  );
  assert.equal(
    rec.ofType("astra.frame.text").length,
    1,
    `criterion 7: exactly the one message, got ${rec.ofType("astra.frame.text").length}`,
  );
  assert.ok(maxInFlight <= 1, `criterion 7: in-flight per kind peaked at ${maxInFlight}`);
  assert.ok(gate.stats().dropped >= 5, `criterion 7: the five refused offers are counted`);
});

// --- criterion 16: never dedup ------------------------------------------------------------------

test("gate: never dedups identical text", opts, async () => {
  // AssistState.texts_for only returns text within ASSIST_HORIZON_MS (60 s) *of the turn's time*
  // (T-0012 ruling A9). A page that sent a stable DOM dump once would have that assist silently
  // vanish from the prompt 60 s later while the badge still said it was on. Re-send at the rate
  // limit; do not cache-and-skip.
  const mod = await gateModule();
  const { gate, clk, rec } = newGate(mod, {
    extractors: { page_text: () => "byte for byte the same text every time" },
  });
  gate.onState(stateMessage({ page_text: true }));
  for (let i = 0; i < 10; i += 1) {
    if (i > 0) clk.advance(1100);
    await within(gate.offer("screen"), 2000, `offer ${i}`);
  }
  const texts = rec.ofType("astra.frame.text");
  assert.equal(
    texts.length,
    10,
    "criterion 16: 10 offers over 9.9 s, each past the rate window, is 10 sends — suppressing a " +
      `resend because the text is unchanged starves the server's 60 s horizon (got ${texts.length})`,
  );
  assert.equal(new Set(texts.map((m) => m.text)).size, 1, "premise: the text really was identical");
  const gaps = texts.slice(1).map((m, i) => m.ts_ms - texts[i].ts_ms);
  assert.ok(
    gaps.every((g) => g >= 1000),
    `criterion 7 still holds while criterion 16 does: gaps were ${gaps}`,
  );
});

// --- criterion 9: the size cap, honestly marked -------------------------------------------------

function loneSurrogates(text) {
  const bad = [];
  for (let i = 0; i < text.length; i += 1) {
    const c = text.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff) {
      const next = text.charCodeAt(i + 1);
      if (!(next >= 0xdc00 && next <= 0xdfff)) bad.push(i);
      else i += 1;
    } else if (c >= 0xdc00 && c <= 0xdfff) {
      bad.push(i);
    }
  }
  return bad;
}

test("truncateUtf8: the cap is bytes and a short text is returned untouched", opts, async () => {
  const mod = await gateModule();
  assert.equal(mod.MAX_TEXT_BYTES, 32768, "criterion 9: the panel constant is 32 KiB");
  assert.ok(
    typeof mod.TRUNCATION_MARKER === "string" && mod.TRUNCATION_MARKER.length > 0,
    "criterion 9: the marker is in-band text the model can see (AstraFrameText has no " +
      "`truncated` field and adding one is a contract task)",
  );

  const short = "a short page of text";
  const kept = mod.truncateUtf8(short);
  assert.equal(kept.text, short, "criterion 9: nothing under the cap is touched");
  assert.equal(kept.truncated, false);
  assert.ok(
    !kept.text.includes(mod.TRUNCATION_MARKER),
    "criterion 9: an untruncated text must not claim to be truncated",
  );

  const long = "a".repeat(mod.MAX_TEXT_BYTES + 4096);
  const cut = mod.truncateUtf8(long);
  assert.equal(cut.truncated, true);
  assert.ok(
    utf8(cut.text) <= mod.MAX_TEXT_BYTES,
    `criterion 9: the cap is UTF-8 bytes, marker included; got ${utf8(cut.text)} bytes`,
  );
  assert.ok(
    cut.text.endsWith(mod.TRUNCATION_MARKER),
    "criterion 9: the marker sits at the end of the text, where the model reads it — a silent " +
      "truncation would let the model claim it read a whole document",
  );
});

test("truncateUtf8 adversarial: the cut is on a code-point boundary, not a UTF-16 unit", opts, async () => {
  // The same class of bug as T-0008 ruling R11: UTF-16 units are not bytes. '☃' is 3 bytes
  // and one unit; '𝄞' (U+1D11E) is 4 bytes and *two* units, so String.slice cuts it in
  // half and produces a lone surrogate, which TextEncoder then turns into U+FFFD on the wire.
  const mod = await gateModule();
  const MARKER_BYTES = utf8(mod.TRUNCATION_MARKER);

  // (1) aimed: the 4-byte character straddles the byte the cut must fall on.
  const budget = mod.MAX_TEXT_BYTES - MARKER_BYTES;
  const q = Math.floor((budget - 1) / 3); // leaves 1..3 bytes of room, never the 4 the astral needs
  const aimed = "☃".repeat(q) + "\u{1D11E}" + "☃".repeat(64);
  // (2) dense: every cut point in this string has a 8-in-10 chance of being mid-character.
  const dense = "☃☃\u{1D11E}".repeat(Math.ceil(mod.MAX_TEXT_BYTES / 10) + 64);

  for (const [name, input] of [
    ["aimed", aimed],
    ["dense", dense],
  ]) {
    const got = mod.truncateUtf8(input);
    assert.equal(got.truncated, true, `${name}: the input is over the cap`);
    assert.ok(
      utf8(got.text) <= mod.MAX_TEXT_BYTES,
      `criterion 9 (${name}): UTF-8 byte length must be <= MAX_TEXT_BYTES, got ${utf8(got.text)} ` +
        "— slicing UTF-16 units keeps ~3x the bytes",
    );
    assert.deepEqual(
      loneSurrogates(got.text),
      [],
      `criterion 9 (${name}): the cut is on a code-point boundary; a lone surrogate at the end ` +
        "is a half character the encoder replaces with U+FFFD",
    );
    assert.ok(
      !got.text.includes("�"),
      `criterion 9 (${name}): no replacement character in the truncated text`,
    );
    assert.ok(got.text.endsWith(mod.TRUNCATION_MARKER), `criterion 9 (${name}): marker in band`);
  }
});

test("gate: an over-cap extraction is truncated, marked and counted", opts, async () => {
  const mod = await gateModule();
  const huge = "☃".repeat(mod.MAX_TEXT_BYTES) + "\u{1D11E}";
  const { gate, rec } = newGate(mod, { extractors: { page_text: () => huge } });
  gate.onState(stateMessage({ page_text: true }));
  await within(gate.offer("screen"), 2000, "offer()");
  const msg = rec.ofType("astra.frame.text")[0];
  assert.ok(msg, "criterion 9: the message still goes out, truncated — it is not dropped silently");
  assert.ok(utf8(msg.text) <= mod.MAX_TEXT_BYTES, `criterion 9: ${utf8(msg.text)} bytes on the wire`);
  assert.ok(msg.text.endsWith(mod.TRUNCATION_MARKER), "criterion 9: the marker is on the wire");
  assert.deepEqual(loneSurrogates(msg.text), [], "criterion 9: no half character on the wire");
  assert.equal(gate.stats().truncated, 1, "criterion 9: the truncation is counted");
});

// --- criterion 4: the badge is the server's string, verbatim ------------------------------------

test("badge: the server's string is shown verbatim, flags and badge are not reconciled", opts, async () => {
  const mod = await gateModule();
  const { gate } = newGate(mod);
  assert.equal(gate.badge(), "", "criterion 4: before any astra.assist.state there is no badge " +
    "to show — the page does not invent one");

  // The wire permits a badge that disagrees with its flags (AstraAssistState validates neither
  // against the other). The page renders both as sent and reconciles nothing.
  gate.onState(stateMessage({ page_text: true }, "pixels"));
  assert.equal(
    gate.badge(),
    "pixels",
    "criterion 4: [data-assist-badge] is astra.assist.state.badge character for character; " +
      "AssistFlags.badge() lives in Python and there is exactly one derivation of it",
  );
  assert.equal(gate.flags().page_text, true, "criterion 4: and the server's flags are the flags");

  gate.onState(stateMessage({ page_text: true, cells: true }, "pixels+dom+cells"));
  assert.equal(gate.badge(), "pixels+dom+cells");
  gate.onState(stateMessage({}, ""));
  assert.equal(gate.badge(), "", "criterion 4: an empty badge from the server is an empty badge");
});

test("badge: no second derivation of the badge exists under hud/src/assists/", opts, async () => {
  // Building a badge from flags in JS is what a second derivation looks like, and the spellings
  // (dom, pdf, a11y) are not AssistFlags field names — they only appear in AssistFlags.badge().
  // "cells" is left out of this scan on purpose: it is a *kind* name as well as a badge spelling,
  // so it legitimately appears in gate.js. Building "pixels+cells" still needs "pixels+".
  const files = jsFilesUnder(ASSISTS_DIR);
  assert.ok(files.length > 0, "T-0025 adds hud/src/assists/{gate,index,extract}.js");
  const offenders = [];
  for (const f of files) {
    const body = stripComments(readFileSync(f, "utf8"));
    for (const spelling of ['"dom"', "'dom'", '"pdf"', "'pdf'", '"a11y"', "'a11y'", "pixels+"]) {
      if (body.includes(spelling)) offenders.push(`${f}: ${spelling}`);
    }
  }
  assert.deepEqual(
    offenders,
    [],
    "criterion 4: the panel contains no code that builds a badge from flags — it renders the " +
      "string the server sent (T-0012 criterion 8: exactly one derivation, in Python)",
  );
});

// --- criterion 8: a kind with no extractor cannot be turned on ----------------------------------

test("availability: with no extractor every kind is disabled, with a reason", opts, async () => {
  const mod = await gateModule();
  const { gate, rec } = newGate(mod, { extractors: {} });
  const available = gate.available();
  assert.deepEqual(
    Object.keys(available),
    KINDS,
    "criterion 2: one row per kind in AssistFlags field order, mirrored from " +
      "local_astra/contracts/perception.py",
  );
  for (const kind of KINDS) {
    assert.equal(
      available[kind].available,
      false,
      `criterion 8: ${kind} has no extractor registered for the live capture source`,
    );
    assert.ok(
      typeof available[kind].reason === "string" && available[kind].reason.trim().length >= 10,
      `criterion 8: ${kind} needs a human-readable reason, got ${JSON.stringify(available[kind].reason)}`,
    );
  }

  for (const kind of KINDS) {
    gate.toggle(kind, true);
    gate.toggle(kind, true); // a second click is a second activation path
  }
  const raising = rec
    .ofType("astra.assist.toggle")
    .filter((m) => KINDS.some((k) => m.flags[k] === true));
  assert.deepEqual(
    raising,
    [],
    "criterion 8: no click path on an unavailable row produces an astra.assist.toggle that sets " +
      "its flag true — the checkbox is disabled and the page must not let the user tick a box " +
      "into a regime it cannot honour",
  );
  assert.deepEqual(gate.flags(), Object.fromEntries(KINDS.map((k) => [k, false])));
});

test("availability: an available kind is the one with an extractor, and only it", opts, async () => {
  const mod = await gateModule();
  const { gate } = newGate(mod, { extractors: { page_text: () => "x" } });
  const available = gate.available();
  assert.equal(available.page_text.available, true, "criterion 8: an extractor makes a kind real");
  for (const kind of ["pdf_text", "cells", "at_spi"]) {
    assert.equal(available[kind].available, false);
    assert.ok(
      available[kind].reason.trim().length >= 10,
      `criterion 8: ${kind}'s reason names why (a browser tab cannot read another tab's DOM; ` +
        "those kinds arrive with the native daemon, docs/01 D6, M2+)",
    );
  }
});

test("registry: v1 registers exactly one extractor, page_text over the synthetic canvas", opts, async () => {
  const extract = await load("assists/extract.js");
  const synthetic = extract.registryFor({ source: "screen", synthetic: true, doc: null });
  assert.deepEqual(
    Object.keys(synthetic.extractors),
    ["page_text"],
    "criterion 8: v1 registers exactly one extractor — page_text over the synthetic canvas " +
      "source (?source=canvas), whose text the page itself drew and therefore genuinely knows",
  );
  for (const capture of [
    { source: "screen", synthetic: false },
    { source: "window", synthetic: false },
    { source: "camera", synthetic: false },
  ]) {
    const registry = extract.registryFor({ ...capture, doc: null });
    assert.deepEqual(
      Object.keys(registry.extractors),
      [],
      `criterion 8: page_text over a real getDisplayMedia/getUserMedia stream (${capture.source}) ` +
        "registers no extractor: getDisplayMedia hands the page a MediaStreamTrack and nothing " +
        "else, so a tab cannot read another tab's DOM (planner's finding 1)",
    );
    for (const kind of KINDS) {
      assert.ok(
        typeof registry.reasons[kind] === "string" && registry.reasons[kind].trim().length >= 10,
        `criterion 8: ${kind} needs a reason for ${capture.source}, got ` +
          JSON.stringify(registry.reasons[kind]),
      );
    }
  }
});

test("gate: a confirmed kind with no extractor sends nothing", opts, async () => {
  // Availability and confirmation are independent: the server may confirm a kind this client
  // cannot honour. That must be silence, not an empty or invented astra.frame.text.
  const mod = await gateModule();
  const { gate, clk, rec } = newGate(mod, { extractors: {} });
  gate.onState(stateMessage({ page_text: true, pdf_text: true, cells: true, at_spi: true }, "pixels+dom"));
  for (let i = 0; i < 5; i += 1) {
    clk.advance(1100);
    await within(gate.offer("screen"), 2000, "offer()");
  }
  assert.deepEqual(rec.sent, [], "criterion 8: no extractor, no text — and nothing invented");
});

// --- criterion 1: it mounts through the seam and cannot break the page --------------------------

test("mount({}) does not throw, renders nothing and sends nothing", opts, async () => {
  const index = await load("assists/index.js");
  assert.equal(
    typeof index.mount,
    "function",
    "criterion 1: hud/src/assists/index.js exports mount(host) — the contract panels.js calls",
  );
  let result;
  await assert.doesNotReject(
    async () => {
      result = await index.mount({});
    },
    "criterion 1: a host that predates this seam (no document, no send, no subscribe) degrades " +
      "instead of taking the socket down — T-0008 criterion 10: 'a loader that throws is caught " +
      "and leaves every other panel and the socket working'",
  );
  assert.ok(result === undefined || typeof result === "object");
});

test("mount(host) subscribes through the seam and sends nothing on its own", opts, async () => {
  // Criterion 13: host gains send (the raw socket.send) and subscribe (a registry called with
  // every decoded astra.* message). Mounting must not put a message on the wire by itself: the
  // first astra.frame.text may only follow a user click *and* a server echo (criterion 5).
  const index = await load("assists/index.js");
  const rec = recorder();
  const handlers = [];
  const host = {
    document: undefined,
    send: rec.send,
    subscribe: (handler) => {
      handlers.push(handler);
      return () => {};
    },
    log: () => {},
  };
  await assert.doesNotReject(async () => index.mount(host), "criterion 1: mount never throws");
  assert.deepEqual(rec.sent, [], "criterion 5: mounting sends nothing — not even a hello toggle");
  for (const handler of handlers) {
    assert.doesNotThrow(
      () => handler(stateMessage({ page_text: true }, "pixels+dom")),
      "criterion 1: a panel with no document must survive an inbound message",
    );
    assert.doesNotThrow(
      () => handler({ type: "astra.turn.state", state: "THINKING", turn_id: "t1" }),
      "criterion 1: subscribers see every astra.* message, not only their own",
    );
  }
  assert.deepEqual(rec.sent, [], "criterion 5: and still nothing goes out");
});

// --- T-0162 group B: the assist panel moves behind the Settings menu (B-6) ------------------------
//
// docs/tasks/T-0162.md B-6: "assists/index.js mounts into host.settingsSlot when it is given one,
// else into main as before"; the popover holds "the same #assist-panel element, data-assist-*
// hooks and gate behaviour; the heading 'Context helpers'; the plain sentence above, as a
// paragraph". `host.settingsSlot` is read as an element (the task says "mounts into"; a function
// like `cardSlot` would also be a defensible reading — the test-author's choice, named here).
//
// The rows run through the T-0162 group B Status-phrase gate (the same mechanism as pendingT0159B
// in tests/hud/denoise.test.mjs): strict xfail until docs/tasks/T-0162.md's **Status:** records
// "group B: tests …, activated <sha>". The "else into main as before" row is ungated: it holds
// today and must keep holding.

const T0162_TASK = fileURLToPath(new URL("docs/tasks/T-0162.md", REPO));
const T0162_STATUS = existsSync(T0162_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0162_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0162_B_ACTIVATED = /\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const T0162_B_PENDING = !T0162_B_ACTIVATED.test(T0162_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0162B(name, fn) {
  test(name + (T0162_B_PENDING ? " [pending T-0162 B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!T0162_B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0162 group B row passed while docs/tasks/T-0162.md's status does " +
          "not record group B's activation (" + JSON.stringify(T0162_STATUS.slice(0, 80)) + ").",
      );
    }
    t.diagnostic("expected failure (T-0162 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

/** D12a's plain sentence (docs/tasks/T-0162.md Goal); curly apostrophes compare as straight. */
const D12A_SENTENCE =
  "Gearvis reads your screen from pixels only; these optional helpers also send it the page's " +
  "text, a PDF's text layer, spreadsheet cells or the accessibility tree. They are all off by " +
  "default, and every turn records which ones were on.";
const plain = (s) => String(s).replace(/[‘’]/g, "'").replace(/\s+/g, " ").trim();

/** A small permissive DOM double: enough for the panel's buildPanel (no layout, no events fired). */
function miniDocument() {
  const el = (tag) => {
    const kids = [];
    const attrs = new Map();
    const listeners = [];
    const node = {
      nodeType: 1,
      localName: tag,
      tagName: tag.toUpperCase(),
      id: "",
      className: "",
      title: "",
      type: "",
      disabled: false,
      checked: false,
      parentNode: null,
      childNodes: kids,
      get children() {
        return kids.filter((k) => k.nodeType === 1);
      },
      get textContent() {
        return kids.map((k) => k.textContent).join("");
      },
      set textContent(v) {
        kids.length = 0;
        if (v !== "" && v !== null && v !== undefined) kids.push({ nodeType: 3, textContent: String(v) });
      },
      setAttribute: (n, v) => {
        attrs.set(String(n), String(v));
        if (n === "id") node.id = String(v);
      },
      getAttribute: (n) => (n === "id" ? node.id || null : attrs.has(n) ? attrs.get(n) : null),
      hasAttribute: (n) => attrs.has(n),
      removeAttribute: (n) => attrs.delete(n),
      appendChild: (c) => {
        if (typeof c === "string") c = { nodeType: 3, textContent: c };
        if (c.parentNode && c.parentNode.childNodes) {
          const i = c.parentNode.childNodes.indexOf(c);
          if (i >= 0) c.parentNode.childNodes.splice(i, 1);
        }
        c.parentNode = node;
        kids.push(c);
        return c;
      },
      append: (...cs) => cs.forEach((c) => node.appendChild(c)),
      addEventListener: (type, fn) => listeners.push([type, fn]),
      removeEventListener() {},
      _listeners: listeners,
      contains: (other) => {
        for (let n = other; n; n = n.parentNode) if (n === node) return true;
        return false;
      },
      all: () => kids.filter((k) => k.nodeType === 1).flatMap((k) => [k, ...k.all()]),
      querySelector: (sel) => node.all().find((k) => k.localName === sel || "#" + k.id === sel) || null,
    };
    return node;
  };
  const html = el("html");
  const head = el("head");
  const body = el("body");
  const main = el("main");
  html.append(head, body);
  body.appendChild(main);
  return {
    head,
    body,
    main,
    createElement: (t) => el(String(t).toLowerCase()),
    createTextNode: (t) => ({ nodeType: 3, textContent: String(t) }),
    querySelector: (sel) => html.querySelector(sel),
    getElementById: (id) => html.querySelector("#" + id),
    addEventListener() {},
  };
}

async function mountWith(doc, extra = {}) {
  const index = await load("assists/index.js");
  const realSetInterval = globalThis.setInterval;
  globalThis.setInterval = () => 0; // the offer tick would keep node alive; nothing is offered here
  try {
    await index.mount({ document: doc, subscribe: () => () => {}, log: () => {}, ...extra });
  } finally {
    globalThis.setInterval = realSetInterval;
  }
}

test("T-0162 B-6 (kept): without host.settingsSlot the panel mounts into <main> as before", opts, async () => {
  const doc = miniDocument();
  await mountWith(doc);
  const panel = doc.getElementById("assist-panel");
  assert.ok(panel, "the #assist-panel element is built");
  assert.equal(panel.parentNode, doc.main, "B-6: with no settingsSlot, into main as before");
});

pendingT0162B("T-0162 B-6: with host.settingsSlot the same #assist-panel mounts into the slot, not into <main>", async () => {
  const doc = miniDocument();
  const slot = doc.createElement("div");
  doc.body.appendChild(slot);
  await mountWith(doc, { settingsSlot: slot });
  const panel = doc.getElementById("assist-panel");
  assert.ok(panel, "B-6: the same #assist-panel id (its own <style> and the tests follow it)");
  assert.ok(slot.contains(panel), "B-6: the panel is inside host.settingsSlot");
  assert.ok(!doc.main.contains(panel), "B-6: …and no longer in <main> (the rail)");
  const kinds = panel.all().map((e) => e.getAttribute("data-assist-kind")).filter(Boolean);
  assert.deepEqual(kinds, KINDS, "B-6: the same data-assist-* rows, in AssistFlags order");
  assert.ok(panel.all().some((e) => e.hasAttribute("data-assist-badge")), "B-6: the badge hook stays");
});

pendingT0162B("T-0162 B-6: the moved panel is headed 'Context helpers' and carries D12a's sentence as a paragraph", async () => {
  const doc = miniDocument();
  const slot = doc.createElement("div");
  doc.body.appendChild(slot);
  await mountWith(doc, { settingsSlot: slot });
  const panel = doc.getElementById("assist-panel");
  assert.ok(panel && slot.contains(panel), "premise: the panel is in the slot");
  const headings = panel.all().filter((e) => /^h[1-6]$/.test(e.localName)).map((e) => plain(e.textContent));
  assert.deepEqual(headings, ["Context helpers"], "B-6: the heading 'Context helpers'");
  const paras = panel.all().filter((e) => e.localName === "p").map((e) => plain(e.textContent));
  assert.ok(paras.includes(plain(D12A_SENTENCE)), `B-6: the plain sentence as a paragraph; paragraphs: ${JSON.stringify(paras)}`);
});

pendingT0162B("T-0162 B-6: the slot-mounted switches keep the gate behaviour (a change sends one whole-set toggle)", async () => {
  const doc = miniDocument();
  const slot = doc.createElement("div");
  doc.body.appendChild(slot);
  const rec = recorder();
  // ?source=canvas: the synthetic capture, the one v1 source where page_text has an extractor.
  const hadLocation = "location" in globalThis;
  const realLocation = globalThis.location;
  globalThis.location = { search: "?source=canvas" };
  try {
    await mountWith(doc, { settingsSlot: slot, send: rec.send });
  } finally {
    if (hadLocation) globalThis.location = realLocation;
    else delete globalThis.location;
  }
  const panel = doc.getElementById("assist-panel");
  assert.ok(panel && slot.contains(panel), "premise: the panel is in the slot");
  const row = panel.all().find((e) => e.getAttribute("data-assist-kind") === "page_text");
  const box = row.all().find((e) => e.localName === "input");
  assert.ok(box, "premise: the page_text row has its checkbox");
  box.checked = true;
  for (const [type, fn] of box._listeners) if (type === "change") fn({ target: box });
  const toggles = rec.ofType("astra.assist.toggle");
  assert.equal(toggles.length, 1, "B-6: one toggle per change, through the same gate");
  assert.deepEqual(toggles[0].flags, { page_text: true, pdf_text: false, cells: false, at_spi: false }, "B-6: the whole flag set");
});
