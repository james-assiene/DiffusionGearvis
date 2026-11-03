// T-0008 criteria 1 and 2 — hud/src/protocol.js: hello, the client whitelist, and a decode that
// is *total* in exactly the way local_astra/contracts/codec.decode is.
//
// The corpus is tests/fixtures/hud/decode_corpus.jsonl, shared with
// tests/unit/test_hud_constants_parity.py, which checks every declared outcome against the real
// Python codec. So this file compares the page to the contract, never to a second copy of itself.
//
// T-0027: the corpus also carries the multibyte payloads that pin the *byte* limit (the euro and
// astral rows, and the two rows on either side of exactly 8,388,608 bytes). They are built at
// read time by rawOf()/buildRaw() below, in the same way build_raw() builds them in Python, and
// are checked against the real Python codec by tests/unit/test_hud_constants_parity.py. The page
// is already right (T-0008 ruling R11), so those rows are green here from the first commit; it
// was contracts/codec.py that counted code points. The inline R11 test further down stays: it
// also pins the ordering of the size check against JSON.parse, which a corpus row cannot.
//
// API this pins (docs/tasks/T-0008.md leaves the JS names to the implementer; these are them):
//   decode(raw) -> { kind: "astra" | "unknown" | "realtime" | "malformed", type?, msg?, error? }
//   hello() -> the astra.hello object the page sends first
//   PROTOCOL_VERSION, MAX_MESSAGE_BYTES, SERVER_TO_CLIENT (iterable of type strings)
//   CLIENT_TYPES — the only client->server types this task's code may emit
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";
import { join, relative, sep } from "node:path";

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

const CORPUS = JSON.parse(
  "[" +
    readFileSync(fileURLToPath(new URL("tests/fixtures/hud/decode_corpus.jsonl", REPO)), "utf8")
      .split("\n")
      .filter((l) => l.trim())
      .join(",") +
    "]",
);

// T-0030 group C adds astra.spoken.played to the page's dispatch table and its field list
// (criterion 21). The sentinel is hud/src/state.js — group C lands state.js and protocol.js
// together — so the row below pins the nine-type list before that merge and the ten-type list
// after it, and never derives the expectation from protocol.js itself. The hard cross-language
// pin is tests/unit/test_hud_constants_parity.py, which compares this page's list with
// contracts/protocol.py SERVER_TO_CLIENT and is red on purpose between the group A and group C
// merges (docs/tasks/T-0030.md implementation notes).
const PLAYED_INSTALLED = (() => {
  const state = fileURLToPath(new URL("hud/src/state.js", REPO));
  return existsSync(state) && readFileSync(state, "utf8").includes("astra.spoken.played");
})();
const playedOpts = PLAYED_INSTALLED
  ? {}
  : {
      skip:
        "T-0030 group C is not implemented yet: hud/src/state.js does not mention " +
        "astra.spoken.played (docs/tasks/T-0030.md criteria 17-21). Node tests cannot carry " +
        "pytest's `pending` marker and tests/unit/test_hud_js.py requires `node --test " +
        "tests/hud` to exit 0.",
    };

// T-0043 criterion 20: the page's SERVER_TO_CLIENT gains astra.log.state. The sentinel is
// protocol.js's own source, for the same reason T-0030's is state.js: this file must not redden
// the fast suite before the task lands, and the hard cross-language pin (the page's list equals
// contracts/protocol.py SERVER_TO_CLIENT, field table included) is
// tests/unit/test_hud_constants_parity.py.
const LOG_STATE_INSTALLED = (() => {
  const file = fileURLToPath(new URL("hud/src/protocol.js", REPO));
  return existsSync(file) && readFileSync(file, "utf8").includes("astra.log.state");
})();
const logStateOpts = LOG_STATE_INSTALLED
  ? {}
  : {
      skip:
        "T-0043 group B is not implemented yet: hud/src/protocol.js does not mention " +
        "astra.log.state (docs/tasks/T-0043.md criterion 20). Node tests cannot carry pytest's " +
        "`pending` marker and tests/unit/test_hud_js.py requires `node --test tests/hud` to exit 0.",
    };

const MAX_MESSAGE_BYTES = 8 * 1024 * 1024; // local_astra/contracts/codec.py

const utf8len = (s) => new TextEncoder().encode(s).length;

/**
 * Build a corpus payload from a `build` spec — the JS half of the builder contract written out
 * in tests/fixtures/hud/MANIFEST.md; the Python half is build_raw() in
 * tests/unit/test_hud_constants_parity.py. The multibyte payloads are 8-10 MB, so they are built
 * at read time on both sides rather than stored in git.
 *
 * `prefix + char.repeat(count) + suffix`, where exactly one of `count` / `fill_to_bytes` gives
 * the count; with `fill_to_bytes` it is
 * `(fill_to_bytes + delta_bytes - utf8len(prefix) - utf8len(suffix)) / utf8len(char)` and must
 * divide exactly, and `fill_to_bytes` must equal MAX_MESSAGE_BYTES so the boundary rows move
 * with the constant instead of quietly testing some other boundary.
 *
 * Anything else throws (docs/tasks/T-0027.md criterion 12): decode(undefined) is
 * {kind:"malformed"}, so a builder that returned undefined for a spec it did not understand
 * would make every "expect":"malformed" row pass for the wrong reason.
 */
function buildRaw(spec, name = "?") {
  if (!spec || typeof spec !== "object" || Array.isArray(spec))
    throw new Error(`corpus case ${name}: 'build' must be an object`);
  const known = ["prefix", "char", "suffix", "count", "fill_to_bytes", "delta_bytes"];
  const unknown = Object.keys(spec).filter((k) => !known.includes(k));
  if (unknown.length)
    throw new Error(
      `corpus case ${name}: unknown build key(s) ${unknown} (the builder contract in ` +
        `tests/fixtures/hud/MANIFEST.md allows ${known})`,
    );
  for (const k of ["prefix", "char", "suffix"])
    if (typeof spec[k] !== "string") throw new Error(`corpus case ${name}: build.${k} is missing`);
  if (spec.char === "") throw new Error(`corpus case ${name}: build.char must be non-empty`);
  const hasCount = "count" in spec;
  const hasFill = "fill_to_bytes" in spec;
  if (hasCount === hasFill)
    throw new Error(`corpus case ${name}: exactly one of 'count' / 'fill_to_bytes'`);
  let count;
  if (hasCount) {
    if ("delta_bytes" in spec)
      throw new Error(`corpus case ${name}: 'delta_bytes' needs 'fill_to_bytes'`);
    count = spec.count;
  } else {
    if (spec.fill_to_bytes !== MAX_MESSAGE_BYTES)
      throw new Error(
        `corpus case ${name}: fill_to_bytes is ${spec.fill_to_bytes} but MAX_MESSAGE_BYTES is ` +
          `${MAX_MESSAGE_BYTES}; the boundary rows must move with the constant`,
      );
    const room =
      spec.fill_to_bytes + (spec.delta_bytes || 0) - utf8len(spec.prefix) - utf8len(spec.suffix);
    const unit = utf8len(spec.char);
    if (room < 0 || room % unit !== 0)
      throw new Error(
        `corpus case ${name}: ${room} bytes of room does not divide by ${unit} bytes per char ` +
          `(the count must be exact, not floored)`,
      );
    count = room / unit;
  }
  return spec.prefix + spec.char.repeat(count) + spec.suffix;
}

function rawOf(c) {
  if (c.oversized)
    return (
      '{"type":"astra.transcript","turn_id":"t","text":"' +
      "x".repeat(MAX_MESSAGE_BYTES + 64) +
      '"}'
    );
  if ("build" in c) return buildRaw(c.build, c.name);
  if (typeof c.raw !== "string")
    throw new Error(`corpus case ${c.name}: neither a string 'raw' nor a 'build' spec`);
  return c.raw;
}

function jsFilesUnder(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) out.push(...jsFilesUnder(p));
    else if (p.endsWith(".js") || p.endsWith(".mjs") || p.endsWith(".html")) out.push(p);
  }
  return out;
}

function stripComments(text) {
  return text
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/(^|[^:])\/\/[^\n]*/g, "$1 ")
    .replace(/<!--[\s\S]*?-->/g, " ");
}

// --- the T-0128c group B strict-xfail gate ------------------------------------------------------
//
// T-0128c B1 re-mirrors astra.canvas.step from AstraCanvasStep (Finding 1: today's mirror rejects
// every real tap frame). Its corpus rows carry "pending_js": "T-0128c": the Python half of
// tests/unit/test_hud_constants_parity.py checks them against the real codec now (they pass there,
// which is the point: the Python side is the spec), while this page-side half runs them through
// `pendingT0128cB` until docs/tasks/T-0128c.md's **Status:** records group B's activation
// ("group B: tests <a>..<b>, activated <sha>"). Same gate as tests/hud/denoise.test.mjs;
// `ASTRA_RUN_PENDING=1` runs them for real now.
const T0128C_TASK = fileURLToPath(new URL("docs/tasks/T-0128c.md", REPO));
const T0128C_STATUS = existsSync(T0128C_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0128C_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0128C_B_ACTIVATED = /\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const T0128C_B_PENDING = !T0128C_B_ACTIVATED.test(T0128C_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0128cB(name, fn) {
  test(name + (T0128C_B_PENDING ? " [pending T-0128c B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!T0128C_B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0128c group B row passed while docs/tasks/T-0128c.md's status does " +
          "not record group B's activation. Either group B has landed (record 'group B: tests …, " +
          "activated <sha>' in the status) or the row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0128c B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

// --- the T-0159 group A strict-xfail gate --------------------------------------------------------
//
// T-0159 A1/A6 add argmax_pieces / committed_pieces to astra.canvas.step. The corpus rows that only
// decode once the fields exist carry "pending": "T-0159" (the Python loader turns that key into the
// pending marker). This half reads the same key: while any corpus row still carries it, the rows
// below run as strict xfail (a row that passes is XPASS and fails), so removing the key from the
// corpus is the one activation step for both languages. `ASTRA_RUN_PENDING=1` runs them for real.
const T0159_A_PENDING = CORPUS.some((c) => c.pending === "T-0159") && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0159A(name, fn) {
  test(name + (T0159_A_PENDING ? " [pending T-0159 A: strict xfail while corpus rows carry pending T-0159]" : ""), async (t) => {
    if (!T0159_A_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0159 group A row passed while tests/fixtures/hud/decode_corpus.jsonl " +
          'still carries "pending": "T-0159". Either group A has landed (remove the key from the ' +
          "corpus rows) or the row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0159 A pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

// --- the T-0162 group A strict-xfail gate --------------------------------------------------------
//
// T-0162 A-1 adds astra.session.cleared to SERVER_TO_CLIENT and its field table to protocol.js. The
// rows below run as strict xfail until docs/tasks/T-0162.md's **Status:** records group A's
// activation ("group A: tests <a>..<b>, activated <sha>"; group B's own entry does not count).
// `ASTRA_RUN_PENDING=1` runs them for real now.
const T0162_TASK = fileURLToPath(new URL("docs/tasks/T-0162.md", REPO));
const T0162_STATUS = existsSync(T0162_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0162_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0162_A_ACTIVATED = /\bgroup A:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const T0162_A_PENDING = !T0162_A_ACTIVATED.test(T0162_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0162A(name, fn) {
  test(name + (T0162_A_PENDING ? " [pending T-0162 A: strict xfail until group A is activated]" : ""), async (t) => {
    if (!T0162_A_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0162 group A row passed while docs/tasks/T-0162.md's status does " +
          "not record group A's activation. Either group A has landed (record 'group A: tests …, " +
          "activated <sha>' in the status) or the row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0162 A pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

// --- the T-0180 group A strict-xfail gate --------------------------------------------------------
//
// T-0180 A1/A2 add astra.thought (the turn's committed thought text, for the "Model thoughts"
// disclosure). Its corpus rows carry "pending": "T-0180" (the Python loader turns that key into the
// pending marker). As for T-0159: while any corpus row still carries the key, the T-0180 rows below
// run as strict xfail, so removing the key from the corpus is the one activation step for both
// languages. `ASTRA_RUN_PENDING=1` runs them for real now.
const T0180_A_PENDING = CORPUS.some((c) => c.pending === "T-0180") && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0180A(name, fn) {
  test(name + (T0180_A_PENDING ? " [pending T-0180 A: strict xfail while corpus rows carry pending T-0180]" : ""), async (t) => {
    if (!T0180_A_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0180 group A row passed while tests/fixtures/hud/decode_corpus.jsonl " +
          'still carries "pending": "T-0180". Either group A has landed (remove the key from the ' +
          "corpus rows) or the row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0180 A pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

// --- the T-0181 group A strict-xfail gate --------------------------------------------------------
//
// T-0181 A1 adds astra.mic.mute (client->server) and astra.mic.state (server->client) to
// contracts/protocol.py and their field tables to hud/src/protocol.js. The rows below run as strict
// xfail while tests/contract/test_t0181_mic_wire.py (the Python half of A1) still carries its
// pending(task="T-0181") marker, so removing that marker is the one activation step for both
// languages. `ASTRA_RUN_PENDING=1` runs them for real now.
const T0181_WIRE = fileURLToPath(new URL("tests/contract/test_t0181_mic_wire.py", REPO));
const T0181_A_PENDING =
  existsSync(T0181_WIRE) &&
  readFileSync(T0181_WIRE, "utf8").includes('pending(task="T-0181")') &&
  process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0181A(name, fn) {
  test(name + (T0181_A_PENDING ? " [pending T-0181 A: strict xfail while test_t0181_mic_wire.py is pending]" : ""), async (t) => {
    if (!T0181_A_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0181 group A row passed while tests/contract/test_t0181_mic_wire.py " +
          'still carries pending(task="T-0181"). Either group A has landed (remove the marker) or the ' +
          "row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0181 A pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

/** A corpus row this page-side half holds back while its task is pending. */
const heldBack = (c) =>
  (c.pending_js === "T-0128c" && T0128C_B_PENDING) ||
  (c.pending === "T-0159" && T0159_A_PENDING) ||
  (c.pending === "T-0180" && T0180_A_PENDING);

test("criterion 2: every corpus message reaches the outcome the real codec gives it", async () => {
  const { decode } = await load("protocol.js", "criterion 2");
  const wrong = [];
  for (const c of CORPUS) {
    if (heldBack(c)) continue; // run below, through the T-0128c B gate
    let got;
    try {
      got = decode(rawOf(c));
    } catch (err) {
      wrong.push(`${c.name}: decode threw ${err} (decode is total and never throws)`);
      continue;
    }
    if (!got || got.kind !== c.expect) {
      wrong.push(`${c.name}: expected kind ${c.expect}, got ${got && got.kind}`);
    } else if (c.type !== null && c.type !== undefined && got.type !== c.type) {
      wrong.push(`${c.name}: expected type ${c.type}, got ${got.type}`);
    }
  }
  assert.deepEqual(wrong, [], `criterion 2 (the four decode outcomes):\n${wrong.join("\n")}`);
});

test("criterion 12: an unbuildable corpus case throws rather than decoding as malformed", async () => {
  // docs/tasks/T-0027.md criterion 12: "a corpus row nobody can build fails loudly, on both
  // sides". The trap it closes is one line below: decode of a payload the builder failed to
  // produce is malformed, which is what most rows declare.
  const { decode } = await load("protocol.js", "criterion 2");
  assert.equal(decode(undefined).kind, "malformed", "the outcome a silent builder would fake");

  assert.throws(() => rawOf({ name: "unknown_key", build: { nonsense: 1 } }), /unknown build key/);
  assert.throws(() => rawOf({ name: "no_raw_no_build" }), /neither a string/);
  const boundary = {
    prefix: '{"type":"astra.transcript","turn_id":"t","text":"',
    char: "x",
    suffix: '"}',
    fill_to_bytes: MAX_MESSAGE_BYTES,
  };
  assert.throws(() => rawOf({ name: "both", build: { ...boundary, count: 3 } }), /exactly one/);
  assert.throws(
    () => rawOf({ name: "stale", build: { ...boundary, fill_to_bytes: 1024 } }),
    /must move with the constant/,
  );
  // 8_388_608 - 49 - 2 = 8_388_557 bytes of room, which is not a multiple of 3: a floored count
  // would build a payload two bytes short of the boundary it claims to sit on.
  assert.throws(
    () => rawOf({ name: "euro_boundary", build: { ...boundary, char: "\u20ac" } }),
    /does not divide/,
  );
  // ... and the four real rows do build, to the byte counts docs/tasks/T-0027.md computes.
  const sizes = {};
  // T-0180 A2: the thought_ build rows sit on the 16000-code-point thought cap, not on the byte
  // limit; their counts are pinned by the T-0180 rows at the end of this module.
  for (const c of CORPUS) if (c.build && !c.name.startsWith("thought_")) sizes[c.name] = utf8len(rawOf(c));
  assert.deepEqual(sizes, {
    multibyte_oversized_euro: 9000051,
    astral_oversized_defeats_a_times_three_shortcut: 10000051,
    ascii_exactly_at_the_limit: 8388608,
    ascii_one_byte_over_the_limit: 8388609,
  });
});

test("criterion 2 adversarial: valid JSON of a known type with a bad field is malformed", async () => {
  const { decode } = await load("protocol.js", "criterion 2");
  const raw =
    '{"type":"astra.frame","source":"screen","ts_ms":1700000000000,"width":0,"height":882,"jpeg_b64":"/9j/"}';
  const got = decode(raw);
  assert.equal(
    got.kind,
    "malformed",
    "AstraFrame.width has ge=1: width 0 must be classified malformed (counted, ignored), " +
      "never dispatched as a typed astra.frame — a dispatched half-valid message is the silent one",
  );
});

test("criterion 2: a message over MAX_MESSAGE_BYTES is malformed without being parsed", async () => {
  const { decode, MAX_MESSAGE_BYTES: MAX } = await load("protocol.js", "criterion 2");
  assert.equal(MAX, MAX_MESSAGE_BYTES, "the page mirrors codec.MAX_MESSAGE_BYTES");
  const raw =
    '{"type":"astra.spoken","turn_id":"t","utterance_id":"u","text":"' + "x".repeat(MAX + 64) + '"}';
  const realParse = JSON.parse;
  const seen = [];
  JSON.parse = (s, ...rest) => {
    seen.push(typeof s === "string" ? s.length : -1);
    return realParse(s, ...rest);
  };
  try {
    const got = decode(raw);
    assert.equal(got.kind, "malformed");
  } finally {
    JSON.parse = realParse;
  }
  assert.ok(
    !seen.some((n) => n > MAX),
    `criterion 2: the size check comes before JSON.parse (parsed lengths: ${seen})`,
  );
});

test("criterion 2 (ruling R11): the oversize guard counts UTF-8 bytes, not UTF-16 units", async () => {
  const { decode, MAX_MESSAGE_BYTES: MAX } = await load("protocol.js", "criterion 2");
  // "€" is one UTF-16 unit and three UTF-8 bytes, so `raw.length > MAX` lets a message of up to
  // ~3x the limit through — bytes the socket really carried and the Conductor would call
  // Malformed. Any text channel (a transcript, an error message) can be all multibyte.
  const payload = "€".repeat(3_000_000);
  const raw = '{"type":"astra.transcript","turn_id":"t","text":"' + payload + '"}';
  assert.ok(raw.length < MAX, "premise: under the limit counted in UTF-16 units");
  assert.ok(
    new TextEncoder().encode(raw).length > MAX,
    "premise: over the limit counted in the bytes the codec counts",
  );
  const realParse = JSON.parse;
  const seen = [];
  JSON.parse = (s, ...rest) => {
    seen.push(typeof s === "string" ? s.length : -1);
    return realParse(s, ...rest);
  };
  try {
    assert.equal(
      decode(raw).kind,
      "malformed",
      "criterion 2: a message over codec.MAX_MESSAGE_BYTES is Malformed — the limit is bytes " +
        "(contracts/codec.py measures len(raw) on bytes and on str only after utf-8 decoding)",
    );
  } finally {
    JSON.parse = realParse;
  }
  assert.ok(
    !seen.some((n) => n >= payload.length),
    `ruling R11: the byte-length guard runs before JSON.parse (parsed lengths: ${seen})`,
  );
});

test("criterion 2: decode never throws, whatever arrives on the socket", async () => {
  const { decode } = await load("protocol.js", "criterion 2");
  const nasty = [
    "",
    " ",
    "null",
    "123",
    '"a string"',
    "[1,2,3]",
    "{",
    '{"type":null}',
    '{"type":{"a":1}}',
    '{"type":"astra."}',
    String.fromCharCode(0, 1),
    undefined,
    null,
    new ArrayBuffer(8),
    new Uint8Array([1, 2, 3]),
  ];
  for (const raw of nasty) {
    let got;
    assert.doesNotThrow(() => {
      got = decode(raw);
    }, `decode(${String(raw).slice(0, 20)}) threw; the socket must stay open (codec.py docstring)`);
    assert.ok(
      ["astra", "unknown", "realtime", "malformed"].includes(got.kind),
      `decode(${String(raw).slice(0, 20)}) returned ${JSON.stringify(got)}`,
    );
  }
});

test("criterion 1: hello is the astra.hello the contract describes", async () => {
  const { hello, PROTOCOL_VERSION } = await load("protocol.js", "criterion 1");
  assert.equal(PROTOCOL_VERSION, 1, "contracts/protocol.py PROTOCOL_VERSION");
  assert.deepEqual(hello(), {
    type: "astra.hello",
    protocol_version: PROTOCOL_VERSION,
    client: "hud",
  });
});

test("criterion 1: the page may only ever emit astra.hello and astra.frame", async () => {
  const { CLIENT_TYPES } = await load("protocol.js", "criterion 1");
  assert.deepEqual(
    [...CLIENT_TYPES].sort(),
    ["astra.frame", "astra.hello"],
    "criterion 1 negative: astra.frame.text and astra.assist.toggle are T-0012's and " +
      "astra.ui.action is T-0010's; this task's code emits neither (contracts CLIENT_TO_SERVER)",
  );
});

// --- criterion 1 negative, re-scoped per owner (T-0025 criterion 12) ---------------------------
//
// T-0008 criterion 1 says "the only client->server types **this task's code** can emit"; its test
// implemented that as "no file under hud/", which the arrival of a panel directory necessarily
// outgrows. docs/tasks/T-0025.md criterion 12 names the amendment: the walk excludes
// hud/src/assists/, hud/src/ui/ and hud/src/denoise/, and each excluded directory is checked
// against **its own** declared whitelist instead. What moved: the three panel directories, from
// "may mention nothing" to "may mention only its own types". What did not move, and is asserted
// below just as strictly: every other file under hud/ — including hud/src/main.js, which T-0025
// amends — may still mention none of them, and astra.ui.action may appear nowhere outside
// hud/src/ui/. Net coverage does not fall: the property (each task's code emits only its own
// types) is preserved and is now checked per owner rather than globally.
// T-0039 criterion 11: the typed-question panel joins the same arrangement. `astra.text` is a
// client->server type in contracts/protocol.py CLIENT_TO_SERVER (T-0039 criterion 1) and
// protocol.js CLIENT_TYPES is *not* widened to hold it (the test above pins CLIENT_TYPES to
// T-0008's two entries, unmodified) — so every mention of it outside hud/src/ask/ is a leak.
// T-0043 criterion 19/20: the session-log toggle joins the same arrangement — it is the log
// panel's egress and `protocol.js` CLIENT_TYPES is *not* widened to hold it, so every mention of
// it outside hud/src/log/ is a leak.
const OTHER_CLIENT_TYPES = [
  "astra.frame.text",
  "astra.assist.toggle",
  "astra.ui.action",
  "astra.text",
  "astra.log.toggle",
];

/** Panel directory (relative to hud/) -> the client->server types its owner declared. */
const PANEL_WHITELISTS = {
  "src/assists": ["astra.assist.toggle", "astra.frame.text"], // T-0025 ASSIST_CLIENT_TYPES
  "src/ui": ["astra.ui.action"], // T-0010/T-0010b
  "src/denoise": [], // T-0011/T-0011c: the denoising view is display-only
  "src/ask": ["astra.text"], // T-0039 ASK_CLIENT_TYPES
  "src/log": ["astra.log.toggle"], // T-0043 LOG_CLIENT_TYPES
};

const relPosix = (root, file) => relative(root, file).split(sep).join("/");
const inPanel = (rel) => Object.keys(PANEL_WHITELISTS).find((dir) => rel.startsWith(`${dir}/`));

/** Types mentioned outside a comment in files under `root` that belong to no panel directory. */
function shellOffenders(root) {
  const offenders = [];
  for (const f of jsFilesUnder(root)) {
    const rel = relPosix(root, f);
    if (inPanel(rel)) continue;
    const body = stripComments(readFileSync(f, "utf8"));
    for (const t of OTHER_CLIENT_TYPES) if (body.includes(t)) offenders.push(`${rel}: ${t}`);
  }
  return offenders;
}

/** Types mentioned outside a comment in a panel directory that are not on its own whitelist. */
function panelOffenders(root) {
  const offenders = [];
  for (const [dir, allowed] of Object.entries(PANEL_WHITELISTS)) {
    const abs = join(root, dir);
    if (!existsSync(abs)) continue; // the panel's task has not run yet
    for (const f of jsFilesUnder(abs)) {
      const body = stripComments(readFileSync(f, "utf8"));
      for (const t of OTHER_CLIENT_TYPES) {
        if (body.includes(t) && !allowed.includes(t)) offenders.push(`${relPosix(root, f)}: ${t}`);
      }
    }
  }
  return offenders;
}

test("criterion 1 negative (T-0025/12): no file under hud/ outside a panel directory mentions another client->server type", async () => {
  const hud = fileURLToPath(new URL("hud", REPO));
  const files = jsFilesUnder(hud);
  assert.ok(files.length > 0, "T-0008 adds hud/index.html and hud/src/*.js");
  assert.ok(
    files.some((f) => f.endsWith("/src/main.js")),
    "the shell this scan is about must be inside it: hud/src/main.js is where T-0025 adds the " +
      "send/subscribe seam, and criterion 13 says main.js mentions no assist type string",
  );
  assert.deepEqual(
    shellOffenders(hud),
    [],
    "criterion 1 negative: astra.frame.text and astra.assist.toggle belong to T-0025's panel and " +
      "astra.ui.action to T-0010's; the shell emits neither (a mention outside a comment is " +
      "close enough to a bug)",
  );
});

test("criterion 1 negative (T-0025/12): each panel directory mentions only the types its owner declared", async () => {
  const hud = fileURLToPath(new URL("hud", REPO));
  assert.deepEqual(
    panelOffenders(hud),
    [],
    "T-0025 criterion 12: hud/src/assists/** may mention only astra.assist.toggle and " +
      "astra.frame.text, hud/src/ui/** only astra.ui.action, hud/src/denoise/** none — " +
      "astra.ui.action in particular stays forbidden in the assist panel",
  );
  // The one global statement that survives verbatim: outside T-0010's own directory, nothing
  // under hud/ knows the A2UI action type at all.
  const strays = jsFilesUnder(hud)
    .map((f) => relPosix(hud, f))
    .filter(
      (rel) =>
        !rel.startsWith("src/ui/") &&
        stripComments(readFileSync(join(hud, rel), "utf8")).includes("astra.ui.action"),
    );
  assert.deepEqual(strays, [], "astra.ui.action appears nowhere under hud/ outside hud/src/ui/");
});

test("criterion 1 negative (T-0025/12) adversarial: the re-scoped scan still catches a stray outside a panel directory", async () => {
  // The risk in narrowing a scan is that it stops catching anything. Rebuild the real tree in a
  // temp directory, drop two probe files into it, and run the very same functions: the shell probe
  // must be caught by the re-scoped walk and the panel probe by its owner's whitelist.
  const hud = fileURLToPath(new URL("hud", REPO));
  const root = join(mkdtempSync(join(tmpdir(), "astra-hud-scan-")), "hud");
  cpSync(hud, root, { recursive: true });
  writeFileSync(
    join(root, "src", "probe_shell.js"),
    'export const leak = { type: "astra.assist.toggle" };\n',
  );
  mkdirSync(join(root, "src", "assists"), { recursive: true });
  writeFileSync(
    join(root, "src", "assists", "probe_panel.js"),
    'export const leak = { type: "astra.ui.action" };\n',
  );
  // T-0039 criterion 11: the new panel directory is probed too, so its whitelist entry cannot be
  // the thing that makes the scan vacuous.
  mkdirSync(join(root, "src", "ask"), { recursive: true });
  writeFileSync(
    join(root, "src", "ask", "probe_panel.js"),
    'export const leak = { type: "astra.assist.toggle" };\n',
  );
  assert.deepEqual(
    shellOffenders(root),
    ["src/probe_shell.js: astra.assist.toggle"],
    "a file under hud/src/ that is not in a panel directory must still fail the scan",
  );
  assert.deepEqual(
    panelOffenders(root).sort(),
    [
      "src/ask/probe_panel.js: astra.assist.toggle",
      "src/assists/probe_panel.js: astra.ui.action",
    ],
    "T-0025 criterion 12 / T-0039 criterion 11: a panel that emits another owner's type must " +
      "fail its own whitelist, hud/src/ask/ included",
  );
});

test("criterion 2: SERVER_TO_CLIENT is what the page dispatches, and nothing else", async () => {
  const { SERVER_TO_CLIENT, decode } = await load("protocol.js", "criterion 2");
  const set = new Set(SERVER_TO_CLIENT);
  // Transcribed from local_astra/contracts/protocol.py SERVER_TO_CLIENT (the contract).
  const expected = [
    "astra.assist.state",
    "astra.canvas.commit",
    "astra.canvas.step",
    "astra.canvas.synthetic",
    "astra.error",
    "astra.spoken",
    "astra.transcript",
    "astra.turn.state",
    "astra.ui.surface",
  ];
  if (PLAYED_INSTALLED) expected.push("astra.spoken.played"); // T-0030 criterion 21
  if (LOG_STATE_INSTALLED) expected.push("astra.log.state"); // T-0043 criterion 20
  // T-0054 criterion 13. FENCE_INSTALLED is declared further down this module; the callback runs
  // after evaluation, so the reference is resolved by then.
  if (FENCE_INSTALLED) expected.push("astra.ui.fence");
  // T-0090 criterion 2: no sentinel — the pending marker came off when /implement T-0090 started,
  // so this row is red until hud/src/protocol.js mirrors astra.tool.call.
  expected.push("astra.tool.call");
  // T-0162 A-1: the reset echo joins the page's dispatch table once group A is activated (its own
  // pending rows below are red until then; this exact list stays today's on main meanwhile).
  if (!T0162_A_PENDING) expected.push("astra.session.cleared");
  // T-0177 A1 (pin move A-P2): the vision state echo and the published frame, server->client only;
  // astra.vision.select stays out (client->server). Red until hud/src/protocol.js mirrors them.
  expected.push("astra.vision.state", "astra.vision.frame");
  // T-0180 A1/A2: the thought message joins once its corpus rows are activated (pending rows below
  // are red until then; this exact list stays today's meanwhile).
  if (!T0180_A_PENDING) expected.push("astra.thought");
  // T-0181 A1: the mic state echo joins once group A is activated (gate below); astra.mic.mute stays
  // out (client->server).
  if (!T0181_A_PENDING) expected.push("astra.mic.state");
  // T-0213 R-5 named re-pin (A2r-1, the new type): the read-only settings state, server->client only.
  expected.push("astra.settings.state");
  assert.deepEqual(
    [...set].sort(),
    expected.sort(),
    "criterion 2: known astra.* in SERVER_TO_CLIENT -> typed dispatch",
  );
  // astra.hello is a *known* astra type but arrives from the client; the page does not dispatch it.
  const got = decode('{"type":"astra.hello","protocol_version":1,"client":"hud"}');
  assert.ok(got.kind !== "malformed", "a well-formed astra.hello is not malformed");
  assert.ok(!set.has("astra.hello"), "astra.hello is client->server only");
  assert.ok(!set.has("astra.vision.select"), "T-0177 A1: astra.vision.select is client->server only");
});

// --- T-0030 criterion 21: the played message on the page's side of the contract ----------------

test("T-0030 criterion 21: astra.spoken.played decodes with its four-field entries", playedOpts, async () => {
  const { decode, SERVER_TO_CLIENT } = await load("protocol.js", "T-0030 criterion 21");
  assert.ok(
    new Set(SERVER_TO_CLIENT).has("astra.spoken.played"),
    "criterion 21: the page dispatches the message that says which words were heard",
  );
  const message = {
    type: "astra.spoken.played",
    turn_id: "t1",
    played_ms: 1400,
    utterances: [
      { utterance_id: "u1", played_text: "It is a stack trace. ", played_words: 5, complete: true },
      { utterance_id: "u2", played_text: "The error", played_words: 2, complete: false },
    ],
  };
  const got = decode(JSON.stringify(message));
  assert.equal(got.kind, "astra", `a valid message must decode: ${JSON.stringify(got)}`);
  assert.deepEqual(got.msg, message, "and arrive with its nested entries intact");

  // The mirror is as strict as contracts/base.Contract is (this file's own rule): a missing
  // field, a negative cut index and an extra key are all malformed, not "mostly fine".
  const without = (key) => {
    const copy = JSON.parse(JSON.stringify(message));
    delete copy[key];
    return JSON.stringify(copy);
  };
  for (const key of ["turn_id", "played_ms", "utterances"]) {
    assert.equal(decode(without(key)).kind, "malformed", `${key} is required`);
  }
  const negative = JSON.parse(JSON.stringify(message));
  negative.utterances[1].played_words = -1;
  assert.equal(
    decode(JSON.stringify(negative)).kind,
    "malformed",
    "played_words is ge=0 in contracts/protocol.py; a negative cut index would make the panel " +
      "slice from the end of the string and dim the wrong words",
  );
  const extra = JSON.parse(JSON.stringify(message));
  extra.surprise = 1;
  assert.equal(decode(JSON.stringify(extra)).kind, "malformed", "unknown fields are errors");

  // Ruling A8: the *nested* entry is a Contract too, and contracts/base.Contract is strict all
  // the way down. Without this the page would accept an entry carrying anything at all beside
  // the four declared keys -- including a "type", which the top-level extra-key loop skips by
  // design and which a nested model must therefore refuse itself.
  const mutate = (fn) => {
    const copy = JSON.parse(JSON.stringify(message));
    fn(copy.utterances[0]);
    return decode(JSON.stringify(copy));
  };
  assert.equal(
    mutate((e) => {
      e.surprise = 1;
    }).kind,
    "malformed",
    "an undeclared key inside a PlayedUtterance is an error, as it is at the top level",
  );
  assert.equal(
    mutate((e) => {
      e.type = "x";
    }).kind,
    "malformed",
    "ruling A8: and so is a nested 'type' -- PlayedUtterance declares none, so the top-level " +
      "exemption for the discriminator must not reach into it",
  );
  assert.equal(
    mutate((e) => {
      delete e.played_text;
    }).kind,
    "malformed",
    "every declared field of the entry is required: a missing played_text would leave the pane " +
      "guessing which words were heard",
  );
});

// --- T-0043 criterion 20: the log state on the page's side of the contract ----------------------
//
// The landmine this row exists for is the one T-0037 hit: contracts/base.Contract is strict and
// hud/src/protocol.js mirrors that literally, per key. A field on the Python side that the page's
// table does not declare turns **every** live astra.log.state into kind:"malformed" — counted,
// ignored, invisible — so the checkbox never moves and no error appears anywhere. The page would
// then show "unknown" forever (criterion 19's safe state, which is why the failure is silent
// rather than loud) while the session writes its file.

test("T-0043 criterion 20: astra.log.state decodes, strictly, with nullable path and error", logStateOpts, async () => {
  const { decode, SERVER_TO_CLIENT } = await load("protocol.js", "T-0043 criterion 20");
  assert.ok(
    new Set(SERVER_TO_CLIENT).has("astra.log.state"),
    "criterion 20: the page dispatches the message that says whether this session is recorded",
  );
  const message = {
    type: "astra.log.state",
    on: true,
    path: "/home/u/astra/logs/sessions/20260916-041500-12345.jsonl",
    records: 412,
    dropped: 0,
    bytes: 190233,
    error: null,
  };
  const got = decode(JSON.stringify(message));
  assert.equal(got.kind, "astra", `a valid message must decode: ${JSON.stringify(got)}`);
  assert.deepEqual(got.msg, message, "…and arrive intact, path string included");

  const off = { ...message, on: false, path: null };
  assert.equal(
    decode(JSON.stringify(off)).kind,
    "astra",
    "criterion 16: path is nullable — logging off has no file, and that is the state the user " +
      "most needs the page to render",
  );
  const failed = { ...message, on: false, path: null, error: "cannot open /x: is a directory" };
  assert.equal(decode(JSON.stringify(failed)).kind, "astra", "criterion 4: error is nullable too");

  const without = (key) => {
    const copy = { ...message };
    delete copy[key];
    return JSON.stringify(copy);
  };
  // `on` is the one field that cannot have a sensible default: a state that does not say which
  // way is not a state, and criterion 19's page would have to guess — which is the whole failure.
  // Whether the *counters* carry defaults is the contract's call, and the page is held to
  // whatever it decided by tests/unit/test_hud_constants_parity.py, field by field, rather than
  // by a second hand-written claim here.
  assert.equal(
    decode(without("on")).kind,
    "malformed",
    "criterion 16/19: `on` is required — the page must never have to infer it",
  );
  assert.equal(
    decode(JSON.stringify({ ...message, on: null })).kind,
    "malformed",
    "criterion 19: `on` is not nullable; 'the server said null' must not be reachable, because " +
      "the page's three-state rendering already has a state for 'the server has not said'",
  );
  assert.equal(
    decode(JSON.stringify({ ...message, records: "412" })).kind,
    "malformed",
    "criterion 20: the counters are integers on both sides",
  );
  assert.equal(
    decode(JSON.stringify({ ...message, surprise: 1 })).kind,
    "malformed",
    "contracts/base.Contract is extra='forbid' and the page mirrors it",
  );
});

// --- T-0054 criterion 13: the fence notice's field table ---------------------------------------
//
// The page's table and contracts/protocol.AstraUiFence must be the same field for field, and the
// hard cross-language pin for that is tests/unit/test_hud_constants_parity.py. What is added here
// is the behaviour a set comparison cannot see: each field's *wrong type* must decode
// `malformed`, one row per field. A page laxer than the contract renders a message the server can
// never send; a page stricter than it answers a legal one with malformed, and the pane then never
// learns that the interface it is showing was abandoned.
//
// Sentinel, as for astra.spoken.played and astra.log.state above: protocol.js's own source. Node
// tests cannot carry pytest's `pending` marker and tests/unit/test_hud_js.py requires
// `node --test tests/hud` to exit 0.
const FENCE_INSTALLED = (() => {
  const file = fileURLToPath(new URL("hud/src/protocol.js", REPO));
  return existsSync(file) && readFileSync(file, "utf8").includes("astra.ui.fence");
})();
const fenceOpts = FENCE_INSTALLED
  ? {}
  : {
      skip:
        "T-0054 group B is not implemented yet: hud/src/protocol.js does not mention " +
        "astra.ui.fence (docs/tasks/T-0054.md criterion 13). The red-now pin is " +
        "tests/unit/test_hud_constants_parity.py.",
    };

const FENCE_OK = {
  type: "astra.ui.fence",
  turn_id: "t6",
  canvases: 2,
  close: "inferred",
  continued: true,
  speech_refused: 1,
};

test("T-0054 criterion 13: a well-formed astra.ui.fence decodes as a typed message", fenceOpts, async () => {
  const { decode } = await load("protocol.js", "criterion 13");
  const got = decode(JSON.stringify(FENCE_OK));
  assert.equal(got.kind, "astra", `a legal fence notice must dispatch, not be dropped: ${JSON.stringify(got)}`);
  assert.equal(got.type, "astra.ui.fence");
  assert.deepEqual(got.msg, FENCE_OK, "…carrying every field the server sent, unchanged");

  // The optional fields really are optional: contracts/protocol.AstraUiFence gives close,
  // continued and speech_refused defaults, so a server that omits them sends a legal message.
  const minimal = { type: "astra.ui.fence", turn_id: "t6", canvases: 0 };
  assert.equal(
    decode(JSON.stringify(minimal)).kind,
    "astra",
    "criterion 13: close/continued/speech_refused carry defaults in the contract and must be " +
      "optional here; a page that requires them refuses a legal message",
  );
  assert.equal(
    decode(JSON.stringify({ ...FENCE_OK, close: null })).kind,
    "astra",
    "criterion 13: close is nullable — null is 'no fence was opened', the refused-only case",
  );
});

const FENCE_WRONG_TYPES = [
  ["turn_id", 6, "a turn id that is not a string"],
  ["canvases", "2", "a count that is a string"],
  ["canvases", -1, "a negative count"],
  ["canvases", 1.5, "a fractional count"],
  ["close", "salvaged", "a close value outside the four FenceClose literals"],
  ["close", 1, "a close value that is not a string"],
  ["continued", "true", "a continued flag that is a string"],
  ["speech_refused", -1, "a negative refusal count"],
  ["speech_refused", null, "a null refusal count (the contract's default is 0, not None)"],
];

for (const [field, value, why] of FENCE_WRONG_TYPES) {
  test(`T-0054 criterion 13: the field table rejects ${why}`, fenceOpts, async () => {
    const { decode } = await load("protocol.js", "criterion 13");
    const got = decode(JSON.stringify({ ...FENCE_OK, [field]: value }));
    assert.equal(
      got.kind,
      "malformed",
      `criterion 13: ${field} = ${JSON.stringify(value)} is not what ` +
        `contracts/protocol.AstraUiFence accepts, so the page must decode it malformed — ignored ` +
        `and counted — rather than rendering it. decode() said ${JSON.stringify(got)}`,
    );
    assert.equal(got.type, "astra.ui.fence", "…and says which type it was, for the counter");
  });
}

test("T-0054 criterion 13: an unknown field on the fence notice is refused", fenceOpts, async () => {
  const { decode } = await load("protocol.js", "criterion 13");
  // contracts/base.Contract forbids unknown fields, and criterion 8 says extra='forbid' holds.
  // `replay` in particular: this message is deliberately not journalled and never replayed, so a
  // page that accepted the flag would be ready for a catch-up the server cannot send.
  for (const extra of [{ replay: true }, { final: true }, { fence_close: "written" }]) {
    const got = decode(JSON.stringify({ ...FENCE_OK, ...extra }));
    assert.equal(
      got.kind,
      "malformed",
      `criterion 8/13: ${JSON.stringify(extra)} is not a field of AstraUiFence; decode() said ` +
        JSON.stringify(got),
    );
  }
});


// --- T-0090 criterion 2: astra.tool.call on the page's side of the contract ----------------------
//
// The field table is pinned against contracts/protocol.AstraToolCall itself by
// tests/unit/test_hud_constants_parity.py; what is added here is the behaviour a set comparison
// cannot see — each wrong value decodes `malformed`, and every legal shape decodes `astra`. Expected
// values are criterion 1's text (docs/tasks/T-0090.md), never read back from protocol.js.

const TOOL_OK = {
  type: "astra.tool.call",
  turn_id: "t7",
  call_id: "t7-c1",
  name: "type_at",
  arguments: { x: 410, y: 72, text: "quarterly report", press_enter: true },
  status: "dry_run",
  source: "repair",
  action: "would type \"quarterly report\" at (525, 79) and press Enter",
  reason: null,
  query_problem: null,
};

test("T-0090 criterion 2: a well-formed astra.tool.call decodes as a typed message", async () => {
  const { decode, SERVER_TO_CLIENT } = await load("protocol.js", "T-0090 criterion 2");
  assert.ok(
    new Set(SERVER_TO_CLIENT).has("astra.tool.call"),
    "T-0090 criterion 2: the page dispatches astra.tool.call; SERVER_TO_CLIENT is " +
      JSON.stringify([...SERVER_TO_CLIENT]),
  );
  const got = decode(JSON.stringify(TOOL_OK));
  assert.equal(got.kind, "astra", `a legal tool call must dispatch: ${JSON.stringify(got)}`);
  assert.equal(got.type, "astra.tool.call");
  assert.deepEqual(got.msg, TOOL_OK, "…carrying every field the server sent, unchanged");

  // codec.encode is exclude_none: a proposal goes on the wire without action/reason/query_problem
  // and without replay, so the page must accept their absence.
  const minimal = {
    type: "astra.tool.call",
    turn_id: "t7",
    call_id: "t7-c1",
    name: "scroll",
    arguments: { direction: "down" },
    status: "proposed",
    source: "draft",
  };
  assert.equal(decode(JSON.stringify(minimal)).kind, "astra", "the minimal proposal decodes");
  for (const status of ["proposed", "repairing", "dry_run", "executed", "refused", "cancelled"]) {
    assert.equal(
      decode(JSON.stringify({ ...TOOL_OK, status })).kind,
      "astra",
      `criterion 1: status ${status} is one of the six`,
    );
  }
  assert.equal(decode(JSON.stringify({ ...TOOL_OK, source: "draft" })).kind, "astra");
  assert.equal(
    decode(JSON.stringify({ ...TOOL_OK, replay: true })).kind,
    "astra",
    "criterion 3: a journalled card replayed to a reloading page carries replay=true",
  );
  assert.equal(
    decode(JSON.stringify({ ...TOOL_OK, query_problem: "query is a single stopword" })).kind,
    "astra",
    "H7: query_problem is a string when set",
  );
});

const TOOL_WRONG = [
  ["status", "executing", "a status outside the six"],
  ["status", null, "a null status"],
  ["source", "model", "a source outside draft|repair"],
  ["arguments", ["down"], "arguments that are a list"],
  ["arguments", "q=x", "arguments that are a string"],
  ["arguments", null, "null arguments"],
  ["call_id", 1, "a numeric call id"],
  ["turn_id", null, "a null turn id"],
  ["name", 5, "a numeric tool name"],
  ["action", 3, "a numeric action"],
  ["reason", false, "a boolean reason"],
  ["query_problem", ["x"], "a list query_problem"],
  ["replay", "true", "a string replay flag"],
];

for (const [field, value, why] of TOOL_WRONG) {
  test(`T-0090 criterion 2: the tool-call table rejects ${why}`, async () => {
    const { decode } = await load("protocol.js", "T-0090 criterion 2");
    const got = decode(JSON.stringify({ ...TOOL_OK, [field]: value }));
    assert.equal(
      got.kind,
      "malformed",
      `criterion 2: ${field} = ${JSON.stringify(value)} is not what AstraToolCall accepts; ` +
        `decode() said ${JSON.stringify(got)}`,
    );
    assert.equal(got.type, "astra.tool.call", "…and says which type it was, for the counter");
  });
}

test("T-0090 criterion 2: a missing required field or an undeclared one is malformed", async () => {
  const { decode } = await load("protocol.js", "T-0090 criterion 2");
  for (const key of ["turn_id", "call_id", "name", "arguments", "status", "source"]) {
    const copy = { ...TOOL_OK };
    delete copy[key];
    const got = decode(JSON.stringify(copy));
    assert.equal(got.kind, "malformed", `criterion 1: ${key} is required; got ${JSON.stringify(got)}`);
  }
  for (const extra of [{ url: "https://www.google.com/search?q=x" }, { href: "x" }, { final: true }]) {
    const got = decode(JSON.stringify({ ...TOOL_OK, ...extra }));
    assert.equal(
      got.kind,
      "malformed",
      `contracts/base.Contract is extra='forbid' and the page mirrors it: ${JSON.stringify(extra)}`,
    );
  }
});

// --- T-0128c B1: astra.canvas.step mirrors AstraCanvasStep ---------------------------------------
//
// The corpus rows tagged "pending_js": "T-0128c" are the shared cross-language cases (the real
// codec's verdict on each is checked by tests/unit/test_hud_constants_parity.py). The rows below
// add the page-only facts: the message comes through unchanged, and the old shape's requirements
// (accepted/entropy, step < max_steps) are gone.

const T0128C_ROWS = CORPUS.filter((c) => c.pending_js === "T-0128c");

test("T-0128c B1 premise: the corpus carries the accepted and the malformed canvas-step rows the task names", () => {
  const names = new Set(T0128C_ROWS.map((c) => c.name));
  for (const name of [
    "canvas_step_denoise_real_shape_cl8",
    "canvas_step_commit_real_shape",
    "canvas_step_denoise_at_max_steps",
    "canvas_step_commit_at_step_zero",
    "canvas_step_denoise_missing_stable",
    "canvas_step_commit_with_rows",
    "canvas_step_commit_with_empty_ids",
    "canvas_step_stable_too_short",
  ])
    assert.ok(names.has(name), `T-0128c Tests to write names the corpus row ${name}`);
  const old = CORPUS.find((c) => c.name === "canvas_step_lengths_disagree");
  assert.ok(old && old.expect === "malformed" && !old.pending_js, "canvas_step_lengths_disagree stays malformed, ungated");
});

pendingT0128cB("T-0128c B1: every T-0128c corpus row reaches the real codec's verdict on the page", async () => {
  const { decode } = await load("protocol.js", "T-0128c B1");
  const wrong = [];
  for (const c of T0128C_ROWS) {
    let got;
    try {
      got = decode(rawOf(c));
    } catch (err) {
      wrong.push(`${c.name}: decode threw ${err}`);
      continue;
    }
    if (!got || got.kind !== c.expect || got.type !== c.type)
      wrong.push(`${c.name}: expected ${c.expect} ${c.type}, got ${got && got.kind} ${got && got.type} (${got && got.error})`);
  }
  assert.deepEqual(wrong, [], `T-0128c B1 (the page mirrors AstraCanvasStep):\n${wrong.join("\n")}`);
});

/** The first frame of the committed real trajectory's shape (plugin tap.py): no accepted, no entropy. */
const REAL_DENOISE = {
  type: "astra.canvas.step",
  request_id: "chatcmpl-b8d75dac2b8f054f-b19b91f7",
  canvas_idx: 0,
  step: 1,
  max_steps: 48,
  canvas_length: 8,
  phase: "denoise",
  canvas: [259472, 1, 2, 3, 4, 5, 6, 7],
  argmax: [259472, 9, 9, 9, 9, 9, 9, 9],
  stable: "00",
};

pendingT0128cB("T-0128c B1: a real tap frame decodes as astra and comes through field for field", async () => {
  const { decode } = await load("protocol.js", "T-0128c B1");
  const got = decode(JSON.stringify(REAL_DENOISE));
  assert.equal(got.kind, "astra", `T-0128c Finding 1: the page must not refuse a real tap frame: ${got.error}`);
  assert.deepEqual(got.msg, REAL_DENOISE, "the decoded message is the wire object, unchanged");
});

pendingT0128cB("T-0128c B1: the optional fields stay optional and typed — accepted is hex, entropy numbers, segment the enum", async () => {
  const { decode } = await load("protocol.js", "T-0128c B1");
  const ok = (extra, why) => {
    const got = decode(JSON.stringify({ ...REAL_DENOISE, ...extra }));
    assert.equal(got.kind, "astra", `${why}: ${JSON.stringify(extra)} -> ${got.error}`);
  };
  const bad = (extra, why) => {
    const got = decode(JSON.stringify({ ...REAL_DENOISE, ...extra }));
    assert.equal(got.kind, "malformed", `${why}: ${JSON.stringify(extra)} must be refused`);
  };
  ok({ accepted: null, entropy: null, segment: null, text_preview: null }, "all four are nullable in AstraCanvasStep");
  ok({ segment: "answer" }, "segment is a Segment");
  bad({ segment: "draft" }, "segment outside the Segment enum");
  bad({ accepted: "zz" }, "accepted must match ^[0-9a-f]*$");
  bad({ accepted: "0" }, "accepted is whole bytes, like stable");
  bad({ entropy: [0.1, 0.2] }, "entropy, when present, has canvas_length entries");
  bad({ entropy: ["x", 0, 0, 0, 0, 0, 0, 0] }, "entropy entries are numbers");
  bad({ canvas: [1, 2, 3, 4, 5, 6, 7, "8"] }, "canvas ids are integers");
  bad({ argmax: [1, 2, 3, 4, 5, 6, 7, 8.5] }, "argmax ids are integers");
  bad({ committed_tokens: "x" }, "committed_tokens is a list of ints");
  bad({ phase: null }, "phase is not nullable");
  bad({ entropies: [] }, "contracts/base.Contract is extra='forbid' and the page mirrors it");
});

pendingT0128cB("T-0128c B1 (the bound both ways): denoise and commit accept 0..max_steps and refuse max_steps + 1", async () => {
  const { decode } = await load("protocol.js", "T-0128c B1");
  const commitFrame = { type: "astra.canvas.step", request_id: "r", canvas_idx: 2, max_steps: 48, canvas_length: 8, phase: "commit", committed_tokens: [1] };
  for (const [frame, what] of [
    [REAL_DENOISE, "denoise"],
    [commitFrame, "commit"],
  ]) {
    for (const s of [0, 1, 47, 48]) assert.equal(decode(JSON.stringify({ ...frame, step: s })).kind, "astra", `${what} step ${s}/48 (R14/R15)`);
    assert.equal(decode(JSON.stringify({ ...frame, step: 49 })).kind, "malformed", `${what} step 49/48`);
    assert.equal(decode(JSON.stringify({ ...frame, step: -1 })).kind, "malformed", `${what} step -1`);
  }
});

// --- T-0159 A6: the page's decoder mirrors argmax_pieces / committed_pieces ----------------------
//
// docs/tasks/T-0159.md A1/A6: argmax_pieces on a denoise frame only, canvas_length strings;
// committed_pieces on a commit frame only, len(committed_tokens) strings; both optional and
// nullable; non-string items refused. The shared corpus rows (verdicts checked against the real
// codec by tests/unit/test_hud_constants_parity.py) plus the page-only facts: the strings come
// through unchanged (markup is data, the page never decodes ids) and every presence/phase/length
// rule holds both ways.

/** The T-0159 corpus rows, hand-listed (accepted, then malformed) so deleting one is visible. */
const T0159_CORPUS = {
  canvas_step_denoise_with_pieces: "astra",
  canvas_step_commit_with_pieces: "astra",
  canvas_step_denoise_pieces_markers_newline_markup: "astra",
  canvas_step_argmax_pieces_length_disagrees: "malformed",
  canvas_step_committed_pieces_length_disagrees: "malformed",
  canvas_step_denoise_with_committed_pieces: "malformed",
  canvas_step_commit_with_argmax_pieces: "malformed",
  canvas_step_argmax_pieces_non_string_item: "malformed",
  canvas_step_committed_pieces_null_item: "malformed",
  canvas_step_commit_with_empty_argmax_pieces: "malformed",
  canvas_step_no_phase_with_argmax_pieces: "astra",
  canvas_step_no_phase_with_committed_pieces: "malformed",
};

test("T-0159 A6 premise: the corpus carries the canvas-step piece rows the task names, with their verdicts", () => {
  const byName = new Map(CORPUS.map((c) => [c.name, c]));
  for (const [name, expect] of Object.entries(T0159_CORPUS)) {
    const c = byName.get(name);
    assert.ok(c, `T-0159 A6 names the corpus row ${name}`);
    assert.equal(c.expect, expect, `${name} declares ${expect}`);
    assert.equal(c.type, "astra.canvas.step");
  }
});

pendingT0159A("T-0159 A6: every T-0159 corpus row reaches the real codec's verdict on the page", async () => {
  const { decode } = await load("protocol.js", "T-0159 A6");
  const wrong = [];
  for (const c of CORPUS.filter((row) => row.name in T0159_CORPUS)) {
    let got;
    try {
      got = decode(rawOf(c));
    } catch (err) {
      wrong.push(`${c.name}: decode threw ${err}`);
      continue;
    }
    if (!got || got.kind !== c.expect || (c.expect === "astra" && got.type !== c.type))
      wrong.push(`${c.name}: expected ${c.expect}, got ${got && got.kind} (${got && got.error})`);
  }
  assert.deepEqual(wrong, [], `T-0159 A6 (the page mirrors the piece fields):\n${wrong.join("\n")}`);
});

/** REAL_DENOISE (CL 8) with pieces, and a commit of three ids with its three pieces. */
const PIECED_DENOISE = {
  ...REAL_DENOISE,
  argmax_pieces: ["<|channel>", "thought", "\n", "</script><b>x", "<0xE2>", " the", "  ", "<channel|>"],
};
const PIECED_COMMIT = {
  type: "astra.canvas.step",
  request_id: "chatcmpl-b8d75dac2b8f054f-b19b91f7",
  canvas_idx: 1,
  step: 18,
  max_steps: 48,
  canvas_length: 8,
  phase: "commit",
  committed_tokens: [818, 1144, 107],
  committed_pieces: [" Here", " it", "\n"],
};

pendingT0159A("T-0159 A6: a frame with pieces decodes as astra and its strings come through unchanged", async () => {
  const { decode } = await load("protocol.js", "T-0159 A6");
  for (const frame of [PIECED_DENOISE, PIECED_COMMIT]) {
    const got = decode(JSON.stringify(frame));
    assert.equal(got.kind, "astra", `${frame.phase}: ${got.error}`);
    assert.deepEqual(got.msg, frame, "the decoded message is the wire object, pieces included, unchanged");
  }
  const msg = decode(JSON.stringify(PIECED_DENOISE)).msg;
  assert.equal(msg.argmax_pieces[3], "</script><b>x", "markup in a piece is a string, not parsed or escaped");
});

pendingT0159A("T-0159 A6: presence, phase, length and item rules, both ways", async () => {
  const { decode } = await load("protocol.js", "T-0159 A6");
  const ok = (frame, why) => {
    const got = decode(JSON.stringify(frame));
    assert.equal(got.kind, "astra", `${why}: ${got.error}`);
  };
  const bad = (frame, why) => {
    const got = decode(JSON.stringify(frame));
    assert.equal(got.kind, "malformed", `${why} must be refused`);
  };
  const eight = ["a", "b", "c", "d", "e", "f", "g", "h"];
  ok(PIECED_DENOISE, "denoise with canvas_length argmax_pieces");
  ok(PIECED_COMMIT, "commit with one piece per committed id");
  ok({ ...REAL_DENOISE, argmax_pieces: null, committed_pieces: null }, "both nullable on a denoise frame");
  ok({ ...PIECED_COMMIT, committed_pieces: null, argmax_pieces: null }, "both nullable on a commit frame");
  ok({ ...REAL_DENOISE, argmax_pieces: ["", "", "", "", "", "", "", ""] }, "an empty string is a string");
  bad({ ...REAL_DENOISE, argmax_pieces: eight.slice(0, 7) }, "argmax_pieces one short of canvas_length");
  bad({ ...REAL_DENOISE, argmax_pieces: [...eight, "i"] }, "argmax_pieces one past canvas_length");
  bad({ ...REAL_DENOISE, argmax_pieces: [] }, "empty argmax_pieces");
  bad({ ...PIECED_COMMIT, committed_pieces: [" Here", " it"] }, "committed_pieces one short of the ids");
  bad({ ...PIECED_COMMIT, committed_pieces: eight }, "committed_pieces of canvas_length, not of the id count");
  bad({ ...PIECED_DENOISE, committed_pieces: eight }, "committed_pieces on a denoise frame (even of canvas_length)");
  bad({ ...REAL_DENOISE, committed_pieces: [] }, "committed_pieces on a denoise frame (empty is still carried)");
  bad({ ...PIECED_COMMIT, argmax_pieces: [" Here", " it", "\n"] }, "argmax_pieces on a commit frame (id count)");
  bad({ ...PIECED_COMMIT, argmax_pieces: eight }, "argmax_pieces on a commit frame (canvas_length)");
  bad({ ...PIECED_COMMIT, argmax_pieces: [] }, "argmax_pieces on a commit frame (empty is still carried)");
  {
    const { phase: _p, ...noPhase } = PIECED_DENOISE;
    ok(noPhase, "no phase key: a denoise frame by default, so argmax_pieces is allowed");
    const { phase: _q, ...noPhasePlain } = REAL_DENOISE;
    bad({ ...noPhasePlain, committed_pieces: eight }, "no phase key: a denoise frame never carries committed_pieces");
  }
  for (const item of [7, null, true, {}, ["x"]]) {
    bad({ ...REAL_DENOISE, argmax_pieces: [...eight.slice(0, 7), item] }, `argmax_pieces item ${JSON.stringify(item)}`);
    bad({ ...PIECED_COMMIT, committed_pieces: [" Here", " it", item] }, `committed_pieces item ${JSON.stringify(item)}`);
  }
  bad({ ...REAL_DENOISE, argmax_pieces: "abcdefgh" }, "argmax_pieces is a list, not a string of canvas_length");
});

// --- T-0162 A-1: astra.session.cleared on the page's side of the contract ------------------------
//
// AstraSessionCleared(resets: int >= 1, ended_turn_id: str | None, memory_forgotten: int >= 0,
// replay: bool = False), server->client. codec.encode drops None, so a reset from IDLE arrives
// without ended_turn_id. astra.session.reset is client->server: the page never dispatches it.

const CLEARED_OK = {
  type: "astra.session.cleared",
  resets: 2,
  ended_turn_id: "t7",
  memory_forgotten: 13,
  replay: false,
};

pendingT0162A("T-0162 A-1: astra.session.cleared is dispatched and decodes as a typed message", async () => {
  const { decode, SERVER_TO_CLIENT } = await load("protocol.js", "T-0162 A-1");
  const set = new Set(SERVER_TO_CLIENT);
  assert.ok(
    set.has("astra.session.cleared"),
    "A-1: the page dispatches the reset echo; SERVER_TO_CLIENT is " + JSON.stringify([...set]),
  );
  assert.ok(!set.has("astra.session.reset"), "A-1: astra.session.reset is client->server only");
  const got = decode(JSON.stringify(CLEARED_OK));
  assert.equal(got.kind, "astra", `a legal echo must dispatch: ${JSON.stringify(got)}`);
  assert.equal(got.type, "astra.session.cleared");
  assert.deepEqual(got.msg, CLEARED_OK, "…carrying every field the server sent, unchanged");
  const fromIdle = { type: "astra.session.cleared", resets: 1, memory_forgotten: 0 };
  assert.equal(decode(JSON.stringify(fromIdle)).kind, "astra", "a reset from IDLE has no ended_turn_id on the wire");
  assert.equal(decode(JSON.stringify({ ...fromIdle, ended_turn_id: null })).kind, "astra", "an explicit null is accepted");
  assert.equal(decode(JSON.stringify({ ...CLEARED_OK, replay: true })).kind, "astra", "the hello-reply prefix is replay=true");
});

const CLEARED_WRONG = [
  ["resets", 0, "resets 0 (>= 1)"],
  ["resets", -1, "a negative reset count"],
  ["resets", 1.5, "a fractional reset count"],
  ["resets", null, "a null reset count"],
  ["memory_forgotten", -1, "a negative memory_forgotten"],
  ["ended_turn_id", 7, "a numeric turn id"],
  ["replay", "true", "a string replay flag"],
];

for (const [field, value, why] of CLEARED_WRONG) {
  pendingT0162A(`T-0162 A-1: the cleared table rejects ${why}`, async () => {
    const { decode } = await load("protocol.js", "T-0162 A-1");
    const got = decode(JSON.stringify({ ...CLEARED_OK, [field]: value }));
    assert.equal(got.kind, "malformed", `A-1: ${field} = ${JSON.stringify(value)}; decode() said ${JSON.stringify(got)}`);
    assert.equal(got.type, "astra.session.cleared", "…and says which type it was, for the counter");
  });
}

pendingT0162A("T-0162 A-1: a missing count or an undeclared field is malformed", async () => {
  const { decode } = await load("protocol.js", "T-0162 A-1");
  for (const key of ["resets", "memory_forgotten"]) {
    const copy = { ...CLEARED_OK };
    delete copy[key];
    const got = decode(JSON.stringify(copy));
    assert.equal(got.kind, "malformed", `A-1: ${key} is required; got ${JSON.stringify(got)}`);
  }
  for (const extra of [{ history: [] }, { session_id: "abc" }, { final: true }]) {
    const got = decode(JSON.stringify({ ...CLEARED_OK, ...extra }));
    assert.equal(got.kind, "malformed", `unknown fields are malformed: ${JSON.stringify(extra)}`);
  }
});

// --- T-0162 group C: AstraToolCall.step and AstraToolCall.result (C-1) ----------------------------
//
// protocol.js mirrors the two fields (docs/tasks/T-0162.md C-1): `step: int >= 1 | null` and
// `result: ToolCallResult | null`, set only on a final message (dry_run, executed, refused,
// cancelled), with `outcome` one of executed/dry_run/refused/failed/cancelled/timeout, `hosts` a list
// of at most 3 plain host names, `hosts_more: int >= 0` and `results: int | null`. ToolCallStatus is
// not widened. The same hand-written cases are checked against the real Python codec by
// tests/contract/test_tool_call_result_wire.py, so the page and the contract agree.
//
// Gate: strict xfail until the **Status:** line records "group C: tests <a>..<b>, activated <sha>".

const T0162_C_ACTIVATED = /\bgroup C:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const T0162_C_PENDING = !T0162_C_ACTIVATED.test(T0162_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0162C(name, fn) {
  test(name + (T0162_C_PENDING ? " [pending T-0162 C: strict xfail until group C is activated]" : ""), async (t) => {
    if (!T0162_C_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0162 group C row passed while docs/tasks/T-0162.md's status does " +
          "not record group C's activation. Either group C has landed (record 'group C: tests …, " +
          "activated <sha>' in the status) or the row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0162 C pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0162 C gate: only group C's own phrase activates the C rows", () => {
  assert.equal(T0162_C_ACTIVATED.test("in progress (group A: tests 2decefd7..f18ad08c, activated f18ad08c; group D: tests 7d570443..9b5f9d1e, activated 9b5f9d1e)"), false);
  assert.equal(T0162_C_ACTIVATED.test("in progress (group C: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6)"), true);
  assert.equal(T0162_C_ACTIVATED.test("in progress (group C: tests 1a2b3c4..5d6e7f8, not yet activated)"), false);
});

const C_RESULT = { outcome: "executed", hosts: ["shop.example", "cdn.shop.example"], hosts_more: 0 };
const C_FINAL = {
  type: "astra.tool.call",
  turn_id: "t3",
  call_id: "t3-call",
  name: "click_at",
  arguments: { x: 500, y: 250 },
  status: "executed",
  source: "draft",
  action: "clicked at (640, 180)",
  step: 1,
  result: C_RESULT,
};

/** The C-1 mirror is present: a final message with a step and a result decodes. */
async function cDecode() {
  const { decode } = await load("protocol.js", "T-0162 C-1");
  const probe = decode(JSON.stringify(C_FINAL));
  assert.equal(probe.kind, "astra", `T-0162 C-1: protocol.js mirrors step and result; a legal final call decoded as ${JSON.stringify(probe)}`);
  return decode;
}

pendingT0162C("T-0162 C-1: a final call with step and result decodes, field for field", async () => {
  const decode = await cDecode();
  for (const status of ["dry_run", "executed", "refused", "cancelled"]) {
    const got = decode(JSON.stringify({ ...C_FINAL, status }));
    assert.equal(got.kind, "astra", `C-1: a result on the final status ${status}; ${JSON.stringify(got)}`);
    assert.deepEqual(got.msg, { ...C_FINAL, status });
  }
  for (const outcome of ["executed", "dry_run", "refused", "failed", "cancelled", "timeout"]) {
    const got = decode(JSON.stringify({ ...C_FINAL, result: { ...C_RESULT, outcome } }));
    assert.equal(got.kind, "astra", `C-1: outcome ${outcome}`);
  }
  const search = { ...C_FINAL, name: "google_search", arguments: { query: "q" }, step: 4, result: { outcome: "executed", hosts: ["a.example", "b.example", "c.example"], hosts_more: 2, results: 9 } };
  assert.equal(decode(JSON.stringify(search)).kind, "astra", "C-1: a search result with its count and 3 hosts");
  assert.equal(decode(JSON.stringify({ ...search, result: { ...search.result, results: null } })).kind, "astra", "results is nullable");
  assert.equal(decode(JSON.stringify({ ...C_FINAL, step: null, result: null })).kind, "astra", "both are nullable");
  const host253 = ["a".repeat(63), "b".repeat(63), "c".repeat(63), "d".repeat(61)].join(".");
  assert.equal(host253.length, 253);
  assert.equal(decode(JSON.stringify({ ...C_FINAL, result: { ...C_RESULT, hosts: [host253] } })).kind, "astra", "T2: a 253-character host is legal");
  const proposed = { ...C_FINAL, status: "proposed", step: 2 };
  delete proposed.result;
  assert.equal(decode(JSON.stringify(proposed)).kind, "astra", "C-1: a proposal carries its step and no result");
});

const C_WRONG = [
  [{ step: 0 }, "step 0 (>= 1)"],
  [{ step: -1 }, "a negative step"],
  [{ step: 1.5 }, "a fractional step"],
  [{ status: "failed" }, "status failed (ToolCallStatus is not widened)"],
  [{ status: "timeout" }, "status timeout (ToolCallStatus is not widened)"],
  [{ status: "proposed" }, "a result on a proposal (final messages only)"],
  [{ status: "repairing" }, "a result while repairing (final messages only)"],
  [{ result: "executed" }, "a result that is a string"],
  [{ result: { ...C_RESULT, outcome: "failure" } }, "an outcome outside the six"],
  [{ result: { ...C_RESULT, outcome: "proposed" } }, "a wire status as an outcome"],
  [{ result: { ...C_RESULT, hosts_more: -1 } }, "a negative hosts_more"],
  [{ result: { ...C_RESULT, hosts: "shop.example" } }, "hosts that are a string"],
  [{ result: { ...C_RESULT, hosts: [1] } }, "a numeric host"],
  [{ result: { ...C_RESULT, results: "2" } }, "a string count"],
  [{ result: { ...C_RESULT, results: -1 } }, "a negative count (ruling: results >= 0)"],
  [{ result: { ...C_RESULT, engine: "brave" } }, "an engine field (T-0125 gap 3 not adopted)"],
  [{ result: { ...C_RESULT, query: "weather in Paris" } }, "a query field (no query text)"],
  [{ result: { ...C_RESULT, type: "x" } }, "a type key inside the nested result"],
  [{ result: { ...C_RESULT, hosts: ["a.example", "b.example", "c.example", "d.example"] } }, "four hosts (at most 3)"],
  [{ result: { ...C_RESULT, hosts: ["<img src=x onerror=alert(1)>"] } }, "markup as a host"],
  [{ result: { ...C_RESULT, hosts: ["https://evil.example"] } }, "a URL as a host"],
  [{ result: { ...C_RESULT, hosts: ["evil.example/path"] } }, "a path in a host"],
  [{ result: { ...C_RESULT, hosts: ["evil.example:443"] } }, "a port in a host"],
  [{ result: { ...C_RESULT, hosts: [""] } }, "an empty host"],
  [{ result: { ...C_RESULT, hosts: [["a".repeat(63), "b".repeat(63), "c".repeat(63), "d".repeat(62)].join(".")] } }, "a 254-character host (reviewer A T2: the cap is 253)"],
  [{ result: { hosts: [], hosts_more: 0 } }, "a result without an outcome (reviewer A T5)"],
];

for (const [over, why] of C_WRONG) {
  pendingT0162C(`T-0162 C-1: the tool-call table rejects ${why}`, async () => {
    const decode = await cDecode();
    const got = decode(JSON.stringify({ ...C_FINAL, ...over }));
    assert.equal(got.kind, "malformed", `C-1: ${why}; decode() said ${JSON.stringify(got)}`);
    assert.equal(got.type, "astra.tool.call", "…and says which type it was, for the counter");
  });
}

// --- T-0180 group A: astra.thought on the page's side of the contract ------------------------------
//
// docs/tasks/T-0180.md A1/A2. The corpus rows named thought_* are checked against the real Python
// codec by tests/unit/test_hud_constants_parity.py (the Python side is the spec); here the page's
// decoder must reach the same verdict for each. Expected strings are written out by hand.

const THOUGHT_ROWS = CORPUS.filter((c) => c.name.startsWith("thought_"));

const THOUGHT_OK = {
  type: "astra.thought",
  turn_id: "t4",
  request_id: "chatcmpl-s1-t4-r1",
  attempt_id: "t4-a1",
  step: 1,
  kind: "step",
  text: "The page shows a search box.",
  omitted_chars: 0,
  end: "open",
};

test("T-0180 A2: the thought corpus rows are the task's list, and the cap rows sit on 16000", () => {
  // A fixture premise, green before the implementation: the rows the page half below relies on.
  const names = THOUGHT_ROWS.map((c) => c.name).sort();
  assert.deepEqual(names, [
    "thought_at_the_cap_in_astral_code_points",
    "thought_at_the_cap_with_omitted_chars",
    "thought_done_replayed",
    "thought_empty_attempt_id_is_malformed",
    "thought_extra_field_is_malformed",
    "thought_from_the_client_is_the_wrong_direction",
    "thought_markup_and_channel_tokens_stay_data",
    "thought_omitted_without_the_cap_is_malformed",
    "thought_open",
    "thought_over_the_cap_is_malformed",
    "thought_step_zero_is_malformed",
    "thought_unknown_end_is_malformed",
    "thought_unknown_kind_is_malformed",
  ]);
  const built = Object.fromEntries(
    THOUGHT_ROWS.filter((c) => c.build).map((c) => [c.name, [c.build.char, c.build.count, c.expect]]),
  );
  assert.deepEqual(built, {
    thought_at_the_cap_with_omitted_chars: ["x", 16000, "astra"],
    thought_over_the_cap_is_malformed: ["x", 16001, "malformed"],
    thought_at_the_cap_in_astral_code_points: ["\u{1F600}", 16000, "astra"],
  });
  const cap = JSON.parse(rawOf(THOUGHT_ROWS.find((c) => c.name === "thought_at_the_cap_with_omitted_chars")));
  assert.equal(cap.text.length, 16000);
  assert.equal(cap.omitted_chars, 4000, "A6: the at-cap row says how much was not carried");
});

pendingT0180A("T-0180 A2: every thought corpus row reaches the outcome the real codec gives it", async () => {
  const { decode } = await load("protocol.js", "T-0180 A2");
  const wrong = [];
  for (const c of THOUGHT_ROWS) {
    const got = decode(rawOf(c));
    if (got.kind !== c.expect) wrong.push(`${c.name}: expected ${c.expect}, got ${JSON.stringify(got).slice(0, 200)}`);
    else if (got.type !== "astra.thought") wrong.push(`${c.name}: type ${got.type}`);
  }
  assert.deepEqual(wrong, [], "A2: the page's verdicts equal the Python codec's, row by row");
});

pendingT0180A("T-0180 A1/A2: astra.thought is server-to-client on the page and never emitted by it", async () => {
  const { SERVER_TO_CLIENT, CLIENT_TYPES } = await load("protocol.js", "T-0180 A2");
  assert.ok(new Set(SERVER_TO_CLIENT).has("astra.thought"), `A2: SERVER_TO_CLIENT is ${JSON.stringify(SERVER_TO_CLIENT)}`);
  assert.ok(!CLIENT_TYPES.includes("astra.thought"), "A1: the page never sends a thought");
  const row = THOUGHT_ROWS.find((c) => c.name === "thought_from_the_client_is_the_wrong_direction");
  assert.equal(row.direction_verdict, "wrong_direction", "the corpus row's server-side verdict (pinned in Python)");
});

pendingT0180A("T-0180 A2: markup and channel tokens in a thought decode as data, byte for byte", async () => {
  const { decode } = await load("protocol.js", "T-0180 A2");
  const row = THOUGHT_ROWS.find((c) => c.name === "thought_markup_and_channel_tokens_stay_data");
  const got = decode(rawOf(row));
  assert.equal(got.kind, "astra", JSON.stringify(got));
  assert.equal(got.msg.text, "<|channel>thought\nline two</script><b>x</b>");
  assert.deepEqual(got.msg, JSON.parse(rawOf(row)), "…every field the server sent, unchanged");
});

pendingT0180A("T-0180 A1/A2: the cap counts code points, not UTF-16 units", async () => {
  const { decode } = await load("protocol.js", "T-0180 A2");
  const astral = "\u{1F600}"; // one code point, two UTF-16 units, four UTF-8 bytes
  // 8001 code points are 16002 UTF-16 units: under the cap in code points, over it in units.
  const under = decode(JSON.stringify({ ...THOUGHT_OK, text: astral.repeat(8001) }));
  assert.equal(under.kind, "astra", `A1: 8001 code points are under 16000; got ${JSON.stringify(under).slice(0, 200)}`);
  const atCap = decode(JSON.stringify({ ...THOUGHT_OK, text: astral.repeat(16000), omitted_chars: 3 }));
  assert.equal(atCap.kind, "astra", "A1: omitted_chars > 0 is legal with text at the cap, in code points");
  const over = decode(JSON.stringify({ ...THOUGHT_OK, text: astral.repeat(16001) }));
  assert.equal(over.kind, "malformed", "A1: 16001 code points are over the cap");
  const short = decode(JSON.stringify({ ...THOUGHT_OK, text: astral.repeat(15999), omitted_chars: 1 }));
  assert.equal(short.kind, "malformed", "A1: omitted_chars > 0 with text one code point below the cap");
});

pendingT0180A("T-0180 A1/A2: every kind and every end decode; a missing required field does not", async () => {
  const { decode } = await load("protocol.js", "T-0180 A2");
  for (const kind of ["step", "reask", "retry", "repair", "reground"])
    assert.equal(decode(JSON.stringify({ ...THOUGHT_OK, kind })).kind, "astra", `kind ${kind}`);
  for (const end of ["open", "done", "stopped_repeating", "interrupted", "error"])
    assert.equal(decode(JSON.stringify({ ...THOUGHT_OK, end })).kind, "astra", `end ${end}`);
  for (const field of ["turn_id", "request_id", "attempt_id", "step", "kind", "text", "omitted_chars", "end"]) {
    const missing = { ...THOUGHT_OK };
    delete missing[field];
    const got = decode(JSON.stringify(missing));
    assert.equal(got.kind, "malformed", `A1: ${field} is required; decode() said ${JSON.stringify(got)}`);
    assert.equal(got.type, "astra.thought");
  }
  for (const [field, value] of [["turn_id", ""], ["request_id", ""], ["omitted_chars", -1]]) {
    const got = decode(JSON.stringify({ ...THOUGHT_OK, [field]: value }));
    assert.equal(got.kind, "malformed", `A1: ${field} = ${JSON.stringify(value)}`);
  }
  assert.equal(decode(JSON.stringify({ ...THOUGHT_OK, replay: true })).kind, "astra", "replay is a bool field");
});

// --- T-0181 group A: astra.mic.mute / astra.mic.state on the page's side of the contract ----------
//
// AstraMicMute(muted: bool), client->server; AstraMicState(muted: bool, available: bool,
// reason: str | None = None), server->client. Malformed: available false with muted true, available
// false with a null (or absent) reason, available true with a non-null reason; booleans are strict
// (no "true", no 1). codec.encode drops None, so an available state arrives without reason. The
// Python half (and the parity of every case with the real codec) is
// tests/contract/test_t0181_mic_wire.py.

const MIC_LIVE = { type: "astra.mic.state", muted: false, available: true };
const MIC_GONE = {
  type: "astra.mic.state",
  muted: false,
  available: false,
  reason: "this audio path has no capture seam",
};

pendingT0181A("T-0181 A1: astra.mic.state is dispatched and decodes as a typed message; astra.mic.mute is not dispatched", async () => {
  const { decode, SERVER_TO_CLIENT } = await load("protocol.js", "T-0181 A1");
  const set = new Set(SERVER_TO_CLIENT);
  assert.ok(set.has("astra.mic.state"), "A1: the page dispatches the mic state; SERVER_TO_CLIENT is " + JSON.stringify([...set]));
  assert.ok(!set.has("astra.mic.mute"), "A1: astra.mic.mute is client->server only");
  for (const ok of [MIC_LIVE, { ...MIC_LIVE, muted: true }, { ...MIC_LIVE, reason: null }, MIC_GONE]) {
    const got = decode(JSON.stringify(ok));
    assert.equal(got.kind, "astra", `a legal state must dispatch: ${JSON.stringify(ok)} -> ${JSON.stringify(got)}`);
    assert.equal(got.type, "astra.mic.state");
    assert.deepEqual(got.msg, ok, "...carrying every field the server sent, unchanged");
  }
  for (const muted of [true, false]) {
    const got = decode(JSON.stringify({ type: "astra.mic.mute", muted }));
    assert.equal(got.kind, "astra", `the page's own egress decodes (parity with the contract): ${JSON.stringify(got)}`);
  }
});

const MIC_WRONG = [
  [{ ...MIC_LIVE, muted: true, available: false, reason: "x" }, "available false and muted true"],
  [{ ...MIC_GONE, reason: null }, "available false and a null reason"],
  [{ type: "astra.mic.state", muted: false, available: false }, "available false and no reason"],
  [{ ...MIC_LIVE, reason: "a reason" }, "available true and a reason"],
  [{ ...MIC_LIVE, muted: "false" }, "a string muted"],
  [{ ...MIC_LIVE, available: 1 }, "an integer available"],
  [{ type: "astra.mic.state", available: true }, "no muted"],
  [{ type: "astra.mic.state", muted: false }, "no available"],
  [{ ...MIC_LIVE, replay: true }, "an undeclared field (no replay on this type)"],
  [{ ...MIC_GONE, reason: 5 }, "a numeric reason"],
];

for (const [wire, why] of MIC_WRONG) {
  pendingT0181A(`T-0181 A1: the mic state table rejects ${why}`, async () => {
    const { decode } = await load("protocol.js", "T-0181 A1");
    const got = decode(JSON.stringify(wire));
    assert.equal(got.kind, "malformed", `A1: ${why}; decode() said ${JSON.stringify(got)}`);
    assert.equal(got.type, "astra.mic.state", "...and says which type it was, for the counter");
  });
}

pendingT0181A("T-0181 A1: the mute table is strict too", async () => {
  const { decode } = await load("protocol.js", "T-0181 A1");
  for (const wire of [
    { type: "astra.mic.mute" },
    { type: "astra.mic.mute", muted: "true" },
    { type: "astra.mic.mute", muted: 1 },
    { type: "astra.mic.mute", muted: null },
    { type: "astra.mic.mute", muted: true, source: "mic" },
  ]) {
    const got = decode(JSON.stringify(wire));
    assert.equal(got.kind, "malformed", `A1: ${JSON.stringify(wire)}; decode() said ${JSON.stringify(got)}`);
  }
});

// --- T-0193 group A: astra.turn.stop on the page's side of the contract (mirror census) -----------
//
// AstraTurnStop(turn_id: non-empty str), client->server only; no server->client type is added (the
// echo is astra.turn.state). Strict xfail while tests/contract/test_t0193_stop_wire.py (the Python
// half of A-1, with the parity of every case against the real codec) carries pending(task="T-0193");
// removing that marker is the one activation step for both languages.
const T0193_WIRE = fileURLToPath(new URL("tests/contract/test_t0193_stop_wire.py", REPO));
const T0193_A_PENDING =
  existsSync(T0193_WIRE) &&
  readFileSync(T0193_WIRE, "utf8").includes('pending(task="T-0193")') &&
  process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0193A(name, fn) {
  test(name + (T0193_A_PENDING ? " [pending T-0193 A: strict xfail while test_t0193_stop_wire.py is pending]" : ""), async (t) => {
    if (!T0193_A_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0193 group A row passed while tests/contract/test_t0193_stop_wire.py " +
          'still carries pending(task="T-0193"). Either group A has landed (remove the marker) or the ' +
          "row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0193 A pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

pendingT0193A("T-0193 A-1: astra.turn.stop is mirrored (decodes as a typed message) and never dispatched", async () => {
  const { decode, SERVER_TO_CLIENT } = await load("protocol.js", "T-0193 A-1");
  assert.ok(!new Set(SERVER_TO_CLIENT).has("astra.turn.stop"), "A-1: astra.turn.stop is client->server only");
  for (const turn_id of ["t1", "t12", " "]) {
    const wire = { type: "astra.turn.stop", turn_id };
    const got = decode(JSON.stringify(wire));
    assert.equal(got.kind, "astra", `the page's own egress decodes (parity with the contract): ${JSON.stringify(got)}`);
    assert.deepEqual(got.msg, wire);
  }
});

pendingT0193A("T-0193 A-1: the stop table is strict", async () => {
  const { decode } = await load("protocol.js", "T-0193 A-1");
  for (const wire of [
    { type: "astra.turn.stop" },
    { type: "astra.turn.stop", turn_id: "" },
    { type: "astra.turn.stop", turn_id: 3 },
    { type: "astra.turn.stop", turn_id: null },
    { type: "astra.turn.stop", turn_id: true },
    { type: "astra.turn.stop", turn_id: ["t1"] },
    { type: "astra.turn.stop", turn_id: "t1", x: 1 },
  ]) {
    const got = decode(JSON.stringify(wire));
    assert.equal(got.kind, "malformed", `A-1: ${JSON.stringify(wire)}; decode() said ${JSON.stringify(got)}`);
    assert.equal(got.type, "astra.turn.stop", "...and says which type it was, for the counter");
  }
});
