// T-0213 A2r-1 — hud/src/protocol.js decodes astra.settings.state as contracts/protocol.py does: the
// instructions cap, both sides. `instructions` of exactly `instructions_max` code points decodes; one more
// is malformed. The text is astral (U+1F600: one code point, two UTF-16 units), so a cap counted in
// `string.length` refuses the at-cap state. Values are typed from the criterion (A1-4's 1000), not read off
// the page. Kills `>` -> `>=` (at cap), a dropped check (over cap) and a UTF-16 count (at cap, astral).
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { CLIENT_TYPES, SERVER_TO_CLIENT, decode } from "../../hud/src/protocol.js";

const CAP = 1000;
const ASTRAL = "\u{1F600}";

function state(instructions) {
  return {
    type: "astra.settings.state",
    rev: 0,
    items: [
      {
        key: "action_outcome",
        value: false,
        source: "default",
        kind: "next_turn",
        changeable: true,
      },
    ],
    system_sha12: "0123456789ab",
    instructions,
    instructions_max: CAP,
    refused: [],
    replay: false,
  };
}

test("A2r-1: instructions at the cap (astral, 1000 code points) decode", () => {
  const text = ASTRAL.repeat(CAP);
  assert.equal(text.length, 2 * CAP, "premise: the text is 2000 UTF-16 units");
  const got = decode(JSON.stringify(state(text)));
  assert.equal(got.kind, "astra", `at the cap is legal; got ${JSON.stringify(got).slice(0, 300)}`);
  assert.equal(got.type, "astra.settings.state");
  assert.equal(got.msg.instructions, text);
});

test("A2r-1: instructions at the cap (ASCII, 1000 code points) decode", () => {
  const got = decode(JSON.stringify(state("a".repeat(CAP))));
  assert.equal(got.kind, "astra", `at the cap is legal; got ${JSON.stringify(got).slice(0, 300)}`);
});

test("A2r-1: instructions one code point over the cap are malformed", () => {
  for (const text of [ASTRAL.repeat(CAP + 1), "a".repeat(CAP + 1)]) {
    const got = decode(JSON.stringify(state(text)));
    assert.equal(got.kind, "malformed", `over the cap; got ${got.kind}`);
    assert.equal(got.type, "astra.settings.state");
  }
});

test("A2r-1: the cap is the message's own instructions_max", () => {
  const small = { ...state("abcd"), instructions_max: 3 };
  assert.equal(decode(JSON.stringify(small)).kind, "malformed");
  const exact = { ...state("abc"), instructions_max: 3 };
  assert.equal(decode(JSON.stringify(exact)).kind, "astra");
});

// T-0213 R-10: an item's effective kind may be "not_here" (not changeable in this mode, no relaunch); the page's
// decoder accepts it as contracts/protocol.py does, and still refuses a kind outside the five.
test(
  "R-10: an item of kind not_here decodes; an unknown kind stays malformed",
  () => {
    const ok = state("");
    ok.items = [{ key: "reground", source: "default", kind: "not_here", changeable: false }];
    const got = decode(JSON.stringify(ok));
    assert.equal(
      got.kind,
      "astra",
      `not_here is a wire kind (R-10); got ${JSON.stringify(got).slice(0, 300)}`,
    );
    assert.equal(got.msg.items[0].kind, "not_here");
    const bad = state("");
    bad.items = [{ key: "reground", source: "default", kind: "nowhere", changeable: false }];
    assert.equal(decode(JSON.stringify(bad)).kind, "malformed", "an unknown kind is still refused");
  },
);

// --- T-0213 A2a-1 / C: astra.settings.set on the page's side --------------------------------------------------
//
// AstraSettingsSet {rev: int, changes: object, stop_turn: bool = false}, client -> server only. The page's decoder
// mirrors it for parity with the contract (as astra.mic.mute and astra.turn.stop are); it is never dispatched (not
// in SERVER_TO_CLIENT), and T-0008's CLIENT_TYPES is not widened: the panel's own gate (group C,
// hud/src/settings/gate.js SETTINGS_CLIENT_TYPES = ["astra.settings.set"]) is what may send it.
//
// Pending rule (as T-0193 A's rows in protocol.test.mjs): the A2a-1 rows are strict xfail while
// tests/unit/test_t0213_settings_log.py carries pending(task="T-0213"); the gate row is group C's (hud/src/settings/
// gate.js does not exist yet), strict xfail while tests/integration/test_hud_settings_instructions.py (group C's
// test module) is absent or carries that marker. ASTRA_RUN_PENDING=1 runs both for real.
const T0213_REPO = new URL("../../", import.meta.url);
const T0213_MARK = 'pending(task="T-0213")';

function pendingWhile(relPath, absentIsPending) {
  if (process.env.ASTRA_RUN_PENDING === "1") return false;
  const file = fileURLToPath(new URL(relPath, T0213_REPO));
  if (!existsSync(file)) return absentIsPending;
  return readFileSync(file, "utf8").includes(T0213_MARK);
}

const A2A_PENDING = pendingWhile("tests/unit/test_t0213_settings_log.py", false);
const C_PENDING = pendingWhile("tests/integration/test_hud_settings_instructions.py", true);

function strictXfail(pending, label, name, fn) {
  test(name + (pending ? ` [pending ${label}: strict xfail]` : ""), async (t) => {
    if (!pending) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null)
      assert.fail(`XPASS(strict): this ${label} row passed while it is pending; remove the marker or fix the row`);
    t.diagnostic(`expected failure (${label} pending): ` + String(failure && failure.message).split("\n")[0]);
  });
}

const SET_TYPE = "astra.settings.set";

strictXfail(
  A2A_PENDING,
  "T-0213 A2a-1",
  "A2a-1: astra.settings.set is mirrored (decodes as a typed message) and never dispatched",
  () => {
    assert.ok(!new Set(SERVER_TO_CLIENT).has(SET_TYPE), "A2a-1: client -> server only");
    assert.ok(!new Set(CLIENT_TYPES).has(SET_TYPE), "T-0008's CLIENT_TYPES is not widened (the panel's gate sends it)");
    for (const wire of [
      { type: SET_TYPE, rev: 0, changes: { visual_tokens: 560 } },
      { type: SET_TYPE, rev: 3, changes: { user_instructions: "be brief", action_outcome: true }, stop_turn: true },
      { type: SET_TYPE, rev: 1, changes: {}, stop_turn: false },
    ]) {
      const got = decode(JSON.stringify(wire));
      assert.equal(got.kind, "astra", `the page's own egress decodes (parity): ${JSON.stringify(got).slice(0, 300)}`);
      assert.equal(got.type, SET_TYPE);
      assert.deepEqual(got.msg, wire);
    }
  },
);

strictXfail(
  A2A_PENDING,
  "T-0213 A2a-1",
  "A2a-1: the set table is strict (unknown field, missing or mistyped rev/changes/stop_turn)",
  () => {
    for (const wire of [
      { type: SET_TYPE, rev: 0, changes: {}, extra: 1 },
      { type: SET_TYPE, changes: {} },
      { type: SET_TYPE, rev: 0 },
      { type: SET_TYPE, rev: "0", changes: {} },
      { type: SET_TYPE, rev: 0.5, changes: {} },
      { type: SET_TYPE, rev: 0, changes: [] },
      { type: SET_TYPE, rev: 0, changes: null },
      { type: SET_TYPE, rev: 0, changes: {}, stop_turn: "yes" },
    ]) {
      const got = decode(JSON.stringify(wire));
      assert.equal(got.kind, "malformed", `A2a-1: ${JSON.stringify(wire)}; decode() said ${JSON.stringify(got)}`);
      assert.equal(got.type, SET_TYPE, "...and says which type it was, for the counter");
    }
  },
);

strictXfail(
  C_PENDING,
  "T-0213 C",
  "C: the settings gate allows astra.settings.set (its only client type), which the decoder accepts",
  async () => {
    let gate;
    try {
      gate = await import(new URL("hud/src/settings/gate.js", T0213_REPO).href);
    } catch (err) {
      assert.fail(`T-0213 C: hud/src/settings/gate.js does not exist yet (${String(err && err.message).split("\n")[0]})`);
    }
    assert.deepEqual(gate.SETTINGS_CLIENT_TYPES, [SET_TYPE], "T-0213 C: SETTINGS_CLIENT_TYPES = [astra.settings.set]");
    const got = decode(JSON.stringify({ type: gate.SETTINGS_CLIENT_TYPES[0], rev: 0, changes: {} }));
    assert.equal(got.kind, "astra", "what the gate allows, the page's decoder accepts");
  },
);
