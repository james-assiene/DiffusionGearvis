// T-0037 group B — the page side of the hello replay: hud/src/protocol.js mirrors the `replay`
// field and hud/src/state.js applies a replayed message idempotently.
//
// Criteria 10, 11, 12 and 13 of docs/tasks/T-0037.md.
//
// This file *skips itself* while the page does not mention `replay`: node test files cannot carry
// pytest's `pending` marker and tests/unit/test_hud_js.py asserts that `node --test tests/hud`
// exits 0. The sentinel is the pair (protocol.js, state.js) — group B lands both together — so
// every case below runs for real the moment the implementer touches either one. Flipping the
// sentinel by writing the word "replay" into a comment and leaving the reducer alone turns these
// into ordinary failing tests, which is the intended failure mode.
//
// THE SEAM (the task leaves the JS shape to the test-author; this is it):
//
//   hud/src/protocol.js — the field tables of astra.transcript, astra.spoken, astra.ui.surface,
//     astra.spoken.played and astra.turn.state each gain an optional `replay: bool()`. Nothing
//     else moves: a genuinely unknown key is still "extra inputs are not permitted" (criterion
//     10's negative — the strictness mirror is extended by one field, not removed).
//
//   hud/src/state.js — reduce(state, message, nowMs) treats a replayed message as the message it
//     is a copy of, idempotently:
//       * astra.spoken: one entry per utterance_id, whatever the order and however often it
//         arrives (this is already withUtterance's rule; replay must not bypass it);
//       * astra.transcript: transcript lines carry their turn_id, so the replayed final line of
//         a turn the page already has *replaces* that turn's line instead of appending a second
//         copy — transcriptLines(state) -> [{ text, final, turn_id }];
//       * astra.spoken.played: unchanged (T-0030 criteria 17/18) — the cut comes from the
//         server's measurement, and a replayed astra.spoken arriving after it still drops it.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);
const STATE_JS = fileURLToPath(new URL("hud/src/state.js", REPO));
const PROTOCOL_JS = fileURLToPath(new URL("hud/src/protocol.js", REPO));
const mentions = (path) => existsSync(path) && readFileSync(path, "utf8").includes("replay");
const INSTALLED = mentions(STATE_JS) || mentions(PROTOCOL_JS);
const opts = INSTALLED
  ? {}
  : {
      skip:
        "T-0037 group B is not implemented yet: neither hud/src/protocol.js nor hud/src/state.js " +
        "mentions `replay`. This file is the acceptance suite of docs/tasks/T-0037.md criteria " +
        "10-13 (the strictness mirror and the idempotent reducer); it runs the moment group B " +
        "lands. Node tests cannot carry pytest's `pending` marker and tests/unit/test_hud_js.py " +
        "requires `node --test tests/hud` to exit 0.",
    };

// node 18 treats a bare .js file as CommonJS unless a package.json in scope says
// {"type":"module"}; the sources are copied once into a temp directory that carries the flag.
// Same shim as tests/hud/played.test.mjs.
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
      `T-0037 group B edits hud/src/${name}, which must stay importable in node as an ES module ` +
        `with no DOM at module top level (T-0008's rule for state.js). node said: ${err.message}`,
    );
  }
}

// --- a DOM double: enough of Document/Element for renderSpoken (copied from played.test.mjs) ----

function fakeDoc() {
  const doc = {
    createElement(tag) {
      return element(doc, tag);
    },
    createTextNode(text) {
      return { nodeType: 3, children: [], get textContent() { return text; } };
    },
  };
  return doc;
}

function element(doc, tag) {
  const el = {
    tagName: String(tag).toUpperCase(),
    ownerDocument: doc,
    className: "",
    dataset: {},
    children: [],
    _text: "",
    classList: {
      add(...names) {
        const have = el.className ? el.className.split(/\s+/) : [];
        for (const n of names) if (!have.includes(n)) have.push(n);
        el.className = have.join(" ");
      },
      contains: (n) => (el.className || "").split(/\s+/).includes(n),
    },
    get textContent() {
      return el._text + el.children.map((c) => c.textContent).join("");
    },
    set textContent(value) {
      el.children = [];
      el._text = String(value);
    },
    append(...nodes) {
      for (const node of nodes) el.children.push(typeof node === "string" ? doc.createTextNode(node) : node);
    },
    appendChild(node) {
      el.append(node);
      return node;
    },
    replaceChildren(...nodes) {
      el.children = [];
      el._text = "";
      el.append(...nodes);
    },
    setAttribute(name, value) {
      if (name.startsWith("data-")) {
        const key = name.slice(5).replace(/-([a-z])/g, (_m, c) => c.toUpperCase());
        el.dataset[key] = String(value);
      } else if (name === "class") {
        el.className = String(value);
      }
    },
    getAttribute(name) {
      if (name.startsWith("data-")) {
        const key = name.slice(5).replace(/-([a-z])/g, (_m, c) => c.toUpperCase());
        return key in el.dataset ? el.dataset[key] : null;
      }
      return name === "class" ? el.className : null;
    },
  };
  return el;
}

function walk(el) {
  const out = [];
  for (const child of el.children || []) {
    if (child.nodeType === 3) continue;
    out.push(child, ...walk(child));
  }
  return out;
}

const withUtteranceId = (root) => walk(root).filter((e) => e.dataset && e.dataset.utteranceId);
const unplayedIn = (el) => walk(el).filter((e) => e.dataset && "unplayed" in e.dataset);
const squeeze = (s) => String(s).replace(/\s+/g, " ").trim();

// --- the wire, live and replayed ----------------------------------------------------------------

const U1 = "Here is a quick check.";
const U2 = "Try the five questions carefully please."; // 6 words; the cut below falls after 3
const QUESTION = "what is on this screen";

const spoken = (turn, utterance, text, replay = false) => ({
  type: "astra.spoken",
  turn_id: turn,
  utterance_id: utterance,
  text,
  replay,
});
const transcript = (turn, text, final, replay = false) => ({
  type: "astra.transcript",
  turn_id: turn,
  text,
  final,
  replay,
});
const played = (turn, utterances, replay = false) => ({
  type: "astra.spoken.played",
  turn_id: turn,
  played_ms: 1400,
  utterances,
  replay,
});
const entry = (utterance_id, played_text, played_words, complete) => ({
  utterance_id,
  played_text,
  played_words,
  complete,
});

/** The page as it was before the socket dropped: one turn, two utterances, the question. */
async function pageMidTurn() {
  const { initialState, reduce } = await load("state.js");
  let state = initialState();
  state = reduce(state, { type: "astra.turn.state", state: "SPEAKING", turn_id: "t1" }, 0);
  state = reduce(state, transcript("t1", QUESTION, true), 10);
  state = reduce(state, spoken("t1", "u1", U1), 20);
  state = reduce(state, spoken("t1", "u2", U2), 30);
  return { state, reduce, initialState };
}

const utterancesOf = (state, turnId = "t1") =>
  (state.spoken.find((g) => g.turn_id === turnId) || { utterances: [] }).utterances;

// --- criterion 10 ---------------------------------------------------------------------------------

test("T-0037 criterion 10: a replayed message decodes; an unknown field still does not", opts, async () => {
  const { decode } = await load("protocol.js");

  const marked = [
    spoken("t1", "u1", U1, true),
    transcript("t1", QUESTION, true, true),
    played("t1", [entry("u1", "Here is", 2, false)], true),
    { type: "astra.turn.state", state: "INTERRUPTED", turn_id: "t1", replay: true },
    {
      type: "astra.ui.surface",
      surface_id: "turn-t1",
      canvas_idx: 1,
      final: true,
      messages: [{ version: "v0.9.1" }],
      repaired: 0,
      replay: true,
    },
  ];
  for (const message of marked) {
    const out = decode(JSON.stringify(message));
    assert.equal(
      out.kind,
      "astra",
      `criterion 10: hud/src/protocol.js mirrors contracts/protocol.py, which adds replay to ` +
        `${message.type}. Without the mirror every replayed message is kind:"malformed" — ` +
        `counted, ignored, invisible, and the reloading page stays empty. decode said ` +
        `${JSON.stringify(out)}`,
    );
    assert.equal(out.msg.replay, true, "the flag survives decoding");
  }

  // ... and the strictness mirror is extended by one field, not removed (the landmine).
  const junk = decode(JSON.stringify({ ...spoken("t1", "u1", U1, true), nope: 1 }));
  assert.equal(
    junk.kind,
    "malformed",
    "criterion 10, negative: contracts/base.Contract forbids unknown fields and this page " +
      "mirrors that literally; a message with a genuinely unknown key stays malformed",
  );
  assert.match(junk.error, /extra inputs are not permitted/);

  const wrongType = decode(JSON.stringify({ ...spoken("t1", "u1", U1), replay: "yes" }));
  assert.equal(
    wrongType.kind,
    "malformed",
    "criterion 10: replay is a bool on the Python side, so a string is malformed here too",
  );

  const absent = decode(JSON.stringify({ type: "astra.spoken", turn_id: "t1", utterance_id: "u1", text: U1 }));
  assert.equal(absent.kind, "astra", "replay has a default: a live message may omit it entirely");
});

// --- criterion 11 ---------------------------------------------------------------------------------

test("T-0037 criterion 11: replaying a turn the page already has leaves one entry per utterance and one transcript line", opts, async () => {
  // The adversarial shape: ws.js's backoff reconnect. The socket dropped, the JS state survived,
  // the page re-hellos and the server replays the open turn. Nothing may double.
  const { state, reduce } = await pageMidTurn();
  let next = state;
  for (const message of [
    transcript("t1", QUESTION, true, true),
    spoken("t1", "u1", U1, true),
    spoken("t1", "u2", U2, true),
  ]) {
    next = reduce(next, message, 100);
  }

  const utterances = utterancesOf(next);
  assert.deepEqual(
    utterances.map((u) => u.utterance_id),
    ["u1", "u2"],
    "criterion 11: exactly one entry per utterance_id after the replay — never a second copy",
  );
  assert.deepEqual(utterances.map((u) => u.text), [U1, U2]);
  assert.equal(next.spoken.length, 1, "one group for the one turn");

  const lines = next.transcript;
  assert.equal(
    lines.length,
    1,
    `criterion 11: exactly one transcript line per turn. The page already had the turn's final ` +
      `line; the replayed copy must replace it, not append a duplicate final line. Got ` +
      `${JSON.stringify(lines)}`,
  );
  assert.equal(lines[0].text, QUESTION);
  assert.equal(lines[0].final, true);
  assert.equal(
    lines[0].turn_id,
    "t1",
    "criterion 11: transcript lines carry their turn_id — that is what lets a replayed line " +
      "replace the line of *that* turn rather than whatever happens to be last",
  );

  // ... and a second turn still opens its own line (the replacement is per turn, not global).
  const two = reduce(next, transcript("t2", "and now", false), 200);
  assert.equal(two.transcript.length, 2, "a new turn is a new line");
  assert.deepEqual(two.transcript.map((l) => l.turn_id), ["t1", "t2"]);
});

// --- criterion 12 ---------------------------------------------------------------------------------

test("T-0037 criterion 12: a replayed played message restores the struck tail", opts, async () => {
  const { state, reduce } = await pageMidTurn();
  const cut = [entry("u1", "Here is a quick check. ", 5, true), entry("u2", "Try the five", 3, false)];

  // The reload order of criterion 1: the spoken lines first, then the played report.
  let next = reduce(state, spoken("t1", "u1", U1, true), 100);
  next = reduce(next, spoken("t1", "u2", U2, true), 101);
  next = reduce(next, played("t1", cut, true), 102);

  const [first, second] = utterancesOf(next);
  assert.equal(first.complete, true, "the whole first utterance was heard");
  assert.equal(
    second.played_words,
    3,
    "criterion 12: the cut comes from the server's measurement in the replayed " +
      "astra.spoken.played and from nothing else (T-0030 criteria 17/18)",
  );
  assert.equal(second.complete, false);
  assert.equal(second.text, U2, "the pane keeps AstraSpoken.text, not played_text");

  // The negative twin: T-0030's rule is unchanged by the replay marker. A replayed astra.spoken
  // arriving *after* a played message for the same utterance still drops the cut — which is why
  // criterion 1 puts every played message last in the reply.
  const rewritten = reduce(next, spoken("t1", "u2", U2, true), 103);
  for (const key of ["played_words", "complete", "played_text"]) {
    assert.equal(
      utterancesOf(rewritten)[1][key],
      undefined,
      `criterion 12: an astra.spoken for an utterance drops its played state whether it is a ` +
        `replay or not (${key} survived). The ordering in the hello reply is what protects the ` +
        `strike, not a special case in the reducer`,
    );
  }

  // ... and no page-side timing dims a word: a replayed INTERRUPTED chip and a minute of clock
  // change nothing (T-0030 criterion 22, re-run over the replay).
  let timed = reduce(state, spoken("t1", "u1", U1, true), 200);
  for (let t = 0; t <= 60000; t += 10000) {
    timed = reduce(timed, { type: "astra.turn.state", state: "INTERRUPTED", turn_id: "t1", replay: true }, t);
  }
  assert.ok(timed.interrupted.includes("t1"), "premise: the turn really was marked interrupted");
  for (const u of utterancesOf(timed)) {
    assert.equal(u.played_words, undefined, "criterion 12, negative: no chip and no clock cuts a word");
  }
});

// --- criterion 13 ---------------------------------------------------------------------------------

test("T-0037 criterion 13: the spoken pane renders one [data-utterance-id] per utterance after replay", opts, async () => {
  const { renderSpoken } = await load("panels/spoken.js");
  const { state, reduce } = await pageMidTurn();
  let next = state;
  for (const message of [
    spoken("t1", "u1", U1, true),
    spoken("t1", "u2", U2, true),
    played("t1", [entry("u2", "Try the five", 3, false)], true),
  ]) {
    next = reduce(next, message, 300);
  }

  const doc = fakeDoc();
  const root = element(doc, "div");
  renderSpoken(root, next.spoken, next.interrupted);

  const ids = withUtteranceId(root).map((e) => e.dataset.utteranceId);
  assert.deepEqual(
    ids,
    ["u1", "u2"],
    "criterion 13: after a replay the pane holds exactly one [data-utterance-id] element per " +
      "utterance — replay is a reducer and wire concern and hud/src/panels/spoken.js does not " +
      `move. The pane holds ${JSON.stringify(ids)}`,
  );
  const second = withUtteranceId(root)[1];
  assert.equal(squeeze(second.textContent), U2, "the full text is in the DOM, struck not removed");
  const tails = unplayedIn(second);
  assert.equal(tails.length, 1, "criterion 12/13: the struck tail is rendered after a reload");
  assert.equal(squeeze(tails[0].textContent), "questions carefully please.");
});
