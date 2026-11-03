// T-0030 group C — the unheard tail: hud/src/state.js reduces astra.spoken.played and
// hud/src/panels/spoken.js renders the words nobody heard inside a [data-unplayed] element.
//
// Criteria 17, 18, 19 and 22 of docs/tasks/T-0030.md. The style (criterion 20) and the rendered
// strike-through are DOM-in-a-real-browser and live in tests/integration/test_hud_headless.py;
// everything here runs on the pure reducer and a DOM double, which is why it is in the fast
// `node --test tests/hud` suite.
//
// This file *skips itself* while hud/src/state.js does not mention astra.spoken.played: node test
// files cannot carry pytest's `pending` marker and tests/unit/test_hud_js.py asserts that
// `node --test tests/hud` exits 0. The moment group C's reducer lands, every case below runs for
// real. (The sentinel is state.js, so the protocol.js rows in tests/hud/protocol.test.mjs flip at
// the same moment: group C lands both files together.)
//
// THE SEAM (the task leaves the JS shape to the test-author; this is it):
//
//   hud/src/state.js — reduce(state, message, nowMs)
//     case "astra.spoken.played": for each entry whose { turn_id, utterance_id } the page has
//     already seen through astra.spoken, the matching utterance in state.spoken gains
//
//         { utterance_id, text, played_words: int, complete: bool, played_text: string }
//
//     Nothing else changes: no group is created, no line is created, no text is rewritten (the
//     pane keeps displaying AstraSpoken.text — landmine 1: played_text is built from the
//     committed source, which is a *different string*; played_words is the cut index and it is
//     valid in both). A later astra.spoken for the same utterance drops the three played keys.
//
//   hud/src/panels/spoken.js — renderSpoken(root, groups, interrupted)
//     inside the existing [data-utterance-id] element: the words from index played_words onward
//     go inside one [data-unplayed] element; complete === true -> no such element; no played
//     state -> today's rendering. The full text is always in the DOM, dimmed and struck, never
//     removed (docs/02 §4: "the HUD may show unspoken text struck through").
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);
const STATE_JS = fileURLToPath(new URL("hud/src/state.js", REPO));
const INSTALLED = existsSync(STATE_JS) && readFileSync(STATE_JS, "utf8").includes("astra.spoken.played");
const opts = INSTALLED
  ? {}
  : {
      skip:
        "T-0030 group C is not implemented yet: hud/src/state.js does not reduce " +
        "astra.spoken.played. This file is the acceptance suite of docs/tasks/T-0030.md " +
        "criteria 17-19 and 22 (the played state and the [data-unplayed] tail); it runs the " +
        "moment that reducer lands. Node tests cannot carry pytest's `pending` marker and " +
        "tests/unit/test_hud_js.py requires `node --test tests/hud` to exit 0.",
    };

// node 18 treats a bare .js file as CommonJS unless a package.json in scope says
// {"type":"module"}; the sources are copied once into a temp directory that carries the flag.
// Same shim as tests/hud/protocol.test.mjs and tests/hud/assists.test.mjs.
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
      `T-0030 group C edits hud/src/${name}, which must stay importable in node as an ES module ` +
        `with no DOM at module top level (T-0008's rule for state.js). node said: ${err.message}`,
    );
  }
}

// --- a DOM double: enough of Document/Element for renderSpoken ---------------------------------

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
        const key = name
          .slice(5)
          .replace(/-([a-z])/g, (_m, c) => c.toUpperCase());
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

/** Every descendant of `el`, in document order (the double has no querySelectorAll). */
function walk(el) {
  const out = [];
  for (const child of el.children || []) {
    if (child.nodeType === 3) continue;
    out.push(child, ...walk(child));
  }
  return out;
}

const byUtterance = (root, id) => walk(root).find((e) => e.dataset && e.dataset.utteranceId === id);
const unplayedIn = (el) => walk(el).filter((e) => e.dataset && "unplayed" in e.dataset);
const squeeze = (s) => String(s).replace(/\s+/g, " ").trim();

// --- fixtures ---------------------------------------------------------------------------------

const U1 = "It is a stack trace.";
const U2 = "The error is on line four."; // 6 words; the cut below falls after 3
const spoken = (turn, utterance, text) => ({
  type: "astra.spoken",
  turn_id: turn,
  utterance_id: utterance,
  text,
});
const played = (turn, utterances, played_ms = 1400) => ({
  type: "astra.spoken.played",
  turn_id: turn,
  played_ms,
  utterances,
});
const entry = (utterance_id, played_text, played_words, complete) => ({
  utterance_id,
  played_text,
  played_words,
  complete,
});

async function stateWithTwoUtterances() {
  const { initialState, reduce } = await load("state.js");
  let state = initialState();
  state = reduce(state, spoken("t1", "u1", U1), 0);
  state = reduce(state, spoken("t1", "u2", U2), 10);
  return { state, reduce, initialState };
}

const utterancesOf = (state, turnId = "t1") =>
  (state.spoken.find((g) => g.turn_id === turnId) || { utterances: [] }).utterances;

// --- criterion 17 -------------------------------------------------------------------------------

test("T-0030 criterion 17: played state lands on the right utterance and nowhere else", opts, async () => {
  const { state, reduce } = await stateWithTwoUtterances();
  const next = reduce(
    state,
    played("t1", [entry("u1", "It is a stack trace. ", 5, true), entry("u2", "The error is", 3, false)]),
    20,
  );
  const [first, second] = utterancesOf(next);
  assert.equal(first.utterance_id, "u1");
  assert.equal(first.played_words, 5);
  assert.equal(first.complete, true);
  assert.equal(first.played_text, "It is a stack trace. ");
  assert.equal(second.played_words, 3);
  assert.equal(second.complete, false);
  assert.equal(second.played_text, "The error is");
  assert.equal(
    second.text,
    U2,
    "criterion 17: the pane keeps displaying AstraSpoken.text; played_text is a different " +
      "string (the committed source, landmine 1) and never overwrites it",
  );
  assert.equal(next.spoken.length, 1, "no group is created or duplicated");
  assert.deepEqual(next.transcript, state.transcript, "nothing else in the state moves");
  assert.deepEqual(next.interrupted, state.interrupted);
});

// --- criterion 18 -------------------------------------------------------------------------------

test("T-0030 criterion 18: a played message for an unknown utterance changes nothing", opts, async () => {
  const { state, reduce } = await stateWithTwoUtterances();
  const before = JSON.parse(JSON.stringify(state));

  const unknownUtterance = reduce(state, played("t1", [entry("u9", "Never sent.", 2, true)]), 30);
  assert.deepEqual(JSON.parse(JSON.stringify(unknownUtterance)), before,
    "criterion 18: an utterance the page never saw creates no line and no text");

  const unknownTurn = reduce(state, played("t9", [entry("u1", "It is", 2, false)]), 40);
  assert.deepEqual(JSON.parse(JSON.stringify(unknownTurn)), before,
    "criterion 18: nor does a turn the page never saw — and it must not fall through to the " +
      "utterance id in some other group");
});

test("T-0030 criterion 18, adversarial: a re-published astra.spoken clears the played state", opts, async () => {
  const { state, reduce } = await stateWithTwoUtterances();
  const cut = reduce(state, played("t1", [entry("u2", "The error is", 3, false)]), 50);
  assert.equal(utterancesOf(cut)[1].played_words, 3, "premise: the cut was recorded");

  const rewritten = reduce(cut, spoken("t1", "u2", "The error is on line five."), 60);
  const second = utterancesOf(rewritten)[1];
  assert.equal(second.text, "The error is on line five.");
  for (const key of ["played_words", "complete", "played_text"]) {
    assert.equal(
      second[key],
      undefined,
      `criterion 18: the cut was measured against the text that was played; a new text has not ` +
        `been measured, so ${key} must be dropped rather than carried over (the page must never ` +
        `strike a string it did not receive a measurement for)`,
    );
  }
});

test("T-0030 criterion 17: reduce stays pure", opts, async () => {
  const { state, reduce } = await stateWithTwoUtterances();
  const snapshot = JSON.stringify(state);
  const next = reduce(
    state,
    played("t1", [entry("u1", "It is", 2, false), entry("u2", "", 0, false)]),
    70,
  );
  assert.equal(JSON.stringify(state), snapshot, "reduce mutated the state it was handed");
  assert.notEqual(next, state, "and it returns a new state when something changed");
});

// --- criterion 19 -------------------------------------------------------------------------------

async function renderWith(playedEntries) {
  const { renderSpoken } = await load("panels/spoken.js");
  const { state, reduce } = await stateWithTwoUtterances();
  const next = playedEntries ? reduce(state, played("t1", playedEntries), 80) : state;
  const doc = fakeDoc();
  const root = element(doc, "div");
  renderSpoken(root, next.spoken, next.interrupted);
  return root;
}

test("T-0030 criterion 19: the tail renders inside [data-unplayed] and the full text stays in the DOM", opts, async () => {
  // (a) complete -> no [data-unplayed] at all; (b) a 3-of-6 cut -> the tail only; (c) zero words
  // played -> the whole utterance inside it. Hand-computed from U2's six words.
  const root = await renderWith([
    entry("u1", "It is a stack trace. ", 5, true),
    entry("u2", "The error is", 3, false),
  ]);
  const first = byUtterance(root, "u1");
  const second = byUtterance(root, "u2");
  assert.ok(first && second, "criterion 19: one [data-utterance-id] element per utterance, as today");

  assert.deepEqual(unplayedIn(first), [], "complete === true -> no [data-unplayed] element at all");
  assert.equal(squeeze(first.textContent), U1);

  const tails = unplayedIn(second);
  assert.equal(tails.length, 1, "exactly one [data-unplayed] element holds the tail");
  assert.equal(
    squeeze(tails[0].textContent),
    "on line four.",
    "criterion 19: the words from index played_words (3) onward, and only those",
  );
  assert.equal(
    squeeze(second.textContent),
    U2,
    "criterion 19: the full text is always present in the DOM — dimmed and struck, never removed",
  );

  const zero = await renderWith([entry("u2", "", 0, false)]);
  const nothingHeard = byUtterance(zero, "u2");
  const whole = unplayedIn(nothingHeard);
  assert.equal(whole.length, 1);
  assert.equal(squeeze(whole[0].textContent), U2, "played_words === 0 -> the whole utterance");

  // Ruling A8: `complete` wins over `played_words`. A drained run reports the whole utterance as
  // heard, and the word count it carries is the *source's*, which need not equal the count of
  // the displayed (TTS-normalised) text -- landmine 1. A renderer that cuts on played_words
  // whenever it is present strikes through the tail of a sentence the user heard in full.
  const drained = await renderWith([entry("u2", U2, 3, true)]);
  assert.deepEqual(
    unplayedIn(byUtterance(drained, "u2")),
    [],
    "criterion 19: complete === true -> no [data-unplayed] element at all, whatever played_words " +
      "says",
  );
  assert.equal(squeeze(byUtterance(drained, "u2").textContent), U2);

  const plain = await renderWith(null);
  assert.deepEqual(
    unplayedIn(byUtterance(plain, "u2")),
    [],
    "no played state -> today's rendering, unchanged",
  );
  assert.equal(squeeze(byUtterance(plain, "u2").textContent), U2);
});

// --- criterion 22, adversarial -------------------------------------------------------------------

test("T-0030 criterion 22: with no played message nothing is dimmed however much time passes", opts, async () => {
  const { renderSpoken } = await load("panels/spoken.js");
  const { state, reduce } = await stateWithTwoUtterances();
  let next = state;
  // 60 s of page clock, a THINKING meter and an INTERRUPTED turn: none of it is a measurement of
  // what the device rendered, and none of it may dim a word (docs/02 §4 — the HUD shows what the
  // server measured, never a trajectory it invented; state.js's old comment: "must not be faked
  // from timing").
  for (let t = 0; t <= 60_000; t += 5_000) {
    next = reduce(next, { type: "astra.turn.state", state: "SPEAKING", turn_id: "t1" }, t);
    next = reduce(next, { type: "astra.turn.state", state: "INTERRUPTED", turn_id: "t1" }, t + 1);
  }
  assert.ok(next.interrupted.includes("t1"), "premise: the turn really was marked interrupted");
  const doc = fakeDoc();
  const root = element(doc, "div");
  renderSpoken(root, next.spoken, next.interrupted);
  for (const id of ["u1", "u2"]) {
    assert.deepEqual(
      unplayedIn(byUtterance(root, id)),
      [],
      "criterion 22: no astra.spoken.played on the socket means nothing is struck through, no " +
        "matter how much time elapsed or how many INTERRUPTED messages arrived",
    );
  }
  assert.equal(squeeze(byUtterance(root, "u2").textContent), U2);
});
