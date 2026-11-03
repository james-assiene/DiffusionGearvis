// T-0225: hud/src/settings/prompts.js, the prompt library behind "Your instructions to Gearvis".
//
// Run: node --test tests/hud/settings_prompts.test.mjs   (also part of `node --test tests/hud`, which
// tests/unit/test_hud_js.py runs in the fast suite).
//
// docs/tasks/T-0225.md fixes: four built-ins in this order with these exact texts (Default is empty);
// each under the 1000-code-point limit and free of chat-template syntax (prompt_builder.validate_instructions);
// saved prompts live in localStorage under one key, "gearvis.prompts.v1", as a JSON list of {name, text};
// names are 1-60 characters, trimmed; a saved name replaces an existing user prompt of that name (the panel asks
// first); built-ins cannot be overwritten or deleted; at most 50 user prompts; texts over the limit cannot be
// saved; a blocked or corrupt store leaves the built-ins working and reports an error (the page's note).
//
// The task names the file but not its exports. The surface these rows ASSUME (relayed by the orchestrator; the
// implementer writes exactly these names, the report lists them):
//
//   BUILTIN_PROMPTS: Array<{id, name, text, builtin: true}>   the four, in the task's order
//   STORAGE_KEY = "gearvis.prompts.v1"
//   MAX_USER_PROMPTS = 50
//   loadPrompts(storage) -> {prompts, error}
//       prompts: the four built-ins, then the user's prompts sorted by name; error: null when the store read
//       cleanly (or holds nothing), truthy when it is blocked or corrupt (the built-ins are still returned).
//   savePrompt(storage, name, text) -> {ok, reason}
//       stores {name: name.trim(), text}; an existing user name is replaced (the in-page confirmation is the
//       panel's, not this function's); refuses (ok false, a reason) a bad name, a built-in name, a text over
//       1000 code points, a 51st prompt, or a store that throws. Never throws.
//   deletePrompt(storage, name) -> {ok, reason}
//       removes a user prompt; refuses a built-in or an unknown name. Never throws.
//   validName(name) -> reason | null
//
// Criterion 7 (Export / Import, added to the task at dc8fbf9a), the surface relayed by the orchestrator:
//
//   exportPrompts(prompts) -> string
//       the JSON text of {"version": 1, "prompts": [{name, text}]}, user prompts only (the built-ins in the
//       list it is given are left out).
//   parseImport(text) -> {entries, error}
//       error (truthy, a reason) and no entries for: invalid JSON, a version other than 1, no prompts list, a
//       text over 256 KB (read here as 256 * 1024 bytes of UTF-8); otherwise the valid entries (name rules, text
//       at most 1000 code points, no special-token strings); invalid entries never reach the store.
//   mergeImport(storage, entries, {replace}) -> {added, replaced, skipped, error}
//       adds new names; an existing user name is replaced only with replace: true (the page asks once for the
//       whole import, then calls with replace: true), left as it was with replace: false; built-in names are
//       skipped; the 50 cap applies (entries beyond it are skipped); a blocked store gives an error and never
//       throws.
//
// `storage` is the Web Storage interface (getItem/setItem/removeItem); the rows pass a Map-backed fake.
// Every expected value is written by hand from the task text, never computed by a copy of the module.
//
// Pending rule (the mechanism of tests/hud/settings_state_wire.test.mjs): every row that needs prompts.js is a
// strict xfail while tests/integration/test_hud_settings_prompts.py (T-0225's integration module) is absent or
// still carries the T-0225 pending marker string, so removing those markers activates these rows too.
// ASTRA_RUN_PENDING=1 runs them for real now.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);
const PROMPTS_FILE = fileURLToPath(new URL("hud/src/settings/prompts.js", REPO));
const T0225_MARK = 'pending(task="T-0225")';

function pendingWhile(relPath, absentIsPending) {
  if (process.env.ASTRA_RUN_PENDING === "1") return false;
  const file = fileURLToPath(new URL(relPath, REPO));
  if (!existsSync(file)) return absentIsPending;
  return readFileSync(file, "utf8").includes(T0225_MARK);
}

const PENDING = pendingWhile("tests/integration/test_hud_settings_prompts.py", true);

function row(name, fn) {
  test(name + (PENDING ? " [pending T-0225: strict xfail]" : ""), async (t) => {
    if (!PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null)
      assert.fail("XPASS(strict): this T-0225 row passed while it is pending; remove the marker or fix the row");
    t.diagnostic("expected failure (T-0225 pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

// The same ES-module shim as tests/hud/settings_model.test.mjs.
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

// Origin proof (.claude/rules/tests.md): where the module under test is imported from.
console.log(`# settings prompts origin: ${PROMPTS_FILE} exists=${existsSync(PROMPTS_FILE)}`);

async function load() {
  let mod;
  try {
    mod = await import(new URL("settings/prompts.js", srcRoot()).href);
  } catch (err) {
    assert.fail(
      "T-0225: hud/src/settings/prompts.js must import in node as an ES module with no DOM at module top " +
        `level; node could not import it: ${String(err && err.message).split("\n")[0]}`,
    );
  }
  for (const name of [
    "BUILTIN_PROMPTS",
    "STORAGE_KEY",
    "MAX_USER_PROMPTS",
    "loadPrompts",
    "savePrompt",
    "deletePrompt",
    "validName",
  ]) {
    assert.ok(name in mod, `T-0225: prompts.js exports ${name} (the surface this file names)`);
  }
  return mod;
}

// --- the task's text, verbatim -------------------------------------------------------------------------------

const KEY = "gearvis.prompts.v1";
// T-0229 A1: the tutor checks the work rather than presupposing a mistake (T-0225's text is historical)
const TUTOR =
  "You are a patient tutor. When I show you my work, check it line by line. If a line is wrong, point to it, " +
  "explain why, and ask me a guiding question before giving the fix; then give the corrected step and the " +
  "correct final answer. If everything is right, say so plainly and don't invent a problem. Speak in plain " +
  "words, not symbols, and keep each reply short.";
const GAMING =
  "You are my gaming companion. I share my game screen with you. Answer out loud in one or two short " +
  "sentences, like a friend sitting next to me. Read what is on the screen before you answer. When I ask what " +
  "to do, give one clear plan and the reason. Do not use your own browser. Use google_search only for game " +
  "facts you are not sure about, and say where you found them.";
// T-0230 A2: the Browser specialist goes as far as checkout and stops at sign-in, payment or the final order button
// (T-0225's text, "Always ask me before buying, sending, posting or deleting anything.", is historical)
const BROWSER =
  "When I ask you to do something on a website, do it yourself in your own browser, one step at a time, and " +
  "check the screen after each step. Do not narrate every click. When you are done, tell me in one sentence " +
  "what you did and what you see. If you are stuck, say what is blocking you. When I ask you to buy or order " +
  "something, do every step you can: find it, add it to the cart and go to checkout. Stop at sign-in, payment " +
  "or the final order button and tell me what is left for me to do. Never type passwords or card numbers, and " +
  "ask me before sending, posting or deleting anything.";
const BUILTINS = [
  ["Default", ""],
  ["Education tutor", TUTOR],
  ["Gaming companion", GAMING],
  ["Browser specialist", BROWSER],
];
const BUILTIN_NAMES = BUILTINS.map(([n]) => n);

// A mirror of prompt_builder.INSTRUCTIONS_FORBIDDEN at 377bc592: the delimiter fragments plus
// toolcall.special_tokens.SPECIAL_TOKENS (printed from the Python module, copied by hand).
const FORBIDDEN = [
  "<|", "|>", "<turn", "turn|>",
  "<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<|tool>", "<tool|>", "<|tool_call>", "<tool_call|>",
  "<|tool_response>", "<tool_response|>", '<|"|>', "<|think|>", "<|channel>", "<channel|>", "<|turn>",
  "<turn|>", "<|image>", "<|audio>", "<|image|>", "<|audio|>", "<image|>", "<audio|>", "<|video|>",
];

const ASTRAL = "\u{1F600}";
const codePoints = (s) => [...s].length;

/** A Map-backed Web Storage fake (getItem returns null for a missing key, as the real one does). */
function fakeStorage(initial = {}) {
  const map = new Map(Object.entries(initial));
  return {
    map,
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => void map.set(k, String(v)),
    removeItem: (k) => void map.delete(k),
    clear: () => map.clear(),
    key: (i) => [...map.keys()][i] ?? null,
    get length() {
      return map.size;
    },
  };
}

/** A store whose every access throws (Safari private mode, a blocked third-party store, a disabled profile). */
function throwingStorage() {
  const boom = () => {
    throw new Error("SecurityError: the operation is insecure");
  };
  return { getItem: boom, setItem: boom, removeItem: boom, clear: boom, key: boom, get length() { return boom(); } };
}

/** Reads fine, but every write throws (a full quota). */
function fullStorage(initial = {}) {
  const s = fakeStorage(initial);
  s.setItem = () => {
    throw new Error("QuotaExceededError");
  };
  return s;
}

const stored = (s) => JSON.parse(s.map.get(KEY));
const names = (prompts) => prompts.map((p) => p.name);
const userOnly = (prompts) => prompts.filter((p) => !p.builtin).map((p) => ({ name: p.name, text: p.text }));

// --- built-ins -------------------------------------------------------------------------------------------------

row("the four built-ins are Default, Education tutor, Gaming companion, Browser specialist, verbatim, in order", async () => {
  const m = await load();
  assert.deepEqual(
    m.BUILTIN_PROMPTS.map((p) => [p.name, p.text]),
    BUILTINS,
    "the task's four built-ins, their exact texts, their order",
  );
  for (const p of m.BUILTIN_PROMPTS) {
    assert.equal(p.builtin, true, `${p.name} is marked built-in`);
    assert.ok(typeof p.id === "string" && p.id.length > 0, `${p.name} has an id`);
  }
  assert.equal(new Set(m.BUILTIN_PROMPTS.map((p) => p.id)).size, 4, "the ids are distinct");
});

row("T-0229 A1: the education_tutor built-in checks the work and does not presuppose a mistake", async () => {
  const m = await load();
  const tutor = m.BUILTIN_PROMPTS.find((p) => p.id === "education_tutor");
  assert.ok(tutor, "an education_tutor built-in");
  assert.equal(tutor.name, "Education tutor");
  // the failure mode T-0226 B measured (5 of 15 correct pages given an invented mistake): T-0225's text
  assert.ok(!tutor.text.includes("find the line with the mistake"), "T-0225's presupposing text is gone");
  assert.ok(tutor.text.includes("check it line by line"), "the tutor checks the work line by line");
  assert.ok(
    tutor.text.includes("If everything is right, say so plainly and don't invent a problem."),
    "the tutor is told a correct page is a possible outcome",
  );
  assert.equal(tutor.text, TUTOR, "verbatim (T-0229 A1)");
});

row("T-0230 A2: the browser_specialist built-in goes to checkout and stops at sign-in, payment or the order button", async () => {
  const m = await load();
  const b = m.BUILTIN_PROMPTS.find((p) => p.id === "browser_specialist");
  assert.ok(b, "a browser_specialist built-in");
  assert.equal(b.name, "Browser specialist");
  // the failure mode James named (2026-11-02): a blanket "ask me before buying" stopped it before the cart
  assert.ok(!b.text.includes("Always ask me before buying"), "T-0225's ask-before-buying sentence is gone");
  assert.ok(b.text.includes("add it to the cart and go to checkout"), "it is told to go as far as checkout");
  assert.ok(
    b.text.includes("Stop at sign-in, payment or the final order button"),
    "it is told where to stop",
  );
  assert.ok(b.text.includes("Never type passwords or card numbers"), "the secret-typing rule is stated");
  assert.equal(b.text, BROWSER, "verbatim (T-0230 A2)");
  assert.ok(codePoints(b.text) <= 1000, `${codePoints(b.text)} code points`);
});

row("T-0230 A2: the other three built-ins are unchanged", async () => {
  const m = await load();
  const byId = Object.fromEntries(m.BUILTIN_PROMPTS.map((p) => [p.id, p]));
  assert.equal(byId.default?.text, "", "Default is empty");
  assert.equal(byId.education_tutor?.text, TUTOR, "Education tutor is T-0229's text");
  assert.equal(byId.gaming_companion?.text, GAMING, "Gaming companion is T-0225's text");
});

row("every built-in is at most 1000 code points and free of chat-template syntax (validate_instructions' specials)", async () => {
  const m = await load();
  for (const p of m.BUILTIN_PROMPTS) {
    assert.ok(codePoints(p.text.trim()) <= 1000, `${p.name}: ${codePoints(p.text)} code points`);
    const squeezed = p.text.replace(/\s+/g, "");
    for (const lit of FORBIDDEN) {
      assert.ok(!p.text.includes(lit) && !squeezed.includes(lit), `${p.name} holds ${JSON.stringify(lit)}`);
    }
    // R-2: no control or format character other than newline and tab
    assert.ok(!/[\p{Cc}\p{Cf}\p{Cs}]/u.test(p.text.replace(/[\n\t]/g, "")), `${p.name} holds a control character`);
  }
});

row("the storage key and the cap are the task's", async () => {
  const m = await load();
  assert.equal(m.STORAGE_KEY, KEY);
  assert.equal(m.MAX_USER_PROMPTS, 50);
});

// --- load ------------------------------------------------------------------------------------------------------

row("an empty store loads the four built-ins and no error", async () => {
  const m = await load();
  const s = fakeStorage();
  const got = m.loadPrompts(s);
  assert.deepEqual(names(got.prompts), BUILTIN_NAMES);
  assert.ok(!got.error, `nothing stored is not an error: ${JSON.stringify(got.error)}`);
  assert.equal(s.map.has(KEY), false, "loading writes nothing");
});

row("a stored list loads after the built-ins, sorted by name, each marked not built-in", async () => {
  const m = await load();
  const s = fakeStorage({
    [KEY]: JSON.stringify([
      { name: "Gamma", text: "g" },
      { name: "Alpha", text: "a" },
      { name: "Beta", text: "b" },
    ]),
  });
  const got = m.loadPrompts(s);
  assert.ok(!got.error, JSON.stringify(got.error));
  assert.deepEqual(names(got.prompts), [...BUILTIN_NAMES, "Alpha", "Beta", "Gamma"]);
  assert.deepEqual(userOnly(got.prompts), [
    { name: "Alpha", text: "a" },
    { name: "Beta", text: "b" },
    { name: "Gamma", text: "g" },
  ]);
  for (const p of got.prompts.slice(4)) assert.ok(!p.builtin, `${p.name} is the user's`);
});

for (const [label, raw] of [
  ["corrupt JSON", "{nope"],
  ["JSON that is not a list", '{"name": "x", "text": "y"}'],
  ["a JSON string", '"hello"'],
]) {
  row(`a store holding ${label} still loads the four built-ins, with an error for the note`, async () => {
    const m = await load();
    const s = fakeStorage({ [KEY]: raw });
    let got;
    assert.doesNotThrow(() => {
      got = m.loadPrompts(s);
    }, "a corrupt store never throws");
    assert.deepEqual(names(got.prompts).slice(0, 4), BUILTIN_NAMES, "the built-ins keep working");
    assert.deepEqual(got.prompts[1].text, TUTOR);
    assert.ok(got.error, `a corrupt store is reported (the note): ${JSON.stringify(got)}`);
  });
}

row("a list with malformed entries never throws and keeps the built-ins first and whole", async () => {
  const m = await load();
  const s = fakeStorage({ [KEY]: JSON.stringify([null, 7, { name: 3 }, { text: "x" }, { name: "Ok", text: "fine" }]) });
  let got;
  assert.doesNotThrow(() => {
    got = m.loadPrompts(s);
  });
  assert.deepEqual(got.prompts.slice(0, 4).map((p) => [p.name, p.text]), BUILTINS);
  for (const p of got.prompts) {
    assert.equal(typeof p.name, "string", `every listed prompt has a string name: ${JSON.stringify(p)}`);
    assert.equal(typeof p.text, "string", `every listed prompt has a string text: ${JSON.stringify(p)}`);
  }
});

row("a store that throws on every access loads the four built-ins, with an error, and never throws", async () => {
  const m = await load();
  let got;
  assert.doesNotThrow(() => {
    got = m.loadPrompts(throwingStorage());
  });
  assert.deepEqual(got.prompts.map((p) => [p.name, p.text]), BUILTINS);
  assert.ok(got.error, "a blocked store is reported (the note)");
});

row("no storage at all (null: window.localStorage itself threw) loads the built-ins with an error", async () => {
  const m = await load();
  let got;
  assert.doesNotThrow(() => {
    got = m.loadPrompts(null);
  });
  assert.deepEqual(names(got.prompts), BUILTIN_NAMES);
  assert.ok(got.error);
});

// --- save / replace --------------------------------------------------------------------------------------------

row("save stores one {name, text} under the key and the next load lists it", async () => {
  const m = await load();
  const s = fakeStorage();
  const res = m.savePrompt(s, "Mine", "be brief");
  assert.equal(res.ok, true, JSON.stringify(res));
  assert.deepEqual(stored(s), [{ name: "Mine", text: "be brief" }], "a JSON list of {name, text}, user prompts only");
  assert.equal(s.map.size, 1, "one key, nothing else written");
  const got = m.loadPrompts(s);
  assert.deepEqual(names(got.prompts), [...BUILTIN_NAMES, "Mine"]);
  assert.equal(got.prompts[4].text, "be brief");
});

row("the name is stored trimmed, and a trimmed duplicate replaces rather than adds", async () => {
  const m = await load();
  const s = fakeStorage();
  assert.equal(m.savePrompt(s, "  Mine  ", "one").ok, true);
  assert.deepEqual(stored(s), [{ name: "Mine", text: "one" }]);
  assert.equal(m.savePrompt(s, "Mine", "two").ok, true, "saving under an existing user name replaces it");
  assert.deepEqual(stored(s), [{ name: "Mine", text: "two" }], "replaced, not duplicated");
});

row("replacing one prompt leaves the others untouched", async () => {
  const m = await load();
  const s = fakeStorage();
  m.savePrompt(s, "Alpha", "a");
  m.savePrompt(s, "Beta", "b");
  m.savePrompt(s, "Alpha", "A2");
  assert.deepEqual(userOnly(m.loadPrompts(s).prompts), [
    { name: "Alpha", text: "A2" },
    { name: "Beta", text: "b" },
  ]);
});

for (const builtin of BUILTIN_NAMES) {
  row(`a built-in name (${builtin}) cannot be overwritten: refused with a reason, the store unchanged`, async () => {
    const m = await load();
    const s = fakeStorage({ [KEY]: JSON.stringify([{ name: "Mine", text: "x" }]) });
    const before = s.map.get(KEY);
    const res = m.savePrompt(s, builtin, "overwritten");
    assert.equal(res.ok, false, JSON.stringify(res));
    assert.ok(typeof res.reason === "string" && res.reason.length > 0, "a reason to show");
    assert.equal(s.map.get(KEY), before, "nothing written");
    const got = m.loadPrompts(s).prompts;
    assert.deepEqual(got.slice(0, 4).map((p) => [p.name, p.text]), BUILTINS, "the built-in still has its own text");
    assert.equal(got.filter((p) => p.name === builtin).length, 1, "listed once");
  });
}

row("saving never mutates BUILTIN_PROMPTS", async () => {
  const m = await load();
  const s = fakeStorage();
  m.savePrompt(s, "Mine", "x");
  m.savePrompt(s, "Default", "x");
  m.deletePrompt(s, "Education tutor");
  assert.deepEqual(m.BUILTIN_PROMPTS.map((p) => [p.name, p.text]), BUILTINS);
});

row("names and texts are stored and loaded verbatim, markup included (never escaped, never parsed)", async () => {
  const m = await load();
  const s = fakeStorage();
  const name = '<img src=x onerror="alert(1)">';
  const text = "<b>bold</b> &amp; <script>x()</script>";
  assert.equal(m.savePrompt(s, name, text).ok, true);
  assert.deepEqual(stored(s), [{ name, text }]);
  assert.deepEqual(userOnly(m.loadPrompts(s).prompts), [{ name, text }]);
});

// --- name rules ------------------------------------------------------------------------------------------------

row("validName: 1 to 60 characters once trimmed", async () => {
  const m = await load();
  for (const bad of ["", " ", "   \t ", "a".repeat(61), " " + "a".repeat(61) + " "]) {
    const reason = m.validName(bad);
    assert.ok(typeof reason === "string" && reason.length > 0, `refused: ${JSON.stringify(bad)} (${reason})`);
  }
  for (const good of ["a", "Mine", "a".repeat(60), "  " + "a".repeat(60) + "  ", "  x  "]) {
    assert.equal(m.validName(good), null, `accepted: ${JSON.stringify(good)}`);
  }
});

row("validName counts characters, not UTF-16 units: 60 astral characters are legal, 61 are not", async () => {
  const m = await load();
  assert.equal(ASTRAL.repeat(60).length, 120, "premise: 120 UTF-16 units");
  assert.equal(m.validName(ASTRAL.repeat(60)), null);
  assert.ok(m.validName(ASTRAL.repeat(61)));
});

row("save refuses a bad name and writes nothing", async () => {
  const m = await load();
  for (const bad of ["", "   ", "a".repeat(61)]) {
    const s = fakeStorage();
    const res = m.savePrompt(s, bad, "text");
    assert.equal(res.ok, false, JSON.stringify(bad));
    assert.ok(res.reason, "with a reason");
    assert.equal(s.map.has(KEY), false, "nothing written");
  }
  const s = fakeStorage();
  assert.equal(m.savePrompt(s, "a".repeat(60), "text").ok, true, "60 characters is legal (not too strict)");
});

// --- the text limit --------------------------------------------------------------------------------------------

row("a text of 1000 code points saves; 1001 does not (counted in code points, astral included)", async () => {
  const m = await load();
  const s = fakeStorage();
  assert.equal(m.savePrompt(s, "AtCap", "a".repeat(1000)).ok, true, "1000 is at the limit");
  assert.equal(m.savePrompt(s, "AtCapAstral", ASTRAL.repeat(1000)).ok, true, "1000 astral (2000 units) is legal");
  const over = m.savePrompt(s, "Over", "a".repeat(999) + ASTRAL + ASTRAL);
  assert.equal(over.ok, false, "1001 code points cannot be saved");
  assert.ok(over.reason);
  assert.deepEqual(names(userOnly(m.loadPrompts(s).prompts)), ["AtCap", "AtCapAstral"]);
});

// --- the 50 cap ------------------------------------------------------------------------------------------------

row("50 user prompts save; the 51st is refused with a reason; replacing one at the cap still works", async () => {
  const m = await load();
  const s = fakeStorage();
  const fifty = Array.from({ length: 50 }, (_, i) => `P${String(i).padStart(2, "0")}`);
  for (const n of fifty) assert.equal(m.savePrompt(s, n, `t ${n}`).ok, true, `${n} saves (under the cap)`);
  assert.equal(stored(s).length, 50);
  const res = m.savePrompt(s, "P50", "one too many");
  assert.equal(res.ok, false, "the 51st is refused");
  assert.ok(res.reason);
  assert.equal(stored(s).length, 50);
  assert.equal(m.savePrompt(s, "P07", "replaced").ok, true, "a replacement does not grow the list");
  const got = userOnly(m.loadPrompts(s).prompts);
  assert.equal(got.length, 50);
  assert.deepEqual(got.find((p) => p.name === "P07"), { name: "P07", text: "replaced" });
  assert.equal(m.deletePrompt(s, "P00").ok, true);
  assert.equal(m.savePrompt(s, "P50", "now it fits").ok, true, "after a delete there is room again");
});

// --- delete ----------------------------------------------------------------------------------------------------

row("delete removes the named user prompt only", async () => {
  const m = await load();
  const s = fakeStorage();
  m.savePrompt(s, "Alpha", "a");
  m.savePrompt(s, "Mine", "m");
  const res = m.deletePrompt(s, "Mine");
  assert.equal(res.ok, true, JSON.stringify(res));
  assert.deepEqual(stored(s), [{ name: "Alpha", text: "a" }]);
  assert.deepEqual(names(m.loadPrompts(s).prompts), [...BUILTIN_NAMES, "Alpha"]);
});

for (const builtin of BUILTIN_NAMES) {
  row(`a built-in (${builtin}) cannot be deleted: refused with a reason, still listed`, async () => {
    const m = await load();
    const s = fakeStorage({ [KEY]: JSON.stringify([{ name: "Mine", text: "x" }]) });
    const before = s.map.get(KEY);
    const res = m.deletePrompt(s, builtin);
    assert.equal(res.ok, false);
    assert.ok(res.reason);
    assert.equal(s.map.get(KEY), before);
    assert.deepEqual(names(m.loadPrompts(s).prompts), [...BUILTIN_NAMES, "Mine"]);
  });
}

row("deleting a name that is not saved is refused and writes nothing", async () => {
  const m = await load();
  const s = fakeStorage({ [KEY]: JSON.stringify([{ name: "Mine", text: "x" }]) });
  const before = s.map.get(KEY);
  assert.equal(m.deletePrompt(s, "Nobody").ok, false);
  assert.equal(s.map.get(KEY), before);
});

// --- a blocked store on write ----------------------------------------------------------------------------------

row("save and delete over a throwing or full store return ok false and never throw", async () => {
  const m = await load();
  for (const s of [throwingStorage(), fullStorage({ [KEY]: JSON.stringify([{ name: "Mine", text: "x" }]) }), null]) {
    let save;
    let del;
    assert.doesNotThrow(() => {
      save = m.savePrompt(s, "New", "text");
    });
    assert.doesNotThrow(() => {
      del = m.deletePrompt(s, "Mine");
    });
    assert.equal(save.ok, false, "a write that throws is not a save");
    assert.equal(del.ok, false, "a write that throws is not a delete");
  }
});

row("save over a corrupt store does not throw; built-ins stay listed", async () => {
  const m = await load();
  const s = fakeStorage({ [KEY]: "{nope" });
  let res;
  assert.doesNotThrow(() => {
    res = m.savePrompt(s, "Mine", "x");
  });
  assert.equal(typeof res.ok, "boolean");
  const got = m.loadPrompts(s);
  assert.deepEqual(got.prompts.slice(0, 4).map((p) => [p.name, p.text]), BUILTINS);
});

// --- criterion 7: Export / Import -------------------------------------------------------------------------------

async function loadIO() {
  const m = await load();
  for (const name of ["exportPrompts", "parseImport", "mergeImport"]) {
    assert.ok(name in m, `T-0225 criterion 7: prompts.js exports ${name} (the surface this file names)`);
  }
  return m;
}

const fileOf = (prompts, version = 1) => JSON.stringify({ version, prompts });
const byName = (list) => [...list].sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));

row("export is {version: 1, prompts: [{name, text}]} of the user's prompts only", async () => {
  const m = await loadIO();
  const s = fakeStorage();
  m.savePrompt(s, "Mine", "be brief");
  m.savePrompt(s, "Alpha", "<b>a</b>");
  const out = m.exportPrompts(m.loadPrompts(s).prompts);
  assert.equal(typeof out, "string");
  const parsed = JSON.parse(out);
  assert.deepEqual(Object.keys(parsed).sort(), ["prompts", "version"], "exactly the two fields");
  assert.equal(parsed.version, 1);
  assert.deepEqual(
    byName(parsed.prompts),
    [
      { name: "Alpha", text: "<b>a</b>" },
      { name: "Mine", text: "be brief" },
    ],
    "user prompts only, each exactly {name, text}",
  );
  assert.deepEqual(JSON.parse(m.exportPrompts(m.loadPrompts(fakeStorage()).prompts)), { version: 1, prompts: [] });
});

row("export, parse and merge round-trip into a fresh store", async () => {
  const m = await loadIO();
  const a = fakeStorage();
  const want = [
    { name: "Alpha", text: "a" },
    { name: "Mine", text: `astral ${ASTRAL} and "quotes" and \\ and\nlines` },
    { name: '<img src=x onerror="1">', text: "<script>x()</script>" },
  ];
  for (const p of want) assert.equal(m.savePrompt(a, p.name, p.text).ok, true, p.name);
  const parsed = m.parseImport(m.exportPrompts(m.loadPrompts(a).prompts));
  assert.ok(!parsed.error, JSON.stringify(parsed.error));
  assert.deepEqual(byName(parsed.entries), byName(want));
  const b = fakeStorage();
  const res = m.mergeImport(b, parsed.entries, { replace: false });
  assert.ok(!res.error, JSON.stringify(res));
  assert.equal(res.added, 3);
  assert.equal(res.replaced, 0);
  assert.equal(res.skipped, 0);
  assert.deepEqual(byName(stored(b)), byName(want));
});

for (const [label, text] of [
  ["invalid JSON", "{nope"],
  ["version 2", fileOf([{ name: "Mine", text: "x" }], 2)],
  ["no version", JSON.stringify({ prompts: [{ name: "Mine", text: "x" }] })],
  ["prompts not a list", JSON.stringify({ version: 1, prompts: { name: "Mine", text: "x" } })],
  ["a bare list", JSON.stringify([{ name: "Mine", text: "x" }])],
  ["over 256 KB", fileOf([{ name: "Mine", text: "x" }, { name: "Pad", text: "p".repeat(300 * 1024) }])],
]) {
  row(`an import of ${label} gives a reason and no entries`, async () => {
    const m = await loadIO();
    let got;
    assert.doesNotThrow(() => {
      got = m.parseImport(text);
    });
    assert.ok(
      typeof got.error === "string" && got.error.length > 0,
      `a reason to show: ${JSON.stringify(got).slice(0, 200)}`,
    );
    assert.deepEqual(got.entries ?? [], [], "nothing to merge, so nothing changes");
  });
}

row("an import just under 256 KB is accepted (the size guard is not too wide)", async () => {
  const m = await loadIO();
  // 200 entries of 1000 code points: about 205 KB, every entry legal
  const prompts = Array.from({ length: 200 }, (_, i) => ({ name: `N${i}`, text: "t".repeat(1000) }));
  const text = fileOf(prompts);
  assert.ok(Buffer.byteLength(text, "utf8") < 256 * 1024, "premise: under the limit");
  const got = m.parseImport(text);
  assert.ok(!got.error, JSON.stringify(got.error));
  assert.equal(got.entries.length, 200);
});

row("invalid entries never reach the store: bad names, texts over 1000 code points, special-token strings", async () => {
  const m = await loadIO();
  const text = fileOf([
    { name: "Good", text: "fine" },
    { name: "", text: "empty name" },
    { name: "a".repeat(61), text: "long name" },
    { name: "TooLong", text: "a".repeat(999) + ASTRAL + ASTRAL },
    { name: "Think", text: "be brief <|think|> always" },
    { name: "Turn", text: "x <turn|> y" },
    { name: 7, text: "number name" },
    { name: "NoText" },
  ]);
  const parsed = m.parseImport(text);
  const s = fakeStorage();
  if (!parsed.error) m.mergeImport(s, parsed.entries, { replace: true });
  const got = s.map.has(KEY) ? stored(s).map((p) => p.name) : [];
  assert.deepEqual(got, ["Good"], "only the one valid entry is merged (an invalid entry does not sink the file)");
});

row("an existing name is replaced only with replace: true; new names are added either way", async () => {
  const m = await loadIO();
  const entries = [
    { name: "Mine", text: "imported" },
    { name: "New", text: "new" },
  ];
  const keep = fakeStorage();
  m.savePrompt(keep, "Mine", "original");
  const r1 = m.mergeImport(keep, entries, { replace: false });
  assert.ok(!r1.error, JSON.stringify(r1));
  assert.equal(r1.added, 1);
  assert.equal(r1.replaced, 0);
  assert.deepEqual(
    byName(stored(keep)),
    [
      { name: "Mine", text: "original" },
      { name: "New", text: "new" },
    ],
    "without the confirmation the existing text stays",
  );

  const swap = fakeStorage();
  m.savePrompt(swap, "Mine", "original");
  const r2 = m.mergeImport(swap, entries, { replace: true });
  assert.ok(!r2.error, JSON.stringify(r2));
  assert.equal(r2.added, 1);
  assert.equal(r2.replaced, 1);
  assert.deepEqual(
    byName(stored(swap)),
    [
      { name: "Mine", text: "imported" },
      { name: "New", text: "new" },
    ],
    "after the confirmation it is replaced, not duplicated",
  );
});

row("built-in names are skipped on import and keep their own texts", async () => {
  const m = await loadIO();
  const s = fakeStorage();
  const res = m.mergeImport(
    s,
    BUILTIN_NAMES.map((n) => ({ name: n, text: "hijacked" })).concat([{ name: "Mine", text: "m" }]),
    { replace: true },
  );
  assert.equal(res.added, 1);
  assert.equal(res.replaced, 0);
  assert.equal(res.skipped, 4, "the four built-in names are skipped");
  assert.deepEqual(stored(s), [{ name: "Mine", text: "m" }]);
  assert.deepEqual(m.loadPrompts(s).prompts.slice(0, 4).map((p) => [p.name, p.text]), BUILTINS);
});

row("the 50 cap applies on import: 48 stored + 5 new adds 2 and skips 3; a replacement still fits", async () => {
  const m = await loadIO();
  const s = fakeStorage();
  for (let i = 0; i < 48; i++) assert.equal(m.savePrompt(s, `P${String(i).padStart(2, "0")}`, "t").ok, true);
  const res = m.mergeImport(
    s,
    [{ name: "P00", text: "replaced" }, ...["Q1", "Q2", "Q3", "Q4", "Q5"].map((n) => ({ name: n, text: n }))],
    { replace: true },
  );
  assert.equal(stored(s).length, 50, "never more than 50");
  assert.equal(res.added, 2);
  assert.equal(res.replaced, 1, "a replacement does not count against the cap");
  assert.equal(res.skipped, 3);
  assert.deepEqual(stored(s).find((p) => p.name === "P00"), { name: "P00", text: "replaced" });
});

row("merge over a blocked or full store reports an error and never throws", async () => {
  const m = await loadIO();
  for (const s of [throwingStorage(), fullStorage(), null]) {
    let res;
    assert.doesNotThrow(() => {
      res = m.mergeImport(s, [{ name: "Mine", text: "x" }], { replace: true });
    });
    assert.ok(res.error, `a store that throws is reported: ${JSON.stringify(res)}`);
  }
});

// --- review rows (T-0225 review, 2026-11-01): mutants that survived the rows above -----------------------------
//
// Expected values come from the task text and from prompt_builder.validate_instructions (A1-4, the server's own
// judge, which the task names as the rule for imported texts): every case below was checked against it by hand
// (`uv run python -c "...validate_instructions(...)"`): "x <pa d> y" -> refused ('<pad>'), "a\u0007b" and
// "a‎b" -> refused (control / format character), "a\tb\nc" -> accepted.

/** The names a one-entry file holding `text` leaves in a fresh store after parseImport + mergeImport (either
 *  step may drop an invalid entry; the criterion is that it is never merged). */
function keptOf(m, text) {
  const got = m.parseImport(fileOf([{ name: "Probe", text }]));
  if (got.error) return [];
  const s = fakeStorage();
  m.mergeImport(s, got.entries, { replace: true });
  return s.map.has(KEY) ? stored(s).map((p) => p.name) : [];
}

row("import drops an entry holding any one of validate_instructions' forbidden strings (each, alone)", async () => {
  const m = await loadIO();
  for (const lit of FORBIDDEN) {
    assert.deepEqual(keptOf(m, `be brief ${lit} always`), [], `${JSON.stringify(lit)} is refused`);
  }
  assert.deepEqual(keptOf(m, "be brief, pad it out, use a mask, think it over; 3 < 4 and 5 > 2"), ["Probe"], "plain words pass (not too wide)");
});

row("import drops a forbidden string split by whitespace ('<pa d>' squeezes to '<pad>')", async () => {
  const m = await loadIO();
  assert.deepEqual(keptOf(m, "x <pa d> y"), []);
  assert.deepEqual(keptOf(m, "x <ma\n sk> y"), []);
});

row("import drops control and format characters but keeps tabs and line breaks", async () => {
  const m = await loadIO();
  assert.deepEqual(keptOf(m, "a\u0007b"), [], "a C0 control (BEL)");
  assert.deepEqual(keptOf(m, "a‎b"), [], "a format character (LRM)");
  assert.deepEqual(keptOf(m, "a\u0085b"), [], "a C1 control (NEL)");
  assert.deepEqual(keptOf(m, "a\tb\nc"), ["Probe"], "tab and newline are plain text");
});

row("256 KB is 256 * 1024 bytes: a file of about 259 KB (over 256 000 bytes) is accepted", async () => {
  const m = await loadIO();
  const prompts = Array.from({ length: 251 }, (_, i) => ({ name: `N${i}`, text: "t".repeat(1000) }));
  const bare = fileOf(prompts);
  const pad = 260_000 - Buffer.byteLength(bare, "utf8");
  assert.ok(pad > 0, "premise: room to pad");
  const text = bare.slice(0, -1) + " ".repeat(pad) + "}"; // JSON whitespace: the same document, 260 000 bytes
  const bytes = Buffer.byteLength(text, "utf8");
  assert.ok(bytes > 256 * 1000 && bytes <= 256 * 1024, `premise: ${bytes} bytes, between the two readings`);
  const got = m.parseImport(text);
  assert.ok(!got.error, `under 256 KiB is accepted: ${JSON.stringify(got.error)}`);
  assert.equal(got.entries.length, 251);
  const over = m.parseImport(text.replace(/\}$/, " ".repeat(256 * 1024 - bytes + 1) + "}"));
  assert.ok(over.error, "one byte over 256 * 1024 is refused");
});

row("a name repeated in the import file is stored once", async () => {
  const m = await loadIO();
  const parsed = m.parseImport(
    fileOf([
      { name: "Mine", text: "first" },
      { name: "Mine", text: "second" },
      { name: " Mine ", text: "third" },
    ]),
  );
  assert.ok(!parsed.error, JSON.stringify(parsed.error));
  const s = fakeStorage();
  const res = m.mergeImport(s, parsed.entries, { replace: true });
  assert.ok(!res.error, JSON.stringify(res));
  assert.equal(stored(s).filter((p) => p.name === "Mine").length, 1, `one Mine: ${JSON.stringify(stored(s))}`);
  assert.equal(stored(s).length, 1);
  assert.equal(res.added, 1, "the repeats are not extra additions");
  assert.equal(res.replaced, 0, "a repeat inside the file is not a replacement of a saved prompt");
});

row("mergeImport is safe on its own: repeated names in the entries it is given are stored once", async () => {
  const m = await loadIO();
  const s = fakeStorage();
  m.mergeImport(
    s,
    [
      { name: "Mine", text: "first" },
      { name: "Mine", text: "second" },
    ],
    { replace: true },
  );
  assert.equal(stored(s).filter((p) => p.name === "Mine").length, 1, JSON.stringify(stored(s)));
});

row("a stored entry whose name is not trimmed or is repeated is never listed as such", async () => {
  const m = await loadIO();
  const s = fakeStorage({
    [KEY]: JSON.stringify([
      { name: " Spaced ", text: "x" },
      { name: "Twice", text: "1" },
      { name: "Twice", text: "2" },
      { name: "Ok", text: "fine" },
    ]),
  });
  const got = m.loadPrompts(s).prompts.slice(4);
  for (const p of got) assert.equal(p.name, p.name.trim(), `a listed name is trimmed: ${JSON.stringify(p.name)}`);
  assert.equal(got.filter((p) => p.name === "Twice").length, 1, `a repeated stored name is listed once: ${JSON.stringify(got)}`);
  assert.ok(got.some((p) => p.name === "Ok"), "the clean entry is listed");
});

row("astral text counts in code points on save and on import: 600 emoji (1200 UTF-16 units) is legal", async () => {
  const m = await loadIO();
  const text = ASTRAL.repeat(600);
  assert.equal(text.length, 1200, "premise: over 1000 UTF-16 units");
  const s = fakeStorage();
  assert.equal(m.savePrompt(s, "Emoji", text).ok, true, "saves");
  assert.deepEqual(keptOf(m, text), ["Probe"], "imports");
  const t = fakeStorage();
  m.mergeImport(t, m.parseImport(fileOf([{ name: "Emoji", text }])).entries, { replace: false });
  assert.deepEqual(stored(t), [{ name: "Emoji", text }]);
});
