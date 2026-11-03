// T-0213 group B-read (B-1, B-2): hud/src/settings/model.js, the Settings panel's pure model.
//
// Run: node --test tests/hud/settings_model.test.mjs   (also part of `node --test tests/hud`, which
// tests/unit/test_hud_js.py runs in the fast suite).
//
// Location: the task names hud/src/settings/model.test.mjs, but the path hook lets the test-author
// write under tests/ only, and tests/hud/ is where the node layer lives and is run from. Reported.
//
// What docs/tasks/T-0213.md fixes and these rows pin: the panel renders the groups in catalog
// order (B-1: Seeing the screen; Acting in the browser; Checking its work; Guidance and prompt;
// Search and web; Speech; Engine (relaunch only)); each item carries a kind chip reading
// "next turn", "next turn, slower first reply" or "needs relaunch" (B-1); a relaunch-only item is
// read-only and shows its relaunch command (B-2); every catalog string is text, never markup
// (B-1 "all set with textContent"; the adversarial row of "Tests to write", Node).
//
// The task names the file ("pure: grouping, chips, the relaunch text") but not its exports. The
// surface these rows ASSUME (the implementer provides exactly these names; the report lists them):
//
//   SETTINGS_GROUPS: readonly string[]
//       the seven B-1 group names, in render order.
//   kindChip(kind: string) -> string
//       "next_turn" -> "next turn"; "next_turn_cold" -> "next turn, slower first reply";
//       "relaunch" -> "needs relaunch"; "not_here" -> "doesn't apply in this mode" (R-10). A
//       "fixed" row or an unknown kind never gets one of the two changeable chips (the text it
//       does get is not pinned).
//   valueText(value) -> string   (R-11)
//       null / undefined -> "doesn't apply here"; "not reported" never appears.
//   groupCatalog(catalog: CatalogEntry[]) -> Array<{ group: string, entries: CatalogEntry[] }>
//       the groups in SETTINGS_GROUPS order (only groups that have entries), each entry once, in
//       the order the catalog lists it; the input is not mutated.
//   relaunchText(item: SettingItem) -> string | null
//       item.relaunch verbatim (no trimming, no HTML escaping); null when it is null, absent or "".
//   itemView(entry: CatalogEntry, item: SettingItem) -> {
//       key, label, chip, source, what, useful_when, not_useful_when, evidence, cost,
//       relaunch: string | null, readOnly: boolean }
//       plain data for index.js to put into textContent. chip = kindChip(item.kind) (R-10 re-pin:
//       the item's effective kind, never the catalog's entry.kind);
//       relaunch = relaunchText(item); readOnly is true for a relaunch or fixed kind and whenever
//       item.changeable is false, false for a changeable next_turn / next_turn_cold item.
//
// CatalogEntry / SettingItem are the A2r-1 wire shapes (astra.settings.state): CatalogEntry
// carries the A1-1 text fields under local_astra/settings_catalog.py's attribute names (key,
// label, group, kind, what, useful_when, not_useful_when, evidence, cost); SettingItem is
// {key, value, source, kind, changeable, pending, relaunch}. Every expected value below is
// written by hand from the task text, never computed by a copy of the model.
//
// Activated with B-read (T-0213): the rows used to skip while model.js was absent; they now run
// unconditionally, so a missing or broken model.js fails load() with a named message, never skips.
//
// R-10 / R-11 rows (panel review rulings, 2026-10-31) run unconditionally since their activation.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);
const MODEL_FILE = fileURLToPath(new URL("hud/src/settings/model.js", REPO));
const INDEX_FILE = fileURLToPath(new URL("hud/src/settings/index.js", REPO));


// The same ES-module shim as tests/hud/assists.test.mjs: the sources are copied once into a temp
// directory that carries {"type":"module"} itself.
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
console.log(`# settings model origin: ${MODEL_FILE} exists=${existsSync(MODEL_FILE)}`);

async function load() {
  let mod;
  try {
    mod = await import(new URL("settings/model.js", srcRoot()).href);
  } catch (err) {
    assert.fail(
      "T-0213 B-1: hud/src/settings/model.js must import in node as an ES module with no DOM at " +
        `module top level; node could not import it: ${err.message}`,
    );
  }
  for (const name of ["SETTINGS_GROUPS", "kindChip", "groupCatalog", "relaunchText", "itemView"]) {
    assert.ok(name in mod, `T-0213 B-read: model.js exports ${name} (the surface this file names)`);
  }
  return mod;
}

// B-1, verbatim.
const B1_GROUPS = [
  "Seeing the screen",
  "Acting in the browser",
  "Checking its work",
  "Guidance and prompt",
  "Search and web",
  "Speech",
  "Engine (relaunch only)",
];

function entry(key, group, kind, extra = {}) {
  return {
    key,
    label: `label of ${key}`,
    group,
    kind,
    what: `what ${key} is`,
    useful_when: `${key} helps`,
    not_useful_when: `${key} does not help`,
    evidence: `T-0000 ${key} n=1`,
    cost: `${key} costs`,
    ...extra,
  };
}

function item(key, kind, extra = {}) {
  return {
    key,
    value: false,
    source: "default",
    kind,
    changeable: kind === "next_turn" || kind === "next_turn_cold",
    pending: null,
    relaunch: null,
    ...extra,
  };
}

// A catalog whose order interleaves the groups and whose keys are anti-alphabetical within a
// group, so a model that sorts by key or label, or keeps first-seen group order, fails.
function interleavedCatalog() {
  return [
    entry("zz_speech", "Speech", "relaunch"),
    entry("engine", "Engine (relaunch only)", "relaunch"),
    entry("zeta_screen", "Seeing the screen", "next_turn_cold"),
    entry("search", "Search and web", "relaunch"),
    entry("mid_check", "Checking its work", "next_turn"),
    entry("alpha_screen", "Seeing the screen", "next_turn"),
    entry("guidance", "Guidance and prompt", "next_turn_cold"),
    entry("act_b", "Acting in the browser", "next_turn"),
    entry("act_a", "Acting in the browser", "relaunch"),
    entry("a_check", "Checking its work", "next_turn"),
    entry("thinking", "Guidance and prompt", "fixed"),
    entry("aa_speech", "Speech", "fixed"),
  ];
}

// --- B-1: the groups and their order -------------------------------------------------------------

test("B-1: SETTINGS_GROUPS is the seven groups of B-1, in B-1's order", async () => {
  const m = await load();
  assert.deepEqual([...m.SETTINGS_GROUPS], B1_GROUPS);
});

test(
  "B-1: groupCatalog renders the groups in B-1 order, whatever order the catalog interleaves them in",
  async () => {
    const m = await load();
    const groups = m.groupCatalog(interleavedCatalog());
    assert.deepEqual(
      groups.map((g) => g.group),
      B1_GROUPS,
      "B-1: group order is the fixed panel order, not first-seen order",
    );
  },
);

test(
  "B-1: inside a group the entries keep the catalog's order (not sorted by key or label)",
  async () => {
    const m = await load();
    const byGroup = Object.fromEntries(
      m.groupCatalog(interleavedCatalog()).map((g) => [g.group, g.entries.map((e) => e.key)]),
    );
    assert.deepEqual(byGroup["Seeing the screen"], ["zeta_screen", "alpha_screen"]);
    assert.deepEqual(byGroup["Acting in the browser"], ["act_b", "act_a"]);
    assert.deepEqual(byGroup["Checking its work"], ["mid_check", "a_check"]);
    assert.deepEqual(byGroup["Guidance and prompt"], ["guidance", "thinking"]);
    assert.deepEqual(byGroup["Search and web"], ["search"]);
    assert.deepEqual(byGroup["Speech"], ["zz_speech", "aa_speech"]);
    assert.deepEqual(byGroup["Engine (relaunch only)"], ["engine"]);
  },
);

test(
  "B-1: every catalog entry appears exactly once, and the input catalog is not mutated",
  async () => {
    const m = await load();
    const catalog = interleavedCatalog();
    const before = JSON.stringify(catalog);
    const keys = m.groupCatalog(catalog).flatMap((g) => g.entries.map((e) => e.key));
    assert.equal(keys.length, 12, `twelve entries in, twelve out: ${keys}`);
    assert.equal(new Set(keys).size, 12, `no entry twice: ${keys}`);
    assert.equal(JSON.stringify(catalog), before, "groupCatalog is pure: the catalog is unchanged");
  },
);

test("B-1: a group with no entries is not rendered as an empty heading", async () => {
  const m = await load();
  const groups = m.groupCatalog([
    entry("b", "Speech", "relaunch"),
    entry("a", "Seeing the screen", "next_turn"),
  ]);
  assert.deepEqual(
    groups.map((g) => [g.group, g.entries.map((e) => e.key)]),
    [
      ["Seeing the screen", ["a"]],
      ["Speech", ["b"]],
    ],
  );
});

// --- B-1: the kind chip --------------------------------------------------------------------------

test("B-1: the chip text for each change kind is the task's, verbatim", async () => {
  const m = await load();
  assert.equal(m.kindChip("next_turn"), "next turn");
  assert.equal(m.kindChip("next_turn_cold"), "next turn, slower first reply");
  assert.equal(m.kindChip("relaunch"), "needs relaunch");
});

test(
  "adversarial: a fixed row or an unknown kind never claims to be changeable at the next turn",
  async () => {
    const m = await load();
    const changeable = new Set(["next turn", "next turn, slower first reply"]);
    for (const kind of [
      "fixed",
      "not_here",
      "later",
      "",
      "next_turn ",
      "NEXT_TURN",
      undefined,
      null,
    ]) {
      const chip = m.kindChip(kind);
      assert.ok(
        !changeable.has(chip),
        `kind ${JSON.stringify(kind)} got the changeable chip ${JSON.stringify(chip)}: only ` +
          "next_turn and next_turn_cold may say a change arrives at the next turn",
      );
    }
  },
);

// --- B-2: the relaunch command text --------------------------------------------------------------

test("B-2: relaunchText is the server's command verbatim", async () => {
  const m = await load();
  // The engine row's command (settings_catalog.py) holds "&&"; a model that escapes it for HTML
  // would show "&amp;&amp;" once index.js sets it with textContent.
  const engine = "diffusion_gearvis down && diffusion_gearvis up --profile lean";
  assert.equal(m.relaunchText(item("engine", "relaunch", { relaunch: engine })), engine);
  const steps = "diffusion_gearvis session --tools live --max-tool-steps N";
  assert.equal(m.relaunchText(item("max_tool_steps", "relaunch", { relaunch: steps })), steps);
  const toml = "set browser_viewport in astra.toml, then relaunch diffusion_gearvis session";
  assert.equal(m.relaunchText(item("browser_viewport", "relaunch", { relaunch: toml })), toml);
});

test("B-2: no command means no relaunch text (no empty Copy)", async () => {
  const m = await load();
  assert.equal(m.relaunchText(item("reground", "next_turn", { relaunch: null })), null);
  assert.equal(m.relaunchText(item("reground", "next_turn", { relaunch: "" })), null);
  const noField = item("reground", "next_turn");
  delete noField.relaunch;
  assert.equal(m.relaunchText(noField), null);
});

// --- B-1/B-2: the item view ----------------------------------------------------------------------

test(
  "B-1: itemView carries the catalog's text fields and the item's source, verbatim",
  async () => {
    const m = await load();
    const e = entry("action_outcome", "Checking its work", "next_turn", {
      label: "Action outcome",
      what: "A measured fact after each updated screen.",
      useful_when: "experiments",
      not_useful_when: "default use (no measured gain)",
      evidence: "T-0136 D n=100: 71 vs 69, no change",
      cost: "one clause per step",
    });
    const v = m.itemView(e, item("action_outcome", "next_turn", { value: true, source: "flag" }));
    assert.equal(v.key, "action_outcome");
    assert.equal(v.label, "Action outcome");
    assert.equal(v.what, "A measured fact after each updated screen.");
    assert.equal(v.useful_when, "experiments");
    assert.equal(v.not_useful_when, "default use (no measured gain)");
    assert.equal(v.evidence, "T-0136 D n=100: 71 vs 69, no change");
    assert.equal(v.cost, "one clause per step");
    assert.equal(v.source, "flag");
    assert.equal(v.chip, "next turn");
    assert.equal(v.relaunch, null);
    assert.equal(v.readOnly, false, "a changeable next_turn item is not read-only in the model");
  },
);

test(
  "B-2: relaunch and fixed items are read-only and a relaunch item carries its command",
  async () => {
    const m = await load();
    const cmd = "diffusion_gearvis session --tools live --search off";
    const rl = m.itemView(
      entry("search", "Search and web", "relaunch"),
      item("search", "relaunch", { relaunch: cmd, value: "ddgs" }),
    );
    assert.equal(rl.readOnly, true);
    assert.equal(rl.chip, "needs relaunch");
    assert.equal(rl.relaunch, cmd);
    const fixed = m.itemView(
      entry("thinking", "Guidance and prompt", "fixed"),
      item("thinking", "fixed"),
    );
    assert.equal(fixed.readOnly, true, "thinking is fixed (A2a-6): never a control");
    // A "later" row: a next-turn kind the server says is not changeable in v1 (wait_tool).
    const later = m.itemView(
      entry("wait_tool", "Acting in the browser", "next_turn_cold"),
      item("wait_tool", "next_turn_cold", {
        changeable: false,
        relaunch: "diffusion_gearvis session --tools live --wait-tool off",
      }),
    );
    assert.equal(later.readOnly, true, "changeable=false from the server makes the item read-only");
    assert.equal(later.relaunch, "diffusion_gearvis session --tools live --wait-tool off");
  },
);

test(
  "adversarial: a relaunch-kind item the server marks changeable is still read-only",
  async () => {
    const m = await load();
    const v = m.itemView(
      entry("browser_web", "Search and web", "relaunch"),
      item("browser_web", "relaunch", {
        changeable: true,
        relaunch: "diffusion_gearvis session --tools live --no-web",
      }),
    );
    assert.equal(v.readOnly, true, "B-2: relaunch-only items are read-only whatever the flag says");
  },
);

// --- adversarial: catalog text is text -----------------------------------------------------------

test(
  "adversarial: an evidence string holding <img src=x onerror=...> comes back as the same text, not markup",
  async () => {
    const m = await load();
    delete globalThis.__t0213_pwned;
    const evil = '<img src=x onerror="globalThis.__t0213_pwned=1">';
    const evilLabel = "<b>bold</b> &amp; label";
    const v = m.itemView(
      entry("reground", "Seeing the screen", "next_turn", {
        evidence: evil,
        label: evilLabel,
        cost: "<script>x</script>",
      }),
      item("reground", "next_turn"),
    );
    assert.equal(
      v.evidence,
      evil,
      "the evidence must reach textContent unchanged: not stripped, not HTML-escaped (escaping " +
        "before textContent shows &lt;img&gt; to the user), not parsed",
    );
    assert.equal(v.label, evilLabel);
    assert.equal(v.cost, "<script>x</script>");
    assert.equal(globalThis.__t0213_pwned, undefined, "nothing in the string ran");
    for (const field of [
      "key",
      "label",
      "chip",
      "source",
      "what",
      "useful_when",
      "not_useful_when",
      "evidence",
      "cost",
    ]) {
      assert.equal(typeof v[field], "string", `itemView.${field} is a plain string for textContent`);
    }
  },
);

const stripComments = (s) =>
  s.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/(^|[^:])\/\/[^\n]*/g, "$1");

test(
  "adversarial: the settings modules never write markup (no innerHTML, outerHTML, insertAdjacentHTML, document.write)",
  async () => {
    const offenders = [];
    for (const file of [MODEL_FILE, INDEX_FILE]) {
      if (!existsSync(file)) continue;
      const code = stripComments(readFileSync(file, "utf8"));
      for (const sink of [
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "DOMParser",
        "createContextualFragment",
      ]) {
        if (code.includes(sink)) offenders.push(`${file}: ${sink}`);
      }
    }
    assert.deepEqual(offenders, [], "B-1: every catalog string is set with textContent");
  },
);

test("B-1: model.js is pure: it touches no DOM global and opens no socket", async () => {
  await load(); // importing in node, where document and WebSocket do not exist, is itself a check
  const code = stripComments(readFileSync(MODEL_FILE, "utf8"));
  for (const name of ["document.", "window.", "WebSocket", "localStorage", "navigator."]) {
    assert.ok(!code.includes(name), `T-0213 owned files: model.js is pure; it mentions ${name}`);
  }
});

// --- R-10: the chip comes from the item's effective kind -----------------------------------------

const NOT_HERE_CHIP = "doesn't apply in this mode"; // R-10, verbatim (ASCII apostrophe)

test("R-10: kind not_here gives the chip \"doesn't apply in this mode\"", async () => {
  const m = await load();
  assert.equal(m.kindChip("not_here"), NOT_HERE_CHIP);
  for (const changeable of ["next turn", "next turn, slower first reply"]) {
    assert.notEqual(m.kindChip("not_here"), changeable, "not_here is never a next-turn chip");
  }
});

test(
  "R-10: itemView's chip is the item's kind, not the catalog's (relaunch over a next_turn_cold entry)",
  async () => {
    const m = await load();
    // page_overview in a live session: the catalog says next_turn_cold, the server says relaunch.
    const v = m.itemView(
      entry("page_overview", "Seeing the screen", "next_turn_cold"),
      item("page_overview", "relaunch", { changeable: false, relaunch: "--page-overview" }),
    );
    assert.equal(v.chip, "needs relaunch", `R-10: the item's kind decides the chip; got ${v.chip}`);
    assert.equal(v.readOnly, true);
    assert.equal(v.relaunch, "--page-overview");
  },
);

test(
  "R-10 adversarial: a not_here item over a next_turn entry never shows a next-turn chip",
  async () => {
    const m = await load();
    // reground outside live: the catalog says next_turn, the server says not_here.
    const v = m.itemView(
      entry("reground", "Seeing the screen", "next_turn"),
      item("reground", "not_here", { changeable: false, value: null }),
    );
    assert.equal(v.chip, NOT_HERE_CHIP, `R-10: got ${JSON.stringify(v.chip)}`);
    assert.equal(v.readOnly, true, "a not_here item is read-only");
    // and the reverse: a changeable item whose kind matches the catalog keeps its chip
    const w = m.itemView(
      entry("visual_tokens", "Seeing the screen", "next_turn_cold"),
      item("visual_tokens", "next_turn_cold", { value: 1120 }),
    );
    assert.equal(w.chip, "next turn, slower first reply");
    assert.equal(w.readOnly, false);
  },
);

// --- R-11: a null value -------------------------------------------------------------------------

const NULL_VALUE_TEXT = "doesn't apply here"; // R-11, verbatim (ASCII apostrophe)

test("R-11: valueText(null) is \"doesn't apply here\"", async () => {
  const m = await load();
  assert.ok(typeof m.valueText === "function", "model.js exports valueText (R-11 surface)");
  assert.equal(m.valueText(null), NULL_VALUE_TEXT);
  assert.equal(m.valueText(undefined), NULL_VALUE_TEXT);
  const v = m.itemView(
    entry("max_tool_steps", "Acting in the browser", "relaunch"),
    item("max_tool_steps", "relaunch", { value: null, changeable: false }),
  );
  assert.equal(v.valueText, NULL_VALUE_TEXT, "the row's value line for a null value");
  // premise / anti-vacuity: a real value still reads as itself
  assert.equal(m.valueText(1120), "1120");
  assert.equal(m.valueText(false), "off");
});

test(
  "R-11 adversarial: \"not reported\" appears nowhere: not in any itemView string, not in model.js or index.js",
  async () => {
    const m = await load();
    for (const value of [null, undefined]) {
      const v = m.itemView(entry("k", "Speech", "fixed"), item("k", "fixed", { value }));
      for (const [field, text] of Object.entries(v)) {
        if (typeof text === "string") {
          assert.ok(!/not reported/i.test(text), `R-11: itemView.${field} reads ${text}`);
        }
      }
    }
    for (const file of [MODEL_FILE, INDEX_FILE]) {
      const code = stripComments(readFileSync(file, "utf8"));
      assert.ok(
        !/not reported/i.test(code),
        `R-11: ${file} still holds "not reported" (comments are stripped; a string literal trips this)`,
      );
    }
  },
);
