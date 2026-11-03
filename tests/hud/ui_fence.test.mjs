// T-0054 group B — the fence panel: hud/src/ui/fence.js and the new code in hud/src/ui/index.js.
//
// Criteria 14-18 of docs/tasks/T-0054.md. The failure this file exists for is the one the human
// met on 2026-09-16: a surface that arrived in pieces, a surface abandoned at the bound and a
// turn that never had a surface at all all produced the same pane, so a half interface on screen
// was indistinguishable from a whole one. After this task the pane tells them apart — and, the
// part D5 is about, it tells them apart *only* by echoing what the server said.
//
// This file skips itself while hud/src/ui/fence.js exports no describeFence. Node tests cannot
// carry pytest's `pending` marker and tests/unit/test_hud_js.py requires `node --test tests/hud`
// to exit 0 (the convention of tests/hud/log_panel.test.mjs and tests/hud/ui_bridge.test.mjs).
// The hard, red-now pins for group B are Python: tests/unit/test_hud_constants_parity.py (the
// mirror, the field table, the close enum, and that this is a new type rather than a new field)
// and tests/unit/test_hud_no_audio_no_egress.py (the file exists and sends nothing).
//
// API this pins (docs/tasks/T-0054.md criterion 14 fixes the shape; these are the names):
//
//   hud/src/ui/fence.js — no DOM, no clock, importable in node
//     FENCE_TYPE = "astra.ui.fence"
//     describeFence(message) -> { status, label, canvases, refused }
//       status in {"none","whole","reassembled","incomplete","abandoned","refused-only","unknown"}
//       A pure function of one decoded message: no state across calls, no clock, and nothing
//       derived that the server did not send (D5).
//
//   hud/src/ui/index.js — the panel's own container carries
//     [data-ui-fence]           the status
//     [data-ui-fence-canvases]  the count the server sent
//     [data-ui-fence-refused]   the refusal count the server sent
//   (T-0154 B3 moves these three marks onto the turn's meta line in the conversation thread; the
//   rows that read them now live in tests/hud/thread.test.mjs — see "criteria 16 and 17" below.)
//
// Every expected value below is computed by hand from criterion 15's table and from
// local_astra/contracts/protocol.AstraUiFence — never from the module under test.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);

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

// Origin proof (.claude/rules/tests.md, 2026-09-16): a mutation run started in a copy can import
// the tree it meant to leave behind. Printed once, from inside this process.
console.log(`# ui fence origin: repo=${fileURLToPath(REPO)} modules=${new URL("ui/", srcRoot()).href}`);

async function tryLoad(name) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch {
    return null;
  }
}

const FENCE = await tryLoad("ui/fence.js");
const INSTALLED = FENCE !== null && typeof FENCE.describeFence === "function";
const opts = INSTALLED
  ? {}
  : {
      skip:
        "T-0054 group B is not implemented yet: hud/src/ui/fence.js exports no describeFence " +
        "(docs/tasks/T-0054.md criteria 14-17). Node tests cannot carry pytest's `pending` " +
        "marker and tests/unit/test_hud_js.py requires `node --test tests/hud` to exit 0; the " +
        "red-now pins for this group are tests/unit/test_hud_constants_parity.py and " +
        `tests/unit/test_hud_no_audio_no_egress.py. (Origin: ${new URL("ui/fence.js", srcRoot()).href})`,
    };

async function load(name) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch (err) {
    assert.fail(
      `docs/tasks/T-0054.md (group B) owns hud/src/${name}; it must be importable in node as an ` +
        `ES module with no DOM at module top level. node could not import it: ${err.message}`,
    );
  }
}

/** One decoded astra.ui.fence, as contracts/protocol.AstraUiFence encodes it (criterion 8). */
const fence = ({ turn_id = "t6", canvases = 0, close = null, continued = false, speech_refused = 0 } = {}) => ({
  type: "astra.ui.fence",
  turn_id,
  canvases,
  close,
  continued,
  speech_refused,
});

// --- criterion 15: the seven rows, each with its own status --------------------------------
//
// The table is transcribed from docs/tasks/T-0054.md criterion 15. `status` is what tests and
// screenshots key on and `label` is English the human may still move (ambiguity 4), so the two
// are asserted separately: a wording change must not be able to break a status pin.

const ROWS = [
  {
    name: "a whole surface says nothing new",
    message: fence({ canvases: 1, close: "written" }),
    status: "whole",
    label: "",
  },
  {
    name: "a fence the client inferred, on one canvas, is still whole",
    message: fence({ canvases: 1, close: "inferred" }),
    status: "whole",
    label: "",
  },
  {
    name: "a surface carried across canvases is reassembled",
    message: fence({ canvases: 2, close: "inferred", continued: true }),
    status: "reassembled",
    contains: ["2", "reassembled"],
  },
  {
    name: "a fence the model wrote, carried, is reassembled too",
    message: fence({ canvases: 3, close: "written", continued: true }),
    status: "reassembled",
    contains: ["3", "reassembled"],
  },
  {
    name: "a canvas that ended inside the fence is incomplete",
    message: fence({ canvases: 1, close: "open" }),
    status: "incomplete",
    contains: ["cut off"],
  },
  {
    name: "a carry abandoned at the bound is abandoned",
    message: fence({ canvases: 5, close: "abandoned", continued: true }),
    status: "abandoned",
    contains: ["never finished"],
  },
  {
    name: "a refusal with nothing shown is refused-only",
    message: fence({ canvases: 0, close: null, speech_refused: 1 }),
    status: "refused-only",
    contains: ["not spoken", "nothing was shown"],
  },
];

for (const row of ROWS) {
  test(`criterion 15: ${row.name}`, opts, async () => {
    const mod = await load("ui/fence.js");
    const got = mod.describeFence(row.message);
    assert.equal(
      got.status,
      row.status,
      `criterion 15: ${JSON.stringify(row.message)} reads "${row.status}"; describeFence said ` +
        JSON.stringify(got),
    );
    assert.equal(got.canvases, row.message.canvases, "criterion 14: the count is echoed, not derived");
    assert.equal(got.refused, row.message.speech_refused, "criterion 14: so is the refusal count");
    if (row.label === "") {
      assert.equal(
        String(got.label || ""),
        "",
        `criterion 15/16: "${row.status}" says nothing — today's pane, unchanged. It said ` +
          JSON.stringify(got.label),
      );
    }
    for (const needle of row.contains || []) {
      assert.ok(
        String(got.label || "").includes(needle),
        `criterion 15: the label for "${row.status}" names ${JSON.stringify(needle)}; it reads ` +
          JSON.stringify(got.label),
      );
    }
  });
}

test("criterion 15: no message for the turn is 'none', and is never inferred from silence", opts, async () => {
  const mod = await load("ui/fence.js");
  // The absence of a message is not a message: describeFence is a pure function of one decoded
  // message (D5), so "none" is what it answers when handed nothing — never a state it reaches by
  // remembering that no message has arrived. A panel that accumulated would contradict the server
  // after a reconnect, and this message is deliberately never replayed (criterion 8).
  for (const nothing of [undefined, null]) {
    const got = mod.describeFence(nothing);
    assert.equal(
      got.status,
      "none",
      `criterion 15: with no message the status is "none" and T-0008's placeholder stands; got ` +
        JSON.stringify(got),
    );
    assert.equal(String(got.label || ""), "", "criterion 15: and it says nothing");
  }
});

test("criterion 15: a refusal is reported next to whatever the fence did", opts, async () => {
  const mod = await load("ui/fence.js");
  const got = mod.describeFence(fence({ canvases: 2, close: "inferred", continued: true, speech_refused: 3 }));
  assert.equal(
    got.status,
    "reassembled",
    `criterion 15's last row: speech_refused > 0 does not change the fence's own status when a ` +
      `surface was shown; got ${JSON.stringify(got)}`,
  );
  assert.equal(got.refused, 3, "criterion 14: the count is the server's");
  assert.ok(
    String(got.label || "").includes("3") && /not spoken/.test(String(got.label || "")),
    `criterion 15: …plus "N pieces of interface data were not spoken"; the label reads ` +
      JSON.stringify(got.label),
  );
});

// --- adversarial: a close value the page does not know ---------------------------------------

test("adversarial: a close value the page does not know never claims completeness", opts, async () => {
  const mod = await load("ui/fence.js");
  const protocol = await load("protocol.js");

  // First the decoder: a future contract value must not reach the panel at all.
  const decoded = protocol.decode(JSON.stringify(fence({ canvases: 1, close: "salvaged" })));
  assert.equal(
    decoded.kind,
    "malformed",
    `criterion 13: close is an enum of the four FenceClose values, so "salvaged" decodes ` +
      `malformed and is ignored and counted — never rendered. decode() said ` +
      JSON.stringify(decoded),
  );

  // Then the panel, in case one ever reaches it another way: unknown, never a success.
  let got;
  assert.doesNotThrow(() => {
    got = mod.describeFence(fence({ canvases: 1, close: "salvaged" }));
  }, "criterion 17: describeFence never throws — a panel that throws takes the page's message loop with it");
  assert.equal(
    got.status,
    "unknown",
    `criterion 14: a value this page does not know is "unknown"; it said ${JSON.stringify(got)}`,
  );
  assert.ok(
    got.status !== "whole" && got.status !== "reassembled",
    "a page that silently renders a future close value as success is how the panel starts lying " +
      `again; it said ${JSON.stringify(got)}`,
  );
});

test("adversarial: describeFence holds no state across calls (D5)", opts, async () => {
  const mod = await load("ui/fence.js");
  const abandoned = fence({ canvases: 4, close: "abandoned", continued: true });
  const first = mod.describeFence(abandoned);
  mod.describeFence(fence({ canvases: 1, close: "written" }));
  mod.describeFence(fence({ canvases: 0, close: null, speech_refused: 2 }));
  const again = mod.describeFence(abandoned);
  assert.deepEqual(
    again,
    first,
    "D5: the same message must describe the same way whatever came before it. A panel that " +
      "remembers can contradict the server after a reconnect — and this message is deliberately " +
      "not replayed, so a remembering panel is wrong exactly when a page reloads.",
  );
});

test("adversarial: a malformed or foreign message is safe and claims nothing", opts, async () => {
  const mod = await load("ui/fence.js");
  const cases = [
    { name: "a surface message", msg: { type: "astra.ui.surface", surface_id: "s", canvas_idx: 0, messages: [] } },
    { name: "a fence with no fields", msg: { type: "astra.ui.fence" } },
    { name: "a fence with a string count", msg: fence({ canvases: "2" }) },
    { name: "a number", msg: 7 },
    { name: "a string", msg: "astra.ui.fence" },
  ];
  for (const { name, msg } of cases) {
    let got;
    assert.doesNotThrow(() => {
      got = mod.describeFence(msg);
    }, `criterion 17: describeFence must not throw on ${name}`);
    assert.ok(
      got && typeof got.status === "string",
      `criterion 14: it always answers a { status, label, canvases, refused }; for ${name} it ` +
        `answered ${JSON.stringify(got)}`,
    );
    assert.ok(
      got.status !== "whole" && got.status !== "reassembled",
      `criterion 14/D5: ${name} is not evidence that anything was shown, let alone that it was ` +
        `complete; it read ${JSON.stringify(got)}`,
    );
  }
});

// --- criteria 16 and 17: the panel -----------------------------------------------------------
//
// **Moved by T-0154 B3/B4 (G2).** The panel rows of this section mounted hud/src/ui/index.js into
// the retired #shown-channel (B1) and read the fence marks off the panel's one container. B3 puts
// the fence notice on the turn's meta line in the conversation thread and B4 mounts the panel
// against the thread's card slots, so these five rows now live in tests/hud/thread.test.mjs under
// "B3 — the fence notice on the turn's meta line", re-keyed to the article of the turn the fence
// names (same hooks, same values, same wire orders):
//   "criterion 16: a healthy turn adds nothing to the DOM"
//   "criterion 15: the three states the human could not tell apart reach the DOM"
//   "criterion 17: a fence before any surface, and one for another turn, are safe"
//   "ruling B8: an abandoned turn's verdict does not stay on screen over the next turn"
//   "ruling B11: a turn's own fence survives that turn's own surfaces, and only a new turn clears it"
// Amended, by ruling B8's own reason: the abandoned turn's verdict is no longer *cleared* by the
// next turn's surface; it stays in its own article as history and never appears in the next
// turn's. Kept here: the pure describeFence rows above, the no-document mount row, criterion 18
// and ruling B9.

test("criterion 17: a host with no document leaves everything else working", opts, async () => {
  const index = await load("ui/index.js");
  let result;
  await assert.doesNotReject(async () => {
    result = await index.mount({ subscribe: () => () => {}, log: () => {} });
  }, "criterion 17/T-0008 criterion 10: mount never throws out of mountPanels");
  assert.equal(result.mounted, false, "a host with no document renders nothing and takes nothing down");
});

test("criterion 18: the fence panel adds no CSS rule and no file outside its own two", opts, async () => {
  // T-0056 group B owns hud/src/ui/style.css and the fence line renders unstyled until then —
  // stated as owed in T-0054's ownership section, so a rule appearing here is a task boundary
  // crossed, not a nicety.
  // T-0164 B5 deletes hud/src/ui/style.css; from group B's activation (T-0164's **Status:** phrase
  // "group B: tests …, activated <sha>", or ASTRA_RUN_PENDING=1) the page's one ui sheet is
  // hud/src/ui/card.css, and the meaning is kept: the fence panel's hooks are styled by no ui sheet.
  const t0164 = fileURLToPath(new URL("docs/tasks/T-0164.md", REPO));
  const t0164Status = existsSync(t0164)
    ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(t0164, "utf8")) || [])[1] || "")
    : "";
  const t0164B =
    /\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(t0164Status) || process.env.ASTRA_RUN_PENDING === "1";
  const sheet = t0164B ? "hud/src/ui/card.css" : "hud/src/ui/style.css";
  const css = readFileSync(fileURLToPath(new URL(sheet, REPO)), "utf8");
  for (const needle of ["a2ui-fence", "data-ui-fence"]) {
    assert.ok(
      !css.includes(needle),
      `criterion 18: ${sheet} is not the fence panel's file and styles none of its hooks; it ` +
        `now mentions ${JSON.stringify(needle)}`,
    );
  }
});

// --- ruling B9 (docs/tasks/T-0054.md, group B), the pure half ---------------------------------
//
// It uses the sentinel convention above (`opts`), so it skips while hud/src/ui/fence.js exports
// no describeFence and `node --test tests/hud` still exits 0 for tests/unit/test_hud_js.py. It
// does not pin the wording of the `refused-only` sentence, which ruling B7 is moving. (Ruling B8's
// panel row moved to tests/hud/thread.test.mjs with T-0154 B3; see the section header above.)

test("ruling B9: more than one canvas reads as reassembled even when `continued` is false", opts, async () => {
  // Criterion 15 gives `whole` exactly one canvas and no carry. A pane that keyed its status on
  // `continued` alone would call a surface built out of two canvases whole — claiming more than
  // the server said, which is D5's whole point — and no other row in this file has that
  // combination, so the mistake would be silent.
  const mod = await load("ui/fence.js");
  const got = mod.describeFence(fence({ canvases: 2, close: "inferred", continued: false }));
  assert.equal(
    got.status,
    "reassembled",
    "criterion 15: `whole` is the one-canvas row; two canvases produced what is on screen, so " +
      `the pane says it was put together out of them. describeFence said ${JSON.stringify(got)}`,
  );
  assert.notEqual(
    got.status,
    "whole",
    "a two-canvas surface described as whole is the pane claiming completeness the server did " +
      `not claim (D5); it said ${JSON.stringify(got)}`,
  );
  assert.equal(got.canvases, 2, "criterion 14: the count is the server's, echoed");
  assert.ok(
    String(got.label || "") !== "",
    "criterion 16 makes `whole` and `none` the silent statuses; this one is neither, so the pane " +
      `has something to say about it. Its label is ${JSON.stringify(got.label)}`,
  );
});

