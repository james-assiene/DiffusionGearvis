// T-0181 group B — the HUD microphone control, the node half (docs/tasks/T-0181.md B1-B3, B5, the
// "Group A0 replan" knock-on edits, rulings H-T181-4: button only, no keyboard shortcut for the mic and
// none for Stop sharing; H-T181-5: a session with no mic shows a disabled "No mic" with the reason;
// H-T181-10: an abandoned question's empty final shows no bubble). The shell rows (ids, the header
// order, Stop sharing's label) are in tests/hud/shell.test.mjs; the Chrome half (real clicks and keys,
// the badge's colour and contrast, overflow, loopback) is tests/integration/test_t0181_hud_mic.py.
//
// Every expected string below is copied by hand from the task's criteria (B1's labels and B3's titles
// without "(m)", B2's badge, B3's badge title), never read from an implementation: hud/src/mic.js does
// not exist when this file is written.
//
// Gate. Every row runs through `pendingT0181B`, the Status-phrase strict-xfail gate of
// tests/hud/thoughts.test.mjs and tests/hud/vision.test.mjs, reading docs/tasks/T-0181.md. While its
// **Status:** line does not record "group B: tests <a>..<b>, activated <sha>", each row must FAIL (an
// XPASS fails the row: a stub that satisfies it is vacuous). ASTRA_RUN_PENDING=1 runs them for real
// now. Group A's and A0's wording ("Group A (with A0-S) merged …", "Group B (the HUD mic button …)
// next", "reviewer B approved") does not switch these rows on.
//
// API these rows pin (the task names the file, `hud/src/mic.js`, "DOM-free", and leaves the JS names to
// the test-author; these are them):
//
//   hud/src/mic.js — importable in node, no DOM at module top level.
//     export const MIC_CLIENT_TYPES = ["astra.mic.mute"]   (the convention of VISION_CLIENT_TYPES)
//     export function createMic() -> {
//       accept(msg)  every decoded astra.* message, in wire order; never throws; true iff it was an
//                    astra.mic.state (re-render the control)
//       view()       null before any astra.mic.state; else
//                    { text, title, pressed, disabled, badge }
//                      text      the button's label
//                      title     the button's title attribute
//                      pressed   true | false while the mic is available (aria-pressed); unpinned
//                                (null or absent are both fine) while it is not
//                      disabled  boolean
//                      badge     null while not muted; else { text, title } of [data-mic-badge]
//       click()      the message to send for a click on the button: { type: "astra.mic.mute",
//                    muted: !echoed.muted }, or null (before a state, or with no mic). It changes
//                    nothing: the view moves only on the next echo.
//     }
//     It has no keyboard entry point at all (B3).
//
//   hud/src/panels/transcript.js — renderBubble(el, line), as today. An empty final line leaves the
//     element hidden: `el.hidden === true` (the page's convention for "not shown": #capture-banner,
//     the error strip, the header badge; a class with a CSS rule would pass the Chrome row but not
//     this one — recorded as the test-author's reading).
//
// Readings the task leaves open (stated, not guessed silently):
//   - aria-pressed on the disabled "No mic" control is unpinned (B1 gives text, disabled and title).
//   - A live astra.session.cleared leaves the control as it was: the mute survives a reset (A9,
//     H-T181-3) and the server re-sends the state only on a hello, so a page that forgot it on a reset
//     would hide the control or show a stale "Mute mic" for a muted session.
//   - Whitespace-only finals are unpinned; the server's abandoned final is exactly "" (A0-S4).
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);

// --- the T-0181 group B strict-xfail gate --------------------------------------------------------

const TASK = fileURLToPath(new URL("docs/tasks/T-0181.md", REPO));
const STATUS = existsSync(TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(TASK, "utf8")) || [])[1] || "").trim()
  : "";
/** "group B: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6" — the phrase the orchestrator writes. */
const B_ACTIVATED = /\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const B_PENDING = !B_ACTIVATED.test(STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0181B(name, fn) {
  test(name + (B_PENDING ? " [pending T-0181 B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0181 group B row passed while docs/tasks/T-0181.md's status does " +
          "not record group B's activation (" + JSON.stringify(STATUS.slice(0, 80)) + "). Either " +
          "group B has landed (record 'group B: tests …, activated <sha>' in the status) or the " +
          "row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0181 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0181 B gate: the status line is readable and only group B's own phrase activates the rows", () => {
  assert.ok(STATUS.length > 0, "docs/tasks/T-0181.md must carry a **Status:** line");
  // Hand-written samples, the first three today's wording for groups A, A0 and B.
  assert.equal(B_ACTIVATED.test("in progress. Group A (with A0-S) merged 2026-10-28 at 6c299004 (reviewer B approved; …)"), false);
  assert.equal(B_ACTIVATED.test("Group B (the HUD mic button, no `m` key; re-read test_t0177_hud_vision's `m`-key row) next"), false);
  assert.equal(B_ACTIVATED.test("Group A0 merged 2026-10-28 at aea18d0a (implementation 9dbddf96; reviewers A and B approved)"), false);
  assert.equal(B_ACTIVATED.test("in progress (group A: tests 1a2b3c4, activated 9a8b7c6; group B: tests planned)"), false);
  assert.equal(B_ACTIVATED.test("in progress (group B: tests a1b2c3d..e4f5a6b, activated c3d4e5f)"), true);
  assert.equal(B_ACTIVATED.test("in progress. group B: tests a1b2c3d, activated c3d4e5f; nothing else"), true);
});

// --- loading the modules --------------------------------------------------------------------------

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

// Origin proof (.claude/rules/tests.md): where the modules under test were imported from.
console.log(`# mic origin: repo=${fileURLToPath(REPO)} modules=${srcRoot().href}`);

async function load(name) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch (err) {
    assert.fail(
      `T-0181 B: hud/src/${name} must be importable in node as an ES module with no DOM at module ` +
        `top level. node could not import it: ${err.message}`,
    );
  }
}

/** The module's code with comments removed (a source guard must not match prose; strings are kept). */
function codeOf(rel) {
  const src = readFileSync(fileURLToPath(new URL(rel, REPO)), "utf8");
  return src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:\\])\/\/.*$/gm, "$1");
}

/** A wire message through the page's real decoder (hud/src/protocol.js), as main.js hands it on. */
async function wire(obj) {
  const { decode } = await load("protocol.js");
  const got = decode(JSON.stringify(obj));
  assert.equal(got.kind, "astra", `premise: ${JSON.stringify(obj)} decodes as a typed message; got ${JSON.stringify(got)}`);
  return got.msg;
}

async function mic() {
  const mod = await load("mic.js");
  assert.equal(typeof mod.createMic, "function", "hud/src/mic.js exports createMic()");
  return mod.createMic();
}

// --- the expected copy, by hand from B1/B2 as amended by B3 (no "(m)") ------------------------------

const LIVE = {
  text: "Mute mic",
  pressed: false,
  title: "Mute the microphone: Gearvis hears nothing until you unmute; its own speech still plays and typing still works",
};
const MUTED = { text: "Unmute mic", pressed: true, title: "Unmute the microphone" };
const BADGE = { text: "Mic muted", title: "Gearvis can't hear you. Click Unmute mic to let it listen again." };
const NO_MIC = "No mic";

// The three reasons of A3, verbatim, plus one carrying markup-looking text (it is a title, never HTML).
const REASONS = [
  "this audio path has no capture seam",
  "this session has no microphone (--no-mic)",
  "this session has no microphone (--mic none)",
  "this session is silent: no audio device is open (--silent)",
  'a "quoted" <b>reason</b> & more',
];

// Wire shapes: codec.encode drops None, so an available state arrives with no `reason` key (A-R8).
const S_LIVE = { type: "astra.mic.state", muted: false, available: true };
const S_MUTED = { type: "astra.mic.state", muted: true, available: true };
const s_gone = (reason) => ({ type: "astra.mic.state", muted: false, available: false, reason });

function assertAvailable(view, want, why) {
  assert.ok(view, `${why}: the control is shown (view() is not null)`);
  assert.equal(view.text, want.text, `${why}: B1 label`);
  assert.equal(view.title, want.title, `${why}: B1/B3 title (no "(m)": H-T181-4)`);
  assert.equal(view.pressed, want.pressed, `${why}: B1 aria-pressed`);
  assert.equal(view.disabled, false, `${why}: an available mic's control is enabled`);
}

// --- B1: the module ------------------------------------------------------------------------------

pendingT0181B("T-0181 B1: mic.js imports in node with no DOM and names its one client type, astra.mic.mute", async () => {
  assert.equal(typeof globalThis.document, "undefined", "premise: node has no document");
  const mod = await load("mic.js");
  assert.deepEqual(mod.MIC_CLIENT_TYPES, ["astra.mic.mute"], "B5: the only new message the page sends is astra.mic.mute");
  assert.equal(typeof mod.createMic, "function", "createMic() is exported");
});

// --- B1: nothing before a state ------------------------------------------------------------------

pendingT0181B("T-0181 B1: no control before an astra.mic.state, whatever else arrives; a click sends nothing", async () => {
  const m = await mic();
  assert.equal(m.view(), null, "B1: the control is shown only after an astra.mic.state arrives");
  assert.equal(m.click(), null, "B1: before any state there is no echoed value to invert, so nothing is sent");
  const others = [
    { type: "astra.turn.state", state: "LISTENING", turn_id: "t1", thinking_ms: 0 },
    { type: "astra.transcript", turn_id: "t1", text: "how many", final: false },
    { type: "astra.vision.state", source: "browser", choices: ["browser", "shared"] },
    { type: "astra.session.cleared", resets: 1, ended_turn_id: null, memory_forgotten: 0 },
  ];
  for (const raw of others) {
    const msg = await wire(raw);
    assert.equal(m.accept(msg), false, `accept() is true only for astra.mic.state; ${raw.type} said true`);
    assert.equal(m.view(), null, `B1: ${raw.type} does not show the control`);
  }
  // The sender-only refusal of A2 is an astra.error, not a state: it does not show the control either.
  const err = await wire({ type: "astra.error", code: "mic_unavailable", message: "this session has no microphone (--no-mic)" });
  m.accept(err);
  assert.equal(m.view(), null, "B1: an astra.error(mic_unavailable) is not a state");
  assert.equal(m.click(), null, "…and still nothing to send");
});

// --- B1: the three label sets ----------------------------------------------------------------------

pendingT0181B("T-0181 B1: available and live reads 'Mute mic', aria-pressed false, the B3 title", async () => {
  const m = await mic();
  assert.equal(m.accept(await wire(S_LIVE)), true, "accept() is true for an astra.mic.state");
  assertAvailable(m.view(), LIVE, "live");
  assert.equal(m.view().badge, null, "B2: no badge while not muted");
});

pendingT0181B("T-0181 B1/B2: muted reads 'Unmute mic', aria-pressed true, and the badge says 'Mic muted'", async () => {
  const m = await mic();
  m.accept(await wire(S_MUTED));
  const v = m.view();
  assertAvailable(v, MUTED, "muted");
  assert.deepEqual(v.badge, BADGE, "B2/B3: the header badge 'Mic muted' with its title (no 'Press m')");
});

for (const reason of REASONS) {
  pendingT0181B(`T-0181 B1 (H-T181-5): no mic reads 'No mic', disabled, the reason verbatim as title: ${JSON.stringify(reason)}`, async () => {
    const m = await mic();
    m.accept(await wire(s_gone(reason)));
    const v = m.view();
    assert.ok(v, "B1: a session with no mic still shows the control (disabled), so the reason is visible");
    assert.equal(v.text, NO_MIC, "B1: the label is 'No mic'");
    assert.equal(v.disabled, true, "B1: the control is disabled");
    assert.equal(v.title, reason, "B1 (H-T181-5): the server's reason, verbatim, is the title");
    assert.equal(v.badge, null, "B2: an unavailable mic is never muted, so no badge");
    assert.equal(m.click(), null, "A2/B1: nothing to send for a click on a disabled control");
  });
}

// --- B1: the state is the echo, never the click ------------------------------------------------------

pendingT0181B("T-0181 B1: a click sends astra.mic.mute {muted: !echoed} and changes nothing until the echo", async () => {
  const m = await mic();
  m.accept(await wire(S_LIVE));
  const before = JSON.stringify(m.view());
  assert.deepEqual(m.click(), { type: "astra.mic.mute", muted: true }, "B1: live → the click asks to mute");
  assert.equal(JSON.stringify(m.view()), before, "B1: the click itself changes nothing on the control");
  // Adversarial (the stub withholds the echo): a second click still inverts the *echo*, not a local flag.
  assert.deepEqual(
    m.click(),
    { type: "astra.mic.mute", muted: true },
    "B1: with no echo yet, the second click still sends muted:true (a page that toggles locally sends false here)",
  );
  assert.equal(JSON.stringify(m.view()), before, "…and the control is still live");
  m.accept(await wire(S_MUTED));
  assertAvailable(m.view(), MUTED, "after the muted echo");
  assert.deepEqual(m.click(), { type: "astra.mic.mute", muted: false }, "B1: muted → the click asks to unmute");
  assert.deepEqual(m.click(), { type: "astra.mic.mute", muted: false }, "…twice, with no echo in between");
  assertAvailable(m.view(), MUTED, "the unmute click alone does not unmute the control");
});

pendingT0181B("T-0181 B1: a redundant echo (the server echoes every routed mute) leaves the control as echoed", async () => {
  const m = await mic();
  m.accept(await wire(S_MUTED));
  assert.equal(m.accept(await wire(S_MUTED)), true, "a redundant state is still a state");
  assertAvailable(m.view(), MUTED, "a second muted echo");
  assert.deepEqual(m.click(), { type: "astra.mic.mute", muted: false });
});

pendingT0181B("T-0181 B1: the control always reflects the last echo, in any order of the three states", async () => {
  const m = await mic();
  const seq = [S_LIVE, S_MUTED, s_gone(REASONS[1]), S_MUTED, S_LIVE, s_gone(REASONS[3]), S_LIVE];
  for (const raw of seq) {
    m.accept(await wire(raw));
    const v = m.view();
    if (!raw.available) {
      assert.equal(v.text, NO_MIC, `after ${JSON.stringify(raw)}`);
      assert.equal(v.disabled, true);
      assert.equal(v.title, raw.reason);
      assert.equal(v.badge, null);
      assert.equal(m.click(), null);
    } else if (raw.muted) {
      assertAvailable(v, MUTED, `after ${JSON.stringify(raw)}`);
      assert.deepEqual(v.badge, BADGE);
      assert.deepEqual(m.click(), { type: "astra.mic.mute", muted: false });
    } else {
      assertAvailable(v, LIVE, `after ${JSON.stringify(raw)}`);
      assert.equal(v.badge, null, "B2: the badge is hidden once the echo is not muted");
      assert.deepEqual(m.click(), { type: "astra.mic.mute", muted: true });
    }
  }
});

pendingT0181B("T-0181 B1 (A9, H-T181-3): a live astra.session.cleared leaves a muted control muted", async () => {
  const m = await mic();
  m.accept(await wire(S_MUTED));
  for (const raw of [
    { type: "astra.session.cleared", resets: 1, ended_turn_id: "t3", memory_forgotten: 0 },
    { type: "astra.session.cleared", resets: 1, ended_turn_id: null, memory_forgotten: 0, replay: true },
  ]) {
    m.accept(await wire(raw));
    assertAvailable(m.view(), MUTED, `after ${JSON.stringify(raw)}: the mute survives Restart and is re-sent only on a hello`);
    assert.deepEqual(m.view().badge, BADGE);
  }
});

pendingT0181B("T-0181 B1: accept() never throws on anything the page may hand it", async () => {
  const m = await mic();
  for (const junk of [undefined, null, 0, "astra.mic.state", {}, { type: 7 }, []]) {
    assert.doesNotThrow(() => m.accept(junk), `accept(${JSON.stringify(junk)}) must not throw`);
  }
  assert.equal(m.view(), null, "junk shows no control");
});

// --- B3: no keyboard shortcut (H-T181-4) -------------------------------------------------------------
//
// The keydown decision table, as B3 rules it, is one line: every key, on every target, sends nothing and
// stops nothing. Node cannot run main.js's listeners (that is the Chrome module's `m` everywhere row);
// these rows fence the source: no key handling in mic.js, and no single-character key comparison
// anywhere in hud/src (comments stripped; strings are not, so a string like '"m"' in code trips them).

pendingT0181B("T-0181 B3: mic.js has no keyboard entry point (no key event, no event.key)", async () => {
  const path = "hud/src/mic.js";
  assert.ok(existsSync(fileURLToPath(new URL(path, REPO))), `B: ${path} exists (Owned files: new, DOM-free)`);
  const code = codeOf(path);
  assert.ok(!/\bkey(down|up|press)\b/i.test(code), "B3: mic.js handles no key event; only the button toggles the mic");
  assert.ok(!/\.key\b|\bkeyCode\b|\.code\b/.test(code), "B3: mic.js reads no event.key / keyCode / code");
  const mod = await load("mic.js");
  const named = Object.keys(mod).filter((k) => /key|shortcut/i.test(k));
  assert.deepEqual(named, [], "B3: mic.js exports nothing key- or shortcut-shaped");
});

pendingT0181B("T-0181 B3: no hud/src module compares a key with a single letter or digit (no single-key shortcut)", async () => {
  const { readdirSync, statSync } = await import("node:fs");
  const hits = [];
  const visit = (dir, rel) => {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      if (statSync(full).isDirectory()) visit(full, rel + entry + "/");
      else if (/\.(js|mjs)$/.test(entry)) {
        const code = codeOf("hud/src/" + rel + entry);
        for (const m of code.matchAll(/\.key\s*[!=]==?\s*["'`]([A-Za-z0-9])["'`]|["'`]([A-Za-z0-9])["'`]\s*[!=]==?\s*[A-Za-z_$.]*\.key\b|\.code\s*[!=]==?\s*["'`]Key[A-Z]["'`]/g)) {
          hits.push(`${rel}${entry}: ${m[0]}`);
        }
      }
    }
  };
  visit(fileURLToPath(new URL("hud/src", REPO)), "");
  assert.deepEqual(
    hits,
    [],
    "B3 (H-T181-4, James 2026-10-28: 'remove it'): the page registers no single-key shortcut — neither " +
      "`m` for Stop sharing nor any key for the mic. Comparisons found:\n  " + hits.join("\n  "),
  );
});

// --- the transcript bubble of an abandoned question (A0-S4, H-T181-10, Group B gains transcript.js) ---

/** The smallest element renderBubble writes: className, textContent, hidden, attributes, style. */
function bubbleDouble() {
  const attrs = new Map();
  return {
    className: "bubble",
    textContent: "",
    hidden: false,
    style: {},
    classList: {
      _set: new Set(),
      add(...c) { for (const x of c) this._set.add(x); },
      remove(...c) { for (const x of c) this._set.delete(x); },
      toggle(c, on) { const want = on === undefined ? !this._set.has(c) : Boolean(on); if (want) this._set.add(c); else this._set.delete(c); return want; },
      contains(c) { return this._set.has(c); },
    },
    setAttribute(k, v) { attrs.set(k, String(v)); if (k === "hidden") this.hidden = true; },
    getAttribute(k) { return attrs.has(k) ? attrs.get(k) : null; },
    hasAttribute(k) { return attrs.has(k); },
    removeAttribute(k) { attrs.delete(k); if (k === "hidden") this.hidden = false; },
    toggleAttribute(k, on) { const want = on === undefined ? !attrs.has(k) : Boolean(on); if (want) this.setAttribute(k, ""); else this.removeAttribute(k); return want; },
  };
}

const shown = (el) => el.hidden !== true && !el.hasAttribute("hidden");

pendingT0181B("T-0181 B (A0-S4): an empty final line renders no visible bubble; a worded one still shows", async () => {
  const { renderBubble } = await load("panels/transcript.js");
  // The wire order of an abandoned question: partials with the words heard, then one empty final.
  const el = bubbleDouble();
  renderBubble(el, { turn_id: "t7", text: "what is the", final: false });
  assert.ok(shown(el), "premise: a partial with words shows");
  assert.equal(el.textContent, "what is the");
  renderBubble(el, { turn_id: "t7", text: "", final: true });
  assert.ok(!shown(el), "A0-S4/H-T181-10: the abandoned question's empty final leaves no visible bubble (hidden)");
  assert.equal(el.textContent, "", "…and none of the dropped words stays in it");
  // Too wide: a worded final and a worded partial are untouched.
  const ok = bubbleDouble();
  renderBubble(ok, { turn_id: "t8", text: "how many eighths", final: true });
  assert.ok(shown(ok), "a worded final still shows (a guard that hides every final is too wide)");
  assert.equal(ok.textContent, "how many eighths");
  const fresh = bubbleDouble();
  renderBubble(fresh, { turn_id: "t9", text: "", final: true });
  assert.ok(!shown(fresh), "an empty final as the first line also shows nothing");
});
