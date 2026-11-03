// T-0177 group C — the vision selector in the HUD (docs/tasks/T-0177.md C1-C5), the node half.
//
// Every row is derived from the acceptance criteria and rulings H-T177-1 / H-T177-2, never from an
// implementation (hud/src/vision.js does not exist when this file is written). The Chrome half (the
// combo in the What I see pane, the Share click, Mute, the red-capture/blue-frame adversarial, both
// themes, no egress) is tests/integration/test_t0177_hud_vision.py.
//
// Gate. The rows run through `pendingT0177C`, the Status-phrase strict-xfail gate of
// tests/hud/tools.test.mjs, reading docs/tasks/T-0177.md. While its **Status:** line does not record
// "group C: tests <a>..<b>, activated <sha>", each row must FAIL (an XPASS fails the row: a stub that
// satisfies it is vacuous). ASTRA_RUN_PENDING=1 runs them for real now. Group A's or B's activation
// does not switch these rows on.
//
// API these rows pin (the task names hud/src/vision.js "DOM-free" and leaves the JS names to the
// test-author; these are them):
//
//   hud/src/vision.js — DOM-free, importable in node
//     createVision() -> v
//       v.accept(msg)       every decoded astra.* message, in wire order. It reads
//                           astra.vision.state, astra.vision.frame and astra.session.cleared and
//                           ignores everything else. Never throws.
//       v.live()            true once an astra.vision.state has arrived (that is what "live" means
//                           to the page: the session only echoes a choice in live, A3/A4).
//       v.combo(capture)    null until a state has arrived (C1: never shown outside live); then
//                           { value, label, options: [{ value, label, disabled }] } where
//                             value   = the last echoed source (C1: "always the last echo")
//                             label   = "Looking at"
//                             options = browser "Gearvis's browser", then shared "Shared screen"
//                                       ("Shared camera" for a camera capture); shared is disabled
//                                       unless a capture is running; browser never is.
//       v.choose(source)    the user changed the combo: returns the message to send,
//                           { type: "astra.vision.select", source }, or null before any state.
//                           It never moves combo().value (the echo does).
//       v.captureStarted(origin)
//                           origin "share" (a click on Share) | "url" (?source=canvas|synthetic at
//                           load). Returns { type: "astra.vision.select", source: "shared" } for a
//                           Share in live, else null (C2, H-T177-1).
//       v.captureStopped()  Returns { type: "astra.vision.select", source: "browser" } iff live and
//                           the last echo is "shared", else null (C2).
//       v.panel(capture)    what the What I see pane shows (C3, C4):
//                           { large, thumbs, caption, empty, live }
//                             large   the newest image of the newest astra.vision.frame, or null
//                             thumbs  the earlier images of that frame, in order
//                             each image = { src, width, height, kind, identity }, src the data URL
//                               "data:image/jpeg;base64," + jpeg_b64 (the session's bytes, verbatim)
//                             caption "<source name> · <W>x<H> · turn <id> step <k>" (+ " · +N earlier
//                               images" when history_images > 0); "" with no frame
//                             empty   the pane's empty sentence, or null when an image is shown:
//                               a blind frame -> "No image was sent for this question"
//                               no frame, live -> "Nothing sent yet: Gearvis looks when you ask"
//                               no frame, not live -> null (the page keeps T-0154's HTML copy)
//                             live    the `.live` badge: true iff the newest image is a shared one
//                               and a capture is running (C4)
//     capture = null | { running: boolean, source: "screen" | "camera" }
//
//   hud/src/capture.js — createCapture({ getStream, doc, log, onEnded }): onEnded() is called when a
//     video track of the running stream fires `ended` (the browser's own "Stop sharing", Owned files:
//     "the track ended event reported to the caller only").
//
// Readings the task leaves open (stated, not guessed silently):
//   - "the newest large, the earlier ones as thumbnails" is read within one astra.vision.frame: its
//     images are "the final user message's ImageParts, in order" (B7), so the newest is the last.
//     Whether earlier *frames* also stay as thumbnails is not pinned.
//   - "+N earlier images" is pinned for N = 3; N = 1 may read "image" or "images".
//   - A blind frame's caption is only pinned not to describe the previous image.
//   - The separator is " · " (U+00B7 between spaces), as written in C4.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);

// --- the T-0177 group C strict-xfail gate --------------------------------------------------------

const TASK = fileURLToPath(new URL("docs/tasks/T-0177.md", REPO));
const STATUS = existsSync(TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(TASK, "utf8")) || [])[1] || "").trim()
  : "";
/** "group C: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6" — the phrase the orchestrator writes. */
const C_ACTIVATED = /\bgroup C:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const C_PENDING = !C_ACTIVATED.test(STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0177C(name, fn) {
  test(name + (C_PENDING ? " [pending T-0177 C: strict xfail until group C is activated]" : ""), async (t) => {
    if (!C_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0177 group C row passed while docs/tasks/T-0177.md's status does " +
          "not record group C's activation (" + JSON.stringify(STATUS.slice(0, 80)) + "). Either " +
          "group C has landed (record 'group C: tests …, activated <sha>' in the status) or the " +
          "row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0177 C pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0177 C gate: the status line is readable and only group C's own phrase activates the rows", () => {
  assert.ok(STATUS.length > 0, "docs/tasks/T-0177.md must carry a **Status:** line");
  assert.equal(C_ACTIVATED.test("in progress 2026-10-28. Group A merged at ab093fef. Groups B, C, D planned."), false);
  assert.equal(C_ACTIVATED.test("in progress (group A: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6)"), false);
  assert.equal(C_ACTIVATED.test("in progress (group B: tests 1a2b3c4, activated 9a8b7c6; group C: tests planned)"), false);
  assert.equal(C_ACTIVATED.test("in progress (group C: tests a1b2c3d..e4f5a6b, activated c3d4e5f)"), true);
  assert.equal(C_ACTIVATED.test("in progress. group C: tests a1b2c3d, activated c3d4e5f; group D planned"), true);
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

// Origin proof (.claude/rules/tests.md): print where the modules under test were imported from.
console.log(`# vision origin: repo=${fileURLToPath(REPO)} modules=${srcRoot().href}`);

async function load(name) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch (err) {
    assert.fail(
      `T-0177 C adds hud/src/${name}, importable in node as an ES module with no DOM at module ` +
        `top level (Owned files: "hud/src/vision.js — group C, new, DOM-free"). node could not ` +
        `import it: ${err.message}`,
    );
  }
}

async function vision() {
  const mod = await load("vision.js");
  assert.equal(typeof mod.createVision, "function", "C: hud/src/vision.js exports createVision()");
  const v = mod.createVision();
  for (const name of ["accept", "live", "combo", "choose", "captureStarted", "captureStopped", "panel"])
    assert.equal(typeof v[name], "function", `C: createVision() returns an object with ${name}()`);
  return v;
}

// --- the wire, as the session sends it (contracts/protocol.py, T-0177 A1) --------------------------

// Distinct byte strings per image, so "which image is shown" is an equality on bytes, not a count.
const B64 = {
  browser1: Buffer.from("browser-frame-1 ÿØ").toString("base64"),
  browser2: Buffer.from("browser-frame-2 ÿØ").toString("base64"),
  shared1: Buffer.from("shared-frame-1 ÿØ").toString("base64"),
  overview: Buffer.from("overview-frame ÿØ").toString("base64"),
  camera: Buffer.from("camera-frame ÿØ").toString("base64"),
};
const url = (b64) => "data:image/jpeg;base64," + b64;

const state = (source, replay = false) => ({
  type: "astra.vision.state",
  source,
  choices: ["browser", "shared"],
  replay,
});

const img = (b64, { identity, kind = "window", width = 1280, height = 800 } = {}) => ({
  identity: identity || "id-" + b64.slice(0, 8),
  kind,
  width,
  height,
  jpeg_b64: b64,
});

const frame = ({ turn_id = "t1", step = 1, source = "browser", images = [], blind = false, history_images = 0, replay = false } = {}) => ({
  type: "astra.vision.frame",
  turn_id,
  step,
  source,
  blind,
  images,
  history_images,
  replay,
});

const cleared = (resets, replay = false) => ({
  type: "astra.session.cleared",
  resets,
  ended_turn_id: null,
  memory_forgotten: 0,
  replay,
});

const RUNNING = { running: true, source: "screen" };
const CAMERA = { running: true, source: "camera" };
const NONE = null;

/** Every message the rows feed must be one the page's own decoder accepts (A1), so a row never
 *  passes on a message the session could not send. */
async function decodes(msg) {
  const { decode } = await load("protocol.js");
  const got = decode(JSON.stringify(msg));
  assert.equal(got.kind, "astra", `premise: ${msg.type} decodes through hud/src/protocol.js: ${JSON.stringify(got)}`);
}

async function feed(v, ...msgs) {
  for (const m of msgs) {
    await decodes(m);
    v.accept(m);
  }
}

/** A select the page sends: exactly {type, source}, and accepted by the page's own decoder. */
async function assertSelect(msg, source, why) {
  assert.deepEqual(msg, { type: "astra.vision.select", source }, why);
  await decodes(msg);
}

const norm = (s) => String(s ?? "").replace(/[‘’]/g, "'");

// --- C1: the combo box, from the echo only ---------------------------------------------------------

pendingT0177C("T-0177 C1: no combo before an astra.vision.state, whatever else arrives", async () => {
  const v = await vision();
  assert.equal(v.live(), false, "C1: before any state the page is not in live");
  assert.equal(v.combo(RUNNING), null, "C1: the combo is rendered only after an astra.vision.state arrives");
  // A frame arrives in every mode (B7), so a frame alone must not make the combo appear.
  await feed(v, frame({ images: [img(B64.browser1)] }), cleared(1));
  assert.equal(v.live(), false, "C1: an astra.vision.frame is not a state; non-live sessions publish frames too");
  assert.equal(v.combo(RUNNING), null, "C1: still no combo after a frame and a reset with no state");
});

pendingT0177C("T-0177 C1: the combo appears on the first state, labelled 'Looking at', with the two options", async () => {
  const v = await vision();
  await feed(v, state("browser"));
  assert.equal(v.live(), true, "C1: a state makes the page live");
  const c = v.combo(RUNNING);
  assert.ok(c, "C1: the combo is rendered once a state has arrived");
  assert.equal(c.label, "Looking at", "C1: the combo is labelled 'Looking at'");
  assert.equal(c.value, "browser", "C1: its value is the echoed source");
  assert.deepEqual(
    c.options.map((o) => [o.value, norm(o.label)]),
    [
      ["browser", "Gearvis's browser"],
      ["shared", "Shared screen"],
    ],
    "C1: options browser 'Gearvis's browser' and shared 'Shared screen', in that order",
  );
});

pendingT0177C("T-0177 C1: the shared option reads 'Shared camera' for a camera capture", async () => {
  const v = await vision();
  await feed(v, state("browser"));
  const shared = v.combo(CAMERA).options.find((o) => o.value === "shared");
  assert.equal(norm(shared.label), "Shared camera", "C1: 'Shared camera' for a camera capture");
  const screen = v.combo(RUNNING).options.find((o) => o.value === "shared");
  assert.equal(norm(screen.label), "Shared screen", "C1: and 'Shared screen' for a screen capture");
});

pendingT0177C("T-0177 C1: shared is enabled exactly while a capture runs; browser never disabled", async () => {
  const v = await vision();
  await feed(v, state("browser"));
  const opt = (cap, value) => v.combo(cap).options.find((o) => o.value === value);
  assert.equal(opt(NONE, "shared").disabled, true, "C1: shared is disabled with no capture");
  assert.equal(opt({ running: false, source: "screen" }, "shared").disabled, true, "C1: …and with a stopped one");
  assert.equal(opt(RUNNING, "shared").disabled, false, "C1: shared is enabled while a capture runs");
  assert.equal(opt(CAMERA, "shared").disabled, false, "C1: …a camera capture too");
  for (const cap of [NONE, RUNNING]) assert.equal(opt(cap, "browser").disabled, false, "C1: browser is always selectable");
});

pendingT0177C("T-0177 C1: the value is always the last echo, never the click", async () => {
  const v = await vision();
  await feed(v, state("browser"));
  const sent = v.choose("shared");
  await assertSelect(sent, "shared", "C1: changing the combo sends astra.vision.select {source: shared}");
  assert.equal(v.combo(RUNNING).value, "browser", "C1: the click does not move the value; the echo does (A3)");
  await feed(v, state("shared"));
  assert.equal(v.combo(RUNNING).value, "shared", "C1: the echo moves it");
  // An echo the page did not ask for (another page's click, A3 echoes to every connection) moves it too.
  await feed(v, state("browser"));
  assert.equal(v.combo(RUNNING).value, "browser", "C1: every echo moves it, asked for or not");
  // A replayed state (the hello) is an echo as well.
  await feed(v, state("shared", true));
  assert.equal(v.combo(RUNNING).value, "shared", "C1: a replayed state is the last echo too");
});

pendingT0177C("T-0177 C1/C5: choose() sends nothing before a state", async () => {
  const v = await vision();
  assert.equal(v.choose("shared"), null, "C5: no astra.vision.select before the session echoed a choice");
  assert.equal(v.choose("browser"), null, "C5: …for either source");
});

// --- C2: auto-switch on Share (H-T177-1) ------------------------------------------------------------

pendingT0177C("T-0177 C2: a Share click in live sends select shared", async () => {
  const v = await vision();
  await feed(v, state("browser"));
  await assertSelect(v.captureStarted("share"), "shared", "C2 (H-T177-1): a user-initiated Share in live selects shared");
});

pendingT0177C("T-0177 C2 adversarial: a capture started by the URL never auto-switches, live or not", async () => {
  const v = await vision();
  assert.equal(v.captureStarted("url"), null, "C2: ?source=canvas|synthetic at load sends nothing (not live)");
  await feed(v, state("browser"));
  assert.equal(
    v.captureStarted("url"),
    null,
    "C2: ?source=canvas|synthetic never auto-switches, even in live (the existing live HUD rigs keep the browser source)",
  );
});

pendingT0177C("T-0177 C2/C5: a Share outside live sends nothing", async () => {
  const v = await vision();
  await feed(v, frame({ images: [img(B64.browser1)] }));
  assert.equal(v.captureStarted("share"), null, "C2/C5: no state, no live session: Share sends no select");
});

pendingT0177C("T-0177 C2: a stopped capture selects browser only while shared is the echoed choice", async () => {
  const v = await vision();
  assert.equal(v.captureStopped(), null, "C2: not live: nothing");
  await feed(v, state("browser"));
  assert.equal(v.captureStopped(), null, "C2 (too wide): browser echoed: stopping sends nothing");
  // The page asked for shared but no echo came yet: the echoed choice is still browser.
  v.choose("shared");
  assert.equal(v.captureStopped(), null, "C2: the echoed choice decides, not the click");
  await feed(v, state("shared"));
  await assertSelect(v.captureStopped(), "browser", "C2 (absent): shared echoed: stopping selects browser");
});

// --- C3: the panel shows the session's frame (H-T177-2) --------------------------------------------

pendingT0177C("T-0177 C3: with no frame, live says so in words and shows nothing", async () => {
  const v = await vision();
  const before = v.panel(RUNNING);
  assert.equal(before.large, null, "C3: never fed by the local capture: no image before a frame, capture or not");
  assert.deepEqual(before.thumbs, [], "C3: no thumbnails");
  assert.equal(before.caption, "", "C4: no frame, empty caption");
  assert.equal(before.empty, null, "C3: outside live the page keeps T-0154's existing empty text (null here)");
  await feed(v, state("browser"));
  const live = v.panel(RUNNING);
  assert.equal(live.large, null, "C3: live, no frame: no image although a capture runs");
  assert.equal(live.empty, "Nothing sent yet: Gearvis looks when you ask", "C3: live's empty sentence");
  assert.equal(live.caption, "", "C4: no frame, empty caption");
});

pendingT0177C("T-0177 C3: the newest frame's newest image is large, its earlier images thumbnails, bytes verbatim", async () => {
  const v = await vision();
  await feed(
    v,
    state("browser"),
    frame({
      turn_id: "t3",
      step: 1,
      images: [img(B64.overview, { kind: "overview", width: 1280, height: 2400 }), img(B64.browser1, { width: 1280, height: 800 })],
    }),
  );
  const p = v.panel(RUNNING);
  assert.ok(p.large, "C3: a frame with images shows one large");
  assert.equal(p.large.src, url(B64.browser1), "C3: the newest (last, B7 order) image is large, the session's bytes verbatim");
  assert.deepEqual(p.thumbs.map((t) => t.src), [url(B64.overview)], "C3: the earlier image is a thumbnail");
  assert.equal(p.empty, null, "C3: no empty sentence while an image is shown");
  assert.equal(p.large.width, 1280);
  assert.equal(p.large.height, 800);

  await feed(v, frame({ turn_id: "t3", step: 2, images: [img(B64.browser2)] }));
  const q = v.panel(RUNNING);
  assert.equal(q.large.src, url(B64.browser2), "C3: a newer frame replaces the large image");
});

pendingT0177C("T-0177 C3: a frame shows in non-live too (frames are published in all modes)", async () => {
  const v = await vision();
  await feed(v, frame({ images: [img(B64.browser1)] }));
  assert.equal(v.panel(NONE).large && v.panel(NONE).large.src, url(B64.browser1), "C3: the panel shows astra.vision.frame in every mode");
});

pendingT0177C("T-0177 C3: a blind frame clears the image and says no image was sent", async () => {
  const v = await vision();
  await feed(v, state("shared"), frame({ turn_id: "t1", source: "shared", images: [img(B64.shared1, { kind: "screen", width: 1568, height: 590 })] }));
  assert.ok(v.panel(RUNNING).large, "premise: an image is shown");
  await feed(v, frame({ turn_id: "t2", source: "shared", blind: true, images: [] }));
  const p = v.panel(RUNNING);
  assert.equal(p.large, null, "C3: a blind frame shows no image (the previous one is not kept)");
  assert.deepEqual(p.thumbs, [], "C3: …and no thumbnails");
  assert.equal(p.empty, "No image was sent for this question", "C3: the blind sentence");
  assert.ok(!p.caption.includes("1568x590"), `C4: the caption does not describe the image no longer shown: ${JSON.stringify(p.caption)}`);
  assert.equal(p.live, false, "C4: no image shown, no Live badge");
});

pendingT0177C("T-0177 C3: a reset clears the panel; the choice survives it", async () => {
  const v = await vision();
  await feed(v, state("shared"), frame({ source: "shared", images: [img(B64.shared1, { kind: "screen" })] }));
  assert.ok(v.panel(RUNNING).large, "premise: an image is shown");
  await feed(v, cleared(1));
  const p = v.panel(RUNNING);
  assert.equal(p.large, null, "C3: a reset clears the panel");
  assert.deepEqual(p.thumbs, []);
  assert.equal(p.caption, "", "C4: no frame after the reset, empty caption");
  assert.equal(p.empty, "Nothing sent yet: Gearvis looks when you ask", "C3: back to live's empty sentence");
  assert.equal(v.combo(RUNNING).value, "shared", "B9/T-0162 D-3: the choice survives a reset; the combo keeps the last echo");
});

pendingT0177C("T-0177 C3: in the hello's wire order (cleared, state, frame replay) the replayed frame is shown", async () => {
  // .claude/rules/tests.md: a clear must be tested in the order the server publishes. B9: the hello
  // reply carries the reset replay first and the latest frame since the reset last, after A4's state.
  const v = await vision();
  await feed(
    v,
    cleared(2, true),
    state("browser", true),
    frame({ turn_id: "t9", step: 2, images: [img(B64.browser2)], replay: true }),
  );
  assert.equal(v.panel(RUNNING).large && v.panel(RUNNING).large.src, url(B64.browser2), "B9 order: the frame after the replayed reset stays");
  // A live reset later in the same session clears it, and the next frame after it shows.
  await feed(v, cleared(3));
  assert.equal(v.panel(RUNNING).large, null, "C3: the live reset clears");
  await feed(v, frame({ turn_id: "t10", images: [img(B64.browser1)] }));
  assert.equal(v.panel(RUNNING).large.src, url(B64.browser1), "C3: the first frame after the reset shows");
});

pendingT0177C("T-0177 C3: other messages never touch the panel or the combo, and nothing throws", async () => {
  const v = await vision();
  await feed(v, state("browser"), frame({ images: [img(B64.browser1)] }));
  const others = [
    { type: "astra.turn.state", turn_id: "t2", state: "THINKING" },
    { type: "astra.frame", source: "screen", ts_ms: 1, width: 2, height: 2, jpeg_b64: B64.shared1, phash: "0".repeat(16) },
    { type: "astra.error", code: "vision_source_unavailable", message: "not live" },
    { type: "astra.vision.select", source: "shared" },
    {},
    null,
    "astra.vision.frame",
  ];
  for (const m of others) assert.doesNotThrow(() => v.accept(m), `C3: accept(${JSON.stringify(m)}) never throws`);
  assert.equal(v.panel(RUNNING).large.src, url(B64.browser1), "C3: only astra.vision.frame feeds the panel");
  assert.equal(v.combo(RUNNING).value, "browser", "C1: only astra.vision.state moves the combo");
});

// --- C4: the caption and the Live badge --------------------------------------------------------------

pendingT0177C("T-0177 C4: the caption names the source, the size, the turn and the step", async () => {
  const cases = [
    [frame({ turn_id: "t3", step: 2, source: "browser", images: [img(B64.browser1, { kind: "window", width: 1280, height: 800 })] }),
      "Gearvis's browser · 1280x800 · turn t3 step 2"],
    [frame({ turn_id: "t14", step: 1, source: "shared", images: [img(B64.shared1, { kind: "screen", width: 1568, height: 590 })] }),
      "Shared screen · 1568x590 · turn t14 step 1"],
    [frame({ turn_id: "t5", step: 1, source: "shared", images: [img(B64.shared1, { kind: "window", width: 1200, height: 700 })] }),
      "Shared screen · 1200x700 · turn t5 step 1"],
    [frame({ turn_id: "t6", step: 1, source: "shared", images: [img(B64.camera, { kind: "camera", width: 640, height: 480 })] }),
      "Shared camera · 640x480 · turn t6 step 1"],
    [frame({ turn_id: "t7", step: 1, source: "browser", images: [img(B64.overview, { kind: "overview", width: 1280, height: 2400 })] }),
      "Page overview · 1280x2400 · turn t7 step 1"],
  ];
  for (const [f, want] of cases) {
    const v = await vision();
    await feed(v, state("browser"), f);
    assert.equal(norm(v.panel(RUNNING).caption), want, `C4: the caption of ${JSON.stringify(f).slice(0, 120)}…`);
  }
});

pendingT0177C("T-0177 C4: '+N earlier images' iff history_images > 0", async () => {
  const v = await vision();
  await feed(v, state("browser"), frame({ turn_id: "t3", step: 4, images: [img(B64.browser2)], history_images: 3 }));
  assert.equal(norm(v.panel(RUNNING).caption), "Gearvis's browser · 1280x800 · turn t3 step 4 · +3 earlier images", "C4: with history");
  await feed(v, frame({ turn_id: "t4", step: 1, images: [img(B64.browser1)], history_images: 0 }));
  const c = norm(v.panel(RUNNING).caption);
  assert.equal(c, "Gearvis's browser · 1280x800 · turn t4 step 1", "C4: no suffix at history_images 0");
  await feed(v, frame({ turn_id: "t4", step: 2, images: [img(B64.browser2)], history_images: 1 }));
  assert.match(norm(v.panel(RUNNING).caption), / · \+1 earlier images?$/, "C4: N = 1");
});

pendingT0177C("T-0177 C4: the Live badge shows only while the newest image is shared and a capture runs", async () => {
  const v = await vision();
  await feed(v, state("shared"), frame({ source: "browser", images: [img(B64.browser1)] }));
  assert.equal(v.panel(RUNNING).live, false, "C4 (too wide): a browser image with a capture running: no badge");
  await feed(v, frame({ turn_id: "t2", source: "shared", images: [img(B64.shared1, { kind: "screen" })] }));
  assert.equal(v.panel(RUNNING).live, true, "C4 (absent): a shared image with a capture running: the badge");
  assert.equal(v.panel(NONE).live, false, "C4: a shared image but no capture: no badge");
  assert.equal(v.panel({ running: false, source: "screen" }).live, false, "C4: a stopped capture: no badge");
  // B1: the step images follow the browser after an action; the badge follows the panel, not the choice.
  await feed(v, frame({ turn_id: "t2", step: 2, source: "browser", images: [img(B64.browser2)] }));
  assert.equal(v.panel(RUNNING).live, false, "C4: the newest image is the browser again: no badge, although shared is chosen");
});

pendingT0177C("T-0177 C4: an empty panel has no Live badge, in live with a capture running", async () => {
  const v = await vision();
  await feed(v, state("shared"));
  assert.equal(v.panel(RUNNING).live, false, "C4: nothing shown, no badge (today's badge keys on the capture alone)");
});

// --- capture.js: the track `ended` event, reported to the caller (Owned files) ----------------------

function endableStream() {
  const tracks = [0, 1].map((i) => {
    const t = new EventTarget();
    Object.assign(t, { kind: "video", id: `v${i}`, enabled: true, readyState: "live", onended: null });
    t.stop = function () {
      this.readyState = "ended"; // a real track.stop() does not fire `ended`
    };
    t.fireEnded = function () {
      this.readyState = "ended";
      const ev = new Event("ended");
      this.dispatchEvent(ev);
      if (typeof this.onended === "function") this.onended(ev);
    };
    return t;
  });
  return {
    __size: [640, 400],
    tracks,
    getTracks: () => tracks,
    getVideoTracks: () => tracks,
    getAudioTracks: () => [],
    addEventListener() {},
    removeEventListener() {},
  };
}

function videoDoc() {
  return {
    createElement(tag) {
      if (tag !== "video") return { width: 0, height: 0, getContext: () => ({ drawImage() {} }), toDataURL: () => "data:," };
      return {
        muted: false,
        playsInline: false,
        videoWidth: 0,
        videoHeight: 0,
        _s: null,
        get srcObject() {
          return this._s;
        },
        set srcObject(s) {
          this._s = s;
          [this.videoWidth, this.videoHeight] = s && s.__size ? s.__size : [0, 0];
        },
        play: () => Promise.resolve(),
      };
    },
  };
}

pendingT0177C("T-0177 C2: capture.js reports a track's `ended` (the browser's Stop sharing) to its caller", async () => {
  const { createCapture } = await load("capture.js");
  const stream = endableStream();
  let ended = 0;
  const capture = createCapture({
    getStream: async () => ({ stream, synthetic: false, source: "screen" }),
    doc: videoDoc(),
    onEnded: () => {
      ended += 1;
    },
  });
  await capture.start();
  assert.equal(ended, 0, "premise: nothing ended yet");
  stream.tracks[0].fireEnded();
  assert.ok(ended >= 1, "C2: a video track's `ended` event reaches the caller's onEnded (so the page can select browser)");
});

// --- Group C review rulings (docs/tasks/T-0177.md "## Group C review rulings") ----------------------
//
// C-R1. The too-large image: the server omits the bytes of an image (A1: jpeg_b64 null, omitted
// "too_large"). The pane shows no picture (never a data: URL built from null), no thumbnail with a null
// src, the sentence below, and the caption still describes the image that was sent. C-R2 (one Share,
// one select) and C-R3 (render on change only) are page wiring: the Chrome module pins them.

const TOO_LARGE = "The image sent was too large to show here";

const tooLarge = ({ identity = "id-too-large", kind = "window", width = 1280, height = 800 } = {}) => ({
  identity,
  kind,
  width,
  height,
  jpeg_b64: null,
  omitted: "too_large",
});

pendingT0177C("T-0177 C-R1: a newest image without bytes shows no picture, says so, and keeps the caption", async () => {
  const v = await vision();
  await feed(
    v,
    state("browser"),
    frame({ turn_id: "t4", step: 1, images: [img(B64.browser1)] }),
  );
  assert.equal(v.panel(RUNNING).large.src, url(B64.browser1), "premise: an image is shown before the too-large frame");
  await feed(
    v,
    frame({
      turn_id: "t5",
      step: 2,
      images: [img(B64.overview, { kind: "overview", width: 1280, height: 2400 }), tooLarge({ width: 1568, height: 980 })],
    }),
  );
  const p = v.panel(RUNNING);
  assert.ok(p.large !== null && typeof p.large === "object", "C-R1: the newest image is still the large slot's image (its caption describes it)");
  assert.strictEqual(p.large.src, null, `C-R1: large.src is null, never a data: URL built from null; got ${JSON.stringify(p.large.src)}`);
  assert.deepEqual(
    p.thumbs.map((t) => t.src),
    [url(B64.overview)],
    "C-R1: the earlier image (with bytes) is the one thumbnail; the omitted newest is not demoted to a null thumbnail",
  );
  for (const t of p.thumbs) assert.ok(typeof t.src === "string" && t.src.startsWith("data:image/jpeg;base64,") && !t.src.endsWith("null"), `C-R1: no thumbnail with a null src: ${JSON.stringify(t.src)}`);
  assert.equal(p.empty, TOO_LARGE, "C-R1: the pane says the image sent was too large to show here");
  assert.equal(
    norm(p.caption),
    "Gearvis's browser · 1568x980 · turn t5 step 2",
    "C-R1: the caption still describes the image that was sent (not the previous one, not the overview)",
  );
});

pendingT0177C("T-0177 C-R1: a single too-large image: no picture, no thumbnails, the sentence; the next frame with bytes shows", async () => {
  const v = await vision();
  await feed(v, state("shared"), frame({ turn_id: "t6", step: 1, source: "shared", images: [tooLarge({ kind: "screen", width: 3840, height: 2160 })] }));
  const p = v.panel(RUNNING);
  assert.strictEqual(p.large && p.large.src, null, `C-R1: no picture: ${JSON.stringify(p.large)}`);
  assert.deepEqual(p.thumbs, [], "C-R1: no thumbnails");
  assert.equal(p.empty, TOO_LARGE, "C-R1: the sentence");
  assert.equal(norm(p.caption), "Shared screen · 3840x2160 · turn t6 step 1", "C-R1: the caption describes the image sent");
  await feed(v, frame({ turn_id: "t6", step: 2, source: "shared", images: [img(B64.shared1, { kind: "screen" })] }));
  const q = v.panel(RUNNING);
  assert.equal(q.large.src, url(B64.shared1), "C3: the next frame with bytes is shown");
  assert.equal(q.empty, null, "C-R1: the too-large sentence goes with the image it described");
});
