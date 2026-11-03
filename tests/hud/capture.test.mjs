// T-0008 criterion 6 (mute is at the pipeline, not the consumer) and criterion 7 (synthetic
// capture is labelled), through the injected `getStream` seam of hud/src/capture.js (ruling R6/R12).
//
// Nothing here touches getDisplayMedia/getUserMedia: criterion 19 says no automated test in this
// task may claim the live path, and none does. What is tested is the code that runs *after* a
// stream arrives, with a fake stream whose tracks report whether stop() was really called.
//
// API this pins:
//   createCapture({ getStream, doc, log }) -> { start, stop, grab, ready, active, captureSize,
//                                               synthetic, source }
//   getStream() -> MediaStream | { stream, synthetic?, source?, onStop? }
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

// node 18 treats a bare .js file as CommonJS unless a package.json in scope says
// {"type":"module"}; the browser decides by MIME type. The sources are copied once into a temp
// directory that carries {"type":"module"} so the very same files can be imported here.
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
        `at module top level (docs/tasks/T-0008.md implementation notes). node could not import ` +
        `it: ${err.message}`,
    );
  }
}

// --- a fake stream and a fake document --------------------------------------------------------

/** A MediaStream whose tracks say, honestly, whether stop() was called on them. */
function fakeStream({ width = 640, height = 400, tracks = 2, deaf = false } = {}) {
  const made = [];
  for (let i = 0; i < tracks; i++) {
    made.push({
      kind: "video",
      id: `v${i}`,
      enabled: true,
      readyState: "live",
      stops: 0,
      stop() {
        this.stops += 1;
        // A real MediaStreamTrack goes to "ended" and the browser's sharing indicator disappears.
        if (!deaf) this.readyState = "ended";
      },
    });
  }
  return {
    __size: [width, height],
    audioCalls: 0,
    getTracks: () => made,
    getVideoTracks: () => made,
    getAudioTracks() {
      this.audioCalls += 1; // criterion 13: the page never holds an audio track
      return [];
    },
    addEventListener() {},
  };
}

/** Just enough document for capture.js: a <video> that reports the stream's size, and canvases. */
function fakeDoc() {
  const created = { video: [], canvas: [] };
  const doc = {
    created,
    createElement(tag) {
      if (tag === "video") {
        const video = {
          muted: false,
          playsInline: false,
          videoWidth: 0,
          videoHeight: 0,
          _srcObject: null,
          get srcObject() {
            return this._srcObject;
          },
          set srcObject(stream) {
            this._srcObject = stream;
            const [w, h] = stream && stream.__size ? stream.__size : [0, 0];
            this.videoWidth = w;
            this.videoHeight = h;
          },
          play: () => Promise.resolve(),
        };
        created.video.push(video);
        return video;
      }
      const canvas = {
        width: 0,
        height: 0,
        getContext: () => ({
          drawImage() {},
          fillRect() {},
          fillText() {},
          getImageData: (_x, _y, w, h) => ({
            // a left-to-right ramp, so the dHash is well defined and not all zeros
            data: Uint8ClampedArray.from(
              { length: w * h * 4 },
              (_v, i) => (Math.floor(i / 4) % w) * 24,
            ),
          }),
        }),
        toDataURL: () => "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
      };
      created.canvas.push(canvas);
      return canvas;
    },
  };
  return doc;
}

/** What "muted" has to mean (criterion 6). Used both ways round, so it cannot be vacuous. */
function assertMuted(stream, video) {
  for (const track of stream.getTracks()) {
    assert.equal(
      track.readyState,
      "ended",
      `criterion 6: mute calls stop() on every track, so the browser's capture session and its ` +
        `sharing indicator end; track ${track.id} is still ${track.readyState}`,
    );
    assert.equal(track.stops, 1, `criterion 6: track ${track.id} was stopped exactly once`);
    assert.ok(
      track.enabled,
      "criterion 6 negative: track.enabled = false leaves the capture session alive and is not " +
        "mute (.claude/rules/perception-memory.md: stop capture at the pipeline level)",
    );
  }
  assert.equal(video.srcObject, null, "criterion 6: the stream is released, not just stopped");
}

// --- criterion 6 --------------------------------------------------------------------------------

test("criterion 6: mute stops every track and releases the stream", async () => {
  const { createCapture } = await load("capture.js", "criterion 6");
  const stream = fakeStream({ tracks: 3 });
  const doc = fakeDoc();
  const capture = createCapture({ getStream: async () => ({ stream, source: "screen" }), doc });
  await capture.start();
  const video = doc.created.video[0];
  assert.ok(capture.active() && capture.ready(), "premise: the capture is running");
  capture.stop();
  assertMuted(stream, video);
  assert.equal(capture.active(), false);
  assert.equal(capture.ready(), false);
  assert.equal(stream.audioCalls, 0, "criterion 13: the page never asks for an audio track");
});

test("criterion 6 adversarial: a stream still alive after mute fails this suite", async () => {
  // The wrong implementation the criterion names: the consumer stops looking, the capture session
  // (and the browser's red indicator) stays up. `deaf` tracks accept stop() and ignore it, which
  // is what that looks like from here. If assertMuted let this pass it would be worthless.
  const stream = fakeStream({ tracks: 2, deaf: true });
  const video = { srcObject: null };
  assert.throws(
    () => assertMuted(stream, video),
    /criterion 6: mute calls stop\(\)/,
    "assertMuted must reject a stream that is still live after mute",
  );
});

test("criterion 6: after mute the offer loop produces nothing", async () => {
  const { createCapture } = await load("capture.js", "criterion 6");
  const doc = fakeDoc();
  const capture = createCapture({ getStream: async () => ({ stream: fakeStream() }), doc });
  await capture.start();
  // main.js's loop guard, verbatim in spirit: offer only while the capture is ready.
  let offers = 0;
  const tick = () => {
    if (capture.ready()) offers += 1;
  };
  for (let i = 0; i < 5; i++) tick();
  assert.equal(offers, 5, "premise: the loop offers while the capture is live");
  capture.stop();
  for (let i = 0; i < 5; i++) tick();
  assert.equal(offers, 5, "criterion 6: afterwards no astra.frame is produced for any source");
  assert.equal(capture.grab(), null, "criterion 6: grab() has nothing to give after mute");
});

test("criterion 6: unmuting needs a fresh acquisition, and start() does not double-acquire", async () => {
  const { createCapture } = await load("capture.js", "criterion 6");
  const streams = [];
  const getStream = async () => {
    const stream = fakeStream();
    streams.push(stream);
    return { stream };
  };
  const doc = fakeDoc();
  const capture = createCapture({ getStream, doc });
  await capture.start();
  await capture.start(); // already live: one click, one permission dialog, one stream
  assert.equal(streams.length, 1, "criterion 6: start() while live does not acquire again");
  capture.stop();
  await capture.start();
  assert.equal(
    streams.length,
    2,
    "criterion 6: unmuting requires a fresh acquisition call (the browser asks again)",
  );
  assert.equal(streams[0].getTracks()[0].readyState, "ended", "the muted stream stays dead");
  assert.equal(streams[1].getTracks()[0].readyState, "live");
});

test("criterion 6: stop() before start() and twice in a row are both no-ops", async () => {
  const { createCapture } = await load("capture.js", "criterion 6");
  const stream = fakeStream();
  const doc = fakeDoc();
  const capture = createCapture({ getStream: async () => ({ stream }), doc });
  assert.doesNotThrow(() => capture.stop(), "mute with nothing captured must not throw");
  await capture.start();
  capture.stop();
  assert.doesNotThrow(() => capture.stop());
  for (const track of stream.getTracks()) {
    assert.equal(track.stops, 1, "a second mute does not stop an already dead track again");
  }
});

// --- criterion 7 --------------------------------------------------------------------------------

test("criterion 7: a stream that does not declare a real provider is synthetic", async () => {
  const { createCapture } = await load("capture.js", "criterion 7");
  const cases = [
    [async () => fakeStream(), true, "a bare stream of unknown provenance"],
    [async () => ({ stream: fakeStream() }), true, "no synthetic flag at all"],
    [async () => ({ stream: fakeStream(), synthetic: true }), true, "declared synthetic"],
    [
      async () => ({ stream: fakeStream(), synthetic: false, source: "screen" }),
      false,
      "the real display provider",
    ],
  ];
  for (const [getStream, expected, why] of cases) {
    const capture = createCapture({ getStream, doc: fakeDoc() });
    await capture.start();
    assert.equal(
      capture.synthetic,
      expected,
      `criterion 7: ${why} -> synthetic=${expected}; unknown provenance is labelled, never ` +
        "rendered as the real thing (docs/01 D15)",
    );
  }
});

// --- criterion 5 through the same seam ------------------------------------------------------------

test("criterion 5: grab() applies the resolution policy and returns a 16-hex dHash", async () => {
  const { createCapture } = await load("capture.js", "criterion 5");
  const doc = fakeDoc();
  const capture = createCapture({ getStream: async () => ({ stream: fakeStream() }), doc });
  assert.equal(capture.grab(), null, "nothing to grab before start()");
  await capture.start();
  const frame = capture.grab();
  assert.deepEqual(
    [frame.width, frame.height],
    [640, 400],
    "criterion 5: 640x400 is inside the screen_max bucket and is never upscaled",
  );
  assert.deepEqual([frame.captureWidth, frame.captureHeight], [640, 400]);
  assert.match(frame.phash, /^[0-9a-f]{16}$/, "criterion 5: 16-character lowercase-hex dHash");
  assert.ok(frame.jpeg_b64.startsWith("/9j/"), "criterion 5: the base64 of a JPEG, header stripped");
  assert.ok(!frame.jpeg_b64.includes("data:"), "the data-URL prefix never reaches the wire");
});

test("criterion 5: a 4K capture is downscaled to the screen_max bucket", async () => {
  const { createCapture } = await load("capture.js", "criterion 5");
  const doc = fakeDoc();
  const capture = createCapture({
    getStream: async () => ({ stream: fakeStream({ width: 3840, height: 2160 }) }),
    doc,
  });
  await capture.start();
  const frame = capture.grab();
  // policy.target_size(3840, 2160, screen_max) == (1568, 882), computed by hand and confirmed
  // against local_astra/perception/policy.py.
  assert.deepEqual([frame.width, frame.height], [1568, 882]);
});
