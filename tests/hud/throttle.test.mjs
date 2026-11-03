// T-0008 criterion 4 (rate and single-flight) and criterion 3 (backpressure), with a fake clock.
// No DOM, no timers, no randomness: 60 fps in, 10 frames out over 10 s of simulated time, and a
// slow encode *drops* frames instead of queueing them.
//
// API this pins (hud/src/frames.js, injected per the implementation notes so criteria 4/7/9 are
// testable without a browser and without a permission dialog):
//
//   createFramePipeline({ now, encode, send, maxFps? }) -> {
//       offer(source),   // called by the capture loop; resolves to "sent" | "dropped" when the
//                        // pipeline is done with that offer. Never queues more than one frame.
//       stats(),         // { sent, dropped, inFlight, pending }
//   }
//   encode(source, hint) -> Promise<{ width, height, jpeg_b64, phash }>
//   send(message)        -> Promise<void>   (resolves when the socket has taken the frame)
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

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

/** Let every queued microtask and the pipeline's own async steps run. */
const flush = () => new Promise((resolve) => setImmediate(resolve));

const ENCODED = {
  width: 1568,
  height: 980,
  jpeg_b64: "/9j/AAAA",
  phash: "0123456789abcdef",
};

function harness({ encode, send } = {}) {
  const h = {
    t: 0,
    sent: [],
    inFlight: 0,
    maxInFlight: 0,
    encodes: 0,
  };
  const wrap = (fn) => async (...args) => {
    h.inFlight += 1;
    h.maxInFlight = Math.max(h.maxInFlight, h.inFlight);
    try {
      return await fn(...args);
    } finally {
      h.inFlight -= 1;
    }
  };
  h.now = () => h.t;
  h.encode = wrap(
    encode ||
      (async () => {
        h.encodes += 1;
        return ENCODED;
      }),
  );
  h.send = wrap(
    send ||
      (async (msg) => {
        h.sent.push(msg);
      }),
  );
  return h;
}

test("criterion 4: 60 fps in for 10 s gives 10 +/- 1 frames out, and nothing is queued", async () => {
  const { createFramePipeline } = await load("frames.js", "criterion 4");
  const h = harness();
  const pipe = createFramePipeline({ now: h.now, encode: h.encode, send: h.send });
  const offers = 600; // 60 fps x 10 s of controlled time
  for (let i = 0; i < offers; i++) {
    h.t = Math.round((i * 1000) / 60);
    await pipe.offer("screen");
  }
  const s = pipe.stats();
  assert.ok(
    s.sent >= 9 && s.sent <= 11,
    `criterion 4: at most one astra.frame per source per second (policy.MAX_FPS = 1.0): ` +
      `expected 10 +/- 1 over 10 s, got ${s.sent}`,
  );
  assert.equal(
    s.sent + s.dropped + (s.pending || 0),
    offers,
    "every offer is either sent or dropped — no third fate, no unbounded queue",
  );
  assert.ok(h.maxInFlight <= 1, `criterion 4: single-flight, saw ${h.maxInFlight} in flight`);
});

test("criterion 4: the emitted message carries the injected clock and the encoder's size", async () => {
  const { createFramePipeline } = await load("frames.js", "criterion 4");
  const h = harness();
  const pipe = createFramePipeline({ now: h.now, encode: h.encode, send: h.send });
  h.t = 1_700_000_123_456;
  await pipe.offer("screen");
  assert.equal(h.sent.length, 1);
  const msg = h.sent[0];
  assert.equal(msg.type, "astra.frame");
  assert.equal(msg.source, "screen");
  assert.equal(
    msg.ts_ms,
    1_700_000_123_456,
    "criterion 5: ts_ms == Date.now() — here the injected page clock",
  );
  assert.equal(msg.width, ENCODED.width);
  assert.equal(msg.height, ENCODED.height);
  assert.equal(msg.jpeg_b64, ENCODED.jpeg_b64);
  assert.equal(msg.phash, ENCODED.phash);
});

test("criterion 4: a 3 s encode drops the frames behind it, it does not queue them", async () => {
  const { createFramePipeline } = await load("frames.js", "criterion 4");
  let release;
  const slow = new Promise((resolve) => {
    release = resolve;
  });
  let first = true;
  const encode = async () => {
    if (first) {
      first = false;
      await slow; // the 3 s encode
    }
    return ENCODED;
  };
  const h2 = harness({ encode });
  const pipe = createFramePipeline({ now: h2.now, encode: h2.encode, send: h2.send });
  const inflightOffer = pipe.offer("screen"); // starts the slow encode at t = 0
  await flush();
  const outcomes = [];
  for (let i = 1; i <= 180; i++) {
    h2.t = Math.round((i * 3000) / 180); // 3 s of 60 fps offers while the encode is stuck
    outcomes.push(pipe.offer("screen"));
    await flush();
  }
  // Asserted while the encode is still stuck: nothing has piled up behind it.
  assert.ok(
    h2.maxInFlight <= 1,
    `criterion 4: no two encodes or sends in flight for one source, saw ${h2.maxInFlight}`,
  );
  assert.ok(
    pipe.stats().pending <= 1,
    `criterion 4: a slow encode causes a drop, never a queue; ${pipe.stats().pending} pending`,
  );
  release(ENCODED);
  const settled = await Promise.all([inflightOffer, ...outcomes]);
  const s = pipe.stats();
  assert.ok(
    s.sent <= 2,
    `criterion 4: the 3 s encode produces one frame (plus at most the single retained pending one ` +
      `of criterion 3), never the backlog; got ${s.sent} sends`,
  );
  assert.ok(
    s.dropped >= 178,
    `criterion 4: the offers behind a stuck encode are dropped (got ${s.dropped}), including the ` +
      "ones at t = 1000/2000/3000 that the 1 fps rule alone would have let through — that is the " +
      "single-flight guard, and the drop count is what criterion 3 displays",
  );
  assert.equal(settled.length, 181);
});

test("criterion 3: a socket that never takes the frame keeps at most one pending", async () => {
  const { createFramePipeline } = await load("frames.js", "criterion 3");
  let releaseSend;
  const stuck = new Promise((resolve) => {
    releaseSend = resolve;
  }); // a send that does not complete: the 10 s disconnection
  const h = harness({ send: () => stuck });
  const offered = [];
  const pipe = createFramePipeline({ now: h.now, encode: h.encode, send: h.send });
  const offers = 600;
  for (let i = 0; i < offers; i++) {
    h.t = Math.round((i * 1000) / 60);
    offered.push(pipe.offer("screen"));
    await flush();
  }
  const s = pipe.stats();
  assert.ok(
    s.pending <= 1,
    `criterion 3: while a send is in flight at most ONE pending frame is retained, older ones are ` +
      `dropped; ${s.pending} were kept (that is the unbounded outbound queue the criterion forbids)`,
  );
  assert.ok(s.inFlight <= 1, `criterion 3: one send in flight, saw ${s.inFlight}`);
  assert.equal(
    s.sent + s.dropped + s.pending + s.inFlight,
    offers,
    `conservation: every offer is accounted for exactly once (${JSON.stringify(s)})`,
  );
  assert.ok(
    s.dropped > 500,
    `criterion 3: the drop count is what the page displays, got ${s.dropped}`,
  );
  releaseSend(); // let the socket come back so nothing is left hanging in the runner
  await Promise.all(offered);
});

test("criterion 4: the 1 fps rule is per source, not global", async () => {
  const { createFramePipeline } = await load("frames.js", "criterion 4");
  const h = harness();
  const pipe = createFramePipeline({ now: h.now, encode: h.encode, send: h.send });
  h.t = 5_000;
  await pipe.offer("screen");
  await pipe.offer("camera");
  await pipe.offer("screen"); // same second, same source: dropped
  const sources = h.sent.map((m) => m.source);
  assert.deepEqual(sources, ["screen", "camera"], "at most one frame per source per second");
  assert.equal(pipe.stats().dropped, 1);
});

test("criterion 4: MAX_FPS is the policy value, not a page opinion", async () => {
  const { MAX_FPS, JPEG_QUALITY } = await load("frames.js", "criterion 5");
  assert.equal(MAX_FPS, 1.0, "perception/policy.py MAX_FPS");
  assert.equal(JPEG_QUALITY, 85, "perception/policy.py JPEG_QUALITY");
});

test("criterion 5: targetSize downscales to the bucket and never upscales", async () => {
  const { targetSize, bucketFor, SCREEN_MAX, CAMERA_BUCKET } = await load("frames.js", "criterion 5");
  // Hand-computed against local_astra/perception/policy.target_size:
  //   3840x2160 into screen_max(1568x1176): scale = min(1, 1568/3840, 1176/2160) = 0.40833..
  //   -> 1568 x 882 (even dimensions, aspect preserved)
  assert.deepEqual(targetSize(3840, 2160, SCREEN_MAX), [1568, 882]);
  //   640x400 is already inside the bucket: no upscale.
  assert.deepEqual(targetSize(640, 400, SCREEN_MAX), [640, 400]);
  //   1920x1080 into camera(896x896): scale = min(1, 896/1920, 896/1080) = 0.46666..
  //   -> 896 x 504
  assert.deepEqual(targetSize(1920, 1080, CAMERA_BUCKET), [896, 504]);
  assert.equal(bucketFor("screen", 3840, 2160).name, "screen_max");
  assert.equal(bucketFor("window", 800, 600).name, "screen_max");
  assert.equal(bucketFor("camera", 1280, 720).name, "camera");
});
