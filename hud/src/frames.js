// The frame pipeline: resolution policy, 1 fps by default (T-0227: the interval is a setting),
// single flight (criteria 3, 4, 5).
//
// The constants below are a mirror of local_astra/perception/policy.py and are compared with it
// by tests/unit/test_hud_constants_parity.py — a divergence fails there rather than silently
// changing what the model sees (.claude/rules/perception-memory.md: "changing a bucket requires
// re-running B2"). Change policy.py first, this second, never only one.
//
// No DOM here: `node --test tests/hud` imports this module; the pixels arrive through the
// injected `encode` callback.

/** policy.SCREEN_BUCKETS[0] — pixel-first: screen frames get the maximum visual-token budget. */
export const SCREEN_MAX = { name: "screen_max", max_long_side: 1568, max_short_side: 1176 };
/** policy.CAMERA_BUCKET */
export const CAMERA_BUCKET = { name: "camera", max_long_side: 896, max_short_side: 896 };
/** policy.JPEG_QUALITY */
export const JPEG_QUALITY = 85;
/** policy.MAX_FPS — at most one astra.frame per source per second. */
export const MAX_FPS = 1.0;
/** policy.RING_BUFFER_S — the server's ring; here only to keep the mirror complete. */
export const RING_BUFFER_S = 60;

/** config.FRAME_INTERVALS_MS — the capture intervals the page accepts, ascending, in ms (T-0227 B-1).
 *  Compared with config.py by tests/unit/test_hud_constants_parity.py and tests/hud/frame_interval.test.mjs. */
export const FRAME_INTERVALS_MS = Object.freeze([250, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000]);

/** The capture loop's offer tick at the default interval (main.js's fixed period before T-0227). */
const DEFAULT_CAPTURE_TICK_MS = 250;

const isAllowedInterval = (ms) => Number.isInteger(ms) && FRAME_INTERVALS_MS.includes(ms);

/**
 * captureTickMs(intervalMs) -> the capture loop's offer period at that interval (T-0227 B-3).
 *
 * The throttle is strict (never faster than the interval), so a tick equal to the interval drops
 * every frame whose tick fires a millisecond early and cuts the rate by up to half. A quarter of
 * the interval keeps the worst gap near 1.25x; it is capped at 250 ms so long intervals stay as
 * responsive to a capture becoming ready as today. 250 ms at the default 1000 ms, as before.
 */
export function captureTickMs(intervalMs) {
  const ms = isAllowedInterval(intervalMs) ? intervalMs : 1000 / MAX_FPS;
  return Math.min(DEFAULT_CAPTURE_TICK_MS, ms / 4);
}

/**
 * frameIntervalFromState(message) -> number | null (T-0227 B-2).
 *
 * The `value` in force (never `pending`) of the `frame_interval_ms` item of an
 * astra.settings.state, hello replays included, iff it is an integer in FRAME_INTERVALS_MS;
 * null for anything else, so the caller keeps the interval it has.
 */
export function frameIntervalFromState(message) {
  if (!message || typeof message !== "object" || message.type !== "astra.settings.state") return null;
  if (!Array.isArray(message.items)) return null;
  const item = message.items.find((it) => it && typeof it === "object" && it.key === "frame_interval_ms");
  if (!item) return null;
  return isAllowedInterval(item.value) ? item.value : null;
}

/** A frame whose base64 is bigger than this is refused and logged rather than shipped: the codec
 *  would accept up to 8 MiB and the socket would stall on it (T-0008 implementation notes). */
export const MAX_FRAME_BYTES = 1048576; // 1 MiB

/** policy.bucket_for (without the crop/lean options, which this task does not have). */
export function bucketFor(source, _width, _height) {
  return source === "camera" ? CAMERA_BUCKET : SCREEN_MAX;
}

/** policy.target_size: downscale (never upscale) into the bucket, aspect kept, even dimensions. */
export function targetSize(width, height, bucket) {
  if (width < 1 || height < 1) throw new Error("dimensions must be positive");
  const longSide = Math.max(width, height);
  const shortSide = Math.min(width, height);
  const scale = Math.min(1, bucket.max_long_side / longSide, bucket.max_short_side / shortSide);
  const even = (v) => Math.max(2, Math.floor(Math.trunc(v * scale) / 2) * 2);
  return [even(width), even(height)];
}

/**
 * createFramePipeline({ now, encode, send, maxFps, log }) -> { offer, stats, setIntervalMs }
 *
 *   offer(source) -> Promise<"sent" | "dropped">   the capture loop calls this as fast as it likes
 *   stats() -> { sent, dropped, inFlight, pending }
 *   setIntervalMs(ms)   T-0227 B-1: the minimum gap, judged from the next accepted frame on; ms must
 *                       be an integer in FRAME_INTERVALS_MS, anything else is ignored (logged) and the
 *                       gap in force is kept. Without a call the gap is 1000 / maxFps.
 *
 * Invariants: at most one encode-and-send in flight per source, at most one *pending* frame
 * retained behind it (the newest; older ones are dropped and counted), and at most one frame per
 * source per 1/MAX_FPS seconds. A slow encode or a stuck socket therefore drops frames — it never
 * queues them.
 */
export function createFramePipeline({
  now,
  encode,
  send,
  maxFps = MAX_FPS,
  maxBytes = MAX_FRAME_BYTES,
  log = () => {},
}) {
  let minGapMs = 1000 / maxFps;
  const sources = new Map();
  let sent = 0;
  let dropped = 0;

  const stateOf = (source) => {
    let state = sources.get(source);
    if (!state) {
      state = { busy: false, pending: null, lastAcceptedAt: -Infinity };
      sources.set(source, state);
    }
    return state;
  };

  const drop = (job) => {
    dropped += 1;
    job.resolve("dropped");
  };

  function run(state, job) {
    state.busy = true;
    state.lastAcceptedAt = job.at;
    (async () => {
      let outcome = "dropped";
      try {
        const encoded = await encode(job.source, { ts_ms: job.at, quality: JPEG_QUALITY });
        if (!encoded || typeof encoded.jpeg_b64 !== "string") throw new Error("encoder gave nothing");
        if (encoded.jpeg_b64.length > maxBytes) {
          log(`frame refused: ${encoded.jpeg_b64.length} bytes is over the ${maxBytes} byte limit`);
        } else {
          await send({
            type: "astra.frame",
            source: job.source,
            ts_ms: job.at,
            width: encoded.width,
            height: encoded.height,
            jpeg_b64: encoded.jpeg_b64,
            phash: encoded.phash,
          });
          outcome = "sent";
        }
      } catch (err) {
        log(`frame dropped: ${err && err.message}`);
      }
      state.busy = false;
      if (outcome === "sent") sent += 1;
      else dropped += 1;
      job.resolve(outcome);

      const next = state.pending;
      state.pending = null;
      if (next) {
        next.at = now();
        if (next.at - state.lastAcceptedAt < minGapMs) drop(next);
        else run(state, next);
      }
    })();
  }

  return {
    offer(source) {
      const state = stateOf(source);
      const at = now();
      return new Promise((resolve) => {
        const job = { source, at, resolve };
        if (state.busy) {
          // criterion 3: keep at most ONE pending frame; the older one is dropped, not queued.
          if (state.pending) drop(state.pending);
          state.pending = job;
          return;
        }
        if (at - state.lastAcceptedAt < minGapMs) {
          drop(job);
          return;
        }
        run(state, job);
      });
    },
    setIntervalMs(ms) {
      if (!isAllowedInterval(ms)) {
        log(`frame interval refused: ${String(ms)} is not one of ${FRAME_INTERVALS_MS.join(", ")} ms; keeping ${minGapMs} ms`);
        return;
      }
      minGapMs = ms;
    },
    stats() {
      let inFlight = 0;
      let pending = 0;
      for (const state of sources.values()) {
        if (state.busy) inFlight += 1;
        if (state.pending) pending += 1;
      }
      return { sent, dropped, inFlight, pending };
    },
  };
}
