// Capture: acquiring a stream, drawing it through the resolution policy, and stopping it.
//
// The HUD never touches audio (criterion 13): getDisplayMedia is called with audio: false and no
// audio track is ever requested or held — the Conductor owns PipeWire capture and playback
// (docs/02 ADR-013, docs/01 D7).
//
// Mute is at the pipeline, not the consumer (criterion 6): stop() calls track.stop() on every
// track of the stream and releases it, so the browser's sharing session — and its indicator —
// ends. Setting track.enabled = false would leave both alive and is not mute.
//
// `getStream` is injected (ruling R6) so this file is drivable without a permission dialog. A
// stream whose provider does not declare itself is treated as SYNTHETIC: unknown provenance is
// labelled, never rendered as the real thing (criterion 7, the honesty rule of docs/01 D15).
//
// A track's `ended` event (the browser's own "Stop sharing") is reported to the caller through
// `onEnded` and nothing else (T-0177 C2).

import { JPEG_QUALITY, bucketFor, targetSize } from "./frames.js";
import { DHASH_COLUMNS, DHASH_ROWS, dhash, grayFromRgba } from "./phash.js";

export const SYNTHETIC_SIZE = { width: 640, height: 400, fps: 5 };

/** The real screen provider. One human click; nothing in this repo automates it (criterion 19). */
export function displayStreamProvider() {
  return async () => {
    const media = globalThis.navigator && globalThis.navigator.mediaDevices;
    if (!media || !media.getDisplayMedia) throw new Error("this browser cannot share a tab");
    const stream = await media.getDisplayMedia({ video: { frameRate: 5 }, audio: false });
    return { stream, synthetic: false, source: "screen" };
  };
}

/** The real camera provider. Video only — see the audio note above. */
export function cameraStreamProvider() {
  return async () => {
    const media = globalThis.navigator && globalThis.navigator.mediaDevices;
    if (!media || !media.getUserMedia) throw new Error("this browser has no camera API");
    const stream = await media.getUserMedia({ video: { frameRate: 5 }, audio: false });
    return { stream, synthetic: false, source: "camera" };
  };
}

/**
 * A deterministic moving pattern on a canvas, for headless runs and screenshots (?source=canvas).
 * It is a synthetic stream and the page says so, permanently and visibly.
 */
export function syntheticStreamProvider({ doc = globalThis.document, size = SYNTHETIC_SIZE } = {}) {
  return async () => {
    const canvas = doc.createElement("canvas");
    canvas.width = size.width;
    canvas.height = size.height;
    const ctx = canvas.getContext("2d");
    let tick = 0;
    const paint = () => {
      tick += 1;
      ctx.fillStyle = "#101820";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#2f6f4f";
      ctx.fillRect((tick * 7) % canvas.width, 40, 120, 120);
      ctx.fillStyle = "#c8d6df";
      ctx.font = "20px monospace";
      ctx.fillText(`synthetic capture ${tick}`, 24, 260);
    };
    paint();
    const timer = globalThis.setInterval(paint, 1000 / size.fps);
    const stream = canvas.captureStream(size.fps);
    stream.addEventListener?.("inactive", () => globalThis.clearInterval(timer));
    return { stream, synthetic: true, source: "screen", onStop: () => globalThis.clearInterval(timer) };
  };
}

/**
 * createCapture({ getStream, doc, log, onEnded }) -> { start, stop, ready, info, grab }
 *
 *   start()  acquires the stream and attaches it to an off-screen <video>
 *   stop()   stops every track of the stream and releases it (mute, criterion 6)
 *   grab()   one frame through the resolution policy: { width, height, jpeg_b64, phash, ... }
 *
 * onEnded() is called once when a video track of the running stream fires `ended` (the browser's
 * own "Stop sharing", T-0177 C2); it is only reported, the caller decides what to do. stop()
 * itself never reports it.
 */
export function createCapture({ getStream, doc = globalThis.document, log = () => {}, onEnded = null } = {}) {
  let stream = null;
  let video = null;
  let onStop = null;
  let synthetic = true;
  let sourceName = "screen";
  let frameCanvas = null;
  let hashCanvas = null;

  const canvasOf = (width, height) => {
    const canvas = doc.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    return canvas;
  };

  const api = {
    get synthetic() {
      return synthetic;
    },
    get source() {
      return sourceName;
    },
    active: () => Boolean(stream),
    ready: () => Boolean(video) && video.videoWidth > 0 && video.videoHeight > 0,
    captureSize: () => (api.ready() ? [video.videoWidth, video.videoHeight] : null),

    async start() {
      if (stream) return api;
      const got = await getStream();
      const acquired = got && got.stream ? got.stream : got;
      if (!acquired) throw new Error("the stream provider returned nothing");
      // Unknown provenance is synthetic: only the two real providers above say otherwise.
      synthetic = !(got && got.synthetic === false);
      sourceName = (got && got.source) || "screen";
      onStop = (got && got.onStop) || null;
      stream = acquired;
      if (typeof onEnded === "function" && typeof acquired.getVideoTracks === "function") {
        let reported = false;
        const ended = () => {
          if (reported || stream !== acquired) return; // stopped, or already reported
          reported = true;
          onEnded();
        };
        for (const track of acquired.getVideoTracks()) track.addEventListener?.("ended", ended);
      }
      video = doc.createElement("video");
      video.muted = true;
      video.playsInline = true;
      video.srcObject = stream;
      try {
        await video.play();
      } catch (err) {
        log(`video.play() refused: ${err && err.message}`);
      }
      return api;
    },

    /** Mute: stop the capture itself, not its consumer. Unmuting needs a fresh acquisition. */
    stop() {
      if (stream) {
        for (const track of stream.getTracks()) track.stop();
      }
      if (onStop) onStop();
      onStop = null;
      stream = null;
      if (video) {
        video.srcObject = null;
        video = null;
      }
    },

    /** One frame: policy size, JPEG at policy quality, 16-hex dHash. null while not ready. */
    grab() {
      if (!api.ready()) return null;
      const width = video.videoWidth;
      const height = video.videoHeight;
      const [targetWidth, targetHeight] = targetSize(
        width,
        height,
        bucketFor(sourceName, width, height),
      );
      if (!frameCanvas) frameCanvas = canvasOf(targetWidth, targetHeight);
      frameCanvas.width = targetWidth;
      frameCanvas.height = targetHeight;
      const ctx = frameCanvas.getContext("2d");
      ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
      const url = frameCanvas.toDataURL("image/jpeg", JPEG_QUALITY / 100);
      const jpeg_b64 = url.slice(url.indexOf(",") + 1);

      if (!hashCanvas) hashCanvas = canvasOf(DHASH_COLUMNS, DHASH_ROWS);
      const hctx = hashCanvas.getContext("2d", { willReadFrequently: true });
      hctx.drawImage(video, 0, 0, DHASH_COLUMNS, DHASH_ROWS);
      const pixels = hctx.getImageData(0, 0, DHASH_COLUMNS, DHASH_ROWS).data;

      return {
        width: targetWidth,
        height: targetHeight,
        jpeg_b64,
        phash: dhash(grayFromRgba(pixels)),
        captureWidth: width,
        captureHeight: height,
      };
    },
  };
  return api;
}
