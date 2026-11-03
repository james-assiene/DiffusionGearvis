// The "What I see" pane's model (T-0177 group C): the vision source combo and the images the
// session actually sent.
//
// The session is the source of truth (docs/tasks/T-0177.md, H-T177-2): the pane shows only the
// images of astra.vision.frame, the exact bytes of the final user message's ImageParts, and never
// the page's own capture. The combo's value is always the session's last astra.vision.state echo,
// never the click (A3). A state only arrives in live, so "live" here means "a state has arrived".
//
// No DOM here: `node --test tests/hud` imports this module. hud/src/main.js only wires it.

import { createClearTracker } from "./session/gate.js";

/** The only client->server type this module may emit. */
export const VISION_CLIENT_TYPES = ["astra.vision.select"];
const [SELECT_TYPE] = VISION_CLIENT_TYPES;

export const VISION_SOURCES = ["browser", "shared"];

export const COMBO_LABEL = "Looking at";
export const LIVE_EMPTY = "Nothing sent yet: Gearvis looks when you ask";
export const BLIND_EMPTY = "No image was sent for this question";
export const TOO_LARGE_EMPTY = "The image sent was too large to show here";

const SEP = " · ";
const BROWSER_NAME = "Gearvis's browser";
const SHARED_SCREEN = "Shared screen";
const SHARED_CAMERA = "Shared camera";
const OVERVIEW_NAME = "Page overview";

const select = (source) => ({ type: SELECT_TYPE, source });
const isObject = (value) => value !== null && typeof value === "object";

/** One sent image as the pane shows it; `src` is the session's bytes verbatim, or null when the
 *  server omitted them (A1: `omitted: "too_large"`). */
function shownImage(image) {
  const b64 = typeof image.jpeg_b64 === "string" && image.jpeg_b64 !== "" ? image.jpeg_b64 : null;
  return {
    src: b64 === null ? null : "data:image/jpeg;base64," + b64,
    width: image.width,
    height: image.height,
    kind: image.kind,
    identity: image.identity,
  };
}

/** C4: the name of the source of the image shown. */
function sourceName(frame, image) {
  if (image.kind === "overview") return OVERVIEW_NAME;
  if (frame.source === "shared") return image.kind === "camera" ? SHARED_CAMERA : SHARED_SCREEN;
  return BROWSER_NAME;
}

function captionOf(frame, image) {
  let text = `${sourceName(frame, image)}${SEP}${image.width}x${image.height}${SEP}turn ${frame.turn_id} step ${frame.step}`;
  const earlier = frame.history_images;
  if (Number.isInteger(earlier) && earlier > 0) {
    text += `${SEP}+${earlier} earlier ${earlier === 1 ? "image" : "images"}`;
  }
  return text;
}

const running = (capture) => Boolean(capture && capture.running);

/**
 * createVision() -> { accept, live, combo, choose, captureStarted, captureStopped, panel }
 *
 * accept(msg)            every decoded astra.* message, in wire order; never throws; true iff it
 *                        was an astra.vision.* message or a new reset (re-render the pane)
 * live()                 an astra.vision.state has arrived
 * combo(capture)         null before a state; else { value, label, options }
 * choose(source)         the select to send for a combo change, or null before a state
 * captureStarted(origin) "share" | "url": the select to send (shared, on a Share in live), or null
 * captureStopped()       the select to send (browser, while shared is echoed in live), or null
 * panel(capture)         { large, thumbs, caption, empty, live }
 *
 * capture = null | { running: boolean, source: "screen" | "camera" }
 */
export function createVision() {
  let echoed = null; // the last astra.vision.state's source; null before any
  let frame = null; // the newest astra.vision.frame since the last reset
  const clears = createClearTracker();

  /** Returns true iff the pane may need a re-render: an astra.vision.* message or a new reset
   *  (C-R3: every other message leaves the pane untouched). */
  function accept(msg) {
    if (!isObject(msg)) return false;
    if (msg.type === "astra.vision.state") {
      if (VISION_SOURCES.includes(msg.source)) echoed = msg.source;
      return true;
    }
    if (msg.type === "astra.vision.frame") {
      if (Array.isArray(msg.images)) frame = msg;
      return true;
    }
    if (clears.accept(msg)) {
      // C3: a reset clears the panel. The choice survives it (B9, T-0162 D-3).
      frame = null;
      return true;
    }
    return false;
  }

  const live = () => echoed !== null;

  function combo(capture) {
    if (!live()) return null;
    const camera = Boolean(capture && capture.source === "camera");
    return {
      value: echoed,
      label: COMBO_LABEL,
      options: [
        { value: "browser", label: BROWSER_NAME, disabled: false },
        { value: "shared", label: camera ? SHARED_CAMERA : SHARED_SCREEN, disabled: !running(capture) },
      ],
    };
  }

  function choose(source) {
    if (!live() || !VISION_SOURCES.includes(source)) return null;
    return select(source);
  }

  function captureStarted(origin) {
    return live() && origin === "share" ? select("shared") : null;
  }

  function captureStopped() {
    return live() && echoed === "shared" ? select("browser") : null;
  }

  function panel(capture) {
    const none = { large: null, thumbs: [], caption: "", live: false };
    if (frame === null) return { ...none, empty: live() ? LIVE_EMPTY : null };
    if (frame.blind || frame.images.length === 0) return { ...none, empty: BLIND_EMPTY };
    const images = frame.images.filter(isObject);
    if (images.length === 0) return { ...none, empty: BLIND_EMPTY };
    const newest = images[images.length - 1];
    const large = shownImage(newest);
    return {
      large,
      thumbs: images.slice(0, -1).map(shownImage),
      caption: captionOf(frame, newest),
      empty: large.src === null ? TOO_LARGE_EMPTY : null,
      live: frame.source === "shared" && running(capture),
    };
  }

  return { accept, live, combo, choose, captureStarted, captureStopped, panel };
}
