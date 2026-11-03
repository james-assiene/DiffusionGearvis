// The v1 extractor registry (T-0025 criterion 8): which assist kinds this page can honestly
// report, and why the rest cannot.
//
// Planner's finding 1, recorded in docs/tasks/T-0025.md: `getDisplayMedia` hands the page a
// `MediaStreamTrack` and nothing else — a browser tab cannot read another tab's text, its PDF text
// layer, its spreadsheet cells or its accessibility tree. So a real share registers *no* extractor
// at all, and the panel says so in each row instead of letting the user tick a box into a regime
// the page cannot honour. Those kinds arrive with the native daemon (docs/01 D6, M2+).
//
// The one exception is the synthetic canvas source (`?source=canvas`): `hud/src/capture.js` draws
// its caption itself, so the page genuinely knows that text (ruling A5 — availability keys on
// `synthetic`, because capture.js reports `source === "screen"` for a real share and for the
// canvas alike). It exists so the egress path can be exercised end to end; it is not a real
// assist and no number may be quoted from it (docs/tasks/T-0025.md "Out of scope", Q9).

import { ASSIST_KINDS } from "./gate.js";

/** The caption `syntheticStreamProvider` paints into the canvas, minus its frame counter. */
export const SYNTHETIC_CAPTION = "synthetic capture";

const NATIVE_ONLY = {
  pdf_text: "a browser tab cannot reach another tab's PDF text layer (native daemon, docs/01 D6, M2+)",
  cells: "a browser tab cannot reach a spreadsheet's cells (native daemon, docs/01 D6, M2+)",
  at_spi: "a browser tab cannot reach the accessibility tree (native daemon, docs/01 D6, M2+)",
};

const sharedSurfaceReason = (source) =>
  `a live ${source} capture hands this page a video track and nothing else, so it cannot read the ` +
  "text of the surface it shares (native daemon, docs/01 D6, M2+)";

const SYNTHETIC_PAGE_TEXT_REASON =
  "the page drew this text itself: the caption hud/src/capture.js paints into the synthetic " +
  "canvas (?source=canvas)";

/** True while the shell is actually grabbing frames: `main.js` writes the capture size on every
 *  grab and clears it on mute (T-0008's `[data-capture-size]` hook). Reporting the caption of a
 *  capture that is no longer running would be assist text with no pixels beside it, which is the
 *  dishonesty this whole task is about. With no document (node) there is nothing to check. */
function captureIsLive(doc) {
  if (!doc || typeof doc.querySelector !== "function") return true;
  const el = doc.querySelector("[data-capture-size]");
  return Boolean(el && (el.getAttribute("data-capture-size") || "").trim());
}

function syntheticCaptionExtractor(doc) {
  return () => {
    if (!captureIsLive(doc)) return null;
    return (
      `${SYNTHETIC_CAPTION}: the caption hud/src/capture.js paints onto the canvas this page is ` +
      "sharing, followed by a frame counter this extractor does not read. There is no other text " +
      "on this surface."
    );
  };
}

/**
 * registryFor({ source, synthetic, doc }) -> { extractors, reasons }
 *
 * `extractors` holds one entry per kind this page can honestly report for the *live* capture —
 * exactly one (`page_text`) over the synthetic canvas, none otherwise. `reasons` always holds all
 * four kinds, so every disabled row can say why it is disabled.
 */
export function registryFor({ source = "screen", synthetic = false, doc = null } = {}) {
  const extractors = {};
  if (synthetic) extractors.page_text = syntheticCaptionExtractor(doc);
  const reasons = {
    ...NATIVE_ONLY,
    page_text: synthetic ? SYNTHETIC_PAGE_TEXT_REASON : sharedSurfaceReason(source),
  };
  for (const kind of ASSIST_KINDS) {
    if (!reasons[kind]) reasons[kind] = "no extractor is registered for this capture source";
  }
  return { extractors, reasons };
}
