// The turn chip and the thinking meter (criterion 9, docs/01 D11).
//
// The chip always equals the server's turn state. The meter shows thinking_ms while the state is
// THINKING and advances with the page clock between messages; every message snaps it back to the
// server value (state.js does that). Outside THINKING no elapsed number is shown at all.

import { THINKING_MARKS_MS, thinkingMs } from "../state.js";

/** The meter bar spans 1.5x the last mark, so both marks sit inside it. */
const FULL_SCALE_MS = THINKING_MARKS_MS[THINKING_MARKS_MS.length - 1] * 1.5;

export function renderThinking({ chip, meter, bar }, state, nowMs) {
  if (chip) {
    const label = state.turnState || "IDLE";
    chip.dataset.turnState = label;
    chip.textContent = label;
  }
  if (!meter) return;
  const ms = thinkingMs(state, nowMs);
  if (ms === null) {
    meter.dataset.thinkingMs = "";
    meter.textContent = "—";
    if (bar) bar.style.setProperty("--fill", "0%");
    return;
  }
  const rounded = Math.round(ms);
  meter.dataset.thinkingMs = String(rounded);
  meter.textContent = `${(rounded / 1000).toFixed(1)} s`;
  if (bar) bar.style.setProperty("--fill", `${Math.min(100, (rounded / FULL_SCALE_MS) * 100)}%`);
}

/** The 3000 ms (thinking budget) and 4000 ms (interim update) marks, drawn once. */
export function drawMarks(bar) {
  if (!bar) return;
  const doc = bar.ownerDocument;
  for (const mark of THINKING_MARKS_MS) {
    const tick = doc.createElement("span");
    tick.className = "tick";
    tick.dataset.markMs = String(mark);
    tick.style.left = `${(mark / FULL_SCALE_MS) * 100}%`;
    tick.title = `${mark} ms`;
    bar.append(tick);
  }
}
