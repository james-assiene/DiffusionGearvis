// The page: one socket, one capture, two output channels and the thinking meter (T-0008).
//
// Everything testable without a browser lives in the DOM-free modules (protocol, ws, frames,
// phash, state, panels); this file is the wiring: it owns the elements, the clock and the
// intervals. No build step, no npm, no network beyond loopback.

import { createCapture, cameraStreamProvider, displayStreamProvider, syntheticStreamProvider } from "./capture.js";
import { captureTickMs, createFramePipeline, frameIntervalFromState } from "./frames.js";
import { mountPanels, openAfterMount } from "./panels.js";
import { createThread } from "./panels/thread.js";
import { drawMarks, renderThinking } from "./panels/thinking.js";
import { decode, hello } from "./protocol.js";
import { connect, resolveWsUrl } from "./ws.js";
import { initialState, reduce } from "./state.js";
import { createClearTracker } from "./session/gate.js";
import { createVision } from "./vision.js";
import { createMic } from "./mic.js";
import { createStop } from "./stop.js";

// The capture loop only offers; the pipeline throttles to the interval in force (1000 ms by
// default, the server's frame_interval_ms setting after T-0227). The tick follows the interval
// (captureTickMs: 250 ms at 1000 ms) so a tick firing a little early never halves the rate.
let frameIntervalMs = 1000;
let captureTimer = null;
const METER_TICK_MS = 100;

const doc = document;
const el = (id) => doc.getElementById(id);
const now = () => Date.now();
const log = (line) => console.info("[hud]", line);

const ui = {
  chip: el("turn-chip"),
  meter: el("thinking-meter"),
  bar: el("thinking-bar"),
  thread: doc.querySelector("[data-thread]"),
  banner: el("capture-banner"),
  errors: el("error-strip"),
  status: el("conn-status"),
  framesSent: el("frames-sent"),
  framesDropped: el("frames-dropped"),
  captureSize: el("capture-size"),
  ignored: el("ignored-count"),
  preview: el("preview"),
  share: el("share-button"),
  mute: el("mute-button"),
  mic: el("mic-button"),
  stop: el("stop-button"),
  micBadge: doc.querySelector("header [data-mic-badge]"),
  caption: doc.querySelector("[data-capture-caption]"),
  seePane: doc.querySelector("main > .pane.thumb"),
  visionLabel: doc.querySelector(".pane.thumb .vision-source"),
  visionCombo: doc.querySelector(".pane.thumb select[data-vision-source]"),
  visionThumbs: doc.querySelector(".pane.thumb [data-vision-thumbs]"),
  seeEmpty: doc.querySelector(".pane.thumb [data-empty]"),
};

// --- T-0177 C: the What I see pane shows what the session sent, and the source combo ---------------

const vision = createVision();
const nonLiveEmpty = ui.seeEmpty ? ui.seeEmpty.textContent : "";

/** What vision.js needs to know about the capture. */
function captureInfo() {
  return capture && capture.active() ? { running: true, source: capture.source } : null;
}

function sendVision(message) {
  if (!message) return;
  socket.send(message).catch((err) => log(`vision select failed: ${err.message}`));
}

/** C1/C3/C4: the combo, #preview, the thumbnails, the empty sentence, the caption, the badge. */
function renderVision() {
  const info = captureInfo();
  const combo = vision.combo(info);
  if (ui.visionLabel && ui.visionCombo) {
    ui.visionLabel.hidden = combo === null;
    if (combo) {
      for (const option of combo.options) {
        const node = ui.visionCombo.querySelector(`option[value="${option.value}"]`);
        if (!node) continue;
        node.textContent = option.label;
        node.disabled = option.disabled;
      }
      ui.visionCombo.value = combo.value;
    }
  }
  const panel = vision.panel(info);
  if (panel.large && panel.large.src) {
    if (ui.preview.getAttribute("src") !== panel.large.src) ui.preview.src = panel.large.src;
    ui.preview.hidden = false;
  } else {
    ui.preview.hidden = true;
    ui.preview.removeAttribute("src");
  }
  if (ui.visionThumbs) {
    ui.visionThumbs.replaceChildren(
      ...panel.thumbs
        .filter((t) => t.src)
        .map((t) => {
          const img = doc.createElement("img");
          img.dataset.visionThumb = "";
          img.alt = "an earlier image sent with the same question";
          img.src = t.src;
          return img;
        }),
    );
  }
  if (ui.seeEmpty) ui.seeEmpty.textContent = panel.empty ?? nonLiveEmpty;
  if (ui.caption) ui.caption.textContent = panel.caption;
  if (ui.seePane) {
    if (panel.live) ui.seePane.dataset.visionLive = "";
    else delete ui.seePane.dataset.visionLive;
  }
}

if (ui.visionCombo) {
  ui.visionCombo.addEventListener("change", () => {
    sendVision(vision.choose(ui.visionCombo.value));
    renderVision(); // C1: the value is always the last echo; the echo moves it, not the click
  });
  // Keys typed on the combo stay there.
  ui.visionCombo.addEventListener("keydown", (event) => event.stopPropagation());
}

/** T-0154 A4: Share reads "Sharing screen" / "Sharing camera" (green) iff a real capture runs.
 *  A synthetic capture is never called sharing: the page cannot vouch for that stream. */
function renderShare() {
  const sharing = Boolean(capture && !capture.synthetic);
  ui.share.textContent = sharing ? `Sharing ${capture.source === "camera" ? "camera" : "screen"}` : "Share";
  if (sharing) ui.share.dataset.sharing = "";
  else delete ui.share.dataset.sharing;
}

let state = initialState();
const ignored = { unknown: 0, malformed: 0, realtime: 0 };
const localErrors = [];
let capture = null;
const subscribers = new Set(); // panels that asked for the inbound stream (T-0025 criterion 13)
let astraReceived = 0; // decoded astra.* messages handed to the subscribers so far
const clears = createClearTracker(); // T-0162 D-2: the page's one source of truth for resets

drawMarks(ui.bar);

// --- rendering ---------------------------------------------------------------------------------

function renderErrors() {
  const all = [...localErrors, ...state.errors];
  ui.errors.hidden = all.length === 0;
  ui.errors.replaceChildren(
    ...all.map((e) => {
      const line = doc.createElement("p");
      line.className = "error-line";
      line.dataset.errorCode = e.code;
      line.textContent = `${e.code}: ${e.message}`;
      return line;
    }),
  );
}

function renderCounters() {
  const stats = pipeline.stats();
  ui.framesSent.dataset.framesSent = String(stats.sent);
  ui.framesSent.textContent = String(stats.sent);
  ui.framesDropped.dataset.framesDropped = String(stats.dropped);
  ui.framesDropped.textContent = String(stats.dropped);
  ui.ignored.textContent = `${ignored.unknown}u/${ignored.malformed}m/${ignored.realtime}r`;
  ui.ignored.title = "ignored and counted: unknown astra.* / malformed / Realtime passthrough";
}

function render() {
  renderThinking(ui, state, now());
  renderErrors();
  renderCounters();
}

// --- the socket --------------------------------------------------------------------------------

function onMessage(raw) {
  const decoded = decode(raw);
  if (decoded.kind === "astra") {
    state = reduce(state, decoded.msg, now());
    // T-0227 B-2: the capture interval in force (the item's value, never its pending), hello
    // replays included; anything unreadable leaves the current interval.
    const intervalMs = frameIntervalFromState(decoded.msg);
    if (intervalMs !== null) applyFrameInterval(intervalMs);
    // T-0162 D-2: the page's one reset tracker, advanced before any panel sees the message, so a
    // panel that mounts later seeds from it (host.lastResets) and never takes a replay of a reset
    // it missed for a new one.
    const firstEver = astraReceived === 0;
    const newReset = clears.accept(decoded.msg);
    for (const fn of subscribers)
      try { fn(decoded.msg); } catch (err) { log(`panel handler failed: ${err && err.message}`); }
    astraReceived += 1;
    if (newReset) onNewReset(decoded.msg, firstEver);
  } else {
    ignored[decoded.kind] += 1; // ignored and counted, exactly as contracts/codec.decode is
  }
  render();
}

let wsUrl = null;
try {
  wsUrl = resolveWsUrl(globalThis.location.search);
} catch (err) {
  localErrors.push({ code: "ws_url_refused", message: err.message });
  log(err.message);
}

const connectionListeners = new Set(); // T-0162 D-1: the Restart button is disabled while closed
let socketsOpened = 0;

const socket = connect({
  url: wsUrl,
  onMessage,
  onStatus: (status) => {
    ui.status.textContent = status;
    const open = socket.isOpen();
    for (const fn of connectionListeners)
      try { fn(open); } catch (err) { log(`connection listener failed: ${err && err.message}`); }
  },
  // The first message on every socket, and exactly one per socket (criterion 1).
  onOpen: (api) => {
    socketsOpened += 1;
    api.send(hello()).catch((err) => log(`hello failed: ${err.message}`));
  },
});

// --- capture and the frame pipeline --------------------------------------------------------------

const pipeline = createFramePipeline({
  now,
  encode: async () => {
    const grabbed = capture && capture.grab();
    if (!grabbed) throw new Error("capture is not ready");
    const size = `${grabbed.captureWidth}x${grabbed.captureHeight}`;
    ui.captureSize.dataset.captureSize = size;
    ui.captureSize.textContent = size;
    // T-0177 C3: #preview is never fed by the local capture, only by astra.vision.frame.
    return grabbed;
  },
  send: (message) => socket.send(message),
  log,
});

function providerFor(kind) {
  if (kind === "canvas" || kind === "synthetic") return syntheticStreamProvider({ doc });
  if (kind === "camera") return cameraStreamProvider();
  return displayStreamProvider();
}

/** `origin` is "share" (a click on Share) or "url" (?source= at load): only a Share in live selects
 *  the shared screen (T-0177 C2, H-T177-1). */
async function startCapture(kind, origin) {
  const replaced = capture !== null;
  stopCapture({ restarting: true });
  const started = createCapture({
    getStream: providerFor(kind),
    doc,
    log,
    // C2: the browser's own "Stop sharing" stops the capture as Stop sharing does.
    onEnded: () => {
      if (capture !== started) return;
      stopCapture();
      render();
    },
  });
  capture = started;
  try {
    await started.start();
  } catch (err) {
    if (capture === started) capture = null;
    localErrors.push({ code: "capture_failed", message: err.message });
    // The capture this one replaced is gone too: the shared choice has nothing to look at.
    if (replaced) sendVision(vision.captureStopped());
    renderVision();
    render();
    return;
  }
  if (capture !== started) {
    // C-R2: stopped while it was starting (the picker was still pending): stop the stream that
    // arrived afterwards, so no capture and no sharing bar is left behind, and send nothing for it.
    started.stop();
    return;
  }
  sendVision(vision.captureStarted(origin));
  // Criterion 7: a stream whose provenance we cannot vouch for is labelled, permanently.
  if (capture.synthetic) {
    ui.banner.textContent = "SYNTHETIC CAPTURE";
    ui.banner.hidden = false;
  }
  ui.mute.disabled = false;
  renderShare();
  renderVision();
  render();
}

/** Stop sharing: stop the capture itself (criterion 6). Sharing again requires a fresh acquisition.
 *  T-0177 C2: a stopped capture selects the browser while shared is the echoed choice, unless a
 *  new capture is replacing it (`restarting`). */
function stopCapture({ restarting = false } = {}) {
  const stopped = capture !== null;
  if (capture) capture.stop();
  capture = null;
  ui.banner.textContent = "";
  ui.banner.hidden = true;
  ui.mute.disabled = true;
  ui.captureSize.dataset.captureSize = "";
  ui.captureSize.textContent = "—";
  renderShare();
  if (stopped && !restarting) sendVision(vision.captureStopped());
  renderVision();
}

ui.share.addEventListener("click", () => {
  const kind = new URLSearchParams(globalThis.location.search).get("source") || "screen";
  startCapture(kind, "share");
});
ui.mute.addEventListener("click", () => {
  stopCapture();
  render();
});

// --- T-0162 B-6/B-7: the Settings popover and the header badge ---------------------------------

const settings = {
  button: el("settings-button"),
  popover: el("settings-popover"),
  slot: doc.querySelector("[data-settings-slot]"),
  badge: el("assist-header-badge"),
};

function settingsOpen() {
  return Boolean(settings.popover && !settings.popover.hidden);
}

function setSettingsOpen(open, { focusButton = false } = {}) {
  if (!settings.popover || !settings.button) return;
  settings.popover.hidden = !open;
  settings.button.setAttribute("aria-expanded", String(open));
  if (open) {
    const first = settings.popover.querySelector("input:not([disabled]), button, select");
    if (first) first.focus();
  } else if (focusButton) settings.button.focus();
}

if (settings.button && settings.popover) {
  settings.button.addEventListener("click", () => setSettingsOpen(!settingsOpen()));
  // Keys typed in the popover stay in it, and Escape closes it.
  settings.popover.addEventListener("keydown", (event) => {
    event.stopPropagation();
    if (event.key === "Escape") setSettingsOpen(false, { focusButton: true });
  });
  doc.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && settingsOpen()) setSettingsOpen(false, { focusButton: true });
  });
  // An outside click closes it; a click on the gear is the gear's own toggle. Both events, so a
  // scripted element.click() (no pointer events) closes it too.
  const closeIfOutside = (event) => {
    if (!settingsOpen()) return;
    const inside = settings.popover.contains(event.target) || settings.button.contains(event.target);
    if (!inside) setSettingsOpen(false);
  };
  doc.addEventListener("pointerdown", closeIfOutside);
  doc.addEventListener("click", closeIfOutside);
}

/** D12a: the server's badge, verbatim, while it is not `pixels`; nothing before the first echo. */
function renderHeaderBadge(message) {
  if (!settings.badge) return;
  const badge = typeof message.badge === "string" ? message.badge : "";
  const show = badge !== "" && badge !== "pixels";
  settings.badge.textContent = show ? badge : "";
  settings.badge.hidden = !show;
  if (show) settings.badge.title = `Context helpers on: ${badge}`;
  else settings.badge.removeAttribute("title");
}

subscribers.add((message) => {
  if (message && message.type === "astra.assist.state") renderHeaderBadge(message);
});

// T-0177 C: the vision state, its frames and the reset reach the What I see pane before onNewReset
// stops the capture (so a reset's stop sees the echoed choice, C2).
// C-R3: only an astra.vision.* message or a new reset re-renders the pane (and capture start/stop).
subscribers.add((message) => {
  if (vision.accept(message)) renderVision();
});

// --- T-0162 D-2/D-3: the page-wide clear ------------------------------------------------------------
//
// Every panel clears itself once per new `resets` value of astra.session.cleared (state.js, the
// thread, the cards, the Tools pane and the denoising view each hold the same rule; they seed it
// from `clears`, the page's tracker, through host.lastResets — normally null at mount since T-0165
// opens the socket only after the mount, but a panel mounted late past the ceiling needs it). Here: the
// reset stops the capture, as Stop sharing does, so the next Share picks the new page (H-T162-2). Errors in
// the strip, the assist flags, the log checkbox, the theme and the diagnostics counters are kept.
//
// One exception, and only for the capture: a replay that is the very first message this page ever
// received is the hello of a page opened *after* the reset. Its capture belongs to the new session
// already, so it is not stopped (a page that missed a live reset while it was reconnecting has
// received messages before, and is stopped).
function onNewReset(message, firstEver) {
  const openedAfterTheReset = message.replay === true && socketsOpened <= 1 && firstEver;
  if (!openedAfterTheReset) stopCapture();
  log(`session cleared (resets ${message.resets}${message.replay ? ", replayed" : ""})`);
}

// T-0181 B3 (H-T181-4): the page has no keyboard shortcut. Stop sharing and the mic mute are
// buttons only.

// --- T-0181 B: the mic control --------------------------------------------------------------------
//
// hud/src/mic.js holds the last astra.mic.state echo; the button and the badge show only that. A
// click sends astra.mic.mute {muted: !echoed} and changes nothing until the echo arrives (B1).
const mic = createMic();

function renderMic() {
  const view = mic.view();
  if (ui.mic) {
    ui.mic.hidden = view === null;
    if (view) {
      ui.mic.textContent = view.text;
      ui.mic.title = view.title;
      ui.mic.disabled = view.disabled;
      if (typeof view.pressed === "boolean") ui.mic.setAttribute("aria-pressed", String(view.pressed));
      else ui.mic.removeAttribute("aria-pressed");
    }
  }
  if (ui.micBadge) {
    const badge = view ? view.badge : null;
    ui.micBadge.hidden = badge === null;
    ui.micBadge.textContent = badge ? badge.text : "";
    if (badge) ui.micBadge.title = badge.title;
    else ui.micBadge.removeAttribute("title");
  }
}

if (ui.mic) {
  ui.mic.addEventListener("click", () => {
    const message = mic.click();
    if (!message) return;
    socket.send(message).catch((err) => log(`mic mute failed: ${err.message}`));
  });
}

subscribers.add((message) => {
  if (mic.accept(message)) renderMic();
});

// --- T-0193 B: the Interrupt Gearvis button -------------------------------------------------------
//
// hud/src/stop.js follows the astra.turn.state echo: hidden before the first one, enabled only in
// THINKING, SPEAKING and PAUSED. A click sends astra.turn.stop for the echoed turn and keeps the
// button disabled until the next echo (B-2). A dropped socket disarms it (B-R4). No key stops a turn.
const stopper = createStop();

function renderStop() {
  const view = stopper.view();
  if (!ui.stop) return;
  ui.stop.hidden = view === null;
  if (view) {
    ui.stop.textContent = view.text;
    ui.stop.title = view.title;
    ui.stop.disabled = view.disabled;
  }
}

if (ui.stop) {
  ui.stop.addEventListener("click", () => {
    const message = stopper.click();
    renderStop();
    if (!message) return;
    socket.send(message).catch((err) => log(`turn stop failed: ${err.message}`));
  });
}

subscribers.add((message) => {
  if (stopper.accept(message)) renderStop();
});

connectionListeners.add((open) => {
  stopper.connection(open);
  renderStop();
});

// --- the later tasks' panels, and the loop --------------------------------------------------------

// T-0154 B1-B7: the conversation thread — one article per turn: the user bubble, the utterances,
// the meta line, the card slot and the turn's action cards (T-0090, display-only). It is handed
// every decoded astra.* message first, so a turn's article exists before the ui panel asks for
// its card slot; the ui panel renders each surface into the slot its id names.
const thread = ui.thread ? createThread(ui.thread) : null;
if (thread) subscribers.add(thread.handle);
else log("no [data-thread] in the page: the conversation is not rendered");

const mounting = mountPanels({
  host: {
    cardSlot: thread ? (surfaceId) => thread.cardSlot(surfaceId) : undefined,
    document: doc,
    settingsSlot: settings.slot || undefined, // T-0162 B-6: the assist panel goes behind the gear
    // T-0162 D-1: the header slot of the Restart button, and fn(open) now and on every change of
    // the socket (the button is disabled while it is closed).
    sessionSlot: doc.querySelector("[data-session-slot]") || undefined,
    // T-0162 D-2 (review I-1): the last reset count the page accepted; panels seed from it at mount.
    lastResets: () => clears.last(),
    onConnection: (fn) => {
      connectionListeners.add(fn);
      fn(socket.isOpen());
      return () => connectionListeners.delete(fn);
    },
    send: (message) => socket.send(message),
    subscribe: (fn) => (subscribers.add(fn), () => subscribers.delete(fn)),
  },
  log,
}).then((result) => {
  log(`panels mounted: ${result.mounted.length}, missing: ${result.missing.length}, failed: ${result.failed.length}`);
});

// T-0213 B-1: the Settings panel, in its own slot below Context helpers in the same popover. It
// mounts beside the panels (and before the socket opens, so it sees the hello's settings state);
// a failure to load it leaves the page and the socket working.
const settingsMounting = import("./settings/index.js")
  .then((module) =>
    module.mount({
      document: doc,
      slot: doc.querySelector("[data-settings-panel-slot]") || undefined,
      subscribe: (fn) => (subscribers.add(fn), () => subscribers.delete(fn)),
      // T-0213 C: the panel's changes; ./settings/gate.js sends only astra.settings.set
      send: (message) => socket.send(message),
      log,
    }),
  )
  .catch((err) => log(`settings panel failed to mount: ${err && err.message}`));

// T-0165 C1/C5: the socket opens once every panel has mounted (so each one sees the first hello
// reply and its journal replay), or after the mount ceiling. `<html data-panels>` is a test seam,
// set once, just before the open: "mounted", or "timeout" on the ceiling path.
openAfterMount(
  Promise.all([mounting, settingsMounting]),
  (outcome) => {
    doc.documentElement.dataset.panels = outcome;
    if (wsUrl) socket.open();
  },
  { log },
);

const requested = new URLSearchParams(globalThis.location.search).get("source");
if (requested === "canvas" || requested === "synthetic") startCapture(requested, "url");

function startCaptureLoop() {
  if (captureTimer !== null) globalThis.clearInterval(captureTimer);
  captureTimer = globalThis.setInterval(() => {
    // Every offer updates the counters as soon as its outcome is known (sent or dropped), so the
    // displayed numbers are never one tick behind what went on the wire.
    if (capture && capture.ready() && socket.isOpen()) pipeline.offer(capture.source).then(renderCounters);
    renderCounters();
  }, captureTickMs(frameIntervalMs));
}

function applyFrameInterval(ms) {
  if (ms === frameIntervalMs) return;
  pipeline.setIntervalMs(ms);
  frameIntervalMs = ms;
  startCaptureLoop();
  log(`frame interval: ${ms} ms (offer tick ${captureTickMs(ms)} ms)`);
}

startCaptureLoop();

globalThis.setInterval(() => renderThinking(ui, state, now()), METER_TICK_MS);

render();
