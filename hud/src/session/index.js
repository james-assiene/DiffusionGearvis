// The header's Restart button and its two-step inline confirm (T-0162 D-1).
//
// This is the DOM half; what may be sent, and when, is ./gate.js. `mount(host)` is the contract
// hud/src/panels.js calls and never throws: without a document or a session slot nothing is
// rendered, and without `send` nothing is ever emitted.
//
// Restart is a click on the header button (which only opens the confirm), then a click on the
// confirm's Restart — never a key shortcut (D-4). Keys typed in the confirm stay in it, so `m`
// there never mutes, and Escape cancels it. The button is disabled while the socket is closed
// (host.onConnection) and while a sent reset waits for the server's astra.session.cleared.

import { CONFIRM_TEXT, SESSION_CLEARED_TYPE, createSessionGate } from "./gate.js";

function el(doc, tag, className, text) {
  const node = doc.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function button(doc, className, text, hook) {
  const b = el(doc, "button", className, text);
  b.type = "button";
  b.setAttribute(hook, "");
  return b;
}

function buildControl(doc, slot) {
  const restart = button(doc, "restart-button", "Restart", "data-restart-button");
  restart.id = "restart-button";
  restart.setAttribute("aria-haspopup", "dialog");
  restart.setAttribute("aria-expanded", "false");
  restart.setAttribute("aria-controls", "restart-confirm");
  restart.title = "Start a new session: clears the conversation and Gearvis's memory";
  restart.disabled = true;

  const confirm = el(doc, "div", "restart-confirm");
  confirm.id = "restart-confirm";
  confirm.setAttribute("role", "alertdialog");
  confirm.setAttribute("aria-label", "Restart");
  confirm.setAttribute("aria-describedby", "restart-confirm-text");
  confirm.setAttribute("data-restart-confirm", "");
  confirm.hidden = true;
  const text = el(doc, "p", "restart-text", CONFIRM_TEXT);
  text.id = "restart-confirm-text";
  const actions = el(doc, "div", "restart-actions");
  const yes = button(doc, "restart-yes", "Restart", "data-restart-confirm-yes");
  const no = button(doc, "restart-no", "Cancel", "data-restart-cancel");
  actions.append(yes, no);
  confirm.append(text, actions);

  slot.append(restart, confirm);
  return { restart, confirm, yes, no };
}

/** mount(host) — host is { document, sessionSlot, send, subscribe, onConnection(fn), log }. */
export function mount(host) {
  const h = host && typeof host === "object" ? host : {};
  const doc = h.document;
  const log = typeof h.log === "function" ? h.log : () => {};
  const send = typeof h.send === "function" ? h.send : null;

  const gate = createSessionGate({ send, log });
  let open = false; // the socket, as host.onConnection last said; closed until it says otherwise
  let confirming = false;
  let refocus = false; // a reset this panel sent: its echo hands the focus back to Restart
  let ui = null;

  function render() {
    if (!ui) return;
    const usable = open && !gate.pending();
    if (!usable) confirming = false;
    ui.restart.disabled = !usable;
    ui.restart.setAttribute("aria-expanded", String(confirming));
    ui.confirm.hidden = !confirming;
    ui.yes.disabled = !confirming;
  }

  if (typeof h.subscribe === "function") {
    h.subscribe((message) => {
      if (!message || message.type !== SESSION_CLEARED_TYPE) return;
      gate.onCleared(message);
      confirming = false; // whoever reset the session, the question has been answered
      render();
      // The confirmed Yes left the focus on a hidden button (the body, in a browser): once the
      // echo enables Restart again, the focus goes back to it.
      if (refocus && ui && !ui.restart.disabled) {
        refocus = false;
        const active = doc.activeElement;
        if (!active || active === doc.body || ui.confirm.contains(active)) ui.restart.focus();
      }
    });
  }
  if (typeof h.onConnection === "function") {
    h.onConnection((isOpen) => {
      open = Boolean(isOpen);
      if (!open) gate.abandon();
      render();
    });
  }

  const slot =
    h.sessionSlot ||
    (doc && typeof doc.querySelector === "function" ? doc.querySelector("[data-session-slot]") : null);
  if (!doc || typeof doc.createElement !== "function" || !slot) {
    log("session panel: no document or no session slot, so no Restart button is rendered");
    return { gate };
  }
  ui = buildControl(doc, slot);

  function close({ focusRestart = false } = {}) {
    if (!confirming) return;
    confirming = false;
    render();
    if (focusRestart) ui.restart.focus();
  }

  // The header button only ever opens the confirm: a second click on it is not a confirm.
  ui.restart.addEventListener("click", () => {
    if (!open || gate.pending() || confirming) return;
    confirming = true;
    render();
    ui.no.focus(); // the safe answer holds the focus
  });
  ui.yes.addEventListener("click", () => {
    if (!confirming || !open) return;
    confirming = false;
    refocus = gate.reset();
    render();
  });
  ui.no.addEventListener("click", () => close({ focusRestart: true }));
  // Keys typed in the confirm stay in it: `m` there never mutes, and Escape cancels.
  ui.confirm.addEventListener("keydown", (event) => {
    event.stopPropagation();
    if (event.key === "Escape") close({ focusRestart: true });
  });
  doc.addEventListener("keydown", (event) => {
    if (event && event.key === "Escape") close();
  });
  // An outside click cancels; a click on the button or inside the confirm does not.
  const closeIfOutside = (event) => {
    if (!confirming || !event) return;
    const inside = ui.confirm.contains(event.target) || ui.restart.contains(event.target);
    if (!inside) close();
  };
  doc.addEventListener("pointerdown", closeIfOutside);
  doc.addEventListener("click", closeIfOutside);

  render();
  log("session panel mounted: Restart asks before it sends astra.session.reset");
  return { gate };
}
