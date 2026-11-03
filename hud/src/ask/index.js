// The typed-question panel, DOM half (T-0039 criteria 10, 12, 13; ruling A19).
//
// `mount(host)` is the contract `hud/src/panels.js` calls, and it must never throw: a host that
// predates this seam (no document, no send, no subscribe) degrades to nothing rendered and
// nothing sent, leaving every other panel and the socket working (T-0008 criterion 10).
//
// The container is declared by hud/index.html ([data-ask-form] / [data-ask-input] /
// [data-ask-submit]) — this panel only wires it up. hud/src/main.js and hud/src/state.js are not
// modified: the question goes out over host.send and the answer comes back through the existing
// reducer and renderers (criterion 14).
//
// Three compositions with the rest of the page, all found by the A19 review:
//   * main.js listens for a bare `m` on the *document* and stops the capture with it, so every
//     key event that starts in the box has its propagation stopped here — otherwise typing
//     "how many" kills perception mid-question;
//   * the controls ship `disabled` and the form is `method="dialog"`, so before this panel mounts
//     (or on a page where it is missing) a submit neither navigates nor blanks the query string
//     that carries `?ws=` and `?source=`; mount() is what enables them;
//   * the box is cleared only once `send` has resolved — a rejected send keeps the question, which
//     is the one thing the user cannot get back.

import { createAskForm } from "./form.js";

/** main.js's connection line: "connected" while the socket is open (hud/src/ws.js onStatus). */
const STATUS_ID = "conn-status";
const CONNECTED = "connected";

/** Key events that main.js's document-level handler reads (it acts on `m` in keydown). */
const KEY_EVENTS = ["keydown", "keypress", "keyup"];

/**
 * mount(host) — host is { shown, document, send, subscribe } (hud/src/main.js, unmodified).
 *
 * `send` is the raw `socket.send`; the panel validates its own egress in ./form.js against its
 * own whitelist. `subscribe` is called with every decoded astra.* message.
 */
export function mount(host) {
  const h = host && typeof host === "object" ? host : {};
  const doc = h.document;
  const log =
    typeof h.log === "function" ? h.log : (line) => globalThis.console.info("[hud]", line);
  const send = typeof h.send === "function" ? h.send : null;

  // Assigned by the DOM half below; read *at submit time*, never cached, because the socket
  // state the page shows is the socket state the question has to travel on.
  let statusEl = null;
  let pending = null; // the promise of the send this submit made

  function socketIsOpen() {
    if (typeof h.isOpen === "function") return Boolean(h.isOpen());
    // main.js exposes no socket state to panels (criterion 10 forbids touching it), so the page's
    // own connection line is the seam. A page that has no such line cannot be asked, and the
    // send's own rejection stays the backstop.
    if (!statusEl) return true;
    return String(statusEl.textContent || "").trim() === CONNECTED;
  }

  const form = createAskForm({
    send: (message) => {
      const result = send ? send(message) : Promise.reject(new Error("this host has no socket"));
      pending = Promise.resolve(result);
      return pending;
    },
    isOpen: socketIsOpen,
    log,
  });

  // Declared before the subscription because an inbound message may arrive before — or without —
  // any rendering at all.
  let ui = null;

  function render() {
    if (!ui) return;
    const on = form.enabled();
    ui.input.disabled = !on;
    if (ui.submit) ui.submit.disabled = !on;
  }

  if (typeof h.subscribe === "function") {
    h.subscribe((message) => {
      form.onMessage(message);
      render();
    });
  }

  if (!doc || typeof doc.querySelector !== "function") {
    log("ask panel: this host has no document, so no box is wired and nothing is sent");
    return { form };
  }

  statusEl = doc.querySelector(`#${STATUS_ID}`);
  const formEl = doc.querySelector("[data-ask-form]");
  const input = formEl && formEl.querySelector("[data-ask-input]");
  if (!formEl || !input) {
    log(
      "ask panel: this page carries no [data-ask-form]/[data-ask-input], so there is nothing to wire",
    );
    return { form };
  }
  ui = { form: formEl, input, submit: formEl.querySelector("[data-ask-submit]") };

  // A19 (1): main.js stops the capture on a bare `m` anywhere in the document. A question with
  // an `m` in it must not be a mute button, so the box swallows its own key events.
  for (const type of KEY_EVENTS) {
    input.addEventListener(type, (event) => {
      if (event && typeof event.stopPropagation === "function") event.stopPropagation();
    });
  }

  formEl.addEventListener("submit", (event) => {
    // Criterion 12 adversarial: a <form> whose submit is not prevented navigates — the page
    // reloads, the WebSocket drops and the turn the user just opened dies with it.
    if (event && typeof event.preventDefault === "function") event.preventDefault();
    const question = input.value;
    if (form.submit(question)) {
      // A19 (3): cleared only once the socket has really taken it. A rejected send keeps the
      // question in the box, where the user can send it again.
      const sent = pending;
      Promise.resolve(sent).then(
        () => {
          if (input.value === question) input.value = "";
          render();
        },
        () => render(), // the refusal itself is logged by ./form.js
      );
    }
    render();
  });

  render(); // A19 (2): the controls ship disabled; mounting is what enables them
  log("ask panel mounted: type a question and press Enter");
  return { form };
}
