// T-0111c fixture recorder: every event a page receives is appended to window.__log.
//
// One record per event, captured on window/document in the capture phase so that no page
// handler can hide it. Fields (absent when the event has no such property):
//   type, t (performance.now(), ms), target (element id, or "document"/"window"/nodeName),
//   x, y, button, buttons            pointer and mouse events (CSS pixels, clientX/clientY)
//   key, code                        keyboard events
//   ctrlKey, altKey, shiftKey, metaKey
//   value, inputType                 input events (the target's value after the input)
//   scrollX, scrollY                 scroll events (the window's scroll offsets)
//   clip                             copy/cut/paste: clipboardData text/plain
// window.__state() returns a snapshot the tests read after an action.
(() => {
  "use strict";
  const log = [];
  window.__log = log;
  const targetName = (t) => {
    if (t === window) return "window";
    if (t === document) return "document";
    if (t && t.id) return t.id;
    return t && t.nodeName ? t.nodeName.toLowerCase() : "";
  };
  const rec = (e) => {
    const r = { type: e.type, t: Math.round(performance.now() * 1000) / 1000, target: targetName(e.target) };
    if ("clientX" in e) {
      r.x = e.clientX;
      r.y = e.clientY;
      r.button = e.button;
      r.buttons = e.buttons;
    }
    if ("key" in e) {
      r.key = e.key;
      r.code = e.code;
    }
    if ("ctrlKey" in e) {
      r.ctrlKey = e.ctrlKey;
      r.altKey = e.altKey;
      r.shiftKey = e.shiftKey;
      r.metaKey = e.metaKey;
    }
    if (e.type === "input") {
      r.value = e.target && "value" in e.target ? e.target.value : null;
      r.inputType = e.inputType;
    }
    if (e.type === "scroll") {
      r.scrollX = window.scrollX;
      r.scrollY = window.scrollY;
    }
    if (e.clipboardData) r.clip = e.clipboardData.getData("text/plain");
    log.push(r);
  };
  const DOC = [
    "pointerdown", "pointerup", "pointermove", "click", "dblclick", "contextmenu", "auxclick",
    "keydown", "keyup", "input", "submit", "copy", "cut", "paste", "focusin",
  ];
  for (const t of DOC) document.addEventListener(t, rec, true);
  window.addEventListener("scroll", rec, true);
  window.__state = () => {
    const a = document.activeElement;
    return {
      active: a ? targetName(a) : null,
      scrollX: window.scrollX,
      scrollY: window.scrollY,
      selection: String(window.getSelection()),
      selStart: a && "selectionStart" in a ? a.selectionStart : null,
      selEnd: a && "selectionEnd" in a ? a.selectionEnd : null,
      values: Object.fromEntries(
        Array.from(document.querySelectorAll("input[id], textarea[id]")).map((el) => [el.id, el.value]),
      ),
      dialogOpen: Array.from(document.querySelectorAll("dialog")).some((d) => d.open),
    };
  };
})();
