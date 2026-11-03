// T-0213 B-read and C: the Settings panel, below "Context helpers" in the header's Settings popover.
//
// This is the DOM half; ./model.js decides what each row says and ./gate.js is the only way out.
// The panel renders the last astra.settings.state the page received (A2r-1): the catalog arrives
// on the hello reply only, so the last catalog is kept and every later state updates the values.
// Every string is set with textContent.
//
// T-0213 C (A2-apply's page half): the value control of a next_turn / next_turn_cold row is
// enabled; a change sends one astra.settings.set {rev, changes, stop_turn: false} through the gate
// and the row shows the server's echo, never the click. A change the server keeps pending (a turn
// is in flight) reads "applies after this turn" with "Stop and apply", which resends it with
// stop_turn: true. A refusal is shown on its row. relaunch / not_here / fixed rows stay read-only
// (B-2, R-10..R-12). Under "Guidance and prompt": the instructions box (C-1, C-3, C-4) and the
// read-only system message (C-2). Typing in the box sends nothing; only Apply and Reset do.
// T-0225: under the box, a prompt library (./prompts.js) fills it; it sends nothing itself.
//
// `mount(host)` never throws: without a document or a slot nothing is rendered; without
// `subscribe` the panel says it is waiting; without `send` every control stays disabled.
// host is { document, slot, subscribe, send, log, storage? }; storage defaults to the page's
// localStorage (T-0225's saved prompts) and may be null.

import {
  INSTRUCTIONS_LABEL,
  PENDING_TEXT,
  STOP_AND_APPLY_TEXT,
  SYSTEM_NOTE,
  SYSTEM_PREVIEW_SUMMARY,
  changeFor,
  counterText,
  codePointCount,
  groupCatalog,
  itemView,
  refusalsFor,
} from "./model.js";
import { SETTINGS_STATE_TYPE, createSettingsGate } from "./gate.js";
import {
  EXPORT_FILENAME,
  MAX_IMPORT_BYTES,
  MAX_PROMPT_CHARS,
  deletePrompt,
  exportPrompts,
  importConflicts,
  loadPrompts,
  mergeImport,
  parseImport,
  savePrompt,
  validName,
} from "./prompts.js";

export { SETTINGS_STATE_TYPE };
/** B-3, verbatim: the title of a control that cannot send (no socket in this host). */
export const DISABLED_TITLE = "Changes from the panel arrive with the next update";
export const SETTINGS_HEADING = "Settings";
export const SETTINGS_SENTENCE =
  "What Gearvis runs with now, where each value came from, and what the measured runs say about " +
  "it. Rows that need a relaunch show what to add to your session command.";
const WAITING_LINE = "Waiting for the session to report its settings.";
const COPIED_MS = 2000;
const INSTRUCTIONS_KEY = "user_instructions";
const INSTRUCTIONS_GROUP = "Guidance and prompt";
const NO_PREVIEW_LINE = "The session did not send its system message.";
/** Key events main.js's document-level handlers could read; typing in the panel keeps them here. */
const KEY_EVENTS = ["keydown", "keypress", "keyup"];

/** The instructions row's editor is the box under its group (C-1), not a row control. */
const NO_CONTROL_KEYS = new Set([INSTRUCTIONS_KEY]);

function el(doc, tag, className, text) {
  const node = doc.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function slotOf(host) {
  const slot = typeof host.slot === "function" ? host.slot() : host.slot;
  return slot && typeof slot.appendChild === "function" ? slot : null;
}

const hasValue = (v) => v !== null && v !== undefined;

/** The value text of one allowed value, as the row shows it. */
function itemViewValue(value) {
  return itemView({}, { value }).valueText;
}

/** Keys typed into a panel control stay in it (C-4: `m` in the box never reaches a page handler).
 *  Escape still bubbles to the popover, which closes on it. */
function keepKeys(node) {
  for (const type of KEY_EVENTS) {
    node.addEventListener(type, (event) => {
      if (event && event.key === "Escape") return;
      if (event && typeof event.stopPropagation === "function") event.stopPropagation();
    });
  }
}

/** A row's value control: a switch for an on/off row, a select for a row with a value list. It
 *  shows the value the row will have (the pending one when a change waits); a change calls
 *  `onChange(value)` with the allowed value itself (never its option text). Without a send path it
 *  is disabled with DISABLED_TITLE. */
function buildControl(doc, view, shown, canSend, onChange) {
  const isBool = typeof shown === "boolean";
  let control;
  let values = [];
  if (isBool) {
    control = doc.createElement("input");
    control.type = "checkbox";
    control.checked = shown;
  } else {
    control = doc.createElement("select");
    values = view.values.length ? view.values.slice() : [shown];
    if (!values.some((v) => v === shown)) values.push(shown);
    values.forEach((value) => {
      const option = el(doc, "option", "", itemViewValue(value));
      option.value = String(value); // the catalog value; the change maps it back to the typed one
      if (value === shown) option.selected = true;
      control.appendChild(option);
    });
  }
  control.className = "settings-control";
  control.setAttribute("data-setting-control", "");
  control.setAttribute("aria-label", view.label);
  if (!canSend) {
    control.disabled = true;
    control.title = DISABLED_TITLE;
    return control;
  }
  control.title =
    view.kind === "next_turn_cold" ? "Applies at the next turn; that reply is slower" : "Applies at the next turn";
  keepKeys(control);
  control.addEventListener("change", () => {
    if (isBool) {
      onChange(Boolean(control.checked));
      return;
    }
    const chosen = values.find((v) => String(v) === control.value);
    if (chosen !== undefined) onChange(chosen);
  });
  return control;
}

function field(doc, list, term, hook, text) {
  const dt = el(doc, "dt", "", term);
  const dd = el(doc, "dd", "", text);
  dd.setAttribute(hook, "");
  list.append(dt, dd);
}

function copyText(doc, text, code) {
  const clip = globalThis.navigator && globalThis.navigator.clipboard;
  if (clip && typeof clip.writeText === "function") return Promise.resolve(clip.writeText(text));
  // No async clipboard (an insecure origin): select the command and use the legacy copy.
  const selection = globalThis.getSelection ? globalThis.getSelection() : null;
  if (selection && typeof doc.createRange === "function") {
    const range = doc.createRange();
    range.selectNodeContents(code);
    selection.removeAllRanges();
    selection.addRange(range);
  }
  const ok = typeof doc.execCommand === "function" && doc.execCommand("copy");
  return ok ? Promise.resolve() : Promise.reject(new Error("the browser refused to copy"));
}

function stopAndApplyButton(doc, hook, onClick) {
  const button = el(doc, "button", "settings-stop-apply", STOP_AND_APPLY_TEXT);
  button.type = "button";
  button.setAttribute(hook, "");
  button.title = "Stop the turn in flight, then apply the waiting change";
  button.addEventListener("click", onClick);
  return button;
}

/** ctx: { canSend, change(key, value), stopAndApply(), reasons(key) } */
function buildRow(doc, entry, item, openKeys, log, ctx) {
  const view = itemView(entry, item);
  const row = el(doc, "li", "settings-row");
  row.setAttribute("data-setting-key", view.key);
  row.setAttribute("data-setting-readonly", String(view.readOnly));
  const pending = item && hasValue(item.pending) ? item.pending : null;

  const head = el(doc, "div", "settings-head");
  const label = el(doc, "span", "settings-label", view.label);
  label.setAttribute("data-setting-label", "");
  const value = el(doc, "span", "settings-value", view.valueText);
  value.setAttribute("data-setting-value", "");
  head.append(label, value);
  if (!view.readOnly && !NO_CONTROL_KEYS.has(view.key)) {
    const shown = pending !== null ? pending : view.value;
    head.append(buildControl(doc, view, shown, ctx.canSend, (v) => ctx.change(view.key, v)));
  }
  row.append(head);

  const tags = el(doc, "div", "settings-tags");
  const source = el(doc, "span", "settings-source", view.sourceText);
  source.setAttribute("data-setting-source", view.source);
  const chip = el(doc, "span", "settings-kind", view.chip);
  chip.setAttribute("data-setting-kind", view.kind);
  tags.append(source, chip);
  if (pending !== null) {
    // the instructions' waiting text is in their box, never repeated here
    const text = view.key === INSTRUCTIONS_KEY ? PENDING_TEXT : `${PENDING_TEXT}: ${itemViewValue(pending)}`;
    const note = el(doc, "span", "settings-pending", text);
    note.setAttribute("data-setting-pending", "");
    tags.append(note);
    if (ctx.canSend && !view.readOnly && view.key !== INSTRUCTIONS_KEY) {
      tags.append(stopAndApplyButton(doc, "data-setting-stop-apply", () => ctx.stopAndApply()));
    }
  }
  row.append(tags);

  const reasons = view.key === INSTRUCTIONS_KEY ? [] : ctx.reasons(view.key);
  if (reasons.length) {
    const refusal = el(doc, "p", "settings-refusal", `Not changed: ${reasons.join("; ")}`);
    refusal.setAttribute("data-setting-refusal", "");
    refusal.setAttribute("role", "status");
    row.append(refusal);
  }

  const what = el(doc, "p", "settings-what", view.what);
  what.setAttribute("data-setting-what", "");
  row.append(what);

  const details = el(doc, "details", "settings-details");
  details.open = openKeys.has(view.key);
  details.addEventListener("toggle", () => {
    if (details.open) openKeys.add(view.key);
    else openKeys.delete(view.key);
  });
  details.append(el(doc, "summary", "", "When it helps, and the evidence"));
  const list = el(doc, "dl", "settings-facts");
  field(doc, list, "Useful when", "data-setting-useful", view.useful_when);
  field(doc, list, "Not useful when", "data-setting-not-useful", view.not_useful_when);
  field(doc, list, "Evidence", "data-setting-evidence", view.evidence);
  field(doc, list, "Cost", "data-setting-cost", view.cost);
  details.append(list);
  row.append(details);

  if (view.relaunch !== null) {
    const box = el(doc, "div", "settings-relaunch");
    const label = el(doc, "span", "settings-relaunch-label", view.relaunchLabel);
    label.setAttribute("data-setting-relaunch-label", "");
    const code = el(doc, "code", "", view.relaunch);
    code.setAttribute("data-setting-relaunch", "");
    const copy = el(doc, "button", "settings-copy", "Copy");
    copy.type = "button";
    copy.setAttribute("data-setting-copy", "");
    copy.title = "Copy the flags to add to your session command";
    const status = el(doc, "span", "settings-copied");
    status.setAttribute("aria-live", "polite");
    let timer = null;
    copy.addEventListener("click", () => {
      copyText(doc, view.relaunch, code).then(
        () => {
          status.textContent = "Copied";
          if (timer !== null) globalThis.clearTimeout(timer);
          timer = globalThis.setTimeout(() => {
            status.textContent = "";
            timer = null;
          }, COPIED_MS);
        },
        (err) => {
          status.textContent = "Could not copy; select the command instead";
          log(`settings: copy failed: ${err && err.message}`);
        },
      );
    });
    box.append(label, code, copy, status);
    row.append(box);
  }
  return row;
}

function smallButton(doc, className, text, hook) {
  const button = el(doc, "button", className, text);
  button.type = "button";
  button.setAttribute(hook, "");
  return button;
}

/** T-0225: the prompt library under the box. Choosing a prompt only fills the box (`fill`); saving
 *  and deleting touch the page's storage only. Replacing a saved prompt and deleting one ask first,
 *  in the page. Nothing here sends anything. */
function buildLibrary(doc, { storage, fill, currentText, maxChars, log }) {
  const wrap = el(doc, "div", "settings-prompts");
  wrap.setAttribute("data-settings-prompts-library", "");
  const pickRow = el(doc, "div", "settings-prompts-row");
  const pickLabel = el(doc, "label", "settings-prompts-label", "Prompt");
  const select = doc.createElement("select");
  select.id = "settings-prompts-select";
  select.className = "settings-prompts-select";
  select.setAttribute("data-settings-prompts", "");
  pickLabel.htmlFor = select.id;
  keepKeys(select);
  pickRow.append(pickLabel, select);

  const saveRow = el(doc, "div", "settings-prompts-row");
  const name = doc.createElement("input");
  name.type = "text";
  name.className = "settings-prompts-name";
  name.placeholder = "Name for this prompt";
  name.setAttribute("aria-label", "Name for this prompt");
  name.setAttribute("data-settings-prompt-name", "");
  keepKeys(name);
  const save = smallButton(doc, "settings-prompts-save", "Save as…", "data-settings-prompt-save");
  save.title = "Save the text in the box under this name, in this browser";
  const del = smallButton(doc, "settings-prompts-delete", "Delete", "data-settings-prompt-delete");
  del.title = "Delete the chosen saved prompt from this browser";
  saveRow.append(name, save, del);

  const confirm = el(doc, "div", "settings-prompts-confirm");
  confirm.setAttribute("data-settings-prompt-confirm", "");
  confirm.setAttribute("role", "alertdialog");
  confirm.hidden = true;
  const question = el(doc, "span", "settings-prompts-question");
  question.setAttribute("data-settings-prompt-confirm-text", "");
  const yes = smallButton(doc, "settings-prompts-yes", "", "data-settings-prompt-confirm-yes");
  const no = smallButton(doc, "settings-prompts-no", "Cancel", "data-settings-prompt-confirm-no");
  confirm.append(question, yes, no);

  const status = el(doc, "p", "settings-prompts-status");
  status.setAttribute("data-settings-prompt-status", "");
  status.setAttribute("role", "status");
  status.hidden = true;
  const note = el(doc, "p", "settings-prompts-note");
  note.setAttribute("data-settings-prompts-note", "");
  note.hidden = true;
  // criterion 7: the saved prompts to and from a file, in the page only
  const fileRow = el(doc, "div", "settings-prompts-row");
  const exportButton = smallButton(doc, "settings-prompts-export", "Export", "data-settings-prompts-export");
  exportButton.title = `Download your saved prompts as ${EXPORT_FILENAME}`;
  const importLabel = el(doc, "label", "settings-prompts-import-label", "Import");
  const importInput = doc.createElement("input");
  importInput.type = "file";
  importInput.accept = "application/json,.json";
  importInput.id = "settings-prompts-import";
  importInput.className = "settings-prompts-import";
  importInput.setAttribute("data-settings-prompts-import", "");
  importInput.title = "Add saved prompts from a prompts file";
  importLabel.htmlFor = importInput.id;
  keepKeys(importInput);
  fileRow.append(exportButton, importLabel, importInput);
  wrap.append(pickRow, saveRow, fileRow, confirm, status, note);

  let prompts = [];
  let pendingAction = null; // the action the in-page confirmation would carry out
  let pendingCancel = null; // and what its Cancel does
  let enabled = true;

  const say = (text) => {
    status.textContent = text;
    status.hidden = !text;
  };
  const chosen = () => prompts.find((p) => p.name === select.value && select.value !== "") || null;
  const syncButtons = () => {
    const p = chosen();
    save.disabled = !enabled;
    del.disabled = !enabled || !p || p.builtin;
    select.disabled = !enabled;
    name.disabled = !enabled;
    yes.disabled = !enabled;
    no.disabled = !enabled;
    exportButton.disabled = !enabled;
    importInput.disabled = !enabled;
  };
  const showNote = (text) => {
    note.textContent = text || "";
    note.hidden = !text;
  };
  const closeConfirm = () => {
    pendingAction = null;
    pendingCancel = null;
    confirm.hidden = true;
    question.textContent = "";
  };
  const ask = (text, yesText, action, onCancel = null) => {
    pendingAction = action;
    pendingCancel = onCancel;
    question.textContent = text;
    yes.textContent = yesText;
    confirm.hidden = false;
  };

  /** Re-read the store and rebuild the list, keeping `keep` chosen when it is still there. */
  function reload(keep) {
    const { prompts: list, error } = loadPrompts(storage);
    prompts = list;
    showNote(error);
    const builtins = el(doc, "optgroup");
    builtins.label = "Built-in";
    const saved = el(doc, "optgroup");
    saved.label = "Saved in this browser";
    for (const p of prompts) {
      const option = el(doc, "option", "", p.name);
      option.value = p.name;
      option.setAttribute("data-builtin", String(p.builtin));
      (p.builtin ? builtins : saved).appendChild(option);
    }
    const groups = [builtins];
    if (saved.children.length) groups.push(saved);
    select.replaceChildren(...groups);
    // nothing chosen until the user picks (so even Default fires a change); no placeholder option
    if (prompts.some((p) => p.name === keep)) select.value = keep;
    else select.selectedIndex = -1;
    syncButtons();
  }

  select.addEventListener("change", () => {
    closeConfirm();
    say("");
    const p = chosen();
    syncButtons();
    if (!p) return;
    if (!p.builtin) name.value = p.name;
    fill(p.text); // fills the box only; Apply sends it
  });

  function doSave() {
    closeConfirm();
    const typed = name.value.trim();
    const bad = validName(typed);
    if (bad) return say(bad);
    const text = currentText();
    const cap = maxChars() > 0 ? Math.min(maxChars(), MAX_PROMPT_CHARS) : MAX_PROMPT_CHARS;
    const n = codePointCount(text);
    if (n > cap) return say(`Not saved: the text is too long (${n} / ${cap} characters)`);
    const write = () => {
      closeConfirm();
      const result = savePrompt(storage, typed, text);
      reload(result.ok ? typed : select.value);
      if (result.ok) {
        name.value = typed;
        say(`Saved "${typed}" in this browser`);
        log("settings: a prompt was saved in this browser");
      } else say(`Not saved: ${result.reason}`);
    };
    if (prompts.some((p) => !p.builtin && p.name === typed)) {
      ask(`Replace the saved prompt "${typed}"?`, "Replace", write);
    } else write();
  }

  save.addEventListener("click", doSave);
  name.addEventListener("keydown", (event) => {
    if (event && event.key === "Enter") {
      if (typeof event.preventDefault === "function") event.preventDefault();
      doSave();
    }
  });
  del.addEventListener("click", () => {
    closeConfirm();
    const p = chosen();
    if (!p || p.builtin) return say("Built-in prompts cannot be deleted");
    ask(`Delete the saved prompt "${p.name}"?`, "Delete", () => {
      closeConfirm();
      const result = deletePrompt(storage, p.name);
      reload(result.ok ? "" : p.name);
      if (result.ok) {
        say(`Deleted "${p.name}"`);
        log("settings: a saved prompt was deleted");
      } else say(`Not deleted: ${result.reason}`);
    });
  });
  yes.addEventListener("click", () => {
    if (pendingAction) pendingAction();
  });
  no.addEventListener("click", () => {
    const cancel = pendingCancel;
    closeConfirm();
    say("");
    if (cancel) cancel();
  });

  exportButton.addEventListener("click", () => {
    closeConfirm();
    const url = globalThis.URL;
    if (typeof globalThis.Blob !== "function" || !url || typeof url.createObjectURL !== "function") {
      return say("Not exported: this browser cannot make a file");
    }
    const href = url.createObjectURL(
      new globalThis.Blob([exportPrompts(prompts)], { type: "application/json" }),
    );
    const link = doc.createElement("a");
    link.href = href;
    link.download = EXPORT_FILENAME;
    link.hidden = true;
    wrap.appendChild(link); // inside the popover: a click outside it would close it
    link.click();
    link.remove();
    globalThis.setTimeout(() => url.revokeObjectURL(href), 1000);
    const n = prompts.filter((p) => !p.builtin).length;
    say(`Exported ${n} saved prompt${n === 1 ? "" : "s"} to ${EXPORT_FILENAME}`);
  });

  /** Merge the file's entries; `replace` was confirmed when any name already exists. */
  function finishImport(entries, replace) {
    closeConfirm();
    const result = mergeImport(storage, entries, { replace });
    reload(select.value);
    if (result.error) {
      showNote(result.error);
      return say("Not imported: nothing changed");
    }
    say(`Imported: ${result.added} added, ${result.replaced} replaced, ${result.skipped} skipped.`);
    log("settings: saved prompts were imported");
  }

  importInput.addEventListener("change", () => {
    closeConfirm();
    say("");
    const file = importInput.files && importInput.files[0];
    if (!file) return;
    reload(select.value);
    if (typeof file.size === "number" && file.size > MAX_IMPORT_BYTES) {
      importInput.value = ""; // the same file can be chosen again
      showNote("The file is larger than 256 KB");
      return say("Not imported: nothing changed");
    }
    Promise.resolve()
      .then(() => file.text())
      .then(
        (text) => {
          importInput.value = "";
          const { entries, error } = parseImport(text);
          if (error) {
            showNote(error);
            return say("Not imported: nothing changed");
          }
          const clashes = importConflicts(storage, entries);
          if (clashes === 0) return finishImport(entries, false);
          ask(
            `${clashes} imported prompt${clashes === 1 ? " has a name" : "s have names"} you already ` +
              "saved. Replace them?",
            "Replace",
            () => finishImport(entries, true),
            () => say("Import cancelled: nothing changed"),
          );
        },
        () => {
          importInput.value = "";
          showNote("The file could not be read");
          say("Not imported: nothing changed");
        },
      );
  });

  reload("");
  return {
    wrap,
    setEnabled(on) {
      enabled = Boolean(on);
      if (!enabled) closeConfirm();
      syncButtons();
    },
  };
}

/** The page's own storage, or null when the browser blocks it (reading it can throw). */
function pageStorage(host) {
  if (host.storage !== undefined) return host.storage;
  try {
    return globalThis.localStorage || null;
  } catch {
    return null;
  }
}

/** C-1/C-3/C-4: the instructions box. Built once and kept across states, so a draft, its caret
 *  and its focus survive every echo. Typing sends nothing; Apply and Reset each send one set. */
function buildInstructions(doc, { onApply, onStopAndApply, storage, log }) {
  const box = el(doc, "div", "settings-instructions");
  box.setAttribute("data-settings-instructions", "");
  box.hidden = true;
  const label = el(doc, "label", "settings-instructions-label", INSTRUCTIONS_LABEL);
  const text = doc.createElement("textarea");
  text.id = "settings-instructions-text";
  text.className = "settings-instructions-text";
  text.rows = 5;
  text.setAttribute("data-settings-instructions-text", "");
  label.htmlFor = text.id;
  keepKeys(text);
  const count = el(doc, "span", "settings-instructions-count");
  count.setAttribute("data-settings-instructions-count", "");
  count.setAttribute("aria-live", "polite");
  const apply = el(doc, "button", "settings-instructions-apply", "Apply");
  apply.type = "button";
  apply.setAttribute("data-settings-instructions-apply", "");
  const reset = el(doc, "button", "settings-instructions-reset", "Reset to default");
  reset.type = "button";
  reset.setAttribute("data-settings-instructions-reset", "");
  const actions = el(doc, "div", "settings-instructions-actions");
  actions.append(count, apply, reset);
  const pending = el(doc, "p", "settings-pending");
  pending.setAttribute("data-settings-instructions-pending", "");
  pending.hidden = true;
  // its Stop and apply exists only while the instructions wait (one such button per waiting change)
  const stop = stopAndApplyButton(doc, "data-settings-instructions-stop-apply", () => onStopAndApply());
  pending.append(el(doc, "span", "", PENDING_TEXT));
  const refusal = el(doc, "p", "settings-refusal");
  refusal.setAttribute("data-settings-instructions-refusal", "");
  refusal.setAttribute("role", "status");
  refusal.hidden = true;
  const state = { max: 0, dirty: false };
  const recount = () => {
    count.textContent = counterText(text.value, state.max);
    count.setAttribute("data-over", String(codePointCount(text.value) > state.max));
  };
  // T-0225: a chosen prompt is a draft like typed text; only Apply sends it
  const library = buildLibrary(doc, {
    storage,
    fill(value) {
      text.value = value;
      state.dirty = true;
      recount();
    },
    currentText: () => text.value,
    maxChars: () => state.max,
    log,
  });
  box.append(label, text, actions, pending, refusal, library.wrap);
  text.addEventListener("input", () => {
    state.dirty = true;
    recount();
  });
  apply.addEventListener("click", () => {
    state.dirty = true; // the box keeps this text until the server's state carries it (C-3)
    onApply(text.value);
  });
  reset.addEventListener("click", () => {
    text.value = "";
    state.dirty = true;
    recount();
    onApply("");
  });

  return {
    box,
    /** view: { present, instructions, max, canSend, pending, reasons } */
    update(view) {
      box.hidden = !view.present;
      state.max = view.max;
      if (state.dirty && view.reasons.length === 0 && !view.pending && text.value === view.instructions) {
        state.dirty = false; // the server applied what the box holds
      }
      if (!state.dirty) text.value = view.instructions;
      recount();
      // a session that cannot change them gets no editable box (R-24: enabled by `changeable`)
      text.disabled = !view.canSend;
      library.setEnabled(view.canSend);
      for (const b of [apply, reset]) {
        b.disabled = !view.canSend;
        if (view.canSend) b.removeAttribute("title");
        else b.title = DISABLED_TITLE;
      }
      pending.hidden = !view.pending;
      if (view.pending && view.canSend) {
        if (stop.parentNode !== pending) pending.append(stop);
      } else if (stop.parentNode) stop.remove();
      refusal.hidden = view.reasons.length === 0;
      refusal.textContent = view.reasons.length ? `Not applied: ${view.reasons.join("; ")}` : "";
    },
  };
}

/** C-2: the server's assembled system message, read-only text, behind a disclosure. */
function buildPreview(doc) {
  const wrap = el(doc, "div", "settings-system");
  wrap.setAttribute("data-settings-system", "");
  wrap.hidden = true;
  const note = el(doc, "p", "settings-system-note", SYSTEM_NOTE);
  note.setAttribute("data-settings-system-note", "");
  const details = el(doc, "details", "settings-details settings-system-details");
  details.setAttribute("data-settings-system-details", "");
  details.append(el(doc, "summary", "", SYSTEM_PREVIEW_SUMMARY));
  const pre = el(doc, "pre", "settings-system-preview");
  pre.setAttribute("data-settings-system-preview", "");
  details.append(pre);
  wrap.append(note, details);
  return {
    wrap,
    update(preview) {
      wrap.hidden = false;
      const text = typeof preview === "string" ? preview : NO_PREVIEW_LINE;
      if (pre.textContent !== text) pre.textContent = text;
    },
  };
}

/** Make `parent`'s children exactly `nodes`, in order, without detaching a node that is already in
 *  place: a detached node loses the focus, so the kept instructions row must never move. */
function syncChildren(parent, nodes) {
  const keep = new Set(nodes);
  for (const child of Array.from(parent.children)) if (!keep.has(child)) child.remove();
  let at = parent.firstElementChild;
  for (const node of nodes) {
    if (node === at) {
      at = at.nextElementSibling;
      continue;
    }
    parent.insertBefore(node, at);
  }
}

function buildPanel(doc, slot) {
  const panel = el(doc, "section", "pane settings-panel");
  panel.id = "settings-panel";
  panel.setAttribute("data-settings-panel", "");
  panel.append(el(doc, "h2", "", SETTINGS_HEADING));
  panel.append(el(doc, "p", "settings-explain", SETTINGS_SENTENCE));
  const empty = el(doc, "p", "settings-empty", WAITING_LINE);
  const groups = el(doc, "div", "settings-groups");
  const footer = el(doc, "p", "settings-footer");
  footer.hidden = true;
  panel.append(empty, groups, footer);
  slot.appendChild(panel);
  return { panel, empty, groups, footer };
}

/** mount(host) — host is { document, slot, subscribe, send, log, storage? }; never throws. */
export function mount(host) {
  const h = host && typeof host === "object" ? host : {};
  const doc = h.document;
  const log = typeof h.log === "function" ? h.log : () => {};
  const slot = slotOf(h);
  if (!doc || typeof doc.createElement !== "function" || !slot) {
    log("settings panel: this host has no document or slot, so nothing is rendered");
    return {};
  }
  const canSend = typeof h.send === "function";
  const gate = createSettingsGate({ send: canSend ? (m) => h.send(m) : undefined, log });
  const ui = buildPanel(doc, slot);
  const openKeys = new Set(); // the rows whose evidence the user opened survive a new state
  const sections = new Map(); // group -> { section, list }: kept, so the box under one survives
  let catalog = null; // the last catalog (the hello's); later states carry none
  let last = null;

  const itemsOf = () => (last && Array.isArray(last.items) ? last.items : []);

  /** A2a-3: resend every waiting change in one set, with stop_turn: true. */
  function stopAndApply() {
    const changes = {};
    for (const item of itemsOf()) {
      if (item && typeof item.key === "string" && hasValue(item.pending)) changes[item.key] = item.pending;
    }
    if (gate.set(changes, { stopTurn: true })) log("settings: stop and apply sent");
  }

  const instructions = buildInstructions(doc, {
    onApply: (text) => {
      // the text itself is never logged (Q5): only that a set went out
      if (gate.set({ [INSTRUCTIONS_KEY]: text })) log("settings: instructions change sent");
    },
    onStopAndApply: stopAndApply,
    storage: pageStorage(h),
    log,
  });
  const preview = buildPreview(doc);

  const ctx = {
    canSend,
    change(key, value) {
      if (gate.set(changeFor(key, value, itemsOf()))) log(`settings: change to ${key} sent`);
    },
    stopAndApply,
    reasons: (key) => refusalsFor(last, key, gate.lastKeys()),
  };

  function sectionFor(group) {
    let s = sections.get(group);
    if (!s) {
      const section = el(doc, "section", "settings-group");
      section.setAttribute("data-settings-group", group);
      section.append(el(doc, "h3", "", group));
      const list = el(doc, "ul", "settings-rows");
      section.append(list);
      if (group === INSTRUCTIONS_GROUP) section.append(preview.wrap);
      s = { section, list };
      sections.set(group, s);
    }
    return s;
  }

  // C-1: the instructions box lives in the user_instructions row. That row element is kept across
  // states (only its other parts are rebuilt), so the box's draft, caret and focus survive an echo.
  let instructionsRow = null;
  function rowFor(entry, item) {
    const row = buildRow(doc, entry, item, openKeys, log, ctx);
    if (entry.key !== INSTRUCTIONS_KEY) return row;
    if (!instructionsRow) {
      instructionsRow = row;
      row.append(instructions.box);
      return row;
    }
    for (const child of Array.from(instructionsRow.childNodes)) {
      if (child !== instructions.box) child.remove();
    }
    instructionsRow.setAttribute("data-setting-readonly", row.getAttribute("data-setting-readonly"));
    for (const child of Array.from(row.childNodes)) instructionsRow.insertBefore(child, instructions.box);
    return instructionsRow;
  }

  function render() {
    if (!last || !catalog) {
      ui.empty.hidden = false;
      ui.groups.replaceChildren();
      sections.clear();
      instructionsRow = null;
      ui.footer.hidden = true;
      return;
    }
    const items = new Map();
    for (const item of itemsOf()) {
      if (item && typeof item.key === "string") items.set(item.key, item);
    }
    const order = groupCatalog(catalog).map(({ group, entries }) => {
      const s = sectionFor(group);
      syncChildren(s.list, entries.map((e) => rowFor(e, items.get(e.key))));
      return s.section;
    });
    // Sections already in place stay put: moving one would blur a focused instructions box.
    syncChildren(ui.groups, order);
    for (const [group, s] of sections) {
      if (!order.includes(s.section)) sections.delete(group);
    }
    ui.empty.hidden = true;

    const item = items.get(INSTRUCTIONS_KEY);
    instructions.update({
      present: Boolean(item),
      instructions: typeof last.instructions === "string" ? last.instructions : "",
      max: Number.isInteger(last.instructions_max) ? last.instructions_max : 0,
      canSend: canSend && Boolean(item) && item.changeable === true,
      pending: Boolean(item) && hasValue(item.pending),
      reasons: ctx.reasons(INSTRUCTIONS_KEY),
    });
    preview.update(last.system_preview);

    const sha = typeof last.system_sha12 === "string" ? last.system_sha12 : "";
    ui.footer.textContent = `Settings revision ${last.rev}${sha ? `, system message ${sha}` : ""}`;
    ui.footer.hidden = false;
  }

  if (typeof h.subscribe === "function") {
    h.subscribe((message) => {
      if (!message || message.type !== SETTINGS_STATE_TYPE) return;
      if (Array.isArray(message.catalog)) catalog = message.catalog;
      last = message;
      gate.onState(message);
      render();
    });
  }
  render();
  log(`settings panel mounted (${canSend ? "changes go to the session" : "read-only: this host has no socket"})`);
  return {};
}
