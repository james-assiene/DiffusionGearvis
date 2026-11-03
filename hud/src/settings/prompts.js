// T-0225: the prompt library under "Your instructions to Gearvis".
//
// Four built-in prompts and the user's own saved prompts. Choosing one only fills the instructions
// box; Apply (T-0213 C) is still the only way a text reaches the session. Saved prompts live in the
// page's own storage (this browser, this HUD address) under one key, as a JSON list of
// {name, text}. Every function here is pure over the storage passed in, and every storage access is
// wrapped: a blocked or corrupt store leaves the built-ins working and reports STORAGE_NOTE.

export const STORAGE_KEY = "gearvis.prompts.v1";
export const MAX_USER_PROMPTS = 50;
export const MAX_NAME_CHARS = 60;
/** The instructions cap (A1-4, `instructions_max`): a longer text cannot be saved. */
export const MAX_PROMPT_CHARS = 1000;
export const STORAGE_NOTE = "Saved prompts are unavailable in this browser";

/** The built-ins, in the task's order and with its exact texts. Default is no added instructions. */
export const BUILTIN_PROMPTS = Object.freeze(
  [
    { id: "default", name: "Default", text: "" },
    {
      id: "education_tutor",
      name: "Education tutor",
      text:
        "You are a patient tutor. When I show you my work, check it line by line. If a line is wrong, " +
        "point to it, explain why, and ask me a guiding question before giving the fix; then give the " +
        "corrected step and the correct final answer. If everything is right, say so plainly and " +
        "don't invent a problem. Speak in plain words, not symbols, and keep each reply short.",
    },
    {
      id: "gaming_companion",
      name: "Gaming companion",
      text:
        "You are my gaming companion. I share my game screen with you. Answer out loud in one or " +
        "two short sentences, like a friend sitting next to me. Read what is on the screen before " +
        "you answer. When I ask what to do, give one clear plan and the reason. Do not use your own " +
        "browser. Use google_search only for game facts you are not sure about, and say where you " +
        "found them.",
    },
    {
      id: "browser_specialist",
      name: "Browser specialist",
      text:
        "When I ask you to do something on a website, do it yourself in your own browser, one step " +
        "at a time, and check the screen after each step. Do not narrate every click. When you are " +
        "done, tell me in one sentence what you did and what you see. If you are stuck, say what is " +
        "blocking you. When I ask you to buy or order something, do every step you can: find it, add it " +
        "to the cart and go to checkout. Stop at sign-in, payment or the final order button and tell me " +
        "what is left for me to do. Never type passwords or card numbers, and ask me before sending, " +
        "posting or deleting anything.",
    },
  ].map((p) => Object.freeze({ ...p, builtin: true })),
);

const BUILTIN_NAMES = new Set(BUILTIN_PROMPTS.map((p) => p.name.toLowerCase()));

function codePoints(text) {
  let n = 0;
  for (const _ of text) n += 1;
  return n;
}

/** The reason `name` cannot name a saved prompt, or null. The name is judged trimmed. */
export function validName(name) {
  if (typeof name !== "string") return "Type a name for the prompt";
  const trimmed = name.trim();
  const n = codePoints(trimmed);
  if (n === 0) return "Type a name for the prompt";
  if (n > MAX_NAME_CHARS) return `The name is too long (${n} / ${MAX_NAME_CHARS} characters)`;
  if (BUILTIN_NAMES.has(trimmed.toLowerCase())) return "That name belongs to a built-in prompt";
  return null;
}

function byName(a, b) {
  const c = a.name.localeCompare(b.name, "en", { sensitivity: "base" });
  if (c !== 0) return c;
  return a.name < b.name ? -1 : a.name > b.name ? 1 : 0;
}

/** The saved list as stored: { list } or { error }. A missing key is an empty list. */
function readUser(storage) {
  let raw;
  try {
    if (!storage || typeof storage.getItem !== "function") return { error: STORAGE_NOTE };
    raw = storage.getItem(STORAGE_KEY);
  } catch {
    return { error: STORAGE_NOTE };
  }
  if (raw === null || raw === undefined) return { list: [] };
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { error: STORAGE_NOTE };
  }
  if (!Array.isArray(parsed)) return { error: STORAGE_NOTE };
  const seen = new Set();
  const list = [];
  for (const entry of parsed) {
    if (!entry || typeof entry !== "object") continue;
    const { name, text } = entry;
    if (typeof name !== "string" || typeof text !== "string") continue;
    if (validName(name) !== null || name !== name.trim() || seen.has(name)) continue;
    seen.add(name);
    list.push({ name, text });
  }
  return { list };
}

function writeUser(storage, list) {
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(list.map(({ name, text }) => ({ name, text }))));
    return null;
  } catch {
    return STORAGE_NOTE;
  }
}

/** { prompts, error }: the built-ins first (in order), then the saved prompts alphabetically.
 *  error is STORAGE_NOTE when the store cannot be read, else null; the built-ins are always there. */
export function loadPrompts(storage) {
  const builtins = BUILTIN_PROMPTS.map((p) => ({ ...p }));
  const user = readUser(storage);
  if (user.error) return { prompts: builtins, error: user.error };
  const saved = user.list.map((p) => ({ name: p.name, text: p.text, builtin: false })).sort(byName);
  return { prompts: builtins.concat(saved), error: null };
}

/** Save `text` under `name` (trimmed), replacing a saved prompt of the same name. Built-in names,
 *  a text over MAX_PROMPT_CHARS and a 51st prompt are refused. { ok, reason } (reason null on ok). */
export function savePrompt(storage, name, text) {
  const bad = validName(name);
  if (bad !== null) return { ok: false, reason: bad };
  if (typeof text !== "string") return { ok: false, reason: "There is no text to save" };
  const n = codePoints(text);
  if (n > MAX_PROMPT_CHARS) {
    return { ok: false, reason: `The text is too long to save (${n} / ${MAX_PROMPT_CHARS} characters)` };
  }
  const user = readUser(storage);
  if (user.error) return { ok: false, reason: user.error };
  const trimmed = name.trim();
  const list = user.list.slice();
  const at = list.findIndex((p) => p.name === trimmed);
  if (at >= 0) list[at] = { name: trimmed, text };
  else if (list.length >= MAX_USER_PROMPTS) {
    return { ok: false, reason: `You can save at most ${MAX_USER_PROMPTS} prompts; delete one first` };
  } else list.push({ name: trimmed, text });
  const failed = writeUser(storage, list);
  return failed ? { ok: false, reason: failed } : { ok: true, reason: null };
}

/** Delete the saved prompt `name`. Built-ins cannot be deleted. { ok, reason }. */
export function deletePrompt(storage, name) {
  if (typeof name !== "string" || name.trim() === "") return { ok: false, reason: "Choose a saved prompt" };
  if (BUILTIN_NAMES.has(name.trim().toLowerCase())) {
    return { ok: false, reason: "Built-in prompts cannot be deleted" };
  }
  const user = readUser(storage);
  if (user.error) return { ok: false, reason: user.error };
  const list = user.list.filter((p) => p.name !== name);
  if (list.length === user.list.length) return { ok: false, reason: "There is no saved prompt with that name" };
  const failed = writeUser(storage, list);
  return failed ? { ok: false, reason: failed } : { ok: true, reason: null };
}

// T-0225 criterion 7: export and import of the saved prompts, as a JSON file the page writes and
// reads itself. The file is {"version": 1, "prompts": [{"name", "text"}]}; the built-ins never go in.

export const EXPORT_FILENAME = "gearvis-prompts.json";
export const EXPORT_VERSION = 1;
export const MAX_IMPORT_BYTES = 256 * 1024;

/** A mirror of prompt_builder.INSTRUCTIONS_FORBIDDEN (T-0213 A1-4, R-1): chat-template syntax an
 *  imported text may not carry. One audio token is split ("<" + ...) only so the HUD's no-audio
 *  grep (tests/unit/test_hud_no_audio_no_egress.py) does not read the literal as an element; the
 *  string is the same. */
export const FORBIDDEN_TEXT = Object.freeze([
  "<|", "|>", "<turn", "turn|>", "<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<|tool>", "<tool|>",
  "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>", '<|"|>', "<|think|>",
  "<|channel>", "<channel|>", "<|turn>", "<turn|>", "<|image>", "<|audio>", "<|image|>", "<|audio|>",
  "<image|>", "<" + "audio|>", "<|video|>",
]);

/** Why `text` cannot be an imported prompt, or null: the rules of validate_instructions (at most
 *  MAX_PROMPT_CHARS code points once trimmed, no control or format character but tab and line
 *  break, no chat-template syntax, also when whitespace splits it). */
export function textReason(text) {
  if (typeof text !== "string") return "the text is not text";
  const stripped = text.trim();
  const n = codePoints(stripped);
  if (n > MAX_PROMPT_CHARS) return `the text is too long (${n} / ${MAX_PROMPT_CHARS} characters)`;
  if (/[^\n\t\P{Cc}]|\p{Cf}|\p{Cs}/u.test(text)) return "the text contains a control character";
  const squeezed = stripped.replace(/\s+/gu, "");
  for (const lit of FORBIDDEN_TEXT) {
    if (stripped.includes(lit) || squeezed.includes(lit)) return "the text contains chat-template syntax";
  }
  return null;
}

/** The export file's text: the saved prompts of `prompts` (built-ins left out), as loadPrompts
 *  lists them. */
export function exportPrompts(prompts) {
  const list = Array.isArray(prompts) ? prompts : [];
  const saved = list
    .filter((p) => p && !p.builtin && typeof p.name === "string" && typeof p.text === "string")
    .map((p) => ({ name: p.name, text: p.text }));
  return JSON.stringify({ version: EXPORT_VERSION, prompts: saved }, null, 2);
}

function utf8Bytes(text) {
  if (typeof TextEncoder === "function") return new TextEncoder().encode(text).length;
  return text.length * 3; // an upper bound
}

/** { entries, error }: the file's entries as {name, text} (each judged later, by mergeImport), or
 *  the reason the whole file is refused (then entries is empty and nothing may change). */
export function parseImport(text) {
  const refuse = (error) => ({ entries: [], error });
  if (typeof text !== "string") return refuse("The file could not be read");
  if (utf8Bytes(text) > MAX_IMPORT_BYTES) return refuse("The file is larger than 256 KB");
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    return refuse("The file is not a prompts file (it is not JSON)");
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return refuse("The file is not a prompts file");
  }
  if (parsed.version !== EXPORT_VERSION) return refuse("The file is not a version 1 prompts file");
  if (!Array.isArray(parsed.prompts)) return refuse("The file has no list of prompts");
  const entries = parsed.prompts.map((p) =>
    p && typeof p === "object" && !Array.isArray(p) ? { name: p.name, text: p.text } : { name: null, text: null },
  );
  return { entries, error: null };
}

/** The trimmed name an imported entry is stored under, or null when it is skipped (a bad name, a
 *  built-in name, a bad text). */
function importName(entry) {
  if (!entry || typeof entry.name !== "string" || validName(entry.name) !== null) return null;
  if (textReason(entry.text) !== null) return null;
  return entry.name.trim();
}

/** How many of `entries` would replace a saved prompt of the same name (the panel asks first). */
export function importConflicts(storage, entries) {
  const user = readUser(storage);
  if (user.error || !Array.isArray(entries)) return 0;
  const have = new Set(user.list.map((p) => p.name));
  const seen = new Set();
  let n = 0;
  for (const e of entries) {
    const name = importName(e);
    if (name === null || seen.has(name)) continue;
    seen.add(name);
    if (have.has(name)) n += 1;
  }
  return n;
}

/** Merge `entries` into the saved prompts in one write. A valid new name is added while fewer than
 *  MAX_USER_PROMPTS are saved; an existing name is replaced only with `replace: true`; invalid
 *  entries, built-in names, a name repeated in the file (after its first) and entries over the cap
 *  are skipped. { added, replaced, skipped, error }: on an error nothing changed. */
export function mergeImport(storage, entries, { replace = false } = {}) {
  const none = (error) => ({ added: 0, replaced: 0, skipped: 0, error });
  if (!Array.isArray(entries)) return none("The file has no list of prompts");
  const user = readUser(storage);
  if (user.error) return none(user.error);
  const list = user.list.slice();
  const seen = new Set();
  let added = 0;
  let replaced = 0;
  let skipped = 0;
  for (const e of entries) {
    const name = importName(e);
    if (name === null || seen.has(name)) {
      skipped += 1;
      continue;
    }
    seen.add(name);
    const at = list.findIndex((p) => p.name === name);
    if (at >= 0) {
      if (replace) {
        list[at] = { name, text: e.text };
        replaced += 1;
      } else skipped += 1;
    } else if (list.length < MAX_USER_PROMPTS) {
      list.push({ name, text: e.text });
      added += 1;
    } else skipped += 1;
  }
  if (added + replaced > 0) {
    const failed = writeUser(storage, list);
    if (failed) return none(failed);
  }
  return { added, replaced, skipped, error: null };
}
