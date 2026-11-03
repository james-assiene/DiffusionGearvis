// T-0213 B-read: the Settings panel's pure model (grouping, chips, the relaunch text, the view of
// one row). DOM-free and socket-free: ./index.js puts what these return into textContent.
//
// The catalog and the items are the A2r-1 wire shapes of astra.settings.state: a CatalogEntry
// carries local_astra/settings_catalog.py's text fields (key, label, group, kind, what,
// useful_when, not_useful_when, evidence, cost); a SettingItem is {key, value, source, kind,
// changeable, pending, relaunch}. Nothing here escapes, strips or parses a string: catalog text
// is shown as text, verbatim.

/** B-1: the panel's groups, in render order (settings_catalog.GROUPS). */
export const SETTINGS_GROUPS = Object.freeze([
  "Seeing the screen",
  "Acting in the browser",
  "Checking its work",
  "Guidance and prompt",
  "Search and web",
  "Speech",
  "Engine (relaunch only)",
]);

/** B-1: the chip of each change kind. Only the two next-turn kinds say a change arrives soon. */
const CHIPS = Object.freeze({
  next_turn: "next turn",
  next_turn_cold: "next turn, slower first reply",
  relaunch: "needs relaunch",
  fixed: "fixed",
  not_here: "doesn't apply in this mode",
});
const UNKNOWN_CHIP = "not changeable";

/** The kinds a row may be changed from the panel in (A2-apply); anything else is read-only. */
const CHANGEABLE_KINDS = new Set(["next_turn", "next_turn_cold"]);

/** The source badge's text (A2r-1 SettingSource). */
const SOURCES = Object.freeze({
  default: "default",
  profile: "profile",
  flag: "flag",
  ui: "changed in UI",
});

export function kindChip(kind) {
  return typeof kind === "string" && Object.hasOwn(CHIPS, kind) ? CHIPS[kind] : UNKNOWN_CHIP;
}

export function sourceText(source) {
  if (typeof source === "string" && Object.hasOwn(SOURCES, source)) return SOURCES[source];
  return typeof source === "string" && source !== "" ? source : "unknown";
}

/** B-1: the groups in SETTINGS_GROUPS order (only those with entries), each entry once, in the
 *  catalog's own order. An entry whose group is not one of the seven comes last, under its own
 *  group name, so nothing the server sends is silently dropped. The input is not mutated. */
export function groupCatalog(catalog) {
  const byGroup = new Map();
  const seen = new Set();
  for (const entry of Array.isArray(catalog) ? catalog : []) {
    if (!entry || typeof entry !== "object" || typeof entry.key !== "string") continue;
    if (seen.has(entry.key)) continue;
    seen.add(entry.key);
    const group = typeof entry.group === "string" ? entry.group : "";
    if (!byGroup.has(group)) byGroup.set(group, []);
    byGroup.get(group).push(entry);
  }
  const out = [];
  for (const group of SETTINGS_GROUPS) {
    if (byGroup.has(group)) out.push({ group, entries: byGroup.get(group) });
  }
  for (const [group, entries] of byGroup) {
    if (!SETTINGS_GROUPS.includes(group)) out.push({ group, entries });
  }
  return out;
}

/** R-12: the relaunch box's label when it shows flags to add to the user's own session command. */
export const RELAUNCH_LABEL = "Add to your session command:";
/** R-12: its label when the row has no flag (the server's text says what to do instead). */
export const RELAUNCH_NOTE_LABEL = "To change it:";

/** B-2/R-12: the server's relaunch text, verbatim (the server sends only the flag(s) to add, or
 *  for a row with no flag what to do instead); null when there is none. */
export function relaunchText(item) {
  const command = item && typeof item === "object" ? item.relaunch : null;
  return typeof command === "string" && command !== "" ? command : null;
}

/** R-12: the relaunch box's label: flags to add, or (no flag) what to do instead. */
export function relaunchLabel(entry) {
  return relaunchFlag(entry) === "" ? RELAUNCH_NOTE_LABEL : RELAUNCH_LABEL;
}

function relaunchFlag(entry) {
  return entry && typeof entry === "object" && typeof entry.flag === "string" ? entry.flag : "";
}

/** The current value as one line of text (never empty, so a row always says something). */
export function valueText(value) {
  if (value === true) return "on";
  if (value === false) return "off";
  if (value === null || value === undefined) return "doesn't apply here";
  if (typeof value === "string") return value === "" ? "empty" : value;
  if (typeof value === "number") return String(value);
  if (Array.isArray(value)) return value.length ? value.map(String).join(" × ") : "none";
  return String(value);
}

const str = (v) => (typeof v === "string" ? v : "");

/** B-1/B-2: plain data for one row. readOnly is true for a relaunch, fixed or not_here kind, for an
 *  unknown kind, and whenever the server says the item is not changeable. R-10: the kind (and its
 *  chip) is the item's effective kind in this session, never the catalog's. */
export function itemView(entry, item) {
  const e = entry && typeof entry === "object" ? entry : {};
  const i = item && typeof item === "object" ? item : {};
  const kind = str(i.kind);
  // R-10a: a row that does not apply in this mode offers no flag to add, whatever the item carries
  const relaunch = kind === "not_here" ? null : relaunchText(i);
  return {
    key: str(e.key),
    label: str(e.label),
    kind,
    chip: kindChip(kind),
    source: str(i.source),
    sourceText: sourceText(i.source),
    value: i.value,
    valueText: valueText(i.value),
    values: Array.isArray(e.values) ? e.values.slice() : [],
    what: str(e.what),
    useful_when: str(e.useful_when),
    not_useful_when: str(e.not_useful_when),
    evidence: str(e.evidence),
    cost: str(e.cost),
    relaunch,
    relaunchLabel: relaunch === null ? null : relaunchLabel(e),
    readOnly: !CHANGEABLE_KINDS.has(kind) || i.changeable !== true,
  };
}

// --- T-0213 C: changes from the panel, and the instructions box --------------------------------

/** A2a-3: what a row says while its validated change waits for the turn boundary. */
export const PENDING_TEXT = "applies after this turn";
/** A2a-3: the button that resends the waiting change with stop_turn: true. */
export const STOP_AND_APPLY_TEXT = "Stop and apply";
/** C-1: the instructions box's label (the catalog row's label). */
export const INSTRUCTIONS_LABEL = "Your instructions to Gearvis";
/** C-2: the disclosure's summary. */
export const SYSTEM_PREVIEW_SUMMARY = "Show the full system message";
/** C-2, verbatim. */
export const SYSTEM_NOTE =
  "Safety rules (no passwords or card numbers, ask before buying or sending, the web fence) are " +
  "enforced by Gearvis's code, not by this text.";

/** Code points, not UTF-16 units: an astral character counts once (A1-4, the server's own count). */
export function codePointCount(text) {
  if (typeof text !== "string") return 0;
  let n = 0;
  for (const _ of text) n += 1;
  return n;
}

/** C-1: the live counter's text, "N / max". */
export function counterText(text, max) {
  const cap = Number.isInteger(max) && max >= 0 ? max : 0;
  return `${codePointCount(text)} / ${cap}`;
}

/** The reasons the server gave for refusing `key` in `state.refused`. A "*" refusal (a stale
 *  revision, an unknown key) belongs to every key of the set that was sent last. */
export function refusalsFor(state, key, lastKeys = []) {
  const refused = state && Array.isArray(state.refused) ? state.refused : [];
  const out = [];
  for (const r of refused) {
    if (!r || typeof r.reason !== "string") continue;
    if (r.key === key || (r.key === "*" && lastKeys.includes(key))) out.push(r.reason);
  }
  return out;
}

/** The change set one control sends. R-4: turning ReGround on while the overview frame is on
 *  also turns the overview frame off, in the same set (Profile refuses the two together). */
export function changeFor(key, value, items) {
  const changes = { [key]: value };
  if (key === "reground" && value === true) {
    const list = Array.isArray(items) ? items : [];
    const frame = list.find((i) => i && i.key === "overview_frame");
    const on = frame && (frame.pending !== null && frame.pending !== undefined ? frame.pending : frame.value);
    if (on === true) changes.overview_frame = false;
  }
  return changes;
}
