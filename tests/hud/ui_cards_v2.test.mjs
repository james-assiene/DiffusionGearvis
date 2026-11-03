// T-0184 group B: the HUD renderer for the v2 cards (catalog `gearvis:cards/v2`), node rows.
// docs/tasks/T-0184.md B1, B2, B4, B5, B6, B7, B11 and the group A review's hand-off ("never resolve or
// render `accessibility` on a v2 kind"). The look (B3), contrast (B8), overflow (B9), the real click (B5)
// and loopback (B11) are measured in Chrome by tests/integration/test_t0184_genui_v2.py.
//
// What drives the renderer: the production panel, `hud/src/ui/index.js` `mount(host)`, with a fake host
// (card slots, a subscribe seam, a send that records), so B1's "index.js creates the engine with both ids"
// is exercised and not assumed. The DOM is the trapping double of tests/hud/ui_render.test.mjs (imported
// from its `// <fake-dom>` region, one double for every renderer module): it counts every mutation, throws
// a named ForbiddenSink on every markup sink and fetch path, and records every DOM name it does not model.
// index.js links the card sheet with a `<link>`, which the double refuses (and index.js logs and carries
// on); that one recorded violation is cleared right after mount, before any card is rendered.
//
// Where the expected values come from (never from the renderer):
//   * tests/fixtures/a2ui/cards_v2_exemplars.jsonl (group A's 19 payloads, MANIFEST.md) for the inputs;
//   * EXPECT below, written by hand from those payloads and B4's rule: the model's strings in the model's
//     order (a Comparison is read row-major: the column headers, then each row's label and cells), plus
//     exactly B4's four derived strings, each in its slot:
//       (i)   String(value) of each bar, immediately before that bar's unit when the model gave one;
//       (ii)  the Bars ticks: String(n) of "nice" numbers from 0, at most five, equally spaced;
//       (iii) "<percent>%" of each tile that has a percent (one text node, or "<percent>" then "%");
//       (iv)  the step ordinals "1".."n", each before its step's title;
//   * the T-0164 census fixture for B2's differential (a basic payload under the v2 id must render as it
//     does under the basic id), and component_repair.py's RB rule for the gate-made Button B5 compares with.
//
// Readings the task leaves open, stated so a reviewer can disagree:
//   * "appears as textContent, byte-equal" is read per text node: each model string is the whole data of
//     one text node (so "828 m" in one node would not show the unit "m" byte-equal). The unit is shown at
//     each bar's end, once per bar (B3: "the value plus the model's unit at each bar's end").
//   * Ticks: "from 0 to the max" is read as: the first tick is 0, the last is at least half the largest
//     value and less than one step beyond it (so 0..9,000 for 8,849, as the approved Chart board draws);
//     "nice" as one significant digit (or 2.5) times a power of ten, which admits the board's 3,000 step.
//     With every bar 0 there is no scale: no tick, or a lone "0".
//   * An Actions button's `sourceComponentId` is the Actions component's id (the only model id it has).
//   * Hooks: every component's outer element keeps T-0164's `data-a2-id` / `data-a2-kind` (implementation
//     note 5 of T-0164). No other hook is assumed here: rows, tiles and cells are found through the model's
//     own text nodes.
//
// Pending convention (node tests cannot carry pytest's `pending`; the T-0162/T-0164 mechanism): every row
// that needs the implementation runs as a strict xfail until docs/tasks/T-0184.md's **Status:** block
// records "group B: tests <a>..<b>, activated <sha>". A row that passes while pending fails (XPASS strict).
// ASTRA_RUN_PENDING=1 runs them for real. The rows that test this module's own oracle are not pending.

import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";
import { join } from "node:path";

const REPO = fileURLToPath(new URL("../../", import.meta.url));
/** The basic catalog id our gate emits (local_astra/ui; hand-copied, never read from the JS under test). */
const GATE_CATALOG_ID = "https://a2ui.org/specification/v0_9_1/catalogs/basic/catalog.json";
/** H-T184-4 (local_astra.ui.a2ui_schema.CARDS_V2_CATALOG_ID, hand-copied). */
const V2_ID = "gearvis:cards/v2";
/** The deleted vendor's v0_9 id: foreign. */
const VENDOR_CATALOG_ID = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json";
const V2_KINDS = ["Stat", "StatTiles", "Comparison", "Bars", "Steps", "Source", "Actions"];
const SVG_NS = "http://www.w3.org/2000/svg";
/** The nine Stat icons (A2's table). */
const ICONS = ["sun", "partlyCloudy", "cloud", "rain", "storm", "snow", "fog", "wind", "night"];

// --- the pending gate (T-0184 group B) --------------------------------------------------------------------

const T0184_TASK = join(REPO, "docs/tasks/T-0184.md");
/** The **Status:** block: from the bold label to the next bold paragraph (the task's Status is a list). */
const T0184_STATUS = existsSync(T0184_TASK)
  ? ((/^\*\*Status:\*\*([\s\S]*?)\n\n\*\*/m.exec(readFileSync(T0184_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const B_ACTIVATED = /\bgroup B:[^;)\n]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const B_PENDING = !B_ACTIVATED.test(T0184_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function rowB(name, optsOrFn, maybeFn) {
  const fn = maybeFn || optsOrFn;
  const opts = maybeFn ? optsOrFn : {};
  test(name + (B_PENDING ? " [pending T-0184 B: strict xfail until group B is activated]" : ""), opts, async (t) => {
    if (!B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail("XPASS(strict): this T-0184 group B row passed while docs/tasks/T-0184.md's Status does not record group B's activation.");
    }
    t.diagnostic("expected failure (T-0184 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0184 B gate: the Status block is readable and only group B's own phrase activates the rows", () => {
  assert.ok(T0184_STATUS.length > 0, "docs/tasks/T-0184.md must carry a **Status:** block");
  assert.equal(B_ACTIVATED.test("group A done 2026-10-28.\n- Implementation 2f587843; tests up to 9c39701a."), false, "group A's wording is not group B's");
  assert.equal(B_ACTIVATED.test("in progress (group A done; group B: tests eb10b6e6..1234abcd, activated 5678abcd)"), true);
  assert.equal(B_ACTIVATED.test("in progress (group B: tests eb10b6e6..1234abcd, not yet activated)"), false, "activation names its commit");
  assert.equal(B_ACTIVATED.test("in progress (group C: tests 1a2b3c4..5d6e7f8, activated 9a8b7c6)"), false, "group C's activation is not group B's");
  assert.equal(B_ACTIVATED.test("group B tests written\n- group C: tests …, activated 1234abc"), false, "a phrase on another line does not count");
});

// --- module loading (the ESM shim of ui_render.test.mjs) ----------------------------------------------------

let ESM_ROOT = null;
function srcRoot() {
  if (ESM_ROOT) return ESM_ROOT;
  const dir = mkdtempSync(join(tmpdir(), "astra-hud-t0184b-"));
  writeFileSync(join(dir, "package.json"), '{"type":"module","private":true}');
  cpSync(join(REPO, "hud/src"), join(dir, "src"), { recursive: true });
  ESM_ROOT = pathToFileURL(join(dir, "src") + "/");
  return ESM_ROOT;
}
async function load(name) {
  if (!existsSync(join(REPO, "hud/src/ui", name))) assert.fail(`hud/src/ui/${name} does not exist`);
  try {
    return await import(new URL(`ui/${name}`, srcRoot()).href);
  } catch (err) {
    assert.fail(`hud/src/ui/${name} must import in node with no document: ${err && err.stack}`);
  }
}

let FAKE = null;
async function fakeDom() {
  if (!FAKE) {
    const src = readFileSync(join(REPO, "tests/hud/ui_render.test.mjs"), "utf8");
    const m = /\/\/ <fake-dom>[^\n]*\n([\s\S]*?)\n\/\/ <\/fake-dom>/.exec(src);
    assert.ok(m, "tests/hud/ui_render.test.mjs holds the fake DOM between // <fake-dom> markers");
    FAKE = await import("data:text/javascript," + encodeURIComponent(m[1]));
  }
  return FAKE.makeDom();
}

// --- fixtures -----------------------------------------------------------------------------------------------

const EXEMPLARS = readFileSync(join(REPO, "tests/fixtures/a2ui/cards_v2_exemplars.jsonl"), "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l));
const ex = (key) => {
  const e = EXEMPLARS.find((x) => x.key === key);
  assert.ok(e, `exemplar ${key} is missing from cards_v2_exemplars.jsonl`);
  return e;
};
const sidOf = (messages) => messages.find((m) => m.createSurface).createSurface.surfaceId;
const componentsOf = (messages) => messages.flatMap((m) => (m.updateComponents ? m.updateComponents.components : []));

const CENSUS = readFileSync(join(REPO, "tests/fixtures/a2ui/genui_census_20261027.jsonl"), "utf8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l))
  .slice(1);

const CREATE = (id, catalogId = V2_ID) => ({ version: "v0.9.1", createSurface: { surfaceId: id, catalogId } });
const COMPONENTS = (id, components) => ({ version: "v0.9.1", updateComponents: { surfaceId: id, components } });
const DATA = (id, path, value) => ({ version: "v0.9.1", updateDataModel: { surfaceId: id, path, value } });
const v2 = (id, components) => [CREATE(id), COMPONENTS(id, components)];
const basic = (id, components) => [CREATE(id, GATE_CATALOG_ID), COMPONENTS(id, components)];

// --- the panel harness: hud/src/ui/index.js mount(host) on the trapping double --------------------------

async function panel() {
  const { mount } = await load("index.js");
  const dom = await fakeDom();
  const sent = [];
  const logs = [];
  const slots = new Map();
  let listener = null;
  const host = {
    document: dom.document,
    cardSlot(id) {
      return dom.quiet(() => {
        const slot = dom.document.createElement("div");
        slot.setAttribute("data-card-slot", id);
        dom.document.body.appendChild(slot);
        slots.set(id, slot);
        return slot;
      });
    },
    subscribe(fn) {
      listener = fn;
      return () => {
        listener = null;
      };
    },
    send(message) {
      sent.push(JSON.parse(JSON.stringify(message)));
      return Promise.resolve();
    },
    log(line) {
      logs.push(String(line));
    },
  };
  const mounted = await mount(host);
  assert.ok(mounted && mounted.mounted === true, `index.js mount(host) mounted against the fake host; logs: ${logs.join(" | ")}`);
  assert.ok(listener, "mount(host) subscribed through host.subscribe");
  // index.js links card.css with a <link>; the double refuses that tag by design (it is the one element
  // a card must never create) and index.js logs and carries on. Cleared here so the card rows see only
  // what the renderer did.
  dom.rec.violations.length = 0;
  dom.rec.mutations.length = 0;
  const idx = new Map();
  const deliver = (surfaceId, messages, { final = false } = {}) => {
    const next = idx.has(surfaceId) ? idx.get(surfaceId) + 1 : 0;
    idx.set(surfaceId, next);
    listener({ type: "astra.ui.surface", surface_id: surfaceId, canvas_idx: next, messages, final });
  };
  const root = (id) => {
    const slot = slots.get(id);
    return slot ? slot.querySelector("[data-a2ui-root]") : null;
  };
  const card = (id) => {
    const r = root(id);
    assert.ok(r, `no card was rendered for surface ${id}; panel log: ${logs.slice(-6).join(" | ")}`);
    return r;
  };
  const wrapper = (id) => {
    const slot = slots.get(id);
    return slot ? slot.querySelector("[data-ui-surface]") : null;
  };
  const rejected = () => logs.filter((l) => /rejected by the renderer|did not render|render failed/i.test(l));
  return { dom, sent, logs, slots, deliver, root, card, wrapper, rejected };
}

async function showExemplar(key) {
  const h = await panel();
  const e = ex(key);
  const sid = sidOf(e.messages);
  h.deliver(sid, e.messages);
  assert.deepEqual(h.rejected(), [], `${key}: the renderer rejected or failed a message`);
  return { h, root: h.card(sid), sid, e };
}

// --- DOM reading helpers (independent of how the renderer builds things) -----------------------------------

function* walk(node) {
  for (const c of node.childNodes) {
    yield c;
    if (c.nodeType === 1) yield* walk(c);
  }
}
const elements = (root) => [...walk(root)].filter((n) => n.nodeType === 1);
const comp = (root, cid) => elements(root).find((e) => e.getAttribute("data-a2-id") === cid) || null;
const kindEls = (root, kind) => elements(root).filter((e) => e.getAttribute("data-a2-kind") === kind);
/** The text nodes a person can read, in document order: their data, verbatim. */
const chunks = (root) => [...walk(root)].filter((n) => n.nodeType === 3 && n.data.trim()).map((n) => n.data);
const textNodes = (root) => [...walk(root)].filter((n) => n.nodeType === 3 && n.data.trim());
const userClick = (dom, el) => dom.quiet(() => el.click());
const inRoot = (root, m) => (m.node && root.contains(m.node)) || (m.detail && typeof m.detail === "object" && root.contains(m.detail));
/** A node's identity in the double's mutation log. The double records a `style` write against the
 *  element itself rather than the proxy the renderer holds; both share the element's attribute map. */
const nodeKey = (n) => (n && n.nodeType === 1 ? n._attrs : n);
const keysOf = (nodes) => new Set([...nodes].map(nodeKey));
const mutationsIn = (dom, root) => dom.rec.mutations.filter((m) => inRoot(root, m));
const summary = (ms) => ms.slice(0, 8).map((m) => `${m.op}:${m.node && m.node.nodeName}${typeof m.detail === "string" ? "(" + m.detail + ")" : ""}`).join(", ");

/** Every attribute value and inline style value in the subtree, for "is this string anywhere" checks. */
function attributeText(root) {
  const out = [];
  for (const el of [root, ...elements(root)]) {
    for (const name of el.getAttributeNames()) out.push(`${name}=${el.getAttribute(name)}`);
    const css = el.style && el.style.cssText;
    if (css) out.push(`style{${css}}`);
  }
  return out;
}

// --- the B4 oracle: model strings in order, plus exactly the closed list of derived strings -----------------
//
// spec = { m: [ item, ... ], d: [ derived, ... ], ticks: null | { max } }
//   item:     a string (must be one text node's whole data), or { any: [s, ...] } (an unordered group)
//   derived:  { v, after, before, adjacentTo }: v is the derived string; it must be a text node strictly
//             after model item `after` and before model item `before` (indices into m; null = card end);
//             `adjacentTo`, when given, is the model item it must immediately precede (the unit after a value)
//   ticks:    the card holds one Bars whose largest value is `max`: every other leftover must be a tick.

/** A "nice" tick step: one significant digit, or 2.5, times a power of ten. Read generously on purpose:
 *  the approved Chart board steps by 3,000; what this rules out is a step like 2,212.25 (max / 4). */
const NICE = [1, 2, 2.5, 3, 4, 5, 6, 7, 8, 9];

/** null when `ticks` (strings, in document order) are B4's ticks for a largest value `max`; else why not. */
function tickProblem(ticks, max) {
  if (max === 0) {
    if (ticks.length === 0 || (ticks.length === 1 && ticks[0] === "0")) return null;
    return `all bars are 0: ticks may only be none or "0", got ${JSON.stringify(ticks)}`;
  }
  if (ticks.length < 2 || ticks.length > 5) return `${ticks.length} ticks; B3/B4: tick labels, at most five (note 5), and a scale needs two`;
  const nums = ticks.map(Number);
  for (const [i, t] of ticks.entries()) {
    if (!Number.isFinite(nums[i]) || String(nums[i]) !== t) return `tick ${JSON.stringify(t)} is not String(n) of a number (note 5: String(n), never toLocaleString)`;
  }
  if (nums[0] !== 0) return `the first tick is ${ticks[0]}, not 0`;
  const step = nums[1] - nums[0];
  for (let i = 1; i < nums.length; i++) {
    if (Math.abs(nums[i] - nums[i - 1] - step) > 1e-9 * Math.max(1, step)) return `ticks ${JSON.stringify(ticks)} are not equally spaced`;
  }
  const exp = Math.floor(Math.log10(step));
  const mant = step / 10 ** exp;
  if (!NICE.some((n) => Math.abs(mant - n) < 1e-9)) return `the tick step ${step} is not a nice number (one significant digit, or 2.5, times a power of ten)`;
  const last = nums[nums.length - 1];
  if (last < max / 2) return `the last tick ${last} covers less than half of the largest value ${max}`;
  if (last - step >= max) return `the last tick ${last} is a whole step beyond the largest value ${max}`;
  return null;
}

/** [] when `got` (text-node data in document order) is exactly the spec; else the problems. */
function closedListProblems(got, spec) {
  const problems = [];
  const items = spec.m.map((it) => (typeof it === "string" ? { any: [it], ordered: true } : { any: [...it.any], ordered: false }));
  const pos = []; // first/last chunk index of each model item
  let i = 0;
  const used = new Set();
  for (const [k, it] of items.entries()) {
    const want = new Map();
    for (const s of it.any) want.set(s, (want.get(s) || 0) + 1);
    let first = null;
    let last = null;
    while (i < got.length && want.size) {
      const c = got[i];
      if (want.has(c)) {
        want.set(c, want.get(c) - 1);
        if (want.get(c) === 0) want.delete(c);
        used.add(i);
        if (first === null) first = i;
        last = i;
      }
      i += 1;
    }
    if (want.size) {
      problems.push(`model string(s) ${JSON.stringify([...want.keys()])} (item ${k}) not shown byte-equal in order`);
      return problems;
    }
    pos.push([first, last]);
  }
  // leftovers, with a "<n>" text node directly followed by "%" read as one "<n>%"
  let left = [];
  for (let j = 0; j < got.length; j++) {
    if (used.has(j)) continue;
    const prev = left[left.length - 1];
    if (got[j] === "%" && prev && prev.at === j - 1 && /^\d+(\.\d+)?$/.test(prev.v)) prev.v += "%";
    else left.push({ v: got[j], at: j });
  }
  for (const d of spec.d || []) {
    const lo = d.after === null || d.after === undefined ? -1 : pos[d.after][1];
    const hi = d.before === null || d.before === undefined ? got.length : pos[d.before][0];
    const hit = left.findIndex((l) => l.v === d.v && l.at > lo && l.at < hi && (d.adjacentTo === undefined || pos[d.adjacentTo][0] === l.at + 1));
    if (hit < 0) {
      problems.push(`derived string ${JSON.stringify(d.v)} not found in its place${d.adjacentTo !== undefined ? " (right before its unit)" : ""}`);
      continue;
    }
    left.splice(hit, 1);
  }
  if (spec.ticks) {
    const p = tickProblem(left.map((l) => l.v), spec.ticks.max);
    if (p) problems.push(`ticks: ${p}`);
  } else if (left.length) {
    problems.push(`synthesised text, outside B4's closed list: ${JSON.stringify(left.map((l) => l.v))}`);
  }
  return problems;
}

/** Bars spec items: the title (if any), then label, [value derived], unit per bar. */
function barsSpec({ title, unit, bars }, prefix = []) {
  const m = [...prefix];
  const d = [];
  if (title !== undefined) m.push(title);
  bars.forEach(([label, value], k) => {
    m.push(label);
    const labelIdx = m.length - 1;
    if (unit !== undefined) {
      m.push(unit);
      d.push({ v: String(value), after: labelIdx, before: m.length - 1, adjacentTo: m.length - 1 });
    } else {
      d.push({ v: String(value), after: labelIdx, before: null, nextLabel: true });
    }
  });
  // without a unit, a value's window ends at the next bar's label
  for (const x of d) {
    if (x.nextLabel) {
      const next = m.findIndex((s, j) => j > x.after && bars.some(([l]) => l === s));
      x.before = next < 0 ? null : next;
      delete x.nextLabel;
    }
  }
  return { m, d, ticks: { max: Math.max(...bars.map(([, v]) => v)) } };
}

/** Steps spec: ordinal k sits after the previous step's last string and before step k's title. */
function stepsSpec(steps, prefix = []) {
  const m = [...prefix];
  const d = [];
  let prevEnd = prefix.length ? prefix.length - 1 : null;
  steps.forEach(([title, detail], k) => {
    m.push(title);
    d.push({ v: String(k + 1), after: prevEnd, before: m.length - 1 });
    if (detail !== undefined) m.push(detail);
    prevEnd = m.length - 1;
  });
  return { m, d, ticks: null };
}

/** StatTiles spec: "<p>%" (String(p) then "%") sits after tile k's title and before tile k+1's title. */
function tilesSpec(tiles, prefix = [], suffix = []) {
  const m = [...prefix];
  const starts = [];
  for (const [title, value, secondary] of tiles) {
    starts.push(m.length);
    m.push(title, value);
    if (secondary !== undefined) m.push(secondary);
  }
  const end = suffix.length ? m.length : null; // the first string after the tiles, if any
  m.push(...suffix);
  const d = [];
  tiles.forEach(([, , , percent], k) => {
    if (percent !== undefined) d.push({ v: `${String(percent)}%`, after: starts[k], before: k + 1 < tiles.length ? starts[k + 1] : end });
  });
  return { m, d, ticks: null };
}

/** EXPECT: hand-written from cards_v2_exemplars.jsonl (one entry per exemplar; a row checks coverage). */
const EXPECT = {
  "stat-minimal": { m: ["BATTERY", "82%"], d: [], ticks: null },
  "stat-icons": {
    m: [
      ["MADRID", "[31°]", "Sunny"],
      ["LYON", "[19°]", "Partly cloudy"],
      ["DUBLIN", "[14°]", "Overcast"],
      ["BERGEN", "[11°]", "Showers"],
      ["MIAMI", "[28°]", "Thunderstorms"],
      ["OSLO", "[-3°]", "Light snow"],
      ["LONDON", "[9°]", "Fog"],
      ["CHICAGO", "[12°]", "Windy"],
      ["ROME · TONIGHT", "[16°]", "Clear"],
    ].flat(),
    d: [],
    ticks: null,
  },
  "stattiles-storage": tilesSpec([
    ["Photos", "212 GB", "of 1 TB", 21],
    ["Apps", "88 GB", undefined, 8.8],
    ["System", "41 GB", "reserved", 4],
    ["Free", "659 GB"],
  ]),
  "stattiles-one": tilesSpec([["Steps today", "8 412"]]),
  "stattiles-percent-bounds": tilesSpec([
    ["Download", "Not started", undefined, 0],
    ["Upload", "Done", "3 files", 100],
  ]),
  "stattiles-four-wide": tilesSpec([
    ["Q1", "123456789012", "units"],
    ["Q2", "234567890123", "units"],
    ["Q3", "345678901234", "units"],
    ["Q4", "456789012345", "units"],
  ]),
  "comparison-laptops": {
    m: [{ any: ["Laptop A"] }, { any: ["Laptop B"] }, { any: ["Laptop C", "Lightest"] }, "Weight", "1.6 kg", "1.4 kg", "1.1 kg", "Battery", "12 h", "18 h", "7 h", "Ports", "4", "3", "2"],
    d: [],
    ticks: null,
  },
  "comparison-highlight-out-of-range": { m: ["Train", "Plane", "Door to door", "3 h 10", "3 h 40"], d: [], ticks: null },
  "bars-titled": barsSpec({ title: "Tallest buildings", unit: "m", bars: [["Burj Khalifa", 828], ["Merdeka 118", 678.9], ["Shanghai Tower", 632]] }),
  "bars-all-zero": barsSpec({ bars: [["Monday", 0], ["Tuesday", 0], ["Wednesday", 0]] }),
  "bars-single": barsSpec({ unit: "km", bars: [["Nile", 6650]] }),
  "bars-huge": barsSpec({ title: "Market value", unit: "USD", bars: [["Company A", 1000000000000], ["Company B", 3]] }),
  "steps-omelette": stepsSpec([["Whisk the eggs", "Three eggs, a pinch of salt."], ["Heat the pan"], ["Cook and fold", "Low heat, fold when just set."]]),
  "source-only": { m: ["From a web search"], d: [], ticks: null },
  "actions-three": { m: ["Show more", "Compare prices", "Start over"], d: [], ticks: null },
  "mixed-basic-on-v2": stepsSpec([["Stretch"], ["Hydrate", "Half a litre."]], ["Morning run", "DISTANCE", "10.2 km", "Personal best", "Pace 4:55 /km", "Time 50:09"]),
  "board-main": tilesSpec(
    [
      ["Friday", "[19°]", "[11°]", 20],
      ["Saturday", "[22°]", "[13°]", 10],
      ["Sunday", "[16°]", "[10°]", 60],
    ],
    ["PARIS · NOW", "[17°]", "Partly cloudy"],
    ["From a web search"],
  ),
  "board-compare": {
    m: [
      { any: ["Deluxe hardcover", "Recommended"] },
      { any: ["Regular volumes"] },
      "Volumes", "[1]", "[3]",
      "Price per book", "[€45]", "[€18]",
      "Format", "[Hardcover, slipcase]", "[Paperback]",
      "Best for", "[Gifts and display]", "[Reading on the go]",
      "[Show the deluxe edition]", "[Compare prices]",
    ],
    d: [],
    ticks: null,
  },
  "board-chart": (() => {
    const s = barsSpec({ unit: "m", bars: [["Everest", 8849], ["K2", 8611], ["Kangchenjunga", 8586], ["Mont Blanc", 4806]] });
    s.m.push("en.wikipedia.org");
    return s;
  })(),
};

// =========================================================================================================
// The oracle itself (not pending): hand-made chunk lists, both directions
// =========================================================================================================

test("oracle: the closed-list check passes the right text and fails each way a renderer could synthesise or drop words", () => {
  const cmp = EXPECT["board-compare"];
  const good = ["Recommended", "Deluxe hardcover", "Regular volumes", "Volumes", "[1]", "[3]", "Price per book", "[€45]", "[€18]", "Format", "[Hardcover, slipcase]", "[Paperback]", "Best for", "[Gifts and display]", "[Reading on the go]", "[Show the deluxe edition]", "[Compare prices]"];
  assert.deepEqual(closedListProblems(good, cmp), [], "the badge may sit before or after its title");
  // the adversarial mutant the task names: "Recommended" with no badge
  const noBadge = { m: [{ any: ["Deluxe hardcover"] }, { any: ["Regular volumes"] }, "Volumes", "[1]", "[3]"], d: [], ticks: null };
  assert.notDeepEqual(closedListProblems(["Recommended", "Deluxe hardcover", "Regular volumes", "Volumes", "[1]", "[3]"], noBadge), [], "a synthesised Recommended is caught");
  assert.notDeepEqual(closedListProblems(["From a web search", "en.wikipedia.org"], { m: ["en.wikipedia.org"], d: [], ticks: null }), [], "a synthesised From a web search is caught");
  assert.notDeepEqual(closedListProblems(good.filter((s) => s !== "[3]"), cmp), [], "a dropped cell is caught");
  const swapped = [...good];
  [swapped[4], swapped[5]] = [swapped[5], swapped[4]];
  assert.notDeepEqual(closedListProblems(swapped, cmp), [], "cells out of order are caught");
  assert.notDeepEqual(closedListProblems([...good, "—"], cmp), [], "a placeholder dash is caught");

  const chart = EXPECT["board-chart"];
  const bars = ["Everest", "8849", "m", "K2", "8611", "m", "Kangchenjunga", "8586", "m", "Mont Blanc", "4806", "m"];
  assert.deepEqual(closedListProblems([...bars, "0", "3000", "6000", "9000", "en.wikipedia.org"], chart), [], "the approved board's ticks 0..9000");
  assert.deepEqual(closedListProblems(["0", "2500", "5000", "7500", ...bars, "en.wikipedia.org"], chart), [], "ticks may sit above the bars");
  assert.notDeepEqual(closedListProblems([...bars, "0", "3,000", "6,000", "9,000", "en.wikipedia.org"], chart), [], "toLocaleString ticks are caught");
  assert.notDeepEqual(closedListProblems([...bars, "0", "1000", "2000", "3000", "4000", "5000", "6000", "7000", "8000", "9000", "en.wikipedia.org"], chart), [], "more than five ticks are caught");
  assert.notDeepEqual(closedListProblems([...bars, "0", "3000", "7000", "9000", "en.wikipedia.org"], chart), [], "unequal ticks are caught");
  assert.notDeepEqual(closedListProblems([...bars, "0", "5000", "10000", "15000", "en.wikipedia.org"], chart), [], "a scale a whole step past the max is caught");
  assert.notDeepEqual(closedListProblems(["Everest", "m", "8849", ...bars.slice(3), "0", "9000", "en.wikipedia.org"], chart), [], "a unit before its value is caught");
  assert.notDeepEqual(closedListProblems(["Everest", "8,849", "m", ...bars.slice(3), "0", "9000", "en.wikipedia.org"], chart), [], "a formatted value is caught");
  assert.notDeepEqual(closedListProblems(["Everest", "8849 m", ...bars.slice(3), "0", "9000", "en.wikipedia.org"], chart), [], "value and unit in one node do not show the unit byte-equal");
  assert.notDeepEqual(closedListProblems([...bars, "0", "NaN", "en.wikipedia.org"], chart), [], "NaN is caught");
  assert.deepEqual(closedListProblems(["Monday", "0", "Tuesday", "0", "Wednesday", "0"], EXPECT["bars-all-zero"]), [], "all-zero bars with no ticks");
  assert.notDeepEqual(closedListProblems(["Monday", "0", "Tuesday", "0", "Wednesday", "0", "0", "1"], EXPECT["bars-all-zero"]), [], "an invented scale for all-zero bars is caught");

  const steps = EXPECT["steps-omelette"];
  assert.deepEqual(closedListProblems(["1", "Whisk the eggs", "Three eggs, a pinch of salt.", "2", "Heat the pan", "3", "Cook and fold", "Low heat, fold when just set."], steps), []);
  assert.notDeepEqual(closedListProblems(["Whisk the eggs", "1", "Three eggs, a pinch of salt.", "2", "Heat the pan", "3", "Cook and fold", "Low heat, fold when just set."], steps), [], "an ordinal after its title is caught");
  assert.notDeepEqual(closedListProblems(["Step 1", "Whisk the eggs", "Three eggs, a pinch of salt.", "2", "Heat the pan", "3", "Cook and fold", "Low heat, fold when just set."], steps), [], "a synthesised word in an ordinal is caught");
  assert.notDeepEqual(closedListProblems(["1.", "Whisk the eggs", "Three eggs, a pinch of salt.", "2.", "Heat the pan", "3.", "Cook and fold", "Low heat, fold when just set."], steps), [], "an ordinal with a full stop is not String(k)");

  const tiles = EXPECT["stattiles-storage"];
  const t = ["Photos", "212 GB", "of 1 TB", "21%", "Apps", "88 GB", "8.8", "%", "System", "41 GB", "reserved", "4%", "Free", "659 GB"];
  assert.deepEqual(closedListProblems(t, tiles), [], "a percent as one node or as <n> then %");
  assert.notDeepEqual(closedListProblems([...t.slice(0, 13), "0%", "659 GB"], tiles), [], "a percent the model did not give is caught");
  assert.notDeepEqual(closedListProblems(["Photos", "212 GB", "of 1 TB", "Apps", "21%", ...t.slice(5)], tiles), [], "a percent in the wrong tile is caught");
});

test("oracle: EXPECT covers every exemplar, and its model strings are the payloads' (hand table vs fixture)", () => {
  assert.deepEqual(Object.keys(EXPECT).sort(), EXEMPLARS.map((e) => e.key).sort(), "one hand-written expectation per exemplar");
  for (const e of EXEMPLARS) {
    // every string anywhere in the payload's components except ids, kinds and icon names
    const strings = [];
    const visit = (v, key) => {
      if (typeof v === "string") {
        if (!["id", "component", "icon", "variant", "child"].includes(key)) strings.push(v);
      } else if (Array.isArray(v)) {
        if (key !== "children") v.forEach((x) => visit(x, key === "cells" || key === "labels" ? key : null));
      } else if (v && typeof v === "object") for (const [k, x] of Object.entries(v)) if (k !== "tone") visit(x, k);
    };
    for (const c of componentsOf(e.messages)) visit(c, null);
    const spec = EXPECT[e.key];
    const flat = spec.m.flatMap((x) => (typeof x === "string" ? [x] : x.any));
    const units = new Set(componentsOf(e.messages).filter((c) => c.component === "Bars" && c.unit).map((c) => c.unit));
    const dedupe = flat.filter((s, i) => !(units.has(s) && flat.indexOf(s) !== i));
    assert.deepEqual([...dedupe].sort(), [...strings].sort(), `${e.key}: EXPECT's model strings are exactly the payload's (each unit once per bar in EXPECT)`);
  }
});

// =========================================================================================================
// B1: the catalog on the page
// =========================================================================================================

rowB("B1: a2ui.js exports CARDS_V2_CATALOG_ID = gearvis:cards/v2 and keeps the basic id", async () => {
  const a2ui = await load("a2ui.js");
  assert.equal(a2ui.CARDS_V2_CATALOG_ID, V2_ID);
  assert.equal(a2ui.GATE_CATALOG_ID, GATE_CATALOG_ID, "the basic id is unchanged");
  assert.equal(a2ui.A2UI_SPEC_VERSION, "v0.9.1");
});

rowB("B1: the panel (index.js) renders a v2 surface and a basic surface; any other catalog id renders nothing", async () => {
  const h = await panel();
  h.deliver("ok-v2", v2("ok-v2", [{ id: "root", component: "Stat", label: "V2 LABEL", value: "v2 value" }]));
  h.deliver("ok-basic", basic("ok-basic", [{ id: "root", component: "Text", text: "basic words" }]));
  assert.ok(h.root("ok-v2"), `the v2 surface has a card; log: ${h.logs.join(" | ")}`);
  assert.deepEqual(chunks(h.root("ok-v2")), ["V2 LABEL", "v2 value"]);
  assert.deepEqual(chunks(h.card("ok-basic")), ["basic words"]);
  const foreign = [VENDOR_CATALOG_ID, "gearvis:cards/v1", "gearvis:cards/v3", "gearvis:cards/v2/", "GEARVIS:CARDS/V2", " gearvis:cards/v2", "gearvis:cards", ""];
  foreign.forEach((id, k) => {
    const sid = `foreign-${k}`;
    h.deliver(sid, [{ version: "v0.9.1", createSurface: { surfaceId: sid, catalogId: id } }, COMPONENTS(sid, [{ id: "root", component: "Stat", label: "FOREIGN", value: "must not render" }])]);
    assert.equal(h.root(sid), null, `catalogId ${JSON.stringify(id)}: no card (T-0164 B3)`);
  });
  assert.ok(!h.dom.document.body.textContent.includes("must not render"), "no foreign word reached the page");
  assert.deepEqual(h.dom.rec.violations, []);
});

// =========================================================================================================
// B2: per-surface kinds
// =========================================================================================================

rowB("B2: every v2 kind renders on a v2 surface (data-a2-kind on its element)", async () => {
  const seen = new Set();
  for (const e of EXEMPLARS) {
    const { root } = await showExemplar(e.key);
    for (const c of componentsOf(e.messages)) {
      if (!V2_KINDS.includes(c.component)) continue;
      const el = comp(root, c.id) || (c.id === "root" ? kindEls(root, c.component)[0] : null);
      assert.ok(el, `${e.key}: ${c.component} ${c.id} has an element with data-a2-id`);
      assert.equal(el.getAttribute("data-a2-kind"), c.component, `${e.key}: data-a2-kind names ${c.component}`);
      seen.add(c.component);
    }
  }
  assert.deepEqual([...seen].sort(), [...V2_KINDS].sort(), "the exemplars cover the seven kinds");
});

const ALL_V2_ON_ONE = (prefix = "") => [
  { id: "root", component: "Column", children: ["before", "st", "tl", "cmp", "br", "sp", "src", "act", "after"] },
  { id: "before", component: "Text", text: "before" },
  { id: "st", component: "Stat", label: `${prefix}STAT LABEL`, value: `${prefix}stat value`, detail: `${prefix}stat detail`, icon: "sun" },
  { id: "tl", component: "StatTiles", tiles: [{ title: `${prefix}tile title`, value: `${prefix}tile value`, percent: 50 }] },
  { id: "cmp", component: "Comparison", columns: [{ title: `${prefix}col one` }, { title: `${prefix}col two`, badge: `${prefix}badge` }], rows: [{ label: `${prefix}row`, cells: [`${prefix}c1`, { text: `${prefix}c2`, tone: "ok" }] }] },
  { id: "br", component: "Bars", title: `${prefix}bars title`, unit: `${prefix}u`, bars: [{ label: `${prefix}bar`, value: 5 }] },
  { id: "sp", component: "Steps", steps: [{ title: `${prefix}step`, detail: `${prefix}step detail` }] },
  { id: "src", component: "Source", text: `${prefix}source` },
  { id: "act", component: "Actions", labels: [`${prefix}act one`, `${prefix}act two`] },
  { id: "after", component: "Text", text: "after" },
];

rowB("B2: on a basic surface every v2 kind renders nothing, logs once per kind, and its siblings stay", async () => {
  const h = await panel();
  const comps = ALL_V2_ON_ONE("leak ");
  h.deliver("b2", basic("b2", comps));
  const root = h.card("b2");
  assert.deepEqual(chunks(root), ["before", "after"], "only the basic siblings' words show, in order");
  for (const kind of V2_KINDS) assert.deepEqual(kindEls(root, kind), [], `no ${kind} element on a basic surface`);
  assert.ok(!root.textContent.includes("leak"), "no v2 string leaked");
  assert.equal(elements(root).filter((e) => e.localName === "button").length, 0, "no Actions button on a basic surface");
  const lines = h.logs.filter((l) => !/mounted/.test(l));
  for (const kind of V2_KINDS) {
    assert.equal(lines.filter((l) => new RegExp(`\\b${kind}\\b`).test(l)).length, 1, `one log line names ${kind}; got ${JSON.stringify(lines)}`);
  }
  // two more canvases of the same surface add no line (once per surface and kind); a benign surface's
  // re-sends are the baseline, as in T-0164 A3
  const mid = h.logs.length;
  h.deliver("b2", [COMPONENTS("b2", comps)]);
  h.deliver("b2", [COMPONENTS("b2", comps)]);
  const g = await panel();
  g.deliver("benign", basic("benign", [{ id: "root", component: "Text", text: "benign" }]));
  const base = g.logs.length;
  g.deliver("benign", [COMPONENTS("benign", [{ id: "root", component: "Text", text: "benign" }])]);
  g.deliver("benign", [COMPONENTS("benign", [{ id: "root", component: "Text", text: "benign" }])]);
  assert.equal(h.logs.length - mid, g.logs.length - base, `re-sends add no fallback line: ${JSON.stringify(h.logs.slice(mid))}`);
  // the same canvas under the v2 id renders every kind (the guard is not "never render v2 kinds")
  h.deliver("b2v2", v2("b2v2", comps));
  for (const kind of V2_KINDS) assert.equal(kindEls(h.card("b2v2"), kind).length, 1, `${kind} renders under the v2 id`);
});

/** The element shape a person sees: tag, component id and kind, type and role, per element in order. */
const shape = (root) => elements(root).map((e) => [e.localName, e.getAttribute("data-a2-id"), e.getAttribute("data-a2-kind"), e.getAttribute("type"), e.getAttribute("role")].join("|"));

rowB("B2: every T-0164 census payload renders under the v2 id exactly as under the basic id (differential)", async () => {
  for (const s of CENSUS) {
    const base = await panel();
    for (const cv of s.canvases) base.deliver(s.surface_id, cv);
    const want = base.card(s.surface_id);
    const moved = await panel();
    for (const cv of s.canvases) moved.deliver(s.surface_id, JSON.parse(JSON.stringify(cv).replaceAll(JSON.stringify(GATE_CATALOG_ID), JSON.stringify(V2_ID))));
    assert.deepEqual(moved.rejected(), [], `${s.key}: the v2 surface was rejected`);
    const got = moved.card(s.surface_id);
    assert.deepEqual(chunks(got), chunks(want), `${s.key}: the same words in the same order`);
    assert.deepEqual(shape(got), shape(want), `${s.key}: the same elements`);
    assert.deepEqual(moved.dom.rec.violations, []);
  }
});

rowB("B2 / note 7: a root Card around a v2 kind is unwrapped into the frame (H-T164-8)", async () => {
  const h = await panel();
  h.deliver("unwrap", v2("unwrap", [{ id: "root", component: "Card", child: "s" }, { id: "s", component: "Stat", label: "WRAPPED", value: "1" }]));
  const root = h.card("unwrap");
  assert.deepEqual(kindEls(root, "Card"), [], "no Card tile drawn inside the frame");
  assert.ok(comp(root, "s"), "the Stat renders");
  assert.deepEqual(chunks(root), ["WRAPPED", "1"]);
});

// =========================================================================================================
// B4: verbatim, and nothing synthesised beyond the closed list
// =========================================================================================================

rowB("B4: every exemplar shows its model strings byte-equal in tree order, plus exactly B4's derived strings", async () => {
  const problems = [];
  for (const e of EXEMPLARS) {
    const { root, h } = await showExemplar(e.key);
    for (const p of closedListProblems(chunks(root), EXPECT[e.key])) problems.push(`${e.key}: ${p}\n    shown ${JSON.stringify(chunks(root)).slice(0, 400)}`);
    assert.deepEqual(h.dom.rec.violations, [], `${e.key}: no sink`);
  }
  assert.deepEqual(problems, [], "B4");
});

rowB("B4 (adversarial): no Recommended without a badge, no From a web search without it in Source.text, no unit the model did not give", async () => {
  const h = await panel();
  h.deliver("nobadge", v2("nobadge", [
    { id: "root", component: "Column", children: ["c", "s", "b"] },
    { id: "c", component: "Comparison", columns: [{ title: "Left" }, { title: "Right" }], highlight: 0, rows: [{ label: "Row", cells: ["x", "y"] }] },
    { id: "s", component: "Source", text: "en.wikipedia.org" },
    { id: "b", component: "Bars", bars: [{ label: "Only", value: 4 }] },
  ]));
  const root = h.card("nobadge");
  const text = root.textContent;
  for (const word of ["Recommended", "Best", "From a web search", "web", "search", "Source", "%", "units"]) assert.ok(!text.includes(word), `"${word}" was synthesised: ${JSON.stringify(chunks(root))}`);
  const spec = { m: [{ any: ["Left"] }, { any: ["Right"] }, "Row", "x", "y", "en.wikipedia.org", "Only"], d: [{ v: "4", after: 6, before: null }], ticks: { max: 4 } };
  assert.deepEqual(closedListProblems(chunks(root), spec), []);
});

const ADVERSARIAL = {
  markup: "<img src=x onerror=alert(1)>",
  closer: "</div><script>",
  entity: "&amp;",
  markdown: "**bold**",
  newline: "line one\nline two",
  rtl: "שלום مرحبا ‮evil",
  emoji: "\u{1F373} \u{1F1F5}\u{1F1F9} \u{1F468}‍\u{1F469}‍\u{1F467}",
  long: "x".repeat(10000),
};

/** Every string role of every v2 kind, holding `s`. 19 roles. */
const everyRole = (s) => [
  { id: "root", component: "Column", children: ["st", "tl", "cmp", "br", "sp", "src", "act"] },
  { id: "st", component: "Stat", label: s, value: s, detail: s },
  { id: "tl", component: "StatTiles", tiles: [{ title: s, value: s, secondary: s }] },
  { id: "cmp", component: "Comparison", columns: [{ title: s, badge: s }, { title: s }], rows: [{ label: s, cells: [s, { text: s, tone: "warn" }] }] },
  { id: "br", component: "Bars", title: s, unit: s, bars: [{ label: s, value: 0 }] },
  { id: "sp", component: "Steps", steps: [{ title: s, detail: s }] },
  { id: "src", component: "Source", text: s },
  { id: "act", component: "Actions", labels: [s] },
];
const ROLES = 19;

rowB("B4 (adversarial): the T-0164 A4 strings in every text role render as that exact string and create no element", async () => {
  const render = async (s) => {
    const h = await panel();
    const before = h.dom.rec.creates.length;
    h.deliver("adv", v2("adv", everyRole(s)));
    const root = h.card("adv");
    return { h, root, made: h.dom.rec.creates.slice(before).filter((n) => n.nodeType === 1).map((n) => n.localName) };
  };
  const plain = await render("plain");
  assert.equal(textNodes(plain.root).filter((n) => n.data === "plain").length, ROLES, `premise: the ${ROLES} roles each show "plain"`);
  for (const [name, s] of Object.entries(ADVERSARIAL)) {
    const { h, root, made } = await render(s);
    assert.equal(textNodes(root).filter((n) => n.data === s).length, ROLES, `${name}: each of the ${ROLES} roles shows the exact string`);
    assert.deepEqual(made, plain.made, `${name}: the same elements as for "plain" (the string made none)`);
    assert.deepEqual(h.dom.rec.violations, [], `${name}: no sink`);
  }
});

rowB("B4: a v2 component no reachable parent names renders nothing", async () => {
  const h = await panel();
  h.deliver("reach", v2("reach", [{ id: "root", component: "Source", text: "reached" }, { id: "orphan", component: "Stat", label: "ORPHAN", value: "unreached" }]));
  assert.deepEqual(chunks(h.card("reach")), ["reached"]);
});

rowB("hand-off (group A review): accessibility on a v2 kind is never rendered, as text or as an attribute", async () => {
  const lit = { label: "A11Y-LABEL-SENTINEL", description: "A11Y-DESC-SENTINEL" };
  const bound = { label: { path: "/a11y/label" }, description: { path: "/a11y/desc" } };
  const comps = (acc) => ALL_V2_ON_ONE().map((c) => (V2_KINDS.includes(c.component) ? { ...c, accessibility: acc } : c));
  const h = await panel();
  h.deliver("acc-lit", v2("acc-lit", comps(lit)));
  h.deliver("acc-bound", [CREATE("acc-bound"), COMPONENTS("acc-bound", comps(bound)), DATA("acc-bound", "/", { a11y: { label: "A11Y-BOUND-SENTINEL", desc: "A11Y-BOUND-DESC" } })]);
  for (const sid of ["acc-lit", "acc-bound"]) {
    const root = h.card(sid);
    for (const kind of V2_KINDS) assert.equal(kindEls(root, kind).length, 1, `${sid}: premise, ${kind} renders`);
    const everywhere = [root.textContent, ...attributeText(root)].join("\n");
    assert.ok(!/A11Y-/.test(everywhere), `${sid}: an accessibility string reached the card: ${everywhere.match(/[^\n]*A11Y-[^\n]*/)}`);
  }
});

// =========================================================================================================
// B3 (structure; the look itself is the Chrome module's)
// =========================================================================================================

rowB("B3: a Stat draws one inline SVG glyph per known icon (nine distinct), none for an absent or unknown icon", async () => {
  const { root } = await showExemplar("stat-icons");
  const glyphs = [];
  for (const k of [1, 2, 3, 4, 5, 6, 7, 8, 9]) {
    const svgs = elements(comp(root, `s${k}`)).filter((e) => e.localName === "svg");
    assert.equal(svgs.length, 1, `s${k} (${ICONS[k - 1]}): one svg glyph`);
    assert.equal(svgs[0].namespaceURI, SVG_NS, "made with createElementNS in the SVG namespace (B11)");
    glyphs.push(JSON.stringify(elements(svgs[0]).map((e) => [e.localName, ...e.getAttributeNames().sort().map((n) => `${n}=${e.getAttribute(n)}`)])));
  }
  assert.equal(new Set(glyphs).size, 9, "nine icons draw nine different glyphs");
  const h = await panel();
  h.deliver("noicon", v2("noicon", [
    { id: "root", component: "Column", children: ["a", "u"] },
    { id: "a", component: "Stat", label: "ABSENT", value: "1" },
    { id: "u", component: "Stat", label: "UNKNOWN", value: "2", icon: "tornado" },
  ]));
  const r = h.card("noicon");
  for (const cid of ["a", "u"]) assert.deepEqual(elements(comp(r, cid)).filter((e) => e.namespaceURI === SVG_NS), [], `${cid}: no glyph`);
  assert.deepEqual(chunks(r), ["ABSENT", "1", "UNKNOWN", "2"], "and no icon name as text");
});

rowB("B3: Steps is an ordered list (ol/li or list/listitem) with one item per step, each holding its ordinal text node", async () => {
  const { root } = await showExemplar("steps-omelette");
  const el = comp(root, "root") || kindEls(root, "Steps")[0];
  const items = elements(el).filter((e) => e.localName === "li" || e.getAttribute("role") === "listitem");
  assert.equal(items.length, 3, "three list items");
  assert.ok(elements(el).concat([el]).some((e) => e.localName === "ol" || e.getAttribute("role") === "list"), "inside an ol (or role=list)");
  items.forEach((li, k) => {
    const texts = textNodes(li).map((n) => n.data);
    assert.equal(texts[0], String(k + 1), `item ${k + 1} starts with its ordinal as a text node (note 5: not ::before)`);
  });
});

rowB("B3: a Source is its text with a glyph; Actions is one <button type=button> per label, in order", async () => {
  const { root: s } = await showExemplar("source-only");
  assert.equal(elements(s).filter((e) => e.localName === "svg" && e.namespaceURI === SVG_NS).length, 1, "the Source chip carries one glyph");
  const { root } = await showExemplar("actions-three");
  const buttons = elements(root).filter((e) => e.localName === "button");
  assert.deepEqual(buttons.map((b) => b.textContent), ["Show more", "Compare prices", "Start over"]);
  assert.ok(buttons.every((b) => b.getAttribute("type") === "button"), "type=button");
});

// =========================================================================================================
// B5: nothing is interactive
// =========================================================================================================

const FOCUSABLE_TAGS = new Set(["button", "input", "select", "textarea", "a", "summary", "details", "iframe"]);

rowB("B5: no input, select, textarea, contenteditable or positive tabindex in a v2 component; only Actions buttons are focusable", async () => {
  const problems = [];
  for (const e of EXEMPLARS) {
    const { root } = await showExemplar(e.key);
    for (const host of elements(root).filter((x) => V2_KINDS.includes(x.getAttribute("data-a2-kind")))) {
      const kind = host.getAttribute("data-a2-kind");
      for (const el of [host, ...elements(host)]) {
        const where = `${e.key}: ${kind} <${el.localName}>`;
        if (["input", "select", "textarea"].includes(el.localName)) problems.push(`${where}: a form control`);
        if (el.hasAttribute("contenteditable")) problems.push(`${where}: contenteditable`);
        const ti = el.getAttribute("tabindex");
        if (ti !== null && Number(ti) > 0) problems.push(`${where}: tabindex ${ti}`);
        const focusable = FOCUSABLE_TAGS.has(el.localName) || (ti !== null && Number(ti) >= 0);
        if (focusable && !(kind === "Actions" && el.localName === "button" && el.getAttribute("type") === "button")) problems.push(`${where}: focusable`);
      }
    }
  }
  assert.deepEqual(problems, [], "B5");
});

/** Clicks the button with `label` in surface `sid` with Date.prototype.toISOString frozen; -> h.sent. */
function clickFrozen(h, root, label) {
  const button = elements(root).find((e) => e.localName === "button" && e.textContent === label);
  assert.ok(button, `a <button> labelled ${JSON.stringify(label)}`);
  const real = Date.prototype.toISOString;
  Date.prototype.toISOString = function () {
    return "2026-10-28T09:10:11.123Z";
  };
  try {
    userClick(h.dom, button);
  } finally {
    Date.prototype.toISOString = real;
  }
}

rowB("B5: an Actions click sends exactly one astra.ui.action, byte-identical to a gate-made Button with that label", async () => {
  const sid = "turn-t7";
  const label = "Compare prices";
  const a = await panel();
  a.deliver(sid, v2(sid, [{ id: "root", component: "Actions", labels: ["Show more", label, "Start over"] }]));
  clickFrozen(a, a.card(sid), label);
  assert.equal(a.sent.length, 1, `exactly one message; sent ${JSON.stringify(a.sent)}`);
  assert.deepEqual(a.sent[0], {
    type: "astra.ui.action",
    surface_id: sid,
    name: "__gate.clicked",
    context: { label, a2ui: { sourceComponentId: "root", timestamp: "2026-10-28T09:10:11.123Z" } },
    data_model: null,
  });
  // the gate's own Button (component_repair.py RB: child "__gate_<id>_label", event __gate.clicked, context {label})
  const g = await panel();
  g.deliver(sid, basic(sid, [
    { id: "root", component: "Button", child: "__gate_root_label", action: { event: { name: "__gate.clicked", context: { label } } } },
    { id: "__gate_root_label", component: "Text", text: label },
  ]));
  clickFrozen(g, g.card(sid), label);
  assert.equal(g.sent.length, 1);
  assert.equal(JSON.stringify(a.sent[0]), JSON.stringify(g.sent[0]), "byte-identical to the gate-made Button's message (T-0164 B4)");
});

rowB("B5: each Actions button sends its own label; the first click on one button sends once, not once per button", async () => {
  const { h, root, sid } = await showExemplar("actions-three");
  for (const label of ["Start over", "Show more"]) clickFrozen(h, root, label);
  assert.deepEqual(h.sent.map((m) => [m.surface_id, m.name, m.context.label]), [[sid, "__gate.clicked", "Start over"], [sid, "__gate.clicked", "Show more"]]);
});

rowB("B5 (negative): any other click, key or focus in a v2 card sends nothing", async () => {
  for (const key of ["board-main", "board-compare", "board-chart", "steps-omelette", "stat-icons", "stattiles-storage", "actions-three"]) {
    const { h, root } = await showExemplar(key);
    const actionButtons = new Set(kindEls(root, "Actions").flatMap((a) => elements(a).filter((e) => e.localName === "button")));
    h.dom.quiet(() => {
      for (const el of [root, ...elements(root)]) {
        if (!actionButtons.has(el)) el.click();
        for (const k of ["Enter", " ", "ArrowRight", "Tab"]) el.dispatchEvent(new h.dom.FakeEvent("keydown", { key: k }));
        for (const k of ["Enter", " "]) el.dispatchEvent(new h.dom.FakeEvent("keyup", { key: k }));
        el.focus();
        el.blur();
        for (const type of ["pointerdown", "pointerup", "mousedown", "mouseup", "dblclick", "contextmenu", "wheel", "scroll"]) el.dispatchEvent(new h.dom.FakeEvent(type));
      }
    });
    assert.deepEqual(h.sent, [], `${key}: nothing but an Actions click sends (keys on an Actions button included: the browser's own synthesised click is the one path)`);
  }
});

rowB("B5: an older v2 card is inert and its Actions buttons send nothing; the newest sends", async () => {
  const h = await panel();
  h.deliver("older", v2("older", [{ id: "root", component: "Actions", labels: ["Old choice"] }]));
  h.deliver("newer", v2("newer", [{ id: "root", component: "Actions", labels: ["New choice"] }]));
  h.card("older");
  h.card("newer");
  assert.equal(h.wrapper("older").inert, true, "the older card's wrapper is inert (T-0154 B8)");
  clickFrozen(h, h.card("older"), "Old choice");
  assert.deepEqual(h.sent, [], "an inert card's button sends nothing");
  clickFrozen(h, h.card("newer"), "New choice");
  assert.deepEqual(h.sent.map((m) => [m.surface_id, m.context.label]), [["newer", "New choice"]]);
});

// =========================================================================================================
// B6: morph
// =========================================================================================================

rowB("B6: an identical v2 canvas re-applied makes zero DOM mutations in the card (every exemplar, 3 re-sends)", async () => {
  for (const e of EXEMPLARS) {
    const { h, root, sid } = await showExemplar(e.key);
    for (let k = 0; k < 3; k++) {
      h.dom.rec.mutations.length = 0;
      h.deliver(sid, e.messages);
      const ms = mutationsIn(h.dom, root);
      assert.equal(ms.length, 0, `${e.key}: re-send ${k + 1} mutated the card: ${summary(ms)}`);
    }
  }
});

/** The highest ancestor of `textNode` inside `within` that holds no other bar's label: that bar's row. */
function rowOf(textNode, within, otherLabels) {
  let row = textNode.parentNode;
  while (row.parentNode && row.parentNode !== within && !otherLabels.some((t) => t !== textNode && row.parentNode.contains(t))) row = row.parentNode;
  return row;
}

const BARS = (bars) => [{ id: "root", component: "Column", children: ["b", "tail"] }, { id: "b", component: "Bars", unit: "m", bars }, { id: "tail", component: "Source", text: "tail" }];

async function growBars(extra) {
  const h = await panel();
  const first = [{ label: "Alpha", value: 40 }, { label: "Beta", value: 100 }, { label: "Gamma", value: 70 }];
  h.deliver("grow", v2("grow", BARS(first)));
  const root = h.card("grow");
  const barsEl = comp(root, "b");
  const labels = textNodes(barsEl).filter((n) => first.some((b) => b.label === n.data));
  assert.equal(labels.length, 3, "premise: three label text nodes");
  const rows = labels.map((t) => rowOf(t, barsEl, labels));
  assert.equal(new Set(rows).size, 3, "premise: three distinct rows");
  const before = rows.map((r) => [r, ...walk(r)]);
  const tickNodes = textNodes(barsEl).filter((n) => !rows.some((r) => r.contains(n)));
  h.dom.rec.mutations.length = 0;
  h.deliver("grow", [COMPONENTS("grow", BARS([...first, extra]))]);
  return { h, root, barsEl, rows, before, tickNodes, labels };
}

rowB("B6: Bars grown by one smaller bar keeps every old row's nodes === and writes nothing in them (the max is unchanged)", async () => {
  const { h, barsEl, before, tickNodes } = await growBars({ label: "Delta", value: 10 });
  assert.ok(textNodes(barsEl).some((n) => n.data === "Delta"), "the new bar rendered");
  const old = keysOf(before.flat());
  for (const nodes of before) for (const n of nodes) assert.ok(barsEl.contains(n), `an old ${n.nodeName} was replaced or detached`);
  const writes = h.dom.rec.mutations.filter((m) => old.has(nodeKey(m.node)) && m.op !== "create");
  assert.deepEqual(writes.map((m) => `${m.op}:${m.node.nodeName}${typeof m.detail === "string" ? "(" + m.detail + ")" : ""}`), [], "no attribute, style, text or move in an old row (B6: widths only change with the maximum)");
  for (const t of tickNodes) assert.ok(barsEl.contains(t), "the tick labels are kept (the max did not change)");
  const ticks = keysOf(tickNodes);
  assert.deepEqual(h.dom.rec.mutations.filter((m) => ticks.has(nodeKey(m.node))).map((m) => m.op), [], "and not rewritten");
});

// B6a (reviewer A round 1): the value room comes from the maximum only, so a smaller value with a
// longer string (12.75 beside a maximum of 100) writes nothing in any old row and leaves the ticks alone.
rowB("B6a: Bars grown by a smaller value with a longer string (12.75 under a max of 100) writes nothing in any old row or tick", async () => {
  const { h, barsEl, before, tickNodes } = await growBars({ label: "Delta", value: 12.75 });
  assert.ok(textNodes(barsEl).some((n) => n.data === "12.75"), "the new value rendered as String(12.75)");
  for (const nodes of before) for (const n of nodes) assert.ok(barsEl.contains(n), `an old ${n.nodeName} was replaced or detached`);
  const old = keysOf(before.flat());
  const writes = h.dom.rec.mutations.filter((m) => old.has(nodeKey(m.node)) && m.op !== "create");
  assert.deepEqual(writes.map((m) => `${m.op}:${m.node.nodeName}${typeof m.detail === "string" ? "(" + m.detail + ")" : ""}`), [], "no write in an old row (its fill included)");
  for (const t of tickNodes) assert.ok(barsEl.contains(t), "the tick labels are kept");
  const ticks = keysOf(tickNodes);
  assert.deepEqual(h.dom.rec.mutations.filter((m) => ticks.has(nodeKey(m.node))).map((m) => m.op), [], "and not rewritten");
});

rowB("B6: Bars grown by a new maximum keeps every old row's nodes === and rewrites every old row's width", async () => {
  const { h, barsEl, rows, before } = await growBars({ label: "Delta", value: 250 });
  for (const nodes of before) for (const n of nodes) assert.ok(barsEl.contains(n), `an old ${n.nodeName} was replaced or detached`);
  rows.forEach((row, k) => {
    const sub = keysOf([row, ...walk(row)]);
    const widthWrites = h.dom.rec.mutations.filter((m) => sub.has(nodeKey(m.node)) && (m.op === "style" || (m.op === "attr" && m.detail === "style")));
    assert.ok(widthWrites.length >= 1, `old row ${k}: its width was rewritten for the new maximum`);
  });
});

rowB("B6: a kind change replaces only that component's subtree; siblings keep their nodes and get no write", async () => {
  const h = await panel();
  const kids = (x) => [{ id: "root", component: "Column", children: ["a", "x", "b"] }, { id: "a", component: "Stat", label: "KEEP A", value: "1" }, x, { id: "b", component: "Source", text: "keep b" }];
  h.deliver("kc", v2("kc", kids({ id: "x", component: "Bars", bars: [{ label: "bar", value: 3 }] })));
  const root = h.card("kc");
  const [a, x, b] = ["a", "x", "b"].map((id) => comp(root, id));
  const keep = keysOf([a, ...walk(a), b, ...walk(b)]);
  h.dom.rec.mutations.length = 0;
  h.deliver("kc", [COMPONENTS("kc", kids({ id: "x", component: "Steps", steps: [{ title: "now steps" }] }))]);
  assert.equal(comp(root, "a"), a);
  assert.equal(comp(root, "b"), b);
  assert.notEqual(comp(root, "x"), x, "the changed kind gets a new node");
  assert.equal(comp(root, "x").getAttribute("data-a2-kind"), "Steps");
  assert.equal(x.isConnected, false, "the old Bars node is detached");
  // T-0164's syncChildren may re-insert a following sibling's own node (a move, the node is kept);
  // nothing inside a sibling is written, replaced or removed
  const writes = h.dom.rec.mutations.filter((m) => keep.has(nodeKey(m.node)) && m.op !== "create" && !(m.op === "insert" && (m.node === a || m.node === b)));
  assert.deepEqual(writes.map((m) => `${m.op}:${m.node.nodeName}`), [], "no write in the siblings");
});

// =========================================================================================================
// B7: edge cases — no throw, no NaN
// =========================================================================================================

const EDGE_KEYS = ["bars-all-zero", "bars-single", "bars-huge", "comparison-highlight-out-of-range", "stattiles-one", "stattiles-four-wide", "stattiles-percent-bounds"];

rowB("B7: the edge exemplars render with no rejection, no render error and no NaN, Infinity or undefined anywhere", async () => {
  for (const key of EDGE_KEYS) {
    const { h, root } = await showExemplar(key);
    assert.ok(chunks(root).length >= 1, `${key}: something rendered`);
    const everywhere = [root.textContent, ...attributeText(root)];
    const bad = everywhere.filter((s) => /NaN|Infinity|undefined|null(?![a-z])/i.test(s));
    assert.deepEqual(bad, [], `${key}: a non-number leaked into text, an attribute or a style`);
    const own = h.logs.filter((l) => !/card sheet|mounted/.test(l));
    assert.deepEqual(own.filter((l) => /fail|error|throw|reject/i.test(l)), [], `${key}: no failure logged`);
  }
});

// B7a (group B amendment, 2026-10-28): when tick labels would collide, fewer are drawn, still 0 and the
// scale end, from the same closed list of steps, each String(n). The overlap itself needs layout and is
// measured in Chrome (tests/integration/test_t0184_genui_v2.py, test_b7a_*); here, the text half on a
// 13-digit maximum (the same payload as that module's t184-13digit).
rowB("B7a: a 13-digit maximum shows ticks that are String(n) of a closed-list scale from 0 (no abbreviation, no suffix, no NaN)", async () => {
  const h = await panel();
  const bars = [["Huge", 4321098765432], ["Tiny", 7]];
  h.deliver("t13", v2("t13", [{ id: "root", component: "Bars", title: "Thirteen digits", unit: "units", bars: bars.map(([label, value]) => ({ label, value })) }]));
  const root = h.card("t13");
  assert.deepEqual(closedListProblems(chunks(root), barsSpec({ title: "Thirteen digits", unit: "units", bars })), [], JSON.stringify(chunks(root)));
  const bad = [root.textContent, ...attributeText(root)].filter((s) => /NaN|Infinity|undefined/i.test(s));
  assert.deepEqual(bad, [], "a non-number leaked into text, an attribute or a style");
});

// =========================================================================================================
// Comparison: extra and missing cells
// =========================================================================================================

rowB("B4: a row with more cells than columns shows every cell; a row with fewer shows nothing in their place", async () => {
  const h = await panel();
  h.deliver("cells", v2("cells", [{ id: "root", component: "Comparison", columns: [{ title: "A" }, { title: "B" }], rows: [{ label: "R1", cells: ["a1", "b1", "x1-extra"] }, { label: "R2", cells: ["a2"] }, { label: "R3", cells: [] }] }]));
  const root = h.card("cells");
  assert.deepEqual(closedListProblems(chunks(root), { m: ["A", "B", "R1", "a1", "b1", "x1-extra", "R2", "a2", "R3"], d: [], ticks: null }), [], JSON.stringify(chunks(root)));
});

// =========================================================================================================
// B11: no sink, no fetch path, glyphs only through createElementNS
// =========================================================================================================

rowB("B11: no forbidden element, attribute, style url( or sink over every exemplar; every svg node is in the SVG namespace", async () => {
  const BAD_TAGS = ["img", "picture", "source", "video", "audio", "iframe", "object", "embed", "link", "script", "style", "a", "use", "image", "foreignobject"];
  for (const e of EXEMPLARS) {
    const { h, root } = await showExemplar(e.key);
    assert.deepEqual(h.dom.rec.violations, [], `${e.key}: the trapping DOM recorded a sink`);
    for (const el of elements(root)) {
      assert.ok(!BAD_TAGS.includes(el.localName.toLowerCase()), `${e.key}: no <${el.localName}>`);
      for (const name of el.getAttributeNames()) assert.ok(!["src", "srcset", "href", "xlink:href", "action", "formaction", "poster"].includes(name) && !/^on/.test(name), `${e.key}: no ${name}=`);
      assert.ok(!(el.getAttribute("style") || "").includes("url("), `${e.key}: no url( in a style`);
      if (["svg", "path", "circle", "g", "line", "rect", "polyline", "polygon", "ellipse"].includes(el.localName)) assert.equal(el.namespaceURI, SVG_NS, `${e.key}: <${el.localName}> made with createElementNS(SVG)`);
    }
  }
});

rowB("fake DOM: every DOM name the v2 renderer read over the exemplars is one the double models", async () => {
  const gaps = new Set();
  for (const e of EXEMPLARS) {
    const { h } = await showExemplar(e.key);
    for (const g of h.dom.rec.gaps) gaps.add(g);
  }
  assert.deepEqual([...gaps], [], "the renderer read a DOM name the double does not model: extend the double in ui_render.test.mjs's <fake-dom> region (test-author), do not change the renderer");
});
