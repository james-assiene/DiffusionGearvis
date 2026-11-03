// T-0056 criteria 3, 4 and 12 — hud/index.html as a *contract*, not as markup someone owns.
//
// The restyle (docs/tasks/T-0056.md) is allowed to touch exactly two files, and one of them is
// this shell. Four other files that the task may not edit reach into it by id, by attribute, by
// class name and by literal text; a modern-looking refactor of the shell breaks them silently,
// because nothing in the page throws when `getElementById` returns null — the panel simply never
// appears. Every row below therefore names *what reads the string* in its failure message, so the
// implementer who trips it knows which file they just disconnected.
//
// Textual, deliberately: there is no DOM here (node, no jsdom — the house rule is that
// tests/hud/*.test.mjs run under plain `node --test`, see tests/unit/test_hud_js.py). The tiny
// element scanner below is enough to answer the two structural questions criterion 4 asks:
// *is this `section.pane` a direct child of `<main>`*, and *what is inside each pane*.
//
// Status when written (before the restyle exists): every row except the two empty-state rows
// passes against the current shell. That is their point — they are the fence the restyle must
// not cross, and a row that only goes red after the feature lands would not be one.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const REPO = new URL("../../", import.meta.url);
const read = (rel) => readFileSync(fileURLToPath(new URL(rel, REPO)), "utf8");

const HTML = read("hud/index.html");
const STYLES = read("hud/styles.css");

// --- a very small element scanner ---------------------------------------------------------
//
// Not an HTML parser: it handles exactly what hud/index.html is allowed to contain — comments,
// void elements, and normal nested elements with quoted attributes. The first row below is the
// anti-vacuity guard: if the scanner ever stops understanding the shell, it fails there rather
// than letting every "still present" assertion pass over an empty tree.

const VOID = new Set([
  "area", "base", "br", "col", "embed", "hr", "img", "input",
  "link", "meta", "param", "source", "track", "wbr",
]);

const TAG_RE = '<!--[\\s\\S]*?-->|<(\\/?)([a-zA-Z!][a-zA-Z0-9-]*)((?:"[^"]*"|\'[^\']*\'|[^>])*?)(\\/?)>';

/** Parse `html` into a tree of {tag, attrs, parent, children, start}. */
function parse(html) {
  const root = { tag: "#root", attrs: "", parent: null, children: [], start: 0 };
  let current = root;
  const re = new RegExp(TAG_RE, "g");
  let match;
  while ((match = re.exec(html)) !== null) {
    if (match[0].startsWith("<!--")) continue;
    const [whole, closing, rawTag, attrs, selfClose] = match;
    const tag = rawTag.toLowerCase();
    if (tag.startsWith("!")) continue; // <!doctype html>
    if (closing) {
      let node = current;
      while (node && node.tag !== tag) node = node.parent;
      if (node && node.parent) current = node.parent;
      continue;
    }
    const node = { tag, attrs, parent: current, children: [], start: match.index + whole.length };
    current.children.push(node);
    if (!VOID.has(tag) && !selfClose) current = node;
  }
  return root;
}

const DOC = parse(HTML);

function walk(node, fn) {
  for (const child of node.children) {
    fn(child);
    walk(child, fn);
  }
}

function all(pred) {
  const out = [];
  walk(DOC, (n) => {
    if (pred(n)) out.push(n);
  });
  return out;
}

const classesOf = (node) => {
  const m = /\sclass\s*=\s*"([^"]*)"/.exec(" " + node.attrs);
  return m ? m[1].trim().split(/\s+/).filter(Boolean) : [];
};
const hasClass = (node, name) => classesOf(node).includes(name);
const hasAttr = (node, name) =>
  new RegExp("(^|\\s)" + name + "(\\s|=|$)", "i").test(" " + node.attrs.trim());
const attrValue = (node, name) => {
  const m = new RegExp("(?:^|\\s)" + name + '\\s*=\\s*"([^"]*)"', "i").exec(" " + node.attrs);
  return m ? m[1] : null;
};

/** The source between a node's `>` and its matching closing tag (depth-aware). */
function innerHtml(node) {
  const re = new RegExp(TAG_RE, "g");
  re.lastIndex = node.start;
  let depth = 0;
  let m;
  while ((m = re.exec(HTML)) !== null) {
    if (m[0].startsWith("<!--")) continue;
    const tag = (m[2] || "").toLowerCase();
    if (tag.startsWith("!") || VOID.has(tag) || m[4]) continue;
    if (m[1]) {
      if (depth === 0 && tag === node.tag) return HTML.slice(node.start, m.index);
      depth -= 1;
    } else if (tag === node.tag) {
      depth += 1;
    }
  }
  return HTML.slice(node.start);
}

const textOf = (node) =>
  innerHtml(node)
    .replace(/<!--[\s\S]*?-->/g, "")
    .replace(/<[^>]*>/g, "")
    .replace(/\s+/g, " ")
    .trim();

test("the scanner really reads hud/index.html (the guard on every row below)", () => {
  assert.ok(HTML.length > 500, "hud/index.html is the page shell (T-0008); it cannot be empty");
  const mains = all((n) => n.tag === "main");
  assert.equal(
    mains.length,
    1,
    "the page has exactly one <main> — the mount point hud/src/assists/index.js and " +
      "hud/src/log/index.js resolve with querySelector('main') (criterion 4)",
  );
  const sections = all((n) => n.tag === "section");
  assert.ok(
    sections.length >= 4,
    "the scanner found " + sections.length + " <section> elements; the shell declares four panes " +
      "(spoken, you said, shown, what the model sees). A lower number means the scanner, not the " +
      "page, is broken — and a broken scanner would make every row in this module vacuous.",
  );
  assert.ok(
    textOf(all((n) => n.tag === "h2")[0]).length > 0,
    "innerHtml()/textOf() return the text of an element; if they returned '' the heading row " +
      "below would compare four empty strings and pass against any shell",
  );
});

// --- criterion 3: the ids, the attributes, the headings, the ask form, the placeholder -------

/** id -> the file that looks it up, and what breaks when it is gone. */
const IDS = {
  "turn-chip": 'hud/src/main.js el("turn-chip") -> panels/thinking.js writes the turn state on it',
  "thinking-bar": "hud/src/main.js -> panels/thinking.js drawMarks() and the inline --fill",
  "thinking-meter": "hud/src/main.js -> panels/thinking.js writes data-thinking-ms and the text",
  // Amended by T-0154 B1 (G2): "spoken-pane", "transcript-pane" and "shown-channel" are retired
  // by name ("the pins of T-0056 criterion 3 and T-0128 criterion 3 on them are superseded here by
  // name"); the conversation lives in [data-thread]. Their absence is the B1 row at the end.
  "capture-banner": "hud/src/main.js: the SYNTHETIC CAPTURE banner (T-0008 criterion 7)",
  "error-strip": "hud/src/main.js renderErrors(); tests/hud/ui_action.test.mjs reads it too",
  "conn-status": "hud/src/main.js onStatus(); hud/src/ask/index.js reads it (tests/hud/ask.test.mjs)",
  "frames-sent": "hud/src/main.js renderCounters()",
  "frames-dropped": "hud/src/main.js renderCounters() (T-0008 criterion 3: the drop count shows)",
  "capture-size": "hud/src/main.js: the capture size before the resolution policy",
  "ignored-count": "hud/src/main.js renderCounters(): unknown/malformed/realtime",
  preview: "hud/src/main.js: the frame thumbnail — the page's pixel-honesty pane",
  "share-button": "hud/src/main.js: the only way a human starts capture",
  // T-0181 B0/B3: the id stays (the hook of ~8 test modules and styles.css); its label is now
  // "Stop sharing" and it has no keyboard shortcut (H-T181-4).
  "mute-button": "hud/src/main.js: the Stop sharing control (stops the screen or camera capture)",
};

test("T-0056 criterion 3: every id the page's JS looks up is still spelled the same", () => {
  const present = new Set(
    all((n) => attrValue(n, "id") !== null).map((n) => attrValue(n, "id")),
  );
  for (const [id, reader] of Object.entries(IDS)) {
    assert.ok(
      present.has(id),
      'hud/index.html no longer declares id="' + id + '". It is load-bearing: ' + reader + ". " +
        "getElementById() returns null and the page keeps running with that feature silently " +
        "missing — no exception, no red row anywhere else — which is why criterion 3 freezes the " +
        "spelling for a restyle whose whole claim is that it changed nothing but presentation.",
    );
  }
});

/** data-attribute -> the file that selects on it. */
const DATA_ATTRS = {
  "data-turn-state":
    "panels/thinking.js writes it; tests/integration/test_hud_headless.py selects on it",
  "data-thinking-ms":
    "panels/thinking.js writes it; test_hud_headless.py reads the value as integer ms",
  "data-frames-sent": "hud/src/main.js renderCounters(); test_hud_headless.py asserts the count",
  "data-frames-dropped": "hud/src/main.js renderCounters(); T-0008 criterion 3 reads it",
  "data-capture-size": 'hud/src/main.js; test_hud_headless.py reads "WxH" before the policy',
  // Amended by T-0154 B1/B3 (G2): "data-transcript" was #transcript-pane's attribute; with that
  // pane retired, [data-transcript] is the user bubble the thread creates per turn
  // (tests/hud/thread.test.mjs), so the shell no longer declares it. [data-thread] is pinned by
  // the B1 row at the end instead.
  "data-ask-form": "hud/src/ask/index.js mount(): the form it attaches its submit handler to",
  "data-ask-input": "hud/src/ask/index.js: the input it enables and reads (T-0039)",
  "data-ask-submit": "hud/src/ask/index.js: the button it enables (T-0039)",
};

test("T-0056 criterion 3: every data-* hook a panel or a test selects on is still in the shell", () => {
  for (const [attr, reader] of Object.entries(DATA_ATTRS)) {
    const nodes = all((n) => hasAttr(n, attr));
    assert.ok(
      nodes.length >= 1,
      "hud/index.html no longer carries [" + attr + "]. " + reader + ". A selector that matches " +
        "nothing throws nowhere: the panel mounts into undefined and the feature disappears from " +
        "the page while every other test stays green. Criterion 3 freezes this list.",
    );
  }
});

// T-0128 amends this pin (docs/tasks/T-0128.md criterion 3, "Named pin changes"): the thumb
// pane's heading becomes "What I see", the headings become sentence case, and the shell declares
// two new rail panes, Thinking and Diagnostics. The spoken, you-said and shown headings keep their
// words. The comparison is case-insensitive here because "sentence case" is a *rendered* property
// (an `h2::first-letter { text-transform: uppercase }` and an HTML edit both satisfy it); the
// rendered casing is asserted in tests/integration/test_hud_redesign.py. The order is not pinned
// any more: the rail's order is the layout's business, and T-0128 adds panes to it.
//
// Removed by T-0128 (criterion 3): the T-0056 assertion that the <h2> list is exactly
// ["spoken", "you said", "shown", "what the model sees"], in that order and in that case.
//
// **Amended by T-0154 B1** (G2, "the pins of T-0056 criterion 3 and T-0128 criterion 3 on them are
// superseded here by name"): the spoken, you-said and shown panes are retired into one
// conversation pane, so their three headings leave the list. What stays pinned: the shell's rail
// headings are exactly the three below (the assist and log panes append their own). Removed
// assertion: the case-folded <h2> list equals ["spoken", "you said", "shown", "what i see",
// "thinking", "diagnostics"]. The conversation pane's own heading, if it has one, is the B1 row's
// business at the end of this file (the mockup's conversation pane has none; B1 names none).
const HEADINGS_RAIL_T0154 = ["what i see", "thinking", "diagnostics"];
const RETIRED_HEADINGS_T0154 = ["spoken", "you said", "shown"];
// T-0128's rows used to skip while hud/src/theme.js was absent (the node stand-in for `pending`).
// T-0128 is implemented, so they always run (reviewer finding, low).

/** The one direct child of <main> that holds [data-thread] (T-0154 B1), or null. */
const conversationPane = () => {
  const main = all((n) => n.tag === "main")[0];
  if (!main) return null;
  const holders = main.children.filter((c) => hasAttr(c, "data-thread") || all((n) => hasAttr(n, "data-thread")).some((t) => {
    for (let p = t.parent; p; p = p.parent) if (p === c) return true;
    return false;
  }));
  return holders.length === 1 ? holders[0] : null;
};
const insideNode = (node, ancestor) => {
  for (let p = node.parent; p; p = p.parent) if (p === ancestor) return true;
  return false;
};

test("T-0128 criterion 3 (amended by T-0154 B1): the thumb pane's heading is \"What I see\"", () => {
  const thumb = all((n) => n.tag === "section" && hasClass(n, "thumb"))[0];
  assert.ok(thumb, 'the frame pane is still <section class="pane thumb"> (criterion 4 names it)');
  const thumbHeading = all((n) => n.tag === "h2").find((h) => {
    for (let p = h.parent; p; p = p.parent) if (p === thumb) return true;
    return false;
  });
  assert.equal(
    thumbHeading ? textOf(thumbHeading).toLowerCase() : null,
    "what i see",
    "criterion 3: it is the *thumb* pane whose heading becomes \"What I see\"",
  );
});

test("T-0056 criterion 3: the ask form is still method=dialog with both controls disabled", () => {
  const form = all((n) => n.tag === "form" && hasAttr(n, "data-ask-form"))[0];
  assert.ok(form, "hud/index.html declares the ask form (T-0039 criterion 10) with [data-ask-form]");
  assert.equal(
    attrValue(form, "method"),
    "dialog",
    'the ask form must stay method="dialog". T-0039 ruling A19: with any other method a submit ' +
      "that reaches the browser navigates, and the navigation drops the ?ws= and ?source= query " +
      "the page was opened with — the socket then points at the default and the session the human " +
      "was in is gone. This is the shell's own guard for the seconds before src/ask/index.js " +
      "mounts, and for a page where that panel failed to load.",
  );
  for (const attr of ["data-ask-input", "data-ask-submit"]) {
    const control = all((n) => hasAttr(n, attr))[0];
    assert.ok(control, "hud/index.html declares [" + attr + "]");
    assert.ok(
      hasAttr(control, "disabled"),
      "[" + attr + "] must ship with the `disabled` attribute in the HTML. T-0039 ruling A19 (2): " +
        "the controls are inert until hud/src/ask/index.js calls mount() and enables them, so " +
        "before the panel mounts a keystroke cannot submit anything. Removing `disabled` because " +
        "the restyle draws its own disabled look re-opens exactly the bug that attribute was " +
        "added for.",
    );
  }
});

// Removed by T-0154 B1 (G2): "T-0056 criterion 3: the shown-channel placeholder is byte-for-byte
// what ui/index.js removes" — B1 retires `no surface yet (T-0010)` and #shown-channel by name. Its
// absence is asserted by the B1 row at the end of this file.
const PLACEHOLDER_TEXT = "no surface yet (T-0010)";

// --- criterion 4: <main> and .pane are a cross-file contract ----------------------------------

test("T-0056 criterion 4, both sides read: the panels mount .pane into main, and main holds panes", () => {
  // One-sided would be worthless. If only the JS side were asserted, a shell that renamed the
  // class would pass; if only the shell were asserted, a panel that stopped using it would pass.
  for (const rel of ["hud/src/assists/index.js", "hud/src/log/index.js"]) {
    const src = read(rel);
    assert.match(
      src,
      /className\s*=\s*"pane"/,
      rel + ' mounts its panel with panel.className = "pane" — that is the whole of how it ' +
        "inherits the pane treatment from hud/styles.css. The file is not in T-0056's Owned " +
        "files, so the restyle may not follow a class rename into it: the class name is fixed on " +
        "this side and hud/styles.css has to keep styling it.",
    );
    assert.match(
      src,
      /querySelector\(\s*"main"\s*\)/,
      rel + ' mounts with doc.querySelector("main").appendChild(panel). If <main> stops existing ' +
        "the panel falls back to document.body (the `|| doc.body` on that line) and lands outside " +
        "the grid entirely — visible, unstyled and below everything, with nothing thrown.",
    );
  }
  const main = all((n) => n.tag === "main")[0];
  const sections = main.children.filter((n) => n.tag === "section");
  assert.ok(sections.length >= 4, "the shell declares its panes as <section> children of <main>");
  for (const section of sections) {
    assert.ok(
      hasClass(section, "pane"),
      "a <section> child of <main> without class=\"pane\" (classes: " +
        (classesOf(section).join(" ") || "none") + "). Criterion 4: a declared pane and a pane a " +
        "panel appends must get the same padding, background, radius and grid placement; the only " +
        "way that holds for both is if both are `main > .pane`.",
    );
  }
});

test("T-0056 criterion 4, adversarial: no wrapper element between <main> and the panes", () => {
  // The most likely shell failure of this task. `main > div.grid > section.pane` is how a modern
  // CSS grid is usually written, and it is invisible in a screenshot of today's page: the four
  // declared panes look right, and every panel that appends itself to <main> (assists, log, and
  // T-0047's listen panel) lands *outside* the grid container as an unstyled block.
  const main = all((n) => n.tag === "main")[0];
  const offenders = all((n) => hasClass(n, "pane")).filter((n) => n.parent !== main);
  assert.deepEqual(
    offenders.map((n) => "<" + n.tag + " " + n.attrs.trim() + ">"),
    [],
    'every element with class="pane" must be a *direct* child of <main>. ' +
      "hud/src/assists/index.js and hud/src/log/index.js (and T-0047's listen panel) do " +
      'doc.querySelector("main").appendChild(panel) with panel.className = "pane"; they cannot ' +
      "reach a wrapper, and T-0056 may not edit them to teach them about one. The grid container " +
      "therefore has to be <main> itself.",
  );
  const wrappers = main.children.filter((n) => n.tag !== "section" && !VOID.has(n.tag));
  assert.deepEqual(
    wrappers.map((n) => "<" + n.tag + " " + n.attrs.trim() + ">"),
    [],
    "<main> has a non-<section> element child. Even an empty wrapper changes what `main > .pane` " +
      "selects and where an appended panel lands relative to the declared ones (criterion 4's " +
      "'same grid placement'), and it is the shape a grid refactor reaches for first.",
  );
});

// --- criterion 12: an honest empty state, written in HTML --------------------------------------

/** The declared panes that need their own empty line, by an id inside them.
 *  **Amended by T-0154 B1** (G2): the "spoken-pane" and "transcript-pane" rows are retired with
 *  those panes; the thread's empty line (B1: "HTML copy hidden by :has()") is the B1 row at the
 *  end of this file. Removed too: the assertion that #shown-channel's empty state is the frozen
 *  placeholder (B1 retires both). */
const EMPTY_PANES = [
  ["preview", "the frame pane shows nothing until the human presses Share"],
];

test("T-0056 criterion 12: each declared pane carries a human-readable empty line in the HTML", () => {
  const main = all((n) => n.tag === "main")[0];
  const panes = main.children.filter((n) => n.tag === "section");
  for (const [rootId, why] of EMPTY_PANES) {
    const pane = panes.find((p) => innerHtml(p).includes('id="' + rootId + '"'));
    assert.ok(pane, "the pane holding #" + rootId + " is still declared in hud/index.html");
    const empties = [];
    walk(pane, (n) => {
      if (hasAttr(n, "data-empty")) empties.push(n);
    });
    assert.ok(
      empties.length >= 1,
      "the pane holding #" + rootId + " declares no [data-empty] element. Criterion 12: " + why +
        ", and the page must say so in words instead of showing a blank box. The copy lives in " +
        "hud/index.html — not in a CSS `content:` string — so that it is real text: in the " +
        "accessibility tree, selectable, and present in the DOM that " +
        "tests/integration/*.py and scripts/hud_dom.sh read. [data-empty] is the attribute the " +
        "task's implementation notes use in the hiding rule " +
        "(`.pane:has(#spoken-pane:not(:empty)) [data-empty] { display: none }`), and " +
        "tests/integration/test_hud_chrome.py drives the same attribute; both would have to " +
        "change together if the implementer wants another name.",
    );
    for (const node of empties) {
      assert.ok(
        textOf(node).length >= 3,
        "[data-empty] in the pane holding #" + rootId + " has no words in it (" +
          JSON.stringify(textOf(node)) + "). An empty empty-state is the blank box criterion 12 " +
          "exists to remove.",
      );
    }
  }
});

test("T-0056 criterion 12: no pane's empty copy is smuggled into a CSS content: string", () => {
  // `content: "nothing spoken yet"` renders the same and is invisible to everything that is not
  // a pair of eyes.
  const declarations = [...STYLES.matchAll(/content\s*:\s*([^;}]+)/g)].map((m) => m[1].trim());
  const prose = declarations.filter((value) => {
    const quoted = [...value.matchAll(/"([^"]*)"|'([^']*)'/g)].map((m) => m[1] ?? m[2]);
    return quoted.some((s) => /[A-Za-z]{3,}/.test(s));
  });
  assert.deepEqual(
    prose,
    [],
    "hud/styles.css declares content: with prose in it. Criterion 12 puts every empty-state line " +
      "in hud/index.html: generated content is not text a screen reader reliably announces, is " +
      "not selectable, and does not appear in the DOM dumps the integration tests and " +
      "scripts/hud_dom.sh read — so a pane that 'says it is empty' only in CSS says it to nobody " +
      "this project can test. (content: \"\" and content: none for decorative rules are fine; " +
      "this row only matches quoted strings with words in them.)",
  );
});

// --- T-0128: the header, the rail panes, the Thinking pane's honest empty state, the control -----
//
// Class names, as the task writes them: `pane-thinking` and `pane-diagnostics` are pane classes in
// the same family as `pane-spoken` / `pane-said` / `pane-shown`, and "pane thumb" is the frame
// pane's existing class pair. `[data-denoise-root]` is the mount point T-0128c's
// `./denoise/index.js` (already in PANEL_SPECS) will fill.

const childOf = (node, ancestor) => {
  for (let p = node.parent; p; p = p.parent) if (p === ancestor) return true;
  return false;
};
const mainNode = () => all((n) => n.tag === "main")[0];
const paneWithClass = (cls) =>
  mainNode().children.filter((n) => n.tag === "section" && hasClass(n, "pane") && hasClass(n, cls));
const byId = (id) => all((n) => attrValue(n, "id") === id)[0];
const header = () => all((n) => n.tag === "header")[0];

/** Criterion 5, verbatim. Straight apostrophe and an em dash, as the task writes it. */
const DENOISE_EMPTY = "no denoising data yet — this page does not receive the plugin's per-step canvases";
const decodeEntities = (s) =>
  s
    .replace(/&#39;|&apos;|&#x27;/gi, "'")
    .replace(/&mdash;|&#8212;|&#x2014;/gi, "—")
    .replace(/&amp;/g, "&");

test("T-0128 criteria 4-5: the rail panes are declared as direct children of <main>", () => {
  for (const cls of ["thumb", "pane-thinking", "pane-diagnostics"]) {
    const panes = paneWithClass(cls);
    assert.equal(
      panes.length,
      1,
      `hud/index.html must declare exactly one <section class="pane ${cls}"> as a direct child of ` +
        `<main> (found ${panes.length}). Criterion 4 names the five rail panes: pane thumb, ` +
        "pane-thinking, the assist pane, the log pane and pane-diagnostics; the assist and log " +
        "panes are appended by their own JS, the other three are the shell's.",
    );
  }
});

test("T-0128 criterion 5: #thinking-meter and #thinking-bar live in the Thinking pane", () => {
  const [pane] = paneWithClass("pane-thinking");
  assert.ok(pane, "premise: the Thinking pane is declared");
  for (const id of ["thinking-meter", "thinking-bar"]) {
    const node = byId(id);
    assert.ok(node, `#${id} is still declared (T-0056 criterion 3's id list)`);
    assert.ok(
      childOf(node, pane),
      `criterion 5: #${id} must live inside section.pane-thinking. It is the real measurement ` +
        "(thinking_ms, written by hud/src/panels/thinking.js) and moves into the Thinking pane by " +
        "id (implementation note 2); left in the header it is a second thinking display.",
    );
  }
});

test("T-0128 criterion 5 (negative): [data-denoise-root] is empty and its empty line is HTML copy", () => {
  const [pane] = paneWithClass("pane-thinking");
  assert.ok(pane, "premise: the Thinking pane is declared");
  const roots = all((n) => hasAttr(n, "data-denoise-root"));
  assert.equal(roots.length, 1, "exactly one [data-denoise-root] in the shell (criterion 5)");
  const root = roots[0];
  assert.ok(childOf(root, pane), "[data-denoise-root] is inside the Thinking pane");
  const inner = innerHtml(root).replace(/<!--[\s\S]*?-->/g, "");
  assert.equal(
    /<[a-zA-Z]/.test(inner),
    false,
    "criterion 5: [data-denoise-root] holds no element children in the HTML (found " +
      JSON.stringify(inner.trim().slice(0, 120)) + "). No placeholder grid, shimmer or pulse is " +
      "drawn (implementation note 2; the mockup's .noise grid is not carried over): the conductor " +
      "does not relay the plugin tap to the page, so anything drawn here would be a fake " +
      "trajectory presented as real (CLAUDE.md, D15).",
  );
  assert.equal(inner.trim(), "", "and no text either: the empty line sits beside the root, not in it");
  // The copy itself: T-0128 criterion 5's line until T-0128c group B is activated, then B5's
  // (T-0128c amends criterion 5's copy only, H-T128c-6). The row below pins B5's.
  const paneText = decodeEntities(textOf(pane));
  const want = T0128C_B_PENDING ? DENOISE_EMPTY : DENOISE_EMPTY_T0128C;
  assert.ok(
    paneText.includes(want),
    "criterion 5: the Thinking pane's empty line reads, in the HTML: " + JSON.stringify(want) +
      ". The pane's text is " + JSON.stringify(paneText) + ". It is real text so that a screen " +
      "reader, a DOM dump and this test can all read it (T-0056 criterion 12's rule).",
  );
});

// --- T-0128c group B: the empty line's new copy and the label's slot (B4, B5) -----------------
//
// Gated like tests/hud/denoise.test.mjs: strict xfail until docs/tasks/T-0128c.md's **Status:**
// records "group B: tests <a>..<b>, activated <sha>"; `ASTRA_RUN_PENDING=1` runs it now.

/** T-0128c B5, verbatim (H-T128c-6). */
const DENOISE_EMPTY_T0128C = "no denoising data yet — the model's real canvases appear here while it thinks";
const T0128C_STATUS = (() => {
  const file = fileURLToPath(new URL("docs/tasks/T-0128c.md", REPO));
  return existsSync(file) ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(file, "utf8")) || [])[1] || "").trim() : "";
})();
const T0128C_B_PENDING =
  !/\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0128C_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0128cB(name, fn) {
  test(name + (T0128C_B_PENDING ? " [pending T-0128c B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!T0128C_B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null)
      assert.fail("XPASS(strict): this T-0128c group B row passed before group B's activation is recorded.");
    t.diagnostic("expected failure (T-0128c B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

pendingT0128cB("T-0128c B5: the Thinking pane's empty line is B5's copy, and the old copy is gone", () => {
  const [pane] = paneWithClass("pane-thinking");
  assert.ok(pane, "premise: the Thinking pane is declared");
  const empties = all((n) => hasAttr(n, "data-denoise-empty"));
  assert.equal(empties.length, 1, "one [data-denoise-empty] (the line hud/src/denoise/index.js rewrites for denoise_no_tap and synthetic)");
  assert.ok(childOf(empties[0], pane), "the empty line is in the Thinking pane");
  assert.equal(
    decodeEntities(textOf(empties[0])).replace(/\s+/g, " ").trim(),
    DENOISE_EMPTY_T0128C,
    "T-0128c B5: the empty line reads, in the HTML, " + JSON.stringify(DENOISE_EMPTY_T0128C),
  );
  assert.equal(decodeEntities(textOf(pane)).includes(DENOISE_EMPTY), false, "the T-0128 copy ('this page does not receive …') is no longer true and is gone");
});

pendingT0128cB("T-0128c B4: one empty [data-denoise-label] beside the Thinking heading, outside the root; the heading is unchanged", () => {
  const [pane] = paneWithClass("pane-thinking");
  assert.ok(pane, "premise: the Thinking pane is declared");
  const labels = all((n) => hasAttr(n, "data-denoise-label"));
  assert.equal(labels.length, 1, "T-0128c B4: exactly one [data-denoise-label] in the shell");
  const label = labels[0];
  assert.ok(childOf(label, pane), "B4: the label sits in the Thinking pane's heading row");
  const [root] = all((n) => hasAttr(n, "data-denoise-root"));
  assert.ok(!childOf(label, root), "the label is not inside [data-denoise-root] (B5: the root holds no element until a frame)");
  const h2s = pane.children.filter((n) => n.tag === "h2");
  assert.equal(h2s.length, 1, "implementation note: the heading markup T-0154's parity rows measure is not restructured");
  assert.ok(!childOf(label, h2s[0]), "a sibling label element, not text inside the h2");
  assert.equal(decodeEntities(textOf(h2s[0])).trim(), "Thinking", "the heading still reads Thinking");
  assert.equal(textOf(label).trim(), "", "B5: no label text before the first frame (nothing fabricated in the HTML)");
});

test("T-0128 header: the turn pill, Share, Mute and the Settings control; the counters left it", () => {
  const bar = header();
  assert.ok(bar, "the page still has a <header>");
  for (const id of ["turn-chip", "share-button", "mute-button"]) {
    assert.ok(byId(id) && childOf(byId(id), bar), `the Goal's header holds #${id}`);
  }
  const [diag] = paneWithClass("pane-diagnostics");
  assert.ok(diag, "premise: the Diagnostics pane is declared");
  for (const id of ["frames-sent", "frames-dropped", "capture-size", "ignored-count", "conn-status"]) {
    const node = byId(id);
    assert.ok(node, `#${id} is still declared (T-0056 criterion 3)`);
    assert.ok(
      childOf(node, diag),
      `Owned files: section.pane-diagnostics holds "the counters and #conn-status"; #${id} is not in it`,
    );
  }
});

/** The header's <select> whose option values are the three preferences. */
function themeSelects() {
  const bar = header();
  if (!bar) return [];
  return all((n) => n.tag === "select" && childOf(n, bar));
}

test("T-0128 criterion 8: the Settings control is one native, labelled select with exactly three options", () => {
  const selects = themeSelects();
  assert.equal(
    selects.length,
    1,
    "criterion 8: the Settings control is a *native* control in the header — read here as one " +
      "<select> (the only native control that holds exactly three options as one control). Found " +
      selects.length + " <select> in <header>.",
  );
  const select = selects[0];
  const options = [];
  walk(select, (n) => {
    if (n.tag === "option") options.push([attrValue(n, "value"), textOf(n)]);
  });
  assert.deepEqual(
    options,
    [
      ["system", "System"],
      ["light", "Light"],
      ["dark", "Dark"],
    ],
    "criterion 8: exactly three options, System, Light and Dark, whose values are the three " +
      "preference strings of criterion 6 (system | light | dark). Found " + JSON.stringify(options),
  );
  const id = attrValue(select, "id");
  const labelled =
    (attrValue(select, "aria-label") || "").trim().length > 0 ||
    (attrValue(select, "aria-labelledby") || "").trim().length > 0 ||
    (id && all((n) => n.tag === "label" && attrValue(n, "for") === id).length > 0) ||
    all((n) => n.tag === "label").some((l) => childOf(select, l));
  assert.ok(
    labelled,
    "criterion 8: the control is labelled — aria-label, aria-labelledby, a <label for> or a " +
      "wrapping <label>. An unlabelled select is announced as \"combo box, System\" and nothing else.",
  );
});

test("T-0128: the theme module is loaded by a relative <script> of its own", () => {
  const html = HTML.replace(/<!--[\s\S]*?-->/g, " ");
  const srcs = [...html.matchAll(/<script\b[^>]*src\s*=\s*"([^"]*)"/g)].map((m) => m[1]);
  assert.ok(
    srcs.some((s) => /^(\.\/)?src\/theme\.js$/.test(s)),
    "Owned files: hud/index.html carries \"the theme script tag(s)\"; hud/src/main.js is not this " +
      "task's to edit, so nothing else can load hud/src/theme.js. Script srcs found: " + JSON.stringify(srcs),
  );
});

// --- criterion 5, the project rule underneath it: no randomness anywhere a trajectory is drawn ----

/** hud/-relative paths of the page files (any of .js/.mjs/.cjs/.html/.css) under `dir` that name
 *  Math.random or getRandomValues, but hud/src/ws.js. No directory is skipped (T-0164 D2: the
 *  vendor/ and node_modules/ skips died with the vendored bundle). */
function randomOffenders(dir) {
  const offenders = [];
  const visit = (here, rel) => {
    for (const entry of readdirSync(here)) {
      const full = join(here, entry);
      const relPath = rel ? rel + "/" + entry : entry;
      if (statSync(full).isDirectory()) {
        visit(full, relPath);
      } else if (/\.(js|mjs|cjs|html|css)$/.test(entry)) {
        if (relPath === "src/ws.js") continue;
        const text = readFileSync(full, "utf8");
        if (/Math\.random|getRandomValues/.test(text)) offenders.push("hud/" + relPath);
      }
    }
  };
  visit(dir, "");
  return offenders;
}

const HUD_DIR = fileURLToPath(new URL("hud", REPO));

test("T-0128 criterion 5: Math.random and getRandomValues appear nowhere in hud/ but ws.js's backoff", () => {
  // A fence: true on the base commit (measured: hud/src/ws.js's backoffDelay is the only hit), and
  // what stops a "denoising placeholder" from being a random grid. Literal text, comments
  // included, because the criterion says "appear nowhere" — prose naming them trips it too.
  // T-0164 D2: every directory under hud/ is descended; a re-added vendor/ (or node_modules/) file
  // naming either is an offender like any other.
  const offenders = randomOffenders(HUD_DIR);
  assert.deepEqual(
    offenders,
    [],
    "criterion 5 / T-0164 D2: Math.random / crypto.getRandomValues appear in " + offenders.join(", ") +
      ". The only allowed use is hud/src/ws.js's reconnect backoff (no vendor/ exemption since " +
      "T-0164 D). Randomness in the page is how a fake denoising trajectory gets drawn.",
  );
  assert.match(read("hud/src/ws.js"), /Math\.random/, "anti-vacuity: the scan can see the one allowed use");
});

// --- T-0154 group A: the header row, the Mute shortcut, no dead controls, the caption hook --------
//
// docs/tasks/T-0154.md A4 and A6, rulings O-A. Two kinds of row below:
//
//   * fences (plain `test`): kept contracts that hold on today's shell and must keep holding
//     after the restyle — Mute keeps `m`, and no header element is a control with nothing behind
//     it. They pass before the implementation, which is their point (this file's header).
//   * rows written ahead of the implementation (`pendingT0154`). Node tests cannot carry pytest's
//     `pending` marker. The precedent (tests/hud/assists.test.mjs lines 8-10 and 50-56) skips a
//     file while the module it tests does not exist — but hud/index.html exists and is *edited*
//     by T-0154, so there is nothing to wait for. The stand-in reads the task's own **Status:**
//     line instead: while it begins with "planned", each such row runs and must FAIL (strict
//     xfail: a row that passes throws "XPASS(strict)" and turns `node --test` red, so a vacuous
//     row or an unannounced implementation is loud). As soon as the status leaves "planned" (the
//     test-author moves it to "in progress" when /implement T-0154 starts, as for the pytest
//     markers), every row runs for real. `ASTRA_RUN_PENDING=1 node --test tests/hud` runs them for
//     real now (the node twin of `pytest --pending`), which is how their failures are shown.

const T0154_TASK = fileURLToPath(new URL("docs/tasks/T-0154.md", REPO));
const T0154_STATUS = existsSync(T0154_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0154_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0154_PENDING = /^planned\b/i.test(T0154_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0154(name, fn) {
  test(name + (T0154_PENDING ? " [pending T-0154: strict xfail while the task is planned]" : ""), async (t) => {
    if (!T0154_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0154 row passed while docs/tasks/T-0154.md's status is " +
          JSON.stringify(T0154_STATUS.slice(0, 60)) + ". Either the implementation has landed (move " +
          "the status to \"in progress\" and the row runs for real) or the row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0154 planned): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0154 pending stand-in: the task file's status is readable (the guard on the rows below)", () => {
  assert.ok(
    T0154_STATUS.length > 0,
    "docs/tasks/T-0154.md must exist and carry a **Status:** line; without it every pendingT0154 " +
      "row would silently run as an ordinary row (or, with a typo, as an xfail forever)",
  );
});

pendingT0154("T-0154 A4: #capture-banner sits inside <header>, not in a second row under it", () => {
  const bar = header();
  assert.ok(bar, "premise: the page has a <header>");
  const banner = byId("capture-banner");
  assert.ok(banner, "#capture-banner is still declared (T-0056 criterion 3; hud/src/main.js writes it)");
  assert.ok(
    childOf(banner, bar),
    "A4: during a synthetic capture #capture-banner (SYNTHETIC CAPTURE, T-0008 criterion 7) sits in " +
      "the header row; there is no second row under the header. It is declared outside <header> " +
      "today, as a sibling between the header and <main>.",
  );
});

pendingT0154("T-0154 A6: the What I see pane declares one empty [data-capture-caption]", () => {
  const [thumb] = paneWithClass("thumb");
  assert.ok(thumb, "premise: section.pane.thumb is declared");
  const caps = all((n) => hasAttr(n, "data-capture-caption"));
  assert.equal(
    caps.length,
    1,
    "A6: the caption line (source kind + capture size) is one element carrying " +
      "[data-capture-caption] — the hook tests/integration/test_hud_redesign.py and " +
      "test_hud_chrome.py's contrast row select on (found " + caps.length + ")",
  );
  assert.ok(childOf(caps[0], thumb), "A6: the caption lives in the What I see pane (section.pane.thumb)");
  assert.equal(
    textOf(caps[0]),
    "",
    "A6 (O-A): the caption is empty with no capture; the shell ships it empty and hud/src/main.js " +
      "fills it from the running capture. Copy in the HTML would show a size with nothing captured.",
  );
});

// T-0154 A4's fence "Mute keeps its `m` shortcut" is superseded by T-0181 B3 (H-T181-4, James
// 2026-10-28: "yes, remove it"): #mute-button becomes "Stop sharing" with no keyboard shortcut. Its
// re-scoped rows ("`m` does nothing") are the T-0181 B rows at the end of this file.

/** Every hud/src .js/.mjs file, as text: what a header control's hook must be referenced by. */
function srcTexts() {
  const out = [];
  const visit = (dir) => {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      if (statSync(full).isDirectory()) visit(full);
      else if (/\.(js|mjs)$/.test(entry)) out.push(readFileSync(full, "utf8"));
    }
  };
  visit(fileURLToPath(new URL("hud/src", REPO)));
  return out;
}

const CONTROL_TAGS = new Set(["button", "select", "input", "textarea", "a", "summary"]);
/** Header things the mockup draws that have no data here (T-0154 "Intentional differences"). */
// T-0181 B: the mic now has data behind it (astra.mic.state, the #mic-button control and its badge), so
// "mic" left this list; the mockup's "Headset" and its ⋯ overflow still have none.
const DATALESS = [/\bheadset\b/i, /⋯|…|&hellip;|&#8943;|&#x22ef;/i];

test("T-0154 A4 (negative, fence): no header element is a control with nothing behind it", () => {
  const bar = header();
  assert.ok(bar, "premise: the page has a <header>");
  const src = srcTexts();
  assert.ok(src.length >= 5, "anti-vacuity: the scan reads hud/src's modules");
  const dash = (s) => s.replace(/[A-Z]/g, (c) => "-" + c.toLowerCase());
  const referenced = (hook) =>
    src.some((text) => {
      if (text.includes('"' + hook + '"') || text.includes("'" + hook + "'") || text.includes("[" + hook + "]")) return true;
      // a data-* hook may be read through dataset.camelCase
      for (const m of text.matchAll(/dataset\.([A-Za-z0-9]+)/g)) if ("data-" + dash(m[1]) === hook) return true;
      return false;
    });
  const controls = [];
  walk(bar, (n) => {
    const roleButton = /^(button|switch|menuitem|tab|link)$/i.test(attrValue(n, "role") || "");
    const focusable = attrValue(n, "tabindex") !== null;
    const looksLikeButton = classesOf(n).some((c) => /^(btn|button)(-|$)/i.test(c));
    if (CONTROL_TAGS.has(n.tag) || roleButton || focusable || looksLikeButton) controls.push(n);
  });
  assert.ok(controls.length >= 3, "anti-vacuity: Share, Mute and the theme select are header controls");
  const dead = [];
  for (const n of controls) {
    const hooks = [];
    const id = attrValue(n, "id");
    if (id) hooks.push(id);
    for (const m of (" " + n.attrs).matchAll(/\s(data-[a-z0-9-]+)/gi)) hooks.push(m[1].toLowerCase());
    if (n.tag !== "button" && n.tag !== "select" && n.tag !== "input" && classesOf(n).some((c) => /^(btn|button)(-|$)/i.test(c))) {
      dead.push("<" + n.tag + " " + n.attrs.trim() + "> is drawn as a button but is not one");
      continue;
    }
    if (!hooks.some(referenced)) {
      dead.push("<" + n.tag + " " + n.attrs.trim() + "> (" + JSON.stringify(textOf(n)) + "): no id or data-* hook of it is read by hud/src");
    }
  }
  const text = textOf(bar);
  for (const re of DATALESS) if (re.test(text) || re.test(innerHtml(bar))) dead.push("the header shows " + re + ", which the page has no data for");
  assert.deepEqual(
    dead,
    [],
    "A4 (negative): every header control has something behind it — a hook a hud/src module reads — " +
      "and the mockup's dataless controls (\"Mic · Headset\", the ⋯ overflow, H-T154-3) are not built:\n  " +
      dead.join("\n  "),
  );
});

// --- T-0154 group B: one conversation pane (B1) -------------------------------------------------
//
// Strict-xfail gate keyed on group B's activation phrase in the task's **Status:** line (the same
// gate as tests/hud/thread.test.mjs); ASTRA_RUN_PENDING=1 runs the rows for real.
const B_ACTIVATED_T0154 = /\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0154_STATUS);
const B_PENDING_T0154 = !B_ACTIVATED_T0154 && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0154B(name, fn) {
  test(name + (B_PENDING_T0154 ? " [pending T-0154 B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!B_PENDING_T0154) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) assert.fail("XPASS(strict): this T-0154 group B row passed before group B's activation is recorded");
    t.diagnostic("expected failure (T-0154 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

pendingT0154B("T-0154 B1: one conversation pane, a direct child of <main>, holds [data-thread] and the ask form after it", () => {
  const threads = all((n) => hasAttr(n, "data-thread"));
  assert.equal(threads.length, 1, "B1: exactly one [data-thread] in the shell (found " + threads.length + ")");
  const pane = conversationPane();
  assert.ok(pane, "B1: exactly one direct child of <main> holds [data-thread]");
  assert.ok(hasClass(pane, "pane"), "B1/T-0056 criterion 4: the conversation pane is a .pane");
  const form = all((n) => n.tag === "form" && hasAttr(n, "data-ask-form"))[0];
  assert.ok(form && insideNode(form, pane), "B1: the ask form is inside the conversation pane");
  assert.ok(!insideNode(form, threads[0]), "B1: the ask form is not inside [data-thread] (it sits at its bottom)");
  assert.ok(form.start > threads[0].start, "B1: the ask form comes after the thread in document order");
  assert.equal(attrValue(form, "method"), "dialog", "T-0039 A19: method=dialog kept");
  for (const attr of ["data-ask-input", "data-ask-submit"]) {
    const c = all((n) => hasAttr(n, attr))[0];
    assert.ok(c && insideNode(c, form) && hasAttr(c, "disabled"), `T-0039 A19: [${attr}] in the form, disabled in the HTML`);
  }
  const h2 = all((n) => n.tag === "h2");
  const rail = h2.filter((h) => !insideNode(h, pane)).map((h) => textOf(h).toLowerCase());
  assert.deepEqual([...rail].sort(), [...HEADINGS_RAIL_T0154].sort(), "T-0128 criterion 3 as amended by B1: the shell's rail headings");
  const convo = h2.filter((h) => insideNode(h, pane)).map((h) => textOf(h).toLowerCase());
  assert.deepEqual(convo.filter((t) => RETIRED_HEADINGS_T0154.includes(t)), [], "B1: no retired pane heading survives in the conversation pane");
});

pendingT0154B("T-0154 B1: the retired ids, classes and the placeholder are gone (T-0056 c3 / T-0128 c3 pins superseded by name)", () => {
  const ids = all((n) => attrValue(n, "id") !== null).map((n) => attrValue(n, "id"));
  for (const id of ["spoken-pane", "transcript-pane", "shown-channel"]) assert.ok(!ids.includes(id), `B1: #${id} is retired`);
  for (const cls of ["pane-spoken", "pane-said", "pane-shown"]) assert.equal(all((n) => hasClass(n, cls)).length, 0, `B1: .${cls} is retired`);
  assert.ok(!HTML.replace(/<!--[\s\S]*?-->/g, "").includes(PLACEHOLDER_TEXT), "B1: the 'no surface yet (T-0010)' placeholder is retired");
});

pendingT0154B("T-0154 B1: the thread's empty state is HTML copy in the conversation pane, hidden by a :has() rule", () => {
  const pane = conversationPane();
  assert.ok(pane, "premise: the conversation pane");
  const empties = [];
  walk(pane, (n) => {
    if (hasAttr(n, "data-empty")) empties.push(n);
  });
  assert.equal(empties.length, 1, "B1 (T-0056 criterion 12 form): one [data-empty] line in the conversation pane");
  assert.ok(textOf(empties[0]).length >= 3, "…with words in it");
  const thread = all((n) => hasAttr(n, "data-thread"))[0];
  assert.ok(!insideNode(empties[0], thread), "the empty line is not a child of [data-thread] (the thread's children are articles)");
  const css = STYLES.replace(/\/\*[\s\S]*?\*\//g, " ");
  const rules = [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].filter(([, sel, body]) =>
    /:has\(/.test(sel) && /data-thread/.test(sel) && /\[data-empty\]/.test(sel) && /display\s*:\s*none/.test(body));
  assert.ok(rules.length >= 1, "B1: hud/styles.css hides the conversation pane's [data-empty] with a :has() rule keyed on [data-thread]");
});

// --- T-0181 groups B0 and B: Stop sharing, the mic control, no single-key shortcut ----------------
//
// docs/tasks/T-0181.md B0-1, B1, B2, B3 (H-T181-4: button only; "the `m` shortcut for Stop sharing is
// removed"). Strict-xfail gate keyed on group B's activation phrase in the task's **Status:** line
// ("group B: tests <a>..<b>, activated <sha>", the gate of tests/hud/mic.test.mjs); B0 folded into B
// (it did not merge before T-0177 C started). ASTRA_RUN_PENDING=1 runs the rows for real. Expected copy
// is written out by hand from the criteria.

const T0181_TASK = fileURLToPath(new URL("docs/tasks/T-0181.md", REPO));
const T0181_STATUS = existsSync(T0181_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0181_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0181_B_PENDING =
  !/\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0181_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0181B(name, fn) {
  test(name + (T0181_B_PENDING ? " [pending T-0181 B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!T0181_B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0181 group B row passed while docs/tasks/T-0181.md's status does not " +
          "record group B's activation (" + JSON.stringify(T0181_STATUS.slice(0, 80)) + ").",
      );
    }
    t.diagnostic("expected failure (T-0181 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0181 B gate (shell): the status line is readable", () => {
  assert.ok(T0181_STATUS.length > 0, "docs/tasks/T-0181.md must carry a **Status:** line");
});

const STOP_SHARING_TITLE_T0181 = "Stop sharing: end the screen or camera capture";

pendingT0181B("T-0181 B0-1/B3: #mute-button keeps its id and reads 'Stop sharing', titled without (m), no aria-keyshortcuts", () => {
  const stop = byId("mute-button");
  assert.ok(stop && stop.tag === "button", "B0-1: #mute-button keeps its id and stays a <button>");
  assert.equal(textOf(stop), "Stop sharing", "B0-1: its text is 'Stop sharing'");
  assert.equal(attrValue(stop, "title"), STOP_SHARING_TITLE_T0181, "B3: its title has no '(m)'");
  assert.ok(!hasAttr(stop, "aria-keyshortcuts"), "B3: #mute-button loses aria-keyshortcuts (no key stops the capture)");
  assert.ok(hasAttr(stop, "disabled"), "B0-1 (behaviour kept): it ships disabled; main.js enables it while a capture runs");
});

pendingT0181B("T-0181 B3: no element of the shell announces a keyboard shortcut", () => {
  const announced = all((n) => hasAttr(n, "aria-keyshortcuts"));
  assert.deepEqual(
    announced.map((n) => "<" + n.tag + " " + n.attrs.trim() + ">"),
    [],
    "B3: 'the page registers no other single-key shortcut'; an aria-keyshortcuts left in the shell " +
      "announces a key that does nothing",
  );
  const titles = all((n) => attrValue(n, "title") !== null).map((n) => attrValue(n, "title"));
  assert.deepEqual(titles.filter((t) => /\(m\)/i.test(t)), [], "B3: no title still says '(m)'");
});

pendingT0181B("T-0181 B3: hud/src/main.js no longer stops the capture on `m` (code, comments stripped)", () => {
  const code = read("hud/src/main.js").replace(/\/\*[\s\S]*?\*\//g, "").replace(/(^|[^:\\])\/\/.*$/gm, "$1");
  assert.ok(code.length > 1000, "anti-vacuity: main.js was read");
  assert.doesNotMatch(code, /\.key\s*===?\s*["'`][mM]["'`]/, "B3: the document keydown `m` shortcut is removed");
});

pendingT0181B("T-0181 B1: <button id=\"mic-button\"> sits in the header before Share, hidden until a state arrives", () => {
  const bar = header();
  assert.ok(bar, "premise: the page has a <header>");
  const mic = byId("mic-button");
  assert.ok(mic && mic.tag === "button", "B1: a <button id=\"mic-button\">");
  assert.ok(childOf(mic, bar), "B1: in the header");
  const share = byId("share-button");
  assert.ok(share && childOf(share, bar), "premise: Share is in the header");
  assert.ok(mic.start < share.start, "B1: the mic control comes before Share in the header");
  assert.ok(hasAttr(mic, "hidden"), "B1: it is shown only after an astra.mic.state arrives, so the shell ships it hidden");
  assert.ok(attrValue(mic, "type") === "button", "a type=button (never a form submit)");
});

pendingT0181B("T-0181 B2: one [data-mic-badge] in the header, hidden and empty in the shell", () => {
  const bar = header();
  const badges = all((n) => hasAttr(n, "data-mic-badge"));
  assert.equal(badges.length, 1, "B2: exactly one [data-mic-badge]");
  assert.ok(childOf(badges[0], bar), "B2: a header badge");
  assert.ok(hasAttr(badges[0], "hidden"), "B2: hidden unless the echoed state is muted (nothing is muted at load)");
  assert.equal(textOf(badges[0]), "", "B2: no copy in the HTML (it would claim a mute nobody echoed)");
  assert.ok(!CONTROL_TAGS.has(badges[0].tag), "B2: the badge is a label, not a control");
});
