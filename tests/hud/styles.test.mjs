// T-0056 criteria 2, 5, 9 and 13 — hud/styles.css read as text, by the properties a renderer
// cannot answer: what the sheet *contains*, in what order, and which names it declares.
//
// The companion module is tests/integration/test_hud_chrome.py, which answers the questions only
// a renderer can (what a colour computes to, what a grid does with seven panes). The division is
// deliberate: three of the four failures this task is most likely to produce are invisible to a
// renderer on this box —
//
//   * a token rename that every consumer survives because it passed a fallback (`var(--dim, #8fa0aa)`
//     renders identically whether or not --dim exists; four panels silently stop following the
//     page palette and no pixel changes on the machine the implementer is looking at);
//   * a layout declaration inside the light-scheme block, which only appears on a machine whose
//     system theme is light — not this one;
//   * a @font-face or an @import, which works perfectly on a box that has a network route and
//     leaves the page half-styled on the one that does not.
//
// Status when written (before the restyle exists): the URL, font-stack and consumer rows pass
// against today's sheet; the token-closure, colour-scheme, reduced-motion and light-theme rows
// fail, because those four things do not exist yet.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, posix } from "node:path";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const REPO = new URL("../../", import.meta.url);
const abs = (rel) => fileURLToPath(new URL(rel, REPO));
const read = (rel) => readFileSync(abs(rel), "utf8");

const SHEET = "hud/styles.css";

// T-0164 group B (docs/tasks/T-0164.md B5): hud/src/ui/style.css (the --a2ui-* bindings for the
// vendored renderer's shadow roots) is deleted and hud/src/ui/card.css becomes the one light-DOM
// card sheet. The rows that read ui/style.css keep reading it until T-0164's **Status:** records
// "group B: tests <a>..<b>, activated <sha>" (so they stay green on main), and read card.css in its
// place from then on; the --a2ui-* exemption of the var() scan is retired at the same moment.
// `ASTRA_RUN_PENDING=1` flips the switch now. New B5 rows run through `pendingT0164B`.
const T0164_TASK_B = fileURLToPath(new URL("docs/tasks/T-0164.md", REPO));
const T0164_STATUS_B = existsSync(T0164_TASK_B)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0164_TASK_B, "utf8")) || [])[1] || "").trim()
  : "";
const T0164_B = /\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0164_STATUS_B) || process.env.ASTRA_RUN_PENDING === "1";
/** The ui sheet the theme/literal/origin rows read: style.css until T-0164 B, card.css after. */
const UI_SHEET = T0164_B ? "hud/src/ui/card.css" : "hud/src/ui/style.css";

function pendingT0164B(name, fn) {
  test(name + (T0164_B ? "" : " [pending T-0164 B: strict xfail until group B is activated]"), async (t) => {
    if (T0164_B) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail("XPASS(strict): this T-0164 group B row passed while docs/tasks/T-0164.md does not record group B's activation.");
    }
    t.diagnostic("expected failure (T-0164 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}
const CSS = read(SHEET);
/** Comments carry URLs and property names legitimately; every scan below runs on this. */
const CODE = CSS.replace(/\/\*[\s\S]*?\*\//g, (m) => " ".repeat(m.length));

// --- brace-aware helpers ------------------------------------------------------------------

/** The `{ … }` block starting at the first `{` at or after `from`, as [start, end) of the body. */
function blockAt(text, from) {
  const open = text.indexOf("{", from);
  if (open < 0) return null;
  let depth = 0;
  for (let i = open; i < text.length; i += 1) {
    if (text[i] === "{") depth += 1;
    else if (text[i] === "}") {
      depth -= 1;
      if (depth === 0) return { open, bodyStart: open + 1, bodyEnd: i, end: i + 1 };
    }
  }
  return null;
}

/** Every top-level `@media` at-rule: {prelude, body, start, end}. */
function mediaBlocks(text) {
  const out = [];
  const re = /@media([^{]*)/g;
  let m;
  while ((m = re.exec(text)) !== null) {
    const block = blockAt(text, m.index);
    if (!block) continue;
    out.push({
      prelude: m[1].trim(),
      body: text.slice(block.bodyStart, block.bodyEnd),
      start: m.index,
      bodyStart: block.bodyStart,
      end: block.end,
    });
    re.lastIndex = block.end;
  }
  return out;
}

/** Split a rule body into [property, value] pairs (no nested blocks expected). */
function declarations(body) {
  return body
    .split(";")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => {
      const i = s.indexOf(":");
      return i < 0 ? [s, ""] : [s.slice(0, i).trim(), s.slice(i + 1).trim()];
    });
}

/** Split a stylesheet fragment into {selector, body} rules (flat; at-rules skipped by caller). */
function rules(text) {
  const out = [];
  let cursor = 0;
  while (cursor < text.length) {
    const block = blockAt(text, cursor);
    if (!block) break;
    out.push({
      selector: text.slice(cursor, block.open).trim(),
      body: text.slice(block.bodyStart, block.bodyEnd),
      start: cursor,
      end: block.end,
    });
    cursor = block.end;
  }
  return out;
}

/** The sheet with every @media block removed — "the base cascade". */
function withoutMedia(text) {
  let out = text;
  for (const block of mediaBlocks(text).reverse()) {
    out = out.slice(0, block.start) + " ".repeat(block.end - block.start) + out.slice(block.end);
  }
  return out;
}

/** Every declaration in every `:root` rule outside a media query. */
function rootDeclarations() {
  const base = withoutMedia(CODE);
  const out = [];
  for (const rule of rules(base)) {
    const selectors = rule.selector.split(",").map((s) => s.trim());
    if (!selectors.includes(":root") && !selectors.includes("html")) continue;
    for (const pair of declarations(rule.body)) out.push(pair);
  }
  return out;
}

const ROOT = rootDeclarations();
const ROOT_NAMES = new Set(ROOT.filter(([p]) => p.startsWith("--")).map(([p]) => p));

// --- criterion 5: the token names other files consume keep resolving -------------------------

/** The nine names criterion 5 pins by hand, so the scan below cannot pass vacuously. */
const REQUIRED_TOKENS = [
  ["--ink", "the page's body text colour; hud/styles.css is the only declarer"],
  ["--dim", "secondary text in four panels: assists, log, the counters, the placeholder"],
  ["--edge", "every border on the page, and hud/src/log/index.js's and assists/index.js's rows"],
  ["--accent", "hud/src/assists/index.js's badge and 'assist on' label, log/index.js's on state"],
  ["--warn", "assists/index.js .assist-pending, log/index.js .log-pending and .log-error"],
  ["--bad", "the error strip (hud/index.html #error-strip / .error-line)"],
  ["--fill", "panels/thinking.js writes it inline on #thinking-bar; the bar reads it"],
  ["--fg", "hud/src/ui/style.css binds the A2UI surface's text colour to it (group B)"],
  ["--mono", "hud/src/log/index.js sets the session-log path in it; group B's numbers too"],
];

test("T-0056 criterion 5: the nine token names are declared at :root", () => {
  assert.ok(
    ROOT.length >= 5,
    "hud/styles.css declares a :root block (the token layer). Found " + ROOT.length +
      " declarations in :root — if this is 0 the parser above is broken and every row in this " +
      "file is vacuous.",
  );
  for (const [name, why] of REQUIRED_TOKENS) {
    assert.ok(
      ROOT_NAMES.has(name),
      "hud/styles.css must declare " + name + " at :root. " + why + ". Criterion 5: --fg and " +
        "--mono are declared for the *first* time by this task — today hud/src/ui/style.css and " +
        "hud/src/log/index.js reference them and fall through to hard-coded fallbacks, which is " +
        "why the shown channel and the session-log path have never used the page's palette. The " +
        "other seven are kept: a rename would require editing a panel this task may not touch.",
    );
    const value = ROOT.filter(([p]) => p === name).map(([, v]) => v).pop();
    if (ROOT_NAMES.has(name)) {
      assert.ok(
        value && value.trim().length > 0,
        name + " is declared at :root with an empty value. An empty custom property resolves to " +
          "the empty token, not to the consumer's fallback — every var(" + name + ", …) in a " +
          "panel then produces an invalid declaration rather than the fallback colour.",
      );
    }
  }
});

/** Every file under hud/ that could reference a token, minus the vendored bundle. */
function hudFiles() {
  const out = [];
  const walk = (dir, rel) => {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      const relPath = rel ? rel + "/" + entry : entry;
      if (statSync(full).isDirectory()) {
        if (relPath === "vendor") continue; // T-0024: the pinned bundle declares its own --a2ui-*
        walk(full, relPath);
      } else if (/\.(css|js|mjs|html)$/.test(entry)) {
        out.push(["hud/" + relPath, readFileSync(full, "utf8")]);
      }
    }
  };
  walk(abs("hud"), "");
  return out;
}

test("T-0056 criterion 5, the row that catches the rename: every var(--x) resolves at :root", () => {
  const files = hudFiles();
  assert.ok(files.length >= 10, "hud/ holds the shell, the sheet and hud/src/** — found " + files.length);
  const referenced = new Map(); // name -> [files]
  for (const [rel, text] of files) {
    const code = text.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/<!--[\s\S]*?-->/g, " ");
    for (const m of code.matchAll(/var\(\s*(--[a-zA-Z0-9_-]+)/g)) {
      const name = m[1];
      // The vendored renderer's own theming surface: declared by hud/src/ui/style.css on the
      // <a2ui-surface> host (group B, criterion 16), never at :root. Not a page token.
      // T-0164 B5 retires this exemption: with the bundle unreferenced, an --a2ui-* name is a
      // dangling token like any other.
      if (!T0164_B && name.startsWith("--a2ui-")) continue;
      if (!referenced.has(name)) referenced.set(name, []);
      if (!referenced.get(name).includes(rel)) referenced.get(name).push(rel);
    }
  }
  assert.ok(
    referenced.size >= 6,
    "the scan found " + referenced.size + " distinct var(--x) references under hud/ (excluding " +
      "hud/vendor/**). Today there are at least nine; a number this low means the scan broke, " +
      "not that the page stopped using tokens.",
  );
  const dangling = [...referenced.entries()]
    .filter(([name]) => !ROOT_NAMES.has(name))
    .map(([name, where]) => name + " (referenced by " + where.join(", ") + ")");
  assert.deepEqual(
    dangling,
    [],
    "these custom properties are referenced under hud/ but never declared at :root in " + SHEET +
      ". This is asserted *including* references that carry a fallback, because a resolving " +
      "fallback is exactly how this failure hides: var(--dim, #8fa0aa) renders the same whether " +
      "or not --dim exists, so a rename degrades four panels to hard-coded literals while every " +
      "screenshot and every other test stays green. Today's var(--fg, #d8e0e5) and " +
      "var(--mono, monospace) are the proof — both have been dangling for two tasks and nobody " +
      "noticed. Criterion 5: add tokens, never substitute one of the nine, and never follow a " +
      "rename into a file T-0056 does not own.",
  );
});

test("T-0056 criterion 5, the demand side: the panels still ask for the names they ask for today", () => {
  // The supply side is the row above. This is the other half: if the implementer 'fixed' a
  // dangling name by editing the consumer, that consumer is outside Owned files and the edit is
  // the escalation trigger in Blast radius, not a fix.
  const CONSUMERS = [
    ["hud/src/assists/index.js", ["--accent", "--dim", "--warn"]],
    ["hud/src/log/index.js", ["--mono", "--dim", "--warn", "--accent"]],
    ["hud/src/ui/style.css", ["--fg"]],
  ].filter(([rel]) => !(T0164_B && rel === "hud/src/ui/style.css")); // T-0164 B5: deleted
  for (const [rel, names] of CONSUMERS) {
    const text = read(rel);
    for (const name of names) {
      assert.ok(
        text.includes("var(" + name),
        rel + " no longer references var(" + name + "…). Criterion 5 keeps the *names* stable " +
          "precisely so that no file outside T-0056's Owned files needs a diff; if this row is " +
          "red the token layer was renamed and a panel was edited to follow it, which is the " +
          "escalation trigger in the task's Blast radius section. (hud/src/ui/style.css is group " +
          "B's own file and may bind --fg to more places — it may not stop reading it.)",
      );
    }
  }
});

// --- criterion 2: no new origin, no new dependency -------------------------------------------

const ORIGIN_FILES = ["hud/index.html", SHEET, UI_SHEET]; // T-0164 B5: card.css replaces style.css

/**
 * Every egress problem in one origin file's text: @import, @font-face, a url() that does not
 * resolve inside hud/, an absolute http(s) URL outside a comment. One function so that the real
 * row and T-0128's adversarial row below exercise the *same* check (T-0128 Tests to write T2:
 * "an injected @font-face or url(https://…) fails the egress row"). Refactored out of T-0056's
 * row by T-0128 with every message and every condition kept; the one addition is that a relative
 * url() climbing out of hud/ with `../` is now caught (criterion 2: "no url() resolving outside
 * hud/").
 */
function egressProblems(rel, text) {
  const problems = [];
  const code = text.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/<!--[\s\S]*?-->/g, " ");
  if (/@import/.test(code)) {
    problems.push(
      rel + " contains @import. Everything on this box binds to loopback and the page must render " +
        "with no network route at all (CLAUDE.md, docs/02 §9); an @import of anything not already " +
        "under hud/ is a request that cannot succeed, and the page then renders unstyled rather " +
        "than styled-but-slower.",
    );
  }
  if (/@font-face/.test(code)) {
    problems.push(
      rel + " contains @font-face. Criterion 2: font families are system stacks only. Vendoring a " +
        "face is a separate task with a provenance row in a manifest (the rule that governs " +
        "hud/vendor/**), not a line in a restyle.",
    );
  }
  for (const m of code.matchAll(/url\(\s*(['"]?)([^'")]*)\1\s*\)/g)) {
    const arg = m[2].trim();
    const relative = !/^([a-zA-Z][a-zA-Z0-9+.-]*:|\/\/|\/)/.test(arg) || arg.startsWith("data:");
    const resolved = posix.normalize(posix.join(posix.dirname(rel), arg.split(/[?#]/)[0]));
    const inside = arg.startsWith("data:") || resolved.startsWith("hud/");
    if (!relative || !inside) {
      problems.push(
        rel + " has url(" + arg + "), which does not resolve inside hud/. Criterion 2 allows only " +
          "a relative path (or a data: URI): anything else is an origin this box is not supposed " +
          "to be able to reach, and tests/unit/test_hud_no_audio_no_egress.py is the blunt fence " +
          "that must keep passing unedited.",
      );
    }
  }
  for (const m of code.matchAll(/https?:\/\/[^\s'"()]+/g)) {
    problems.push(
      rel + " contains the absolute URL " + m[0] + " outside a comment. Criterion 2: no new " +
        "origin. If it is documentation, it belongs in a /* comment */ — which this scan strips.",
    );
  }
  return problems;
}

test("T-0056 criterion 2: the sheet and the shell fetch nothing", () => {
  for (const rel of ORIGIN_FILES) {
    const problems = egressProblems(rel, read(rel));
    assert.deepEqual(problems, [], problems.join("\n"));
  }
});

test("T-0128 criterion 2, adversarial: an injected @font-face or url(https://…) fails the egress row", () => {
  // The check above, fed the four injections a mockup port would bring with it, each appended to
  // the real sheet. A check that cannot see them is a row that passes against the mockup's
  // `@import url(https://fonts.googleapis.com/…)` and its "Inter" face.
  const base = read(SHEET);
  assert.deepEqual(egressProblems(SHEET, base), [], "premise: the real sheet is clean");
  const injections = {
    "@font-face with a remote src":
      '@font-face { font-family: "Inter"; src: url(https://fonts.gstatic.com/s/inter/v13/x.woff2); }',
    "@font-face with a local src": '@font-face { font-family: "Inter"; src: url(fonts/inter.woff2); }',
    "a remote background image": ".pane { background-image: url(https://example.com/noise.png); }",
    "a protocol-relative url": '.pane { background-image: url("//cdn.example.com/noise.png"); }',
    "an @import of a web font": '@import url("https://fonts.googleapis.com/css2?family=Inter");',
    "a relative url that climbs out of hud/": ".pane { background-image: url(../../docs/mock.png); }",
  };
  for (const [what, css] of Object.entries(injections)) {
    assert.ok(
      egressProblems(SHEET, base + "\n" + css + "\n").length > 0,
      "the egress check did not flag " + what + " (" + css + "). If this row is red the check, not " +
        "the sheet, is broken — and every green run of the row above means nothing.",
    );
  }
  // …and the forgiving side, so the check is not simply "any url() fails" (a guard needs a row for
  // being too wide as well as for being absent, .claude/rules/tests.md).
  for (const css of [".a { background: url(img/x.png); }", '.a { background: url("data:image/png;base64,AAAA"); }']) {
    assert.deepEqual(egressProblems(SHEET, base + "\n" + css), [], "a relative or data: url() is allowed: " + css);
  }
});

test("T-0128 criterion 2: no font, icon or image file is added under hud/", () => {
  // A fence (true on the base commit): only hud/vendor/** may carry binary assets, and it is not
  // this task's. A vendored face is a separate task with a provenance row.
  const found = [];
  const visit = (dir, relPath) => {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      const r = relPath ? relPath + "/" + entry : entry;
      if (statSync(full).isDirectory()) {
        if (r === "vendor" || r === "node_modules") continue;
        visit(full, r);
      } else if (/\.(woff2?|ttf|otf|eot|svg|png|jpe?g|gif|webp|avif|ico|bmp)$/i.test(entry)) {
        found.push("hud/" + r);
      }
    }
  };
  visit(abs("hud"), "");
  assert.deepEqual(found, [], "criterion 2: no font, icon or image file is added outside hud/vendor/");
});

test("T-0056 criterion 2: every <link> and <script> in the shell is a relative path", () => {
  const html = read("hud/index.html").replace(/<!--[\s\S]*?-->/g, " ");
  const refs = [
    ...[...html.matchAll(/<link\b[^>]*href\s*=\s*"([^"]*)"/g)].map((m) => ["link", m[1]]),
    ...[...html.matchAll(/<script\b[^>]*src\s*=\s*"([^"]*)"/g)].map((m) => ["script", m[1]]),
  ];
  assert.ok(
    refs.length >= 2,
    "the shell links hud/styles.css and loads src/main.js; the scan found " + refs.length +
      " references, which means it is not reading the shell",
  );
  for (const [kind, href] of refs) {
    assert.ok(
      !/^([a-zA-Z][a-zA-Z0-9+.-]*:|\/\/)/.test(href),
      "<" + kind + "> points at " + href + ". Criterion 2: no <link>/<script> whose href/src is " +
        "not a relative path. An absolute one is a second origin even when it happens to be " +
        "127.0.0.1 — the page is served from an ephemeral port that changes every run.",
    );
  }
});

const FONT_ALLOWLIST = new Set(
  [
    // generics and the two UA system stacks: present on every box, zero bytes fetched
    "system-ui", "ui-sans-serif", "ui-serif", "ui-monospace", "ui-rounded",
    "sans-serif", "serif", "monospace", "cursive", "fantasy",
    "-apple-system", "BlinkMacSystemFont", "inherit", "initial", "unset", "revert",
    // named faces that ship with a mainstream desktop (the stack in T-0056 Implementation notes)
    "Segoe UI", "Segoe UI Emoji", "Segoe UI Symbol", "Roboto", "Noto Sans", "Noto Sans Mono",
    "DejaVu Sans", "DejaVu Sans Mono", "Liberation Sans", "Liberation Mono", "Cantarell",
    "Helvetica Neue", "Helvetica", "Arial", "Menlo", "Consolas", "Courier New",
    "Apple Color Emoji", "Ubuntu", "Ubuntu Mono",
  ].map((s) => s.toLowerCase()),
);

test("T-0056 criterion 2: every font family is a system stack, not a face this box has to find", () => {
  const seen = [];
  for (const rel of ORIGIN_FILES) {
    const code = read(rel)
      .replace(/\/\*[\s\S]*?\*\//g, " ")
      .replace(/<!--[\s\S]*?-->/g, " ");
    for (const m of code.matchAll(/\bfont(?:-family)?\s*:\s*([^;}"']*(?:"[^"]*"[^;}]*)*)/g)) {
      const value = m[1].trim();
      if (!value || value.includes("var(")) continue; // var(--mono, monospace) is checked elsewhere
      const parts = value.split(",").map((s) => s.trim()).filter(Boolean);
      // A `font:` shorthand puts size/line-height/weight before the first family.
      parts[0] = parts[0].split(/\s+/).pop();
      for (const part of parts) {
        const name = part.replace(/^["']|["']$/g, "").trim();
        if (!name || /^\d/.test(name)) continue;
        seen.push([rel, name]);
        assert.ok(
          FONT_ALLOWLIST.has(name.toLowerCase()),
          rel + ' asks for the font family "' + name + '", which is not in this test\'s ' +
            "allow-list of generic and system families. Criterion 2 forbids fetching a face, and " +
            "this row is what makes that a property instead of a promise: a CDN-less " +
            'font-family: "Inter" passes the URL grep and silently falls back to the next stack ' +
            "entry, so the page the human sees is not the page the implementer designed. Adding " +
            "a name here is a claim that the face is installed on this box with nothing " +
            "downloaded; vendoring one is a separate task with a provenance row.",
        );
      }
    }
  }
  assert.ok(
    seen.length >= 2,
    "the scan found " + seen.length + " font family names under hud/; today's sheet declares a " +
      "mono stack and two `font: inherit`s, so a lower number means this row is vacuous",
  );
});

// --- criterion 9: reduced motion ---------------------------------------------------------------

function reduceBlock() {
  return mediaBlocks(CODE).filter((b) => /prefers-reduced-motion\s*:\s*reduce/.test(b.prelude)).pop();
}

test("T-0056 criterion 9: a universal reduced-motion block neutralises animation and transition", () => {
  const block = reduceBlock();
  assert.ok(
    block,
    SHEET + " contains no @media (prefers-reduced-motion: reduce) block. Criterion 9: the human " +
      "sits in front of this page while talking, and motion in the corner of the eye during a " +
      "conversation is the cost. The block is also what makes the 'at most one animation' budget " +
      "(ruling A1 (6)) safe to spend: it can be switched off wholesale.",
  );
  const universal = rules(block.body).find((r) => {
    const selectors = r.selector.split(",").map((s) => s.trim());
    return selectors.includes("*") && selectors.includes("*::before") && selectors.includes("*::after");
  });
  assert.ok(
    universal,
    "the reduced-motion block has no `*, *::before, *::after` rule (selectors found: " +
      rules(block.body).map((r) => JSON.stringify(r.selector)).join(", ") + "). A block that " +
      "lists the animations it knows about neutralises exactly the ones that existed the day it " +
      "was written; the next animation anyone adds is not covered, and nothing goes red.",
  );
  const props = declarations(universal.body);
  for (const family of ["animation", "transition"]) {
    const hits = props.filter(([p]) => p === family || p.startsWith(family + "-"));
    assert.ok(
      hits.length > 0,
      "the universal reduced-motion rule sets no " + family + " property. Criterion 9 requires " +
        "both families: an animation-only block leaves every transition running, and a " +
        "transition-only block leaves every keyframe animation running.",
    );
    for (const [prop, value] of hits) {
      assert.ok(
        /!important/.test(value),
        "the universal reduced-motion rule declares " + prop + ": " + value + " without " +
          "!important. A `*` selector has specificity 0, so every rule above it in the sheet — " +
          "and every rule hud/src/assists/index.js and hud/src/log/index.js inject into " +
          "document.head *after* this sheet loads — wins the tie and the block does nothing.",
      );
    }
  }
});

test("T-0056 criterion 9, adversarial: nothing declares motion after the neutraliser", () => {
  const block = reduceBlock();
  assert.ok(block, "criterion 9's @media (prefers-reduced-motion: reduce) block must exist");
  const after = [];
  for (const m of CODE.matchAll(/(?:^|[;{\s])(animation|transition)(-[a-z-]+)?\s*:/g)) {
    if (m.index >= block.start && m.index < block.end) continue;
    if (m.index > block.end) after.push(m[1] + (m[2] || "") + " at offset " + m.index);
  }
  assert.deepEqual(
    after,
    [],
    "hud/styles.css declares motion *after* the reduced-motion block. Two reasons this is the " +
      "row and not a style preference. (a) !important beats source order only against normal " +
      "declarations: one `transition: … !important` later in the file wins outright, and that is " +
      "a one-word edit away from any rule written below the block. (b) The block is the sheet's " +
      "last word by construction, so 'is motion neutralised?' is answerable by reading the end of " +
      "the file rather than by simulating the cascade — which is what makes the fence hold for " +
      "the next person as well as for this task. The behavioural twin of this row is in " +
      "tests/integration/test_hud_chrome.py, which emulates prefers-reduced-motion: reduce and " +
      "walks every element's computed durations.",
  );
});

// --- T-0128 criterion 9: one token layer; the theme is only tokens ------------------------------
//
// T-0128 supersedes T-0056 criterion 13's two textual pins (named in T-0128 criterion 9:
// "Superseded: T-0056 criterion 13's `color-scheme: light dark` and its single-@media-block
// pins"). The theme is now resolved by hud/src/theme.js and written as <html data-theme>, so:
//
// Removed by T-0128 (criterion 9):
//   1. "T-0056 criterion 13: :root declares color-scheme: light dark" — its one assertion,
//      `:root color-scheme` equals "light dark". Under T-0128 :root is the dark theme and
//      color-scheme follows data-theme (criterion 6); the computed value is asserted in
//      tests/integration/test_hud_redesign.py.
//   2. "T-0056 criterion 13, adversarial: the light block re-binds tokens and changes nothing
//      else" — its four assertions: exactly one @media (prefers-color-scheme: light) block; every
//      rule in it has the selector :root; every declaration in it is a `--*`; it re-binds every
//      colour token of REQUIRED_TOKENS. All four survive below, re-keyed to the
//      :root[data-theme="light"] block (with `color-scheme` added to what it may declare), and
//      the @media count becomes zero instead of one.

// T-0128's rows used to skip while hud/src/theme.js was absent (the node stand-in for `pending`).
// T-0128 is implemented, so they always run (reviewer finding, low).

const LIGHT_SELECTOR = /^:root\[data-theme=(["']?)light\1\]$/;

/** Every rule at any depth (inside @media, @supports, @layer, @keyframes too), with its at-rule path. */
function allRules(text, path = []) {
  const out = [];
  let cursor = 0;
  while (cursor < text.length) {
    const block = blockAt(text, cursor);
    if (!block) break;
    const selector = text.slice(cursor, block.open).trim();
    const body = text.slice(block.bodyStart, block.bodyEnd);
    if (selector.startsWith("@")) {
      out.push(...allRules(body, [...path, selector]));
    } else {
      out.push({ selector, body, path });
    }
    cursor = block.end;
  }
  return out;
}

/** Strip @import-like statements without blocks, so allRules sees only real rules. */
const statementsStripped = (text) => text.replace(/@(import|charset|namespace)[^;{}]*;/g, " ");

const SHEET_RULES = allRules(statementsStripped(CODE));
const lightBlocks = () => SHEET_RULES.filter((r) => r.path.length === 0 && LIGHT_SELECTOR.test(r.selector));
const isRootBlock = (r) => r.path.length === 0 && r.selector === ":root";

test("T-0128 criterion 9 (supersedes T-0056 criterion 13): exactly one :root[data-theme=\"light\"] block, declarations only", () => {
  const blocks = lightBlocks();
  assert.equal(
    blocks.length,
    1,
    "hud/styles.css must contain exactly one top-level :root[data-theme=\"light\"] rule (found " +
      blocks.length + "). Criterion 9: the light theme is one block that re-binds the dark tokens " +
      ":root declares — a palette, not a second stylesheet.",
  );
  for (const [prop, value] of declarations(blocks[0].body)) {
    assert.ok(
      prop.startsWith("--") || prop === "color-scheme",
      "the light block declares `" + prop + ": " + value + "`. Criterion 9 allows only `--*` " +
        "declarations plus `color-scheme`. A layout property here is a change that exists only " +
        "in one theme — invisible in the other theme's screenshots and in every review of them.",
    );
  }
  assert.equal(
    /\{/.test(blocks[0].body),
    false,
    "the light block contains a nested rule; it holds declarations only (criterion 9)",
  );
  const lightNames = new Set(declarations(blocks[0].body).map(([p]) => p));
  // The colour half of the token layer. --fill is written inline by panels/thinking.js and --mono
  // is a font stack: neither is a palette entry (kept from T-0056's row).
  for (const [name] of REQUIRED_TOKENS) {
    if (name === "--fill" || name === "--mono") continue;
    assert.ok(
      lightNames.has(name),
      "the light block does not re-bind " + name + ". A palette that re-binds some of its colour " +
        "tokens and inherits the rest gives a mixed page whose contrast fails in one theme only.",
    );
  }
});

test("T-0128 criterion 9, adversarial: no prefers-color-scheme block and no other theme-keyed rule", () => {
  for (const [rel, text] of [
    [SHEET, CODE],
    [UI_SHEET, read(UI_SHEET).replace(/\/\*[\s\S]*?\*\//g, " ")], // T-0164 B5: card.css after B
  ]) {
    const media = mediaBlocks(text).filter((b) => /prefers-color-scheme/.test(b.prelude));
    assert.deepEqual(
      media.map((b) => "@media " + b.prelude),
      [],
      rel + " still carries a prefers-color-scheme block. The system scheme is now read by " +
        "hud/src/theme.js and written as data-theme (criterion 6). A left-over @media (light) " +
        "block re-binds the tokens a second time and breaks criterion 6 (a): under a light " +
        "desktop it turns a stored Dark choice light again.",
    );
    const keyed = allRules(statementsStripped(text)).filter(
      (r) => /data-theme/.test(r.selector) && !(rel === SHEET && r.path.length === 0 && LIGHT_SELECTOR.test(r.selector)),
    );
    assert.deepEqual(
      keyed.map((r) => (r.path.length ? r.path.join(" ") + " " : "") + r.selector),
      [],
      rel + ": rules keyed on data-theme other than the one light block. Criterion 9: the theme is " +
        "only tokens — a `[data-theme=light] .pane { … }` rule is a second stylesheet for one theme.",
    );
  }
});

// --- colour literals -----------------------------------------------------------------------------

/** CSS Color 4's named colours (the 148, lower-case). `transparent` and `currentcolor` are keywords. */
const NAMED_COLOURS = new Set(
  (
    "aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue " +
    "blueviolet brown burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk " +
    "crimson cyan darkblue darkcyan darkgoldenrod darkgray darkgreen darkgrey darkkhaki " +
    "darkmagenta darkolivegreen darkorange darkorchid darkred darksalmon darkseagreen " +
    "darkslateblue darkslategray darkslategrey darkturquoise darkviolet deeppink deepskyblue " +
    "dimgray dimgrey dodgerblue firebrick floralwhite forestgreen fuchsia gainsboro ghostwhite " +
    "gold goldenrod gray green greenyellow grey honeydew hotpink indianred indigo ivory khaki " +
    "lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan " +
    "lightgoldenrodyellow lightgray lightgreen lightgrey lightpink lightsalmon lightseagreen " +
    "lightskyblue lightslategray lightslategrey lightsteelblue lightyellow lime limegreen linen " +
    "magenta maroon mediumaquamarine mediumblue mediumorchid mediumpurple mediumseagreen " +
    "mediumslateblue mediumspringgreen mediumturquoise mediumvioletred midnightblue mintcream " +
    "mistyrose moccasin navajowhite navy oldlace olive olivedrab orange orangered orchid " +
    "palegoldenrod palegreen paleturquoise palevioletred papayawhip peachpuff peru pink plum " +
    "powderblue purple rebeccapurple red rosybrown royalblue saddlebrown salmon sandybrown " +
    "seagreen seashell sienna silver skyblue slateblue slategray slategrey snow springgreen " +
    "steelblue tan teal thistle tomato turquoise violet wheat white whitesmoke yellow yellowgreen"
  ).split(/\s+/),
);
/** Properties whose values are names, not colours ("Linen" could be a font; "tan" a grid area). */
const NOT_COLOUR_PROPS = /^(font|font-family|content|grid-area|grid-template-areas|animation-name|animation|counter-reset|counter-increment|list-style-type|quotes|will-change|transition-property|font-feature-settings)$/;

/** The colour literals in one declaration value: hex, colour functions, named colours. */
function colourLiterals(prop, value) {
  const v = value
    .replace(/"[^"]*"|'[^']*'/g, " ")
    .replace(/url\([^)]*\)/g, " ")
    .replace(/--[A-Za-z0-9_-]+/g, " "); // var(--accent-red) is a token name, not a literal
  const out = [];
  for (const m of v.matchAll(/#[0-9a-fA-F]{3,8}\b/g)) out.push(m[0]);
  for (const m of v.matchAll(/\b(rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)\(/gi)) out.push(m[0] + "…)");
  if (!NOT_COLOUR_PROPS.test(prop)) {
    for (const m of v.matchAll(/[A-Za-z]+/g)) if (NAMED_COLOURS.has(m[0].toLowerCase())) out.push(m[0]);
  }
  return out;
}

function strayLiterals(rel, rulesList, allowed) {
  const out = [];
  for (const rule of rulesList) {
    if (allowed(rule)) continue;
    for (const [prop, value] of declarations(rule.body.replace(/\{[\s\S]*\}/g, " "))) {
      for (const lit of colourLiterals(prop, value)) {
        out.push(`${rel}: ${(rule.path.length ? rule.path.join(" ") + " " : "") + rule.selector} { ${prop}: ${value} } — ${lit}`);
      }
    }
  }
  return out;
}

test("T-0128 criterion 9: no colour literal outside :root and the light block, none in ui/style.css", () => {
  // The anti-vacuity guard: the detector must see the literals :root itself is made of.
  const rootLiterals = SHEET_RULES.filter(isRootBlock).flatMap((r) =>
    declarations(r.body).flatMap(([p, v]) => colourLiterals(p, v)),
  );
  assert.ok(
    rootLiterals.length >= 10,
    "the colour-literal detector found " + rootLiterals.length + " literals in :root, which declares " +
      "the whole dark palette — the detector, not the sheet, is broken",
  );
  const stray = [
    ...strayLiterals(SHEET, SHEET_RULES, (r) => isRootBlock(r) || (r.path.length === 0 && LIGHT_SELECTOR.test(r.selector))),
    ...strayLiterals(
      UI_SHEET, // T-0164 B5: hud/src/ui/card.css once group B is activated (style.css is deleted)
      allRules(statementsStripped(read(UI_SHEET).replace(/\/\*[\s\S]*?\*\//g, " "))),
      () => false,
    ),
  ];
  assert.deepEqual(
    stray,
    [],
    "criterion 9: no colour literal outside the two token blocks of hud/styles.css, and none at " +
      "all in hud/src/ui/style.css (which binds --a2ui-* to page tokens only). A literal is a colour " +
      "that does not change with the theme — and that includes a var(--x, #fallback): the fallback " +
      "is a literal, and it is what renders the day the token is renamed.\n  " + stray.join("\n  "),
  );
});

test("T-0128 criterion 9: every var(--x) resolves in both themes (no light-only token)", () => {
  const blocks = lightBlocks();
  assert.equal(blocks.length, 1, "premise: the light block exists");
  const lightOnly = declarations(blocks[0].body)
    .map(([p]) => p)
    .filter((p) => p.startsWith("--") && !ROOT_NAMES.has(p));
  assert.deepEqual(
    lightOnly,
    [],
    "the light block declares tokens :root does not: " + lightOnly.join(", ") + ". A var() of one " +
      "of them resolves in the light theme and is invalid in the dark one. Criterion 9: :root " +
      "declares the tokens, the light block only re-binds them. (The other half — every var(--x) " +
      "under hud/ is declared at :root — is T-0056 criterion 5's row above, which still runs.)",
  );
  for (const name of ["--ink", "--dim", "--edge", "--accent", "--warn", "--bad", "--fill", "--fg", "--mono"]) {
    assert.ok(ROOT_NAMES.has(name), "criterion 9: " + name + " stays declared at :root");
  }
});

// --- T-0128 criterion 11: the spoken channel's honesty styling ---------------------------------

// A fence: true on the base commit, and the place a port of the mockup's dimmed not-yet-voiced
// words would land. It runs now, ungated.
//
// **Widened by T-0154 B3** (G2: the selector set grows, nothing is removed). The spoken text now
// lives in the thread: `[data-thread]`, one `article` per turn, one `[data-utterance-id]` per
// utterance, and an interrupted turn's article carries `[data-interrupted]`. The old tokens
// (`.utterance`, `#spoken-pane`, `.pane-spoken`, `.turn`) stay. A selector naming any of them is
// "spoken": keyed on the turn state, or animated/transitioned, it is an offender as before. New
// (B3: "no rule dims or hides an utterance by turn state or time; an interrupted turn carries
// data-interrupted" — a mark, not a dimmer): a rule keyed on `[data-interrupted]` whose subject is
// an utterance, an article, the thread or the interrupted element itself, and which sets a
// property that dims, hides or recolours it, is an offender. The interrupted *mark* in the meta
// line (a different subject) stays allowed.
//
// **Widened again by O-B3 (3), F2** (nothing removed): the utterances sit inside the assistant
// row, `.assistant[data-assistant] > .assistant-body > [data-utterance-id]`, so a dimmer aimed at
// the row or its body dims the spoken text exactly as one aimed at the article does. Reviewer 2's
// surviving mutants were `[data-interrupted] .assistant-body { opacity }`,
// `.turn[data-interrupted] > .assistant { opacity }` and
// `body:has(.chip[data-turn-state="INTERRUPTED"]) .assistant-body { opacity }`. Both lists now name
// `.assistant`, `.assistant-body` and `[data-assistant]`; `.assistant-x` and `[data-assistant-x]`
// (other names) stay out.
const ASSISTANT_TOKENS = String.raw`\.assistant(?:-body)?(?![\w-])|\[data-assistant(?![\w-])`;
const SPOKEN_SELECTOR = new RegExp(
  String.raw`\.utterance\b|#spoken-pane\b|\.pane-spoken\b|\.turn\b(?!-)|\[data-thread\b|\[data-utterance-id\b|(?<![\w.#\-[="'])article\b(?!-)|\[data-interrupted\b|` +
    ASSISTANT_TOKENS,
);
/** The subject compound of each selector in a list: the part after the last combinator. */
const subjects = (selectorList) =>
  selectorList.split(",").map((part) => part.trim().split(/\s*[\s>+~]\s*/).filter(Boolean).pop() || "");
const DIMMED_SUBJECT = new RegExp(
  String.raw`\.utterance\b|\[data-utterance-id\b|(?<![\w.#\-[="'])article\b(?!-)|\[data-thread\b|\[data-interrupted\b|\.turn\b(?!-)|^\*$|` +
    ASSISTANT_TOKENS,
);
const DIMMING_PROPERTY =
  /^(opacity|filter|visibility|display|color|-webkit-text-fill-color|text-decoration(-line|-color|-style)?|content-visibility|clip-path|mask(-.*)?)$/;

/** Criterion 11 (T-0128) / B3 (T-0154) offenders in a list of {selector, body} rules. */
function spokenHonestyOffenders(rulesList) {
  const offenders = [];
  for (const rule of rulesList) {
    const sel = rule.selector;
    if (!SPOKEN_SELECTOR.test(sel)) continue;
    const decls = declarations(rule.body.replace(/\{[\s\S]*\}/g, " "));
    if (/data-turn-state|turn-state/.test(sel)) offenders.push(sel + " — keyed on the turn state");
    for (const [prop] of decls) {
      if (/^(animation|transition)/.test(prop)) offenders.push(sel + " { " + prop + " } — changes with time");
    }
    for (const part of sel.split(",")) {
      if (!/\[data-interrupted\b/.test(part)) continue;
      const subject = subjects(part)[0];
      if (!DIMMED_SUBJECT.test(subject)) continue;
      for (const [prop] of decls) {
        if (DIMMING_PROPERTY.test(prop)) {
          offenders.push(part.trim() + " { " + prop + " } — dims an interrupted turn's spoken text");
        }
      }
    }
  }
  return offenders;
}

test("T-0154 B3: the criterion 11 fence catches the thread's selectors and an interrupted-turn dimmer", () => {
  // Planted offenders: each would make the spoken text's look depend on something other than
  // [data-unplayed]. Expected list written by hand from B3 and T-0128 criterion 11.
  const planted = allRules(`
    :root[data-turn-state="speaking"] [data-thread] { opacity: 1; }
    [data-utterance-id] { transition: opacity 200ms; }
    article { animation: fade 1s; }
    [data-turn-state] article[data-turn-id] { color: red; }
    [data-interrupted] [data-utterance-id] { opacity: 0.5; }
    article[data-interrupted] { filter: grayscale(1); }
    [data-thread] [data-interrupted] { color: gray; }
    [data-interrupted] .utterance, main [data-interrupted] > article { visibility: hidden; }
  `);
  assert.deepEqual(spokenHonestyOffenders(planted), [
    ':root[data-turn-state="speaking"] [data-thread] — keyed on the turn state',
    "[data-utterance-id] { transition } — changes with time",
    "article { animation } — changes with time",
    "[data-turn-state] article[data-turn-id] — keyed on the turn state",
    "[data-interrupted] [data-utterance-id] { opacity } — dims an interrupted turn's spoken text",
    "article[data-interrupted] { filter } — dims an interrupted turn's spoken text",
    "[data-thread] [data-interrupted] { color } — dims an interrupted turn's spoken text",
    "[data-interrupted] .utterance { visibility } — dims an interrupted turn's spoken text",
    "main [data-interrupted] > article { visibility } — dims an interrupted turn's spoken text",
  ]);
  // Allowed (the fence must not be too wide): the honest strike from astra.spoken.played, the
  // interrupted mark in the meta line, plain thread layout, and names that merely contain "article".
  const allowed = allRules(`
    [data-unplayed] { text-decoration-line: line-through; opacity: 0.55; }
    [data-interrupted] [data-turn-meta]::before { content: "interrupted"; color: var(--warn); }
    [data-thread] { padding: 26px 34px; overflow-y: auto; }
    article[data-turn-id] { display: grid; gap: 14px; }
    [data-utterance-id] { color: var(--text); font-size: 16.5px; }
    .article-list, [data-article] { opacity: 0.5; transition: opacity 1s; }
  `);
  assert.deepEqual(spokenHonestyOffenders(allowed), []);
});

test("O-B3 F2: the criterion 11 fence names the assistant row and body the utterances sit in", () => {
  // Reviewer 2's three CSS mutants, verbatim, each of which the previous fence let through (the
  // fourth, an inline style written by markInterrupted, is the Chrome guard's: no sheet holds it).
  // Expected list written by hand from B3 and O-B3 (3).
  const planted = allRules(`
    [data-interrupted] .assistant-body { opacity: 0.55; }
    .turn[data-interrupted] > .assistant { opacity: 0.55; }
    body:has(.chip[data-turn-state="INTERRUPTED"]) .assistant-body { opacity: 0.55; }
    article[data-interrupted] [data-assistant] { filter: saturate(0.2); }
    .assistant-body { transition: opacity 300ms; }
  `);
  assert.deepEqual(spokenHonestyOffenders(planted), [
    "[data-interrupted] .assistant-body { opacity } — dims an interrupted turn's spoken text",
    ".turn[data-interrupted] > .assistant { opacity } — dims an interrupted turn's spoken text",
    'body:has(.chip[data-turn-state="INTERRUPTED"]) .assistant-body — keyed on the turn state',
    "article[data-interrupted] [data-assistant] { filter } — dims an interrupted turn's spoken text",
    ".assistant-body { transition } — changes with time",
  ]);
  // Not too wide: other names that merely start with "assistant", the decorative avatar's own
  // rule under [data-interrupted] (a different subject), and plain layout of the row.
  const allowed = allRules(`
    .assistant-tools, [data-assistant-hint] { opacity: 0.5; transition: opacity 1s; }
    [data-interrupted] .avatar { opacity: 0.5; }
    .assistant { display: grid; grid-template-columns: 30px 1fr; gap: 14px; }
    .assistant-body { min-width: 0; }
  `);
  assert.deepEqual(spokenHonestyOffenders(allowed), []);
});

test("T-0128 criterion 11 (H-T128-2): no rule dims, hides or recolours .utterance by turn state or time", () => {
  const offenders = spokenHonestyOffenders(SHEET_RULES);
  assert.deepEqual(
    offenders,
    [],
    "criterion 11 / H-T128-2: the mockup's dimmed not-yet-voiced words have no wire source, so " +
      "they are dropped. The spoken text's look may depend on [data-unplayed] (what the device " +
      "really did not play) and on nothing else — not on the turn state and not on a clock.\n  " +
      offenders.join("\n  "),
  );
  const strikes = SHEET_RULES.filter((r) => /line-through/.test(r.body)).map((r) => r.selector);
  assert.deepEqual(
    strikes,
    ["[data-unplayed]"],
    "criterion 11: [data-unplayed] stays the only line-through rule in hud/styles.css",
  );
});

// --- T-0128 criterion 5: no new animation anywhere under hud/ -----------------------------------
//
// The Chrome module measures the Thinking pane's computed animation-name (element, ::before and
// ::after). This is its textual twin, so a shimmer is caught on a box with no Chrome too: the set
// of @keyframes under hud/ (minus the pinned vendor bundle) is exactly the one at the task's HEAD,
// chip-pulse (T-0056 ruling A1 (6): "the one animation on the page"). Counted by hand with
// `grep -rn keyframes hud/ --exclude-dir=vendor` at 8fed448: one hit, hud/styles.css.
const KEYFRAMES_AT_HEAD = ["chip-pulse"];

test("T-0128 criterion 5, adversarial: no @keyframes under hud/ other than chip-pulse", () => {
  const found = [];
  for (const [rel, text] of hudFiles()) {
    const code = /\.css$/.test(rel) ? text.replace(/\/\*[\s\S]*?\*\//g, " ") : text;
    for (const m of code.matchAll(/@(?:-webkit-|-moz-)?keyframes\s+([A-Za-z0-9_-]+|"[^"]*"|'[^']*')/g)) {
      found.push(m[1].replace(/^["']|["']$/g, ""));
    }
  }
  assert.deepEqual(
    [...found].sort(),
    [...KEYFRAMES_AT_HEAD].sort(),
    "T-0128 criterion 5: the Thinking pane shows real per-step canvases or an empty state, never a " +
      "placeholder animation; a new @keyframes is how a shimmer arrives. Found " +
      JSON.stringify(found) + " under hud/ (vendor excluded); expected exactly " +
      JSON.stringify(KEYFRAMES_AT_HEAD) + ". A legitimate new animation needs a task entry that " +
      "updates this list.",
  );
});

// --- T-0154 A7: new tokens are tokens, declared only in the two token blocks ---------------------
//
// docs/tasks/T-0154.md A7, keeping T-0128 criterion 9: "one token layer, no colour literal outside
// the two token blocks; new tokens such as a logo glow are tokens". The colour-literal row above
// already refuses `.logo { box-shadow: 0 0 24px rgba(…) }`; this row closes the other way round a
// "token" can be introduced: a custom property declared on a component rule
// (`.logo { --glow: … }`, `.btn.on { --x: … }`) is a second token layer that the light block
// cannot re-bind. A fence: it holds on the base commit (every `--*` declaration in hud/styles.css
// is in :root or :root[data-theme="light"]) and must hold after the restyle. `--fill` is written
// inline by hud/src/panels/thinking.js, never in the sheet, so it is not an exception here.

test("T-0154 A7 (fence): every custom property in hud/styles.css is declared in one of the two token blocks", () => {
  assert.ok(ROOT_NAMES.size >= 10, "anti-vacuity: the detector sees the :root tokens (" + ROOT_NAMES.size + ")");
  const isTokenBlock = (r) => isRootBlock(r) || (r.path.length === 0 && LIGHT_SELECTOR.test(r.selector));
  const stray = [];
  for (const rule of SHEET_RULES) {
    if (isTokenBlock(rule)) continue;
    for (const [prop, value] of declarations(rule.body.replace(/\{[\s\S]*\}/g, " "))) {
      if (prop.startsWith("--")) {
        stray.push((rule.path.length ? rule.path.join(" ") + " " : "") + rule.selector + " { " + prop + ": " + value + " }");
      }
    }
  }
  assert.deepEqual(
    stray,
    [],
    "T-0154 A7 / T-0128 criterion 9: a custom property declared outside :root and the light block " +
      "is a token the theme cannot re-bind (a logo glow, a sharing fill, a switch knob colour). " +
      "Declare it in both token blocks and use var() at the component:\n  " + stray.join("\n  "),
  );
  // the detector, proven on a hand-made sheet: one stray token, one legitimate pair
  const probe = allRules(":root { --glow: #fff; } :root[data-theme=\"light\"] { --glow: #000; } .logo { --glow2: var(--glow); }");
  const found = probe.filter((r) => !isTokenBlock(r)).flatMap((r) => declarations(r.body).filter(([p]) => p.startsWith("--")));
  assert.deepEqual(found, [["--glow2", "var(--glow)"]], "the detector finds exactly the component-level token");
});

// --- T-0154 C1: the page's card sheet, hud/src/ui/card.css ----------------------------------------
//
// docs/tasks/T-0154.md C1 (H-T154-1, ruled "Yes"): one same-origin, token-only stylesheet, added to
// each surface's shadow root; "tokens are the only colours"; "the model's text is never altered (no
// `text-transform` on model text, no `content:` strings)". Tests to write: "`card.css` covered by the
// egress and colour-literal rows". The same two checks as the rows above (`egressProblems`,
// `colourLiterals`/`strayLiterals`) are applied to card.css — not copies of them — plus the
// T-0128 criterion 9 theme rule (no prefers-color-scheme block, no data-theme-keyed rule) and the
// C1 text rule. Every rule in card.css paints the model's surface, so `content` and a non-`none`
// `text-transform` are refused anywhere in the sheet, not only on a text selector.
//
// Gate (O-P): these rows need group C's file, so they run through `pendingT0154C` — strict xfail
// until docs/tasks/T-0154.md's **Status:** records "group C: tests <a>..<b>, activated <sha>";
// `ASTRA_RUN_PENDING=1` runs them for real. Group A's and group B's activation phrases do not
// switch them on (hand-written samples in the gate row).

const CARD_SHEET = "hud/src/ui/card.css";
const T0154_TASK_C = abs("docs/tasks/T-0154.md");
const T0154_STATUS_C = existsSync(T0154_TASK_C)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0154_TASK_C, "utf8")) || [])[1] || "").trim()
  : "";
const C_ACTIVATED = /\bgroup C:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i;
const C_PENDING = !C_ACTIVATED.test(T0154_STATUS_C) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0154C(name, fn) {
  test(name + (C_PENDING ? " [pending T-0154 C: strict xfail until group C is activated]" : ""), async (t) => {
    if (!C_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail(
        "XPASS(strict): this T-0154 group C row passed while docs/tasks/T-0154.md's status does not " +
          "record group C's activation (" + JSON.stringify(T0154_STATUS_C.slice(0, 80)) + "). Either " +
          "group C has landed (record 'group C: tests …, activated <sha>') or the row is vacuous.",
      );
    }
    t.diagnostic("expected failure (T-0154 C pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

test("T-0154 C gate: the status line is readable and only group C's own phrase activates the rows", () => {
  assert.ok(T0154_STATUS_C.length > 0, "docs/tasks/T-0154.md must carry a **Status:** line");
  // hand-written samples: A's and B's activation wording must not switch group C's rows on
  assert.equal(C_ACTIVATED.test("in progress (group A: tests 433d454..7a97f95, activated 7030bd5)"), false);
  assert.equal(C_ACTIVATED.test("in progress (group B: tests b650070..291c79f, activated 291c79f, landed 8285570; group C: tests on T-0154-c-tests e45e4ed; group A: …)"), false, "B's activation, C's tests only");
  assert.equal(C_ACTIVATED.test("in progress (group C: tests e45e4ed..1a2b3c4, activated 1a2b3c4; group A: …, activated 7030bd5)"), true);
  assert.equal(C_ACTIVATED.test("in progress (group C: tests e45e4ed..1a2b3c4, not yet activated)"), false, "activation names its commit");
});

/** card.css, comments blanked; throws a readable assertion if group C's file is absent. */
function cardCode() {
  assert.ok(existsSync(abs(CARD_SHEET)), "T-0154 C1: " + CARD_SHEET + " does not exist (group C's file)");
  return read(CARD_SHEET).replace(/\/\*[\s\S]*?\*\//g, (m) => " ".repeat(m.length));
}

/** C1's text rule on one sheet's rules: every `content` declaration, every text-transform but none. */
function textAlterations(rel, rulesList) {
  const out = [];
  for (const rule of rulesList) {
    for (const [prop, value] of declarations(rule.body.replace(/\{[\s\S]*\}/g, " "))) {
      const where = rel + ": " + (rule.path.length ? rule.path.join(" ") + " " : "") + rule.selector;
      if (prop.toLowerCase() === "content") out.push(where + " { content: " + value + " }");
      if (prop.toLowerCase() === "text-transform" && !/^none\s*(!important)?$/i.test(value.trim())) {
        out.push(where + " { text-transform: " + value + " }");
      }
    }
  }
  return out;
}

pendingT0154C("T-0154 C1 (T-0056 criterion 2 extended): the card sheet fetches nothing", () => {
  const problems = egressProblems(CARD_SHEET, cardCode());
  assert.deepEqual(problems, [], problems.join("\n"));
});

pendingT0154C("T-0154 C1 (T-0128 criterion 9 extended): no colour literal in the card sheet; tokens only", () => {
  const code = cardCode();
  const cardRules = allRules(statementsStripped(code));
  assert.ok(cardRules.length >= 3, "anti-vacuity: card.css styles the card, its heading and its rows; found " + cardRules.length + " rules");
  const refs = [...code.matchAll(/var\(\s*(--[a-zA-Z0-9_-]+)/g)].map((m) => m[1]);
  assert.ok(refs.length >= 1, "C1: card.css paints with the page's tokens, so it reads at least one var(--token)");
  const stray = strayLiterals(CARD_SHEET, cardRules, () => false);
  assert.deepEqual(
    stray,
    [],
    "C1: tokens are the only colours in " + CARD_SHEET + " — no hex, no colour function, no named " +
      "colour, and no var(--x, #fallback) (the fallback is the literal that renders the day the " +
      "token is renamed):\n  " + stray.join("\n  "),
  );
  const media = mediaBlocks(code).filter((b) => /prefers-color-scheme/.test(b.prelude));
  assert.deepEqual(media.map((b) => "@media " + b.prelude), [], "T-0128 criterion 9: the theme is tokens, not a prefers-color-scheme block in card.css");
  const keyed = cardRules.filter((r) => /data-theme/.test(r.selector) || r.path.some((p) => /data-theme/.test(p)));
  assert.deepEqual(keyed.map((r) => r.selector), [], "T-0128 criterion 9: no data-theme-keyed rule in card.css (one sheet for both themes)");
});

pendingT0154C("T-0154 C1: the card sheet writes no content: string and transforms no text", () => {
  const offenders = textAlterations(CARD_SHEET, allRules(statementsStripped(cardCode())));
  assert.deepEqual(
    offenders,
    [],
    "C1: the model's text is never altered. A `content:` string paints words the model did not " +
      "write (an eyebrow, a tick, a bullet), and `text-transform` changes the words it did:\n  " +
      offenders.join("\n  "),
  );
});

test("T-0154 C1, adversarial: the card-sheet checks see an injected literal, url, content and transform", () => {
  // Runs ungated: it proves the detectors on hand-made card sheets, not on group C's file.
  const clean = ".a2ui-card { border: 1px solid var(--line); color: var(--text-2); text-transform: none; }";
  const rulesOf = (css) => allRules(statementsStripped(css));
  assert.deepEqual(strayLiterals(CARD_SHEET, rulesOf(clean), () => false), [], "premise: a token-only card rule is clean");
  assert.deepEqual(egressProblems(CARD_SHEET, clean), [], "premise: no url, no import");
  assert.deepEqual(textAlterations(CARD_SHEET, rulesOf(clean)), [], "premise: text-transform: none is allowed");
  const literal = {
    "a hex colour": ".a2ui-card { background: #14171d; }",
    "a fallback literal": ".a2ui-card h3 { color: var(--text, #eceef3); }",
    "a colour function in a shadow": ".a2ui-card { box-shadow: 0 12px 32px rgba(0, 0, 0, 0.3); }",
    "a named colour": ".a2ui-divider { border-top-color: white; }",
  };
  for (const [what, css] of Object.entries(literal)) {
    assert.ok(strayLiterals(CARD_SHEET, rulesOf(css), () => false).length > 0, "the literal check missed " + what + ": " + css);
  }
  for (const css of [
    '@import url("https://fonts.googleapis.com/css2?family=Inter");',
    ".a2ui-card { background-image: url(https://example.com/noise.png); }",
    ".a2ui-card { background-image: url(../../../docs/mock.png); }",
  ]) {
    assert.ok(egressProblems(CARD_SHEET, css).length > 0, "the egress check missed: " + css);
  }
  // card.css sits in hud/src/ui/, so a relative url that stays inside hud/ is allowed
  assert.deepEqual(egressProblems(CARD_SHEET, ".a2ui-card { background: url(../../img/x.png); }"), []);
  const alterations = {
    "an eyebrow": '.a2ui-card h3::before { content: "SHOPPING"; }',
    "a bullet": '.a2ui-list-item-none::before { content: "\\2022"; }',
    "an uppercase heading": ".a2ui-card h3 { text-transform: uppercase; }",
    "a capitalised caption": ".a2ui-text.caption em { text-transform: capitalize; }",
  };
  for (const [what, css] of Object.entries(alterations)) {
    assert.ok(textAlterations(CARD_SHEET, rulesOf(css)).length > 0, "the text check missed " + what + ": " + css);
  }
});

// --- T-0164 B5: one light-DOM card sheet; ui/style.css deleted ---------------------------------
//
// The T-0154 C1 rows above (no fetch, tokens only with no fallback, no content:, no text-transform)
// keep applying to the rewritten card.css unchanged. These add what B5 changes: no shadow root
// exists, so a shadow-scoped selector in the sheet is dead CSS, and the vendor-binding sheet goes.

/** Code tokens of a hud/src JS file: comments and string/template contents blanked (rules/tests.md:
 *  a source grep matches prose; the import specifier is a string, so it is kept as its own list). */
function importSpecifiers(text) {
  const code = text.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/(^|[^:])\/\/.*$/gm, "$1");
  return [...code.matchAll(/(?:\bimport\b|\bnew\s+URL)\s*\(?[^'"`;]*?(['"`])([^'"`]+)\1/g)].map((m) => m[2]);
}

pendingT0164B("T-0164 B5: hud/src/ui/style.css is deleted and no module under hud/src loads it", () => {
  assert.equal(existsSync(abs("hud/src/ui/style.css")), false, "B5: hud/src/ui/style.css (the --a2ui-* bindings) is deleted");
  const loaders = [];
  const walk = (dir, rel) => {
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      const r = rel + "/" + entry;
      if (statSync(full).isDirectory()) walk(full, r);
      else if (/\.(js|mjs|html)$/.test(entry) && importSpecifiers(readFileSync(full, "utf8")).some((s) => /(^|\/)style\.css$/.test(s))) loaders.push(r);
    }
  };
  walk(abs("hud/src"), "hud/src");
  assert.deepEqual(loaders, [], "B5: no import or new URL(…) under hud/src names style.css (the link mount() added per page is retired)");
});

pendingT0164B("T-0164 B5: card.css is a light-DOM sheet — no :host, ::part, ::slotted or a2ui-surface selector", () => {
  const rules = allRules(statementsStripped(cardCode()));
  assert.ok(rules.length >= 3, "anti-vacuity: card.css styles the card, its text and its rows; found " + rules.length + " rules");
  const shadowScoped = rules
    .map((r) => [...r.path, r.selector].join(" "))
    .filter((sel) => /:host\b|::part\(|::slotted\(|\ba2ui-surface\b/i.test(sel));
  assert.deepEqual(
    shadowScoped,
    [],
    "B5 (amends T-0154 C1 by name): \"adopted into each surface's shadow root\" becomes \"one page sheet; no shadow " +
      "root exists\", so a selector scoped to a shadow host or to the vendor's element matches nothing in the page",
  );
});

test("T-0164 B5 gate sample: the shadow-selector check sees each shadow form (the row above is not vacuous)", () => {
  for (const css of [":host .a2ui-card { color: var(--text); }", "a2ui-surface::part(card) { color: var(--text); }", "x::slotted(p) { color: var(--text); }"]) {
    const sels = allRules(statementsStripped(css)).map((r) => [...r.path, r.selector].join(" "));
    assert.ok(sels.some((s) => /:host\b|::part\(|::slotted\(|\ba2ui-surface\b/i.test(s)), "missed: " + css);
  }
  assert.deepEqual(
    allRules(statementsStripped("[data-a2ui-root] .kv { color: var(--text); }")).map((r) => [...r.path, r.selector].join(" ")).filter((s) => /:host\b|::part\(|::slotted\(|\ba2ui-surface\b/i.test(s)),
    [],
    "a light-DOM selector is not flagged",
  );
});
