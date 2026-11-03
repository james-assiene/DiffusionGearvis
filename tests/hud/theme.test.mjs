// T-0128 criteria 6 and 7 — the theme module, hud/src/theme.js, under plain `node --test`.
//
// The human's choice 3 (docs/tasks/T-0128.md): the theme follows prefers-color-scheme, a Settings
// control overrides it (System / Light / Dark), and the choice persists per browser in
// localStorage, wrapped in try/catch. Implementation note 3: there is ONE implementation of the
// resolution order and it is testable under node. This file is that test. The browser half (the
// control, `color-scheme`, a reload, an init script that makes localStorage throw, the module
// 404ing) is tests/integration/test_hud_redesign.py.
//
// This file used to *skip itself* while hud/src/theme.js was absent (the node stand-in for a
// `pending` marker while T-0128 was unimplemented). T-0128 is implemented, so the skip is gone
// (reviewer finding, low): a missing theme.js is now a red row, not a silent skip. The first row
// below asserts the file exists, so the failure names the cause instead of an import error.
//
// API this pins (the task leaves the JS names to the test-author; these are them):
//
//   hud/src/theme.js — an ES module; importing it under node (no `document`, no `window`) must not
//   throw. If it initialises itself in the browser, that is guarded on a document being present.
//
//     THEME_KEY = "astra.hud.theme"          the localStorage key (criterion 6, verbatim)
//     initTheme({ win, root }) -> controller
//       win   window-like: `win.localStorage` (may be undefined, or a getter that throws) and
//             `win.matchMedia(query)` (may be absent). Every access is the module's to guard.
//       root  the <html> element: setAttribute / getAttribute / removeAttribute (and `dataset`,
//             which the double below keeps in sync, so either way of writing it is fine).
//       Applies the resolved theme to `root` synchronously, before returning.
//     controller.preference() -> "system" | "light" | "dark"   the preference in force
//     controller.theme()      -> "light" | "dark"              what `data-theme` says
//     controller.set(pref)    pref in {"system","light","dark"}: applies at once, persists if it
//                             can, and never throws.
//
// What is deliberately NOT pinned: how the module logs (only how much, criterion 7), whether it
// writes `color-scheme` inline or lets the `:root[data-theme="light"]` block do it (the Chrome
// module measures the computed value), whether `system` is stored as "system" or as an absent
// key (criterion 6 (d) says "removes the override"; both readings resolve identically), and which
// of the two prefers-color-scheme queries it asks (the double answers both consistently).
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";

const REPO = new URL("../../", import.meta.url);
const THEME_JS = fileURLToPath(new URL("hud/src/theme.js", REPO));
const KEY = "astra.hud.theme";

test("T-0128 criteria 6-7: hud/src/theme.js exists (no skip once the task is implemented)", () => {
  assert.ok(
    existsSync(THEME_JS),
    "hud/src/theme.js is missing. T-0128 is implemented; its theme module is the one " +
      "implementation of the resolution order (implementation note 3), and every row in this " +
      "file exercises it. " + THEME_JS,
  );
});

async function load() {
  try {
    return await import(pathToFileURL(THEME_JS).href);
  } catch (err) {
    assert.fail(
      "T-0128 adds hud/src/theme.js, importable in node as an ES module with nothing at module " +
        "top level that needs a DOM (implementation note 3: one implementation of the " +
        "resolution order, testable under node). node said: " + (err && err.message),
    );
  }
}

async function initFrom(win, root) {
  const mod = await load();
  assert.equal(typeof mod.initTheme, "function", "hud/src/theme.js exports initTheme({ win, root })");
  return mod.initTheme({ win, root });
}

// --- doubles ----------------------------------------------------------------------------------

/** A Storage with the five members the real one has; `map` is the test's view of what persisted. */
function memoryStorage(initial = {}) {
  const map = new Map(Object.entries(initial));
  return {
    map,
    getItem: (k) => (map.has(String(k)) ? map.get(String(k)) : null),
    setItem: (k, v) => void map.set(String(k), String(v)),
    removeItem: (k) => void map.delete(String(k)),
    clear: () => map.clear(),
    key: (i) => [...map.keys()][i] ?? null,
    get length() {
      return map.size;
    },
  };
}

/**
 * matchMedia with a live system scheme. Each returned list answers both prefers-color-scheme
 * queries consistently, supports both listener APIs and `onchange`, and `matches` is live (as on
 * a real MediaQueryList), so an implementation is free in which of these it uses.
 */
function fakeMedia(systemLight) {
  let light = systemLight;
  const lists = [];
  const evaluate = (query) => {
    if (/prefers-color-scheme\s*:\s*light/.test(query)) return light;
    if (/prefers-color-scheme\s*:\s*dark/.test(query)) return !light;
    return false;
  };
  function matchMedia(query) {
    const listeners = new Set();
    const mql = {
      media: String(query),
      get matches() {
        return evaluate(String(query));
      },
      onchange: null,
      addEventListener(type, fn) {
        if (type === "change" && typeof fn === "function") listeners.add(fn);
      },
      removeEventListener(type, fn) {
        listeners.delete(fn);
      },
      addListener(fn) {
        if (typeof fn === "function") listeners.add(fn);
      },
      removeListener(fn) {
        listeners.delete(fn);
      },
      dispatchEvent() {
        return true;
      },
      _listeners: listeners,
    };
    lists.push(mql);
    return mql;
  }
  function flip(toLight) {
    if (light === toLight) return;
    light = toLight;
    for (const mql of lists) {
      const event = { type: "change", matches: mql.matches, media: mql.media };
      for (const fn of [...mql._listeners]) fn.call(mql, event);
      if (typeof mql.onchange === "function") mql.onchange.call(mql, event);
    }
  }
  return { matchMedia, flip, queried: () => lists.map((l) => l.media) };
}

/** The <html> element: attributes, plus a dataset kept in sync with them. */
function fakeRoot() {
  const attrs = new Map();
  const toAttr = (prop) => "data-" + String(prop).replace(/[A-Z]/g, (c) => "-" + c.toLowerCase());
  return {
    attrs,
    style: {},
    setAttribute: (k, v) => void attrs.set(String(k).toLowerCase(), String(v)),
    getAttribute: (k) => (attrs.has(String(k).toLowerCase()) ? attrs.get(String(k).toLowerCase()) : null),
    removeAttribute: (k) => void attrs.delete(String(k).toLowerCase()),
    hasAttribute: (k) => attrs.has(String(k).toLowerCase()),
    toggleAttribute: (k, force) => {
      const name = String(k).toLowerCase();
      const on = force === undefined ? !attrs.has(name) : Boolean(force);
      if (on) attrs.set(name, "");
      else attrs.delete(name);
      return on;
    },
    dataset: new Proxy(
      {},
      {
        get: (_, p) => (typeof p === "string" && attrs.has(toAttr(p)) ? attrs.get(toAttr(p)) : undefined),
        set: (_, p, v) => (attrs.set(toAttr(p), String(v)), true),
        deleteProperty: (_, p) => (attrs.delete(toAttr(p)), true),
        has: (_, p) => attrs.has(toAttr(p)),
      },
    ),
  };
}

const themeOf = (root) => root.getAttribute("data-theme");
const settle = () => new Promise((resolve) => setTimeout(resolve, 20));

/** Every console line written while `fn` runs. */
async function consoleLines(fn) {
  const kinds = ["log", "info", "warn", "error", "debug", "trace"];
  const saved = Object.fromEntries(kinds.map((k) => [k, console[k]]));
  const lines = [];
  for (const k of kinds) console[k] = (...args) => lines.push(k + ": " + args.map(String).join(" "));
  try {
    await fn();
  } finally {
    for (const k of kinds) console[k] = saved[k];
  }
  return lines;
}

const domError = (name) =>
  typeof DOMException === "function" ? new DOMException(name + " (test double)", name) : Object.assign(new Error(name), { name });

// --- the key ----------------------------------------------------------------------------------

test("T-0128 criterion 6: the preference lives under astra.hud.theme", async () => {
  const mod = await load();
  assert.equal(
    mod.THEME_KEY,
    KEY,
    "criterion 6 names the key verbatim: astra.hud.theme. A different key is a preference that " +
      "every other reader (the README's theme section, a human clearing it in devtools) looks for " +
      "in the wrong place.",
  );
  const storage = memoryStorage();
  const root = fakeRoot();
  const ctl = await initFrom({ localStorage: storage, matchMedia: fakeMedia(false).matchMedia }, root);
  ctl.set("light");
  assert.equal(
    storage.map.get(KEY),
    "light",
    "choosing Light must persist the string \"light\" under astra.hud.theme; the storage double " +
      "holds " + JSON.stringify(Object.fromEntries(storage.map)),
  );
});

// --- criterion 6 (a)/(b): the resolution table, by hand -----------------------------------------

/**
 * Hand-written: [stored value or ABSENT, system is light, expected data-theme, expected preference].
 * (a) a valid stored light/dark wins; (b) absent, "system" or any other value follows the system.
 * The odd rows are the ones an implementation gets wrong: "Dark" (case), "" (falsy but present),
 * "1" (truthy junk), and two JSON shapes a future serialiser might write.
 */
const ABSENT = Symbol("absent");
const TABLE = [
  [ABSENT, true, "light", "system"],
  [ABSENT, false, "dark", "system"],
  ["system", true, "light", "system"],
  ["system", false, "dark", "system"],
  ["light", true, "light", "light"],
  ["light", false, "light", "light"],
  ["dark", true, "dark", "dark"],
  ["dark", false, "dark", "dark"],
  ["Dark", true, "light", "system"],
  ["Dark", false, "dark", "system"],
  ["", true, "light", "system"],
  ["", false, "dark", "system"],
  ["1", true, "light", "system"],
  ["1", false, "dark", "system"],
  ['"dark"', true, "light", "system"],
  ['"light"', false, "dark", "system"],
  ['{"theme":"dark"}', true, "light", "system"],
  ['{"theme":"light"}', false, "dark", "system"],
];

for (const [stored, systemLight, expected, preference] of TABLE) {
  const label = stored === ABSENT ? "absent" : JSON.stringify(stored);
  test(
    `T-0128 criterion 6 (a)/(b): stored ${label}, system ${systemLight ? "light" : "dark"} -> ${expected}`,
    async () => {
      const storage = memoryStorage(stored === ABSENT ? {} : { [KEY]: stored });
      const root = fakeRoot();
      const ctl = await initFrom({ localStorage: storage, matchMedia: fakeMedia(systemLight).matchMedia }, root);
      assert.equal(
        themeOf(root),
        expected,
        `criterion 6: with astra.hud.theme = ${label} and the system ${systemLight ? "light" : "dark"}, ` +
          `<html data-theme> must be "${expected}" as soon as initTheme returns (got ${JSON.stringify(themeOf(root))}). ` +
          "Only an exact \"light\" or \"dark\" overrides the system; anything else — a case variant, " +
          "an empty string, JSON a later serialiser wrote — resolves from matchMedia.",
      );
      assert.equal(ctl.theme(), expected, "controller.theme() agrees with what data-theme says");
      assert.equal(
        ctl.preference(),
        preference,
        `criterion 6/8: the preference in force for stored ${label} is "${preference}" (the control ` +
          "reflects it at load, so an invalid stored value must read as System, not as itself)",
      );
    },
  );
}

test("T-0128 criterion 6: data-theme is always light or dark, never absent or 'system'", async () => {
  for (const systemLight of [true, false]) {
    for (const stored of [ABSENT, "system", "garbage"]) {
      const root = fakeRoot();
      const storage = memoryStorage(stored === ABSENT ? {} : { [KEY]: stored });
      await initFrom({ localStorage: storage, matchMedia: fakeMedia(systemLight).matchMedia }, root);
      assert.ok(
        ["light", "dark"].includes(themeOf(root)),
        "criterion 6: the page always ends with <html data-theme> set to light or dark; the token " +
          "block is keyed on it. Got " + JSON.stringify(themeOf(root)) + " for stored " +
          String(stored === ABSENT ? "absent" : stored),
      );
    }
  }
});

// --- criterion 6 (c): a live system change ------------------------------------------------------

test("T-0128 criterion 6 (c): while the preference is system, a system flip re-themes live", async () => {
  const media = fakeMedia(false);
  const root = fakeRoot();
  await initFrom({ localStorage: memoryStorage(), matchMedia: media.matchMedia }, root);
  assert.equal(themeOf(root), "dark", "premise: system dark, nothing stored -> dark");
  media.flip(true);
  await settle();
  assert.equal(
    themeOf(root),
    "light",
    "criterion 6 (c): the system scheme changed to light while the preference is System, and the " +
      "page must re-theme without a reload. The module has to subscribe to the MediaQueryList " +
      "(addEventListener('change') or addListener); reading matchMedia once at load is the bug.",
  );
  media.flip(false);
  await settle();
  assert.equal(themeOf(root), "dark", "…and back: a second flip is followed too (the listener stays)");
});

// --- adversarial: the stored override must survive the system --------------------------------------

test("T-0128 criterion 6 (c), adversarial: a stored dark survives a system flip to light", async () => {
  for (const startLight of [false, true]) {
    const media = fakeMedia(startLight);
    const root = fakeRoot();
    const ctl = await initFrom({ localStorage: memoryStorage({ [KEY]: "dark" }), matchMedia: media.matchMedia }, root);
    assert.equal(themeOf(root), "dark", `premise: stored dark wins at load over a ${startLight ? "light" : "dark"} system`);
    media.flip(!startLight);
    await settle();
    media.flip(true);
    await settle();
    assert.equal(
      themeOf(root),
      "dark",
      "criterion 6 (c): while the preference is Dark, a system change does nothing. A listener " +
        "that re-resolves from matchMedia unconditionally turns the human's explicit choice into " +
        "a suggestion the desktop overrides at sunset.",
    );
    assert.equal(ctl.preference(), "dark", "the preference itself is untouched by the system");
  }
});

test("T-0128 criterion 6 (c): a choice made through the control also ignores later system flips", async () => {
  const media = fakeMedia(true);
  const root = fakeRoot();
  const ctl = await initFrom({ localStorage: memoryStorage(), matchMedia: media.matchMedia }, root);
  assert.equal(themeOf(root), "light", "premise: system light");
  ctl.set("dark");
  assert.equal(themeOf(root), "dark", "criterion 6/8: choosing Dark applies at once");
  media.flip(false);
  await settle();
  media.flip(true);
  await settle();
  assert.equal(themeOf(root), "dark", "criterion 6 (c): an override set at runtime is not undone by the system");
});

// --- adversarial: choosing system removes the override and re-resolves -------------------------------

test("T-0128 criterion 6 (d), adversarial: choosing system after dark re-resolves at once and follows again", async () => {
  const media = fakeMedia(true);
  const storage = memoryStorage();
  const root = fakeRoot();
  const ctl = await initFrom({ localStorage: storage, matchMedia: media.matchMedia }, root);
  ctl.set("dark");
  assert.equal(themeOf(root), "dark", "premise: Dark chosen under a light system");
  ctl.set("system");
  assert.equal(
    themeOf(root),
    "light",
    "criterion 6 (d): choosing System must re-resolve *at once* from the system (light here). " +
      "Storing 'system' and waiting for the next media change leaves the page dark under a light " +
      "desktop until the sun moves.",
  );
  assert.equal(ctl.preference(), "system", "the preference in force is System");
  const persisted = storage.map.has(KEY) ? storage.map.get(KEY) : null;
  assert.ok(
    persisted === null || persisted === "system",
    "criterion 6 (d): System removes the override; astra.hud.theme must be absent or \"system\" " +
      "afterwards, not still " + JSON.stringify(persisted),
  );
  media.flip(false);
  await settle();
  assert.equal(themeOf(root), "dark", "…and a later system flip is followed again (criterion 6 (c))");
  // A fresh page on the same storage: the override is gone across a reload too.
  const reloaded = fakeRoot();
  await initFrom({ localStorage: storage, matchMedia: fakeMedia(true).matchMedia }, reloaded);
  assert.equal(themeOf(reloaded), "light", "after System, a reload under a light system resolves light");
});

test("T-0128 criterion 6: a stored choice survives a reload (a second init on the same storage)", async () => {
  const storage = memoryStorage();
  const first = fakeRoot();
  const ctl = await initFrom({ localStorage: storage, matchMedia: fakeMedia(false).matchMedia }, first);
  ctl.set("light");
  const second = fakeRoot();
  const again = await initFrom({ localStorage: storage, matchMedia: fakeMedia(false).matchMedia }, second);
  assert.equal(themeOf(second), "light", "human choice 3: the choice persists per browser");
  assert.equal(again.preference(), "light", "and the control reflects it at load (criterion 8)");
});

// --- criterion 7: persistence degrades without breaking the page -------------------------------------

/** The four failures criterion 7 names, each as a window-like double. */
const FAILURES = {
  "localStorage is undefined": (matchMedia) => ({ localStorage: undefined, matchMedia }),
  "reading localStorage throws (the getter)": (matchMedia) => {
    const win = { matchMedia };
    Object.defineProperty(win, "localStorage", {
      configurable: true,
      enumerable: true,
      get() {
        throw domError("SecurityError");
      },
    });
    return win;
  },
  "getItem throws SecurityError": (matchMedia) => {
    const storage = memoryStorage();
    storage.getItem = () => {
      throw domError("SecurityError");
    };
    return { localStorage: storage, matchMedia };
  },
  "setItem throws QuotaExceededError": (matchMedia) => {
    const storage = memoryStorage();
    storage.setItem = () => {
      throw domError("QuotaExceededError");
    };
    storage.removeItem = () => {
      throw domError("QuotaExceededError");
    };
    return { localStorage: storage, matchMedia };
  },
};

for (const [kind, makeWin] of Object.entries(FAILURES)) {
  for (const systemLight of [true, false]) {
    const system = systemLight ? "light" : "dark";
    test(`T-0128 criterion 7: ${kind} (system ${system}) — renders, resolves, switches, logs at most once`, async () => {
      const media = fakeMedia(systemLight);
      const win = makeWin(media.matchMedia);
      const root = fakeRoot();
      let ctl;
      const lines = await consoleLines(async () => {
        try {
          ctl = await initFrom(win, root);
        } catch (err) {
          if (err && err.code === "ERR_ASSERTION") throw err;
          assert.fail(
            `criterion 7 (${kind}): initTheme threw ${err && err.name}: ${err && err.message}. ` +
              "Every localStorage access is wrapped in try/catch (human choice 3); an uncaught " +
              "throw here is a page whose theme script dies on a privacy-mode browser.",
          );
        }
        assert.equal(
          themeOf(root),
          system,
          `criterion 7 (${kind}): with storage unusable the theme resolves from the system ` +
            `(${system}); got ${JSON.stringify(themeOf(root))}`,
        );
        for (const pref of ["light", "dark", "light", "dark", "system"]) {
          try {
            ctl.set(pref);
          } catch (err) {
            assert.fail(`criterion 7 (${kind}): set(${JSON.stringify(pref)}) threw ${err && err.name}: ${err && err.message}`);
          }
          const expected = pref === "system" ? system : pref;
          assert.equal(
            themeOf(root),
            expected,
            `criterion 7 (${kind}): the control still switches the theme for the life of the page — ` +
              `after set(${JSON.stringify(pref)}) data-theme is ${JSON.stringify(themeOf(root))}, expected ${expected}`,
          );
          assert.equal(ctl.preference(), pref, `criterion 7 (${kind}): the in-memory preference is ${pref}`);
        }
        ctl.set("dark");
        media.flip(!systemLight);
        await settle();
        assert.equal(
          themeOf(root),
          "dark",
          `criterion 6 (c) with ${kind}: an in-memory Dark choice still ignores the system`,
        );
      });
      assert.ok(
        lines.length <= 1,
        `criterion 7 (${kind}): at most one console line per kind of failure; the module wrote ` +
          `${lines.length} across one init and six choices:\n  ` + lines.join("\n  ") +
          "\nA warning on every set() is the console flood a privacy-mode user sees on each click.",
      );
    });
  }
}

test("T-0128 criterion 7 (adversarial): matchMedia missing or throwing does not break the module", async () => {
  const variants = {
    "matchMedia is absent": { localStorage: memoryStorage() },
    "matchMedia throws": {
      localStorage: memoryStorage(),
      matchMedia() {
        throw new TypeError("matchMedia unavailable (test double)");
      },
    },
  };
  for (const [kind, win] of Object.entries(variants)) {
    const root = fakeRoot();
    let ctl;
    try {
      ctl = await initFrom(win, root);
    } catch (err) {
      if (err && err.code === "ERR_ASSERTION") throw err;
      assert.fail(`implementation note 3 (${kind}): every matchMedia access is wrapped; initTheme threw ${err && err.message}`);
    }
    assert.ok(
      ["light", "dark"].includes(themeOf(root)),
      `criterion 6 (${kind}): data-theme still ends as light or dark; got ${JSON.stringify(themeOf(root))}`,
    );
    ctl.set("light");
    assert.equal(themeOf(root), "light", `${kind}: an explicit choice still applies`);
    ctl.set("dark");
    assert.equal(themeOf(root), "dark", `${kind}: an explicit choice still applies`);
  }
});
