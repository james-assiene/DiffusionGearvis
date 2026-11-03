// The theme setting (T-0128 criteria 6-8): System / Light / Dark, the one implementation of the
// resolution order, testable under node (tests/hud/theme.test.mjs).
//
// Resolution order: a valid stored "light" or "dark" wins; anything else (absent, "system", junk)
// resolves from the system scheme and follows it live. The result is written as
// <html data-theme="light|dark">; hud/styles.css keys exactly one token block on it, and that block
// also sets `color-scheme`, so native controls follow the theme too.
//
// Every localStorage and matchMedia access is wrapped: a privacy-mode browser, a storage quota or a
// missing API costs the persistence, never the page. At most one console line is written per page.
//
// Loaded by its own <script type="module"> in hud/index.html. Module scripts are deferred, so a
// stored Light choice shows one dark frame first (implementation note 3 allows it; :root is the
// dark theme). If this file fails to load, the page renders dark and everything else still works.

export const THEME_KEY = "astra.hud.theme";
export const PREFERENCES = Object.freeze(["system", "light", "dark"]);

const LIGHT_QUERY = "(prefers-color-scheme: light)";
const KEY_EVENTS = ["keydown", "keypress", "keyup"];

/**
 * Resolve and apply the theme on `root`, and return the controller the Settings control drives.
 *
 * @param {{win?: any, root: any}} options  `win` is window-like (localStorage, matchMedia, both
 *   optional); `root` is the <html> element.
 * @returns {{preference: () => string, theme: () => string, set: (pref: string) => void}}
 */
export function initTheme({ win, root } = {}) {
  // The first storage failure of any kind retires storage for this page: a store that cannot be
  // read is not written to either (a later load could not read the choice back, and a half-working
  // store would make the next load resolve from a value this page never showed), and the one
  // console line stays one.
  let usable = true;
  const warn = (what, err) => {
    if (!usable) return;
    usable = false;
    try {
      globalThis.console.warn(
        `[hud/theme] ${what} (${(err && err.name) || err}); the theme choice lasts for this page only`,
      );
    } catch {
      // nothing more to do: the console itself is unavailable
    }
  };

  const storage = () => {
    if (!usable) return undefined;
    try {
      return win ? win.localStorage : undefined;
    } catch (err) {
      warn("localStorage is not readable", err);
      return undefined;
    }
  };

  const readStored = () => {
    const store = storage();
    if (!store) return null;
    try {
      return store.getItem(THEME_KEY);
    } catch (err) {
      warn("localStorage.getItem failed", err);
      return null;
    }
  };

  const persist = (pref) => {
    const store = storage();
    if (!store) return;
    try {
      if (pref === "system") store.removeItem(THEME_KEY);
      else store.setItem(THEME_KEY, pref);
    } catch (err) {
      warn("localStorage could not be written", err);
    }
  };

  let media = null;
  try {
    if (win && typeof win.matchMedia === "function") media = win.matchMedia(LIGHT_QUERY);
  } catch {
    media = null;
  }
  const systemTheme = () => {
    try {
      return media && media.matches ? "light" : "dark";
    } catch {
      return "dark";
    }
  };

  const stored = readStored();
  let preference = stored === "light" || stored === "dark" ? stored : "system";
  let theme = "dark";

  const apply = () => {
    theme = preference === "system" ? systemTheme() : preference;
    try {
      root.setAttribute("data-theme", theme);
    } catch {
      // no root to write to (a test double without attributes): the controller still answers
    }
  };

  const onSystemChange = () => {
    if (preference === "system") apply();
  };
  try {
    if (media && typeof media.addEventListener === "function") {
      media.addEventListener("change", onSystemChange);
    } else if (media && typeof media.addListener === "function") {
      media.addListener(onSystemChange);
    }
  } catch {
    // no live updates; the theme still resolves at load and on every choice
  }

  apply();

  return {
    preference: () => preference,
    theme: () => theme,
    set(pref) {
      if (!PREFERENCES.includes(pref)) return;
      preference = pref;
      apply();
      persist(pref);
    },
  };
}

/**
 * Wire the header's <select data-theme-control> to a controller: it shows the preference in force,
 * a change applies at once, and its key events stop at the control, because hud/src/main.js mutes
 * the capture on a bare `m` anywhere in the document (the same fix as the ask box, T-0039 A19).
 */
export function mountThemeControl(control, controller) {
  if (!control || !controller) return;
  control.value = controller.preference();
  const choose = () => controller.set(control.value);
  control.addEventListener("change", choose);
  control.addEventListener("input", choose);
  for (const type of KEY_EVENTS) control.addEventListener(type, (event) => event.stopPropagation());
}

if (typeof window !== "undefined" && typeof document !== "undefined" && document.documentElement) {
  const controller = initTheme({ win: window, root: document.documentElement });
  const mount = () => mountThemeControl(document.querySelector("[data-theme-control]"), controller);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mount, { once: true });
  else mount();
}
