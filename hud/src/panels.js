// The three extension points of the later HUD tasks (criterion 10). With all three absent — the
// state today — the page renders completely and logs one line per missing panel; a loader that
// throws is caught and leaves every other panel and the socket working. This is what keeps those
// tasks out of T-0008's files.

export const PANEL_SPECS = [
  "./ui/index.js",
  "./denoise/index.js",
  "./assists/index.js",
  "./tools/index.js", // T-0162 B-1: the Tools pane, in the rail slot the Context pane left
  "./ask/index.js", // T-0039 criterion 10: the typed-question box, through the same seam
  "./log/index.js", // T-0043 criterion 19: the session-log checkbox, through the same seam
  "./session/index.js", // T-0162 D-1: the header's Restart button and its inline confirm
];

/** node says ERR_MODULE_NOT_FOUND, a browser says "failed to fetch dynamically imported module". */
const NOT_INSTALLED = /ERR_MODULE_NOT_FOUND|Cannot find module|dynamically imported module|404/i;

function notInstalled(err) {
  if (!err) return false;
  if (err.code === "ERR_MODULE_NOT_FOUND") return true;
  return NOT_INSTALLED.test(String(err.message || err));
}

const defaultImporter = (spec) => import(new URL(spec, import.meta.url).href);

/** importer(spec), started now, as a promise that never rejects: { ok, module } or { ok, error }. */
function startImport(importer, spec) {
  try {
    return Promise.resolve(importer(spec)).then(
      (module) => ({ ok: true, module }),
      (error) => ({ ok: false, error }),
    );
  } catch (error) {
    return Promise.resolve({ ok: false, error });
  }
}

/**
 * Mount whatever is installed. Never rejects. -> { mounted, missing, failed }
 *
 * T-0165 C6: every import starts before any is awaited (one wave of fetches, not seven in a row);
 * the mounts still run one after another in PANEL_SPECS order (T-0162 B-1: panes self-append in
 * mount order), whatever order the imports resolve in.
 */
export async function mountPanels({ importer = defaultImporter, host = {}, log = () => {} } = {}) {
  const mounted = [];
  const missing = [];
  const failed = [];
  const imports = PANEL_SPECS.map((spec) => startImport(importer, spec));
  for (const [index, spec] of PANEL_SPECS.entries()) {
    const { ok, module, error: err } = await imports[index];
    if (!ok) {
      if (notInstalled(err)) {
        missing.push(spec);
        log(`panel ${spec} is not installed yet — the page runs without it`);
      } else {
        failed.push(spec);
        log(`panel ${spec} failed to load: ${err && err.message}`);
      }
      continue;
    }
    try {
      if (!module || typeof module.mount !== "function")
        throw new TypeError("the panel exports no mount(host)");
      await module.mount(host);
      mounted.push(spec);
    } catch (err) {
      failed.push(spec);
      log(`panel ${spec} failed to mount: ${err && err.message}`);
    }
  }
  return { mounted, missing, failed };
}

/** T-0165 Q1 ruling: how long the page waits for its panels before opening the socket anyway. */
export const MOUNT_CEILING_MS = 3000;

/**
 * T-0165 C1/C5: call `open` exactly once — when `mounting` settles (fulfilled or rejected) or when
 * `ceilingMs` elapses, whichever comes first; never before either. The ceiling path logs one line
 * (a panel import that never settles must not leave the page connected to nothing). `open` is
 * passed the outcome, "mounted" | "timeout" | "failed", and the returned promise resolves to it.
 */
export function openAfterMount(
  mounting,
  open,
  {
    ceilingMs = MOUNT_CEILING_MS,
    setTimeout = globalThis.setTimeout,
    clearTimeout = globalThis.clearTimeout,
    log = () => {},
  } = {},
) {
  return new Promise((resolve) => {
    let done = false;
    let timer = null;
    const finish = (outcome) => {
      if (done) return;
      done = true;
      if (timer !== null) clearTimeout(timer);
      timer = null;
      open(outcome);
      resolve(outcome);
    };
    timer = setTimeout(() => {
      timer = null;
      if (done) return;
      log(`panels still mounting after ${ceilingMs} ms — opening the socket anyway`);
      finish("timeout");
    }, ceilingMs);
    Promise.resolve(mounting).then(
      () => finish("mounted"),
      () => finish("failed"),
    );
  });
}
