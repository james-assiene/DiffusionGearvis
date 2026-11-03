// T-0008 criterion 10 — hud/src/panels.js: the three later tasks mount here, and their absence
// (the state today) is the normal case, not an error.
//
// API this pins:
//   PANEL_SPECS === ["./ui/index.js", "./denoise/index.js", "./assists/index.js", "./ask/index.js"]
//   mountPanels({ importer, host, log }) -> Promise<{ mounted: [...], missing: [...], failed: [...] }>
//     importer(spec) -> Promise<module>, defaulting to dynamic import relative to panels.js;
//     a module mounts through its exported mount(host). mountPanels never rejects.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

// node 18 treats a bare .js file as CommonJS unless a package.json in scope says
// {"type":"module"}, and T-0008 creates no package.json under hud/ (T-0010 owns hud/package.json,
// per the split in docs/tasks/T-0008.md). In the browser the MIME type decides and the very same
// files are plain ES modules; so to import them here they are copied once into a temp directory
// that does carry {"type":"module"}. If hud/ ever gains that package.json this shim stays correct.
let ESM_ROOT = null;
function srcRoot() {
  if (ESM_ROOT) return ESM_ROOT;
  const real = fileURLToPath(new URL("../../hud/src", import.meta.url));
  const dir = mkdtempSync(join(tmpdir(), "astra-hud-esm-"));
  writeFileSync(join(dir, "package.json"), '{"type":"module","private":true}');
  if (existsSync(real)) cpSync(real, join(dir, "src"), { recursive: true });
  ESM_ROOT = pathToFileURL(join(dir, "src") + "/");
  return ESM_ROOT;
}

async function load(name, criterion) {
  try {
    return await import(new URL(name, srcRoot()).href);
  } catch (err) {
    assert.fail(
      `T-0008 ${criterion} adds hud/src/${name}, importable in node as an ES module with no DOM ` +
        `at module top level (docs/tasks/T-0008.md implementation notes: "keep DOM out of ` +
        `state.js, protocol.js, frames.js, phash.js, ws.js"). node could not import it: ` +
        `${err.message}`,
    );
  }
}

const UI = "./ui/index.js";
const DENOISE = "./denoise/index.js";
const ASSISTS = "./assists/index.js";
// T-0039 criterion 10: the typed-question panel mounts through this very seam, which is what
// keeps hud/src/main.js at zero diff (criterion 20).
const ASK = "./ask/index.js";
// T-0043 criterion 19: the session-log checkbox mounts through the same seam, which is what keeps
// hud/src/main.js and hud/src/state.js at zero diff (criterion 23).
const LOG = "./log/index.js";
// T-0162 B-1: the Tools pane, mounted through PANEL_SPECS in the slot where Context was.
const TOOLS = "./tools/index.js";
// T-0162 B-1, the same amendment the ask panel made one panel earlier: once hud/src/tools/index.js
// is installed it is one of "every other panel" the two criterion-10 isolation rows below expect.
const TOOLS_EXPECTED = existsSync(fileURLToPath(new URL("../../hud/src/tools/index.js", import.meta.url)))
  ? [TOOLS]
  : [];
// T-0162 D-1 (amendment 2026-10-27): the header's Restart button mounts through the same seam, after
// the session log. Admitted here by name, and only once hud/src/session/index.js is installed.
const SESSION = "./session/index.js";
const SESSION_EXPECTED = existsSync(fileURLToPath(new URL("../../hud/src/session/index.js", import.meta.url)))
  ? [SESSION]
  : [];

// T-0039 criterion 10, the same amendment one panel later: with hud/src/ask/index.js installed the
// truth becomes "ui, assists and ask mounted, denoise missing".
const ASK_INSTALLED = existsSync(
  fileURLToPath(new URL("../../hud/src/ask/index.js", import.meta.url)),
);

function notFound(spec) {
  const err = new Error(`Cannot find module '${spec}'`);
  err.code = "ERR_MODULE_NOT_FOUND";
  return err;
}

test("criterion 10: the extension points are T-0010, T-0011, T-0012 and T-0039's", async () => {
  const { PANEL_SPECS } = await load("panels.js", "criterion 10");
  assert.deepEqual(
    [...PANEL_SPECS].slice(0, 3),
    [UI, DENOISE, ASSISTS],
    "criterion 10: ./ui/index.js (T-0010), ./denoise/index.js (T-0011), ./assists/index.js " +
      "(T-0012) — this is what keeps the three later tasks out of T-0008's files, and T-0039 " +
      "adds to the list rather than reordering it",
  );
  assert.deepEqual(
    // T-0043 criterion 19 adds ./log/index.js to this allowed set — the scope fence is updated
    // with the task named, not deleted: PANEL_SPECS still may not grow anything nobody owns.
    // T-0162 B-1 adds ./tools/index.js (the Tools pane) to the same allowed set, named here.
    // T-0162 D-1 adds ./session/index.js (the Restart button), named here.
    [...PANEL_SPECS].filter((spec) => ![UI, DENOISE, ASSISTS, ASK, LOG, TOOLS, SESSION].includes(spec)),
    [],
    "…and nothing else joined it: the loader contract is unchanged (T-0039 criterion 20, " +
      "T-0043 criterion 19)",
  );
});

// T-0039 criterion 10, held by the same source sentinel as tests/hud/ask.test.mjs: node tests
// cannot carry pytest's `pending` marker, and tests/unit/test_hud_js.py runs this file inside the
// fast suite, so a row that is red before its task lands would redden the gate for every agent.
const askOpts = ASK_INSTALLED
  ? {}
  : {
      skip:
        "T-0039 group B is not implemented yet: hud/src/ask/index.js does not exist. This row " +
        "is criterion 10 of docs/tasks/T-0039.md and runs the moment the panel lands.",
    };

test("T-0039 criterion 10: PANEL_SPECS gains ./ask/index.js", askOpts, async () => {
  const { PANEL_SPECS } = await load("panels.js", "criterion 10");
  assert.ok(
    [...PANEL_SPECS].includes(ASK),
    "T-0039 criterion 10: hud/src/ask/index.js is loaded through the existing panels.js seam — " +
      "which is what keeps hud/src/main.js at zero diff (criterion 20). PANEL_SPECS is " +
      `${JSON.stringify([...PANEL_SPECS])}`,
  );
});

// T-0043 criterion 19, held by the same kind of source sentinel: the stub modules exist (so this
// file and tests/hud/log_panel.test.mjs import), and the panel is "installed" once index.js
// exports a mount.
const LOG_INSTALLED = (() => {
  const file = fileURLToPath(new URL("../../hud/src/log/index.js", import.meta.url));
  if (!existsSync(file)) return false;
  return /export\s+(?:async\s+)?function\s+mount\b|export\s*\{[^}]*\bmount\b/.test(
    readFileSync(file, "utf8"),
  );
})();
const logOpts = LOG_INSTALLED
  ? {}
  : {
      skip:
        "T-0043 group B is not implemented yet: hud/src/log/index.js exports no mount(host). " +
        "This row is criterion 19 of docs/tasks/T-0043.md and runs the moment the panel lands.",
    };

test("T-0043 criterion 19: PANEL_SPECS gains ./log/index.js", logOpts, async () => {
  const { PANEL_SPECS } = await load("panels.js", "criterion 10");
  assert.ok(
    [...PANEL_SPECS].includes(LOG),
    "T-0043 criterion 19: hud/src/log/index.js mounts through the existing panels.js seam — " +
      "hud/index.html, hud/styles.css, hud/src/main.js and hud/src/state.js are not touched. " +
      `PANEL_SPECS is ${JSON.stringify([...PANEL_SPECS])}`,
  );
});

test("criterion 10: with all three panels absent the page still mounts, and logs one line each", async () => {
  const { mountPanels, PANEL_SPECS } = await load("panels.js", "criterion 10");
  const logged = [];
  const result = await mountPanels({
    importer: (spec) => Promise.reject(notFound(spec)),
    host: {},
    log: (line) => logged.push(String(line)),
  });
  assert.deepEqual([...result.mounted], [], "nothing mounts when nothing is there");
  assert.deepEqual([...result.missing].sort(), [...PANEL_SPECS].sort());
  assert.equal(
    logged.length,
    PANEL_SPECS.length,
    `criterion 10: one log line per missing panel, got ${logged.length}: ${logged.join(" | ")}`,
  );
  for (const spec of PANEL_SPECS) {
    assert.ok(
      logged.some((l) => l.includes(spec)),
      `criterion 10: the log line must name the missing panel (${spec}), got ${logged.join(" | ")}`,
    );
  }
});

test("criterion 10: a loader that throws leaves every other panel mounted", async () => {
  const { mountPanels } = await load("panels.js", "criterion 10");
  const mounted = [];
  const importer = (spec) => {
    if (spec === UI) return Promise.reject(new TypeError("boom inside the A2UI bundle"));
    return Promise.resolve({
      mount: (host) => {
        mounted.push(spec);
        return { host };
      },
    });
  };
  const result = await mountPanels({ importer, host: {}, log: () => {} });
  assert.deepEqual(
    mounted.sort(),
    [ASSISTS, DENOISE, ASK, LOG, ...TOOLS_EXPECTED, ...SESSION_EXPECTED].sort(),
    "criterion 10: a throwing loader is caught and leaves *every* other panel working — the " +
      "T-0039 ask panel and the T-0043 log panel included, both of which joined PANEL_SPECS and " +
      "are mounted through this same seam. The list is PANEL_SPECS minus the one that threw, so " +
      "a panel quietly dropped from the seam reddens this row as well as the one above",
  );
  assert.deepEqual([...result.failed], [UI]);
  assert.ok([...result.mounted].includes(DENOISE));
});

test("criterion 10: a panel whose mount() throws does not take the others down", async () => {
  const { mountPanels } = await load("panels.js", "criterion 10");
  const mounted = [];
  const importer = (spec) =>
    Promise.resolve({
      mount: () => {
        if (spec === DENOISE) throw new Error("mount blew up");
        mounted.push(spec);
      },
    });
  const result = await mountPanels({ importer, host: {}, log: () => {} });
  assert.deepEqual(
    mounted.sort(),
    [ASSISTS, UI, ASK, LOG, ...TOOLS_EXPECTED, ...SESSION_EXPECTED].sort(),
    "criterion 10: a mount() that throws is caught and leaves every other panel mounted — the " +
      "T-0039 ask panel and the T-0043 log panel included (PANEL_SPECS minus the one that threw)",
  );
  assert.deepEqual([...result.failed], [DENOISE]);
});

// T-0025 amendment (docs/tasks/T-0025.md criterion 1, "Tests to write"): the "all three absent"
// case with the *real* default importer becomes "the two absent" as soon as the assist panel
// lands. PANEL_SPECS is unchanged and the injected-importer cases above are unchanged — the
// amendment is the arrival of one panel, nothing else. The branch below keeps this file green on
// today's tree and tightens itself the moment hud/src/assists/index.js exists; no marker is
// possible here because node tests cannot carry pytest's `pending`.
const ASSISTS_INSTALLED = existsSync(
  fileURLToPath(new URL("../../hud/src/assists/index.js", import.meta.url)),
);

// T-0024 criterion 11, the same amendment one panel later: when hud/src/ui/index.js lands, the
// truth becomes "ui and assists mounted, denoise missing". Under `node --test` there is no DOM,
// so mount({}) either degrades to a no-op (mounted) or reports itself (failed) — both are
// acceptable here and neither may take the other panels down; what is *not* acceptable is the
// panel disappearing from the report or mountPanels rejecting (criterion 2).
const UI_INSTALLED = existsSync(
  fileURLToPath(new URL("../../hud/src/ui/index.js", import.meta.url)),
);


test("criteria 10 and T-0025/1: the real default importer mounts what is installed and never rejects", async () => {
  const { mountPanels, PANEL_SPECS } = await load("panels.js", "criterion 10");
  // No importer injected: this exercises the actual dynamic import of ./ui/index.js &c. The page
  // must render completely whether or not a panel is there, and mount() is called with host {} —
  // T-0025 criterion 1: "mount({}) ... returns without throwing, renders nothing and sends
  // nothing, so a host that predates this seam degrades instead of taking the socket down".
  const result = await mountPanels({ host: {}, log: () => {} });
  if (UI_INSTALLED) {
    assert.deepEqual(
      [...result.missing],
      [...PANEL_SPECS].filter(
        (spec) =>
          !existsSync(fileURLToPath(new URL(`../../hud/src/${spec.slice(2)}`, import.meta.url))),
      ),
      "T-0024 criterion 2/11: with hud/src/ui/index.js installed only ./denoise/index.js is " +
        "still missing (T-0011c) — T-0008 criterion 10 keeps holding, one panel later. T-0039 " +
        "criterion 10 adds ./ask/index.js to that reckoning.",
    );
    if (ASK_INSTALLED) {
      assert.ok(
        [...result.mounted, ...result.failed].includes(ASK),
        "T-0039 criterion 10: the ask panel is reported in `mounted` (or, with no DOM under " +
          "node --test, in `failed`) — never silently absent",
      );
      assert.ok(
        ![...result.failed].includes(ASSISTS) && ![...result.failed].includes(UI),
        "T-0008 criterion 10 negative: the new panel leaves the others working",
      );
    }
    assert.ok(
      [...result.mounted].includes(ASSISTS),
      "T-0025 criterion 1: the assist panel keeps mounting through the same seam",
    );
    assert.ok(
      [...result.mounted, ...result.failed].includes(UI),
      "T-0024 criterion 2: the A2UI panel is reported in `mounted` (or, with no DOM under " +
        "node --test, in `failed`) — never silently absent",
    );
    assert.ok(
      ![...result.failed].includes(ASSISTS),
      "T-0024 criterion 2 negative: a panel that cannot load leaves the others working",
    );
  } else if (ASSISTS_INSTALLED) {
    assert.deepEqual(
      [...result.mounted],
      [ASSISTS],
      "T-0025 criterion 1: the assist panel mounts through the seam and is reported in `mounted`",
    );
    assert.deepEqual(
      [...result.missing],
      [UI, DENOISE],
      "T-0025 criterion 1: the other two specs stay in `missing` (T-0010b and T-0011c)",
    );
    assert.deepEqual([...result.failed], [], "T-0025 criterion 1: mount({}) must not throw");
  } else {
    assert.deepEqual([...result.mounted], []);
    assert.equal([...result.missing].length, PANEL_SPECS.length);
  }
});

// --- T-0162 group B: the Tools pane joins PANEL_SPECS (B-1) ---------------------------------------
//
// docs/tasks/T-0162.md B-1: "a new hud/src/tools/index.js, mounted through PANEL_SPECS. It renders
// a rail pane headed 'Tools' in the slot where Context was (between Thinking and the session log)."
// Panes self-append to <main> in mount order, so "before the session log" is "before ./log/index.js
// in PANEL_SPECS". Gated by docs/tasks/T-0162.md's Status phrase "group B: tests …, activated <sha>"
// (the pendingT0159B mechanism of tests/hud/denoise.test.mjs): strict xfail until then.

const T0162_TASK = fileURLToPath(new URL("../../docs/tasks/T-0162.md", import.meta.url));
const T0162_STATUS = existsSync(T0162_TASK)
  ? ((/^\*\*Status:\*\*\s*(.*)$/m.exec(readFileSync(T0162_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0162_B_PENDING =
  !/\bgroup B:[^;)]*\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0162_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0162B(name, fn) {
  test(name + (T0162_B_PENDING ? " [pending T-0162 B: strict xfail until group B is activated]" : ""), async (t) => {
    if (!T0162_B_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail("XPASS(strict): this T-0162 group B row passed while docs/tasks/T-0162.md's status does not record group B's activation.");
    }
    t.diagnostic("expected failure (T-0162 B pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

pendingT0162B("T-0162 B-1: PANEL_SPECS gains ./tools/index.js, once, before ./log/index.js; the earlier order is kept", async () => {
  const { PANEL_SPECS } = await load("panels.js", "criterion 10");
  const specs = [...PANEL_SPECS];
  assert.equal(specs.filter((s) => s === TOOLS).length, 1, `B-1: ./tools/index.js is in PANEL_SPECS once; got ${JSON.stringify(specs)}`);
  assert.ok(specs.indexOf(TOOLS) < specs.indexOf(LOG), "B-1: the Tools pane mounts before the session log (the slot where Context was)");
  // T-0162 D-1 (amendment 2026-10-27): ./session/index.js may follow ./log/index.js, once, and nothing else joins.
  assert.ok(specs.filter((s) => s === SESSION).length <= 1, "D-1: ./session/index.js at most once");
  if (specs.includes(SESSION)) assert.ok(specs.indexOf(SESSION) > specs.indexOf(LOG), "D-1: the Restart panel mounts after the session log");
  assert.deepEqual(specs.filter((s) => s !== TOOLS && s !== SESSION), [UI, DENOISE, ASSISTS, ASK, LOG], "B-1: nothing else moves");
});

// --- T-0165 C5 and C6: the socket opens after the mount; imports in parallel, mounts in order ------
//
// docs/tasks/T-0165.md C5 (a DOM-free helper in hud/src/panels.js decides when main.js calls
// socket.open()) and C6 (mountPanels starts every PANEL_SPECS import before awaiting any, and mounts
// in PANEL_SPECS order, each after the previous mount settled). Contract pinned here, from C5's
// "proposed contract":
//   openAfterMount(mounting: Promise, open: () => void, { ceilingMs, setTimeout, clearTimeout, log })
//     calls `open` exactly once: when `mounting` settles (fulfilled or rejected) or when `ceilingMs`
//     elapses, whichever is first; one log line on the ceiling path; never before either event.
// Timers are injected (no wall clock). Gated like pendingT0162B: strict xfail until T-0165's
// Status line records "activated <sha>" (or ASTRA_RUN_PENDING=1).

const T0165_TASK = fileURLToPath(new URL("../../docs/tasks/T-0165.md", import.meta.url));
const T0165_STATUS = existsSync(T0165_TASK)
  ? ((/^\**Status:\**\s*(.*)$/m.exec(readFileSync(T0165_TASK, "utf8")) || [])[1] || "").trim()
  : "";
const T0165_PENDING =
  !/\bactivated\s+[0-9a-f]{7,40}\b/i.test(T0165_STATUS) && process.env.ASTRA_RUN_PENDING !== "1";

function pendingT0165(name, fn) {
  test(name + (T0165_PENDING ? " [pending T-0165: strict xfail until the tests are activated]" : ""), async (t) => {
    if (!T0165_PENDING) return fn(t);
    let failure = null;
    try {
      await fn(t);
    } catch (err) {
      failure = err;
    }
    if (failure === null) {
      assert.fail("XPASS(strict): this T-0165 row passed while docs/tasks/T-0165.md's Status does not record its activation.");
    }
    t.diagnostic("expected failure (T-0165 pending): " + String(failure && failure.message).split("\n")[0]);
  });
}

/** A promise with its settle functions outside. */
function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** Let every queued microtask and the I/O turn behind it run (no timers involved). */
async function flush(rounds = 5) {
  for (let i = 0; i < rounds; i += 1) await new Promise((r) => setImmediate(r));
}

const SPECS_T0165 = [UI, DENOISE, ASSISTS, TOOLS, ASK, LOG, SESSION];

/** An importer whose every spec answers with a pre-made deferred; records the order of calls. */
function deferredImporter(log) {
  const pending = new Map(SPECS_T0165.map((spec) => [spec, deferred()]));
  const calls = [];
  const importer = (spec) => {
    calls.push(spec);
    log.push(`import ${spec}`);
    const d = pending.get(spec);
    assert.ok(d, `T-0165 C6: mountPanels asked for a spec that is not in PANEL_SPECS: ${spec}`);
    return d.promise;
  };
  return { importer, calls, pending };
}

function recordingModule(spec, log, mountFn) {
  return {
    mount: (host) => {
      log.push(`mount ${spec}`);
      return mountFn ? mountFn(host) : undefined;
    },
  };
}

pendingT0165("T-0165 C6: every import starts before any resolves", async () => {
  const { mountPanels, PANEL_SPECS } = await load("panels.js", "T-0165 C6");
  assert.deepEqual([...PANEL_SPECS], SPECS_T0165, "premise (C9 g): PANEL_SPECS keeps its content and order");
  const log = [];
  const { importer, calls, pending } = deferredImporter(log);
  const done = mountPanels({ importer, host: {}, log: () => {} });
  await flush();
  assert.deepEqual(
    [...calls].sort(),
    [...SPECS_T0165].sort(),
    "T-0165 C6: mountPanels starts every PANEL_SPECS import before awaiting any; with no import " +
      `resolved yet it had started ${calls.length} of ${SPECS_T0165.length}: ${JSON.stringify(calls)}`,
  );
  assert.equal(calls.length, SPECS_T0165.length, `C6: each import is started once; calls ${JSON.stringify(calls)}`);
  for (const spec of SPECS_T0165) pending.get(spec).resolve(recordingModule(spec, log));
  const result = await done;
  assert.deepEqual([...result.mounted], SPECS_T0165, "C6: and then every panel mounts");
});

pendingT0165("T-0165 C6: mounts run in PANEL_SPECS order when imports resolve in reverse", async () => {
  // Adversarial: a parallel loader that mounts each panel in its import's .then() mounts them in
  // resolution order, here last-first. Panes self-append to <main> in mount order (T-0162 B-1), so
  // that would reverse the rail.
  const { mountPanels } = await load("panels.js", "T-0165 C6");
  const log = [];
  const { importer, calls, pending } = deferredImporter(log);
  const done = mountPanels({ importer, host: {}, log: () => {} });
  await flush();
  for (const spec of [...SPECS_T0165].reverse()) {
    assert.ok(
      calls.includes(spec),
      `T-0165 C6: the import of ${spec} must already be in flight before any import resolves ` +
        `(started so far: ${JSON.stringify(calls)}); without that the reverse order below is not real`,
    );
    pending.get(spec).resolve(recordingModule(spec, log));
    await flush();
  }
  const result = await done;
  const mounts = log.filter((line) => line.startsWith("mount ")).map((line) => line.slice(6));
  assert.deepEqual(
    mounts,
    SPECS_T0165,
    `T-0165 C6: mount(host) is called in PANEL_SPECS order whatever order the imports resolve in; got ${JSON.stringify(mounts)}`,
  );
  assert.deepEqual([...result.mounted], SPECS_T0165, "C6: `mounted` keeps PANEL_SPECS order");
  assert.deepEqual([...result.missing], []);
  assert.deepEqual([...result.failed], []);
});

pendingT0165("T-0165 C6: mount i+1 is not called before mount i settles", async () => {
  const { mountPanels } = await load("panels.js", "T-0165 C6");
  const log = [];
  const calls = [];
  const gate = deferred();
  const importer = (spec) => {
    calls.push(spec);
    return Promise.resolve(
      recordingModule(spec, log, () => {
        if (spec !== DENOISE) return undefined;
        return gate.promise.then(() => log.push(`settled ${spec}`));
      }),
    );
  };
  const done = mountPanels({ importer, host: {}, log: () => {} });
  await flush();
  const mountsWhilePending = log.filter((line) => line.startsWith("mount "));
  assert.equal(
    calls.length,
    SPECS_T0165.length,
    "T-0165 C6: while one mount is still pending every import has already started; started " +
      JSON.stringify(calls),
  );
  assert.deepEqual(
    mountsWhilePending,
    [`mount ${UI}`, `mount ${DENOISE}`],
    "T-0165 C6: mount(host) of the next panel waits until the previous mount settled; while " +
      `${DENOISE}'s mount was pending the loader had called ${JSON.stringify(mountsWhilePending)}`,
  );
  gate.resolve();
  const result = await done;
  const order = log.filter((line) => line.startsWith("mount ") || line.startsWith("settled "));
  assert.deepEqual(
    order,
    [`mount ${UI}`, `mount ${DENOISE}`, `settled ${DENOISE}`, ...SPECS_T0165.slice(2).map((s) => `mount ${s}`)],
    `C6: the later mounts follow the settle, in PANEL_SPECS order; got ${JSON.stringify(order)}`,
  );
  assert.deepEqual([...result.mounted], SPECS_T0165);
});

pendingT0165(
  "T-0165 C6: an import that rejects mid-flight leaves the others mounted in order, and the result lists keep PANEL_SPECS order",
  async () => {
    const { mountPanels } = await load("panels.js", "T-0165 C6");
    const log = [];
    const { importer, calls, pending } = deferredImporter(log);
    const lines = [];
    const done = mountPanels({ importer, host: {}, log: (line) => lines.push(String(line)) });
    let rejected = null;
    done.catch((err) => {
      rejected = err;
    });
    await flush();
    assert.equal(calls.length, SPECS_T0165.length, `C6 premise: every import is in flight; started ${JSON.stringify(calls)}`);
    // Mid-flight, out of order: LOG fails to load (a real error), then DENOISE is not installed,
    // both before UI (the first spec) has resolved; the rest resolve last-first.
    pending.get(LOG).reject(new TypeError("boom inside the log panel"));
    await flush();
    pending.get(DENOISE).reject(notFound(DENOISE));
    await flush();
    for (const spec of [SESSION, ASK, TOOLS, ASSISTS, UI]) {
      pending.get(spec).resolve(recordingModule(spec, log));
      await flush();
    }
    const result = await done;
    assert.equal(rejected, null, "C6: mountPanels never rejects");
    const expected = [UI, ASSISTS, TOOLS, ASK, SESSION];
    const mounts = log.filter((line) => line.startsWith("mount ")).map((line) => line.slice(6));
    assert.deepEqual(mounts, expected, `C6: the others mount, in PANEL_SPECS order; got ${JSON.stringify(mounts)}`);
    assert.deepEqual([...result.mounted], expected, "C6: `mounted` in PANEL_SPECS order");
    assert.deepEqual([...result.missing], [DENOISE], "C6: a not-installed import is `missing`");
    assert.deepEqual([...result.failed], [LOG], "C6: an import that throws is `failed`");
    assert.ok(lines.some((l) => l.includes(DENOISE)) && lines.some((l) => l.includes(LOG)), `C6: one line each; ${lines.join(" | ")}`);
  },
);

// --- C5: openAfterMount, with injected timers --------------------------------------------------

const CEILING_MS = 3000; // ruling Q1 (2026-10-27, provisional); the helper takes it as a parameter

/** setTimeout/clearTimeout stand-ins: nothing fires until the row says so. */
function fakeTimers() {
  const live = new Map(); // id -> fn, while armed
  const armed = [];
  const cleared = [];
  let next = 1;
  return {
    armed,
    cleared,
    setTimeout: (fn, ms) => {
      const id = next;
      next += 1;
      live.set(id, fn);
      armed.push({ id, ms });
      return id;
    },
    clearTimeout: (id) => {
      cleared.push(id);
      live.delete(id);
    },
    /** Fire every timer still armed, as the clock reaching it would. */
    fire() {
      for (const [id, fn] of [...live]) {
        live.delete(id);
        fn();
      }
    },
  };
}

async function loadHelper() {
  const mod = await load("panels.js", "T-0165 C5");
  assert.equal(
    typeof mod.openAfterMount,
    "function",
    "T-0165 C5: hud/src/panels.js exports openAfterMount(mounting, open, { ceilingMs, setTimeout, clearTimeout, log })",
  );
  return mod.openAfterMount;
}

function harness(openAfterMount, mounting) {
  const clock = fakeTimers();
  const opened = [];
  const lines = [];
  openAfterMount(mounting, () => opened.push("open"), {
    ceilingMs: CEILING_MS,
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    log: (line) => lines.push(String(line)),
  });
  return { clock, opened, lines };
}

pendingT0165("T-0165 C5: open is not called while mounting is pending, and once when it resolves", async () => {
  const openAfterMount = await loadHelper();
  const mounting = deferred();
  const h = harness(openAfterMount, mounting.promise);
  assert.equal(h.opened.length, 0, "C5: open is never called at the helper's entry, before mounting settles");
  await flush();
  assert.equal(h.opened.length, 0, "C5: …nor while mounting is pending and the ceiling has not fired");
  assert.deepEqual(
    h.clock.armed.map((t) => t.ms),
    [CEILING_MS],
    `C5: one ceiling timer, armed at ceilingMs; armed ${JSON.stringify(h.clock.armed)}`,
  );
  mounting.resolve({ mounted: [], missing: [], failed: [] });
  await flush();
  assert.equal(h.opened.length, 1, "C5: open is called once when mounting resolves");
  assert.deepEqual(h.lines, [], "C5: the settle path logs nothing about a ceiling");
});

pendingT0165("T-0165 C5: open is called once when mounting rejects", async () => {
  const openAfterMount = await loadHelper();
  const mounting = deferred();
  const h = harness(openAfterMount, mounting.promise);
  await flush();
  assert.equal(h.opened.length, 0, "C5: not before the rejection");
  mounting.reject(new Error("the loader itself blew up"));
  await flush();
  assert.equal(h.opened.length, 1, "C5: a rejected mount still opens the socket, once");
});

pendingT0165("T-0165 C5: a mount that never settles opens once at the ceiling, with one log line", async () => {
  const openAfterMount = await loadHelper();
  const h = harness(openAfterMount, new Promise(() => {}));
  await flush();
  assert.equal(h.opened.length, 0, "C5: nothing before the ceiling");
  h.clock.fire();
  await flush();
  assert.equal(h.opened.length, 1, "C5: the ceiling opens the socket, once");
  assert.equal(h.lines.length, 1, `C5: the ceiling path logs one line; got ${JSON.stringify(h.lines)}`);
});

pendingT0165("T-0165 C5: the ceiling fires and then mounting resolves: open is still called once", async () => {
  // Adversarial: the late mount must not open a second socket (a second hello, C9 b).
  const openAfterMount = await loadHelper();
  const mounting = deferred();
  const h = harness(openAfterMount, mounting.promise);
  await flush();
  h.clock.fire();
  await flush();
  assert.equal(h.opened.length, 1, "C5 premise: the ceiling opened it");
  mounting.resolve({ mounted: [], missing: [], failed: [] });
  await flush();
  assert.equal(h.opened.length, 1, `C5: exactly once, though the mount finished after the ceiling; open ran ${h.opened.length} times`);
  assert.equal(h.lines.length, 1, `C5: still the one ceiling line; ${JSON.stringify(h.lines)}`);
});

pendingT0165("T-0165 C5: mounting settles and then the ceiling: open once, and the timer was cleared", async () => {
  const openAfterMount = await loadHelper();
  const mounting = deferred();
  const h = harness(openAfterMount, mounting.promise);
  await flush();
  mounting.resolve({ mounted: [], missing: [], failed: [] });
  await flush();
  assert.equal(h.opened.length, 1, "C5 premise: the settle opened it");
  const [timer] = h.clock.armed;
  assert.ok(timer, "C5: a ceiling timer was armed");
  assert.ok(h.clock.cleared.includes(timer.id), `C5: the ceiling timer is cleared once mounting settled; cleared ${JSON.stringify(h.clock.cleared)}`);
  // The clock then passes the ceiling: whatever is still armed fires.
  h.clock.fire();
  await flush();
  assert.equal(h.opened.length, 1, `C5: exactly once; open ran ${h.opened.length} times`);
  assert.deepEqual(h.lines, [], `C5: no ceiling line on the settle path; ${JSON.stringify(h.lines)}`);
});

pendingT0165("T-0165 C5: with no ceilingMs given, the ceiling is ruling Q1's 3000 ms", async () => {
  // main.js relies on the helper's default (review, 2026-10-27): every other C5 row passes
  // ceilingMs explicitly, so a default of 15000 survived them. Both directions are pinned by the
  // exact value: a ceiling far above 3000 leaves a hung page dead for longer than a human waits; one
  // below it opens before slow panels on a loaded box (docs/tasks/T-0165.md Q1).
  const openAfterMount = await loadHelper();
  const clock = fakeTimers();
  const opened = [];
  openAfterMount(new Promise(() => {}), () => opened.push("open"), {
    setTimeout: clock.setTimeout,
    clearTimeout: clock.clearTimeout,
    log: () => {},
  });
  await flush();
  assert.deepEqual(
    clock.armed.map((t) => t.ms),
    [CEILING_MS],
    `C5/Q1: the default ceiling is ${CEILING_MS} ms; armed ${JSON.stringify(clock.armed)}`,
  );
  assert.equal(opened.length, 0, "C5: nothing before the ceiling");
  clock.fire();
  await flush();
  assert.equal(opened.length, 1, "C5: the default ceiling opens once");
});
