// T-0008 criterion 12 (loopback only) and the bounded-backoff half of criterion 3 — hud/src/ws.js.
//
// API this pins:
//   DEFAULT_WS_URL === "ws://127.0.0.1:8000/v1/realtime"
//   resolveWsUrl(search) -> the URL string to connect to; THROWS for any host that is not
//     127.0.0.1 / localhost / [::1] (the page shows the error instead of connecting).
//     `search` is a location.search string, e.g. "?ws=ws://localhost:8000/v1/realtime".
//   backoffDelay(attempt) -> ms, 250 initial, 8000 cap, jittered.
import { test } from "node:test";
import assert from "node:assert/strict";
import { cpSync, existsSync, mkdtempSync, writeFileSync } from "node:fs";
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

test("criterion 12: with no query parameter the page talks to loopback:8000", async () => {
  const { resolveWsUrl, DEFAULT_WS_URL } = await load("ws.js", "criterion 12");
  assert.equal(DEFAULT_WS_URL, "ws://127.0.0.1:8000/v1/realtime");
  assert.equal(resolveWsUrl(""), DEFAULT_WS_URL);
  assert.equal(resolveWsUrl("?source=canvas"), DEFAULT_WS_URL);
});

test("criterion 12: ?ws= is honoured for the three loopback hosts", async () => {
  const { resolveWsUrl } = await load("ws.js", "criterion 12");
  for (const url of [
    "ws://127.0.0.1:8000/v1/realtime",
    "ws://127.0.0.1:41234/v1/realtime",
    "ws://localhost:8000/v1/realtime",
    "ws://[::1]:8000/v1/realtime",
    "wss://127.0.0.1:8443/v1/realtime",
  ]) {
    assert.equal(
      resolveWsUrl("?ws=" + encodeURIComponent(url)),
      url,
      `criterion 12: ${url} is loopback and must be accepted`,
    );
  }
});

test("criterion 12: every non-loopback host is refused, including the lookalikes", async () => {
  const { resolveWsUrl } = await load("ws.js", "criterion 12");
  const refused = [
    "ws://127.0.0.1.evil.example/v1/realtime", // suffix trick
    "ws://[::1]@evil.example/v1/realtime", // userinfo trick
    "ws://localhost@evil.example/v1/realtime",
    "ws://user:pass@evil.example/v1/realtime",
    "ws://evil.example/v1/realtime",
    "wss://example.com/v1/realtime",
    "ws://127.0.0.1evil.example/v1/realtime",
    "ws://0.0.0.0:8000/v1/realtime", // binds everywhere: not loopback (docs/02 §9)
    "ws://192.168.1.10:8000/v1/realtime",
    "http://127.0.0.1:8000/v1/realtime", // not a WebSocket URL at all
    "javascript:alert(1)",
    "//evil.example/v1/realtime",
  ];
  for (const url of refused) {
    assert.throws(
      () => resolveWsUrl("?ws=" + encodeURIComponent(url)),
      `criterion 12: ${url} must be refused with a visible error, never connected to ` +
        "(CLAUDE.md: all services bind to loopback only)",
    );
  }
});

test("criterion 3: reconnect backoff starts at 250 ms, is jittered, and is capped at 8 s", async () => {
  const { backoffDelay } = await load("ws.js", "criterion 3");
  const sample = (attempt, n = 50) => Array.from({ length: n }, () => backoffDelay(attempt));
  const first = sample(0);
  assert.ok(
    first.every((d) => d > 0 && d <= 250),
    `criterion 3: the initial delay is 250 ms (jitter may only shorten it), got ${first.slice(0, 5)}`,
  );
  for (let attempt = 0; attempt <= 20; attempt++) {
    const ds = sample(attempt, 20);
    assert.ok(
      ds.every((d) => d >= 0 && d <= 8000),
      `criterion 3: bounded exponential backoff, 8 s cap; attempt ${attempt} gave ${ds.slice(0, 5)}`,
    );
  }
  assert.ok(
    Math.max(...sample(4)) > Math.max(...first),
    "criterion 3: the delay grows with the attempt count",
  );
  assert.ok(
    new Set(sample(4)).size > 1,
    "criterion 3: jitter — every client must not come back at the same instant",
  );
  assert.ok(
    Math.max(...sample(20)) >= 4000,
    "criterion 3: the backoff reaches the 8 s cap rather than collapsing to nothing",
  );
});
