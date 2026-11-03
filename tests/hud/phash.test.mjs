// T-0008 criterion 5, the phash half: hud/src/phash.js must produce a 16-character lowercase-hex
// dHash. The oracle is computed by hand below, not by a second implementation.
//
// API this pins:
//   dhash(gray) -> string of 16 lowercase hex characters
//     gray: 72 grayscale samples, 9 columns x 8 rows, row-major (the classic dHash grid: one
//     extra column so each row yields 8 horizontal comparisons).
//     bit = gray[right] > gray[left] ? 1 : 0, MSB first within a row, rows top to bottom.
//     Ties (equal neighbours) are 0, as in the imagehash reference (image[:,1:] > image[:,:-1]).
//   A grid of the wrong size throws: a short hash on the wire reaches policy.scene_changed, which
//   raises ValueError on differing hash lengths (criterion 14, implementation notes).
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

const W = 9;
const H = 8;

/** A 9x8 grid built row by row from a function of (x, y). */
function grid(f) {
  const out = new Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) out[y * W + x] = f(x, y);
  return out;
}

test("criterion 5: a left-to-right gradient is all ones (every right neighbour is brighter)", async () => {
  const { dhash } = await load("phash.js", "criterion 5");
  assert.equal(dhash(grid((x) => x * 28)), "ffffffffffffffff");
});

test("criterion 5: a right-to-left gradient is all zeros", async () => {
  const { dhash } = await load("phash.js", "criterion 5");
  assert.equal(dhash(grid((x) => 224 - x * 28)), "0000000000000000");
});

test("criterion 5: a flat image is all zeros — equal neighbours are not a change", async () => {
  const { dhash } = await load("phash.js", "criterion 5");
  assert.equal(dhash(grid(() => 120)), "0000000000000000");
});

test("criterion 5: bit order is MSB-first within a row, rows top to bottom", async () => {
  const { dhash } = await load("phash.js", "criterion 5");
  // Row y is built so that its 8 comparisons spell the byte y: start at 128 and step +10 for a
  // 1 bit, -10 for a 0 bit, reading y's bits from the most significant. Expected hash, by hand:
  // bytes 0x00 0x01 0x02 0x03 0x04 0x05 0x06 0x07 -> "0001020304050607".
  const rows = [];
  for (let y = 0; y < H; y++) {
    const row = [128];
    for (let x = 0; x < 8; x++) {
      const bit = (y >> (7 - x)) & 1;
      row.push(row[x] + (bit ? 10 : -10));
    }
    rows.push(row);
  }
  assert.equal(dhash(rows.flat()), "0001020304050607");
});

test("criterion 5: one flipped comparison flips exactly one bit", async () => {
  const { dhash } = await load("phash.js", "criterion 5");
  const g = grid((x) => x * 10);
  // Row 3, comparison 5 (between columns 5 and 6) is made to go the other way: 50 -> 45.
  g[3 * W + 6] = 45;
  assert.equal(dhash(g), "fffffffbffffffff"); // row 3 loses bit 1<<2
  const a = BigInt("0x" + dhash(grid((x) => x * 10)));
  const b = BigInt("0x" + dhash(g));
  let diff = a ^ b;
  let bits = 0;
  while (diff) {
    bits += Number(diff & 1n);
    diff >>= 1n;
  }
  assert.equal(bits, 1, "a one-pixel-pair change is a Hamming distance of 1");
});

test("criterion 5: the output is always 16 lowercase hex characters", async () => {
  const { dhash } = await load("phash.js", "criterion 5");
  let seed = 12345; // deterministic LCG, no Math.random in a test
  const next = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed % 256;
  };
  for (let i = 0; i < 50; i++) {
    const h = dhash(grid(() => next()));
    assert.match(h, /^[0-9a-f]{16}$/, `criterion 5: 16-character lowercase-hex dHash, got ${h}`);
  }
});

test("criterion 5/14: a grid of the wrong size throws instead of returning a short hash", async () => {
  const { dhash } = await load("phash.js", "criterion 5");
  for (const bad of [[], new Array(71).fill(0), new Array(73).fill(0), new Array(64).fill(0)]) {
    assert.throws(
      () => dhash(bad),
      `a hash of the wrong length reaches policy.scene_changed, which raises ValueError on ` +
        `differing hash lengths (${bad.length} samples must not produce a hash)`,
    );
  }
});
