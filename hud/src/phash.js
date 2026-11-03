// dHash (difference hash) over a 9x8 grayscale grid — the `phash` field of astra.frame
// (criterion 5). The page computes it because it has the pixels; the server validates its shape
// only (local_astra/perception/browser_source.py). No DOM here.
//
// One extra column so each row yields 8 horizontal comparisons:
//   bit = gray[right] > gray[left] ? 1 : 0, MSB first within a row, rows top to bottom.
// Ties are 0, as in the imagehash reference (image[:,1:] > image[:,:-1]).

export const DHASH_COLUMNS = 9;
export const DHASH_ROWS = 8;
export const DHASH_SAMPLES = DHASH_COLUMNS * DHASH_ROWS;
/** Comparisons per row — one fewer than the columns, and the bit width of a row. */
export const DHASH_BITS_PER_ROW = DHASH_COLUMNS - 1;

/** 72 grayscale samples (row-major, 9 columns x 8 rows) -> 16 lowercase hex characters. */
export function dhash(gray) {
  if (!gray || typeof gray.length !== "number" || gray.length !== DHASH_SAMPLES) {
    // A hash of the wrong length reaches policy.scene_changed, which raises on differing
    // lengths (criterion 14). Refuse to produce one instead.
    throw new Error(
      `dhash needs ${DHASH_SAMPLES} samples (${DHASH_COLUMNS}x${DHASH_ROWS}), got ${
        gray ? gray.length : gray
      }`,
    );
  }
  let out = "";
  for (let y = 0; y < DHASH_ROWS; y++) {
    let byte = 0;
    for (let x = 0; x < DHASH_BITS_PER_ROW; x++) {
      const left = gray[y * DHASH_COLUMNS + x];
      const right = gray[y * DHASH_COLUMNS + x + 1];
      byte = (byte << 1) | (right > left ? 1 : 0);
    }
    out += byte.toString(16).padStart(2, "0");
  }
  return out;
}

/** RGBA bytes of a 9x8 draw -> the grayscale grid dhash() wants (ITU-R BT.601 luma). */
export function grayFromRgba(rgba) {
  const gray = new Array(DHASH_SAMPLES);
  for (let i = 0; i < DHASH_SAMPLES; i++) {
    const o = i * 4;
    gray[i] = 0.299 * rgba[o] + 0.587 * rgba[o + 1] + 0.114 * rgba[o + 2];
  }
  return gray;
}
