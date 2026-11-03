# tests/fixtures/t0204 — T-0204 page-readiness fixtures

Provenance: written by hand by the T-0204 test-author on 2026-10-29 (no generator kept; the tile palette below is
the hand-chosen list). Licence: Apache-2.0, as the repository. Served only by the tests' own loopback
`http.server` on 127.0.0.1 (and `localhost` for the cross-origin row); no network.

Group A rows use `tiles.html`; groups B and C add their own rows here (`lazy_scroll.html`, `animated.html`,
`noise.html`, `caret.html`, per docs/tasks/T-0204.md Owned files).

## tiles.html

Geometry in CSS px at a 1280x800 viewport, scroll 0 (by hand from the stylesheet; `body` margin 0,
`overflow-anchor: none` so Chromium never compensates a shift by scrolling):

- `#head`: y 0-100, full width, rgb(40, 40, 40).
- `#grid`: x 40-1240, y 100-1000; 6 columns of 200 px, rows of 150 px; tile *n* (1-36) is at column `(n-1) % 6`,
  row `(n-1) // 6`, i.e. x `40 + 200*col`, y `100 + 150*row`, 200x150, its number in white bold 48 px centred.
  Tile 7: x 40-240, y 250-400 (centre (140, 325)), rgb(240, 120, 20).
- `#tail`: y 1000-1600, rgb(250, 250, 250). Document height 1600.

Tile colours (rgb, 1-36, row by row): 200,40,40 / 40,160,60 / 40,70,200 / 220,180,30 / 150,50,170 / 30,170,170 /
240,120,20 / 90,90,90 / 120,200,60 / 200,60,140 / 60,120,160 / 170,140,90 / 100,30,30 / 30,100,40 / 30,40,110 /
160,160,40 / 90,30,110 / 20,110,110 / 250,170,120 / 170,170,170 / 180,240,140 / 240,150,200 / 140,190,230 /
220,210,170 / 130,70,20 / 70,130,20 / 20,70,130 / 230,90,90 / 90,230,90 / 90,90,230 / 60,60,20 / 20,60,60 /
60,20,60 / 250,230,60 / 60,230,250 / 250,60,230.

Test hooks (read or called by the test through Playwright, never by the driver or the model):

- `window.__clicked`: the numbers of the tiles clicked, in order (group B).
- `window.__input`: a count of `pointerdown`/`mousedown`/`keydown`/`wheel`/`touchstart` events (A1: a region
  screenshot dispatches no input).
- `window.t0204.insertAbove(px)`: inserts a `px`-high rgb(200, 30, 30) block right above `#grid` (a real layout
  shift of the grid and the tail). `insertBelow(px)` inserts after `#tail` (nothing visible moves);
  `insertBelowGrid(px)` inserts between the grid and the tail. `transformGrid(px)` moves the grid with a CSS
  `transform` (no layout-shift entry, A2 / threat model).
- `?lazy=PX`: on the first `scroll` event, 50 ms later, `insertAbove(PX)` (the lazy-load-after-scroll case).

Expected shift for `?lazy=300` after a 200 px scroll (by hand): the grid's visible part moves from (40, 0, 1200, 800)
to (40, 200, 1200, 600) in viewport CSS px. Reading R1 (Chromium 156 headless) reported exactly that pair.

## caret.html (group B, B9.4)

Hand-written. A text input `#q` at x 340-940, y 380-420 (1 px rgb(120, 120, 120) border, 4 px horizontal padding,
24 px sans-serif, white), above it a static label. No timers, no animation: the only change over time is Chromium's own
caret blink while `#q` is focused (the test focuses it). Playwright's `screenshot()` hides the caret by default
(`caret="hide"`), so that blink never reaches a capture (reading, 2026-10-29, Chromium headless: ten region captures
over 1.2 s identical). `?drawn=1` adds `#drawn`, a page-drawn 2x22 px black caret at x 345-347, y 389-411, toggled
every 530 ms (a `visibility` change: no layout shift), so the blink does reach the captures. Test hooks: `window.__keys` (keydown count),
`window.__clicks` (click count).

## animated.html (group B, B9.4/B9.5; group C may reuse it)

Hand-written. Background rgb(245, 245, 245); two static blocks: `#a` x 100-300, y 100-250 rgb(40, 70, 200) and
`#b` x 980-1180, y 550-700 rgb(240, 120, 20).

- `?spinner=1`: `#spin`, a 60x12 rgb(20, 20, 20) bar at x 810-870, y 394-406 rotating with a CSS animation (0.6 s per
  turn; a `transform`, so no layout shift). Its nearest pixel is 170 px right of (640, 400) and its centre 200 px.
- `?block=1`: `#noise`, a 300x250 canvas at x 490-790, y 275-525 (centred on (640, 400)) repainted with fresh grey
  noise every 100 ms from a fixed LCG seed (12345); two paints never match.

Test hook: `window.__clicks` (click count).

## blocks.html (ruling A-R1; the reviewer's probe, 2026-10-29)

A 10 px spacer `#top`, then 12 sibling `position: relative` blocks, each 60 px high: block *i* (0-11) at y
`10 + 60 i`, x 0 to `300 + 20 i`, rgb(`20 i`, 80, 80), its number as text. `window.t0204.insertTop(px)` inserts a
`px`-high empty block after `#top`, so every block moves down by `px`. Chromium 156 names at most 5 sources per
layout-shift entry: for `insertTop(100)` it reported blocks 6-10 only, so block 2 ((150, 160) -> (150, 260)) is
covered only if a 5-source entry is read as the whole viewport.

## animated.html `?steps=1` (group C, C5)

Added by the T-0204 C test-author, 2026-10-30. `#steps`, a 300x250 block at x 480-780, y 280-530 (left/top on the
40 px multiples of the settle's 20 px grid cells), whose background grey steps 30, 60, ..., 240 and round again every
100 ms (a `background` change: no layout shift). Any two states within the cycle differ by at least 30 grey levels, so
every cell the block covers (even a half-covered one) differs by more than `STABLE_CELL_DELTA = 6` whenever a repaint
fell between two shots. `?block=1` (B's per-pixel noise) is unchanged.

## animated.html `?carousel=1` (group C, ruling C-R1)

Added by the T-0204 C test-author, 2026-10-30. `#car`, a 300x250 window at x 480-780, y 280-530 (`overflow: hidden`)
over a strip of eight 300x250 slides of greys 30, 60, ..., 240; every 100 ms the strip jumps one slide left by a CSS
`transform` (no transition), so the window shows the next grey. A transform reports no layout-shift entry (A's R1), so
the settle's shift check (C-R1) does not hold it and the animation rule ignores it.

## lazy_scroll.html (group C, C4)

Hand-written by the T-0204 C test-author, 2026-10-30. `overflow-anchor: none`. `#top` y 0-600 (an 80 px rgb(40, 40,
40) top border, then rgb(250, 250, 250)); `#ins` (empty) at y 600; `#bands`: ten 120 px bands alternating rgb(60, 60,
60) and rgb(180, 180, 180). On the first `scroll` event, every 80 ms five times (at 80..400 ms), a 120 px block is
inserted at the top of `#ins`, colours in order rgb(200, 30, 30), rgb(30, 160, 60), rgb(40, 70, 200), rgb(220, 180,
30), rgb(150, 50, 170). After the executor's `scroll down` (600 px: 3/4 of 800) `#ins` is at the viewport top, so the
newest block is at viewport y 0-120: the final layout's colour at (640, 60) is rgb(150, 50, 170) and at (640, 660)
rgb(60, 60, 60) (the five blocks fill viewport y 0-600, then band 1, the dark one, y 600-720). `?n=N` inserts N blocks
(the colours cycling; `?n=10` inserts for 800 ms and still ends on rgb(150, 50, 170), the 10th colour; ruling C-R1's row). Test hook: `window.__inserted` (the count).

## noise.html (group C, C6)

Hand-written by the T-0204 C test-author, 2026-10-30. A 320x200 canvas stretched to 1280x800 (`image-rendering:
pixelated`, 4x4 CSS px per sample), repainted with fresh grey noise every 50 ms from LCG seed 20404; the top 10 sample
rows (40 CSS px) stay black. Each 20x20 grid cell averages 25 samples, so between two shots a random ~3/4 of the cells
differ by more than 6 and a different random subset each pair: never stable under C2's rule. Test hook:
`window.__clicks`.
