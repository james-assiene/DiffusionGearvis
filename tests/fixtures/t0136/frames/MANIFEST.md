# tests/fixtures/t0136/frames (and ../labels.jsonl)

T-0136 group A2/A2b calibration fixtures (docs/tasks/T-0136.md, A2-5, amendment A1c/A2b: A2b-1, A2b-2; ruling
H-T136-1).

## Provenance

- **Corpus:** `logs/t0136-outcome-corpus-610f36f/` (gitignored), the A1c-5 rebuild by `scripts/build_outcome_corpus.py`
  at 610f36f from `t0151-overview-frame-ab-4471255` and `t0134-feedback-ab-e59aa69` with `--probes` (both probe pages,
  `outcomes.html` then `outcomes_b.html`); manifest sha256
  `3c8c1c6ff5502b4d71c3e64b25799c7d64239f4c7df5a0c83478433fda48fbe0` (the task file's Rulings (A1c/A2b)). 442 distinct
  pairs, 241 frames, 236 jitter clusters (`frame_groups.json`). Earlier corpora: cfd4127 (182 pairs), 20104cc (274
  pairs, manifest sha256 `0b12243295b40a0820383fc4bd9a41942f208b18d7be7ce025681e951cec0f6e`).
- **Recording conditions:** Playwright 1.64.0, headless Chromium 156, 1280x800, DSF 1, loopback pages,
  `runtime.screenshot` JPEGs at the executor's quality. CPU only.
- **These frames:** 34 JPEGs copied byte-for-byte from the 610f36f corpus `frames/` (file name = sha256 of the bytes;
  1 962 484 bytes in all, under the 2 MB cap). They are the `before`/`after` frames of the 27 records in
  `../labels.jsonl` with `"subset": true`: 18 kept from the A2 subset (identical frames, hover tint, un-hover outside
  the near box, the off-screen cart change (H-T136-4), the caret probe, checkbox, typed text, focus ring, the badge,
  toasts, the status pill, three navigations; two A2 pairs dropped for size), plus the F3 landmines of A2b: n-hover-3's
  underline, the c-label-3 Start -> Stop relabel, two page-b long-link underlines (540 and 560 px), three C-SMALL pairs
  (qty, count, red hint), one C-COLOUR pair (star) and one N-PRE pair (un-hover then a no-effect click). Every subset
  group is `seen` and in dev (A2b-2).
- **Licence:** rendered from this project's own loopback pages and test fixtures; project-internal.

## labels.jsonl

One record per distinct pair of the 610f36f corpus (442):

- **260 carried** 20104cc ids keep their label and region from the A2 labels at ad8268d (pinned by
  `test_carried_labels_unchanged`); 14 20104cc ids vanished (JPEG jitter changed their bytes) and are dropped.
- **182 new** pairs, labelled **blind to any detector** by the A2b test-author on 2026-10-25 from the A1-6 contact
  sheets (before / after / |delta| x4, click point and near box drawn), tiled six to a mosaic captioned with the pair id
  only: 176 page-b probes and 6 jitter-drifted replay pairs. Each replay pair was labelled from its own sheet and then
  checked against the 20104cc pair with jitter-equivalent frames and the same points (all 6 agree; listed in
  `INHERITED` in `tests/unit/test_t0136_action_outcome.py`). The labels commit (2294f37) precedes any detector run on
  the new pairs. The 176 probe labels were compared with the probe table's `intended` class and, for `elsewhere`, the
  ninth of its `changes` rect after labelling: no disagreement.

Fields: `pair_id`; `label` (`none` / `near` / `elsewhere` / `page`); `region` (one of the 9 ninths, set exactly when
`label == "elsewhere"`); `split` (A2b-2: by group; seen groups dev; unseen groups stratified by class, ordered by
sha256 of the group string, test, dev, test, ...); `group` (`"<before cluster>:<after cluster>"`, clusters from
`frame_groups.json`, the O-A1 (7) jitter rule); `seen` (the group is jitter-equivalent to a 20104cc dev pair's frames,
or holds a subset pair); `trivial` (before and after in one cluster); `stress` (from the page-b probe table, false
otherwise); `kind` (the probe table's `kind`, or `replay`); `probe_page` (`outcomes.html`, `outcomes_b.html` or null);
`frame_kind` (`live` / `overview`); `subset`; `carried`; `point`, `prev_point` (copied from the corpus record);
`before`/`after` (sha256, subset records only); `source`; a short `note`.

Label definition (A2-3's geometry, A2-5): hover styling under a pointer (current or previous), including link
underlines, button brightness, chip tints and the stress hovers (shadow, lift, ghost border, row tint), and caret
blink are `none`; a focus ring is a change; a change intersecting the near box (+-0.25 W x +-0.10 H around the click)
is `near`; another change is `elsewhere (<ninth holding most of it>)`; `page` is a new page in the viewport -- a
navigation, or the whole viewport shifted (the overview-frame scroll). Real changes that are thin or equal in luma
(strikethrough toggles, equal-luma dot swaps) are labelled as changes (`near`); they are stress kinds, report-only.
