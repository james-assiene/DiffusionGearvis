# Tool-call eval set v1 (T-0080b criterion 1)

140 cases: 120 tool cases (30 google_search, 30 click_at, 30 type_at, 15 scroll, 15 press_key) and 20 no-tool cases.
Split fixed in `cases.jsonl` (`split`): dev 40 + 8 no-tool, test 80 + 12 no-tool, stratified 1:2 per tool; no screen is
shared between splits. `MANIFEST.json` pins the sha256 of the cases file, the split, every screen file and the generator.

Provenance: synthetic screens (human decision Q4), 24 self-contained HTML pages (`screens.py`, inline CSS, no external
resource) rendered offline by headless Chrome 147.0.7727.101 over the DevTools Protocol from `file://`, network emulated
offline and every non-file URL blocked; viewport 1280 x 800, device scale 1, JPEG quality 85. Bboxes are read from the
DOM (`getBoundingClientRect` of `[data-t]` elements) and normalised to the 0-999 grid (rule in `build.py`). Cases are
authored in `cases_src.py` by the test-author; licence: project-original, no third-party content.

    uv run python -m tests.eval.toolcall.build          # regenerate (rewrites screens/, cases.jsonl, MANIFEST.json)
    uv run python -m tests.eval.toolcall.build --check  # re-render in a temp dir and compare bboxes

Two renders on this machine gave byte-identical JPEGs (2026-10-08); other machines/fonts may not, so the committed
images are the set. Integrity rows: `tests/unit/test_toolcall_set_integrity.py` (fast suite; ruling R-T46). Reviewed by the human at H-T80.
