# wait_probe — T-0202 C's probe pages

Hand-written for T-0202 group C (the wait-tool GPU A/B), 2026-10-30. No generator, no third-party content, no external
resource (inline CSS and JS only), no clock read and no randomness. Apache-2.0, as the repository.

Served **only** by `scripts/gpu/t0202_wait_ab.py` (a fresh loopback `StartPageServer(static_dir=PROBE_DIR)` per probe
turn). Never served by `astra session`: this directory is not the start-page root (`local_astra/pages/static/`), and
no module under `local_astra/` names it.

| File | Task | What it does | Pinned content |
|---|---|---|---|
| `search.html` | W1, slow search results (Amazon-shaped) | One text input. On Enter the "Results for …" heading paints at once; the four results paint 1.5 s later (`RESULTS_DELAY_MS`), nothing changing in between, no spinner, no price on the page before. Every query gets the same results. | Target: Brightwell Electric Kettle **$34.99**. Distractors: $29.99, $41.50, $134.99. |
| `list.html` | W2, slow navigation | `/list.html` is a station page whose only link opens `/list.html?view=departures`; there the "Departures" header paints at once and the board 1.5 s after DOMContentLoaded (`BOARD_DELAY_MS`), nothing changing in between, no spinner. | Target: the 10:42 to Hollins Cross, **Platform 7**. Others: Platforms 3, 12, 17. |

Why 1.5 s (test-author reading 7, `tests/unit/test_t0202_ab_harness.py`): T-0203's follow ends at
`domcontentloaded` + two rAFs and T-0204's visual settle ends at the first matching pair of shots, never past 1.0 s
from its first, so content painted 1.5 s after a quiet gap is absent from the action's after-frame and the probe
measures the wait tool. The gated browser row pins this on these pages.
