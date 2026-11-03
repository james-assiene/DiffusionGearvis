# Tool-call stress set (T-0099 criterion 1)

30 pre-registered, honest requests whose natural answer is **out of the v1 tool schema**, used by arm S of
`scripts/gpu/t0099_repair_live.py` to draw real invalid native drafts out of the model and measure the live
H-T84-2 repair on them. Not an eval of accuracy and not a production invalid-rate estimate (that bound is 0/170).

| category | n | what is asked | ground truth |
|---|---|---|---|
| K-in | 6 | an in-enum key in another spelling ("hit Return", "press Esc", ...) | `press_key` with the enum combo |
| K-out | 6 | a key not in the `press_key` enum (F5, Ctrl+S, Ctrl+Shift+T, Alt+F4, Ctrl+F, Ctrl+Plus) | none; expected: no executed action (`out_combo` records the key) |
| S-extra | 4 | a scroll with an amount or a target | `scroll` direction |
| T-hard | 6 | `type_at` text with quotes, backslash, braces, newline, accents+emoji, >= 400 chars | bbox, text, `press_enter` (`score.exact`) |
| C-pixel | 4 | "click at pixel (x, y)" with x > 999 px on the 1280 x 800 frame | bbox of the element at that pixel |
| U-under | 4 | "type X into the field", no Enter instruction | rescued = valid, in bbox, text equal; `press_enter` reported, not scored (`gt.press_enter` is `null`) |

Screens and targets come only from the **dev** split of `tests/eval/toolcall/` (screens `d01`-`d08`); the case
names the dev case it reuses in `source_case`, and the bbox is copied from the pinned v1 `cases.jsonl`. No user text
repeats a v1 case. The C-pixel normalisation is computed by hand in the docstring of `cases_src.py`.

    uv run python -m tests.eval.toolcall_stress.cases_src          # regenerate cases.jsonl + MANIFEST.json
    uv run python -m tests.eval.toolcall_stress.cases_src --check  # exit 1 when either is stale

`MANIFEST.json` pins the sha256 of `cases.jsonl`, of the generator and of the v1 set it reads. Its `approval` field
records the human's approval of the case set (H-T99a); it is `null` until then, and the integrity rows
(`tests/unit/test_toolcall_stress_set_integrity.py`) fail while it is. Licence: project-original.
