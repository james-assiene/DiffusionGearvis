# tests/fixtures/a2ui

| file | provenance | licence |
|---|---|---|
| `quiz_example.jsonl` | Hand-written on 2026-09-12 as the U8 "quiz + flashcards from the chapter on screen" surface (docs/11 §3, docs/12 §1): 3 A2UI v0.9.1 messages (createSurface, updateComponents with 38 components, updateDataModel). Validated against the vendored `server_to_client.json` / basic `catalog.json` (a2ui-agent-sdk 0.6.0). Concatenated pretty-printed JSON, not strict JSONL, on purpose: that is what models emit. | Apache-2.0 (this repo) |

Schemas themselves live in `local_astra/ui/schemas/a2ui_v0_9_1/` (from the `a2ui-agent-sdk` 0.6.0
wheel, `a2ui/assets/0.9.1/`; Apache-2.0). SHA-256 at vendoring time:

- catalog.json `8cc94d0a482e67048f9fc989964ca5da56fe42f531d919315a508989fb22e13e`
- common_types.json `ac79788e95e5bdf0a39808953593a53c1bc9fcdcdb55480f4610613c6591e94c`
- server_to_client.json `ed1ee11b4f1e94dd08ebe0d57bedb436cb452cc546d26cae407858ae477af288`

## flight_booking_stutter_20260917.jsonl

The two `ui` canvases of turn `t2` from the human's desk session `logs/sessions/20260917-165104-374798.jsonl` (2026-09-17 16:51,
commit 9f3e426 at session start, engine `vllm/vllm-openai:v0.29.0`, `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev 2ea8372, canvas 256,
48 steps, thinking on, no seed possible), verbatim from the `llm.commit` records. The request was "a flight booking UI"; the model put a
complete A2UI surface in the `ui` segment (T-0055 held) but the JSON carries diffusion artefacts: `{"{"id"` (a doubled `{"`) three
times and `"]}` where `"}` belongs four times. `UiGate.repair` split the one `updateComponents` message into four fragments, three of
which are not A2UI messages, so validation failed with `'createSurface' is a required property` ×3 — reported in `schema_errors`, which
the session's `ui.gate` record does not log (`problems: []`), so the pane showed nothing and the assistant said "I have created a flight
booking interface for you". Origin of T-0062.

## t0108_canvases.jsonl

T-0108 criterion 13 (i): 17 real `ui` canvases, verbatim, from the Q12 positive trials of two stored runs: the T-0103 run
`logs/t0103-c3-confirm-6720e56/trials.jsonl` and the T-0098c run `logs/t0098c-prompt-ab-3ac00a1/trials.jsonl` (both read
only; regime lean, vLLM v0.29.0, `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev 2ea8372, canvas 256, 48 steps, thinking on, no
seed, typed text, no Session). Classes: duplicated_opener 2, missing_separator 2 (one trial, canvases 0 and 1: the only one in
either run), wrong_closer_drop 2, wrong_closer_insert 3, stray_quote 2, duplicate_id 1, missing_component_key 1,
inline_children 1, header_cut 2, non_a2ui_json (D3) 1. Each row carries the run, request id, arm, case, ui index, canvas_idx, the
trial's earlier ui texts and `baseline_fresh_gate`: the outcome of a fresh `UiGate()` at 14cde78 (surface, ok, repaired,
problems, messages). Selected by the test-author on 2026-10-16 and built with an inline `uv run python -c` at d525329, whose
`local_astra/` is identical to 14cde78 (CPU, Python 3.11.16, json_repair 0.63.4). The golden repaired texts are not in this
file; they are hand-written in `tests/unit/test_bracket_repair.py`.

## t0108_baseline_outcomes.json

T-0108 criterion 13 (ii): per request, one bool per `ui` commit (did the gate return a surface), for every Q12 positive trial
(case `expect_surface` true in `tests/eval/surface_placement_cases.jsonl`, kind `q12`) of the same two runs: 320 trials of
6720e56 (A0 160, C3 160) and 140 of 3ac00a1 (A0, C3, C4, C5, D1, D3, D4, 20 each); `[]` when a trial had no ui commit. Reading:
a fresh `UiGate()` per trial at 14cde78, every ui commit in order, `final=False`, turn id = request id, the commit's
canvas_idx (the planner's *Evidence* reading of T-0108). Built in the same command as `t0108_canvases.jsonl`; it reproduces the
task's baseline table exactly (6720e56 A0 last call OK 103/153, C3 80/128; 3ac00a1 pooled 58/88; D3 0 emitted of 16).
The two `source` paths: identifiers replaced 2026-10-25 (T-0157) (the home directory reads `/home/user`); no other byte changed.

## t0108b_components.jsonl

T-0108b (*Tests to write*, per-rule goldens): 55 rows of real A2UI component lists, verbatim as the new component stage
receives them (the strict parse of the `ui` canvas, or, when that fails, `repair_structure` + the unguided json_repair
fix grounded against the raw canvas, before salvage), from the Q12 positive trials of three stored runs, read only:
`t0108a-launch/logs/t0108a-ui-fields-ab-e6c3a81/trials.jsonl` (T-0108a; A0, K1, K2), `t0103-launch/logs/t0103-c3-confirm-6720e56/trials.jsonl`
(T-0103; A0, C3) and `t0098c-launch/logs/t0098c-prompt-ab-3ac00a1/trials.jsonl` (T-0098c; 7 arms); paths under
`.claude/worktrees/`. Run regime: lean, vLLM v0.29.0, `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev 2ea8372, canvas 256,
48 steps, thinking on, no seed. Each row carries `row_id`, `class`, the run, request id, arm, case, ui index, canvas_idx,
`path` (strict / unguided), the message index, `component_index` (the target; `null` for `all_valid`) and the whole
`components` list. Classes (the test-author's classifier of the target's defect, from the rule definitions; up to 3 rows
each, one per run/arm/case/path): button_child_inline, button_children_id, button_children_inline, button_text_only,
choice_no_options, choice_no_value, choice_string_options, choice_string_value, container_inline_children,
datetime_no_value, list_items, textfield_extra_keys, textfield_label_and_placeholder, textfield_placeholder,
textfield_text 3 each; button_children_two 2; button_text_extra 1; choice_mixed_options 1; all_valid 6 (last canvases
with at least 4 components, all catalog-valid). Built by the test-author on 2026-10-17 with an inline `uv run python -c`
at 08c7776 (branch base; `local_astra/` identical to 749312d), CPU, Python 3.11.16, json_repair 0.63.4. The golden
outputs are not in this file; they are hand-written in `tests/unit/test_component_repair.py`.

## t0108b_baseline_outcomes.json

T-0108b criteria 10-11 baseline: for every trial of the same three runs that is a Q12 positive (kind `q12`, case
`expect_surface` true in `tests/eval/surface_placement_cases.jsonl`) or has at least one `ui` commit: `kind`, `case_id`,
`arm`, `positive`, `outcomes` (one bool per ui commit: did the gate return a surface), `pruned_last` and `complete` (the
T-0108a criterion 4 reading: a fresh `UiGate()` per trial, every ui commit in order, `final=False`, turn id = request
id, canvas_idx = the commit's; pruned = invalid ids on `_repair(last).messages` | `_repair(last).dropped`; complete =
last call OK and nothing pruned). Each run also records its source path and the sha256 of its `trials.jsonl`. Built in
the same session at 08c7776 with the gate as at 749312d. It reproduces the task's baseline exactly: complete e6c3a81 A0
54, K1 64, K2 66, 6720e56 A0 63, C3 32 (of 160); last-OK 142 / 143 / 137 / 149 / 114; shown 146 / 152 / 139 / 153 /
124; calls OK 357 / 332 / 324 / 340 / 295; 3ac00a1 D3 0 surfaces of 16 calls; one non-positive trial with a ui commit
(an e6c3a81 K2 tool trial, 2 surfaces).

## desk_20261020_card_children.jsonl

T-0127 criterion 8: 34 records copied verbatim (byte for byte, one line each, in log order) from the human's desk
session `logs/sessions/20261020-165855-1581291.jsonl` (gitignored, read only; sha256 of the whole log
`a85e218fcf476bd120f750fd53990e18b40bc464f045a661e8d52bf0756da69e`, 18 436 lines). For each of the six card turns:
every `llm.commit` with segment `ui`, every `ui.gate` record, and the final (`final: true`) `astra.ui.surface`
`wire.out` record. Seqs:

| turn | request | `ui` commits | `ui.gate` | final surface |
|---|---|---|---|---|
| t3 | omelette | 3250, 3274 | 3253, 3277 | 3362 |
| t5 | iPhone vs Pixel | 6697, 6726, 6772 | 6700, 6728, 6775 | 6900 |
| t6 | bike tyre | 7692, 7714 | 7695, 7717 | 7808 |
| t8 | spreadsheet | 9796, 9806 | 9798, 9808 | 9867 |
| t9 | shopping list | 10625, 10652 | 10628, 10655 | 10745 |
| t10 | "Show me a card with the 10 largest countries by area and their populations." (typed, seq 11541) | 11654, 11668, 11684 | 11657, 11670, 11687 | 11870 |

The sixth case (criterion 8(d)) is `t10`: a tool-loop turn (`google_search` dry run in `t10-gen`, the card in
`t10-gen-s2`, whose `ui` commits start at canvas_idx 1). Its final surface also carries a third message re-sending
`l`, `r1`, `r9`, the same `_restore_references` overwrite as t8's `r3` (criterion 7).

Regime: `log.started` carries only `pid` and `path`; the regime is read from the session's `turn.record` lines:
profile lean, canvas 256, 48 denoising steps, thinking on, `vllm/vllm-openai:v0.29.0`,
`nvidia/diffusiongemma-26B-A4B-it-NVFP4` (revision as pinned in `services/versions.lock`; not logged), STT
`kyutai/stt-1b-en_fr`, TTS `pocket-tts@3.1.0`, headset, assists pixels only, tools `native:bbfc53ccdd80:dry_run:loop8`,
seed null, git commit ef4c68d at session start, typed input. Copied by the T-0127 test-author on 2026-10-20 with
Write/Edit from the log lines; verbatim-ness checked by comparing every fixture line with the log line of the same
seq (34/34 equal). Licence: Apache-2.0 (this repo); the content is the model's output in the human's own session.

## genui_census_20261027.jsonl

T-0164 group A (the renderer core's node rows, `tests/hud/ui_render.test.mjs` and `tests/hud/ui_engine.test.mjs`). Built by the
T-0164 A test-author on 2026-10-27 with a generator kept in the session scratchpad (`t0164a_build_census_fixture.py`, sha256
`40da78de5cf327be44c7373d78a4e8552cc63cc74b5853a7191d42e504f32a59`; not committed: the task's owned files do not include a script),
run as `uv run python t0164a_build_census_fixture.py <worktree> <out>` at commit `06a7acf7` (main fc684b82 + the T-0164 task file;
`local_astra/` identical to fc684b82), CPU only, deterministic. sha256 of the file:
`441660e2ee980efa91b0a830bc6eb01d8f505d6266d1847c44bf83cf3176427c`.

Line 1 is a header: `gate_commit`, a committed copy of `ui_instructions.ADVERTISED_ORDER` (the node rows cross-check it against
the Python source), the census population and the oracle's rules. Every further line is one surface:
`{key, source, surface_id, canvas_idxs, canvases: [[a2ui messages]...], send_data_model, expected_texts, kinds}`. Every surface is
the stored `ui` commits pushed through a **fresh `UiGate()` per turn** (as the T-0164 option-A mock's `gate_payloads.py` does),
keeping every emitted `astra.ui.surface` (all canvases, `messages` verbatim as `model_dump_json(by_alias=True, exclude_none=True)`).

- `desk-t3`, `desk-t5`, `desk-t6`, `desk-t8`, `desk-t9`, `desk-t10`: the `ui` commits of `desk_20261020_card_children.jsonl` above
  (regime as stated there).
- `trial-omelette` (`t0108a-r11-omelette_steps-K2`) and `trial-flight` (`t0108a-r2-flight_booking-A0`): the run
  `t0108a-ui-fields-ab-e6c3a81` (`.claude/worktrees/t0108a-launch/logs/…/trials.jsonl`, commit e6c3a81; regime lean, vLLM v0.29.0,
  `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev 2ea8372, canvas 256, 48 steps, thinking on, no seed, typed text, no Session).
- `quiz`: `quiz_example.jsonl` above, gated as turn `t1`, one final canvas (so its surface id is `turn-t1`, `sendDataModel: true`).
  It holds **no CheckBox** (the task expected one); CheckBox, Modal and Icon rows use hand-written surfaces in the test modules.
- `census-01` … `census-04`: a greedy cover of the GPU-trial census, the 801 gated surfaces of the 17 distinct `trials.jsonl`
  files under `logs/` and `.claude/worktrees/*/logs/` of the main checkout (the 799 of the option-A census plus 2). Feature = a
  component kind, or (kind, `variant`/`justify`/`align`/`direction`/`axis`/`displayStyle`/`fit` value), or (kind, prop, value
  shape: literal type / `path` / `call` / `event` / list). 34 features in the census; the greedy cover starts from what the 8 named
  payloads and the quiz already hold, and four census surfaces cover the rest (`census_features_uncovered: []`). No Image, Video, AudioPlayer, FunctionCall or
  template ChildList occurs in the census.

`expected_texts` is computed in Python from the accumulated tree and data model, never by the JS under test: a walk from `root`
with `local_astra.ui.a2ui_schema.component_children` (the gate's own reference helper), a dangling id or an id already on the
root-to-leaf path rendering nothing; per kind, in tree order: `Text.text`; CheckBox/Slider `label`; TextField/DateTimeInput
`label` then `value`; ChoicePicker `label` then option labels; Tabs every title then the **first** tab's child only; Modal its
trigger only (H-T164-5: content closed); Image/Video/AudioPlayer/unknown kinds/template lists nothing; a `{path}` resolved
against the data model and counted only when it is a string; empty strings skipped. Licence: Apache-2.0 (this repo); the
content is model output from the runs named.

## cards_v2_exemplars.jsonl

T-0184 group A (A3; also B's renderer rows and B10's screenshots): 19 hand-written A2UI v0.9.1 surfaces under the custom
catalog `gearvis:cards/v2`, one per line, `{key, board, messages}` (`messages` = `createSurface` + one
`updateComponents`, strict JSONL). Written by the T-0184 A test-author on 2026-10-28 with Write, from A2's table in
`docs/tasks/T-0184.md`, before the catalog existed; no model output, no generator. sha256 of the file:
`41fd48d7d2c3fa8d36b6ddef7379fed31e4e6b8eb8fab3eb5b5ddf5b41b4e482`.

- Per kind: `stat-minimal`, `stat-icons` (all nine icons), `stattiles-storage`, `comparison-laptops` (string and toned
  cells, a badge), `bars-titled`, `steps-omelette`, `source-only`, `actions-three`, and `mixed-basic-on-v2` (Card,
  Column, Row, Text, Divider beside Stat and Steps on a v2 surface).
- B7's edge cases: `bars-all-zero`, `bars-single`, `bars-huge` (1e12), `comparison-highlight-out-of-range`
  (`highlight: 7` with two columns), `stattiles-one`, `stattiles-four-wide` (12-digit values), `stattiles-percent-bounds`
  (0 and 100).
- The three approved mockup boards (James, 2026-10-28) rebuilt as v2 payloads (`board` = `main` / `compare` /
  `chart`): Main is a `Stat` "PARIS · NOW" "[17°]" "Partly cloudy" `partlyCloudy`, `StatTiles` Friday / Saturday
  (highlighted) / Sunday and a `Source`; Compare is a `Comparison` of "Deluxe hardcover" (badge "Recommended",
  highlighted) against "Regular volumes" with rows Volumes, Price per book, Format and Best for (toned), then `Actions`;
  Chart is `Bars` Everest 8849, K2 8611, Kangchenjunga 8586, Mont Blanc 4806, unit "m", and a `Source`
  "en.wikipedia.org". Bracketed values (`[19°]`, `[€45]`, …) are placeholders, not the mockup's figures; the board
  contents named in the task are pinned in `tests/unit/test_t0184_catalog.py`.

Licence: Apache-2.0 (this repo).

## q12_trials_sample.jsonl

T-0184 group C (C1's differential: the gate after C against `tests/unit/_frozen_ui_gate_t0184.py`, BASE 3cb090f1). 17 stored
Q12 positive trials, every `ui` commit of each verbatim (`text`, `canvas_idx`, `thought_open`), in commit order: 30 commits,
20 of which a fresh BASE `UiGate()` per trial (`final=False`, turn id = request id) turns into a surface, 14 complete. Line 1
is a header naming the three source runs and the sha256 of each `trials.jsonl` (read only, paths under `.claude/worktrees/` of
the main checkout): `t0098c-prompt-ab-3ac00a1` (T-0098c; 7 arms), `t0103-c3-confirm-6720e56` (T-0103; A0, C3) and
`t0108a-ui-fields-ab-e6c3a81` (T-0108a; A0, K1, K2). Run regime: lean, vLLM v0.29.0, `nvidia/diffusiongemma-26B-A4B-it-NVFP4`
rev 2ea8372, canvas 256, 48 steps, thinking on, no seed, typed text, no Session. Every other line is
`{run, request_id, case_id, arm, ui_commits}`.

Selection (the test-author, 2026-10-28, CPU, Python 3.11.16, at 3cb090f1): of the 826 positive trials (case `expect_surface`
true in `tests/eval/surface_placement_cases.jsonl`) with at least one `ui` commit, a greedy set cover, smallest text first,
over (a) the BASE gate's per-call features (surface / no surface, structural, component and fixer edits, lost, complete,
incomplete, unreachable, resample, id-less, pruned reference, reserved `__gate_` ids, a `thought_open` ui commit), (b) every
(run, case) and (c) every (run, arm). No negative Q12 trial (`capital_of_france`, `spider_legs`, `cloud_joke`) has a `ui`
commit in any of the three runs, so none is in the sample. Scripts kept in the session scratchpad, not committed (the task
owns no script): `features.py` (sha256 `437bb05523c6246f8c6e8f701b833c30deb7233cb3aeaa62b0818cb7f7bca82a`), `cover.py`
(`635214e68e79968cc623e3f37b721d24df80ed8ef674dd9dd2a84b0f95f649e4`) and `build_q12.py`
(`655eb07ff2bbee8f80e4582b1320156667be421c4253166b706b0518d51e0436`), run as `uv run python build_q12.py <main checkout>
chosen.json <out>`. sha256 of the file: `f8ed38de403a1c0934954d5a1349d7dfd3668092b6a619d76ce7186f4f9cef38`. Licence:
Apache-2.0 (this repo); the content is model output from the runs named.
