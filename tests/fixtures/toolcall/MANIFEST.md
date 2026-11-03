# tests/fixtures/toolcall

## native_t4_eval.json

- **What.** What the T-0084e/T-0084f native eval sent for the cell `native/T4/named/perstep_mar+seed`:
  - per v1 tool: the automaton's `automaton_sha256`, `n_states` and `bucket` (identical on every row of both artifacts; the
    generator checks this);
  - one seeded row per tool (the first dev case naming it): the pass-1 `draft_call_text`, the first stop id of its pass-1
    `token_ids`, the stored `seed_ids` and the stored `knobs` (without `record_tag`);
  - the regime's `seed_draft` block, `tools_sha256`, `compiler_version`, `tokenizer_sha256`, `model_revision`, `whitespace`
    and the `perstep_mar` calibration sha.
- **Source.** The stored artifacts, read-only, in the main checkout's `logs/` (untracked):
  - T-0084e dev: `logs/t0084-native-dev-794161d/20261011T022440.808068Z-toolcall-dev.jsonl`, sha256 `06a71fc7…`;
  - T-0084f test: `logs/t0084-native-test-794161d/20261011T031521.606364Z-toolcall-test.jsonl`, sha256 `14728634…`.
- **Regime.** vLLM v0.29.0, nvidia/diffusiongemma-26B-A4B-it-NVFP4 rev 2ea8372, tokenizer `cc8d3a0c…`, tools_v1 `bbfc53cc…`,
  compiler `dgfa_compile@27c7d0d+local_astra.T-0063;outlines-core==0.2.14`, whitespace `json`, CL 256, calibration
  `4084026c…` (entropy bound 1.0, confidence threshold 0.005), seed noise 0.0, start step 24.
- **Generator.** `tests/fixtures/toolcall/build_native_t4_eval.py` (stdlib only). Written 2026-10-12 by the T-0091
  test-author. The cd group (`tests/cd/unit/test_toolcall_live_parity.py`) re-derives every value from the artifacts and from
  `compile_toolcall`; at bd2c124 the current compile reproduces the five stored shas.
- **Used by.** `tests/unit/test_toolcall_live.py` (the payload file's identity and the seeded request, in the fast suite, without
  the cd stack).
- **Licence.** Project-internal measurement data.

## t0134_t0132_answers.json

- **What.** From every one of the 300 T-0132 trials: the 60 S4 trials' arm, repeat, final cart, voiced sources and logged
  T-0111g `score` verdict; the 240 S1-S3/S5 trials' task, arm, repeat, final `__state` (an index into 14 distinct states)
  and logged verdict. Nothing else (no thoughts, no URLs).
- **Source.** `logs/t0132-crop-budget-ab-2ccff41/trials.jsonl` in the main checkout (untracked; sha256 `3baf0610…`), the
  T-0132 run at 2ccff41 on 2026-10-22 (regime: docs/tasks/T-0132.md; 1120 visual tokens, ReGround off / on-1120 / on-280,
  n = 100 per arm, seed 0). Read-only.
- **Generator.** `tests/fixtures/toolcall/build_t0134_t0132_answers.py <out>` (stdlib only), written 2026-10-22 by the T-0134
  test-author; re-running it reproduces this file byte for byte (sha256 `e47b5efa…`).
- **Used by.** `tests/unit/test_t0134_ab_harness.py::test_score_v2_s4` (H-T134-3: every logged enamel-mug answer fails
  `score_v2`; S1-S3/S5 verdicts equal T-0111g's).
- **Licence.** Project-internal measurement data.
