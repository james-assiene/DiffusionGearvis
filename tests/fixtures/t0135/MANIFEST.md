# tests/fixtures/t0135

## thought_tails.jsonl

- **What.** 59 requests copied verbatim (text unchanged, fields reduced) from the T-0132 run, one JSON object per line,
  each tagged with a `class`:
  - `firing` (29): every `generate`/`tool_step` request that kept no `tool` commit, whose last non-tool commit is an open
    `thought` commit, and whose final open thought block ends in a complete tagged call (T-0134 D-5 as re-derived in
    T-0135: S3 18, S1 7, S5 4). Each carries `expect`: the tail and its `live.classify` verdict at f3712c8 (28 `valid`;
    1 `repairable`, `20261022-210810-1975518.jsonl` `t1-gen-s2`, "text after call", the doubled `<tool_call|>`).
  - `open_no_call` (3): the same open-thought ending with no call tag (two repetition collapses, one answer left inside
    the thought).
  - `pass2` (10): ReGround pass-2 (`reground`) responses of the firing shape (out of scope for T-0135, counted only).
  - `reground_other` (2), `closed` (6), `answer_call_prose` (6: an answer-channel `tool` commit beside non-tool text
    naming a v1 tool) and `other_phase` (3: one `prefill`, both `tool_repair` requests): negatives.
- **Fields.** Per row: `source` (the session file name under `sessions/`), the request's log `seq`, `turn_id`, `phase`,
  `request_id`, `attempt_id`, `step`, `call_id`; `commits` (each commit's `seq`, `canvas_idx`, `segment`, `text`,
  `thought_open`); `trial` (the trial's `task`, `repeat`, `success`, `tool_loop_end` from `trials.jsonl`); `expect`
  (`firing` rows only). No prompts, screenshots or URLs.
- **Source.** `logs/t0132-crop-budget-ab-2ccff41/` in the main checkout (untracked): `trials.jsonl` (sha256
  `3baf06102d33f2f9…`) and `sessions/*.jsonl`, the T-0132 run at 2ccff41 on 2026-10-22 (regime: docs/tasks/T-0135.md
  "Regime": profile lean, CL 256, 48 steps, thinking on, vLLM 0.29.0, NVFP4 diffusiongemma-26B-A4B rev 2ea8372; tools
  native bbfc53ccdd80, live, repair step 1 only, loop 8; 1120 visual tokens, ReGround off / on-1120 / on-280, n = 100
  per arm, order seed 0, concurrency 1, engine unseeded; typed, fake STT/TTS). Read-only.
- **Generator.** `tests/fixtures/t0135/build_thought_tails.py <out>` (stdlib plus `local_astra.toolcall.live` for the
  expected verdicts), written 2026-10-22 by the T-0135 test-author. Its selection is the test-author's own census,
  independent of `toolcall/thought_call.py`; it asserts the planner's census (860 eligible, 29 firing by task, 28/1, the
  named repairable, 3 open no-call, 10 pass-2) before writing. Re-running it reproduces this file byte for byte (sha256
  `ae629cc9f9aad6c7…`).
- **Used by.** `tests/unit/test_t0135_replay.py` (A2 on a run directory rebuilt from these rows) and
  `tests/unit/test_t0135_thought_call.py` (the `…-203318-…` reopened-channel row of `test_recover_table`).
- **Licence.** Project-internal measurement data.
