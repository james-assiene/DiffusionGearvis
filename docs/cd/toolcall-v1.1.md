# Tool-call v1.1: the `</cc>` strip re-measured (T-0082c)

**Regime (both runs).**
- **Build and model:** commit 48313a9, a pinned detached launch worktree, clean. Profile lean (engine only).
  nvidia/diffusiongemma-26B-A4B-it-NVFP4 rev 2ea8372, vLLM v0.29.0, RTX 5090, driver 595.84.
- **Engine:** CL 256, 48 steps, gpu-mem 0.86, prefix caching on, canvas tap on. The plugin, `services/vllm.sh` and the JSON grammar
  payloads are byte-identical to T-0080c at 0dc9134.
- **Pass 1:** **stored, not re-sampled.** These are the T-0080c drafts: dev artifact `b0fd71cc…`, test `1c69a9c2…`, thinking on.
  They are re-routed with the T-0082a strip of a terminal `</cc>`.
- **Pass 2:** sent fresh, with one cache warm-up per routed case beforehand (human ruling R10: the case's pass-1 prompt, `max_tokens` 1,
  never scored).
- **Cells:** T1/named × {end_projection, perstep_mar (eb 1.0 / ct 0.005), perstep_mar_qargmax, perstep_mar+seed (noise 0, start 24,
  R2 trim)}, Williams order, concurrency 1, no sampler seed.
- **Pass-1 walls** come from the T-0080c boot; pass-2 walls come from this boot.
- **Evidence:** `logs/t0082-replay-{dev,test}-48313a9/` in the launch worktree.

**Caveat (pre-registration).** The strip was designed after seeing the T-0080c test split, so this is not a held-out confirmation. The
selection rule and its tolerances are T-0080c criterion 3, unchanged.

## Outcome: the pre-registered rule picks `cd_on_invalid:T1/named/end_projection`
**Test split** (n 80 tool + 12 no-tool; 76 routed). The chosen policy:
- exact 66/80, schema 76/76, CS 24/24;
- end-to-end p50 / p90 316.4 / 558.1 ms on tool cases;
- pass 2 is sent on 24/76 routed cases only (the others' drafts are valid after the strip).

How the rule got there:
- The best eligible exact is 66/80 (end_projection and qargmax under `cd_on_invalid`), so the bar is 62/80.
- **Both seeded arms (60/80) fall below the bar**, as do the unseeded per-step `cd_always` arms (56 and 57).
- `cd_always:end_projection` passes the accuracy filter at exactly 62/80, then drops out at the 5 % p50 window (577.2 > 332.2 ms).
- Every remaining `cd_on_invalid` arm has the same e2e p50, 316.4 ms, because the median routed case sends no pass 2. The tie goes to
  the lower p90: end_projection 558.1 vs qargmax 1229.1 vs perstep_mar 1271.4 ms.

**This differs from the provisional default** (`cd_on_invalid:T1/named/perstep_mar+seed`, H-T80c). Per T-0082c criterion 4, the
provisional default stayed until the human ruled at H-T82. **H-T82 (2026-10-10): the human switched the default to
`cd_on_invalid:T1/named/end_projection`.**

| policy (test) | exact | e2e p50 / p90 ms | pass-2 p50 ms | steps |
|---|---|---|---|---|
| draft_only' (`</cc>` stripped) | 47/80 (schema 52/76) | 247.9 / 343.6 | — | — |
| cd_on_invalid:end_projection | **66/80** | **316.4 / 558.1** | 293.7 | 9.6 |
| cd_on_invalid:perstep_mar_qargmax | 66/80 | 316.4 / 1229.1 | 811.9 | 19.5 |
| cd_on_invalid:perstep_mar | 64/80 | 316.4 / 1271.4 | 827.3 | 19.2 |
| cd_on_invalid:perstep_mar+seed | 60/80 | 316.4 / 516.3 | 236.8 | 3.9 |
| cd_always:end_projection | 62/80 | 577.2 / 724.0 | 319.9 | 9.7 |
| cd_always:perstep_mar+seed | 60/80 | 449.6 / 607.1 | 197.4 | 2.8 |

## The grammar's own value (criterion 5), over the stripped draft
- **What CD adds:** on the 24 drafts still invalid after the strip, `cd_on_invalid` adds **19** exact calls with end_projection and
  qargmax, 17 with perstep_mar and 13 with the seed, against `draft_only'`. McNemar b 19, c 0, p 3.8e-6 (Bonferroni-adjusted 5.3e-5).
  Every discordant pair is a still-invalid draft: on the 52 drafts valid after the strip, `cd_on_invalid` is identical to the draft.
- **Sentence:** over the stripped draft, constrained decoding adds 19 exact calls on 24 still-invalid drafts (p 3.8e-6, end_projection).
- **`cd_always` harm on valid drafts** (52; draft-right / CD-wrong vs the reverse):
  - perstep_mar 8/0 (p 0.008);
  - qargmax 9/0;
  - end_projection 5/1;
  - **seed 0/0**: the seeded pass keeps the valid draft in 49/52 cases.

## Seeds (criterion 6)
- **Coverage:** seeded on 75/76 routed cases; 1 case went through the name picker.
- **Seed contents:** 0/75 seeds contain `</cc>`. R2 trimmed whitespace in 0/75.
- **Copy rate:** 40/52 on valid drafts, 0/23 on invalid drafts.
- **Finding (review A, F1):** on genuinely invalid drafts, seeding yields fewer *exact* calls than end_projection: 13 vs 19 of 24.
  - Within this run the paired split is 7 vs 1 (exact two-sided p 0.07).
  - **It replicates.** On the same 24 drafts, the T-0080c boot gave end_projection 16 vs seed 11, and dev gave 8/9 vs 4/9 (paired 4 vs 0).
    The seed arm is nearly deterministic across boots: 74/76 exact flags agree.
  - So the gap is not significant at n 24 on its own, but it is consistent across three samples, and it should not be dismissed as noise.
  - A plausible reading, not tested here: the remaining invalid drafts are wrong in content, and starting from them anchors the repair.
    T-0081 (BFCL) found no anchoring (16 vs 16, p 1.0).
  - Schema validity is unaffected: all arms make 24/24 of those drafts schema-valid.

## Noise floor, counterfactuals and timing (criterion 7)
- **Run-to-run noise** (same prompts, different boot, no sampler seed). The `exact` flag of replayed vs stored T-0080c pass-2 rows
  agrees on 61/76 (end_projection), 51/76 (perstep_mar) and 58/76 (qargmax) routed cases; byte-identical outputs are 0–1/76. So
  differences of a few calls between policies at n 80 sit inside run-to-run noise. The selection rule's tolerance absorbs some of this,
  but the ranking among close policies is not stable.
- **Offline re-score** (T-0082a, counterfactual): `draft_only'` 52/76 schema-valid and 47/80 exact, the same as measured here (pass 1 is
  stored).
- **Pass-2 call rate under `cd_on_invalid'`:** 24/76, down from 49/76 at T-0080c.
- **e2e change against T-0080c (p50/p90, review A F2):**
  - T-0080c's choice (`cd_on_invalid:seed`) went from 426.2/609.1 at 58/80 to the new choice at 316.4/558.1, 66/80.
  - Same policy, before and after the strip: `cd_on_invalid:end_projection` 445.1/697.2 → 316.4/558.1; `cd_on_invalid:seed` 426.2/609.1 →
    316.4/516.3.
- **Boot shift (review A).** This replay boot scored higher on every cell: `cd_always` 57→62 (end_projection), 51→56, 51→57, and the seed
  58→60. None of these is significant alone. So "58/80 → 66/80" mixes three things (the strip, the policy change and a boot shift) and must
  not be read as the strip's effect alone. The strip's own effect on the draft is the offline count: 24/80 → 47/80 exact.
- **Warm-up check (review B).** The uncached pass-2 tokens per cell are identical to the live T-0080c run (median 10, max 32, pooled
  prefix hit 0.9915), so the warm-up reproduced the live cache state. Only the very first warm-up was cold.
- **JIT warm-up spikes.** Triton JIT compiles land on the first case's per-step request in both runs (test 3.5 s, dev 4.3 s); the dev
  `cd_on_invalid:perstep_mar` p90 of 4295 ms is mostly that one spike. Re-deriving the selection with the first 1–10 cases dropped gives the
  same choice.
- **Dev report's selection line.** The verbatim dev report prints a stage-1 `selection:` line. That is the report's own computation; no
  selection file was written and none is used (T-0082b R6).
- **Walls by position** (warm-up check): pass-2 p50 is 405.5 ms for the first cell after a warm-up and 438.2 ms for later cells. So the
  warm-up reproduced the live cache state.
- **Unchanged by construction:** route recall 76/80, false tags 0/12.

## Dev pilot (n 40 + 8; reported, not selected on)
- `draft_only'` 27/40.
- `cd_on_invalid` exact: end_projection 35, qargmax 34, perstep_mar 31, seed 31.
- Seeded pass 2 keeps the valid draft in 27/29 cases. CS is 1.000 on every cell, and 0/38 seeds contain `</cc>`.

## Per-tool gap (T-0080c F3, restated)
The per-tool table gives exact, schema_valid and e2e p50 only. Per-tool CS, pass-2 walls, steps, bucket and prefix hit are not broken
out.

---

## R-test driver report (verbatim)
# Two-pass tool-call report (T-0080b driver)

- stage test, commit 48313a99834035ff5f54eab8e67142adddc489e9, cells ['T1/named/end_projection', 'T1/named/perstep_mar', 'T1/named/perstep_mar_qargmax', 'T1/named/perstep_mar+seed'], n 92, system prompt sha256 fa5dace3bd1c9c89476c922c8e3c7bb4e8030d8a8fffc8ffd349982c903bbad3, tools sha256 bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707, cases sha256 460cd1c337dddf0cacc64cca149ec6af21a90899c42450ea5791fc3409afef85, calibration {'perstep_mar': {'path': '/home/user/demos/local_astra/.claude/worktrees/t0082c-launch/docs/cd/calibration-bfcl.json', 'sha256': '4084026c580a3148b4351699af64832e22b8e756fb288b551a17e642585ba74a', 'entropy_bound': 1.0, 'confidence_threshold': 0.005}, 'perstep_mar_qargmax': {'path': '/home/user/demos/local_astra/.claude/worktrees/t0082c-launch/docs/cd/calibration-bfcl-perstep_mar_qargmax.json', 'sha256': 'cb81bea43c274a446a01a34187d8d1a6e2b484eb4c417f82f90c4462e54ae086', 'entropy_bound': 0.3, 'confidence_threshold': 0.005}}, thinking on, pass-1 max_tokens 1024, pass-2 {'T1': 1024, 'T2': 256, 'T4': 256}, concurrency 1, no seed, versions {'served_model': 'nvidia/diffusiongemma-26B-A4B-it-NVFP4', 'harness': 'local_astra.eval.cd (T-0070)', 'versions_lock_sha256': '5bab0570362d68c841f82f8d56cd58ddf1dd3253a2d1208f3ea2fe7109246f25', 'dgfa_commit': '27c7d0d3cc9c9e338d2fe4f6a1e64d3da01c8c03', 'compiler_version': 'dgfa_compile@27c7d0d+local_astra.T-0063;outlines-core==0.2.14', 'tokenizer_sha256': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f', 'model_revision': '2ea837236295d617ac27f8c17d61228081932c40'}
- pass 1 replayed (T-0082b) from /home/user/demos/local_astra/logs/t0080-toolcall-test-0dc9134/20261009T152205.274601Z-toolcall-test.jsonl (sha256 1c69a9c2fcb0465db2399d65f470ffd3b702af63097788e5789d4220b77b6ce1, commit 0dc91342a8c8bf33eb0a0a940bae123448873735, run 44c5b3ccf450): no pass-1 request was sent; the pass-1 rows are that artifact's, re-scored, and the pass-1 walls (and e2e walls) come from that artifact's boot
- stopped: no
- route recall (tool cases): 76/80 = 0.950 [0.878, 0.980]; false-tag rate (no-tool cases): 0/12 = 0.000 [0.000, 0.243]
- drafts with a stripped `</cc>` close (T-0082a), logs/t0082-replay-test-48313a9/20261010T113830.525698Z-toolcall-test.jsonl: 33/76 routed
- selection: {'rule': 'stage2', 'choice': 'cd_on_invalid:T1/named/end_projection'}
- pass-2 wall ms by position within a case (T-0082b R10; first cell vs later cells): first p50/p90 405.5/1356.8 (n 76); later p50/p90 438.2/1202.0 (n 228)

| policy | exact | schema_valid | CS | pass-2 wall p50/p90 ms | e2e p50/p90 ms (tool) | e2e p50 ms (no-tool) | steps mean/p90 | prefix hit (pooled; median) |
|---|---|---|---|---|---|---|---|---|
| draft_only | 47/80 = 0.588 [0.478, 0.689] | 52/76 = 0.684 [0.573, 0.778] | n/a | n/a/n/a | 247.9/343.6 | 219.0 | n/a/n/a | n/a; n/a |
| cd_always:T1/named/end_projection | 62/80 = 0.775 [0.672, 0.853] | 76/76 = 1.000 [0.952, 1.000] | 76/76 = 1.000 [0.952, 1.000] | 319.9/404.2 | 577.2/724.0 | 219.0 | 9.7/11.0 | 0.992; 0.994 |
| cd_always:T1/named/perstep_mar | 56/80 = 0.700 [0.592, 0.789] | 76/76 = 1.000 [0.952, 1.000] | 76/76 = 1.000 [0.952, 1.000] | 720.3/1542.9 | 991.8/1736.3 | 219.0 | 17.0/31.0 | 0.992; 0.994 |
| cd_always:T1/named/perstep_mar_qargmax | 57/80 = 0.713 [0.605, 0.800] | 76/76 = 1.000 [0.952, 1.000] | 76/76 = 1.000 [0.952, 1.000] | 782.4/1683.4 | 1039.9/1992.5 | 219.0 | 18.8/33.0 | 0.992; 0.994 |
| cd_always:T1/named/perstep_mar+seed | 60/80 = 0.750 [0.645, 0.832] | 76/76 = 1.000 [0.952, 1.000] | 76/76 = 1.000 [0.952, 1.000] | 197.4/274.9 | 449.6/607.1 | 219.0 | 2.8/4.0 | 0.992; 0.994 |
| cd_on_invalid:T1/named/end_projection | 66/80 = 0.825 [0.727, 0.893] | 76/76 = 1.000 [0.952, 1.000] | 24/24 = 1.000 [0.862, 1.000] | 293.7/483.9 | 316.4/558.1 | 219.0 | 9.6/15.0 | 0.990; 0.986 |
| cd_on_invalid:T1/named/perstep_mar | 64/80 = 0.800 [0.700, 0.873] | 76/76 = 1.000 [0.952, 1.000] | 24/24 = 1.000 [0.862, 1.000] | 827.3/1776.2 | 316.4/1271.4 | 219.0 | 19.2/34.0 | 0.990; 0.986 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | 66/80 = 0.825 [0.727, 0.893] | 76/76 = 1.000 [0.952, 1.000] | 24/24 = 1.000 [0.862, 1.000] | 811.9/1683.4 | 316.4/1229.1 | 219.0 | 19.5/35.0 | 0.990; 0.986 |
| cd_on_invalid:T1/named/perstep_mar+seed | 60/80 = 0.750 [0.645, 0.832] | 76/76 = 1.000 [0.952, 1.000] | 24/24 = 1.000 [0.862, 1.000] | 236.8/379.0 | 316.4/516.3 | 219.0 | 3.9/6.0 | 0.990; 0.986 |

| policy | tool | exact | schema_valid | e2e p50 ms |
|---|---|---|---|---|
| draft_only | click_at | 17/20 = 0.850 [0.640, 0.948] | 19/20 = 0.950 [0.764, 0.991] | 287.4 |
| draft_only | google_search | 4/20 = 0.200 [0.081, 0.416] | 4/18 = 0.222 [0.090, 0.452] | 191.6 |
| draft_only | press_key | 2/10 = 0.200 [0.057, 0.510] | 4/8 = 0.500 [0.215, 0.785] | 262.1 |
| draft_only | scroll | 7/10 = 0.700 [0.397, 0.892] | 7/10 = 0.700 [0.397, 0.892] | 202.4 |
| draft_only | type_at | 17/20 = 0.850 [0.640, 0.948] | 18/20 = 0.900 [0.699, 0.972] | 279.4 |
| cd_always:T1/named/end_projection | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 617.1 |
| cd_always:T1/named/end_projection | google_search | 13/20 = 0.650 [0.433, 0.819] | 18/18 = 1.000 [0.824, 1.000] | 443.2 |
| cd_always:T1/named/end_projection | press_key | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 579.7 |
| cd_always:T1/named/end_projection | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 469.4 |
| cd_always:T1/named/end_projection | type_at | 17/20 = 0.850 [0.640, 0.948] | 20/20 = 1.000 [0.839, 1.000] | 629.5 |
| cd_always:T1/named/perstep_mar | click_at | 13/20 = 0.650 [0.433, 0.819] | 20/20 = 1.000 [0.839, 1.000] | 1003.0 |
| cd_always:T1/named/perstep_mar | google_search | 12/20 = 0.600 [0.387, 0.781] | 18/18 = 1.000 [0.824, 1.000] | 960.0 |
| cd_always:T1/named/perstep_mar | press_key | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 939.6 |
| cd_always:T1/named/perstep_mar | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 807.0 |
| cd_always:T1/named/perstep_mar | type_at | 17/20 = 0.850 [0.640, 0.948] | 20/20 = 1.000 [0.839, 1.000] | 1051.4 |
| cd_always:T1/named/perstep_mar_qargmax | click_at | 16/20 = 0.800 [0.584, 0.919] | 20/20 = 1.000 [0.839, 1.000] | 1088.4 |
| cd_always:T1/named/perstep_mar_qargmax | google_search | 12/20 = 0.600 [0.387, 0.781] | 18/18 = 1.000 [0.824, 1.000] | 860.3 |
| cd_always:T1/named/perstep_mar_qargmax | press_key | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 1072.3 |
| cd_always:T1/named/perstep_mar_qargmax | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 787.3 |
| cd_always:T1/named/perstep_mar_qargmax | type_at | 15/20 = 0.750 [0.531, 0.888] | 20/20 = 1.000 [0.839, 1.000] | 1161.9 |
| cd_always:T1/named/perstep_mar+seed | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 494.2 |
| cd_always:T1/named/perstep_mar+seed | google_search | 11/20 = 0.550 [0.342, 0.742] | 18/18 = 1.000 [0.824, 1.000] | 392.4 |
| cd_always:T1/named/perstep_mar+seed | press_key | 4/10 = 0.400 [0.168, 0.687] | 8/8 = 1.000 [0.676, 1.000] | 520.7 |
| cd_always:T1/named/perstep_mar+seed | scroll | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 409.7 |
| cd_always:T1/named/perstep_mar+seed | type_at | 17/20 = 0.850 [0.640, 0.948] | 20/20 = 1.000 [0.839, 1.000] | 494.3 |
| cd_on_invalid:T1/named/end_projection | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 287.4 |
| cd_on_invalid:T1/named/end_projection | google_search | 15/20 = 0.750 [0.531, 0.888] | 18/18 = 1.000 [0.824, 1.000] | 409.7 |
| cd_on_invalid:T1/named/end_projection | press_key | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 511.0 |
| cd_on_invalid:T1/named/end_projection | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 238.0 |
| cd_on_invalid:T1/named/end_projection | type_at | 19/20 = 0.950 [0.764, 0.991] | 20/20 = 1.000 [0.839, 1.000] | 284.9 |
| cd_on_invalid:T1/named/perstep_mar | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 287.4 |
| cd_on_invalid:T1/named/perstep_mar | google_search | 13/20 = 0.650 [0.433, 0.819] | 18/18 = 1.000 [0.824, 1.000] | 960.0 |
| cd_on_invalid:T1/named/perstep_mar | press_key | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 593.1 |
| cd_on_invalid:T1/named/perstep_mar | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 238.0 |
| cd_on_invalid:T1/named/perstep_mar | type_at | 19/20 = 0.950 [0.764, 0.991] | 20/20 = 1.000 [0.839, 1.000] | 284.9 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 287.4 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | google_search | 15/20 = 0.750 [0.531, 0.888] | 18/18 = 1.000 [0.824, 1.000] | 759.1 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | press_key | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 602.8 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 238.0 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | type_at | 19/20 = 0.950 [0.764, 0.991] | 20/20 = 1.000 [0.839, 1.000] | 284.9 |
| cd_on_invalid:T1/named/perstep_mar+seed | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 287.4 |
| cd_on_invalid:T1/named/perstep_mar+seed | google_search | 11/20 = 0.550 [0.342, 0.742] | 18/18 = 1.000 [0.824, 1.000] | 380.4 |
| cd_on_invalid:T1/named/perstep_mar+seed | press_key | 4/10 = 0.400 [0.168, 0.687] | 8/8 = 1.000 [0.676, 1.000] | 491.5 |
| cd_on_invalid:T1/named/perstep_mar+seed | scroll | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 238.0 |
| cd_on_invalid:T1/named/perstep_mar+seed | type_at | 17/20 = 0.850 [0.640, 0.948] | 20/20 = 1.000 [0.839, 1.000] | 284.9 |

McNemar (exact, tool cases; Bonferroni over 14):

- cd_always:T1/named/end_projection vs draft_only: n 80, b 20, c 5, p 0.004077 (adj 0.05708); both parsed n 52, b 1, c 5, p 0.2188
- cd_always:T1/named/perstep_mar vs draft_only: n 80, b 17, c 8, p 0.1078 (adj 1); both parsed n 52, b 0, c 8, p 0.007812
- cd_always:T1/named/perstep_mar_qargmax vs draft_only: n 80, b 19, c 9, p 0.08716 (adj 1); both parsed n 52, b 0, c 9, p 0.003906
- cd_always:T1/named/perstep_mar+seed vs draft_only: n 80, b 13, c 0, p 0.0002441 (adj 0.003418); both parsed n 52, b 0, c 0, p 1
- cd_on_invalid:T1/named/end_projection vs draft_only: n 80, b 19, c 0, p 3.815e-06 (adj 5.341e-05); both parsed n 52, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar vs draft_only: n 80, b 17, c 0, p 1.526e-05 (adj 0.0002136); both parsed n 52, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar_qargmax vs draft_only: n 80, b 19, c 0, p 3.815e-06 (adj 5.341e-05); both parsed n 52, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar+seed vs draft_only: n 80, b 13, c 0, p 0.0002441 (adj 0.003418); both parsed n 52, b 0, c 0, p 1
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar: n 80, b 10, c 4, p 0.1796 (adj 1); both parsed n 76, b 10, c 4, p 0.1796
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar_qargmax: n 80, b 8, c 3, p 0.2266 (adj 1); both parsed n 76, b 8, c 3, p 0.2266
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar+seed: n 80, b 8, c 6, p 0.7905 (adj 1); both parsed n 76, b 8, c 6, p 0.7905
- cd_always:T1/named/perstep_mar vs cd_always:T1/named/perstep_mar_qargmax: n 80, b 8, c 9, p 1 (adj 1); both parsed n 76, b 8, c 9, p 1
- cd_always:T1/named/perstep_mar vs cd_always:T1/named/perstep_mar+seed: n 80, b 5, c 9, p 0.424 (adj 1); both parsed n 76, b 5, c 9, p 0.424
- cd_always:T1/named/perstep_mar_qargmax vs cd_always:T1/named/perstep_mar+seed: n 80, b 8, c 11, p 0.6476 (adj 1); both parsed n 76, b 8, c 11, p 0.6476

T1 `101` position: {'T1/named/end_projection': {'101': 1, '103': 1, '104': 1, '106': 1, '109': 1, '112': 1, '114': 1, '117': 1, '121': 2, '122': 1, '144': 1, '164': 1, '17': 1, '173': 1, '18': 1, '19': 1, '200': 1, '345': 1, '35': 1, '38': 1, '40': 1, '43': 4, '431': 1, '44': 2, '45': 3, '46': 2, '47': 2, '48': 1, '49': 1, '53': 3, '54': 1, '57': 2, '58': 1, '59': 1, '60': 1, '62': 2, '64': 3, '65': 1, '66': 1, '67': 1, '68': 1, '71': 3, '76': 1, '77': 1, '78': 1, '79': 1, '82': 1, '83': 1, '85': 1, '86': 2, '90': 1, '91': 1, '92': 1, '93': 1, '94': 1, '95': 2, '99': 1}, 'T1/named/perstep_mar': {'100': 2, '104': 1, '105': 1, '108': 1, '109': 1, '110': 1, '111': 1, '115': 1, '117': 1, '118': 1, '122': 2, '148': 1, '156': 1, '164': 1, '17': 2, '18': 2, '19': 2, '20': 1, '23': 1, '36': 1, '38': 1, '39': 1, '409': 1, '42': 1, '43': 1, '44': 2, '45': 2, '46': 2, '47': 2, '49': 1, '50': 2, '51': 1, '52': 2, '54': 1, '59': 1, '60': 2, '62': 1, '64': 2, '65': 3, '67': 1, '68': 1, '70': 1, '71': 1, '73': 2, '76': 1, '80': 4, '83': 1, '84': 1, '85': 1, '87': 3, '90': 2, '93': 1, '94': 1, '98': 1}, 'T1/named/perstep_mar_qargmax': {'100': 1, '102': 1, '104': 2, '106': 1, '108': 2, '109': 1, '110': 1, '118': 1, '120': 3, '146': 1, '17': 1, '18': 1, '256': 1, '281': 1, '316': 1, '333': 1, '35': 1, '39': 1, '40': 3, '43': 1, '44': 3, '45': 1, '46': 1, '47': 2, '48': 3, '49': 1, '52': 1, '53': 2, '55': 1, '57': 3, '60': 2, '61': 1, '62': 2, '63': 2, '65': 1, '67': 1, '68': 2, '70': 1, '71': 1, '72': 2, '74': 1, '75': 1, '80': 1, '81': 2, '84': 2, '85': 1, '88': 1, '92': 2, '93': 1, '94': 1, '95': 3, '96': 1}, 'T1/named/perstep_mar+seed': {'104': 1, '108': 1, '115': 1, '121': 1, '132': 1, '146': 1, '159': 1, '160': 1, '27': 1, '28': 1, '29': 1, '30': 4, '31': 10, '32': 3, '33': 1, '34': 1, '35': 2, '36': 2, '38': 2, '43': 1, '46': 1, '47': 1, '49': 1, '53': 1, '54': 1, '55': 1, '56': 1, '57': 1, '58': 3, '59': 2, '60': 2, '62': 1, '64': 1, '65': 1, '69': 2, '70': 1, '74': 1, '75': 1, '76': 2, '78': 1, '79': 2, '83': 1, '84': 1, '87': 1, '90': 2, '91': 1, '94': 2, '95': 1, '96': 1, '99': 1}}

- T1/named/end_projection: pass 2 equals the valid draft 19/52 = 0.365 [0.248, 0.501]; invalid drafts repaired 24/24 = 1.000 [0.862, 1.000]; name-picker rows 1
- T1/named/perstep_mar: pass 2 equals the valid draft 15/52 = 0.288 [0.183, 0.423]; invalid drafts repaired 24/24 = 1.000 [0.862, 1.000]; name-picker rows 1
- T1/named/perstep_mar_qargmax: pass 2 equals the valid draft 19/52 = 0.365 [0.248, 0.501]; invalid drafts repaired 24/24 = 1.000 [0.862, 1.000]; name-picker rows 1
- T1/named/perstep_mar+seed: pass 2 equals the valid draft 49/52 = 0.942 [0.844, 0.980]; invalid drafts repaired 24/24 = 1.000 [0.862, 1.000]; name-picker rows 1
- seed T1/named/perstep_mar+seed: seeded 75/76 routed cases; no seed {'picker': 1}
- seed copy T1/named/perstep_mar+seed (first canvas equals the seed through its first stop): valid drafts 40/52 = 0.769 [0.639, 0.863]; invalid drafts 0/23 = 0.000 [0.000, 0.143]
- seed trim T1/named/perstep_mar+seed (seeded rows whose draft lost trailing whitespace, R2): 0/75 = 0.000 [0.000, 0.049]

---

## R-dev driver report (verbatim)
# Two-pass tool-call report (T-0080b driver)

- stage dev, commit 48313a99834035ff5f54eab8e67142adddc489e9, cells ['T1/named/end_projection', 'T1/named/perstep_mar', 'T1/named/perstep_mar_qargmax', 'T1/named/perstep_mar+seed'], n 48, system prompt sha256 fa5dace3bd1c9c89476c922c8e3c7bb4e8030d8a8fffc8ffd349982c903bbad3, tools sha256 bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707, cases sha256 460cd1c337dddf0cacc64cca149ec6af21a90899c42450ea5791fc3409afef85, calibration {'perstep_mar': {'path': '/home/user/demos/local_astra/.claude/worktrees/t0082c-launch/docs/cd/calibration-bfcl.json', 'sha256': '4084026c580a3148b4351699af64832e22b8e756fb288b551a17e642585ba74a', 'entropy_bound': 1.0, 'confidence_threshold': 0.005}, 'perstep_mar_qargmax': {'path': '/home/user/demos/local_astra/.claude/worktrees/t0082c-launch/docs/cd/calibration-bfcl-perstep_mar_qargmax.json', 'sha256': 'cb81bea43c274a446a01a34187d8d1a6e2b484eb4c417f82f90c4462e54ae086', 'entropy_bound': 0.3, 'confidence_threshold': 0.005}}, thinking on, pass-1 max_tokens 1024, pass-2 {'T1': 1024, 'T2': 256, 'T4': 256}, concurrency 1, no seed, versions {'served_model': 'nvidia/diffusiongemma-26B-A4B-it-NVFP4', 'harness': 'local_astra.eval.cd (T-0070)', 'versions_lock_sha256': '5bab0570362d68c841f82f8d56cd58ddf1dd3253a2d1208f3ea2fe7109246f25', 'dgfa_commit': '27c7d0d3cc9c9e338d2fe4f6a1e64d3da01c8c03', 'compiler_version': 'dgfa_compile@27c7d0d+local_astra.T-0063;outlines-core==0.2.14', 'tokenizer_sha256': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f', 'model_revision': '2ea837236295d617ac27f8c17d61228081932c40'}
- pass 1 replayed (T-0082b) from /home/user/demos/local_astra/logs/t0080-toolcall-dev-0dc9134/20261009T134249.641182Z-toolcall-dev.jsonl (sha256 b0fd71cccdc5b8fb3704407f1ff58bf5fc74911ee0d6e6924c9c8e8c740f1d82, commit 0dc91342a8c8bf33eb0a0a940bae123448873735, run 94e3fdbf0f8b): no pass-1 request was sent; the pass-1 rows are that artifact's, re-scored, and the pass-1 walls (and e2e walls) come from that artifact's boot
- stopped: no
- route recall (tool cases): 38/40 = 0.950 [0.835, 0.986]; false-tag rate (no-tool cases): 0/8 = 0.000 [0.000, 0.324]
- drafts with a stripped `</cc>` close (T-0082a), logs/t0082-replay-dev-48313a9/20261010T111949.137517Z-toolcall-dev.jsonl: 10/38 routed
- selection: {'rule': 'stage1', 'choice': 'T1/named/perstep_mar'}
- pass-2 wall ms by position within a case (T-0082b R10; first cell vs later cells): first p50/p90 437.5/1391.0 (n 38); later p50/p90 391.8/1085.5 (n 114)

| policy | exact | schema_valid | CS | pass-2 wall p50/p90 ms | e2e p50/p90 ms (tool) | e2e p50 ms (no-tool) | steps mean/p90 | prefix hit (pooled; median) |
|---|---|---|---|---|---|---|---|---|
| draft_only | 27/40 = 0.675 [0.520, 0.799] | 29/38 = 0.763 [0.608, 0.870] | n/a | n/a/n/a | 251.6/364.9 | 207.2 | n/a/n/a | n/a; n/a |
| cd_always:T1/named/end_projection | 28/40 = 0.700 [0.546, 0.819] | 38/38 = 1.000 [0.908, 1.000] | 38/38 = 1.000 [0.908, 1.000] | 298.3/402.9 | 553.9/683.2 | 207.2 | 8.9/11.0 | 0.992; 0.993 |
| cd_always:T1/named/perstep_mar | 23/40 = 0.575 [0.422, 0.715] | 38/38 = 1.000 [0.908, 1.000] | 38/38 = 1.000 [0.908, 1.000] | 798.0/1391.0 | 1043.2/1588.5 | 207.2 | 16.3/25.0 | 0.992; 0.993 |
| cd_always:T1/named/perstep_mar_qargmax | 25/40 = 0.625 [0.470, 0.758] | 38/38 = 1.000 [0.908, 1.000] | 38/38 = 1.000 [0.908, 1.000] | 802.5/1680.8 | 1053.4/2045.8 | 207.2 | 16.6/30.0 | 0.992; 0.993 |
| cd_always:T1/named/perstep_mar+seed | 31/40 = 0.775 [0.625, 0.877] | 38/38 = 1.000 [0.908, 1.000] | 38/38 = 1.000 [0.908, 1.000] | 196.8/243.6 | 460.2/563.1 | 207.2 | 2.4/3.0 | 0.992; 0.993 |
| cd_on_invalid:T1/named/end_projection | 35/40 = 0.875 [0.739, 0.945] | 38/38 = 1.000 [0.908, 1.000] | 9/9 = 1.000 [0.701, 1.000] | 278.2/454.2 | 273.7/474.9 | 207.2 | 8.1/14.0 | 0.990; 0.987 |
| cd_on_invalid:T1/named/perstep_mar | 31/40 = 0.775 [0.625, 0.877] | 38/38 = 1.000 [0.908, 1.000] | 9/9 = 1.000 [0.701, 1.000] | 769.2/4295.5 | 273.7/918.5 | 207.2 | 15.6/25.0 | 0.990; 0.987 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | 34/40 = 0.850 [0.709, 0.929] | 38/38 = 1.000 [0.908, 1.000] | 9/9 = 1.000 [0.701, 1.000] | 768.5/1680.8 | 273.7/1239.3 | 207.2 | 16.2/33.0 | 0.990; 0.987 |
| cd_on_invalid:T1/named/perstep_mar+seed | 31/40 = 0.775 [0.625, 0.877] | 38/38 = 1.000 [0.908, 1.000] | 9/9 = 1.000 [0.701, 1.000] | 231.7/379.0 | 273.7/471.8 | 207.2 | 3.0/6.0 | 0.990; 0.987 |

| policy | tool | exact | schema_valid | e2e p50 ms |
|---|---|---|---|---|
| draft_only | click_at | 6/10 = 0.600 [0.313, 0.832] | 7/8 = 0.875 [0.529, 0.978] | 290.3 |
| draft_only | google_search | 5/10 = 0.500 [0.237, 0.763] | 5/10 = 0.500 [0.237, 0.763] | 199.8 |
| draft_only | press_key | 3/5 = 0.600 [0.231, 0.882] | 4/5 = 0.800 [0.376, 0.964] | 224.3 |
| draft_only | scroll | 4/5 = 0.800 [0.376, 0.964] | 4/5 = 0.800 [0.376, 0.964] | 203.3 |
| draft_only | type_at | 9/10 = 0.900 [0.596, 0.982] | 9/10 = 0.900 [0.596, 0.982] | 273.7 |
| cd_always:T1/named/end_projection | click_at | 4/10 = 0.400 [0.168, 0.687] | 8/8 = 1.000 [0.676, 1.000] | 637.1 |
| cd_always:T1/named/end_projection | google_search | 8/10 = 0.800 [0.490, 0.943] | 10/10 = 1.000 [0.722, 1.000] | 442.8 |
| cd_always:T1/named/end_projection | press_key | 2/5 = 0.400 [0.118, 0.769] | 5/5 = 1.000 [0.566, 1.000] | 520.7 |
| cd_always:T1/named/end_projection | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 474.9 |
| cd_always:T1/named/end_projection | type_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 629.0 |
| cd_always:T1/named/perstep_mar | click_at | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 924.7 |
| cd_always:T1/named/perstep_mar | google_search | 4/10 = 0.400 [0.168, 0.687] | 10/10 = 1.000 [0.722, 1.000] | 899.7 |
| cd_always:T1/named/perstep_mar | press_key | 3/5 = 0.600 [0.231, 0.882] | 5/5 = 1.000 [0.566, 1.000] | 1177.9 |
| cd_always:T1/named/perstep_mar | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 758.3 |
| cd_always:T1/named/perstep_mar | type_at | 6/10 = 0.600 [0.313, 0.832] | 10/10 = 1.000 [0.722, 1.000] | 1096.5 |
| cd_always:T1/named/perstep_mar_qargmax | click_at | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 1086.3 |
| cd_always:T1/named/perstep_mar_qargmax | google_search | 5/10 = 0.500 [0.237, 0.763] | 10/10 = 1.000 [0.722, 1.000] | 831.9 |
| cd_always:T1/named/perstep_mar_qargmax | press_key | 3/5 = 0.600 [0.231, 0.882] | 5/5 = 1.000 [0.566, 1.000] | 971.4 |
| cd_always:T1/named/perstep_mar_qargmax | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 953.7 |
| cd_always:T1/named/perstep_mar_qargmax | type_at | 7/10 = 0.700 [0.397, 0.892] | 10/10 = 1.000 [0.722, 1.000] | 1549.1 |
| cd_always:T1/named/perstep_mar+seed | click_at | 7/10 = 0.700 [0.397, 0.892] | 8/8 = 1.000 [0.676, 1.000] | 474.3 |
| cd_always:T1/named/perstep_mar+seed | google_search | 6/10 = 0.600 [0.313, 0.832] | 10/10 = 1.000 [0.722, 1.000] | 401.6 |
| cd_always:T1/named/perstep_mar+seed | press_key | 3/5 = 0.600 [0.231, 0.882] | 5/5 = 1.000 [0.566, 1.000] | 506.2 |
| cd_always:T1/named/perstep_mar+seed | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 439.6 |
| cd_always:T1/named/perstep_mar+seed | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 479.5 |
| cd_on_invalid:T1/named/end_projection | click_at | 7/10 = 0.700 [0.397, 0.892] | 8/8 = 1.000 [0.676, 1.000] | 290.3 |
| cd_on_invalid:T1/named/end_projection | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 309.4 |
| cd_on_invalid:T1/named/end_projection | press_key | 3/5 = 0.600 [0.231, 0.882] | 5/5 = 1.000 [0.566, 1.000] | 315.6 |
| cd_on_invalid:T1/named/end_projection | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 226.3 |
| cd_on_invalid:T1/named/end_projection | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 273.7 |
| cd_on_invalid:T1/named/perstep_mar | click_at | 7/10 = 0.700 [0.397, 0.892] | 8/8 = 1.000 [0.676, 1.000] | 290.3 |
| cd_on_invalid:T1/named/perstep_mar | google_search | 7/10 = 0.700 [0.397, 0.892] | 10/10 = 1.000 [0.722, 1.000] | 410.7 |
| cd_on_invalid:T1/named/perstep_mar | press_key | 3/5 = 0.600 [0.231, 0.882] | 5/5 = 1.000 [0.566, 1.000] | 315.6 |
| cd_on_invalid:T1/named/perstep_mar | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 226.3 |
| cd_on_invalid:T1/named/perstep_mar | type_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 273.7 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | click_at | 7/10 = 0.700 [0.397, 0.892] | 8/8 = 1.000 [0.676, 1.000] | 290.3 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | google_search | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 441.5 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | press_key | 3/5 = 0.600 [0.231, 0.882] | 5/5 = 1.000 [0.566, 1.000] | 315.6 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 226.3 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 273.7 |
| cd_on_invalid:T1/named/perstep_mar+seed | click_at | 7/10 = 0.700 [0.397, 0.892] | 8/8 = 1.000 [0.676, 1.000] | 290.3 |
| cd_on_invalid:T1/named/perstep_mar+seed | google_search | 6/10 = 0.600 [0.313, 0.832] | 10/10 = 1.000 [0.722, 1.000] | 293.9 |
| cd_on_invalid:T1/named/perstep_mar+seed | press_key | 3/5 = 0.600 [0.231, 0.882] | 5/5 = 1.000 [0.566, 1.000] | 315.6 |
| cd_on_invalid:T1/named/perstep_mar+seed | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 226.3 |
| cd_on_invalid:T1/named/perstep_mar+seed | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 273.7 |

McNemar (exact, tool cases; Bonferroni over 14):

- cd_always:T1/named/end_projection vs draft_only: n 40, b 8, c 7, p 1 (adj 1); both parsed n 29, b 0, c 7, p 0.01562
- cd_always:T1/named/perstep_mar vs draft_only: n 40, b 4, c 8, p 0.3877 (adj 1); both parsed n 29, b 0, c 8, p 0.007812
- cd_always:T1/named/perstep_mar_qargmax vs draft_only: n 40, b 7, c 9, p 0.8036 (adj 1); both parsed n 29, b 0, c 9, p 0.003906
- cd_always:T1/named/perstep_mar+seed vs draft_only: n 40, b 4, c 0, p 0.125 (adj 1); both parsed n 29, b 0, c 0, p 1
- cd_on_invalid:T1/named/end_projection vs draft_only: n 40, b 8, c 0, p 0.007812 (adj 0.1094); both parsed n 29, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar vs draft_only: n 40, b 4, c 0, p 0.125 (adj 1); both parsed n 29, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar_qargmax vs draft_only: n 40, b 7, c 0, p 0.01562 (adj 0.2188); both parsed n 29, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar+seed vs draft_only: n 40, b 4, c 0, p 0.125 (adj 1); both parsed n 29, b 0, c 0, p 1
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar: n 40, b 10, c 5, p 0.3018 (adj 1); both parsed n 38, b 10, c 5, p 0.3018
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar_qargmax: n 40, b 7, c 4, p 0.5488 (adj 1); both parsed n 38, b 7, c 4, p 0.5488
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar+seed: n 40, b 4, c 7, p 0.5488 (adj 1); both parsed n 38, b 4, c 7, p 0.5488
- cd_always:T1/named/perstep_mar vs cd_always:T1/named/perstep_mar_qargmax: n 40, b 2, c 4, p 0.6875 (adj 1); both parsed n 38, b 2, c 4, p 0.6875
- cd_always:T1/named/perstep_mar vs cd_always:T1/named/perstep_mar+seed: n 40, b 1, c 9, p 0.02148 (adj 0.3008); both parsed n 38, b 1, c 9, p 0.02148
- cd_always:T1/named/perstep_mar_qargmax vs cd_always:T1/named/perstep_mar+seed: n 40, b 3, c 9, p 0.146 (adj 1); both parsed n 38, b 3, c 9, p 0.146

T1 `101` position: {'T1/named/end_projection': {'105': 1, '107': 1, '109': 1, '111': 1, '114': 1, '167': 1, '17': 1, '190': 1, '37': 1, '41': 6, '42': 1, '43': 1, '46': 2, '48': 1, '49': 2, '50': 1, '52': 2, '54': 1, '63': 2, '66': 1, '67': 2, '79': 1, '80': 1, '81': 1, '91': 1, '96': 1, '97': 1, '98': 1}, 'T1/named/perstep_mar': {'100': 1, '101': 1, '103': 1, '107': 1, '109': 1, '116': 1, '128': 1, '136': 1, '17': 1, '18': 1, '36': 1, '39': 2, '42': 3, '43': 1, '44': 1, '45': 1, '46': 1, '48': 1, '51': 1, '54': 1, '64': 2, '65': 1, '67': 1, '68': 2, '69': 1, '71': 1, '72': 1, '81': 1, '85': 1, '86': 2, '90': 1, '93': 1}, 'T1/named/perstep_mar_qargmax': {'100': 1, '102': 1, '108': 2, '125': 1, '132': 1, '17': 2, '36': 1, '38': 1, '41': 3, '42': 2, '44': 2, '45': 1, '46': 1, '47': 1, '49': 1, '51': 1, '55': 1, '64': 2, '66': 2, '67': 1, '68': 2, '69': 1, '74': 1, '75': 1, '79': 1, '86': 1, '94': 1, '95': 1, '98': 1}, 'T1/named/perstep_mar+seed': {'102': 1, '130': 1, '136': 1, '26': 1, '28': 1, '29': 4, '30': 2, '31': 3, '32': 1, '33': 1, '34': 1, '36': 1, '48': 1, '51': 1, '55': 2, '56': 2, '59': 1, '61': 2, '66': 2, '75': 1, '79': 1, '80': 1, '83': 1, '85': 1, '86': 1, '89': 1, '93': 1, '96': 1}}

- T1/named/end_projection: pass 2 equals the valid draft 9/29 = 0.310 [0.173, 0.492]; invalid drafts repaired 9/9 = 1.000 [0.701, 1.000]; name-picker rows 0
- T1/named/perstep_mar: pass 2 equals the valid draft 8/29 = 0.276 [0.147, 0.457]; invalid drafts repaired 9/9 = 1.000 [0.701, 1.000]; name-picker rows 0
- T1/named/perstep_mar_qargmax: pass 2 equals the valid draft 7/29 = 0.241 [0.122, 0.421]; invalid drafts repaired 9/9 = 1.000 [0.701, 1.000]; name-picker rows 0
- T1/named/perstep_mar+seed: pass 2 equals the valid draft 27/29 = 0.931 [0.780, 0.981]; invalid drafts repaired 9/9 = 1.000 [0.701, 1.000]; name-picker rows 0
- seed T1/named/perstep_mar+seed: seeded 38/38 routed cases; no seed none
- seed copy T1/named/perstep_mar+seed (first canvas equals the seed through its first stop): valid drafts 24/29 = 0.828 [0.655, 0.924]; invalid drafts 0/9 = 0.000 [0.000, 0.299]
- seed trim T1/named/perstep_mar+seed (seeded rows whose draft lost trailing whitespace, R2): 0/38 = 0.000 [0.000, 0.092]
