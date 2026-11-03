# Tool-call v1: two-pass constrained decoding, results (T-0080c)

Regime (both stages): commit 0dc9134 (pinned detached launch worktree, clean), profile lean (engine only), nvidia/diffusiongemma-26B-A4B-it-NVFP4
rev 2ea8372, vLLM v0.29.0, RTX 5090, driver 595.84, CL 256, 48 steps, gpu-mem 0.86, prefix caching on, concurrency 1, thinking on, pass-1
max_tokens 1024. Eval set toolcall-v1 (cases sha256 460cd1c3…; human-approved at H-T80, 2026-10-09). Knobs: perstep_mar eb 1.0 / ct 0.005
(calibration-bfcl.json sha256 4084026c…), perstep_mar_qargmax from calibration-bfcl-perstep_mar_qargmax.json, seed noise 0 / start step 24
(T-0081 `mar_n0_s24`) with the R2 trailing-whitespace trim. Pass-2 scope named only (K-D2). Canvas tap on (T-0011: +0.48 % on B16). Every number below is from the artifacts in
`logs/t0080-toolcall-{dev,test}-0dc9134/` of the launch worktree; the two driver reports are reproduced verbatim at the end.

## Choice (pre-registered rule, T-0080c criterion 3)
**`cd_on_invalid:T1/named/perstep_mar+seed`**: send the call as drafted when it is schema-valid; otherwise run one constrained pass
seeded from the draft (1/49 invalid drafts went through the unseeded two-request name picker instead). Test split (n 80 tool + 12 no-tool): exact 58/80 = 0.725, schema 76/76, CS 49/49, end-to-end p50 / p90 426.2 /
609.1 ms on tool cases, pass-2 p50 232.9 ms, 3.4 executed steps per pass-2 call (mean).

How the rule got there:
- The best eligible exact is 60/80 (`cd_on_invalid:…/end_projection`), so the bar is 60/80 − 0.05 = 56/80. That leaves every policy except
  `cd_always:perstep_mar` and `cd_always:perstep_mar_qargmax` (51/80 each).
- The lowest end-to-end p50 is the chosen policy, at 426.2 ms. `cd_on_invalid:…/end_projection` is within 5 % at 445.1 ms. The tie-break
  goes to the lower p90 (609.1 vs 697.2 ms), so the chosen policy wins.
- `draft_only` is ineligible: schema 27/76.

## Gaps against criterion 6 (review A, F3)
The per-tool table gives exact, schema_valid and end-to-end p50 only. Per-tool CS, pass-2 p50/p90, end-to-end p90, steps, bucket and
prefix hit are not broken out. Bucket appears only in the run log (T1: every tool case is in the 128 bucket). The pooled values are in the
policy table.

## Stage 1 (dev, n 40 + 8): ruling T1/named
Under `cd_always`:
| thought | exact | pass-2 p50 ms | steps mean |
|---|---|---|---|
| T1 | 21/40 | 799.5 | 18.4 |
| T2 | 6/40 | 2671.2 | 34.6 |
| T4 | 12/40 | 434.4 | 6.1 |

The bar is best − 0.10, which only T1 clears.

## The human's expectation ("the constrained pass mostly adjusts the draft")
- **Without a seed, it did not hold.** Pass 2 equalled a valid draft in 10/27 (end_projection), 10/27 (perstep_mar) and 9/27 (qargmax)
  cases on test. On dev it was 4/19 at T1. `cd_always` turned correct drafts wrong: draft-right / CD-wrong 3, 8 and 7 against 0, on the 27
  cases where both the draft and pass 2 parsed.
- **With the seed, it did hold.** Pass 2 equalled the valid draft in 25/27 cases, and there were 0 draft-right / CD-wrong cases.
- **Repairs.** All 49 invalid drafts were repaired by every arm, and none of the 48 seeded ones was a copy (copy 0/48, as expected: an invalid draft
  cannot pass the grammar). Most of these "repairs" are not grammar fixes; see the next point.
- **Drafts — most "invalid" drafts are a closing `</cc>` tag (review A, F2).** The model often closes the tag. The router keeps
  `</cc>` in the draft call text, so a complete, correct call fails to parse.
  - Parse failures on test: 41 not JSON, 7 no arguments, 1 args not an object (49). 33 of the drafts end in `</cc>`, and 30 of those are
    valid JSON once it is stripped. On dev, 10 of the 19 invalid drafts end in `</cc>`.
  - Counterfactual with `</cc>` stripped: `draft_only` on test would be 52/76 schema-valid (not 27/76) and 47/80 exact (not 24/80).
  - So of the ~34 extra exact calls that `cd_on_invalid` shows over `draft_only` (b 34–36, c 0, p ≤ 1.2e-10), about 23 come from removing
    `</cc>`. The grammar's own contribution is about 11 (58–60 vs ~47), which this run does not test directly.
  - Review B: 25 of those 33 are schema-valid once `</cc>` is stripped, 23 of them exact, and the seeded pass 2 got exactly those 23
    right. It is also why the seed copy rate on invalid drafts is 0/48: those seeds carry `</cc>`, which no grammar accepts.
  - The selection is unaffected: the rule ran as written on the reviewed scorer.
  - Stripping `</cc>` in the router would also make `cd_on_invalid` cheaper, since about 25 more drafts would skip pass 2. That is a
    separate task for the human; it is not a re-run of this launch.
- **Seeding cost.** The seed cut pass-2 time from 665 to 233 ms (p50) and steps from 15.0 to 3.4 per canvas, against unseeded perstep_mar.
- **Seed coverage.** 75/76 routed cases were seeded (1 went through the name picker). The R2 trim never fired (0/75).
- **Seed vs end projection.** Among `cd_on_invalid` arms, the seeded perstep_mar and end_projection differ by 2 exact calls. The report
  compares only the `cd_always` arms pairwise, and none of those differences is significant (adjusted p = 1). `cd_on_invalid` arms were not
  compared pairwise, so read the 2-call gap as noise at n 80, not as evidence.
- **Prefix cache.** The median prefix-hit fraction is 0.99, so "same prefix, warm cache" holds.
- **Routing.** Route recall was 76/80 and the false-tag rate 0/12.

---

## Stage 1 driver report (verbatim)
# Two-pass tool-call report (T-0080b driver)

- stage dev, commit 0dc91342a8c8bf33eb0a0a940bae123448873735, cells ['T1/named/perstep_mar', 'T2/named/perstep_mar', 'T4/named/perstep_mar'], n 48, system prompt sha256 fa5dace3bd1c9c89476c922c8e3c7bb4e8030d8a8fffc8ffd349982c903bbad3, tools sha256 bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707, cases sha256 460cd1c337dddf0cacc64cca149ec6af21a90899c42450ea5791fc3409afef85, calibration {'perstep_mar': {'path': '/home/user/demos/local_astra/.claude/worktrees/t0080c-launch/docs/cd/calibration-bfcl.json', 'sha256': '4084026c580a3148b4351699af64832e22b8e756fb288b551a17e642585ba74a', 'entropy_bound': 1.0, 'confidence_threshold': 0.005}}, thinking on, pass-1 max_tokens 1024, pass-2 {'T1': 1024, 'T2': 256, 'T4': 256}, concurrency 1, no seed, versions {'served_model': 'nvidia/diffusiongemma-26B-A4B-it-NVFP4', 'harness': 'local_astra.eval.cd (T-0070)', 'versions_lock_sha256': '5bab0570362d68c841f82f8d56cd58ddf1dd3253a2d1208f3ea2fe7109246f25', 'dgfa_commit': '27c7d0d3cc9c9e338d2fe4f6a1e64d3da01c8c03', 'compiler_version': 'dgfa_compile@27c7d0d+local_astra.T-0063;outlines-core==0.2.14', 'tokenizer_sha256': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f', 'model_revision': '2ea837236295d617ac27f8c17d61228081932c40'}
- stopped: no
- route recall (tool cases): 38/40 = 0.950 [0.835, 0.986]; false-tag rate (no-tool cases): 0/8 = 0.000 [0.000, 0.324]
- selection: {'rule': 'stage1', 'choice': 'T1/named/perstep_mar'}

| policy | exact | schema_valid | CS | pass-2 wall p50/p90 ms | e2e p50/p90 ms (tool) | e2e p50 ms (no-tool) | steps mean/p90 | prefix hit (pooled; median) |
|---|---|---|---|---|---|---|---|---|
| draft_only | 18/40 = 0.450 [0.307, 0.602] | 19/38 = 0.500 [0.348, 0.652] | n/a | n/a/n/a | 251.6/364.9 | 207.2 | n/a/n/a | n/a; n/a |
| cd_always:T1/named/perstep_mar | 21/40 = 0.525 [0.375, 0.671] | 38/38 = 1.000 [0.908, 1.000] | 38/38 = 1.000 [0.908, 1.000] | 799.5/1844.7 | 1053.1/1954.4 | 207.2 | 18.4/33.0 | 0.992; 0.993 |
| cd_always:T2/named/perstep_mar | 6/40 = 0.150 [0.071, 0.291] | 38/38 = 1.000 [0.908, 1.000] | 38/38 = 1.000 [0.908, 1.000] | 2671.2/2926.3 | 2816.6/3206.7 | 207.2 | 34.6/48.0 | 0.992; 0.993 |
| cd_always:T4/named/perstep_mar | 12/40 = 0.300 [0.181, 0.454] | 38/38 = 1.000 [0.908, 1.000] | 38/38 = 1.000 [0.908, 1.000] | 434.4/670.4 | 649.6/954.7 | 207.2 | 6.1/9.0 | 0.982; 0.982 |
| cd_on_invalid:T1/named/perstep_mar | 29/40 = 0.725 [0.572, 0.839] | 38/38 = 1.000 [0.908, 1.000] | 19/19 = 1.000 [0.832, 1.000] | 775.2/1943.6 | 406.9/1763.3 | 207.2 | 18.8/34.0 | 0.991; 0.992 |
| cd_on_invalid:T2/named/perstep_mar | 22/40 = 0.550 [0.398, 0.693] | 38/38 = 1.000 [0.908, 1.000] | 19/19 = 1.000 [0.832, 1.000] | 2675.8/2923.6 | 406.9/3172.7 | 207.2 | 33.3/48.0 | 0.991; 0.992 |
| cd_on_invalid:T4/named/perstep_mar | 26/40 = 0.650 [0.495, 0.779] | 38/38 = 1.000 [0.908, 1.000] | 19/19 = 1.000 [0.832, 1.000] | 431.5/574.5 | 372.8/825.9 | 207.2 | 5.2/8.0 | 0.980; 0.981 |

| policy | tool | exact | schema_valid | e2e p50 ms |
|---|---|---|---|---|
| draft_only | click_at | 5/10 = 0.500 [0.237, 0.763] | 6/8 = 0.750 [0.409, 0.929] | 290.3 |
| draft_only | google_search | 4/10 = 0.400 [0.168, 0.687] | 4/10 = 0.400 [0.168, 0.687] | 199.8 |
| draft_only | press_key | 1/5 = 0.200 [0.036, 0.624] | 1/5 = 0.200 [0.036, 0.624] | 224.3 |
| draft_only | scroll | 2/5 = 0.400 [0.118, 0.769] | 2/5 = 0.400 [0.118, 0.769] | 203.3 |
| draft_only | type_at | 6/10 = 0.600 [0.313, 0.832] | 6/10 = 0.600 [0.313, 0.832] | 273.7 |
| cd_always:T1/named/perstep_mar | click_at | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 956.4 |
| cd_always:T1/named/perstep_mar | google_search | 3/10 = 0.300 [0.108, 0.603] | 10/10 = 1.000 [0.722, 1.000] | 1421.0 |
| cd_always:T1/named/perstep_mar | press_key | 2/5 = 0.400 [0.118, 0.769] | 5/5 = 1.000 [0.566, 1.000] | 1062.0 |
| cd_always:T1/named/perstep_mar | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 821.5 |
| cd_always:T1/named/perstep_mar | type_at | 6/10 = 0.600 [0.313, 0.832] | 10/10 = 1.000 [0.722, 1.000] | 994.9 |
| cd_always:T2/named/perstep_mar | click_at | 0/10 = 0.000 [0.000, 0.278] | 8/8 = 1.000 [0.676, 1.000] | 2816.6 |
| cd_always:T2/named/perstep_mar | google_search | 3/10 = 0.300 [0.108, 0.603] | 10/10 = 1.000 [0.722, 1.000] | 2753.1 |
| cd_always:T2/named/perstep_mar | press_key | 0/5 = 0.000 [0.000, 0.434] | 5/5 = 1.000 [0.566, 1.000] | 1328.7 |
| cd_always:T2/named/perstep_mar | scroll | 3/5 = 0.600 [0.231, 0.882] | 5/5 = 1.000 [0.566, 1.000] | 1009.1 |
| cd_always:T2/named/perstep_mar | type_at | 0/10 = 0.000 [0.000, 0.278] | 10/10 = 1.000 [0.722, 1.000] | 3199.5 |
| cd_always:T4/named/perstep_mar | click_at | 2/10 = 0.200 [0.057, 0.510] | 8/8 = 1.000 [0.676, 1.000] | 845.5 |
| cd_always:T4/named/perstep_mar | google_search | 1/10 = 0.100 [0.018, 0.404] | 10/10 = 1.000 [0.722, 1.000] | 503.2 |
| cd_always:T4/named/perstep_mar | press_key | 1/5 = 0.200 [0.036, 0.624] | 5/5 = 1.000 [0.566, 1.000] | 638.7 |
| cd_always:T4/named/perstep_mar | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 467.5 |
| cd_always:T4/named/perstep_mar | type_at | 3/10 = 0.300 [0.108, 0.603] | 10/10 = 1.000 [0.722, 1.000] | 759.8 |
| cd_on_invalid:T1/named/perstep_mar | click_at | 6/10 = 0.600 [0.313, 0.832] | 8/8 = 1.000 [0.676, 1.000] | 290.3 |
| cd_on_invalid:T1/named/perstep_mar | google_search | 6/10 = 0.600 [0.313, 0.832] | 10/10 = 1.000 [0.722, 1.000] | 659.7 |
| cd_on_invalid:T1/named/perstep_mar | press_key | 3/5 = 0.600 [0.231, 0.882] | 5/5 = 1.000 [0.566, 1.000] | 1044.3 |
| cd_on_invalid:T1/named/perstep_mar | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 561.9 |
| cd_on_invalid:T1/named/perstep_mar | type_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 286.5 |
| cd_on_invalid:T2/named/perstep_mar | click_at | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 290.3 |
| cd_on_invalid:T2/named/perstep_mar | google_search | 6/10 = 0.600 [0.313, 0.832] | 10/10 = 1.000 [0.722, 1.000] | 978.0 |
| cd_on_invalid:T2/named/perstep_mar | press_key | 1/5 = 0.200 [0.036, 0.624] | 5/5 = 1.000 [0.566, 1.000] | 1217.7 |
| cd_on_invalid:T2/named/perstep_mar | scroll | 4/5 = 0.800 [0.376, 0.964] | 5/5 = 1.000 [0.566, 1.000] | 680.2 |
| cd_on_invalid:T2/named/perstep_mar | type_at | 6/10 = 0.600 [0.313, 0.832] | 10/10 = 1.000 [0.722, 1.000] | 286.5 |
| cd_on_invalid:T4/named/perstep_mar | click_at | 6/10 = 0.600 [0.313, 0.832] | 8/8 = 1.000 [0.676, 1.000] | 290.3 |
| cd_on_invalid:T4/named/perstep_mar | google_search | 5/10 = 0.500 [0.237, 0.763] | 10/10 = 1.000 [0.722, 1.000] | 417.7 |
| cd_on_invalid:T4/named/perstep_mar | press_key | 2/5 = 0.400 [0.118, 0.769] | 5/5 = 1.000 [0.566, 1.000] | 589.8 |
| cd_on_invalid:T4/named/perstep_mar | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 461.1 |
| cd_on_invalid:T4/named/perstep_mar | type_at | 8/10 = 0.800 [0.490, 0.943] | 10/10 = 1.000 [0.722, 1.000] | 286.5 |

McNemar (exact, tool cases; Bonferroni over 9):

- cd_always:T1/named/perstep_mar vs draft_only: n 40, b 11, c 8, p 0.6476 (adj 1); both parsed n 19, b 0, c 8, p 0.007812
- cd_always:T2/named/perstep_mar vs draft_only: n 40, b 4, c 16, p 0.01182 (adj 0.1064); both parsed n 19, b 0, c 16, p 3.052e-05
- cd_always:T4/named/perstep_mar vs draft_only: n 40, b 8, c 14, p 0.2863 (adj 1); both parsed n 19, b 0, c 14, p 0.0001221
- cd_on_invalid:T1/named/perstep_mar vs draft_only: n 40, b 11, c 0, p 0.0009766 (adj 0.008789); both parsed n 19, b 0, c 0, p 1
- cd_on_invalid:T2/named/perstep_mar vs draft_only: n 40, b 4, c 0, p 0.125 (adj 1); both parsed n 19, b 0, c 0, p 1
- cd_on_invalid:T4/named/perstep_mar vs draft_only: n 40, b 8, c 0, p 0.007812 (adj 0.07031); both parsed n 19, b 0, c 0, p 1
- cd_always:T1/named/perstep_mar vs cd_always:T2/named/perstep_mar: n 40, b 17, c 2, p 0.0007286 (adj 0.006557); both parsed n 38, b 17, c 2, p 0.0007286
- cd_always:T1/named/perstep_mar vs cd_always:T4/named/perstep_mar: n 40, b 11, c 2, p 0.02246 (adj 0.2021); both parsed n 38, b 11, c 2, p 0.02246
- cd_always:T2/named/perstep_mar vs cd_always:T4/named/perstep_mar: n 40, b 2, c 8, p 0.1094 (adj 0.9844); both parsed n 38, b 2, c 8, p 0.1094

T1 `101` position: {'T1/named/perstep_mar': {'102': 2, '109': 1, '116': 1, '123': 1, '138': 1, '17': 1, '18': 1, '317': 1, '38': 4, '41': 1, '42': 2, '43': 2, '44': 1, '45': 1, '49': 1, '51': 1, '57': 1, '61': 1, '63': 2, '65': 1, '69': 1, '74': 1, '80': 1, '81': 2, '87': 2, '96': 3, '99': 1}}

- T1/named/perstep_mar: pass 2 equals the valid draft 4/19 = 0.211 [0.085, 0.433]; invalid drafts repaired 19/19 = 1.000 [0.832, 1.000]; name-picker rows 0
- T2/named/perstep_mar: pass 2 equals the valid draft 1/19 = 0.053 [0.009, 0.246]; invalid drafts repaired 19/19 = 1.000 [0.832, 1.000]; name-picker rows 0
- T4/named/perstep_mar: pass 2 equals the valid draft 2/19 = 0.105 [0.029, 0.314]; invalid drafts repaired 19/19 = 1.000 [0.832, 1.000]; name-picker rows 0

---

## Stage 2 driver report (verbatim)
# Two-pass tool-call report (T-0080b driver)

- stage test, commit 0dc91342a8c8bf33eb0a0a940bae123448873735, cells ['T1/named/end_projection', 'T1/named/perstep_mar', 'T1/named/perstep_mar_qargmax', 'T1/named/perstep_mar+seed'], n 92, system prompt sha256 fa5dace3bd1c9c89476c922c8e3c7bb4e8030d8a8fffc8ffd349982c903bbad3, tools sha256 bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707, cases sha256 460cd1c337dddf0cacc64cca149ec6af21a90899c42450ea5791fc3409afef85, calibration {'perstep_mar': {'path': '/home/user/demos/local_astra/.claude/worktrees/t0080c-launch/docs/cd/calibration-bfcl.json', 'sha256': '4084026c580a3148b4351699af64832e22b8e756fb288b551a17e642585ba74a', 'entropy_bound': 1.0, 'confidence_threshold': 0.005}, 'perstep_mar_qargmax': {'path': '/home/user/demos/local_astra/.claude/worktrees/t0080c-launch/docs/cd/calibration-bfcl-perstep_mar_qargmax.json', 'sha256': 'cb81bea43c274a446a01a34187d8d1a6e2b484eb4c417f82f90c4462e54ae086', 'entropy_bound': 0.3, 'confidence_threshold': 0.005}}, thinking on, pass-1 max_tokens 1024, pass-2 {'T1': 1024, 'T2': 256, 'T4': 256}, concurrency 1, no seed, versions {'served_model': 'nvidia/diffusiongemma-26B-A4B-it-NVFP4', 'harness': 'local_astra.eval.cd (T-0070)', 'versions_lock_sha256': '5bab0570362d68c841f82f8d56cd58ddf1dd3253a2d1208f3ea2fe7109246f25', 'dgfa_commit': '27c7d0d3cc9c9e338d2fe4f6a1e64d3da01c8c03', 'compiler_version': 'dgfa_compile@27c7d0d+local_astra.T-0063;outlines-core==0.2.14', 'tokenizer_sha256': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f', 'model_revision': '2ea837236295d617ac27f8c17d61228081932c40'}
- stopped: no
- route recall (tool cases): 76/80 = 0.950 [0.878, 0.980]; false-tag rate (no-tool cases): 0/12 = 0.000 [0.000, 0.243]
- selection: {'rule': 'stage2', 'choice': 'cd_on_invalid:T1/named/perstep_mar+seed'}

| policy | exact | schema_valid | CS | pass-2 wall p50/p90 ms | e2e p50/p90 ms (tool) | e2e p50 ms (no-tool) | steps mean/p90 | prefix hit (pooled; median) |
|---|---|---|---|---|---|---|---|---|
| draft_only | 24/80 = 0.300 [0.211, 0.408] | 27/76 = 0.355 [0.257, 0.467] | n/a | n/a/n/a | 247.9/343.6 | 219.0 | n/a/n/a | n/a; n/a |
| cd_always:T1/named/end_projection | 57/80 = 0.713 [0.605, 0.800] | 76/76 = 1.000 [0.952, 1.000] | 76/76 = 1.000 [0.952, 1.000] | 299.8/391.9 | 544.4/712.7 | 219.0 | 9.0/12.0 | 0.992; 0.994 |
| cd_always:T1/named/perstep_mar | 51/80 = 0.637 [0.528, 0.734] | 76/76 = 1.000 [0.952, 1.000] | 76/76 = 1.000 [0.952, 1.000] | 649.2/1530.7 | 903.4/1659.5 | 219.0 | 14.6/25.0 | 0.992; 0.994 |
| cd_always:T1/named/perstep_mar_qargmax | 51/80 = 0.637 [0.528, 0.734] | 76/76 = 1.000 [0.952, 1.000] | 76/76 = 1.000 [0.952, 1.000] | 622.5/1337.2 | 898.1/1542.3 | 219.0 | 14.4/26.0 | 0.992; 0.994 |
| cd_always:T1/named/perstep_mar+seed | 58/80 = 0.725 [0.619, 0.811] | 76/76 = 1.000 [0.952, 1.000] | 76/76 = 1.000 [0.952, 1.000] | 225.4/291.3 | 478.5/634.8 | 219.0 | 3.1/4.0 | 0.992; 0.994 |
| cd_on_invalid:T1/named/end_projection | 60/80 = 0.750 [0.645, 0.832] | 76/76 = 1.000 [0.952, 1.000] | 49/49 = 1.000 [0.927, 1.000] | 308.8/394.0 | 445.1/697.2 | 219.0 | 8.7/12.0 | 0.992; 0.993 |
| cd_on_invalid:T1/named/perstep_mar | 59/80 = 0.738 [0.632, 0.821] | 76/76 = 1.000 [0.952, 1.000] | 49/49 = 1.000 [0.927, 1.000] | 665.4/1541.5 | 750.4/1385.9 | 219.0 | 15.0/27.0 | 0.992; 0.993 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | 58/80 = 0.725 [0.619, 0.811] | 76/76 = 1.000 [0.952, 1.000] | 49/49 = 1.000 [0.927, 1.000] | 619.5/1337.2 | 723.6/1258.3 | 219.0 | 14.1/26.0 | 0.992; 0.993 |
| cd_on_invalid:T1/named/perstep_mar+seed | 58/80 = 0.725 [0.619, 0.811] | 76/76 = 1.000 [0.952, 1.000] | 49/49 = 1.000 [0.927, 1.000] | 232.9/334.9 | 426.2/609.1 | 219.0 | 3.4/5.0 | 0.992; 0.993 |

| policy | tool | exact | schema_valid | e2e p50 ms |
|---|---|---|---|---|
| draft_only | click_at | 10/20 = 0.500 [0.299, 0.701] | 11/20 = 0.550 [0.342, 0.742] | 287.4 |
| draft_only | google_search | 3/20 = 0.150 [0.052, 0.360] | 3/18 = 0.167 [0.058, 0.392] | 191.6 |
| draft_only | press_key | 1/10 = 0.100 [0.018, 0.404] | 3/8 = 0.375 [0.137, 0.694] | 262.1 |
| draft_only | scroll | 7/10 = 0.700 [0.397, 0.892] | 7/10 = 0.700 [0.397, 0.892] | 202.4 |
| draft_only | type_at | 3/20 = 0.150 [0.052, 0.360] | 3/20 = 0.150 [0.052, 0.360] | 279.4 |
| cd_always:T1/named/end_projection | click_at | 17/20 = 0.850 [0.640, 0.948] | 20/20 = 1.000 [0.839, 1.000] | 638.8 |
| cd_always:T1/named/end_projection | google_search | 12/20 = 0.600 [0.387, 0.781] | 18/18 = 1.000 [0.824, 1.000] | 411.4 |
| cd_always:T1/named/end_projection | press_key | 3/10 = 0.300 [0.108, 0.603] | 8/8 = 1.000 [0.676, 1.000] | 562.7 |
| cd_always:T1/named/end_projection | scroll | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 482.4 |
| cd_always:T1/named/end_projection | type_at | 15/20 = 0.750 [0.531, 0.888] | 20/20 = 1.000 [0.839, 1.000] | 624.3 |
| cd_always:T1/named/perstep_mar | click_at | 10/20 = 0.500 [0.299, 0.701] | 20/20 = 1.000 [0.839, 1.000] | 903.3 |
| cd_always:T1/named/perstep_mar | google_search | 15/20 = 0.750 [0.531, 0.888] | 18/18 = 1.000 [0.824, 1.000] | 764.0 |
| cd_always:T1/named/perstep_mar | press_key | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 891.7 |
| cd_always:T1/named/perstep_mar | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 850.0 |
| cd_always:T1/named/perstep_mar | type_at | 12/20 = 0.600 [0.387, 0.781] | 20/20 = 1.000 [0.839, 1.000] | 1002.0 |
| cd_always:T1/named/perstep_mar_qargmax | click_at | 14/20 = 0.700 [0.481, 0.855] | 20/20 = 1.000 [0.839, 1.000] | 996.8 |
| cd_always:T1/named/perstep_mar_qargmax | google_search | 9/20 = 0.450 [0.258, 0.658] | 18/18 = 1.000 [0.824, 1.000] | 723.6 |
| cd_always:T1/named/perstep_mar_qargmax | press_key | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 905.3 |
| cd_always:T1/named/perstep_mar_qargmax | scroll | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 723.7 |
| cd_always:T1/named/perstep_mar_qargmax | type_at | 13/20 = 0.650 [0.433, 0.819] | 20/20 = 1.000 [0.839, 1.000] | 958.3 |
| cd_always:T1/named/perstep_mar+seed | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 495.4 |
| cd_always:T1/named/perstep_mar+seed | google_search | 11/20 = 0.550 [0.342, 0.742] | 18/18 = 1.000 [0.824, 1.000] | 387.4 |
| cd_always:T1/named/perstep_mar+seed | press_key | 3/10 = 0.300 [0.108, 0.603] | 8/8 = 1.000 [0.676, 1.000] | 595.4 |
| cd_always:T1/named/perstep_mar+seed | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 433.2 |
| cd_always:T1/named/perstep_mar+seed | type_at | 17/20 = 0.850 [0.640, 0.948] | 20/20 = 1.000 [0.839, 1.000] | 508.0 |
| cd_on_invalid:T1/named/end_projection | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 382.2 |
| cd_on_invalid:T1/named/end_projection | google_search | 13/20 = 0.650 [0.433, 0.819] | 18/18 = 1.000 [0.824, 1.000] | 398.6 |
| cd_on_invalid:T1/named/end_projection | press_key | 3/10 = 0.300 [0.108, 0.603] | 8/8 = 1.000 [0.676, 1.000] | 516.9 |
| cd_on_invalid:T1/named/end_projection | scroll | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 238.0 |
| cd_on_invalid:T1/named/end_projection | type_at | 16/20 = 0.800 [0.584, 0.919] | 20/20 = 1.000 [0.839, 1.000] | 598.5 |
| cd_on_invalid:T1/named/perstep_mar | click_at | 15/20 = 0.750 [0.531, 0.888] | 20/20 = 1.000 [0.839, 1.000] | 382.2 |
| cd_on_invalid:T1/named/perstep_mar | google_search | 16/20 = 0.800 [0.584, 0.919] | 18/18 = 1.000 [0.824, 1.000] | 650.0 |
| cd_on_invalid:T1/named/perstep_mar | press_key | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 763.4 |
| cd_on_invalid:T1/named/perstep_mar | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 238.0 |
| cd_on_invalid:T1/named/perstep_mar | type_at | 14/20 = 0.700 [0.481, 0.855] | 20/20 = 1.000 [0.839, 1.000] | 1002.0 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | click_at | 17/20 = 0.850 [0.640, 0.948] | 20/20 = 1.000 [0.839, 1.000] | 382.2 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | google_search | 12/20 = 0.600 [0.387, 0.781] | 18/18 = 1.000 [0.824, 1.000] | 688.6 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | press_key | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 726.4 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | scroll | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 238.0 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | type_at | 14/20 = 0.700 [0.481, 0.855] | 20/20 = 1.000 [0.839, 1.000] | 958.3 |
| cd_on_invalid:T1/named/perstep_mar+seed | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 382.2 |
| cd_on_invalid:T1/named/perstep_mar+seed | google_search | 11/20 = 0.550 [0.342, 0.742] | 18/18 = 1.000 [0.824, 1.000] | 377.7 |
| cd_on_invalid:T1/named/perstep_mar+seed | press_key | 3/10 = 0.300 [0.108, 0.603] | 8/8 = 1.000 [0.676, 1.000] | 514.4 |
| cd_on_invalid:T1/named/perstep_mar+seed | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 238.0 |
| cd_on_invalid:T1/named/perstep_mar+seed | type_at | 17/20 = 0.850 [0.640, 0.948] | 20/20 = 1.000 [0.839, 1.000] | 491.8 |

McNemar (exact, tool cases; Bonferroni over 14):

- cd_always:T1/named/end_projection vs draft_only: n 80, b 36, c 3, p 3.609e-08 (adj 5.052e-07); both parsed n 27, b 0, c 3, p 0.25
- cd_always:T1/named/perstep_mar vs draft_only: n 80, b 35, c 8, p 4.193e-05 (adj 0.0005871); both parsed n 27, b 0, c 8, p 0.007812
- cd_always:T1/named/perstep_mar_qargmax vs draft_only: n 80, b 34, c 7, p 2.532e-05 (adj 0.0003545); both parsed n 27, b 0, c 7, p 0.01562
- cd_always:T1/named/perstep_mar+seed vs draft_only: n 80, b 34, c 0, p 1.164e-10 (adj 1.63e-09); both parsed n 27, b 0, c 0, p 1
- cd_on_invalid:T1/named/end_projection vs draft_only: n 80, b 36, c 0, p 2.91e-11 (adj 4.075e-10); both parsed n 27, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar vs draft_only: n 80, b 35, c 0, p 5.821e-11 (adj 8.149e-10); both parsed n 27, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar_qargmax vs draft_only: n 80, b 34, c 0, p 1.164e-10 (adj 1.63e-09); both parsed n 27, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar+seed vs draft_only: n 80, b 34, c 0, p 1.164e-10 (adj 1.63e-09); both parsed n 27, b 0, c 0, p 1
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar: n 80, b 16, c 10, p 0.3269 (adj 1); both parsed n 76, b 16, c 10, p 0.3269
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar_qargmax: n 80, b 13, c 7, p 0.2632 (adj 1); both parsed n 76, b 13, c 7, p 0.2632
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar+seed: n 80, b 6, c 7, p 1 (adj 1); both parsed n 76, b 6, c 7, p 1
- cd_always:T1/named/perstep_mar vs cd_always:T1/named/perstep_mar_qargmax: n 80, b 11, c 11, p 1 (adj 1); both parsed n 76, b 11, c 11, p 1
- cd_always:T1/named/perstep_mar vs cd_always:T1/named/perstep_mar+seed: n 80, b 9, c 16, p 0.2295 (adj 1); both parsed n 76, b 9, c 16, p 0.2295
- cd_always:T1/named/perstep_mar_qargmax vs cd_always:T1/named/perstep_mar+seed: n 80, b 8, c 15, p 0.21 (adj 1); both parsed n 76, b 8, c 15, p 0.21

T1 `101` position: {'T1/named/end_projection': {'100': 1, '101': 1, '104': 1, '105': 1, '106': 1, '111': 1, '113': 1, '114': 1, '115': 1, '130': 1, '154': 1, '18': 2, '19': 1, '20': 1, '35': 1, '356': 1, '38': 1, '40': 1, '41': 1, '42': 2, '44': 3, '45': 3, '46': 4, '52': 1, '53': 2, '55': 1, '56': 1, '59': 3, '60': 2, '61': 4, '62': 1, '63': 1, '64': 1, '65': 2, '67': 1, '69': 2, '70': 1, '71': 2, '72': 1, '74': 1, '79': 1, '82': 3, '83': 1, '84': 1, '87': 1, '90': 3, '91': 2, '92': 1, '94': 1, '95': 1, '96': 1, '99': 1}, 'T1/named/perstep_mar': {'101': 2, '102': 1, '103': 1, '105': 2, '106': 1, '107': 1, '110': 1, '111': 1, '113': 1, '114': 2, '129': 1, '145': 1, '161': 1, '18': 2, '19': 3, '20': 1, '21': 1, '22': 1, '35': 1, '38': 2, '39': 1, '41': 1, '43': 1, '44': 3, '45': 3, '47': 2, '50': 1, '54': 2, '55': 3, '56': 1, '59': 2, '62': 1, '65': 4, '69': 2, '70': 3, '71': 2, '72': 1, '74': 1, '75': 2, '78': 1, '82': 1, '86': 1, '87': 2, '91': 2, '93': 2, '94': 3, '96': 1}, 'T1/named/perstep_mar_qargmax': {'100': 1, '101': 3, '103': 2, '105': 1, '109': 2, '114': 1, '115': 1, '116': 2, '119': 1, '123': 1, '137': 1, '159': 1, '165': 1, '17': 1, '18': 1, '19': 1, '256': 1, '36': 2, '38': 1, '40': 1, '42': 1, '43': 1, '44': 4, '45': 2, '47': 2, '48': 1, '49': 1, '50': 2, '52': 1, '54': 1, '55': 1, '57': 2, '58': 1, '59': 3, '61': 1, '63': 2, '64': 1, '65': 1, '67': 1, '70': 3, '72': 2, '73': 1, '75': 1, '76': 1, '80': 1, '81': 1, '83': 2, '86': 1, '89': 1, '90': 1, '91': 2, '92': 1, '95': 1, '96': 1, '97': 1}, 'T1/named/perstep_mar+seed': {'104': 1, '108': 1, '121': 1, '132': 1, '146': 1, '159': 1, '160': 1, '27': 1, '28': 1, '29': 1, '30': 4, '31': 10, '32': 3, '33': 1, '34': 1, '35': 2, '36': 2, '38': 2, '43': 1, '46': 1, '47': 1, '49': 1, '51': 1, '53': 1, '54': 1, '55': 1, '56': 1, '57': 1, '58': 3, '59': 1, '60': 2, '62': 1, '64': 1, '65': 1, '69': 2, '70': 1, '74': 1, '75': 1, '76': 2, '78': 1, '79': 2, '83': 1, '84': 1, '85': 1, '87': 1, '90': 2, '91': 1, '94': 2, '95': 1, '96': 1, '99': 1}}

- T1/named/end_projection: pass 2 equals the valid draft 10/27 = 0.370 [0.215, 0.558]; invalid drafts repaired 49/49 = 1.000 [0.927, 1.000]; name-picker rows 1
- T1/named/perstep_mar: pass 2 equals the valid draft 10/27 = 0.370 [0.215, 0.558]; invalid drafts repaired 49/49 = 1.000 [0.927, 1.000]; name-picker rows 1
- T1/named/perstep_mar_qargmax: pass 2 equals the valid draft 9/27 = 0.333 [0.186, 0.522]; invalid drafts repaired 49/49 = 1.000 [0.927, 1.000]; name-picker rows 1
- T1/named/perstep_mar+seed: pass 2 equals the valid draft 25/27 = 0.926 [0.766, 0.979]; invalid drafts repaired 49/49 = 1.000 [0.927, 1.000]; name-picker rows 1
- seed T1/named/perstep_mar+seed: seeded 75/76 routed cases; no seed {'picker': 1}
- seed copy T1/named/perstep_mar+seed (first canvas equals the seed through its first stop): valid drafts 20/27 = 0.741 [0.553, 0.868]; invalid drafts 0/48 = 0.000 [0.000, 0.074]
- seed trim T1/named/perstep_mar+seed (seeded rows whose draft lost trailing whitespace, R2): 0/75 = 0.000 [0.000, 0.049]
