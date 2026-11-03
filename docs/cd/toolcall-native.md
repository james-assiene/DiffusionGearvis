# Native Gemma 4 tool-call format vs `<cc>` (T-0084e, T-0084f)

## Dev pilot (T-0084e, run 2 at 794161d, 2026-10-11) — winner: native
**Regime.**
- **Engine and model:** lean profile, engine only. nvidia/diffusiongemma-26B-A4B-it-NVFP4 rev 2ea8372, vLLM v0.29.0 (versions.lock
  5bab0570…), RTX 5090, driver 595.84, gpu-mem 0.86. The plugin and `services/vllm.sh` are unchanged since 0dc9134.
- **Decoding:** CL 256, 48 steps, thinking on, prefix caching on, canvas tap on.
- **Split and run shape:** dev 40 tool + 8 no-tool cases, both formats with a fresh pass 1 in one boot, alternated per case (24/24), 16 cells,
  n = 1 run, run_id 701e00a6d7f9.
- **Prompts:** native prompt sha e0b54a58… (approved at H-T84-0), tools field 9bfd7685…, `tool_choice "none"`. The cc prompt is fa5dace3….
- **Knobs:** as T-0080c/T-0082c.
- **Evidence:** `logs/t0084-native-dev-794161d/` (artifact sha 06a71fc7…, selection sha 7bb5c4de…).
- **Caveat:** dev pilot, n 40; not a held-out result.

**Run 1 (9aee66d)** stopped after 11 cases on an HTTP 400: a stray `<channel|>` in the native spoken prefix broke the T4 continuation. T-0085
fixed this. The partial (31f9fd7f…) is evidence only.

**Decision (T-0084d criterion 1, re-derived independently by two reviewers):**
- **Step 1 picks θ_n = T4.** Native perstep_mar exact: T1 37/40, T2 13/40, T4 33/40. The bar is 33/40, which keeps T1 and T4, and T4's
  median pass-2 wall (443.8 ms) is the only one within 5 % of the fastest. **θ_n sits exactly on the inclusive bar:** dropping any one case
  where T1 and T4 are both exact flips θ_n to T1. This affects only the native constrained cells, not the winning policy.
- **Step 2 picks `draft_only:native`, winner native.**
  - The best eligible exact is 39/40, so the bar is 35/40.
  - The best cc policy is 32/40 (`cd_on_invalid:T1/named/{end_projection, perstep_mar}`). The cc draft alone is ineligible (schema 34/37).
  - The kept native policies tie at e2e p50/p90 224.0/315.9 ms; kind order picks `draft_only`.

| | native | `<cc>` |
|---|---|---|
| route recall (own tag; any tag) | 40/40 | 37/40 |
| draft schema-valid | 40/40 | 34/37 |
| draft exact | 39/40 | 31/40 |
| best policy exact | 39/40 (`draft_only:native`) | 32/40 |
| e2e p50/p90, tool cases | 224.0/315.9 ms | 276.8/371.8 ms (cd_on_invalid end_projection) |
| draft parse failures | 0 | 3 (`no arguments` 2, `not JSON` 1; all google_search without the `arguments` wrapper) |
| false tags (no-tool) | 0/8 | 0/8 |
| spoken sentence before the call | **27/40** (median 47 chars) | 37/37 (median 49 chars) |
| median prompt tokens | 1071.5 | 1538.5 |

**Findings:**
- **Constrained decoding adds nothing for native on dev.** Every native draft was valid, so `cd_on_invalid:native/*` is identical to the
  draft. CD stays as a safety net.
- **Product finding (Q6).** Native skips the spoken sentence on 13/40 tool calls, including all 10 google_search calls, although the
  prompt asks for one.
  **Delta (2026-10-16, H-T104, docs/01 D16):** not a defect for routine steps; D16 supersedes the target "a spoken sentence before every tool call" for them. The results, the selection and the safety rows are unchanged.
- **Dependency on T-0085.** 23/48 native pass-1 texts contain the stray `<channel|>` (`…<channel|>spoken<channel|><|tool_call>…`), and the
  spoken prefix is clean only because T-0085's router removes it. The live voice path must use the same router.
- **Statistics.** The cross-format McNemar (exact, paired): b 0, c 8, raw p 0.0078, Bonferroni (×105) 0.82. Reported only.
- **Guarantees.**
  - CS host replay: 628/628, with tampered controls rejected.
  - Native prompts carry 5 declarations each, and T1/T2 keep the pass-1 prompt.
  - All 160 native T4 continuations are clean, including click-01.
  - Joins are 1:1.
  - No preemption, OOM or errors. The JIT spike on the first case does not change the selection.

**H-T84 / H-T84-1 (human, 2026-10-11):** run the test split (T-0084f), its third use, run once, with the pre-registered margin of 0.05.

---

## Dev driver report (verbatim)
# Two-pass tool-call report (T-0080b driver; T-0084d format comparison)

**dev pilot; not a held-out result**

- stage dev, commit 794161d963cbc63553aa111f545542abe8eafa4d, cells ['T1/named/end_projection', 'T1/named/perstep_mar', 'T1/named/perstep_mar_qargmax', 'T1/named/perstep_mar+seed', 'native/T1/named/end_projection', 'native/T1/named/perstep_mar', 'native/T1/named/perstep_mar_qargmax', 'native/T1/named/perstep_mar+seed', 'native/T2/named/end_projection', 'native/T2/named/perstep_mar', 'native/T2/named/perstep_mar_qargmax', 'native/T2/named/perstep_mar+seed', 'native/T4/named/end_projection', 'native/T4/named/perstep_mar', 'native/T4/named/perstep_mar_qargmax', 'native/T4/named/perstep_mar+seed'], n 48, system prompt sha256 fa5dace3bd1c9c89476c922c8e3c7bb4e8030d8a8fffc8ffd349982c903bbad3, tools sha256 bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707, cases sha256 460cd1c337dddf0cacc64cca149ec6af21a90899c42450ea5791fc3409afef85, calibration {'perstep_mar': {'path': '/home/user/demos/local_astra/.claude/worktrees/t0084e-launch2/docs/cd/calibration-bfcl.json', 'sha256': '4084026c580a3148b4351699af64832e22b8e756fb288b551a17e642585ba74a', 'entropy_bound': 1.0, 'confidence_threshold': 0.005}, 'perstep_mar_qargmax': {'path': '/home/user/demos/local_astra/.claude/worktrees/t0084e-launch2/docs/cd/calibration-bfcl-perstep_mar_qargmax.json', 'sha256': 'cb81bea43c274a446a01a34187d8d1a6e2b484eb4c417f82f90c4462e54ae086', 'entropy_bound': 0.3, 'confidence_threshold': 0.005}}, thinking on, pass-1 max_tokens 1024, pass-2 {'T1': 1024, 'T2': 256, 'T4': 256}, concurrency 1, no seed, versions {'served_model': 'nvidia/diffusiongemma-26B-A4B-it-NVFP4', 'harness': 'local_astra.eval.cd (T-0070)', 'versions_lock_sha256': '5bab0570362d68c841f82f8d56cd58ddf1dd3253a2d1208f3ea2fe7109246f25', 'dgfa_commit': '27c7d0d3cc9c9e338d2fe4f6a1e64d3da01c8c03', 'compiler_version': 'dgfa_compile@27c7d0d+local_astra.T-0063;outlines-core==0.2.14', 'tokenizer_sha256': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f', 'model_revision': '2ea837236295d617ac27f8c17d61228081932c40'}
- stopped: no
- formats: cc, native, paired by case in one boot. The formats differ jointly in the prompt instructions, where the declarations live, the call syntax and the grammar (T-0084).
- format winner: native (policy draft_only:native; native thought θ_n T4; rule T-0084d criterion 1, step-2 tolerance 1/10)

| policy | exact | schema_valid | CS | pass-2 wall p50/p90 ms | e2e p50/p90 ms (tool) | e2e p50 ms (no-tool) | steps mean/p90 | prefix hit (pooled; median) |
|---|---|---|---|---|---|---|---|---|
| draft_only | 31/40 = 0.775 [0.625, 0.877] | 34/37 = 0.919 [0.787, 0.972] | n/a | n/a/n/a | 271.3/346.3 | 230.4 | n/a/n/a | n/a; n/a |
| draft_only:native | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | n/a | n/a/n/a | 224.0/315.9 | 210.8 | n/a/n/a | n/a; n/a |
| cd_always:T1/named/end_projection | 28/40 = 0.700 [0.546, 0.819] | 37/37 = 1.000 [0.906, 1.000] | 37/37 = 1.000 [0.906, 1.000] | 292.2/333.1 | 555.0/656.8 | 230.4 | 7.9/10.0 | 0.992; 0.994 |
| cd_always:T1/named/perstep_mar | 21/40 = 0.525 [0.375, 0.671] | 37/37 = 1.000 [0.906, 1.000] | 37/37 = 1.000 [0.906, 1.000] | 745.5/1721.2 | 989.8/1844.0 | 230.4 | 16.6/32.0 | 0.992; 0.994 |
| cd_always:T1/named/perstep_mar_qargmax | 23/40 = 0.575 [0.422, 0.715] | 37/37 = 1.000 [0.906, 1.000] | 37/37 = 1.000 [0.906, 1.000] | 696.0/1380.9 | 1020.4/1604.6 | 230.4 | 17.1/28.0 | 0.992; 0.994 |
| cd_always:T1/named/perstep_mar+seed | 31/40 = 0.775 [0.625, 0.877] | 37/37 = 1.000 [0.906, 1.000] | 37/37 = 1.000 [0.906, 1.000] | 191.5/276.4 | 461.1/553.2 | 230.4 | 2.3/4.0 | 0.992; 0.994 |
| cd_always:native/T1/named/end_projection | 38/40 = 0.950 [0.835, 0.986] | 40/40 = 1.000 [0.912, 1.000] | 40/40 = 1.000 [0.912, 1.000] | 263.6/355.2 | 512.4/614.3 | 210.8 | 7.5/11.0 | 0.982; 0.984 |
| cd_always:native/T1/named/perstep_mar | 37/40 = 0.925 [0.801, 0.974] | 40/40 = 1.000 [0.912, 1.000] | 40/40 = 1.000 [0.912, 1.000] | 521.0/1149.0 | 751.1/1452.3 | 210.8 | 12.2/24.0 | 0.982; 0.984 |
| cd_always:native/T1/named/perstep_mar_qargmax | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | 40/40 = 1.000 [0.912, 1.000] | 651.2/930.9 | 871.9/1246.8 | 210.8 | 11.8/19.0 | 0.982; 0.984 |
| cd_always:native/T1/named/perstep_mar+seed | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | 40/40 = 1.000 [0.912, 1.000] | 183.4/194.2 | 415.8/502.0 | 210.8 | 2.0/2.0 | 0.982; 0.984 |
| cd_always:native/T2/named/end_projection | 11/40 = 0.275 [0.161, 0.428] | 40/40 = 1.000 [0.912, 1.000] | 40/40 = 1.000 [0.912, 1.000] | 287.7/359.4 | 513.8/702.0 | 210.8 | 7.5/10.0 | 0.982; 0.984 |
| cd_always:native/T2/named/perstep_mar | 13/40 = 0.325 [0.201, 0.480] | 40/40 = 1.000 [0.912, 1.000] | 40/40 = 1.000 [0.912, 1.000] | 1219.6/2614.5 | 1486.2/2850.9 | 210.8 | 25.1/48.0 | 0.982; 0.984 |
| cd_always:native/T2/named/perstep_mar_qargmax | 13/40 = 0.325 [0.201, 0.480] | 40/40 = 1.000 [0.912, 1.000] | 40/40 = 1.000 [0.912, 1.000] | 1642.1/2641.4 | 1825.8/2940.0 | 210.8 | 29.5/48.0 | 0.982; 0.984 |
| cd_always:native/T2/named/perstep_mar+seed | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | 40/40 = 1.000 [0.912, 1.000] | 209.7/222.6 | 428.0/528.6 | 210.8 | 2.0/2.0 | 0.982; 0.984 |
| cd_always:native/T4/named/end_projection | 33/40 = 0.825 [0.680, 0.913] | 40/40 = 1.000 [0.912, 1.000] | 40/40 = 1.000 [0.912, 1.000] | 198.4/289.3 | 436.3/592.3 | 210.8 | 5.6/8.0 | 0.976; 0.975 |
| cd_always:native/T4/named/perstep_mar | 33/40 = 0.825 [0.680, 0.913] | 40/40 = 1.000 [0.912, 1.000] | 40/40 = 1.000 [0.912, 1.000] | 443.8/1536.0 | 701.3/1662.1 | 210.8 | 9.8/27.0 | 0.981; 0.978 |
| cd_always:native/T4/named/perstep_mar_qargmax | 34/40 = 0.850 [0.709, 0.929] | 40/40 = 1.000 [0.912, 1.000] | 40/40 = 1.000 [0.912, 1.000] | 419.7/1792.2 | 735.1/1944.2 | 210.8 | 11.9/32.0 | 0.981; 0.979 |
| cd_always:native/T4/named/perstep_mar+seed | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | 40/40 = 1.000 [0.912, 1.000] | 208.1/220.2 | 441.5/529.2 | 210.8 | 2.0/2.0 | 0.980; 0.977 |
| cd_on_invalid:T1/named/end_projection | 32/40 = 0.800 [0.652, 0.895] | 37/37 = 1.000 [0.906, 1.000] | 3/3 = 1.000 [0.438, 1.000] | 221.9/278.1 | 276.8/371.8 | 230.4 | 6.3/8.0 | 0.984; 0.984 |
| cd_on_invalid:T1/named/perstep_mar | 32/40 = 0.800 [0.652, 0.895] | 37/37 = 1.000 [0.906, 1.000] | 3/3 = 1.000 [0.438, 1.000] | 909.6/916.2 | 276.8/403.8 | 230.4 | 14.7/18.0 | 0.984; 0.984 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | 31/40 = 0.775 [0.625, 0.877] | 37/37 = 1.000 [0.906, 1.000] | 3/3 = 1.000 [0.438, 1.000] | 1333.5/1380.9 | 276.8/403.8 | 230.4 | 22.0/28.0 | 0.984; 0.984 |
| cd_on_invalid:T1/named/perstep_mar+seed | 31/40 = 0.775 [0.625, 0.877] | 37/37 = 1.000 [0.906, 1.000] | 3/3 = 1.000 [0.438, 1.000] | 191.5/277.2 | 276.8/354.3 | 230.4 | 2.7/4.0 | 0.984; 0.984 |
| cd_on_invalid:native/T1/named/end_projection | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | n/a | n/a/n/a | 224.0/315.9 | 210.8 | n/a/n/a | n/a; n/a |
| cd_on_invalid:native/T1/named/perstep_mar | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | n/a | n/a/n/a | 224.0/315.9 | 210.8 | n/a/n/a | n/a; n/a |
| cd_on_invalid:native/T1/named/perstep_mar_qargmax | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | n/a | n/a/n/a | 224.0/315.9 | 210.8 | n/a/n/a | n/a; n/a |
| cd_on_invalid:native/T1/named/perstep_mar+seed | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | n/a | n/a/n/a | 224.0/315.9 | 210.8 | n/a/n/a | n/a; n/a |
| cd_on_invalid:native/T2/named/end_projection | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | n/a | n/a/n/a | 224.0/315.9 | 210.8 | n/a/n/a | n/a; n/a |
| cd_on_invalid:native/T2/named/perstep_mar | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | n/a | n/a/n/a | 224.0/315.9 | 210.8 | n/a/n/a | n/a; n/a |
| cd_on_invalid:native/T2/named/perstep_mar_qargmax | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | n/a | n/a/n/a | 224.0/315.9 | 210.8 | n/a/n/a | n/a; n/a |
| cd_on_invalid:native/T2/named/perstep_mar+seed | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | n/a | n/a/n/a | 224.0/315.9 | 210.8 | n/a/n/a | n/a; n/a |
| cd_on_invalid:native/T4/named/end_projection | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | n/a | n/a/n/a | 224.0/315.9 | 210.8 | n/a/n/a | n/a; n/a |
| cd_on_invalid:native/T4/named/perstep_mar | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | n/a | n/a/n/a | 224.0/315.9 | 210.8 | n/a/n/a | n/a; n/a |
| cd_on_invalid:native/T4/named/perstep_mar_qargmax | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | n/a | n/a/n/a | 224.0/315.9 | 210.8 | n/a/n/a | n/a; n/a |
| cd_on_invalid:native/T4/named/perstep_mar+seed | 39/40 = 0.975 [0.871, 0.996] | 40/40 = 1.000 [0.912, 1.000] | n/a | n/a/n/a | 224.0/315.9 | 210.8 | n/a/n/a | n/a; n/a |

| policy | tool | exact | schema_valid | e2e p50 ms |
|---|---|---|---|---|
| draft_only | click_at | 7/10 = 0.700 [0.397, 0.892] | 9/9 = 1.000 [0.701, 1.000] | 281.5 |
| draft_only | google_search | 6/10 = 0.600 [0.313, 0.832] | 6/9 = 0.667 [0.354, 0.879] | 191.8 |
| draft_only | press_key | 4/5 = 0.800 [0.376, 0.964] | 5/5 = 1.000 [0.566, 1.000] | 267.7 |
| draft_only | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 225.4 |
| draft_only | type_at | 9/10 = 0.900 [0.596, 0.982] | 9/9 = 1.000 [0.701, 1.000] | 306.3 |
| draft_only:native | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 256.1 |
| draft_only:native | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 144.2 |
| draft_only:native | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 202.1 |
| draft_only:native | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 160.0 |
| draft_only:native | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 276.2 |
| cd_always:T1/named/end_projection | click_at | 7/10 = 0.700 [0.397, 0.892] | 9/9 = 1.000 [0.701, 1.000] | 605.8 |
| cd_always:T1/named/end_projection | google_search | 5/10 = 0.500 [0.237, 0.763] | 9/9 = 1.000 [0.701, 1.000] | 439.1 |
| cd_always:T1/named/end_projection | press_key | 3/5 = 0.600 [0.231, 0.882] | 5/5 = 1.000 [0.566, 1.000] | 565.2 |
| cd_always:T1/named/end_projection | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 472.9 |
| cd_always:T1/named/end_projection | type_at | 8/10 = 0.800 [0.490, 0.943] | 9/9 = 1.000 [0.701, 1.000] | 622.7 |
| cd_always:T1/named/perstep_mar | click_at | 6/10 = 0.600 [0.313, 0.832] | 9/9 = 1.000 [0.701, 1.000] | 1298.1 |
| cd_always:T1/named/perstep_mar | google_search | 2/10 = 0.200 [0.057, 0.510] | 9/9 = 1.000 [0.701, 1.000] | 1081.1 |
| cd_always:T1/named/perstep_mar | press_key | 2/5 = 0.400 [0.118, 0.769] | 5/5 = 1.000 [0.566, 1.000] | 892.3 |
| cd_always:T1/named/perstep_mar | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 666.8 |
| cd_always:T1/named/perstep_mar | type_at | 6/10 = 0.600 [0.313, 0.832] | 9/9 = 1.000 [0.701, 1.000] | 987.2 |
| cd_always:T1/named/perstep_mar_qargmax | click_at | 6/10 = 0.600 [0.313, 0.832] | 9/9 = 1.000 [0.701, 1.000] | 1095.8 |
| cd_always:T1/named/perstep_mar_qargmax | google_search | 4/10 = 0.400 [0.168, 0.687] | 9/9 = 1.000 [0.701, 1.000] | 744.0 |
| cd_always:T1/named/perstep_mar_qargmax | press_key | 2/5 = 0.400 [0.118, 0.769] | 5/5 = 1.000 [0.566, 1.000] | 1604.6 |
| cd_always:T1/named/perstep_mar_qargmax | scroll | 4/5 = 0.800 [0.376, 0.964] | 5/5 = 1.000 [0.566, 1.000] | 799.2 |
| cd_always:T1/named/perstep_mar_qargmax | type_at | 7/10 = 0.700 [0.397, 0.892] | 9/9 = 1.000 [0.701, 1.000] | 998.5 |
| cd_always:T1/named/perstep_mar+seed | click_at | 7/10 = 0.700 [0.397, 0.892] | 9/9 = 1.000 [0.701, 1.000] | 478.7 |
| cd_always:T1/named/perstep_mar+seed | google_search | 6/10 = 0.600 [0.313, 0.832] | 9/9 = 1.000 [0.701, 1.000] | 393.5 |
| cd_always:T1/named/perstep_mar+seed | press_key | 4/5 = 0.800 [0.376, 0.964] | 5/5 = 1.000 [0.566, 1.000] | 450.4 |
| cd_always:T1/named/perstep_mar+seed | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 420.6 |
| cd_always:T1/named/perstep_mar+seed | type_at | 9/10 = 0.900 [0.596, 0.982] | 9/9 = 1.000 [0.701, 1.000] | 530.9 |
| cd_always:native/T1/named/end_projection | click_at | 8/10 = 0.800 [0.490, 0.943] | 10/10 = 1.000 [0.722, 1.000] | 557.6 |
| cd_always:native/T1/named/end_projection | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 320.9 |
| cd_always:native/T1/named/end_projection | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 480.8 |
| cd_always:native/T1/named/end_projection | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 373.1 |
| cd_always:native/T1/named/end_projection | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 597.2 |
| cd_always:native/T1/named/perstep_mar | click_at | 8/10 = 0.800 [0.490, 0.943] | 10/10 = 1.000 [0.722, 1.000] | 807.0 |
| cd_always:native/T1/named/perstep_mar | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 445.6 |
| cd_always:native/T1/named/perstep_mar | press_key | 4/5 = 0.800 [0.376, 0.964] | 5/5 = 1.000 [0.566, 1.000] | 1295.6 |
| cd_always:native/T1/named/perstep_mar | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 650.6 |
| cd_always:native/T1/named/perstep_mar | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 989.2 |
| cd_always:native/T1/named/perstep_mar_qargmax | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 947.0 |
| cd_always:native/T1/named/perstep_mar_qargmax | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 450.7 |
| cd_always:native/T1/named/perstep_mar_qargmax | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 909.6 |
| cd_always:native/T1/named/perstep_mar_qargmax | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 811.4 |
| cd_always:native/T1/named/perstep_mar_qargmax | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 965.6 |
| cd_always:native/T1/named/perstep_mar+seed | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 446.7 |
| cd_always:native/T1/named/perstep_mar+seed | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 324.4 |
| cd_always:native/T1/named/perstep_mar+seed | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 384.7 |
| cd_always:native/T1/named/perstep_mar+seed | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 346.4 |
| cd_always:native/T1/named/perstep_mar+seed | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 460.0 |
| cd_always:native/T2/named/end_projection | click_at | 0/10 = 0.000 [0.000, 0.278] | 10/10 = 1.000 [0.722, 1.000] | 568.2 |
| cd_always:native/T2/named/end_projection | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 320.1 |
| cd_always:native/T2/named/end_projection | press_key | 0/5 = 0.000 [0.000, 0.434] | 5/5 = 1.000 [0.566, 1.000] | 499.3 |
| cd_always:native/T2/named/end_projection | scroll | 1/5 = 0.200 [0.036, 0.624] | 5/5 = 1.000 [0.566, 1.000] | 406.7 |
| cd_always:native/T2/named/end_projection | type_at | 0/10 = 0.000 [0.000, 0.278] | 10/10 = 1.000 [0.722, 1.000] | 622.4 |
| cd_always:native/T2/named/perstep_mar | click_at | 0/10 = 0.000 [0.000, 0.278] | 10/10 = 1.000 [0.722, 1.000] | 2836.4 |
| cd_always:native/T2/named/perstep_mar | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 1342.9 |
| cd_always:native/T2/named/perstep_mar | press_key | 0/5 = 0.000 [0.000, 0.434] | 5/5 = 1.000 [0.566, 1.000] | 1285.8 |
| cd_always:native/T2/named/perstep_mar | scroll | 3/5 = 0.600 [0.231, 0.882] | 5/5 = 1.000 [0.566, 1.000] | 727.3 |
| cd_always:native/T2/named/perstep_mar | type_at | 0/10 = 0.000 [0.000, 0.278] | 10/10 = 1.000 [0.722, 1.000] | 1760.6 |
| cd_always:native/T2/named/perstep_mar_qargmax | click_at | 0/10 = 0.000 [0.000, 0.278] | 10/10 = 1.000 [0.722, 1.000] | 2857.6 |
| cd_always:native/T2/named/perstep_mar_qargmax | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 1894.2 |
| cd_always:native/T2/named/perstep_mar_qargmax | press_key | 0/5 = 0.000 [0.000, 0.434] | 5/5 = 1.000 [0.566, 1.000] | 1373.8 |
| cd_always:native/T2/named/perstep_mar_qargmax | scroll | 3/5 = 0.600 [0.231, 0.882] | 5/5 = 1.000 [0.566, 1.000] | 798.2 |
| cd_always:native/T2/named/perstep_mar_qargmax | type_at | 0/10 = 0.000 [0.000, 0.278] | 10/10 = 1.000 [0.722, 1.000] | 2173.8 |
| cd_always:native/T2/named/perstep_mar+seed | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 463.8 |
| cd_always:native/T2/named/perstep_mar+seed | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 352.1 |
| cd_always:native/T2/named/perstep_mar+seed | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 412.7 |
| cd_always:native/T2/named/perstep_mar+seed | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 377.6 |
| cd_always:native/T2/named/perstep_mar+seed | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 493.0 |
| cd_always:native/T4/named/end_projection | click_at | 6/10 = 0.600 [0.313, 0.832] | 10/10 = 1.000 [0.722, 1.000] | 503.0 |
| cd_always:native/T4/named/end_projection | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 371.4 |
| cd_always:native/T4/named/end_projection | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 371.4 |
| cd_always:native/T4/named/end_projection | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 316.0 |
| cd_always:native/T4/named/end_projection | type_at | 7/10 = 0.700 [0.397, 0.892] | 10/10 = 1.000 [0.722, 1.000] | 512.4 |
| cd_always:native/T4/named/perstep_mar | click_at | 7/10 = 0.700 [0.397, 0.892] | 10/10 = 1.000 [0.722, 1.000] | 751.3 |
| cd_always:native/T4/named/perstep_mar | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 964.7 |
| cd_always:native/T4/named/perstep_mar | press_key | 4/5 = 0.800 [0.376, 0.964] | 5/5 = 1.000 [0.566, 1.000] | 442.8 |
| cd_always:native/T4/named/perstep_mar | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 374.0 |
| cd_always:native/T4/named/perstep_mar | type_at | 7/10 = 0.700 [0.397, 0.892] | 10/10 = 1.000 [0.722, 1.000] | 696.9 |
| cd_always:native/T4/named/perstep_mar_qargmax | click_at | 7/10 = 0.700 [0.397, 0.892] | 10/10 = 1.000 [0.722, 1.000] | 735.1 |
| cd_always:native/T4/named/perstep_mar_qargmax | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 1203.6 |
| cd_always:native/T4/named/perstep_mar_qargmax | press_key | 4/5 = 0.800 [0.376, 0.964] | 5/5 = 1.000 [0.566, 1.000] | 458.5 |
| cd_always:native/T4/named/perstep_mar_qargmax | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 381.6 |
| cd_always:native/T4/named/perstep_mar_qargmax | type_at | 8/10 = 0.800 [0.490, 0.943] | 10/10 = 1.000 [0.722, 1.000] | 726.4 |
| cd_always:native/T4/named/perstep_mar+seed | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 460.6 |
| cd_always:native/T4/named/perstep_mar+seed | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 355.8 |
| cd_always:native/T4/named/perstep_mar+seed | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 411.9 |
| cd_always:native/T4/named/perstep_mar+seed | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 365.4 |
| cd_always:native/T4/named/perstep_mar+seed | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 488.1 |
| cd_on_invalid:T1/named/end_projection | click_at | 7/10 = 0.700 [0.397, 0.892] | 9/9 = 1.000 [0.701, 1.000] | 281.5 |
| cd_on_invalid:T1/named/end_projection | google_search | 7/10 = 0.700 [0.397, 0.892] | 9/9 = 1.000 [0.701, 1.000] | 240.8 |
| cd_on_invalid:T1/named/end_projection | press_key | 4/5 = 0.800 [0.376, 0.964] | 5/5 = 1.000 [0.566, 1.000] | 267.7 |
| cd_on_invalid:T1/named/end_projection | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 225.4 |
| cd_on_invalid:T1/named/end_projection | type_at | 9/10 = 0.900 [0.596, 0.982] | 9/9 = 1.000 [0.701, 1.000] | 306.3 |
| cd_on_invalid:T1/named/perstep_mar | click_at | 7/10 = 0.700 [0.397, 0.892] | 9/9 = 1.000 [0.701, 1.000] | 281.5 |
| cd_on_invalid:T1/named/perstep_mar | google_search | 7/10 = 0.700 [0.397, 0.892] | 9/9 = 1.000 [0.701, 1.000] | 240.8 |
| cd_on_invalid:T1/named/perstep_mar | press_key | 4/5 = 0.800 [0.376, 0.964] | 5/5 = 1.000 [0.566, 1.000] | 267.7 |
| cd_on_invalid:T1/named/perstep_mar | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 225.4 |
| cd_on_invalid:T1/named/perstep_mar | type_at | 9/10 = 0.900 [0.596, 0.982] | 9/9 = 1.000 [0.701, 1.000] | 306.3 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | click_at | 7/10 = 0.700 [0.397, 0.892] | 9/9 = 1.000 [0.701, 1.000] | 281.5 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | google_search | 6/10 = 0.600 [0.313, 0.832] | 9/9 = 1.000 [0.701, 1.000] | 240.8 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | press_key | 4/5 = 0.800 [0.376, 0.964] | 5/5 = 1.000 [0.566, 1.000] | 267.7 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 225.4 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | type_at | 9/10 = 0.900 [0.596, 0.982] | 9/9 = 1.000 [0.701, 1.000] | 306.3 |
| cd_on_invalid:T1/named/perstep_mar+seed | click_at | 7/10 = 0.700 [0.397, 0.892] | 9/9 = 1.000 [0.701, 1.000] | 281.5 |
| cd_on_invalid:T1/named/perstep_mar+seed | google_search | 6/10 = 0.600 [0.313, 0.832] | 9/9 = 1.000 [0.701, 1.000] | 240.8 |
| cd_on_invalid:T1/named/perstep_mar+seed | press_key | 4/5 = 0.800 [0.376, 0.964] | 5/5 = 1.000 [0.566, 1.000] | 267.7 |
| cd_on_invalid:T1/named/perstep_mar+seed | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 225.4 |
| cd_on_invalid:T1/named/perstep_mar+seed | type_at | 9/10 = 0.900 [0.596, 0.982] | 9/9 = 1.000 [0.701, 1.000] | 306.3 |
| cd_on_invalid:native/T1/named/end_projection | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 256.1 |
| cd_on_invalid:native/T1/named/end_projection | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 144.2 |
| cd_on_invalid:native/T1/named/end_projection | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 202.1 |
| cd_on_invalid:native/T1/named/end_projection | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 160.0 |
| cd_on_invalid:native/T1/named/end_projection | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 276.2 |
| cd_on_invalid:native/T1/named/perstep_mar | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 256.1 |
| cd_on_invalid:native/T1/named/perstep_mar | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 144.2 |
| cd_on_invalid:native/T1/named/perstep_mar | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 202.1 |
| cd_on_invalid:native/T1/named/perstep_mar | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 160.0 |
| cd_on_invalid:native/T1/named/perstep_mar | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 276.2 |
| cd_on_invalid:native/T1/named/perstep_mar_qargmax | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 256.1 |
| cd_on_invalid:native/T1/named/perstep_mar_qargmax | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 144.2 |
| cd_on_invalid:native/T1/named/perstep_mar_qargmax | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 202.1 |
| cd_on_invalid:native/T1/named/perstep_mar_qargmax | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 160.0 |
| cd_on_invalid:native/T1/named/perstep_mar_qargmax | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 276.2 |
| cd_on_invalid:native/T1/named/perstep_mar+seed | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 256.1 |
| cd_on_invalid:native/T1/named/perstep_mar+seed | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 144.2 |
| cd_on_invalid:native/T1/named/perstep_mar+seed | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 202.1 |
| cd_on_invalid:native/T1/named/perstep_mar+seed | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 160.0 |
| cd_on_invalid:native/T1/named/perstep_mar+seed | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 276.2 |
| cd_on_invalid:native/T2/named/end_projection | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 256.1 |
| cd_on_invalid:native/T2/named/end_projection | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 144.2 |
| cd_on_invalid:native/T2/named/end_projection | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 202.1 |
| cd_on_invalid:native/T2/named/end_projection | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 160.0 |
| cd_on_invalid:native/T2/named/end_projection | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 276.2 |
| cd_on_invalid:native/T2/named/perstep_mar | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 256.1 |
| cd_on_invalid:native/T2/named/perstep_mar | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 144.2 |
| cd_on_invalid:native/T2/named/perstep_mar | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 202.1 |
| cd_on_invalid:native/T2/named/perstep_mar | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 160.0 |
| cd_on_invalid:native/T2/named/perstep_mar | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 276.2 |
| cd_on_invalid:native/T2/named/perstep_mar_qargmax | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 256.1 |
| cd_on_invalid:native/T2/named/perstep_mar_qargmax | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 144.2 |
| cd_on_invalid:native/T2/named/perstep_mar_qargmax | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 202.1 |
| cd_on_invalid:native/T2/named/perstep_mar_qargmax | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 160.0 |
| cd_on_invalid:native/T2/named/perstep_mar_qargmax | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 276.2 |
| cd_on_invalid:native/T2/named/perstep_mar+seed | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 256.1 |
| cd_on_invalid:native/T2/named/perstep_mar+seed | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 144.2 |
| cd_on_invalid:native/T2/named/perstep_mar+seed | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 202.1 |
| cd_on_invalid:native/T2/named/perstep_mar+seed | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 160.0 |
| cd_on_invalid:native/T2/named/perstep_mar+seed | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 276.2 |
| cd_on_invalid:native/T4/named/end_projection | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 256.1 |
| cd_on_invalid:native/T4/named/end_projection | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 144.2 |
| cd_on_invalid:native/T4/named/end_projection | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 202.1 |
| cd_on_invalid:native/T4/named/end_projection | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 160.0 |
| cd_on_invalid:native/T4/named/end_projection | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 276.2 |
| cd_on_invalid:native/T4/named/perstep_mar | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 256.1 |
| cd_on_invalid:native/T4/named/perstep_mar | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 144.2 |
| cd_on_invalid:native/T4/named/perstep_mar | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 202.1 |
| cd_on_invalid:native/T4/named/perstep_mar | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 160.0 |
| cd_on_invalid:native/T4/named/perstep_mar | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 276.2 |
| cd_on_invalid:native/T4/named/perstep_mar_qargmax | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 256.1 |
| cd_on_invalid:native/T4/named/perstep_mar_qargmax | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 144.2 |
| cd_on_invalid:native/T4/named/perstep_mar_qargmax | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 202.1 |
| cd_on_invalid:native/T4/named/perstep_mar_qargmax | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 160.0 |
| cd_on_invalid:native/T4/named/perstep_mar_qargmax | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 276.2 |
| cd_on_invalid:native/T4/named/perstep_mar+seed | click_at | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 256.1 |
| cd_on_invalid:native/T4/named/perstep_mar+seed | google_search | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 144.2 |
| cd_on_invalid:native/T4/named/perstep_mar+seed | press_key | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 202.1 |
| cd_on_invalid:native/T4/named/perstep_mar+seed | scroll | 5/5 = 1.000 [0.566, 1.000] | 5/5 = 1.000 [0.566, 1.000] | 160.0 |
| cd_on_invalid:native/T4/named/perstep_mar+seed | type_at | 10/10 = 1.000 [0.722, 1.000] | 10/10 = 1.000 [0.722, 1.000] | 276.2 |

## Format cc

- cases 48; route recall by the `cc` tag (tool cases): 37/40 = 0.925 [0.801, 0.974]; by any tool-call tag: 37/40 = 0.925 [0.801, 0.974]; false-tag rate (no-tool cases): 0/8 = 0.000 [0.000, 0.324]
- routes with another format's tag (wrong_tag): 0; open_thought_call: n/a
- draft parse failures: {'no arguments': 2, 'not JSON': 1}
- spoken prefix non-empty (routed tool cases): 37/37 = 1.000 [0.906, 1.000]; median chars 49.0; text after the close: n/a
- pass-1 wall ms p50/p90 253.7/346.3 (n 48); pass-1 prompt tokens p50 n/a (n 0); pass-1 prefix hit median 0.977
- drafts with a stripped close, logs/t0084-native-dev-794161d/20261011T022440.808068Z-toolcall-dev.jsonl: 20/37 routed
- cc stage line: {'rule': 'stage1', 'choice': 'T1/named/perstep_mar'}
- pass-2 wall ms by position within a case (T-0082b R10; first cell vs later cells): first p50/p90 273.9/581.9 (n 37); later p50/p90 547.7/1336.9 (n 111)

McNemar (exact, tool cases; Bonferroni over 14):

- cd_always:T1/named/end_projection vs draft_only: n 40, b 1, c 4, p 0.375 (adj 1); both parsed n 34, b 0, c 4, p 0.125
- cd_always:T1/named/perstep_mar vs draft_only: n 40, b 1, c 11, p 0.006348 (adj 0.08887); both parsed n 34, b 0, c 11, p 0.0009766
- cd_always:T1/named/perstep_mar_qargmax vs draft_only: n 40, b 0, c 8, p 0.007812 (adj 0.1094); both parsed n 34, b 0, c 8, p 0.007812
- cd_always:T1/named/perstep_mar+seed vs draft_only: n 40, b 0, c 0, p 1 (adj 1); both parsed n 34, b 0, c 0, p 1
- cd_on_invalid:T1/named/end_projection vs draft_only: n 40, b 1, c 0, p 1 (adj 1); both parsed n 34, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar vs draft_only: n 40, b 1, c 0, p 1 (adj 1); both parsed n 34, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar_qargmax vs draft_only: n 40, b 0, c 0, p 1 (adj 1); both parsed n 34, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar+seed vs draft_only: n 40, b 0, c 0, p 1 (adj 1); both parsed n 34, b 0, c 0, p 1
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar: n 40, b 8, c 1, p 0.03906 (adj 0.5469); both parsed n 37, b 8, c 1, p 0.03906
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar_qargmax: n 40, b 6, c 1, p 0.125 (adj 1); both parsed n 37, b 6, c 1, p 0.125
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar+seed: n 40, b 1, c 4, p 0.375 (adj 1); both parsed n 37, b 1, c 4, p 0.375
- cd_always:T1/named/perstep_mar vs cd_always:T1/named/perstep_mar_qargmax: n 40, b 5, c 7, p 0.7744 (adj 1); both parsed n 37, b 5, c 7, p 0.7744
- cd_always:T1/named/perstep_mar vs cd_always:T1/named/perstep_mar+seed: n 40, b 1, c 11, p 0.006348 (adj 0.08887); both parsed n 37, b 1, c 11, p 0.006348
- cd_always:T1/named/perstep_mar_qargmax vs cd_always:T1/named/perstep_mar+seed: n 40, b 0, c 8, p 0.007812 (adj 0.1094); both parsed n 37, b 0, c 8, p 0.007812

T1 `101` position: {'T1/named/end_projection': {'106': 1, '110': 1, '127': 1, '17': 1, '35': 1, '36': 1, '38': 1, '41': 1, '42': 3, '43': 1, '48': 4, '52': 1, '55': 1, '59': 1, '62': 1, '65': 2, '66': 1, '67': 1, '68': 2, '71': 1, '72': 1, '83': 2, '84': 1, '86': 1, '91': 1, '93': 1, '95': 1, '97': 1, '98': 1}, 'T1/named/perstep_mar': {'100': 1, '102': 1, '106': 1, '120': 2, '133': 1, '36': 1, '39': 1, '40': 2, '42': 3, '43': 1, '44': 1, '45': 1, '47': 1, '48': 1, '53': 1, '55': 1, '57': 1, '59': 1, '64': 2, '65': 2, '66': 3, '67': 1, '68': 1, '70': 1, '77': 1, '82': 1, '85': 1, '89': 1, '93': 1}, 'T1/named/perstep_mar_qargmax': {'108': 1, '119': 1, '122': 1, '137': 1, '140': 1, '17': 1, '37': 1, '39': 1, '40': 1, '44': 1, '45': 4, '47': 1, '48': 2, '50': 1, '56': 1, '60': 1, '62': 1, '63': 1, '65': 3, '67': 2, '72': 1, '74': 1, '83': 1, '84': 1, '85': 1, '93': 2, '94': 1, '97': 1, '98': 1}, 'T1/named/perstep_mar+seed': {'125': 1, '128': 1, '130': 1, '18': 1, '25': 1, '28': 2, '29': 1, '30': 3, '31': 1, '32': 1, '33': 1, '34': 1, '43': 1, '47': 1, '49': 2, '52': 1, '54': 1, '55': 1, '56': 1, '57': 1, '60': 2, '61': 1, '63': 1, '68': 1, '76': 1, '77': 1, '82': 1, '83': 2, '85': 1, '89': 1, '94': 1}}

- T1/named/end_projection: pass 2 equals the valid draft 16/34 = 0.471 [0.315, 0.633]; invalid drafts repaired 3/3 = 1.000 [0.438, 1.000]; name-picker rows 0
- T1/named/perstep_mar: pass 2 equals the valid draft 10/34 = 0.294 [0.168, 0.462]; invalid drafts repaired 3/3 = 1.000 [0.438, 1.000]; name-picker rows 0
- T1/named/perstep_mar_qargmax: pass 2 equals the valid draft 11/34 = 0.324 [0.191, 0.492]; invalid drafts repaired 3/3 = 1.000 [0.438, 1.000]; name-picker rows 0
- T1/named/perstep_mar+seed: pass 2 equals the valid draft 33/34 = 0.971 [0.851, 0.995]; invalid drafts repaired 3/3 = 1.000 [0.438, 1.000]; name-picker rows 0
- seed T1/named/perstep_mar+seed: seeded 37/37 routed cases; no seed none
- seed copy T1/named/perstep_mar+seed (first canvas equals the seed through its first stop): valid drafts 30/34 = 0.882 [0.734, 0.953]; invalid drafts 0/3 = 0.000 [0.000, 0.562]
- seed trim T1/named/perstep_mar+seed (seeded rows whose draft lost trailing whitespace, R2): 1/37 = 0.027 [0.005, 0.138]

## Format native

- cases 48; route recall by the `native` tag (tool cases): 40/40 = 1.000 [0.912, 1.000]; by any tool-call tag: 40/40 = 1.000 [0.912, 1.000]; false-tag rate (no-tool cases): 0/8 = 0.000 [0.000, 0.324]
- routes with another format's tag (wrong_tag): 0; open_thought_call: 0/48 = 0.000 [0.000, 0.074]
- draft parse failures: none
- spoken prefix non-empty (routed tool cases): 27/40 = 0.675 [0.520, 0.799]; median chars 47.0; text after the close: 0/40 = 0.000 [0.000, 0.088]
- pass-1 wall ms p50/p90 215.7/315.9 (n 48); pass-1 prompt tokens p50 1071.5 (n 48); pass-1 prefix hit median 0.971
- drafts with a stripped close, logs/t0084-native-dev-794161d/20261011T022440.808068Z-toolcall-dev.jsonl: 0/40 routed
- native stage line: {'rule': 'stage1', 'choice': 'native/T4/named/perstep_mar'}
- pass-2 wall ms by position within a case (T-0082b R10; first cell vs later cells): first p50/p90 231.2/1351.3 (n 40); later p50/p90 314.7/1573.0 (n 440)

McNemar (exact, tool cases; Bonferroni over 90):

- cd_always:native/T1/named/end_projection vs draft_only:native: n 40, b 0, c 1, p 1 (adj 1); both parsed n 40, b 0, c 1, p 1
- cd_always:native/T1/named/perstep_mar vs draft_only:native: n 40, b 0, c 2, p 0.5 (adj 1); both parsed n 40, b 0, c 2, p 0.5
- cd_always:native/T1/named/perstep_mar_qargmax vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_always:native/T1/named/perstep_mar+seed vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_always:native/T2/named/end_projection vs draft_only:native: n 40, b 0, c 28, p 7.451e-09 (adj 6.706e-07); both parsed n 40, b 0, c 28, p 7.451e-09
- cd_always:native/T2/named/perstep_mar vs draft_only:native: n 40, b 0, c 26, p 2.98e-08 (adj 2.682e-06); both parsed n 40, b 0, c 26, p 2.98e-08
- cd_always:native/T2/named/perstep_mar_qargmax vs draft_only:native: n 40, b 0, c 26, p 2.98e-08 (adj 2.682e-06); both parsed n 40, b 0, c 26, p 2.98e-08
- cd_always:native/T2/named/perstep_mar+seed vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_always:native/T4/named/end_projection vs draft_only:native: n 40, b 0, c 6, p 0.03125 (adj 1); both parsed n 40, b 0, c 6, p 0.03125
- cd_always:native/T4/named/perstep_mar vs draft_only:native: n 40, b 1, c 7, p 0.07031 (adj 1); both parsed n 40, b 1, c 7, p 0.07031
- cd_always:native/T4/named/perstep_mar_qargmax vs draft_only:native: n 40, b 0, c 5, p 0.0625 (adj 1); both parsed n 40, b 0, c 5, p 0.0625
- cd_always:native/T4/named/perstep_mar+seed vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_on_invalid:native/T1/named/end_projection vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_on_invalid:native/T1/named/perstep_mar vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_on_invalid:native/T1/named/perstep_mar_qargmax vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_on_invalid:native/T1/named/perstep_mar+seed vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_on_invalid:native/T2/named/end_projection vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_on_invalid:native/T2/named/perstep_mar vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_on_invalid:native/T2/named/perstep_mar_qargmax vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_on_invalid:native/T2/named/perstep_mar+seed vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_on_invalid:native/T4/named/end_projection vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_on_invalid:native/T4/named/perstep_mar vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_on_invalid:native/T4/named/perstep_mar_qargmax vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_on_invalid:native/T4/named/perstep_mar+seed vs draft_only:native: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_always:native/T1/named/end_projection vs cd_always:native/T1/named/perstep_mar: n 40, b 1, c 0, p 1 (adj 1); both parsed n 40, b 1, c 0, p 1
- cd_always:native/T1/named/end_projection vs cd_always:native/T1/named/perstep_mar_qargmax: n 40, b 0, c 1, p 1 (adj 1); both parsed n 40, b 0, c 1, p 1
- cd_always:native/T1/named/end_projection vs cd_always:native/T1/named/perstep_mar+seed: n 40, b 0, c 1, p 1 (adj 1); both parsed n 40, b 0, c 1, p 1
- cd_always:native/T1/named/end_projection vs cd_always:native/T2/named/end_projection: n 40, b 27, c 0, p 1.49e-08 (adj 1.341e-06); both parsed n 40, b 27, c 0, p 1.49e-08
- cd_always:native/T1/named/end_projection vs cd_always:native/T2/named/perstep_mar: n 40, b 25, c 0, p 5.96e-08 (adj 5.364e-06); both parsed n 40, b 25, c 0, p 5.96e-08
- cd_always:native/T1/named/end_projection vs cd_always:native/T2/named/perstep_mar_qargmax: n 40, b 25, c 0, p 5.96e-08 (adj 5.364e-06); both parsed n 40, b 25, c 0, p 5.96e-08
- cd_always:native/T1/named/end_projection vs cd_always:native/T2/named/perstep_mar+seed: n 40, b 0, c 1, p 1 (adj 1); both parsed n 40, b 0, c 1, p 1
- cd_always:native/T1/named/end_projection vs cd_always:native/T4/named/end_projection: n 40, b 5, c 0, p 0.0625 (adj 1); both parsed n 40, b 5, c 0, p 0.0625
- cd_always:native/T1/named/end_projection vs cd_always:native/T4/named/perstep_mar: n 40, b 6, c 1, p 0.125 (adj 1); both parsed n 40, b 6, c 1, p 0.125
- cd_always:native/T1/named/end_projection vs cd_always:native/T4/named/perstep_mar_qargmax: n 40, b 5, c 1, p 0.2188 (adj 1); both parsed n 40, b 5, c 1, p 0.2188
- cd_always:native/T1/named/end_projection vs cd_always:native/T4/named/perstep_mar+seed: n 40, b 0, c 1, p 1 (adj 1); both parsed n 40, b 0, c 1, p 1
- cd_always:native/T1/named/perstep_mar vs cd_always:native/T1/named/perstep_mar_qargmax: n 40, b 0, c 2, p 0.5 (adj 1); both parsed n 40, b 0, c 2, p 0.5
- cd_always:native/T1/named/perstep_mar vs cd_always:native/T1/named/perstep_mar+seed: n 40, b 0, c 2, p 0.5 (adj 1); both parsed n 40, b 0, c 2, p 0.5
- cd_always:native/T1/named/perstep_mar vs cd_always:native/T2/named/end_projection: n 40, b 26, c 0, p 2.98e-08 (adj 2.682e-06); both parsed n 40, b 26, c 0, p 2.98e-08
- cd_always:native/T1/named/perstep_mar vs cd_always:native/T2/named/perstep_mar: n 40, b 24, c 0, p 1.192e-07 (adj 1.073e-05); both parsed n 40, b 24, c 0, p 1.192e-07
- cd_always:native/T1/named/perstep_mar vs cd_always:native/T2/named/perstep_mar_qargmax: n 40, b 24, c 0, p 1.192e-07 (adj 1.073e-05); both parsed n 40, b 24, c 0, p 1.192e-07
- cd_always:native/T1/named/perstep_mar vs cd_always:native/T2/named/perstep_mar+seed: n 40, b 0, c 2, p 0.5 (adj 1); both parsed n 40, b 0, c 2, p 0.5
- cd_always:native/T1/named/perstep_mar vs cd_always:native/T4/named/end_projection: n 40, b 5, c 1, p 0.2188 (adj 1); both parsed n 40, b 5, c 1, p 0.2188
- cd_always:native/T1/named/perstep_mar vs cd_always:native/T4/named/perstep_mar: n 40, b 5, c 1, p 0.2188 (adj 1); both parsed n 40, b 5, c 1, p 0.2188
- cd_always:native/T1/named/perstep_mar vs cd_always:native/T4/named/perstep_mar_qargmax: n 40, b 5, c 2, p 0.4531 (adj 1); both parsed n 40, b 5, c 2, p 0.4531
- cd_always:native/T1/named/perstep_mar vs cd_always:native/T4/named/perstep_mar+seed: n 40, b 0, c 2, p 0.5 (adj 1); both parsed n 40, b 0, c 2, p 0.5
- cd_always:native/T1/named/perstep_mar_qargmax vs cd_always:native/T1/named/perstep_mar+seed: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_always:native/T1/named/perstep_mar_qargmax vs cd_always:native/T2/named/end_projection: n 40, b 28, c 0, p 7.451e-09 (adj 6.706e-07); both parsed n 40, b 28, c 0, p 7.451e-09
- cd_always:native/T1/named/perstep_mar_qargmax vs cd_always:native/T2/named/perstep_mar: n 40, b 26, c 0, p 2.98e-08 (adj 2.682e-06); both parsed n 40, b 26, c 0, p 2.98e-08
- cd_always:native/T1/named/perstep_mar_qargmax vs cd_always:native/T2/named/perstep_mar_qargmax: n 40, b 26, c 0, p 2.98e-08 (adj 2.682e-06); both parsed n 40, b 26, c 0, p 2.98e-08
- cd_always:native/T1/named/perstep_mar_qargmax vs cd_always:native/T2/named/perstep_mar+seed: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_always:native/T1/named/perstep_mar_qargmax vs cd_always:native/T4/named/end_projection: n 40, b 6, c 0, p 0.03125 (adj 1); both parsed n 40, b 6, c 0, p 0.03125
- cd_always:native/T1/named/perstep_mar_qargmax vs cd_always:native/T4/named/perstep_mar: n 40, b 7, c 1, p 0.07031 (adj 1); both parsed n 40, b 7, c 1, p 0.07031
- cd_always:native/T1/named/perstep_mar_qargmax vs cd_always:native/T4/named/perstep_mar_qargmax: n 40, b 5, c 0, p 0.0625 (adj 1); both parsed n 40, b 5, c 0, p 0.0625
- cd_always:native/T1/named/perstep_mar_qargmax vs cd_always:native/T4/named/perstep_mar+seed: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_always:native/T1/named/perstep_mar+seed vs cd_always:native/T2/named/end_projection: n 40, b 28, c 0, p 7.451e-09 (adj 6.706e-07); both parsed n 40, b 28, c 0, p 7.451e-09
- cd_always:native/T1/named/perstep_mar+seed vs cd_always:native/T2/named/perstep_mar: n 40, b 26, c 0, p 2.98e-08 (adj 2.682e-06); both parsed n 40, b 26, c 0, p 2.98e-08
- cd_always:native/T1/named/perstep_mar+seed vs cd_always:native/T2/named/perstep_mar_qargmax: n 40, b 26, c 0, p 2.98e-08 (adj 2.682e-06); both parsed n 40, b 26, c 0, p 2.98e-08
- cd_always:native/T1/named/perstep_mar+seed vs cd_always:native/T2/named/perstep_mar+seed: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_always:native/T1/named/perstep_mar+seed vs cd_always:native/T4/named/end_projection: n 40, b 6, c 0, p 0.03125 (adj 1); both parsed n 40, b 6, c 0, p 0.03125
- cd_always:native/T1/named/perstep_mar+seed vs cd_always:native/T4/named/perstep_mar: n 40, b 7, c 1, p 0.07031 (adj 1); both parsed n 40, b 7, c 1, p 0.07031
- cd_always:native/T1/named/perstep_mar+seed vs cd_always:native/T4/named/perstep_mar_qargmax: n 40, b 5, c 0, p 0.0625 (adj 1); both parsed n 40, b 5, c 0, p 0.0625
- cd_always:native/T1/named/perstep_mar+seed vs cd_always:native/T4/named/perstep_mar+seed: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_always:native/T2/named/end_projection vs cd_always:native/T2/named/perstep_mar: n 40, b 0, c 2, p 0.5 (adj 1); both parsed n 40, b 0, c 2, p 0.5
- cd_always:native/T2/named/end_projection vs cd_always:native/T2/named/perstep_mar_qargmax: n 40, b 0, c 2, p 0.5 (adj 1); both parsed n 40, b 0, c 2, p 0.5
- cd_always:native/T2/named/end_projection vs cd_always:native/T2/named/perstep_mar+seed: n 40, b 0, c 28, p 7.451e-09 (adj 6.706e-07); both parsed n 40, b 0, c 28, p 7.451e-09
- cd_always:native/T2/named/end_projection vs cd_always:native/T4/named/end_projection: n 40, b 0, c 22, p 4.768e-07 (adj 4.292e-05); both parsed n 40, b 0, c 22, p 4.768e-07
- cd_always:native/T2/named/end_projection vs cd_always:native/T4/named/perstep_mar: n 40, b 0, c 22, p 4.768e-07 (adj 4.292e-05); both parsed n 40, b 0, c 22, p 4.768e-07
- cd_always:native/T2/named/end_projection vs cd_always:native/T4/named/perstep_mar_qargmax: n 40, b 0, c 23, p 2.384e-07 (adj 2.146e-05); both parsed n 40, b 0, c 23, p 2.384e-07
- cd_always:native/T2/named/end_projection vs cd_always:native/T4/named/perstep_mar+seed: n 40, b 0, c 28, p 7.451e-09 (adj 6.706e-07); both parsed n 40, b 0, c 28, p 7.451e-09
- cd_always:native/T2/named/perstep_mar vs cd_always:native/T2/named/perstep_mar_qargmax: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_always:native/T2/named/perstep_mar vs cd_always:native/T2/named/perstep_mar+seed: n 40, b 0, c 26, p 2.98e-08 (adj 2.682e-06); both parsed n 40, b 0, c 26, p 2.98e-08
- cd_always:native/T2/named/perstep_mar vs cd_always:native/T4/named/end_projection: n 40, b 0, c 20, p 1.907e-06 (adj 0.0001717); both parsed n 40, b 0, c 20, p 1.907e-06
- cd_always:native/T2/named/perstep_mar vs cd_always:native/T4/named/perstep_mar: n 40, b 0, c 20, p 1.907e-06 (adj 0.0001717); both parsed n 40, b 0, c 20, p 1.907e-06
- cd_always:native/T2/named/perstep_mar vs cd_always:native/T4/named/perstep_mar_qargmax: n 40, b 0, c 21, p 9.537e-07 (adj 8.583e-05); both parsed n 40, b 0, c 21, p 9.537e-07
- cd_always:native/T2/named/perstep_mar vs cd_always:native/T4/named/perstep_mar+seed: n 40, b 0, c 26, p 2.98e-08 (adj 2.682e-06); both parsed n 40, b 0, c 26, p 2.98e-08
- cd_always:native/T2/named/perstep_mar_qargmax vs cd_always:native/T2/named/perstep_mar+seed: n 40, b 0, c 26, p 2.98e-08 (adj 2.682e-06); both parsed n 40, b 0, c 26, p 2.98e-08
- cd_always:native/T2/named/perstep_mar_qargmax vs cd_always:native/T4/named/end_projection: n 40, b 0, c 20, p 1.907e-06 (adj 0.0001717); both parsed n 40, b 0, c 20, p 1.907e-06
- cd_always:native/T2/named/perstep_mar_qargmax vs cd_always:native/T4/named/perstep_mar: n 40, b 0, c 20, p 1.907e-06 (adj 0.0001717); both parsed n 40, b 0, c 20, p 1.907e-06
- cd_always:native/T2/named/perstep_mar_qargmax vs cd_always:native/T4/named/perstep_mar_qargmax: n 40, b 0, c 21, p 9.537e-07 (adj 8.583e-05); both parsed n 40, b 0, c 21, p 9.537e-07
- cd_always:native/T2/named/perstep_mar_qargmax vs cd_always:native/T4/named/perstep_mar+seed: n 40, b 0, c 26, p 2.98e-08 (adj 2.682e-06); both parsed n 40, b 0, c 26, p 2.98e-08
- cd_always:native/T2/named/perstep_mar+seed vs cd_always:native/T4/named/end_projection: n 40, b 6, c 0, p 0.03125 (adj 1); both parsed n 40, b 6, c 0, p 0.03125
- cd_always:native/T2/named/perstep_mar+seed vs cd_always:native/T4/named/perstep_mar: n 40, b 7, c 1, p 0.07031 (adj 1); both parsed n 40, b 7, c 1, p 0.07031
- cd_always:native/T2/named/perstep_mar+seed vs cd_always:native/T4/named/perstep_mar_qargmax: n 40, b 5, c 0, p 0.0625 (adj 1); both parsed n 40, b 5, c 0, p 0.0625
- cd_always:native/T2/named/perstep_mar+seed vs cd_always:native/T4/named/perstep_mar+seed: n 40, b 0, c 0, p 1 (adj 1); both parsed n 40, b 0, c 0, p 1
- cd_always:native/T4/named/end_projection vs cd_always:native/T4/named/perstep_mar: n 40, b 2, c 2, p 1 (adj 1); both parsed n 40, b 2, c 2, p 1
- cd_always:native/T4/named/end_projection vs cd_always:native/T4/named/perstep_mar_qargmax: n 40, b 1, c 2, p 1 (adj 1); both parsed n 40, b 1, c 2, p 1
- cd_always:native/T4/named/end_projection vs cd_always:native/T4/named/perstep_mar+seed: n 40, b 0, c 6, p 0.03125 (adj 1); both parsed n 40, b 0, c 6, p 0.03125
- cd_always:native/T4/named/perstep_mar vs cd_always:native/T4/named/perstep_mar_qargmax: n 40, b 2, c 3, p 1 (adj 1); both parsed n 40, b 2, c 3, p 1
- cd_always:native/T4/named/perstep_mar vs cd_always:native/T4/named/perstep_mar+seed: n 40, b 1, c 7, p 0.07031 (adj 1); both parsed n 40, b 1, c 7, p 0.07031
- cd_always:native/T4/named/perstep_mar_qargmax vs cd_always:native/T4/named/perstep_mar+seed: n 40, b 0, c 5, p 0.0625 (adj 1); both parsed n 40, b 0, c 5, p 0.0625

T1 `101` position: {'native/T1/named/end_projection': {'100': 1, '104': 1, '111': 1, '130': 1, '27': 1, '28': 2, '29': 2, '30': 2, '31': 2, '32': 1, '34': 1, '35': 2, '42': 1, '46': 1, '59': 1, '61': 1, '63': 1, '66': 2, '69': 1, '70': 1, '72': 1, '73': 1, '75': 1, '77': 1, '80': 2, '85': 1, '87': 1, '91': 2, '93': 1, '95': 1, '97': 1, '98': 1}, 'native/T1/named/perstep_mar': {'101': 1, '104': 1, '116': 1, '119': 1, '129': 1, '137': 1, '156': 1, '253': 1, '28': 2, '29': 1, '30': 4, '31': 1, '32': 1, '33': 1, '36': 2, '39': 1, '42': 1, '49': 1, '59': 1, '60': 1, '65': 1, '66': 1, '67': 2, '71': 1, '73': 1, '78': 1, '79': 1, '80': 1, '83': 1, '85': 1, '87': 1, '90': 1, '93': 1, '95': 1}, 'native/T1/named/perstep_mar_qargmax': {'103': 1, '104': 1, '110': 1, '111': 1, '117': 1, '129': 1, '144': 1, '181': 1, '28': 3, '29': 4, '30': 1, '31': 1, '36': 1, '38': 1, '41': 2, '42': 1, '43': 1, '53': 1, '60': 1, '65': 1, '67': 1, '69': 1, '70': 1, '71': 1, '72': 1, '73': 1, '75': 1, '76': 1, '79': 1, '86': 2, '89': 1, '92': 1, '97': 1}, 'native/T1/named/perstep_mar+seed': {'18': 1, '27': 1, '28': 2, '29': 1, '30': 5, '31': 3, '33': 1, '35': 1, '40': 1, '50': 1, '51': 1, '54': 1, '56': 1, '59': 2, '61': 2, '64': 2, '68': 2, '69': 1, '73': 1, '75': 1, '79': 1, '81': 1, '85': 1, '88': 3, '94': 2, '97': 1}}

- native/T1/named/end_projection: pass 2 equals the valid draft 21/40 = 0.525 [0.375, 0.671]; invalid drafts repaired n/a; name-picker rows 0
- native/T1/named/perstep_mar: pass 2 equals the valid draft 22/40 = 0.550 [0.398, 0.693]; invalid drafts repaired n/a; name-picker rows 0
- native/T1/named/perstep_mar_qargmax: pass 2 equals the valid draft 23/40 = 0.575 [0.422, 0.715]; invalid drafts repaired n/a; name-picker rows 0
- native/T1/named/perstep_mar+seed: pass 2 equals the valid draft 40/40 = 1.000 [0.912, 1.000]; invalid drafts repaired n/a; name-picker rows 0
- native/T2/named/end_projection: pass 2 equals the valid draft 1/40 = 0.025 [0.004, 0.129]; invalid drafts repaired n/a; name-picker rows 0
- native/T2/named/perstep_mar: pass 2 equals the valid draft 3/40 = 0.075 [0.026, 0.199]; invalid drafts repaired n/a; name-picker rows 0
- native/T2/named/perstep_mar_qargmax: pass 2 equals the valid draft 3/40 = 0.075 [0.026, 0.199]; invalid drafts repaired n/a; name-picker rows 0
- native/T2/named/perstep_mar+seed: pass 2 equals the valid draft 38/40 = 0.950 [0.835, 0.986]; invalid drafts repaired n/a; name-picker rows 0
- native/T4/named/end_projection: pass 2 equals the valid draft 10/40 = 0.250 [0.142, 0.402]; invalid drafts repaired n/a; name-picker rows 0
- native/T4/named/perstep_mar: pass 2 equals the valid draft 9/40 = 0.225 [0.123, 0.375]; invalid drafts repaired n/a; name-picker rows 0
- native/T4/named/perstep_mar_qargmax: pass 2 equals the valid draft 9/40 = 0.225 [0.123, 0.375]; invalid drafts repaired n/a; name-picker rows 0
- native/T4/named/perstep_mar+seed: pass 2 equals the valid draft 39/40 = 0.975 [0.871, 0.996]; invalid drafts repaired n/a; name-picker rows 0
- seed native/T1/named/perstep_mar+seed: seeded 40/40 routed cases; no seed none
- seed copy native/T1/named/perstep_mar+seed (first canvas equals the seed through its first stop): valid drafts 36/40 = 0.900 [0.769, 0.960]; invalid drafts n/a
- seed trim native/T1/named/perstep_mar+seed (seeded rows whose draft lost trailing whitespace, R2): 0/40 = 0.000 [0.000, 0.088]
- seed native/T2/named/perstep_mar+seed: seeded 40/40 routed cases; no seed none
- seed copy native/T2/named/perstep_mar+seed (first canvas equals the seed through its first stop): valid drafts 37/40 = 0.925 [0.801, 0.974]; invalid drafts n/a
- seed trim native/T2/named/perstep_mar+seed (seeded rows whose draft lost trailing whitespace, R2): 0/40 = 0.000 [0.000, 0.088]
- seed native/T4/named/perstep_mar+seed: seeded 40/40 routed cases; no seed none
- seed copy native/T4/named/perstep_mar+seed (first canvas equals the seed through its first stop): valid drafts 39/40 = 0.975 [0.871, 0.996]; invalid drafts n/a
- seed trim native/T4/named/perstep_mar+seed (seeded rows whose draft lost trailing whitespace, R2): 0/40 = 0.000 [0.000, 0.088]

## Format comparison (T-0084d criterion 1)

Step 1 (θ_n = T4; tolerance 1/10):

| step-1 cell | exact | schema_valid | pass-2 median ms | steps mean |
|---|---|---|---|---|
| cd_always:native/T1/named/perstep_mar | 37/40 | 40/40 | 521.0 | 12.2 |
| cd_always:native/T2/named/perstep_mar | 13/40 | 40/40 | 1219.6 | 25.1 |
| cd_always:native/T4/named/perstep_mar | 33/40 | 40/40 | 443.8 | 9.8 |

Step 2 (tolerance 1/10; best eligible exact 39/40, bar 7/8):

| policy | format | eligible | kept | within 5 % | exact | schema_valid | CS false | e2e p50/p90 ms (tool) | steps mean |
|---|---|---|---|---|---|---|---|---|---|
| draft_only | cc | False | False | False | 31/40 | 34/37 | 0 | 271.3/346.3 | 0.0 |
| draft_only:native | native | True | True | True | 39/40 | 40/40 | 0 | 224.0/315.9 | 0.0 |
| cd_always:T1/named/end_projection | cc | True | False | False | 28/40 | 37/37 | 0 | 555.0/656.8 | 7.3 |
| cd_on_invalid:T1/named/end_projection | cc | True | False | False | 32/40 | 37/37 | 0 | 276.8/371.8 | 0.5 |
| cd_always:T1/named/perstep_mar | cc | True | False | False | 21/40 | 37/37 | 0 | 989.8/1844.0 | 15.4 |
| cd_on_invalid:T1/named/perstep_mar | cc | True | False | False | 32/40 | 37/37 | 0 | 276.8/403.8 | 1.1 |
| cd_always:T1/named/perstep_mar_qargmax | cc | True | False | False | 23/40 | 37/37 | 0 | 1020.4/1604.6 | 15.8 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | cc | True | False | False | 31/40 | 37/37 | 0 | 276.8/403.8 | 1.6 |
| cd_always:T1/named/perstep_mar+seed | cc | True | False | False | 31/40 | 37/37 | 0 | 461.1/553.2 | 2.1 |
| cd_on_invalid:T1/named/perstep_mar+seed | cc | True | False | False | 31/40 | 37/37 | 0 | 276.8/354.3 | 0.2 |
| cd_always:native/T4/named/end_projection | native | True | False | False | 33/40 | 40/40 | 0 | 436.3/592.3 | 5.6 |
| cd_on_invalid:native/T4/named/end_projection | native | True | True | True | 39/40 | 40/40 | 0 | 224.0/315.9 | 0.0 |
| cd_always:native/T4/named/perstep_mar | native | True | False | False | 33/40 | 40/40 | 0 | 701.3/1662.1 | 9.8 |
| cd_on_invalid:native/T4/named/perstep_mar | native | True | True | True | 39/40 | 40/40 | 0 | 224.0/315.9 | 0.0 |
| cd_always:native/T4/named/perstep_mar_qargmax | native | True | False | False | 34/40 | 40/40 | 0 | 735.1/1944.2 | 11.9 |
| cd_on_invalid:native/T4/named/perstep_mar_qargmax | native | True | True | True | 39/40 | 40/40 | 0 | 224.0/315.9 | 0.0 |
| cd_always:native/T4/named/perstep_mar+seed | native | True | True | False | 39/40 | 40/40 | 0 | 441.5/529.2 | 2.0 |
| cd_on_invalid:native/T4/named/perstep_mar+seed | native | True | True | True | 39/40 | 40/40 | 0 | 224.0/315.9 | 0.0 |

- cross-format McNemar (reported only), cd_on_invalid:T1/named/perstep_mar+seed vs draft_only:native (b: cc exact and native not; c: native exact and cc not): n 40, b 0, c 8, p 0.007812 (Bonferroni over 105: 0.8203); both parsed n 37, b 0, c 5


---

## Test (T-0084f at 794161d, 2026-10-11, run once) — winner: native, confirmed
**Regime.** As for dev, except:
- test split, 80 tool + 12 no-tool cases;
- 8 cells (cc T1/named × 4 + native T4/named × 4);
- step-2 tolerance 1/20;
- θ_n T4, taken from the dev selection `logs/t0084-native-dev-794161d/selection-native-dev.json` (sha 7bb5c4de…);
- run_id de51faf2ea3c.

**Evidence:** `logs/t0084-native-test-794161d/` (artifact 14728634…, selection 89ec58e0…). The pre-registration note: the native prompt, the
rule and θ_n were fixed on dev before any test request.

**Decision** (re-derived independently by two reviewers): the bar is 70/80. No cc policy is kept (best 63/80). The kept native policies tie
at p50/p90 227.4/352.3 ms, and kind order gives **`draft_only:native`, winner native**. The decision is robust: it is unchanged after
dropping any one case or the first 1–10 cases.

| | native draft | incumbent `cd_on_invalid:T1/named/end_projection` (H-T82) | best cc (`cd_on_invalid:T1/named/perstep_mar`) |
|---|---|---|---|
| exact | **74/80** | 62/80 | 63/80 |
| schema-valid | 80/80 | 69/69 | 69/69 |
| route recall | 80/80 | 69/80 | 69/80 |
| e2e p50/p90 (tool) | **227.4/352.3 ms** | 269.7/436.6 ms | 269.7/842.0 ms |
| draft parse failures | 0 | 12 (`not JSON` 8, `no arguments` 4) | — |
| spoken sentence | 54/80 (google_search 1/20) | 69/69 | 69/69 |

- **Per tool, native draft vs the incumbent:** search 20 vs 15, click 18 vs 16, type 19 vs 17, scroll 9 vs 7, key 8 vs 7.
- **McNemar (exact; b = cc-only exact, c = native-only exact)**, Bonferroni over this report's family of 29:
  - vs the incumbent: b 2, c 14, p 0.0042 (0.12 adjusted);
  - vs the best cc: b 2, c 13, p 0.0074 (0.21 adjusted).
- **Dev vs test:** 97.5 % and 92.5 % for the native draft; 80 % and 78.75 % for the best cc. The effect is consistent.

**Findings:**
1. **The scorer's keyword rule hides a native-T4 CD defect.** Unseeded native T4 and T2 constrained calls for google_search write the
   leaked thought plus garbage into the query (19/20 per cell on test, 10/10 on dev), and the lenient `exact` (keywords contained) counts
   them as correct.
   - The `cd_always:native/T4` unseeded exact counts (65/66/68 of 80) are therefore **not real**.
   - On dev this decided θ_n = T4. Without it, θ_n would have been T1.
   - The choice and the winner are unaffected, because `draft_only:native` never runs CD.
   - A native CD fallback should use the seed arm or T1, not unseeded T4. The keyword rule needs tightening (follow-up).
2. **The cc recall of 69/80** (76/80 in T-0080c's earlier sample) is a fresh stochastic sample: 10 cases lost and 3 gained, p 0.09. 7 of the
   11 misses are malformed `<cc {` opens (T-0083). Recovering them would give cc at most about 69/80 exact, still below the bar.
3. **The stray `<channel|>` appears in 41/80 native answers**, all of them spoken cases. Native's spoken channel depends on T-0085's router.
4. **Constrained decoding adds nothing for native:** all 80 native drafts are valid.


### H-T84-2 (human, 2026-10-11): ADOPT NATIVE
The default tool-call policy becomes **`cd_on_invalid:native/T4/named/perstep_mar+seed`**:
- a schema-valid native draft (`<|tool_call>call:…<tool_call|>` parsed by `native.parse_call`) is sent as written, which on test is
  identical to the measured winner `draft_only:native` (74/80, 227/352 ms);
- an invalid native draft gets one seeded constrained native pass 2. That is the seed arm, whose search queries were clean; the unseeded T4
  CD wrote garbage queries.

This replaces H-T82's `cd_on_invalid:T1/named/end_projection`.

**Caveat:** no native draft was invalid in 120 dev and test cases, so the fallback path is unmeasured on real invalid drafts.

**H-T99d/H-T99e amendment (human, 2026-10-15; implemented by T-0100).** Not every invalid native draft is repaired. The live judge
(`toolcall/live.py::_judge`, shared by `classify` and `repair_verdict`) refuses at detection, with no pass 2:
- **H-T99d:** a `press_key` draft whose `combo` is not a string that maps into the list after a small alias table (whitespace squashed,
  casefolded, whole `+` parts mapped Esc→Escape, Return→Enter, Control→Ctrl, Del→Delete, Up/Down/Left/Right→Arrow*; Cmd is not mapped):
  reason `"combo not in enum"`; or a `press_key` draft with no readable combo: reason `"no combo"`;
- **H-T99e:** a `click_at`/`type_at` draft whose read-back `x` or `y` is not an integer in 0–999: reason `"coordinate out of range"`.

A spelling that maps into the list stays repairable (the repair writes the exact string); in-range click/type drafts invalid for another
reason, and every `scroll` and `google_search` draft, are repaired as before. The eval (`eval.cd`) is unchanged: it measures the model,
these are live policy. In the T-0099 run at a794a98 the repair coerced Ctrl+S/Ctrl+F to Ctrl+C (5×) and truncated x > 999 to a wrong point
(4×). Re-deriving that run's report at a commit after T-0100 gives different numbers (those drafts now count as refused); the report as
published is reproducible only at a commit ≤ 0756893.

**H-T101c amendment (human, 2026-10-15; implemented by T-0102).** A not-valid native draft with more model text after its call is refused
at detection with reason `"model text after call"`, after the H-T99d/H-T99e rules (which keep their reasons), for all five tools. The
call's end is found by a string-aware depth walk from `call:NAME{` (the closing brace, past a following `<tool_call|>`, or a call tag);
the remainder holds model text iff, with every special literal replaced by a space, it has a letter or digit. Clean cuts (no end found),
remainders of whitespace, special literals or punctuation, and a second call after the first keep today's verdict; the stop id is not
used. In the T-0101 run at df379e4 (kout-03 rep 2) the model closed `press_key{combo:Ctrl+T}` and then wrote that it could not do
Ctrl+Shift+T; the repair closed it as Ctrl+T and Ctrl+T ran as a dry run. The eval is unchanged. Definitions in `docs/tasks/T-0102.md`.

**Follow-ups:**
- tune the native prompt so the assistant speaks before calls (human: later, as its own measured task);
- tighten the google_search `exact` rule (keywords contained);
- the conductor's live path (T-0080d) must use T-0085's router for the spoken prefix.

---

## Test driver report (verbatim)
# Two-pass tool-call report (T-0080b driver; T-0084d format comparison)

- stage test, commit 794161d963cbc63553aa111f545542abe8eafa4d, cells ['T1/named/end_projection', 'T1/named/perstep_mar', 'T1/named/perstep_mar_qargmax', 'T1/named/perstep_mar+seed', 'native/T4/named/end_projection', 'native/T4/named/perstep_mar', 'native/T4/named/perstep_mar_qargmax', 'native/T4/named/perstep_mar+seed'], n 92, system prompt sha256 fa5dace3bd1c9c89476c922c8e3c7bb4e8030d8a8fffc8ffd349982c903bbad3, tools sha256 bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707, cases sha256 460cd1c337dddf0cacc64cca149ec6af21a90899c42450ea5791fc3409afef85, calibration {'perstep_mar': {'path': '/home/user/demos/local_astra/.claude/worktrees/t0084e-launch2/docs/cd/calibration-bfcl.json', 'sha256': '4084026c580a3148b4351699af64832e22b8e756fb288b551a17e642585ba74a', 'entropy_bound': 1.0, 'confidence_threshold': 0.005}, 'perstep_mar_qargmax': {'path': '/home/user/demos/local_astra/.claude/worktrees/t0084e-launch2/docs/cd/calibration-bfcl-perstep_mar_qargmax.json', 'sha256': 'cb81bea43c274a446a01a34187d8d1a6e2b484eb4c417f82f90c4462e54ae086', 'entropy_bound': 0.3, 'confidence_threshold': 0.005}}, thinking on, pass-1 max_tokens 1024, pass-2 {'T1': 1024, 'T2': 256, 'T4': 256}, concurrency 1, no seed, versions {'served_model': 'nvidia/diffusiongemma-26B-A4B-it-NVFP4', 'harness': 'local_astra.eval.cd (T-0070)', 'versions_lock_sha256': '5bab0570362d68c841f82f8d56cd58ddf1dd3253a2d1208f3ea2fe7109246f25', 'dgfa_commit': '27c7d0d3cc9c9e338d2fe4f6a1e64d3da01c8c03', 'compiler_version': 'dgfa_compile@27c7d0d+local_astra.T-0063;outlines-core==0.2.14', 'tokenizer_sha256': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f', 'model_revision': '2ea837236295d617ac27f8c17d61228081932c40'}
- stopped: no
- formats: cc, native, paired by case in one boot. The formats differ jointly in the prompt instructions, where the declarations live, the call syntax and the grammar (T-0084).
- format winner: native (policy draft_only:native; native thought θ_n T4; rule T-0084d criterion 1, step-2 tolerance 1/20)

| policy | exact | schema_valid | CS | pass-2 wall p50/p90 ms | e2e p50/p90 ms (tool) | e2e p50 ms (no-tool) | steps mean/p90 | prefix hit (pooled; median) |
|---|---|---|---|---|---|---|---|---|
| draft_only | 54/80 = 0.675 [0.566, 0.768] | 57/69 = 0.826 [0.720, 0.898] | n/a | n/a/n/a | 249.8/373.2 | 225.9 | n/a/n/a | n/a; n/a |
| draft_only:native | 74/80 = 0.925 [0.846, 0.965] | 80/80 = 1.000 [0.954, 1.000] | n/a | n/a/n/a | 227.4/352.3 | 213.0 | n/a/n/a | n/a; n/a |
| cd_always:T1/named/end_projection | 58/80 = 0.725 [0.619, 0.811] | 69/69 = 1.000 [0.947, 1.000] | 69/69 = 1.000 [0.947, 1.000] | 303.0/413.7 | 547.2/773.5 | 225.9 | 9.2/12.0 | 0.991; 0.993 |
| cd_always:T1/named/perstep_mar | 48/80 = 0.600 [0.490, 0.700] | 69/69 = 1.000 [0.947, 1.000] | 69/69 = 1.000 [0.947, 1.000] | 712.8/1446.4 | 974.4/1522.6 | 225.9 | 17.0/29.0 | 0.991; 0.993 |
| cd_always:T1/named/perstep_mar_qargmax | 54/80 = 0.675 [0.566, 0.768] | 69/69 = 1.000 [0.947, 1.000] | 69/69 = 1.000 [0.947, 1.000] | 764.8/1490.1 | 992.6/1577.8 | 225.9 | 16.9/29.0 | 0.991; 0.993 |
| cd_always:T1/named/perstep_mar+seed | 58/80 = 0.725 [0.619, 0.811] | 69/69 = 1.000 [0.947, 1.000] | 69/69 = 1.000 [0.947, 1.000] | 192.5/243.0 | 436.3/583.7 | 225.9 | 2.2/3.0 | 0.991; 0.993 |
| cd_always:native/T4/named/end_projection | 65/80 = 0.812 [0.713, 0.883] | 80/80 = 1.000 [0.954, 1.000] | 80/80 = 1.000 [0.954, 1.000] | 224.3/288.3 | 453.6/610.5 | 213.0 | 5.7/8.0 | 0.979; 0.977 |
| cd_always:native/T4/named/perstep_mar | 66/80 = 0.825 [0.727, 0.893] | 80/80 = 1.000 [0.954, 1.000] | 80/80 = 1.000 [0.954, 1.000] | 393.5/1546.4 | 661.6/1680.7 | 213.0 | 10.0/27.0 | 0.982; 0.981 |
| cd_always:native/T4/named/perstep_mar_qargmax | 68/80 = 0.850 [0.756, 0.912] | 80/80 = 1.000 [0.954, 1.000] | 80/80 = 1.000 [0.954, 1.000] | 426.9/1135.2 | 717.6/1379.6 | 213.0 | 9.5/20.0 | 0.982; 0.981 |
| cd_always:native/T4/named/perstep_mar+seed | 74/80 = 0.925 [0.846, 0.965] | 80/80 = 1.000 [0.954, 1.000] | 80/80 = 1.000 [0.954, 1.000] | 208.5/218.5 | 438.6/577.4 | 213.0 | 2.0/2.0 | 0.979; 0.977 |
| cd_on_invalid:T1/named/end_projection | 62/80 = 0.775 [0.672, 0.853] | 69/69 = 1.000 [0.947, 1.000] | 12/12 = 1.000 [0.757, 1.000] | 222.7/303.7 | 269.7/436.6 | 225.9 | 6.6/9.0 | 0.988; 0.985 |
| cd_on_invalid:T1/named/perstep_mar | 63/80 = 0.787 [0.686, 0.863] | 69/69 = 1.000 [0.947, 1.000] | 12/12 = 1.000 [0.757, 1.000] | 648.9/1188.5 | 269.7/842.0 | 225.9 | 15.5/24.0 | 0.988; 0.985 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | 62/80 = 0.775 [0.672, 0.853] | 69/69 = 1.000 [0.947, 1.000] | 12/12 = 1.000 [0.757, 1.000] | 702.6/1183.9 | 269.7/1021.1 | 225.9 | 16.0/24.0 | 0.988; 0.985 |
| cd_on_invalid:T1/named/perstep_mar+seed | 58/80 = 0.725 [0.619, 0.811] | 69/69 = 1.000 [0.947, 1.000] | 12/12 = 1.000 [0.757, 1.000] | 212.7/280.7 | 269.7/442.9 | 225.9 | 2.7/4.0 | 0.988; 0.985 |
| cd_on_invalid:native/T4/named/end_projection | 74/80 = 0.925 [0.846, 0.965] | 80/80 = 1.000 [0.954, 1.000] | n/a | n/a/n/a | 227.4/352.3 | 213.0 | n/a/n/a | n/a; n/a |
| cd_on_invalid:native/T4/named/perstep_mar | 74/80 = 0.925 [0.846, 0.965] | 80/80 = 1.000 [0.954, 1.000] | n/a | n/a/n/a | 227.4/352.3 | 213.0 | n/a/n/a | n/a; n/a |
| cd_on_invalid:native/T4/named/perstep_mar_qargmax | 74/80 = 0.925 [0.846, 0.965] | 80/80 = 1.000 [0.954, 1.000] | n/a | n/a/n/a | 227.4/352.3 | 213.0 | n/a/n/a | n/a; n/a |
| cd_on_invalid:native/T4/named/perstep_mar+seed | 74/80 = 0.925 [0.846, 0.965] | 80/80 = 1.000 [0.954, 1.000] | n/a | n/a/n/a | 227.4/352.3 | 213.0 | n/a/n/a | n/a; n/a |

| policy | tool | exact | schema_valid | e2e p50 ms |
|---|---|---|---|---|
| draft_only | click_at | 13/20 = 0.650 [0.433, 0.819] | 15/18 = 0.833 [0.608, 0.942] | 264.4 |
| draft_only | google_search | 11/20 = 0.550 [0.342, 0.742] | 11/19 = 0.579 [0.363, 0.769] | 179.1 |
| draft_only | press_key | 7/10 = 0.700 [0.397, 0.892] | 8/8 = 1.000 [0.676, 1.000] | 285.1 |
| draft_only | scroll | 6/10 = 0.600 [0.313, 0.832] | 6/7 = 0.857 [0.487, 0.974] | 226.9 |
| draft_only | type_at | 17/20 = 0.850 [0.640, 0.948] | 17/17 = 1.000 [0.816, 1.000] | 282.9 |
| draft_only:native | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 257.5 |
| draft_only:native | google_search | 20/20 = 1.000 [0.839, 1.000] | 20/20 = 1.000 [0.839, 1.000] | 155.3 |
| draft_only:native | press_key | 8/10 = 0.800 [0.490, 0.943] | 10/10 = 1.000 [0.722, 1.000] | 306.7 |
| draft_only:native | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 197.2 |
| draft_only:native | type_at | 19/20 = 0.950 [0.764, 0.991] | 20/20 = 1.000 [0.839, 1.000] | 251.4 |
| cd_always:T1/named/end_projection | click_at | 16/20 = 0.800 [0.584, 0.919] | 18/18 = 1.000 [0.824, 1.000] | 595.6 |
| cd_always:T1/named/end_projection | google_search | 15/20 = 0.750 [0.531, 0.888] | 19/19 = 1.000 [0.832, 1.000] | 401.4 |
| cd_always:T1/named/end_projection | press_key | 6/10 = 0.600 [0.313, 0.832] | 8/8 = 1.000 [0.676, 1.000] | 629.0 |
| cd_always:T1/named/end_projection | scroll | 6/10 = 0.600 [0.313, 0.832] | 7/7 = 1.000 [0.646, 1.000] | 459.7 |
| cd_always:T1/named/end_projection | type_at | 15/20 = 0.750 [0.531, 0.888] | 17/17 = 1.000 [0.816, 1.000] | 600.7 |
| cd_always:T1/named/perstep_mar | click_at | 12/20 = 0.600 [0.387, 0.781] | 18/18 = 1.000 [0.824, 1.000] | 936.3 |
| cd_always:T1/named/perstep_mar | google_search | 12/20 = 0.600 [0.387, 0.781] | 19/19 = 1.000 [0.832, 1.000] | 957.8 |
| cd_always:T1/named/perstep_mar | press_key | 5/10 = 0.500 [0.237, 0.763] | 8/8 = 1.000 [0.676, 1.000] | 1095.7 |
| cd_always:T1/named/perstep_mar | scroll | 7/10 = 0.700 [0.397, 0.892] | 7/7 = 1.000 [0.646, 1.000] | 777.3 |
| cd_always:T1/named/perstep_mar | type_at | 12/20 = 0.600 [0.387, 0.781] | 17/17 = 1.000 [0.816, 1.000] | 1031.9 |
| cd_always:T1/named/perstep_mar_qargmax | click_at | 14/20 = 0.700 [0.481, 0.855] | 18/18 = 1.000 [0.824, 1.000] | 957.4 |
| cd_always:T1/named/perstep_mar_qargmax | google_search | 12/20 = 0.600 [0.387, 0.781] | 19/19 = 1.000 [0.832, 1.000] | 964.6 |
| cd_always:T1/named/perstep_mar_qargmax | press_key | 6/10 = 0.600 [0.313, 0.832] | 8/8 = 1.000 [0.676, 1.000] | 1029.1 |
| cd_always:T1/named/perstep_mar_qargmax | scroll | 7/10 = 0.700 [0.397, 0.892] | 7/7 = 1.000 [0.646, 1.000] | 826.4 |
| cd_always:T1/named/perstep_mar_qargmax | type_at | 15/20 = 0.750 [0.531, 0.888] | 17/17 = 1.000 [0.816, 1.000] | 1136.5 |
| cd_always:T1/named/perstep_mar+seed | click_at | 15/20 = 0.750 [0.531, 0.888] | 18/18 = 1.000 [0.824, 1.000] | 448.0 |
| cd_always:T1/named/perstep_mar+seed | google_search | 12/20 = 0.600 [0.387, 0.781] | 19/19 = 1.000 [0.832, 1.000] | 389.3 |
| cd_always:T1/named/perstep_mar+seed | press_key | 7/10 = 0.700 [0.397, 0.892] | 8/8 = 1.000 [0.676, 1.000] | 475.3 |
| cd_always:T1/named/perstep_mar+seed | scroll | 7/10 = 0.700 [0.397, 0.892] | 7/7 = 1.000 [0.646, 1.000] | 429.6 |
| cd_always:T1/named/perstep_mar+seed | type_at | 17/20 = 0.850 [0.640, 0.948] | 17/17 = 1.000 [0.816, 1.000] | 471.9 |
| cd_always:native/T4/named/end_projection | click_at | 15/20 = 0.750 [0.531, 0.888] | 20/20 = 1.000 [0.839, 1.000] | 518.6 |
| cd_always:native/T4/named/end_projection | google_search | 20/20 = 1.000 [0.839, 1.000] | 20/20 = 1.000 [0.839, 1.000] | 340.1 |
| cd_always:native/T4/named/end_projection | press_key | 6/10 = 0.600 [0.313, 0.832] | 10/10 = 1.000 [0.722, 1.000] | 488.6 |
| cd_always:native/T4/named/end_projection | scroll | 8/10 = 0.800 [0.490, 0.943] | 10/10 = 1.000 [0.722, 1.000] | 353.2 |
| cd_always:native/T4/named/end_projection | type_at | 16/20 = 0.800 [0.584, 0.919] | 20/20 = 1.000 [0.839, 1.000] | 507.4 |
| cd_always:native/T4/named/perstep_mar | click_at | 15/20 = 0.750 [0.531, 0.888] | 20/20 = 1.000 [0.839, 1.000] | 683.9 |
| cd_always:native/T4/named/perstep_mar | google_search | 20/20 = 1.000 [0.839, 1.000] | 20/20 = 1.000 [0.839, 1.000] | 965.2 |
| cd_always:native/T4/named/perstep_mar | press_key | 6/10 = 0.600 [0.313, 0.832] | 10/10 = 1.000 [0.722, 1.000] | 591.2 |
| cd_always:native/T4/named/perstep_mar | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 414.8 |
| cd_always:native/T4/named/perstep_mar | type_at | 16/20 = 0.800 [0.584, 0.919] | 20/20 = 1.000 [0.839, 1.000] | 639.2 |
| cd_always:native/T4/named/perstep_mar_qargmax | click_at | 15/20 = 0.750 [0.531, 0.888] | 20/20 = 1.000 [0.839, 1.000] | 720.0 |
| cd_always:native/T4/named/perstep_mar_qargmax | google_search | 20/20 = 1.000 [0.839, 1.000] | 20/20 = 1.000 [0.839, 1.000] | 941.3 |
| cd_always:native/T4/named/perstep_mar_qargmax | press_key | 8/10 = 0.800 [0.490, 0.943] | 10/10 = 1.000 [0.722, 1.000] | 628.7 |
| cd_always:native/T4/named/perstep_mar_qargmax | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 427.4 |
| cd_always:native/T4/named/perstep_mar_qargmax | type_at | 16/20 = 0.800 [0.584, 0.919] | 20/20 = 1.000 [0.839, 1.000] | 700.3 |
| cd_always:native/T4/named/perstep_mar+seed | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 466.1 |
| cd_always:native/T4/named/perstep_mar+seed | google_search | 20/20 = 1.000 [0.839, 1.000] | 20/20 = 1.000 [0.839, 1.000] | 356.6 |
| cd_always:native/T4/named/perstep_mar+seed | press_key | 8/10 = 0.800 [0.490, 0.943] | 10/10 = 1.000 [0.722, 1.000] | 518.3 |
| cd_always:native/T4/named/perstep_mar+seed | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 405.6 |
| cd_always:native/T4/named/perstep_mar+seed | type_at | 19/20 = 0.950 [0.764, 0.991] | 20/20 = 1.000 [0.839, 1.000] | 461.3 |
| cd_on_invalid:T1/named/end_projection | click_at | 16/20 = 0.800 [0.584, 0.919] | 18/18 = 1.000 [0.824, 1.000] | 269.7 |
| cd_on_invalid:T1/named/end_projection | google_search | 15/20 = 0.750 [0.531, 0.888] | 19/19 = 1.000 [0.832, 1.000] | 252.8 |
| cd_on_invalid:T1/named/end_projection | press_key | 7/10 = 0.700 [0.397, 0.892] | 8/8 = 1.000 [0.676, 1.000] | 285.1 |
| cd_on_invalid:T1/named/end_projection | scroll | 7/10 = 0.700 [0.397, 0.892] | 7/7 = 1.000 [0.646, 1.000] | 248.0 |
| cd_on_invalid:T1/named/end_projection | type_at | 17/20 = 0.850 [0.640, 0.948] | 17/17 = 1.000 [0.816, 1.000] | 282.9 |
| cd_on_invalid:T1/named/perstep_mar | click_at | 16/20 = 0.800 [0.584, 0.919] | 18/18 = 1.000 [0.824, 1.000] | 269.7 |
| cd_on_invalid:T1/named/perstep_mar | google_search | 16/20 = 0.800 [0.584, 0.919] | 19/19 = 1.000 [0.832, 1.000] | 252.8 |
| cd_on_invalid:T1/named/perstep_mar | press_key | 7/10 = 0.700 [0.397, 0.892] | 8/8 = 1.000 [0.676, 1.000] | 285.1 |
| cd_on_invalid:T1/named/perstep_mar | scroll | 7/10 = 0.700 [0.397, 0.892] | 7/7 = 1.000 [0.646, 1.000] | 248.0 |
| cd_on_invalid:T1/named/perstep_mar | type_at | 17/20 = 0.850 [0.640, 0.948] | 17/17 = 1.000 [0.816, 1.000] | 282.9 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | click_at | 16/20 = 0.800 [0.584, 0.919] | 18/18 = 1.000 [0.824, 1.000] | 269.7 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | google_search | 15/20 = 0.750 [0.531, 0.888] | 19/19 = 1.000 [0.832, 1.000] | 252.8 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | press_key | 7/10 = 0.700 [0.397, 0.892] | 8/8 = 1.000 [0.676, 1.000] | 285.1 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | scroll | 7/10 = 0.700 [0.397, 0.892] | 7/7 = 1.000 [0.646, 1.000] | 248.0 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | type_at | 17/20 = 0.850 [0.640, 0.948] | 17/17 = 1.000 [0.816, 1.000] | 282.9 |
| cd_on_invalid:T1/named/perstep_mar+seed | click_at | 15/20 = 0.750 [0.531, 0.888] | 18/18 = 1.000 [0.824, 1.000] | 269.7 |
| cd_on_invalid:T1/named/perstep_mar+seed | google_search | 12/20 = 0.600 [0.387, 0.781] | 19/19 = 1.000 [0.832, 1.000] | 252.8 |
| cd_on_invalid:T1/named/perstep_mar+seed | press_key | 7/10 = 0.700 [0.397, 0.892] | 8/8 = 1.000 [0.676, 1.000] | 285.1 |
| cd_on_invalid:T1/named/perstep_mar+seed | scroll | 7/10 = 0.700 [0.397, 0.892] | 7/7 = 1.000 [0.646, 1.000] | 248.0 |
| cd_on_invalid:T1/named/perstep_mar+seed | type_at | 17/20 = 0.850 [0.640, 0.948] | 17/17 = 1.000 [0.816, 1.000] | 282.9 |
| cd_on_invalid:native/T4/named/end_projection | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 257.5 |
| cd_on_invalid:native/T4/named/end_projection | google_search | 20/20 = 1.000 [0.839, 1.000] | 20/20 = 1.000 [0.839, 1.000] | 155.3 |
| cd_on_invalid:native/T4/named/end_projection | press_key | 8/10 = 0.800 [0.490, 0.943] | 10/10 = 1.000 [0.722, 1.000] | 306.7 |
| cd_on_invalid:native/T4/named/end_projection | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 197.2 |
| cd_on_invalid:native/T4/named/end_projection | type_at | 19/20 = 0.950 [0.764, 0.991] | 20/20 = 1.000 [0.839, 1.000] | 251.4 |
| cd_on_invalid:native/T4/named/perstep_mar | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 257.5 |
| cd_on_invalid:native/T4/named/perstep_mar | google_search | 20/20 = 1.000 [0.839, 1.000] | 20/20 = 1.000 [0.839, 1.000] | 155.3 |
| cd_on_invalid:native/T4/named/perstep_mar | press_key | 8/10 = 0.800 [0.490, 0.943] | 10/10 = 1.000 [0.722, 1.000] | 306.7 |
| cd_on_invalid:native/T4/named/perstep_mar | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 197.2 |
| cd_on_invalid:native/T4/named/perstep_mar | type_at | 19/20 = 0.950 [0.764, 0.991] | 20/20 = 1.000 [0.839, 1.000] | 251.4 |
| cd_on_invalid:native/T4/named/perstep_mar_qargmax | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 257.5 |
| cd_on_invalid:native/T4/named/perstep_mar_qargmax | google_search | 20/20 = 1.000 [0.839, 1.000] | 20/20 = 1.000 [0.839, 1.000] | 155.3 |
| cd_on_invalid:native/T4/named/perstep_mar_qargmax | press_key | 8/10 = 0.800 [0.490, 0.943] | 10/10 = 1.000 [0.722, 1.000] | 306.7 |
| cd_on_invalid:native/T4/named/perstep_mar_qargmax | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 197.2 |
| cd_on_invalid:native/T4/named/perstep_mar_qargmax | type_at | 19/20 = 0.950 [0.764, 0.991] | 20/20 = 1.000 [0.839, 1.000] | 251.4 |
| cd_on_invalid:native/T4/named/perstep_mar+seed | click_at | 18/20 = 0.900 [0.699, 0.972] | 20/20 = 1.000 [0.839, 1.000] | 257.5 |
| cd_on_invalid:native/T4/named/perstep_mar+seed | google_search | 20/20 = 1.000 [0.839, 1.000] | 20/20 = 1.000 [0.839, 1.000] | 155.3 |
| cd_on_invalid:native/T4/named/perstep_mar+seed | press_key | 8/10 = 0.800 [0.490, 0.943] | 10/10 = 1.000 [0.722, 1.000] | 306.7 |
| cd_on_invalid:native/T4/named/perstep_mar+seed | scroll | 9/10 = 0.900 [0.596, 0.982] | 10/10 = 1.000 [0.722, 1.000] | 197.2 |
| cd_on_invalid:native/T4/named/perstep_mar+seed | type_at | 19/20 = 0.950 [0.764, 0.991] | 20/20 = 1.000 [0.839, 1.000] | 251.4 |

## Format cc

- cases 92; route recall by the `cc` tag (tool cases): 69/80 = 0.863 [0.770, 0.921]; by any tool-call tag: 69/80 = 0.863 [0.770, 0.921]; false-tag rate (no-tool cases): 0/12 = 0.000 [0.000, 0.243]
- routes with another format's tag (wrong_tag): 0; open_thought_call: n/a
- draft parse failures: {'no arguments': 4, 'not JSON': 8}
- spoken prefix non-empty (routed tool cases): 69/69 = 1.000 [0.947, 1.000]; median chars 50.0; text after the close: n/a
- pass-1 wall ms p50/p90 246.9/368.4 (n 92); pass-1 prompt tokens p50 n/a (n 0); pass-1 prefix hit median 0.977
- drafts with a stripped close, logs/t0084-native-test-794161d/20261011T031521.606364Z-toolcall-test.jsonl: 32/69 routed
- cc stage line: {'rule': 'stage2', 'choice': 'cd_on_invalid:T1/named/end_projection'}
- pass-2 wall ms by position within a case (T-0082b R10; first cell vs later cells): first p50/p90 284.2/795.7 (n 69); later p50/p90 515.9/1209.4 (n 207)

McNemar (exact, tool cases; Bonferroni over 14):

- cd_always:T1/named/end_projection vs draft_only: n 80, b 8, c 4, p 0.3877 (adj 1); both parsed n 57, b 0, c 4, p 0.125
- cd_always:T1/named/perstep_mar vs draft_only: n 80, b 9, c 15, p 0.3075 (adj 1); both parsed n 57, b 0, c 15, p 6.104e-05
- cd_always:T1/named/perstep_mar_qargmax vs draft_only: n 80, b 8, c 8, p 1 (adj 1); both parsed n 57, b 0, c 8, p 0.007812
- cd_always:T1/named/perstep_mar+seed vs draft_only: n 80, b 4, c 0, p 0.125 (adj 1); both parsed n 57, b 0, c 0, p 1
- cd_on_invalid:T1/named/end_projection vs draft_only: n 80, b 8, c 0, p 0.007812 (adj 0.1094); both parsed n 57, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar vs draft_only: n 80, b 9, c 0, p 0.003906 (adj 0.05469); both parsed n 57, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar_qargmax vs draft_only: n 80, b 8, c 0, p 0.007812 (adj 0.1094); both parsed n 57, b 0, c 0, p 1
- cd_on_invalid:T1/named/perstep_mar+seed vs draft_only: n 80, b 4, c 0, p 0.125 (adj 1); both parsed n 57, b 0, c 0, p 1
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar: n 80, b 14, c 4, p 0.03088 (adj 0.4324); both parsed n 69, b 14, c 4, p 0.03088
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar_qargmax: n 80, b 7, c 3, p 0.3438 (adj 1); both parsed n 69, b 7, c 3, p 0.3438
- cd_always:T1/named/end_projection vs cd_always:T1/named/perstep_mar+seed: n 80, b 4, c 4, p 1 (adj 1); both parsed n 69, b 4, c 4, p 1
- cd_always:T1/named/perstep_mar vs cd_always:T1/named/perstep_mar_qargmax: n 80, b 7, c 13, p 0.2632 (adj 1); both parsed n 69, b 7, c 13, p 0.2632
- cd_always:T1/named/perstep_mar vs cd_always:T1/named/perstep_mar+seed: n 80, b 5, c 15, p 0.04139 (adj 0.5795); both parsed n 69, b 5, c 15, p 0.04139
- cd_always:T1/named/perstep_mar_qargmax vs cd_always:T1/named/perstep_mar+seed: n 80, b 4, c 8, p 0.3877 (adj 1); both parsed n 69, b 4, c 8, p 0.3877

T1 `101` position: {'T1/named/end_projection': {'100': 1, '101': 1, '104': 1, '105': 1, '109': 2, '119': 1, '127': 1, '136': 1, '148': 1, '166': 1, '17': 1, '18': 1, '205': 1, '337': 1, '36': 1, '37': 1, '38': 1, '40': 1, '41': 1, '43': 1, '44': 2, '45': 6, '46': 3, '47': 3, '48': 1, '50': 1, '51': 1, '54': 1, '58': 2, '59': 2, '60': 1, '61': 2, '64': 1, '66': 1, '68': 2, '69': 1, '70': 1, '71': 1, '73': 1, '76': 1, '79': 1, '85': 1, '86': 2, '88': 1, '92': 1, '94': 1, '95': 3, '96': 1, '97': 2, '98': 1}, 'T1/named/perstep_mar': {'102': 1, '105': 1, '109': 1, '110': 1, '111': 2, '117': 1, '128': 1, '130': 1, '153': 1, '178': 1, '18': 2, '23': 1, '39': 2, '40': 3, '42': 1, '44': 4, '444': 1, '45': 2, '46': 1, '47': 6, '48': 2, '51': 2, '53': 1, '54': 1, '57': 2, '58': 1, '59': 1, '61': 1, '62': 1, '63': 1, '67': 1, '68': 1, '69': 1, '70': 1, '71': 1, '74': 1, '75': 2, '76': 1, '78': 1, '79': 1, '81': 1, '85': 1, '86': 1, '88': 1, '91': 1, '92': 1, '93': 1, '94': 1, '96': 1, '98': 1, '99': 1}, 'T1/named/perstep_mar_qargmax': {'101': 1, '102': 1, '105': 2, '106': 1, '108': 1, '112': 1, '113': 1, '114': 2, '118': 1, '122': 1, '140': 1, '145': 1, '157': 1, '166': 1, '175': 1, '18': 1, '19': 1, '35': 1, '38': 2, '41': 1, '42': 3, '44': 4, '45': 3, '451': 1, '46': 1, '47': 1, '48': 3, '49': 2, '50': 1, '52': 1, '54': 2, '57': 2, '59': 1, '61': 1, '64': 1, '65': 2, '66': 1, '67': 1, '69': 2, '72': 2, '74': 1, '75': 1, '78': 2, '80': 1, '83': 1, '85': 1, '86': 1, '87': 1, '94': 1, '97': 1}, 'T1/named/perstep_mar+seed': {'104': 1, '105': 2, '137': 1, '178': 1, '25': 1, '27': 1, '28': 1, '30': 6, '31': 8, '32': 4, '33': 1, '34': 3, '36': 1, '37': 1, '44': 1, '46': 1, '49': 2, '50': 2, '53': 2, '56': 1, '58': 3, '59': 2, '60': 1, '61': 1, '64': 1, '66': 2, '67': 1, '71': 1, '76': 1, '77': 2, '78': 1, '79': 1, '84': 2, '85': 2, '86': 1, '87': 2, '89': 1, '92': 1, '96': 1, '98': 1}}

- T1/named/end_projection: pass 2 equals the valid draft 31/57 = 0.544 [0.416, 0.666]; invalid drafts repaired 12/12 = 1.000 [0.757, 1.000]; name-picker rows 0
- T1/named/perstep_mar: pass 2 equals the valid draft 17/57 = 0.298 [0.195, 0.427]; invalid drafts repaired 12/12 = 1.000 [0.757, 1.000]; name-picker rows 0
- T1/named/perstep_mar_qargmax: pass 2 equals the valid draft 21/57 = 0.368 [0.255, 0.498]; invalid drafts repaired 12/12 = 1.000 [0.757, 1.000]; name-picker rows 0
- T1/named/perstep_mar+seed: pass 2 equals the valid draft 57/57 = 1.000 [0.937, 1.000]; invalid drafts repaired 12/12 = 1.000 [0.757, 1.000]; name-picker rows 0
- seed T1/named/perstep_mar+seed: seeded 69/69 routed cases; no seed none
- seed copy T1/named/perstep_mar+seed (first canvas equals the seed through its first stop): valid drafts 54/57 = 0.947 [0.856, 0.982]; invalid drafts 0/12 = 0.000 [0.000, 0.243]
- seed trim T1/named/perstep_mar+seed (seeded rows whose draft lost trailing whitespace, R2): 0/69 = 0.000 [0.000, 0.053]

## Format native

- cases 92; route recall by the `native` tag (tool cases): 80/80 = 1.000 [0.954, 1.000]; by any tool-call tag: 80/80 = 1.000 [0.954, 1.000]; false-tag rate (no-tool cases): 0/12 = 0.000 [0.000, 0.243]
- routes with another format's tag (wrong_tag): 0; open_thought_call: 0/92 = 0.000 [0.000, 0.040]
- draft parse failures: none
- spoken prefix non-empty (routed tool cases): 54/80 = 0.675 [0.566, 0.768]; median chars 46.0; text after the close: 0/80 = 0.000 [0.000, 0.046]
- pass-1 wall ms p50/p90 224.5/327.2 (n 92); pass-1 prompt tokens p50 1070.5 (n 92); pass-1 prefix hit median 0.976
- drafts with a stripped close, logs/t0084-native-test-794161d/20261011T031521.606364Z-toolcall-test.jsonl: 0/80 routed
- native stage line: {'rule': 'stage2', 'choice': 'draft_only:native'}
- pass-2 wall ms by position within a case (T-0082b R10; first cell vs later cells): first p50/p90 213.4/571.3 (n 80); later p50/p90 267.0/880.7 (n 240)

McNemar (exact, tool cases; Bonferroni over 14):

- cd_always:native/T4/named/end_projection vs draft_only:native: n 80, b 0, c 9, p 0.003906 (adj 0.05469); both parsed n 80, b 0, c 9, p 0.003906
- cd_always:native/T4/named/perstep_mar vs draft_only:native: n 80, b 0, c 8, p 0.007812 (adj 0.1094); both parsed n 80, b 0, c 8, p 0.007812
- cd_always:native/T4/named/perstep_mar_qargmax vs draft_only:native: n 80, b 0, c 6, p 0.03125 (adj 0.4375); both parsed n 80, b 0, c 6, p 0.03125
- cd_always:native/T4/named/perstep_mar+seed vs draft_only:native: n 80, b 0, c 0, p 1 (adj 1); both parsed n 80, b 0, c 0, p 1
- cd_on_invalid:native/T4/named/end_projection vs draft_only:native: n 80, b 0, c 0, p 1 (adj 1); both parsed n 80, b 0, c 0, p 1
- cd_on_invalid:native/T4/named/perstep_mar vs draft_only:native: n 80, b 0, c 0, p 1 (adj 1); both parsed n 80, b 0, c 0, p 1
- cd_on_invalid:native/T4/named/perstep_mar_qargmax vs draft_only:native: n 80, b 0, c 0, p 1 (adj 1); both parsed n 80, b 0, c 0, p 1
- cd_on_invalid:native/T4/named/perstep_mar+seed vs draft_only:native: n 80, b 0, c 0, p 1 (adj 1); both parsed n 80, b 0, c 0, p 1
- cd_always:native/T4/named/end_projection vs cd_always:native/T4/named/perstep_mar: n 80, b 1, c 2, p 1 (adj 1); both parsed n 80, b 1, c 2, p 1
- cd_always:native/T4/named/end_projection vs cd_always:native/T4/named/perstep_mar_qargmax: n 80, b 0, c 3, p 0.25 (adj 1); both parsed n 80, b 0, c 3, p 0.25
- cd_always:native/T4/named/end_projection vs cd_always:native/T4/named/perstep_mar+seed: n 80, b 0, c 9, p 0.003906 (adj 0.05469); both parsed n 80, b 0, c 9, p 0.003906
- cd_always:native/T4/named/perstep_mar vs cd_always:native/T4/named/perstep_mar_qargmax: n 80, b 0, c 2, p 0.5 (adj 1); both parsed n 80, b 0, c 2, p 0.5
- cd_always:native/T4/named/perstep_mar vs cd_always:native/T4/named/perstep_mar+seed: n 80, b 0, c 8, p 0.007812 (adj 0.1094); both parsed n 80, b 0, c 8, p 0.007812
- cd_always:native/T4/named/perstep_mar_qargmax vs cd_always:native/T4/named/perstep_mar+seed: n 80, b 0, c 6, p 0.03125 (adj 0.4375); both parsed n 80, b 0, c 6, p 0.03125

T1 `101` position: {}

- native/T4/named/end_projection: pass 2 equals the valid draft 20/80 = 0.250 [0.168, 0.355]; invalid drafts repaired n/a; name-picker rows 0
- native/T4/named/perstep_mar: pass 2 equals the valid draft 18/80 = 0.225 [0.147, 0.328]; invalid drafts repaired n/a; name-picker rows 0
- native/T4/named/perstep_mar_qargmax: pass 2 equals the valid draft 20/80 = 0.250 [0.168, 0.355]; invalid drafts repaired n/a; name-picker rows 0
- native/T4/named/perstep_mar+seed: pass 2 equals the valid draft 80/80 = 1.000 [0.954, 1.000]; invalid drafts repaired n/a; name-picker rows 0
- seed native/T4/named/perstep_mar+seed: seeded 80/80 routed cases; no seed none
- seed copy native/T4/named/perstep_mar+seed (first canvas equals the seed through its first stop): valid drafts 78/80 = 0.975 [0.913, 0.993]; invalid drafts n/a
- seed trim native/T4/named/perstep_mar+seed (seeded rows whose draft lost trailing whitespace, R2): 0/80 = 0.000 [0.000, 0.046]

## Format comparison (T-0084d criterion 1)

Step 2 (tolerance 1/20; best eligible exact 37/40, bar 7/8):

| policy | format | eligible | kept | within 5 % | exact | schema_valid | CS false | e2e p50/p90 ms (tool) | steps mean |
|---|---|---|---|---|---|---|---|---|---|
| draft_only | cc | False | False | False | 54/80 | 57/69 | 0 | 249.8/373.2 | 0.0 |
| draft_only:native | native | True | True | True | 74/80 | 80/80 | 0 | 227.4/352.3 | 0.0 |
| cd_always:T1/named/end_projection | cc | True | False | False | 58/80 | 69/69 | 0 | 547.2/773.5 | 7.9 |
| cd_on_invalid:T1/named/end_projection | cc | True | False | False | 62/80 | 69/69 | 0 | 269.7/436.6 | 1.0 |
| cd_always:T1/named/perstep_mar | cc | True | False | False | 48/80 | 69/69 | 0 | 974.4/1522.6 | 14.7 |
| cd_on_invalid:T1/named/perstep_mar | cc | True | False | False | 63/80 | 69/69 | 0 | 269.7/842.0 | 2.3 |
| cd_always:T1/named/perstep_mar_qargmax | cc | True | False | False | 54/80 | 69/69 | 0 | 992.6/1577.8 | 14.6 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | cc | True | False | False | 62/80 | 69/69 | 0 | 269.7/1021.1 | 2.4 |
| cd_always:T1/named/perstep_mar+seed | cc | True | False | False | 58/80 | 69/69 | 0 | 436.3/583.7 | 1.9 |
| cd_on_invalid:T1/named/perstep_mar+seed | cc | True | False | False | 58/80 | 69/69 | 0 | 269.7/442.9 | 0.4 |
| cd_always:native/T4/named/end_projection | native | True | False | False | 65/80 | 80/80 | 0 | 453.6/610.5 | 5.7 |
| cd_on_invalid:native/T4/named/end_projection | native | True | True | True | 74/80 | 80/80 | 0 | 227.4/352.3 | 0.0 |
| cd_always:native/T4/named/perstep_mar | native | True | False | False | 66/80 | 80/80 | 0 | 661.6/1680.7 | 10.0 |
| cd_on_invalid:native/T4/named/perstep_mar | native | True | True | True | 74/80 | 80/80 | 0 | 227.4/352.3 | 0.0 |
| cd_always:native/T4/named/perstep_mar_qargmax | native | True | False | False | 68/80 | 80/80 | 0 | 717.6/1379.6 | 9.5 |
| cd_on_invalid:native/T4/named/perstep_mar_qargmax | native | True | True | True | 74/80 | 80/80 | 0 | 227.4/352.3 | 0.0 |
| cd_always:native/T4/named/perstep_mar+seed | native | True | True | False | 74/80 | 80/80 | 0 | 438.6/577.4 | 2.0 |
| cd_on_invalid:native/T4/named/perstep_mar+seed | native | True | True | True | 74/80 | 80/80 | 0 | 227.4/352.3 | 0.0 |

- cross-format McNemar (reported only), cd_on_invalid:T1/named/end_projection vs draft_only:native (b: cc exact and native not; c: native exact and cc not): n 80, b 2, c 14, p 0.004181 (Bonferroni over 29: 0.1212); both parsed n 69, b 2, c 6

## Re-score under the plausible-query `exact` rule (T-0086, pointer)
The T-0084f review found unseeded native T4/T2 queries that carry the leaked thought and call syntax but were scored exact. T-0086 re-scores
these artifacts offline, CPU only, under `score.query_problem`. The tables are in [`toolcall-rescore-exact.md`](toolcall-rescore-exact.md).

In short:
- **Unchanged:**
  - `draft_only:native` (39/40 dev, 74/80 test);
  - every seeded cell;
  - the T-0084e and T-0084f winner, `draft_only:native`.
- **Unseeded native T4 `cd_always` falls:** from 65/66/68 to 46/47/49 of 80 on test, and from 33/33/34 to 23/23/24 of 40 on dev.
- **Counterfactual θ_n on dev is T1, not T4.**

This is a counterfactual on data already seen. Decisions stand as recorded unless the human rules otherwise.
