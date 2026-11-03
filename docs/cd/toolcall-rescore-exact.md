# Tool-call artifacts re-scored under the plausible-query `exact` rule (T-0086)

**Caveat (pre-registration).** The plausible-query rule (`score.query_problem`: empty, more than 32 words, control character, markup,
tool syntax, narration) was written after reading the stored queries. This re-score is therefore a counterfactual on data already seen,
not a pre-registered measurement. No model was run: pass-1 drafts are re-scored from their stored text, and pass-2 calls from their stored
call.

**Decisions stand as recorded unless the human rules otherwise.** Every "new" choice below is a counterfactual and not a decision
(H-T80c, H-T82, H-T84, H-T84-1 and H-T84-2 are unchanged).

## Regime
- **Run:** CPU only, no GPU, no network. `scripts/cd_toolcall_rescore_exact.py` at cfe1cf5, on a clean tree, run once per artifact (n = 1
  each). Reports are in `logs/t0086-rescore-cfe1cf5/<artifact>/report-rescore-exact.{json,md}` in the main checkout (gitignored).
- **Pinned inputs:** tokenizer `cc8d3a0c…` (rev 2ea8372), `tools_v1.json` `bbfc53cc…`, `cases.jsonl` `460cd1c3…`.
- **Old vs new:**
  - "Old" for pass 1 is the stored text re-routed by today's router and scored under the 4c7dfdc rule (ruling S1). For the T-0080c
    artifacts this is the `</cc>`-stripped draft, so their old `draft_only` is 27/40 (dev) and 47/80 (test), not the stored 18/40 and 24/80.
  - "Old" for pass 2 is the stored `exact`.
  - "New" is `score.exact` at cfe1cf5 on the same calls.
- **Selection rule per artifact (ruling S3):**
  - cc-only dev: `select("stage1")`;
  - cc-only test and both T-0082c replays: `select("stage2")`;
  - mixed artifacts: `select_format`. On test it uses the recorded θ_n = T4 for both lines.
- **Seeded cells** are derived and re-scored (ruling S4). The T-0080c seeded cells are T-0082a's "stale" seeds, built before the strip.

| run | split | artifact sha256 | report json sha256 |
|---|---|---|---|
| T-0080c | dev | `b0fd71cc…` | `ba0b5d78…` |
| T-0080c | test | `1c69a9c2…` | `a51c6bc1…` |
| T-0082c | dev | `e5148afd…` | `7312d683…` |
| T-0082c | test | `41628bdb…` | `15f1b8e8…` |
| T-0084e | dev | `06a71fc7…` | `06c821f7…` |
| T-0084f | test | `14728634…` | `4f138bd4…` |

## Headline
- **No pass-1 draft flips** in any artifact (0 flipped pass-1 rows in all six). The native win is unchanged: `draft_only:native` stays at
  39/40 (dev) and 74/80 (test).
- **No seeded cell flips.** Every `perstep_mar+seed` policy is unchanged in all six artifacts.
- **Unseeded native T4 collapses.** On test, `cd_always` falls from 65/66/68 to 46/47/49 of 80, with google_search going from 20/20 to
  1/20. On dev it falls from 33/33/34 to 23/23/24 of 40, with google_search 10/10 to 0/10. Unseeded native T2 on dev falls from 11/13/13
  to 1/3/3 of 40.
- **Unseeded cc T1 cells lose 0–8 per cell.**
- **Flips by reason, per artifact (from the reports):**

  | artifact | flipped | markup | narration |
  |---|---|---|---|
  | T-0080c dev | 5 | 3 | 2 |
  | T-0080c test | 15 | 3 | 12 |
  | T-0082c dev | 5 | 0 | 5 |
  | T-0082c test | 13 | 2 | 11 |
  | T-0084e dev | 62 | 60 | 2 |
  | T-0084f test | 64 | 58 | 6 |

  No other reason occurs.
- **Recorded choices that would flip** (counterfactual):
  - **T-0082c stage 2** (test): the recorded and adopted `cd_on_invalid:T1/named/end_projection` (H-T82) becomes
    `cd_on_invalid:T1/named/perstep_mar+seed`.
  - **T-0084e θ_n**: T4 becomes T1. The T-0084e winner and choice do not change (`draft_only:native`, native).
  - **Unchanged:** T-0080c stage 1 (T1), the T-0080c stage-2 recorded choice (`cd_on_invalid:T1/named/perstep_mar+seed`; see below for
    how) and the T-0084f winner (`draft_only:native`, native).

## Recorded choices against the counterfactual

| recorded choice | recorded | old line (re-score, 4c7dfdc rule) | new line (T-0086 rule) | flips? |
|---|---|---|---|---|
| T-0080c stage 1 (dev) | `T1/named/perstep_mar` | `T1/named/perstep_mar` | `T1/named/perstep_mar` | no |
| T-0080c stage 2 (test) | `cd_on_invalid:T1/named/perstep_mar+seed` (pre-strip drafts) | `cd_on_invalid:T1/named/end_projection` | `cd_on_invalid:T1/named/perstep_mar+seed` | no (see note) |
| T-0082c stage 2 (test replay) | `cd_on_invalid:T1/named/end_projection` (H-T82) | `cd_on_invalid:T1/named/end_projection` | `cd_on_invalid:T1/named/perstep_mar+seed` | **yes** |
| T-0082c (dev replay, stage-2 rule) | not a recorded decision | `cd_on_invalid:T1/named/end_projection` | `cd_on_invalid:T1/named/end_projection` | — |
| T-0084e θ_n (dev) | T4 | T4 | **T1** | **yes** |
| T-0084e winner (dev) | `draft_only:native`, native | `draft_only:native`, native | `draft_only:native`, native (at θ_n T1) | no |
| T-0084f winner (test, θ_n T4) | `draft_only:native`, native | `draft_only:native`, native | `draft_only:native`, native | no |

### How the flips happen (figures from the reports)
- **T-0082c stage 2.**
  - Under the new rule the best eligible exact is 64/80 (`cd_on_invalid` end_projection and qargmax), so the bar is 60/80.
  - `cd_on_invalid:…/perstep_mar+seed` (60/80) now clears the bar, where it fell below the old bar of 62/80.
  - Every remaining `cd_on_invalid` arm shares the e2e p50 of 316.4 ms. The p90 then decides: seed 516.3 ms against end_projection
    558.1 ms.
- **T-0080c stage 2 (note).** The recorded choice was made on the pre-strip drafts (`draft_only` schema 27/76). The old line here applies
  the strip (ruling S1) and so reproduces T-0082a's counterfactual, end_projection. The new line returns to `perstep_mar+seed` by the
  same p90 route as T-0082c (bar 58/80, seed 525.3 ms against end_projection 539.8 ms). The recorded label is therefore unchanged, but
  for a different reason than the one it was recorded for.
- **Both `+seed` outcomes have a margin of zero.**
  - T-0082c stage 2: `+seed` has 60/80 against a bar of 60/80 (best 64/80 − 1/20).
  - T-0080c test: `+seed` has 58/80 against a bar of 58/80 (best 62/80 − 1/20).
  - One row either way reverses both. One fewer exact `+seed` row, or one more exact row in the best policy, puts `+seed` below the bar.
    The choice then returns to `cd_on_invalid:T1/named/end_projection`.
- **T-0084e θ_n.**
  - Step 1's native perstep_mar exact: T1 37/40 (unchanged), T2 3/40 (was 13), T4 23/40 (was 33).
  - The bar is 33/40, which keeps only T1.
  - Step 2 over T1's family picks `draft_only:native`.
  - The T-0084f test ran only the T4 family, so a test-split check of θ_n = T1 is not available from these artifacts.
  - The H-T84-2 default fallback (`cd_on_invalid:native/T4/named/perstep_mar+seed`) is clean under the new rule: 39/40 dev, 74/80 test,
    unchanged. Whether a T1 seeded fallback would differ on real invalid drafts is the possible follow-up named in the human decision 3.

## Per artifact: exact, old against new

Flipped rows are counted per cell (pass 2) and for pass 1. The reasons are `query_problem` over the flipped rows.

### T-0080c dev (stage 1; flipped 5: markup 3, narration 2; pass 1: 0)
| policy | old exact | new exact | old google_search | new google_search |
|---|---|---|---|---|
| draft_only | 27/40 | 27/40 | 5/10 | 5/10 |
| cd_always:T1/named/perstep_mar | 21/40 | 19/40 | 3/10 | 1/10 |
| cd_always:T2/named/perstep_mar | 6/40 | 3/40 | 3/10 | 0/10 |
| cd_always:T4/named/perstep_mar | 12/40 | 12/40 | 1/10 | 1/10 |
| cd_on_invalid:T1/named/perstep_mar | 32/40 | 30/40 | 7/10 | 5/10 |
| cd_on_invalid:T2/named/perstep_mar | 30/40 | 28/40 | 7/10 | 5/10 |
| cd_on_invalid:T4/named/perstep_mar | 31/40 | 31/40 | 6/10 | 6/10 |

### T-0080c test (stage 2; flipped 15: narration 12, markup 3; pass 1: 0)
| policy | old exact | new exact | old google_search | new google_search |
|---|---|---|---|---|
| draft_only | 47/80 | 47/80 | 4/20 | 4/20 |
| cd_always:T1/named/end_projection | 57/80 | 53/80 | 12/20 | 8/20 |
| cd_always:T1/named/perstep_mar | 51/80 | 43/80 | 15/20 | 7/20 |
| cd_always:T1/named/perstep_mar_qargmax | 51/80 | 48/80 | 9/20 | 6/20 |
| cd_always:T1/named/perstep_mar+seed | 58/80 | 58/80 | 11/20 | 11/20 |
| cd_on_invalid:T1/named/end_projection | 63/80 | 60/80 | 13/20 | 10/20 |
| cd_on_invalid:T1/named/perstep_mar | 65/80 | 59/80 | 16/20 | 10/20 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | 64/80 | 62/80 | 12/20 | 10/20 |
| cd_on_invalid:T1/named/perstep_mar+seed | 58/80 | 58/80 | 11/20 | 11/20 |

### T-0082c dev replay (flipped 5: narration 5; pass 1: 0)
| policy | old exact | new exact | old google_search | new google_search |
|---|---|---|---|---|
| draft_only | 27/40 | 27/40 | 5/10 | 5/10 |
| cd_always:T1/named/end_projection | 28/40 | 27/40 | 8/10 | 7/10 |
| cd_always:T1/named/perstep_mar | 23/40 | 21/40 | 4/10 | 2/10 |
| cd_always:T1/named/perstep_mar_qargmax | 25/40 | 23/40 | 5/10 | 3/10 |
| cd_always:T1/named/perstep_mar+seed | 31/40 | 31/40 | 6/10 | 6/10 |
| cd_on_invalid:T1/named/end_projection | 35/40 | 34/40 | 10/10 | 9/10 |
| cd_on_invalid:T1/named/perstep_mar | 31/40 | 30/40 | 7/10 | 6/10 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | 34/40 | 33/40 | 9/10 | 8/10 |
| cd_on_invalid:T1/named/perstep_mar+seed | 31/40 | 31/40 | 6/10 | 6/10 |

### T-0082c test replay (flipped 13: narration 11, markup 2; pass 1: 0)
| policy | old exact | new exact | old google_search | new google_search |
|---|---|---|---|---|
| draft_only | 47/80 | 47/80 | 4/20 | 4/20 |
| cd_always:T1/named/end_projection | 62/80 | 59/80 | 13/20 | 10/20 |
| cd_always:T1/named/perstep_mar | 56/80 | 48/80 | 12/20 | 4/20 |
| cd_always:T1/named/perstep_mar_qargmax | 57/80 | 55/80 | 12/20 | 10/20 |
| cd_always:T1/named/perstep_mar+seed | 60/80 | 60/80 | 11/20 | 11/20 |
| cd_on_invalid:T1/named/end_projection | 66/80 | 64/80 | 15/20 | 13/20 |
| cd_on_invalid:T1/named/perstep_mar | 64/80 | 58/80 | 13/20 | 7/20 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | 66/80 | 64/80 | 15/20 | 13/20 |
| cd_on_invalid:T1/named/perstep_mar+seed | 60/80 | 60/80 | 11/20 | 11/20 |

### T-0084e dev (mixed; flipped 62: markup 60, narration 2; pass 1: 0)
Flipped per cell:
- native T2 and T4 `end_projection`, `perstep_mar` and `perstep_mar_qargmax`: 10 each;
- cc T1 `end_projection` and `perstep_mar_qargmax`: 1 each.

| policy | old exact | new exact | old google_search | new google_search |
|---|---|---|---|---|
| draft_only | 31/40 | 31/40 | 6/10 | 6/10 |
| draft_only:native | 39/40 | 39/40 | 10/10 | 10/10 |
| cd_always:T1/named/end_projection | 28/40 | 27/40 | 5/10 | 4/10 |
| cd_always:T1/named/perstep_mar | 21/40 | 21/40 | 2/10 | 2/10 |
| cd_always:T1/named/perstep_mar_qargmax | 23/40 | 22/40 | 4/10 | 3/10 |
| cd_always:T1/named/perstep_mar+seed | 31/40 | 31/40 | 6/10 | 6/10 |
| cd_always:native/T1/named/end_projection | 38/40 | 38/40 | 10/10 | 10/10 |
| cd_always:native/T1/named/perstep_mar | 37/40 | 37/40 | 10/10 | 10/10 |
| cd_always:native/T1/named/perstep_mar_qargmax | 39/40 | 39/40 | 10/10 | 10/10 |
| cd_always:native/T1/named/perstep_mar+seed | 39/40 | 39/40 | 10/10 | 10/10 |
| cd_always:native/T2/named/end_projection | 11/40 | 1/40 | 10/10 | 0/10 |
| cd_always:native/T2/named/perstep_mar | 13/40 | 3/40 | 10/10 | 0/10 |
| cd_always:native/T2/named/perstep_mar_qargmax | 13/40 | 3/40 | 10/10 | 0/10 |
| cd_always:native/T2/named/perstep_mar+seed | 39/40 | 39/40 | 10/10 | 10/10 |
| cd_always:native/T4/named/end_projection | 33/40 | 23/40 | 10/10 | 0/10 |
| cd_always:native/T4/named/perstep_mar | 33/40 | 23/40 | 10/10 | 0/10 |
| cd_always:native/T4/named/perstep_mar_qargmax | 34/40 | 24/40 | 10/10 | 0/10 |
| cd_always:native/T4/named/perstep_mar+seed | 39/40 | 39/40 | 10/10 | 10/10 |
| cd_on_invalid:T1/named/end_projection | 32/40 | 32/40 | 7/10 | 7/10 |
| cd_on_invalid:T1/named/perstep_mar | 32/40 | 32/40 | 7/10 | 7/10 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | 31/40 | 31/40 | 6/10 | 6/10 |
| cd_on_invalid:T1/named/perstep_mar+seed | 31/40 | 31/40 | 6/10 | 6/10 |
| cd_on_invalid:native/{T1,T2,T4}/named/{all four arms} (12 policies) | 39/40 each | 39/40 each | 10/10 each | 10/10 each |

### T-0084f test (mixed, θ_n T4; flipped 64: markup 58, narration 6; pass 1: 0)
Flipped per cell:
- native T4 `end_projection`, `perstep_mar` and `perstep_mar_qargmax`: 19 each;
- cc T1 `perstep_mar`: 3;
- cc T1 `end_projection` and `perstep_mar_qargmax`: 2 each.

| policy | old exact | new exact | old google_search | new google_search |
|---|---|---|---|---|
| draft_only | 54/80 | 54/80 | 11/20 | 11/20 |
| draft_only:native | 74/80 | 74/80 | 20/20 | 20/20 |
| cd_always:T1/named/end_projection | 58/80 | 56/80 | 15/20 | 13/20 |
| cd_always:T1/named/perstep_mar | 48/80 | 45/80 | 12/20 | 9/20 |
| cd_always:T1/named/perstep_mar_qargmax | 54/80 | 52/80 | 12/20 | 10/20 |
| cd_always:T1/named/perstep_mar+seed | 58/80 | 58/80 | 12/20 | 12/20 |
| cd_always:native/T4/named/end_projection | 65/80 | 46/80 | 20/20 | 1/20 |
| cd_always:native/T4/named/perstep_mar | 66/80 | 47/80 | 20/20 | 1/20 |
| cd_always:native/T4/named/perstep_mar_qargmax | 68/80 | 49/80 | 20/20 | 1/20 |
| cd_always:native/T4/named/perstep_mar+seed | 74/80 | 74/80 | 20/20 | 20/20 |
| cd_on_invalid:T1/named/end_projection | 62/80 | 61/80 | 15/20 | 14/20 |
| cd_on_invalid:T1/named/perstep_mar | 63/80 | 61/80 | 16/20 | 14/20 |
| cd_on_invalid:T1/named/perstep_mar_qargmax | 62/80 | 62/80 | 15/20 | 15/20 |
| cd_on_invalid:T1/named/perstep_mar+seed | 58/80 | 58/80 | 12/20 | 12/20 |
| cd_on_invalid:native/T4/named/{all four arms} (4 policies) | 74/80 each | 74/80 each | 20/20 each | 20/20 each |
