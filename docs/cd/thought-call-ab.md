# Thought-call recovery A/B: T-0135 (2026-10-23, e1ae76e)

## Regime

| Setting | Value |
|---|---|
| Profile | lean, canvas 256, 48 steps, thinking on |
| Engine | vLLM 0.29.0, NVFP4 diffusiongemma-26B-A4B rev 2ea8372, `services/vllm.sh` 806da12, one boot |
| Tools | native bbfc53ccdd80, live, repair on, loop 8 |
| Live default (both arms) | 1120 visual tokens, ReGround off, echo in model units, "screen now" label (T-0134c), read from `cli.grounding_of`/`cli.feedback_of` |
| Browser | headless Chromium 1280×800, loopback start pages S1–S5 |
| Turns | typed, n = 100 per arm (20 per task), order seed 0, concurrency 1, engine unseeded, STT down |

The harness exited 0; the eval exited 1 (one row; see below). Log: `logs/t0135-thought-call-ab-e1ae76e/report.md`.

## Arms

| Arm | Recovery | Success (Wilson 95 %) | S1 | S2 | S3 | S4 | S5 | False claims | Over-clicks | Silent call in thought | Recoveries |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | off | 66/100 [0.56, 0.75] | 17 | 18 | 16 | 0 | 15 | 27 | 3/60 | 4 | 0 |
| B | on | 67/100 [0.57, 0.75] | 18 | 17 | **20** | 0 | 12 | 27 | 4/60 | **0** | 11 (2 turns with 2+) |

Newcombe 95 % intervals, B − A:
- success: [−0.12, +0.14]
- S3: [−0.00, +0.42]
- silent call in thought: [−0.10, +0.00]

## Decision (H-T135-1): `inconclusive`

Arm B is ineligible on acceptance because of one trial, B S2#5:
- At step 2 the model collapsed into "the. the. the. …" in the **answer channel**.
- That text was voiced: about 200 "the." utterances went to TTS.
- The turn hit the 60 s turn timeout. Its cart was already correct.

Had B been eligible, every H-T135-1 clause holds (success 67 ≥ 63, S3 20 ≥ 16, over-clicks 4 ≤ 5, false claims 27 ≤ 30, safety 0), so the label would have been **`recover`**.

The eval exited 1 on `test_arm_setting_reached_every_turn`. A timed-out turn writes no turn line, so the A8 key is missing for B S2#5. This is an eval defect: the rule should exempt turns with no turn line. Every metric, finding, probe and decision row re-derived and matched.

## What it shows

- **Recovery fixes the D-5 failure.** S3 goes from 16 to 20/20, and silent tail calls in thought from 4 to 0. 11 calls were recovered.
- **No measurable harm elsewhere.** False claims are 27 vs 27 and over-clicks 3 vs 4. S5 at 15 → 12 is within noise at n = 20.
- **A new failure mode.** A repetition collapse in the answer channel is spoken aloud. T-0142 currently guards thought only (R-o, "no evidence of answer degeneracy"); this run is that evidence.
- **The single-timeout ineligibility rule decided a run for the second time** (T-0134 arm C, T-0135 arm B). T-0140's rule already relaxes it to more than 2 engine errors per arm.

## Ruling (James, 2026-10-23)

- Recovery becomes the live default (T-0135b).
- The loop guard is extended to the spoken answer channel (T-0142b).

**Adopted:** T-0135b merged 2026-10-23 at 876bf85: recovery is on by default for `astra session --tools live` (`--thought-call-recovery off` restores arm A). The answer-channel guard is T-0142b.
