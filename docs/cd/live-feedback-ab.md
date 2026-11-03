# Live feedback A/B: T-0134 (2026-10-23, e59aa69)

## Regime

- Model: profile lean, CL 256, 48 steps, thinking on, vLLM 0.29.0, NVFP4 diffusiongemma-26B-A4B rev 2ea8372, `services/vllm.sh` 806da12, one boot.
- Tools: native bbfc53ccdd80, live, repair on, loop 8, `LIVE_TOOL_GUIDANCE` 768a973b.
- Every arm: 1120 visual tokens, ReGround off.
- Browser: headless Chromium 156 at 1280×800, on loopback start pages S1–S5.
- Turns: typed, n = 100 per arm (20 per task), order seed 0, concurrency 1, engine unseeded, STT down.
- Run: 05:24, exit 0, eval 18/18 re-derived. Log: `logs/t0134-feedback-ab-e59aa69/report.md`.

## Arms

| Arm | What | Success (Wilson 95 %) | S1 | S2 | S3 | S4 | S5 | False claims (non-S4) | Over-clicks | `no_see` | coord mismatch | silent call in thought |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | baseline: pixel echo, no label | 61/100 [0.51, 0.70] | 15 | 16 | 13 | 0 | 17 | 25 (5) | 4/60 | 25 | 4 | 13 |
| B | echo in model units + "screen now" label | 64/100 [0.54, 0.73] | 15 | 17 | 17 | 0 | 15 | 24 (4) | 3/60 | 24 | 0 | 6 |
| C | B + page extent | 70/100 [0.60, 0.78] | 17 | 20 | 17 | 0 | 16 | 25 (5) | 2/60 | 26 | 1 | 3 |

Newcombe 95 % intervals on the success differences:
- B − A: [−0.10, +0.16]
- C − B: [−0.07, +0.19]
- C − A: [−0.04, +0.22]

## Decision (H-T134-1): `inconclusive`

Arm C is ineligible on acceptance. Its first failing trial was S2#15.
- At step 2 the model fell into a repetition collapse: "the. the. …", then a run of dots across 19 canvases. The LLM call timed out and the loop ended with `error`.
- The cart was nevertheless already correct, because step 1 had executed.
- That one engine timeout makes C ineligible, and the rule (C6b / O-11) turns any ineligible arm into `inconclusive`.

Had C been eligible:
- (y) would hold: B is no worse on success, and has fewer non-S4 false claims and fewer over-clicks.
- (x) would fail: S4 is 0 in both B and C.

So the label would have been `feedback only`.

## What it shows

- **The page extent did not make the model scroll.** S4 is 0/20 in every arm, with 0 scrolls. The arm-C label said "Page beyond this screen: about 0.5 screens below". The thoughts still list the six visible deals and pick the $11.50 enamel mug. Telling the model that the page continues is not enough.
- **The model-unit echo removes the coordinate confusion:** coord-mismatch thoughts fall 4 → 0/1, and over-clicks 4 → 3 → 2.
- **The label does not cure `no_see`.** "I don't see the new screenshot" thoughts stay at about 25 per arm.
- **Silent calls in thought drop 13 → 6 → 3** with the label. That is unexplained, and it is the D-5 failure T-0135 targets.
- **Success rises 61 → 64 → 70**, with gains in S2 (16 → 20) and S3 (13 → 17). It is within noise at n = 100 (C − A interval [−0.04, +0.22]).
- **Non-S4 false claims are unchanged at 4–5 per 80.** The S4 claims (20 per arm) are the wrong enamel-mug answer.

## Ruling (James, 2026-10-23)

Adopt arm B (echo + label) as the live default in T-0134c. The page extent stays off. S4 goes next to a page overview tool (T-0140).
