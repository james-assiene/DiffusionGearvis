# Live grounding A/Bs: T-0131 and T-0132 (2026-10-22)

These are two GPU A/B runs of live computer use in the agent's own browser. They use the five loopback start-page tasks S1–S5 from T-0111g. Each turn is typed, so there is no audio. One turn is one task, and it succeeds when the page state matches the goal.

**Shared regime:**
- profile lean, canvas length 256, 48 denoising steps, thinking on;
- vLLM 0.29.0, NVFP4 diffusiongemma-26B-A4B rev 2ea8372, `services/vllm.sh` at 806da12, `max_num_seqs` 3, prefix caching on, KV 4 GiB bf16;
- tools native:bbfc53ccdd80 live, repair on, loop of 8 steps, live guidance on;
- headless Chromium at 1280×800;
- concurrency 1, order seed 0, engine unseeded;
- no headset or speakers.

## T-0131 (c70baf0): image budget × ReGround, n = 25 per arm

| Arm | Success | Hit rate | Step done p95 | ReGround p50 |
|---|---|---|---|---|
| 280/off | 5/25 | 0.35 | 1523 ms | – |
| 280/on | 15/25 | 0.88 | 1157 ms | 407 ms |
| 560/off | 10/25 | 0.88 | 1796 ms | – |
| 560/on | 11/25 | 0.875 | 2372 ms | 425 ms |
| 1120/off | 15/25 | 1.00 | 1901 ms | – |
| 1120/on | 18/25 | 1.00 | 1756 ms | 762 ms |

The pre-registered rule (D5) labelled this "no change". Its gate "step done p95 ≤ 1.5 s" was **mis-calibrated**: the 280/off control itself failed it, at 1523 ms. The human overruled the label on 2026-10-22 and made live mode default to 1120 + ReGround. The earlier C-G1 smoke at ca9576f had 4/25 succeed, at an implicit 280 with no guidance.

**What T-0131 showed:**
- **The image budget is the big lever.** At 280 tokens, one token covers about 61 px, more than a 41-px button.
- **ReGround rescues a low budget.** At 280 it took success from 5/25 to 15/25.
- **At 1120, ReGround's +3/25 was within noise.**

## T-0132 (2ccff41): does ReGround still help at 1120, and can its crop be cheaper? n = 100 per arm

This was 300 turns, shuffled jointly. The run took 42 min, and the boot plus probes P1–P4 passed. Every request was accepted, every safety count was 0, there were 0 configuration findings, and every arm was eligible. The post-run eval re-derived everything: 33/33 rows passed. Log: `logs/t0132-crop-budget-ab-2ccff41/report.md`.

| Arm | Success (Wilson 95 %) | Hit rate | Pass-1 hit rate | False claims | Step done p95 | Pass-2 done p50 / p95 | Turn wall p50 |
|---|---|---|---|---|---|---|---|
| 1120/off | 65/100 [0.55, 0.74] | 0.977 (n = 176) | – | 25 | 1878 ms | – | 5.99 s |
| 1120/on, crop 1120 | 66/100 [0.56, 0.75] | 0.994 (n = 162) | 0.994 | 23 | 1909 ms | 724 / 2099 ms | 7.47 s |
| 1120/on, crop 280 | 64/100 [0.54, 0.73] | 0.994 (n = 154) | 0.994 | 23 | 1878 ms | 352 / 1531 ms | 6.00 s |

Per task, out of 20 each:

| Arm | S1 | S2 | S3 | S4 | S5 |
|---|---|---|---|---|---|
| 1120/off | 14 | 18 | 16 | 0 | 17 |
| 1120/on, crop 1120 | 18 | 18 | 13 | 0 | 17 |
| 1120/on, crop 280 | 17 | 19 | 11 | 0 | 17 |

Differences in success (Newcombe 95 %):
- on1120 − off: [−0.12, +0.14]
- on280 − off: [−0.14, +0.12]
- on280 − on1120: [−0.15, +0.11]

**Decision (H-T132-1, with H-T132-4/5): `reground off`.**
- Step (b) chose the 280 crop: it was not worse, and it ran pass 2 at about half the median latency (352 against 724 ms).
- Step (a) then found its success gain over 1120/off was under 5 of 100. The gain was −1.

**What T-0132 showed:**
- **At 1120 the first look already lands.** In both on arms, pass 1's counterfactual hit rate equals the final hit rate (0.994). Pass 2 moved no click from a miss to a hit. The off arm's 0.977 is 4 misses in 176 clicks.
- **ReGround's cost is real:**
  - 0.35–0.72 s median per coordinate step;
  - 2 timeouts in 162 at a 1120 crop;
  - 26 unparseable pass-2 answers at a 1120 crop.
  
  It bought no success.
- **A 280 crop halves the cost** and loses nothing measurable against a 1120 crop. This is the answer to "why is ReGround slow": the processor upscales the 512×320 crop to fill the request budget.
- **The remaining failures are not about grounding:**
  - **S4 ("find the cheapest") is 0/20 in every arm.** The model never scrolls there: 0 scrolls per S4 turn.
  - **False claims run to 23–25 per 100 turns.** That is, success claimed without verifying.
- **P4, the worst-case window,** is 9351 tokens out of 16128 (4 frames at 1120 plus a full 14 000-character continuation), and it is served. Q-4 is answered and Q-5 is not needed.

## Caveats

- n = 100 per arm resolves true gaps of about 13 pp with about 0.9 probability. A small ReGround benefit, under about 10 pp, cannot be ruled out.
- The tasks are five loopback pages with 41-px buttons. Denser real websites may need ReGround again. This was not tested.
- T-0131 is history, not like-for-like: n = 25, a different commit, and a different arm set.
- Per T-0132 C4, adopting a default from the label is a separate one-line task, and the human decides.
- 2026-10-22: adopted by T-0133 (c4497c6): live defaults to 1120 with ReGround off; `--reground` uses a 280 crop at 1120.
