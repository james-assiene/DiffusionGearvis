# T-0151 B: overview-frame A/B (2026-10-24, commit 4471255)

**Result: `no change` under H-T151-1, and only because of the over-click guard.**
- **Every other margin passes, most by a wide margin.**
  - S4 goes from 0/20 to **20/20**.
  - S5 goes from 15/20 to **19/20**; the guard needed at least 12.
  - Success goes from 72/100 to **94/100**; Newcombe B − A is [0.118, 0.320].
  - Non-S4 false claims go from 3 to 4 of 80, inside the +2 margin.
- **The guard that fails:** over-clicks go from 1/60 to 4/60, against a margin of +2.
  - All five over-clicks (A 1, B 4) are double "blue mug" adds on S1 or S2.
  - Those tasks never get the overview frame: P8 and the turn lines show `fits` on all 60 arm-B S1–S3 turns. A2-3 makes their step-1 requests byte-identical to arm A's.
  - So the over-click difference is noise by construction, as the report's "which guards can move" line states.
  - The rule as approved still labels the run `no change`. Overriding the label is the human's decision.

## Regime

- **Engine:** vLLM 0.29.0 (`services/vllm.sh` e6a1abc0 at 806da12), NVFP4 `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev 2ea8372; `max_num_seqs` 3, prefix caching on, KV 4 GiB bf16.
- **Generation:** lean profile, CL 256, 48 steps, thinking on; repair on, loop 8; `LIVE_TOOL_GUIDANCE` 768a973b; tools v1 bbfc53cc in both arms.
- **Browser:** headless Chromium 156 (Playwright 1.64), 1280×800, loopback start pages S1–S5, one fresh browser per turn.
- **Turns:** typed, with fake STT/TTS/audio and STT down. n = 100 per arm (20 per task), order seed 0, concurrency 1, engine unseeded, 60 s turn timeout.
- **Arms:**
  - A, the live default: 1120, ReGround off, "screen now" label, echo in model units, recovery on, guard on; page, auto and frame overviews all off.
  - B: A + `overview_frame`.
- **Run:** `scripts/locked/t0151_overview_frame_ab.sh` (chmod 444), 22:32 to 23:00 (28 min). Harness exit 0; the eval re-derived everything (18/18).
- **Logs:** `logs/t0151-overview-frame-ab-4471255/`.

## Probes (all passed)

- **P1–P3:** passed.
- **P6:** deals 1.525 screens, 838×798, scroll unchanged; the hostile page reads 3.5 screens.
- **P7:** a step-1 request with the overview frame is 2,602 of 16,128 tokens, 1 image (the live-frame step 1 is 2,560). Overview tokens 1,100.
- **P8:** S1–S3 `fits`; S4 and S5 `shown` at 1.525.
- **P9:** `click_at(160, 933)` and `(160, 935)` on the overview frame each scrolled 420 px and added exactly the cotton tea towel.

## Arms

| arm | S4 | S5 | S1–S3 | success | non-S4 false claims (S5 + S1–S3) | over-clicks | S4/S5 shown | S4 names cheapest / mug | degenerate | engine errors |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0/20 [0, 0.16] | 15/20 [0.53, 0.89] | 57/60 | 72/100 [0.63, 0.80] | 3/80 (2 + 1) | 1/60 | 0 | 0 / 20 | 4 | 0 |
| B | **20/20** [0.84, 1.00] | **19/20** [0.76, 0.99] | 55/60 | **94/100** [0.88, 0.97] | 4/80 (0 + 4) | **4/60** | 20/20 | 20 / 0 | 0 | 0 |

- Both arms are accepted and eligible, with no safety violations, no configuration findings and no voids.
- Per task:
  - A: S1 20, S2 17, S3 20, S4 0, S5 15.
  - B: S1 18, S2 17, S3 20, S4 20, S5 19.
- Over-click turns:
  - A: S2 r11.
  - B: S1 r8, S2 r9, r11 and r17.
  - Every one is a cart of `['blue mug', 'blue mug']`.

## What it shows

- **The coordinate fix works.** On S5 arm B, the model's step-1 click is on the overview frame in all 20 turns. It is executed in 19 (one call never ran); all 19 scroll, and all 19 have the tea towel in the cart right after step 1.
  - There were no B9 refusals and no `OVERVIEW_NOT_REACHED`.
  - This is the failure T-0150 showed (S5 0/20, clicks at overview y on the live screen), fixed by reading the click in the overview's own coordinates.
- **S4 stays fixed:** 20/20 turns answer with the true cheapest item, none with the enamel mug.
- **Fewer steps:** arm B's median steps per turn is 2, against 3 in arm A. Arm B also has no degenerate ends (A had 4).
- **Step-1 latency is unchanged:** first commit p50 is 277 ms with the frame shown, against 299 ms without.
- **The over-click guard trips on tasks the treatment does not touch.** The +2 margin has no "only where the overview is shown" clause. On 60 identical S1–S3 requests per arm, 1 against 4 double-adds is well within chance.
- **Harness gap (minor):** the decision's `reason` is empty for `no change`; it should name the failing guard.

## Decision (H-T151-1)

`no change` by the approved rule, so the overview frame stays off by default and `--overview-frame` is available as an opt-in.

**Adopted 2026-10-25 (James, overriding the over-click guard):** T-0152 (c58f800) makes the overview frame the live default; `--no-overview-frame` restores the old one.

## For the human

1. **Adopt `--overview-frame` as the live default, overriding the over-click guard.** The guard failed only on S1/S2, where the arms send byte-identical requests.
   - Needs: a small task flipping `LIVE_DEFAULT_OVERVIEW_FRAME`, its cli and doc lines, and the re-pins.
   - Then H-T151-6, the headed desk check.
2. **Re-run the A/B with the over-click guard restricted to S5,** the only over-click task the frame touches. Same harness and n; about 30 GPU minutes.
3. **Keep it opt-in** and move on, for example to T-0138's harder task suite, before choosing a default.
