# T-0150 B: automatic page-overview A/B (2026-10-24, commit 133f0aa)

**Result: `no change` under H-T150-1.**
- **S4 goes from 0/20 to 20/20:** every arm-B S4 turn saw the whole page and named the true cheapest item.
- **S5 collapses from 13/20 to 0/20:** non-S4 false claims rise from 7 to 20 of 80, which fails the +2 guard.
- **Why S5 collapses:** the model clicks the overview's coordinates on the live screen before scrolling.

## Regime

- **Engine:** vLLM 0.29.0 (`services/vllm.sh` e6a1abc0 at 806da12), NVFP4 `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev 2ea8372; `max_num_seqs` 3, prefix caching on, KV 4 GiB bf16.
- **Decoding:** lean profile, CL 256, 48 steps, thinking on; repair on, loop 8; `LIVE_TOOL_GUIDANCE` 768a973b; tools v1 bbfc53cc in both arms.
- **Browser:** headless Chromium 156, 1280×800, loopback start pages S1–S5.
- **Turns:** typed, with fake STT/TTS/audio and STT down. n = 100 per arm (20 per task), order seed 0, concurrency 1, engine unseeded.
- **Arms:**
  - A, the live default: 1120, ReGround off, "screen now" label, echo in model units, recovery on, guard on.
  - B: A + `auto_overview`.
- **Run:** `scripts/locked/t0150_auto_overview_ab.sh`, 17:17 to 17:44 (27 min). Harness exit 0; the eval re-derived everything (17/17).
- **Logs:** `logs/t0150-auto-overview-ab-133f0aa/`.

## Probes (all passed)

- P1–P3 passed.
- P6: deals 1.525 screens, 838×798, scroll unchanged.
- P7: a step-1 request with the overview is 3,676 of 16,128 tokens, 2 images; the overview costs **1,116 tokens**.
- P8 census: S1, S2 and S3 fit; S4 and S5 are `shown` at 1.525 screens.

## Arms

| arm | S4 | S5 | S1–S3 | success | non-S4 false claims (S5 + S1–S3) | over-clicks | S4/S5 shown | S4 names cheapest / mug | degenerate | engine errors |
|---|---|---|---|---|---|---|---|---|---|---|
| A | 0/20 [0, 0.16] | 13/20 [0.43, 0.82] | 53/60 | 66/100 [0.56, 0.75] | 7/80 (4 + 3) | 3/60 | 0 | 0 / 20 | 4 | 0 |
| B | **20/20** [0.84, 1.00] | **0/20** [0, 0.16] | 52/60 | **72/100** [0.63, 0.80] | **20/80** (13 + 7) | 7/60 | 20/20 | 20 / 0 | 7 | 0 |

- Both arms are accepted and eligible, with no safety violations, no configuration findings and no voids.
- S1–S3 get no overview: their requests are byte-identical to arm A's, so 53 against 52 is noise.

## What it shows

- **Seeing the whole page fixes S4.** It solves the "it doesn't know the page continues" failure from T-0140 completely: 20/20, and the answer names the true cheapest item every time instead of the enamel mug. The mechanism worked on all 20 S4 turns.
- **It breaks S5 through coordinate confusion.**
  - **Arm A:** in every S5 turn the model scrolls first (20 scrolls), then clicks "Add to cart" at about y = 900 in model units on the scrolled screen, and succeeds 13 times out of 20.
  - **Arm B:** it sees the tea towel in the overview and clicks at about y = 935 **on the unscrolled live screen**, before scrolling; that is where the button sits in the overview's geometry, not on the screen. Then it scrolls and announces "I have added the cotton tea towel": 0/20, with 13 false claims.
  - **Why the label does not stop it:** the label says the next image is the browser screen, and the tool description says the picture is for looking only. The model still acts on the overview's coordinates.
- **Overall success rises** (66 → 72), because +20 on S4 outweighs −13 on S5. Under the approved rule the false-claim guard decides: an assistant that says "done" when it is not done is the failure the guard exists to stop.

## Decision (H-T150-1)

`no change`, so auto overview stays off by default. `--auto-overview` remains available as an opt-in for look-and-answer use.

## Next options (for the human)

1. **Mark the visible screen on the overview.**
   - Draw a rectangle on the overview where the current screen sits, and say so in the label, e.g. "the outlined box is what the browser screen shows now". Below-the-box targets then read as "not on screen yet" in pixels, which pushes the model to scroll before clicking.
   - Same A/B on S1–S5; the fact is in pixels, not instructions.
   - This is the smallest change aimed at the observed failure.
2. **Refuse a click whose target was found in the overview.** The executor cannot know that, so this needs reasoning about the model's intent. Rejected.
3. **Show the overview only after a failed attempt** (e.g. after an answer turn) or only for questions. This needs task intent the session does not have.
4. **Accept the trade** and adopt it for look-and-answer use only. It is not adoptable as a default while S5 is at 0/20.
