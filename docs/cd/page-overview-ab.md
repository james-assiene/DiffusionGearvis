# T-0140 B: page-overview A/B (2026-10-24, commit 8697960)

**Result: `overview unused`.** Arm B was offered `page_overview` on every turn. The model never called it, in any of the 100 turns or any task. S4 stays 0/20 in both arms.

## Regime

- **Engine:** vLLM 0.29.0 (container `vllm/vllm-openai:v0.29.0`, `services/vllm.sh` e6a1abc0 at ref 806da12), serving NVFP4 `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev 2ea8372; `max_num_seqs` 3, prefix caching on, KV 4 GiB bf16.
- **Generation:** lean profile, canvas 256, 48 steps, thinking on, repair on, max 8 tool steps, live guidance 768a973b.
- **Browser:** headless Chromium 156 (Playwright 1.64), 1280×800, loopback start pages S1–S5, one fresh browser per turn.
- **Turns:** typed; fake STT, TTS and audio; STT down.
- **Sampling:** n = 100 per arm (20 per task), order seed 0, concurrency 1, engine unseeded, 60 s turn timeout.
- **Arms:**
  - A, the live default: 1120 visual tokens, ReGround off, "screen now" label, echo in model units, thought-call recovery on, degeneracy guard on; tools v1 bbfc53cc.
  - B: A plus `--page-overview` (tools 8048bdb7).
- **Run:** `scripts/locked/t0140_overview_ab.sh` from 12:41 to 13:09 (27 min). The harness and the eval both exited 0. The eval re-derived every number from the records (16/16).
- **Logs:** `logs/t0140-overview-ab-8697960/`.

## Probes (all passed)

- **P1–P3:** image budget deltas at 280/560/1120, one image, and text only.
- **P6:** on S4's deals.html the overview shows 1.525 screens at 838×798 and scroll_y stays 0 → 0. The 400 px page gives `fits`. The hostile page shows its true 3.5 screens.
- **P7:** a realistic step-2 request with the overview is 4,916 of 16,128 tokens and carries 3 images. The overview costs **1,116 tokens**. HTTP 200, engine healthy afterwards.

## Arms

| arm | S4 (Wilson 95 %) | success (Wilson 95 %) | false claims, all (non-S4) | over-clicks | overview calls | S4 scrolls | degenerate | engine errors |
|---|---|---|---|---|---|---|---|---|
| A | 0/20 [0, 0.16] | **71/100** [0.62, 0.79] | 22 (2/80) | 2/60 | 0 | 0 | 3 | 0 |
| B | 0/20 [0, 0.16] | 66/100 [0.56, 0.75] | 28 (8/80) | 6/60 | 0 | 0 | 2 | 0 |

- **Per task, A:** S1 19, S2 19, S3 19, S4 0, S5 14 (out of 20).
- **Per task, B:** S1 18, S2 14, S3 20, S4 0, S5 14.
- **Difference in success (B − A):** Newcombe [−0.18, +0.08].
- **Both arms:** accepted, eligible, no safety violations, no configuration findings, no voids.

## What it shows

- **The model does not know the page continues, so it never asks to see more.**
  - Every S4 turn in arm B was one step: the model answers with the cheapest of the six deals visible on screen ("the enamel mug at eleven dollars and fifty cents"), and the real answer is further down the page.
  - A tool the model has no reason to call changes nothing.
  - T-0134's arm C showed the same thing, even with the page extent stated as a fact in the label.
- **Offering an unused tool may cost a little.** B is 5 points lower on success; its non-S4 false claims are 8 against 2, and its over-clicks 6 against 2.
  - All of this is within noise at n = 100 (the success difference's interval spans 0).
  - The only differences between the arms are the extra tool declaration in the prompt and the judge's tool set.
  - It is not a decision input, but it argues against offering tools that are not used.
- **Arm A at 71/100 is the best live-default number so far:** T-0135's B arm was 67, and T-0134's B arm 64.
  - Different commits, so this is history, not like-for-like.
  - The T-0142 loop guard (3 degenerate ends caught in A) and T-0135b recovery are the main changes since.

## Decision (H-T140-1)

`overview unused`, so the overview is **not adopted**. `--page-overview` stays available and off by default.

Per T-0140's implementation notes, this result is the finding. No sentence is added to `LIVE_TOOL_GUIDANCE` to make the model call the tool; that would be its own task with its own A/B.

## Next options for S4 (for the human)

1. **Show the overview without asking.** When the page is taller than about 1.05 screens, the harness puts the whole-page picture into the first step's request, so the model needs no reason to look. This tests directly whether seeing the whole page fixes S4. Cost: about 1.1k tokens on those turns.
2. **A guidance sentence:** a fact, not an instruction, e.g. "Pages can continue below the screen; page_overview shows the whole page." Its own A/B, because T-0134 R-a froze the guidance.
3. **Leave S4** until the harder task suite (T-0138) shows whether long pages matter beyond this one task.
