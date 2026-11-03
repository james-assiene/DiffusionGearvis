# 14 — Harness engineering for computer use: review and adoption plan

Written 2026-10-22, from a literature review plus this repo's measurements. The research agent's full source list is at the end. The measured numbers below are kept separate from vendor claims.

## Where we are

**T-0132** (n = 100 per arm, 1120 visual tokens; regime in `docs/cd/live-grounding-ab.md`):
- 65/100 success at the live default.
- Click hit rate 0.977–0.994, so **grounding is not the bottleneck** on these pages.
- The failures are:
  - **D-1:** the model never scrolls (S4 0/20).
  - **D-2:** 23–25 false claims per 100 turns, 20 of them on S4.
  - **D-3:** it does not trust that the screenshot is current (136/300 turns).
  - **D-4:** the echo is in pixels while the model uses 0–999 units.
  - **D-5:** it writes the call inside an unclosed thought (15/300 turns).

  Details are in `docs/tasks/T-0134.md`.

**The owner's OSWorld work with the same model family** (`<cua-diffusion-gemma clone>`):
- Base 15–20%.
- Every fine-tuned checkpoint was worse.
- The only replicated harness gain was letting the decode finish and anchoring to the last action line: +2.6 pp, p = 0.013 pooled.
- Self-reported "it worked" carries no information (99.8% vs 99.7%).
- Exact-hash change detection mislabelled 51% of true no-ops.
- Repeated clicks were 32.6% of actions.
- 321 of 350 episodes hit the step cap.
- Base won 1 of 27 infeasible tasks.

**Prompt-only fixes are measured nulls three times:**
- T-0107;
- T-0131's `LIVE_TOOL_GUIDANCE`;
- cua's Prompt V2 (58 vs 58 wins).

## The rule this review arrives at

**Give the model facts it lacks; don't give it more instructions.**

Things the harness can measure and state cost no extra model calls and fit pixel-first. Those things are:
- which image is current;
- what the last action did, in the model's units;
- whether anything visibly changed, and where;
- how much page lies beyond the screen;
- that a click repeats an earlier one.

Extra model calls (verifiers, judges, zoom) come second. They are triggered by measured trouble, never on by default.

## Techniques, ranked for this stack

Cost basis: one extra request is about 0.7–1.9 s. One 1280×800 image at 1120 is about 1092 tokens, in a 16k context.

| # | Technique | Evidence | Cost here | Status |
|---|---|---|---|---|
| 1 | Observation facts and measured action outcomes (echo, "screen now" label, extent, a pixel-diff outcome label) | EFR 2026: +5.94 pp OSWorld-Verified (diff-grounded reflection); Agent-E change observation; cua: self-reports are uninformative | 0 calls, <60 tokens | echo, label, extent in **T-0134**; outcome label in **T-0136** |
| 2 | Decode completion: recover a call lost in an open thought by letting the model close it; never execute a quoted call | the only replicated gain in this model family (cua) | 1 request on ~5% of steps | **T-0135** |
| 3 | Completion gate before claiming success | VLAA-GUI 2026: −1.4 to −2.9 pp when removed; GUI-tuned self-verifiers are weak; LLM judges are noisy | ~1 s per action turn, plus holding the answer back from TTS (highest blast radius) | **T-0139**, only if T-0134 arm B leaves non-S4 false claims ≥ baseline |
| 4 | Repeat facts and no-progress stop (no coordinate perturbation) | VLAA-GUI loop breaker halved wasted steps; cua's perturbation breaker was inert | 0 | **T-0137** |
| 5 | Page extent, then an on-demand overview tool and scroll overlap | Anthropic reference prompt; T-0134 arm C is the direct test | ~1092 tokens per overview, on request | extent in T-0134; overview in **T-0140** if arm C misses its S4 margin |
| 6 | Zoom/crop re-grounding triggered by errors | RegionFocus (ICCV 2025): +28.9–31.8% relative on ScreenSpot-Pro; UI-Zoomer; **null here at 1120** (T-0132) | 0.35 s per triggered step | **T-0141**, on dense pages only, after a no-change click |
| 7 | Best-of-N with a comparative judge | Agent S3 bBoN: 62.6 → 69.9% OSWorld (N = 10); GTA1 +3.9 pp | N× wall time plus resettable state | offline evaluation only; never live |
| 8 | Screenshot history and memory | OpenCUA: prior screenshots help; naive image memory raises grounding errors | ~1092 tokens per frame; no images allowed in the continuation | low priority; crops around past clicks if needed |
| 9 | Set-of-marks or accessibility tree | VisualWebArena SoM 16.4% vs a11y 15.1%; solves grounding, already solved here | tokens plus injection surface | off-by-default HUD checkbox only (pixel-first) |
| 10 | Planner/grounder split | Agent S3 gained +9.1 pp by **removing** hierarchy | second model's VRAM | skip |
| 11 | Infeasibility: an explicit `fail` action plus infeasible test tasks | cua base 1/27 | 0 | `fail` in **T-0140** (with tools_v2), tasks in **T-0138** |

## Adoption plan (each step behind a keyword, default off, then a locked A/B)

1. **T-0134**, in progress: echo in model units, "screen now" label, page extent, and the S4 scorer that accepts spelled-out prices. Arms A/B/C, n = 100.
2. **T-0135:** thought-call recovery (D-5; the re-derived count is 29/300, which is 18 of S3's 20 failures). The planner chose deterministic recovery of the call at the tail of the open thought, with no extra request, over a re-request; see the T-0135 rulings. The original proposal for this step follows. If a step ends in an open thought with no call, send one never-voiced continuation that closes the thought, so the model emits the call itself. Never lift a call out of the thought text.
   - Metric: silent-call rate 15/300 → ~0, and S3 success.
   - First check: in the logs, did those steps end on EOS or on a token cap?
3. **T-0136:** a measured outcome label. A pixel diff of the before and after frames, with the pointer and hover masked, in a closed vocabulary: `no visible change` / `changed near your click` / `changed elsewhere (<region>)` / `page changed`.
   - Calibrate first on hand-labelled frames from the T-0132 logs.
   - Metrics: over-clicks, non-S4 false claims.
4. **T-0137:** repeat facts and a no-progress stop. The payload gains `repeat_of_step` with that step's outcome. After 2 no-change repeats or 3 recurring screens, the call is refused with a fixed reason, and "stuck" speech becomes eligible (D16).
5. **T-0138:** a harder loopback suite, S6–S15, built before step 6. *Status 2026-10-25: suite built (792a149) and baselined (f0ec9f2, `docs/cd/suite-baseline.md`): default 70/100; S11 early-Enter 0/20, S9 inner panel 6/20, S8 far target 9/20, S14 injection followed 4/20.* Today's five tasks are near ceiling apart from S4. The new suite covers:
   - dense small targets;
   - targets 2–4 screens down;
   - an inner scrolling div;
   - multi-page navigation;
   - early-Enter forms;
   - two infeasible tasks;
   - an on-screen prompt-injection page;
   - a non-idempotent action.

   Use more tasks at fewer repeats, with paired McNemar tests.
6. **T-0139 / T-0140 / T-0141:** conditional on the earlier labels (see the table).
7. **Beyond loopback, in this order:**
   1. A self-hosted WebArena-lite subset: Docker, bound to loopback, about 50 tasks with programmatic checks, never an LLM judge.
   2. An OSWorld Chrome-domain subset on cua's VM setup. Never quote a non-random subset as an OSWorld estimate: cua's 39-task subset gave 27.8% where the full run gave 9.2%.
   3. Real websites. This needs an egress ruling and CLAUDE.md amendment from the human, and must pass the injection fixture first.
      *Status 2026-10-27 (T-0160):* James ruled on 2026-10-26 to state the injection risk rather than fix it first, and to ship real websites in v0.1, guarded and on by default in live (`--no-web` turns it off). The guards are an egress proxy that reaches public `https` hosts on port 443 only, a fresh profile with no logins, no passwords or card numbers typed, no downloads or uploads, Chromium's sandbox required, and ask-first and honour-no unchanged. The injection fixture was **not** passed first: the risk is stated in the startup notice, on `/web.html` and in the README (S14 followed 4/20, T-0138), and the injection defence is the follow-up T-0161 (bar ≤ 1/40 on S14 at n = 40 per arm).

## What not to do

- **Fine-tune.** Every cua checkpoint was below base, and `done()` precision fell with training.
- **Add prompt-only instructions.** Three measured nulls.
- **Use a coordinate-perturbation loop breaker, or majority voting.** The breaker was inert, and voting scored below pass@1.
- **Turn ReGround back on by default.** Null at 1120.
- **Add a second model or a planner split.**
- **Default to SoM or the accessibility tree.**
- **Attach full-screenshot history.**
- **Run any live judge, or use an LLM judge as the success metric.**
- **Execute calls found inside thinking or quoted text.**
- **Toggle thinking per step.**
- **Decide from single runs or non-random subsets.**

## Sources (selection; primary unless noted)

- Agashe et al., Agent S3 / bBoN, 2025, https://arxiv.org/abs/2510.02250
- Agashe et al., Agent S2, 2025, https://arxiv.org/abs/2504.00906
- Yang et al., GTA1, 2025, https://arxiv.org/abs/2507.05791
- Luo et al., RegionFocus (ICCV 2025), https://arxiv.org/abs/2505.00684
- UI-Zoomer, 2026, https://arxiv.org/abs/2604.14113
- VLAA-GUI ("Knowing When to STOP, RECOVER, and SEARCH"), 2026, https://arxiv.org/abs/2604.21375
- EFR ("Reflection with Action-Induced Visual Differences"), 2026, https://arxiv.org/abs/2608.24015
- "Naive Visual Memory is Not Enough", 2026, https://arxiv.org/abs/2606.14106
- OpenCUA, 2025, https://opencua.xlang.ai/
- Jedi, 2025, https://arxiv.org/abs/2505.13227
- OpenAI Computer-Using Agent (vendor figures), 2025, https://openai.com/index/computer-using-agent/
- Anthropic computer-use tool docs and the quickstarts `loop.py`
- VisualWebArena, 2024, https://arxiv.org/abs/2401.13649
- OSWorld, 2024, https://arxiv.org/abs/2404.07972
- WebVoyager, 2024, https://arxiv.org/abs/2401.13919
- SeeAct, 2024, https://arxiv.org/abs/2401.01614
- Agent-E, 2024, https://arxiv.org/abs/2407.13032
- Scale Labs, "Who Grades the Graders?", 2026, https://labs.scale.com/blog/verifier-design-for-cua
- `<cua-diffusion-gemma clone>`: README, EXPERIMENTS.md, outputs/notes/FAILURE_MODES.md (the owner's measurements)
