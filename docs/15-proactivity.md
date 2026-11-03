# 15 — Proactivity: literature review and proposal (gaming companion)

Written 2026-10-22 from a literature review. Labels:
- **[M]** measured on this box;
- **[P]** published and measured by the authors;
- **[C]** vendor or demo claim;
- **[H]** estimate, not measured: it needs a `/launch` bench before anyone relies on it.

The milestone was set by the human on 2026-10-22, as the likely last v1 milestone. The assistant speaks not only in reply, but when something important happens in the audio or visual context, where the user defines what counts. The use case is a gaming companion: comment on play, warn or advise when something important happens on screen, and search the web. The owner's baseline is to ask the model every second whether something important happened.

## What the field does

- **A "speak or stay silent" decision per frame inside a video LLM.** Examples: VideoLLM-online's streaming EOS; ProAssist's thresholded P(EOS); EgoVoice's per-second Listen/Speak tokens; MMDuet2 outputting "NO REPLY".
  - All of them need fine-tuning.
  - Zero-shot they are poor: EgoVoice's zero-shot backbone reaches onset F1 0.11 [P].
  - Repetition is the default failure: MMDuet duplicates 81% of its replies, while MMDuet2 with an RL repetition penalty duplicates 4% [P].
- **A cheap gate, then the big model.** StreamMind runs a "cognition gate" at 100 fps; StreamBridge runs a 0.5B activation model; Dispider and Flash-VStream use asynchronous decision/response processes.
  - The class imbalance is extreme: 71:1 to 310:1 silence to response [P].
  - Em-Garde (2026) has the LLM compile the user's query into visual proposals, which a lightweight matcher then checks continuously. This is the closest template for user-defined rules here.
- **What industry ships.**
  - NVIDIA/KRAFTON PUBG Ally is event-driven from **engine game state as text**, not pixels, with a behaviour tree for reflexes [C].
  - G-Assist and Gaming Copilot are mostly on-demand. Users described Copilot's beta prompts as "constantly in your face" [C].
  - Astra's "choosing when to talk" mechanism is undisclosed [C].
  - Gemini Live's "proactive audio" filters speech not addressed to the model. It does not trigger speech from vision [C].
- **The cost of interrupting people (HCI).**
  - Horvitz: act only when the expected value beats the cost of bothering.
  - Interrupting at task breakpoints cuts frustration (Iqbal 2008) [P].
  - In CHI 2025's proactive-assistant study, the aggressive arm's user preference fell from 80–90% to 47% [P].
  - In a 2026 field study, mid-task suggestions were dismissed 62% of the time [P].
- **Duplex speech** (Moshi, VAP turn-taking). What transfers: never interject while the user holds the floor, keep pause-then-cancel barge-in (already in the turn machine), and treat silence as an active output.
- **Cheap continuous front ends.**
  - The pHash scene change already exists [M].
  - Region-of-interest (ROI) bar fill or colour probes on HUD elements, and OCR.
  - SigLIP text-to-frame similarity on CPU.
  - Audio tagging: YAMNet (3.7M params, about 12 ms) up to BEATs (mAP 0.48) [P].

## The 1 Hz baseline on this stack [H unless marked]

- **Cost.**
  - Each poll is about 650 ms of first commit with a 1120 image [M], plus a thought canvas because thinking is on. That is about 1.0–1.3 s per poll.
  - At 1 Hz, single-flight cannot keep up. It holds one of the three vLLM slots continuously and queues user turns behind it.
  - The cheapest variant is a 280-token crop on a separate prefix with one canvas: about 375 ms per poll [M-derived], or 35–40% of one slot.
- **Cache.**
  - A new image every second forces a re-prefill of at least 1120 tokens.
  - The 4 GiB KV pool (about 17k tokens) turns over in about 10–15 s. That evicts the conversation prefix, so the first user question after a quiet minute pays a full prefill, which undoes D9.
- **Latency from event to audio:** capture phase 0–1 s, plus the judge 1.0–1.3 s, plus escalation to a turn 0.65–2 s, plus TTS 0.2–0.35 s, for **about 2.5–4.5 s**. That is fine for "you're low on potions" and useless for "enemy behind you".
- **False alarms.**
  - At 60 polls a minute and one real event a minute, reaching precision ≥ 0.9 needs a per-poll false-positive rate ≤ ~0.15%. Zero-shot judges are nowhere near that.
  - Persistent states (low health for 20 s) re-fire unless the dedup state lives outside the model.

## Shared plumbing (all architectures)

1. **A text-free turn-machine event, `proactive.trigger(rule_id, severity, t_ms, frame_ref)`.** It is accepted only in IDLE; otherwise it is dropped and counted.
2. **A proactive turn is an ordinary turn.** It goes through `llm.start`, commit, the speech gate and TTS. History stores only what was played, marked `[proactive:<rule>]`. Barge-in is unchanged.
3. **Cooldowns are pure state advanced by `Tick`.** A global minimum gap, a per-rule cooldown, and a latch with hysteresis for persistent conditions.
4. **The judge is a single-flight worker, like the captioner.** It is aborted at user speech onset and never runs outside IDLE.

## Candidate architectures

**1. CPU tripwires, then an LLM verifier, then the speaking policy. Build this first.**
- **Trigger stage.** Pure functions in `perception/`: ROI bar fill with hysteresis, ROI flash or pHash, whole-frame pHash, and later OCR and a game-audio tagger. Optionally, an ROI sampler at 4–10 fps in the HUD page.
- **Verifier.** It runs only on candidates. Its prompt is a short judge prompt plus the user's rules plus a 280-token ROI crop (optionally the full frame). It returns `{speak, rule_id, line ≤ 15 words}`, parsed from committed text only.
- **Escalation.** Only "advice" severity escalates to a full turn, which may search.
- **Critical rules skip the LLM.** The user writes the line, Pocket TTS pre-renders it when the rule is created, and the reflex lane plays it. It is committed text by construction.
- **HUD.** A rule editor (natural language, severity, cooldown), ROI drawing on the "what the model sees" view, "quiet for N min" by button and by voice, and a per-turn badge showing which rules were on.
- **Budget [H].** Tens of ms of CPU per frame. On the GPU, at most about 6 verifications a minute, which is at most about 7% of one slot. A critical clip plays about 0.2–0.6 s after the event; a verified line about 0.8–1.6 s after.
- **Failure modes.**
  - ROI setup is needed per game, and HUD or resolution changes break it.
  - Events with no visual or audio signature are invisible.
  - On speakers, game audio bleeds into STT, so it is headset first.
  - The verifier still produces false positives, but per candidate rather than per second.

**2. A gated low-rate sentinel for commentary and advice.**
- **When it runs.** The same judge, run on a pHash scene change or an adaptive 5–15 s heartbeat that backs off after each silent verdict. It sees a 280-token frame and keeps a rolling summary of what it has said, so it doesn't repeat itself.
- **Breakpoint queue.** Comment-severity lines wait for a game breakpoint (loading screen, menu, death screen, lull) and expire after 30–90 s.
- **Search** only when IDLE.
- **Budget [H]:** about 2.5–7.5% of one slot.
- **Failure modes.** It misses fast events by design, can become repetitive or sycophantic, and its queued advice can go stale.

**3. A rule compiler, Em-Garde style. Later.**
- **How it works.** The LLM, given a frame, compiles each plain-language rule ("warn me if my shield breaks") into a detector. The detector is one of `roi_bar` / `roi_text` / `audio_class` / `clip_text`, with `llm_check` as the fallback. The model proposes an ROI box and the user confirms it.
- **At runtime** this is Architecture 1's CPU stage.
- **Risks:** the proposed boxes can be mis-grounded, SigLIP needs calibrating per user, and detectors break across game updates.

A trained activation model, as in StreamBridge, is a v2 route. It needs fine-tuning, which is a non-goal, and about 1 GiB of VRAM, while 4.8 GiB is free and 3 GiB of that is reserve.

## Evaluation

- **Clip set.**
  - 3–4 games of different pace, about 2 h with roughly 150–300 labelled events, plus about 30 min of negative-only footage. Recorded with OBS.
  - Replayed through the same frame contract at 1 fps, plus an optional ROI stream, so runs are deterministic on CPU.
  - Each clip is labelled under 2–3 different rule sets, so "the user defines important" is tested directly.
- **Labels.**
  - Each event has an onset, type and severity, and an acceptance window: critical [0, +2 s], advice [0, +5 s], comment [0, +20 s or the next breakpoint].
  - Persistent spans are marked, for dedup.
  - Do-not-speak spans are marked: the user talking, and cutscene dialogue.
  - A second rater labels 20%, and κ is reported.
- **Metrics.**
  - Event-level precision, recall and F1, using one-to-one matching inside the window.
  - Time from event to first audio, p50 and p95 per severity.
  - False alarms per minute; per hour on negative footage.
  - Duplicate rate.
  - AMOC (time to detect against false-positive rate).
  - Speaking during the user's speech must be 0. This is an invariant.
  - Cost to the reactive assistant: user-turn first-audio p50 and p95 with proactivity on vs off (the 10% regression rule applies), GPU slot-seconds per minute, and the prefix hit rate on the first user turn after a 60 s lull.
  - Annoyance at the desk: a 1–7 rating, barge-ins on proactive speech, and "quiet" commands per hour.
- **Arms.**

  | Arm | Description |
  |---|---|
  | A0 | Silent |
  | A1 | The owner's baseline: 1 Hz, thinking on, 1120, conversation prefix |
  | A1′ | Cheapest baseline: 1 Hz, 280 crop, separate prefix |
  | A2 | Architecture 1 |
  | A3 | Architecture 2 |
  | A4 | A2 + A3 |
  | Oracle | The judge is given the true onset frames |

  A pre-registered decision rule picks the default.

## Task sequence (each through the protocol, CPU-testable with fakes until the bench)

1. **Spec.**
   - docs/01 gains D17 "Proactivity": speaking policy, severities, cooldowns, breakpoints, and the critical-clip path.
   - docs/02 §4 gains the new machine rows and invariants.
   - Human rulings needed (H-items):
     - a thinking-off judge on a separate prefix?
     - default caps;
     - search allowed in proactive turns?
2. **Contracts.**
   - `ProactiveRule`, `TriggerCandidate`, `JudgeVerdict`, and the `proactive.trigger` event.
   - `TurnPolicy` gains `proactive` (default false) and its cooldown fields.
   - Fakes in `tests/fakes/`.
3. **Turn-machine rows (pure), as Hypothesis properties.**
   - No proactive turn outside IDLE.
   - The user always wins.
   - Minimum gap and latch.
   - History is a subsequence of what was played.
   - A differential property: with proactivity off, the machine equals a frozen copy of today's.
4. **Tripwire library.** Pure numpy ROI bar, ROI flash, and breakpoint detectors, with mutation-tested thresholds.
5. **Judge worker.** Single-flight, abort on onset, a rate limit, parsing of committed text only, and a fallback for an unclosed thought.
6. **Session wiring and speaking policy.** Tripwire to judge to trigger, the critical-clip path through the reflex lane, a deferral queue with expiry, and a per-turn log of which rules were on.
7. **HUD rule editor.** ROI drawing, "quiet for N min" by button and by voice, and the proactive badge, under `make browser`.
8. **Replay harness `scripts/replay_proactive.py`.** Precision/recall/F1, time-to-alert, false alarms per minute, duplicates and AMOC. Its tests include broken matchers.
9. **Clip set and labels** (human).
10. **GPU bench `astra bench B-proactive` via `/launch`.** Arms A0–A4 plus Oracle, with the user-latency regression arm.
11. **Later:** a game-audio tagger (PipeWire monitor plus YAMNet), ROI OCR, SigLIP rules, and the Architecture 3 rule compiler.

## Bottom line

- **A 1 Hz thinking-on judge does not fit this box [H].** It would hold a slot continuously, churn the KV pool and evict the conversation prefix, alert 2.5–4.5 s after the event, and repeat itself on any persistent state.
- **The literature and the shipped products agree on the shape.** A cheap always-on gate, then an expensive responder, then explicit silence and cooldown logic.
- **First things to measure:** the verifier's real cost at 280 tokens on a separate prefix, and how much a background judge inflates user-turn latency.

## Sources

Primary sources:
- VideoLLM-online (CVPR 2024) https://arxiv.org/abs/2406.11816
- MMDuet https://arxiv.org/abs/2411.17991
- MMDuet2 https://arxiv.org/html/2512.06810
- Dispider (CVPR 2025) https://arxiv.org/abs/2501.03218
- StreamMind (ICCV 2025) https://arxiv.org/abs/2503.06220
- StreamBridge (NeurIPS 2025) https://arxiv.org/abs/2505.05467
- ProAssist (EMNLP 2025) https://arxiv.org/abs/2506.05904
- EgoVoice (2026) https://arxiv.org/html/2610.12248
- Flash-VStream (ICCV 2025) https://arxiv.org/html/2506.23825v2
- Em-Garde (2026) https://arxiv.org/abs/2603.19054
- OmniPro (2026) https://arxiv.org/html/2605.18577v1
- ProactiveVideoQA https://arxiv.org/abs/2507.09313
- StreamingBench https://arxiv.org/abs/2411.03628
- OVO-Bench https://arxiv.org/abs/2501.05510
- Pause-aware game commentary (LREC 2026) https://aclanthology.org/2026.lrec-1.722/
- Low-latency game commentary (IJCAI 2026) https://arxiv.org/abs/2606.13322
- ProactiveBench https://arxiv.org/abs/2410.12361
- Horvitz (CHI 1999) https://erichorvitz.com/chi99horvitz.pdf
- Iqbal & Bailey (CHI 2008) https://interruptions.net/literature/Iqbal-CHI08.pdf
- "Need Help?" (CHI 2025) https://arxiv.org/abs/2410.04596
- Proactive AI field study (2026) https://arxiv.org/abs/2601.10253
- Moshi https://arxiv.org/abs/2410.00037
- VAP https://arxiv.org/abs/2401.04868
- Full-Duplex-Bench https://arxiv.org/abs/2503.04721
- PANNs https://arxiv.org/abs/1912.10211
- VideoGameQA-Bench https://arxiv.org/abs/2505.15952
- Hoai & De la Torre (AMOC)

Vendor sources [C]:
- NVIDIA PUBG Ally/ACE blog
- G-Assist
- Xbox Gaming Copilot beta
- Gemini Live API "proactive audio" docs
- Project Astra page

Some figures come from abstracts only: Em-Garde, P³IBench, and the LREC commentary paper.
