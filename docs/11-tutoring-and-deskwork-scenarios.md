# Tutoring and desk-work scenarios for DiffusionGemma — grounded in its published numbers

**Status:** 2026-09-12. Scope set by James: education/tutoring and at-desk, on-screen knowledge work; coding is out. Every scenario below is tied to a benchmark the model has actually been scored on, to the thinking-vs-no-thinking delta on that benchmark, and to an estimated wall time on the RTX 5090. Where the numbers say the model is weak, the scenario says so and the mitigation is part of the scenario.

## 1. What the numbers actually support

All figures are DiffusionGemma-26B-A4B-it in text-diffusion mode from the tech report (arXiv 2608.00146, Tables 3–4) and the model card, with Gemma 4 12B from the Gemma 4 report for the comparison the demo will run.

| Capability | Benchmark | DG think | DG no-think | Δ think | Gemma 4 12B | Reading |
|---|---|---|---|---|---|---|
| School maths | GSM8K | 96.3 | 95.8 | +0.5 | — | solved; thinking unnecessary |
| Mid-level maths | HiddenMath (Google) | 80.6 | 74.3 | +6.3 | — | strong, thinking pays |
| Multilingual maths | MGSM | 84.8 | 80.7 | +4.1 | — | good; French tutoring plausible |
| Competition maths | AIME 2026 | 69.1 | 50.8 | **+18.3** | 77.5 | thinking essential; 12B more accurate |
| Undergraduate maths | Putnam | 67.4 | 57.1 | **+10.3** | — | thinking essential |
| Graduate science QA | GPQA Diamond | 73.2 | 64.6 | **+8.6** | 78.8 | strong; 12B slightly better |
| Hard reasoning | BIG-Bench Extra Hard | 47.6 | 40.0 | +7.6 | 53.0 | weakest reasoning number |
| Knowledge | MMLU-Pro | 77.6 | 77.9 | −0.3 | 77.2 | on par with 12B; no thinking needed |
| Multilingual knowledge | MMMLU | 81.5 | 76.3 | +5.2 | 83.4 | good |
| Instruction following | IFEval | 97.4 | 94.5 | +2.9 | 97.2 | a genuine strength: format-constrained artifacts |
| Policy-following multi-step | Tau2 retail / airline | 71.5 / 69.0 | 57.5 / 49.0 | **+14 / +20** | 77.6 / 75.0 | rule application benefits most from thinking |
| Visual maths | MATH-Vision | 70.5 | — | — | 79.7 | usable; 12B clearly better |
| Visual reasoning | MMMU-Pro | 54.3* | 66.0 | *artifact | 69.1 | see §2.1 — thinking is slightly *better* once the thought is closed |
| Document parsing | OmniDocBench 1.5 (edit distance, lower better) | 0.319 | — | — | 0.164 | **weak**: worse than every Gemma 4 size, including E2B |
| Long-context retrieval | MRCR v2 8-needle 128k | 32.0 | — | — | 43.4 | **weak**: keep inputs bounded |
| Frontier knowledge | HLE (no tools) | 11.0 | — | — | 5.2 | the one number above Gemma 4 26B (8.7) |

Two things the table says loudly. First, the model is at its best when the task is *text reasoning with a rulebook*: maths, science QA, policy application and format-constrained writing, and thinking is worth 6–20 points there. Second, its two documented weaknesses are exactly the ones a screen assistant leans on — dense document parsing and long-context retrieval — so the design has to route around them (§2.2) rather than hope.

### 1.1 Latency tiers on the 5090, derived from Table 4

Table 4 gives total tokens per problem with thinking on an H100. Dividing by the 5090's reported long-form rate (~700 tok/s; short replies are slower) gives a planning estimate, and dividing by ~140 tok/s gives the same for a 12B AR model on the same card (a 12B thinks *longer*, not shorter — Gemma 4 26B averages 7,207 tokens with thinking against DiffusionGemma's 4,001 — so these AR estimates are generous).

| Tier | Benchmark class | Tokens (think) | DiffusionGemma on 5090 | 12B AR on 5090 | Interaction pattern |
|---|---|---|---|---|---|
| A — quick think | GSM8K 883 · MGSM 1,085 · IFEval 1,464 | ≤ 2,000 | **1.3–2.1 s** | 6–10 s | one acknowledgement, then the answer; inside the socially acceptable pause |
| B — "give me a moment" | MMMU-Pro 3,178 · Putnam 4,725 · GPQA 5,647 · AIME 6,445 | 3,000–6,500 | **4.5–9 s** | 23–46 s | contextual acknowledgement, thinking meter on the HUD, an interim spoken update at ~4 s |
| C — "I'll work on it while we talk" | BBEH 9,062 | > 8,000 | **~13 s** | ~65 s | the GPT-Live pattern: keep the conversation going, deliver when done |

The demo argument is tier B. Tier A is where the 12B is also fine. Tier C is rare in tutoring. Tier B — a university physics problem, a proof, a multi-constraint policy check — is possible in a voice loop with DiffusionGemma and simply is not with an AR model of comparable quality on this card: nobody waits 40 seconds in a conversation.

## 2. Four design rules that fall out of the numbers

### 2.1 Close the thought (James's finding, matching report §10)

On multimodal prompts the model often puts the answer inside the thought channel and never emits the closing tag, so the parser returns nothing and the benchmark records a miss — the mechanism behind MMMU-Pro 54.3 vs 66.0. Explicitly instructing the model to close its thought before answering closes the gap, with thinking slightly ahead. Consequences: the system prompt carries an explicit "finish your thinking with the closing tag, then answer" instruction; the LLM client treats a canvas that ends inside an unclosed thought as *not yet an answer* and never voices it; X7 records the closing-tag rate per scenario class; and once the constrained-decoding plugin exists, a two-state automaton that requires the closing tag before end-of-turn is its first production grammar — the smallest possible |S|, and exactly the "structure encodes the correctness condition" case where constraints are free. Thinking therefore stays on for visual scenarios too (D11 holds).

### 2.2 Pixel-first, with text assists behind checkboxes (revised 2026-09-12)

The model reads text better than pictures of text (OmniDocBench 0.319), but the point of the demo is raw visual understanding, so the default input is pixels only and every text assist is an explicit, visible toggle, off by default: page text from the browser-tab source (`astra.frame.text`), the PDF text layer, spreadsheet cell contents, the accessibility tree. The pixel-only mitigations are: the maximum visual-token budget for documents (Gemma 4's own numbers show InfographicVQA collapsing from 89 to 78 when tokens drop from 1120 to 280); *focus crops* — the whole screen plus a high-resolution crop of the active window or the pointer region, as two images; a model-requested zoom crop if grounding proves reliable (X13); and, for camera input, the read-back in §2.4. The desk-work eval runs with assists on and off so the cost of pixel-first is measured (Q9); the demo runs with them off, and the HUD badge says so.

### 2.3 A tutoring state object instead of long history (route around multi-turn weakness)

Tau2-telecom (28.1) and the independent multi-turn reports say long dialogues are where this model drifts. A tutoring session is a long dialogue. So the Conductor keeps a compact, structured *session state* — problem statement, the student's current step, misconceptions already identified, hints already given, hint-ladder level — and re-injects it each turn immediately *after* the frame, in the uncached tail (it changes every turn), while the raw history is capped at three turns of spoken text; the system prompt, older context and the frame stay cached. Once the first solve exists it is stored in the state too, which is what makes later hint turns single-canvas (§7). The same object drives the hint ladder below.

### 2.4 Pedagogy by construction, and read-back for handwriting

The only tutoring evaluation of the Gemma 4 family found (Pedagogical Suitability Index, arXiv 2608.05411) scores Gemma 4 27B lowest-but-one on *scaffolding density* — it tells rather than scaffolds. The two-channel design fixes most of that structurally: the spoken channel is a hint ladder (question → hint → partial step → full step; the student has to ask to climb), and the shown channel reveals nothing until the ladder reaches it. For camera-captured handwriting (no published OCR number for this model at all), the first canvas is a *read-back* ("I read this as: the derivative of x squared over two…") that the student confirms with one word — a no-think canvas costing ~0.4 s, which is also good tutoring practice.

## 3. Tutoring scenarios

Each row: what is on screen or camera; what the student says; what the model has to think through; the benchmark that says it can; the tier; spoken vs shown; the risk and its mitigation. Eval items are drawn one-to-one from these.

**U1 — "Where did I go wrong?" (GCSE/A-level maths, camera on the exercise book).** Student holds up a page of algebra or calculus with a wrong answer. Thinking: solve it independently, align the student's steps, find the *first* wrong step. Grounding: GSM8K 96.3 and HiddenMath 80.6 for the maths, MATH-Vision 70.5 for reading a maths image. Tier A–B. Spoken: "check the sign when you moved the 3x across" (hint level 1); shown: nothing until asked, then the corrected line. Risk: handwriting misread — read-back first (§2.4). Why diffusion: the solve-and-align step is 1–3k tokens; the 12B would take 10–20 s, which turns a tutoring exchange into a wait.

**U2 — "Is my setup right?" (university physics or chemistry problem set on screen).** A PDF problem and the student's typed attempt. Thinking: work the problem to the point where the setup is validated, check units and assumptions. Grounding: GPQA Diamond 73.2 (+8.6 with thinking). Tier B (~8 s). Spoken: yes/no and the one assumption that is off; shown: the corrected setup. Risk: the 12B is 5 points better on GPQA — the demo should show that the 12B needs ~40 s for the same answer. Text-first: the PDF text layer is sent, not OCR.

**U3 — "Does this proof hold?" (induction, limits, linear algebra; screen).** Thinking: check each inference and the base case; find the gap. Grounding: Putnam 67.4 (+10.3), AIME 69.1 (+18.3). Tier B. Spoken: "the inductive step assumes what you're proving in line 3"; shown: the annotated proof. Risk: hard proofs beyond the model — the spoken channel must be allowed to say "I can't verify step 4" rather than bluff (a rubric item).

**U4 — French tutor (the only local one with a French voice).** Student writes a paragraph in French on screen, or speaks French; the assistant corrects, explains each correction in English or French, and can hold a short role-play. Grounding: MMMLU 81.5, MGSM 84.8 (French text quality); Kyutai STT/TTS and Pocket TTS are EN/FR. Tier A. Spoken: the two most important corrections; shown: the marked-up paragraph. Why this one matters for the demo: no cloud voice assistant runs a private French tutor on your desk, and the artifact burst (a fully annotated paragraph appearing at once) is visible on video.

**U5 — Marking against a mark scheme (screen: the student's essay answer and the official scheme).** Thinking: apply the scheme's criteria one by one, cite evidence from the answer, decide marks. Grounding: Tau2 retail/airline (+14/+20 with thinking) as the closest proxy for rule application; IFEval 97.4 for producing the marks table in the required format. Tier A–B. Spoken: the mark and the biggest lost opportunity; shown: the criterion-by-criterion table. Two dense documents: focus crops on each, and this is the scenario most likely to need the PDF-text assist — the toggle is on camera when used.

**U6 — "Explain this diagram" (textbook figure: circuit, pulley, cell, food web; screen or camera).** Thinking: identify components and relations before explaining at the student's level. Grounding: MMMU-Pro 66.0 (thinking on, thought closed), MATH-Vision 70.5; AI2D/ScienceQA are not reported. Tier A–B. Spoken: the one-sentence principle; shown: a labelled list. Risk: the 12B is better at visual reasoning (69.1) — position this as adequate, not a strength; maximum visual tokens.

**U7 — Concept explanation with a worked example, at level.** "Explain eigenvalues like I'm in first year" or "why does entropy increase?" Grounding: MMLU-Pro 77.6 (no thinking gain, so tier A even with thinking on), MMMLU 81.5. Spoken: the analogy; shown: the worked example. This is where the model is *fast* — the whole exchange lands in ~1.5 s — and it is the bread-and-butter of tutoring even if it is not a thinking showcase.

**U8 — Quiz and flashcards from the chapter on screen.** "Make me twenty questions on this section with answers, hardest first." Grounding: IFEval 97.4 (format), MMLU-Pro (content). Tier A; the value is the artifact burst — an interactive A2UI surface (MCQ with feedback, a flippable deck) appearing at once while the voice says "done, start with number one" (`12` §1). Risk: long chapters hit the long-context weakness (MRCR 32.0) — bound the input to the visible section plus the page text.

**U9 — Data literacy: "is this conclusion supported by this graph?"** A chart on screen and a claim about it. Thinking: read axes and trend, check whether the claim follows. Grounding: MMMU-Pro 66.0 (no ChartQA number exists). Tier A–B. Spoken: supported / not, and why; shown: the numbers read off the chart. Risk: chart reading is unmeasured for this model — treat as experimental and include a read-back ("I see the blue line peaking in March at about 40").

**U10 — Revision plan from syllabus + calendar (two windows).** Thinking: constraint satisfaction — topics, weights, available slots, spacing. Grounding: BBEH 47.6 is the honest proxy (weak-ish), IFEval for the table. Tier B. Spoken: "you're short on mechanics; I've put it on Tuesday and Thursday"; shown: the plan. Risk: reasoning quality — keep it to two weeks and a dozen slots.

## 4. Desk knowledge-work scenarios (non-coding)

**W1 — "Does this add up?" (spreadsheet or budget on screen).** Thinking: recompute totals and rates, find the inconsistent cell. Grounding: GSM8K/HiddenMath arithmetic; the *reading* is the risk (OmniDocBench) — pixel-first with a focus crop of the relevant range; the *cells* assist is the checkbox to flip if the read-back is wrong. Tier A–B. Spoken: "row 14 is summed twice"; shown: the recomputed column. Value: data analysis is the lowest-completion category in Microsoft's Copilot data — unmet demand.

**W2 — Expense claim or invoice against the policy (two documents on screen).** Thinking: apply each policy rule to each line. Grounding: Tau2 retail/airline (+14/+20 with thinking), IFEval. Tier A–B. Spoken: "two lines fail: the hotel exceeds the cap and the taxi has no receipt"; shown: the line-by-line check as an A2UI table.

**W3 — Decision memo from visible options (three vendor quotes in tabs, or a P&L and a proposal).** Thinking: weigh options against stated constraints and produce a recommendation with reasons. Grounding: BBEH/Tau2 for the reasoning (moderate), IFEval for the memo structure; "making decisions and solving problems" is 14.9 % of work messages in OpenAI's data. Tier B. Spoken: the recommendation and the main trade-off; shown: the options table and memo.

**W4 — Thread → minutes and action items (email or chat on screen).** Thinking: light — who committed to what, by when. Grounding: IFEval 97.4; documents/reports are 20 % of work conversations (AEI). Tier A. Spoken: three sentences; shown: minutes with owners and dates. Bound the thread length (long-context weakness).

**W5 — Constrained rewrite (the user's draft on screen).** "Make this decline politely, under 120 words, and give me a French version." Grounding: IFEval 97.4 (the model's strongest relative number), MMMLU 81.5. Tier A. Spoken: "done — I softened the second paragraph"; shown: both versions. Writing is 40 % of work messages and two-thirds of it is editing.

**W6 — Research synthesis of two short papers or reports (abstracts and one section each).** Thinking: where do they agree, disagree, and why. Grounding: GPQA 73.2 for scientific comprehension, HLE 11.0 for frontier knowledge; long-context weakness bounds the input to a few pages per source (focus crops per page; PDF-text assist optional). Tier B. Spoken: the disagreement in one sentence; shown: the comparison table.

**W7 — Form filling and structured extraction (a scanned letter or PDF → the form fields).** Thinking: light; the difficulty is reading. Grounding: OmniDocBench 0.319 makes this the hardest pixel-only scenario — it is the honest stress test of visual understanding, and the PDF-text assist is the checkbox that turns it trivial; the structured output is a JSON canvas that converges in 2–3 steps (report §9.3). Tier A. Spoken: "filled, two fields I couldn't find"; shown: the form.

**W8 — Scheduling across calendar and messages.** "Can I fit a 45-minute call with Anna before Thursday?" Thinking: constraint satisfaction over two sources. Grounding: BBEH (moderate). Tier A–B. Spoken: the slot; shown: the conflicts. Pixel-first from the calendar view; focus crop on the week.

**W9 — Compliance or checklist review of a document against a rulebook (procurement rules, style guide, submission requirements).** Thinking: rule-by-rule application. Grounding: Tau2 (+14/+20). Tier B. Spoken: pass/fail and the blocking item; shown: the checklist.

**W10 — Translation QA of a bilingual document (source and target side by side).** Thinking: meaning preservation, terminology consistency. Grounding: MMMLU 81.5, MGSM 84.8; WMT24++ is not reported. Tier A–B. Spoken: the worst mistranslation; shown: a table of flagged segments. Position as experimental.

## 5. What to put in the demo video

Six scenes, chosen for value × thinking × visibility, with the Gemma 4 12B side-by-side on the three tier-B ones:

1. **U2** (physics problem set, tier B): the clearest "give me a moment… here's what's wrong with your setup" moment; the 12B side-by-side shows the same answer arriving 30 s later.
2. **U1** (homework via camera with read-back): the Astra-like physical moment, plus the hint ladder — it refuses to give the answer until asked.
3. **U4** (French tutor): unique to a local stack with French voices; the annotated paragraph appears in one burst.
4. **W2** (expense claim vs policy, tier A–B): rule application, the model's best thinking delta, and the two-channel output.
5. **W3** (decision memo from three quotes, tier B): "making decisions" is the work activity people actually do; the memo lands at once.
6. **U8** (twenty flashcards in one burst): the throughput made visible.

And the honest caption for the whole video: on the tier-A scenes the 12B would have been just as good; the reason to run diffusion is the tier-B scenes, where thinking for several thousand tokens is what makes the answer right and only this model does it inside a conversation.

## 7. The single-canvas regime — where DiffusionGemma alone shines

A canvas commits after ~12 effective denoising steps, and a step costs the same whether 60 or 256 of its slots carry content; on the 5090 that is roughly 0.3–0.5 s per canvas (X1 measures it). So any exchange whose *thought plus answer* fits in one 256-token canvas arrives complete in ~0.4 s — the wall time an AR model needs for its first sentence, and a quarter of the ~1.8 s a 12B needs to finish 250 tokens. Because Pocket TTS consumes whole sentences, a complete reply at 0.4 s is better for speech than a token stream that begins at 0.2 s. Table 4 bounds the regime: with thinking off, IFEval-class replies average 239 tokens, GSM8K 298 and MGSM 291; with thinking on, even GSM8K averages 883. The regime is therefore real but narrow — a brief thought (≤ ~100 tokens, "think in at most three short lines") plus a ≤ 120-token answer — and it is where the constrained decoder has a precise job: a length-bounded thought channel plus a required closing tag turns the single-canvas latency from a distribution into a guarantee (`07`). A second consequence for tutoring: the first turn's solution is stored in the `TutorState`, so every subsequent hint, check or grade is a single-canvas turn.

| | Scenario | Per-turn shape | Why DG alone |
|---|---|---|---|
| S1 | Rapid-fire drills — vocabulary (FR↔EN), mental arithmetic, formulas, dates, anatomy: the assistant asks, the student answers by voice, it grades with a one-line correction or mnemonic | ~60–120 tokens incl. a two-line check | tempo: ~0.4 s per turn; French voices available |
| S2 | Flashcard grading with partial credit ("I said X", the card says Y) | ~100 | small judgement, fits |
| S3 | Hint-ladder turns after the first solve ("next hint", "is this step right?") | ~40–150 | the solve is cached in `TutorState` |
| S4 | One-step homework checks ("is 3/8 more than 0.4?", "does this equation balance?", "is this sentence correct?") | brief thought + short answer | GSM8K-class at ~0.4 s |
| S5 | Multiple-choice practice from the page on screen (it asks, the student answers, it confirms with a one-line rationale) | ~80–120 | MMLU-Pro 77.6 without thinking; image tokens are in the cached prefix |
| S6 | Lecture or meeting companion ("what did she mean by heteroskedasticity?" → three sentences) | ~120, no thought | the complete explanation in 0.4 s while the speaker continues |
| S7 | Unit and sanity checks ("is 70 mph about 110 km/h?") | ~60 with a one-line computation | |
| S8 | Quick reads off a chart or table on screen ("which bar is highest, by how much?") | ~80–100 | read-back included |
| S9 | One-paragraph correction in language learning (corrected paragraph plus two notes) | ~150–250 | the whole artifact at once; IFEval 97.4 |
| S10 | Single-constraint checks ("does this slot fit?", "is this deadline a working day?") | ~80 | |
| S11 | Mnemonics and acrostics for memorisation (a four-line rhyme, an acrostic on a term) | ~60–120 | a plausible diffusion-specific strength — bidirectional planning within a canvas, as LLaDA showed on constrained short text — unproven for DG, so an experiment (X10), but pedagogically useful and visibly different from an AR model |

The boundary is honest: anything that needs a real think (a GPQA item averages 726 tokens *without* thinking) leaves the regime and goes to tier A/B (§1.1), where the talker holds the floor.

## 8. The talker: Gemma 4 E2B (decided as the `talker` profile, 2026-09-12)

**Role.** The talker owns the *interaction*, never the *answer*: an instant grounded acknowledgement ("I can see the integral — give me a moment"), clarifying questions, small talk, hosting a drill, interim updates during a tier-B think, and — optionally — hearing the user. It never asserts content the thinker has not produced, so there is no consistency check; Pocket TTS voices DG's spoken channel unchanged. Gemma 4 E2B (5.1B parameters with per-layer embeddings, ~2B effective; text, image and audio in) is enough for that role and no more — it must not answer GSM8K-class questions, which DG does in ~1.3 s and better.

**Fit.** DG needs ≈ 24.5 GiB (17.5 weights + 2 KV + 2–3 activations/graphs + 2.5 sampler transient) and the desktop/contexts ≈ 1 GiB, leaving ~6 GiB. A second vLLM instance costs 5–6 GiB even at int4; Kyutai STT costs 2.5. So the talker runs as **E2B on llama.cpp at Q4 on the GPU (~3–3.5 GiB all-in)**, Kyutai STT leaves the GPU, and recognition comes either from Nemotron on the CPU or from E2B itself: after end-of-turn (Silero + smart-turn) it emits the transcript and the acknowledgement in one ~60-token call (~0.4 s). The E2B-hears variant trades Kyutai's streaming partial transcript and ~0.15 s of head start for 2.5 GiB, paralinguistic cues and one model fewer — one evening's experiment (X9: transcript WER, ack latency, and DG start time versus the Kyutai path). E4B (~5 GiB) does not fit next to the sampler transient. Two GPU processes time-slice, so E2B's decode will jitter while DG denoises; the talker's outputs stay short enough not to care, and CUDA MPS is the fallback if they do not.

**What it replaces.** The E4B "interim opener" and, for now, the dual-branch `decode_mode` patch (`08`): E2B gives the low-latency spoken opener without touching vLLM's scheduler, at the price of no shared prefix — which does not matter once the talker is forbidden from answering.

## 6. Evaluation additions

Q8 *Tutoring rubric*: 40 items across U1–U10 scored with a PSI-style rubric (knowledge correctness, scaffolding density — does it hint before telling, error localisation — first wrong step found, read-back accuracy for camera items, refusal to bluff on unverifiable steps), by a local judge (Gemma 4 12B) with a 20-item human spot check. Q9 *Desk-work rubric*: 40 items across W1–W10 scored on correctness of the spoken sentence and completeness/format of the artifact, with text-first grounding on and off (the on/off delta is the OmniDocBench mitigation made measurable). Both suites record thinking tokens, time to first spoken word and closing-tag rate per item, and both run against Gemma 4 12B for the side-by-side. X7 gains the closing-tag rate as a required column. Q10 *Single-canvas drill set*: 100 drill turns (S1–S5) measuring per-turn latency (target ≤ 0.6 s p95), single-canvas rate (fraction of turns whose thought + answer fit one canvas, target ≥ 90 % with the brevity instruction) and grading accuracy. X9: E2B-hears versus Kyutai STT (WER, acknowledgement latency, DG start time, VRAM). X10: mnemonic/acrostic quality DG vs 12B (constraint satisfaction and human preference).

## Sources

DiffusionGemma tech report https://arxiv.org/pdf/2608.00146 (Tables 3, 4; §9.3; §10) · model card https://ai.google.dev/gemma/docs/diffusiongemma/model_card · Gemma 4 model card https://ai.google.dev/gemma/docs/core/model_card_4 and report https://arxiv.org/pdf/2607.02770 (Tables 5, 6, 9, 12) · NVIDIA NVFP4 card (independent re-run) https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4 · Google developer guide (700+ tok/s on RTX 5090) https://developers.googleblog.com/en/diffusiongemma-the-developer-guide · 5090 hands-on https://note.com/unco3/n/n2b1edf5b8567 · Pedagogical Suitability Index https://arxiv.org/html/2608.05411 · Anthropic Economic Index (Jun 2026) https://www.anthropic.com/research/economic-index-june-2026-report · OpenAI, How People Use ChatGPT https://www.nber.org/system/files/working_papers/w34255/w34255.pdf · Microsoft, Working with AI https://arxiv.org/pdf/2507.07935 · latency tolerance: Maslych et al. CUI 2025 https://arxiv.org/pdf/2507.22352, Boukaram et al. HAI 2021, Shiwa et al. 2008.
