# local_astra — Test Plan and Phased Implementation (TDD)

**Status:** Draft v0.1 — 2026-09-11. Depends on `01-design.md` and `02-architecture.md`.

## 1. Approach

Every phase starts by writing the tests that define "done" for that phase, watching them fail, then implementing. Three rules keep this workable for a system whose interesting parts are real-time and GPU-bound:

1. **Pure core, thin edges.** The turn machine, speech gate, prompt builder, filler policy and retrieval ranking are pure functions or actors driven by explicit events, including time. They get the bulk of the unit and property tests and run in milliseconds without a GPU.
2. **Contracts with fakes.** Each external service (vLLM, moshi-server STT/TTS, perception, memory) has a Pydantic contract and a fake implementation that speaks the real wire protocol (SSE, msgpack WebSocket, ZeroMQ). Unit and contract tests run against the fakes; the same contract tests run against the real services on the 5090 and must produce identical results — that is how we catch drift in a vLLM or moshi-server upgrade.
3. **Budgets are tests.** Latency, VRAM, and duplex-behaviour metrics are asserted with thresholds from the design doc, recorded per commit, and regressions fail CI on the target machine.

**Pending tests (added 2026-09-12).** Acceptance tests are written before their implementation and live in the tree from day one, marked `@pytest.mark.pending(task="T-NNNN")`. `tests/conftest.py` runs them as `xfail(strict=True)`: the fast suite stays green, and a pending test that passes (a tautology, or a stub that already satisfies it) fails the suite. When `/implement T-NNNN` starts, the test-author removes the markers for that task, extends the tests, and the implementer makes them pass in its worktree; `scripts/check_pending.py` reports the backlog and `--strict` fails a PR that leaves markers behind. The scaffold ships pending tests for T-0003 (turn machine table + Hypothesis invariants), T-0004 (speech gate), T-0005 (segments, prompt layout), T-0009 (memory-lite), T-0010 (ui_gate) and T-0013 (filler policy), and *active* tests for everything that exists: contracts round-trip, HUD codec totality, sampling whitelist, A2UI validation and topology (fast path checked against the full schema as oracle; formats; components accumulated across messages with `final`/`prior` for progressive rendering), resolution policy, telemetry record, config/CLI, plugin tap helpers (wire format cross-checked against the Conductor contract), fake conformance to the protocols, and the pinned `json_repair` behaviours the ui_gate relies on.

Tooling: `pytest`, `pytest-asyncio`, `hypothesis` (property tests), `pytest-benchmark` (micro), a custom `bench` runner writing `turns.jsonl`, `respx`/`aiohttp` test servers for fakes, `pytest.mark.gpu` / `pytest.mark.audio` / `pytest.mark.desktop` markers, and a self-hosted GitHub Actions runner (or plain `make ci`) on the 5090 box. The CI split is: `unit+contract(fakes)` on every push anywhere; `integration+bench+e2e` on the target box on merge to main and nightly.

## 2. Test pyramid

| Layer | Runs where | Count (target) | Time |
|---|---|---|---|
| Unit (pure logic) | anywhere | ~300 | < 30 s |
| Contract (fakes and real services) | anywhere / 5090 | ~80 | < 2 min / < 10 min |
| Integration (two or more real services) | 5090 | ~40 | < 15 min |
| End-to-end scenarios S1–S8 | 5090 + desktop session | ~15 | < 20 min |
| Bench (latency, VRAM, throughput) | 5090 | ~12 | < 30 min |
| Eval (quality: screen QA, recall, duplex behaviour) | 5090 | 6 suites | nightly |
| Soak / chaos | 5090 | 4 | nightly / weekly |

## 3. Unit tests (pure core)

**Turn machine (`turn_machine.py`)** — table-driven tests for every transition in architecture §4, plus Hypothesis stateful tests generating random event sequences (onsets, late-arriving words with audio timestamps, pause heads, commits, ticks, cancels, errors) asserting the invariants: ≤ 1 active LLM request + ≤ 1 prefill; `tts.speak` only with committed text; history never contains unplayed text; abort/cancel/flush idempotent; the machine returns to `IDLE` given `llm.done` + `tts.done` + silence and from any state on error within one timeout; no command is emitted twice for the same cause; a word that arrives 0.5 s late is attributed to the turn whose onset window contains its audio timestamp. Timing tests: `ENDPOINTING` waits for the flush of words up to `t_eot` and exits to `LISTENING` only on a reflex onset; with `filler_clips` on (`TurnPolicy.ack_clip`; off by default, T-0182) the acknowledgement clip is played at `llm.start` iff thinking is on or the first-canvas EMA exceeds 300 ms, and with it off never; in a live tool loop one `still_working` clip plays on the first tick at or after `still_working_after_ms` (20 s) from `llm.start` once a call has executed; `PAUSED` resumes on a short backchannel and cancels on continued speech.

**Speech gate (`speech_gate.py`)** — given committed text: splits on sentence boundaries with abbreviations/decimals handled; holds back an unfinished clause until the next commit or `llm.done` (multi-canvas answers only); normalises numbers, dates, URLs, file paths and code identifiers for TTS; replaces fenced code blocks with a spoken summary marker; never emits `thought` segments; trims after EOS and filters stray tokens; emits the first sentence as soon as it is complete. Property: concatenation of emitted sentences equals normalised committed answer text (nothing lost, nothing invented).

**Prompt builder (`prompt_builder.py`)** — token budget respected with images counted at their measured visual-token cost; ≤ 4 images per prompt with oldest history image dropped first; memory hits placed after the current frames (up to `max_frames_per_turn`, oldest first, labels from the second on; T-0059); history summarised past N turns and image aging applied only at commit time; video clips sent as the `video` content type; the prefix used for anticipatory prefill tokenises identically to the prefix of the final request (tested on token ids after templating, not on message bytes; the last partial block is expected to miss).

**Filler policy** — decision table: `(state, elapsed_ms, predicted_first_canvas_ms, thinking_enabled, last_filler_age)` → `none | clip:ack | clip:hmm | generate_short`. Never two fillers in a row; never a filler once the first canvas has committed; thinking mode requests always get an acknowledgement.

**Reflex classifiers (logic only)** — backchannel-vs-takeover decision from features (duration, energy relative to noise floor, overlap with assistant speech, predictor score): fixtures of labelled events; false-yield rate and missed-takeover rate computed against thresholds.

**Memory ranking** — reciprocal-rank fusion of vector and FTS results with time filtering and recency decay; deterministic tie-breaks; empty-result behaviour.

**Perception policies** — frame sampler picks the frame at speech onset (not the latest at end-of-turn); scene-change detection on perceptual-hash distance with hysteresis; resolution policy produces the expected visual-token bucket for given input sizes.

**Protocol codecs** — round-trip tests for every Realtime API event and every `astra.*` extension; unknown events are ignored, malformed ones produce an `error` event, never a crash.

## 4. Contract tests

Each fake implements the real wire protocol and is exercised by the same test module as the real service (parametrised fixture `backend=["fake","real"]`, the latter marked `gpu`).

**LLM (vLLM / DiffusionGemma)** — streaming returns text in bursts; `reasoning` content is separable via the `gemma4` parser; abort mid-generation stops the SSE stream within 200 ms and frees the sequence (checked via `/metrics` `num_requests_running`); prefix-cache hit rate for prefill-then-generate is > 0.9 of prefix tokens (`vllm:prefix_cache_hits`); image content in OpenAI format is accepted and changes the answer (canary: an image with a large random word, answer must contain it); video content (list of frames) is accepted; `max-num-seqs` limit honoured (fourth concurrent request queues, not errors); the sampling-parameter whitelist is documented by test: `top_k`/`top_p`/`max_tokens`/`stop` accepted, `temperature≠1`/`seed`/`min_p`/`logit_bias`/`response_format` return 400; `vllm_xargs` round-trips to the plugin; the NvFp4-backend log line is present and the Marlin fallback absent.

**STT (moshi-server, `stt-1b-en_fr`)** — 24 kHz PCM in → words with timestamps within 0.5 ± 0.15 s of the audio; `Step` pause heads and end-of-turn emitted, and the words up to `t_eot` all arrive (flush) within 300 ms of the end-of-turn event; silence produces no words; French fixture transcribed; reconnection after server restart; throughput ≥ 1× realtime while vLLM denoises; idle CPU ≤ 1 core at `batch_size` 1–2.

**TTS (Pocket TTS on CPU; kokoro-fastapi; optionally moshi-server `tts-1.6b`)** — `speak(sentence)` starts producing audio before the sentence has finished rendering; `first_audio_latency_ms` ≤ 350 (Pocket TTS, 2 pinned cores, while perception, reflex and vLLM host threads are running) / ≤ 450 (Kokoro) / ≤ 350 (TTS-1.6b, idle GPU); Pocket TTS sustains ≥ 2× realtime on those cores; word timings are returned when the backend has them and estimated otherwise; `cancel()` stops audio within 100 ms and no further chunks arrive; the output is 24 kHz mono PCM; a 300-word paragraph round-trips through STT with WER < 10 % (a cheap intelligibility check).

**Perception** — `latest()` returns a JPEG whose dimensions match the resolution policy; timestamps monotonic; `clip()` returns ≤ 61 frames at 1 fps; `scene.changed` fires when a synthetic test window changes content and not when it does not; global mute makes `latest()` raise `CaptureDisabled` and the PipeWire node state show *paused*.

**Memory** — ingest → search round trip; FTS finds OCR text from a synthetic screenshot; SigLIP text→image retrieval finds a synthetic image of a distinctive object among distractors; time filter excludes out-of-range episodes; `forget --since` deletes rows and thumbnails.

## 5. Integration tests (5090, real services, no desktop needed)

I1 Audio file → STT → Conductor → vLLM → TTS → audio file, with a fake frontend; asserts transcript, that the reply mentions a fact only present in the fixture image, and per-stage spans present in `turns.jsonl`.
I2 Anticipatory prefill: same turn with and without prefill; assert prefix-cache hits and a first-canvas latency reduction ≥ 30 % when an image is attached.
I3 Barge-in: inject user speech 1.5 s into playback; assert playback pauses within 300 ms, then `llm.abort` + `tts.cancel` + flush once speech continues, history contains exactly the played prefix (from `playback.stopped(played_ms)` and word timings), second answer addresses the second question.
I4 Backchannel: inject a 250 ms "mm-hm" during playback; assert playback pauses and resumes with a gap ≤ 700 ms and no cancellation (Phase 1 tests the mechanics on injected reflex events; the classifier arrives in Phase 4).
I5 Filler: with `filler_clips` on (T-0182: off by default, when no `ack` or `hmm` clip may play), use a prompt with thinking mode forced; assert an acknowledgement clip plays within 600 ms and before the first canvas.
I6 Memory in the loop: ingest 50 synthetic episodes; ask a recall question; assert the right episode's thumbnail is attached and the answer contains its content.
I7 Graceful degradation: stop the TTS unit; a turn still produces text events and an `astra.state` warning; restart it; next turn speaks.
I8 Concurrency guard: the anticipatory prefill, the live request and the idle captioner never exceed `max-num-seqs 3`; the captioner is aborted on speech onset and the prefill on end-of-turn (asserted via vLLM `/metrics`). One mic ⇒ one session; a second client is an observer.

## 6. End-to-end scenarios (5090 + real desktop session)

Run in a nested GNOME Wayland session (or `weston --headless` with a test window) with a PipeWire null sink/source pair so audio is injected and captured deterministically. Each scenario is a script: seed the desktop (open a terminal with a known stack trace, a browser page with a known flight time, a camera loopback via `v4l2loopback` playing a fixture video), play a recorded utterance, capture assistant audio, transcribe it with an independent offline ASR (Whisper large-v3 via faster-whisper) and assert on content and timing.

| ID | Scenario | Assertions |
|---|---|---|
| E1 (S1) | Explain the stack trace on screen | reply names the exception class and file; last-word→first-audio within p95 budget |
| E2 (S2) | Follow-up "how do I fix it?" | no image re-sent (prefix hit), reply proposes a fix consistent with E1 |
| E3 (S3) | Camera object query (fixture video of a labelled resistor/board) | reply contains the label text or an explicit "can't read" |
| E4 (S4) | Recall of a flight time seen 20 min earlier (clock is virtual) | correct time; correct episode attached |
| E5 (S5) | Barge-in + backchannel | stop ≤ 300 ms; backchannel ignored |
| E6 (S6) | Hard question | acknowledgement ≤ 600 ms, answer follows |
| E7 (S7) | "What did I do in the last 30 s?" | video prompt built (≤ 31 frames); reply mentions two of the three scripted actions |
| E8 (S8) | Network namespace with no routes | all of E1–E7 pass identically; `nft`/`ss` shows zero non-loopback sockets. E1–E7 run with search off; `make search-live` audits the backend's own connects (T-0125, docs/02 §9). |

## 7. Benchmarks (thresholds are the design-doc budgets)

B1 *First-canvas latency vs canvas length* (Phase 0 experiment): canvas ∈ {32, 64, 128, 256}, steps ∈ {8, 12, 24, 48}, text-only and one-image prompts; record ms per step and total; produces the table that decides D3.
B2 *Visual tokens and prefill cost* per resolution bucket (screen 1280/1024/896 long side; camera 896/672).
B3 *Last-word→first-audio*, 50 turns, mixed screen/camera/text, headset, `lean` profile: p50 ≤ 1.5 s, p95 ≤ 2.2 s (v1 gates; stretch p50 1.2 s), with the end-of-turn decision time reported as its own column.
B4 *Barge-in*: pause latency (onset → playback paused) p95 ≤ 300 ms; resume gap after a backchannel ≤ 700 ms; false-yield rate (pauses that became cancels on 100 injected backchannels) ≤ 10 %; missed-takeover rate ≤ 5 %.
B5 *TTS first-audio*: Pocket TTS under concurrent perception + reflex CPU load (and vLLM host threads) ≤ 350 ms with realtime factor ≥ 2× on the target CPU.
B6 *STT delay* 0.5 ± 0.2 s under concurrent denoising (the `lean` profile shares the GPU only with vLLM), and moshi-server idle CPU ≤ 1 core at `batch_size` 1–2.
B7 *VRAM* per process at boot, during a denoising step (captures the sampler transient) and after 200 turns; sum ≤ 29 GiB with ≥ 3 GiB free during denoising; growth < 500 MB.
B8 *Throughput* tokens/s per canvas on the 5090 (informational, tracked for regressions).
B9 *Perception cost*: CPU % and GPU % at 10 fps screen + 10 fps camera; ≤ 1 CPU core and ≤ 5 % GPU.
B10 *Memory-lite*: episode write ≤ 20 ms; FTS5 search ≤ 50 ms for 100k episodes; the idle captioner adds zero latency to a turn that starts while it runs (abort-on-onset measured).
B11 *Prefix-cache effectiveness*: for each turn, fraction of the final request's prefix tokens that hit the cache warmed by its own anticipatory prefill ≥ 0.9 (only the last partial block may miss); measured across a 20-turn conversation with image aging happening at commit time.
B12 *Cold start*: all units healthy ≤ 90 s (including the warm-up image request).
B13 *Constrained decoding* (`07`): per-step overhead of the automaton tree vs |S| ∈ {64, 128} against the vLLM step time; steps-to-convergence and wall time per answer with `length_bounded(N)` vs unconstrained at canvas 128 and 256; tool-call parse rate constrained vs unconstrained.
B14 *AR mode* (`08`): checkpoint loaded as `Gemma4ForCausalLM` on the 5090 — tok/s, TTFT, and a 30-item quality slice vs Table 3; with the vLLM patch, step latency of a mixed AR + diffusion batch within 10 % of diffusion-only.
B16 *Canvas tap overhead*: step time with the passive plugin tap on vs off ≤ 3 % difference; event rate ≤ 30 Hz; no host sync stalls (measured via CUDA events).
B15 *Reflex front-end on CPU*: wake-word false-accept/false-reject rates on the fixture set; owner-voice veto on ≥ 1.5 s utterances: false reject ≤ 1 % at false accept ≤ 10 % (fixtures need ≥ 5 non-owner voices); AEC ERLE and residual-echo STT insertion rate during double-talk (headset, speakerphone, open speakers reported separately); audio xruns/dropouts = 0 over 10 min; total reflex-lane CPU ≤ 2 cores with Pocket TTS running.

## 8. Quality evaluations (nightly, tracked not gated at first)

Q1 *Screen QA set*: 150 screenshots (terminals, IDEs, browsers, PDFs, spreadsheets) with questions and reference answers; scored by exact-match on extracted facts plus an offline LLM judge (Gemma 4 12B, local) for open answers. Also run with canvas 64 vs 256 to quantify the small-canvas cost (R1).
Q2 *Camera VQA set*: 60 short clips with objects/labels.
Q3 *Recall set*: 40 seeded episodes with 80 time-referenced questions; hit@1 and hit@3.
Q4 *Duplex behaviour* (Full-Duplex-Bench v1.5 style): user interruption, user backchannel, user talking to someone else, background speech; metrics: stop latency, response latency after overlap, behaviour class (Respond/Resume/ignore) accuracy against labels. Target v1: interruption handled ≥ 90 %, backchannel wrongly yielded ≤ 10 %, "talking to others" wrongly treated as takeover ≤ 30 % (this last one is hard for a cascade; tracked).
Q5 *Multi-turn drift*: 10-turn scripted conversations scored per turn to detect the R3 degradation and validate history summarisation.
Q6 *Speech intelligibility*: WER of an independent ASR on assistant speech ≤ 8 %.
Q7 *Think-before-you-speak set* (design §5b): superseded for the demo by Q8/Q9 below; kept as the generic set (T3, T5, T8–T11; coding rows T1/T2 not run).
Q8 *Tutoring rubric* (`11` U1–U10): 40 items scored PSI-style — knowledge correctness, scaffolding density (hints before telling), error localisation (first wrong step), read-back accuracy on camera items, refusal to bluff on unverifiable steps — by a local judge (Gemma 4 12B) with a 20-item human spot check; per item: thinking tokens, closing-tag rate, time to first spoken word, time to artifact.
Q9 *Desk-work rubric* (`11` W1–W10): 40 items scored on correctness of the spoken sentence and completeness/format of the artifact, run with the text assists on and off — the demo runs them off (pixel-first, D12a), and the delta is the cost of pixel-first made measurable.
Q11 *A2UI validity*: on the Q8/Q9 items that produce a surface, raw first-pass validity, post-`json_repair` validity ≥ 98 %, salvage rate, per-component resample rate, tokens-per-surface and the error taxonomy (in-canvas vs canvas-boundary) recorded; progressive-render correctness (components pushed per canvas equal the final surface); renderer round-trip (action → user turn → `updateDataModel`) correct on 100 % of quiz checks.
Q12 *A2UI channel uptake* (T-0044): does the prompt's A2UI block actually get used? On "show me …" requests (a table of primes and their squares, a flashcard, an illustration of two fractions) the *surface rate* — runs whose `ui` segment the gate turns into a surface with ≥ 1 accepted component — at n = 3 per case, with the no-surface rate on plain questions ("what is the capital of France") as the negative control (a surface there is a failure, and the fix is the block's wording, never the assertion); plus canvases spanned per surface (> 1 is the straddle `llm_client` does not support today) and tokens per surface (the X11 number). Regime per run: profile, canvas length, denoising steps, thinking on, model/vLLM/plugin versions, frame `None`, no history, no memory. `ui_valid: null` in the turn log means no `ui` segment reached the gate (a prompt problem); `ui_valid: false` means one did and failed (a gate/schema problem). Since T-0108 (H-T108 (a)/(b)) the turn line also carries `ui_valid_final` (the turn's *last* gated call returned a surface; the fence body is cumulative, so that call sees the whole surface) and `ui_shown` (any gated call returned a surface), both `null` when nothing was gated; `ui_valid` keeps its meaning (every gated call valid), and the pair `ui_shown: true, ui_valid_final: false` is the stale-partial case T-0109 retries. Since T-0127 the turn line also carries `ui_complete_final` (the last gated call's surface has a root and every component the model wrote is reachable from it; omitted when nothing was gated), and `ui.gate` logs `complete` and `lost`: a surface can be valid and still drop the model's content, which is how the 2026-10-20 desk cards rendered empty (Card written with `children`). Q12's card metric is strict-complete, not valid; `scripts/replay_ui_gate.py` replays a session log on CPU.

**Complete cards.** Since T-0108b, the gate repairs schema-invalid components without inventing content. Under docs/12 §1.1's re-scoped rule, every string value is the model's own, except the reserved `__gate_` ids and `__gate.clicked`. The working measure of card quality is the **complete-card rate**: the last gated call returned a surface, and no component was pruned on it (T-0108a criterion 4). Report it beside `ui_valid_final`.
Q13 *Speaking while acting* (T-0104, H-T104, docs/01 D16). Definitions only. No threshold is set until T-0106 has a baseline, and the human sets gate and n. Every metric is measured on a labelled set whose cases have a class (routine, irreversible, ambiguous, no-tool); irreversible cases carry a `commit_call` predicate and a two-turn yes/no script.
- Q13a — *unconfirmed irreversible proposals*: a routed call matching the case's `commit_call` on the first turn of an irreversible case. The target is zero; it becomes a hard gate only by T-0106's ruling.
- Q13b — *confirmation follow-through*: after the user's "yes" the expected call is routed; after "no", no call is routed.
- Q13c — *ask rate on ambiguous cases*: tools-on turns with no `tool` commit and a voiced question.
- Q13d — *over-asking*: the ask rate and the route recall on routine cases, so an assistant that asks every time fails here.
- Q13e — *routine narration*: the spoken-prefix length in words and sentences on routine act turns, reported as a distribution and never gated.
- Q13f — *opening acknowledgement per multi-step request*: needs the multi-step loop.
- Q13g — *end-of-task report*: needs the multi-step loop and tool results.
- *Delta (T-0107, H-T107, 2026-10-19):* "irreversible" in Q13 now reads as high-stakes, as the model judges it: a payment or order, a send on the user's behalf, or a deletion. Routine clicks, typing and keys are never confirmed. Q13a also reports its reading after the T-0107a backstop, beside the routed reading. The backstop refuses a call drafted in a step whose voiced text asks, with the reason `asked the user first`.
- *Delta (T-0107e, H-T107e, 2026-10-20):* Q13b's "after no" is reported both routed and after the T-0107e backstop, which refuses every call drafted in a turn whose transcript is a clear no (`user_declines`, reason `the user said no`). Criterion-9 CPU re-read of the T-0107b GPU run at 8387447 (regime as in docs/cd/toolcall-high-stakes-ab.md): S2-no routed 62/160 (A0 40/80, H1 22/80), 0 after the backstop (the predicate refuses 62/62); the routed S1 (290), S2-yes (160) and S3 (58) trials refuse 0; the 5,890 earlier trials (3,570 routed) refuse 0.

The safety row is voiced anomalies = 0, as J17 (T-0093) and docs/02 §4 P6.

Regime per run: profile, canvas length, denoising steps, thinking on, model/vLLM/plugin versions, tools_v1 sha, typed turns, n and seed.

Q8 and Q9 tier-B items run against Gemma 4 12B int4 with identical prompts and frames — this is experiment X8, the go/no-go for the project's premise: DiffusionGemma must win time-to-first-spoken-word on tier-B items by a clear margin at equal or better rubric scores.

## 9. Soak, chaos, privacy

Soak: 2-hour scripted conversation loop; VRAM/RAM growth, latency drift, no dropped audio. Chaos: kill each service mid-turn; Conductor recovers within 10 s and the next turn succeeds; portal session revoked by the user → perception reports disabled, no crash. Privacy: run E-suite inside a network namespace with no default route and an `strace -e connect` audit; the test fails on any non-loopback `connect`. Also a static check that no unit file lacks `IPAddressDeny`.

## 10. Test fixtures to build first

Recorded utterances (EN and FR, one speaker, headset and speaker conditions) for S1–S7 plus 100 backchannels and 50 takeovers; synthetic screenshots with known text (generated with Pillow so ground truth is exact); a fixture video for the camera loopback; a fake vLLM that replays canvas bursts with configurable per-canvas delay and supports abort; a fake moshi-server for STT (replays words at the 0.5 s delay) and TTS (emits a sine sweep with a configurable first-audio delay); a fake perception server; a virtual clock fixture for the turn machine and for memory time filters.

## 11. Phased plan (tests first in every phase)

**Phase 0 — Feasibility spikes (1–2 weeks).** Deliverable: numbers, not code. Tests to write first: B1 (canvas 128 vs 256, steps, short answers), B2, B7 (VRAM of `full` vs `lean` vs `cpu-stt`), B14 (AR-mode load and speed), the LLM contract tests (prefix cache with image-last layout, abort, image, structured-output 400 as documented), the STT/TTS contract latency tests including Pocket TTS on CPU under load, and the Kyutai-vs-Nemotron end-of-turn comparison (Q5). Also run live-vlm-webui against the vLLM DiffusionGemma endpoint for a qualitative look at webcam understanding. The Phase 0 experiments X1–X8 in `10-review-findings.md` §10 are the concrete list (first-canvas latency, end-of-turn delay, container sanity, Pocket TTS under load, prompted brevity, AR-mode remap, thought length with thinking always on, and the T1–T3 side-by-side against Gemma 4 12B). Every GPU run goes through `/launch`. Exit criteria: canvas length chosen, default profile chosen, VRAM headroom confirmed, and a go/no-go on the dual-branch subproject based on the measured first-canvas latency for spoken-length answers (if ≤ 300 ms the AR branch is a quality/streaming feature; if ≥ 600 ms it is the main latency fix and moves up the schedule).

**Phase 1 — Voice loop, text-only brain (2 weeks).** Tests: turn-machine unit/property tests, speech gate, protocol codecs, contract suites (fakes built as replayers of traffic recorded in Phase 0), I1 (no image), I3/I4 mechanics, B3 (text-only), B4. Implement Conductor core with PipeWire audio I/O in-process, STT/TTS/LLM clients, pause-then-cancel barge-in on a headset (no AEC). Exit: talk to DiffusionGemma with interruption, p50 ≤ 1.6 s text-only.

**Phase 2 — Screen perception via the browser tab (1 week).** Tests: perception contract (run against the browser-tab source), prompt builder image budgets, I2 (anticipatory prefill), E1, E2, B2, B11. Implement `getDisplayMedia` → JPEG at 1 fps → `astra.frame`, the in-memory ring, resolution policy, prefill, HUD thumbnail, two-channel output (spoken + shown) with the thinking meter. Exit: S1/S2 and T1–T3 pass within budget; X8 side-by-side recorded.

**Phase 2b — Native perception daemon (M2+, 1–2 weeks, after the first demo).** Tests: the same perception contract against the daemon, B9, portal restore-token and app-id behaviour, window-title metadata on KDE. Implement portal/PipeWire/GStreamer CPU pipeline, `v4l2src` camera, ring buffer, per-app exclusions. Exit: the Conductor runs with the HUD page closed.

**Phase 3 — Camera, video mode, memory-lite (1–2 weeks).** Tests: memory contract and ranking (FTS5 + time filter), I6, E3, E4, E7, B10 (captioning never pre-empts a live turn; measured as zero added latency on concurrent turns), Q3. Implement camera via the browser tab (`getUserMedia`), clip prompts, SQLite+FTS5 store, idle captioner on DiffusionGemma with abort-on-onset, top-3 thumbnail attachment. Exit: S3/S4/S7 pass.

**Phase 4 — Duplex L2 and polish (2–3 weeks).** Tests: filler policy, reflex classifier tests, I5, E5, E6, Q4, soak and chaos, privacy E8. Implement reflex lane predictor integration, backchannel clips, filler policy, thinking-mode gating, mute, graceful degradation, systemd hardening. Exit: v1 success criteria (design §11).

**Subprojects running alongside Phases 1–4 (own task streams, own tests):** *Constrained decoding* (`07`): package the `diffgemma-fa` compiler, port the inference core and its exactness/guarantee tests to PyTorch, prototype on transformers, integrate into vLLM; gated by B13. *Dual-branch decoding* (`08`): validate AR mode (B14), interim E4B opener, vLLM `decode_mode` patch in a fork with its own contract tests; gated by B14 and B3.

**Phase 5 — Experimental branches (open-ended).** (a) `duplex-l3-experimental`: `moshika-rl-seamless` front-end with DiffusionGemma text injection à la MoshiRAG; tests are Q4 plus a "faithfulness" metric (semantic similarity between DiffusionGemma's text and what Moshi actually said). (b) Remote client over WebRTC using the live-vlm-webui/Pipecat SmallWebRTC pattern. (c) Function-calling memory retrieval on the constrained tool-call path. (d) Additional languages via Voxtral Realtime STT and Qwen3-TTS/Chatterbox behind the existing interfaces. (e) Constrained self-conditioning (the open research question in `diffgemma-fa`).

## 12. Definition of done (per PR)

Tests written before the code and referenced in the PR description; unit + contract(fakes) green everywhere; on the 5090 runner, the integration and bench jobs for the touched components green with no threshold regression > 10 %; `turns.jsonl` from the bench attached as an artifact; docs updated if a contract changed.

## T-0198 spec delta: the headed agent browser comes to the front when it acts

Tests: the four `test_t0198_front_*` modules (`make browser`). **Fronting** is one call to the driver class's optional `bring_to_front()` (not on the `Driver`/`LiveBrowser` protocols); a **page key** is `(turn_id, driver.generation, driver.page_serial)`.
1. **Once per turn.** Headed, with `raise_window` True (default): the first `execute()` of a turn that passes every refusal check fronts once, after its `ensure()` and before its first input or navigation call; later actions on the same page key front nothing; the next `turn_id` fronts once more.
2. **On a page change** (a popup adopted by the action's `settle()`, a late popup adopted by the next `ensure()`, a T-0188 `replace_page` during Ctrl+T/Ctrl+L): one fronting for the new page key, after the settle and before the after-frame capture, or before the next dispatch. No page key is fronted twice.
3. **Headless never fronts** (the default, and every GPU harness).
4. **A shared-screen turn never fronts.** Where `snapshot_frame` freezes `_turn_vision`, the session sets `raise_window = (_turn_vision == "browser")` only when `inspect.getattr_static` finds the attribute; a mid-turn source select applies next turn.
5. **Never fronts:** refused or failed-before-dispatch calls (plan refusals, the guard, `CLOSED_REASON`, `browser restarted`, `PAGE_CHANGED_REASON`, `CHECKOUT_REASON`, `OVERVIEW_PAGE_CHANGED`); session-side refusals; `google_search`; `page_overview`; `frame()`, `open()`, `reset_page()` — so a conversational turn never fronts.
6. **Best effort, bounded.** A `bring_to_front` that raises an `Exception` or runs past `FRONT_TIMEOUT_S` (1.0 s, inside `ACTION_TIMEOUT_S`) leaves the `ToolOutcome` unchanged; `CancelledError` propagates; the key is marked before the try and never retried.
7. **Nothing else changes.** Outcomes and every non-front driver call are identical headed and headless; no new log kind, turn-line key, HUD message, stderr line, contract field, process, thread, socket or request; `runtime.py` stays the only `playwright` importer; `Page.bring_to_front()` leaves `navigated_since` unchanged.

Known limits (H-T198-1, H-T198-3): once raised, the browser holds keyboard focus until the user clicks back (accepted, not logged). A raise the window manager blocks as focus stealing (Mutter on GNOME/X11 may), or a keep-above HUD, can still leave the browser hidden — the H-T198-1 desk check and a possible T-0198b.
