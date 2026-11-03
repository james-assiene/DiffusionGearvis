# local_astra — documentation index

An on-device voice-and-vision assistant inspired by research demos like Google's Project Astra, fully offline, for a Debian/Ubuntu desktop with one RTX 5090, with `diffusiongemma-26B-A4B-it` as the brain.

| Doc | What it answers |
|---|---|
| `01-design.md` | What we are building and why: goals, scenarios, the ten key decisions, duplex levels (L0–L3), latency and VRAM budgets, risks, v1 success criteria |
| `02-architecture.md` | How it is built: components, one-turn data flow, the turn state machine, contracts, process/port model, profiles, observability, repo layout, ADR index, Phase 0 open questions |
| `03-test-plan.md` | How we know it works: test pyramid, unit/contract/integration/e2e/bench/eval suites with thresholds, fixtures, phased plan with tests-first ordering |
| `04-tool-landscape.md` | What exists in the ecosystem as of 2026-09-11, what we picked, what we rejected, and every source URL |
| `05-agentic-coding-review.md` | Literature review of multi-agent AI coding practice (Anthropic, Ng, practitioners, METR) and the principles the repo setup follows |
| `06-second-pass-risks.md` | The "what did we miss" pass: CPU alternatives, server-global diffusion params, in-app AEC, history hygiene, Blackwell pins, KDE, wake word/owner check |
| `07-constrained-decoding.md` | Audit of `diffgemma-fa`, the literature, serving support, and the plan to bring exact automaton-constrained decoding into the vLLM path |
| `08-dual-branch-decoding.md` | AR opener + diffusion thinker from one prefill: the facts, the cost, the vLLM obstacle, the plan |
| `09-agent-workflow.md` | How code is written here with Claude Code: roles, hooks, skills, gates, PR flow |
| `10-review-findings.md` | Results of the nine-reviewer verification swarm: what was wrong, what changed, what is left to decide, the Phase 0 experiments |
| `12-generative-ui-and-visualization.md` | Generative UI via A2UI v0.9.1 + an `edu` catalog (framework survey and decision), the denoising view fed by the plugin's per-step tap, and the pixel-first perception policy with optional text assists |
| `11-tutoring-and-deskwork-scenarios.md` | Tutoring and desk-work scenarios grounded in the model's published numbers, latency tiers on the 5090, four design rules (close the thought, text-first grounding, tutoring state, read-back), the six demo scenes, the single-canvas regime where DG alone shines, and the Gemma 4 E2B talker profile |

## The design in eleven lines

1. Brain: DiffusionGemma 26B-A4B (Apache 2.0, image+video in, text out, no audio) on vLLM v0.29.0 (cu130), NVFP4 (17.5 GiB), `TRITON_ATTN`, `max-num-seqs 3`, `--kv-cache-memory-bytes 2GiB`.
2. Speech is a cascade: Kyutai STT 1b on the GPU → DiffusionGemma → Kyutai Pocket TTS on the CPU (sentence-in; zero VRAM); Kokoro is the alternative, TTS-1.6b only if STT moves to the CPU. The Conductor owns the audio devices; the HUD is display-only.
3. Speak only committed canvases (uniform-state diffusion revises drafts); a spoken answer fits in one canvas, so first-canvas latency is answer latency; canvas length is engine-global, so latency comes from prefill caching, an acknowledgement clip, the AR opener branch, and (hypothesis) constraint-driven termination — Phase 0 measures which matter.
4. Anticipatory prefill on speech onset: image + history hit the prefix cache before the user finishes talking.
5. Perception is sequenced: v1 from the HUD page (`getDisplayMedia`/`getUserMedia` → JPEG at 1 fps over the WebSocket); the native portal → PipeWire → GStreamer daemon (CPU at ≤ 2 fps, restore tokens, window metadata, V4L2 camera) replaces it in M2+ behind the same interface.
6. Memory-lite: SQLite + FTS5, thumbnails on scene change, captions written by DiffusionGemma when idle, top-3 frames re-attached on recall; embeddings only if that fails.
7. Duplex: L1 pause-then-cancel barge-in and L2 fillers/backchannels via a CPU reflex lane (hotkey or self-trained wake word with a session window → VAD → STT; owner-voice *veto* at end-of-turn; AEC in the Conductor or PipeWire, none on a headset); L3 native overlap is an experimental Moshi-graft branch only.
8. The Conductor's design is derived from Unmute's handler (MIT), no code copied (Realtime event protocol, pause-probability end-of-turn, vLLM streaming); it grows a pure turn machine, DiffusionGemma commit semantics, in-process audio, and two output channels; thinking is always on; Pipecat is the escape hatch.
9. Output is two channels: spoken text plus an interactive A2UI surface (quizzes, decks, tables, maths) validated before rendering; the HUD also shows the real denoising trajectory from the plugin tap. Perception is pixel-first; text assists are checkboxes, off by default.
10. Everything is loopback-only and systemd-enforced; privacy is a test, not a promise.
11. TDD: pure turn machine + protocol-faithful fakes + budgets asserted on the 5090 runner.

## Immediate next steps

The repository is scaffolded (2026-09-12): package layout per `02` §10, decided contracts as Pydantic models, the A2UI validator, fakes, and acceptance tests that are *pending* until their task runs (`make check` is green: 277 active tests, 67 pending). What comes next, in order:

1. **Pure core through the loop, no GPU needed** — `/implement T-0003` (turn machine; its pending table and Hypothesis invariants are the spec), then T-0004 (speech gate), T-0013 (filler policy), T-0005 (LLM client + prompt layout against the fake).
2. **Phase 0 on the 5090** (see `03` §11, `10` §10): stand up vLLM with the NVFP4 checkpoint via `services/vllm.sh` and `moshi-server` via `services/stt.sh`; run X1–X13 through `/launch`; answer Q1–Q6 in `02` §12. The answers fix the default profile, the canvas strategy and the sampler field names the plugin tap needs (T-0011).
3. **Record traffic** during Phase 0 and turn the fakes into replayers (T-0001); then T-0006/T-0007 (audio, Unmute fork), T-0008 (HUD frames), T-0009 (memory-lite), T-0010 (generative UI), T-0012 (assists).
