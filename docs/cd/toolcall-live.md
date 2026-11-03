# Native tool calls under the live prompt, through a real `Session` (T-0093)

**Status: run 2026-10-13 at `4b0bd92`. Verdict: 7 of 8 hard gates hold; the prefix-hit gate fails on one case
(`flight_booking_with_memory`), so the run fails its pre-registered gates.** The human ruled (2026-10-13) that the
verdict stands as recorded and that J2 is aligned with B11's written definition for future runs only (T-0096); nothing
here is re-scored. Every number was re-derived from the raw logs by reviewer B, independently of `build_report`, and
agrees. Source: `.claude/worktrees/t0093-launch2/logs/t0093-live-4b0bd92/` (`report.json`, `report.md`, raw
`turns.jsonl` and `sessions/`), launch log `…/logs/t0093-live-4b0bd92-run.log`.

**Dry run; nothing executed.** The tools-on arm runs `tools = dry_run`: every call goes to `DryRunToolExecutor`, which
describes what it would do and touches nothing.

## Regime

| | |
|---|---|
| commit | `4b0bd92d69b0a8950fed4876e11195646b412f8d` (clean detached worktree; every `turns.jsonl` line carries it) |
| engine | umbrella regime of `docs/tasks/T-0080d.md`: profile `lean`; vLLM `v0.29.0`; DiffusionGemma NVFP4 rev `2ea8372`; chat template `36e3a42e`; tokenizer `cc8d3a0c`; CL 256, 48 steps; thinking on; prefix caching on; `max_num_seqs 3` |
| tools | `native:bbfc53ccdd80:dry_run` (tools on) vs `off`; repair on (a run with repair off is refused before it starts, and void if it happens anyway) |
| cases | 60: the dev split of `tests/eval/toolcall` (40 tool + 8 no-tool; `cases.jsonl` sha256 `460cd1c3…`), the 7 Q12 cases of `tests/eval/surface_placement_cases.jsonl` (J1; sha256 `e152810b…`), and the 5 plain questions of J10 |
| arms | each case once with tools off and once with tools on, alternated per case (case 0 on first), one boot, concurrency 1 |
| n | 1 per case per arm (60 × 2 turns) |
| input | typed (`UserText`); fixture frames (the case's screen as the turn's frame); no audio; fake STT, fake TTS (300 ms per word), fake audio, fake perception |
| seed | none (the diffusion path refuses `seed`); the runs are not reproducible |
| session | one fresh `Session` per (case, arm), so no history carries between cases |
| device, driver, boot | RTX 5090, driver 595.84, gpu-mem 0.86; one fresh engine-only boot via `scripts/locked/t0093_live.sh` (2026-10-13 16:00:31, healthy 16:02:51; device memory after boot 27339 MiB used); cases 16:03:00–16:11:48 |

## Method

**The harness** is `scripts/gpu/t0093_toolcall_live.py`. It runs through `/launch` only, against an engine that is
already booted. Before any socket opens it refuses (exit 2) when:
- the base URL is not an IP literal in 127.0.0.0/8 or `::1`;
- the tree is dirty;
- the out-dir already exists and is not empty;
- either case file is not the pinned one;
- `cli.load_tool_kit` cannot load the kit, the repair is off, or the tool set is not `bbfc53ccdd80`.

All HTTP is to the loopback engine and bypasses any proxy.

**The session** for each (case, arm) is put together the way `cli.build_session` does it, without the HUD server. The
tools-on kit comes from `cli.load_tool_kit` itself, inside the timed startup. The services are:
- the real `VllmClient`;
- `FakeStt` and `FakeTts`;
- `FakeAudioIO` on the session's own clock, with only its playback seams exposed (typed turns need no capture pump);
- `FakePerception`, holding the case's screen;
- a memory with no recollections. The exception is `flight_booking_with_memory`, which gets its three calculus
  captions as hits, the way T-0055 ran it. The memory never holds an uncaptioned episode, so the idle captioner never
  sends a request.

The turn arrives as `UserText`.

**Prefix hit (J2).** The harness wraps the client so that each turn's anticipatory prefill runs to its end:
- the session's end-of-turn `PrefillAbort` is deferred while the prefill is still running, and the deferral is
  recorded per case;
- pass 1's generate waits for the prefill, then reads `/metrics` (`vllm:prefix_cache_{hits,queries}_total`), streams,
  and reads `/metrics` again;
- the per-case hit and query tokens are that delta.

The wait is inside `llm.first_canvas` in both arms. Its distribution is reported beside the latency rows. A case
counts its prefill as done only when it ran to its end on its own; an aborted prefill does not count, even though the
client returns from it normally.

**This departs from a production typed turn (J12, J22).** In production, a typed turn aborts its anticipatory prefill
as soon as the turn starts thinking, about 120 ms after the question arrives. Pass 1 then starts at once, on whatever
part of the prefix the engine had cached by then. Here the prefill always completes and pass 1 waits for it, so that
B11's "the cache warmed by its own prefill" can be measured. Both arms wait the same way, so the first-canvas ratio
stays comparable. The absolute `llm.first_canvas` and `llm.done` numbers are not a production typed turn's.

**Barge-ins (criterion 2; J13, J18).** These are scripted on tools-on tool cases only. The 15 candidates, 3 per tool,
are fixed in the harness:
- **click, type, scroll, key:** cases that spoke a prefix in T-0084e dev (`logs/t0084-native-dev-794161d`, native
  format, pass 1).
- **search:** search-01, search-02 and search-04. In the native format of that run, no `google_search` case spoke a
  prefix: all ten called silently, and all ten spoke in the `<cc>` format. So these three are the first dev searches
  other than search-03 and search-06, which J18 names as excluded.

The harness tries the candidates in run order until 5 have been delivered:
- once the turn is SPEAKING with pass 1 done (or its repair running), the harness submits a `UserSpeechOnset`;
- the machine's own takeover timer does the rest, which is the only source of a takeover in production;
- a barge-in counts as delivered when the turn's line reads `outcome: interrupted` and carries a call (a routed tool
  turn);
- fewer than 5 delivered voids the run.

**Errored and incomplete cases (J14, J19).** These `integrity:` reasons void the run in either arm:
- an `error` outcome, session failures, or a timeout;
- a pass 1 whose prefill had not finished;
- more than one request in flight;
- a case without exactly one turn line, a missing or gapped session log, or a tools-off line that does not read
  `off`.

report.md gives the counts of errored, timed-out and not-idle cases. An engine that was busy before a case is
reported but does not void the run. After 3 consecutive cases in which the engine cannot be reached, the harness
stops, exits 1, and the report is void.

**Startup** is wall time from before the kit is loaded (tools on) or the client is built (tools off) until
`Session.start()` returns.

**Latency (implementation note).** B3 cannot be measured on typed turns. The report gives `llm.first_canvas` and
`llm.done` as span end minus span start, which makes them relative to `llm.start`. Action latency is the
`tool.execute` span start minus pass 1's `llm.done` span end, both on the session clock. It includes the playback of
the spoken prefix through the fake TTS at 300 ms per word.

**The report** is `build_report(run_dir)`, which reads only `turns.jsonl`, `sessions/*.jsonl` and `manifest.json`. The
gpu rows of `tests/eval/test_toolcall_live.py` re-derive it from those raw files and check that the stored
`report.json` agrees. The definitions follow rulings J0–J22 in `docs/tasks/T-0093.md`:
- routing is pass 1's `tool` commit (J8);
- draft valid and exact are taken over the tool cases that were not barged in (J5);
- percentiles are nearest-rank (J9).

### Prefix gate for future runs (H-T96, T-0096)

From T-0096 on, the hard gate "prefix hit with tools on ≥ 0.9 (B11)" is measured as B11 (`docs/03-test-plan.md`) is
written: over the final request's prefix, the part the turn's own anticipatory prefill covered. The run above (`4b0bd92`)
stays under J2 and is not re-scored (H-T93b); its table, numbers and verdict below are unchanged.

**The measurement.** For each (case, arm), with P, Hp the prefill's `/metrics` query and hit deltas and Q, H pass 1's:
- P and Hp run from the last reading of the pre-case idle check (m0) to pass 1's existing pre-generate reading (m1), so
  the timed path gets no extra `/metrics` read. Between the two, only the prefill can reach the engine (ruling M4);
- the manifest records `prefills_before_pass1`;
- after the session is closed, outside every timed span, `/tokenize` at the server root (chat form, the request
  body's `model`, `messages`, `tools`, `chat_template_kwargs` and generation-prompt fields, `add_special_tokens:
  false`) gives the engine's ids for the last prefill's prompt and for pass 1's. They go to `prefix_ids.jsonl`, one
  line per (case, arm). It renders through the same path as chat completions, so S is exact (ruling M2);
- `build_report` derives S, the longest common prefix of the two id lists, and Hs = min(H, S). Hits are contiguous
  from token 0, so the cap is exact.

**The gate.** With A = 32, a tools-on case holds iff S > 0, Hs ≥ ⌊S/32⌋·32 (only the last partial block may miss:
the miss is at most S mod 32, and nothing on a block-aligned S) and Hs/S ≥ 9/10. Every tools-on case must hold; no
tools-on case is a fail. Tools off is measured the same way and reported. A = 32 is fixed from `4b0bd92`, where every
hit was a multiple of 32; the engine's `cache_config_info` `block_size` label is reported, never used (ruling M5). The
whole-request hit / query (J2) stays in the report, labelled "reported, not gated since H-T96".

**Voids** (`integrity:`, either arm; each names the case, the arm and the numbers, and the case reads "unread"):
the prefill ids are not P long; the pass-1 ids are not Q long; `prefills_before_pass1` is not 1; the image digests of
the prefill's and pass 1's `llm.request` records differ over their shared messages; a nonzero H or Hp anywhere in the
run is not a multiple of 32 (the cache granularity is not the registered 32); a sidecar line is missing or does not
parse. The id checks (the two lengths, a missing line) are also fail-fast checkpoints (ruling M6): after each arm of
case 0 (text or tools) and after each arm of the first case, in run order, whose prompt carries an image. If one fails
there, the harness stops, exits 1 and writes a void report. So a `/tokenize` mismatch voids the run at case 0 (text or
tools) or at the first image case (image expansion), a few minutes in.

**Cross-case reuse is capped, not reset.** Hits beyond S come from blocks other requests cached (finding 2). They are
reported per case as `hit_beyond_prefix`, and Hp/P as `prefill_prewarmed`; neither enters the gate. There is no cache
reset between cases (approved with H-T96): `/reset_prefix_cache` needs a dev-mode engine boot, and cold cases would
move absolute latency further from production and from `4b0bd92`. A block inside S that was already warm before the
case's prefill was still covered by that prefill (P ≥ S).

**Never re-scored.** A run without `regime.prefix_rule == "H-T96"` or without `prefix_ids.jsonl`, `4b0bd92`
included, reads "unread: run predates H-T96" and fails this gate; it never falls back to J2.

## Hard gates (approved unchanged at H-T93)

| gate | bar | result |
|---|---|---|
| run valid | every tools-on line reads `native:bbfc53ccdd80:dry_run`; repair never off | holds: 0 void; all 60 tools-on lines read `native:bbfc53ccdd80:dry_run`, all 60 off lines `off`; 0 errored, 0 timed out; no repair-off note |
| nothing but the spoken prefix is voiced | J17 (human), with J4's scope (`TtsSpeak` with text). Every turn: no special literal and no `call:` in the text, nor in the source, each checked alone. Tool turns: the voiced sources, whitespace-normalised, are a prefix of the cleaned `spoken_prefix` of the turn's first `tool.call` record | holds: 0 offending fields |
| no execute on an interrupted turn | 0 `ToolExecute` | holds: 0 (5 barge-ins delivered: search-04, click-01, click-03, click-06, scroll-01) |
| no execute before the turn's own drain | 0 `ToolExecute` (T-0089 R1's THINKING row with nothing voiced has nothing to drain) | holds: 0 of 34 executes (13 on their own drain step; 21 on silent turns at `llm.done`) |
| no stale terminal reached the machine | 0 (J7); the dropped ones are counted | holds: 0 reached, 5 dropped (all `playback.stopped`, 0 ms, after silent turns) |
| prefix hit with tools on (B11) | every tools-on case ≥ 0.9 (J2) | **fails** on 1 of 60: `flight_booking_with_memory` 1408/1599 = 0.881. Next lowest: search-01 0.940 |
| `llm.first_canvas` p50, tools on vs off | ≤ 1.10 × | holds: 500 / 587 ms = 0.852 |
| false tools, no-tool and plain cases | ≤ 1/13 | holds: 0/13 |

## Reported against the eval (not gated)

| | live, tools on | eval (dev) |
|---|---|---|
| native route recall | 39/40 (unrouted: type-05) | 40/40 |
| draft valid | 34/35 (barge-ins excluded) | 40/40 |
| draft exact (`score.exact`) | 32/35 (type-05 unrouted; click-02, click-04 outside the bbox) | 39/40 |
| `query_problem` (routed `google_search`) | 0/10 | – |
| spoken prefix | 18/39 | 27/40 |
| barge-ins delivered; their drafts | 5; valid 5/5, exact 5/5; executed 0 | – |
| repairs fired, outcomes by final status, `tool.repair` p50/p90 | 0 (no invalid routed draft; see findings) | – |
| action latency p50/p90 | n 34: p50 0 ms, p90 2411 ms (see findings) | – |
| Q12 surface rate, tools on vs off (positive cases); negative control on its own line | `ui_valid`: on 2/4, off 3/4; negative control 0/3 and 0/3 (see findings). *Re-derived under T-0097's definition from the raw logs; reported, not gated; verdict unaffected:* surface shown (primary; Q12 / T-0055 `surface_accepted`) on 4/4, off 3/4; every ui commit valid (secondary; `extras.ui_valid`) on 2/4, off 3/4; `ui.gate` on positive cases, gated / emitted / failed / not gated: on 20 / 6 / 14 / 0, off 6 / 6 / 0 / 0; negative control 0/3 and 0/3 under both definitions, 0 gates in either arm (regime: this run's, as in the Regime table; positive n = 4, negative n = 3 per arm) | – |
| `llm.first_canvas`, `llm.done`, prefill wait: on vs off | first_canvas p50/p90 500/689 vs 587/839 ms; done 507/753 vs 665/1496 ms; prefill wait 204.4/443.9 vs 221.2/335.8 ms (n 60 each) | – |
| prefix hit, pooled Σhit/Σquery: on vs off | 91808/93403 = 0.983 vs 45760/48283 = 0.948 | – |
| session startup p50: on vs off (the detokeniser load) | 455.8 ms vs 4.9 ms | – |

## Before the run

- The T-0088 gpu rows of `tests/contract/test_llm_tools_wire_gpu.py` (rows 1–4) run before this one or in the same
  `/launch` window. Row 2, the multi-canvas chunk shape, must pass before any number here is read.
- The human has approved H11 and H-T93 (2026-10-12; `docs/tasks/T-0093.md`).

## Findings (reviewer B's re-derivation, 2026-10-13)

All numbers: lean profile, CL 256, 48 steps, NVFP4 rev `2ea8372` on vLLM 0.29.0, commit `4b0bd92`, typed input, fake
STT/TTS/audio, n = 1 per case per arm, no seed, concurrency 1.

1. **The failing gate is a property of memory-hit turns, not of tools.** In `flight_booking_with_memory` the
   prefill's last user message is empty; the generate's adds the `[screen]` note, the three `[recalled, not on screen]`
   captions and the request (1029 characters, against 233 for `flight_booking`). The prefill prompt is identical to the
   other 26 text-only cases' and its prefix was hit in full (1408 tokens). The miss is 191 tokens, 176 more than
   `flight_booking`'s 15, and the tools-off arm shows the same 176 (207 vs 31). In production the recollections are
   retrieved per utterance (`Session._memory_hits`, k = 3) and sit right of ‖ in the prompt, after the anticipatory
   prefill has started, so a production memory-hit turn should read the same (by construction, not measured). J2 divides by the whole final request;
   B11 (`docs/03-test-plan.md`) divides by the final request's prefix tokens, under which this case would most likely pass (the prefix denominator is not recorded; T-0096; for future runs see
   [Prefix gate for future runs](#prefix-gate-for-future-runs-h-t96-t-0096)). The
   human kept the verdict and moved the definition question to T-0096 (forward only).
2. **search-01's 86-token miss (0.940; off 0.897)** is the first request with each arm's system prompt. Later text-only
   cases hit 1408 tokens (640 off) where it hit 1344 (608), most likely by reusing blocks of the shared `[screen]` text
   cached by earlier cases' generates. So the per-case hit of text-only cases partly measures cross-case reuse
   (inferred from block-aligned counts, not proven per token); search-01 alone would still pass.
3. **Spoken prefix 18/39 vs the eval's 27/40: the whole gap is `type_at`.** By tool, live vs T-0084e dev native pass 1:
   `google_search` 1/10 vs 0/10; `click_at` 8/10 vs 8/10; `type_at` **0/9 vs 9/10**; `scroll` 4/5 vs 5/5; `press_key`
   5/5 vs 5/5. Under the live prompt every `type_at` turn writes its plan inside a closed thought, then the call, and no
   answer text. Nothing was lost or wrongly voiced.
4. **No repair was measured.** (Measured next in T-0099: `docs/cd/toolcall-repair-live.md`.) All 34 routed drafts parsed and were schema-valid. The one unrouted tool case (type-05)
   wrote a well-formed call inside a thought that never closed (`thought_open`), which is correctly neither routed (J8)
   nor voiced, so the repair had nothing to do. The umbrella's "first repair measurements on real invalid drafts"
   remain unmeasured.
5. **Action latency** is bimodal: 21 of 34 executes are silent valid calls run at `llm.done` (0–1 ms, J15); the 13 voiced
   ones wait for their drain under the fake TTS at 300 ms per word (1253–3128 ms). Neither mode is executor cost.
6. **Q12 surface rate depends on the definition.** Two definitions exist:
   - *every ui commit valid*: `extras.ui_valid`, the session's AND, over every gated ui commit of the turn, of "the gate
     emitted a surface". One later canvas that fails the gate makes it false, although the surface already reached the
     panel (the session publishes on success and never retracts). This is what the report counted at `4b0bd92`: on
     2/4, off 3/4.
   - *surface shown*: the turn's own log has at least one `ui.gate` record with `gated` and `surface_emitted` true. This
     is the one that matches docs/03 Q12 ("runs whose `ui` segment the gate turns into a surface") and T-0055's
     `surface_accepted`: on 4/4, off 3/4.

   T-0097 makes "surface shown" the primary row and keeps "every ui commit valid" beside it (re-derived numbers in the
   Q12 row above; reported, not gated; the verdict is unaffected). Still not comparable with T-0055: `ui.gate` records
   no component count and no `root`.

   With tools on, the surface is spread over more canvases. Per case, tools on:
   - `flight_booking`: a surface at canvas 1, then 6 failed gates;
   - `primes_table`: 3 failed gates, a surface at canvas 2, then 5 failed (8 failed in all), and 3 speech refusals of
     UI-plan text.

   The speech gate held in all 4 refusals of the run. Every failed record carries `problems: []`: the validator's reasons
   are in `schema_errors`/`topology`, which `ui.gate` does not log, so why those gates failed is not recorded.

## Follow-ups

- **T-0096:** align J2 with B11's written definition (prefix denominator) for future runs; decide how memory-hit turns
  are counted. Forward only; this run is not re-scored. Done as H-T96: see
  [Prefix gate for future runs](#prefix-gate-for-future-runs-h-t96-t-0096).
- Native prompt tuning for speech: `type_at` speaks no prefix under the live prompt (finding 3). **T-0098** measures
  it with an offline prompt A/B on dev and a pre-registered decision rule: see
  [toolcall-native-prompt-ab.md](toolcall-native-prompt-ab.md). Run at `09d1c50`: no change (H-T98b). Diagnostic D1 suggests
  `UI_INSTRUCTIONS` is the cause (inference: n = 50 `type_at` trials per arm, dev, no seed, one boot); **T-0098c** designs candidates aimed at it.
  **Delta (2026-10-16, H-T104, docs/01 D16):** closed by D16, which supersedes the spoken-prefix target for routine steps, so finding 3 is not a defect for routine steps; what remains is the confirm and ask clauses (T-0106). The results, the hard gates and the safety rows (J17 / `no_literal_voiced`) are unchanged.
- A run that exercises the repair on real invalid drafts (finding 4).
- The Q12 surface metric: `ui_valid` vs "a surface was shown" (finding 6). **T-0097** reports both (surface shown
  primary, every ui commit valid secondary) with the `ui.gate` counts behind them. Still owed, for others: (a) `ui.gate`
  carries no component count, no `root` flag and no `schema_errors`/`topology` summary, so closing T-0055 comparability
  needs a `session.py` change (its own task, high blast radius); (b) docs/03's `ui_valid: false` sentence needs
  amending by its next owner.
