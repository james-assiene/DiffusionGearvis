# The live tool-call repair on real invalid drafts and on forced valid drafts (T-0099)

**Status: method and pre-registration only; not run.** Written before the run (criterion 11). Results and the regime
table are appended after the `/launch`, every number with its regime. Task: `docs/tasks/T-0099.md` (criteria 1-11,
H-T99a/b/c, rulings Q1-Q7). Origin: finding 4 of `docs/cd/toolcall-live.md`.

**Dry run; nothing executed.** Every session runs `tools = dry_run`: every call goes to `DryRunToolExecutor`.

## Why

The H-T84-2 repair (one seeded constrained native T4 pass 2 on an invalid native draft that names a v1 tool; T-0091
`live.repair_request`/`repair_verdict`, wired by T-0092c, span `tool.repair`) has never fired on a real draft. Every
recorded native pass-1 draft re-judges valid or refused (0 invalid of 170 routed; rule-of-three 95 % upper bound about
1.8 % per draft). This run draws invalid drafts out with honest out-of-schema requests (arm S) and forces the repair on
valid drafts (arm F) to see the repair's floor on the live path.

## Method

**The harness** is `scripts/gpu/t0099_repair_live.py`, run once through `/launch` by `scripts/locked/t0099_repair_live.sh`
(one fresh engine-only boot in the regime of `scripts/locked/t0093_live.sh`: `services/vllm.sh` unchanged since
`806da12`, prefix caching on, plugin wheel rebuilt; then the harness; then the gpu rows of
`tests/eval/test_t0099_repair_live.py` with `T0099_RUN_DIR`; then the engine is stopped). Before any socket opens the
harness refuses (exit 2) when the base URL is not a loopback IP literal, the tree is dirty, the out-dir exists and is
not empty, a T-0093 case file is not the pinned one, the tool kit cannot load, its repair is off or its tool set is not
`bbfc53ccdd80`, or the stress file is not the pinned one (sha256 `dea52a37…`, H-T99a). All HTTP is to the loopback
engine and bypasses any proxy. No request carries `seed`.

**The session** is built as T-0093 builds it, one fresh `Session` per (case, repetition, arm) with one typed turn
(`UserText`): the real `VllmClient` behind T-0093's `_MeteredLlm`; fake STT, fake TTS at 300 ms per word, fake audio
and fake perception holding the case's screen as the turn's frame; no recollections; tools on
(`native:bbfc53ccdd80:dry_run`), repair on, repair deadline 3000 ms. Concurrency 1, one boot.

**Arm S (stress).** The 30 cases of `tests/eval/toolcall_stress/cases.jsonl` (H-T99a), dev-split screens only, in six
categories: K-in 6 (an in-enum key in another spelling), K-out 6 (a key outside the enum; no ground truth), S-extra 4
(a scroll with an amount or target), T-hard 6 (`type_at` text with quotes, a backslash, braces, a newline, accents or
emoji, and one text of 400 characters or more), C-pixel 4 (a click at a pixel coordinate above 999 on a 1280×800
screen), U-under 4 (type into a field, no Enter instruction). File order, k = 3, repetition-major: 90 turns. The
drafts are the model's own on a deliberately shifted input distribution: **the arm-S yield is not a production
invalid-rate estimate.**

**Arm F (forced).** The 40 dev tool cases of `tests/eval/toolcall/cases.jsonl`, file order, once each: 40 turns. For
each arm-F session, and only there, the harness replaces `local_astra.toolcall.live.classify` with a wrapper that turns
a `valid` verdict into `repairable` with the reason `forced: valid draft (T-0099 arm F)` and returns every other
verdict unchanged; the previous object is restored in a `finally` (H-T99c). The repair's own verdict
(`repair_verdict`) is never forced. Arm F measures the seeded path's floor on valid drafts, not repair on invalid
drafts.

**Barge-in during a repair (criterion 4).** On arm-F cases that spoke a prefix at `4b0bd92` (search-04, click-01/02/03/
05/06/08/09/10, scroll-01/03/04/05, key-01..05), in run order, one `UserSpeechOnset` is submitted once the turn is
SPEAKING and the live generate is the turn's repair request, on **at most 5** candidates (ruling Q8), so at most 5 of the
40 arm-F turns are interrupted. The path is production's: the onset pauses the turn and the machine's own takeover
timer (`takeover_ms` = 600 ms) is the only source of the takeover and of the repair's `LlmAbort`; no synthetic
`UserTakeover` is sent. Three counts are reported separately: **attempted** (an onset was submitted); **delivered** (the
line reads `outcome == "interrupted"`, it carries a call, and the session log has the `LlmAbort` of the repair
request); **repair done before the takeover** (the repair's `llm.done` with its verdict was accepted by a step whose
`state_before` is `PAUSED`, i.e. the repair finished inside the takeover window). The last is itself a finding: the
forced repair's offline floor (p50/p90 438.6/577.4 ms, T-0084f) is shorter than the 600 ms window. Fewer than 3
delivered makes the row unread; it does not void the run.

**The report** (`build_report`) reads only `turns.jsonl`, `sessions/*.jsonl` and `manifest.json`. An *invalid draft*
is pass 1's kept `tool` commit judged `repairable` by the **unpatched** `live.classify`, re-run over the logged commit
text, never read from a `tool.call` record. A repair's result is its first `tool` commit judged by
`live.repair_verdict`. The final status of a call is its last `tool.call` record's status (J6; `repair_failed` shows
as `refused`, `repair_timeout` as `cancelled` with that reason). Percentiles are nearest-rank (J9) with n.

## Pre-registration

### Hard gates (H-T99b), per arm; no number is read unless every one holds

- **G1 run valid:** every line `native:bbfc53ccdd80:dry_run`; no repair-off note; 0 `error` outcomes, 0 timeouts, 0
  session failures; exactly one turn line per session; the forced marker only in `forced: true` sessions; and
  (criterion 8) no arm-S repair on a draft the unpatched judge calls valid (`integrity: repair on a valid draft outside
  arm F`), no arm-S record carrying the forced marker. Any of these voids the run.
- **G2:** nothing but the spoken prefix is voiced (T-0093 J4/J17), and no `TtsSpeak` comes from a repair request's
  commits (R9).
- **G3:** 0 `ToolExecute` on interrupted turns. **G4:** 0 `ToolExecute` before the turn's own drain. **G5:** 0 stale
  terminals reached the machine.
- **G6:** at most one `tool_repair` request per turn.
- **G7:** no `ToolExecute` for a call whose final status is `refused`, `repair_failed`, `repair_timeout` or
  `cancelled`; every executed call with `source == "repair"` is schema-valid under `tools_v1`, re-judged from the log.
- **G8:** every repair ends with a verdict less than 3000 ms after it started, or as `repair_timeout` with its abort
  logged (or aborted by a barge-in); none open at session close.

### Reported, not gated

Arm S: (a) yield = invalid drafts / routed tool turns whose draft is not refused, by category and by parse reason;
refused (nameless) drafts are counted only on their own and are not in the denominator (ruling Q5); unrouted turns
and turns with no call are counted separately; (b) repairs fired; (c) outcomes by final status; (d)
`tool.repair` p50/p90 with n; (e) rescue = repaired to valid / repairs fired, K-out excluded from the numerator, Wilson
95 %; (f) rescue-exact = repaired valid and exact (`score.exact`; U-under: inside the bbox and the text equal,
`press_enter` reported, not scored) / repairs fired on gt-bearing categories; (g) coercion = K-out turns whose repair
produced a valid call, with the combo chosen (never a rescue); (h) action latency of repaired calls
(`tool.execute` start minus pass 1's `llm.done` end), p50/p90.

Arm F: repairs fired / routed; agreement (repaired call equal to the draft); exact against the ground truth;
`tool.repair` p50/p90 with n, labelled "floor on valid drafts, not repair latency on invalid drafts"; `repair_timeout`
count; criterion 4's barge-in row.

### Reading rules (H-T99b)

- Rescue and rescue-exact are read only with **at least 10 repairs fired on gt-bearing arm-S turns**. Otherwise the
  row reads "insufficient yield (n = …)", with the yield and its upper bound, and no conclusion on rescue; finding 4
  then closes as "native drafts stayed valid even on out-of-schema requests (yield k/n, upper bound …); the repair
  stays an unmeasured safety net".
- Arm-F numbers never enter an arm-S figure. A forced repair is never counted as an invalid draft, a repair on an
  invalid draft, or a rescue.
- If coercion is 1 or more, the human decides whether out-of-enum drafts should be refused instead of repaired (a
  policy change to H-T84-2, its own task).
- Any `repair_timeout`, or an arm-S `tool.repair` p90 above 1500 ms, triggers a follow-up task on the deadline or the
  budget.
- A T-hard text of 400 characters or more may not fit the repair (`repair_request` cuts the seed at 256 ids;
  `REPAIR_OPTIONS.max_tokens` 256): it is expected to end refused (`repair_failed`), never as a truncated execute (G7),
  and is reported as a repair limit.
- No threshold applies to rescue.

## Regime (planned)

The T-0093 umbrella regime as recorded at `4b0bd92`: profile `lean`, engine only, fresh boot; vLLM `v0.29.0`; NVFP4 rev
`2ea8372`; chat template `36e3a42e`; tokenizer `cc8d3a0c`; CL 256, 48 steps; thinking on; prefix caching on;
`max_num_seqs 3`; RTX 5090, driver 595.84, gpu-mem 0.86. `native:bbfc53ccdd80:dry_run`, repair on (calibration
`4084026c…`, payloads `native_T4.json`), deadline 3000 ms; typed turns, fake STT/audio/perception, fake TTS 300 ms per
word, fixture frames; concurrency 1; no seed. Arm S 30 × k = 3 = 90 turns; arm F 40 turns. The live system prompt's
sha256 is recorded by the run (from the first session's `llm.request`), since the run follows T-0098b.

## Results

Not run yet.

## Cold first use (T-0101)

Written before the confirmation run (criterion 11); results and the regime table are appended after it.

### Evidence (planner, 2026-10-15, read-only, over the a794a98 run dir)

- **Every slow repair was the first on its automaton.** Pairing the `tool_repair` `llm.request`/`llm.done` records by
  request id and sorting by wall: press_key's first repair took 2545 ms (S 80, bucket 128), scroll's first 2031 ms
  (S 39, bucket 64), click_at's first 1892 ms (S 29, bucket 32), type_at's first 1469 ms (S 66, bucket 128); the
  second press_key and click_at repairs took 341 and 268 ms, and all 51 others 213–341 ms. google_search's first
  repair took 231 ms: it shares np2 = 64 and S mod 16 ≠ 0 with scroll, warmed four minutes earlier.
- **The compile key.** The plugin's Triton `log_matmul` kernel is built once per process and specialised per
  (`COMP`, `BLOCK_N`, `BLOCK_K`) plus Triton's integer specialisations (value 1, divisibility by 16); the products run
  on the live `[·, S, S]` block, so the variant depends on the automaton's state count S. The five payloads fall into
  four variant groups. The very first CD request of a boot also carries the automaton-independent compiles (the H(q)
  entropy kernel over `[L, V]`).
- **Nothing persists the compile across boots.** `services/vllm.sh` runs the engine with `docker run --rm` and mounts
  no Triton cache, so every boot compiles again.
- **Other per-automaton caches are cheap** (decoded automata LRU 64, device tables LRU 16, the stop tensor); a request
  through the real path fills them and the Triton variants without guessing Triton's key.

Under the 3000 ms deadline the first press_key repair spent 85 % of it compiling; that is the H-T99b trigger (arm-S
`tool.repair` p90 2034 ms > 1500 ms at a794a98).

### Method (H-T101a)

- `local_astra/toolcall/warmup.py`: `warm_repair(llm, kit)` sends one repair request per distinct payload
  `automaton_sha256`, sequentially, in tool-name order, through the caller's real client and CD path. Tools sharing a
  bucket or a kernel variant are **not** merged (one request per automaton); two tools sharing a sha are warmed once
  and both carry the outcome. Each request's `cd` is `live.repair_request(name, ids, 50, tag, payloads, calibration)`
  (only `draft` and `record_tag` differ from a turn's), options `live.REPAIR_OPTIONS`, the prompt `live.repair_prompt`
  of one fixed minimal pass-1 prompt (one user line, the session's `tools` field, no screen, memory or user data); the
  draft is the engine's ids of a fixed valid native call for the tool.
- Out of sight: `xargs` `{"astra_tap": "off"}` (no denoising view), request ids `warmup-…`, record tags `warmup:…`
  (the engine's `ASTRA_CD` records carry that tag; anything counting records filters by it). Commits are counted and
  discarded; nothing reaches TTS, the HUD, history, memory or a session log.
- Never gating: a failed, 4xx/5xx or overrunning request (deadline = the remaining 30 s total budget, aborted) is
  recorded and the next one is still tried; once the budget is spent the rest are `skipped`. The kit and `repair_off`
  are never touched. The output's `live.repair_verdict` status is reported only.
- Placement: `cli.run_session` awaits it **before** `session.start()` (which starts the STT pump), once per conductor
  process, only with tools on, repair on and payloads loaded, and prints one line
  `astra session: repair warm-up: <tool> <ms> ms <status>, …, total <ms> ms`. An engine restarted under a running
  session is cold again (out of scope).
- Unchanged: the 3000 ms deadline (`tool_repair_timeout_ms`), `REPAIR_OPTIONS.max_tokens` 256, every threshold, the
  turn machine, `session.py`, `live.py`, the plugin and `services/`. VRAM: no new steady state beyond what the first
  real repair allocates; B7 unchanged. The warm-up's prompt differs from the session prefix and invalidates nothing;
  thinking is on as in every session request.
- Harness: `scripts/gpu/t0099_repair_live.py --warm-repair` (default off) runs the warm-up once with its own
  `VllmClient` after the idle and loopback checks and before case 0, and writes `regime.repair_warmup = {"on": true,
  "per_tool": {…}, "total_ms": …, "ended_wall": …}` (off: `{"on": false}`). A warm-up leaving any tool not `ok` voids
  the run (`integrity: repair warm-up incomplete (<tool>: <status>)`). The report adds `first_use_repair` (per payload
  sha, the first `tool_repair` request in manifest session order, either arm) and `first_use_repair_max_ms`, and
  report.md one table. `scripts/locked/t0101_warm_rerun.sh` boots fresh, runs the harness with `--warm-repair` into
  `logs/t0101-warm-rerun-<sha>/`, then `tests/eval/test_t0101_warm_rerun.py` when `report.json` exists.

### Pre-registration (H-T101b; read only if G1–G8 and criterion 6 hold)

Fresh boot at the T-0101 merge commit with `--warm-repair`:

- **C1** every payload tool appears in `first_use_repair` and every first-use `tool.repair` span is ≤ **600 ms**.
- **C2** arm-S `tool.repair` p90 (nearest rank) ≤ **1500 ms** when arm S fires ≥ 1 repair; otherwise "unread (n = 0)"
  and C1 decides (after T-0100, arm-S K-out and out-of-range C-pixel drafts are refused, so arm S fires fewer repairs
  than at a794a98).
- **C3** arm-F `tool.repair` p50 in **[203, 248] ms** (±10 % of a794a98's 225 ms).
- **C4** (reported) the slowest warm-up request is ≥ 1000 ms.

Readings: **R1** C1–C3 hold → the H-T99b trigger is closed by fixing the cause; the deadline stays 3000 ms. **R2** C1
fails for a tool whose warm-up was `ok` → cold state is not process-wide; re-plan, no deadline change. **R3** C1 holds,
C4 does not → the fix holds, attribution unconfirmed on this boot; inspect for a persisted cache. **R4** C1 holds, C2
fails → slow repairs are not first-use; open the budget/deadline task H-T99b named; the human decides. **R5** C3 fails
→ the warm-up changed a steady-state number; the fix is not accepted as is. Not allowed: dropping the first sample or
calling a cold first use an outlier.

Planned regime: T-0099's as recorded at a794a98 plus `--warm-repair` (profile `lean`, engine only, fresh boot; vLLM
v0.29.0; NVFP4 rev `2ea8372`; template `36e3a42e`; tokenizer `cc8d3a0c`; CL 256, 48 steps; thinking on; prefix caching
on; `max_num_seqs 3`; RTX 5090, driver 595.84, gpu-mem 0.86; `native:bbfc53ccdd80:dry_run`, repair on, calibration
`4084026c…`, payloads `native_T4.json`, deadline 3000 ms; typed turns; fake STT/audio; fake TTS 300 ms/word; fixture
frames; concurrency 1; no seed; stress `dea52a37…`; arm S 30 × k = 3; arm F 40; Q8 barge cap 5). Baseline: the
a794a98 run (warm-up off).

### Results (T-0101)

Confirmation run 2026-10-15 at the T-0101 merge commit `df379e4` (`scripts/locked/t0101_warm_rerun.sh`, exit 0;
harness exit 0, eval rows 8/8 passed). Read from `logs/t0101-warm-rerun-df379e4/` (`report.json` sha256 `842cabcb…`,
`report.md`, `manifest.json`), the console log `logs/t0101-warm-rerun-df379e4-run.log` and the engine log
`logs/vllm.log` of that boot. Not void; G1–G8 hold in both arms. Reviewer B re-derived every figure from the raw
files with an independent stdlib script (all agree). Reading: **R1**.

**Regime.**

| | |
|---|---|
| commit | `df379e4ee5496997ff64fec9a27075c9b4513034` (baseline: `a794a98`, warm-up off) |
| boot | fresh, engine only (`services/vllm.sh`), booted 20:09:35, healthy 20:11:55 (+01:00) |
| engine | vLLM `0.29.0`, `nvidia/diffusiongemma-26B-A4B-it-NVFP4` (rev `2ea8372`), template `36e3a42e`, tokenizer `cc8d3a0c`; `versions.lock` sha256 `5bab0570…` |
| device | RTX 5090, driver 595.84, gpu-mem 0.86; 27323 MiB used / 4764 MiB free after boot, 27331 / 4756 at the end |
| decoding | profile `lean`, CL 256, 48 steps, thinking on, prefix caching on, `max_num_seqs 3` |
| tools | `native:bbfc53ccdd80:dry_run`, repair on, deadline 3000 ms, calibration `4084026c…`, payloads `native_T4.json` |
| warm-up | `--warm-repair`: on |
| input | typed turns; fixture frames; fake STT/audio/perception; fake TTS 300 ms/word |
| sets | stress `dea52a37…` (arm S 30 × k = 3 = 90), cases `460cd1c3…` (arm F 40); 130 sessions |
| run | concurrency 1, no seed; Q8 barge cap 5; system prompt sha256 `3605c012…` (same as a794a98) |

**Warm-up** (`regime.repair_warmup`; one request per payload automaton, all `ok`, each output `valid`, 1 commit each):

| tool | wall ms | status |
|---|---|---|
| click_at | 4493.0 | ok |
| google_search | 1887.3 | ok |
| press_key | 1045.0 | ok |
| scroll | 213.9 | ok |
| type_at | 1631.9 | ok |
| **total** | **9271.3** | |

The warm-up ended at 20:12:06.021; the first session's `log.started` is 20:12:06.533 (512 ms later). No `warmup-`
request id appears in any session log. click_at was first in name order, so it also carried the boot's first CD
request's automaton-independent compiles (below).

**First use per automaton** (`first_use_repair`: the first `tool_repair` request per payload sha in manifest session
order, either arm; `tool.repair` span ms). The a794a98 column is the same report built from that run's files (no
warm-up):

| tool | df379e4 (warm): arm, case, rep | ms | a794a98 (cold): arm, case, rep | ms |
|---|---|---|---|---|
| click_at | F, click-01, 0 | 220 | S, cpix-03, 0 | 1895 |
| google_search | F, search-01, 0 | 223 | F, search-01, 0 | 234 |
| press_key | S, kout-03, 2 | 228 | S, kout-02, 0 | 2549 |
| scroll | F, scroll-01, 0 | 232 | S, sx-01, 0 | 2034 |
| type_at | F, type-01, 0 | 222 | F, type-01, 0 | 1473 |
| **max** | | **232** | | **2549** |

**C1–C4 (H-T101b).**

| | bar | value | n | holds |
|---|---|---|---|---|
| C1 every payload tool has a first use, each ≤ 600 ms | ≤ 600 ms | 220–232 ms (max 232) | 5 tools | yes |
| C2 arm-S `tool.repair` p90 (nearest rank) | ≤ 1500 ms | 228 ms (p50 228) | 1 repair | yes (see below) |
| C3 arm-F `tool.repair` p50 (nearest rank) | [203, 248] ms | 225 ms (p90 228; a794a98 225 / 232, n 39) | 40 repairs | yes |
| C4 slowest warm-up request (reported) | ≥ 1000 ms | 4493 ms (click_at) | 5 requests | yes |

**Reading R1.** C1–C3 hold: the H-T99b trigger (arm-S p90 2034 ms > 1500 ms at a794a98) is closed by fixing its
cause, cold first use. The deadline stays 3000 ms and `REPAIR_OPTIONS.max_tokens` stays 256. C4 holds (the slowest warm-up paid
seconds, as a cold first use did at a794a98), so the attribution to first-use compiles is consistent on this boot
(R3 does not apply).

**The compiles land in the warm-up (engine log).** The engine's JIT monitor logged four "Triton kernel JIT compilation
during inference" warnings (engine log stamps are UTC; +01:00 above). Three came at 19:11:57–19:11:59: `_lm_kernel` (the
plugin's `cd/kernels/logmatmul.py`) and `_stage1` / `_combine` (`cd/kernels/entropy.py`, the H(q) kernel). All three
fall between the first warm-up chat completion and the `ASTRA_CD` record tagged `warmup:click_at` (19:12:01), so inside
the click_at warm-up. The fourth, `_bmm_outer_product_kernel` (not a plugin kernel; that it is upstream vLLM is an
inference), came at 19:12:06. That is after the last warm-up record (`warmup:type_at`), on the first session's first
chat completion, a tap-on turn request and not a repair. No warning was logged on any session's repair (reviewer B's check, and none
after 19:12:06 in the log). The five `ASTRA_CD`
records tagged `warmup:` are the warm-ups; harnesses counting CD records filter by that tag. *Inference, not
verified:* the monitor appears to warn once per kernel name and not per specialisation. That would explain why the
google_search, type_at and press_key warm-ups (1045–1887 ms) were slow without a fresh warning: they compiled new
variants of already-named kernels (per-S specialisations; Evidence above). scroll's 214 ms is consistent with it
sharing google_search's variant group (np2 = 64, S mod 16 ≠ 0), as at a794a98.

**Arm S fired 1 repair, so C2 carries little information.** With T-0100 in force, arm-S K-out and out-of-range
C-pixel drafts are refused at detection instead of repaired. Arm S's yield was 1/63 (by parse reason: one `no close`),
against 11 repairs at a794a98. C2's n = 1 means its p90 is a single sample; C2 holding is weak evidence on its own, and
C1 (every tool's first use, mostly from arm F) decides, as pre-registered. Rescue reads "insufficient yield (n = 0)".

**kout-03 rep 2: the one counted coercion is the model's own substitution.** The report counts one coercion
(K-out repaired to a valid call: kout-03 rep 2 → `Ctrl+T`). From the session log, as recorded in `docs/tasks/T-0101.md` *Result*: in pass 1 the model drafted
the exact enum string `Ctrl+T`, left the call unclosed, and retracted it in the same commit ("I cannot perform
Ctrl+Shift+T … I should inform the user") after voicing "I will press Ctrl Shift T". The repair only closed the call.
The repair did not choose the substitute; the model did. That repair is also press_key's first use (228 ms).
**Ruling H-T101c (human, 2026-10-15):** a draft whose call did not close and is followed by more model text is
refused instead of repaired; clean truncations stay repairable. It is its own task, **T-0102**.

Other numbers from the run, reported only: arm F 40/40 repairs fired, agreement 40/40, exact 37/40, outcomes
`dry_run` 35 / `cancelled` 5, repair_timeout 0. Barge-in: 0 delivered of 5 attempted, with all 5 repairs done before
the takeover, so it reads "unread", as T-0099 allows. Arm S: 19 unrouted turns, 8 refused drafts, repair_timeout 0.
