# Native tool turns under the live prompt: an offline prompt A/B on dev (T-0098)

**Status: run 2026-10-15 at `09d1c50` (exit 0). Result: no change.** Neither candidate passes the pre-registered
rule. C2 fails P1. C1 fails P1 and P4. R1 and R2 both hold. The gpu eval row re-derived the same decision from
`trials.jsonl`. The human accepted "no change" (H-T98b, 2026-10-15), so T-0098b is not needed. The diagnostic arm D1
points at `UI_INSTRUCTIONS`; follow-up **T-0098c** designs candidates aimed at it. The method and the pre-registration
below are unchanged from before the run; the measurements are under [Results](#results).

**Delta (2026-10-16, H-T104, docs/01 D16):** for routine steps D16 supersedes the target "a spoken sentence before every tool call"; P1, P2, R1 and R2 may not be reused as a target for routine steps. The pre-registration, the results (no change) and the safety row P6 are unchanged.

Origin: finding 3 of [toolcall-live.md](toolcall-live.md). At `4b0bd92`, through a real `Session`, `type_at` turns spoke a
prefix on 0 of 9 routed turns. Under the T-0084e/f eval prompt they spoke on 9/10 (dev) and 18/20 (test). Task:
[`docs/tasks/T-0098.md`](../tasks/T-0098.md).

## Regime (pre-registered; confirmed by the run's manifest, see [Results](#results))

| | |
|---|---|
| engine | profile `lean`; vLLM `v0.29.0`; DiffusionGemma NVFP4 rev `2ea8372`; chat template `36e3a42e`; tokenizer `cc8d3a0c`; CL 256, 48 steps; thinking on; prefix caching on; `max_num_seqs 3` |
| device, boot | RTX 5090, driver 595.84, gpu-mem 0.86; one fresh engine-only boot via `scripts/locked/t0098_prompt_ab.sh` (the boot regime of `scripts/locked/t0093_live.sh`: `services/vllm.sh` unchanged since `806da12`, port and container refusals, proxies unset) |
| tools | `native:bbfc53ccdd80`, all tools on every arm (`tool_choice: "none"`); repair on (a kit with repair off is refused). Pass 1 only: the repair never runs here |
| cases | 53: the dev split of `tests/eval/toolcall` (40 tool + 8 no-tool, file order; `cases.jsonl` sha256 `460cd1c3…`), then T-0093's 5 `PLAIN_QUESTIONS` (J10). The test split is never read |
| arms | six (below), one request each per (case, repeat) |
| n | 5 repeats per (case, arm): 53 × 6 × 5 = 1590 requests; 50 trials per tool per arm |
| seed | none (the diffusion path refuses `seed`, so it is never sent); repeats are the only replication, and runs are not reproducible |
| concurrency | 1; one boot; no prefill, no continuation |
| input | typed text; the case's fixture screen as the turn's one frame; no `Session`, no audio, no memory hits, no assists |

## Method

**The harness** is `scripts/gpu/t0098_native_prompt_ab.py`. It runs through `/launch` only, against an engine that is
already booted. Before any socket opens it refuses (exit 2) when:
- the base URL is not an IP literal in 127.0.0.0/8 or `::1`;
- the tree is dirty (the dirty paths are named);
- the out-dir already exists and is not empty;
- `--split` is anything but `dev`;
- `cases.jsonl` is not the pinned one (`CASES_SHA256`);
- `cli.load_tool_kit` cannot load the kit, the repair is off, or the tool set is not `bbfc53ccdd80`;
- the A0 system text built by `PromptBuilder` does not hash to `3605c012…`. That would mean the live prompt has drifted
  since `4b0bd92`, and the run would no longer measure the finding;
- an arm's request body differs from A0's anywhere but `messages[0]`.

All HTTP goes to the loopback engine over an explicit `httpx.AsyncHTTPTransport()`, so no proxy is picked up from the
environment.

**The prompts.** For each case, A0 is the pass-1 prompt a fresh `Session` builds for one typed turn:
- `PromptBuilder.prefix` with no history, the case's frame and no assists;
- then `PromptBuilder.full` with the case's text and no memory hits, so a text-only case carries the `[screen]` note.

A unit row checks that this prompt equals the one a real `Session` over the fakes sends, and that the client's body for
it is byte-equal to the session's. Every other arm is A0 with `messages[0]` replaced. No module global is patched.

**The request.** Each trial sends pass 1 through the real `VllmClient` in id mode, with the kit's detokeniser and
`LlmOptions()`, as the session sends it. The bodies carry no `seed`, no `max_tokens: 1` prefill and no continuation.

**Order.** The loops run repeats outermost, then the cases in order. For each (case, repeat) the six arms are sent
starting at offset `(case_index + repeat) mod 6` of the arm order `A0 A1 D1 D2 C2 C1`. So every arm takes every slot
equally often, and no arm always follows another.

**The raw record.** `trials.jsonl` has one line per request, in request order. Each line holds:
- the case, arm, repeat, and `order` (the arm's slot 0–5 in its block);
- the sha256 of the `messages[0]` text actually sent;
- every commit (`segment`, `text`, `thought_open`, `canvas_idx`, `stop_id`);
- every `token_ids` id the engine streamed, stop id included, read off the wire by a tee on the transport;
- `error`, `wall_ms` and `first_canvas_ms`.

`manifest.json` holds the regime and the six arm texts with their shas. After 10 consecutive errored requests the run
stops (exit 1) and the manifest says why. `report.json` is `build_report(run_dir)`: it reads only `trials.jsonl`,
`manifest.json` and the pinned `cases.jsonl`. `report.md` renders it.

## Pre-registration (H-T98a, approved 2026-10-14)

### Arms

| arm | system message | role |
|---|---|---|
| A0 | live, exactly `PromptBuilder(DEFAULT_SYSTEM_PROMPT, tools=kit.tools).system_message()`: persona, `CLOSE_THE_THOUGHT`, `SPOKEN_STYLE`, `UI_INSTRUCTIONS`, `NATIVE_TOOL_INSTRUCTIONS` joined by `"\n"` (sha `3605c012…`, 2744 characters) | control |
| A1 | `twopass.NATIVE_SYSTEM_PROMPT` (sha `e0b54a588619…`) | positive control (the eval prompt) |
| D1 | A0 without `UI_INSTRUCTIONS` | diagnostic, never selected |
| D2 | A0 without `SPOKEN_STYLE` | diagnostic, never selected |
| C2 | A0 with `"\n\n"` (not `"\n"`) before `NATIVE_TOOL_INSTRUCTIONS`; wording unchanged | candidate 1 (smallest) |
| C1 | A0 with one sentence added after "…then call exactly one of the declared tools.", so that the paragraph reads: "Answer the user briefly. When an action is needed, first say in one short sentence what you will do, then call exactly one of the declared tools. Say that sentence in your reply, after you have finished thinking: what you write while thinking is never heard. Use at most one call per answer. When no action is needed, answer normally and call no tool." | candidate 2 |

### Definitions (criterion 5; the J-rulings carried over)

- *routed*: a `tool` commit outside an open thought (J8).
- *spoken*: the trial is routed and `drop_special_literals("".join(answer commits)).strip() != ""`, over the `answer` commits with
  `thought_open == False` only (ruling N4). This is the session's
  `_AnswerCleaner.spoken_prefix` (I15). Spoken rates are over routed trials.
- *valid* and *exact*: `toolcall.score` on the routed call, as T-0093 scores them, over routed trials.
- *route recall*: routed trials over all trials of the tool cases. A trial whose stream errored before any tool commit counts as unrouted; one that errored after its tool commit counts as routed (harness and oracle agree). Errored trials above 2 % void the arm.
- *false tool*: a routed trial on a no-tool or plain case (65 per arm).
- *thought_open*: a tool trial whose last thought never closed.
- *voiced anomaly*: a spoken prefix that contains `call:` or any special literal.
- *void arm*: an arm with more than 2 % errored trials. A void arm cannot be selected.

### Reproduction readings (labels, not gates)

- **R1:** A0 `type_at` spoken ≤ 0.3. If it fails, the report reads "finding not reproduced", **no candidate is
  selected** (N3), and the follow-up is a T-0093-harness re-run with n ≥ 3.
- **R2:** A1 `type_at` spoken ≥ 0.7. If it fails, the report carries "eval baseline not reproduced", and selection still
  proceeds against A0.

### Decision rule (criterion 7)

Candidate C passes iff all seven conditions hold, each against A0 in the same run:

| | condition |
|---|---|
| P1 | `type_at` spoken ≥ 0.80 over its routed trials, **and** at least 8 of the 10 type cases spoke in a majority of their routed trials |
| P2 | pooled click + scroll + key spoken ≥ A0 − 0.05 |
| P3 | pooled route recall (200) ≥ A0 − 0.02, **and** `type_at` route recall (50) ≥ A0 − 0.04 |
| P4 | valid ≥ A0 − 0.02 **and** exact ≥ A0 − 0.03, over routed trials |
| P5 | false tools ≤ 5/65 **and** ≤ A0 + 2 |
| P6 | voiced anomalies = 0 |
| P7 | thought_open on tool trials ≤ A0 + 4 |

**Winner:** C2 if it passes, else C1. If neither passes, the result is "no change". D1 and D2 are reported only (N2).
The rule is computed with exact fractions. The gpu eval row `tests/eval/test_toolcall_prompt_ab.py` re-derives every
P-condition, both readings, the void arms and the winner from `trials.jsonl` with its own oracle, and checks that they
agree with `report.json`.

### Ruled after the run (H-T98b)

Before the run, two questions were left to the human: which candidate wins, and, if C1 won, whether twopass changes too
or stays frozen with the parity test amended. The run selected no candidate; see [Human ruling](#human-ruling-h-t98b-2026-10-15).

## Results

Every number below comes from the run's own files, read-only:
`.claude/worktrees/t0098-launch/logs/t0098-prompt-ab-09d1c50/`:
- `report.json`, sha256 `defbb513…`;
- `trials.jsonl`, sha256 `212f5eb0…`;
- `manifest.json` and `report.md`;
- the launch log `…/logs/t0098-prompt-ab-09d1c50-run.log`.

The exception is a set of regime entries that are pins, not run outputs. The model rev, chat template, tokenizer,
gpu-mem and `max_num_seqs` come from `services/versions.lock` (sha256 `5bab0570…`, recorded in the manifest),
`services/vllm.sh` and `local_astra/config.py` at the run's commit.

Each rate is given as the raw k/n behind it. Spoken, valid and exact are over routed trials.

### Regime as run

| | |
|---|---|
| commit | `09d1c508fbce4c6393baa3255aae5a9abba293ce` (clean tree; the harness and the launch script both refuse a dirty one) |
| engine | profile `lean`; vLLM `0.29.0` (from `/version`); DiffusionGemma NVFP4 rev `2ea8372`; chat template `36e3a42e`; tokenizer `cc8d3a0c`; CL 256, 48 steps; thinking on; prefix caching on; `max_num_seqs 3`; `versions.lock` sha256 `5bab0570…` |
| device | RTX 5090, driver 595.84, gpu-mem 0.86; fresh engine-only boot via `scripts/locked/t0098_prompt_ab.sh`; device memory 27337 MiB used after boot |
| tools | `native:bbfc53ccdd80` on every arm, repair on (unused: pass 1 only) |
| cases | 53 dev cases (40 tool, 8 no-tool, 5 plain); `cases.jsonl` sha256 `460cd1c3…`; the test split was not read |
| design | 6 arms × 5 repeats = 1590 requests; rotated arm order; concurrency 1; no seed; typed text, no `Session`. **n = 50 trials per tool per arm** (25 for `scroll` and `press_key`) |
| arm texts (manifest) | A0 `3605c012a0fe…` 2744 chars; A1 `e0b54a588619…` 483; D1 `b795e27dd646…` 873; D2 `c6293ec5ff3f…` 2580; C2 `6a56ef583a3a…` 2745; C1 `9a4819a573fd…` 2857 |
| timing | boot started 2026-10-15 01:10:37, engine healthy 01:13:02, harness step 01:13:02–01:19:31 (1590 requests in 6 min 29 s, including its start-up and report writing), gpu eval row 01:19:31, engine stopped 01:19:33 |
| box | quiet: load average 0.13 (1-min) at boot; 391 MiB of device memory in use before the boot |
| outcome | 1590 trials, **0 errored**, no void arm; harness exit 0; gpu eval row (`tests/eval/test_toolcall_prompt_ab.py`, 2 passed) re-derived every P-condition, R1, R2, the void arms and the winner from `trials.jsonl` and agreed with `report.json`; script exit 0 |

### Reproduction readings

- **R1 holds.** A0 `type_at` spoke on 1 of its 50 routed trials (0.020 ≤ 0.3). That one trial is type-09, repeat 4:
  "I will type the password into the password field and press enter." T-0093 had 0/9 live at `4b0bd92`;
  the prompt alone reproduces it without a `Session`.
- **R2 holds.** A1 (the eval prompt) `type_at` spoke on 48 of its 48 routed trials (1.000 ≥ 0.7). T-0084e had 9/10.

### Decision table (criterion 7, each candidate against A0 in this run)

| | A0 (reference) | C2 | C1 | bar |
|---|---|---|---|---|
| P1 `type_at` spoken | 1/50 (0.020) | 12/50 (0.240) **FAIL** | 12/49 (0.245) **FAIL** | ≥ 0.80 |
| P1 type cases spoken in a majority | 0/10 | 2/10 **FAIL** | 2/10 **FAIL** | ≥ 8/10 |
| P2 click + scroll + key spoken | 78/99 (0.788) | 87/99 (0.879) pass | 73/96 (0.760) pass | ≥ A0 − 0.05 = 0.738 |
| P3 pooled route recall | 199/200 (0.995) | 199/200 (0.995) pass | 195/200 (0.975) pass (at the bar) | ≥ A0 − 0.02 = 0.975 |
| P3 `type_at` route recall | 50/50 (1.000) | 50/50 pass | 49/50 (0.980) pass | ≥ A0 − 0.04 = 0.96 |
| P4 valid | 199/199 (1.000) | 199/199 pass | 195/195 pass | ≥ A0 − 0.02 = 0.98 |
| P4 exact | 185/199 (0.930) | 182/199 (0.915) pass | 175/195 (0.897) **FAIL** | ≥ A0 − 0.03 = 17903/19900 ≈ 0.8997 |
| P5 false tools | 0/65 | 0/65 pass | 0/65 pass | ≤ 5 and ≤ A0 + 2 = 2 |
| P6 voiced anomalies | 0 | 0 pass | 0 pass | = 0 |
| P7 thought_open, tool trials | 0/200 | 0/200 pass | 1/200 pass | ≤ A0 + 4 = 4 |
| **passes** | | **no** (P1) | **no** (P1, P4) | |

Notes on the table:
- **C1's P4 miss** is less than one trial wide: 176/195 = 0.9026 would have passed. The P1 miss is not close on either
  candidate.
- **P7 is nowhere near its bar** (0 or 1 open thoughts per arm). The reviewer's caveat therefore does not arise: a
  thought spanning a canvas boundary would read as open on every arm alike.
- **C1's one open thought** is type-02, repeat 4: the call sits inside a thought that never closed ("…I will use
  `type_at` tool.<|tool_call>call:type_at{…}"). It is unrouted and outside the spoken denominator.

### Every arm (D1 and D2 are diagnostics, never selected)

| arm | type spoken | type majority cases | click+scroll+key spoken | route recall | type recall | valid | exact | false tools | anomalies | thought_open | errored |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 1/50 | 0 | 78/99 | 199/200 | 50/50 | 199/199 | 185/199 | 0/65 | 0 | 0/200 | 0/265 |
| A1 | 48/48 | 10 | 93/100 | 198/200 | 48/50 | 198/198 | 189/198 | 0/65 | 0 | 0/200 | 0/265 |
| D1 | 45/50 | 10 | 92/100 | 200/200 | 50/50 | 200/200 | 190/200 | 1/65 | 0 | 0/200 | 0/265 |
| D2 | 4/50 | 0 | 78/100 | 200/200 | 50/50 | 200/200 | 182/200 | 0/65 | 0 | 0/200 | 0/265 |
| C2 | 12/50 | 2 | 87/99 | 199/200 | 50/50 | 199/199 | 182/199 | 0/65 | 0 | 0/200 | 0/265 |
| C1 | 12/49 | 2 | 73/96 | 195/200 | 49/50 | 195/195 | 175/195 | 0/65 | 0 | 1/200 | 0/265 |

### Spoken prefix by tool, spoken/routed (n = 50 per tool per arm; 25 for scroll and press_key)

| arm | google_search | click_at | type_at | scroll | press_key |
|---|---|---|---|---|---|
| A0 | 18/50 | 43/50 | 1/50 | 16/25 | 19/24 |
| A1 | 5/50 | 48/50 | 48/48 | 23/25 | 22/25 |
| D1 | 13/50 | 46/50 | 45/50 | 22/25 | 24/25 |
| D2 | 14/50 | 45/50 | 4/50 | 13/25 | 20/25 |
| C2 | 24/50 | 50/50 | 12/50 | 16/25 | 21/24 |
| C1 | 27/50 | 42/50 | 12/49 | 14/25 | 17/21 |

Per-tool exact over routed trials, `click_at` / `type_at`:

| arm | `click_at` exact | `type_at` exact |
|---|---|---|
| A0 | 40/50 | 47/50 |
| A1 | 45/50 | 45/48 |
| D1 | 45/50 | 45/50 |
| D2 | 40/50 | 46/50 |
| C2 | 38/50 | 48/50 |
| C1 | 38/50 | 44/49 |

Every `google_search` and `scroll` call was exact on every arm. `press_key` exact: A0 23/24, A1 24/25, D1 25/25,
D2 21/25, C2 21/24, C1 18/21. The `press_key` recall of C1 (21/25) is where most of its pooled recall loss lies.

### Per case: spoken k / routed, of 5 trials

| case | A0 | A1 | D1 | D2 | C2 | C1 |
|---|---|---|---|---|---|---|
| search-01 | 2/5 | 0/5 | 1/5 | 2/5 | 3/5 | 2/5 |
| search-02 | 2/5 | 0/5 | 2/5 | 1/5 | 1/5 | 2/5 |
| search-03 | 1/5 | 0/5 | 1/5 | 0/5 | 2/5 | 3/5 |
| search-04 | 1/5 | 3/5 | 3/5 | 1/5 | 4/5 | 4/5 |
| search-05 | 1/5 | 0/5 | 1/5 | 0/5 | 4/5 | 3/5 |
| search-06 | 4/5 | 0/5 | 1/5 | 2/5 | 2/5 | 2/5 |
| search-07 | 0/5 | 0/5 | 0/5 | 1/5 | 1/5 | 2/5 |
| search-08 | 1/5 | 1/5 | 1/5 | 1/5 | 1/5 | 2/5 |
| search-09 | 2/5 | 0/5 | 3/5 | 4/5 | 3/5 | 4/5 |
| search-10 | 4/5 | 1/5 | 0/5 | 2/5 | 3/5 | 3/5 |
| click-01 | 3/5 | 5/5 | 4/5 | 3/5 | 5/5 | 3/5 |
| click-02 | 2/5 | 5/5 | 5/5 | 4/5 | 5/5 | 4/5 |
| click-03 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
| click-04 | 4/5 | 5/5 | 4/5 | 5/5 | 5/5 | 5/5 |
| click-05 | 5/5 | 5/5 | 4/5 | 5/5 | 5/5 | 5/5 |
| click-06 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 4/5 |
| click-07 | 5/5 | 4/5 | 5/5 | 5/5 | 5/5 | 4/5 |
| click-08 | 5/5 | 5/5 | 4/5 | 5/5 | 5/5 | 5/5 |
| click-09 | 4/5 | 5/5 | 5/5 | 4/5 | 5/5 | 2/5 |
| click-10 | 5/5 | 4/5 | 5/5 | 4/5 | 5/5 | 5/5 |
| type-01 | 0/5 | 5/5 | 5/5 | 0/5 | 0/5 | 0/5 |
| type-02 | 0/5 | 5/5 | 4/5 | 0/5 | 0/5 | 0/4 |
| type-03 | 0/5 | 4/4 | 5/5 | 1/5 | 2/5 | 3/5 |
| type-04 | 0/5 | 5/5 | 5/5 | 0/5 | 3/5 | 2/5 |
| type-05 | 0/5 | 5/5 | 4/5 | 1/5 | 3/5 | 4/5 |
| type-06 | 0/5 | 5/5 | 4/5 | 0/5 | 2/5 | 0/5 |
| type-07 | 0/5 | 5/5 | 4/5 | 0/5 | 1/5 | 0/5 |
| type-08 | 0/5 | 4/4 | 4/5 | 0/5 | 0/5 | 0/5 |
| type-09 | 1/5 | 5/5 | 5/5 | 2/5 | 1/5 | 2/5 |
| type-10 | 0/5 | 5/5 | 5/5 | 0/5 | 0/5 | 1/5 |
| scroll-01 | 3/5 | 4/5 | 5/5 | 2/5 | 3/5 | 3/5 |
| scroll-02 | 1/5 | 4/5 | 5/5 | 5/5 | 4/5 | 3/5 |
| scroll-03 | 4/5 | 5/5 | 4/5 | 4/5 | 3/5 | 3/5 |
| scroll-04 | 3/5 | 5/5 | 4/5 | 0/5 | 1/5 | 0/5 |
| scroll-05 | 5/5 | 5/5 | 4/5 | 2/5 | 5/5 | 5/5 |
| key-01 | 2/4 | 3/5 | 5/5 | 2/5 | 4/4 | 1/1 |
| key-02 | 4/5 | 5/5 | 5/5 | 5/5 | 4/5 | 4/5 |
| key-03 | 5/5 | 5/5 | 5/5 | 4/5 | 4/5 | 3/5 |
| key-04 | 3/5 | 4/5 | 5/5 | 5/5 | 5/5 | 4/5 |
| key-05 | 5/5 | 5/5 | 4/5 | 4/5 | 4/5 | 5/5 |

**False tools, no-tool and plain cases.** Every case on every arm reads 0/5, except D1 on plain-02 (1/5).

### Timing (reported only, not gated)

The box was quiet at boot (1-minute load 0.13, 391 MiB of device memory in use). These timings describe this offline harness, not a live turn.

| arm | first canvas ms p50 / p90 | wall ms p50 / p90 |
|---|---|---|
| A0 | 236.6 / 309.7 | 237.0 / 310.2 |
| A1 | 220.4 / 303.2 | 220.9 / 303.7 |
| D1 | 235.0 / 304.4 | 235.4 / 304.8 |
| D2 | 238.1 / 327.8 | 238.6 / 328.3 |
| C2 | 237.4 / 326.8 | 237.8 / 327.3 |
| C1 | 238.9 / 316.6 | 239.5 / 316.9 |

n = 265 per arm; nearest-rank percentiles. The first-canvas and wall medians differ by under 1 ms, so these
pass-1 responses were almost always complete at their first committed canvas. The median spoken prefix is 9–10 words on
every arm.

### What D1 and D2 suggest about the cause

*Inference, not a finding.* It rests on n = 50 `type_at` trials per arm, with no seed, one boot and the dev split only.
Neither diagnostic arm was eligible for selection (N2).

- **Removing `UI_INSTRUCTIONS` (D1) restores the spoken prefix.** `type_at` spoke 45/50, in a majority on 10/10 type
  cases, against A0's 1/50. Nothing else got materially worse: false tools 1/65 against A0's 0/65, and `type_at`
  exact 45/50 against 47/50, with every P-bar still holding. Recall was 200/200, valid 200/200 and exact 190/200. D1 is close to the eval prompt A1 (48/48). **The UI block is the likeliest cause.**
- **Removing `SPOKEN_STYLE` (D2) does almost nothing** (4/50). It is not a main cause.
- **What D1 cannot separate.** D1 takes out about 1870 of A0's 2744 characters (D1 is 873). It therefore cannot tell
  apart three explanations:
  - the UI block's *content*;
  - the sheer length of the system message ahead of the tool paragraph;
  - the tool paragraph reading as a continuation of the UI list.
- **The continuation explanation is at most a small part.** C2 adds only a blank line to break that continuation. It
  lifted `type_at` from 1/50 to 12/50.
- **Nor is explicit instruction enough.** C1 tells the model outright to say the sentence after thinking. It also
  reached only 12/49.
- **The effect is concentrated on `type_at`.** In A0, `click_at` (43/50), `scroll` (16/25) and `press_key` (19/24)
  still speak. In D1 they rise only modestly (46/50, 22/25, 24/25).

Designing candidates that target the UI block — shorter, moved, or reworded — is T-0098c's job, not this page's.

### Winner

**No change.** Neither C2 nor C1 passes. The live prompt stays as it is.

### Human ruling (H-T98b, 2026-10-15)

The human accepted "no change". T-0098b (ship the winner) is not needed. A follow-up, **T-0098c**, designs new
candidates aimed at `UI_INSTRUCTIONS`. They are pre-registered, run on the dev split only, and go back to the human
before any run. T-0099 runs on the current prompt. D1 itself is not shippable: it removes the UI channel's instructions.

T-0098c's method and pre-registration (H-T98c; the same harness under `--table T-0098c`): [toolcall-ui-prompt-ab.md](toolcall-ui-prompt-ab.md).
