# The UI block silences `type_at`: a pre-registered dev A/B aimed at `UI_INSTRUCTIONS`, with a UI-channel guard (T-0098c)

**Status: run 2026-10-15 at `3ac00a1` through `/launch` (`scripts/locked/t0098c_prompt_ab.sh`). Winner: no change.
C3 fails P8 only, and goes to a pre-registered confirmation run, T-0103 (human ruling 2026-10-15).** See
[Results](#results). Everything between this paragraph and *Results* is the pre-registration as approved; none of it was
edited after the run. Task: [`docs/tasks/T-0098c.md`](../tasks/T-0098c.md).

**Delta (2026-10-16, H-T104, docs/01 D16):** for routine steps D16 supersedes the target "a spoken sentence before every tool call", so "the UI block silences `type_at`" is closed as not a defect for routine steps; P1, P2, R1 and R2c may not be reused as a target for routine steps. The pre-registration, the results (no change; C3 retired by T-0103) and the safety row P6 are unchanged.

Origin: T-0098 ([toolcall-native-prompt-ab.md](toolcall-native-prompt-ab.md)). Under the live system message, `type_at`
turns spoke a prefix on 1/50 trials in A0. Removing `UI_INSTRUCTIONS` (D1) restored 45/50 without hurting routing. D1
cannot ship, because it removes the shown channel's instructions. This A/B tests three shippable candidates aimed at the
UI block, and guards the UI channel while doing it.

## Regime (pre-registered)

| | |
|---|---|
| engine | profile `lean`; vLLM `v0.29.0`; DiffusionGemma NVFP4 rev `2ea8372`; chat template `36e3a42e`; tokenizer `cc8d3a0c`; CL 256, 48 steps; thinking on; prefix caching on; `max_num_seqs 3` |
| device, boot | RTX 5090, driver 595.84, gpu-mem 0.86; one fresh engine-only boot via `scripts/locked/t0098c_prompt_ab.sh` (T-0098's boot regime: `services/vllm.sh` unchanged since `806da12`, port and container refusals, proxies unset) |
| tools | `native:bbfc53ccdd80`, all tools on every arm (`tool_choice: "none"`); repair on (a kit with repair off is refused). Pass 1 only: the repair never runs here |
| cases | 60, in this order: T-0098's 53 (the dev split of `tests/eval/toolcall`, 40 tool + 8 no-tool, `cases.jsonl` sha256 `460cd1c3…`, then T-0093's 5 `PLAIN_QUESTIONS`; indices 0–52 unchanged), then the 7 Q12 cases of `tests/eval/surface_placement_cases.jsonl` (J1; sha256 `e152810b…`; 4 positive + 3 negative; file order; indices 53–59). The test split is never read |
| arms | seven (below), one request each per (case, repeat) |
| n | 5 repeats per (case, arm): 60 × 7 × 5 = 2100 requests; per arm 50 trials per tool (25 for `scroll` and `press_key`), 20 Q12 positive and 15 Q12 negative trials |
| seed | none (the diffusion path refuses `seed`, so it is never sent); repeats are the only replication, and runs are not reproducible |
| concurrency | 1; one boot; no prefill, no repair, no continuation |
| input | typed text; no `Session`, no audio, no assists. Tool cases carry their fixture screen as the turn's one frame. No-tool, plain and Q12 cases have no frame, so they carry the `[screen]` note. `flight_booking_with_memory` carries its three captions as memory hits, built as T-0093's `_CaseMemory` serves them, with no `now_ms` (as the session; N10) |
| UI gate | `UiGate` and the vendored A2UI v0.9.1 schema at the run commit |

Estimated duration: about 10–14 minutes of requests plus about 2.5 minutes of boot. This is scaled from T-0098's 6.5
minutes for 1590 trials; it is not measured.

## Method

**One harness, two tables (criterion 1; N5).** The T-0098 harness, `scripts/gpu/t0098_native_prompt_ab.py`, gains a
registry `TABLES` keyed by task. A table carries:
- the arms in table order, and their roles;
- the candidates in winner order;
- the arm-text builder and the A0 pin;
- the case loader and its sources;
- the P-conditions and the rule id.

`--table` picks a table; the default is `T-0098`. Under the default, every existing name behaves as at `3ee1636`, and
the T-0098 tests pass unmodified. The manifest records `table` and `rule`. `build_report` reads the table from the
manifest; a manifest with no `table` key is read as T-0098, which keeps the `09d1c50` run reproducible.

**Refusals (criterion 2).** Before any socket opens, the harness refuses (exit 2) on every T-0098 refusal. That covers
the loopback URL, a dirty tree, a non-empty out-dir, a split other than dev, the pinned `cases.jsonl`, the tool kit, and
an A0 that does not hash to `3605c012…`. Under `--table T-0098c` it also refuses when:
- `surface_placement_cases.jsonl` does not hash to `e152810b…`, or does not hold exactly 4 positive and 3 negative cases;
- an arm cannot be built exactly as *Arms* defines it. C5's and C4's anchors must each occur exactly once in
  `UI_INSTRUCTIONS`. If an anchor occurs 0 times or 2 or more times, the run is refused, never edited twice or left
  unedited. C4's `UI_INSTRUCTIONS` must be at most 2000 code points.

The arm texts are built and checked before A0's sha is compared, so an anchor refusal names its arm.

**The prompts (criterion 4).** For each case, A0 is the pass-1 prompt a fresh `Session` builds for one typed turn:
- `PromptBuilder.prefix` with no history, the case's frame (none on a Q12 case) and no assists;
- `PromptBuilder.full` with the case's text, the case's memory hits, and no `now_ms`.

Every other arm is A0 with `messages[0]` replaced. No module global is patched. Before any socket, the harness checks
for every case that each arm's request body equals A0's outside `messages[0]` and differs inside it. Unit rows check
that A0 equals what a real `Session` over the fakes sends, for `capital_of_france` and for `flight_booking_with_memory`.

**Order (criterion 3).** Repeats outermost, then the 60 cases in order. For each (case, repeat), the seven arms start
at offset `(case_index + repeat) mod 7` of the table order `A0 D1 D3 D4 C3 C5 C4`.

**The raw record.** `trials.jsonl` has one line per request, as in T-0098. Each line also holds `request_id`, which is
the gate's `turn_id`, and Q12 lines hold `expect_surface`. `manifest.json` holds the regime, the table, the rule, and
the seven arm texts with their shas. `report.json` is `build_report(run_dir)`; it reads only `trials.jsonl`,
`manifest.json` and the pinned case files. `report.md` renders it.

**The gpu eval row (criterion 8).** `tests/eval/test_toolcall_ui_prompt_ab.py` reads `T0098C_RUN_DIR`. From
`trials.jsonl` alone, with its own `UiGate` calls, it re-derives:
- shown and placement;
- every P-condition;
- R1, R2c and R3;
- the void arms and the winner.

It then checks that they agree with `report.json`. The locked script runs it after the harness, and only if
`report.json` exists.

## Pre-registration (H-T98c, approved by the human 2026-10-15)

### Arms

Notation: `P = DEFAULT_SYSTEM_PROMPT`, `CT = CLOSE_THE_THOUGHT`, `SS = SPOKEN_STYLE`, `U = UI_INSTRUCTIONS` (1870
characters, sha12 `4360995b0136`), `N = NATIVE_TOOL_INSTRUCTIONS`. `i = U.index("- Between the fences")` (= 1353, the
start of a line). `U_head = U[:i-1]` has no trailing newline; `U_bullets = U[i:]`.

| arm | system message | role |
|---|---|---|
| A0 | `"\n".join([P, CT, SS, U, N])`, the live text, sha `3605c012…` (refused if it drifted) | control |
| D1 | `"\n".join([P, CT, SS, N])` | positive diagnostic, never selected |
| D3 | `"\n".join([P, CT, SS, U_bullets, N])` (the UI block's first half removed) | diagnostic, never selected |
| D4 | `"\n".join([P, CT, SS, U_head, N])` (the UI block's bullets removed) | diagnostic, never selected |
| C3 | `"\n".join([P, CT, SS, N, U])`: the tool paragraph before the UI block; wording unchanged | candidate 1 |
| C5 | A0 with `U`'s last bullet ending "what you are showing or doing." instead of "what you are showing.". The replacement is `"what you are showing."` → `"what you are showing or doing."`, which keeps the bullet's existing line break after "plain words" (N8). Exactly one occurrence | candidate 2 |
| C4 | A0 with `U`'s line "an empty or decorative surface is worse than none." followed by the new line "Every reply says at least one sentence aloud, after you have finished thinking, whatever else it contains." (wrapped after "whatever else", as `U` wraps). Exactly one occurrence; `U` becomes 1977 code points (N9), under the 2000 limit | candidate 3 |

Winner order: **C3, then C5, then C4**. A1 is dropped: T-0098's R2 already established it. D1 is this run's positive
control.

### Definitions

- *routed*, *spoken*, *valid*, *exact*, *route recall*, *thought_open*, *voiced anomaly*: as T-0098 defines them
  ([Definitions](toolcall-native-prompt-ab.md#definitions-criterion-5-the-j-rulings-carried-over)), over the same
  53-case denominators. Q12 trials never enter P1–P4 or P7.
- *false tool*: a routed trial on a no-tool or plain case (65 per arm). Q12 trials are not counted here; P11 counts
  them.
- *surface shown* (criterion 5; N6): per trial, a fresh `UiGate()` gates every `ui` commit in commit order. Each call
  is `gate(text, turn_id=request_id, canvas_idx=commit.canvas_idx, final=False)`, the session's `_gate_surface` call,
  made whatever the commit's `thought_open` says. The trial is *shown* if at least one call returned a surface. A later
  failing canvas never retracts it.
- *placement failure*: a `ui` commit with `thought_open == True`, or a `thought` commit containing `UI_OPEN` (the
  T-0055 defect). A surface built only inside an unclosed thought therefore counts as both shown (the session publishes
  it) and a placement failure.
- *every ui valid*: every gate call of the trial returned a surface (T-0097 criterion 2). It is reported, not gated.
- *`ui.gate` counts* per arm: gated (calls), emitted (calls that returned a surface), failed (calls that did not).
- *void arm*: more than 2 % errored trials (more than 6 of 300). A void arm cannot be selected, and a void A0 selects
  nothing.

### Readings

- **R1:** A0 `type_at` spoken ≤ 0.3. If it fails, the report reads "finding not reproduced", and **no candidate is
  selected**.
- **R2c:** D1 `type_at` spoken ≥ 0.7. If it fails, the report reads "UI-block cause not reproduced". This is a label
  only.
- **R3:** A0 Q12 positive shown ≥ 14/20. If it fails, the report reads "UI baseline not reproduced: the guard cannot
  detect a regression", and **no candidate is selected**.

### Decision rule (criterion 7)

Candidate C passes iff all eleven conditions hold, each against A0 in the same run:

| | condition |
|---|---|
| P1 | `type_at` spoken ≥ 0.80 over its routed trials, **and** at least 8 of the 10 type cases spoke in a majority of their routed trials |
| P2 | pooled click + scroll + key spoken ≥ A0 − 0.05 |
| P3 | pooled route recall (200) ≥ A0 − 0.02, **and** `type_at` route recall (50) ≥ A0 − 0.04 |
| P4 | valid ≥ A0 − 0.02 **and** exact ≥ A0 − 0.03, over routed tool trials |
| P5 | false tools ≤ 5/65 **and** ≤ A0 + 2 (the 65 no-tool + plain trials) |
| P6 | voiced anomalies = 0 |
| P7 | thought_open on tool trials ≤ A0 + 4 |
| P8 | Q12 positive shown ≥ A0 − 2 (of 20), **and** no positive case that A0 shows in ≥ 3/5 falls below 3/5 |
| P9 | surfaces on the 80 trials where none belongs (3 negative Q12 + 8 no-tool + 5 plain cases) ≤ A0 + 1, **and** on the 200 tool trials ≤ A0 + 2 |
| P10 | placement failures on Q12 positive trials ≤ A0 + 1 |
| P11 | routed tool calls on the 35 Q12 trials ≤ A0 + 2 |

**Winner:** the first passing candidate in the order C3, C5, C4. Selection goes by that order, not by the best numbers.
It happens only on a complete run in which R1 and R3 hold. Otherwise the result is "no change". D1, D3 and D4 are
reported only and are never selected (N7). The rule is computed with exact fractions.

### What the report carries (criterion 6)

Everything T-0098 reports, plus, per arm:
- Q12 positive shown k/20 and negative shown k/15;
- per-case Q12 shown k/5;
- surfaces on the 80 trials where none belongs, on the no-tool + plain trials, and on the 200 tool trials;
- placement failures (on Q12 positive trials and overall);
- tool calls on Q12 trials;
- every-ui-valid over gated trials;
- the `ui.gate` counts.

### What the write-up will say after the run (criterion 9)

- the regime as run;
- the decision table, with the raw k/n behind each rate;
- per-case tables;
- R1, R2c and R3;
- what D3 and D4 suggest about which half of the UI block acts, labelled as inference (n = 50 per tool per arm, no
  seed);
- the winner, or "no change".

If a candidate wins, T-0098d ships it. If C3 wins, H-T98c (e) accepts in principle amending T-0092b criterion 3 ("tool
paragraph last"): tools-off stays byte-identical, and the row is re-pinned, not weakened.

## Results

Every number below comes from the run's own files, read-only, in the detached launch worktree
`.claude/worktrees/t0098c-launch/logs/t0098c-prompt-ab-3ac00a1/`:
- `trials.jsonl`, sha256 `64d4143a…`;
- `report.json`, sha256 `6102112c…`;
- `manifest.json`, sha256 `8b0f664c…`;
- `report.md`, sha256 `e66a8d82…`;
- the launch log `…/logs/t0098c-prompt-ab-3ac00a1-run.log`.

The trial and report shas match the ones the launch log printed. The write-up re-derived every figure quoted here
from `trials.jsonl` alone, with its own aggregation. It called only the library primitives the definitions name:
`UiGate`, `UI_OPEN`, `drop_special_literals`, `native.parse_call` and `toolcall.score`. None of those modules changed
between `3ac00a1` and the write-up's base. Every figure agrees with `report.json` and `report.md`, and **no
disagreement was found**. One difference is representation only: the gpu eval row prints the winner as `None`, where
the report says "no change". The row passed, so the two agree.

Some regime entries are pins, not run outputs:
- the model rev and chat template come from `services/versions.lock` at `3ac00a1` (sha256 `5bab0570…`, as the
  manifest records);
- gpu-mem 0.86, `max_num_seqs 3` and prefix caching on (`--enable-prefix-caching`) come from `services/vllm.sh` at
  `3ac00a1` (no diff from `806da12`);
- the tokenizer `cc8d3a0c` is `TOKENIZER_SHA256` in `local_astra/toolcall/detok.py:50`. `HostDetokenizer.__init__`
  (`detok.py:152-158`) hashes the host `tokenizer.json` and raises on any mismatch. The harness loads it through
  `cli.load_tool_kit` at start-up, so a completed run checked the host copy. The engine's own copy, in the
  rev-`2ea8372` snapshot, was not hashed.

Each rate is given as the raw k/n behind it. Spoken, valid and exact are over routed tool trials.

### Regime as run

| | |
|---|---|
| commit | `3ac00a1b6dbf6313a3b59c7a37725588cae14d54` (clean; the harness refuses a dirty tree), from the detached worktree `.claude/worktrees/t0098c-launch` |
| table, rule | `T-0098c`, `H-T98c` |
| engine | `base_url` `http://127.0.0.1:8001/v1`; `served_model` `nvidia/diffusiongemma-26B-A4B-it-NVFP4` (rev `2ea8372`); `engine_version` `0.29.0` (from `/version`); chat template `36e3a42e`; tokenizer `cc8d3a0c`; `versions_lock_sha256` `5bab0570…` |
| decoding | `profile` `lean`; `canvas_length` 256; `max_denoising_steps` 48; `thinking` true; prefix caching on; `max_num_seqs 3` |
| device, boot | RTX 5090, driver 595.84, gpu-mem 0.86; one fresh engine-only boot. Before the boot: 378 MiB in use, 1-min load 0.07. After the boot: 27320 MiB used / 4767 MiB free. At the end: 27323 / 4764 |
| tools | `tools` `native:bbfc53ccdd80` on every arm; `repair` on (unused); `pass`: pass 1 only, so no prefill, no repair and no continuation; `LlmOptions()` as the session sends it |
| cases | `split` dev; `cases` 60. `cases_sha256` `460cd1c3…` (dev split of `tests/eval/toolcall/cases.jsonl`), plus T-0093 J10 `PLAIN_QUESTIONS`, plus `q12_sha256` `e152810b…` (`tests/eval/surface_placement_cases.jsonl`, J1). That is 40 tool, 8 no-tool, 5 plain and 7 Q12 (4 positive, 3 negative). The test split was not read |
| input | typed text, with no `Session`, no audio and no assists. Tool cases are sent over their fixture frame. No-tool, plain and Q12 cases have no frame and carry the `[screen]` note. `flight_booking_with_memory` carries its captions as memory hits, with no `now_ms` |
| `arms` | A0 D1 D3 D4 C3 C5 C4. System-text sha256 / chars (manifest): A0 `3605c012a0fe…` / 2744 (the pinned live text); D1 `b795e27dd646…` / 873; D3 `386558dec99b…` / 1391; D4 `6680987c06f0…` / 2226; C3 `72c3b92bcff2…` / 2744; C5 `2b90859bffb1…` / 2753; C4 `fc88ea0acc14…` / 2851 (U at 1977 code points, N9) |
| `rotation` | per (case, repeat), the arms start at offset (case_index + repeat) mod 7 of the arm order |
| `repeats`, n | 5. 60 × 7 × 5 = **2100 trials**, 300 per arm. **n = 50 trials per tool per arm** (25 for `scroll` and `press_key`); 20 Q12 positive and 15 Q12 negative trials per arm |
| `seed` | none (`null`; the diffusion path refuses `seed`). Repeats are the only replication, and the run is not reproducible |
| `concurrency` | 1 |
| `surface_shown` | per trial, a fresh `UiGate()` gates every `ui` commit in commit order, with `final=False` and the request id as `turn_id` (`session._gate_surface`). Shown means at least one call returned a surface |
| UI gate | `UiGate` + vendored A2UI v0.9.1 schema at `3ac00a1` (unchanged at `3578b66`) |
| timing | boot 2026-10-15 21:48:06 (+01:00); engine healthy 21:50:26; harness 21:50:26–22:00:31 (2100 requests in about 10 min, including start-up and report writing; the estimate was 10–14 min); gpu eval row 22:00:31–22:00:33; engine stopped 22:00:34 with 0 containers left |
| outcome | 2100/2100 trials, every (case, arm, repeat) key exactly once, **0 errored**, no void arm, `complete: true`. Harness exit 0. The gpu eval row (`tests/eval/test_toolcall_ui_prompt_ab.py`, 2 passed) re-derived the decision from `trials.jsonl` and agreed with `report.json`. Script exit 0 |

### Readings

- **R1 holds.** A0 `type_at` spoke on 8 of its 50 routed trials (0.160 ≤ 0.3). T-0098's A0 used the same system
  text (sha `3605c012…`) and read 1/50. We read the gap, 7 trials in 50 on an unseeded engine, as run-to-run spread
  (inference: two runs; the rotation and the arm set also changed). Read the small differences below against it.
- **R2c holds.** D1 `type_at` spoke on 42/50 (0.840 ≥ 0.7); T-0098's D1 read 45/50. This reading is a label only.
- **R3 holds.** A0 showed a Q12 positive surface on 15/20 (≥ 14/20), so the guard can detect a regression.

### Decision table (criterion 7, each candidate against A0 in this run)

| | A0 (reference) | C3 | C5 | C4 | bar |
|---|---|---|---|---|---|
| P1 `type_at` spoken | 8/50 (0.160) | 47/50 (0.940) pass | 14/50 (0.280) **FAIL** | 4/50 (0.080) **FAIL** | ≥ 0.80 |
| P1 type cases spoken in a majority | 1/10 | 10/10 pass | 1/10 **FAIL** | 1/10 **FAIL** | ≥ 8/10 |
| P2 click + scroll + key spoken | 73/98 (0.745) | 97/98 (0.990) pass | 80/98 (0.816) pass | 65/98 (0.663) **FAIL** | ≥ A0 − 0.05 = 681/980 ≈ 0.695 |
| P3 pooled route recall | 198/200 (0.990) | 198/200 pass | 198/200 pass | 198/200 pass | ≥ A0 − 0.02 = 0.970 |
| P3 `type_at` route recall | 50/50 | 50/50 pass | 50/50 pass | 50/50 pass | ≥ A0 − 0.04 = 0.960 |
| P4 valid | 198/198 (1.000) | 197/198 (0.995) pass | 198/198 pass | 198/198 pass | ≥ A0 − 0.02 = 0.980 |
| P4 exact | 180/198 (0.909) | 183/198 (0.924) pass | 178/198 (0.899) pass | 178/198 (0.899) pass | ≥ A0 − 0.03 = 967/1100 ≈ 0.879 |
| P5 false tools (no-tool + plain) | 0/65 | 0/65 pass | 0/65 pass | 0/65 pass | ≤ 5 and ≤ A0 + 2 = 2 |
| P6 voiced anomalies (all 300 trials, N11) | 0 | 0 pass | 0 pass | 0 pass | = 0 |
| P7 thought_open, tool trials | 0/200 | 1/200 pass | 0/200 pass | 0/200 pass | ≤ A0 + 4 = 4 |
| P8 Q12 positive shown | 15/20 | 12/20 **FAIL** | 15/20 pass | 14/20 pass | ≥ A0 − 2 = 13 |
| P8 per case: each case A0 shows in ≥ 3/5 (all four: flight_booking 4/5, flight_booking_with_memory 4/5, primes_table 3/5, omelette_steps 4/5) stays ≥ 3/5 | — | 3, 3, 4, **2** (omelette_steps 4/5 → 2/5) **FAIL** | 4, 4, 4, 3 pass | **2**, 4, 3, 5 (flight_booking 4/5 → 2/5) **FAIL** | ≥ 3/5 on each |
| P9 surfaces where none belongs | 0/80 | 0/80 pass | 0/80 pass | 0/80 pass | ≤ A0 + 1 = 1 |
| P9 surfaces on tool trials | 0/200 | 0/200 pass | 0/200 pass | 0/200 pass | ≤ A0 + 2 = 2 |
| P10 placement failures, Q12 positive | 1/20 | 1/20 pass | 2/20 pass (at the bar) | 0/20 pass | ≤ A0 + 1 = 2 |
| P11 routed tool calls on Q12 | 0/35 | 0/35 pass | 0/35 pass | 0/35 pass | ≤ A0 + 2 = 2 |
| void (> 6 of 300 errored) | no (0/300) | no (0/300) | no (0/300) | no (0/300) | |
| **passes** | | **no** (P8, both clauses) | **no** (P1) | **no** (P1, P2, P8 per-case clause) | |

Notes on the table:
- **C3 misses P8 by one trial on each clause.** 12/20 against a bar of 13, and omelette_steps 2/5 against 3/5. On
  P1–P7 and P9–P11 it passes, mostly by wide margins: `type_at` 47/50 against A0's 8/50, a majority on 10/10 type
  cases, and click + scroll + key 97/98 against 73/98.
- **C4 passes P8's total (14 ≥ 13) and fails only its per-case clause.** flight_booking fell from A0's 4/5 to 2/5.
  This is the case criterion 11's adversarial row describes.
- **C3's one open thought** is key-05, repeat 3. The call `press_key{combo: Ctrl+T}` sits inside a thought that never
  closed. That makes the trial unrouted, so it is outside the spoken denominator (`press_key` routed 23/25).
- **The placement failures are all of one kind.** Every one, on every arm, is a `thought` commit that holds
  `<a2ui>`. No `ui` commit was ever flagged `thought_open`. They fall on A0 flight_booking r2; C3 flight_booking r4;
  C5 flight_booking_with_memory r0 and flight_booking r1; D4 primes_table r0; D3 flight_booking_with_memory r1 and
  flight_booking r2, r3, r4.

### Every arm (D1, D3 and D4 are diagnostics, never selected)

| arm | type spoken | type majority cases | click+scroll+key spoken | route recall | type recall | valid | exact | false tools | anomalies | thought_open | errored |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 8/50 | 1 | 73/98 | 198/200 | 50/50 | 198/198 | 180/198 | 0/65 | 0 | 0/200 | 0/300 |
| D1 | 42/50 | 9 | 95/100 | 199/200 | 50/50 | 199/199 | 183/199 | 0/65 | 0 | 0/200 | 0/300 |
| D3 | 13/50 | 1 | 91/98 | 198/200 | 50/50 | 196/198 | 178/198 | 0/65 | 0 | 0/200 | 0/300 |
| D4 | 26/50 | 5 | 83/100 | 200/200 | 50/50 | 199/200 | 184/200 | 0/65 | 0 | 0/200 | 0/300 |
| C3 | 47/50 | 10 | 97/98 | 198/200 | 50/50 | 197/198 | 183/198 | 0/65 | 0 | 1/200 | 0/300 |
| C5 | 14/50 | 1 | 80/98 | 198/200 | 50/50 | 198/198 | 178/198 | 0/65 | 0 | 0/200 | 0/300 |
| C4 | 4/50 | 1 | 65/98 | 198/200 | 50/50 | 198/198 | 178/198 | 0/65 | 0 | 0/200 | 0/300 |

The UI channel per arm:

| arm | Q12 positive shown | Q12 negative shown | surfaces where none belongs | surfaces on tool trials | placement failures (Q12 positive / all) | tool calls on Q12 | every ui valid (gated trials) | `ui.gate` gated / emitted / failed |
|---|---|---|---|---|---|---|---|---|
| A0 | 15/20 | 0/15 | 0/80 | 0/200 | 1/20 / 1/300 | 0/35 | 11/18 | 51 / 31 / 20 |
| D1 | 0/20 | 0/15 | 0/80 | 0/200 | 0/20 / 0/300 | 1/35 | 0/0 | 0 / 0 / 0 |
| D3 | 0/20 | 0/15 | 0/80 | 0/200 | 4/20 / 4/300 | 0/35 | 0/13 | 16 / 0 / 16 |
| D4 | 15/20 | 0/15 | 0/80 | 0/200 | 1/20 / 1/300 | 0/35 | 11/17 | 43 / 27 / 16 |
| C3 | 12/20 | 0/15 | 0/80 | 0/200 | 1/20 / 1/300 | 0/35 | 7/16 | 48 / 30 / 18 |
| C5 | 15/20 | 0/15 | 0/80 | 0/200 | 2/20 / 2/300 | 0/35 | 9/17 | 46 / 27 / 19 |
| C4 | 14/20 | 0/15 | 0/80 | 0/200 | 0/20 / 0/300 | 0/35 | 11/20 | 46 / 27 / 19 |

D1's one Q12 tool call is flight_booking_with_memory r4, a `google_search` for "flight booking UI design template".
D1 has no UI instructions, so it emitted no `ui` commit at all.

### Spoken prefix by tool, spoken/routed (n = 50 per tool per arm; 25 for scroll and press_key)

| arm | google_search | click_at | type_at | scroll | press_key |
|---|---|---|---|---|---|
| A0 | 17/50 | 41/50 | 8/50 | 13/25 | 19/23 |
| D1 | 15/49 | 50/50 | 42/50 | 22/25 | 23/25 |
| D3 | 5/50 | 47/50 | 13/50 | 22/25 | 22/23 |
| D4 | 10/50 | 44/50 | 26/50 | 16/25 | 23/25 |
| C3 | 11/50 | 50/50 | 47/50 | 24/25 | 23/23 |
| C5 | 19/50 | 45/50 | 14/50 | 14/25 | 21/23 |
| C4 | 16/50 | 38/50 | 4/50 | 9/25 | 18/23 |

### Per case: `type_at` spoken k / routed, of 5 trials

| case | A0 | D1 | D3 | D4 | C3 | C5 | C4 |
|---|---|---|---|---|---|---|---|
| type-01 | 0/5 | 4/5 | 2/5 | 1/5 | 5/5 | 0/5 | 0/5 |
| type-02 | 0/5 | 5/5 | 0/5 | 3/5 | 3/5 | 0/5 | 0/5 |
| type-03 | 1/5 | 3/5 | 1/5 | 4/5 | 5/5 | 2/5 | 0/5 |
| type-04 | 0/5 | 5/5 | 2/5 | 2/5 | 5/5 | 1/5 | 0/5 |
| type-05 | 4/5 | 2/5 | 3/5 | 4/5 | 5/5 | 2/5 | 3/5 |
| type-06 | 1/5 | 5/5 | 1/5 | 4/5 | 4/5 | 1/5 | 0/5 |
| type-07 | 1/5 | 5/5 | 1/5 | 4/5 | 5/5 | 2/5 | 1/5 |
| type-08 | 0/5 | 4/5 | 2/5 | 0/5 | 5/5 | 1/5 | 0/5 |
| type-09 | 0/5 | 5/5 | 1/5 | 2/5 | 5/5 | 3/5 | 0/5 |
| type-10 | 1/5 | 4/5 | 0/5 | 2/5 | 5/5 | 2/5 | 0/5 |

The per-case tables for the other 30 tool cases, and for the 13 no-tool and plain cases, are in `report.md`. The
re-derivation agrees with them. Every no-tool and plain case reads 0/5 false tools on every arm.

### Per case: Q12 surface shown, k / 5

| case | expect | A0 | D1 | D3 | D4 | C3 | C5 | C4 |
|---|---|---|---|---|---|---|---|---|
| flight_booking | surface | 4/5 | 0/5 | 0/5 | 4/5 | 3/5 | 4/5 | **2/5** |
| flight_booking_with_memory | surface | 4/5 | 0/5 | 0/5 | 5/5 | 3/5 | 4/5 | 4/5 |
| primes_table | surface | 3/5 | 0/5 | 0/5 | 3/5 | 4/5 | 4/5 | 3/5 |
| omelette_steps | surface | 4/5 | 0/5 | 0/5 | 3/5 | **2/5** | 3/5 | 5/5 |
| capital_of_france | none | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| spider_legs | none | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |
| cloud_joke | none | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 |

### What D3 and D4 suggest about which half of the UI block acts

*Inference, not a finding.* It rests on n = 50 `type_at` trials per arm, with no seed, one boot and the dev split only.
A0 itself moved from 1/50 to 8/50 between two runs of the same text. D3 and D4 are diagnostics and were never
eligible for selection (N7).

- **Neither half, removed alone, restores the prefix as D1 does.**
  - D3 removes the first half (S1–S2, the fence, the example and the catalog) and keeps the bullets. `type_at` read
    13/50, a majority on 1/10 cases. That is within the run-to-run spread of A0's 8/50.
  - D4 removes the bullets (S3–S5) and keeps the first half. It read 26/50, a majority on 5/10 cases. That is about
    halfway between A0 (8/50) and D1 (42/50).
  - On this evidence the bullets carry more of the effect than the first half, but not all of it.
- **Length is not the explanation.** D3 (1391 characters) is shorter than D4 (2226) and speaks less. C3 has exactly
  A0's characters, reordered, and speaks 47/50. This weighs against T-0098's length confound. What changes the
  outcome is where the UI block sits relative to the tool paragraph, more than how much text comes before it.
- **Rewording the closest bullet or adding an explicit rule does not help.** C5 changes the bullet nearest the tool
  paragraph ("showing or doing") and read 14/50. C4 adds an explicit "say at least one sentence aloud" line and read
  4/50, with click + scroll + key falling to 65/98. T-0098's C2 and C1 found the same: a blank line and an explicit
  sentence each reached about 12/50. Every arm that keeps the whole UI block ahead of the tool paragraph stays low:
  A0, C5 and C4 here, C2 and C1 in T-0098. The one arm that moves the block after the tool paragraph (C3) speaks.
  This points to order, and to the bullets in particular, not to any one sentence.
- **The first half carries the surface format.** Without it (D3), the model still opened `ui` fences on 13 trials,
  but wrote a JSON format of its own (`{"components": [{"type": "header", …}, {"type": "input", …}]}`). The gate
  rejected all 16 of its `ui` commits: 0/20 shown. D3 also had 4 of the run's 9 placement failures. With the first half and no
  bullets (D4), Q12 matched A0 at 15/20. So the bullets are not needed for uptake on these four cases, though they
  may matter for what this run does not score.

### Winner

**No change.** No candidate passes all of P1–P11. R1 and R3 hold on a complete run with no void arm, so this is the
rule's verdict and not a refusal to select. The live prompt stays as it is, and T-0098d does not ship anything from
this run.

The numbers behind the verdict:
- **C3**, first in winner order, fails only P8: 12/20 against a bar of 13, and omelette_steps 4/5 → 2/5. It passes
  P1 (47/50, 10/10 cases), P2 (97/98), P3 (198/200, 50/50), P4 (197/198 valid, 183/198 exact), P5 (0/65), P6 (0),
  P7 (1/200), P9 (0/80, 0/200), P10 (1/20) and P11 (0/35).
- **C5** fails P1 (14/50, 1/10 cases) and passes the rest.
- **C4** fails P1 (4/50, 1/10), P2 (65/98 < 0.695) and P8's per-case clause (flight_booking 2/5).

C3's every-ui-valid, which is reported and not gated, is 7/16 against A0's 11/18; at this n the two cannot be told
apart. At n = 5 per case, a one-trial miss on each P8 clause cannot be told apart from the run-to-run spread either.
The pre-registered bar still reads it as a fail.

### Human ruling (2026-10-15)

The human chose the option labelled **"Pre-register a C3 confirmation run"**, whose description read: "New task: C3 vs
today's prompt only, with more repeats on the four card cases, under P8 as written. Rule fixed before the run, one more
/launch. The bar is not loosened." That run is **T-0103**. Its two arms are C3 and today's prompt
(A0); C5 and C4 are not in it. T-0098d waits on T-0103's result. If C3 is confirmed, T-0098d ships it, with H-T98c (e)'s
re-pin of T-0092b criterion 3.

## C3 confirmation (T-0103)

**Status: pre-registered (H-T103, ruled by the human 2026-10-15); not run yet.** This section is the pre-registration,
committed before the `/launch`. The results will be appended below it after the run, and this part will not be edited.
Nothing above this section is changed by T-0103. Task: [`docs/tasks/T-0103.md`](../tasks/T-0103.md).

The T-0098c run at `3ac00a1` read C3 as failing only P8, by one trial on each clause, at n = 5 per case with no seed.
T-0103 re-runs A0 and C3 only, with the same texts, 40 repeats on the four Q12 positive cases and 5 on every other
case, under P8 as written (reading S: the same proportions on the larger denominators). The bar is not loosened. The
T-0098c data is not pooled with this run.

### Regime (pre-registered)

T-0098c's regime verbatim, except for the arms, the repeats and the counts:

| | |
|---|---|
| engine | profile `lean`; vLLM `v0.29.0`; DiffusionGemma NVFP4 rev `2ea8372`; chat template `36e3a42e`; tokenizer `cc8d3a0c`; CL 256, 48 steps; thinking on; prefix caching on; `max_num_seqs 3` |
| device, boot | RTX 5090, driver 595.84, gpu-mem 0.86; one fresh engine-only boot via `scripts/locked/t0103_c3_confirm.sh` (T-0098c's boot regime verbatim: `services/vllm.sh` unchanged since `806da12`, port and container refusals, proxies unset) |
| tools | `native:bbfc53ccdd80`, all tools on both arms; repair on (unused: pass 1 only) |
| cases | T-0098c's 60 with the same order and indices: 0–52 T-0098's 53 (40 tool, 8 no-tool, 5 plain), 53–56 the Q12 positives (flight_booking, flight_booking_with_memory, primes_table, omelette_steps), 57–59 the Q12 negatives (capital_of_france, spider_legs, cloud_joke). Pins unchanged: `cases.jsonl` `460cd1c3…`, `surface_placement_cases.jsonl` `e152810b…`; dev split only |
| arms | A0 and C3 (below) |
| n | 40 repeats on indices 53–56, 5 on the other 56: 4 × 40 × 2 + 56 × 5 × 2 = **880 requests**, 440 per arm |
| seed | none; concurrency 1; no prefill, no repair, no continuation |
| input | typed text, no `Session`, no audio, no assists; as T-0098c (Q12 cases no frame; the memory case its captions as hits, no `now_ms`) |
| UI gate | `UiGate` and the vendored A2UI v0.9.1 schema at the run commit |
| out-dir | `logs/t0103-c3-confirm-<sha7>`; request ids `t0103-r{repeat}-{case_id}-{arm}` |

Estimated duration (from the `3ac00a1` per-kind means, not measured): about 7.4 minutes of requests, plus about
2 min 20 s of boot and the plugin wheel build.

### Arms

| arm | system message | sha256 (refused if it drifted) | role |
|---|---|---|---|
| A0 | `"\n".join([P, CT, SS, U, N])`, the builder's own | `3605c012a0fecf70738cb4917f28ca53e3b5aa69caf18ae291a34c0fbab79e57` | control |
| C3 | `"\n".join([P, CT, SS, N, U])`, built by T-0098c's own construction | `72c3b92bcff2e5924901e42475342826ffc38efefa116f66138f8aa43bad9308` | the candidate being confirmed |

Both texts are 2744 characters. There are no diagnostics and no other candidates.

### Design

- **Order.** Repeats are outermost (r = 0…39). Within a repeat, the cases scheduled for it run in index order: all 60
  for r ≤ 4, only 53–56 for r ≥ 5. Within each (case, repeat), both arms run back to back, starting from offset
  `(case_index + repeat) mod 2` of the table order `A0 C3`. Each positive case therefore has A0 first exactly 20 times
  out of 40.
- **Refusals** (exit 2, before any socket): every T-0098c refusal; an A0 text whose sha is not `3605c012…`; a C3 text
  whose sha is not `72c3b92b…` or that is not exactly `"\n".join([P, CT, SS, N, U])`; a repeats map that is not exactly
  40 on the four positive ids and 5 on the other 56.
- **Per-arm denominators.** 50 per tool (25 for `scroll` and `press_key`), 200 tool trials; 65 no-tool + plain trials;
  15 Q12 negative trials; 80 trials where no surface belongs; **160 Q12 positive trials; 175 Q12 trials; 440 trials in
  all.**

### Decision rule (H-T103 (c)–(d); exact fractions; C3 against A0 in the same run)

Shown, placement failure, false tool, spoken, routed, valid, exact, thought_open and anomaly are as defined in
T-0098c's [Definitions](#definitions).

| | condition | bar at this run's denominators |
|---|---|---|
| P1–P7 | verbatim from T-0098c's [Decision rule](#decision-rule-criterion-7), over the same denominators (50 `type_at`, 10 type cases × 5, click + scroll + key routed, 200 tool trials, 65 no-tool + plain). P6 counts every trial, the 320 positives included (N11) | unchanged |
| P8 total | C3 Q12 positive shown ≥ A0 − 1/10 × 160 | **≥ A0 − 16** (`16/160`) |
| P8 per case | a positive case is *protected* iff A0 shows it in ≥ 3/5 of its 40 trials; P8 fails if any protected case has C3 < 3/5 | protected iff A0 ≥ **24/40**; fails at C3 ≤ 23/40 |
| P9 | verbatim: surfaces on the 80 trials where none belongs ≤ A0 + 1, **and** on the 200 tool trials ≤ A0 + 2 | unchanged |
| P10 | C3 placement failures on Q12 positive trials ≤ A0 + 1/20 × 160 | **≤ A0 + 8** (`8/160`) |
| P11 | C3 routed calls on the 175 Q12 trials ≤ A0 + 2/35 × 175 | **≤ A0 + 10** (`10/175`) |
| R1 (control) | A0 `type_at` spoken ≤ 3/10 of its routed trials | as T-0098c |
| R3 (control) | A0 Q12 positive shown ≥ 7/10 × 160 | **≥ 112** (`112/160`) |
| void | an arm with > 2 % errored trials | ≥ 9 of 440 |

Reading S, as ruled (H-T103 (c)): the T-0098c margins keep their proportions of the denominators. The per-case clause
is kept as written (H-T103 (e)); a simulation in the task file estimates that an equal C3 fails it about 25 % of the
time on primes_table alone (inference). R2c is dropped (there is no D1).

**Outcome** (exactly one):
- **"C3 confirmed"** iff the run is complete, neither arm is void, R1 and R3 hold, and C3 passes P1–P11. Then
  T-0098d ships C3.
- **"C3 not confirmed: fails P…"** iff the run is complete, nothing is void, R1 and R3 hold, and some P fails. Every
  failing P is named. C3 is then retired and the live prompt stays.
- **"inconclusive: <reason>"** otherwise: an incomplete run, a void A0 or C3, R1 failing ("finding not reproduced"),
  or R3 failing ("UI baseline not reproduced"). Nothing is selected, and the result goes back to the human.

Only "C3 confirmed" selects anything (`winner` `"C3"`; otherwise `"no change"`). There is no second T-0103 run at
another sha without a new human ruling.

### What the report carries

The T-0098c per-arm keys for A0 and C3, plus: per positive case, k/40 for each arm and whether it is protected; each
scaled threshold as k/n beside the value it is compared with; Q12 positive shown split by repeat block (r 0–4 and
r 5–39), reported and not gated; `outcome` and `winner`. The gpu eval row `tests/eval/test_toolcall_c3_confirm.py`
(`T0103_RUN_DIR`) re-derives the schedule, shown, placement, P1–P11, R1, R3, the void arms and the outcome from
`trials.jsonl` alone and checks them against `report.json`.

After the run this section gains: the regime as run, the decision table with raw k/n, the per-case table, R1/R3, and
the outcome.

### Results (T-0103)

Every number below comes from the run's own files, read-only, in the detached launch worktree
`.claude/worktrees/t0103-launch/logs/t0103-c3-confirm-6720e56/`:
- `trials.jsonl`, sha256 `cbb3e764…`;
- `report.json`, sha256 `b578aeb5…`;
- `manifest.json`, sha256 `15d57325…`;
- `report.md`, sha256 `761d4c36…`;
- the launch log `…/logs/t0103-c3-confirm-6720e56-run.log` (sha256 `85cf5766…`).

The trial and report shas match the ones the launch log printed. The write-up re-derived every figure quoted here
from `trials.jsonl` alone, with its own aggregation. It called only the library primitives the definitions name:
`UiGate`, `UI_OPEN`, `drop_special_literals`, `native.parse_call` and `toolcall.score`. Nothing under `local_astra/`
or `services/` changed between `6720e56` and the write-up's base (`73ed0db`). The re-derivation also rebuilt the
schedule from the manifest's case indices and repeats map, and it matches the file line for line. Every figure agrees
with `report.json` and `report.md`, per arm, per tool, per case and per block, and **no disagreement was found**.

Some regime entries are pins, not run outputs:
- the model rev and chat template come from `services/versions.lock` at `6720e56` (sha256 `5bab0570…`, as the
  manifest records);
- gpu-mem 0.86, `max_num_seqs 3` and prefix caching on (`--enable-prefix-caching`) come from `services/vllm.sh` at
  `6720e56` (no diff from `806da12`);
- the tokenizer `cc8d3a0c` is `TOKENIZER_SHA256` in `local_astra/toolcall/detok.py:50`. `HostDetokenizer.__init__`
  (`detok.py:152-158`) hashes the host `tokenizer.json` and raises on any mismatch, so a completed run checked the host
  copy. The engine's own copy, in the rev-`2ea8372` snapshot, was not hashed.

Each rate is given as the raw k/n behind it. Spoken, valid and exact are over routed tool trials.

#### Regime as run

| | |
|---|---|
| commit | `6720e56c3c290d4614d60c3246f6b2b789faa125` (clean; the harness refuses a dirty tree), from the detached worktree `.claude/worktrees/t0103-launch` |
| table, rule | `T-0103`, `H-T103`; `reading` S (H-T103 (c)) |
| engine | `base_url` `http://127.0.0.1:8001/v1`; `served_model` `nvidia/diffusiongemma-26B-A4B-it-NVFP4` (rev `2ea8372`); `engine_version` `0.29.0` (from `/version`); chat template `36e3a42e`; tokenizer `cc8d3a0c`; `versions_lock_sha256` `5bab0570…` |
| decoding | `profile` `lean`; `canvas_length` 256; `max_denoising_steps` 48; `thinking` true; prefix caching on; `max_num_seqs 3` |
| device, boot | RTX 5090, driver 595.84, gpu-mem 0.86; one fresh engine-only boot via `scripts/locked/t0103_c3_confirm.sh`. Before the boot: 380 MiB in use, 1-min load 0.59. After the boot: 27323 MiB used / 4764 MiB free. At the end: 27324 / 4763 |
| tools | `tools` `native:bbfc53ccdd80` on both arms; `repair` on (unused); `pass`: pass 1 only, so no prefill, no repair and no continuation; `LlmOptions()` as the session sends it |
| cases | `split` dev; `cases` 60, T-0098c's order and indices. `cases_sha256` `460cd1c3…` (dev split of `tests/eval/toolcall/cases.jsonl`), plus T-0093 J10 `PLAIN_QUESTIONS`, plus `q12_sha256` `e152810b…` (`tests/eval/surface_placement_cases.jsonl`, J1). That is 40 tool, 8 no-tool, 5 plain and 7 Q12 (4 positive, 3 negative). The test split was not read |
| input | typed text, with no `Session`, no audio and no assists. Tool cases are sent over their fixture frame. No-tool, plain and Q12 cases have no frame and carry the `[screen]` note. `flight_booking_with_memory` carries its captions as memory hits, with no `now_ms` |
| `arms` | A0 C3. System-text sha256 / chars (report.json): A0 `3605c012a0fe…` / 2744 (the live text); C3 `72c3b92bcff2…` / 2744. Every trial's `system_sha256` is its arm's |
| `repeats` | 40 on flight_booking, flight_booking_with_memory, primes_table and omelette_steps (indices 53–56); 5 on the other 56 |
| `schedule`, `rotation` | repeats outermost; within a repeat, the cases scheduled for it in index order; per (case, repeat) the two arms back to back from offset (case_index + repeat) mod 2 of `A0 C3` |
| n | `requests` 880 = 4 × 40 × 2 + 56 × 5 × 2, 440 per arm. Per arm: **n = 50 trials per tool** (25 for `scroll` and `press_key`), 200 tool trials, 65 no-tool + plain, **160 Q12 positive (40 per case)**, 15 Q12 negative, 80 where no surface belongs, 175 Q12 |
| `seed` | none (`null`; the diffusion path refuses `seed`). Repeats are the only replication, and the run is not reproducible |
| `concurrency` | 1 |
| `thresholds` | `P8_total_margin` `16/160`; `P8_protected_min` `24/40`; `P10_margin` `8/160`; `P11_margin` `10/175`; `R3_min` `112/160` |
| `surface_shown` | per trial, a fresh `UiGate()` gates every `ui` commit in commit order, with `final=False` and the request id as `turn_id` (`session._gate_surface`). Shown means at least one call returned a surface |
| UI gate | `UiGate` + vendored A2UI v0.9.1 schema (`local_astra/ui/schemas/a2ui_v0_9_1/`) at `6720e56` (unchanged at `73ed0db`) |
| timing | boot 2026-10-16 09:27:23 (+01:00); engine healthy 09:29:48; harness 09:29:48–09:36:57 (880 requests in 7 min 9 s, including start-up and report writing; the summed request wall time is 416.9 s, a mean of 0.474 s, against the estimate of about 7.4 min); gpu eval row 09:36:57–09:37:02; engine stopped 09:37:02 with 0 containers left |
| outcome | 880/880 trials (`trials_written` 880), every (case, arm, repeat) key exactly once, **0 errored**, no void arm, `complete: true`. Harness exit 0. The gpu eval row (`tests/eval/test_toolcall_c3_confirm.py`, 2 passed) re-derived the outcome from `trials.jsonl` and agreed with `report.json`. Script exit 0 |

#### Controls

- **R1 holds.** A0 `type_at` spoke on 4 of its 49 routed trials (4/49 ≈ 0.082 ≤ 3/10). One `type_at` trial
  (type-10, r3) put its call inside a thought that never closed, so it is unrouted.
- **R3 holds.** A0 showed a Q12 positive surface on 129/160 (≥ 112/160), so the guard can detect a regression.

#### Decision table (H-T103 (c)–(d), C3 against A0 in this run)

| | A0 (reference) | C3 | bar | C3 |
|---|---|---|---|---|
| P1 `type_at` spoken | 4/49 (0.082) | 47/50 (0.940) | ≥ 0.80 | pass |
| P1 type cases spoken in a majority | 0/10 | 10/10 | ≥ 8/10 | pass |
| P2 click + scroll + key spoken | 70/99 (0.707) | 100/100 (1.000) | ≥ A0 − 0.05 = 1301/1980 ≈ 0.657 | pass |
| P3 pooled route recall | 197/200 (0.985) | 200/200 | ≥ A0 − 0.02 = 193/200 = 0.965 | pass |
| P3 `type_at` route recall | 49/50 (0.980) | 50/50 | ≥ A0 − 0.04 = 47/50 = 0.940 | pass |
| P4 valid | 196/197 (0.995) | 200/200 | ≥ A0 − 0.02 = 9603/9850 ≈ 0.975 | pass |
| P4 exact | 178/197 (0.904) | 184/200 (0.920) | ≥ A0 − 0.03 = 17209/19700 ≈ 0.874 | pass |
| P5 false tools (no-tool + plain) | 0/65 | 0/65 | ≤ 5 and ≤ A0 + 2 = 2 | pass |
| P6 voiced anomalies (all 440 trials, N11) | 0 | 0 | = 0 | pass |
| P7 thought_open, tool trials | 2/200 | 0/200 | ≤ A0 + 4 = 6 | pass |
| P8 total: Q12 positive shown | 129/160 | 109/160 | ≥ A0 − 16/160 × 160 = 113 | **FAIL** (short by 4) |
| P8 per case: every protected case (A0 ≥ 24/40: all four, A0 27, 30, 35, 37) stays ≥ 24/40 | — | 28, 35, 35, **11** (omelette_steps 37/40 → 11/40) | ≥ 24/40 on each | **FAIL** |
| P9 surfaces where none belongs | 0/80 | 0/80 | ≤ A0 + 1 = 1 | pass |
| P9 surfaces on tool trials | 0/200 | 0/200 | ≤ A0 + 2 = 2 | pass |
| P10 placement failures, Q12 positive | 8/160 | 12/160 | ≤ A0 + 8/160 × 160 = 16 | pass |
| P11 routed tool calls on Q12 | 0/175 | 0/175 | ≤ A0 + 10/175 × 175 = 10 | pass |
| void (≥ 9 of 440 errored) | no (0/440) | no (0/440) | | |
| **passes** | | | | **no** (P8, both clauses) |

Notes on the table:
- A0's two open thoughts are type-10 r3 and search-10 r4: each call sits inside a thought that never closed, so both
  trials are unrouted. A0's third unrouted tool trial, key-01 r3, answered "Ctrl Z" in speech without calling the tool.
- The P2 denominators differ (99 and 100) because A0 left key-01 r3 unrouted.
- The placement failures are all of one kind, as in T-0098c: every one, on both arms, is a `thought` commit that
  holds `<a2ui>`. No `ui` commit was flagged `thought_open`.

#### Per positive case: Q12 surface shown, k / 40

| case | A0 | C3 | protected (A0 ≥ 24/40) |
|---|---|---|---|
| flight_booking | 27/40 | 28/40 | yes |
| flight_booking_with_memory | 30/40 | 35/40 | yes |
| primes_table | 35/40 | 35/40 | yes |
| omelette_steps | 37/40 | **11/40** | yes |
| total | 129/160 | 109/160 | |

Both arms showed 0/15 on the Q12 negatives, 0/80 where no surface belongs (of which 0/65 on the no-tool and plain
trials) and 0/200 on the tool trials. Every-ui-valid over gated trials, reported and not gated: A0 87/153, C3 58/128.
`ui.gate` gated / emitted / failed: A0 376 / 250 / 126, C3 359 / 218 / 141.

#### Q12 positive shown by repeat block (reported, not gated)

| arm | r 0–4 | r 5–39 |
|---|---|---|
| A0 | 17/20 | 112/140 |
| C3 | 13/20 | 96/140 |

The r 0–4 block is the one whose design matches T-0098c's five repeats. Per case in that block: A0 3, 4, 5, 5 and C3
3, 5, 5, 0 of 5 (flight_booking, flight_booking_with_memory, primes_table, omelette_steps).

#### Outcome

**C3 not confirmed: fails P8.** The run is complete, neither arm is void, and R1 and R3 hold, so this is the rule's
verdict and not an inconclusive run. C3 fails both P8 clauses: 109/160 against a bar of 113, and omelette_steps 11/40
against 24/40. It passes P1–P7 and P9–P11. `winner` is "no change".

What follows, under H-T103 (f): **C3 is retired and the live prompt stays** (A0, sha `3605c012…`). T-0098d does not
ship C3 and is not started. There is **no second T-0103 run**, at this sha or any other, without a new human ruling.

#### Observations (inference, not findings)

*Inference.* One run, one boot, no seed, the dev split only; trials of one case share one prompt and the prefix cache,
so they are not independent, and the effective n per case is probably below 40.

- **C3's P8 loss sits in one case.** On omelette_steps C3 showed 11/40 against A0's 37/40. On the other three it is
  at or above A0: flight_booking 28 vs 27, flight_booking_with_memory 35 vs 30, primes_table 35 vs 35. Without
  omelette_steps the three-case totals are C3 98/120 against A0 92/120. The total clause fails only because of this
  one case: the 26-trial gap on omelette_steps is larger than the 20-trial gap in the totals.
- **What C3 did on omelette_steps instead of a surface.** Counted from the commits of the 29 C3 trials that showed
  nothing:
  - no `ui` commit at all: **29/29**;
  - a `ui` commit the gate rejected: 0/29;
  - a `thought` holding `<a2ui>`: 0/29.

  Every one closed its thought. 25 of the 29 spoke the steps aloud (each contains 'First'; none is a numbered list).
  The other 4 (r 4, 9, 23, 31; 17–23 words) spoke only an offer to walk through it, with no steps and no surface.
  Across the 29: 22 as one `answer` commit, 7 as two, a median of 98 words (17 to 127). No answer contains `<a2ui`
  or `{`. So the failure is not placement or validity: in 25 of 29 C3 put the steps in speech instead of a surface
  for a step-by-step request. A0's 3 misses on this case are like C3's 4 offers: no `ui` commit, a 21–24-word answer
  announcing steps it does not give. C3's 11 shown trials fall at r 5, 6, 10, 12, 15, 26, 28 and 34–37,
  and r 0–4 showed 0/5.
- **C3's speech gain reproduced.** `type_at` spoke on 47/50 (10/10 type cases in a majority) against A0's 4/49 (0/10),
  and click + scroll + key on 100/100 against 70/99. Every C3 tool trial was routed (200/200).
- **Against T-0098c, side by side and not pooled** (T-0098c at `3ac00a1`, n = 5 per case; T-0103 at `6720e56`):

  | | T-0098c A0 | T-0098c C3 | T-0103 A0 | T-0103 C3 |
  |---|---|---|---|---|
  | `type_at` spoken | 8/50 | 47/50 | 4/49 | 47/50 |
  | click + scroll + key spoken | 73/98 | 97/98 | 70/99 | 100/100 |
  | Q12 positive shown | 15/20 | 12/20 | 129/160 (r 0–4: 17/20) | 109/160 (r 0–4: 13/20) |
  | omelette_steps shown | 4/5 | 2/5 | 37/40 (r 0–4: 5/5) | 11/40 (r 0–4: 0/5) |
  | placement failures, Q12 positive | 1/20 | 1/20 | 8/160 | 12/160 |

  Both runs read the same direction on every row except the placement failures, which were level in T-0098c. The speech gain is about as large in both. The omelette_steps drop
  that T-0098c showed at n = 5 (4/5 → 2/5) is larger at n = 40 (37/40 → 11/40), so it was not a one-trial miss of the
  kind that run's write-up could not tell from spread. The other three cases, which T-0098c read as mixed (C3 3, 3, 4
  against A0 4, 4, 3), read here as level or better for C3. Why moving the tool paragraph ahead of the UI block makes
  this one request spoken instead of shown is not established by this run.

## Required fields (T-0108a)

**Status: pre-registered (H-T108a, (a)–(i) approved as drafted by the human 2026-10-16); not run yet.** This part is
the pre-registration, committed before the `/launch`. The results will be appended below it after the run, and this
part will not be edited then. Nothing above this section is changed by T-0108a. Task:
[`docs/tasks/T-0108a.md`](../tasks/T-0108a.md).

Under the T-0108 gate most flight-style cards reach the user with the last call OK, but with their interactive
components pruned. The model writes a TextField with `placeholder` and no `label`, a Button with `text` instead of
`child` + `action`, a DateTimeInput or ChoicePicker with no `value`, and ChoicePicker options as strings. The live UI
block names only Column, Row and Text. T-0108a runs A0 against two candidate UI blocks, K1 and K2, that name the
required fields compactly. The primary metric is the **complete-card rate** on the Q12 positive cases. Nothing in the
product changes in this task: a selected K ships in T-0108d.

### Baseline (CPU re-gate, not a run of this task)

T-0108's `ui_reading` loop with the definition below, at `ca61f08`, over stored trials (deterministic, n = 1 per
trial): A0 of the T-0103 run `6720e56` is complete on **63/160** Q12 positive trials (flight_booking 0/40,
flight_booking_with_memory 0/40, omelette_steps 27/40, primes_table 36/40), with the last call OK on 149/160. The A0
trials that prune something prune a TextField in 75, a Button in 72, a DateTimeInput in 64 and a ChoicePicker in 40.
C3 of the same run is complete on 32/160; A0 of the T-0098c run `3ac00a1` on 5/20 (flights 0/10). The planner's full
table is in the task's *Evidence*.

### Regime (pre-registered)

T-0103's regime verbatim, except for the arms and the counts:

| | |
|---|---|
| engine | profile `lean`; vLLM `v0.29.0`; DiffusionGemma NVFP4 rev `2ea8372`; chat template `36e3a42e`; tokenizer `cc8d3a0c`; CL 256, 48 steps; thinking on; prefix caching on; `max_num_seqs 3` |
| device, boot | RTX 5090, driver 595.84, gpu-mem 0.86; one fresh engine-only boot via `scripts/locked/t0108a_ui_fields_ab.sh` (T-0103's boot regime verbatim: `services/vllm.sh` unchanged since `806da12`, port and container refusals, proxies unset) |
| tools | `native:bbfc53ccdd80`, all tools on every arm; repair on (unused: pass 1 only) |
| cases | T-0098c's 60 with the same order and indices: 0–52 T-0098's 53 (40 tool, 8 no-tool, 5 plain), 53–56 the Q12 positives, 57–59 the Q12 negatives. Pins unchanged: `cases.jsonl` `460cd1c3…`, `surface_placement_cases.jsonl` `e152810b…`; dev split only |
| arms | A0, K1, K2 (below) |
| n | 40 repeats on indices 53–56, 5 on the other 56: 4 × 40 × 3 + 56 × 5 × 3 = **1320 requests**, 440 per arm |
| seed | none; concurrency 1; no prefill, no repair, no continuation |
| input | typed text, no `Session`, no audio, no assists; as T-0098c (Q12 cases no frame; the memory case its captions as hits, no `now_ms`) |
| UI gate | `UiGate` (T-0108) and the vendored A2UI v0.9.1 schema at the run commit; the manifest records the sha256 of `local_astra/conductor/ui_gate.py`, `local_astra/ui/bracket_repair.py` and `local_astra/ui/a2ui_schema.py` |
| out-dir | `logs/t0108a-ui-fields-ab-<sha7>`; request ids `t0108a-r{repeat}-{case_id}-{arm}`; aux `logs/t0108a-launch-<sha7>` |

Estimated duration (from the `3ac00a1` per-kind means, not measured): about 663 s of requests (about 11 min), about
13.5 min with the boot.

### Arms

Every arm is `"\n".join([P, CT, SS, U′, N])`; only U′ differs. K1 and K2 are built from U by exact single-occurrence
replacements, and the harness refuses the run if any anchor occurs 0 or ≥ 2 times or a U′ exceeds 2000 code points.

- **E1** (shared; makes room) replaces "Copy that catalogId exactly, use the same surfaceId in every message, give the
  tree a component / with id "root", and define every id you reference in children in the same fence." with "Copy
  that catalogId exactly, keep one surfaceId, give one component id "root" and define every / id you reference in the
  same fence." (`/` marks the line break).
- **F1** (two lines): `Required: TextField label; DateTimeInput value ""; ChoicePicker options
  [{"label":"A","value":"a"}]` / `and value []; Button child (a Text id) and action {"event":{"name":"go"}}.`
- **F2**: `Text variant: h1-h5, caption or body; children are ids, never objects.`

| arm | U′ | U′ chars / sha256 | text chars / sha256 (refused if it drifted) | role |
|---|---|---|---|---|
| A0 | U, the live block | 1870 / `4360995b0136…` | 2744 / `3605c012a0fecf70738cb4917f28ca53e3b5aa69caf18ae291a34c0fbab79e57` | control |
| K1 | U with E1, then F1 after "flashcard is a Card." | 1999 / `50f7d7424777…` | 2873 / `22a2ba7e9a022268fb41f847fa3cfad1e0a123f5c323cb120c63d7526fac0702` | candidate |
| K2 | U with E1, then the two-line "There is no table, chart, maths or flashcard component: …" sentence replaced by "A table is a Column of Rows of Text." + F1 + F2 | 1991 / `71765248036…` | 2865 / `23a34c3656547eebaf2f89c65cc6a0a04367f8dcd93924d3bb4b8be79ced687c` | candidate |

The CPU token estimates are 500 (U), 535 (K1) and 535 (K2), under the 600 ceiling of T-0044 5b. The manifest records
each U′'s engine token count from the server's `/tokenize` before the first request.

### Design

- **Order.** Repeats are outermost (r = 0…39). Within a repeat the cases scheduled for it run in index order: all 60
  for r ≤ 4, only 53–56 for r ≥ 5. Within each (case, repeat) the three arms run back to back from offset
  `(case_index + repeat) mod 3` of `A0 K1 K2`; A0 is first 13, 14, 13 and 13 times of 40 on the four positive cases.
- **Refusals** (exit 2, before any socket): every T-0103 refusal (loopback, dirty tree, non-empty out-dir, a split
  other than dev, the case pins, the tool kit, the Q12 pin and its 4 + 3 shape); an A0 text that is not `3605c012…`;
  a K1 or K2 text whose sha is not the one above; an anchor found 0 or ≥ 2 times; a U′ over 2000 code points; a
  repeats map other than 40 on the four positives and 5 on the other 56.
- **Per-arm denominators.** 160 Q12 positive trials (40 per case); 175 Q12 trials; 200 tool trials; 65 no-tool +
  plain trials; 80 trials where no surface belongs; 440 trials in all.

### Definitions (H-T108a (a))

On each Q12 positive trial, a fresh `UiGate()` gates every `ui` commit in commit order with `final=False` and the
request id as `turn_id` (as `session._gate_surface` does), and the same gate is then read on the **last** call:

- **last-OK**: the last call returned a surface;
- **pruned**: the ids that are schema-invalid among the components of `gate._repair(raw).messages` (after salvage),
  together with `gate._repair(raw).dropped` (what salvage threw away). `_repair` is private and is read, never
  changed;
- **complete**: last-OK and nothing pruned. A trial with no `ui` commit is not complete.

Reported, not counted: forward-reference prunes (an id the last canvas references and does not define); the pruned
kinds per arm and per case (trials pruning each kind); and an as-written-clean rate (complete, and the canvas parsed
as written with no fixer edit of any kind). The as-written rate is stricter than the task's "salvage off" baseline
(49/160), which still let the structural repair run. Shown, placement failure, routed, valid, exact, thought_open and
anomaly are as in T-0098c's [Definitions](#definitions).

### Decision rule (H-T108a (c)–(f); exact fractions; each K against A0 in the same run)

| | condition | bar at this run's denominators |
|---|---|---|
| Q1 (primary) | K complete ≥ A0 complete + 3/20 × 160 | **≥ A0 + 24** (`24/160`) |
| Q2 (primary) | a positive case is *protected* iff A0 is complete on ≥ 3/5 of its 40; on each protected case K complete ≥ 3/5 | protected iff A0 ≥ **24/40**; fails at K ≤ 23/40 |
| U1 | K last-OK ≥ A0 − 1/10 × 160, with the same per-case clause on last-OK | **≥ A0 − 16** (`16/160`); `24/40` |
| U2 | T-0103's P8: K shown ≥ A0 − 1/10 × 160, with the same per-case clause on shown | **≥ A0 − 16** (`16/160`); `24/40` |
| P3, P4, P5, P7 | verbatim from T-0098 (recall, valid/exact, false tools over 65, thought open over 200) | unchanged |
| P6 | voiced anomalies over the arm's 440 trials | **= 0** |
| P9 | verbatim: surfaces on the 80 trials where none belongs ≤ A0 + 1, and on the 200 tool trials ≤ A0 + 2 | `1/80`, `2/200` |
| P10 | placement failures on the Q12 positives ≤ A0 + 1/20 × 160 (reading S) | **≤ A0 + 8** (`8/160`) |
| P11 | routed calls on the 175 Q12 trials ≤ A0 + 2/35 × 175 (reading S) | **≤ A0 + 10** (`10/175`) |
| R3 (control) | A0 Q12 positive shown ≥ 7/10 × 160 | **≥ 112** (`112/160`) |
| R4 (control: the degradation reproduced) | A0 complete ≤ 3/5 × 160 | **≤ 96** (`96/160`) |
| void | an arm with > 2 % errored trials | ≥ 9 of 440 |

The protected sets are A0's, never K's. P1, P2 and R1 are dropped per D16 (docs/01; H-T104): the spoken-prefix rate
is reported and never gated.

**Outcome** (exactly one):
- **"K\<i\> selected"** iff the run is complete, no arm is void, R3 and R4 hold, and K\<i\> passes Q1, Q2, U1, U2,
  P3–P7 and P9–P11. If both pass, the higher complete count wins; a tie goes to K1. Then T-0108d ships that U′.
- **"no change: …"**, naming every failing condition of each K, iff the run is complete, nothing is void, R3 and R4
  hold, and no K passes. The UI block stays and T-0108b (resample) is next.
- **"inconclusive: \<reason\>"** otherwise: an incomplete run, a void arm, R3 failing ("UI baseline not
  reproduced") or R4 failing ("degradation not reproduced"). Nothing is selected and the result goes back to the
  human.

Only a selection ships anything (`winner` is the selected K, otherwise `"no change"`). There is no second T-0108a run
without a new human ruling.

### What the report carries

T-0103's per-arm keys for A0, K1 and K2, plus: `q12_positive_complete` and `q12_positive_last_ok` (k/160); per
positive case, complete, last-OK and shown as k/40 per arm, with the Q2, U1 and U2 protected flags; the pruned kinds
per arm and per case, the forward-reference count and the as-written-clean rate; each threshold as k/n beside the
value it is compared with; the speech rates under "reported, not gated (D16)"; each U′'s chars, sha and engine token
count; `outcome`, its reasons and `winner`. The gpu eval row `tests/eval/test_ui_fields_ab.py` (`T0108A_RUN_DIR`)
first compares the manifest's gate-file shas with its working tree and skips when they differ ("re-gates with the live
UiGate; reproduces only at the run commit"); otherwise it re-derives the schedule, the complete-card reading, every
condition, R3, R4, the void arms and the outcome from `trials.jsonl` alone and checks them against `report.json`.

After the run this section gains: the regime as run, the decision table with raw k/n, the per-case tables, the pruned
kinds before and after, and the outcome.

### Results (T-0108a)

Every number below comes from the run's own files, read-only, in the detached launch worktree
`.claude/worktrees/t0108a-launch/logs/t0108a-ui-fields-ab-e6c3a81/`:
- `trials.jsonl`, sha256 `bd27cc28…`;
- `report.json`, sha256 `97a85910…`;
- `manifest.json`, sha256 `af1810b4…`;
- `report.md`, sha256 `a6fc89ec…`;
- the launch log `…/logs/t0108a-ui-fields-ab-e6c3a81-run.log` (sha256 `09721a70…`).

The trial and report shas match the ones the launch log printed. The write-up re-derived every figure quoted here
from `trials.jsonl` alone, with its own aggregation. It called only the library primitives the definitions name:
`UiGate` (including the private `_repair`, read-only), `drop_special_literals`, `native.parse_call` and
`toolcall.score`. The re-gate used the live `UiGate` of the write-up's worktree. Its three gate files hash to the
manifest's `gate_files_sha256` (`ui_gate.py` `635328ad…`, `bracket_repair.py` `623429b0…`, `a2ui_schema.py`
`9c87b9a1…`), and nothing under `local_astra/`, `services/` or `tests/` changed between `e6c3a81` and the
write-up's base (`b4d3ed6`). The re-derivation also rebuilt the schedule from the manifest's case indices and repeats
map, and it matches the file line for line (1320 lines, every (case, arm, repeat) key once). Every figure agrees with
`report.json` and `report.md`, per arm, per case and per kind, and **no disagreement was found**.

Some regime entries are pins, not run outputs:
- the model rev `2ea8372` and chat template `36e3a42e` come from `services/versions.lock` at `e6c3a81` (sha256
  `5bab0570…`, as the manifest records);
- gpu-mem 0.86, `max_num_seqs 3` and prefix caching on come from `services/vllm.sh` at `e6c3a81` (no diff from
  `806da12`);
- the tokenizer `cc8d3a0c` is `TOKENIZER_SHA256` in `local_astra/toolcall/detok.py:50`, which the host detokenizer
  checks on load; the engine's own copy was not hashed.

Each rate is given as the raw k/n behind it. Spoken, valid and exact are over routed tool trials.

#### Regime as run

| | |
|---|---|
| commit | `e6c3a8135c4f33ec74bfece1e338d33828999951` (clean; the harness refuses a dirty tree), from the detached worktree `.claude/worktrees/t0108a-launch` |
| table, rule | `T-0108a`, `H-T108a`; `reading`: complete card (H-T108a (a)): the last gated call returned a surface and nothing is pruned on it (schema-invalid after salvage, or dropped by salvage, read through `UiGate._repair`); forward-reference prunes reported, not counted; U2, P9, P10 and P11 as T-0103 (reading S); P1, P2 and R1 dropped (D16) |
| engine | `base_url` `http://127.0.0.1:8001/v1`; `served_model` `nvidia/diffusiongemma-26B-A4B-it-NVFP4` (rev `2ea8372`); `engine_version` `0.29.0` (from `/version`); chat template `36e3a42e`; tokenizer `cc8d3a0c`; `versions_lock_sha256` `5bab0570…` |
| decoding | `profile` `lean`; `canvas_length` 256; `max_denoising_steps` 48; `thinking` true; prefix caching on; `max_num_seqs 3` |
| device, boot | RTX 5090, driver 595.84, gpu-mem 0.86; one fresh engine-only boot via `scripts/locked/t0108a_ui_fields_ab.sh`. Before the boot: 380 MiB in use, 1-min load 0.65. After the boot: 27323 MiB used / 4764 MiB free. At the end: 27322 / 4765 |
| tools | `tools` `native:bbfc53ccdd80` on every arm; `repair` on (unused); `pass`: pass 1 only, so no prefill, no repair and no continuation; `LlmOptions()` as the session sends it |
| cases | `split` dev; `cases` 60, T-0098c's order and indices. `cases_sha256` `460cd1c3…` (dev split of `tests/eval/toolcall/cases.jsonl`), plus T-0093 J10 `PLAIN_QUESTIONS`, plus `q12_sha256` `e152810b…` (`tests/eval/surface_placement_cases.jsonl`, J1). That is 40 tool, 8 no-tool, 5 plain and 7 Q12 (4 positive, 3 negative). The test split was not read |
| input | typed text, with no `Session`, no audio and no assists. Tool cases are sent over their fixture frame. No-tool, plain and Q12 cases have no frame and carry the `[screen]` note. `flight_booking_with_memory` carries its captions as memory hits, with no `now_ms` |
| `arms` | A0 K1 K2. System-text sha256 / chars (manifest): A0 `3605c012a0fe…` / 2744 (the live text); K1 `22a2ba7e9a02…` / 2873; K2 `23a34c365654…` / 2865. Every trial's `system_sha256` is its arm's |
| U′ (`u_prime`) | A0 1870 chars, sha `4360995b0136…`, **500 engine tokens**; K1 1999 chars, `50f7d7424777…`, **535**; K2 1991 chars, `71765248036…`, **535** (root `/tokenize` before the first request; the CPU estimates were 500 / 535 / 535; ceiling 600) |
| `repeats` | 40 on flight_booking, flight_booking_with_memory, primes_table and omelette_steps (indices 53–56); 5 on the other 56 |
| `schedule`, `rotation` | repeats outermost; within a repeat, the cases scheduled for it in index order; per (case, repeat) the three arms back to back from offset (case_index + repeat) mod 3 of `A0 K1 K2`. A0 ran first 13, 14, 13 and 13 times of 40 on the four positive cases |
| n | `requests` 1320 = 4 × 40 × 3 + 56 × 5 × 3, 440 per arm. Per arm: n = 50 trials per tool (25 for `scroll` and `press_key`), 200 tool trials, 65 no-tool + plain, **160 Q12 positive (40 per case)**, 15 Q12 negative, 80 where no surface belongs, 175 Q12 |
| `seed` | none (`null`). Repeats are the only replication, and the run is not reproducible |
| `concurrency` | 1 |
| `thresholds` | `Q1_margin` `24/160`; `protected_min` `24/40`; `U1_margin` `16/160`; `U2_margin` `16/160`; `P9_no_belong_over_A0` `1/80`; `P9_tool_over_A0` `2/200`; `P10_margin` `8/160`; `P11_margin` `10/175`; `R3_min` `112/160`; `R4_max` `96/160` |
| `surface_shown` | per trial, a fresh `UiGate()` gates every `ui` commit in commit order, with `final=False` and the request id as `turn_id` (`session._gate_surface`). Shown means at least one call returned a surface |
| UI gate | `UiGate` (T-0108) + vendored A2UI v0.9.1 schema at `e6c3a81`; `gate_files_sha256` as above, unchanged at `b4d3ed6` |
| timing | boot 2026-10-17 02:35:09 (+01:00); engine healthy 02:37:29; harness 02:37:29–02:49:08 (1320 requests in 11 min 39 s, including start-up and report writing; the summed request wall time is 677.6 s, a mean of 0.513 s, against the estimate of about 663 s; Q12 positive means A0 0.989 s, K1 0.966 s, K2 0.998 s); gpu eval row 02:49:08–02:49:26; engine stopped 02:49:27 with 0 containers left |
| outcome | 1320/1320 trials (`trials_written` 1320), **0 errored**, no void arm, `complete: true`. Harness exit 0. The gpu eval row (`tests/eval/test_ui_fields_ab.py`, 2 passed) re-derived the outcome from `trials.jsonl` with the run commit's gate and agreed with `report.json`. Script exit 0 |

#### Controls

- **R3 holds.** A0 showed a Q12 positive surface on 146/160 (≥ 112/160).
- **R4 holds.** A0 was complete on 54/160 (≤ 96/160), so the degradation the task targets was reproduced. (The CPU
  re-gate of the T-0103 run read 63/160; this run's A0 is lower on omelette_steps, 16/40 against 27/40.)

#### Decision table (H-T108a (c)–(f), each K against A0 in this run)

| | A0 (reference) | K1 | K2 | bar | K1 | K2 |
|---|---|---|---|---|---|---|
| Q1 complete, Q12 positive | 54/160 | 64/160 | 66/160 | ≥ A0 + 24/160 × 160 = 78 | **FAIL** (short by 14) | **FAIL** (short by 12) |
| Q2 per protected case (A0 ≥ 24/40: primes_table only, A0 37/40) | — | 36/40 | 40/40 | ≥ 24/40 | pass | pass |
| U1 last-OK, Q12 positive | 142/160 | 143/160 | 137/160 | ≥ A0 − 16 = 126 | pass | pass |
| U1 per case (A0 ≥ 24/40 on all four: 34, 38, 37, 33) | — | 35, 32, 36, 40 | 35, 34, 40, 28 | ≥ 24/40 on each | pass | pass |
| U2 shown, Q12 positive | 146/160 | 152/160 | 139/160 | ≥ A0 − 16 = 130 | pass | pass |
| U2 per case (A0 ≥ 24/40 on all four: 35, 38, 40, 33) | — | 36, 36, 40, 40 | 36, 35, 40, 28 | ≥ 24/40 on each | pass | pass |
| P3 pooled route recall | 199/200 (0.995) | 198/200 (0.990) | **194/200 (0.970)** | ≥ A0 − 0.02 = 39/40 = 195/200 | pass | **FAIL** (short by 1) |
| P3 `type_at` route recall | 50/50 | 50/50 | 50/50 | ≥ A0 − 0.04 = 48/50 | pass | pass |
| P4 valid | 197/199 (0.990) | 197/198 (0.995) | 194/194 (1.000) | ≥ A0 − 0.02 = 9651/9950 ≈ 0.970 | pass | pass |
| P4 exact | 175/199 (0.879) | 176/198 (0.889) | 173/194 (0.892) | ≥ A0 − 0.03 = 16903/19900 ≈ 0.849 | pass | pass |
| P5 false tools (no-tool + plain) | 0/65 | 0/65 | 0/65 | ≤ 5 and ≤ A0 + 2 = 2 | pass | pass |
| P6 voiced anomalies (all 440 trials) | 0 | 0 | 0 | = 0 | pass | pass |
| P7 thought_open, tool trials | 0/200 | 0/200 | 0/200 | ≤ A0 + 4 = 4 | pass | pass |
| P9 surfaces where none belongs | 0/80 | 0/80 | 0/80 | ≤ A0 + 1 = 1 | pass | pass |
| P9 surfaces on tool trials | 0/200 | 0/200 | 1/200 | ≤ A0 + 2 = 2 | pass | pass |
| P10 placement failures, Q12 positive | 14/160 | 17/160 | 21/160 | ≤ A0 + 8 = 22 | pass | pass |
| P11 routed tool calls on Q12 | 0/175 | 0/175 | 0/175 | ≤ A0 + 10 = 10 | pass | pass |
| void (≥ 9 of 440 errored) | no (0/440) | no (0/440) | no (0/440) | | | |
| **passes** | | | | | **no** (Q1) | **no** (Q1, P3) |

Reported, not gated (D16): spoken before a tool call A0 99/199, K1 103/198, K2 91/194; `type_at` spoken A0 4/50,
K1 6/50, K2 4/50. All arms showed 0/15 on the Q12 negatives. `ui.gate` gated / emitted / failed: A0 413 / 357 / 56,
K1 391 / 332 / 59, K2 372 / 326 / 46.

#### Complete cards (Q12 positive trials)

| arm | complete | last-OK | shown | as written, clean (reported) | last canvas holds a forward ref (reported) |
|---|---|---|---|---|---|
| A0 | 54/160 | 142/160 | 146/160 | 19/160 | 68/160 |
| K1 | 64/160 | 143/160 | 152/160 | 21/160 | 79/160 |
| K2 | 66/160 | 137/160 | 139/160 | 35/160 | 73/160 |

"As written, clean" is complete with no fixer edit of any kind on the last canvas (`_repair(raw).edits == 0`).

Per positive case, k / 40, as complete · last-OK · shown:

| case | A0 | K1 | K2 | protected for Q2 / U1 / U2 |
|---|---|---|---|---|
| flight_booking | 1 · 34 · 35 | 1 · 35 · 36 | 2 · 35 · 36 | no / yes / yes |
| flight_booking_with_memory | 0 · 38 · 38 | 2 · 32 · 36 | 0 · 34 · 35 | no / yes / yes |
| primes_table | 37 · 37 · 40 | 36 · 36 · 40 | 40 · 40 · 40 | yes / yes / yes |
| omelette_steps | 16 · 33 · 33 | 25 · 40 · 40 | 24 · 28 · 28 | no / yes / yes |
| total | 54 · 142 · 146 | 64 · 143 · 152 | 66 · 137 · 139 | |

#### Pruned kinds, before and after salvage (trials of 160 pruning each kind; reported, not gated)

"Before" is the last canvas as parsed (after the structural repair, before the schema-guided salvage): trials with at
least one schema-invalid component of that kind. "After" is the report's reading: kinds still schema-invalid after
salvage, with everything salvage dropped in one bucket, "(dropped by salvage)". The third row resolves that bucket to
the dropped components' own kinds.

| arm | | Button | TextField | DateTimeInput | ChoicePicker | Row | List | Card | Column | Text | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | before | 66 | 72 | 69 | 39 | 2 | 23 | 4 | — | — | — |
| A0 | after (report; + 45 dropped by salvage) | 30 | 30 | 30 | 15 | 1 | 14 | 1 | — | — | — |
| A0 | pruned, by own kind | 64 | 72 | 69 | 39 | 1 | 14 | 4 | — | — | — |
| K1 | before | 69 | 2 | 1 | 28 | 8 | 31 | 2 | 3 | 3 | — |
| K1 | after (report; + 56 dropped by salvage) | 14 | 2 | 1 | 28 | 8 | 12 | — | 2 | — | — |
| K1 | pruned, by own kind | 68 | 2 | 1 | 28 | 8 | 12 | 2 | 2 | — | — |
| K2 | before | 64 | 4 | 2 | 11 | 11 | 4 | 4 | 2 | 4 | 1 |
| K2 | after (report; + 52 dropped by salvage) | 14 | 3 | 2 | 11 | 11 | 1 | 2 | 2 | 3 | — |
| K2 | pruned, by own kind | 63 | 4 | 2 | 11 | 11 | 1 | 4 | 2 | 3 | 1 |

Per case (the report's "after" reading): primes_table prunes nothing on any arm; omelette_steps prunes List on A0 14,
K1 12, K2 1 (plus a Card or Column and 2 salvage drops per arm); every other prune is on the two flight cases.

#### Outcome

**No change: K1 fails Q1; K2 fails Q1, P3.** The run is complete, no arm is void, and R3 and R4 hold, so this is the
rule's verdict and not an inconclusive run. K1 gained 10 complete cards and K2 12, against a bar of 24. K2 also missed
P3's pooled route recall by one trial. Both pass Q2, U1, U2, P4–P7 and P9–P11. `winner` is "no change".

What follows, under H-T108a (i): **the live UI block stays** (A0, sha `3605c012…`; U sha `4360995b0136…`). T-0108d
does not start. **T-0108b (resample) is next.** There is no second T-0108a run without a new human ruling.

#### Observations (inference, not findings)

*Inference.* One run, one boot, no seed, the dev split only. Trials of one case share one prompt and the prefix
cache, so they are not independent, and the effective n per case is probably below 40. The counts below come from
the last `ui` commit of each Q12 positive trial, read with the live gate. Components are classified by what the
model wrote, before salvage.

**1. What the "(dropped by salvage)" components on flight cards are.** The label is not a defect class. Salvage runs
only on a canvas that did not parse as written. On a canvas that parsed, the same defects are reported under the
component's own kind. On the two flight cases (80 trials per arm), a salvage drop happened in A0 43, K1 54 and K2 50
trials. Every dropped component has an id, a `component` field and a known kind, except one K2 kind (below). Each is
dropped for a missing or wrong required property:

| arm | dropped components (trials) | what they are |
|---|---|---|
| A0 | 194 (43) | TextField with no `label`: with `placeholder` 86 (41), with `text` 3 (2); DateTimeInput with no `value` 45 (39), simply missing it (no other error); Button with `text` and no `child` or `action` 34 (34); ChoicePicker with string options and no `value` 20 (19); other ChoicePicker (no options or no value) 5 (5); Card with no `child` 1 (1) |
| K1 | 54 (54) | all Buttons with `action` and no `child`: `children` holding an inline Text object 34 (34); `children` holding an id list 17 (17); `child` that is not an id string (object or list) 3 (3) |
| K2 | 52 (50) | Button `children` id list 34 (34); Button `children` inline object 12 (12); Button `child` not a string 3 (3); TextField `value: []` 2 (1); an unknown kind 1 (1): the component name holds leaked reasoning (`"TextFieldPicker - wait, the prompt says "TextField label; DateTimeInput value …"`) |

The other shapes in the question are rare:
- **Missing `component`.** No top-level component lacks it. The objects with no `component` sit inline in a Button's
  `children` or `child`, such as `{"id":"btn_text"}`: K1 3 and K2 3 of the inline objects in Buttons (as parsed, flight
  cases), against 45 and 18 that are complete Text objects. A0 writes no inline Button child.
- **Duplicate ids.** Definition: an id carried by two or more objects in the last canvas. That covers the parsed
  component tree, inline `children`/`child` objects included, before salvage. It also covers the raw text, for the one
  case where the parse keeps only one copy.

  On that definition, duplicates occur in 6 K1 flight canvases (r5-fb, r7-fb, r17-fbm, r23-fb, r28-fb, r36-fbm), in 4
  K2 ones (r0-fb, r12-fbm, r19-fbm, r38-fbm) and in no A0 one. Every duplicated id except K2 r19's is `btn_text`.
  - **Inline in the Button and again at top level:** K1 r5, r23, r28 and K2 r38.
  - **Both copies inline in the same `children` list:** K2 r0, K2 r12 and K1 r7. In each, an id-only stub
    `{"id":"btn_text"}` is followed by the full Text.
  - **The Button itself has id `btn_text`, as does its Text:** K1 r17 and r36.
  - **K2 r19:** the leaked "TextFieldPicker…" component is redefined as a ChoicePicker under the same id
    (`passengers`). This one shows only in the raw text, because the parse keeps the first object.
- **Wrong props.** These make up the whole A0 bucket: `placeholder` for `label`, `text` for `child`, strings for
  option objects.

**2. What K1/K2 fixed and what remains** (trials of 160 pruning each kind, by own kind):
- **Fixed by naming the field: TextField 72 → 2 / 4, DateTimeInput 69 → 1 / 2.** F1 says "TextField label;
  DateTimeInput value """, and the model now writes exactly that: 148 (K1) and 150 (K2) `"value":""` literals in the
  last Q12-positive canvases, against 0 in A0.
- **ChoicePicker options fixed, `value` not: 39 → 28 / 11.** String options vanish: 33 A0 flight trials, 0 on either
  K. But the pickers now carry a scalar `value` instead of an array. That is 28 of K1's 28 invalid ChoicePickers and
  11 of K2's 11 (as parsed: `"eco"` K1 20 / K2 6, `"1"` 6 / 4, `""` 2 / 0, `"economy"` 0 / 1). F1's `value []` was
  read as "a value", not "an array".
- **Button unchanged in count, changed in form: 64 → 68 / 63.** The A0 Button defect is `text` with no child or
  action (all 64). The K Buttons carry `action` and a Text, but put it in `children`, the field every container in
  the block uses, instead of the single `child` F1 names. K1 mostly inlines the Text object (44 trials inline, 20 id
  list, 4 a non-string `child`). K2's F2 line ("children are ids, never objects") turns most of these into id lists
  (19 inline, 41 id list, 3 non-string), still under `children`. F2 does fix List: List with inline children falls
  from 14 (A0) and 12 (K1) to 1 (K2), which is K2's gain on omelette_steps.
- **Row is new: 1 → 8 / 11, a side effect of the `""` literal.** Every K Row prune is a merge artefact. The model
  writes the empty string as a lone `"`, as in `"value":"]},{"id":"dest",…` (38 such literals in K1's last canvases
  and 40 in K2's, in 18 and 17 trials; none in A0). The string then swallows the component boundary, and the repair
  attaches the TextField's `label` and `value` to the next Row (`Unevaluated properties … ('label', 'value')`). Every
  pruned K Row carries a `value` string holding swallowed component text. All 14 of K1's pruned Rows (8 trials) start
  with `}` or `]}`. 22 of K2's 23 pruned Rows (covering all 11 trials) start with `}` or `]}`; the 23rd, in
  r27-flight_booking, starts with `:id":"pass_row`. The remaining K TextField/DateTimeInput prunes are not merge
  artefacts:
  - stray `x`/`y` props (K1 1 trial, K2 2);
  - `placeholder` (1 trial each);
  - on K2, also `value: []` (1 trial, r6-flight_booking, dropped by salvage).
- **What binds flights now.** Of the last-OK flight trials, the only pruned kinds are Buttons in K1 34/67 and K2
  40/69; Buttons and ChoicePickers in 55/67 and 47/69; Buttons, ChoicePickers and merged Rows in 62/67 and 57/69. A0
  has 1/72 complete, with every remaining trial pruning at least one input field.

**3. Why K2 failed P3.** K2 left 6 of 200 tool trials unrouted. A0 left 1 (key-01 r2, which answered "The shortcut for
undo is Ctrl Z." in speech) and K1 2 (key-01 r3, the same answer; key-04 r2, which asked which display name to set).
K2's six:
- **key-01 "Press the keyboard shortcut for undo."**, r3 and r4: K2 said "The keyboard shortcut for undo is Ctrl
  Z." and called no tool (A0's failure mode, twice).
- **key-04 "Close this browser tab."**, r1 and r4: the thought concludes "the user didn't ask anything" or "I should
  describe what I see". K2 then asks which display name to set (r1) or describes the settings page (r4).
- **scroll-05 "Scroll down the message list to see older emails."**, r4: the thought ends "I should use the `scroll`
  tool". The answer is "I will scroll down the message list to see older emails.", and no call follows.
- **click-05 "Start writing a new email."**, r3: K2 described two emails from Lena Fischer in speech and drew a list
  surface instead of clicking compose. This is K2's one surface on a tool trial (P9, 1/200, within its bar).

So `press_key` routed 21/25 on K2, against 24/25 on A0 and 23/25 on K1; `scroll` 24/25 and `click_at` 49/50 on K2.
Every `type_at` and `google_search` trial was routed on every arm. Four of the six misses are on the two `press_key`
cases that K1 also misses once each (A0 misses key-01 once and key-04 never). A 1-trial shortfall (194 against a bar
of 195) is the size of change that 5-trial cases can make. Whether K2's shorter "no table" sentence or F2 caused it is not established by this run.
