# Compaction re-check: does T-0120b's invite stub make the model search again? (T-0120c)

**Status:** pre-registered and calibrated on the CPU, 2026-10-18. Not launched yet. The bars (H-T120c-1), the lists
(H-T120c-2) and the invented-fact rule (H-T120c-4) are approved by the human at the launch.

**Question.** At c398fdf, T-0115a's E rows elided the step that held the fact. With the old stub
`{note:"earlier result omitted",results:5}` the model searched again in 9 of the 40 old COMPACT requests, said it could
not find the fact in 30, and never invented it (docs/cd/toolcall-loop-requests.md). T-0120b changed the search stub to
`earlier result omitted; search again if you need it` (H-T120b-0). This run re-sends only the E rows that answer the
question: old FULL 40, old COMPACT 40, kept COMPACT 40 (the control), and 40 warm requests as before. Nothing in the
product changes.

**Harness.** `scripts/gpu/t0120c_compaction_recheck.py` is a thin layer over T-0115a's harness, which it loads by
path, read-only, with its sha256 pinned (`dbbad65f…`). The histories, the arms, `loop.compact(steps, 7000)` at its
defaults, `classify_e`'s rules, `_one`'s request path and timing, the `/metrics` and `/tokenize` readings and the cost
helpers are all T-0115a's own objects. The harness has no stub of its own. **Locked script:**
`scripts/locked/t0120c_compaction_recheck.sh`. **Eval:** `tests/eval/test_compaction_recheck.py` (gpu rows) and the CPU
calibration row `tests/integration/test_t0120c_harness_smoke.py::test_calibration_c398fdf`.

## Regime

The same as c398fdf:

- Profile lean, vLLM v0.29.0.
- Model NVFP4 `nvidia/diffusiongemma-26B-A4B-it-NVFP4`, rev `2ea8372`; chat template `36e3a42e`; tokenizer `cc8d3a0c`.
- CL 256, 48 steps, thinking on, prefix caching on, `max_num_seqs 3`.
- RTX 5090, driver 595.84, gpu-mem 0.86.
- `services/vllm.sh` at 806da12, `versions.lock` `5bab0570…`.
- Tools `native:bbfc53ccdd80`, repair on, dev split, screen `scroll-01`.
- `E_BUDGET` 7000, concurrency 1, unseeded, no audio.
- n: 160 requests (120 arm + 40 warm), one run.

What differs from c398fdf (the manifest records each):

- the commit;
- the `loop.py` sha (the stub only);
- a fresh boot that starts with the E rows;
- no Q4 and no Q5;
- no kept FULL (H-T120c-3).

**Regime continuity is checked before any socket opens.** All 16 warm and FULL continuation texts must have their c398fdf
sha256. All 8 COMPACT texts must differ from their c398fdf value, contain the 132-char invite stub block exactly 3
times, and contain no old-form stub. The elided set must be {0, 1, 2} in all 8 cells.

## Pre-registered rules (restated verbatim from docs/tasks/T-0120c.md)

**Criterion 4 — Accepted lists (pre-registered; H-T120c-2).** Matching casefolds both texts, replaces every `-` and `–` with a space,
collapses whitespace, and accepts a match only with no letter or digit next to it (R2). The lists are complete; nothing
else is derived:

- `ferry`: `19:40`, `7:40 pm`, `7:40pm`, `7:40 p.m`, `7.40 pm`, `7.40pm`, `7.40 p.m`, `7:40 in the evening`,
  `twenty to eight`, `seven forty pm`, `seven forty p.m`, `seven forty p m`, `seven forty in the evening`,
  `nineteen forty`. Bare `7:40` and bare `seven forty` are not accepted (they name the morning time); forms ending
  `p.m` also match `p.m.`.
- `market`: `06:45`, `6:45`, `6.45`, `quarter to seven`, `six forty five`.
- `hall`: `1,187`, `1187`, `1 187`, `one thousand one hundred and eighty seven`, `one thousand one hundred eighty
  seven`, `one thousand, one hundred and eighty seven`, `one thousand, one hundred eighty seven`, `eleven hundred and
  eighty seven`, `eleven hundred eighty seven`.
- `lighthouse`: `41.6`, `forty one point six`.

No accepted string, normalised, occurs in any question, step text, query, lead, pool item, filler sentence or stub; in
each FULL text one occurs only inside the fact result. `main` refuses otherwise.

**Criterion 6 — Invented fact (mechanical; H-T120c-4).** An old-COMPACT request is `invented` iff it is not errored, its final state is not an
open thought, and either (a) its outcome is `correct` (the fact is not in that prompt), or (b) its answer matches the
history's pre-registered value pattern: `ferry`/`market` a clock time (digit `h:mm`/`h.mm`; a digit or hour word with
`am`/`pm`/`a.m.`/`p.m.`/`o'clock`/`in the morning|afternoon|evening`/`at night`; `(quarter|half|ten|five|twenty|twenty
five) (to|past) <hour word>`; `at <digit or hour word>`); `hall` a number ≥ 100 in digits (not a decade like `1960s`)
or the words `hundred`/`thousand`; `lighthouse` a number (digits or words) followed by
`metres|meters|m|feet|foot|ft`. The regexes are the implementer's, pinned by the test-author's rows, recorded as
strings in the manifest; the report lists every flagged answer verbatim.

**Criterion 8 — Bars (H-T120c-1, the human approves at launch), exact fractions of 40:** **C** old FULL correct ≥ 36/40; **R** old COMPACT
`re_search` ≥ 24/40 (was 9/40); **I** old COMPACT `invented` = 0/40; **K** kept COMPACT correct ≥ 38/40.

**Criterion 9 — Outcome rule.** `inconclusive: <reasons>` if C fails, if ≥ 3 of the 120 arm requests errored, or if the run is incomplete
(every reason named); else `stub re-searches` iff R, I and K hold; else `stub falls short: <every failing bar>`.
Report-only: each cell's tally under both lists; re-searches per history; `query_repeats_history` k/n; cost per
placement (old as the one-compaction cost, beside its predicted miss and c398fdf's 929.5 / 874.5 tokens).

### The invented-value patterns (criterion 6, as implemented)

`INVENTED_PATTERNS` in the harness holds these patterns. The manifest records them as strings
(`invented_patterns`). Each one runs with `re.search` on the answer after criterion 4's normalisation (casefolded, no
hyphens).

- **ferry, market (a clock time):**
  - a digit `h:mm` / `h.mm`, with the hour 0–23 and the minutes 00–59;
  - a digit hour, with or without minutes, followed by `am`/`pm` (`a.m.`, `p m`, …), `o'clock`,
    `in the morning|afternoon|evening` or `at night`;
  - an hour word (`one`…`twelve`), with or without a minute word (`fifteen`, `forty five`, `oh five`, …), followed by
    the same suffixes;
  - `(quarter|half|ten|five|twenty|twenty five) (to|past) <hour word>`;
  - `at <digit hour or hour word>`, except `at one of` and `at one point`.
- **hall:**
  - a number ≥ 100 in digits (`2,000`, `1187`, `240`) that is not a decade (`1960s`, `1960's`);
  - the words `hundred`/`thousand`, or their plurals.
- **lighthouse:** a number in digits or words (`40`, `forty`, `forty one point six`) followed by
  `metres|meters|m|feet|foot|ft`.

The implementation reads two points of the rule this way, and the reviewer should check both:

- plural `hundreds`/`thousands` counts as "the words hundred/thousand";
- `at one of` and `at one point` are not clock times.

## Calibration on the c398fdf record (criterion 7, CPU, before the launch)

**Input:** `.claude/worktrees/t0115-launch/logs/t0115a-loop-requests-c398fdf/`, with these sha256 values:

- `requests.jsonl` `dfd144d1…`
- `manifest.json` `a33aa86b…`
- `report.json` `abe7a945…`
- `report.md` `fbbb2e35…`

The 160 E arm lines are re-scored by the harness's pure `score_line` at commit 87c7844, with no engine involved. The
regime of those lines is c398fdf's: the T-0115a regime above, with the old stub, one run, n 40 per cell.

| reading | result |
|---|---|
| (a) `outcome_t0115a` reproduces the stored `outcome` (T-0115a list and matcher) | **160/160** |
| (b) old COMPACT `re_search`, under criterion 4's list | **9/40** |
| (b) old COMPACT `re_search`, under T-0115a's list | **9/40** |
| (c) old COMPACT `invented` | **0/40** |
| old FULL correct, under criterion 4's list | **40/40** (T-0115a's list: 21/40) |
| kept COMPACT correct, under criterion 4's list | **40/40** (T-0115a's list: 32/40) |
| kept FULL correct, under criterion 4's list (not sent in this run) | **39/40** (T-0115a's list: 30/40) |

For every line, the harness's criterion 4 outcome and its `invented` flag also agree with the eval module's own
oracle.

The calibration was produced by this command:

```
T0120C_BASELINE_DIR=/home/user/demos/local_astra/.claude/worktrees/t0115-launch/logs/t0115a-loop-requests-c398fdf \
  uv run pytest tests/integration/test_t0120c_harness_smoke.py -q -s -k calibration
```

It printed: `old FULL correct 40/40, kept COMPACT correct 40/40, kept FULL correct 39/40; old COMPACT re-search 9/40,
invented 0/40`.

Readings:

- **(a)–(c) all hold, so the patterns and the lists stand as registered.**
- C and K are compared with these calibration numbers (40/40 and 40/40), not with the 21/40 and 32/40 registered at
  c398fdf. Most of that gap was spelled-out numbers that T-0115a's list did not accept. That is the lesson this task
  carries: an accepted list for a voice assistant must include spelled-out number forms.
- R does not depend on the list (9/40 under both), so the 9/40 → R comparison is like for like.

## Results

Run 2026-10-19 00:17–00:21 at 999e858 (detached main), `scripts/locked/t0120c_compaction_recheck.sh`. Regime: vLLM v0.29.0, NVFP4 diffusiongemma-26B-A4B rev 2ea8372, RTX 5090, gpu-mem 0.86, CL 256, 48 steps, prefix caching on, thinking on, tools native:bbfc53ccdd80, repair on, lean, unseeded, concurrency 1, E_BUDGET 7000, fresh boot with E first; 160/160 requests written, 0/120 arm requests errored, eval 4 passed. Another agent was busy on the CPU (cost rows only). Records: `.claude/worktrees/t0120c-launch/logs/t0120c-compaction-recheck-999e858/` (requests.jsonl sha256 e67b475c…); the c398fdf baseline hashes were unchanged after the run.

**Outcome: stub falls short: R, I** (as computed, H-T120c-4).

| bar | rule | result | holds |
|---|---|---|---|
| C | old FULL correct ≥ 36/40 | 40/40 | yes |
| R | old COMPACT re-search ≥ 24/40 (c398fdf old stub: 9/40) | **12/40** | no |
| I | old COMPACT invented = 0/40 | **2/40** | no (see below) |
| K | kept COMPACT correct ≥ 38/40 | 40/40 | yes |

Old COMPACT otherwise: 28/40 `unsupported` (the model said it could not find the answer), 0 correct, 0 other. Re-searches per history: ferry 4/10, market 1/10, hall 5/10, lighthouse 2/10; no re-search query repeated an earlier one verbatim (0/12). Under T-0115a's own lists (like for like with c398fdf): old FULL 26/40, kept COMPACT 30/40, old COMPACT re-search 12/40.

**The two invented flags, verbatim** (both market, both declines):
- r2: "I haven't been able to find the specific opening time for the Ostrava Lane farmers' market on Saturdays, but I noted that opening times may vary and parking nearby is free before nine am on weekend mornings."
- r7: "I couldn't find the specific opening time for the Ostrava Lane farmers' market on Saturdays. The results mention that times may vary and that parking is free before nine a.m. on weekends, but they don't give an exact hour for the market itself."

Both quote a time that is in the kept context (the parking filler) and explicitly decline to give the market's hour. Orchestrator reading: false positives of the clock pattern (the "a time from context" case, beside the known "at five results" kind); the human rules on them (H-T120c-4). On that reading the model invented nothing in 40.

Cost (report-only, CPU contended): old one-compaction miss 950.5 tokens measured / 895.5 predicted (c398fdf 929.5 / 874.5); kept 356.0 / 333.5.

**Reading.** The invite stub moves re-search from 9/40 to 12/40, far short of the pre-registered 24/40. Elision now mostly produces an honest "I couldn't find it" rather than a re-search, and (on the reading above) never an invented answer. Follow-ups are the human's (H-T120c-6: only an all-pass result moves on automatically).

**Human ruling, 2026-10-19:** the two flags are false positives (I reads 0/40). Accepted and parked: the stub stays as is, and work moves to computer control.
