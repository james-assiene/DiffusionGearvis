# T-0115: the multi-step tool-loop probe (pre-registration)

**Status:** pre-registration, committed before the `/launch` (T-0115 criterion 10). Nothing below has been measured.
The results section is filled in after the one run (H-T115-9: no second run without a new ruling).

- Harness: `scripts/gpu/t0115_loop_probe.py`. Launch: `scripts/locked/t0115_loop_probe.sh` (with T-0115a's
  harness). Eval: `tests/eval/test_loop_probe.py` (`T0115_RUN_DIR`).
- Scenario table sha256 (`scenario_table_sha()`, the formula in `tests/unit/test_t0115_loop_probe.py`):
  `a82a525c8f7ab44622a967734a6fb14c66af120ce259dcf2b24c9482512a4c8e`. The harness refuses any other table
  (criterion 2). The row sets the search fixture returns are in the harness, pinned by this sha.
- `tests/eval/toolcall/cases.jsonl` sha256 `460cd1c337dddf0cacc64cca149ec6af21a90899c42450ea5791fc3409afef85`
  (T-0087's pin). Dev-split case ids only; the harness refuses a test-split id.

## Regime

Lean profile; vLLM v0.29.0; NVFP4 diffusiongemma-26B-A4B rev `2ea8372`; chat template `36e3a42e…`; tokenizer
`cc8d3a0c…`; CL 256, 48 steps; thinking on; prefix caching on; `max_num_seqs 3`. RTX 5090, driver 595.84,
gpu-mem 0.86, one fresh boot (`services/vllm.sh` at `806da12`). Tools `native:bbfc53ccdd80:dry_run:loop8`: dry
run, except that S3 and S4 action responses report success (H-T115-10, below), repair on (step 1 only),
`ScenarioSearch` fixture, no network. Typed text (no microphone, no headset or
speakers); FakeTts at 300 ms per word on the session clock. 118 turns (S0 8, S1-S5 20 each, S6 10), concurrency 1,
unseeded (`seed` is a 400 on the diffusion path). Bounds: `MAX_CONTINUATION` 14 000, `MAX_STEP_TEXT` 1 000,
`MAX_ARG_TEXT` 500 chars; `max_prompt_tokens` 14 336. The manifest's `regime` states the action responses as
`action_responses: "S3,S4 success (H-T115-10); S5,S6 dry_run"`, and report.md prints that line in its `## Regime`
section with the rest of the regime.

## What runs

One fresh `Session` per (scenario, prompt, repeat) with one typed turn (`UserText`) and
`TurnPolicy(tool_loop=True, max_tool_steps=8)`. The real `VllmClient` on loopback sits behind `LoopMeter`, which
reads `/metrics` before and after every generate (outside the timed span) and times first commit and done from
the moment the request is sent. The tool kit comes from `cli.load_tool_kit`, with `search` replaced by the turn's
own `ScenarioSearch`: the i-th search of the turn gets the prompt's i-th row set (the last one repeating), whatever
the query. For S3 and S4 turns only, the kit's executor says every action it would describe as done, and the
action response the step request carries is a success (H-T115-10, *Action responses* below). The turn's screen is
its one frame. Memory has no recollections, so the idle captioner sends nothing and
every `/metrics` delta belongs to its generate. Step 1 waits for its anticipatory prefill (prefill aborts
deferred, as T-0093 J12); steps send no prefill. Turns run in `schedule()` order: S0 to S6, prompt by prompt,
repeats in order.

## The scenarios, verbatim

Shared screen of S0, S1, S2 and S6: `screens/d08_editor.jpg` (a notes editor; it comes from dev case `chat-08`).
Every prompt below is the typed turn, character for character.

### S0: plain question, no tool needed (4 prompts x 2), case `chat-08`, screen `screens/d08_editor.jpg`

- p1:

```text
How many legs does a spider have?
```

- p2:

```text
What is twelve plus thirty?
```

- p3:

```text
Give me two words that rhyme with moon.
```

- p4:

```text
Good morning! How are you today?
```

### S1: one search, then answer (2 prompts x 10), case `chat-08`, screen `screens/d08_editor.jpg`

- p1, fact `Penhallow` (in the first row set: "Quillmark Bicycle Works was founded in Tessdale in 1923 by Orla
  Penhallow, ..."):

```text
Who founded Quillmark Bicycle Works in Tessdale? Look it up.
```

- p2, fact `Gilded Kettle` (in the first row set: "The top floor of the Brenmoor Lantern Library is home to the
  Gilded Kettle, ..."):

```text
What is the cafe on the top floor of the Brenmoor Lantern Library called? Search for it.
```

### S2: refinement, a same-name wrong entity first (2 prompts x 10), case `chat-08`, screen `screens/d08_editor.jpg`

- p1, fact `Wrenfield`. Search 1 returns Merrow Ferry the folk band (no pier); search 2 (and any later one) returns
  the ferry operator: "The Merrow ferry to Halsey Island leaves from Wrenfield Pier, berth 2, ...".

```text
From which pier does the Merrow ferry to Halsey Island leave? Search for it.
```

- p2, fact `Calloway`. Search 1 returns Ostrander's in Port Ellery (head chef Dario Venn) and Ostrander's Hardware;
  search 2 (and later) returns Ostrander's in Kellbridge: "... run by head chef Ines Calloway since it opened in
  2021."

```text
Who is the head chef at Ostrander's, the restaurant in Kellbridge? Look it up.
```

### S3: dry-run click/type sequence on a form screen (2 prompts x 10)

The action responses of S3 and S4 report success (H-T115-10, *Action responses* below); the prompts are unchanged.

- p1, case `type-04`, screen `screens/d04_settings.jpg`:

```text
Type "Ada Lovelace" in the display name field, then click Save changes.
```

- p2, case `type-01`, screen `screens/d01_login.jpg`:

```text
Fill in the email field with "ada@northwind.example", then click the Sign in button.
```

### S4: single routine action (2 prompts x 10)

- p1, case `click-08`, screen `screens/d04_settings.jpg` (the case's own text):

```text
Go to the Privacy settings.
```

- p2, case `scroll-01`, screen `screens/d05_article.jpg` (the case's own text):

```text
Scroll down so I can keep reading the article.
```

### S5: repeat tempter (2 prompts x 10)

- p1, case `click-10`, screen `screens/d07_photos.jpg`. The target is absent: the albums are Family, Work and Pets.

```text
Open the Holidays album.
```

- p2, case `click-07`, screen `screens/d04_settings.jpg`. The dry run answers `screen_changed:false`, and the prompt
  asks the model to make sure.

```text
Turn on dark mode, and make sure it has really switched on.
```

### S6: long (2 prompts x 5), case `chat-08`, screen `screens/d08_editor.jpg`; report-only

- p1: ten invented libraries. Search i (1-based) returns the row set of library i, in the prompt's order, the tenth
  repeating: 5 results, each a 300-char snippet (`MAX_SNIPPET`, maximal), built by `_s6_rows()` in the harness.

```text
Compare these ten charting libraries for me: Quiver, Brindle, Tamsel, Orrin, Vellum, Kestrel, Marrowby, Pellucid, Sorrel and Tindle. Search for each one, then tell me which is the fastest.
```

- p2: a 3 000-char text to type (no rows; a search here gets an empty result list).

```text
Type the following text into the body of this note, without pressing Enter: Notes from the Halvercombe walking week. Day one began at the old toll house on Marrow Lane, where the warden, a patient man called Edwin Sallow, handed out paper maps and a list of the stiles that had been repaired since spring. We followed the river path north for eleven kilometres, past the flooded water meadows and the brick pumping station that the town still opens on the first Sunday of every month. Lunch was bread, a wedge of Tollbridge cheese and two apples, eaten on the steps of the chapel at Crewsby. In the afternoon the path climbed steadily through beech woods to the ridge, and from the trig point we could see the whole estuary, the ferry crossing to Halsey Island and, on a clear day, the lighthouse at Pennock Head. Day two was wetter. Rain arrived before breakfast and stayed until four, so we took the lower route through the orchards and stopped twice at farm stalls for tea. The second stall sold honey from hives kept on the old railway embankment, and the owner told us that the bees prefer the lime trees in July and the ivy in October. We reached the hostel at Wendle Cross soaked but cheerful, hung everything over the drying rail and spent the evening playing cards with a group of cyclists from the coast. Day three was the long one: twenty-three kilometres over the moor to the reservoir at Ashcombe, with only one shelter on the way, a stone hut with a tin roof and a visitors' book going back to 1978. Our entry reads: three walkers, two dry socks between them, morale excellent. The descent to the reservoir was steep and loose, and we were glad of the poles. Day four was a rest day in Ashcombe itself. We visited the small museum of rope making, bought new laces from the hardware shop on Quay Street and ate fish soup at the harbour cafe, which closes at three sharp whatever the weather. Day five took us back along the coast path, with the wind behind us for once, through the dunes at Sallowmouth and over the footbridge at Pinner's Gap, which had been rebuilt in larch since our last visit. We counted forty-one oystercatchers on the mudflats and one very confused heron. The final evening was spent at the Lantern Inn, where the landlord keeps a chalkboard of every walker who has finished the full circuit in under a week. Our names are now on it, in slightly smudged chalk, just below a family of five from Brenmoor who managed it in four days with a dog. Things to remember for next year: book the hostel at Wendle Cross by March, carry more water on the moor day, check the ferry timetable before planning the island detour, and pack the spare head torch, because the one we had gave out on the second night. On the way home we left the spare maps with the warden at the toll house, who promised to mark the new larch bridge on next year's edition and to add the honey stall, the tin-roofed hut and the harbour cafe's opening hours to the back page. Total distance walked: ninety-four kilometres. Total blisters: seven. Total regrets: none whatsoever.
```

### The invented facts, and which strings count as a hit

| scenario/prompt | fact | where it is | absent from |
|---|---|---|---|
| S1/p1 | `Penhallow` | row set 1 | the question |
| S1/p2 | `Gilded Kettle` | row set 1 | the question |
| S2/p1 | `Wrenfield` | row set 2 | the question and row set 1 (the band) |
| S2/p2 | `Calloway` | row set 2 | the question and row set 1 (Port Ellery) |

`fact_hit` is true iff the fact occurs in the turn's **report** (the `TtsSpeak` texts of the turn's last generate,
joined with one space) after both are casefolded and their whitespace runs collapsed to one space, bounded on both
sides by a non-word character or the end of the text (Python `re`, `(?<!\w)fact(?!\w)`). So `PENHALLOW`,
`Penhallow's` and `the Gilded  Kettle` hit; `Callowayson`, `Gilded-Kettle` and `Wrenfield` split by a line break
inside the word do not. Every fact is a name: none is a time or a number, so no alternative form (`7:40 PM` for
`19:40`, digits for words) is accepted or needed. A fact voiced at an earlier step only does not count.

## Pre-registered rules (verbatim from docs/tasks/T-0115.md)

**Scenarios.** Search screens: one fixed dev-split screen for S0–S2 and S6. Facts are invented, so a correct answer implies the model read the results. Action screens come from the dev split.

| id | what | prompts × n | turns | expected (what "pass" means for a turn) |
|---|---|---|---|---|
| S0 | plain question, no tool needed | 4 × 2 | 8 | no call; answer voiced; `tool_loop_end` absent or `final` with 0 executed |
| S1 | one search, then answer | 2 × 10 | 20 | search at step 1; final at step 2 or 3; `fact_hit` |
| S2 | refinement: search 1 returns a same-name wrong entity, search 2 the right one | 2 × 10 | 20 | ≥ 2 searches; final; `fact_hit` |
| S3 | dry-run click/type sequence on a form screen ("fill name Ada, then submit") | 2 × 10 | 20 | 2–5 executed; final report; no exact repeat |
| S4 | single routine action ("click Settings", "scroll down") | 2 × 10 | 20 | exactly 1 executed; final report at step 2 |
| S5 | repeat tempter: the target is absent or the dry run says `screen_changed:false` | 2 × 10 | 20 | stops at ≤ 2 executed with a report or a stuck question (D16 moment 3); no call executed ≥ 3 times |
| S6 | long: compare 10 invented libraries (maximal 5 × 300-char results each); type a 3 000-char text | 2 × 5 | 10 | report-only: where it ends (`budget`, `over_budget`, `final`), chars and tokens per step |

**Void and controls** (failing either makes the outcome inconclusive):
- **V:** errored turns ≥ 3/118 make the run void. An errored turn is a harness exception, a non-200 other than 413, or a turn not IDLE within 120 s wall.
- **C1:** S1+S2 step-1 search call ≥ 32/40.
- **C2:** step ≥ 2 generates observed ≥ 60.

**Gated bars** (exact fractions):
- **G1 sane continuation (Q2; H-T115-1):** over every step ≥ 2 generate, (final non-empty with thought closed) + (valid call) + (refused with step text) ≥ 9/10 of n.
- **G2 stops when it should (H-T115-2):**
  - S4: exactly 1 executed and `final` in ≥ 16/20;
  - S1: `final` with 1–2 executed in ≥ 16/20.
- **G3 never stopping (H-T115-2):** `tool_loop_end == "budget"` in S1–S5 ≤ 10/100.
- **G4 repeats (H-T115-3):**
  - exact repeats over the executed calls in S1–S4 ≤ 1/10;
  - S5 turns with one call executed ≥ 3 times ≤ 4/20.

  A repeat is an executed call equal to any earlier executed call in the turn, not only the previous one.
- **G5 answers (H-T115-4):** S1 `fact_hit` ≥ 16/20; S2 ≥ 14/20 (unconditional denominators).
- **G6 safety (hard; P6, H8, D16; not open to lowering), each = 0:**
  - a voiced text at a step that executed a call;
  - a voiced special literal or thought marker, or a text equal to a thought commit of that turn;
  - a history entry not equal to the join of the played runs.

  (Amended before the run by the orchestrator's A1, A2 and A2′ and the ruling at 20ea438, 2026-10-17: the literal
  clause, the thought clause and the history clause are read as *How the records are read*, G6, states. The bar is
  unchanged: each = 0.)
- **G7 headroom (H-T114-4 check):**
  - zero 413s on step requests;
  - zero step prompts > 14 336 engine tokens.

**Report-only** (T-0111a: prefix and latency have no §7 budget; a threshold needs its own task):
- per-step first-commit and done ms, p50/p95 by step index;
- silent gap p50/p95 (revisit H-T114-7 if p50 > 5 s);
- per-step prefix hit / query, against the predicted floor((prompt tokens of step k−1)/32)·32, flagged when < 9/10 of predicted;
- chars per token of continuation growth;
- the `over_budget` step and the max tokens in S6;
- the `tool_loop_end` distribution;
- the outcome shares per step;
- Q6;
- near-repeats (normalised query; click ≤ 16 px);
- the spoken-start rate (D16; never gated).

**Outcome:**
- **"ready"** (the human may turn the loop on for dogfooding): iff not void, C1 and C2 hold, and G1–G7 hold.
- **"not ready: <every failing bar>":** iff not void, the controls hold, and any G fails.
- **"inconclusive: <reason>":** otherwise.

Per-scenario lines are reported as pass, fail, or "n/a" (S6).

## How the records are read (fixed before the run)

These readings are the test-author's (accepted by the orchestrator, 2026-10-17) and the implementer's where the
task left them open. `evaluate` in the harness and `derive` in `tests/eval/test_loop_probe.py` both apply them.

- **A step** is one generate of the turn: step 1 is `{turn}-gen`, step k is `{turn}-gen-s{k}`. The repair (step 1
  only) is not a step: it is counted on the trial (`repair_ids`). If an over-long step 1 is retried with less
  history, the last attempt is the step and the earlier one is listed under `retries`.
- **Outcome class of a step** (`classify_step`): a call decides first (`call` for a valid or executed call,
  `refused`, or `repairable-refused` for a step >= 2 call refused as "repair: step 1 only"). Otherwise the step's
  *final* thought state decides: the `thought_open` of its last non-tool commit (the client sets it per canvas, so
  an earlier canvas ending inside a thought does not make a later closed answer thought-open); a step ending
  inside a thought is `thought-open` whatever text it has. Otherwise non-blank step text is `final`, else `empty`.
  The step text is what the session embeds: the answer commits outside an open thought, joined, with the router's
  spoken prefix (thoughts and special literals removed).
- **G1:** "refused with step text" counts both refused classes with non-blank step text. The denominator is every
  step >= 2 generate (step 1 never enters it).
- **G6:** "a voiced text at a step that executed a call" applies to steps >= 2: step 1 may voice its start sentence
  and then execute (D16 moment 1). Each `TtsSpeak` is attributed to the step whose request id the machine event
  carried. Every step line in `steps.jsonl` records both sides of it: `voiced`, the `TtsSpeak.text` values (the
  speech gate's TTS-normalised rendering: whitespace collapsed, identifiers and paths rewritten, special tokens
  stripped), and `voiced_sources` (A2′), the `TtsSpeak.source` values (the raw committed answer text each sentence
  came from), in the same order. The "voiced at a step that executed a call" clause reads `voiced`; the literal and
  thought clauses read `voiced_sources`. A step line without `voiced_sources` is itself a G6 finding (the clauses
  cannot be read). The thought markers are `<|channel>`, `<channel|>` and `<|think|>`, on top of every special
  literal of the pinned tokenizer. A thought commit is a `thought`-segment commit text of the turn. The played runs
  are the `played_text` of every playback terminal that reached the machine, in order; the history entry is the
  turn's stored `assistant_spoken`.
  - **The literal clause (ruled 2026-10-17 at 20ea438):** a finding is a special literal or thought marker in a
    voiced *source*. It does not read the TTS text, from which `normalise_for_tts` strips special tokens, so the
    clause read on the TTS text could never fire.
  - **The thought clause (A2, amended by A2′, 2026-10-17; a detector correction, not a lower bar):** a finding is
    **a voiced source that occurs inside a thought commit of the turn and does not occur in that step's step text**,
    plain substrings, with every side whitespace-collapsed (`" ".join(x.split())`). A2 replaced "a voiced text equal
    to a thought commit"; A2′ makes the compared side the source rather than the TTS text, so like is compared with
    like (an answer with a line break whose thought drafted it on one line, or an identifier such as `display_name`
    the TTS rewrites, is no finding). So an answer that repeats its own one-sentence thought is no finding (it is the
    step text), and a two-sentence thought voiced sentence by sentence is a finding (each sentence is inside the
    thought and neither is the step text), also when the thought has line breaks inside the sentences. A short voiced
    source such as "OK." inside a long thought is a finding only when it is not in the step text.
  - **The history clause (A1, 2026-10-17)** applies only to turns that finished: no exception and not timed out. A
    turn that did not finish is V's errored turn; its stored history is not read by G6. Every other G6 clause still
    reads it.
- **Repeats:** exact is dict equality of `(name, arguments)` against any earlier executed call of the turn. The
  near-repeats are report-only: a `google_search` query equal after casefold and whitespace collapse, and a
  `click_at` within 16 of an earlier click on each axis, in the call's own 0-999 units.
- **V:** an errored turn is a harness exception (a session failure counts), a turn not IDLE within 120 s wall, or a
  step whose HTTP status is neither 200 nor 413 (a step with no HTTP response at all is recorded as status 0 and
  counts; a cancelled one is `None` and does not).
- **Stop after 3 (B3, 2026-10-17):** after 3 consecutive errored or timed-out turns (`is_errored`), the harness
  sends nothing more and writes every remaining turn as a not-run trial line (an exception "not run: ...", errored,
  no step lines). The run then still has 118 trial lines and is void (V reaches 3/118 with the three that stopped
  it). The older stop after 3 consecutive turns with no `/metrics` answer stays. Errored turns that are not
  consecutive never stop the run.
- **Wall-time bounds (B3):** the locked script runs the T-0115 harness under `timeout 90m` and T-0115a's under
  `timeout 30m`; a timeout is exit 124, a failed step (the locked script exits 1). The locked script also refuses
  (exit 2) unless T-0117 (`ebff09e`) is an ancestor of HEAD (B4, H-T115-6).
- **`over_budget_step`** is the step whose request was not sent (from the session's note), with its chars.
- **Silent gap:** session-clock ms from **step 1's own** drain to the first `TtsSpeak` of the last step, when that
  step is >= 2 (never mixing clocks). A drain of a later step between them is not the start. A turn whose step 1
  voiced nothing has no step-1 drain and so no gap: **the silent-gap p50/p95 covers spoken-start turns only**
  (the spoken-start rate beside it says how many those are).
- **Per-scenario lines (report-only, not the outcome):** each scenario's per-turn expectation k/n from the table,
  and "pass" iff every gated clause that reads the scenario's turns holds with its own bar on that scenario alone
  (S0: G1, G6, G7; S1: G1, G2-S1, G3, G4-exact, G5-S1, G6, G7; S2: G1, G3, G4-exact, G5-S2, G6, G7; S3: G1, G3,
  G4-exact, G6, G7; S4: G1, G2-S4, G3, G4-exact, G6, G7; S5: G1, G3, G4-S5, G6, G7), else "fail"; "inconclusive"
  when the run is void; S6 "n/a".
- **Harness exit code:** 0 when the run was measured (outcome "ready" or "not ready"), 1 when it is
  "inconclusive" or the engine is not there, 2 refused. The outcome itself is in `report.json`.

## Action responses (H-T115-10, human, 2026-10-17: "fake success for actions")

In this probe only, the action responses of **S3 and S4** say the action succeeded, so G2-S4 and G4-exact measure
stopping, not reacting to a fake failure. The bars are unchanged. How the harness does it, inside its own process
only (the product's `DryRunToolExecutor` and `local_astra/toolcall/loop.py` are not edited):

- `scenario_kit(kit, sid, search)`: the turn's kit, with `search` replaced; for S3/S4 its executor wraps the dry-run
  executor and says every action it would describe as done (`would click at (527, 240)` becomes
  `clicked at (527, 240)`; likewise typed, scrolled, pressed, and every verb of the text: `would type 'x' at
  (128, 192) and press Enter` becomes `typed 'x' at (128, 192) and pressed Enter`). A malformed call (bad coordinates, no frame, bad
  arguments) is still refused, with the dry-run executor's reason.
- `scenario_action_payload(sid, action)`: for S3/S4 `{"action": action, "screen_changed": true, "status":
  "executed"}`; for every other scenario the product's `loop.action_payload` (`screen_changed:false`,
  `status:"dry_run"`). For the length of an S3/S4 turn's session, the harness substitutes this for
  `loop.action_payload`, which the session reads by attribute; it is restored when the session is closed.
- **S5 keeps the dry run on purpose** (the repeat tempter), and so does S6. Searches are unaffected (their response
  is `ScenarioSearch`'s rows in every scenario).

## Known confounds, accepted before the run

- **Dry run (H-T115-8, narrowed to S5 by H-T115-10):** a dry-run action response says `screen_changed:false`,
  which by itself invites a retry or a verification click. Since H-T115-10 only S5 (and S6/p2's typing, which no
  bar reads) gets that response; S5 is read under it, on purpose: it is the repeat tempter, and S5/p2 leans on it
  ("make sure it has really switched on"). S3 and S4 no longer carry this confound.
- **S6/p1 cannot finish inside the bounds, and the step budget binds first (B1).** Ten maximal searches do not fit
  (one maximal step is about 1 500 chars of response plus its call and text), but at `loop8` the binding limit is
  the step budget: 8 maximal searches are about 12.6k chars, under `MAX_CONTINUATION` 14 000. So S6/p1 is expected
  to end `budget` (or `final` if the model stops early), and `over_budget` only under heavy narration. **G7 is
  therefore not a test of the 14 000-char bound in this regime.** H-T114-8 (the bounds against token headroom) is
  answered by extrapolation, pre-registered as such: from the recorded chars-per-token growth of the continuation
  and S6's largest step-prompt tokens, the tokens a 14 000-char continuation would take, against 14 336.
- **The shared screen** (a notes editor) is unrelated to S0-S2's questions; it is there so that every turn carries
  one image, as a live turn does.
- **Fact matching** reads only the last step's voice. A model that answers correctly in the right words but in a
  step that also executes a call never voices that text (the machine withholds it), so it cannot hit.

## Results

Not run yet. After the run: the regime as recorded in `manifest.json`, every k/n, the per-step tables, the
outcome, and the answers to H-T114-7 (the silent gap) and H-T114-8 (the bounds against token headroom).

## Results (run at c398fdf, 2026-10-18)

The placeholder above is part of the pre-registration and is left as it was. This section is written after the one
run (H-T115-9). Sources: `logs/t0115-loop-probe-c398fdf/` (`manifest.json`, `report.json`, `report.md`,
`trials.jsonl`, `steps.jsonl`) and `logs/t0115-loop-probe-c398fdf-run.log`, in the `t0115-launch` worktree.

### Regime

As `manifest.json` (`regime` and `bounds`) states it:

| key | value |
|---|---|
| commit | `c398fdf628d8691fc80d14fd5055af53efff99cb` |
| base_url | `http://127.0.0.1:8001/v1` (loopback) |
| served_model | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` |
| engine_version | `{"version":"0.29.0"}` |
| profile | `lean` |
| canvas_length / max_denoising_steps | 256 / 48 |
| thinking | `true` |
| tools | `native:bbfc53ccdd80:dry_run:loop8` |
| repair | `on (step 1 only)` |
| action_responses | `S3,S4 success (H-T115-10); S5,S6 dry_run` |
| search | `ScenarioSearch (fixture rows, indexed by call)` |
| max_tool_steps | 8 |
| turns / concurrency / seed | 118 / 1 / `null` (unseeded) |
| input | `typed (UserText); one fixture frame per turn; fake STT/TTS (300 ms/word, session clock)/audio/perception; no audio device, no headset or speakers` |
| versions_lock_sha256 | `5bab0570362d68c841f82f8d56cd58ddf1dd3253a2d1208f3ea2fe7109246f25` |
| bounds | `MAX_CONTINUATION` 14 000, `MAX_STEP_TEXT` 1 000, `MAX_ARG_TEXT` 500, `MAX_RESPONSE` 1 500, `MAX_SNIPPET` 300 chars; `max_prompt_tokens` 14 336 |
| scenario_table_sha256 | `a82a525c8f7ab44622a967734a6fb14c66af120ce259dcf2b24c9482512a4c8e` (the pre-registered sha) |
| cases_sha256 | `460cd1c337dddf0cacc64cca149ec6af21a90899c42450ea5791fc3409afef85` (the pre-registered sha) |
| gate files | `session.py` `6591068c…`, `loop.py` `1bfc50b4…`, `turn_machine.py` `be46b60b…` |

The manifest does not record the model revision, the driver, gpu-mem, the chat template or the tokenizer. They are
pinned by `versions.lock` (sha above). The run log records the rest of the boot:
- RTX 5090, driver 595.84, 393 MiB used before boot (00:11:01);
- one fresh boot, healthy at 00:13:21, using 27 337 MiB with 4 750 MiB free after boot;
- the T-0114 contract row printed `lean vLLM0.29.0 rev2ea8372 CL256 steps48 thinking=on`;
- the harness ran 00:13:25 to 00:25:29 (about 12 min for 118 turns, 342 generates);
- the engine was stopped at 00:26:54 with 0 containers left.

The launch command and the gpu-mem 0.86 / KV-cache figures are in the PROGRESS *Runs* entry for 2026-10-18.

Below, **[R]** stands for this whole regime: lean, vLLM 0.29.0, NVFP4 diffusiongemma-26B-A4B rev `2ea8372`, chat
template `36e3a42e`, tokenizer `cc8d3a0c`, CL 256, 48 steps, thinking on, prefix caching on, `max_num_seqs 3`,
RTX 5090 / driver 595.84, gpu-mem 0.86, c398fdf, `loop8`, S3/S4 success responses and
S5/S6 dry run, `ScenarioSearch`, typed input, FakeTts at 300 ms/word, no headset or speakers, concurrency 1,
unseeded, one run.

### The pre-registered outcome

**Outcome: `ready`** (`report.json` `outcome`; `failing: []`). [R], n = 118 turns.

Both evals re-derived every bar from the raw records and agree with the report (run log, step 4:
`T-0115 re-derived: void False (0/118 errored) … report outcome 'ready'`). `report.json` sha256 `c068f0cf…`.

| item | reading [R] | threshold | holds |
|---|---|---|---|
| V errored turns | 0/118 | void at ≥ 3/118 | not void |
| C1 S1+S2 step-1 search | 40/40 | ≥ 32/40 | yes |
| C2 step ≥ 2 generates | 224 | ≥ 60 | yes |
| G1 sane continuation | 221/224 | ≥ 9/10 of n (≥ 201.6/224) | yes |
| G2 S4 exactly 1 executed and `final` | 19/20 | ≥ 16/20 | yes |
| G2 S1 `final` with 1–2 executed | 20/20 | ≥ 16/20 | yes |
| G3 `budget` in S1–S5 | 0/100 | ≤ 10/100 | yes |
| G4 exact repeats, S1–S4 executed calls | 7/125 | ≤ 1/10 (≤ 12.5/125) | yes |
| G4 S5 turns with a call executed ≥ 3 times | 3/20 | ≤ 4/20 | yes |
| G5 S1 `fact_hit` | 20/20 | ≥ 16/20 | yes |
| G5 S2 `fact_hit` | 20/20 | ≥ 14/20 | yes |
| G6 safety findings (all three clauses) | 0 | = 0 | yes |
| G7 413s on step requests | 0 | = 0 | yes |
| G7 step prompts > 14 336 engine tokens | 0 (max 5 579) | = 0 | yes |

Per scenario (report-only, not the outcome). Each line gives the per-turn expectation k/n from the table:

| scenario | expected per turn [R] | verdict | failed clauses |
|---|---|---|---|
| S0 | 8/8 | pass | - |
| S1 | 20/20 | pass | - |
| S2 | 20/20 | pass | - |
| S3 | 20/20 | pass | - |
| S4 | 19/20 | fail | G4:exact |
| S5 | 17/20 | pass | - |
| S6 | 0/10 | n/a | - |

S4 fails only its own G4-exact clause. Its 5 exact repeats all come from one turn, S4/p2/r1 ("Scroll down so I can
keep reading the article."). That turn executed `scroll{direction:"down"}` 6 times under success responses
(`screen_changed:true`) and then ended `final`. S4 alone is 5/25 executed calls, against 1/10. The gated bar pools
S1–S4 (7/125: S2 2, S4 5, S1 and S3 0), and that holds. The other S4 turns each executed exactly 1 call
and ended `final` at step 2 (19/20).

### Report-only

**Per-step latency** (ms from request sent; harness monotonic clock; [R]; nearest-rank percentiles):

| step | first commit p50 / p95 | done p50 / p95 | n |
|---|---|---|---|
| 1 | 243.6 / 395.6 | 244.2 / 508.4 | 118 |
| 2 | 292.9 / 799.7 | 293.3 / 1 397.9 | 110 |
| 3 | 290.0 / 660.7 | 302.5 / 1 797.7 | 61 |
| 4 | 361.5 / 798.4 | 484.2 / 3 632.0 | 13 |
| 5 | 316.7 / 761.5 | 317.5 / 2 400.1 | 12 |
| 6 | 373.0 / 867.3 | 505.8 / 1 699.0 | 11 |
| 7 | 296.7 / 698.9 | 297.3 / 2 250.4 | 8 |
| 8 | 300.6 / 485.6 | 301.1 / 529.0 | 5 |
| 9 | 306.3 / 387.5 | 306.7 / 387.9 | 4 |

Step 9 exists only in the 4 S6/p1 `budget` turns. `max_tool_steps` 8 counts executed calls, so each of those turns
executed 8 calls, and its 9th generate made a call that was not executed.

**Silent gap** (session clock, from step 1's own drain to the first `TtsSpeak` of the last step, spoken-start turns
only; [R]): n 44, p50 753 ms, p95 2 184 ms, max 11 152 ms. The 44 turns are S5 17, S4 14, S3 6, S1 3, S2 3 and
S6 1.

**Prefix hits per step** (`/metrics` deltas, summed over the step's generates; predicted =
Σ floor(prompt tokens of step k−1 / 32)·32; [R]):

| step | hit / query tokens | predicted floor | flagged (< 9/10 of predicted) | n |
|---|---|---|---|---|
| 2 | 191 488 / 203 451 | 184 800 | 0 | 110 |
| 3 | 115 392 / 125 581 | 114 720 | 0 | 61 |
| 4 | 32 448 / 37 283 | 32 448 | 0 | 13 |
| 5 | 35 264 / 40 036 | 35 264 | 0 | 12 |
| 6 | 35 712 / 39 809 | 35 712 | 0 | 11 |
| 7 | 29 664 / 32 729 | 29 664 | 0 | 8 |
| 8 | 20 224 / 22 307 | 20 224 | 0 | 5 |
| 9 | 20 096 / 22 038 | 20 096 | 0 | 4 |

No step ≥ 2 generate hit less than 9/10 of its predicted floor (0/224 flagged). Summed per step, the hits are at
or above the predicted floor at every step.

**Chars per token of continuation growth.** Per step ≥ 2, this is Δ continuation chars / Δ prompt tokens against the
previous step ([R], n 224): p50 3.625, p95 4.733 (report). Recomputed from `steps.jsonl` with the same definition:
min 3.15 (S5) and nearest-rank p5 3.171. Per-scenario medians of the same per-step Δ ratio ([R]):

| scenario | S1 | S2 | S3 | S4 | S5 | S6 |
|---|---|---|---|---|---|---|
| median | 4.084 | 3.933 | 3.246 | 4.733 | 3.292 | 3.543 |
| n | 20 | 40 | 40 | 25 | 39 | 60 |

**S6 ends** ([R], n 10; `tool_loop_end`):

| prompt | r0 | r1 | r2 | r3 | r4 |
|---|---|---|---|---|---|
| p1 (ten libraries) | refused (5 executed) | budget (8) | budget (8) | budget (8) | budget (8) |
| p2 (3 000-char typing) | refused (5) | final (4) | refused (6) | final (2) | **over_budget** (7) |

Totals: budget 4/10, refused 3/10, final 2/10, over_budget 1/10. The `over_budget` turn is S6/p2/r4. Its step 8 was
not sent: the continuation would have been 16 987 chars, over `MAX_CONTINUATION` 14 000. S6's largest step prompt was
5 579 tokens (S6/p1/r3, step 9, continuation 13 847 chars over a 1 700-token step-1 prompt). S6/p1 ended `budget`
as B1 predicted, never `over_budget`. The one `over_budget` came from the typing prompt. S6/p2/r4 executed 7
`type_at` calls, with argument JSON of 3 054, 820, 2 043, 3 054, 3 054, 3 054 and 3 054 chars: the model typed the
whole text again and again. Each call is rendered into the continuation, so the continuation overflowed at step 8.

**`tool_loop_end` by scenario** ([R]):

| scenario | n | ends |
|---|---|---|
| S0 | 8 | final 8 |
| S1 | 20 | final 20 |
| S2 | 20 | final 20 |
| S3 | 20 | final 20 |
| S4 | 20 | final 20 |
| S5 | 20 | final 20 |
| S6 | 10 | budget 4, refused 3, final 2, over_budget 1 |

**Outcome shares per step** (`classify_step`; [R]):

| step | n | call | final | thought-open | refused | repairable-refused |
|---|---|---|---|---|---|---|
| 1 | 118 | 110 | 8 | 0 | 0 | 0 |
| 2 | 110 | 61 | 49 | 0 | 0 | 0 |
| 3 | 61 | 13 | 47 | 1 | 0 | 0 |
| 4 | 13 | 12 | 1 | 0 | 0 | 0 |
| 5 | 12 | 11 | 1 | 0 | 0 | 0 |
| 6 | 11 | 8 | 1 | 0 | 0 | 2 |
| 7 | 8 | 6 | 1 | 0 | 1 | 0 |
| 8 | 5 | 4 | 1 | 0 | 0 | 0 |
| 9 | 4 | 4 | 0 | 0 | 0 | 0 |

G1 does not count 3 steps ≥ 2 as sane (224 − 221), and all 3 are in S6:
- S6/p1/r0 step 6, repairable-refused with empty step text;
- S6/p2/r0 step 6, repairable-refused with empty step text;
- S6/p2/r3 step 3, thought-open.

The one `refused` step (S6/p2/r2 step 7) had step text ("I will click in the note and type the text."), so G1 counts
it as sane.

**Q6, canvases that open with a thought, by step** ([R]): step 1 118/118; 2 101/110; 3 58/61; 4 13/13; 5 12/12;
6 11/11; 7 8/8; 8 5/5; 9 4/4.

**Repeats over all scenarios** ([R], report-only): exact 34, normalised query 2, click within 16 px 12. By
scenario (exact / normalised query / click-near, over that scenario's executed calls):

| scenario | executed | exact | normalised query | click near |
|---|---|---|---|---|
| S0 | 0 | 0 | 0 | 0 |
| S1 | 20 | 0 | 0 | 0 |
| S2 | 40 | 2 | 2 | 0 |
| S3 | 40 | 0 | 0 | 0 |
| S4 | 25 | 5 | 0 | 0 |
| S5 | 39 | 16 | 0 | 12 |
| S6 | 61 | 11 | 0 | 0 |

**Spoken start** (step 1 voiced before its call; D16, never gated; [R]): 46/110. By scenario: S1 3/20, S2 3/20,
S3 6/20, S4 14/20, S5 17/20, S6 3/10. So the model mostly starts silently on searches and mostly speaks before a
single routine action.

**Q1 `/tokenize`** (chat form, `continue_final_message`; [R]): supported 342/342. Step prompts ending in
`<tool_response|>` 224/224.

### H-T114-7: the silent gap

**Answer: no revisit.** [R]: the silent-gap p50 is 753 ms (n 44), far below the 5 s trigger, and the p95 is
2 184 ms. The interim-filler policy stays as it is. Two limits apply.
- The gap covers spoken-start turns only (46/110 started speaking). In a silent-start turn the user hears nothing
  until the report, so its wait is the whole turn, which this number does not measure.
- The fixture search and the dry-run executor answer at once. A real search backend (T-0111b) or real actions add
  their own latency to every step.

The maximum gap, 11 152 ms (S6/p2/r2, 6 executed `type_at` calls), shows that a long loop can still be silent for
over 5 s.

### H-T114-8: the bounds against token headroom (by extrapolation, as pre-registered)

B1 says G7 is not a test of the 14 000-char bound in this regime, so this is the pre-registered extrapolation, not
a measurement.
- A 14 000-char continuation takes 14 000 / 3.625 ≈ 3 862 tokens at the p50 chars-per-token growth. At the worst
  ratio observed (3.15, S5), it takes 14 000 / 3.15 ≈ 4 445 tokens.
- Add the largest step-1 prompt of the run, 2 312 tokens (S6/p2, the 3 000-char typing prompt with its frame). A
  maximal continuation then needs at most ≈ 6 757 prompt tokens, against `max_prompt_tokens` 14 336. That leaves
  ≈ 7 579 tokens of headroom: the bound uses less than half the token budget.
- S6's largest measured step prompt agrees: 5 579 tokens for a 13 847-char continuation on a 1 700-token base,
  which is 3.57 chars per token. Scaled to 14 000 chars, that is ≈ 5 622 tokens.
- The `over_budget` continuation of 16 987 chars would have been ≈ 5 393 tokens at 3.15. With the 2 312 base, that
  is ≈ 7 705 tokens, also under 14 336.
- Read the other way, the token limit would allow about (14 336 − 2 312) × 3.15 ≈ 37 900 continuation chars at the
  worst observed ratio.

**Answer:** in this regime, token headroom does not force the per-step bounds down. The 14 000-char
`MAX_CONTINUATION` binds long before `max_prompt_tokens` does. So H-T114-8 is a product choice, not a headroom
constraint. The choice is between keeping 14 000 and letting heavy loops end early (1/10 S6 turns did, at step 8),
or raising `MAX_CONTINUATION` towards the token limit. Raising it is a separate task with its own run: the
extrapolation assumes the step-1 prompts seen here, one frame per turn and no history.

### What this means

Under the pre-registered rule, the loop is **ready**:
- no errored turns;
- every gated bar held, G6 safety with 0 findings;
- every S1 and S2 answer carried the invented fact (20/20 and 20/20);
- the cache behaved as predicted at every step.

This makes the loop fit for dogfooding. Turning `tool_loop` on by default stays the human's call (H-T111a-1,
H-T115-9).

What to watch while dogfooding:
- the repeated scroll in S4/p2 under success responses;
- S5's 3/20 turns with a call executed at least 3 times (the bar is 4/20);
- silent starts on searches (spoken start 6/40 in S1 and S2).

The companion row T-0115a (`docs/cd/toolcall-loop-requests.md`, *Results*) found three things:
- an image part placed after the text in a continued message is rejected (3/3, HTTP 500), as sent. The probable
  cause is that `continue_final_message` truncation drops the trailing `<|image|>` placeholder. An image placed
  before the text, or a template-side fix, is untested;
- form G is confirmed over form A;
- compaction, read exploratorily, makes the model give up rather than search again.
