# tests/fixtures/t0207

- `session_claims.jsonl`: a synthetic session log for `scripts/replay_completion_check.py` (T-0207 A1.5), 337
  lines (289 before T-0208 A2.6's t16-t19). The record shapes (`llm.request`, `llm.commit`, `llm.done`, `tool.call`, `tool.result`, `tts.utterance`,
  `note`, `turn.record` with `extras.vision_source`) are those of `logs/sessions/20261029-201813-2716251.jsonl`
  (James's session, read only for the shapes). Turns:
  - `t0`: a chat turn, no tools. No row.
  - `t1` (final request seq **2493**, like seq 2493): `type_at`, `scroll`, then a `click_at` whose thought says
    "Add to cart" and whose `frame_after` equals the scroll's ("no new page loaded"); the final step 4 has no
    thought and answers "I have added volume 7 of Vinland Saga to your cart." (the text the task file quotes).
    Row: `would_check` true, `last_action_unchanged` true, `report_step` false, label `updated`.
  - `t2` (final request seq **5538**, like seq 5538): `type_at`, `click_at`, a *refused* `type_at` with another
    `frame_after`, then a `click_at` with the same `frame_after` as the first click; the final step 5 has no
    thought and answers "I have purchased The Legend of Zelda: Tears of the Kingdom for you." (quoted by the
    task). Row: `would_check` true, `last_action_unchanged` true (only executed actions count), `report_step` false.
  - `t3` (final request seq **16197**, like seq 16197): the report step (`report_fact: true`, the continuation ends
    with `loop.REPORT_FACT`), labelled `unchanged`, an implicit thought, a 159-character answer claiming the
    checkout page was reached. Row: `report_step` true, `thought_present` true, `last_action_unchanged` true.
  - `t4`: a search-only turn (`google_search` executed, no browser action). No row.
  - `t5`: the final request ends with a `click_at` call (cancelled when the turn closed). No row.
  - `t6`: `vision_source: "shared"` (H-T207-9), `press_key`, `scroll`, `press_key`, the last `frame_after` equal
    to the first but not to the one before it. Row: `would_check` false, `last_action_unchanged` false, label
    `shared_screen`.
  - `t7` (adversarial): a `click_at` the user declined (never executed), then an executed `google_search` as the
    last action, then an answer. No row.
  - `t8`: one executed `click_at` (nothing before it), a thought, "I cannot see the screen right now." Row:
    `would_check` true, `last_action_unchanged` false, `claim_match` false.
  - `t9`: two `type_at` with different `frame_after`, a thought, "I am looking at the search results for volume
    8." Row: `last_action_unchanged` false, `claim_match` false.
  - `t10`: step 2 only announced a click; T-0172 re-asked it (`phase: "announce_reask"`, seq 16443) and the re-ask
    answered without a call. Row: `reasked_this_step` true, `would_check` false (H-T207-5).
  - `t11` (review 2026-10-30): the final step's answer tripped T-0142b (note `degenerate answer: request aborted`,
    no `llm.done` for the final request). Row: `would_check` false.
  - `t12` (review): the final step was aborted mid-stream (a barge-in: no `llm.done`, no note). Row: `would_check`
    false.
  - `t13` (review): step 2 was re-asked (T-0172) and the re-ask's click executed; the final step 3 answers with no
    call. Row: `reasked_this_step` false (only the final step counts), `would_check` true.
  - `t14` (review): the only executed action is `wait`. Row: `would_check` true.
  - `t15` (T-0207 A2-R3): a checked step as the A2 session logs it: step 2's draft commits (no `llm.done` of their
    own), then a `phase: "completion_check"` request (seq 16585, a fresh attempt under `t15-gen-s2`, `fresh_frame`),
    its commits, one `llm.done` and the check's answer voiced. Row: the step's `tool_step` request (seq 16579) is the
    final request, never the check; `would_check` true; the answer is the step's draft.
  - `t16`-`t19` (T-0208 A2.6, appended 2026-10-30 at e5301165; lines 1-288 unchanged): shared-source turns
    (`vision_source: "shared"`), step 1 labelled with the shared screen label and an executed `scroll`, then a
    no-call answer at step 2 whose label is `updated` (t16), `unchanged` (t17), `now` (t18) or the shared camera
    label (t19). Rows: `would_check` true for t16-t18 (the final request's image is a browser frame, T-0208
    A2.1), false for t19 (`shared_camera`).
- Seqs: the three named final requests are pinned to 2493, 5538 and 16197 (`final_at` in the builder); every other
  seq follows from the builder, gaps included (the real log interleaves `machine.step`/`wire.out` records).
- Generator: `build_claims.py` (the real `loop.step_prompt`, `loop.with_report_fact`, `announced_call.reask_prompt`,
  `session.log_dump` and `session_log.redact`, so images are redacted as the session log redacts them, `phash`
  kept). Run at BASE c97e5b30 on 2026-10-30 (t11-t14 appended at 8c3725bc the same day; lines 1-216 unchanged; t15 appended at 3281676c the same day, lines
  1-272 unchanged) with `uv run python tests/fixtures/t0207/build_claims.py`, output
  written to the session scratchpad and copied here byte for byte.
- No real user utterance, page text or URL; the only texts from the real log are the two answers the task file
  quotes. Licence: synthetic, project-internal.
