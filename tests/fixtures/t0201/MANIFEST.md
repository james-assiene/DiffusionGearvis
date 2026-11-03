# tests/fixtures/t0201

- `requests.jsonl`: a synthetic session log for `scripts/replay_screen_reminder.py` (T-0201 B9). Line 1 is a
  `log.started` record; then eleven `llm.request` records: a labelled `generate` (seq 10, no row), eight labelled
  `tool_step` requests (seq 11 shaped like `20261029-173950-15317.jsonl` seq 16202: a `type_at` then scrolls,
  labelled `updated`; seq 12 like `20261029-181336-282350.jsonl` seq 3957: a long scroll run; seq 13 `unchanged`,
  14 `now`, 15 `updated` with the T-0136 outcome clause, 16 `updated` with the T-0134 extent clause, 17 the shared
  screen, 18 the camera), an unlabelled `tool_step` (seq 19, no row) and a labelled report step (seq 20,
  `report_fact: true`, no row: B4).
- Generator: `build_requests.py` (the real `loop.step_prompt`, `session.log_dump` and `session_log.redact`, so the
  images are redacted exactly as the session log redacts them, `phash` kept); run at BASE 620b1245 on 2026-10-29,
  output written to the session scratchpad and copied here byte for byte. No real user utterance, page text or URL.
- Licence: synthetic, project-internal.
