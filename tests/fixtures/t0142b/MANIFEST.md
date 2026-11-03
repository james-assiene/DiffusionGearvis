# tests/fixtures/t0142b

- `answers.jsonl`: 41 requests' answer commits copied verbatim from the gitignored A/B logs
  `logs/t0132-crop-budget-ab-2ccff41/` (T-0132 at 2ccff41), `logs/t0134-feedback-ab-e59aa69/` (T-0134 at e59aa69) and
  `logs/t0135-thought-call-ab-e1ae76e/` (T-0135 at e1ae76e): the 1 collapsed request (`class: collapsed`, T-0135 arm B
  S2#5, attempt `5008519a4296-t1-gen-s2-4`, its 3 answer commits and, for the session replay, its 2 thought commits)
  and the 40 clean ones with the highest repeat / character-run / length scores (`class: clean`), each with its source
  session file, request id and attempt id.
- Generator: `build_answers.py` (the test-author's own census, run read-only; its docstring states the selection and
  the checks against the planner's evidence in `docs/tasks/T-0142b.md`: 1,666 requests with answer commits, 1
  collapse, 1,665 clean with longest unit repeat 1 and longest character run 2).
- The chat logs under `logs/sessions/` (H-T142b-3) are read by `tests/eval/test_t0142b_replay_answers.py` only and are
  never copied here: they hold personal documents shown to the assistant.
- Regime: lean, CL 256, 48 steps, NVFP4 rev 2ea8372, vLLM 0.29.0, 1120 visual tokens; T-0132 and T-0134 3 arms x 100,
  T-0135 2 arms x 100 (order seed 0, engine unseeded).
- Licence: model output produced on this box; project-internal.
