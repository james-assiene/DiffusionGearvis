# tests/fixtures/t0142

- `thoughts.jsonl`: 50 requests' thought commits copied verbatim from the gitignored A/B logs
  `logs/t0134-feedback-ab-e59aa69/` (T-0134 at e59aa69) and `logs/t0132-crop-budget-ab-2ccff41/` (T-0132 at 2ccff41):
  the 10 collapsed requests (`class: collapsed`) and 40 clean ones (`class: clean`), each with its source session file,
  request id and attempt id.
- Generator: `build_thoughts.py` (the test-author's own census, run read-only; its docstring states the selection and
  the checks against the planner's evidence in `docs/tasks/T-0142.md`).
- Regime: lean, CL 256, 48 steps, NVFP4 rev 2ea8372, vLLM 0.29.0, 1120 visual tokens; 3 arms x 100 per run.
- Licence: model output produced on this box; project-internal.
