# tests/fixtures/t0172

- `labels.jsonl`: the T-0172 A4 census labels. Line 1 is the provenance (corpus cut-off `20261027-123442`, the
  per-directory file counts, and the counts the builder cross-checked: 2,313 files, 6,681 `generate`/`tool_step`
  requests, 1,987 tools-on no-call answers, 45 tools-off). Then 116 rows, one per tools-on no-call answer holding
  first-person future intent anywhere (a broad filter), each hand-labelled: 108 positive, 8 negative. Every other
  no-call answer is a negative by definition. Each row has its source session file (under `logs/`), the
  `llm.request` seq, request and attempt ids, phase, step, the joined answer text, the label, why, and `smoke`
  (the two T-0125 smoke records at b372eda, answer commits seq 904 and 1047).
- Generator: `build_labels.py` (stdlib only, never imports `local_astra.toolcall.announced_call`; its docstring
  defines the label and lists the ambiguous cases). Run read-only over the gitignored logs on the 5090 box on
  2026-10-27 at main 3e4d45b7; output written to the session scratchpad, then copied here byte for byte.
- The chat logs under `logs/sessions/` (126 files, 11 tools-on and 45 tools-off no-call answers, no candidate) are
  read and counted only. No row, text or file name from them is here.
- Regime: lean, CL 256, 48 steps, thinking on, NVFP4 rev 2ea8372, vLLM 0.29.0, tools native, loop 8, unseeded
  (the A/B runs T-0131..T-0151, the T-0111g live smoke, the T-0125 search smoke and T-0160 web smoke at b372eda).
- Licence: model output produced on this box from loopback test pages and public search smokes; project-internal.
