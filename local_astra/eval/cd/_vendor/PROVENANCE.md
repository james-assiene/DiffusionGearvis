# Vendored code: provenance (T-0070)

- **Source:** diffgemma-fa (github james-assiene, local clone `<diffgemma-fa clone>`), commit
  **`27c7d0d3cc9c9e338d2fe4f6a1e64d3da01c8c03`**, each file read with
  `git -C <diffgemma-fa clone> show 27c7d0d:<path>` (not from the working tree).
- **Why:** T-0070 scores with diffgemma-fa's scorers as they stood after the 2026-08-08 rescoring
  ([AUDIT-A]..[AUDIT-E]), so the local numbers are comparable with docs/RESULTS.md there.
- **Licence:** Copyright James Assiene; licensed Apache-2.0 for this repository (ruling 2026-10-25).
  Vendored under T-0062 human decision 4.
- **Vendored by:** T-0070, 2026-09-30. Excluded from ruff and mypy (pyproject.toml) so it stays verbatim.
- **Dependencies:** stdlib only, plus `jsonschema` (lazily, in `metrics.schema_valid`; a main dependency).

## Files

| file (here) | upstream path | status | sha256 (here) | sha256 (upstream 27c7d0d) |
|---|---|---|---|---|
| `metrics.py` | `diffgemma_fa/eval/metrics.py` | unmodified | 7e786563a5d7b3c1a2b5680b817ff4e49ef2b5db7882be43376cccf6b8d59843 | 7e786563a5d7b3c1a2b5680b817ff4e49ef2b5db7882be43376cccf6b8d59843 |
| `stats.py` | `diffgemma_fa/eval/stats.py` | unmodified | f8396bf7c3ec333bbbb2a323523352fab79975b4b522ac3b8c39551520efbaf1 | f8396bf7c3ec333bbbb2a323523352fab79975b4b522ac3b8c39551520efbaf1 |
| `tasks/__init__.py` | `diffgemma_fa/compile/tasks/__init__.py` (`answer_region`, `NO_ANSWER_REASONS`, `produced_an_answer`) | unmodified | ae2eaad9ada6c6975cf49832380695393dcdcfd0aa2728afdcda20ea9b750cfd | ae2eaad9ada6c6975cf49832380695393dcdcfd0aa2728afdcda20ea9b750cfd |
| `tasks/sudoku.py` | `diffgemma_fa/compile/tasks/sudoku.py` | modified: one import line | ae7e21084315d037670e8b4fa8a85a7a6265382f72d946642fa7ec1a282a3d53 | 59b413d89552693e6f5cd17d9c10bfbe8003d7010fcd57a67835fdf5377388f7 |
| `run_prompt.py` | `diffgemma_fa/eval/run.py` (`PROMPT_STYLES`, `build_prompt`) | excerpt | 919504aca9e765cb218cb4bbabe70be058979a3f7f18c0418eee596539e60719 | 056c938fef035c32de91e3c37914109c6764945deea6a3c8f0bbd49dd52f48ae (whole file) |

## Deviations

- `tasks/sudoku.py`: `from diffgemma_fa.compile.tasks import answer_region` became
  `from local_astra.eval.cd._vendor.tasks import answer_region`. Nothing else changed.
- `run_prompt.py`: the two top-level definitions `PROMPT_STYLES` (with its `#:` comment block) and
  `build_prompt`, copied byte for byte from `eval/run.py`, under a new module docstring and
  `from __future__ import annotations`. The rest of `run.py` imports JAX and is not vendored.
- The BFCL loader (`bfcl_data`) and schema normaliser (`schema.normalize_bfcl_schema`) are not vendored
  again here: T-0063 already vendored `diffgemma_fa/compile/` unmodified for those two files
  (`local_astra/decoding/_vendor/PROVENANCE.md`), and the harness imports them from there.
