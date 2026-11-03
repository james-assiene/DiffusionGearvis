# tests/fixtures/cd/prompts

The oracle for T-0070 criterion 2 (prompts byte-identical to diffgemma-fa's).

| file | provenance | sha256 | licence |
|---|---|---|---|
| `prompts.json` | **generated** 2026-09-30 by `generate.py` in this directory, which executes diffgemma-fa's own `eval/run.py::build_prompt(rec, fn, style="stock")` (with `PROMPT_STYLES`) and `compile/tasks/sudoku.py::build_prompt(rec)` read with `git show 27c7d0d:<path>` from `<diffgemma-fa clone>` (no `local_astra` code involved). Inputs: `../bfcl/BFCL_v4_live_simple.json` (gorilla `6ea57973c7a6`), the ids of `../dgfa_preserved/exp_e5_grammar130_map.json` (BFCL test cut, 130), the single-function live_simple records 130-229 in file order (BFCL dev cut, 100), and `../dgfa_preserved/sudoku_puzzles.json` (Sudoku test 250 / dev 100). | `1fd8b7f7c0b1ff4b506d118dbcb857ae6cf0e87f056c61b8b83813e015748dab` | Apache-2.0 (this repo); prompt text derived from BFCL (Apache-2.0) |
| `generate.py` | test-author code (T-0070). | — | Apache-2.0 (this repo) |

Note: Sudoku record ids collide across the two cuts (`sudoku4_0` … `sudoku4_99` exist in both, seeds 0 and 1); the puzzles do not. Join on `(cut, id)`, never on `id` alone.
