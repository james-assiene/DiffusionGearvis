# tests/fixtures/cd/dgfa_preserved

Preserved diffgemma-fa (JAX baseline) emissions, used by `tests/cd/unit/test_grammar_parity_jax.py` to
replay each recorded text through the host-compiled automaton (T-0063 criteria 5-6) and by
`tests/cd/unit/test_state_gate.py` for the Sudoku cuts (criterion 7).

| file | provenance | sha256 | licence |
|---|---|---|---|
| `exp_e5_grammar130_map.json` | verbatim `git show 27c7d0d:preserved/exp_e5_grammar130_map.json` from `<diffgemma-fa clone>` (github james-assiene). The accuracy-parity headline arm: BFCL-Live simple, n = 130 (offset 0), `j0`/`map`, `whitespace=pretty`, `fence`, `ci_enums`, `expand_wildcard`, `nonempty_strings`, seed 0; written at code commit `4f9a6e3` (code base `1aa8ccf`). All 130 rows `accepted: true`. | `384a1c9c8649685d49b81a934a6e67526726f018c254d7a547d6e35f008ac31d` | Copyright James Assiene; licensed Apache-2.0 for this repository (ruling 2026-10-25); test data only |
| `eval_bfcl_live_simple_unconstrained_sample.json` | verbatim, same commit. Stock DiffusionGemma, n = 130, seed 0. 0/130 `accepted`. | `bfe0e59f7b6c0702fcaf50fab1df1fc27cff1538c861a0a3ae4fc0fdf6a820a6` | as above |
| `task_sudoku_unconstrained.json` | verbatim, same commit. Sudoku 4x4, n = 250, seed 0, unconstrained. 0/250 `accepted`, 216 `ok`. `text` is capped at 300 characters by `eval/run_tasks.py` (every row here is shorter). | `269ac9559db6231b353b59fdfdc68fd58f8684f65a94aed3048ceaba3f93a0bd` | as above |
| `task_sudoku_j0map.json` | verbatim, same commit. Sudoku 4x4, n = 250, seed 0, `j0`/`map`. 250/250 `accepted`, 0 `ok`. Its `why` column comes from the pre-AUDIT-D scorer and is not re-derivable. | `8a5244fa479c1a95e98a7637ca335fe9640bd6ba408b1909bafc91f5bf8127f0` | as above |
| `sudoku_puzzles.json` | **generated**, not preserved: `diffgemma_fa.compile.tasks.sudoku.generate(n, seed=seed, n_blanks=6)` at diffgemma-fa `27c7d0d`, run on the host 2026-09-29 (pure Python, no model). Cut `test` = (250, seed 0), the puzzles of both Sudoku artifacts above; cut `dev` = (100, seed 1) (T-0070 criterion 1). **Validated**: re-scoring the preserved unconstrained texts against these puzzles with diffgemma-fa's current `score_solution` reproduces the recorded `ok` flag on 250/250 rows (the `why` strings differ on 4 rows only by the post-AUDIT-E wording `empty`/`no grid`), and all 250 `j0map` texts replay as accepted against automata built from these givens. | `159e0618ea1713a37cf211b0747691dfdb6b20e83c73349fdba2e65a1865c72f` | Apache-2.0 (this repo) |

**How the rows are read** (the replay rule the tests apply, derived from diffgemma-fa `eval/run.py` and
`eval/run_tasks.py` at 27c7d0d): the JAX harness stripped trailing PAD (0) and stored
`gemma.Tokenizer.decode(tokens)`. EOS (1) is a SentencePiece control piece and decodes to nothing, so the
stop token that ended an accepted emission is not in `text`; `<turn|>` (106) and the channel delimiters
are ordinary pieces and do appear. The replay is therefore `HF tokenizer.encode(text,
add_special_tokens=False) + [1]`; when the text already ends in a visible stop, the extra EOS falls in
the unscored `ACC --Σ--> ACC` tail. A scratch prototype of this rule (vendored compiler + an HF-vocabulary
adapter, not committed) reproduced the recorded `accepted` flag on 130/130 e5 rows and 250/250 rows of
each Sudoku artifact, 2026-09-29, outlines-core 0.2.14, tokenizers 0.23.2.
