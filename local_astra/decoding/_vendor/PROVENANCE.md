# Vendored code: provenance

## `dgfa_compile/` — diffgemma-fa's grammar compiler

- **Source:** `diffgemma_fa/compile/` of diffgemma-fa (github james-assiene, local clone
  `<diffgemma-fa clone>`), commit **`27c7d0d3cc9c9e338d2fe4f6a1e64d3da01c8c03`** ("Preserve eval artifacts,
  logs and frozen queue scripts before VM termination"). Extracted with
  `git -C <diffgemma-fa clone> archive 27c7d0d diffgemma_fa/compile`, not from the working tree (which
  carries a later SPEC-only commit, `ea7c5da`).
- **Licence:** Copyright James Assiene; licensed Apache-2.0 for this repository (ruling 2026-10-25).
  Vendored under T-0062 human decision 4 ("vendoring is allowed").
- **Vendored by:** T-0063, 2026-09-29.
- **Regime it was checked in:** CPU only; outlines-core 0.2.14 (the version diffgemma-fa ran,
  `docs/ENV.md` at 27c7d0d); tokenizers 0.23.2; torch 2.13.0+cpu in the `cd` group, pinned to the torch
  inside `vllm/vllm-openai:v0.29.0` (`torch-2.13.0+cu130`, read from the image's dist-info by the
  orchestrator 2026-09-29, docs/tasks/T-0062.md decision 2); vocabulary from `tokenizer.json` of
  `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev `2ea837236295d617ac27f8c17d61228081932c40`, sha256
  `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.

### Files

One row per vendored file: path relative to `dgfa_compile/`, sha256 of the file **as vendored here**,
status against upstream, and the upstream sha256 at 27c7d0d. `tests/cd/unit/test_vendor_provenance.py`
recomputes both and fails on drift.

| file | sha256 (vendored) | status | sha256 (upstream 27c7d0d) |
|---|---|---|---|
| `__init__.py` | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 | unmodified | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| `automaton.py` | 0110de96683349eed2fd5135531f76e730dadd00538400fe5616c7bbf38da54a | modified | 66267e2b5f2e68ff37a00eb87c008d0289b760863659755a8dc5a6e4c4f9a2e7 |
| `bfcl_data.py` | 655fcfd8e21926a098e9c60630bd8534b28bd42b0425abb68eb97dada24a1f73 | unmodified | 655fcfd8e21926a098e9c60630bd8534b28bd42b0425abb68eb97dada24a1f73 |
| `classes.py` | 59846f304f858eaa2c40c2e09f6b370d2561c8a90f0254fd89578b496ee0fd03 | unmodified | 59846f304f858eaa2c40c2e09f6b370d2561c8a90f0254fd89578b496ee0fd03 |
| `lift.py` | 7aee8fbe8f1901374fc3d3c0f426e9d9c1562ac54055429658ccfdbab3e7c403 | modified | ce1d2da0ddd911ad0f2f66b2a612c6bd07efbfa4a87db51db80f9469e3ec29cf |
| `minimize.py` | bb9a74bc9456602284ecb3a7b5ea855dec2baf6876f48d94bee5676213f3421d | unmodified | bb9a74bc9456602284ecb3a7b5ea855dec2baf6876f48d94bee5676213f3421d |
| `pipeline.py` | e77a893b08b3d17fa43cfb324abd6f60e1bf57e34d82d6e9153ab081bb40156b | modified | 6c646bd0ff5a56626cb23ae964da3b6234d773ffe36b80035dca91a68b600a69 |
| `schema.py` | d242d567157c2e714626d6064ae2cde93eb2358d0dd5bd0f5ffeb962e14a19ca | unmodified | d242d567157c2e714626d6064ae2cde93eb2358d0dd5bd0f5ffeb962e14a19ca |
| `validate.py` | 24bb5b7718d3cc2233108bf6e2ae1f17f5ef561835854ae9d70dffafad77218c | modified | 044ca20ee2d27c3978bb7d8e33853f7d5eb46909f2b4d1491beaaa47b3e7ddbc |
| `vocab.py` | dbbb401bb7ff2324ecc6042d63377af39229262c9ec12b612421059358175ce3 | modified | ab3b96bb72a77240fb4b182fffdf137e78349abebfa468afc4a4b367c9babe13 |
| `tasks/__init__.py` | ae2eaad9ada6c6975cf49832380695393dcdcfd0aa2728afdcda20ea9b750cfd | unmodified | ae2eaad9ada6c6975cf49832380695393dcdcfd0aa2728afdcda20ea9b750cfd |
| `tasks/bfcl.py` | 8dda8d2f09edbd542e1c080862135a06617d649e9df9c50dfef8be99fe7089bb | modified | 98b302c5fc73d3f5037c0fb5c263dc73c2533ddcdf262c9148d6483a456c8fef |
| `tasks/bfcl_python.py` | 53c29defdc969a375004333b50c19e2a08f3986771622e3c42cdb6a0e3ea3a46 | modified | abbb840ccf7abb1fe103d9114801486545f4e4409879b591fbb442687fab7ad2 |
| `tasks/countdown.py` | 3052be9a3e9c84708e542f52d289568b6662945858c8f0d9a96a8703efaabc98 | modified | 6f4fd10f8a805caa0aa03ad8dfdf61addd83809ad3e8f166ea90ca4f0a35f6d0 |
| `tasks/grammars.py` | 6afdab1999a65a7f3af03e99a6d124ba66a0b29d09856ed5ffbcc63263d845f5 | unmodified | 6afdab1999a65a7f3af03e99a6d124ba66a0b29d09856ed5ffbcc63263d845f5 |
| `tasks/sudoku.py` | 7a16b2267c638d60a482b127dd8a01bccfdebffb47adc87589fd392e42ea352b | modified | 59b413d89552693e6f5cd17d9c10bfbe8003d7010fcd57a67835fdf5377388f7 |

Nothing else of diffgemma-fa is vendored. The files are excluded from ruff and from mypy's strict
checking (`pyproject.toml`) so that they stay byte-comparable with upstream.

### Deviations

Every modified file, and why. Nothing else differs from upstream: `diff -r` against the archive shows
only the lines named here.

1. **Import-path rewrites** (`automaton.py`, `lift.py`, `pipeline.py`, `validate.py`, `tasks/bfcl.py`,
   `tasks/bfcl_python.py`, `tasks/countdown.py`, `tasks/sudoku.py`). Every `from diffgemma_fa.compile…`
   import statement, module-level or function-local, is rewritten to
   `from local_astra.decoding._vendor.dgfa_compile…`, with the rest of the line unchanged. Reason:
   diffgemma-fa has no `pyproject.toml` and is not installable, and the package must import under its
   vendored path. Prose that mentions `diffgemma_fa` (docstrings, comments, CLI usage strings) is left
   as it was.
2. **`vocab.py`: the SentencePiece tokenizer is replaced by the HF `tokenizer.json`.** Upstream built the
   `outlines_core.Vocabulary` from the JAX `gemma` package's SentencePiece model
   (`gm.text.Gemma4Tokenizer()._sp`), which pulls in JAX and is not available on the host. Here:
   - `gemma_tokenizer()` returns `HfGemmaTokenizer`, wrapping the `tokenizers.Tokenizer` of the pinned
     revision's `tokenizer.json` (located, sha256-checked and loaded by `local_astra.decoding.vocab`),
     with the `vocab_size` attribute `pipeline.compile_regex` reads (262 144).
   - `build_vocabulary()` walks the same 262 144 ids with the **same piece -> bytes rule** (`<0xXX>`
     byte-fallback pieces -> the raw byte; otherwise U+2581 -> space, UTF-8). SentencePiece's
     `IsControl`/`IsUnknown` has no counterpart in `tokenizer.json`, so the four control/unknown pieces
     upstream skipped are named: `CONTROL_PIECES = {"<pad>", "<eos>", "<bos>", "<unk>"}` (ids 0-3).
     The Vocabulary's EOS id is read as `token_to_id("<eos>")` (= 1, as upstream's
     `special_tokens.EOS`).
   - The module docstring gains a paragraph naming this deviation. `END_TOKENS`, `PAD_TOKEN`,
     `CHANNEL_OPEN`/`CHANNEL_CLOSE`, `RESERVED_TOKENS` and the `build_vocabulary` signature and
     docstring are unchanged.

### Ported, not vendored (live in `local_astra/decoding/grammars.py`)

diffgemma-fa `eval/run.py` at 27c7d0d imports JAX and the `gemma` package, so it is not vendored. The
grammar options of the baseline arm that live there are copied into `grammars.py` verbatim in
behaviour, each with a pointer back: `PRETTY_WS = r"( |\n {0,6})?"`, `WHITESPACE_PATTERNS`, `ALLOW`
and `_case_insensitive_enums` (as `case_insensitive_enums`). The Sudoku call
`sudoku_regex(r.puzzle, row_separator="\n")` with the default 8-token header is from
`eval/run_tasks.py::load_task` at the same commit.
