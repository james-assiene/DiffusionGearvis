# tests/fixtures/cd/model

| file | provenance | sha256 | licence |
|---|---|---|---|
| `generation_config.json` | verbatim copy of `generation_config.json` from the HF snapshot of `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev `2ea837236295d617ac27f8c17d61228081932c40` (the revision pinned in `services/versions.lock`), copied 2026-09-29. Its `eos_token_id` `[1, 106, 50]` is the oracle for the automaton stop set (T-0063 criterion 8). | `99334f763c3dbe8b161aeaca1c150a05344299fda2d2e4a0e1d342c744461200` | model card licence (NVIDIA / Gemma terms); a 357-byte config file |

`tokenizer.json` of the same snapshot is **not** copied (32 MB). The tests read it from the local HF
cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`) and fail loudly if it is
absent or if its sha256 is not
`cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` (measured 2026-09-29, 32 169 626 bytes).
