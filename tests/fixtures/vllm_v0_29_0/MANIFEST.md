# vLLM v0.29.0 stock DiffusionGemma sampling code (vendored, CPU oracle for T-0067)

- **Source:** `vllm/vllm-openai:v0.29.0@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`,
  extracted 2026-09-30 with `docker create` + `docker cp` (image never started).
- **Licence:** Apache-2.0 (vLLM project); SPDX headers kept in each file.
- **Segments:** `manifest.json` lists every vendored segment with its path in the image, 1-based inclusive
  line range and the sha256 of its exact text. In the image's `diffusion_gemma.py`
  (sha256 `c962cf4c823431a22bd74fe2276806bd03b6119a8d0077b516a87c5d7ac0cb89`, 1425 lines),
  `_compiled_sample_step` is **L470–657** (decorator at L469). T-0067's text says "L473–659"; the file read
  here is authoritative.
- **Rule for the step's sha256:** from the line `def _compiled_sample_step(` through the first line
  `    return scaled`, inclusive, newline-terminated. The container probe applies the same rule.
- **Changes:** none inside the `# >>> vendored` / `# <<< vendored` markers. The `@torch.compile(...)`
  decorators are omitted (eager CPU). Imports are redirected to `_shims.py` (ours), which states what each
  stand-in keeps and drops. `apply_top_k_top_p` (ours) routes to the vendored pytorch path.
- **Checked by:** `tests/cd/unit/test_cd_stock_fixture_provenance.py` (fixture ↔ manifest) and
  `tests/contract/test_vllm_cd_coupling_gpu.py` (manifest ↔ image, `-m gpu`).

## Neutral path and consumers (T-0011 R8, 2026-10-02)

This directory is a copy of `tests/fixtures/cd/vllm_v0_29_0/` from branch `T-0062-perstep-cd` @ `e10bcd0`, moved to a
neutral path so both consumers share one vendored oracle. Every file except this one (`__init__.py`, `_shims.py`,
`diffusion_gemma_step.py`, `diffusion_sampler.py`, `manifest.json`) is **byte-identical** to the source; only this
section was added. The merge of main into `T-0062-perstep-cd` removes the `cd/` copy and repoints its importers here.

Consumers:
- **T-0011 (per-step canvas tap):** `tests/unit/test_vllm_fixture_provenance.py` (fixture ↔ manifest, torch-free),
  `tests/stock/world.py` + `tests/stock/test_tap_stock.py` (tier 1b, the tap around the vendored sampler), and
  `tests/contract/test_vllm_tap_fields_gpu.py` (manifest ↔ image, `-m gpu`).
- **T-0062 / T-0067 (per-step constrained decoding):** `tests/cd/fakes/diffusion_sampler.py`, the `tests/cd/unit`
  modules that import it, `test_cd_stock_fixture_provenance.py` and `test_vllm_cd_coupling_gpu.py` (after the merge).
