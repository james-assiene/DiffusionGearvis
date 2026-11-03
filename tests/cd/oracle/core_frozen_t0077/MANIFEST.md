# tests/cd/oracle/core_frozen_t0077 — MANIFEST

**Purpose.** The frozen pre-change path for T-0077's precision-neutrality rows (umbrella `docs/tasks/T-0077.md`):
N3 (exact parts `torch.equal` to this copy), N4 (sum-product parts within T-0065 c1's fp32 tolerances of this copy)
and N5 (per-step records equal to this copy's, on the CPU stock world). Like `../delta_star_seq.py` it is **not**
independent of the implementation: it pins the behaviour the live-state rewrite (T-0077a) and the kernels
(T-0077b–d) must keep. The independent oracles stay `../enumerate.py`, `../dense64.py` and `../viterbi64.py`.

**Provenance.** Copied 2026-10-04 by the T-0077a test-author with
`git show e73cfa7:plugin/astra_vllm_plugin/astra_vllm_plugin/cd/<name>.py > <name>.py` (e73cfa7 = T-0076 merged into
`T-0062-perstep-cd`, the base of `T-0077a-cd-live-states`). Byte-identical: each sha256 below equals the sha256 of
the blob at e73cfa7 (checked at copy time, and re-checked by `tests/cd/unit/test_cd_live_states.py`).

**Import.** No edits: the files keep their relative imports (`from .semiring import ...`), which resolve inside this
package (`tests.cd.oracle.core_frozen_t0077`), so the six modules load side by side with the live
`astra_vllm_plugin.cd.*` and share no module object with them. `budget.py` is not copied (not changed by T-0077a;
nothing here imports it). **Frozen: never edited.**

| File | sha256 (= blob at e73cfa7) |
|---|---|
| `semiring.py` | `0b4ae05f7de105c3ea0642acd89876ad1bd14f725494342aa3a6d1d48577d00c` |
| `tree.py` | `00495db70edae3717f334db3297729c6228645c8768f6b647fea5065ccb44ae2` |
| `emission.py` | `1175409ba665fa415f31e5d547a5bbf29d44d1d0485f47c195315349f8716cb6` |
| `marginals.py` | `c6a48bd0530715be4a2284cc25d6635d543f305661466c46c4927abbe9b85566` |
| `mapdecode.py` | `09f6b7d204f698f8f4e9ba4c41f69421521c3b97412d50d9406f9a6c91af01f5` |
| `draw.py` | `74154c8ee28385ecfb0262b3b52638c8b9d6d63b61a060bd916d424791a8f7ba` |
