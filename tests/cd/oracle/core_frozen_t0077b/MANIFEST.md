# tests/cd/oracle/core_frozen_t0077b — MANIFEST

**Purpose.** The frozen branch-base path for T-0077b (`docs/tasks/T-0077b.md`): criterion 2(b) (on CUDA, the
dispatched `constrained_marginals(...)` — `q`, `entropy`, `log_z`, `log_a`, `log_b` — is `torch.equal` to this
copy's), criterion 5 (on CPU, every field of `constrained_marginals` with `kernels=True` and `kernels=False` is
`torch.equal` to this copy's) and criterion 8 (path A of `scripts/gpu/cd_kernel_ab.py`). Like `../core_frozen_t0077/`
it is **not** independent of the implementation: it pins the torch H(q) tail the kernel must reproduce bit for bit.
The independent oracles stay `../enumerate.py`, `../dense64.py` and `../viterbi64.py`.

**Provenance.** Copied 2026-10-04 by the T-0077b test-author with
`git show 2d90189:plugin/astra_vllm_plugin/astra_vllm_plugin/cd/<name>.py > <name>.py` (2d90189 = the T-0077a merge
into `T-0062-perstep-cd`, the base of `T-0077b-cd-hq-kernel`). Byte-identical: each sha256 below equals the sha256
of the blob at 2d90189 (checked at copy time, and re-checked by `tests/cd/unit/test_cd_kernel_backend.py`).

**Import.** No edits: the files keep their relative imports, which resolve inside this package
(`tests.cd.oracle.core_frozen_t0077b`), so the six modules load side by side with the live `astra_vllm_plugin.cd.*`
and with `core_frozen_t0077` and share no module object with either. **Frozen: never edited.**

| File | sha256 (= blob at 2d90189) |
|---|---|
| `semiring.py` | `0b4ae05f7de105c3ea0642acd89876ad1bd14f725494342aa3a6d1d48577d00c` |
| `tree.py` | `45017614e056dbc2be2f42f9dfa683adc65449998f8a3d7971dd9a7a52ac1a51` |
| `emission.py` | `aaff33c4aa9f0877d4f2800276e48323304a6de40eef0c7e29347f1f8f95ed27` |
| `marginals.py` | `4d9f2c34047fe458cb543f9051d98461aac3c28c7e6247996e460d7db9ec9402` |
| `mapdecode.py` | `1d13fb06c90e7760080742892c50d492214be9d9026f08f8f5a7a92890ce2ee5` |
| `draw.py` | `e45e8899f59d9fb8db0bf1f78a2f4749cf83f5173b2960a5ffb970b3409b9022` |
