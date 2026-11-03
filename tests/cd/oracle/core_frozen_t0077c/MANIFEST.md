# tests/cd/oracle/core_frozen_t0077c — MANIFEST

**Purpose.** The frozen branch-base path for T-0077c (`docs/tasks/T-0077c.md`): criterion 5 (on CUDA, live with the
`log_matmul` kernel vs this copy: N4 for the sum-product parts, `torch.equal` for the N3 parts), criterion 6 (on CPU,
`semiring.log_matmul` with `kernels=True`/`kernels=False` is `torch.equal` to this copy's; every other `semiring.py`
function AST-equal) and criterion 9 (path A of `scripts/gpu/cd_kernel_ab.py`). It is T-0077b's code: the torch
`log_matmul` and T-0077b's Triton H(q) tail (`kernels/`), because the frozen `marginals.py` imports `.kernels`; A vs B
therefore isolates `log_matmul`. Like `../core_frozen_t0077b/` it is **not** independent of the implementation; the
independent oracles stay `../enumerate.py`, `../dense64.py`, `../viterbi64.py` and the fp64/`math.fsum` rows of
`tests/cd/unit/test_cd_logmatmul_spec.py`.

**Provenance.** Copied 2026-10-05 by the T-0077c test-author with
`git show b53457a:plugin/astra_vllm_plugin/astra_vllm_plugin/cd/<name>.py > <name>.py` (b53457a = the T-0077b merge
into `T-0062-perstep-cd`, the base of `T-0077c-cd-logmatmul-kernel`). Byte-identical: each sha256 below equals the
sha256 of the blob at b53457a (checked at copy time, and re-checked by `tests/cd/unit/test_cd_logmatmul_backend.py`).

**Import.** No edits: the files keep their relative imports, which resolve inside this package
(`tests.cd.oracle.core_frozen_t0077c`, `tests.cd.oracle.core_frozen_t0077c.kernels`), so the nine modules load side
by side with the live `astra_vllm_plugin.cd.*` and with `core_frozen_t0077`/`core_frozen_t0077b`, and share no module
object with either. The frozen `kernels/backend.py` logs on the same logger name as the live one
(`astra_vllm_plugin.cd.kernels`) with its own once-per-process state: a process that runs the frozen path on CUDA
sees its record too (the record-count rows read the records before the frozen path runs). **Frozen: never edited.**

| File | sha256 (= blob at b53457a) |
|---|---|
| `semiring.py` | `0b4ae05f7de105c3ea0642acd89876ad1bd14f725494342aa3a6d1d48577d00c` |
| `tree.py` | `45017614e056dbc2be2f42f9dfa683adc65449998f8a3d7971dd9a7a52ac1a51` |
| `emission.py` | `aaff33c4aa9f0877d4f2800276e48323304a6de40eef0c7e29347f1f8f95ed27` |
| `marginals.py` | `a85c404f054b60f5a552e6b6bffd7c58633a209abac5a91edc8bc71e6b6adc0d` |
| `mapdecode.py` | `1d13fb06c90e7760080742892c50d492214be9d9026f08f8f5a7a92890ce2ee5` |
| `draw.py` | `e45e8899f59d9fb8db0bf1f78a2f4749cf83f5173b2960a5ffb970b3409b9022` |
| `kernels/__init__.py` | `99a754b62d55422b55107fd27b8408014c7e1c9e126275b94f1970ca78e558b0` |
| `kernels/backend.py` | `14146a02723c3fac513699103cacdd287649bb1785d9ab31353de1220926346c` |
| `kernels/entropy.py` | `db0302a382d8b3cec8578a21d181626682f8be5d97419c2ebfcc9c292ebf0847` |
