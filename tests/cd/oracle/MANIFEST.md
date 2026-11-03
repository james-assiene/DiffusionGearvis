# tests/cd/oracle — MANIFEST

Test-author code for T-0065 (and reused by T-0066/T-0068). Pure Python + numpy; no torch, no plugin import.

| File | What | Provenance |
|---|---|---|
| `enumerate.py` | Brute-force enumeration over **strings** `x ∈ V^L`, with the NFA's accepting-path multiplicity `N(x)` counted by an integer DP per string. Two arithmetics: `Fraction` (exact, for rational `p`) and float64 log space (`math.fsum` after a max shift, any dynamic range). Also `distance_to_final` by reverse BFS, `advance` (δ on a token string) and `log_matmul_bruteforce`. | Written 2026-09-29 by the T-0065 test-author from SPEC §2.2–2.4 (diffgemma-fa `SPEC.md` at `ea7c5da`), not from any implementation. |
| `enumerate.py` (T-0066 additions) | `support_scores` (every constrained string with its multiplicity `N(x)` and score `Σ log p_i(x_i)`), `string_posterior` (exact joint posterior `∝ N(x)·Π p_i(x_i)`, SPEC §2.6 eq. 8) and `accepts` (host replay). The χ² arbiter of the joint draw and the brute-force argmax of the MAP. | Written 2026-09-29 by the T-0066 test-author from SPEC §2.5–2.7 (`27c7d0d`), not from any implementation. |
| `viterbi64.py` | float64 max-plus forward pass over class maxima of the true member sets (SPEC §2.7 eq. 9): the production-shape MAP-score arbiter; `string_score64` scores a returned string. | Same author and date. |
| `dense64.py` | Float64 numpy forward–backward in log space, per position, longhand (`[L, S, S]` dense, class masses by **direct summation over the true member set** — never a subtraction). The production-shape arbiter for criterion 2, where enumeration is impossible. | Same author and date; follows SPEC eq (1)–(6). |

## Deviation from T-0065 "Tests to write"

The task asks for diffgemma-fa `infer/reference.py` at `27c7d0d` to be vendored here. Copying it into this tree was
**refused by the session's permission classifier** ("untrusted code integration") on 2026-09-29, so it is not
vendored. `dense64.py` stands in as the float64 reference. It uses plain class sums (the `marginals` route, not
`marginals_complement_aware`), so the known >69-nat hole of the upstream complement-aware reference does not apply.
If the human approves vendoring, add the file verbatim with its sha256 here and cross-check `dense64` against it.

## T-0072 addition (frozen implementation copy; torch)

| File | What | Provenance |
|---|---|---|
| `delta_star_seq.py` | The **pre-change** sequential `budget.delta_star` (T-0065) and its `DeltaStar` dataclass, copied verbatim; the one oracle here that imports torch (it is the bit-identity reference for tensors). T-0072 criterion 1 compares the δ\* scan against it with `torch.equal`; criterion 2's liveness controls use it as the op-heavy baseline. **Frozen: never edited.** | Copied 2026-09-30 by the T-0072 test-author from `plugin/astra_vllm_plugin/astra_vllm_plugin/cd/budget.py` at commit `f9304b1` (file sha256 `f4b59e2a0bce3c948c1979e02ca7ce083752f70345091eed4d3423b24e971f98`); the `delta_star` body is byte-identical (checked with `diff`). Unlike the rest of this directory it is *not* independent of the implementation: it pins the behaviour the rewrite must keep, while `enumerate.py`'s `advance` stays the independent δ oracle. |

## T-0084a additions (native Gemma 4 tool-call format)

| File | What | Provenance |
|---|---|---|
| `gemma4_template.py` | Renders the pinned `chat_template.jinja` as `transformers` does (`ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`, `loopcontrols`), refusing any file whose sha256 is not `36e3a42e5cf14cd0020e72d92e1fdd9970f59b82170e421f0cbe1bb42bead3f0`. Returns the call substring and the full model turn of an assistant message with `tool_calls` (and `reasoning`). The language oracle for criterion 6. | Written 2026-10-09 by the T-0084a test-author; reads `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev `2ea837236295d617ac27f8c17d61228081932c40` from the local HF cache. No `local_astra` import. |
| `vllm_gemma4_args.py` | `_parse_gemma4_args` / `_parse_gemma4_array` and the delimiter constants, **verbatim** from vLLM; the server-side parser used as the language-side oracle of the random walks. | Extracted 2026-10-09 (read-only, `docker run --rm --network none --entrypoint cat`) from the local image `vllm/vllm-openai:v0.29.0` (image id `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`), file `vllm/parser/gemma4.py`, sha256 `478c04e66ffdda42a1457cb3b824166ec7de98fd8067953178624ad042879c94` (matches docs/tasks/T-0084a.md). Copied: SPDX Apache-2.0 header, source lines 38–43 and 47–282 byte-identical (checked by substring comparison). Changed: `logger = init_logger(__name__)` → a no-op logger; the unused vLLM imports are dropped. Licence Apache-2.0 (vLLM project). |
