# tests/fixtures/engine

| file | provenance |
|---|---|
| `vllm-coldboot-bd8001b.excerpt.log` | Lines 1–48, byte-for-byte, of `logs/vllm-coldboot-bd8001b.log` (untracked; full-file sha256 `b43427b921c2371e1a372ba2db97240a879494a16cbf9cac3618e878c77a6784`, excerpt sha256 `353b5c6ff0dd66c26791a0e185369c43a115091a64870e63697f745423b83d87`). Cold boot of 2026-09-17 15:42 at commit bd8001b: `vllm/vllm-openai:v0.29.0`, `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev 2ea8372, plugin on (`AstraDiffusionGemma`), **no tap**, flags as in `services/vllm.sh` at bd8001b. Cut by `head -n 48` on 2026-10-02 (T-0011 test-author). The excerpt ends at the `GPU KV cache size: 31,605 tokens, Maximum concurrency … 1.93x` line and contains every line criterion 14 names: the `non-default args` line, `Resolved architecture`, the engine `with config:` line (`kv_cache_dtype=bfloat16`), `Using AttentionBackendEnum.TRITON_ATTN backend`, `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend` (no Marlin fallback), and the KV-cache size/concurrency line. Licence: our own log output. |

Consumer: `tests/bench/test_b16_harness.py` (`local_astra.bench.b16_tap.boot_config_lines`, T-0011 criterion 14).
Consumer: `tests/cd/unit/test_cd_smoke_checks.py` (T-0069 criterion 8, `smoke.boot_problems` R16 rows).
