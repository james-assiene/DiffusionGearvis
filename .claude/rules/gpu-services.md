---
paths:
  - "services/**"
  - "local_astra/conductor/llm_client.py"
  - "local_astra/decoding/**"
  - "scripts/gpu/**"
  - "scripts/demo/**"
  - ".claude/hooks/guard_bash.sh"
---

# vLLM, moshi-server and decoding

- Versions are pinned in `services/versions.lock` (vLLM, torch, CUDA, driver, FlashInfer, moshi-server, Gemma 4 chat template revision). Changing any of them is its own task with bench B1/B3/B7 runs.
- vLLM flags for DiffusionGemma (`services/vllm.sh` is the executable form): `--max-num-seqs 3` (≤ 4 is mandatory — diffusion state scales seqs × canvas × 262K vocab), `--kv-cache-memory-bytes 4294967296` (the pool is spelled in **bytes** and is never sized by the utilisation figure — with `--kv-cache-memory-bytes` set, vLLM skips memory profiling and says so in its log; `--gpu-memory-utilization` is passed **only** as the startup free-memory gate, at a value derived from the measured free fraction with the recogniser resident and **never below the engine's own footprint**, so the gate still refuses a configuration the engine could not survive — 0.86 as of T-0058, from `logs/vllm-coldboot-73a85dd.log` 2026-09-16, where the 0.92 default demanded 28.83 GiB of a 31.33 GiB device and the recogniser had left 27.23, killing the boot 21 s in before a weight was loaded; boot order is **engine first, recogniser second**, which `astra up` already does and a hand start must copy; ≥ 3 GiB must stay free for the unprofiled sampler transient — spell the pool in **bytes**: the parser takes a single-letter suffix where case picks the base, so `"4GiB"` is a hard argparse failure and `4g` would silently mean 4e9), `--attention-backend TRITON_ATTN` (sm_120), `--enable-prefix-caching`, `--reasoning-parser gemma4`, `--tool-call-parser gemma4`, `--diffusion-config` is **engine-global** (no per-request canvas/steps). KV cache stays bf16 in v1: no FP8 KV (docs/01 R2), and NVFP4 KV never (V-scale corruption on the XQA path). **`--kv-cache-dtype bfloat16` is required, not a default**: the checkpoint's `quantization_config.kv_cache_scheme` declares `num_bits: 8`, so under `auto` vLLM silently selects `fp8_e4m3` and — with no k/v scales in the checkpoint — runs it at scaling factor 1.0 (measured 2026-09-13, T-0014). bf16 KV costs 2× per token and 25 of 30 layers are sliding-window, so their cost is a constant that lowering `--max-model-len` does not reduce; at 2 GiB the bf16 ceiling is 8,720 tokens, which is why the budget is 4 GiB at `--max-model-len 16384` (1.68x). Verify with the config-only allocator check before booting, never by reading `GPU KV cache size` — that line is `int(max_concurrency * max_model_len)`, derived from the length, not a capacity.
- Boot assertions: total VRAM per profile ≤ 29 GiB (docs/02 §7, B7); log must not contain the Marlin/"no FP4 support" fallback warning for W4A4 layers; a warm-up request with one image runs before the service reports healthy.
- **The DiffusionGemma image path is patched by us, not by upstream (T-0015).** `diffusion_gemma.py:289-301` borrows seven methods from `Gemma4ForConditionalGeneration` by class-level assignment instead of inheriting. Commit `95c914442` (2026-08-13, #42662 "Support vision tower LoRA") added `self._enable_mm_lora` to Gemma4's `__init__` and read it in the borrowed `_process_image_input`, so **every image request raises `AttributeError` and kills the EngineCore** — not just the request. Broken since 2026-08-13; worked 2026-06-12 → 2026-08-12; v0.29.0 and upstream `main` both ship broken, so there is no release to downgrade to. Filed upstream: https://github.com/vllm-project/vllm/issues/56712. Our `AstraDiffusionGemma` wrapper supplies the attribute. **Delete the shim the moment upstream fixes this** — `tests/contract/test_vllm_shim_gpu.py::test_upstream_defect_is_still_present` goes red when that happens, and red there is good news. Never "fix" that test by relaxing it. Because the wrapper overrides `architectures`, vLLM's `MODELS_CONFIG_MAP` lookup misses and `verify_and_update_config()` never runs: the four explicit flags `--attention-backend`, `--diffusion-config`, `--max-num-seqs`, `--override-generation-config` are what keep plugin-on identical to plugin-off. Do not drop them.
- Sampling params on the diffusion path: only `top_k`, `top_p`, `max_tokens`, `stop` (`SAMPLING_WHITELIST` in `contracts/llm.py`). `temperature`, `min_p`, `seed`, `min_tokens`, `logit_bias`, `bad_words`, `allowed_token_ids` are a hard **400** — measured 2026-09-13, and the rejection is itself useful: it only happens on the diffusion path, so a `temperature` probe returning 200 proves the engine is *not* running diffusion.
- **Output-format fields with tools (T-0088).** Only when `Prompt.tools` is set, the request adds `tools`, `tool_choice: "none"`, `skip_special_tokens: false` and `return_token_ids: true`. T-0087 arm (b) at 724c0c4 measured them accepted on the diffusion path. They never travel through `LlmOptions`, and they never appear in a tools-off body.
- Structured outputs / `response_format` / `logit_bias` do **not** work on the diffusion path in vLLM or SGLang (400). Our constrained decoding is a per-step logits hook (`docs/07`). Tool calls fall back to the `gemma4` regex parser.
- Decoding hook code follows the numerics rules from `docs/07` §0: log space throughout, normalise tree nodes, finite sentinel `-1e30` never `-inf`, `log(max(q, 1e-30))`, differential-test every optimised path against the float64 reference, never let the exactness test go red. Automata on the live path must be ≤ 128 states (build gate).
- `moshi-server` speaks msgpack over WebSocket; contract tests for STT delay (0.5 ± 0.15 s) and TTS first-audio run against both the fake and the real server.
- No `pip install` outside `uv`; no system CUDA changes; never `sudo`.

## Where GPU-spending commands live (T-0016)

- **`scripts/gpu/` is the GPU-spending directory.** Any script that starts an engine, sends it a
  request, or measures it lives there, whatever it is called: `x3_prefix_cache.sh` (moved here from
  `scripts/` by T-0016). `scripts/locked/`, where `/launch` step 2 copies a long script before
  running it, is gated the same way (T-0016 I2). `.claude/hooks/guard_bash.sh` treats *running*
  anything under those paths as GPU-spending by definition, so it needs a `/launch` token and a
  clean tree; a `cd` into the directory does not escape it (I1). Reading, editing, listing,
  linting or committing those files is untouched: `cat`, `ruff check`, `bash -n` and
  `python3 -m py_compile` on a harness need no token (I9), and `--help` does need one, because
  these scripts talk to the engine before they parse their arguments (R2). Put new harnesses here by
  default: a harness anywhere else is invisible to the guard until somebody remembers to add its
  name to a regex, which is exactly how the X3 prefix-cache run escaped review on 2026-09-13 (no
  token, and no dirty-tree check either, so its numbers were measured against a tree two agents
  were editing).
- **Three trigger classes.** *Measurement* — produces numbers, writes into the artifacts
  directory, runs a `scripts/gpu/` harness, selects the gpu pytest marker, or sends an inference
  request — needs a full `/launch` review, a single-use token, a clean tree and a *Runs* entry in
  `PROGRESS.md`. *Engine starts* — `vllm serve`, the STT server (`moshi-server`),
  `docker run --gpus` — take VRAM until something stops them, so they need the same token
  and clean tree. *Routine service starts* — `uv run astra up`,
  `systemctl --user start astra-*`, the commands CLAUDE.md lists among its ordinary
  commands — need **only a clean tree**: no token, no reviewer pass, no *Runs* entry (T-0016 I12,
  which is D2 as the human decided it). The reason for the split: the only thing that writes a
  token is `/launch`, and `/launch` always reviews, so gating `astra up` on a token puts a review
  in front of every start — exactly the friction D2 removed. The hazard D2 does care about is
  measuring later against a tree that was dirty when the service started, and the clean-tree check
  catches that on its own. A dirty-tree refusal **names the dirty paths**: a refusal that only says
  "tree is dirty" is the one that gets an agent to re-run blind.
- Reading the engine's status is free and stays free: `GET /v1/models`, `/metrics`, `/health` spend
  nothing and we poll them constantly. Only a request carrying a body to an inference endpoint
  counts as GPU time.
- The guard matches the *executable shape* of a command, not text it merely quotes: heredoc bodies
  and quoted spans are blanked before the word patterns run, except the argument of `-c` (that is
  how `services/vllm.sh` starts the engine) and heredoc bodies fed to an interpreter. So describing
  a GPU command in a commit message or a `grep` pattern is not running one. The path trigger is the
  exception: it keeps the quotes and strips them per token instead, because a quoted path is still
  a path (I10) — what protects a commit message there is that `git` is not an interpreter. The
  cost of the blanking: a redirect into the artifacts directory written *inside quotes* is
  invisible to the guard.
- The interactive rungs live in **`scripts/demo/`** (`ask.py`, `_listen.py`) and are **not** gated
  (T-0016 ruling D5): they send one request to an engine somebody else already started and produce
  no number anyone will quote, so a `/launch` review per spoken question is the friction that gets
  a hook switched off. The exemption is the *directory*, never a filename — put anything that
  produces a measurement in `scripts/gpu/` instead.
- The path trigger reads the *shape* of the command: quoting a gated path changes nothing
  (`bash "$TOP/scripts/gpu/x3.sh"` is gated, T-0016 I10), `timeout 600 bash <harness>` is gated
  because `timeout` is a wrapper whose duration is consumed with it, and `source`/`.` count as
  running the file (I11).
- **Indirection of any kind defeats the path trigger**, and that is accepted rather than fixed:
  `cat <harness> | bash`, `bash < <harness>`, `xargs`, `find -exec`, a `make` target, an
  `eval "$(cat ...)"`, `docker exec` into a running container, or a symlink pointing at the
  harness all run it without naming it in a program position. The guard is a speed bump against
  unreviewed spend, not a sandbox; if you find yourself reaching for one of these to get a run
  past the hook, that is the moment to ask for the `/launch` review instead.
- Other residual edges, recorded rather than fixed (T-0016 I4-I7): an inline `python -c` inference
  request, a redirect into the artifacts directory written inside quotes, a CDI
  `--device nvidia.com/gpu` container run, and a GPU-spending script kept outside the convention.
