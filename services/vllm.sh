#!/usr/bin/env bash
# Start vLLM for DiffusionGemma with the pinned container and flags (docs/02 §2.4, docs/10 §9).
# Runs only through /launch (guard_bash.sh checks the token). Loopback only. No network inside
# the container: models come from the HF cache filled by `astra fetch-models`, the plugin from a
# prebuilt wheel (`make plugin-wheel`).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
CONTAINER=${ASTRA_VLLM_CONTAINER:-vllm/vllm-openai:v0.29.0}
MODEL=${ASTRA_MODEL:-nvidia/diffusiongemma-26B-A4B-it-NVFP4}
PLUGIN=${ASTRA_PLUGIN:-1}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

PLUGIN_FLAGS=""
INSTALL_PLUGIN=":"
if [ "$PLUGIN" = "1" ]; then
  WHEEL=$(ls "$ROOT"/plugin/astra_vllm_plugin/dist/astra_vllm_plugin-*.whl 2>/dev/null | tail -1 || true)
  [ -n "$WHEEL" ] || { echo "no plugin wheel; run 'make plugin-wheel' (or ASTRA_PLUGIN=0 for the stock model)" >&2; exit 1; }
  # Must match ARCHITECTURE in the plugin's __init__.py. Renamed from
  # AstraDiffusionGemmaForCausalLM, which named a class v0.29.0 does not have.
  #
  # LOAD-BEARING, do not "simplify" the flags below while this override is active.
  # vLLM's MODELS_CONFIG_MAP (models/config.py:1006) is keyed by the literal architecture
  # string, so overriding it means DiffusionGemmaModelForBlockDiffusionConfig
  # .verify_and_update_config() NEVER RUNS. Today that changes nothing, because every branch
  # it would apply is already falsified by a flag we pass explicitly:
  #   - FA4/Triton selection and the use_non_causal branch need attention_config.backend is None
  #     -> we pass --attention-backend TRITON_ATTN
  #   - diffusion_config auto-creation                 -> we pass --diffusion-config
  #   - the max_num_seqs -> 8 default needs >= DEFAULT  -> we pass 3
  #   - the max_new_tokens injection                    -> we pass --override-generation-config
  # Drop any one of those and the plugin-on boot silently diverges from plugin-off. The
  # diffusion path itself is safe: is_diffusion keys off hf_config.canvas_length
  # (config/model.py:1811), not the architecture string, and --hf-overrides leaves it alone.
  PLUGIN_FLAGS="--hf-overrides '{\"architectures\": [\"AstraDiffusionGemma\"]}'"
  INSTALL_PLUGIN="pip install --no-deps --no-index -q /plugin/dist/$(basename "$WHEEL")"
fi

# A custom generation config dir (the model's generation_config.json with our overrides, written by
# `astra fetch-models`) replaces the sampler constants; without it vLLM uses the model's own file.
GEN_FLAGS="--override-generation-config '{\"max_new_tokens\": null}'"
if [ -f "$ROOT/services/generation/generation_config.json" ]; then
  GEN_FLAGS="--generation-config /config/generation $GEN_FLAGS"
fi

# --- the startup gate, and the refusal that costs a second instead of 21 s (T-0058) -------------
#
# On 2026-09-16 22:17 this script started the engine on a box where the recogniser had moved onto
# the GPU, and the container died 21 s in, before a weight was loaded
# (logs/vllm-coldboot-73a85dd.log):
#
#   ValueError: Free memory on device cuda:0 (27.23/31.33 GiB) on startup is less than desired
#   GPU memory utilization (0.92, 28.83 GiB).
#
# The script had never passed --gpu-memory-utilization, so vLLM's v0.29.0 default of 0.92 applied
# and the engine demanded 0.92 of the whole device be FREE before it loaded anything. With the
# recogniser resident only 27.23 GiB was, so every start in that configuration failed; the engine
# that had been up for 23 h was started on an empty card and could not have been restarted where
# it stood. Boot order is engine first, recogniser second (`astra up` already does that).
#
# What the flag does here, because the value is meaningless without it: with
# --kv-cache-memory-bytes set, GPUWorker.determine_available_memory returns those bytes and logs
# that it "does not respect the gpu_memory_utilization config". So on this path the flag gates
# ONLY the startup free-memory check -- it sizes no arena, does not shrink the KV pool, and gives
# the sampler neither more room nor less. The single thing it costs is guard strength, which is
# why the value has a floor as well as a ceiling. All figures are MiB, read from
# logs/vram-coldboot-73a85dd.before.txt and the ValueError above (2026-09-16, commit 73a85dd):
#
#   ceiling  27882/32082 = 0.869   free at the engine's own check over vLLM's own total. Above
#                                  this the engine cannot start with the recogniser resident.
#   floor    27434/32082 = 0.855   the engine's measured steady footprint at KV 4 GiB. A demand
#                                  BELOW what the engine actually uses stops being a guard: the
#                                  startup check then passes boots that die later and deeper.
#   chosen   0.86 -> demands 27591: 291 below the free memory measured with the recogniser up,
#                                  157 above the footprint. (0.85 was rejected for being 164
#                                  below the footprint; its margin is margin on the check, not on
#                                  the physics.) docs/tasks/T-0058.md carries the derivation.
#
# NB the two denominators: vLLM computes against mem_get_info's ~32082 MiB (31.33 GiB), not
# nvidia-smi's 32607 (31.84). At 0.86 the difference is 452 MiB -- enough to refuse a box that
# boots -- while the two FREE readings agree exactly. The guard below therefore uses 32082, and
# subtracts the ~512 MiB of CUDA context vLLM creates before it looks (28391 -> 27882 on that
# run, n = 1, rounded up and deliberately not tuned).
#
# The refusal is a speed bump, not a sandbox: it saves ~20 s and explains a failure, and the
# engine's own check remains the authority. So every unknown -- no nvidia-smi, a driver error,
# an empty answer, [N/A], an unexpected unit -- warns once and PROCEEDS. A launcher that refused
# to start on an unreadable status query would brick every box that is not this one. It runs on
# the dry-run path too: a guard a dry run skips is a guard nothing can test.
VLLM_TOTAL_MIB=32082            # vLLM's own denominator (mem_get_info), NOT nvidia-smi's 32607
ENGINE_INIT_OVERHEAD_MIB=512    # the CUDA context vLLM creates before its check (n = 1)
# Keep this equal to the --gpu-memory-utilization in the argv below; both are pinned to the
# 0.855-0.869 window by tests/unit/test_vllm_launch_flags.py and
# tests/contract/test_vllm_sh_start_guard.py, so a drift that matters reddens on both sides.
STARTUP_GATE_UTILIZATION=0.86

# The driver's own memory.free column, in MiB, or a non-zero exit when it cannot be trusted.
# free is NOT total-used on this device: the columns differ by 520 MiB, in the direction that
# costs a boot. `|| true` because an nvidia-smi that exits 1 inside a command substitution would
# otherwise take the whole script with it under `set -e`.
startup_gate_free_mib() {
  local answer
  answer=$(timeout 10 nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]') || true
  [[ "$answer" =~ ^[0-9]+$ ]] || return 1
  printf '%s' "$answer"
}

# ceil(utilisation x total), which is what vllm/v1/worker/utils.py::request_memory compares.
startup_gate_demand_mib() {
  awk -v util="$STARTUP_GATE_UTILIZATION" -v total="$VLLM_TOTAL_MIB" \
    'BEGIN { want = util * total; whole = int(want); if (want > whole) whole += 1; print whole }'
}

if FREE_MIB=$(startup_gate_free_mib) && DEMAND_MIB=$(startup_gate_demand_mib) &&
  [[ "$DEMAND_MIB" =~ ^[0-9]+$ ]]; then
  # 10# forces base ten: a driver that ever answered "08" would otherwise be an octal syntax
  # error, and under `set -e` that is the guard failing *closed*, which is the one thing it
  # must never do.
  PREDICTED_FREE_MIB=$((10#$FREE_MIB - ENGINE_INIT_OVERHEAD_MIB))
  if [ "$PREDICTED_FREE_MIB" -lt "$DEMAND_MIB" ]; then
    HOLDERS=$(timeout 10 nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
      --format=csv,noheader 2>/dev/null | cut -c1-160 || true)
    [ -n "$HOLDERS" ] || HOLDERS="(nvidia-smi listed no compute apps)"
    {
      echo "services/vllm.sh: refusing to start the engine; it would die at its own startup check."
      echo "  free now, from the driver's memory.free column: $FREE_MIB MiB"
      echo "  minus ~$ENGINE_INIT_OVERHEAD_MIB MiB for the CUDA context vLLM creates before it looks: $PREDICTED_FREE_MIB MiB"
      echo "  demanded at --gpu-memory-utilization $STARTUP_GATE_UTILIZATION of vLLM's own $VLLM_TOTAL_MIB MiB total: $DEMAND_MIB MiB"
      echo "  short by $((DEMAND_MIB - PREDICTED_FREE_MIB)) MiB"
      echo "holding device memory now (pid, process, MiB):"
      printf '  %s\n' "$HOLDERS"
      echo "Boot order is engine first, recogniser second. Stop the recogniser (Kyutai STT,"
      echo "moshi-server: uv run astra down, or stop the astra-stt container), start the engine,"
      echo "then start the recogniser again -- or keep it off the card with ASTRA_STT_CPU=1."
      echo "Refused before the container started; the engine's own check takes 21 s to say this"
      echo "(logs/vllm-coldboot-73a85dd.log, 2026-09-16 22:17)."
    } >&2
    exit 1
  fi
else
  echo "services/vllm.sh: could not read free VRAM from nvidia-smi; starting anyway -- the engine's own startup check (21 s) decides" >&2
fi

DOCKER=docker
[ "${ASTRA_DRY_RUN:-0}" = "1" ] && DOCKER="printf %s\n"   # one argv element per line
# --kv-cache-memory-bytes is parsed by vllm.utils.argparse_utils.human_readable_int, which takes a
# SINGLE-letter suffix: lowercase is decimal (2g = 2e9), uppercase is binary (2G = 2^30). "2GiB" —
# the spelling in versions.lock and .claude/rules/gpu-services.md — matches neither and dies in
# argparse. Spelled in bytes here so a g/G slip cannot silently cost 7% of the KV cache.
# --kv-cache-dtype bfloat16 is REQUIRED, not a default. The checkpoint's quantization_config
# declares kv_cache_scheme num_bits=8, so under `auto` vLLM picks fp8_e4m3 and — since the
# checkpoint ships no k/v scales — runs it at scaling factor 1.0. That violates docs/01 R2
# (bf16 KV in v1) and risks silent accuracy loss.
#
# CORRECTION (T-0014). The comment committed here in 62f6946 claimed "2 GiB holds 54,931 tokens at
# fp8, so bf16 halves it to ~27,465, giving 1.68x at max-model-len 16384". That was wrong twice
# over. The logged "GPU KV cache size: N tokens" is computed as int(max_concurrency *
# max_model_len) — it is derived FROM max_model_len, not a capacity independent of it. And KV here
# is affine, not proportional: 25 of 30 layers are sliding-window capped at 1024, so their cost is
# a CONSTANT that halving max_model_len does not reduce; only the 5 full-attention layers scale.
# At 2 GiB the real bf16 ceiling is 8,720 tokens, so 16384 could not have booted at all.
#
# Modelled with vLLM's own allocator against the real config (no weights loaded), bf16:
#   2 GiB: 4096 -> 2.13x | 8192 -> 1.06x | 12288+ -> fails, max len 8720
#   4 GiB: 4096 -> 4.25x | 8192 -> 2.13x | 16384 -> 1.68x | 32768 -> 1.33x
# Trust the POOL axis, not the length axis. The 32768 cell is refuted: 2 GiB at fp8 and 4 GiB at
# bf16 are byte-identical capacities, and the 2026-09-13 fp8 boot logged 1.68x there, not 1.33x.
# The real 4 GiB @ 16384 boot (logs/vllm-bf16-4gib.log) gave 1.93x / 31,605 tokens, VRAM 26.8 GiB.
# Note the token figure FELL from 54,931 (@32768) on the same bytes: "GPU KV cache size" is
# int(concurrency * max_model_len), derived from the length. It is not a capacity; never size a
# budget from it. 8,720 tokens (the 2 GiB bf16 ceiling) would not have held 4 images (1120-token
# bucket each = 4,480) plus history and a 256 canvas, which is what rules out the 2 GiB pool.
# --network host: the API binds to 127.0.0.1 and the plugin's tap publishes on 127.0.0.1:8002.
exec $DOCKER run --rm --gpus all --ipc=host --network host \
  -v "$HF_HOME:/root/.cache/huggingface" \
  -v "$ROOT/plugin/astra_vllm_plugin:/plugin:ro" \
  -v "$ROOT/services/generation:/config/generation:ro" \
  -e HF_HUB_OFFLINE=1 -e ASTRA_TAP_ENDPOINT=tcp://127.0.0.1:8002 \
  --entrypoint bash "$CONTAINER" -c "$INSTALL_PLUGIN && exec vllm serve $MODEL \
    --host 127.0.0.1 --port 8001 \
    --max-model-len 16384 --max-num-seqs 3 --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.86 \
    --kv-cache-memory-bytes 4294967296 --kv-cache-dtype bfloat16 --enable-prefix-caching \
    --attention-backend TRITON_ATTN \
    --diffusion-config '{\"canvas_length\": 256, \"max_denoising_steps\": 48}' \
    $GEN_FLAGS \
    --reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4 \
    --limit-mm-per-prompt '{\"image\": 4, \"video\": 1}' \
    $PLUGIN_FLAGS"
