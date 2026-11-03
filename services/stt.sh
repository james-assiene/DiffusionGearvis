#!/usr/bin/env bash
# Kyutai STT via moshi-server (GPU, ~2.5 GiB). Loopback only. Runs through /launch.
#
# Containerised rather than a host binary: the pinned server needs the CUDA toolkit to build and
# the CUDA runtime to run, and this box has neither (.claude/rules/gpu-services.md forbids system
# CUDA changes). services/moshi/Dockerfile builds it; see there for why the CUDA version is
# pinned to exactly 12.8.1 and cannot be moved in either direction.
#
#   make stt-image     # build once (~15 min, 350 crates)
#   services/stt.sh    # run
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${ASTRA_MOSHI_IMAGE:-astra-moshi:0.6.4}
HF_HOME_HOST=${HF_HOME:-$HOME/.cache/huggingface}

# Run the WHOLE script under `sg docker -c` when the session lacks the docker group — which is the
# normal case here, since the group was added after login. Sprinkling `sg` inside would need it
# twice with fragile quoting. Distinguish "cannot reach the daemon" from "image is missing":
# conflating them printed "build it first" for an image that existed, which is a 15-minute lie.
if ! docker info >/dev/null 2>&1; then
  echo "cannot reach the docker daemon. Run the whole script under the docker group:" >&2
  echo "  sg docker -c '$0'" >&2
  exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "no $IMAGE — build it first: make stt-image" >&2
  exit 1
fi

# --network host so the server binds 127.0.0.1:8080 in the host namespace, matching vllm.sh and
# the loopback-only rule. The HF cache is mounted rw because the weights named by hf:// paths in
# the config are fetched on first run; everything after that is a cache hit.
# --name so there is a stop command that is not `docker ps | grep`: docker stop astra-stt.
#
# The HF cache is mounted READ-ONLY, and that ":ro" is the load-bearing part. Weights are
# pre-fetched as the user (snapshot_download), because a container writing this cache as root is
# what broke pocket-tts earlier today. HF_HUB_OFFLINE is kept only for parity with vllm.sh and
# HAS NO EFFECT HERE: it is a Python huggingface_hub variable, and this server is Rust using the
# hf-hub crate, which has no offline handling at all. An earlier version of this comment claimed
# the env var prevented a silent download; it does not, and with --network host a cache miss would
# have fetched multiple GB as root. :ro is what actually makes a miss fail, in seconds, loudly.
# /logs is mounted because the config's log_dir must be absolute — TOML does no shell expansion,
# so a "$HOME/..." value creates a directory literally named $HOME at container root, which --rm
# then deletes, silently losing the ASR timing logs B3 needs.
# ASTRA_STT_CPU=1 runs without a CUDA device: the GPU start is blocked until vLLM's footprint
# leaves >= 3 GiB free with STT resident (docs/02 §7). --gpus all stays regardless, because the
# binary links libcuda.so.1 unconditionally and will not even print --help without it.
CPU_FLAG=""
if [ "${ASTRA_STT_CPU:-0}" = "1" ]; then CPU_FLAG="--cpu"; fi
mkdir -p "$HOME/tmp/astra-stt-logs"

exec docker run --rm --name astra-stt --gpus all --network host \
  -e HF_HUB_OFFLINE=1 \
  -v "$HF_HOME_HOST:/root/.cache/huggingface:ro" \
  -v "$HERE/moshi:/config:ro" \
  -v "$HOME/tmp/astra-stt-logs:/logs" \
  "$IMAGE" worker --config /config/stt-1b-en_fr.toml --addr 127.0.0.1 --port 8080 $CPU_FLAG
