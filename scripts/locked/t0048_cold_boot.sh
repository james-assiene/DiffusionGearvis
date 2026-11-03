#!/usr/bin/env bash
# T-0048 ruling A13 cold-engine test: boot the engine with the recogniser already resident.
#
# Approved by launch review 2026-09-16. The pre-flight refusals are the point of this file:
# the experiment is only valid if the recogniser is holding GPU memory when the engine starts,
# and the cheapest way to get a meaningless result is to boot into an empty GPU and not notice.
#
# Orchestrator amendment to the reviewed text: every docker call goes through dock(), because
# this box needs `sg docker` for group membership and a bare `docker ps` under `set -e` would
# abort the script as though the recogniser were absent. Behaviour is otherwise unchanged.
set -euo pipefail
cd /home/user/demos/local_astra

dock() {
  if docker "$@" 2>/dev/null; then return 0; fi
  sg docker -c "docker $(printf '%q ' "$@")"
}

SHA=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSE: dirty tree: $(git status --porcelain | tr '\n' ' ')" >&2; exit 1
fi

# Hard stop 1 - the case under test is "cold engine, recogniser ALREADY on the GPU".
if ! dock ps --format '{{.Names}}' | grep -qx astra-stt; then
  echo "REFUSE: astra-stt is not running; booting into an empty GPU is not this test" >&2; exit 1
fi
if ! nvidia-smi --query-compute-apps=process_name,used_memory --format=csv,noheader | grep -q moshi-server; then
  echo "REFUSE: moshi-server holds no GPU memory (STT on CPU?); not the case under test" >&2; exit 1
fi
FREE_BEFORE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$FREE_BEFORE" -ge 29500 ]; then
  echo "REFUSE: ${FREE_BEFORE} MiB free - too empty for a 3 GiB recogniser to be resident" >&2; exit 1
fi

# Hard stop 2 - 8001 must be free, or the model loads for 90 s and then dies on bind
# (exactly what logs/vllm.log:2695-2730 records for the 2026-09-16 00:11 attempt).
if ss -ltn 2>/dev/null | grep -q '127\.0\.0\.1:8001'; then
  echo "REFUSE: 127.0.0.1:8001 is already bound" >&2; exit 1
fi

LOG="logs/vllm-coldboot-$SHA.log"
CSV="logs/vram-coldboot-$SHA.csv"
if [ -e "$LOG" ] || [ -e "$CSV" ]; then
  echo "REFUSE: $LOG or $CSV exists; move them aside first" >&2; exit 1
fi

{ echo "# commit $SHA  $(date -Is)"
  nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
  dock ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
} > "logs/vram-coldboot-$SHA.before.txt"
cat "logs/vram-coldboot-$SHA.before.txt"

nvidia-smi --query-gpu=timestamp,memory.used,memory.free --format=csv,noheader -lms 200 > "$CSV" 2>&1 &
echo $! > "logs/vram-coldboot-$SHA.smi.pid"

setsid bash "$PWD/services/vllm.sh" > "$LOG" 2>&1 &
echo "engine launched detached. log=$LOG  vram=$CSV  sampler pid=$(cat logs/vram-coldboot-$SHA.smi.pid)"
echo "watch: tail -f $LOG"
