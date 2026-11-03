#!/usr/bin/env bash
# T-0011 tier 3 (criteria 13-14): an engine-only, plugin-on boot with the canvas tap, then the B16
# driver (scripts/gpu/b16_tap.py), then the engine is stopped. /launch only; run detached:
#
#   nohup bash scripts/locked/t0011_tier3.sh > logs/t0011-tier3-run.log 2>&1 &
#
# Engine only (STT, TTS and the conductor stay down: docs/tasks/T-0011.md Regime, tier 3), so the
# engine is started with services/vllm.sh directly rather than `astra up --profile lean`, which
# would also start stt.sh and tts.sh. The boot appends to this worktree's logs/vllm.log; the
# driver slices it to the last boot (R16) before reading the config and the tap's release lines.
set -euo pipefail
cd /home/user/demos/local_astra/.claude/worktrees/t0011

dock() {
  if docker "$@" 2>/dev/null; then return 0; fi
  sg docker -c "docker $(printf '%q ' "$@")"
}

SHA=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSE: dirty tree: $(git status --porcelain | tr '\n' ' ')" >&2; exit 1
fi
if ss -ltn 2>/dev/null | grep -Eq '127\.0\.0\.1:(8001|8002|8080|8081)\b'; then
  echo "REFUSE: an Astra port (8001/8002/8080/8081) is already bound" >&2; exit 1
fi
if [ -n "$(dock ps -q)" ]; then
  echo "REFUSE: containers are running: $(dock ps --format '{{.Names}} {{.Image}}' | tr '\n' ' ')" >&2; exit 1
fi
OUT="logs/t0011-tier3-$SHA"
if [ -e "$OUT" ]; then
  echo "REFUSE: $OUT exists; move it aside first" >&2; exit 1
fi

make plugin-wheel
mkdir -p logs
{ echo "# t0011 tier 3 boot, commit $SHA, $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free --format=csv,noheader
  uptime
} | tee -a logs/vllm.log

setsid bash "$PWD/services/vllm.sh" >> logs/vllm.log 2>&1 &
ENGINE_PID=$!

stop_engine() {
  local ids
  ids=$(dock ps -q --filter ancestor=vllm/vllm-openai:v0.29.0 || true)
  if [ -n "$ids" ]; then dock stop $ids >/dev/null || true; fi
  kill "$ENGINE_PID" 2>/dev/null || true
  echo "engine stopped $(date -Is)"
}
trap stop_engine EXIT

for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then break; fi
  if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "FAIL: the engine exited during boot; see logs/vllm.log" >&2; exit 1
  fi
  sleep 5
done
curl -fsS http://127.0.0.1:8001/health >/dev/null || { echo "FAIL: engine not healthy after 600 s" >&2; exit 1; }
echo "engine healthy $(date -Is)"

uv run python scripts/gpu/b16_tap.py --boot-log logs/vllm.log --profile lean
