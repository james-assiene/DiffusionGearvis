#!/usr/bin/env bash
# T-0071a a2 (criteria 3-5): one engine-only, plugin-on boot (Tap ∘ CD ∘ stock), the B13 driver
# (scripts/gpu/cd_b13.py: tap warm-up, gated strata 64/128, report-only strata 256/512, then the runner's
# limit-10 dev runs), then the engine is stopped. /launch only; run detached from the T-0071a worktree:
#
#   nohup bash scripts/locked/t0071a_b13.sh > logs/t0071a-b13-run.log 2>&1 &
#   echo "exit $?"   # /launch echoes the exit code when the job ends
#
# Engine only (STT, TTS and the conductor stay down: docs/tasks/T-0071.md Regime), so the engine is started
# with services/vllm.sh directly rather than `astra up --profile lean`. The boot appends to this worktree's
# logs/vllm.log; the driver slices it to the last boot (R16).
#
# Exit code: the driver's (0 run complete and valid, whatever B13's verdict -- D-H1; 1 a run defect; 2 refused).
# The evidence and the report are in logs/t0071a-b13-<sha>/ (report.txt is written before the exit code).
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

dock() {
  if docker "$@" 2>/dev/null; then return 0; fi
  sg docker -c "docker $(printf '%q ' "$@")"
}

SHA=$(git rev-parse --short=7 HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSE: dirty tree: $(git status --porcelain | tr '\n' ' ')" >&2; exit 2
fi
if [ -n "$(git diff 806da12..HEAD -- services/vllm.sh)" ]; then
  echo "REFUSE: services/vllm.sh changed since 806da12; the T-0069/T-0011 boot regime no longer applies" >&2; exit 2
fi
if ss -ltn 2>/dev/null | grep -Eq '127\.0\.0\.1:(8001|8002|8080|8081)\b'; then
  echo "REFUSE: an Astra port (8001/8002/8080/8081) is already bound" >&2; exit 2
fi
if [ -n "$(dock ps -q)" ]; then
  echo "REFUSE: containers are running: $(dock ps --format '{{.Names}} {{.Image}}' | tr '\n' ' ')" >&2; exit 2
fi
OUT="logs/t0071a-b13-$SHA"
if [ -e "$OUT" ]; then
  echo "REFUSE: $OUT exists; move it aside first" >&2; exit 2
fi

make plugin-wheel
mkdir -p logs
{ echo "# t0071a a2 B13 boot, commit $SHA, $(date -Is)"
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
  echo "containers left: $(dock ps -q | wc -l)"
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

set +e
uv run --group cd python scripts/gpu/cd_b13.py --boot-log logs/vllm.log --profile lean
DRIVER=$?
set -e
echo "driver exit $DRIVER $(date -Is)"
exit "$DRIVER"
