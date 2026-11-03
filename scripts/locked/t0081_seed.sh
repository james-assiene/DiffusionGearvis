#!/usr/bin/env bash
# T-0081 tier 2 (criteria 8-9): one engine-only, plugin-on boot, the seeded-repair driver
# (scripts/gpu/cd_seed_ab.py: BFCL dev n 100, a draft then nine constrained arms per record), then the engine is
# stopped. /launch only; run detached from the T-0081 worktree:
#
#   nohup bash scripts/locked/t0081_seed.sh > /dev/null 2>&1 &
#   echo "exit $?"   # /launch echoes the exit code when the job ends
#
# Engine only (STT, TTS and the conductor stay down), so the engine is started with services/vllm.sh directly
# (the boot appends to logs/vllm.log). The plugin wheel is built from this commit first.
#
# Everything this script prints is teed to logs/t0081-seed-run-<short sha>.log (refused if it exists).
# Refuses (exit 2) a dirty tree (naming the paths), an existing run log or an existing evidence directory, a
# bound Astra port or running containers. The driver refuses (exit 2) a services/vllm.sh changed since 806da12.
# Exit code: the driver's own (0 = run complete, whatever the criterion-9 reading says; 1 = the run stopped on a
# loud error, a failed join, CS = 0 or a crash; 2 = refused by the driver), 1 when the engine never became
# healthy, or 3 when the exit code and the evidence disagree (exit 0 without logs/t0081-seed-<sha>/report.txt
# ending in "run: COMPLETE").
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2

SHA=$(git rev-parse --short=7 HEAD)
DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
  echo "REFUSE: dirty tree: $(printf '%s' "$DIRTY" | tr '\n' ' ')" >&2
  exit 2
fi
mkdir -p logs
LOG="logs/t0081-seed-run-$SHA.log"
EVID="logs/t0081-seed-$SHA"
if [ -e "$LOG" ]; then
  echo "REFUSE: $LOG exists; move it aside first" >&2
  exit 2
fi
if [ -e "$EVID" ]; then
  echo "REFUSE: $EVID exists; move it aside first" >&2
  exit 2
fi

{
  echo "# t0081 seeded repair at $SHA ($(git rev-parse HEAD)), $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free --format=csv,noheader
  uptime
} 2>&1 | tee "$LOG"

if ss -ltn 2>/dev/null | grep -Eq '127\.0\.0\.1:(8001|8002|8080|8081)\b'; then
  echo "REFUSE: an Astra port (8001/8002/8080/8081) is already bound" | tee -a "$LOG" >&2
  exit 2
fi
RUNNING=$(docker ps -q)
DOCKER=$?
if [ "$DOCKER" -ne 0 ]; then
  echo "FAIL: docker ps exited $DOCKER; cannot check for running containers" | tee -a "$LOG" >&2
  exit 1
fi
if [ -n "$RUNNING" ]; then
  echo "REFUSE: containers are running: $(printf '%s' "$RUNNING" | tr '\n' ' ')" | tee -a "$LOG" >&2
  exit 2
fi

echo "## plugin wheel $(date -Is)" | tee -a "$LOG"
uv --directory plugin/astra_vllm_plugin build --wheel 2>&1 | tee -a "$LOG"
WHEEL=${PIPESTATUS[0]}
if [ "$WHEEL" -ne 0 ]; then
  echo "FAIL: plugin wheel build exited $WHEEL" | tee -a "$LOG" >&2
  exit 1
fi

echo "## engine boot $(date -Is)" | tee -a "$LOG"
setsid bash "$PWD/services/vllm.sh" >> logs/vllm.log 2>&1 &
ENGINE_PID=$!

stop_engine() {
  local ids
  ids=$(docker ps -q --filter ancestor=vllm/vllm-openai:v0.29.0 || true)
  if [ -n "$ids" ]; then docker stop $ids > /dev/null || true; fi
  kill "$ENGINE_PID" 2> /dev/null || true
  echo "engine stopped $(date -Is); containers left: $(docker ps -q | wc -l)" | tee -a "$LOG"
}
trap stop_engine EXIT

HEALTHY=0
for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8001/health > /dev/null 2>&1; then
    HEALTHY=1
    break
  fi
  if ! kill -0 "$ENGINE_PID" 2> /dev/null; then
    echo "FAIL: the engine exited during boot; see logs/vllm.log" | tee -a "$LOG" >&2
    exit 1
  fi
  sleep 5
done
if [ "$HEALTHY" -ne 1 ]; then
  echo "FAIL: engine not healthy after 600 s" | tee -a "$LOG" >&2
  exit 1
fi
echo "engine healthy $(date -Is)" | tee -a "$LOG"

echo "## seeded repair driver (cd_seed_ab.py) $(date -Is)" | tee -a "$LOG"
uv run --group cd python scripts/gpu/cd_seed_ab.py --boot-log logs/vllm.log 2>&1 | tee -a "$LOG"
DRIVER=${PIPESTATUS[0]}
echo "driver exit $DRIVER $(date -Is)" | tee -a "$LOG"

REPORT="$EVID/report.txt"
if [ ! -f "$REPORT" ]; then
  if [ "$DRIVER" -eq 0 ]; then
    echo "FAIL: the driver exited 0 but $REPORT does not exist" | tee -a "$LOG" >&2
    exit 3
  fi
  echo "FAIL: driver exit $DRIVER, no $REPORT (refused or crashed before writing evidence)" | tee -a "$LOG" >&2
  exit "$DRIVER"
fi
if [ "$DRIVER" -eq 0 ]; then
  if [ "$(tail -n 1 "$REPORT")" != "run: COMPLETE" ]; then
    echo "FAIL: the driver exited 0 but $REPORT does not end with 'run: COMPLETE'" | tee -a "$LOG" >&2
    exit 3
  fi
  echo "PASS: run complete, evidence $EVID" | tee -a "$LOG"
  grep -E '^(chosen cell|worth wiring)' "$REPORT" | tee -a "$LOG"
  exit 0
fi
echo "FAIL: driver exit $DRIVER; the PROBLEM lines of $REPORT:" | tee -a "$LOG" >&2
grep -E '^PROBLEM: ' "$REPORT" | tee -a "$LOG" >&2
exit "$DRIVER"
