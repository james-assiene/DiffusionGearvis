#!/usr/bin/env bash
# T-0071b (criteria 1-4): one engine-only, plugin-on boot (Tap ∘ CD ∘ stock) for ONE task, the campaign driver
# (scripts/gpu/cd_campaign.py: CPU bucket histogram, unconstrained stack check on the dev cut, the 18-cell grid
# per calibrated arm, freeze, each arm's SC twin at its chosen values), then the engine is stopped.
# /launch only; one task per launch; run detached from the pinned launch worktree (ruling C10:
# .claude/worktrees/t0071b-launch, detached HEAD at the reviewed SHA, nobody commits to it until the job ends).
# The console log is per SHA and its last line is the script's real exit code (C11), not the backgrounding status:
#
#   ( nohup bash scripts/locked/t0071b_calib.sh sudoku perstep_mar perstep_mar_qargmax > logs/t0071b-calib-sudoku-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0071b-calib-sudoku-<SHA>-run.log ) &   # b1
#   ( nohup bash scripts/locked/t0071b_calib.sh bfcl perstep_mar perstep_mar_qargmax > logs/t0071b-calib-bfcl-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0071b-calib-bfcl-<SHA>-run.log ) &       # b2
#
# Further arguments after the task are calibrated arms, each one of calibration.GRID_ARMS; the default is both,
# perstep_mar perstep_mar_qargmax, in one boot (ruling C6; budgets per arm: Sudoku 4 h, BFCL 6 h). An SC twin
# (perstep_mar_sc, perstep_mar_qargmax_sc) is refused before the boot (C4): it reuses its base arm's file and is
# run and reported by the driver at that arm's chosen values (C5).
# Each arm is frozen separately: perstep_mar at <evidence>/calibration-<task>.json, any other arm at
# <evidence>/calibration-<task>-<arm>.json. The frozen files are written under logs/ (gitignored) so the
# tree stays clean for every runner call of the launch; after review they are copied byte for byte to
# docs/cd/ and committed (the sha256 in the report must match the committed file; C8).
#
# Every early exit prints a REFUSED: (exit 2) or FAIL: (exit 1) line; the driver's outcome is the
# "driver exit N" line. nvidia-smi lines are informational and never decide the exit code (C11).
#
# Engine only (STT, TTS and the conductor stay down: docs/tasks/T-0071.md Regime), so the engine is started
# with services/vllm.sh directly. The boot appends to this worktree's logs/vllm.log; the driver reads the CD
# records from it and writes the last boot's slice (R16) into the evidence directory. The driver reads no tap
# frames (the runner joins on ASTRA_CD engine-log records), so there is no tap warm-up (T-0069 O7 applies only
# to frame readers).
#
# Exit code: the driver's (0 every requested arm frozen; 1 a stop, no eligible cell or a crash; 2 refused).
# The report (<evidence>/campaign-<task>.json) is written before the exit code in every outcome.
set -euo pipefail
# C11: an unexpected failure under set -e still prints a recognisable line before the script exits.
trap 'rc=$?; echo "FAIL: exit $rc at line $LINENO: $BASH_COMMAND" >&2' ERR
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

TASK="${1:-}"
case "$TASK" in
  sudoku|bfcl) ;;
  *) echo "REFUSED: usage: $0 <sudoku|bfcl> [arm ...] (one task per launch; the test cut is never run)" >&2; exit 2 ;;
esac
shift
ARMS=("$@")
if [ "${#ARMS[@]}" -eq 0 ]; then ARMS=(perstep_mar perstep_mar_qargmax); fi
ARM_FLAGS=()
for a in "${ARMS[@]}"; do
  case "$a" in
    perstep_mar|perstep_mar_qargmax) ;;
    *) echo "REFUSED: arm $a is not one of calibration.GRID_ARMS (perstep_mar perstep_mar_qargmax); an SC twin is reported, never calibrated (C4)" >&2; exit 2 ;;
  esac
  ARM_FLAGS+=(--arm "$a")
done

dock() {
  if docker "$@" 2>/dev/null; then return 0; fi
  sg docker -c "docker $(printf '%q ' "$@")"
}

SHA=$(git rev-parse --short=7 HEAD)
HEAD_FULL=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSED: dirty tree: $(git status --porcelain | tr '\n' ' ')" >&2; exit 2
fi
if [ -n "$(git diff 806da12..HEAD -- services/vllm.sh)" ]; then
  echo "REFUSED: services/vllm.sh changed since 806da12; the T-0069/T-0071a boot regime no longer applies" >&2; exit 2
fi
if ss -ltn 2>/dev/null | grep -Eq '127\.0\.0\.1:(8001|8002|8080|8081)\b'; then
  echo "REFUSED: an Astra port (8001/8002/8080/8081) is already bound" >&2; exit 2
fi
if [ -n "$(dock ps -q)" ]; then
  echo "REFUSED: containers are running: $(dock ps --format '{{.Names}} {{.Image}}' | tr '\n' ' ')" >&2; exit 2
fi
OUT="logs/t0071b-calib-$TASK-$SHA"
if [ -e "$OUT" ]; then
  echo "REFUSED: $OUT exists; move it aside first" >&2; exit 2
fi
for a in "${ARMS[@]}"; do
  if [ "$a" = perstep_mar ]; then F="docs/cd/calibration-$TASK.json"; else F="docs/cd/calibration-$TASK-$a.json"; fi
  if [ -e "$F" ]; then
    echo "REFUSED: $F is already committed; a frozen calibration is never re-run without the human" >&2; exit 2
  fi
done

make plugin-wheel
mkdir -p logs
{ echo "# t0071b calibration boot ($TASK; arms ${ARMS[*]}), commit $SHA, $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free --format=csv,noheader || true
  uptime || true
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
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory after boot: /' || true

# the driver's status is captured in an || list, so neither set -e nor the ERR trap fires on it
DRIVER=0
uv run --group cd python scripts/gpu/cd_campaign.py \
  --task "$TASK" --cut dev "${ARM_FLAGS[@]}" \
  --base-url http://127.0.0.1:8001/v1 \
  --engine-log logs/vllm.log --boot-log logs/vllm.log \
  --out-dir "$OUT" --repo "$PWD" --expected-commit "$HEAD_FULL" \
  --freeze-path "$OUT/calibration-$TASK.json" \
  --probe-diffusion || DRIVER=$?
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true
echo "report: $OUT/campaign-$TASK.json"
if [ -f "$OUT/campaign-$TASK.json" ]; then
  python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print("stopped:", r.get("stopped")); print("frozen:", r.get("frozen_by_arm"))' "$OUT/campaign-$TASK.json" || true
fi
echo "driver exit $DRIVER $(date -Is)"
exit "$DRIVER"
