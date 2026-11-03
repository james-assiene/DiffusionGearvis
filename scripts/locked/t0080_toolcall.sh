#!/usr/bin/env bash
# T-0080b criterion 6 / T-0080c: one engine-only, plugin-on boot for ONE stage of the two-pass tool-call launch
# (scripts/gpu/cd_toolcall.py: CPU bucket histogram per thought, pass 1 once per case, every pass-2 cell paired,
# the report from the artifact, and for --stage dev the stage-1 selection), then the engine is stopped.
#
#   t0080_toolcall.sh --stage dev
#   t0080_toolcall.sh --stage test --selection logs/t0080-toolcall-dev-<SHA>/selection-dev.json
#
# /launch only; one stage per launch; run detached from a pinned launch worktree (T-0071b C10: detached HEAD at the
# reviewed SHA, nobody commits to it until the job ends). The console log is per SHA and its last line is the
# script's real exit code (C11), not the backgrounding status:
#
#   ( nohup bash scripts/locked/t0080_toolcall.sh --stage dev > logs/t0080-toolcall-dev-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0080-toolcall-dev-<SHA>-run.log ) &
#
# --stage test is refused unless --selection names a selection-dev.json written by THIS SHA's dev stage (its
# "commit" equals HEAD); the test split runs once (T-0080c criterion 7).
#
# Engine only (STT, TTS and the conductor stay down), so the engine is started with services/vllm.sh directly;
# prefix caching is on there (--enable-prefix-caching), and the driver reads the prefix-cache counters from
# /metrics (reported as unavailable, never guessed, if the counters are absent). The boot appends to logs/vllm.log;
# the driver joins the ASTRA_CD records from it and writes the last boot's slice into the evidence directory.
#
# Every early exit prints a REFUSED: (exit 2) or FAIL: (exit 1) line; the driver's outcome is the "driver exit N"
# line and the script's exit code. nvidia-smi lines are informational and never decide the exit code (C11).
set -euo pipefail
# C11: an unexpected failure under set -e still prints a recognisable line before the script exits.
trap 'rc=$?; echo "FAIL: exit $rc at line $LINENO: $BASH_COMMAND" >&2' ERR
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

STAGE=""
SELECTION=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage) STAGE="${2:-}"; shift 2 || { echo "REFUSED: --stage needs a value" >&2; exit 2; } ;;
    --stage=*) STAGE="${1#--stage=}"; shift ;;
    --selection) SELECTION="${2:-}"; shift 2 || { echo "REFUSED: --selection needs a path" >&2; exit 2; } ;;
    --selection=*) SELECTION="${1#--selection=}"; shift ;;
    *) echo "REFUSED: unknown argument $1; usage: $0 --stage dev | --stage test --selection <selection-dev.json>" >&2; exit 2 ;;
  esac
done
case "$STAGE" in
  dev|test) ;;
  *) echo "REFUSED: usage: $0 --stage dev | --stage test --selection <selection-dev.json> (got stage '$STAGE')" >&2; exit 2 ;;
esac

dock() {
  if docker "$@" 2>/dev/null; then return 0; fi
  sg docker -c "docker $(printf '%q ' "$@")"
}

SHA=$(git rev-parse --short=7 HEAD)
HEAD_FULL=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSED: dirty tree: $(git status --porcelain | tr '\n' ' ')" >&2; exit 2
fi
if [ "$STAGE" = test ]; then
  if [ -z "$SELECTION" ] || [ ! -f "$SELECTION" ]; then
    echo "REFUSED: --stage test needs --selection <selection-dev.json> from this SHA's dev stage (got '$SELECTION')" >&2; exit 2
  fi
  SEL_COMMIT=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("commit") or "") if d.get("stage") == "dev" else print("")' "$SELECTION" 2>/dev/null || true)
  if [ "$SEL_COMMIT" != "$HEAD_FULL" ]; then
    echo "REFUSED: $SELECTION is not a dev selection of HEAD $HEAD_FULL (its commit: '${SEL_COMMIT}')" >&2; exit 2
  fi
fi
if [ -n "$(git diff 806da12..HEAD -- services/vllm.sh)" ]; then
  echo "REFUSED: services/vllm.sh changed since 806da12; the reviewed boot regime (prefix caching on) no longer applies" >&2; exit 2
fi
if ss -ltn 2>/dev/null | grep -Eq '127\.0\.0\.1:(8001|8002|8080|8081)\b'; then
  echo "REFUSED: an Astra port (8001/8002/8080/8081) is already bound" >&2; exit 2
fi
if [ -n "$(dock ps -q)" ]; then
  echo "REFUSED: containers are running: $(dock ps --format '{{.Names}} {{.Image}}' | tr '\n' ' ')" >&2; exit 2
fi
OUT="logs/t0080-toolcall-$STAGE-$SHA"
if [ -e "$OUT" ]; then
  echo "REFUSED: $OUT exists; a stage is run once per SHA (move it aside only with the human)" >&2; exit 2
fi
SEL_FLAGS=()
if [ "$STAGE" = test ]; then SEL_FLAGS=(--selection "$SELECTION"); fi

make plugin-wheel || { echo "FAIL: make plugin-wheel failed" >&2; exit 1; }
mkdir -p logs
{ echo "# t0080 tool-call boot (stage $STAGE), commit $SHA, $(date -Is)"
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
uv run --group cd python scripts/gpu/cd_toolcall.py \
  --stage "$STAGE" "${SEL_FLAGS[@]}" \
  --base-url http://127.0.0.1:8001/v1 \
  --engine-log logs/vllm.log --boot-log logs/vllm.log \
  --out-dir "$OUT" --repo "$PWD" --expected-commit "$HEAD_FULL" || DRIVER=$?
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true
echo "report: $OUT/report-$STAGE.md"
if [ "$STAGE" = dev ] && [ -f "$OUT/selection-dev.json" ]; then
  echo "selection: $OUT/selection-dev.json sha256 $(sha256sum "$OUT/selection-dev.json" | cut -d' ' -f1)" || true
fi
echo "driver exit $DRIVER $(date -Is)"
exit "$DRIVER"
