#!/usr/bin/env bash
# T-0082b criterion 1 / T-0082c: one engine-only, plugin-on boot for ONE pass-2 replay of an earlier artifact's
# stored pass-1 rows (scripts/gpu/cd_toolcall.py --pass1-from/--pass1-sha256: no pass-1 request, the stored rows
# re-scored under the current router, only the cells' pass-2 requests sent), then the engine is stopped.
#
#   t0082_replay.sh --stage dev|test --pass1-from <artifact.jsonl> --pass1-sha256 <hex>
#
# /launch only; one stage per launch; run detached from a pinned launch worktree (T-0071b C10: detached HEAD at the
# reviewed SHA, nobody commits to it until the job ends). The console log is per SHA and its last line is the
# script's real exit code (C11), not the backgrounding status:
#
#   ( nohup bash scripts/locked/t0082_replay.sh --stage test --pass1-from <P> --pass1-sha256 <H> > logs/t0082-replay-test-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0082-replay-test-<SHA>-run.log ) &
#
# No --cell is taken or passed: the driver's REPLAY_CELLS apply (ruling R2). No selection file is needed or written
# (ruling R6). The artifact's sha256 is checked here, before the boot (ruling R1), and again by the driver, which
# also refuses a stage, regime or stopped mismatch before any request. Its out-dir is logs/t0082-replay-<stage>-<SHA>
# and is refused if it exists: a stage is replayed once per SHA.
#
# Engine only (STT, TTS and the conductor stay down), so the engine is started with services/vllm.sh directly, as
# scripts/locked/t0080_toolcall.sh does. The boot appends to logs/vllm.log; the driver joins the ASTRA_CD records
# from it and writes the last boot's slice into the evidence directory.
#
# Every early exit prints a REFUSED: (exit 2) or FAIL: (exit 1) line; the driver's outcome is the "driver exit N"
# line and the script's exit code. nvidia-smi lines are informational and never decide the exit code (C11).
set -euo pipefail
# C11: an unexpected failure under set -e still prints a recognisable line before the script exits.
trap 'rc=$?; echo "FAIL: exit $rc at line $LINENO: $BASH_COMMAND" >&2' ERR
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

USAGE="usage: $0 --stage dev|test --pass1-from <artifact.jsonl> --pass1-sha256 <hex>"
STAGE=""
PASS1_FROM=""
PASS1_SHA256=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage) STAGE="${2:-}"; shift 2 || { echo "REFUSED: --stage needs a value" >&2; exit 2; } ;;
    --stage=*) STAGE="${1#--stage=}"; shift ;;
    --pass1-from) PASS1_FROM="${2:-}"; shift 2 || { echo "REFUSED: --pass1-from needs a path" >&2; exit 2; } ;;
    --pass1-from=*) PASS1_FROM="${1#--pass1-from=}"; shift ;;
    --pass1-sha256) PASS1_SHA256="${2:-}"; shift 2 || { echo "REFUSED: --pass1-sha256 needs a value" >&2; exit 2; } ;;
    --pass1-sha256=*) PASS1_SHA256="${1#--pass1-sha256=}"; shift ;;
    *) echo "REFUSED: unknown argument $1; $USAGE" >&2; exit 2 ;;
  esac
done
case "$STAGE" in
  dev|test) ;;
  *) echo "REFUSED: $USAGE (got stage '$STAGE')" >&2; exit 2 ;;
esac
if [ -z "$PASS1_FROM" ] || [ -z "$PASS1_SHA256" ]; then
  echo "REFUSED: $USAGE (--pass1-from and --pass1-sha256 are both required)" >&2; exit 2
fi
if [ ! -f "$PASS1_FROM" ]; then
  echo "REFUSED: --pass1-from $PASS1_FROM is not a file" >&2; exit 2
fi
# R1: the artifact's sha256 is checked here, before the boot
ART_SHA=$(sha256sum "$PASS1_FROM" | cut -d' ' -f1)
if [ "$ART_SHA" != "$PASS1_SHA256" ]; then
  echo "REFUSED: $PASS1_FROM has sha256 $ART_SHA, not the pinned $PASS1_SHA256" >&2; exit 2
fi

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
  echo "REFUSED: services/vllm.sh changed since 806da12; the reviewed boot regime (prefix caching on) no longer applies" >&2; exit 2
fi
if ss -ltn 2>/dev/null | grep -Eq '127\.0\.0\.1:(8001|8002|8080|8081)\b'; then
  echo "REFUSED: an Astra port (8001/8002/8080/8081) is already bound" >&2; exit 2
fi
if [ -n "$(dock ps -q)" ]; then
  echo "REFUSED: containers are running: $(dock ps --format '{{.Names}} {{.Image}}' | tr '\n' ' ')" >&2; exit 2
fi
OUT="logs/t0082-replay-$STAGE-$SHA"
if [ -e "$OUT" ]; then
  echo "REFUSED: $OUT exists; a stage is replayed once per SHA (move it aside only with the human)" >&2; exit 2
fi

make plugin-wheel || { echo "FAIL: make plugin-wheel failed" >&2; exit 1; }
mkdir -p logs
{ echo "# t0082 replay boot (stage $STAGE, pass 1 from $PASS1_FROM sha256 $PASS1_SHA256), commit $SHA, $(date -Is)"
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
  --stage "$STAGE" --pass1-from "$PASS1_FROM" --pass1-sha256 "$PASS1_SHA256" \
  --base-url http://127.0.0.1:8001/v1 \
  --engine-log logs/vllm.log --boot-log logs/vllm.log \
  --out-dir "$OUT" --repo "$PWD" --expected-commit "$HEAD_FULL" || DRIVER=$?
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true
echo "report: $OUT/report-$STAGE.md"
echo "driver exit $DRIVER $(date -Is)"
exit "$DRIVER"
