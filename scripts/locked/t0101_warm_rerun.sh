#!/usr/bin/env bash
# T-0101: the pre-registered confirmation (H-T101b). One fresh engine-only boot in the regime of
# scripts/locked/t0099_repair_live.sh (services/vllm.sh at 806da12, prefix caching on, plugin wheel rebuilt), then
#   1. scripts/gpu/t0099_repair_live.py --warm-repair: one warm-up repair per payload automaton (tap off, ids
#      warmup-...), then arm S (30 stress cases x k = 3, repetition-major) and arm F (40 dev tool cases, the repair
#      forced on valid drafts), typed turns, concurrency 1, no seed; written to logs/t0101-warm-rerun-<SHA>/;
#   2. tests/eval/test_t0101_warm_rerun.py: the report re-derived from the raw files, G1-G8, criterion 6, C1-C3
#      asserted and C4 printed (no GPU time of its own; T0101_RUN_DIR points it at the run; only if report.json exists);
# then the engine is stopped. /launch only, from a clean tree; the console log's last line is the real exit code:
#
#   ( nohup bash scripts/locked/t0101_warm_rerun.sh > logs/t0101-warm-rerun-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0101-warm-rerun-<SHA>-run.log ) &
#
# The proxy variables are unset for every step (loopback only). Exit: 0 all passed; 1 a step failed; 2 refused.
set -euo pipefail
trap 'rc=$?; echo "FAIL: exit $rc at line $LINENO: $BASH_COMMAND" >&2' ERR
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

if [ "$#" -ne 0 ]; then echo "REFUSED: no arguments expected (got $*)" >&2; exit 2; fi

dock() {
  if docker "$@" 2>/dev/null; then return 0; fi
  sg docker -c "docker $(printf '%q ' "$@")"
}

SHA=$(git rev-parse --short=7 HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSED: dirty tree: $(git status --porcelain | tr '\n' ' ')" >&2; exit 2
fi
if [ -n "$(git diff 806da12..HEAD -- services/vllm.sh)" ]; then
  echo "REFUSED: services/vllm.sh changed since 806da12; the reviewed boot regime (prefix caching on) no longer applies" >&2; exit 2
fi
if ss -ltn 2>/dev/null | grep -Eq '127\.0\.0\.1:(8001|8002|8080|8081)\b'; then
  echo "REFUSED: an Astra port (8001/8002/8080/8081) is already bound (a fresh boot is needed)" >&2; exit 2
fi
if [ -n "$(dock ps -q)" ]; then
  echo "REFUSED: containers are running: $(dock ps --format '{{.Names}} {{.Image}}' | tr '\n' ' ')" >&2; exit 2
fi
OUT="logs/t0101-warm-rerun-$SHA"
AUX="logs/t0101-launch-$SHA"
for d in "$OUT" "$AUX"; do
  if [ -e "$d" ]; then
    echo "REFUSED: $d exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
  fi
done
mkdir -p "$AUX"

make plugin-wheel || { echo "FAIL: make plugin-wheel failed" >&2; exit 1; }
{ echo "# t0101 warm re-run boot, commit $SHA, $(date -Is)"
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

NOPROXY=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY)

echo "== step 1: T-0099 harness with --warm-repair $(date -Is)"
LIVE=0
"${NOPROXY[@]}" uv run --all-extras python scripts/gpu/t0099_repair_live.py \
  --base-url http://127.0.0.1:8001/v1 --out-dir "$OUT" --warm-repair || LIVE=$?
echo "T-0099 harness (warm) exit $LIVE"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true
if [ -f "$OUT/report.json" ]; then
  echo "report: $OUT/report.md"
  echo "report.json sha256 $(sha256sum "$OUT/report.json" | cut -d' ' -f1)"
fi

echo "== step 2: T-0101 gpu eval rows from the run $(date -Is)"
EVAL=0
if [ ! -f "$OUT/report.json" ]; then
  echo "skipped: $OUT/report.json does not exist (harness exit $LIVE); there is no run to read"
  EVAL=1
else
  "${NOPROXY[@]}" T0101_RUN_DIR="$OUT" \
    uv run --all-extras pytest tests/eval/test_t0101_warm_rerun.py -q -m gpu -rA -s \
    --junitxml="$AUX/t0101_eval.xml" || EVAL=$?
fi
echo "T-0101 eval rows exit $EVAL"

echo "summary: harness $LIVE eval $EVAL $(date -Is)"
if [ "$LIVE" -eq 2 ]; then exit 2; fi
if [ "$LIVE" -ne 0 ] || [ "$EVAL" -ne 0 ]; then exit 1; fi
exit 0
