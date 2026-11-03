#!/usr/bin/env bash
# T-0120c: one fresh engine-only boot, then
#   1. scripts/gpu/t0120c_compaction_recheck.py: the compaction re-check (160 E requests: old FULL + COMPACT, kept
#      COMPACT, 40 warm; no Q4/Q5, no kept FULL), concurrency 1, no seed, the T-0120b invite stub;
#   2. tests/eval/test_compaction_recheck.py -m gpu: the outcome re-derived from requests.jsonl and checked against
#      report.json (no GPU time of its own), only if report.json exists;
#   3. the sha256 of report.json and requests.jsonl;
# then the engine is stopped. The boot regime and its refusals are scripts/locked/t0115_loop_probe.sh's verbatim
# (services/vllm.sh at 806da12, prefix caching on, plugin wheel rebuilt, port and container refusals, one run per SHA).
# /launch only, from a clean tree at the T-0120c merge commit; the console log's last line is the real exit code:
#
#   ( nohup bash scripts/locked/t0120c_compaction_recheck.sh > logs/t0120c-compaction-recheck-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0120c-compaction-recheck-<SHA>-run.log ) &
#
# Refuses unless T-0120b's merge (a5d7454) is an ancestor of HEAD and local_astra/toolcall/loop.py carries the invite
# stub. The harness runs under `timeout 20m` (a timeout is exit 124, a failed step). The CPU calibration on the c398fdf
# record (criterion 7) runs before the launch, not here. The proxy variables are unset for every step (loopback only).
# Exit: 0 every step passed; 1 a step failed; 2 refused (here, or by the harness before any socket).
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
if ! git merge-base --is-ancestor a5d7454 HEAD; then
  echo "REFUSED: T-0120b (a5d7454) is not an ancestor of HEAD; the invite stub under test is absent" >&2; exit 2
fi
if ! grep -qF "earlier result omitted; search again if you need it" local_astra/toolcall/loop.py; then
  echo "REFUSED: local_astra/toolcall/loop.py does not carry the T-0120b invite stub (H-T120b-0)" >&2; exit 2
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
H120C=scripts/gpu/t0120c_compaction_recheck.py
EVAL_MODULE=tests/eval/test_compaction_recheck.py
for f in "$H120C" scripts/gpu/t0115a_loop_requests.py "$EVAL_MODULE"; do
  if [ ! -f "$f" ]; then
    echo "REFUSED: $f is absent" >&2; exit 2
  fi
done
OUT="logs/t0120c-compaction-recheck-$SHA"
AUX="logs/t0120c-launch-$SHA"
for d in "$OUT" "$AUX"; do
  if [ -e "$d" ]; then
    echo "REFUSED: $d exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
  fi
done
mkdir -p "$AUX"

make plugin-wheel || { echo "FAIL: make plugin-wheel failed" >&2; exit 1; }
{ echo "# t0120c compaction re-check boot, commit $SHA, $(date -Is)"
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

echo "== step 1: T-0120c compaction re-check harness $(date -Is)"
RECHECK=0
timeout 20m "${NOPROXY[@]}" uv run --all-extras python "$H120C" \
  --base-url http://127.0.0.1:8001/v1 --out-dir "$OUT" || RECHECK=$?
echo "T-0120c harness exit $RECHECK"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true

echo "== step 2: the outcome re-derived from the records $(date -Is)"
EVAL=0
if [ -f "$OUT/report.json" ]; then
  echo "report: $OUT/report.md"
  "${NOPROXY[@]}" T0120C_RUN_DIR="$OUT" \
    uv run --all-extras pytest "$EVAL_MODULE" -q -m gpu -rA \
    --junitxml="$AUX/t0120c_eval.xml" || EVAL=$?
else
  echo "no $OUT/report.json: the T-0120c eval not run" >&2
  EVAL=1
fi
echo "T-0120c eval exit $EVAL"

echo "== step 3: record hashes $(date -Is)"
if [ -f "$OUT/report.json" ]; then
  echo "report.json sha256 $(sha256sum "$OUT/report.json" | cut -d' ' -f1)"
fi
if [ -f "$OUT/requests.jsonl" ]; then
  echo "requests.jsonl sha256 $(sha256sum "$OUT/requests.jsonl" | cut -d' ' -f1)"
fi

echo "summary: recheck $RECHECK eval $EVAL $(date -Is)"
if [ "$RECHECK" -eq 2 ]; then exit 2; fi
if [ "$RECHECK" -ne 0 ] || [ "$EVAL" -ne 0 ]; then
  exit 1
fi
exit 0
