#!/usr/bin/env bash
# T-0108a: one fresh engine-only boot, then
#   1. scripts/gpu/t0098_native_prompt_ab.py --table T-0108a --split dev: 60 dev cases (40 tool + 8 no-tool + 5 plain
#      + 7 Q12) x 3 arms (A0, K1, K2), 40 repeats on the 4 Q12 positive cases and 5 on the other 56 = 1320 pass-1
#      requests, concurrency 1, no seed (the required-fields prompt A/B; H-T108a);
#   2. tests/eval/test_ui_fields_ab.py: the outcome re-derived from trials.jsonl alone (its own UiGate calls) and
#      checked against report.json (criterion 8; no GPU time of its own), only if report.json exists;
# then the engine is stopped. The boot regime is scripts/locked/t0103_c3_confirm.sh's verbatim (services/vllm.sh at
# 806da12, prefix caching on, plugin wheel rebuilt, port and container refusals). /launch only, from a clean tree; the
# console log's last line is the real exit code:
#
#   ( nohup bash scripts/locked/t0108a_ui_fields_ab.sh > logs/t0108a-ui-fields-ab-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0108a-ui-fields-ab-<SHA>-run.log ) &
#
# The proxy variables are unset for every step (loopback only). Exit: 0 both steps passed; 1 a step failed; 2 refused
# (here, or by the harness before any socket).
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
OUT="logs/t0108a-ui-fields-ab-$SHA"
AUX="logs/t0108a-launch-$SHA"
for d in "$OUT" "$AUX"; do
  if [ -e "$d" ]; then
    echo "REFUSED: $d exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
  fi
done
mkdir -p "$AUX"

make plugin-wheel || { echo "FAIL: make plugin-wheel failed" >&2; exit 1; }
{ echo "# t0108a ui fields a/b boot, commit $SHA, $(date -Is)"
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

echo "== step 1: T-0108a harness (--table T-0108a) $(date -Is)"
AB=0
"${NOPROXY[@]}" uv run --all-extras python scripts/gpu/t0098_native_prompt_ab.py \
  --base-url http://127.0.0.1:8001/v1 --out-dir "$OUT" --table T-0108a --split dev || AB=$?
echo "T-0108a harness exit $AB"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true

echo "== step 2: T-0108a outcome re-derived from trials.jsonl $(date -Is)"
EVAL=0
if [ -f "$OUT/report.json" ]; then
  echo "report: $OUT/report.md"
  echo "report.json sha256 $(sha256sum "$OUT/report.json" | cut -d' ' -f1)"
  echo "trials.jsonl sha256 $(sha256sum "$OUT/trials.jsonl" | cut -d' ' -f1)"
  "${NOPROXY[@]}" T0108A_RUN_DIR="$OUT" \
    uv run --all-extras pytest tests/eval/test_ui_fields_ab.py -q -m gpu -rA \
    --junitxml="$AUX/t0108a_eval.xml" || EVAL=$?
else
  echo "no $OUT/report.json: step 2 not run" >&2
  EVAL=1
fi
echo "T-0108a eval exit $EVAL"

echo "summary: harness $AB eval $EVAL $(date -Is)"
if [ "$AB" -eq 2 ]; then exit 2; fi
if [ "$AB" -ne 0 ] || [ "$EVAL" -ne 0 ]; then exit 1; fi
exit 0
