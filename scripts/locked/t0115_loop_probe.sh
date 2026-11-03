#!/usr/bin/env bash
# T-0115 (with T-0115a): one fresh engine-only boot, then
#   1. tests/contract/test_llm_step_continuation.py -m gpu: T-0114's real-engine step-continuation row;
#   2. scripts/gpu/t0115_loop_probe.py: 118 typed turns through a real Session with the tool loop on (max_tool_steps
#      8, dry run, repair on, ScenarioSearch), concurrency 1, no seed (the multi-step loop probe; H-T115-1..9);
#   3. scripts/gpu/t0115a_loop_requests.py: the request-level rows (Q4 form G vs A, Q5 image in a continued
#      message, the compaction A/B), Q5 last inside that harness;
#   4. tests/eval/test_loop_probe.py and tests/eval/test_loop_requests.py: each outcome re-derived from its run's
#      records and checked against its report.json (no GPU time of their own), each only if its report.json exists;
# then the engine is stopped. The boot regime is scripts/locked/t0108a_ui_fields_ab.sh's verbatim (services/vllm.sh
# at 806da12, prefix caching on, plugin wheel rebuilt, port and container refusals, one run per SHA). /launch only,
# from a clean tree at the merge commit of T-0115, T-0115a and T-0120; the console log's last line is the real exit
# code:
#
#   ( nohup bash scripts/locked/t0115_loop_probe.sh > logs/t0115-loop-probe-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0115-loop-probe-<SHA>-run.log ) &
#
# Refuses unless T-0117 (ebff09e) is an ancestor of HEAD (B4). The T-0115 harness runs under `timeout 90m` and
# T-0115a under `timeout 30m` (B3; a timeout is exit 124, a failed step).
# The proxy variables are unset for every step (loopback only). Exit: 0 every step passed; 1 a step failed; 2 refused
# (here, or by a harness before any socket).
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
if ! git merge-base --is-ancestor ebff09e HEAD; then
  echo "REFUSED: T-0117 (ebff09e) is not an ancestor of HEAD; its known thought voicing would trip G6 (H-T115-6, B4)" >&2; exit 2
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
H115=scripts/gpu/t0115_loop_probe.py
H115A=scripts/gpu/t0115a_loop_requests.py
CONTRACT=tests/contract/test_llm_step_continuation.py
for f in "$H115" "$H115A" "$CONTRACT" tests/eval/test_loop_probe.py tests/eval/test_loop_requests.py; do
  if [ ! -f "$f" ]; then
    echo "REFUSED: $f is absent (T-0115a and T-0120 merge before this launch; H-T115-5)" >&2; exit 2
  fi
done
if ! grep -q "^def compact" local_astra/toolcall/loop.py; then
  echo "REFUSED: loop.compact is absent from local_astra/toolcall/loop.py (T-0120 merges before this launch)" >&2; exit 2
fi
OUT="logs/t0115-loop-probe-$SHA"
OUT_A="logs/t0115a-loop-requests-$SHA"
AUX="logs/t0115-launch-$SHA"
for d in "$OUT" "$OUT_A" "$AUX"; do
  if [ -e "$d" ]; then
    echo "REFUSED: $d exists; the run happens once per SHA (move it aside only with the human; H-T115-9)" >&2; exit 2
  fi
done
mkdir -p "$AUX"

make plugin-wheel || { echo "FAIL: make plugin-wheel failed" >&2; exit 1; }
{ echo "# t0115 loop probe boot, commit $SHA, $(date -Is)"
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

echo "== step 1: T-0114 step-continuation contract row (-m gpu) $(date -Is)"
CONTRACT_RC=0
"${NOPROXY[@]}" uv run --all-extras pytest "$CONTRACT" -m gpu -rA \
  --junitxml="$AUX/t0114_contract.xml" || CONTRACT_RC=$?
echo "contract row exit $CONTRACT_RC"

echo "== step 2: T-0115 loop probe harness $(date -Is)"
PROBE=0
# B3: a wall-time bound (the harness also stops itself after 3 consecutive errored turns); exit 124 is a step failure
timeout 90m "${NOPROXY[@]}" uv run --all-extras python "$H115" \
  --base-url http://127.0.0.1:8001/v1 --out-dir "$OUT" || PROBE=$?
echo "T-0115 harness exit $PROBE"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory after step 2: /' || true

echo "== step 3: T-0115a loop requests harness $(date -Is)"
REQUESTS=0
timeout 30m "${NOPROXY[@]}" uv run --all-extras python "$H115A" \
  --base-url http://127.0.0.1:8001/v1 --out-dir "$OUT_A" || REQUESTS=$?
echo "T-0115a harness exit $REQUESTS"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true

echo "== step 4: the outcomes re-derived from the records $(date -Is)"
EVAL=0
if [ -f "$OUT/report.json" ]; then
  echo "report: $OUT/report.md"
  echo "report.json sha256 $(sha256sum "$OUT/report.json" | cut -d' ' -f1)"
  echo "trials.jsonl sha256 $(sha256sum "$OUT/trials.jsonl" | cut -d' ' -f1)"
  echo "steps.jsonl sha256 $(sha256sum "$OUT/steps.jsonl" | cut -d' ' -f1)"
  "${NOPROXY[@]}" T0115_RUN_DIR="$OUT" \
    uv run --all-extras pytest tests/eval/test_loop_probe.py -q -m gpu -rA \
    --junitxml="$AUX/t0115_eval.xml" || EVAL=$?
else
  echo "no $OUT/report.json: the T-0115 eval not run" >&2
  EVAL=1
fi
echo "T-0115 eval exit $EVAL"
EVAL_A=0
if [ -f "$OUT_A/report.json" ]; then
  echo "report: $OUT_A/report.md"
  echo "report.json sha256 $(sha256sum "$OUT_A/report.json" | cut -d' ' -f1)"
  echo "requests.jsonl sha256 $(sha256sum "$OUT_A/requests.jsonl" | cut -d' ' -f1)"
  "${NOPROXY[@]}" T0115A_RUN_DIR="$OUT_A" \
    uv run --all-extras pytest tests/eval/test_loop_requests.py -q -m gpu -rA \
    --junitxml="$AUX/t0115a_eval.xml" || EVAL_A=$?
else
  echo "no $OUT_A/report.json: the T-0115a eval not run" >&2
  EVAL_A=1
fi
echo "T-0115a eval exit $EVAL_A"

echo "summary: contract $CONTRACT_RC probe $PROBE requests $REQUESTS eval $EVAL eval_a $EVAL_A $(date -Is)"
if [ "$PROBE" -eq 2 ] || [ "$REQUESTS" -eq 2 ]; then exit 2; fi
if [ "$CONTRACT_RC" -ne 0 ] || [ "$PROBE" -ne 0 ] || [ "$REQUESTS" -ne 0 ] || [ "$EVAL" -ne 0 ] || [ "$EVAL_A" -ne 0 ]; then
  exit 1
fi
exit 0
