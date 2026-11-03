#!/usr/bin/env bash
# T-0093 (with T-0088's owed gpu rows, per the T-0093 amendment): one fresh engine-only boot, then
#   1. tests/contract/test_llm_tools_wire_gpu.py rows 1-4 (T-0088). If any row fails, T-0093 does not run: row 2's
#      multi-canvas shape must pass before any T-0093 number is read, and a different shape means the fake changes first;
#   2. scripts/gpu/t0093_toolcall_live.py: 60 cases x {tools off, tools on dry_run}, typed turns, one boot;
#   3. tests/eval/test_toolcall_live.py: the hard gates re-read from the run's report (no GPU time of its own);
# then the engine is stopped. Same boot regime as scripts/locked/t0087_probe.sh (services/vllm.sh at 806da12, prefix
# caching on, plugin wheel rebuilt). /launch only, from a clean tree; the console log's last line is the real exit code:
#
#   ( nohup bash scripts/locked/t0093_live.sh > logs/t0093-live-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0093-live-<SHA>-run.log ) &
#
# Row 3's CD payload is the click_at entry of local_astra/toolcall/payloads/native_T4.json (T-0091): the same
# compile_toolcall(tools.load(), scope="click_at", thought="T4", call_format="native") with whitespace "json", which is
# both compile_toolcall's default and TwoPassConfig's. It is written beside the run, not under tests/ or artifacts/.
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
OUT="logs/t0093-live-$SHA"
AUX="logs/t0093-launch-$SHA"
for d in "$OUT" "$AUX"; do
  if [ -e "$d" ]; then
    echo "REFUSED: $d exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
  fi
done
mkdir -p "$AUX"
ENGINE_LOG="$PWD/logs/vllm.log"
PAYLOAD="$PWD/$AUX/cd_payload_click_at.txt"
uv run --all-extras python - "$PAYLOAD" <<'PY'
import json, sys
doc = json.load(open("local_astra/toolcall/payloads/native_T4.json"))
m = doc["manifest"]
assert (m["call_format"], m["thought"], m["whitespace"]) == ("native", "T4", "json"), m
open(sys.argv[1], "w").write(doc["tools"]["click_at"]["automaton"])
print(f"row 3 payload: click_at sha256 {doc['tools']['click_at']['automaton_sha256']} tools {m['tools_sha256'][:12]}")
PY

make plugin-wheel || { echo "FAIL: make plugin-wheel failed" >&2; exit 1; }
{ echo "# t0093 live boot, commit $SHA, $(date -Is)"
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

echo "== step 1: T-0088 gpu rows $(date -Is)"
WIRE=0
"${NOPROXY[@]}" ASTRA_ENGINE_LOG="$ENGINE_LOG" ASTRA_T0088_CD_PAYLOAD="$PAYLOAD" ASTRA_T0088_RECORD_DIR="$PWD/$AUX" \
  uv run --all-extras pytest tests/contract/test_llm_tools_wire_gpu.py -q -m gpu -rA -s \
  --junitxml="$AUX/t0088_gpu.xml" || WIRE=$?
echo "T-0088 gpu rows exit $WIRE"
if [ "$WIRE" -ne 0 ]; then
  echo "FAIL: a T-0088 gpu row failed; T-0093 does not run (the amendment: row 2 passes first, the fake is fixed first)" >&2
  exit 1
fi

echo "== step 2: T-0093 harness $(date -Is)"
LIVE=0
"${NOPROXY[@]}" uv run --all-extras python scripts/gpu/t0093_toolcall_live.py \
  --base-url http://127.0.0.1:8001/v1 --out-dir "$OUT" || LIVE=$?
echo "T-0093 harness exit $LIVE"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true
if [ -f "$OUT/report.json" ]; then
  echo "report: $OUT/report.md"
  echo "report.json sha256 $(sha256sum "$OUT/report.json" | cut -d' ' -f1)"
fi

echo "== step 3: T-0093 gates from the report $(date -Is)"
GATES=0
"${NOPROXY[@]}" T0093_RUN_DIR="$OUT" \
  uv run --all-extras pytest tests/eval/test_toolcall_live.py -q -m gpu -rA \
  --junitxml="$AUX/t0093_gates.xml" || GATES=$?
echo "T-0093 gates exit $GATES"

echo "summary: wire $WIRE harness $LIVE gates $GATES $(date -Is)"
if [ "$LIVE" -eq 2 ]; then exit 2; fi
if [ "$LIVE" -ne 0 ] || [ "$GATES" -ne 0 ]; then exit 1; fi
exit 0
