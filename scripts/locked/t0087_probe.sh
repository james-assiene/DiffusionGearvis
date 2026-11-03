#!/usr/bin/env bash
# T-0087: one fresh engine-only boot (P16), then the native stream probe (scripts/gpu/t0080d_native_stream_probe.py),
# then the engine is stopped. Same boot regime as scripts/locked/t0084_native.sh (services/vllm.sh, prefix caching on,
# plugin wheel rebuilt). /launch only, from a clean tree; the console log's last line is the script's real exit code:
#
#   ( nohup bash scripts/locked/t0087_probe.sh > logs/t0087-probe-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0087-probe-<SHA>-run.log ) &
#
# P10: the proxy variables are unset for the probe, since twopass._metrics uses urlopen. The probe itself refuses a
# non-loopback or non-IP-literal base URL, a dirty tree, a non-empty out-dir and a cases file other than the pinned one.
# Every early exit prints a REFUSED: (exit 2) or FAIL: (exit 1) line.
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
  echo "REFUSED: an Astra port (8001/8002/8080/8081) is already bound (P16 needs a fresh boot)" >&2; exit 2
fi
if [ -n "$(dock ps -q)" ]; then
  echo "REFUSED: containers are running: $(dock ps --format '{{.Names}} {{.Image}}' | tr '\n' ' ')" >&2; exit 2
fi
OUT="logs/t0087-probe-$SHA"
if [ -e "$OUT" ]; then
  echo "REFUSED: $OUT exists; the probe runs once per SHA (move it aside only with the human)" >&2; exit 2
fi

make plugin-wheel || { echo "FAIL: make plugin-wheel failed" >&2; exit 1; }
mkdir -p logs
{ echo "# t0087 probe boot, commit $SHA, $(date -Is)"
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

PROBE=0
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  uv run --all-extras --group cd python scripts/gpu/t0080d_native_stream_probe.py \
  --base-url http://127.0.0.1:8001/v1 --repo "$PWD" --out-dir "$OUT" || PROBE=$?
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true
echo "report: $OUT/report.md"
if [ -f "$OUT/report.json" ]; then
  echo "report.json sha256 $(sha256sum "$OUT/report.json" | cut -d' ' -f1)" || true
fi
echo "probe exit $PROBE $(date -Is)"
exit "$PROBE"
