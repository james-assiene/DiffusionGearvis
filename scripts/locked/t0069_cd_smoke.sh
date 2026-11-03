#!/usr/bin/env bash
# T-0069 tier 3 (criteria 4-9): one engine-only, plugin-on boot (Tap ∘ CD ∘ stock), the smoke driver
# (scripts/gpu/cd_smoke.py), criterion 9's two backend=real contract rows on the same boot, then the
# engine is stopped. /launch only; run detached from the T-0069 worktree:
#
#   nohup bash scripts/locked/t0069_cd_smoke.sh > logs/t0069-smoke-run.log 2>&1 &
#   echo "exit $?"   # /launch echoes the exit code when the job ends
#
# Engine only (STT, TTS and the conductor stay down: docs/tasks/T-0069.md Regime, tier 3), so the engine
# is started with services/vllm.sh directly rather than `astra up --profile lean`; no live prefix cache is
# shared, so the driver's enable_thinking=false costs nothing elsewhere. The boot appends to this
# worktree's logs/vllm.log; the driver slices it to the last boot (R16).
#
# Exit code: the driver's (0 pass, 1 fail, 2 refused); if the driver passed, the contract rows' (pytest's).
# Both always run once the engine is healthy; the evidence and report are in logs/t0069-smoke-<sha>/.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

dock() {
  if docker "$@" 2>/dev/null; then return 0; fi
  sg docker -c "docker $(printf '%q ' "$@")"
}

SHA=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSE: dirty tree: $(git status --porcelain | tr '\n' ' ')" >&2; exit 2
fi
if [ -n "$(git diff 806da12..HEAD -- services/vllm.sh)" ]; then
  echo "REFUSE: services/vllm.sh changed since 806da12; criterion 8's boot reference no longer applies" >&2; exit 2
fi
if [ ! -f logs/t0011-tier3-806da12/boot-reference.txt ]; then
  echo "REFUSE: logs/t0011-tier3-806da12/boot-reference.txt missing (copy it from the t0011 worktree's logs/)" >&2; exit 2
fi
if ss -ltn 2>/dev/null | grep -Eq '127\.0\.0\.1:(8001|8002|8080|8081)\b'; then
  echo "REFUSE: an Astra port (8001/8002/8080/8081) is already bound" >&2; exit 2
fi
if [ -n "$(dock ps -q)" ]; then
  echo "REFUSE: containers are running: $(dock ps --format '{{.Names}} {{.Image}}' | tr '\n' ' ')" >&2; exit 2
fi
OUT="logs/t0069-smoke-$SHA"
if [ -e "$OUT" ]; then
  echo "REFUSE: $OUT exists; move it aside first" >&2; exit 2
fi

make plugin-wheel
mkdir -p logs
{ echo "# t0069 tier 3 smoke boot, commit $SHA, $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free --format=csv,noheader
  uptime
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

set +e
uv run --group cd python scripts/gpu/cd_smoke.py --boot-log logs/vllm.log --profile lean
DRIVER=$?
echo "driver exit $DRIVER $(date -Is)"

# Criterion 9, first bullet: the two backend=real rows of the T-0070 contract module, on this boot.
# The module reads ASTRA_CD_REAL_URL / ASTRA_CD_ENGINE_LOG (its `backend` fixture); `-m gpu` selects the
# real parametrisation (marked gpu), `-k real` excludes the fake rows; tests/cd is collected only with
# ASTRA_CD_TESTS=1.
ASTRA_CD_TESTS=1 ASTRA_CD_REAL_URL=http://127.0.0.1:8001/v1 ASTRA_CD_ENGINE_LOG="$PWD/logs/vllm.log" \
  uv run --all-extras --group cd pytest tests/cd/contract/test_eval_fake_server_contract.py \
  -m gpu -k real -v -p no:cacheprovider 2>&1 | tee "logs/t0069-contract-real-$SHA.log"
CONTRACT=${PIPESTATUS[0]}
echo "contract rows exit $CONTRACT $(date -Is)"
set -e

if [ "$DRIVER" -ne 0 ]; then exit "$DRIVER"; fi
exit "$CONTRACT"
