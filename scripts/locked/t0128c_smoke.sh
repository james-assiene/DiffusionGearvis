#!/usr/bin/env bash
# T-0128c C2: the /launch smoke of the real denoising view. One fresh lean boot (the harness runs
# `diffusion_gearvis up --profile lean` and `down`), the gpu rows of tests/contract/test_canvas_relay_wire.py,
# then `diffusion_gearvis session --silent --serve-hud`, headless Chromium on the HUD page and three typed turns;
# the gates G1-G7 are local_astra/eval/denoise_smoke.py's over logs/t0128c-smoke-<SHA>/evidence.json (never
# artifacts/). Same boot-regime pins as scripts/locked/t0093_live.sh and t0111g_live_smoke.sh (services/vllm.sh at
# 806da12, port and container refusals, plugin wheel rebuilt, one run per SHA). /launch only, from a clean tree at
# (or after) both group merges; the console log's last line is the real exit code:
#
#   ( nohup bash scripts/locked/t0128c_smoke.sh > logs/t0128c-smoke-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0128c-smoke-<SHA>-run.log ) &
#
# Refuses unless the merges of T-0128c group A (the relay) and group B (the page) are both ancestors of HEAD: a
# merge commit reachable from HEAD whose subject starts "Merge T-0128c group A" / "Merge T-0128c group B". The
# plugin wheel is rebuilt offline into an emptied dist/ so services/vllm.sh installs exactly that one wheel; the
# harness refuses (exit 2) unless HEAD:plugin is the branch base's tree (no wheel sha is compared, O-C (2)). A
# failed step still leaves a partial evidence.json naming it (O-C (5)). The harness runs under `timeout` (SIGINT first so its `finally` stops the session and runs `down`; exit
# 124 is a failed step). The proxy variables are unset for every step (loopback only). Exit: 0 every gate passed;
# 1 a gate or a step failed, or a `down` exited non-zero; 2 refused (here, or by the harness before any service starts).
set -euo pipefail
trap 'rc=$?; echo "FAIL: exit $rc at line $LINENO: $BASH_COMMAND" >&2' ERR
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

if [ "$#" -ne 0 ]; then echo "REFUSED: no arguments expected (got $*)" >&2; exit 2; fi

dock() {
  if docker "$@" 2>/dev/null; then return 0; fi
  sg docker -c "docker $(printf '%q ' "$@")"
}

SHA=$(git rev-parse --short=7 HEAD)
DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
  echo "REFUSED: dirty tree: $(printf '%s' "$DIRTY" | tr '\n' ' ')" >&2; exit 2
fi
MERGES=$(git log --merges --format='%H %s' HEAD)
for GROUP in A B; do
  if ! grep -Eq "^[0-9a-f]+ Merge T-0128c group ${GROUP}([^A-Za-z0-9]|\$)" <<<"$MERGES"; then
    echo "REFUSED: no 'Merge T-0128c group ${GROUP}' merge commit reachable from HEAD; the smoke runs after both group merges" >&2; exit 2
  fi
done
if [ -n "$(git diff 806da12..HEAD -- services/vllm.sh)" ]; then
  echo "REFUSED: services/vllm.sh changed since 806da12; the reviewed boot regime (prefix caching on) no longer applies" >&2; exit 2
fi
if ss -ltn 2>/dev/null | grep -Eq ':(8000|1420|8001|8002|8080|8081)\b'; then
  echo "REFUSED: an Astra port (8000/1420/8001/8002/8080/8081) is already bound (a fresh boot is needed)" >&2; exit 2
fi
if [ -n "$(dock ps -q)" ]; then
  echo "REFUSED: containers are running: $(dock ps --format '{{.Names}} {{.Image}}' | tr '\n' ' ')" >&2; exit 2
fi
HARNESS=scripts/gpu/t0128c_denoise_smoke.py
for f in "$HARNESS" local_astra/eval/denoise_smoke.py tests/contract/test_canvas_relay_wire.py; do
  if [ ! -f "$f" ]; then
    echo "REFUSED: $f is absent" >&2; exit 2
  fi
done
OUT="logs/t0128c-smoke-$SHA"
if [ -e "$OUT" ]; then
  echo "REFUSED: $OUT exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
fi

NOPROXY=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY)

rm -rf plugin/astra_vllm_plugin/dist
if ! "${NOPROXY[@]}" UV_OFFLINE=1 make plugin-wheel; then
  echo "FAIL: make plugin-wheel (offline) failed" >&2; exit 1
fi
{ echo "# t0128c denoising smoke boot, commit $SHA, $(date -Is)"
  if ! nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free --format=csv,noheader; then
    echo "nvidia-smi failed"
  fi
  uptime
} | tee -a logs/vllm.log

echo "== T-0128c smoke harness $(date -Is)"
SMOKE=0
timeout --signal=INT --kill-after=3m 40m "${NOPROXY[@]}" uv run --all-extras python "$HARNESS" \
  --out-dir "$OUT" || SMOKE=$?
echo "T-0128c harness exit $SMOKE"

# Belt and braces: the harness's `finally` already ran `down`; a second `down` stops anything a killed harness left.
DOWN=0
"${NOPROXY[@]}" uv run --all-extras diffusion_gearvis down || DOWN=$?
echo "diffusion_gearvis down exit $DOWN"
echo "containers left: $(dock ps -q | wc -l)"

if [ -f "$OUT/evidence.json" ]; then
  echo "report: $OUT/report.md"
  echo "evidence.json sha256 $(sha256sum "$OUT/evidence.json" | cut -d' ' -f1)"
  echo "gates.json sha256 $(sha256sum "$OUT/gates.json" | cut -d' ' -f1)"
else
  echo "no $OUT/evidence.json: the gates were not decided" >&2
fi

echo "summary: smoke $SMOKE down $DOWN $(date -Is)"
if [ "$SMOKE" -eq 2 ]; then exit 2; fi
if [ "$SMOKE" -ne 0 ] || [ ! -f "$OUT/evidence.json" ]; then exit 1; fi
# O-C2 (5): a failed `down` may leave services holding the GPU: the run fails
if [ "$DOWN" -ne 0 ]; then echo "FAIL: diffusion_gearvis down exited $DOWN" >&2; exit 1; fi
exit 0
