#!/usr/bin/env bash
# T-0205 D: one fresh engine-only boot, then scripts/gpu/t0205_real_sites.py --real-sites: the demo requests D1-D6, D8,
# D9 (D7 is refused, H-T205-6) on REAL PUBLIC WEBSITES, in web mode, through each session's own T-0160 egress proxy
# (Chromium's sandbox on, a fresh profile, nothing injected), with web search on through the cli's own start path
# (the T-0111b/T-0125/T-0163 scoped exception); headless; n = 1; every turn starts at the live start page /web.html;
# typed turns, fake STT/TTS/audio; 300 s per turn, a 60 min run budget; then the engine is stopped.
# Report-only: reachability, bot walls, steps, where each run ended, the hosts reached and a screenshot per step.
#
# Regime (state it beside every number, with the date and local time of the run: real sites change): profile lean,
# engine only; vLLM, torch, CUDA, driver, model and chat template per services/versions.lock at HEAD (container
# vllm/vllm-openai:v0.29.0); canvas 256, 48 steps, thinking on, prefix caching on, repair on; HEAD's live web-mode
# defaults as the cli resolves them (recorded in manifest.json: guidance, loop 20, screen reminder, ...); headless
# Chromium, sandbox on; search on; engine unseeded; concurrency 1; clean tree.
#
# The boot regime is scripts/locked/t0207_completion_check_ab.sh's verbatim: services/vllm.sh at 806da12 and
# unchanged, prefix caching on, plugin wheel rebuilt, port and container refusals, one run per SHA, one boot,
# setsid services/vllm.sh, health wait up to 600 s, the load-settle wait, the engine stopped on every exit. STT and
# TTS must be down (ports 8080/8081 are refused below).
#
# /launch only, by James himself (H-T205-7: never queued for an unattended session), from a clean tree at (or after)
# T-0205 D's merge commit. THE TOKEN-GATED COMMAND IS THE CONSENT: this script exports ASTRA_REAL_SITES=1 itself,
# after the merge check, and the harness refuses without it. The console log's last line is the real exit code:
#
#   ( nohup bash scripts/locked/t0205_real_sites.sh > logs/t0205-real-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0205-real-<SHA>-run.log ) &
#
# Refusals (exit 2), here before the boot: any argument; a dirty tree (its paths named); no merge commit reachable
# from HEAD whose subject starts "Merge T-0205-d:" (a branch merge such as "Merge main into T-0205-d-..." does not
# count); services/vllm.sh changed since 806da12; an Astra port (8001/8002/8080/8081) bound; any container running;
# the harness absent; the output directory (or this script's aux directory) already existing. The merge log is
# matched through a here-string, never a pipe (SIGPIPE under pipefail once the log passes 64 KiB).
#
# Network: unlike every other locked script, the harness reaches real websites, but only through the paths a live
# session uses: the browser's page traffic through its own egress proxy (https on 443 to public addresses), and web
# search through the cli. The proxy variables, ddgs/primp's own DDGS_PROXY and PRIMP_PROXY too, are unset (the
# egress proxy and the search backend never go behind a proxy). Nothing is written outside logs/.
#
# Budget: the harness runs under `timeout -k 60s 70m` (a timeout is exit 124, a failed step); its own RUN_BUDGET_S is
# 60 min, after which no turn starts. Estimate 25-50 min for 8 turns.
#
# `uv run --locked`: uv.lock is used as committed, never re-resolved.
# Output: logs/t0205-real-<SHA>/ (manifest.json, records.jsonl, shots/<id>/<repeat>-<step>.jpg, sessions/,
# session_turns.jsonl, report.json, report.md); boot lines appended to logs/vllm.log.
# Exit: 0 every turn ran; 1 a step failed (boot, no healthy engine, a run that stopped early, the timeout); 2 refused
# (here, or by the harness before any socket).
#
# File mode: chmod 444 in the working tree; git records it as 100644. Run it with `bash`, as above.
set -euo pipefail
trap 'rc=$?; echo "FAIL: exit $rc at line $LINENO: $BASH_COMMAND" >&2' ERR
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

if [ "$#" -ne 0 ]; then echo "REFUSED: no arguments expected (got $*)" >&2; exit 2; fi

dock() {
  if docker "$@" 2>/dev/null; then return 0; fi
  sg docker -c "docker $(printf '%q ' "$@")"
}

# the first 7 hex digits exactly, as the harness names its default out-dir (sha[:7]); --short=7 can be longer
SHA=$(git rev-parse HEAD | cut -c1-7)
DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
  echo "REFUSED: dirty tree: $(printf '%s' "$DIRTY" | tr '\n' ' ')" >&2; exit 2
fi
MERGES=$(git log --merges --format='%H %s' HEAD || true)
if ! grep -Eq '^[0-9a-f]+ Merge T-0205-d:' <<<"$MERGES"; then
  echo "REFUSED: T-0205 group D's merge commit is not an ancestor of HEAD (no 'Merge T-0205-d:' merge reachable from HEAD; a 'Merge main into T-0205-d-...' branch merge does not count); the rehearsal runs at the merge commit" >&2; exit 2
fi
# D5r: the consent, exported here (after the merge check), so the token-gated command is James's consent
export ASTRA_REAL_SITES=1
if [ -n "$(git diff 806da12..HEAD -- services/vllm.sh)" ]; then
  echo "REFUSED: services/vllm.sh changed since 806da12; the reviewed boot regime (prefix caching on) no longer applies" >&2; exit 2
fi
if ss -ltn 2>/dev/null | grep -Eq '127\.0\.0\.1:(8001|8002|8080|8081)\b'; then
  echo "REFUSED: an Astra port (8001/8002/8080/8081) is already bound (a fresh boot is needed, and STT/TTS must be down)" >&2; exit 2
fi
if [ -n "$(dock ps -q)" ]; then
  echo "REFUSED: containers are running: $(dock ps --format '{{.Names}} {{.Image}}' | tr '\n' ' ')" >&2; exit 2
fi
HARNESS=scripts/gpu/t0205_real_sites.py
for f in "$HARNESS" local_astra/toolcall/tools_v1.json local_astra/pages/static/web.html; do
  if [ ! -f "$f" ]; then
    echo "REFUSED: $f is absent" >&2; exit 2
  fi
done
OUT="logs/t0205-real-$SHA"
AUX="logs/t0205-real-launch-$SHA"
for d in "$OUT" "$AUX"; do
  if [ -e "$d" ]; then
    echo "REFUSED: $d exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
  fi
done
mkdir -p "$AUX"

make plugin-wheel || { echo "FAIL: make plugin-wheel failed" >&2; exit 1; }
{ echo "# t0205 real-site rehearsal boot, commit $SHA, $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free --format=csv,noheader || true
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true
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
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader | sed 's/^/compute app: /' || true
uptime | sed 's/^/load after boot: /' || true

# the plugin-wheel build and the boot can leave the load high, so wait here (every 10 s, up to 10 min) before the
# turns; never settling is exit 1 (the trap stops the engine)
LOAD_OK=0
for i in $(seq 0 60); do
  LOAD1=$(cut -d' ' -f1 /proc/loadavg)
  echo "load1 $LOAD1 (wait ${i}0 s) $(date -Is)"
  if awk -v l="$LOAD1" 'BEGIN { exit !(l < 2) }'; then LOAD_OK=1; break; fi
  if [ "$i" -lt 60 ]; then sleep 10; fi
done
if [ "$LOAD_OK" -ne 1 ]; then
  echo "FAIL: the 1-min load stayed at or above 2 for 10 min after the boot (last $LOAD1). Find what is loading the box, then /launch again" >&2; exit 1
fi

NOPROXY=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u DDGS_PROXY -u PRIMP_PROXY)

echo "== T-0205 D real-site rehearsal $(date -Is) (local time: $(date '+%Y-%m-%d %H:%M %Z'))"
RUN=0
timeout -k 60s 70m "${NOPROXY[@]}" uv run --locked --all-extras python scripts/gpu/t0205_real_sites.py \
  --base-url http://127.0.0.1:8001/v1 --real-sites || RUN=$?
echo "T-0205 D harness exit $RUN"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true

if [ -f "$OUT/report.json" ]; then
  echo "report: $OUT/report.md"
  for f in report.json manifest.json records.jsonl; do
    if [ -f "$OUT/$f" ]; then echo "$f sha256 $(sha256sum "$OUT/$f" | cut -d' ' -f1)"; fi
  done
else
  echo "no $OUT/report.json" >&2
fi

echo "summary: harness $RUN $(date -Is)"
if [ "$RUN" -eq 2 ]; then exit 2; fi
if [ "$RUN" -ne 0 ]; then exit 1; fi
exit 0
