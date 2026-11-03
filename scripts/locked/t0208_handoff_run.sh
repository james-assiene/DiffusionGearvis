#!/usr/bin/env bash
# T-0208 C6: one fresh engine-only boot, then
#   1. scripts/gpu/t0208_handoff_run.py: the pre-flight probes P1-P3 (T-0131's), P-H1 (each handoff's golden path
#      succeeds and each trap fails under handoff_tasks.judge, through the real web-mode executor on loopback with a
#      deny-all resolver and connector; browser only, no inference) and P-H2 (zero resolver or connector calls and
#      zero non-loopback sockets over P-H1 plus an exhaustive interaction of every scenario page; browser only) -- a
#      failed probe is exit 3 and a void report --, then H1-H3 x {shared, typed} x n = 3 = 18 typed turns at HEAD's
#      live defaults as cli gives them (web mode on loopback, live guidance v3, max_tool_steps 20, the wait tool and
#      the completion check as at HEAD, recorded), search off, shuffled with seed 0, each through a fresh Session, a
#      fresh socketless HudServer over one BrowserFrameSource, one fresh StartPageServer(SCENARIO_DIR) and one fresh
#      web-mode executor per turn; the shared arm posts the picture as an astra.frame and selects shared, the typed
#      arm posts nothing; concurrency 1, 300 s per turn, no engine seed; the report is written straight after;
#   2. tests/eval/test_t0208_handoff_run.py -m gpu: every per-arm count, the probes, P-H3 and the label re-derived
#      from the run's records and checked against its report.json (no GPU time of its own), only if report.json
#      exists;
# then the engine is stopped. The boot regime is scripts/locked/t0205_demo_run.sh's verbatim (services/vllm.sh at
# 806da12 and unchanged, prefix caching on, plugin wheel rebuilt, port and container refusals, one run per SHA, one
# boot). STT must be down (its ports 8080/8081 are refused below).
#
# /launch only, from a clean tree at (or after) T-0208 group B's merge commit; the console log's last line is the real
# exit code:
#
#   ( nohup bash scripts/locked/t0208_handoff_run.sh > logs/t0208-handoff-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0208-handoff-<SHA>-run.log ) &
#
# Refuses unless T-0208 group B's merge is an ancestor of HEAD: a merge commit reachable from HEAD whose subject
# starts "Merge T-0208-b:" (the handoff scenarios, images and judge); a branch merge ("Merge commit '...' into
# T-0208-b-impl") or "Merge T-0208-bb:" does not count. A2 and C are neither required nor refused (the harness records
# whether A2 is in). The merge log is matched through a here-string, never a pipe (under pipefail `printf | grep -q`
# can fail on SIGPIPE once the log passes the pipe buffer). Also refused: services/vllm.sh changed since 806da12. The
# harness runs under `timeout -k 60s 60m` (a timeout is exit 124, a failed step; SIGKILL 60 s after the TERM); its own
# RUN_BUDGET_S is 50 min from process start, the probes included, so the report is still written (estimate 30-45
# min). Both `uv run` steps use --locked. The proxy variables are unset for every step (loopback only).
# Exit: 0 every step passed; 1 a step failed (a failed probe's exit 3 included); 2 refused (here, or by the harness
# before any socket). Report-only: no bar, no threshold file changes.
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
MERGES=$(git log --merges --format='%H %s' HEAD || true)
if ! grep -Eq '^[0-9a-f]+ Merge T-0208-b:' <<<"$MERGES"; then
  echo "REFUSED: T-0208 group B's merge commit is not an ancestor of HEAD (no 'Merge T-0208-b:' merge reachable from HEAD; a branch merge does not count); the handoff scenarios, images and judge are group B's" >&2; exit 2
fi
if [ -n "$(git diff 806da12..HEAD -- services/vllm.sh)" ]; then
  echo "REFUSED: services/vllm.sh changed since 806da12; the reviewed boot regime (prefix caching on) no longer applies" >&2; exit 2
fi
if ss -ltn 2>/dev/null | grep -Eq '127\.0\.0\.1:(8001|8002|8080|8081)\b'; then
  echo "REFUSED: an Astra port (8001/8002/8080/8081) is already bound (a fresh boot is needed, and STT must be down)" >&2; exit 2
fi
if [ -n "$(dock ps -q)" ]; then
  echo "REFUSED: containers are running: $(dock ps --format '{{.Names}} {{.Image}}' | tr '\n' ' ')" >&2; exit 2
fi
HARNESS=scripts/gpu/t0208_handoff_run.py
EVAL=tests/eval/test_t0208_handoff_run.py
# the harness imports T-0205 C's, which imports T-0134's, T-0132's, T-0131's and T-0111g's; the scenario and handoff
# tables, the handoff images, the tool sets
for f in "$HARNESS" "$EVAL" scripts/gpu/t0205_demo_run.py scripts/gpu/t0134_feedback_ab.py scripts/gpu/t0132_crop_budget_ab.py scripts/gpu/t0131_grounding_ab.py scripts/gpu/t0111g_live_smoke.py local_astra/toolcall/tools_v1.json local_astra/toolcall/tools_v1_wait.json local_astra/pages/scenario_tasks.py local_astra/pages/handoff_tasks.py local_astra/pages/handoff/h1_steam_wand.jpg local_astra/pages/handoff/h2_stove_box.jpg local_astra/pages/handoff/h3_chat.jpg local_astra/pages/handoff/MANIFEST.md; do
  if [ ! -f "$f" ]; then
    echo "REFUSED: $f is absent" >&2; exit 2
  fi
done
OUT="logs/t0208-handoff-$SHA"
AUX="logs/t0208-launch-$SHA"
for d in "$OUT" "$AUX"; do
  if [ -e "$d" ]; then
    echo "REFUSED: $d exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
  fi
done
mkdir -p "$AUX"

make plugin-wheel || { echo "FAIL: make plugin-wheel failed" >&2; exit 1; }
{ echo "# t0208 handoff run boot, commit $SHA, $(date -Is)"
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

echo "== step 1: T-0208 C handoff run harness (H1-H3 x {shared, typed} x 3, HEAD's live defaults, web mode on loopback) $(date -Is)"
RUN=0
timeout -k 60s 60m "${NOPROXY[@]}" uv run --locked --all-extras python "$HARNESS" \
  --base-url http://127.0.0.1:8001/v1 --out-dir "$OUT" || RUN=$?
echo "T-0208 harness exit $RUN"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true

echo "== step 2: the per-arm counts, the probes, P-H3 and the label re-derived from the records $(date -Is)"
EVAL_RC=0
if [ -f "$OUT/report.json" ]; then
  echo "report: $OUT/report.md"
  for f in report.json trials.jsonl steps.jsonl probes.json order.json manifest.json; do
    if [ -f "$OUT/$f" ]; then echo "$f sha256 $(sha256sum "$OUT/$f" | cut -d' ' -f1)"; fi
  done
  "${NOPROXY[@]}" T0208_RUN_DIR="$OUT" \
    uv run --locked --all-extras pytest "$EVAL" -q -m gpu -rA \
    --junitxml="$AUX/t0208_eval.xml" || EVAL_RC=$?
else
  echo "no $OUT/report.json: the eval not run" >&2
  EVAL_RC=1
fi
echo "T-0208 eval exit $EVAL_RC"

echo "summary: harness $RUN eval $EVAL_RC $(date -Is)"
if [ "$RUN" -eq 2 ]; then exit 2; fi
if [ "$RUN" -ne 0 ] || [ "$EVAL_RC" -ne 0 ]; then
  exit 1
fi
exit 0
