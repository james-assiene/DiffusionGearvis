#!/usr/bin/env bash
# T-0214 B: one fresh engine-only boot, then
#   1. scripts/gpu/t0214_threejs_both.py: P-A (the "both" arm's Session keywords differ from base only in its two
#      levers -- action_outcome added, reground False -> True, overview_frame dropped -- before any socket), the
#      pre-flight probes P1-P3 (T-0131's), P-M1 and P-M2 (T-0210 C4's; browser only, no inference) -- a failed probe
#      is exit 3 and a void report --, then M1-M5 x n = 3 = 15 typed turns in three blocks (order seed 0), each with
#      cli.LIVE_DEFAULT_ACTION_OUTCOME and cli.LIVE_DEFAULT_REGROUND rebound True for the turn (ReGround pass 2 at a
#      crop budget of 280 from "default"), every turn through T-0210 C's _one_turn as T-0212's harness runs it (a
#      fresh Session, StartPageServer(THREEJS_DIR) and web-mode executor with no profile directory on loopback with a
#      deny-all resolver and connector), search off, concurrency 1, 300 s per turn, no engine seed; the report
#      (T-0212's three arms quoted beside it as an unpaired anchor) is written straight after;
#   2. tests/eval/test_t0214_threejs_both.py -m gpu: the order, the arm evidence, the success, every A8 count, the
#      safety sums, the label and the reading marks re-derived from the run's records and session logs and checked
#      against its report.json (no GPU time of its own), only if report.json exists;
# then the engine is stopped. The boot is scripts/locked/t0212_threejs_arms.sh's lines 41-113 verbatim
# (services/vllm.sh at 806da12 and unchanged, prefix caching on, plugin wheel rebuilt, port and container refusals,
# one run per SHA, one boot); only the merge checks, the file list, the names and the timeout differ. STT must be
# down (its ports 8080/8081 are refused below).
#
# /launch only, from a clean tree at (or after) T-0214's merge commit; the console log's last line is the real
# exit code:
#
#   ( nohup bash scripts/locked/t0214_threejs_both.sh > logs/t0214-threejs-both-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0214-threejs-both-<SHA>-run.log ) &
#
# Refuses unless both T-0214's and T-0212's merges are ancestors of HEAD: merge commits reachable from HEAD whose
# subjects start "Merge T-0214:" and "Merge T-0212:"; a branch merge ("Merge branch 'main' into T-0212-x") does not
# count. The merge log is matched through a here-string, never a pipe (under pipefail `printf | grep -q` can fail on
# SIGPIPE once the log passes the pipe buffer). Also refused: services/vllm.sh changed since 806da12. The harness runs
# under `timeout -k 60s 25m` (a timeout is exit 124, a failed step; SIGKILL 60 s after the TERM); its own budget
# (RUN_BUDGET_S 18 min, and no turn starts unless a 480 s worst-case turn plus the 60 s report fit inside 25 min, both
# from process start, the probes included) is checked between turns, so the report is still written (estimate 11-12
# min). Both `uv run` steps use --locked. The proxy variables are unset for every step (loopback only).
# Exit: 0 every step passed; 1 a step failed (a failed probe's exit 3 and a void run's exit 3 included); 2 refused
# (here, or by the harness before any socket). Report-only: no bar, no threshold file changes, no default changes.
# T-0214 B; the anchor (T-0212 at 5b87b723) is quoted by the harness, never read from logs/ at run time.
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
if ! grep -Eq '^[0-9a-f]+ Merge T-0214:' <<<"$MERGES"; then
  echo "REFUSED: T-0214's merge commit is not an ancestor of HEAD (no 'Merge T-0214:' merge reachable from HEAD; a branch merge does not count); the harness is T-0214's" >&2; exit 2
fi
if ! grep -Eq '^[0-9a-f]+ Merge T-0212:' <<<"$MERGES"; then
  echo "REFUSED: T-0212's merge commit is not an ancestor of HEAD (no 'Merge T-0212:' merge reachable from HEAD; a branch merge does not count); T-0212's harness, which this run imports, is that merge's" >&2; exit 2
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
HARNESS=scripts/gpu/t0214_threejs_both.py
EVAL=tests/eval/test_t0214_threejs_both.py
# the harness imports T-0212's, which imports T-0210 C's, which imports T-0205 C's, which imports T-0134's, T-0132's,
# T-0131's and T-0111g's (and, through them, T-0138's suite table and the stats module); the scenario, suite and
# three.js tables, the copy's provenance, the tool sets
for f in "$HARNESS" "$EVAL" scripts/gpu/t0212_threejs_arms.py scripts/gpu/t0210_threejs_run.py scripts/gpu/t0205_demo_run.py scripts/gpu/t0134_feedback_ab.py scripts/gpu/t0132_crop_budget_ab.py scripts/gpu/t0131_grounding_ab.py scripts/gpu/t0111g_live_smoke.py scripts/gpu/t0138_suite_ab.py local_astra/toolcall/tools_v1.json local_astra/toolcall/tools_v1_wait.json local_astra/toolcall/tools_v1_overview.json local_astra/toolcall/tools_v1_overview_wait.json local_astra/pages/scenario_tasks.py local_astra/pages/suite_tasks.py local_astra/pages/threejs_tasks.py local_astra/pages/threejs/PROVENANCE.md local_astra/pages/threejs/editor/astra_state.js local_astra/pages/threejs/editor/astra_seeds.js local_astra/eval/cd/stats.py; do
  if [ ! -f "$f" ]; then
    echo "REFUSED: $f is absent" >&2; exit 2
  fi
done
OUT="logs/t0214-threejs-both-$SHA"
AUX="logs/t0214-launch-$SHA"
for d in "$OUT" "$AUX"; do
  if [ -e "$d" ]; then
    echo "REFUSED: $d exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
  fi
done
mkdir -p "$AUX"

make plugin-wheel || { echo "FAIL: make plugin-wheel failed" >&2; exit 1; }
{ echo "# t0214 threejs both boot, commit $SHA, $(date -Is)"
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

echo "== step 1: T-0214 both-arm harness (M1-M5 x 3, action outcome + ReGround, web mode on loopback, search off) $(date -Is)"
RUN=0
timeout -k 60s 25m "${NOPROXY[@]}" uv run --locked --all-extras python "$HARNESS" \
  --base-url http://127.0.0.1:8001/v1 --out-dir "$OUT" || RUN=$?
echo "T-0214 harness exit $RUN"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true

echo "== step 2: the order, arm evidence, metrics, safety, label and reading re-derived from the records $(date -Is)"
EVAL_RC=0
if [ -f "$OUT/report.json" ]; then
  echo "report: $OUT/report.md"
  for f in report.json trials.jsonl steps.jsonl probes.json order.json manifest.json; do
    if [ -f "$OUT/$f" ]; then echo "$f sha256 $(sha256sum "$OUT/$f" | cut -d' ' -f1)"; fi
  done
  "${NOPROXY[@]}" T0214_RUN_DIR="$OUT" \
    uv run --locked --all-extras pytest "$EVAL" -q -m gpu -rA \
    --junitxml="$AUX/t0214_eval.xml" || EVAL_RC=$?
else
  echo "no $OUT/report.json: the eval not run" >&2
  EVAL_RC=1
fi
echo "T-0214 eval exit $EVAL_RC"

echo "summary: harness $RUN eval $EVAL_RC $(date -Is)"
if [ "$RUN" -eq 2 ]; then exit 2; fi
if [ "$RUN" -ne 0 ] || [ "$EVAL_RC" -ne 0 ]; then
  exit 1
fi
exit 0
