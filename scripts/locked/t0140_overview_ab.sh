#!/usr/bin/env bash
# T-0140 B6: one fresh engine-only boot, then
#   1. scripts/gpu/t0140_overview_ab.py: the pre-flight probes P1-P3 (T-0131's), P6 (page_overview through the real
#      executor at 1280x800: deals.html, a page that fits, T-0134's hostile fixture; browser only, no inference) and P7
#      (the /tokenize count of an S4-like step-2 request carrying one overview, T-0132 P4's rule) -- a failed probe is
#      exit 3 and a void or inconclusive report --, then 2 arms (A: HEAD's live default, tools v1 bbfc53ccdd80; B: the
#      same plus page_overview, tools_v1_overview.json 8048bdb7aabe), both at HEAD's live default as cli gives it
#      (visual tokens, ReGround, live feedback, echo, thought-call recovery; read once, recorded in the manifest) with
#      the T-0142 guard on, x S1-S5 x n = 20 = 200 typed turns, shuffled jointly with seed 0, each through a fresh
#      Session with tools = live (headless, one browser per turn, loopback start pages only), the tool loop on
#      (max_tool_steps 8, repair on), the live guidance on, every arm value explicit, concurrency 1, no engine seed; the
#      report is written straight after the arms;
#   2. tests/eval/test_t0140_ab.py -m gpu: every metric, the configuration findings, the probes and the decision label
#      re-derived from the run's records and checked against its report.json (no GPU time of its own), only if
#      report.json exists;
# then the engine is stopped. The boot regime is scripts/locked/t0134_feedback_ab.sh's verbatim (services/vllm.sh at
# 806da12 and unchanged, prefix caching on, plugin wheel rebuilt, port and container refusals, one run per SHA, one
# boot). STT must be down (its ports 8080/8081 are refused below). /launch only, from a clean tree at (or after)
# T-0140's merge commit; the console log's last line is the real exit code:
#
#   ( nohup bash scripts/locked/t0140_overview_ab.sh > logs/t0140-overview-ab-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0140-overview-ab-<SHA>-run.log ) &
#
# Refuses unless T-0140's merge commit is an ancestor of HEAD: the merge SHA does not exist when this is written, so
# the check is a merge commit reachable from HEAD whose subject starts "Merge T-0140:" or "Merge T-0140 group B"
# (B-3; the "Merge T-0140 group A1"/"group A2" merges already on main do not count, nor do "Merge T-0140b",
# "Merge branch 'T-0140-...'" or "Merge main into T-0140-..."). The merge log is matched through a here-string, never a
# pipe: under pipefail `printf | grep -q` can fail when grep exits at the first match while printf is still writing
# (SIGPIPE once the log passes the 64 KiB pipe buffer), which would refuse a merged HEAD. The harness runs under
# `timeout -k 60s 75m` (a timeout is exit 124, a failed step; SIGKILL 60 s after the TERM); its own RUN_BUDGET_S is 62
# min from process start, the probes included, so the report is still written (estimate 30-40 min for the 200 turns:
# T-0132 ran 300 in 42 min). Both `uv run` steps use --locked: uv.lock is used as committed, never re-resolved. The
# proxy variables are unset for every step (loopback only). Exit: 0 every step passed; 1 a step failed (a failed
# probe's exit 3 included); 2 refused (here, or by the harness before any socket). Report-only plus the decision rule
# of H-T140-1: no bar, no threshold file changes.
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
if ! grep -Eq '^[0-9a-f]+ Merge T-0140(:| group B\b)' <<<"$MERGES"; then
  echo "REFUSED: T-0140's merge commit is not an ancestor of HEAD (no 'Merge T-0140:' or 'Merge T-0140 group B' merge reachable from HEAD; the group A1/A2 merges do not count); the A/B runs at the merge commit" >&2; exit 2
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
HARNESS=scripts/gpu/t0140_overview_ab.py
EVAL=tests/eval/test_t0140_ab.py
# the harness imports T-0134's, which imports T-0132's, T-0131's and T-0111g's (B1: imported, not copied); arm B's tool set
for f in "$HARNESS" "$EVAL" scripts/gpu/t0134_feedback_ab.py scripts/gpu/t0132_crop_budget_ab.py scripts/gpu/t0131_grounding_ab.py scripts/gpu/t0111g_live_smoke.py local_astra/toolcall/tools_v1_overview.json; do
  if [ ! -f "$f" ]; then
    echo "REFUSED: $f is absent" >&2; exit 2
  fi
done
OUT="logs/t0140-overview-ab-$SHA"
AUX="logs/t0140-launch-$SHA"
for d in "$OUT" "$AUX"; do
  if [ -e "$d" ]; then
    echo "REFUSED: $d exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
  fi
done
mkdir -p "$AUX"

make plugin-wheel || { echo "FAIL: make plugin-wheel failed" >&2; exit 1; }
{ echo "# t0140 page-overview A/B boot, commit $SHA, $(date -Is)"
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

echo "== step 1: T-0140 page-overview A/B harness $(date -Is)"
AB=0
timeout -k 60s 75m "${NOPROXY[@]}" uv run --locked --all-extras python "$HARNESS" \
  --base-url http://127.0.0.1:8001/v1 --out-dir "$OUT" || AB=$?
echo "T-0140 harness exit $AB"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true

echo "== step 2: the metrics, the findings, the probes and the decision re-derived from the records $(date -Is)"
EVAL_RC=0
if [ -f "$OUT/report.json" ]; then
  echo "report: $OUT/report.md"
  for f in report.json trials.jsonl steps.jsonl bodies.jsonl; do
    if [ -f "$OUT/$f" ]; then echo "$f sha256 $(sha256sum "$OUT/$f" | cut -d' ' -f1)"; fi
  done
  "${NOPROXY[@]}" T0140_RUN_DIR="$OUT" \
    uv run --locked --all-extras pytest "$EVAL" -q -m gpu -rA \
    --junitxml="$AUX/t0140_eval.xml" || EVAL_RC=$?
else
  echo "no $OUT/report.json: the eval not run" >&2
  EVAL_RC=1
fi
echo "T-0140 eval exit $EVAL_RC"

echo "summary: harness $AB eval $EVAL_RC $(date -Is)"
if [ "$AB" -eq 2 ]; then exit 2; fi
if [ "$AB" -ne 0 ] || [ "$EVAL_RC" -ne 0 ]; then
  exit 1
fi
exit 0
