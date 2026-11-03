#!/usr/bin/env bash
# T-0223 B: one fresh engine-only boot, then
#   1. scripts/gpu/t0223_handwritten_check.py: the B1 refusals (P-S, the CPU settings probe, among them; all before
#      any socket), P0 (one text and one image generate at max_soft_tokens 1120 answer 200; failing is void, exit 3),
#      then 10 handwritten pages x n = 3 = 30 typed turns (order seed 0), each on a fresh tools-off Session at 1120
#      visual tokens, search off: hello -> astra.settings.set (the tutor instructions) -> echo at rev 1 -> one
#      astra.frame (the page at the HUD screen bucket) through a socketless HudServer -> UserText; concurrency 1,
#      90 s per turn, no engine seed; the report is written straight after;
#   2. tests/eval/test_t0223_handwritten_check.py -m gpu: every B3 check, B4 verdict, k/3, the reading, latency and
#      the label re-derived from the run's session logs, turns.jsonl, order.json and the pinned attempts.json and
#      checked against its report.json (no GPU time of its own), only if report.json exists;
# then the engine is stopped. The boot is scripts/locked/t0212_threejs_arms.sh's lines 41-113 verbatim (services/vllm.sh
# at 806da12 and unchanged, prefix caching on, plugin wheel rebuilt, port and container refusals, one run per SHA, one
# boot); only the merge subjects, the file list and the output names differ. STT must be down (its ports 8080/8081 are
# refused below).
#
# /launch only, from a clean tree at (or after) T-0223 group B's merge commit; the console log's last line is the real
# exit code:
#
#   ( nohup bash scripts/locked/t0223_handwritten_check.sh > logs/t0223-handwritten-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0223-handwritten-<SHA>-run.log ) &
#
# Refuses unless three merge commits are ancestors of HEAD: subjects starting "Merge T-0223-a:" (the attempts and their
# pages), "Merge T-0213-a2a:" (the settings set applied at the turn boundary) and "Merge T-0223-b:" (this harness); a
# branch merge ("Merge branch 'main' into T-0223-b-impl") or "Merge T-0223-ab:" does not count. The merge log is
# matched through a here-string, never a pipe (under pipefail `printf | grep -q` can fail on SIGPIPE once the log
# passes the pipe buffer). Also refused: services/vllm.sh changed since 806da12. The harness runs under
# `timeout -k 30s 15m` (a timeout is exit 124, a failed step; SIGKILL 30 s after the TERM); its own budget
# (RUN_BUDGET_S 14 min (R-4), and no turn starts unless a 90 s turn plus the 45 s report fit, both from process start, the
# probes included) is checked between turns, so the report is still written (estimate 8-14 min after the ~4 min boot).
# Both `uv run` steps use --locked. The proxy variables are unset for every step (loopback only).
# Exit: 0 every step passed; 1 a step failed (a failed P0's exit 3, a void run's exit 3 and an incomplete run's exit 1
# included); 2 refused (here, or by the harness before any socket). Report-only: no bar, no threshold file changes, no
# default changes.
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
if ! grep -Eq '^[0-9a-f]+ Merge T-0223-a:' <<<"$MERGES"; then
  echo "REFUSED: T-0223 group A's merge commit is not an ancestor of HEAD (no 'Merge T-0223-a:' merge reachable from HEAD; a branch merge does not count); the attempts and their pages are group A's" >&2; exit 2
fi
if ! grep -Eq '^[0-9a-f]+ Merge T-0213-a2a:' <<<"$MERGES"; then
  echo "REFUSED: T-0213 A2-apply's merge commit is not an ancestor of HEAD (no 'Merge T-0213-a2a:' merge reachable from HEAD; a branch merge does not count); the settings set every turn sends is A2-apply's" >&2; exit 2
fi
if ! grep -Eq '^[0-9a-f]+ Merge T-0223-b:' <<<"$MERGES"; then
  echo "REFUSED: T-0223 group B's merge commit is not an ancestor of HEAD (no 'Merge T-0223-b:' merge reachable from HEAD; a branch merge does not count); the harness and its eval are group B's" >&2; exit 2
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
HARNESS=scripts/gpu/t0223_handwritten_check.py
EVAL=tests/eval/test_t0223_handwritten_check.py
# the harness imports T-0208 C's, which imports T-0205 C's, which imports T-0134's, T-0132's, T-0131's and T-0111g's;
# group A's attempts and their ten page PNGs (their sha256s are pinned in the harness)
for f in "$HARNESS" "$EVAL" scripts/gpu/t0208_handoff_run.py scripts/gpu/t0205_demo_run.py scripts/gpu/t0134_feedback_ab.py scripts/gpu/t0132_crop_budget_ab.py scripts/gpu/t0131_grounding_ab.py scripts/gpu/t0111g_live_smoke.py local_astra/pages/scenario_tasks.py local_astra/pages/handoff_tasks.py demos/handwritten/attempts.json demos/handwritten/out/M1.png demos/handwritten/out/M2.png demos/handwritten/out/M3.png demos/handwritten/out/M4.png demos/handwritten/out/M5.png demos/handwritten/out/P1.png demos/handwritten/out/P2.png demos/handwritten/out/P3.png demos/handwritten/out/P4.png demos/handwritten/out/P5.png; do
  if [ ! -f "$f" ]; then
    echo "REFUSED: $f is absent" >&2; exit 2
  fi
done
OUT="logs/t0223-handwritten-$SHA"
AUX="logs/t0223-launch-$SHA"
for d in "$OUT" "$AUX"; do
  if [ -e "$d" ]; then
    echo "REFUSED: $d exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
  fi
done
mkdir -p "$AUX"

make plugin-wheel || { echo "FAIL: make plugin-wheel failed" >&2; exit 1; }
{ echo "# t0223 handwritten check boot, commit $SHA, $(date -Is)"
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

echo "== step 1: T-0223 handwritten check (10 pages x 3, tools off, search off, 1120 visual tokens, tutor instructions via settings) $(date -Is)"
RUN=0
timeout -k 30s 15m "${NOPROXY[@]}" uv run --locked --all-extras python "$HARNESS" \
  --base-url http://127.0.0.1:8001/v1 --out-dir "$OUT" || RUN=$?
echo "T-0223 harness exit $RUN"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true

echo "== step 2: the evidence, verdicts, k/3, reading, latency and label re-derived from the records $(date -Is)"
EVAL_RC=0
if [ -f "$OUT/report.json" ]; then
  echo "report: $OUT/report.md"
  for f in report.json trials.jsonl turns.jsonl probes.json order.json manifest.json; do
    if [ -f "$OUT/$f" ]; then echo "$f sha256 $(sha256sum "$OUT/$f" | cut -d' ' -f1)"; fi
  done
  "${NOPROXY[@]}" T0223_RUN_DIR="$OUT" \
    uv run --locked --all-extras pytest "$EVAL" -q -m gpu -rA \
    --junitxml="$AUX/t0223_eval.xml" || EVAL_RC=$?
else
  echo "no $OUT/report.json: the eval not run" >&2
  EVAL_RC=1
fi
echo "T-0223 eval exit $EVAL_RC"

echo "summary: harness $RUN eval $EVAL_RC $(date -Is)"
if [ "$RUN" -eq 2 ]; then exit 2; fi
if [ "$RUN" -ne 0 ] || [ "$EVAL_RC" -ne 0 ]; then
  exit 1
fi
exit 0
