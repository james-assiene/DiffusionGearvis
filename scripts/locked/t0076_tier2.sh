#!/usr/bin/env bash
# T-0076 tier 2 (criteria 11, 12): container only, no engine. /launch only; run from the T-0076 worktree:
#
#   bash scripts/locked/t0076_tier2.sh; echo "exit $?"
#
# Everything is teed to logs/t0076-tier2-<short sha>.log (refused if it already exists).
# 1. Criterion 11: tests/contract/test_vllm_cd_perstep_sync_gpu.py under -m gpu -- one host row wrapping the 11
#    container sync rows (liveness + perstep_mf, perstep_mar, perstep_mar + SC, perstep_mar_qargmax ± SC x buckets
#    64/128, sync-debug "error"); its summary must be exactly "1 passed".
# 2. Criterion 12: tests/contract/test_vllm_cd_perstep_gpu.py under -m gpu -- three host rows (T-0068's two
#    arms-semantics modules on CUDA, and the CUDA/memory module with its 8 rows incl. T-0076's qargmax memory and
#    semantics rows); its summary must be exactly "3 passed". Runs whatever step 1 returned (both are container-only
#    checks; neither feeds the other), so one launch reports both.
#
# Exit code: 0 only if both exited 0 with the exact summaries; otherwise the first non-zero code
# (2 = refused here, 1 = a summary that is not the exact pass count).
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2

SHA=$(git rev-parse --short=7 HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSE: dirty tree: $(git status --porcelain | tr '\n' ' ')" >&2
  exit 2
fi
mkdir -p logs
LOG="logs/t0076-tier2-$SHA.log"
if [ -e "$LOG" ]; then
  echo "REFUSE: $LOG exists; move it aside first" >&2
  exit 2
fi

# check_summary <label> <file> <n>: the last pytest summary line in <file> is exactly "<n> passed" (warnings
# allowed), nothing skipped, deselected, failed or errored.
check_summary() {
  local label=$1 file=$2 n=$3 summary
  summary=$(grep -E '^=+ .* in [0-9.]+s' "$file" | tail -n 1)
  if ! printf '%s\n' "$summary" | grep -Eq "^=+ $n passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))? =+\$"; then
    echo "$label summary is not exactly '$n passed': '$summary'"
    return 1
  fi
  if grep -Eiq '[0-9]+ (skipped|errors?|deselected|failed|xfailed|xpassed)' <<<"$summary"; then
    echo "$label summary reports skipped/error/deselected/failed rows: '$summary'"
    return 1
  fi
  echo "$label summary ok: $summary"
  return 0
}

{
  echo "# t0076 tier 2 at $SHA, $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free --format=csv,noheader
} 2>&1 | tee "$LOG"

TMP_SYNC=$(mktemp)
TMP_MEM=$(mktemp)
trap 'rm -f "$TMP_SYNC" "$TMP_MEM"' EXIT

echo "## criterion 11: sync rows $(date -Is)" | tee -a "$LOG"
uv run pytest tests/contract/test_vllm_cd_perstep_sync_gpu.py -m gpu -rA -p no:cacheprovider 2>&1 \
  | tee "$TMP_SYNC" | tee -a "$LOG"
SYNC=${PIPESTATUS[0]}
echo "sync rows exit $SYNC $(date -Is)" | tee -a "$LOG"
if [ "$SYNC" -eq 0 ]; then
  check_summary "sync" "$TMP_SYNC" 1 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || SYNC=1
fi

echo "## criterion 12: CUDA semantics and memory $(date -Is)" | tee -a "$LOG"
uv run pytest tests/contract/test_vllm_cd_perstep_gpu.py -m gpu -rA -p no:cacheprovider 2>&1 \
  | tee "$TMP_MEM" | tee -a "$LOG"
MEM=${PIPESTATUS[0]}
echo "cuda/memory rows exit $MEM $(date -Is)" | tee -a "$LOG"
if [ "$MEM" -eq 0 ]; then
  check_summary "cuda/memory" "$TMP_MEM" 3 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -eq 0 ] || MEM=1
fi

echo "exit codes: sync $SYNC, cuda/memory $MEM" | tee -a "$LOG"
if [ "$SYNC" -ne 0 ]; then exit "$SYNC"; fi
if [ "$MEM" -ne 0 ]; then exit "$MEM"; fi
exit 0
