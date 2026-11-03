#!/usr/bin/env bash
# T-0071a a1 (criteria 1-2, ruling R6): container only, no engine. /launch only; run from the T-0071a worktree:
#
#   bash scripts/locked/t0071a_a1.sh 2>&1 | tee logs/t0071a-a1-run-<short sha>.log; echo "exit ${PIPESTATUS[0]}"
#
# 1. The per-step sync rows (criterion 1): tests/contract/test_vllm_cd_perstep_sync_gpu.py under -m gpu (exactly 7
#    passes pinned by the module), teed to logs/t0071a-a1-sync-<short sha>.log; its exit code from PIPESTATUS.
# 2. Only if that is 0 AND the log's summary line is exactly "1 passed" (the one host row wrapping the 7 container
#    rows; warnings allowed, nothing skipped/deselected/failed/errored): the microbench (criterion 2), scripts/gpu/cd_microbench.py, which writes
#    logs/t0071a-microbench-<short sha>/. A raise in the sync rows blocks the microbench and a2 (T-0069 O5 pattern).
#
# Exit code: 0 only if both ran and both exited 0; otherwise the first non-zero code (2 = refused here).
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2

SHA=$(git rev-parse --short=7 HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSE: dirty tree: $(git status --porcelain | tr '\n' ' ')" >&2; exit 2
fi
mkdir -p logs
SYNC_LOG="logs/t0071a-a1-sync-$SHA.log"
if [ -e "$SYNC_LOG" ]; then
  echo "REFUSE: $SYNC_LOG exists; move it aside first" >&2; exit 2
fi

echo "# t0071a a1 at $SHA, $(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free --format=csv,noheader

uv run pytest tests/contract/test_vllm_cd_perstep_sync_gpu.py -m gpu -rA -p no:cacheprovider 2>&1 | tee "$SYNC_LOG"
SYNC=${PIPESTATUS[0]}
echo "sync rows exit $SYNC $(date -Is)"

if [ "$SYNC" -ne 0 ]; then
  echo "microbench exit NOT RUN (the sync rows failed: a raise is a cd/** defect; a2 must not run)"
  exit "$SYNC"
fi
# The contract module is one host row wrapping the 7 container rows: its summary must be exactly "1 passed"
# (warnings allowed), with nothing skipped, deselected, failed or errored -- exit 0 alone is not enough.
SUMMARY=$(grep -E '^=+ .* in [0-9.]+s' "$SYNC_LOG" | tail -n 1)
if ! printf '%s\n' "$SUMMARY" | grep -Eq '^=+ 1 passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))? =+$'; then
  echo "microbench exit NOT RUN (the sync pytest exited 0 but its summary is not exactly '1 passed': '$SUMMARY')"
  exit 1
fi
if grep -Eiq '[0-9]+ (skipped|errors?|deselected|failed)' <<<"$SUMMARY"; then
  echo "microbench exit NOT RUN (the sync summary reports skipped/error/deselected/failed rows: '$SUMMARY')"
  exit 1
fi
echo "sync summary ok: $SUMMARY"

uv run --group cd python scripts/gpu/cd_microbench.py
MICRO=$?
echo "microbench exit $MICRO $(date -Is)"
echo "exit codes: sync $SYNC, microbench $MICRO"
if [ "$MICRO" -ne 0 ]; then exit "$MICRO"; fi
exit 0
