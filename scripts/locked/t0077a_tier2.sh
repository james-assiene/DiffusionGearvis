#!/usr/bin/env bash
# T-0077a tier 2 (umbrella criteria 1-3, task criteria 1, 3-4 on CUDA): container only, no engine. /launch only;
# run from the T-0077a worktree:
#
#   bash scripts/locked/t0077a_tier2.sh; echo "exit $?"
#
# Everything is teed to logs/t0077a-tier2-<short sha>.log (refused if it already exists; logs/ is gitignored).
# 1. scripts/locked/t0071a_a1.sh, unedited (umbrella criterion 3): its sync stage (the per-step sync rows,
#    tests/contract/test_vllm_cd_perstep_sync_gpu.py, summary exactly "1 passed" = 11 container rows, 0 raises),
#    then, only if that held, T-0071a's microbench (logs/t0071a-microbench-<sha>/, compared with 9a386ee by the
#    reader). Its own exit code; its own logs (logs/t0071a-a1-sync-<sha>.log).
# 2. T-0073's CUDA numerics module through its wrapper: tests/contract/test_vllm_cd_numerics_gpu.py::
#    test_criterion5_numerics_module_on_cuda (inner exactly 51 passes, N1); host summary exactly "1 passed".
# 3. T-0073's CUDA repeats module through its wrapper: tests/contract/test_vllm_cd_cuda_gpu.py::
#    test_core_repeats_and_memory_on_cuda (inner exactly 18 passes, N1/N2); host summary exactly "1 passed".
# 4. The same-process A/B, scripts/gpu/cd_kernel_ab.py (frozen e73cfa7 vs live per component and bucket; N3 on
#    CUDA, no component > 10 % slower, whole-call peak <= frozen), writing logs/t0077a-ab-<sha>/; exit 0 = PASS.
# Steps 2-4 run only if step 1's sync rows held (a raise there is a cd/** defect, T-0069 O5): read from
# t0071a_a1.sh's own "sync rows exit 0" and "sync summary ok" lines.
#
# Exit code: 0 only if all four exited 0 with the exact summaries; otherwise the first non-zero code
# (2 = refused here, 1 = a summary that is not the exact pass count, 4 = a step not run).
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2

SHA=$(git rev-parse --short=7 HEAD)
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSE: dirty tree: $(git status --porcelain | tr '\n' ' ')" >&2
  exit 2
fi
mkdir -p logs
LOG="logs/t0077a-tier2-$SHA.log"
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
  echo "# t0077a tier 2 at $SHA, $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free --format=csv,noheader
} 2>&1 | tee "$LOG"

TMP_A1=$(mktemp)
TMP_NUM=$(mktemp)
TMP_REP=$(mktemp)
trap 'rm -f "$TMP_A1" "$TMP_NUM" "$TMP_REP"' EXIT

echo "## step 1: t0071a_a1.sh (sync rows, then the microbench) $(date -Is)" | tee -a "$LOG"
bash scripts/locked/t0071a_a1.sh 2>&1 | tee "$TMP_A1" | tee -a "$LOG"
A1=${PIPESTATUS[0]}
echo "t0071a_a1.sh exit $A1 $(date -Is)" | tee -a "$LOG"
SYNC_OK=0
if grep -Eq '^sync rows exit 0 ' "$TMP_A1" && grep -Eq '^sync summary ok: ' "$TMP_A1"; then
  SYNC_OK=1
fi

NUM=4
REP=4
AB=4
if [ "$SYNC_OK" -ne 1 ]; then
  echo "steps 2-4 NOT RUN: the sync rows did not hold (a raise is a cd/** defect)" | tee -a "$LOG"
else
  echo "## step 2: T-0073 numerics module on CUDA (51 inner) $(date -Is)" | tee -a "$LOG"
  uv run pytest "tests/contract/test_vllm_cd_numerics_gpu.py::test_criterion5_numerics_module_on_cuda" \
    -m gpu -rA -p no:cacheprovider 2>&1 | tee "$TMP_NUM" | tee -a "$LOG"
  NUM=${PIPESTATUS[0]}
  echo "numerics wrapper exit $NUM $(date -Is)" | tee -a "$LOG"
  if [ "$NUM" -eq 0 ]; then
    check_summary "numerics" "$TMP_NUM" 1 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || NUM=1
  fi

  echo "## step 3: T-0073 repeats module on CUDA (18 inner) $(date -Is)" | tee -a "$LOG"
  uv run pytest "tests/contract/test_vllm_cd_cuda_gpu.py::test_core_repeats_and_memory_on_cuda" \
    -m gpu -rA -p no:cacheprovider 2>&1 | tee "$TMP_REP" | tee -a "$LOG"
  REP=${PIPESTATUS[0]}
  echo "repeats wrapper exit $REP $(date -Is)" | tee -a "$LOG"
  if [ "$REP" -eq 0 ]; then
    check_summary "repeats" "$TMP_REP" 1 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || REP=1
  fi

  echo "## step 4: same-process A/B, frozen vs live (cd_kernel_ab.py) $(date -Is)" | tee -a "$LOG"
  uv run --group cd python scripts/gpu/cd_kernel_ab.py 2>&1 | tee -a "$LOG"
  AB=${PIPESTATUS[0]}
  echo "A/B exit $AB $(date -Is)" | tee -a "$LOG"
fi

echo "exit codes: t0071a_a1 $A1, numerics $NUM, repeats $REP, ab $AB" | tee -a "$LOG"
for c in "$A1" "$NUM" "$REP" "$AB"; do
  if [ "$c" -ne 0 ]; then exit "$c"; fi
done
exit 0
