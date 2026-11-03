#!/usr/bin/env bash
# T-0077c tier 2 (criteria 1-9 on CUDA; umbrella criteria 1-3): container only, no engine. /launch only; run from
# the T-0077c worktree:
#
#   bash scripts/locked/t0077c_tier2.sh; echo "exit $?"
#
# Everything is teed to logs/t0077c-tier2-<short sha>.log (refused if it already exists; logs/ is gitignored).
# 1. scripts/locked/t0071a_a1.sh, unedited (umbrella criterion 3): its sync stage (the per-step sync rows,
#    tests/contract/test_vllm_cd_perstep_sync_gpu.py, summary exactly "1 passed" = 11 container rows, 0 raises),
#    then, only if that held, T-0071a's microbench (logs/t0071a-microbench-<sha>/, compared with 9a386ee by the
#    reader). Its own exit code; its own logs (logs/t0071a-a1-sync-<sha>.log).
# 2. T-0073's CUDA numerics module through its wrapper: tests/contract/test_vllm_cd_numerics_gpu.py::
#    test_criterion5_numerics_module_on_cuda (inner exactly 51 passes, N1; now through the log_matmul kernel);
#    host summary exactly "1 passed".
# 3. T-0073's CUDA repeats module through its wrapper: tests/contract/test_vllm_cd_cuda_gpu.py::
#    test_core_repeats_and_memory_on_cuda (inner exactly 18 passes, N1/N2); host summary exactly "1 passed".
# 4. T-0077b's CUDA kernel module (as retargeted under ruling O3), twice, through its wrapper (ruling K6):
#    uv run --group cd pytest tests/contract/test_vllm_cd_kernels_gpu.py::
#    test_kernels_module_twice_on_cuda_with_equal_digests -m gpu; host summary exactly "1 passed", and the two
#    inner summaries in its captured output exactly "34 passed" each (K = 34).
# 5. T-0077c's CUDA log_matmul module, twice (two container processes, equal digests), through its wrapper:
#    uv run --group cd pytest tests/contract/test_vllm_cd_logmatmul_gpu.py::
#    test_logmatmul_module_twice_on_cuda_with_equal_digests -m gpu; host summary exactly "1 passed", and the two
#    inner summaries in its captured output exactly "43 passed" each (K_c = 43).
# 6. The same-process A/B, scripts/gpu/cd_kernel_ab.py (frozen b53457a vs live with the log_matmul kernel
#    dispatched, per component and bucket: N3, N4 (dq, dH), log_a/log_b dead sets, repeat, V2 dispatch observed,
#    no row > 10 % slower, whole-call peak <= frozen), writing logs/t0077c-ab-<sha>/; exit 0 and report.txt saying
#    "verdict: PASS" for 4 buckets.
# Steps 2-6 run only if step 1's sync rows held (a raise there is a cd/** defect, T-0069 O5): read from
# t0071a_a1.sh's own "sync rows exit 0" and "sync summary ok" lines.
#
# Exit code: 0 only if all six exited 0 with the exact summaries; otherwise the first non-zero code
# (2 = refused here, 1 = a summary that is not the exact pass count, 3 = the A/B's exit code and its report
# disagree, 4 = a step not run).
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2

SHA=$(git rev-parse --short=7 HEAD)
DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
  echo "REFUSE: dirty tree: $(printf '%s' "$DIRTY" | tr '\n' ' ')" >&2
  exit 2
fi
mkdir -p logs
LOG="logs/t0077c-tier2-$SHA.log"
EVID="logs/t0077c-ab-$SHA"
if [ -e "$LOG" ]; then
  echo "REFUSE: $LOG exists; move it aside first" >&2
  exit 2
fi
if [ -e "$EVID" ]; then
  echo "REFUSE: $EVID exists; move it aside first" >&2
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

# check_inner <label> <file> <k> <runs>: exactly <runs> inner pytest summaries "<k> passed" (warnings allowed) in
# the captured container output, and no inner summary of any other shape.
check_inner() {
  local label=$1 file=$2 k=$3 runs=$4 good all
  good=$(grep -Ec "^=+ $k passed(, [0-9]+ warnings?)? in [0-9.]+s( \([0-9:]+\))? =+\$" "$file")
  all=$(grep -Ec '^=+ .* in [0-9.]+s( \([0-9:]+\))? =+$' "$file")
  # the host's own summary line ("1 passed") is the one summary that is not an inner one
  if [ "$good" -ne "$runs" ] || [ "$all" -ne $((runs + 1)) ]; then
    echo "$label inner summaries: $good of '$k passed' (want $runs), $all summary lines in all (want $((runs + 1)))"
    grep -E '^=+ .* in [0-9.]+s' "$file"
    return 1
  fi
  echo "$label inner summaries ok: $runs x '$k passed'"
  return 0
}

{
  echo "# t0077c tier 2 at $SHA ($(git rev-parse HEAD)), $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free --format=csv,noheader
} 2>&1 | tee "$LOG"

TMP_A1=$(mktemp)
TMP_NUM=$(mktemp)
TMP_REP=$(mktemp)
TMP_KER=$(mktemp)
TMP_LM=$(mktemp)
trap 'rm -f "$TMP_A1" "$TMP_NUM" "$TMP_REP" "$TMP_KER" "$TMP_LM"' EXIT

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
KER=4
LM=4
AB=4
if [ "$SYNC_OK" -ne 1 ]; then
  echo "steps 2-6 NOT RUN: the sync rows did not hold (a raise is a cd/** defect)" | tee -a "$LOG"
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

  echo "## step 4: T-0077b kernel module on CUDA, two processes (34 inner each) $(date -Is)" | tee -a "$LOG"
  uv run --group cd pytest \
    "tests/contract/test_vllm_cd_kernels_gpu.py::test_kernels_module_twice_on_cuda_with_equal_digests" \
    -m gpu -rA -p no:cacheprovider 2>&1 | tee "$TMP_KER" | tee -a "$LOG"
  KER=${PIPESTATUS[0]}
  echo "kernels wrapper exit $KER $(date -Is)" | tee -a "$LOG"
  if [ "$KER" -eq 0 ]; then
    check_summary "kernels" "$TMP_KER" 1 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || KER=1
  fi
  if [ "$KER" -eq 0 ]; then
    check_inner "kernels" "$TMP_KER" 34 2 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || KER=1
  fi

  echo "## step 5: T-0077c log_matmul module on CUDA, two processes (43 inner each) $(date -Is)" | tee -a "$LOG"
  uv run --group cd pytest \
    "tests/contract/test_vllm_cd_logmatmul_gpu.py::test_logmatmul_module_twice_on_cuda_with_equal_digests" \
    -m gpu -rA -p no:cacheprovider 2>&1 | tee "$TMP_LM" | tee -a "$LOG"
  LM=${PIPESTATUS[0]}
  echo "logmatmul wrapper exit $LM $(date -Is)" | tee -a "$LOG"
  if [ "$LM" -eq 0 ]; then
    check_summary "logmatmul" "$TMP_LM" 1 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || LM=1
  fi
  if [ "$LM" -eq 0 ]; then
    check_inner "logmatmul" "$TMP_LM" 43 2 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || LM=1
  fi

  echo "## step 6: same-process A/B, frozen b53457a vs live with the log_matmul kernel (cd_kernel_ab.py) $(date -Is)" \
    | tee -a "$LOG"
  uv run --group cd python scripts/gpu/cd_kernel_ab.py 2>&1 | tee -a "$LOG"
  AB=${PIPESTATUS[0]}
  echo "A/B exit $AB $(date -Is)" | tee -a "$LOG"
  REPORT="$EVID/report.txt"
  if [ "$AB" -eq 0 ]; then
    NB=0
    [ -f "$REPORT" ] && NB=$(grep -Ec '^ +(64|128|256|512) N3 ' "$REPORT")
    if [ ! -f "$REPORT" ] || ! grep -qx 'verdict: PASS' "$REPORT" || [ "$NB" -ne 4 ]; then
      echo "FAIL: the A/B exited 0 but $REPORT does not say 'verdict: PASS' for 4 buckets (N3 lines: $NB)" \
        | tee -a "$LOG"
      AB=3
    fi
  elif [ -f "$REPORT" ]; then
    echo "the PROBLEM lines of $REPORT:" | tee -a "$LOG"
    grep -E '^PROBLEM: ' "$REPORT" | tee -a "$LOG"
  fi
fi

echo "exit codes: t0071a_a1 $A1, numerics $NUM, repeats $REP, kernels $KER, logmatmul $LM, ab $AB" | tee -a "$LOG"
for c in "$A1" "$NUM" "$REP" "$KER" "$LM" "$AB"; do
  if [ "$c" -ne 0 ]; then exit "$c"; fi
done
exit 0
