#!/usr/bin/env bash
# T-0077a ruling L9: the A/B-only re-run (umbrella criteria 2-3 on CUDA). Container only, no engine. /launch only;
# run from the T-0077a worktree:
#
#   bash scripts/locked/t0077a_ab.sh; echo "exit $?"
#
# Runs exactly step 4 of scripts/locked/t0077a_tier2.sh and nothing else: the same host command,
# `uv run --group cd python scripts/gpu/cd_kernel_ab.py`, whose host side starts the one container (the pinned
# vllm/vllm-openai:v0.29.0 by digest, `docker run --rm --gpus all --network none`, the repository read-only at
# /repo, the host-compiled automata read-only at /cdpayloads, PYTHONPATH=/repo:/repo/plugin/astra_vllm_plugin,
# HF_HUB_OFFLINE=1, ASTRA_CONTAINER_DIGEST / ASTRA_COMMIT from the host) and writes the evidence directory
# logs/t0077a-ab-<short sha>/ (report.txt, lines.jsonl, container.log). The sync rows, the T-0073 numerics and
# repeats wrappers and T-0071a's microbench are NOT run here (tier 2 at b555c33 passed them; whether the L9 fix
# needs any of them again is the reviewers' call, and then it is tier 2, not this script).
#
# Everything this script prints is teed to logs/t0077a-ab-run-<short sha>.log (refused if it exists).
# Refuses (exit 2) a dirty tree (naming the paths), an existing run log or an existing evidence directory.
# Exit code: the A/B's own (0 = every bucket measured and every gate held, 1 = a gate failed or a bucket errored,
# 2 = refused by the host side), or 3 when the exit code and the evidence disagree (exit 0 without
# report.txt saying "verdict: PASS" for four buckets, or no evidence directory at all).
set -uo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 2

SHA=$(git rev-parse --short=7 HEAD)
DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
  echo "REFUSE: dirty tree: $(printf '%s' "$DIRTY" | tr '\n' ' ')" >&2
  exit 2
fi
mkdir -p logs
LOG="logs/t0077a-ab-run-$SHA.log"
EVID="logs/t0077a-ab-$SHA"
if [ -e "$LOG" ]; then
  echo "REFUSE: $LOG exists; move it aside first" >&2
  exit 2
fi
if [ -e "$EVID" ]; then
  echo "REFUSE: $EVID exists; move it aside first" >&2
  exit 2
fi

{
  echo "# t0077a A/B-only (ruling L9) at $SHA ($(git rev-parse HEAD)), $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free --format=csv,noheader
} 2>&1 | tee "$LOG"

echo "## same-process A/B, frozen vs live (cd_kernel_ab.py) $(date -Is)" | tee -a "$LOG"
uv run --group cd python scripts/gpu/cd_kernel_ab.py 2>&1 | tee -a "$LOG"
AB=${PIPESTATUS[0]}
echo "A/B exit $AB $(date -Is)" | tee -a "$LOG"

REPORT="$EVID/report.txt"
if [ ! -f "$REPORT" ]; then
  if [ "$AB" -eq 0 ]; then
    echo "FAIL: the A/B exited 0 but $REPORT does not exist" | tee -a "$LOG" >&2
    exit 3
  fi
  echo "FAIL: A/B exit $AB, no $REPORT (refused or crashed before writing evidence)" | tee -a "$LOG" >&2
  exit "$AB"
fi
NB=$(grep -Ec '^ +(64|128|256|512) N3 ' "$REPORT")
if [ "$AB" -eq 0 ]; then
  if ! grep -qx 'verdict: PASS' "$REPORT" || [ "$NB" -ne 4 ]; then
    echo "FAIL: the A/B exited 0 but $REPORT does not say 'verdict: PASS' for 4 buckets (N3 lines: $NB)" \
      | tee -a "$LOG" >&2
    exit 3
  fi
  echo "PASS: verdict PASS, 4 buckets, evidence $EVID" | tee -a "$LOG"
  exit 0
fi
echo "FAIL: A/B exit $AB; the PROBLEM lines of $REPORT:" | tee -a "$LOG" >&2
grep -E '^PROBLEM: ' "$REPORT" | tee -a "$LOG" >&2
exit "$AB"
