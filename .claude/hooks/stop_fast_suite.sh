#!/usr/bin/env bash
# Stop / SubagentStop hook: if python files changed (tracked or untracked) in this worktree, run the fast suite and package mypy.
# Exit 2 blocks finishing and feeds the output back. stop_hook_active guards against loops.
#
# T-0019: failures are classified by module against docs/tasks/*.md (classify_failures.py). A
# failure confined to a module of a task whose **Status:** is `in progress` is the prescribed
# activate-then-implement window and is reported (`expected: T-NNNN (N failing in <module>,
# implementer pending)`) but tolerated; anything else — an unclaimed module, a `planned` task's
# module (markers removed without moving the status), a merged task's module — is a regression,
# listed first, and blocks. Tolerance never comes from `pending` markers in test sources.
set -uo pipefail
. "$(dirname "$0")/_lib.sh"
case "$(jf stop_hook_active)" in true|True) exit 0;; esac
cd "$TOP" || exit 0
# Python changed if the tree is dirty OR the current task branch has commits touching *.py that
# main does not (an implementer that committed must still pass the gate before finishing).
changed=$(git status --porcelain -- '*.py' 2>/dev/null)
if [ -z "$changed" ] && git rev-parse --verify -q main >/dev/null 2>&1; then
  changed=$(git diff --name-only main...HEAD -- '*.py' 2>/dev/null || true)
fi
[ -z "$changed" ] && exit 0
command -v uv >/dev/null 2>&1 || exit 0
# No -x: the classifier needs the whole failure set, and "N failing" must be a real count.
# T-0031: the same selection, run in parallel. `--dist loadfile` keeps a module's tests on one
# worker (the audio modules assert against elapsed time), a fixed `-n 12` (T-0143) rather than `auto`
# because the box also hosts vLLM. Nothing is deselected: `make fast` (serial) is the canonical
# run the verifier quotes; this run is the per-turn gate, budgeted at 60 s (tests/conftest.py
# STOP_HOOK_BUDGET_S). xdist's -rfE summary lines stay unprefixed at line start, which is what
# classify_failures.py reads; a worker crash leaves no FAILED line and takes the abort branch.
if ! out=$(timeout 110 uv run pytest tests/unit tests/contract -q -m "not gpu" -rfE -n 12 --dist loadfile 2>&1); then
  report=$(printf '%s\n' "$out" | python3 "$(dirname "$0")/classify_failures.py" "$TOP" 2>&1)
  rc=$?
  # A report is meaningful only if the classifier attributed something; stderr chatter that
  # exits 0 must not let the turn finish (ruling A14).
  if [ "$rc" -eq 0 ] && printf '%s\n' "$report" | grep -qE '^(expected|regression):'; then
    # Only expected failures: let the turn finish, but keep the red visible and attributed.
    printf '%s\n' "$report"
  elif [ "$rc" -eq 0 ]; then
    # pytest failed but no FAILED/ERROR line was found (collection crash, timeout): show the tail.
    echo "Fast suite failed; fix or report before finishing:" >&2; printf '%s\n' "$out" | tail -40 >&2; exit 2
  else
    echo "Fast suite failed; fix or report before finishing:" >&2
    printf '%s\n' "$report" >&2
    # The failing tests of the regression modules only (never the expected block's noise).
    printf '%s\n' "$report" | sed -n 's/^regression: [0-9]* failing in \([^ ]*\).*/\1/p' | sort -u | while read -r mod; do
      printf '%s\n' "$out" | grep -E "^(FAILED|ERROR) $mod" | head -20 >&2
    done
    exit 2
  fi
fi
if [ -d local_astra ] && ! out=$(uv run mypy local_astra 2>&1); then
  echo "mypy failed:" >&2; echo "$out" | tail -30 >&2; exit 2
fi
exit 0
