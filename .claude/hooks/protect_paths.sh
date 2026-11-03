#!/usr/bin/env bash
# PreToolUse hook for Edit/Write/NotebookEdit. Enforces file ownership by role (agent_type).
# Exit 2 blocks the edit and returns stderr to the model.
set -euo pipefail
. "$(dirname "$0")/_lib.sh"
path=$(jf tool_input.file_path); [ -n "$path" ] || path=$(jf tool_input.notebook_path)
[ -z "$path" ] && exit 0
rel="${path#"$TOP"/}"
# A path inside a worktree under .claude/worktrees/<name>/ is judged by its path inside that
# worktree, not by where the worktree lives (2026-09-17: a test-author writing tests/ in its
# worktree was denied because the toplevel was the main checkout).
case "$rel" in .claude/worktrees/*/*) rel="${rel#.claude/worktrees/*/}";; esac

# Always protected
case "$rel" in
  services/versions.lock|uv.lock) deny "version locks change only via a dedicated task (see CLAUDE.md)";;
esac

# Every agent may write its own memory directory
case "$rel" in .claude/agent-memory/"$ROLE"/*) exit 0;; esac

# The test-author, the reviewer and the implementer may write scratch files (mutant copies, probes,
# dress-rehearsal drivers) in a session
# scratchpad, /tmp/claude-<uid>/<project>/<session>/scratchpad/, and nowhere else outside their
# rules (human, 2026-10-22: blocking it made them improvise heredocs, inline stdin scripts and
# temporary harnesses under tests/; the implementer was added the same day so it can dress-rehearse
# GPU harnesses on CPU). The resolved path must stay inside that scratchpad and outside
# this worktree, so a symlink or `..` cannot turn it into a repo write.
case "$ROLE" in
  test-author|reviewer|implementer)
    case "$path" in
      /tmp/claude-*/*/*/scratchpad/*)
        real=$(realpath -m -- "$path" 2>/dev/null || true)
        pad=${path%%/scratchpad/*}/scratchpad
        realpad=$(realpath -m -- "$pad" 2>/dev/null || true)
        case "$real" in
          "$realpad"/*)
            case "$real" in "$TOP"/*|"$TOP") ;; *) exit 0;; esac;;
        esac;;
    esac;;
esac

case "$ROLE" in
  implementer)
    case "$rel" in
      tests/*) deny "implementer may not edit tests/. Report the problem to the reviewer instead.";;
      .claude/*|CLAUDE.md) deny "implementer may not edit agent configuration.";;
      docs/03-test-plan.md) deny "implementer may not edit the test plan / thresholds.";;
    esac
    [ -n "$TASK" ] || deny "no task branch: run 'git switch -c T-NNNN-<slug>' before editing"
    # One parser for both gates (T-0029, ruling A2): owned_globs in _lib.sh. Fail closed with a
    # name: a missing file, a missing section, a parser failure or an empty list all deny.
    task_file="$TOP/docs/tasks/$TASK.md"
    if ! owned=$(owned_globs "$task_file" 2>/dev/null); then
      deny "cannot read the '## Owned files' section of docs/tasks/$TASK.md ($(owned_globs "$task_file" 2>&1 >/dev/null || true)); fix the task file before editing"
    fi
    [ -n "$owned" ] || deny "the '## Owned files' section of docs/tasks/$TASK.md lists no path; ask the planner to fill it before editing"
    match=0
    while IFS= read -r pat; do case "$rel" in $pat) match=1;; esac; done <<< "$owned"
    [ "$match" = 1 ] || deny "$rel is not in the owned files of task $TASK. Ask the planner to re-scope.";;
  test-author)
    case "$rel" in tests/*) ;; *) deny "test-author may edit tests/ only.";; esac;;
  planner)
    case "$rel" in docs/tasks/*) ;; *) deny "planner may write task files only.";; esac;;
  reviewer|verifier|launch-reviewer|explorer)
    deny "$ROLE is read-only.";;
esac
exit 0
