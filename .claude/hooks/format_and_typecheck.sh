#!/usr/bin/env bash
# PostToolUse hook for Edit/Write: format the touched python file and type-check it.
# NOTE: PostToolUse exit 2 is advisory (the edit already happened); the blocking mypy run is in stop_fast_suite.sh.
set -uo pipefail
. "$(dirname "$0")/_lib.sh"
path=$(jf tool_input.file_path)
case "$path" in *.py) ;; *) exit 0;; esac
cd "$TOP" || exit 0
command -v uv >/dev/null 2>&1 || exit 0
uv run ruff format "$path" >/dev/null 2>&1 || true
uv run ruff check --fix "$path" >/dev/null 2>&1 || true
case "${path#"$TOP"/}" in
  local_astra/*)
    if ! out=$(uv run mypy "$path" 2>&1); then
      echo "mypy reported errors for $path (fix before finishing):" >&2; echo "$out" | tail -20 >&2; exit 2
    fi;;
esac
exit 0
