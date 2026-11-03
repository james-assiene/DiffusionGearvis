#!/usr/bin/env bash
# SubagentStop hook (matcher: implementer|test-author|verifier): the report must contain a pytest command and its summary line.
# Uses last_assistant_message from the hook input (the subagent's own final message), not the transcript.
set -uo pipefail
. "$(dirname "$0")/_lib.sh"
case "$(jf stop_hook_active)" in true|True) exit 0;; esac   # one retry, then let it finish
last=$(jf last_assistant_message)
printf '%s' "$last" | grep -Eq 'pytest' \
  || { echo "Report must include the pytest command you ran and its output (evidence, not claims)." >&2; exit 2; }
printf '%s' "$last" | grep -Eq '([0-9]+ (passed|failed|error)|no tests ran)' \
  || { echo "Report must include the pytest summary line (e.g. '12 passed, 1 failed')." >&2; exit 2; }
exit 0
