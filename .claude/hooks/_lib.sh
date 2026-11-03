#!/usr/bin/env bash
# Shared prelude for all hooks. Source it first: `. "$(dirname "$0")/_lib.sh"`.
# Reads the hook JSON from stdin once and exposes:
#   INPUT  — raw JSON
#   jf KEY — dotted-path lookup (strings printed raw, objects as JSON)
#   CWD    — the hook input's cwd (follows Claude into worktrees; $CLAUDE_PROJECT_DIR does not)
#   TOP    — git toplevel of CWD (worktree root)
#   STATE  — untracked dir shared by all worktrees of this repo (git common dir), for launch tokens
#   ROLE   — agent_type from the hook input ("none" in the main session)
#   TASK   — T-NNNN parsed from the current branch name (empty if none)
#   deny MSG — print BLOCKED and exit 2

# owned_globs TASK-FILE — print one ownership glob per line from the file's `## Owned files`
# section (T-0029). This is the ONE parser both protect_paths.sh (Edit/Write) and guard_bash.sh
# (Bash write shapes) use, so the two gates can never disagree about who owns what.
#   * a bullet (-, *, +; indented or not) yields its FIRST token only, backticks stripped, so an
#     annotation after the path (`*(new — the module)*`) or wrapped onto the next line is ignored;
#   * the section ends at the first heading of any level (^#{1,6}\s) OR at a "not own" marker
#     line: any line containing "does not own", "deliberately does not touch",
#     "ownership conflicts" or "claims no " (case-insensitive; T-0029 A7) — paths listed after it are never owned;
#   * a token is path-shaped only if it matches ^\.?[A-Za-z0-9_][A-Za-z0-9_.\-/]*(\*\*|/\*|\*)?$
#     with no ".." segment: no whitespace, no leading "/", no bold markup (**T-0017**), no lone
#     "*", no URL (":"), and it carries a "/" or a "." unless it is the bullet's only token
#     (`- Makefile`, T-0008); fenced blocks and prose yield nothing.
# Exit 0 with the list (possibly empty), 3 if the file cannot be read or the parser fails —
# the caller decides what an empty or failed parse means (protect_paths.sh: deny, criterion 6).
# It never runs at source time: only a hook that needs ownership calls it (criterion 12).
owned_globs() {
  python3 - "$1" <<'OWNEDPY'
import re, sys
path = sys.argv[1]
try:
    text = open(path, encoding="utf-8").read()
except OSError as e:
    print(f"cannot read {path}: {e.strerror or e}", file=sys.stderr); sys.exit(3)
except UnicodeDecodeError as e:
    print(f"cannot decode {path} as UTF-8 (owned-files parser in .claude/hooks/_lib.sh): {e.reason}",
          file=sys.stderr); sys.exit(3)
except Exception as e:
    print(f"owned-files parser in .claude/hooks/_lib.sh failed on {path}: {e!r}", file=sys.stderr)
    sys.exit(3)
HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
OWNED = re.compile(r"^\s{0,3}##+\s*Owned files\s*$", re.I)
MARKER = re.compile(r"does not own|deliberately does not touch|ownership conflicts|claims no ", re.I)
BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
SHAPE = re.compile(r"^\.?[A-Za-z0-9_][A-Za-z0-9_.\-/]*(\*\*|/\*|\*)?$")
lines = text.splitlines()
# find the heading outside any fenced block (a task file may quote another one's section)
fenced, start = False, None
for i, l in enumerate(lines):
    if l.lstrip().startswith("```"):
        fenced = not fenced
    elif not fenced and OWNED.match(l):
        start = i
        break
if start is None:
    print(f"no '## Owned files' section in {path}", file=sys.stderr); sys.exit(3)
fenced = False
for line in lines[start + 1:]:
    if line.lstrip().startswith("```"):
        fenced = not fenced
        continue
    if fenced:
        continue
    if HEADING.match(line) or MARKER.search(line):
        break
    m = BULLET.match(line)
    if not m:
        continue
    words = m.group(1).split()
    tok = words[0] if words else ""
    tok = tok.strip("`").rstrip(",;:")
    if tok.startswith("./"):
        tok = tok[2:]
    # a path has a separator or an extension ("the", "see", "T-0017" are prose), except that a
    # bullet made of exactly one token is a path whatever it looks like: `- Makefile` (A9)
    pathish = "/" in tok or "." in tok or len(words) == 1
    if not tok or ".." in tok.split("/") or not SHAPE.match(tok) or not pathish:
        continue
    print(tok)
OWNEDPY
}
if [ "${1:-}" = "owned-globs" ] && [ "${BASH_SOURCE[0]}" = "$0" ]; then
  owned_globs "${2:?task file}"; exit $?
fi
INPUT=$(cat)
jf() {
  printf '%s' "$INPUT" | python3 -c '
import sys, json
d = json.load(sys.stdin)
for k in sys.argv[1].split("."):
    d = d.get(k, "") if isinstance(d, dict) else ""
print(d if isinstance(d, str) else json.dumps(d))' "$1"
}
CWD=$(jf cwd); [ -n "$CWD" ] || CWD=$(pwd)
TOP=$(git -C "$CWD" rev-parse --show-toplevel 2>/dev/null || echo "$CWD")
STATE="$(git -C "$CWD" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || echo "$TOP/.git")/local_astra"
mkdir -p "$STATE" 2>/dev/null || true
ROLE=$(jf agent_type); [ -n "$ROLE" ] || ROLE=none
TASK=$(git -C "$CWD" branch --show-current 2>/dev/null | grep -oE 'T-[0-9]{4}[a-z]?' || true)
deny() { echo "BLOCKED ($ROLE): $1" >&2; exit 2; }
