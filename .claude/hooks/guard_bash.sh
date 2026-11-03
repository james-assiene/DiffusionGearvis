#!/usr/bin/env bash
# PreToolUse hook for Bash. Blocks destructive git, test silencing, and unreviewed GPU/artifact runs.
set -euo pipefail
. "$(dirname "$0")/_lib.sh"
cmd=$(jf tool_input.command)
[ -z "$cmd" ] && exit 0
has() { printf '%s' "$cmd" | grep -Eq -- "$1"; }
SQ="'"; Q="[${SQ}\"]"   # either quote character, for use inside patterns

# --- what the GPU rules below are allowed to see (T-0016) ------------------------------------
# They match the executable shape of the command, not text it merely quotes: a commit message or
# a grep pattern that *mentions* an engine launch spends no GPU time. So heredoc bodies and
# quoted spans are blanked before matching, with two exceptions that would re-open defect A:
#   * the argument of `-c` survives -- `--entrypoint bash "$CONTAINER" -c "..."` (services/vllm.sh)
#     and `sg docker -c "..."` are how we actually start the engine;
#   * a heredoc body survives when the command also names an interpreter (T-0016 D1):
#     `bash <<SH ... SH` executes its body; a commit taking its message on stdin only records it.
# Only the GPU rules below use this view. The rules above keep matching the raw text, because
# blanking quoted spans there would let a negated -k selector past the test-silencing guard.
#
# The same pass answers the one question a regex cannot: does this command *execute* a file under
# a gated directory (scripts/gpu/, scripts/locked/)? It prints that verdict as its first line.
#   * a `cd` (or `--directory`) is resolved first, so `cd scripts/gpu && ./x3.sh` is read as the
#     command `bash scripts/gpu/x3.sh` wearing a different working directory (T-0016 I1);
#   * a file is executed when it is the program itself, or the first non-flag argument of an
#     interpreter -- wrappers (`uv run`, `sudo`, `time`, `nohup`) are peeled off to find that
#     program, `-m`/`-c` re-target it (what follows them is what runs), and a shell given `-n`
#     parses without running. So `cat`, `ruff check`, `bash -n` and `py_compile` read a harness
#     for free, while running it under any spelling does not (T-0016 I9);
#   * quotes are kept for this verdict and stripped per token instead, because `bash "$TOP/x3.sh"`
#     is the same command as `bash $TOP/x3.sh` -- an agent quotes a path by reflex (T-0016 I10);
#   * `timeout 600 bash x3.sh` runs the harness, so `timeout` is a wrapper whose duration argument
#     is consumed with it, and `source`/`.` run the file in the current shell (T-0016 I11).
# `SUB` holds the subcommand words a wrapper takes before the real program (`uv run`), `VALUED`
# the flags that swallow the next token so it is not mistaken for that program, and `CHDIR` the
# spellings that move the working directory. Indirection defeats all of this by design: see
# `.claude/rules/gpu-services.md` -- the guard is a speed bump, not a sandbox.
# Falls back to the raw command if python3 cannot normalise it, i.e. to the previous behaviour.
norm=$(printf '%s' "$cmd" | python3 -c '
import os, re, sys
s = sys.stdin.read()
HEREDOC = re.compile(r"<<-?[ \t]*([\x27\"]?)(\w+)\1([^\n]*)\n.*?^\2", re.S | re.M)
QUOTED = re.compile(r"\x27[^\x27]*\x27|\"(?:[^\"\\]|\\.)*\"")
DASH_C = re.compile(r"(^|\s)-[A-Za-z]*c[ \t]*$")
SHELL = re.compile(r"(^|[\s;|&()])(bash|sh|zsh|ksh|dash|sg|ssh|docker|podman|python3?|perl|ruby"
                   r"|node|uv|eval|source|exec)([\s;|&()]|$)")
GATED = re.compile(r"(^|/)scripts/(gpu|locked)/")
SHELLS = ("bash", "sh", "zsh", "ksh", "dash")
INTERP = SHELLS + ("python", "python3", "perl", "ruby", "node", "source", ".")
WRAP = ("sudo", "env", "time", "timeout", "nohup", "watch", "xargs", "command", "exec",
        "stdbuf", "nice", "ionice", "uv", "uvx", "poetry")
TIMED = ("timeout",)
DURATION = re.compile(r"^[0-9]+([.][0-9]+)?[smhd]?$")
SUB = ("run", "tool", "x")
VALUED = ("-m", "--module", "--with", "--python", "-p", "--project", "-C",
          "--directory", "--chdir", "--cwd")
# Flags that take their argument as a *separate* token, per program (T-0026). The same letter
# means different things to different programs -- `-n` is the adjustment for nice (valued) but
# non-interactive for sudo (not) and parse-only for a shell (not) -- so one global list would
# open a hole for every wrapper it closed for another. These are enumerated tables, not a parser:
# an unrecognised flag consumes one token, which keeps the program in place only when that flag
# takes no argument. A flag the table does not know *and* that takes a separate argument
# (`nice -q 10`, `sudo -T 600`) still ends the peel -- extend the table when such a spelling is
# met. Shell clusters (`-euo pipefail`) are read by their last letter when it is `o`/`O`
# (valued) or `c` (re-target); `-n` anywhere in a shell cluster means "parse only" and wins
# (T-0016 I9); other clusters consume nothing beyond themselves. A shell never consults the
# shared VALUED list (`-p`, `-C` are no-argument shell options).
# Two residuals the rules file cannot yet record (T-0026 D-a; it is owned by T-0014/T-0015):
#   * a heredoc fed to `python3 -` is scanned as executable by design (T-0016 D1), so prose inside
#     it that names an engine start is refused -- write the script to a file and run the file;
#   * a `cd ... && ...` written inside a quoted message can be read as a real `cd`; both fail
#     closed (nuisance, not hole).
WRAP_VALUED = {
    "nice": ("-n", "--adjustment"),
    "ionice": ("-c", "--class", "-n", "--classdata", "-p", "--pid"),
    "sudo": ("-u", "--user", "-g", "--group", "-U", "--other-user", "-C", "--close-from",
             "-p", "--prompt", "-r", "--role", "-t", "--type", "-D", "--chdir"),
    "timeout": ("-s", "--signal", "-k", "--kill-after"),
    "env": ("-u", "--unset", "-S", "--split-string", "-C", "--chdir"),
    "stdbuf": ("-i", "--input", "-o", "--output", "-e", "--error"),
    "xargs": ("-n", "-P", "-I", "-d", "-a", "-E", "-L", "-s"),
    "watch": ("-n", "--interval"),
}
SHELL_VALUED = ("-o", "-O", "+o", "+O", "--rcfile", "--init-file")
PY_VALUED = ("-W", "-X", "--check-hash-based-pycs")
CHDIR = ("cd", "-C", "--directory", "--chdir", "--cwd")
SEG = re.compile(r"\|\||&&|[;|&\n]")
ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*=")

def unquote(t):
    return QUOTED.sub(lambda m: m.group(0) if DASH_C.search(t[:m.start()]) else " ", t)

def bare(tok):
    return tok.strip(" \t\n\r\x27\"")

def resolve_cd(text):
    out, cwd, want = [], "", False
    for part in re.split(r"(\s+)", text):
        tok = bare(part)
        if not tok:
            out.append(part)
        elif want:
            want = False
            cwd = cwd if tok.startswith("-") else tok
            out.append(part)
        elif tok in CHDIR:
            want = True
            out.append(part)
        elif cwd and not tok.startswith(("/", "-")) and (
                tok.startswith("./") or tok.endswith((".sh", ".py"))):
            out.append(os.path.normpath(os.path.join(cwd, tok)))
        else:
            out.append(part)
    return "".join(out)

def runs_gated(toks):
    while toks and (ASSIGN.match(toks[0]) or toks[0] in WRAP):
        if toks[0] in WRAP:
            prog = toks[0]
            timed = prog in TIMED
            valued = VALUED + WRAP_VALUED.get(prog, ())
            toks = toks[1:]
            while toks and (toks[0].startswith("-") or toks[0] in SUB):
                toks = toks[2:] if toks[0] in valued else toks[1:]
            while timed and toks and DURATION.match(toks[0]):
                toks = toks[1:]
        else:
            toks = toks[1:]
    if not toks:
        return False
    if GATED.search(toks[0]):
        return True
    if os.path.basename(toks[0]) not in INTERP:
        return False
    shell = os.path.basename(toks[0]) in SHELLS
    i = 1
    while i < len(toks) and toks[i].startswith(("-", "+")):
        tok = toks[i]
        cluster = re.fullmatch(r"-[A-Za-z]+", tok)
        if shell and re.fullmatch(r"-[A-Za-z]*n[A-Za-z]*", tok):
            return False   # parse only, whatever else the cluster says (`-nc` included)
        if tok in ("-m", "--module", "-c") or (shell and cluster and tok.endswith("c")):
            return runs_gated(toks[i + 1:])
        if shell and (tok in SHELL_VALUED or (cluster and tok[-1] in "oO")):
            i += 2
        elif not shell and tok in VALUED:
            i += 2
        elif os.path.basename(toks[0]).startswith("python") and tok in PY_VALUED:
            i += 2   # -W/-X take an argument for python only; perl/ruby/node -W take none
        else:
            # A shell never consults the shared list: `bash -p` (privileged) and `bash -C`
            # (noclobber) take no argument, and would otherwise swallow the script.
            i += 1
    return i < len(toks) and bool(GATED.search(toks[i]))

bodyless = HEREDOC.sub(lambda m: " " + m.group(3) + " ", s)
kept = s if SHELL.search(unquote(bodyless)) else bodyless
scan = unquote(kept)
gated = any(runs_gated([bare(t) for t in seg.split() if bare(t)])
            for seg in SEG.split(resolve_cd(kept)))

# Candidate write targets (T-0029 criterion 9): the file a redirection, tee, sed -i, cp, mv or rm
# lands on. Read from the quote-preserving, heredoc-less view -- a quoted path is still a path
# (T-0016 I10) and a heredoc body is data, not a command -- and tokenised with shell quoting so
# a `>` inside a commit message is one token, not an operator. Bash decides what a target means
# (in-repo? owned?); python only names them. Interpreters that write files from a heredoc body
# (`python3 - <<PY ... open(p, "w")`) are the accepted residual D-a.
import shlex
REDIR = re.compile(r"^[0-9]*&?>{1,2}\|?")
def write_targets(text):
    out = []
    for seg in SEG.split(resolve_cd(text)):
        try:
            toks = shlex.split(seg, posix=False)
        except ValueError:
            continue
        raw = [t for t in toks if bare(t)]
        toks = [bare(t) for t in raw]
        i, prog, args = 0, None, []
        while i < len(toks):
            t, r = toks[i], raw[i]  # operators are read unquoted: a quoted > is a grep pattern, a bare > writes
            m = REDIR.match(r)
            if m:
                # `> f`, `>> f`, `2> f`, `&> f` take the next token; `>f`, `2>/dev/null` carry it;
                # `2>&1` and `>&2` duplicate a descriptor and name no file
                rest = r[m.end():]
                if rest.startswith("&"):
                    i += 1; continue
                if rest:
                    out.append(bare(rest)); i += 1; continue
                if i + 1 < len(toks):
                    out.append(toks[i + 1]); i += 2; continue
                i += 1; continue
            if prog is None and not ASSIGN.match(t):
                prog = t
            elif prog is not None:
                args.append(t)
            i += 1
        base = (prog or "").rsplit("/", 1)[-1]
        # a valued flag consumes the next token (`sed -e SCRIPT`, `sed -f FILE`) so a script is
        # never mistaken for a file
        plain, skip = [], False
        for a in args:
            if skip:
                skip = False; continue
            if a in ("-e", "--expression", "-f", "--file"):
                skip = True; continue
            if not a.startswith("-"):
                plain.append(a)
        if base == "tee":
            out += plain
        elif base == "sed" and any(a.startswith("-i") or a.startswith("--in-place") for a in args):
            scripted = any(a in ("-e", "--expression", "-f", "--file") for a in args)
            out += plain if scripted else plain[1:]
        elif base in ("cp", "mv") and len(plain) >= 2:
            dest = plain[-1]
            if dest.endswith("/"):
                out += [dest + src.rsplit("/", 1)[-1] for src in plain[:-1]]
            else:
                out.append(dest)
        elif base == "rm":
            out += plain
    return [t for t in out if t and not t.startswith(("&", "-"))]

sys.stdout.write("%d%s\n%s" % (gated, "".join("\t" + t for t in write_targets(bodyless)), scan))
') || norm="0
$cmd"
first=${norm%%$'\n'*}
gated=${first%%$'\t'*}  # 1 when the command executes a file under a gated directory
targets=${first#"$gated"} # tab-separated candidate write targets (T-0029), empty when none
scan=${norm#*$'\n'}     # the executable shape of the command, for the patterns below
eff() { printf '%s' "$scan" | grep -Eq -- "$1"; }

# Destructive / history-rewriting git (branch names containing -f are not matched)
has 'git[[:space:]]+(push([[:space:]]+[^[:space:]-][^[:space:]]*)*[[:space:]]+(-[a-zA-Z]*f[a-zA-Z]*|--force(-with-lease|-if-includes)?)([[:space:]]|$|=)|reset[[:space:]]+--hard|clean[[:space:]]+-[a-zA-Z]*[fd][a-zA-Z]*|branch[[:space:]]+(-D|--delete[[:space:]]+--force)|rebase[[:space:]]+(-i|--interactive)|commit([[:space:]]+[^[:space:]]+)*[[:space:]]+--no-verify)' \
  && deny "destructive or history-rewriting git is not allowed here"

# Tests must not be silenced (either quote style; --ignore both forms; addopts overrides)
has "pytest([[:space:]]+[^[:space:]]+)*[[:space:]]+(--deselect|--ignore(-glob)?([[:space:]]|=)|-k[[:space:]]*${Q}?[[:space:]]*not[[:space:]]|-o[[:space:]]*addopts|--override-ini)" \
  && deny "do not deselect or ignore tests to get green; report the failing test"
if [ "$ROLE" != test-author ]; then
  has '(^|[[:space:]])(rm|mv|sed[[:space:]]+-i[^|]*|>>?[[:space:]]*)[^|;&]*tests/[^[:space:]]*\.py' \
    && deny "do not delete, move or overwrite tests"
fi

# Implementer may not push or open PRs (verifier does, after review)
if [ "$ROLE" = implementer ]; then
  has '(^|[[:space:]])(git[[:space:]]+push|gh[[:space:]]+pr[[:space:]]+(create|merge))' \
    && deny "implementer does not push or open PRs; the verifier does after review"
fi

# File ownership through the shell (T-0029 criterion 9): the write shapes an agent reaches for
# (redirection, tee, sed -i, cp, mv, rm) are checked against the task's owned files with the same
# parser protect_paths.sh uses for Edit/Write, so the two gates cannot disagree. Only in-repo
# targets count; the scratchpad, /tmp and /dev/null are free. On a normaliser failure there are
# no targets and this rule does not fire: fail-open for ownership, fail-closed for GPU (the
# fallback above keeps the raw command for the GPU patterns) -- ownership is a courtesy between
# agents, GPU spend is money. Residuals (D-a): an interpreter writing a file from a heredoc body;
# `>|` and `>&word` (the segment splitter sees the `|`/`&` first); `truncate`, `install`, `dd of=`;
# a target behind a variable other than $TOP/$PWD (fail-open).
if [ "$ROLE" = implementer ] && [ -n "$targets" ]; then
  owned=""; owned_rc=-1
  while IFS= read -r t; do
    [ -n "$t" ] || continue
    t=${t//\$\{TOP\}/$TOP}; t=${t//\$TOP/$TOP}; t=${t//\$\{PWD\}/$CWD}; t=${t//\$PWD/$CWD}
    # any other unexpanded variable ($OUT/f.txt) is unknowable here: fail-open, as this rule
    # declares for ownership (A9); /dev/* is never a file of ours
    case "$t" in *\$*|/dev/*) continue;; esac
    abs=$(cd "$CWD" 2>/dev/null && realpath -m -- "$t" 2>/dev/null) || continue
    case "$abs" in "$TOP"/*) rel=${abs#"$TOP"/};; *) continue;; esac
    case "$rel" in .claude/agent-memory/"$ROLE"/*) continue;; esac
    [ -n "$TASK" ] || deny "no task branch: run 'git switch -c T-NNNN-<slug>' before writing $rel"
    if [ "$owned_rc" = -1 ]; then
      task_file="$TOP/docs/tasks/$TASK.md"
      if owned=$(owned_globs "$task_file" 2>/dev/null); then owned_rc=0; else owned_rc=$?; fi
      [ "$owned_rc" = 0 ] || deny "cannot read the '## Owned files' section of docs/tasks/$TASK.md ($(owned_globs "$task_file" 2>&1 >/dev/null || true)); fix the task file before writing $rel"
      [ -n "$owned" ] || deny "the '## Owned files' section of docs/tasks/$TASK.md lists no path; ask the planner to fill it before writing $rel"
    fi
    match=0
    while IFS= read -r pat; do case "$rel" in $pat) match=1;; esac; done <<< "$owned"
    [ "$match" = 1 ] || deny "$rel is not in the owned files of task $TASK (a shell write is an edit). Ask the planner to re-scope."
  done <<< "${targets//$'\t'/$'\n'}"
fi

# GPU / artifact-WRITING runs need a launch token (reads of artifacts/ are fine).
# Two trigger classes (T-0016 D2): measurement, which needs a full /launch review, and service
# starts, which need only the cheap invariants. Both are gated on the same single-use token and
# the same clean tree -- what differs is what /launch must do before writing the token, so only
# the message differs here.
gpu=0
# Produces numbers, detaches a long run, or writes into the artifacts directory. `diffusion_gearvis`
# is the same command as `astra` (T-0156 R4), so both spellings are gated identically.
eff '((astra|diffusion_gearvis)[[:space:]]+bench|local_astra\.eval\.run|nohup[[:space:]]|(>>?|tee([[:space:]]+-a)?|--out(put)?(-dir)?[=[:space:]]|-o)[[:space:]]*[^[:space:]]*artifacts/)' && gpu=1
# Everything under scripts/gpu/ or scripts/locked/ spends GPU time by definition, whatever the
# script is called -- but only when it is run, not when it is read, linted, edited or committed
# (the normaliser above decides which, after resolving any cd; T-0016 I1, I2, I9).
[ "$gated" = 1 ] && gpu=1
# An inference request to a running engine is GPU time (T-0016 D3). A status poll is not, so the
# trigger needs an inference endpoint AND a request body: GET /v1/models and /metrics stay free.
if eff '/v1/(chat/)?completions|/v1/(embeddings|responses)' \
  && eff "(-X[[:space:]]*POST|--request[[:space:]]*POST|(^|[[:space:]])(-d|--data(-raw|-binary|-urlencode)?|--json)([[:space:]]|=))"; then gpu=1; fi
# pytest -m gpu. Anchored at both ends: pytest must really be invoked (not quoted inside a
# message), and -m must be a token of its own -- `curl --max-time 5 ... --query-gpu=...` used to
# match this clause through the -m inside --max-time.
if eff '(^|[[:space:]])pytest([[:space:]]|$)' \
  && has "(^|[[:space:]])-m[[:space:]]*${Q}?[^${SQ}\"]*\bgpu\b" \
  && ! has "(^|[[:space:]])-m[[:space:]]*${Q}?[[:space:]]*not[[:space:]]+gpu\b"; then gpu=1; fi
# Engine starts: they take VRAM but produce no numbers, and they still need a token.
svc=0
eff '(vllm[[:space:]]+serve|moshi-server)' && svc=1
if eff '(docker|podman)[[:space:]]+run' && eff '(^|[[:space:]])--gpus([[:space:]]|=)'; then svc=1; fi
# The routine service starts CLAUDE.md lists among its ordinary commands need a clean tree and
# nothing else -- no token, no reviewer pass, no Runs entry (T-0016 I12, the human's D2): the
# hazard is not that a service was started, it is measuring later against a tree that was dirty
# when it started, and a review per start is the friction that gets the hook switched off.
start=0
eff '((astra|diffusion_gearvis)[[:space:]]+up|systemctl([[:space:]]+--user)?[[:space:]]+(start|restart)[[:space:]]+astra)' && start=1
# What is dirty, by name: a refusal that does not say leaves the agent guessing (T-0016 I12).
# Called only on a triggered path, so an ordinary Bash call still runs no VCS command.
dirty_paths() {
  local out
  out=$(git -C "$TOP" status --porcelain -uno 2>/dev/null) \
    || { echo "unknown -- $TOP is not a worktree this hook can read"; return 0; }
  printf '%s' "$out" | sed 's/^...//' | tr '\n' ' '
}

if [ "$gpu" = 1 ] || [ "$svc" = 1 ]; then
  tok="$STATE/launch_approved"
  why="GPU/artifact-writing commands require /launch review first"
  [ "$gpu" = 1 ] || why="starting an engine takes VRAM until something stops it: it needs a /launch review and a clean tree"
  [ -f "$tok" ] || deny "$why"
  approved=$(cat "$tok")
  printf '%s' "$cmd" | grep -Fq -- "$approved" || deny "this command differs from the one the launch-reviewer approved"
  d=$(dirty_paths)
  [ -z "$d" ] || deny "tree is dirty; never measure a tree an agent is editing. Commit or stash: $d"
  rm -f "$tok"   # single use
elif [ "$start" = 1 ]; then
  d=$(dirty_paths)
  [ -z "$d" ] || deny "a service started against a tree an agent is editing makes every number measured afterwards unattributable. Commit or stash: $d"
fi
exit 0
