#!/usr/bin/env bash
# T-0191 group B (B1-B4): the GPU navigation smoke of live guidance v3 (how to go to a different website), via /launch.
# A copy of scripts/gpu/t0174_web_ab.sh (its refusals, hard stops, strace connect audit, listener checks, per-session
# fresh data dir and startup void check) with one arm, `--live-guidance v3`, in web mode (every other flag is the
# T-0174 smoke's). Each trial is one session with two typed turns (setup, then turn 2):
#
#   N1  "Open amazon.com."                            then "Now go to photopea.com."
#   N2  "Open duckduckgo.com/html/?q=photo+editor."   then "Open Wikipedia."
#   N3  "Open amazon.com."                            then "Search Amazon for bath towels."   (report-only)
#
# 3 scenarios x n = 5 = 15 sessions x 2 turns, shuffled with order seed 0 (order.json holds [scenario, rep] pairs,
# written before any turn). Each session is
# `diffusion_gearvis session --silent --serve-hud --tools live --browser headless --live-guidance v3` (web and
# search are on by default in live), so each trial gets a fresh browser, profile and data dir; two typed turns through
# scripts/demo/text_mode_probe.py with a 60 s turn timeout each; the session runs under
# `strace -f -qq -e trace=connect`. Each trial lives in turns/<scenario>-<rep>/.
#
# Regime (stated with the run; engine unseeded, one run per SHA): profile lean, the pinned versions of
# services/versions.lock at HEAD, NVFP4 revision from the lock, CL 256, 48 steps, thinking on; --live-guidance v3;
# HEAD's other live defaults; web on and web search on by default; headless Chromium 1280x800, sandbox on; --silent
# (no speakers, no headset), typed turns; real websites from the 5090 box, date and time in run.log.
#
# Refuses (exit 2) unless: no arguments; the tree is clean; a `Merge T-0160 group C impl` is an ancestor of HEAD; a
# `Merge T-0174-a:` and a `Merge T-0191-a:` are ancestors of HEAD (the bare names, tests-only merges and look-alikes
# are refused); no proxy variable is set; strace is present; ports 8000 and 1420 are free; the engine, TTS and STT
# are healthy; the output directory does not exist.
#
# Hard stops (exit 1), as T-0160/T-0174: a session that never binds or exits before binding; a non-loopback listener
# (ports 8000/1420, and every TCP LISTEN socket of the session's process group); the web-mode notice absent; port
# 8000 or 1420 still bound after SIGINT; the safety counters (voiced-uncommitted, non-public connects) not 0.
# Void (a report finding, exit 0 on a completed run): a session whose startup output lacks `live guidance: v3 (flag)`.
# Budget (B4): no trial starts once the run has used 2700 s; a budget-stopped run is incomplete (INCONCLUSIVE in
# report.json), and the analysis still runs.
# Decision (B3, pre-registered; scripts/gpu/t0191_nav_smoke_analysis.py writes report.json and trials.jsonl).
# Output: logs/t0191-nav-smoke-<sha7>/. No `cmd | grep -q` pipelines under pipefail: output is captured and grepped
# through here-strings.
#
#   ( nohup bash scripts/gpu/t0191_nav_smoke.sh > logs/t0191-nav-smoke-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0191-nav-smoke-<SHA>-run.log ) &
set -euo pipefail
trap 'rc=$?; echo "FAIL: exit $rc at line $LINENO: $BASH_COMMAND" >&2' ERR
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

if [ "$#" -ne 0 ]; then echo "REFUSED: no arguments expected (got $*)" >&2; exit 2; fi

# listening REGEX: true when a listening socket line matches (ss captured, then a here-string: no pipe)
listening() {
  local l
  l=$(ss -tln 2>/dev/null || true)
  grep -Eq "$1" <<< "$l"
}

SHA=$(git rev-parse --short=7 HEAD)
DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
  echo "REFUSED: dirty tree: ${DIRTY//$'\n'/ }" >&2; exit 2
fi
MERGES=$(git log --merges --format='%H %s' HEAD || true)
# the implementation's merge, not a tests-only merge ("Merge T-0160 group C tests ...")
if ! grep -Eq '^[0-9a-f]+ Merge T-0160 group C impl\b' <<< "$MERGES"; then
  echo "REFUSED: 'Merge T-0160 group C impl' is not an ancestor of HEAD (a tests-only merge does not count); the smoke runs at (or after) the implementation's merge" >&2; exit 2
fi
# T-0174 group A's merge (the colon is required: not the bare name, "Merge T-0174-a tests: ..." or "Merge T-0174-ab")
if ! grep -Eq '^[0-9a-f]+ Merge T-0174-a:' <<< "$MERGES"; then
  echo "REFUSED: 'Merge T-0174-a:' is not an ancestor of HEAD (--live-guidance needs group A's merge; the bare name, a tests-only merge and look-alikes do not count)" >&2; exit 2
fi
# T-0191 group A's merge (v3 exists; the colon is required: not the bare name, "Merge T-0191-a tests: ..." or "Merge T-0191-ab")
if ! grep -Eq '^[0-9a-f]+ Merge T-0191-a:' <<< "$MERGES"; then
  echo "REFUSED: 'Merge T-0191-a:' is not an ancestor of HEAD (--live-guidance v3 needs group A's merge; the bare name, a tests-only merge and look-alikes do not count)" >&2; exit 2
fi
for v in HTTPS_PROXY https_proxy HTTP_PROXY http_proxy ALL_PROXY all_proxy NO_PROXY no_proxy DDGS_PROXY PRIMP_PROXY; do
  if [ -n "${!v:-}" ]; then
    echo "REFUSED: proxy variable $v is set; this smoke runs without one (the browser's only proxy is the session's egress proxy)" >&2; exit 2
  fi
done
if ! command -v strace >/dev/null 2>&1; then
  echo "REFUSED: strace is absent; the connect audit needs it" >&2; exit 2
fi
for P in 8000 1420; do
  if listening ":$P "; then
    echo "REFUSED: port $P already bound" >&2; exit 2
  fi
done
H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health || true)
[ "$H" = 200 ] || { echo "REFUSED: engine (vllm /health) = $H, not healthy" >&2; exit 2; }
T=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8081/health || true)
[ "$T" = 200 ] || { echo "REFUSED: tts /health = $T, not healthy" >&2; exit 2; }
S=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8080/api/asr-streaming || true)
[ "$S" = 400 ] || { echo "REFUSED: stt route = $S, not healthy" >&2; exit 2; }

OUT="logs/t0191-nav-smoke-$SHA"
if [ -e "$OUT" ]; then
  echo "REFUSED: $OUT exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
fi
mkdir -p "$OUT"
LOG="$OUT/run.log"
say() { printf '%s\n' "$*" >> "$LOG"; printf '%s\n' "$*"; }
ROUTE=$(ip -brief route get 1.1.1.1 2>/dev/null || echo unknown)
{
  echo "# T-0191 B navigation smoke (live guidance v3) - commit ${SHA} (clean) - $(date -Is) - host network: ${ROUTE%%$'\n'*}"
  echo "# regime: lean, pinned versions of services/versions.lock at ${SHA}, CL 256, 48 steps, thinking on;"
  echo "#   one arm --live-guidance v3, every other flag the T-0174 smoke's; web mode through the egress proxy;"
  echo "#   headless Chromium 1280x800, sandbox on; web search on (live default); typed turns, fresh browser and"
  echo "#   data dir per session, 60 s per turn; N1/N2/N3 x n = 5 sessions x 2 turns, shuffled with order seed 0;"
  echo "#   engine unseeded; real websites. Budget 2700 s (B4)."
} > "$LOG"

N1_SETUP="Open amazon.com."
N1_TURN2="Now go to photopea.com."
N2_SETUP="Open duckduckgo.com/html/?q=photo+editor."
N2_TURN2="Open Wikipedia."
N3_SETUP="Open amazon.com."
N3_TURN2="Search Amazon for bath towels."
GUIDANCE=v3
N=5
TURN_TIMEOUT_S=60
BUDGET_S=2700
NOTICE="can open public websites"
# the order: (scenario, rep) for N1/N2/N3, rep 0..N-1, shuffled with seed 0 (written before any turn)
uv run python -c "import json, random, sys; o=[(s, r) for s in ('N1', 'N2', 'N3') for r in range($N)]; random.Random(0).shuffle(o); json.dump(o, sys.stdout)" > "$OUT/order.json"
RUN_START=$SECONDS

SPID=""
stop_session() {
  if [ -n "$SPID" ]; then
    kill -INT -"$SPID" 2>/dev/null || kill -INT "$SPID" 2>/dev/null || true
    wait "$SPID" 2>/dev/null || true
    SPID=""
  fi
}
trap stop_session EXIT

# check_listeners WHERE: every TCP LISTEN socket of the session's process group (setsid: the group id is $SPID) must
# be bound to loopback; any other bind is a hard stop
check_listeners() {
  local where="$1" pids l bad="" pid line addr
  pids=$(pgrep -g "$SPID" || true)
  l=$(ss -tlnpH 2>/dev/null || true)
  while read -r pid; do
    [ -n "$pid" ] || continue
    while read -r line; do
      [ -n "$line" ] || continue
      addr=$(awk '{print $4}' <<< "$line")
      case "$addr" in
        127.*|"[::1]":*|"[::ffff:127."*) ;;
        *) bad+="$line; " ;;
      esac
    done < <(grep -F "pid=$pid," <<< "$l" || true)
  done <<< "$pids"
  if [ -n "$bad" ]; then
    say "HARD STOP: $where: a non-loopback listener in the session's process group: $bad"; exit 1
  fi
}

# probe NAME K PROMPT: one typed turn into the running session, its output in probe-K.out
probe() {
  local name="$1" k="$2" prompt="$3" dir="$OUT/turns/$1" rc
  set +e
  PYTHONUNBUFFERED=1 timeout $((TURN_TIMEOUT_S + 30)) uv run python scripts/demo/text_mode_probe.py "$prompt" \
    --timeout-s "$TURN_TIMEOUT_S" < /dev/null > "$dir/probe-$k.out" 2>&1
  rc=$?
  set -e
  tee -a "$LOG" < "$dir/probe-$k.out" > /dev/null
  say "# $name turn $k probe rc $rc"
  echo "$rc" > "$dir/probe-$k.rc"
}

# run_trial SCENARIO REPEAT SETUP TURN2: one session (fresh browser and data dir), two typed turns, SIGINT, the ports
# checked
run_trial() {
  local scen="$1" rep="$2" setup="$3" turn2="$4"
  local name="$scen-$rep"
  local dir="$OUT/turns/$name"
  mkdir -p "$dir/aux"
  say "== $name (--live-guidance $GUIDANCE): session start $(date -Is)"
  PYTHONUNBUFFERED=1 setsid strace -f -qq -e trace=connect -o "$dir/aux/connect.log" \
    uv run --all-extras diffusion_gearvis session --silent --serve-hud --tools live --browser headless \
    --live-guidance "$GUIDANCE" \
    --data-dir "$dir/data" --log-dir "$dir/sessions" --turns "$dir/turns.jsonl" > "$dir/session.out" 2>&1 &
  SPID=$!
  for _ in $(seq 1 120); do
    listening ':8000 ' && break
    if ! kill -0 "$SPID" 2>/dev/null; then
      say "HARD STOP: $name: the session exited before binding"; tee -a "$LOG" < "$dir/session.out"; exit 1
    fi
    sleep 0.5
  done
  listening ':8000 ' || { say "HARD STOP: $name: the conductor endpoint never bound"; exit 1; }
  local l hits
  l=$(ss -tln 2>/dev/null || true)
  hits=$(grep -E ':(8000|1420) ' <<< "$l" || true)
  if [ -n "$hits" ] && grep -qv '127\.0\.0\.1' <<< "$hits"; then
    say "HARD STOP: $name: non-loopback listener: ${hits//$'\n'/ }"; exit 1
  fi
  # the web-mode notice (C-3) prints once the session has started; wait briefly for it, then stop if absent
  for _ in $(seq 1 20); do grep -qF "$NOTICE" "$dir/session.out" && break; sleep 0.5; done
  if ! grep -qF "$NOTICE" "$dir/session.out"; then
    say "HARD STOP: $name: the web-mode notice is absent (is the session in web mode?)"; tee -a "$LOG" < "$dir/session.out"; exit 1
  fi
  check_listeners "$name (started)"
  grep -m1 'regime:' "$dir/session.out" >> "$LOG" || say "# no measured regime line"
  grep -m1 -F "$NOTICE" "$dir/session.out" >> "$LOG"
  # B1 void check (judged by the analysis from session.out): v3's guidance line, from the flag
  if grep -qF "live guidance: $GUIDANCE (flag)" "$dir/session.out"; then
    say "# $name: live guidance: $GUIDANCE (flag)"
  else
    say "# $name: VOID: the startup output lacks 'live guidance: $GUIDANCE (flag)'"
  fi
  probe "$name" 1 "$setup"
  check_listeners "$name (after turn 1)"
  probe "$name" 2 "$turn2"
  check_listeners "$name (after turn 2)"
  stop_session
  for _ in $(seq 1 20); do listening ':(8000|1420) ' || break; sleep 0.5; done
  if listening ':8000 '; then say "HARD STOP: $name: port 8000 still bound after SIGINT"; exit 1; fi
  if listening ':1420 '; then say "HARD STOP: $name: port 1420 still bound after SIGINT"; exit 1; fi
  return 0
}

INCOMPLETE=""
ORDER=$(uv run python -c "import json; [print(s, r) for s, r in json.load(open('$OUT/order.json'))]")
while read -r scen rep; do
  # B4: no trial starts once the budget is spent (checked between trials, so no trial is cut in half)
  if [ $((SECONDS - RUN_START)) -ge "$BUDGET_S" ]; then
    INCOMPLETE="$scen-$rep"
    say "# BUDGET: ${BUDGET_S} s spent before $scen-$rep; the run stops here and is incomplete (B4)"
    break
  fi
  case "$scen" in
    N1) setup="$N1_SETUP"; turn2="$N1_TURN2" ;;
    N2) setup="$N2_SETUP"; turn2="$N2_TURN2" ;;
    N3) setup="$N3_SETUP"; turn2="$N3_TURN2" ;;
    *) say "HARD STOP: unknown scenario $scen in order.json"; exit 1 ;;
  esac
  run_trial "$scen" "$rep" "$setup" "$turn2"
done <<< "$ORDER"
trap - EXIT

# analysis: report.json (analyse()'s return value) and trials.jsonl (its "trials") from each trial's session log,
# turns.jsonl, probe outputs, startup output and connect log; exit 1 when a safety counter is not 0
ARC=0
PYTHONUNBUFFERED=1 uv run python scripts/gpu/t0191_nav_smoke_analysis.py "$OUT" > "$OUT/analysis.out" 2>&1 || ARC=$?
tee -a "$LOG" < "$OUT/analysis.out"
[ "$ARC" -eq 0 ] || exit 1
if [ -n "$INCOMPLETE" ]; then
  say "OK (incomplete: budget stop before $INCOMPLETE): T-0191 B navigation smoke in $LOG; the decision is in report.json"
else
  say "OK: T-0191 B navigation smoke in $LOG (the decision and any void finding are in report.json)"
fi
