#!/usr/bin/env bash
# T-0174 group C-web (CW1-CW3): the quick real-web A/B of live guidance v1 and v2, report-only, via /launch.
# A copy of scripts/locked/t0160_web_smoke.sh (its refusals, hard stops and strace connect audit) with two arms:
#
#   arm A: --live-guidance v1      arm B: --live-guidance v2      (every other flag is the T-0160 smoke's)
#
#   W1 "Open en.wikipedia.org/wiki/Mount_Everest and tell me how high it is."  pass: the answer matches 8[ ,.]?8(48|49)
#   W2 "Go to example.com and read me the page heading."                       pass: the answer contains "Example Domain"
#
# 2 arms x 2 tasks x n = 5 = 20 turns, jointly shuffled with order seed 0 (order.json holds [arm, task, rep] triples,
# written before any turn). Each turn is its own
# `diffusion_gearvis session --silent --serve-hud --tools live --browser headless --live-guidance <v>` (web and
# search are on by default in live: no --no-web, no --search), so each turn gets a fresh browser and a fresh
# profile; one typed turn through scripts/demo/text_mode_probe.py with a 60 s turn timeout; the session runs under
# `strace -f -qq -e trace=connect`. Each trial lives in turns/<arm>-<task>-<rep>/.
#
# Regime (stated with the run; engine unseeded, one run per SHA): profile lean, vLLM v0.29.0, NVFP4
# diffusiongemma-26B-A4B-it rev 2ea8372, CL 256, 48 steps, thinking on, repair on, loop 8, tools v1 bbfc53cc,
# arm A LIVE_TOOL_GUIDANCE 768a973b (v1), arm B LIVE_TOOL_GUIDANCE_V2 (v2); HEAD's other live defaults (T-0172
# reask on); web on and web search on by default; headless Chromium 1280x800, sandbox on; --silent (no speakers,
# no headset), typed turns; real network from the 5090 box. The engine, TTS and STT are already up (this script
# boots nothing); it refuses otherwise.
#
# Refuses (exit 2) unless: no arguments; the tree is clean; a `Merge T-0160 group C impl` is an ancestor of HEAD; a
# `Merge T-0174-a:` is an ancestor of HEAD (so --live-guidance exists; the bare name, a tests-only merge and
# look-alikes are refused); no proxy variable is set; strace is present; ports 8000 and 1420 are free; the engine,
# TTS and STT are healthy; the output directory does not exist.
#
# Hard stops (exit 1), as T-0160: a session that never binds or exits before binding; a non-loopback listener
# (ports 8000/1420, and every TCP LISTEN socket of the session's process group); the web-mode notice absent; port
# 8000 or 1420 still bound after SIGINT; the safety counters (voiced-uncommitted, non-public connects) not 0.
# Void (a report finding, exit 0 on a completed run): an arm-A session whose startup output lacks
# `live guidance: v1 (flag)`, or an arm-B one lacking `live guidance: v2 (flag)`.
# Report-only, per arm (scripts/gpu/t0174_web_ab_analysis.py writes report.json and trials.jsonl): W1/W2 k/5 with
# Wilson 95 %, M1 (C2's DOUBT_RE on step >= 2 first-attempt thoughts), degenerate aborts, re-asks and outcomes, tool
# timeouts, timeouts, hosts, step-latency p50, safety; whether M1_B <= M1_A / 2 and B's successes >= A's. No eval
# module is called (CW3); the analysis is unit-tested instead.
# Output: logs/t0174-web-ab-<sha7>/. No `cmd | grep -q` pipelines under pipefail: output is captured and grepped
# through here-strings.
#
#   ( nohup bash scripts/gpu/t0174_web_ab.sh > logs/t0174-web-ab-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0174-web-ab-<SHA>-run.log ) &
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

OUT="logs/t0174-web-ab-$SHA"
if [ -e "$OUT" ]; then
  echo "REFUSED: $OUT exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
fi
mkdir -p "$OUT"
LOG="$OUT/run.log"
say() { printf '%s\n' "$*" >> "$LOG"; printf '%s\n' "$*"; }
ROUTE=$(ip -brief route get 1.1.1.1 2>/dev/null || echo unknown)
{
  echo "# T-0174 C-web A/B (live guidance v1 vs v2) - commit ${SHA} (clean) - $(date -Is) - host network: ${ROUTE%%$'\n'*}"
  echo "# regime: lean, vLLM v0.29.0, NVFP4 rev 2ea8372, CL 256, 48 steps, thinking on, repair on, loop 8;"
  echo "#   arm A --live-guidance v1, arm B --live-guidance v2, every other flag the T-0160 smoke's;"
  echo "#   headless Chromium 1280x800, sandbox on; web on and web search on (both live defaults); typed turns,"
  echo "#   fresh browser per turn, 60 s turn timeout; 2 arms x 2 tasks x n = 5, jointly shuffled with order seed 0,"
  echo "#   engine unseeded; real network. Report-only."
} > "$LOG"

W1="Open en.wikipedia.org/wiki/Mount_Everest and tell me how high it is."
W2="Go to example.com and read me the page heading."
N=5
TURN_TIMEOUT_S=60
NOTICE="can open public websites"
# the order: (arm, task, rep) for arm A/B, task W1/W2, rep 0..N-1, jointly shuffled with seed 0 (written before any turn)
uv run python -c "import json, random, sys; o=[(a, t, r) for a in ('A', 'B') for t in ('W1', 'W2') for r in range($N)]; random.Random(0).shuffle(o); json.dump(o, sys.stdout)" > "$OUT/order.json"

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

# run_turn ARM GUIDANCE TASK REPEAT PROMPT: one session (fresh browser), one typed turn, SIGINT, the ports checked
run_turn() {
  local arm="$1" guidance="$2" task="$3" rep="$4" prompt="$5"
  local name="$arm-$task-$rep"
  local dir="$OUT/turns/$name"
  mkdir -p "$dir/aux"
  say "== $name (--live-guidance $guidance): session start $(date -Is)"
  PYTHONUNBUFFERED=1 setsid strace -f -qq -e trace=connect -o "$dir/aux/connect.log" \
    uv run --all-extras diffusion_gearvis session --silent --serve-hud --tools live --browser headless \
    --live-guidance "$guidance" \
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
  # CW1 void check (judged by the analysis from session.out): the arm's own guidance line, from the flag
  if grep -qF "live guidance: $guidance (flag)" "$dir/session.out"; then
    say "# $name: live guidance: $guidance (flag)"
  else
    say "# $name: VOID: the startup output lacks 'live guidance: $guidance (flag)'"
  fi
  local rc
  set +e
  PYTHONUNBUFFERED=1 timeout $((TURN_TIMEOUT_S + 30)) uv run python scripts/demo/text_mode_probe.py "$prompt" \
    --timeout-s "$TURN_TIMEOUT_S" > "$dir/probe.out" 2>&1
  rc=$?
  set -e
  tee -a "$LOG" < "$dir/probe.out" > /dev/null
  say "# $name probe rc $rc"
  echo "$rc" > "$dir/probe.rc"
  check_listeners "$name (after the turn)"
  stop_session
  for _ in $(seq 1 20); do listening ':(8000|1420) ' || break; sleep 0.5; done
  if listening ':8000 '; then say "HARD STOP: $name: port 8000 still bound after SIGINT"; exit 1; fi
  if listening ':1420 '; then say "HARD STOP: $name: port 1420 still bound after SIGINT"; exit 1; fi
  return 0
}

ORDER=$(uv run python -c "import json; [print(a, t, r) for a, t, r in json.load(open('$OUT/order.json'))]")
while read -r arm task rep; do
  case "$arm" in
    A) guidance=v1 ;;
    B) guidance=v2 ;;
    *) say "HARD STOP: unknown arm $arm in order.json"; exit 1 ;;
  esac
  if [ "$task" = W1 ]; then prompt="$W1"; else prompt="$W2"; fi
  run_turn "$arm" "$guidance" "$task" "$rep" "$prompt"
done <<< "$ORDER"
trap - EXIT

# analysis: report.json (analyse()'s return value) and trials.jsonl (its "trials") from each turn's session log,
# turns.jsonl, probe output, startup output and connect log; exit 1 when a safety counter is not 0
ARC=0
PYTHONUNBUFFERED=1 uv run python scripts/gpu/t0174_web_ab_analysis.py "$OUT" > "$OUT/analysis.out" 2>&1 || ARC=$?
tee -a "$LOG" < "$OUT/analysis.out"
[ "$ARC" -eq 0 ] || exit 1
say "OK: T-0174 C-web A/B in $LOG (report-only; a void finding is in report.json)"
