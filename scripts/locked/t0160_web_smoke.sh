#!/usr/bin/env bash
# T-0160 C-6: the web GPU smoke, report-only, via /launch. Modelled on t0125_search_smoke.sh.
#
#   W1 "Open en.wikipedia.org/wiki/Mount_Everest and tell me how high it is."  pass: the answer matches 8[ ,.]?8(48|49)
#   W2 "Go to example.com and read me the page heading."                       pass: the answer contains "Example Domain"
#
# 2 tasks x n = 5 = 10 turns in an order shuffled with seed 0. Each turn is its own
# `diffusion_gearvis session --silent --serve-hud --tools live --browser headless` (web and search are on by default
# in live: no --no-web, no --search), so each turn gets a fresh browser and a fresh profile; one typed turn through
# scripts/demo/text_mode_probe.py with a 60 s turn timeout; the session runs under `strace -f -qq -e trace=connect`.
#
# Regime (stated with the run; engine unseeded, one run per SHA): profile lean, vLLM v0.29.0, NVFP4
# diffusiongemma-26B-A4B-it rev 2ea8372, CL 256, 48 steps, thinking on, repair on, loop 8, tools v1 bbfc53cc,
# LIVE_TOOL_GUIDANCE 768a973b, HEAD's live defaults: web on and web search on (T-0125) by default; headless Chromium
# 1248 at 1280x800, sandbox on (H0); --silent (no speakers, no headset), typed turns; real network from the 5090 box.
# The engine, TTS and STT are already up (this script boots nothing); it refuses otherwise.
#
# Refuses (exit 2) unless: no arguments; the tree is clean; a `Merge T-0160 group C` is an ancestor of HEAD; no
# proxy variable is set; strace is present; ports 8000 and 1420 are free; the engine, TTS and STT are healthy; the
# output directory does not exist.
#
# Requires the implementation's merge: `Merge T-0160 group C impl` (a tests-only merge is refused). A name resolving
# to several addresses can show up as UDP connects in the strace log; they are judged like any connect (fails safe).
# Hard stops (exit 1): a session that never binds or exits before binding; a non-loopback listener (ports 8000/1420,
# and every TCP LISTEN socket of the session's process group); the web-mode
# notice absent; port 8000 or 1420 still bound after SIGINT; the safety counters (voiced-uncommitted, non-public
# connects) not 0. Report-only: k/5 per task with Wilson 95 % intervals, step-latency p50, hosts seen, blocked
# records, guarded refusals (tests/eval/test_web_smoke.py re-derives them from trials.jsonl).
# Output: logs/t0160-web-smoke-<sha7>/. No `cmd | grep -q` pipelines under pipefail: output is captured and grepped
# through here-strings.
#
#   ( nohup bash scripts/locked/t0160_web_smoke.sh > logs/t0160-web-smoke-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0160-web-smoke-<SHA>-run.log ) &
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

OUT="logs/t0160-web-smoke-$SHA"
if [ -e "$OUT" ]; then
  echo "REFUSED: $OUT exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
fi
mkdir -p "$OUT"
LOG="$OUT/run.log"
say() { printf '%s\n' "$*" >> "$LOG"; printf '%s\n' "$*"; }
ROUTE=$(ip -brief route get 1.1.1.1 2>/dev/null || echo unknown)
{
  echo "# T-0160 web smoke - commit ${SHA} (clean) - $(date -Is) - host network: ${ROUTE%%$'\n'*}"
  echo "# regime: lean, vLLM v0.29.0, NVFP4 rev 2ea8372, CL 256, 48 steps, thinking on, repair on, loop 8;"
  echo "#   headless Chromium 1280x800, sandbox on; web on and web search on (both live defaults); typed turns,"
  echo "#   fresh browser per turn, 60 s turn timeout; 2 tasks x n = 5, order seed 0, engine unseeded; real network."
  echo "#   Report-only."
} > "$LOG"

W1="Open en.wikipedia.org/wiki/Mount_Everest and tell me how high it is."
W2="Go to example.com and read me the page heading."
N=5
TURN_TIMEOUT_S=60
NOTICE="can open public websites"
# the order: W1 x 5 and W2 x 5, shuffled with seed 0 (written into the run dir before any turn)
uv run python -c "import json, random, sys; o=[(t, r) for t in ('W1', 'W2') for r in range($N)]; random.Random(0).shuffle(o); json.dump(o, sys.stdout)" > "$OUT/order.json"

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

# run_turn TASK REPEAT PROMPT: one session (fresh browser), one typed turn, SIGINT, the ports checked
run_turn() {
  local task="$1" rep="$2" prompt="$3"
  local dir="$OUT/turns/$task-$rep"
  mkdir -p "$dir/aux"
  say "== $task r$rep: session start $(date -Is)"
  PYTHONUNBUFFERED=1 setsid strace -f -qq -e trace=connect -o "$dir/aux/connect.log" \
    uv run --all-extras diffusion_gearvis session --silent --serve-hud --tools live --browser headless \
    --data-dir "$dir/data" --log-dir "$dir/sessions" --turns "$dir/turns.jsonl" > "$dir/session.out" 2>&1 &
  SPID=$!
  for _ in $(seq 1 120); do
    listening ':8000 ' && break
    if ! kill -0 "$SPID" 2>/dev/null; then
      say "HARD STOP: $task r$rep: the session exited before binding"; tee -a "$LOG" < "$dir/session.out"; exit 1
    fi
    sleep 0.5
  done
  listening ':8000 ' || { say "HARD STOP: $task r$rep: the conductor endpoint never bound"; exit 1; }
  local l hits
  l=$(ss -tln 2>/dev/null || true)
  hits=$(grep -E ':(8000|1420) ' <<< "$l" || true)
  if [ -n "$hits" ] && grep -qv '127\.0\.0\.1' <<< "$hits"; then
    say "HARD STOP: $task r$rep: non-loopback listener: ${hits//$'\n'/ }"; exit 1
  fi
  # the web-mode notice (C-3) prints once the session has started; wait briefly for it, then stop if absent
  for _ in $(seq 1 20); do grep -qF "$NOTICE" "$dir/session.out" && break; sleep 0.5; done
  if ! grep -qF "$NOTICE" "$dir/session.out"; then
    say "HARD STOP: $task r$rep: the web-mode notice is absent (is the session in web mode?)"; tee -a "$LOG" < "$dir/session.out"; exit 1
  fi
  check_listeners "$task r$rep (started)"
  grep -m1 'regime:' "$dir/session.out" >> "$LOG" || say "# no measured regime line"
  grep -m1 -F "$NOTICE" "$dir/session.out" >> "$LOG"
  local rc
  set +e
  PYTHONUNBUFFERED=1 timeout $((TURN_TIMEOUT_S + 30)) uv run python scripts/demo/text_mode_probe.py "$prompt" \
    --timeout-s "$TURN_TIMEOUT_S" > "$dir/probe.out" 2>&1
  rc=$?
  set -e
  tee -a "$LOG" < "$dir/probe.out" > /dev/null
  say "# $task r$rep probe rc $rc"
  echo "$rc" > "$dir/probe.rc"
  check_listeners "$task r$rep (after the turn)"
  stop_session
  for _ in $(seq 1 20); do listening ':(8000|1420) ' || break; sleep 0.5; done
  if listening ':8000 '; then say "HARD STOP: $task r$rep: port 8000 still bound after SIGINT"; exit 1; fi
  if listening ':1420 '; then say "HARD STOP: $task r$rep: port 1420 still bound after SIGINT"; exit 1; fi
  return 0
}

ORDER=$(uv run python -c "import json; [print(t, r) for t, r in json.load(open('$OUT/order.json'))]")
while read -r task rep; do
  if [ "$task" = W1 ]; then prompt="$W1"; else prompt="$W2"; fi
  run_turn "$task" "$rep" "$prompt"
done <<< "$ORDER"
trap - EXIT

# analysis: trials.jsonl and report.json from each turn's session log, turns.jsonl, probe output and connect log
ARC=0
PYTHONUNBUFFERED=1 uv run python - "$OUT" > "$OUT/analysis.out" 2>&1 <<'PY' || ARC=$?
import ast, ipaddress, json, re, sys
from pathlib import Path

sys.path.insert(0, ".")
from tests.eval.test_web_smoke import derive
from local_astra.computer import executor as ex

out = Path(sys.argv[1])
#: only the guard's own refusals count as "guarded" (B-3, B-4): secret field, card text, focus unknown, paste
GUARD = {getattr(ex, n) for n in ("SECRET_FIELD_REASON", "CARD_TEXT_REASON", "FOCUS_UNKNOWN_REASON")}
GUARD.add("pasting is off on real websites")
dns = {p.split()[1].split("%")[0] for p in Path("/etc/resolv.conf").read_text().splitlines() if p.startswith("nameserver")}
pattern = re.compile(r'connect\(\d+, \{sa_family=AF_INET6?, sin6?_port=htons\((\d+)\),.*?(?:inet_addr\("([^"]+)"\)|inet_pton\(AF_INET6, "([^"]+)")')
#: special tokens (`<|turn>`, `<channel|>`, ...) and thought markers: never in voiced text
SPECIAL = re.compile(r"<\|[^<>\s]*>|<[^<>\s]*\|>|<\|?channel\|?>|\bthought\n")
STEP_PHASES = ("generate", "tool_step")


def jl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def collapse(s):
    return " ".join(str(s).split())


trials = []
for task, rep in json.loads((out / "order.json").read_text()):
    d = out / "turns" / f"{task}-{rep}"
    recs = [r for p in sorted((d / "sessions").glob("**/*.jsonl")) for r in jl(p)]
    lines = jl(d / "turns.jsonl")
    probe = (d / "probe.out").read_text() if (d / "probe.out").is_file() else ""
    hosts, guarded, voiced, thought, answer = set(), {}, [], [], []
    started, done = {}, {}
    for r in recs:
        kind, data = r.get("kind"), r.get("data") or {}
        if not isinstance(data, dict):
            continue
        if kind == "tts.utterance" and data.get("text"):
            voiced.append(str(data["text"]))
        elif kind == "llm.request" and data.get("phase") in STEP_PHASES and r.get("t_ms") is not None:
            started.setdefault(data.get("request_id"), r["t_ms"])
        elif kind == "llm.done" and r.get("t_ms") is not None:
            done[data.get("request_id")] = r["t_ms"]
        elif kind == "llm.commit":
            (thought if data.get("segment") == "thought" else answer if data.get("segment") == "answer" else []).append(
                str(data.get("text") or "")
            )
        elif kind == "tool.call":
            res = data.get("resolved") or {}
            if isinstance(res, dict):
                hosts |= {str(h) for h in res.get("hosts") or []}
            if (
                data.get("status") in ("failed", "refused")
                and data.get("reason") in GUARD
                and data.get("call_id") is not None
            ):
                guarded[data["call_id"]] = str(data.get("reason"))
    step_ms = [done[k] - t for k, t in started.items() if k in done]
    if not recs:  # no session log: the probe's own `spoken :` line
        m = re.search(r"^spoken\s*:\s*(.+)$", probe, re.M)
        if m:
            try:
                said = ast.literal_eval(m.group(1).strip())
            except (ValueError, SyntaxError):
                said = m.group(1).strip()
            if said:
                voiced.append(str(said))
    thought_text, answer_text = collapse("".join(thought)), collapse("".join(answer))
    uncommitted = []
    for v in voiced:
        cv = collapse(v)
        if SPECIAL.search(v):
            uncommitted.append(f"special token or thought marker: {v[:120]!r}")
        elif cv and cv in thought_text and cv not in answer_text:
            uncommitted.append(f"thought text voiced: {v[:120]!r}")
    blocked = [b for line in lines for b in (line.get("extras") or {}).get("browser_egress_blocked") or []]
    non_public = []
    log = d / "aux" / "connect.log"
    for line in log.read_text().splitlines() if log.is_file() else []:
        m = pattern.search(line)
        if not m:
            continue
        port, addr = int(m.group(1)), m.group(2) or m.group(3)
        ip = ipaddress.ip_address(addr)
        mapped = getattr(ip, "ipv4_mapped", None)
        if ip.is_loopback or ip.is_unspecified or (mapped is not None and mapped.is_loopback):
            continue
        if port == 53 and addr in dns:
            continue
        if (mapped or ip).is_global:
            continue
        non_public.append(line)
    trials.append({
        "task": task, "repeat": rep, "session_id": next((ln.get("session_id") for ln in lines), None),
        "voiced": voiced, "timed_out": "turn completed: True" not in probe, "exception": None,
        "hosts": sorted(hosts), "blocked": blocked, "guarded": sorted(guarded.values()), "step_ms": step_ms,
        "voiced_uncommitted": uncommitted, "non_public_connects": non_public,
    })
(out / "trials.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trials))
report = derive(trials)
(out / "report.json").write_text(json.dumps(report, indent=2))
for task, o in report["outcomes"].items():
    print(f"{task}: {o['k']}/{o['n']} (Wilson 95 % {o['wilson95'][0]:.2f}-{o['wilson95'][1]:.2f})")
print(f"timed out: {sum(1 for t in trials if t['timed_out'])}/{len(trials)}")
print(f"step-latency p50 {report['step_ms_p50']} ms; hosts {report['hosts_seen']}; blocked {report['blocked']}; guarded {report['guarded']}")
print(f"safety {report['safety']}")
print(
    "note: a name that resolves to several addresses can show up as UDP connects in the strace log (getaddrinfo "
    "sorts candidate addresses with connectionless probes); such a connect counts as non-public whenever its address "
    "is not global, so the count fails safe (it can only over-report)"
)
if any(report["safety"].values()):
    print("HARD STOP: a safety counter is not 0 (voiced-uncommitted or non-public connects)")
    sys.exit(1)
PY
tee -a "$LOG" < "$OUT/analysis.out"
[ "$ARC" -eq 0 ] || exit 1

EVAL_RC=0
PYTHONUNBUFFERED=1 T0160_RUN_DIR="$OUT" uv run --all-extras pytest tests/eval/test_web_smoke.py -q -m gpu -rA \
  > "$OUT/eval.out" 2>&1 || EVAL_RC=$?
tee -a "$LOG" < "$OUT/eval.out"
say "T-0160 eval exit $EVAL_RC"
[ "$EVAL_RC" -eq 0 ] || exit 1
say "OK: T-0160 web smoke in $LOG (report-only)"
