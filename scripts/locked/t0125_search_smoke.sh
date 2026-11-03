#!/usr/bin/env bash
# T-0125 criterion 13: the web search GPU smoke, report-only, via /launch. Modelled on t0039_text_mode_live.sh.
#
#   phase 1: `diffusion_gearvis session --silent --serve-hud --tools dry-run --search ddgs` under
#            `strace -f -qq -e trace=connect`, three typed turns through scripts/demo/text_mode_probe.py;
#   phase 2: `--tools live --browser headless --no-web` with NO --search flag (search is on by default in live), turn 1
#            only; the search line must read `search: ddgs (default)`. Phase 2 runs live with web off, flag --no-web
#            (T-0168), so Chromium stays on the loopback start pages and the connect rule below holds exactly; the web
#            path is covered by scripts/locked/t0160_web_smoke.sh.
#
# Regime (stated with the run; n = 1 per query, unseeded, one run per SHA): profile lean, vLLM v0.29.0, NVFP4
# diffusiongemma-26B-A4B rev 2ea8372, canvas 256, 48 steps, thinking on, prefix caching on, versions per
# services/versions.lock, tools_v1, repair on, loop on, max_tool_steps 8, --silent (no speakers, no headset), typed
# turns, ddgs 9.16.1 / primp 2.0.1, ENGINES duckduckgo -> brave -> mojeek, client timeout 5 s per engine, proxies
# unset; phase 2 headless Chromium 1280x800 on the loopback start pages. The engine, TTS and STT are already up
# (this script boots nothing); it refuses otherwise.
#
# Refuses (exit 2) unless: the tree is clean; a `Merge T-0125` and a `Merge T-0160 group C impl` are ancestors of
# HEAD (T-0168 amends 13.1: the impl merge guarantees `--no-web` exists); the engine, TTS and STT are healthy;
# ports 8000 and 1420 are free; strace is present; no proxy variable is set; the output directory does not exist;
# a CPU pre-flight (phase 2's command plus --dry-run) exits 0 with an egress=loopback tools line and no egress=web.
#
# Hard stops (exit 1): no `web search` note with status executed and results >= 1 in a phase ("provider refused"
# when every search failed at the provider, a distinct message from a code failure); any non-loopback connect that
# is not to a resolv.conf nameserver on port 53 or to an ALLOWED_HOSTS address; a non-loopback listener; port 8000
# or 1420 still bound after SIGINT; the phase-2 search line is not `search: ddgs (default)`; phase 2 prints the web
# startup notice.
#
# T-0163 criterion 12: the run log states `brave api: configured|not configured` (the production key loader, in a
# child; only the status word crosses back). Only when configured are brave_api.ALLOWED_HOSTS addresses allowed in
# the connect audit, and the "provider refused" stop names brave_api among the refusers.
#
# Report-only: the spoken answers, and per turn the searches, statuses, engines, ms and hosts. Output:
# logs/t0125-search-smoke-<sha7>/. Run (console log's last line is the real exit code):
#
#   ( nohup bash scripts/locked/t0125_search_smoke.sh > logs/t0125-search-smoke-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0125-search-smoke-<SHA>-run.log ) &
set -euo pipefail
trap 'rc=$?; echo "FAIL: exit $rc at line $LINENO: $BASH_COMMAND" >&2' ERR
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

if [ "$#" -ne 0 ]; then echo "REFUSED: no arguments expected (got $*)" >&2; exit 2; fi

SHA=$(git rev-parse --short=7 HEAD)
DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
  echo "REFUSED: dirty tree: $(printf '%s' "$DIRTY" | tr '\n' ' ')" >&2; exit 2
fi
MERGES=$(git log --merges --format='%H %s' HEAD || true)
# here-strings, never a pipe: under pipefail `printf | grep -q` can die of SIGPIPE (141) once history is large
if ! grep -Eq '^[0-9a-f]+ Merge T-0125\b' <<< "$MERGES"; then
  echo "REFUSED: T-0125's merge commit is not an ancestor of HEAD (no 'Merge T-0125' merge reachable from HEAD)" >&2; exit 2
fi
# the implementation's merge, not a tests-only merge ("Merge T-0160 group C tests ..."): phase 2 needs --no-web
if ! grep -Eq '^[0-9a-f]+ Merge T-0160 group C impl\b' <<< "$MERGES"; then
  echo "REFUSED: 'Merge T-0160 group C impl' is not an ancestor of HEAD (a tests-only merge does not count); phase 2 needs --no-web (T-0168, T-0125 13.1 amended)" >&2; exit 2
fi
for v in DDGS_PROXY PRIMP_PROXY HTTPS_PROXY https_proxy HTTP_PROXY http_proxy ALL_PROXY all_proxy; do
  if [ -n "${!v:-}" ]; then
    echo "REFUSED: proxy variable $v is set; the search backend refuses a proxy and this smoke runs without one" >&2; exit 2
  fi
done
if ! command -v strace >/dev/null 2>&1; then
  echo "REFUSED: strace is absent; the connect audit needs it" >&2; exit 2
fi
for P in 8000 1420; do
  if ss -tln | grep -q ":$P "; then
    echo "REFUSED: port $P already bound" >&2; exit 2
  fi
done
H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health || true)
[ "$H" = 200 ] || { echo "REFUSED: engine (vllm /health) = $H, not healthy" >&2; exit 2; }
T=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8081/health || true)
[ "$T" = 200 ] || { echo "REFUSED: tts /health = $T, not healthy" >&2; exit 2; }
S=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8080/api/asr-streaming || true)
[ "$S" = 400 ] || { echo "REFUSED: stt route = $S, not healthy" >&2; exit 2; }

OUT="logs/t0125-search-smoke-$SHA"
if [ -e "$OUT" ]; then
  echo "REFUSED: $OUT exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
fi
# T-0168 criterion 3a: CPU pre-flight. A real start prints no tools line (only --dry-run does), so phase 2's exact
# command plus --dry-run (opens nothing) shows the egress before any GPU time; its own throwaway dirs.
PRE=$(mktemp -d)
PRE_RC=0
PRE_OUT=$(uv run --all-extras diffusion_gearvis session --silent --serve-hud --tools live --browser headless --no-web \
  --dry-run --data-dir "$PRE/data" --log-dir "$PRE/sessions" 2>&1) || PRE_RC=$?
rm -rf "$PRE"
if [ "$PRE_RC" -ne 0 ]; then
  echo "REFUSED: pre-flight: phase 2's command with --dry-run failed with exit status $PRE_RC: ${PRE_OUT//$'\n'/ }" >&2; exit 2
fi
if ! grep -Eq 'tools=live .*egress=loopback( |$)' <<< "$PRE_OUT"; then
  echo "REFUSED: pre-flight: phase 2's dry-run shows no tools=live line with egress=loopback (web off)" >&2; exit 2
fi
if grep -qF 'egress=web' <<< "$PRE_OUT"; then
  echo "REFUSED: pre-flight: phase 2's dry-run shows egress=web despite --no-web" >&2; exit 2
fi
mkdir -p "$OUT"
LOG="$OUT/run.log"
{
  echo "# T-0125 search smoke - commit ${SHA} (clean) - $(date -Is) - host network: $(ip -brief route get 1.1.1.1 2>/dev/null | head -1 || echo unknown)"
  echo "# regime: profile lean, vLLM v0.29.0, NVFP4 diffusiongemma-26B-A4B rev 2ea8372, canvas 256, steps 48,"
  echo "#   thinking on, --silent, typed turns, n=1 per query, unseeded; ddgs ENGINES duckduckgo -> brave -> mojeek;"
  echo "#   phase 2 live with --no-web (web off; web path: t0160_web_smoke.sh), headless Chromium 1280x800,"
  echo "#   loopback start pages, search on by default. Report-only."
} > "$LOG"
# T-0163 criterion 12: the Brave Search API fallback's status from the production loader in a child; anything but
# the word "configured" (a failed child included) reads as not configured
BRAVE=$(uv run --all-extras python -c 'from local_astra.search import brave_key; print(brave_key.load_key().status)' 2>/dev/null || echo "not configured")
if [ "$BRAVE" != configured ]; then BRAVE="not configured"; fi
echo "# brave api: $BRAVE" | tee -a "$LOG"

PHASE1_TURNS=(
  "Search the web for the height of the Eiffel Tower and tell me in one sentence."
  "Search the web for today's weather in Paris."
  "Search the web for the latest Python release."
)
PHASE2_TURNS=(
  "Search the web for the height of the Eiffel Tower and tell me in one sentence."
)

SPID=""
stop_session() {
  if [ -n "$SPID" ]; then
    kill -INT -"$SPID" 2>/dev/null || kill -INT "$SPID" 2>/dev/null || true
    wait "$SPID" 2>/dev/null || true
    SPID=""
  fi
}
trap stop_session EXIT

# run_phase NAME EXPECTED_SEARCH_LINE TURNS... -- (the session command of each phase is spelled out below)
run_phase() {
  local name="$1" expect="$2"; shift 2
  local turns=()
  while [ "$1" != "--" ]; do turns+=("$1"); shift; done
  shift
  local dir="$OUT/$name" aux="$OUT/$name/aux"
  mkdir -p "$dir" "$aux"
  local session_out="$dir/session.out"
  echo "== $name: session start $(date -Is)" | tee -a "$LOG"
  if [ "$name" = phase1 ]; then
    # phase 1: dry-run, search opted in (unbuffered: stdout goes to a file and the search line is grepped live)
    PYTHONUNBUFFERED=1 setsid strace -f -qq -e trace=connect -o "$aux/connect.log" \
      uv run --all-extras diffusion_gearvis session --silent --serve-hud --tools dry-run --search ddgs \
      --data-dir "$dir/data" --log-dir "$dir/sessions" > "$session_out" 2>&1 &
  else
    # phase 2: live with no search flag (the default), web off (T-0168)
    PYTHONUNBUFFERED=1 setsid strace -f -qq -e trace=connect -o "$aux/connect.log" \
      uv run --all-extras diffusion_gearvis session --silent --serve-hud --tools live --browser headless --no-web \
      --data-dir "$dir/data" --log-dir "$dir/sessions" > "$session_out" 2>&1 &
  fi
  SPID=$!
  for _ in $(seq 1 120); do
    ss -tln | grep -q ':8000 ' && break
    if ! kill -0 "$SPID" 2>/dev/null; then
      echo "HARD STOP: $name: the session exited before binding" | tee -a "$LOG"; tee -a "$LOG" < "$session_out"; exit 1
    fi
    sleep 0.5
  done
  ss -tln | grep -q ':8000 ' || { echo "HARD STOP: $name: the conductor endpoint never bound" | tee -a "$LOG"; exit 1; }
  if ss -tln | grep -E ':(8000|1420) ' | grep -qv '127\.0\.0\.1'; then
    echo "HARD STOP: $name: non-loopback listener" | tee -a "$LOG"; exit 1
  fi
  { grep -m1 'regime:' "$session_out" || echo "# no measured regime line"; } | tee -a "$LOG"
  grep -m1 'web search is ' "$session_out" | tee -a "$LOG" || true
  if ! grep -qxF "astra session: search: $expect" "$session_out"; then
    echo "HARD STOP: $name: the search line is not 'search: $expect': $(grep -m1 'search: ' "$session_out" || echo none)" | tee -a "$LOG"; exit 1
  fi
  if [ "$name" = phase2 ]; then
    # T-0168 criterion 3b: web off on the real start, before any turn (the egress itself is checked by the pre-flight)
    if grep -qF 'web browsing is on' "$session_out"; then
      echo "HARD STOP: phase2: the web startup notice ('web browsing is on') was printed despite --no-web" | tee -a "$LOG"; exit 1
    fi
  fi
  local i=0 rc
  for q in "${turns[@]}"; do
    i=$((i + 1))
    echo "-- $name turn $i: $q" | tee -a "$LOG"
    set +e
    timeout 300 uv run python scripts/demo/text_mode_probe.py "$q" --timeout-s 240 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    set -e
    echo "# $name turn $i probe rc $rc" | tee -a "$LOG"
  done
  stop_session
  for _ in $(seq 1 20); do ss -tln | grep -Eq ':(8000|1420) ' || break; sleep 0.5; done
  ss -tln | grep -q ':8000 ' && { echo "HARD STOP: $name: port 8000 still bound after SIGINT" | tee -a "$LOG"; exit 1; }
  ss -tln | grep -q ':1420 ' && { echo "HARD STOP: $name: port 1420 still bound after SIGINT" | tee -a "$LOG"; exit 1; }
  echo "# $name: last session output lines:" | tee -a "$LOG"
  tail -20 "$session_out" | tee -a "$LOG"
  analyse "$name" "$dir"
}

# analyse NAME DIR: the report rows and the two record-based hard stops (searches, connects)
analyse() {
  local name="$1" dir="$2"
  local arc=0
  uv run python - "$name" "$dir" "$BRAVE" <<'PY' 2>&1 | tee -a "$LOG" || arc=$?
import ipaddress, json, re, socket, sys
from pathlib import Path

from local_astra.search.ddgs_backend import ALLOWED_HOSTS, ENGINES

name, root, brave = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
refusers = ENGINES + ("brave_api",) if brave == "configured" else ENGINES
notes, errors, turns = [], [], []
for path in sorted((root / "sessions").glob("**/*.jsonl")):
    for line in path.read_text().splitlines():
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        data = rec.get("data") or {}
        if rec.get("kind") == "note" and isinstance(data, dict) and data.get("what") == "web search":
            notes.append((rec.get("turn_id"), data))
        elif rec.get("kind") == "error" and isinstance(data, dict) and data.get("what") == "tool.search":
            errors.append((rec.get("turn_id"), data.get("error")))
        elif rec.get("kind") == "turn.record" and isinstance(data, dict):
            turns.append(data)
for turn in turns:
    extras = turn.get("extras") or {}
    print(f"{name} turn {turn.get('turn_id')}: searches={extras.get('searches', 0)} outcome={turn.get('outcome')}")
for turn_id, n in notes:
    print(
        f"{name} search {turn_id} step {n.get('step')}: status={n.get('status')} engine={n.get('engine')} "
        f"results={n.get('results')} ms={n.get('ms')} hosts={n.get('hosts')} (+{n.get('hosts_more')})"
    )
code = 0
if not any(n.get("status") == "executed" and (n.get("results") or 0) >= 1 for _, n in notes):
    failed = [n for _, n in notes if n.get("status") == "failed"]
    # only "error" and "rate_limited" fall through to the next engine; "network" stops at the first one
    # the session's tool.search error is repr(exc): SearchFailed('<kind>: <detail>')
    fell = [e for _, e in errors if e and re.match(r"SearchFailed\(['\"](error|rate_limited):", e)]
    network = [e for _, e in errors if e and re.match(r"SearchFailed\(['\"]network:", e)]
    if failed and len(fell) == len(failed) == len(errors):
        print(f"HARD STOP: {name}: provider refused: no search returned results; every engine ({', '.join(refusers)}) refused")
        for turn_id, e in errors:
            print(f"  {turn_id}: last engine's failure: {e}")
    elif failed and network and len(network) + len(fell) == len(failed) == len(errors):
        print(f"HARD STOP: {name}: provider refused or unreachable: a network failure stops at the first engine, so not every engine was tried")
        for turn_id, e in errors:
            print(f"  {turn_id}: {e}")
    else:
        print(f"HARD STOP: {name}: no web search note with status executed and results >= 1 (a code failure, not the provider)")
        for turn_id, e in errors:
            print(f"  {turn_id}: {e}")
    code = 1

def nameservers():
    out = set()
    for line in Path("/etc/resolv.conf").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "nameserver":
            out.add(parts[1].split("%")[0])
    return out

hosts = set(ALLOWED_HOSTS)
if brave == "configured":
    from local_astra.search.brave_api import ALLOWED_HOSTS as BRAVE_HOSTS
    hosts |= BRAVE_HOSTS
allowed = set()
for host in hosts:
    try:
        allowed |= {info[4][0] for info in socket.getaddrinfo(host, 443)}
    except OSError:
        pass
dns = nameservers()
pattern = re.compile(
    r'connect\(\d+, \{sa_family=AF_INET6?, sin6?_port=htons\((\d+)\),.*?(?:inet_addr\("([^"]+)"\)|inet_pton\(AF_INET6, "([^"]+)")'
)
bad = []
for line in (root / "aux" / "connect.log").read_text().splitlines():
    m = pattern.search(line)
    if not m:
        continue
    port, addr = int(m.group(1)), m.group(2) or m.group(3)
    ip = ipaddress.ip_address(addr)
    mapped = getattr(ip, "ipv4_mapped", None)
    if ip.is_loopback or (mapped is not None and mapped.is_loopback) or ip.is_unspecified:
        continue
    if port == 53 and addr in dns:
        continue
    if addr in allowed or (mapped is not None and str(mapped) in allowed):
        continue
    bad.append(line)
if bad:
    print(f"HARD STOP: {name}: {len(bad)} non-loopback connect(s) outside the resolv.conf nameservers and ALLOWED_HOSTS:")
    for line in bad[:20]:
        print("  " + line)
    code = 1
else:
    print(f"{name}: connects: loopback, resolv.conf nameservers on 53 and ALLOWED_HOSTS addresses only")
sys.exit(code)
PY
  [ "$arc" -eq 0 ] || exit 1
}

PHASE1_LINE="search: ddgs (flag)"
PHASE2_LINE="search: ddgs (default)"  # criterion 13.3: the default, no --search flag
run_phase phase1 "${PHASE1_LINE#search: }" "${PHASE1_TURNS[@]}" --
run_phase phase2 "${PHASE2_LINE#search: }" "${PHASE2_TURNS[@]}" --
trap - EXIT
echo "OK: T-0125 search smoke in $LOG (report-only)"
