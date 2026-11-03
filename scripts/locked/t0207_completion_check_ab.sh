#!/usr/bin/env bash
# T-0207 B: one fresh engine-only boot, then scripts/gpu/t0207_completion_check_ab.py --part both:
#   Part 1 (B3, constructed corpus with ground truth): ten checkable tasks (S6-S11, S14, S15, D4, D5) x two states
#     (DONE: the recorded golden path; NOT_DONE: its decisive click moved to a pinned dead coordinate), each run as
#     scripted tool calls through a real Session(completion_check=True) and the real BrowserToolExecutor in headless
#     Chromium on loopback pages, the draft the pinned claim sentence; only the completion_check request goes to the
#     engine; n = 5 per item, order seed 0 = 100 turns, concurrency 1. The corpus is rebuilt first with a scripted
#     LLM for every request (no engine) and must match its pins, else the harness refuses (exit 2);
#   Part 2 (B4, live A/B): the T-0138 suite S6-S15, whatever the state of T-0205-b (the D tasks are never in Part 2,
#     whether or not a "Merge T-0205-b:" commit is an ancestor; the manifest records which as t0205_b_merged), through
#     a real Session at HEAD's live default (--no-web: loopback suite pages), arms completion_check False/True,
#     every other setting explicit and equal, n = 3 per task per arm, order seed 0 = 60 typed turns, one headless
#     browser per turn;
#   then the decision (B5) and report.json/report.md, written by the harness itself;
# then the engine is stopped.
#
# Regime (state it beside every number): profile lean, engine only (no STT, no TTS); vLLM, torch, CUDA, driver,
# model and chat template per services/versions.lock at HEAD (container vllm/vllm-openai:v0.29.0); canvas 256,
# 48 steps, thinking on, prefix caching on, repair on; tools v1 bbfc53cc; HEAD's live defaults (live guidance v3,
# v2's text with --no-web; the screen reminder as cli.screen_reminder_of resolves it, on since T-0201 D, in both
# parts and recorded in the manifest); visual tokens 1120, ReGround off; max_tool_steps 8; headless Chromium 1280x800,
# loopback only; typed turns, fake STT/TTS/audio; order and corpus seed 0; engine unseeded; concurrency 1; clean
# tree, idle box.
#
# The boot regime is scripts/locked/t0201_screen_reminder_ab.sh's verbatim (itself t0138_suite_ab.sh's):
# services/vllm.sh at 806da12 and unchanged, prefix caching on, plugin wheel rebuilt, port and container refusals,
# one run per SHA, one boot, setsid services/vllm.sh, health wait up to 600 s, the load-settle wait, the engine
# stopped on every exit. STT and TTS must be down (ports 8080/8081 are refused below). The harness itself refuses a
# box that is not idle: any GPU compute process other than the engine (VLLM::EngineCore) -- **James: close desktop
# Chrome first** (it is a GPU compute app), and moshi/TTS must be down --, a 1-min load >= 2, port 8000 or 1420
# bound (no astra session, no HUD), Chromium absent.
#
# /launch only, from a clean tree at (or after) T-0207 B's merge commit; the console log's last line is the real exit
# code:
#
#   ( nohup bash scripts/locked/t0207_completion_check_ab.sh > logs/t0207-completion-check-ab-<SHA>-run.log 2>&1; echo "exit $?" >> logs/t0207-completion-check-ab-<SHA>-run.log ) &
#
# Refusals (exit 2), here before the boot: any argument; a dirty tree (its paths named); no merge commit reachable
# from HEAD whose subject starts "Merge T-0207-a2:" (the completion check in the session) or "Merge T-0207-b:" (this
# harness; a branch merge such as "Merge main into T-0207-b-..." does not count); services/vllm.sh changed since
# 806da12; an Astra port (8001/8002/8080/8081) bound; any container running; the harness or a file it imports absent;
# the output directory (or this script's aux directory) already existing. The merge log is matched through a
# here-string, never a pipe: under pipefail `printf | grep -q` can fail when grep exits at the first match while
# printf is still writing (SIGPIPE once the log passes the 64 KiB pipe buffer), which would refuse a merged HEAD.
# After the engine is healthy the script waits for the 1-min load (/proc/loadavg) to drop below 2, polling every 10 s
# for up to 10 min and logging each reading; if it never does, exit 1 (the engine is stopped), not a harness refusal
# after a wasted boot.
#
# Budget: the harness runs under `timeout -k 60s 240m` (a timeout is exit 124, a failed step; SIGKILL 60 s after the
# TERM). Estimate about 1.5-2.5 h: the corpus build (20 scripted turns, no engine) about 3 min; Part 1's 100 turns at
# about 10-20 s each (a scripted path of 1-6 browser actions plus one check request) is 17-35 min; Part 2's 60 suite
# turns at T-0138's 16-20 s per turn plus the TIP arm's check is 20-40 min, up to the task's 2-3 h estimate when
# turns run long. The harness has no run budget of its own, so a timeout leaves no report.json (samples.jsonl and
# trials.jsonl are appended per turn and survive). Each Part 1 turn is bounded at 240 s by the harness.
#
# `uv run --locked`: uv.lock is used as committed, never re-resolved. The proxy variables are unset (loopback only).
# Output: logs/t0207-completion-check-ab-<SHA>/ (order.json, order_part2.json, corpus.json, manifest.json,
# samples.jsonl, part1/, trials.jsonl, steps.jsonl, sessions/, report.json, report.md); boot lines appended to
# logs/vllm.log.
# Exit: 0 the harness measured (PASS, FAIL and INCONCLUSIVE are all exit 0: read report.md); 1 a step failed (boot,
# no healthy engine, a run that stopped early, the timeout); 2 refused (here, or by the harness: before any socket,
# or on a corpus not as pinned, before the engine is asked).
# Report-only: no bar, no threshold file changes; group C (on by default) waits on this report or James's ruling.
#
# File mode: chmod 444 in the working tree; git records it as 100644 (git keeps only the executable bit), which is
# expected. Run it with `bash`, as above.
set -euo pipefail
trap 'rc=$?; echo "FAIL: exit $rc at line $LINENO: $BASH_COMMAND" >&2' ERR
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

if [ "$#" -ne 0 ]; then echo "REFUSED: no arguments expected (got $*)" >&2; exit 2; fi

dock() {
  if docker "$@" 2>/dev/null; then return 0; fi
  sg docker -c "docker $(printf '%q ' "$@")"
}

# the first 7 hex digits exactly, as the harness names its default out-dir (sha[:7]); --short=7 can be longer
SHA=$(git rev-parse HEAD | cut -c1-7)
DIRTY=$(git status --porcelain)
if [ -n "$DIRTY" ]; then
  echo "REFUSED: dirty tree: $(printf '%s' "$DIRTY" | tr '\n' ' ')" >&2; exit 2
fi
MERGES=$(git log --merges --format='%H %s' HEAD || true)
if ! grep -Eq '^[0-9a-f]+ Merge T-0207-a2:' <<<"$MERGES"; then
  echo "REFUSED: T-0207 group A2's merge commit is not an ancestor of HEAD (no 'Merge T-0207-a2:' merge reachable from HEAD); the TIP arm is A2's completion check" >&2; exit 2
fi
if ! grep -Eq '^[0-9a-f]+ Merge T-0207-b:' <<<"$MERGES"; then
  echo "REFUSED: T-0207 group B's merge commit is not an ancestor of HEAD (no 'Merge T-0207-b:' merge reachable from HEAD; a 'Merge main into T-0207-b-...' branch merge does not count); the A/B runs at the merge commit" >&2; exit 2
fi
if [ -n "$(git diff 806da12..HEAD -- services/vllm.sh)" ]; then
  echo "REFUSED: services/vllm.sh changed since 806da12; the reviewed boot regime (prefix caching on) no longer applies" >&2; exit 2
fi
if ss -ltn 2>/dev/null | grep -Eq '127\.0\.0\.1:(8001|8002|8080|8081)\b'; then
  echo "REFUSED: an Astra port (8001/8002/8080/8081) is already bound (a fresh boot is needed, and STT/TTS must be down)" >&2; exit 2
fi
if [ -n "$(dock ps -q)" ]; then
  echo "REFUSED: containers are running: $(dock ps --format '{{.Names}} {{.Image}}' | tr '\n' ' ')" >&2; exit 2
fi
HARNESS=scripts/gpu/t0207_completion_check_ab.py
# the harness imports T-0201 C's, which imports T-0138's, T-0134's, T-0132's, T-0131's and T-0111g's (B1: imported,
# not copied); T-0138's task table and oracle, T-0205's scenario table and oracle, the check prompt; both arms' tools
for f in "$HARNESS" scripts/gpu/t0201_screen_reminder_ab.py scripts/gpu/t0138_suite_ab.py scripts/gpu/t0134_feedback_ab.py scripts/gpu/t0132_crop_budget_ab.py scripts/gpu/t0131_grounding_ab.py scripts/gpu/t0111g_live_smoke.py local_astra/toolcall/tools_v1.json local_astra/toolcall/completion_check.py local_astra/pages/suite_tasks.py local_astra/pages/scenario_tasks.py; do
  if [ ! -f "$f" ]; then
    echo "REFUSED: $f is absent" >&2; exit 2
  fi
done
OUT="logs/t0207-completion-check-ab-$SHA"
AUX="logs/t0207-launch-$SHA"
for d in "$OUT" "$AUX"; do
  if [ -e "$d" ]; then
    echo "REFUSED: $d exists; the run happens once per SHA (move it aside only with the human)" >&2; exit 2
  fi
done
mkdir -p "$AUX"

make plugin-wheel || { echo "FAIL: make plugin-wheel failed" >&2; exit 1; }
{ echo "# t0207 completion-check A/B boot, commit $SHA, $(date -Is)"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.free --format=csv,noheader || true
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader || true
  uptime || true
} | tee -a logs/vllm.log

setsid bash "$PWD/services/vllm.sh" >> logs/vllm.log 2>&1 &
ENGINE_PID=$!

stop_engine() {
  local ids
  ids=$(dock ps -q --filter ancestor=vllm/vllm-openai:v0.29.0 || true)
  if [ -n "$ids" ]; then dock stop $ids >/dev/null || true; fi
  kill "$ENGINE_PID" 2>/dev/null || true
  echo "engine stopped $(date -Is)"
  echo "containers left: $(dock ps -q | wc -l)"
}
trap stop_engine EXIT

for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then break; fi
  if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "FAIL: the engine exited during boot; see logs/vllm.log" >&2; exit 1
  fi
  sleep 5
done
curl -fsS http://127.0.0.1:8001/health >/dev/null || { echo "FAIL: engine not healthy after 600 s" >&2; exit 1; }
echo "engine healthy $(date -Is)"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory after boot: /' || true
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader | sed 's/^/compute app: /' || true
uptime | sed 's/^/load after boot: /' || true

# the harness refuses a 1-min load >= 2 (B2, exit 2); the plugin-wheel build and the boot can leave it high, so wait
# here (every 10 s, up to 10 min) rather than waste the boot; never settling is exit 1 (the trap stops the engine)
LOAD_OK=0
for i in $(seq 0 60); do
  LOAD1=$(cut -d' ' -f1 /proc/loadavg)
  echo "load1 $LOAD1 (wait ${i}0 s) $(date -Is)"
  if awk -v l="$LOAD1" 'BEGIN { exit !(l < 2) }'; then LOAD_OK=1; break; fi
  if [ "$i" -lt 60 ]; then sleep 10; fi
done
if [ "$LOAD_OK" -ne 1 ]; then
  echo "FAIL: the 1-min load stayed at or above 2 for 10 min after the boot (last $LOAD1); the harness would refuse a busy box. Find what is loading the box, then /launch again" >&2; exit 1
fi

NOPROXY=(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY)

echo "== T-0207 B completion-check A/B harness, both parts $(date -Is)"
AB=0
timeout -k 60s 240m "${NOPROXY[@]}" uv run --locked --all-extras python "$HARNESS" \
  --base-url http://127.0.0.1:8001/v1 --part both || AB=$?
echo "T-0207 harness exit $AB"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/device memory at the end: /' || true

if [ -f "$OUT/report.json" ]; then
  echo "report: $OUT/report.md"
  for f in report.json order.json order_part2.json corpus.json manifest.json samples.jsonl trials.jsonl steps.jsonl; do
    if [ -f "$OUT/$f" ]; then echo "$f sha256 $(sha256sum "$OUT/$f" | cut -d' ' -f1)"; fi
  done
else
  echo "no $OUT/report.json" >&2
fi

echo "summary: harness $AB $(date -Is)"
if [ "$AB" -eq 2 ]; then exit 2; fi
if [ "$AB" -ne 0 ]; then exit 1; fi
exit 0
