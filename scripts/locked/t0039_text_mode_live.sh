#!/usr/bin/env bash
# T-0039 live check: one typed turn through `astra session --silent --serve-hud`.
# Reviewed by the launch-reviewer 2026-09-14; run via /launch. GPU time: one vLLM turn.
set -euo pipefail
cd /home/user/demos/local_astra
[ -z "$(git status --porcelain)" ] || { echo "REFUSE: dirty tree"; exit 10; }
SHA="$(git rev-parse --short HEAD)"
for P in 8000 1420; do
  ss -tln | grep -q ":$P " && { echo "REFUSE: port $P already bound"; exit 11; }
done
H=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8001/health)
[ "$H" = 200 ] || { echo "REFUSE: vllm /health = $H"; exit 12; }
T=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8081/health)
[ "$T" = 200 ] || { echo "REFUSE: tts /health = $T"; exit 13; }
S=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8080/api/asr-streaming)
[ "$S" = 400 ] || { echo "REFUSE: stt route = $S"; exit 14; }
RUNDIR="logs/t0039-live-${SHA}"; mkdir -p "$RUNDIR"
LOG="$RUNDIR/run.log"; SESSION_LOG="$RUNDIR/session.log"
{
  echo "# T-0039 live text-mode check - commit ${SHA} (clean) - $(date -Is)"
  echo "# claimed regime (the measured one is the 'regime:' line below): profile lean, --silent"
  echo "#   (NullAudioDevice: no PortAudio, no mic, TTS runs unheard), vLLM v0.29.0"
  echo "#   diffusiongemma-26B-A4B-it-NVFP4, canvas 256, steps 48, thinking on, STT container CPU"
  echo "#   (connected, fed nothing), NO HUD PAGE -> no frame, text-only prompt (not pixel-first),"
  echo "#   cold memory in ${RUNDIR}/data, n=1 typed question, seed n/a, loopback only."
  echo "# Not a bench number: silent playback, no pixels, n=1."
} > "$LOG"
setsid uv run astra session --silent --serve-hud --data-dir "$RUNDIR/data" > "$SESSION_LOG" 2>&1 &
SPID=$!
trap 'kill -INT -$SPID 2>/dev/null || kill -INT $SPID 2>/dev/null || true; wait $SPID 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  ss -tln | grep -q ':8000 ' && break
  kill -0 $SPID 2>/dev/null || { echo "HARD STOP: session exited before binding"; tee -a "$LOG" < "$SESSION_LOG"; exit 19; }
  sleep 0.5
done
ss -tln | grep -q ':8000 ' || { echo "HARD STOP: the conductor endpoint never bound"; tee -a "$LOG" < "$SESSION_LOG"; exit 20; }
ss -tln | grep -E ':(8000|1420) ' | grep -qv '127\.0\.0\.1' && { echo "HARD STOP: non-loopback listener"; exit 22; }
{ grep -m1 'regime:' "$SESSION_LOG" || echo "# HARD STOP-worthy: no measured regime line"; } | tee -a "$LOG"
START=$(date +%s)
set +e
timeout 240 uv run python scripts/demo/text_mode_probe.py \
  "In one sentence, what is the capital of France?" --timeout-s 180 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
set -e
echo "# probe rc ${RC}, wall $(( $(date +%s) - START ))s" | tee -a "$LOG"
kill -INT -$SPID 2>/dev/null || kill -INT $SPID 2>/dev/null || true
SRC=0; wait $SPID || SRC=$?
trap - EXIT
echo "# session rc ${SRC} (uv may report 130 for SIGINT); last session log lines:" | tee -a "$LOG"
tail -20 "$SESSION_LOG" | tee -a "$LOG"
for _ in $(seq 1 20); do ss -tln | grep -q ':8000 ' || break; sleep 0.5; done
ss -tln | grep -q ':8000 ' && { echo "HARD STOP: 8000 still bound after SIGINT (T-0028 criterion 16)"; exit 23; }
ss -tln | grep -q ':1420 ' && { echo "HARD STOP: 1420 still bound after SIGINT"; exit 24; }
[ "$RC" -eq 0 ] || { echo "HARD STOP: the typed turn did not complete (probe rc=$RC)"; exit 21; }
grep -qi 'paris' "$LOG" || { echo "HARD STOP: the answer never said Paris"; exit 25; }
echo "OK: text mode live in $LOG"
