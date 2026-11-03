#!/usr/bin/env bash
# T-0007a group B census (criteria 12, 13). Approved by the launch-reviewer 2026-09-14; run via /launch.
set -euo pipefail
cd /home/user/demos/local_astra

[ -z "$(git status --porcelain)" ] || { echo "REFUSE: dirty tree"; exit 10; }
SHA="$(git rev-parse --short HEAD)"

H="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8081/health)"
A="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8080/api/asr-streaming)"
[ "$H" = 200 ] || { echo "REFUSE: pocket-tts /health = $H (want 200)"; exit 12; }
[ "$A" = 400 ] || { echo "REFUSE: asr-streaming = $A (want 400, the mounted-route signal)"; exit 13; }

LOG="logs/t0007a-group-b-census-${SHA}.log"
{
  echo "# T-0007a group B census — criteria 12 and 13 (NOT 14: no TTS row in this module)"
  echo "# commit ${SHA} (clean) · $(date -Is)"
  echo "# regime: STT kyutai/stt-1b-en_fr-candle in astra-moshi:0.6.4, moshi-server 0.6.4, --cpu,"
  echo "#   batch_size=2, asr_delay_in_tokens=6, temperature=0.0, container up 15h (pid 355628),"
  echo "#   co-resident with vLLM v0.29.0 / diffusiongemma-26B-A4B-it-NVFP4 idle (26936 MiB, 0% util);"
  echo "#   audio=tests/fixtures/audio/pocket_tts_sentence.wav 24 kHz mono s16le 4.040 s at realtime"
  echo "#   pace + 3.2 s harness silence; n=1, seed n/a; loopback only."
} > "$LOG"

START=$(date +%s)
uv run pytest tests/integration/test_speech_clients_live.py -q -m gpu -rA \
  --junitxml="logs/t0007a-group-b-census-${SHA}.xml" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
WALL=$(( $(date +%s) - START ))

echo "# wall ${WALL}s, pytest rc ${RC}" | tee -a "$LOG"
[ "$RC" -eq 0 ]                    || { echo "HARD STOP: pytest rc=$RC";            exit 20; }
grep -qE '^3 passed' "$LOG"        || { echo "HARD STOP: not 3 passed (skip gate?)"; exit 21; }
grep -qE '[0-9]+ skipped' "$LOG"   && { echo "HARD STOP: a row skipped";            exit 22; }
grep -q 'inbound messages:' "$LOG" || { echo "HARD STOP: no census block captured"; exit 23; }
[ "$WALL" -lt 60 ]                 || { echo "HARD STOP: ${WALL}s > 60s budget";    exit 24; }
echo "OK: census in $LOG"
