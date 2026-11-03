#!/usr/bin/env bash
# Dump the rendered DOM of a page in headless Chrome — the --dump-dom sibling of hud_shot.sh.
#
#   scripts/hud_dom.sh <url> [wait_ms] > logs/hud/hud.dom.html
#   scripts/hud_dom.sh 'http://127.0.0.1:1420/?source=canvas' 4000
#
# No GPU, no display, no node. Same warning as hud_shot.sh: this CANNOT exercise anything behind
# a permission prompt (getDisplayMedia, getUserMedia) — those need a real click.
#
# --virtual-time-budget suits a page that fills itself from timers. It is NOT a way to drive a
# WebSocket conversation: virtual time runs ~15x real time here and a scripted server sequence
# arrives truncated (T-0008 ruling R3) — the integration test uses the DevTools Protocol instead.
set -euo pipefail
url=${1:?url}; wait_ms=${2:-4000}
chrome=$(command -v google-chrome || command -v chromium || command -v chromium-browser) \
  || { echo "no chrome/chromium on PATH" >&2; exit 1; }
"$chrome" --headless=new --disable-gpu --no-sandbox --hide-scrollbars \
  --window-size=1280,800 --virtual-time-budget="$wait_ms" \
  --dump-dom "$url" 2>/dev/null
