#!/usr/bin/env bash
# Render a page in headless Chrome and screenshot it — the eyes for HUD work.
#
#   scripts/hud_shot.sh <url-or-file> <out.png> [WxH] [wait_ms]
#   scripts/hud_shot.sh hud/dist/index.html /tmp/hud.png 1280x720 800
#
# No GPU, no display, no node: works with the DP cable unplugged and the session unattended.
# The PNG is then readable by an agent as an image, so "does it look right" is answerable
# without a human in front of the monitor. What this CANNOT do: anything behind a permission
# prompt (getDisplayMedia, getUserMedia) — those need a real click, and a test that claims to
# exercise them headless is lying.
set -euo pipefail
src=${1:?url or file}; out=${2:?output png}; size=${3:-1280x720}; wait_ms=${4:-500}
case "$src" in http://*|https://*|file://*) url=$src;; *) url="file://$(realpath "$src")";; esac
chrome=$(command -v google-chrome || command -v chromium || command -v chromium-browser) \
  || { echo "no chrome/chromium on PATH" >&2; exit 1; }
# --virtual-time-budget lets timers/fetches settle before the capture, so a page that fills itself
# from a WebSocket or an animation frame is caught after it did so, not before.
"$chrome" --headless=new --disable-gpu --no-sandbox --hide-scrollbars \
  --window-size="${size/x/,}" --virtual-time-budget="$wait_ms" \
  --screenshot="$out" "$url" 2>/dev/null
[ -s "$out" ] || { echo "screenshot produced no file: $out" >&2; exit 1; }
echo "$out ($(stat -c%s "$out") bytes, $size)"
