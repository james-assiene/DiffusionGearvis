#!/usr/bin/env bash
# T-0006 criterion 11 / T-0021 criteria 9 and 11: the PipeWire null sink the audio tests measure
# themselves against, and its monitor as a capture source.
#
# `pactl` is NOT on this box (docs/tasks/T-0006.md), so the sink is created with `pw-cli` and
# verified with `pw-dump`. `object.linger=true` keeps the node alive after pw-cli exits; without
# it the node dies with the process that made it and every rig test would skip.
#
# The monitor of this sink is the oracle for what was actually rendered: it needs no speaker, no
# human and no GPU, and it cannot "pass silently" -- `up` fails loudly if the node or its monitor
# ports do not show up in pw-dump, and the rig tests fail (never skip) if its monitor recording
# is empty. A *monitor* capture is a bit-exact echo path, which is what makes it the strictest
# possible test of the self-speech gate (T-0021 criterion 9).
#
# Capturing it is `pw-record -P stream.capture.sink=true --target astra-test-sink`: without that
# property the session manager finds no *source* by that name and links the stream to the default
# source instead -- on this box the webcam mic, which records the room at a peak of ~0.12 and
# looks exactly like a rig that is working (scripts/audio_loopback.py::Monitor).
#
#   bash scripts/audio_rig.sh up | down | status | monitor
set -euo pipefail

SINK="${ASTRA_TEST_SINK:-astra-test-sink}"
HERE="$(dirname "$0")"

# The pw-dump parsing and the create/destroy calls live in the Python helper next door: shell
# has no JSON parser and `pactl` is not on this box (docs/tasks/T-0006.md).
RIG="$HERE/audio_loopback.py"
ACTION="${1:-status}"

# Monitor ports of a null sink are named "<node.name>:monitor_<CH>" in pw-dump's object.path.
# Checked by name, not merely by the node existing: a sink whose monitor ports never appeared
# gives a capture of digital silence, and "no echo found" through a dead path is the exact
# failure T-0021 criterion 11 refuses to let pass quietly.
monitor_ok() {
  # Buffered, not piped into grep (T-0021 ruling A12): under `set -o pipefail` a `grep -q` that
  # exits at its first match SIGPIPEs pw-dump, and a healthy rig then reports failure -- but only
  # sometimes, which is the worst kind of rig check.
  local dump
  dump=$(pw-dump)
  case "$dump" in
    *"$SINK:monitor_"*) return 0 ;;
    *) return 1 ;;
  esac
}

say_monitor() {
  echo "$SINK.monitor is a capture source (T-0021 criterion 9: a bit-exact echo path)"
  echo "capture it with: pw-record -P stream.capture.sink=true --target $SINK --rate 48000"
}

case "$ACTION" in
  monitor)
    if monitor_ok; then
      say_monitor
      exit 0
    fi
    echo "$SINK has no monitor ports in pw-dump: nothing can capture it" >&2
    exit 1
    ;;
  up)
    python3 "$RIG" --rig up --sink "$SINK"
    for _ in $(seq 50); do
      if monitor_ok; then
        say_monitor
        exit 0
      fi
      sleep 0.1
    done
    echo "$SINK is up but its monitor ports never appeared in pw-dump" >&2
    exit 1
    ;;
  *)
    exec python3 "$RIG" --rig "$ACTION" --sink "$SINK"
    ;;
esac
