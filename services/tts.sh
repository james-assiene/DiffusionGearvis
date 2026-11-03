#!/usr/bin/env bash
# Pocket TTS on CPU, pinned to two cores. Loopback only.
#
# Started through `uvx`, not a `pocket-tts` on PATH: Pocket TTS has only ever been installed here
# as a uv tool run (PROGRESS Runs, 2026-09-14), so `taskset -c 2,3 pocket-tts serve …` failed with
# "taskset: failed to execute pocket-tts" on the first real `astra up` (2026-09-15, T-0041).
# The version is pinned — a bare `uvx pocket-tts` resolves to whatever PyPI serves today — and the
# pin is the one in services/versions.lock ([speech] pocket_tts).
set -euo pipefail
CORES=${ASTRA_TTS_CORES:-2,3}
POCKET_TTS_VERSION=${POCKET_TTS_VERSION:-3.1.0}
exec taskset -c "$CORES" uvx --from "pocket-tts==$POCKET_TTS_VERSION" pocket-tts serve --host 127.0.0.1 --port 8081
