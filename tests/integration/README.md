# tests/integration

Two or more real services or devices (docs/03 §5). Marker `integration`, plus the marker for the
hardware the module actually needs — they are a **pair**; it is not true that every module here
also carries `gpu`:

- `integration` + `gpu` — the 5090 box with the real GPU services (vLLM, the speech server).
- `integration` + `audio` — a PipeWire session with the audio rig: the null sink
  `astra-test-sink` (`scripts/audio_rig.sh`), or a real sink and the mic for the acoustic
  loopback. **No GPU** (`test_audio_null_sink.py`, `test_audio_acoustic_loopback.py`; T-0006
  ruling A4). The acoustic module adds `slow`.
- `integration` + `desktop` — a live desktop session (portal, windows).

Runs via `/launch` only. A module that needs hardware skips with the missing device **named** in
the skip reason, and fails — never skips — when the device is present but renders nothing
(T-0006 criterion 11: the default sink on this box is a Dummy Output while the DP cable is
unplugged, so "no echo found" and "no speaker" look identical).
