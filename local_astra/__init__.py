"""local_astra — an on-device voice-and-vision assistant inspired by research demos like
Google's Project Astra (DiffusionGemma + Kyutai STT + Pocket TTS).

Package map (see docs/02-architecture.md §10):

- ``contracts``  Pydantic models for every internal event, command and wire message.
- ``conductor``  turn machine (pure), session executors, speech/ui gates, prompt builder, clients.
- ``reflex``     CPU-side reflex lane: VAD, turn predictor, owner-voice veto, filler policy.
- ``perception`` frame sources (browser tab in v1), resolution policy, ring buffer.
- ``memory``     memory-lite (SQLite + FTS5) store and idle captioner.
- ``ui``         A2UI v0.9.1 schema registry, validation and topology checks.
- ``telemetry``  span names and the turns.jsonl writer.
"""

__version__ = "0.0.1"
