"""Conductor: replaces Unmute's handler; design derived from Unmute (MIT), no code copied.

Owns audio I/O, the turn machine and both output channels.

Module map (docs/02 §2.2):

- ``turn_machine``  pure ``step(state, event) -> (state, commands)`` — T-0003
- ``session``       executors for commands; asyncio lives here, not in the machine — T-0007
- ``speech_gate``   committed answer text → TTS sentences — T-0004
- ``ui_gate``       ``ui`` segments → validated A2UI surfaces (json_repair → schema → topology) — T-0010
- ``prompt_builder`` prompt layout + budget; ``prefill`` anticipatory prefix — T-0005
- ``llm_client``    vLLM SSE client with commit/segment separation — T-0005
- ``stt_client``, ``tts_client``, ``audio_io`` — T-0006/T-0007
"""
