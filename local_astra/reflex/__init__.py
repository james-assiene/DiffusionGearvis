"""Reflex lane (docs/01 D7): CPU-only, never blocks the loop > 20 ms.

- ``vad``            Silero VAD v6 onset/offset
- ``turn_predictor`` smart-turn v3.2 (or Kyutai semantic VAD from the STT stream)
- ``owner_voice``    speaker check as an end-of-turn *veto* (needs ≥ 1.5 s of speech)
- ``filler_policy``  pure decision table for acknowledgement clips / interim updates — T-0013
"""
