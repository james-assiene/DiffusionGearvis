"""Perception: frame sources and policies (docs/01 D6, docs/12 §3).

v1 source is the HUD page (``browser_source``: ``astra.frame`` over the WebSocket, ≤ 1 fps);
the native portal/PipeWire/GStreamer daemon (M2+) implements the same ``PerceptionClient``
and shares ``tests/contract/test_perception_contract.py``.
"""
