"""Constrained-decoding evaluation harness (T-0070): BFCL-Live simple and Sudoku 4x4 over the loopback
vLLM server, five arms, diffgemma-fa's scorers, host replay, statistics, coded hard stops, calibration.

Needs the ``cd`` dependency group (tokenizers, outlines-core for the host compiler); never imports torch.
Running it against the real engine is GPU time: ``scripts/gpu/cd_eval.py`` goes through ``/launch``.
Import the submodules (``cuts``, ``prompts``, ``runner``, ``stats``, ``report``, ``calibration``).
"""
