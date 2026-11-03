"""float64 max-plus (Viterbi) oracle for the constrained MAP score at production shape (SPEC §2.7 eq. 9).

``score* = max_{x ∈ C} Σ_{i < Leff} log softmax(logits_i)[x_i]``. Class maxima are taken over each class's
**true** member set (``dense64.member_masks``, complement applied, never a subtraction); the forward pass
is longhand, edge by edge, in float64. Written by the T-0066 test-author from the spec; see MANIFEST.md.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from tests.cd.oracle.dense64 import log_softmax64, member_masks
from tests.cd.oracle.enumerate import Toy


def map_score64(
    logits: np.ndarray, toy: Toy, start: Iterable[int], end: Iterable[int], l_eff: int
) -> float:
    logp = log_softmax64(logits)
    mask = member_masks(toy)
    C = mask.shape[0]
    wmax = np.full((l_eff, C), -np.inf)
    for c in range(C):
        if mask[c].any():
            wmax[:, c] = np.max(logp[:l_eff][:, mask[c]], axis=1)
    la = np.full(toy.n_states, -np.inf)
    la[list(start)] = 0.0
    for i in range(l_eff):
        nxt = np.full(toy.n_states, -np.inf)
        for s, d, c in toy.edges:
            nxt[d] = max(nxt[d], la[s] + wmax[i, c])
        la = nxt
    return float(np.max(la[list(end)]))


def string_score64(logits: np.ndarray, tokens: Iterable[int], l_eff: int) -> float:
    """``Σ_{i < Leff} log softmax(logits_i)[x_i]`` in float64 (the score of a returned MAP string)."""
    logp = log_softmax64(np.asarray(logits)[:l_eff])
    toks = list(tokens)[:l_eff]
    return float(np.sum(logp[np.arange(l_eff), toks]))
