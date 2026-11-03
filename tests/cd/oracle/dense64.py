"""float64 numpy forward–backward, longhand, for production shapes (SPEC eq (1)–(6)). See MANIFEST.md.

Class masses are direct sums over each class's **true** member set (a boolean mask over V), so a
negated class is never formed by subtraction. Forward/backward run in log space position by
position; ``q`` is anchored on each position's dominant edge.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from tests.cd.oracle.enumerate import Toy


@dataclasses.dataclass(frozen=True)
class Dense64:
    log_z: float
    log_a: np.ndarray  # [L+1, S]
    log_b: np.ndarray  # [L+1, S]
    q: np.ndarray  # [Leff, V]
    entropy: np.ndarray  # [Leff]
    logp: np.ndarray  # [L, V]


def _lse(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(x - safe), axis=axis, keepdims=True)) + safe
    return np.squeeze(np.where(np.isfinite(m), out, -np.inf), axis=axis)


def log_softmax64(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    return x - _lse(x, axis=-1)[..., None]


def member_masks(toy: Toy) -> np.ndarray:
    """``[C, V] bool``: true membership (complement already applied for negated classes)."""
    C, V = len(toy.classes), toy.vocab_size
    mask = np.zeros((C, V), dtype=bool)
    for c, (stored, neg) in enumerate(toy.classes):
        idx = np.fromiter(sorted(stored), dtype=np.int64, count=len(stored))
        if neg:
            mask[c, :] = True
            mask[c, idx] = False
        else:
            mask[c, idx] = True
    return mask


def forward_backward64(
    logits: np.ndarray,
    toy: Toy,
    start: np.ndarray,
    end: np.ndarray,
    l_eff: int,
) -> Dense64:
    logp = log_softmax64(logits)
    L, V = logp.shape
    S = toy.n_states
    mask = member_masks(toy)
    # log class mass: logsumexp over true members (direct, non-subtractive)
    C = mask.shape[0]
    logw = np.full((L, C), -np.inf)
    for c in range(C):
        if mask[c].any():
            logw[:, c] = _lse(logp[:, mask[c]], axis=1)
    logM = np.full((L, S, S), -np.inf)
    ident = np.full((S, S), -np.inf)
    np.fill_diagonal(ident, 0.0)
    for i in range(L):
        if i >= l_eff:
            logM[i] = ident
            continue
        for s, d, c in toy.edges:
            logM[i, s, d] = np.logaddexp(logM[i, s, d], logw[i, c])
    with np.errstate(divide="ignore"):
        la = np.full((L + 1, S), -np.inf)
        lb = np.full((L + 1, S), -np.inf)
        la[0] = np.log(start.astype(np.float64))
        for i in range(L):
            la[i + 1] = _lse(la[i][:, None] + logM[i], axis=0)
        lb[L] = np.log(end.astype(np.float64))
        for i in range(L - 1, -1, -1):
            lb[i] = _lse(logM[i] + lb[i + 1][None, :], axis=1)
    log_z = float(_lse(la[0] + lb[0], axis=0))
    q = np.zeros((l_eff, V))
    ent = np.zeros(l_eff)
    src = np.array([e[0] for e in toy.edges])
    dst = np.array([e[1] for e in toy.edges])
    cls = np.array([e[2] for e in toy.edges])
    for i in range(l_eff):
        log_u = la[i][src] + lb[i + 1][dst]
        anchor = np.max(log_u)
        u = np.exp(log_u - anchor)
        U = np.zeros(C)
        np.add.at(U, cls, u)
        r = U @ mask  # [V], a sum of non-negative terms
        with np.errstate(divide="ignore"):
            lrow = logp[i] + np.log(r)
        lrow = lrow - _lse(lrow, axis=0)
        row = np.exp(lrow)
        q[i] = row
        pos = row > 0
        ent[i] = -np.sum(row[pos] * lrow[pos])
    return Dense64(log_z=log_z, log_a=la, log_b=lb, q=q, entropy=ent, logp=logp)
