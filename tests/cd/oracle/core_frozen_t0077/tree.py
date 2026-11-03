"""Log-semiring up-sweep retaining the aligned dyadic nodes (T-0065; SPEC §2.6).

``levels[0]`` are the leaves ``log M_i`` (``[L, S, S]``); ``levels[k+1][j] = levels[k][2j] ⊗
levels[k][2j+1]`` in the log semiring, so ``levels[k][j]`` is the transfer over positions
``[j·2^k, (j+1)·2^k)``. An odd level carries its last node up unchanged (a product with the
identity), so ``L`` need not be a power of two; at a power of two the retained nodes number exactly
``2L − 1``.

Every retained node is **normalised** (ruling R4): stored shifted by the integer ``ceil(max)`` so its
largest entry lies in (−1, 0], with the shift kept in ``log_scales`` (``[n_k]`` per level). The true
node is ``levels[k][j] + log_scales[k][j]``. Values therefore stay near 0 whatever the canvas length,
and the per-node scales — scalars that cancel in every conditional — carry the magnitude. An
integer shift makes the subtraction exact wherever it matters and the scale sums exact. A node with
no live entry keeps scale 0 and stays at the sentinel.
"""

from __future__ import annotations

import dataclasses

import torch

from .semiring import log_matmul, normalise_last2


@dataclasses.dataclass(frozen=True)
class LogTree:
    #: ``levels[0]`` is ``[L, S, S]``; each next level is ``ceil(n/2)`` long; the last is ``[1, S, S]``.
    levels: tuple[torch.Tensor, ...]
    #: ``log_scales[k]`` is ``[n_k]``: the true node is ``levels[k][j] + log_scales[k][j]``.
    log_scales: tuple[torch.Tensor, ...]

    @property
    def root(self) -> torch.Tensor:
        return self.levels[-1][0]

    @property
    def root_log_scale(self) -> torch.Tensor:
        return self.log_scales[-1][0]


def up_sweep(log_m: torch.Tensor, *, compensated: bool = False) -> LogTree:
    """Blelloch up-sweep in the log semiring: ``ceil(log2 L)`` batched combines, all nodes kept.

    ``compensated`` selects ``log_matmul(..., compensated=True)`` (used by the gauged pass 2 of
    ``marginals.constrained_marginals``; 2× the elementwise work).
    """
    if log_m.dim() != 3 or log_m.shape[-1] != log_m.shape[-2]:
        raise ValueError(f"up_sweep expects [L, S, S], got {tuple(log_m.shape)}")
    if log_m.shape[0] == 0:
        raise ValueError("up_sweep needs at least one position")
    cur, scale = normalise_last2(log_m)
    levels = [cur]
    scales = [scale]
    while cur.shape[0] > 1:
        n = cur.shape[0]
        half = n // 2
        prod = log_matmul(cur[0 : 2 * half : 2], cur[1 : 2 * half : 2], compensated=compensated)
        prod, m = normalise_last2(prod)
        psc = scale[0 : 2 * half : 2] + scale[1 : 2 * half : 2] + m
        if n % 2:
            prod = torch.cat([prod, cur[n - 1 :]], dim=0)
            psc = torch.cat([psc, scale[n - 1 :]], dim=0)
        cur, scale = prod, psc
        levels.append(cur)
        scales.append(scale)
    return LogTree(levels=tuple(levels), log_scales=tuple(scales))
