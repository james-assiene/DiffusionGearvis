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

**Live states (T-0077a).** The automaton's real states are ``0..S−1`` (``S = n_states``) and the
bucket's padded states ``S..B−1`` are dead: they have no edge, so in ``log M_i`` their rows and
columns are the sentinel, except that an identity leaf (a position ``≥ l_eff``) has 0 on its whole
diagonal. Every product therefore runs on the ``[·, S, S]`` live block; the retained levels are padded
back to ``[n, B, B]`` with exactly what the bucket product would have stored there (the sentinel, and
0 on the padded diagonal of a node that covers only identity positions — a product of identities is
exactly the identity). Nothing reads the padded entries. ``S`` comes from the ``n_states`` keyword,
or from the tag ``emission.log_transfer`` puts on its output (``live_states_of``); neither = the
bucket, today's behaviour.
"""

from __future__ import annotations

import dataclasses

import torch

from .semiring import SENTINEL, _integer_shift, log_matmul, normalise_last2

#: Attribute ``emission.log_transfer`` sets on its ``[L, B, B]`` output: ``(n_states, l_eff)``.
LIVE_STATES_ATTR = "_astra_cd_live_states"


@dataclasses.dataclass(frozen=True)
class LogTree:
    #: ``levels[0]`` is ``[L, S, S]``; each next level is ``ceil(n/2)`` long; the last is ``[1, S, S]``.
    levels: tuple[torch.Tensor, ...]
    #: ``log_scales[k]`` is ``[n_k]``: the true node is ``levels[k][j] + log_scales[k][j]``.
    log_scales: tuple[torch.Tensor, ...]
    #: T-0077a: the live states ``0..n_states−1`` the products ran on (``None``: the whole axis).
    n_states: int | None = None

    @property
    def root(self) -> torch.Tensor:
        return self.levels[-1][0]

    @property
    def root_log_scale(self) -> torch.Tensor:
        return self.log_scales[-1][0]


def tag_live_states(log_m: torch.Tensor, n_states: int, l_eff: int) -> torch.Tensor:
    """Mark ``log_m`` (``[L, B, B]``) as having live states ``0..n_states−1`` and identity leaves at
    positions ``≥ l_eff``; returns ``log_m`` itself. Host ints only."""
    setattr(log_m, LIVE_STATES_ATTR, (int(n_states), int(l_eff)))
    return log_m


def live_states_of(log_m: torch.Tensor) -> tuple[int, int] | None:
    """The ``(n_states, l_eff)`` tag of ``tag_live_states``, or ``None`` (an untagged tensor)."""
    tag = getattr(log_m, LIVE_STATES_ATTR, None)
    if tag is None:
        return None
    n_states, l_eff = tag
    return int(n_states), int(l_eff)


def _new_level(
    n: int, bucket: int, S: int, span: int, l_eff: int, like: torch.Tensor
) -> torch.Tensor:
    """``[n, B, B]`` holding what the bucket product stores outside the live block: the sentinel, and 0 on
    the padded diagonal of every node ``j`` covering only identity positions (``j·span ≥ l_eff``)."""
    out = torch.full((n, bucket, bucket), SENTINEL, dtype=like.dtype, device=like.device)
    j0 = -(-l_eff // span)  # the first node whose positions all lie at or after l_eff
    if j0 < n:
        out[j0:, S:, S:].diagonal(dim1=-2, dim2=-1).fill_(0.0)
    return out


def _normalised_into(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``normalise_last2(x)`` written into ``out`` (a ``[n, S, S]`` view); returns the shift. The same ops,
    so the same bits; no ``[n, S, S]`` copy is kept besides the padded level."""
    shift = _integer_shift(x.amax(dim=(-2, -1)))
    torch.sub(x, shift[..., None, None], out=out)
    return shift


def up_sweep(
    log_m: torch.Tensor, *, compensated: bool = False, n_states: int | None = None
) -> LogTree:
    """Blelloch up-sweep in the log semiring: ``ceil(log2 L)`` batched combines, all nodes kept.

    ``compensated`` selects ``log_matmul(..., compensated=True)`` (used by the gauged pass 2 of
    ``marginals.constrained_marginals``; 2× the elementwise work).

    ``n_states`` (T-0077a): the products run on the live block ``[·, S, S]`` and the levels are padded
    back to ``log_m``'s ``[·, B, B]`` (see the module docstring). Default: the tag of
    ``emission.log_transfer``'s output if present, else the whole axis (the bucket).
    """
    if log_m.dim() != 3 or log_m.shape[-1] != log_m.shape[-2]:
        raise ValueError(f"up_sweep expects [L, S, S], got {tuple(log_m.shape)}")
    if log_m.shape[0] == 0:
        raise ValueError("up_sweep needs at least one position")
    L, B = log_m.shape[0], log_m.shape[-1]
    tag = live_states_of(log_m)
    S = n_states if n_states is not None else (tag[0] if tag is not None else B)
    # identity leaves from the tag; an explicit n_states on an untagged tensor assumes none
    l_eff = tag[1] if tag is not None else L
    if not 1 <= S <= B:
        raise ValueError(f"up_sweep: n_states {S} outside [1, {B}]")
    if S == B:
        return _up_sweep_dense(log_m, compensated)
    span = 1
    cur = _new_level(L, B, S, span, l_eff, log_m)
    scale = _normalised_into(log_m[:, :S, :S], cur[:, :S, :S])
    levels = [cur]
    scales = [scale]
    while cur.shape[0] > 1:
        n = cur.shape[0]
        half = n // 2
        live = cur[:, :S, :S]
        prod = log_matmul(live[0 : 2 * half : 2], live[1 : 2 * half : 2], compensated=compensated)
        span *= 2
        nxt = _new_level(half + n % 2, B, S, span, l_eff, log_m)
        m = _normalised_into(prod, nxt[:half, :S, :S])
        del prod
        psc = scale[0 : 2 * half : 2] + scale[1 : 2 * half : 2] + m
        if n % 2:
            nxt[half] = cur[n - 1]  # the carried node, padded entries included
            psc = torch.cat([psc, scale[n - 1 :]], dim=0)
        cur, scale = nxt, psc
        levels.append(cur)
        scales.append(scale)
    return LogTree(levels=tuple(levels), log_scales=tuple(scales), n_states=S)


def _up_sweep_dense(log_m: torch.Tensor, compensated: bool) -> LogTree:
    """The whole state axis (S = bucket): the e73cfa7 up-sweep, unchanged."""
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
