"""Joint draw of a canvas from the automaton-constrained posterior (T-0066; SPEC §2.6, Algorithm 1).

``P(x) ∝ N(x)·Π_{i<L_eff} p_i(x_i)`` over the strings of length ``L_eff`` read from ``A`` into ``B``,
drawn top-down over T-0065's retained tree:

1. **Boundary pair.** ``(s_0, s_L) ∝ a_start(s_0)·P_{[0,L)}(s_0, s_L)·b_final(s_L)`` from the root.
   Positions ``≥ L_eff`` are identity leaves, so ``s_L = s_{L_eff}``.
2. **Midpoint recursion** (eq. 7), one batched draw per tree level:
   ``P(s_m | s_ℓ, s_r) ∝ P_{[ℓ,m)}(s_ℓ, s_m)·P_{[m,r)}(s_m, s_r)``. An odd level's carried node keeps
   its boundaries.
3. **Tokens** (eq. 8, multiplicity-weighted, never the ``∃`` form): per position an edge
   ``e ∝ 1[src = s_i, dst = s_{i+1}]·W_i(class e)`` (parallel edges each count), then
   ``x_i ∝ p_i(v)·1[v ∈ label(e)]``. Positions ``≥ L_eff`` are drawn from ``p_i`` unconstrained.

**Which tree.** The caller's tree is the *ungauged* pass-1 tree (T-0062 decision 10). Every
conditional above is a ratio inside one node (or one pair of sibling nodes), so the per-node log
scales cancel and only the normalised node entries are read, in log space: a categorical over
entries hundreds of nats apart is a Gumbel-max over their logs, never an exponentiation of the
raw values. fp32 stores those entries to ~ulp(|entry|) (6e-5 at 850 nats), a relative error in a
conditional probability far below what 20 000 draws resolve; the zero pattern (which pairs are
reachable) is exact because dead entries are the finite sentinel. So the pass-1 tree is used as is.

**Randomness.** Every random number comes from ``torch.rand(..., generator=generator)``; the global
RNGs are never touched. Small categoricals are Gumbel-max; the token draw over ``V`` is an inverse
CDF in two levels (block sums, then inside the chosen block) with ``searchsorted(right=True)``, which
can only land on a bin whose weight is strictly positive — a zero-weight (forbidden or filtered)
token is never drawn, whatever the rounding.

Per-step path: no host synchronisation (no ``.item()``, no boolean-mask indexing, no ``nonzero``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import torch

from .emission import CdTables
from .emission import class_log_weights as _class_log_weights
from .semiring import CHUNK_ELEMS, DEAD, SENTINEL
from .tree import LogTree

#: Width of the first level of the two-level inverse CDF over the vocabulary.
_BLOCK = 1024


@dataclasses.dataclass(frozen=True)
class CdDraw:
    tokens: torch.Tensor  # int64 [L]
    states: torch.Tensor  # int64 [L+1]: states[i] is the state before position i
    empty: torch.Tensor  # 0-d bool: the constrained support is empty (Z = 0)


def _mask_to_log(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=like.dtype, device=like.device)
    sent = torch.full((), SENTINEL, dtype=like.dtype, device=like.device)
    return torch.where(mask.to(device=like.device, dtype=torch.bool), zero, sent)


def gumbel_argmax(logw: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Index over the last axis drawn ``∝ exp(logw)`` (Gumbel-max); dead entries never win a live one."""
    u = torch.rand(logw.shape, generator=generator, dtype=logw.dtype, device=logw.device)
    tiny = torch.finfo(logw.dtype).tiny
    g = -torch.log(-torch.log(u.clamp_min(tiny)))
    return torch.argmax(logw.clamp_min(SENTINEL) + g, dim=-1)


def descend(
    levels: tuple[torch.Tensor, ...],
    s_first: torch.Tensor,
    s_last: torch.Tensor,
    choose: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Top-down midpoint recursion over a retained dyadic tree; returns the ``[L+1]`` boundary states.

    ``choose`` maps ``[h, S]`` log scores (``left[s_ℓ, ·] + right[·, s_r]``) to ``[h]`` indices:
    a Gumbel-max for the draw, an argmax for the MAP. The sum ``left + right`` is formed in the
    same order as the tree's combine, so an argmax reproduces the parent's max-plus value exactly.
    """
    bd = torch.stack([s_first, s_last]).to(torch.int64)  # boundaries of the root's single node
    for k in range(len(levels) - 2, -1, -1):
        nodes = levels[k]
        n = nodes.shape[0]
        half = n // 2
        new = torch.empty((n + 1,), dtype=torch.int64, device=bd.device)
        new[0 : 2 * half + 1 : 2] = bd[: half + 1]
        if n % 2:
            new[n] = bd[half + 1]
        if half:
            ar = torch.arange(half, device=bd.device)
            left = nodes[0 : 2 * half : 2][ar, bd[:half]]  # [h, S]: P_left(s_ℓ, ·)
            right = nodes[1 : 2 * half : 2][ar, :, bd[1 : half + 1]]  # [h, S]: P_right(·, s_r)
            new[1 : 2 * half : 2] = choose(left + right)
        bd = new
    return bd


def _allowed(
    tables: CdTables, cls: torch.Tensor, free: torch.Tensor, vocab: int, device: torch.device
) -> torch.Tensor:
    """bool ``[L, V]``: ``v ∈ label class cls[i]`` (true member set), or every ``v`` where ``free``."""
    L = cls.shape[0]
    n_u = tables.union_ids.shape[0]
    hits = torch.zeros((L, n_u), dtype=torch.int32, device=device)
    nnz = tables.memb_class.shape[0]
    kc = max(1, CHUNK_ELEMS // max(1, L))
    for j0 in range(0, nnz, kc):
        j1 = min(nnz, j0 + kc)
        on = (tables.memb_class[j0:j1].unsqueeze(0) == cls.unsqueeze(1)).to(torch.int32)
        hits.index_add_(1, tables.memb_upos[j0:j1], on)
    neg = tables.class_neg.index_select(0, cls)  # [L]
    allowed = (~tables.in_union).unsqueeze(0) & neg.unsqueeze(1)  # V \ U: every negated class
    allowed = allowed.expand(L, vocab).clone()
    allowed.index_copy_(1, tables.union_ids, hits > 0)
    return allowed | free.unsqueeze(1)


def inverse_cdf(w: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """``[L]`` indices drawn ``∝ w`` (``[L, V]``, non-negative) by a two-level inverse CDF.

    ``searchsorted(right=True)`` returns the first bin whose cumulative sum exceeds the target, so
    the chosen bin's weight is strictly positive (``fl(c + 0) = c``); an all-zero row returns
    ``V − 1`` (callers flag that case separately).
    """
    L, V = w.shape
    nb = -(-V // _BLOCK)
    pad = nb * _BLOCK - V
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    blocks = w.view(L, nb, _BLOCK)
    u = torch.rand((L, 2), generator=generator, dtype=w.dtype, device=w.device)
    bcum = torch.cumsum(blocks.sum(dim=-1), dim=-1)  # [L, nb]
    t1 = (u[:, :1] * bcum[:, -1:]).contiguous()
    blk = torch.searchsorted(bcum.contiguous(), t1, right=True).clamp_max(nb - 1)  # [L, 1]
    inner = torch.gather(blocks, 1, blk.unsqueeze(-1).expand(L, 1, _BLOCK)).squeeze(1)
    icum = torch.cumsum(inner, dim=-1)
    t2 = (u[:, 1:] * icum[:, -1:]).contiguous()
    pos = torch.searchsorted(icum.contiguous(), t2, right=True).clamp_max(_BLOCK - 1)
    return (blk * _BLOCK + pos).squeeze(1).clamp_max(V - 1)


def joint_draw(
    logits: torch.Tensor,
    tables: CdTables,
    tree: LogTree,
    a_start: torch.Tensor,
    b_final: torch.Tensor,
    l_eff: int,
    *,
    generator: torch.Generator,
    class_log_weights: torch.Tensor | None = None,
) -> CdDraw:
    """One canvas drawn jointly from the constrained posterior (ruling R1); see the module docstring.

    ``class_log_weights`` (T-0072) is the ``emission.class_log_weights(logits, tables)`` the caller
    already holds; it is used as is, not recomputed.
    """
    L, V = logits.shape
    device = logits.device
    l_eff = max(0, min(int(l_eff), L))
    root = tree.levels[-1][0]
    S = root.shape[-1]

    # 1. boundary pair from the root
    pair = (
        _mask_to_log(a_start, root).unsqueeze(1) + root + _mask_to_log(b_final, root).unsqueeze(0)
    )
    pair = pair.clamp_min(SENTINEL).reshape(S * S)
    empty = pair.amax() <= DEAD
    flat = gumbel_argmax(pair, generator)
    # 2. midpoint recursion
    states = descend(
        tree.levels, flat // S, flat % S, lambda x: gumbel_argmax(x.clamp_min(SENTINEL), generator)
    )

    # 3a. edge per position ∝ W_i(class e) among the edges s_i → s_{i+1} (parallel edges each count)
    cw = _class_log_weights(logits, tables) if class_log_weights is None else class_log_weights
    ew = cw.index_select(1, tables.edge_class)  # [L, E]
    on = (tables.edge_src.unsqueeze(0) == states[:-1].unsqueeze(1)) & (
        tables.edge_dst.unsqueeze(0) == states[1:].unsqueeze(1)
    )
    edge = gumbel_argmax(torch.where(on, ew, torch.full_like(ew, SENTINEL)), generator)  # [L]
    cls = tables.edge_class.index_select(0, edge)
    # 3b. token ∝ p_i(v)·1[v ∈ label(e_i)]; positions ≥ L_eff from p_i unconstrained
    free = torch.arange(L, device=device) >= l_eff
    allowed = _allowed(tables, cls, free, V, device) & (logits > DEAD)
    x = torch.where(allowed, logits, torch.full_like(logits, SENTINEL))
    m = x.amax(dim=1, keepdim=True)
    w = torch.where(allowed, torch.exp(x - m), torch.zeros_like(x))
    tokens = inverse_cdf(w, generator)
    return CdDraw(tokens=tokens, states=states, empty=empty)
