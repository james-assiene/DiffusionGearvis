"""Constrained MAP by max-plus over the same dyadic tree shape (T-0066; SPEC §2.7).

``x̂ = argmax_{x ∈ C} Σ_{i<L_eff} log p_i(x_i)``. Path multiplicity cannot change a max over strings,
so this is exact on NFAs too, and it is temperature-invariant: ``log softmax(ℓ_i/T)`` differs from
``ℓ_i/T`` by a per-position constant.

**Its own sweep.** The MAP does not share the sum-product tree: it builds a max-plus up-sweep of its
own (``ceil(log2 L)`` batched ``max(a + b)`` combines, the retained nodes the same ``2L − 1`` as the
sum tree) and descends it with argmax instead of a draw (``draw.descend``). No backtrace table is
stored: the tree nodes are the only ``S²``-sized state, as in the sum path.

**Leaves in regret form.** Leaf values are ``max_{v ∈ class} (ℓ_i(v) − max_v ℓ_i(v))`` — logits
minus the row max, each a single rounding of an input difference, ≤ 0, and exactly 0 at the
per-position argmax. So when the per-position argmax ``x*`` is in C its path value is exactly 0
(sums of zeros) and every other path is strictly below it unless it ties at the argmax: the
projection identity (criterion 5) holds in fp32 without a margin. Max-plus needs no normalisation:
only additions of non-positive finite values, the sentinel clamped after every combine.

**Token recovery** (the ``(s, s') → class`` step): once the descent fixes ``(s_i, s_{i+1})``, the
edge realising the leaf max is re-derived per position as the argmax over edges ``s_i → s_{i+1}`` of
the class max (``O(L·E)``), and the token is that class's per-position argmax — the same values the
leaf was built from, so the recovered string attains the tree's optimum exactly. Ties break
deterministically (lowest token id within a class; ``torch.argmax``'s first index elsewhere).

The score is ``Σ_{i<L_eff} log softmax(logits_i)[x̂_i]`` of the recovered string. No randomness; no
host synchronisation.

**Live states (T-0077a).** The leaves, the max-plus sweep, the boundary pair and the descent run on the
real states ``0..S−1`` (``S = tables.n_states``), never on the padded bucket: a padded state has no
edge, so in the bucket form it is the sentinel everywhere except the identity tail's diagonal, which
the masks (padded states are neither start nor final) and the midpoint sums keep below every live
entry. Max-plus is exact under any grouping, and row-major order over ``[S, S]`` is the order of the
live pairs over ``[B, B]``, so every ``argmax`` (first index on ties) picks the same state: tokens,
score and ``empty`` are bit-identical to the bucket path.
"""

from __future__ import annotations

import dataclasses

import torch

from .draw import _mask_to_log, descend
from .emission import CdTables
from .semiring import CHUNK_ELEMS, DEAD, SENTINEL


@dataclasses.dataclass(frozen=True)
class CdMap:
    tokens: torch.Tensor  # int64 [L]; positions ≥ l_eff unspecified (the per-position argmax)
    score: torch.Tensor  # 0-d: Σ_{i<l_eff} log softmax(logits_i)[tokens_i]; sentinel when empty
    empty: torch.Tensor  # 0-d bool: the constrained support is empty


#: Elements of one ``[g, r, m, c]`` max-plus temporary (T-0072 note 4): 4× ``semiring.CHUNK_ELEMS``
#: = 2^24 elements, 64 MiB in fp32 (the only large temporary live per chunk), so a level of the MAP
#: sweep at bucket 128 is a few chunks rather than dozens. Max-plus is exact under any grouping, and
#: every chunk still reduces the whole ``m`` axis in one ``amax``, so the output is bit-identical to
#: the ``CHUNK_ELEMS`` grouping.
MAP_CHUNK_ELEMS = CHUNK_ELEMS << 2


def max_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Max-plus product ``[p, n, m] ⊗ [p, m, k] -> [p, n, k]``, chunked like ``log_matmul``."""
    nb, n, m = a.shape
    k = b.shape[-1]
    out = torch.empty((nb, n, k), dtype=a.dtype, device=a.device)
    kc = min(k, max(1, MAP_CHUNK_ELEMS // max(1, m)))
    rc = min(n, max(1, MAP_CHUNK_ELEMS // max(1, m * kc)))
    bg = max(1, MAP_CHUNK_ELEMS // max(1, n * m * kc)) if rc == n else 1
    for b0 in range(0, nb, bg):
        b1 = min(nb, b0 + bg)
        for r0 in range(0, n, rc):
            r1 = min(n, r0 + rc)
            lhs = a[b0:b1, r0:r1, :].unsqueeze(-1)  # [g, r, m, 1]
            for c0 in range(0, k, kc):
                c1 = min(k, c0 + kc)
                # [g, r, m, c], reduced in the same statement: one such temporary is live at a time
                out[b0:b1, r0:r1, c0:c1] = (lhs + b[b0:b1, :, c0:c1].unsqueeze(-3)).amax(dim=-2)
    return out.clamp_min_(SENTINEL)


def max_up_sweep(leaves: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Retained max-plus dyadic tree, same shape and odd-carry rule as ``tree.up_sweep``."""
    cur = leaves
    levels = [cur]
    while cur.shape[0] > 1:
        n = cur.shape[0]
        half = n // 2
        prod = max_matmul(cur[0 : 2 * half : 2], cur[1 : 2 * half : 2])
        if n % 2:
            prod = torch.cat([prod, cur[n - 1 :]], dim=0)
        cur = prod
        levels.append(cur)
    return tuple(levels)


def class_max(
    reg: torch.Tensor, tables: CdTables, row_arg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(max, argmax)`` ``[L, C]`` of ``reg`` over each class's true member set.

    Listed members inside ``U`` by a scatter-max, the argmax as the lowest token attaining it; for
    negated classes also ``V \\ U`` (one masked row max). Dead classes: sentinel, token 0.
    """
    L, V = reg.shape
    C = tables.n_classes
    device = reg.device
    mx = torch.full((L, C), SENTINEL, dtype=reg.dtype, device=device)
    nnz = tables.memb_token.shape[0]
    kc = max(1, CHUNK_ELEMS // max(1, L))
    for j0 in range(0, nnz, kc):
        j1 = min(nnz, j0 + kc)
        idx = tables.memb_class[j0:j1].unsqueeze(0).expand(L, j1 - j0)
        vals = reg.index_select(1, tables.memb_token[j0:j1])
        mx.scatter_reduce_(1, idx, vals, reduce="amax", include_self=True)
    arg = torch.full((L, C), V, dtype=torch.int64, device=device)
    for j0 in range(0, nnz, kc):
        j1 = min(nnz, j0 + kc)
        cls = tables.memb_class[j0:j1]
        tok = tables.memb_token[j0:j1]
        vals = reg.index_select(1, tok)
        hit = (vals == mx.index_select(1, cls)) & (vals > DEAD)
        cand = torch.where(
            hit, tok.unsqueeze(0).expand(L, j1 - j0), torch.full_like(vals, V, dtype=torch.int64)
        )
        arg.scatter_reduce_(
            1, cls.unsqueeze(0).expand(L, j1 - j0), cand, reduce="amin", include_self=True
        )
    if tables.has_outside:
        outside = torch.where(tables.in_union.unsqueeze(0), torch.full_like(reg, SENTINEL), reg)
        out_mx, out_arg = outside.max(dim=1)
        neg = tables.class_neg.unsqueeze(0)
        take_out = neg & (out_mx.unsqueeze(1) > mx)
        mx = torch.where(take_out, out_mx.unsqueeze(1), mx)
        arg = torch.where(take_out, out_arg.unsqueeze(1), arg)
    live = mx > DEAD
    mx = torch.where(live, mx, torch.full_like(mx, SENTINEL))
    arg = torch.where(live & (arg < V), arg, row_arg.unsqueeze(1).expand(L, C))
    return mx, arg


def _row_argmax(logits: torch.Tensor) -> torch.Tensor:
    """``[L]``: the per-position argmax over the live logits (first index on ties)."""
    live_logit = logits > DEAD
    return torch.where(live_logit, logits, torch.full_like(logits, SENTINEL)).max(dim=1)[1]


def class_max_of(logits: torch.Tensor, tables: CdTables) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(max, argmax)`` ``[L, C]`` pair ``constrained_map`` builds its leaves from (T-0072).

    Regret-form logits (``ℓ − max ℓ`` over live entries) through ``class_max``, which is called
    through this module's global so a caller can observe or replace it.
    """
    return _class_max_and_row_argmax(logits, tables)[0]


def _class_max_and_row_argmax(
    logits: torch.Tensor, tables: CdTables
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    live_logit = logits > DEAD
    row_max, row_arg = torch.where(live_logit, logits, torch.full_like(logits, SENTINEL)).max(dim=1)
    reg = torch.where(live_logit, logits - row_max.unsqueeze(1), torch.full_like(logits, SENTINEL))
    return class_max(reg, tables, row_arg), row_arg


def constrained_map(
    logits: torch.Tensor,
    tables: CdTables,
    a_start: torch.Tensor,
    b_final: torch.Tensor,
    l_eff: int,
    *,
    class_max: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> CdMap:
    """Exact constrained MAP (ruling R1); see the module docstring.

    ``class_max`` (T-0072) is the ``(max, argmax)`` pair ``class_max_of(logits, tables)`` returns for
    these logits; a caller running several MAPs of one step computes it once and passes it here.
    """
    L = logits.shape[0]
    S = tables.n_states  # T-0077a: the live states; the bucket's padding is never built
    device = logits.device
    l_eff = max(0, min(int(l_eff), L))
    if class_max is None:
        (wmax, warg), row_arg = _class_max_and_row_argmax(logits, tables)  # [L, C], [L]
    else:
        (wmax, warg), row_arg = class_max, _row_argmax(logits)

    # leaves: max over the edges of each (src, dst) pair; positions ≥ L_eff are the identity
    ew = wmax.index_select(1, tables.edge_class)  # [L, E]
    pw = torch.full((L, tables.n_pairs), SENTINEL, dtype=logits.dtype, device=device)
    pw.scatter_reduce_(
        1, tables.edge_pair.unsqueeze(0).expand(L, -1), ew, reduce="amax", include_self=True
    )
    leaves = torch.full((L, S * S), SENTINEL, dtype=logits.dtype, device=device)
    leaves.index_copy_(1, tables.pair_flat_live, pw)
    leaves = leaves.view(L, S, S)
    if l_eff < L:
        tail = leaves[l_eff:]
        tail.fill_(SENTINEL)
        tail.diagonal(dim1=-2, dim2=-1).fill_(0.0)
    levels = max_up_sweep(leaves)

    root = levels[-1][0]
    top = (
        _mask_to_log(a_start[:S], root).unsqueeze(1)
        + root
        + _mask_to_log(b_final[:S], root).unsqueeze(0)
    )
    top = top.clamp_min(SENTINEL).reshape(S * S)
    empty = top.amax() <= DEAD
    flat = torch.argmax(top)
    states = descend(levels, flat // S, flat % S, lambda x: torch.argmax(x, dim=-1))

    # token recovery: the edge s_i → s_{i+1} attaining the leaf max, then that class's argmax token
    on = (tables.edge_src.unsqueeze(0) == states[:-1].unsqueeze(1)) & (
        tables.edge_dst.unsqueeze(0) == states[1:].unsqueeze(1)
    )
    edge = torch.argmax(torch.where(on, ew, torch.full_like(ew, 2.0 * SENTINEL)), dim=1)  # [L]
    cls = tables.edge_class.index_select(0, edge)
    tok = torch.gather(warg, 1, cls.unsqueeze(1)).squeeze(1)
    scored = torch.arange(L, device=device) < l_eff
    tokens = torch.where(scored, tok, row_arg)

    lse = torch.logsumexp(logits, dim=1)
    picked = torch.gather(logits, 1, tokens.unsqueeze(1)).squeeze(1) - lse
    score = torch.where(scored, picked, torch.zeros_like(picked)).sum()
    score = torch.where(empty, torch.full_like(score, SENTINEL), score)
    return CdMap(tokens=tokens, score=score, empty=empty)
