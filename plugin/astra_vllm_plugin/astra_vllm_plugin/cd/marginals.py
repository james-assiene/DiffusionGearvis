"""Partition, forward/backward vectors, constrained marginals q and H(q) (T-0065; SPEC §2.4).

One constrained slot per call (ruling R3). Per-step path: no host synchronisation.

**Forward/backward by a down-sweep over the retained tree.** For a node covering ``[ℓ, r)`` with
parent ``[ℓ, r')``: the left child starts where its parent starts, the right child starts after the
left child, ``a(right) = a(parent) ⊗ P_left``; symmetrically ``b(left) = P_right ⊗ b(parent)``. Each
boundary vector is therefore ``log2 L`` vector–matrix products from ``a_start``/``b_final``, not
``L`` (fewer roundings), and every product is batched per level.

Vectors are carried normalised (integer shift, max in (−1, 0]) with a separate log-scale, like the
tree nodes; the absolute ``log_a``/``log_b`` are only assembled for output, and
``log(a_i·b_i)`` is ``i``-invariant.

**Marginals.** With ``u_i(e) = a_i(src e)·b_{i+1}(dst e)``, the posterior of class ``c`` at position
``i`` is ``G_i(c) ∝ Σ_{e∈c} u_i(e)·W_i(c)`` and ``q_i(v) = Σ_{c ∋ v} G_i(c)·p_i(v | c)`` (SPEC eq (6)
regrouped by class). The per-token sum runs over the classes that truly contain ``v``: listed
memberships inside ``U``, and every negated class for ``v ∉ U`` — a sum of non-negative terms,
never the subtractive ``Σ_Neg − Σ_{Neg, v∈N_c}`` form that cancels above ~69 nats. It is formed in
linear space as ``e^{x_v − anchor_c}·G_i(c)/sum_c`` (every exponent ≤ 0), so q keeps fp32's
relative precision. Forbidden tokens get ``q = 0`` exactly, and ``H = −Σ q·log(max(q, 1e-30))``
gives them exactly 0.

**Precision: two passes.** fp32 stores a log value ``y`` to ``ulp(|y|)``: 3.8e-6 at 33 nats,
6e-5 at 850. Tree entries and forward/backward values that matter for the posterior routinely sit
tens to thousands of nats from any scalar anchor (a chain automaton's ``P(0 → 9)`` is ~300 nats
below its self-loop; the unscored tail pins ``b`` at 1 while the grammar sits at ``Z``), so a
single fp32 pass loses ~1e-5 in q — measured 10–60× over criterion 2's bar. Pass 1 (the tree the
caller built) gives ``log Z``, ``log_a``/``log_b`` and, from ``log_b``, an integer gauge
``β_i(s) = round(log b_i(s))``. Pass 2 rebuilds the leaves as the gauged (Doob-transformed)
matrices ``log M_i(s, s') + β_{i+1}(s') − β_i(s)`` — rows are near-stochastic, so every entry that
carries posterior mass is within a few nats of 0 — directly from the split class masses (integer
parts added exactly, the fractional log-ratio last), sweeps that tree, and reads the class
posteriors from it. The gauge telescopes along any path, so it changes no posterior; being integer,
adding it is exact. Cost: a second up-sweep and down-sweep (2× the ``S³`` work of pass 1).
"""

from __future__ import annotations

import dataclasses
import math

import torch

from .emission import CdTables, ClassMass, class_log_weights_split, row_split
from .emission import class_mass as _class_mass
from .kernels import backend as _kernel_backend
from .kernels import entropy as _kernel_entropy
from .semiring import (  # noqa: F401  (segment_* kept as module names: T-0073's probe swaps them)
    DEAD,
    SENTINEL,
    compensated_sum,
    layout_logsumexp,
    layout_reduce_blocks,
    layout_reduce_upper,
    layout_sumexp_split,
    log_matmul,
    normalise_last,
    safe_log,
    segment_logsumexp,
    segment_sumexp_split,
)
from .tree import LogTree, up_sweep


@dataclasses.dataclass(frozen=True)
class CdMarginals:
    log_z: torch.Tensor  # 0-d
    q: torch.Tensor  # [L, V]
    entropy: torch.Tensor  # [L], nats; rows ≥ l_eff unspecified
    log_a: torch.Tensor  # [L+1, bucket]: absolute log forward vector before position i
    log_b: torch.Tensor  # [L+1, bucket]: absolute log backward vector at boundary i


def _mask_to_log(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros((), dtype=like.dtype, device=like.device)
    sent = torch.full((), SENTINEL, dtype=like.dtype, device=like.device)
    return torch.where(mask.to(device=like.device, dtype=torch.bool), zero, sent)


def _vec_mat(v: torch.Tensor, m: torch.Tensor, compensated: bool = False) -> torch.Tensor:
    """``[p, S] ⊗ [p, S, S] -> [p, S]``."""
    return log_matmul(v.unsqueeze(-2), m, compensated=compensated).squeeze(-2)


def _mat_vec(m: torch.Tensor, v: torch.Tensor, compensated: bool = False) -> torch.Tensor:
    """``[p, S, S] ⊗ [p, S] -> [p, S]``."""
    return log_matmul(m, v.unsqueeze(-1), compensated=compensated).squeeze(-1)


def _down_sweep(
    tree: LogTree,
    la0: torch.Tensor,
    lb_end: torch.Tensor,
    compensated: bool = False,
    *,
    n_states: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalised prefix/suffix vectors over the leaves.

    Returns ``(a, sa, b, sb)``: ``a[i] + sa[i]`` is log a before leaf ``i`` and ``b[i] + sb[i]``
    log b after leaf ``i`` (``[L, S]`` and ``[L]``).

    T-0077a: runs on the live states ``0..S−1`` only (``n_states``, default ``tree.n_states``, else
    the whole axis): the nodes and the boundary vectors are sliced (views), so the vectors are
    ``[L, S]``. Padded states are dead in every node a vector meets, so nothing is lost.
    """
    levels, scales = tree.levels, tree.log_scales
    S = _live(tree, n_states)
    a, sa = normalise_last(la0[:S].unsqueeze(0))
    b, sb = normalise_last(lb_end[:S].unsqueeze(0))
    for k in range(len(levels) - 2, -1, -1):
        nodes, nsc = levels[k][:, :S, :S], scales[k]
        n = nodes.shape[0]
        half = n // 2
        new_a = torch.empty((n, S), dtype=a.dtype, device=a.device)
        new_sa = torch.empty((n,), dtype=a.dtype, device=a.device)
        new_b = torch.empty((n, S), dtype=a.dtype, device=a.device)
        new_sb = torch.empty((n,), dtype=a.dtype, device=a.device)
        # left children (and a carried odd node) start where their parent starts
        new_a[0::2] = a
        new_sa[0::2] = sa
        # right children end where their parent ends
        new_b[1::2] = b[:half]
        new_sb[1::2] = sb[:half]
        if half:
            ra, rm = normalise_last(_vec_mat(a[:half], nodes[0 : 2 * half : 2], compensated))
            new_a[1::2] = ra
            new_sa[1::2] = sa[:half] + nsc[0 : 2 * half : 2] + rm
            lb, lm = normalise_last(_mat_vec(nodes[1 : 2 * half : 2], b[:half], compensated))
            new_b[0 : 2 * half : 2] = lb
            new_sb[0 : 2 * half : 2] = sb[:half] + nsc[1 : 2 * half : 2] + lm
        if n % 2:
            new_b[n - 1] = b[half]
            new_sb[n - 1] = sb[half]
        a, sa, b, sb = new_a, new_sa, new_b, new_sb
    return a, sa, b, sb


def _live(tree: LogTree, n_states: int | None) -> int:
    """The live-state count: ``n_states``, else ``tree.n_states``, else the tree's whole state axis."""
    if n_states is not None:
        return n_states
    if tree.n_states is not None:
        return tree.n_states
    return int(tree.levels[0].shape[-1])


def _pad_states(v: torch.Tensor, bucket: int) -> torch.Tensor:
    """``[..., S] -> [..., bucket]`` with the sentinel at the padded states (dead, as the bucket path)."""
    S = v.shape[-1]
    if bucket == S:
        return v
    out = torch.full((*v.shape[:-1], bucket), SENTINEL, dtype=v.dtype, device=v.device)
    out[..., :S] = v
    return out


def _gauged_edge_weights(
    cm: ClassMass, tables: CdTables, beta: torch.Tensor, viable: torch.Tensor, l_eff: int
) -> torch.Tensor:
    """``[L, E]``: ``log W_i(class e) + β_{i+1}(dst e) − β_i(src e)``, rounded only near 0.

    ``log W = big + k + small`` (``class_log_weights_split``): the integer gauge and ``k`` are added
    to ``big`` (an exact difference of logits) first, then the near-0 remainder ``small``, so an
    entry that ends up near 0 is accurate to fp32 resolution near 0. Edges into a state with
    ``b_{i+1} = 0`` and every edge at positions ``≥ l_eff`` are the sentinel (they carry nothing).
    """
    L = beta.shape[0] - 1
    cls = tables.edge_class
    big_c, k_c, small_c = class_log_weights_split(cm)  # [L, C]
    anchor = cm.class_anchor.index_select(1, cls)  # [L, E]
    big = big_c.index_select(1, cls)
    gauge = beta[1:].index_select(1, tables.edge_dst) - beta[:-1].index_select(1, tables.edge_src)
    w = (big + (gauge + k_c.index_select(1, cls))) + small_c.index_select(1, cls)
    live = (anchor > DEAD) & viable[1:].index_select(1, tables.edge_dst)
    if l_eff < L:
        pos = torch.arange(L, device=w.device).unsqueeze(1)
        live = live & (pos < l_eff)
    return torch.where(live, w, torch.full_like(w, SENTINEL))


def _gauged_leaves(
    w_edge: torch.Tensor, tables: CdTables, beta: torch.Tensor, viable: torch.Tensor, l_eff: int
) -> torch.Tensor:
    """``[L, S, S]`` gauged transfer matrices ``log M_i(s, s') + β_{i+1}(s') − β_i(s)``.

    Parallel edges add. Positions ``≥ l_eff`` are the gauged identity.

    T-0077a: built on the live states only (``S = tables.n_states``; ``beta``/``viable`` may be
    ``[L+1, S]`` or ``[L+1, bucket]``): a padded state is not viable, so the bucket form's padded block
    is the sentinel throughout and pass 2 never reads it.
    """
    L = w_edge.shape[0]
    S = tables.n_states
    beta, viable = beta[:, :S], viable[:, :S]
    if tables.single_edge_pairs:
        pw = torch.empty((L, tables.n_pairs), dtype=w_edge.dtype, device=w_edge.device)
        pw.index_copy_(1, tables.edge_pair, w_edge)
    else:
        pw = layout_logsumexp(w_edge, tables.edge_pair_layout)
    leaves = torch.full((L, S * S), SENTINEL, dtype=w_edge.dtype, device=w_edge.device)
    leaves.index_copy_(1, tables.pair_flat_live, pw)
    leaves = leaves.view(L, S, S)
    if l_eff < L:
        diag = beta[l_eff + 1 :] - beta[l_eff:L]
        diag = torch.where(viable[l_eff + 1 :], diag, torch.full_like(diag, SENTINEL))
        leaves[l_eff:].diagonal(dim1=-2, dim2=-1).copy_(diag)
    return leaves


def _token_marginals(
    logits: torch.Tensor, tables: CdTables, cm: ClassMass, coef: torch.Tensor
) -> torch.Tensor:
    """``[L, V]`` unnormalised q: listed memberships inside ``U``, the negated classes outside it."""
    L = logits.shape[0]
    zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
    if tables.has_outside:
        # every negated class contains every v ∉ U: q(v) = e^{x_v − M_out}·Σ_neg coef_c·e^{M_out − anchor_c}
        out_m = cm.out_anchor.unsqueeze(1)
        k = torch.where(
            tables.class_neg.unsqueeze(0),
            coef * torch.exp((out_m - cm.class_anchor).clamp_max(0.0)),
            zero,
        ).sum(dim=1, keepdim=True)
        q = torch.exp((logits - out_m).clamp_max(0.0)) * k
    else:
        q = torch.zeros_like(logits)
    # per union token, the sum over its memberships in the fixed order of ``upos_layout`` (T-0073 D4)
    lay = tables.upos_layout
    if lay.width == 0:  # no true membership inside U: every token of U is forbidden
        q.index_fill_(1, tables.union_ids, 0.0)
        return q
    w = lay.block

    def terms(b0: int, b1: int) -> torch.Tensor:
        j0, j1 = b0 * w, b1 * w
        cls = tables.upos_class[j0:j1]
        x = logits.index_select(1, lay.src[j0:j1])
        e = torch.exp((x - cm.class_anchor.index_select(1, cls)).clamp_max(0.0))
        e = torch.where(lay.pad[j0:j1], zero, e * coef.index_select(1, cls))
        return e.view(L, b1 - b0, w)

    q_u = layout_reduce_upper(lay, layout_reduce_blocks(lay, L, terms, "sum", logits), "sum")
    q.index_copy_(1, tables.union_ids, q_u)
    return q


def _absolute(v: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = v + scale.unsqueeze(-1)
    return torch.where(v > DEAD, out, torch.full_like(out, SENTINEL))


def constrained_marginals(
    logits: torch.Tensor,
    tables: CdTables,
    tree: LogTree,
    a_start: torch.Tensor,
    b_final: torch.Tensor,
    l_eff: int,
    *,
    class_mass: ClassMass | None = None,
    kernels: bool = True,
) -> CdMarginals:
    """``log Z``, ``q [L, V]``, ``H(q) [L]`` and the absolute ``log_a``/``log_b`` ``[L+1, bucket]``.

    ``class_mass`` (T-0072) is the ``emission.class_mass(logits, tables)`` the caller already holds;
    it is used as is, not recomputed.

    T-0077a: both passes run on the live states ``0..S−1`` (``S = tables.n_states``; the caller's
    tree is read through ``[:S, :S]`` views); ``log_a``/``log_b`` are padded back to the bucket with
    the sentinel, which is what the bucket path stores for a padded (dead) state.

    T-0077b: ``kernels`` (keyword-only, default True) lets the ``[L, V]`` H(q) tail run as the Triton kernel
    ``kernels.entropy.hq_tail`` when ``q_raw`` is a CUDA tensor and triton imports (``kernels.backend``); the
    kernel is bit-identical to the torch tail below, which stays the CPU path and the reference. ``False``
    forces the torch tail.
    """
    l_eff = max(0, min(int(l_eff), logits.shape[0]))
    S = tables.n_states
    root = tree.levels[-1][0][:S, :S]
    root_scale = tree.log_scales[-1][0]
    la0 = _mask_to_log(a_start[:S], root)
    lb_end = _mask_to_log(b_final[:S], root)

    a, sa, b, sb = _down_sweep(tree, la0, lb_end, n_states=S)
    # the two outer boundaries: a after the last leaf, b before the first
    a_last, a_last_m = normalise_last(_vec_mat(la0.unsqueeze(0), root.unsqueeze(0)))
    b_first, b_first_m = normalise_last(_mat_vec(root.unsqueeze(0), lb_end.unsqueeze(0)))
    sa_last = root_scale + a_last_m
    sb_first = root_scale + b_first_m

    # log Z at boundary 0: a_start · (root ⊗ b_final)
    t0 = la0 + b_first[0]
    log_z = (torch.logsumexp(t0, dim=0) + sb_first[0]).clamp_min(SENTINEL)
    log_z = torch.where(t0.amax() > DEAD, log_z, torch.full_like(log_z, SENTINEL))

    log_a = torch.cat([_absolute(a, sa), _absolute(a_last, sa_last)], dim=0)  # [L+1, S]
    log_b = torch.cat([_absolute(b_first, sb_first), _absolute(b, sb)], dim=0)

    # ---- pass 2: the gauged chain. See the module docstring ("Precision").
    viable = log_b > DEAD  # [L+1, S]: b_i(s) > 0
    beta = torch.where(viable, torch.round(log_b), torch.zeros_like(log_b))
    cm = _class_mass(logits, tables) if class_mass is None else class_mass
    w_edge = _gauged_edge_weights(cm, tables, beta, viable, l_eff)  # [L, E]
    leaves = _gauged_leaves(w_edge, tables, beta, viable, l_eff)  # [L, S, S]
    tree2 = up_sweep(leaves, compensated=True)
    ga0 = torch.where(
        viable[0] & (la0 > DEAD), la0 + beta[0], torch.full_like(la0, SENTINEL)
    )  # log a_0 + β_0
    gbl = torch.where(
        viable[-1] & (lb_end > DEAD), lb_end - beta[-1], torch.full_like(lb_end, SENTINEL)
    )  # log b_L − β_L
    a2, _, b2, _ = _down_sweep(tree2, ga0, gbl, compensated=True, n_states=S)
    # per edge and position: log u_i(e) + log W_i(e) up to a per-position constant, every term small
    joint_e = (a2.index_select(1, tables.edge_src) + b2.index_select(1, tables.edge_dst)) + w_edge
    joint_e = joint_e.clamp_min(SENTINEL)
    c_m, c_s = layout_sumexp_split(joint_e, tables.edge_class_layout)  # [L, C]
    joint = torch.where(c_m > DEAD, c_m + safe_log(c_s), c_m)
    j_m, j_s = row_split(joint)
    # g_c = log G_i(c): the posterior of class c at position i, ≤ 0
    g = torch.where(
        (joint > DEAD) & (j_m > DEAD).unsqueeze(1),
        (joint - j_m.unsqueeze(1)) - safe_log(j_s).unsqueeze(1),
        torch.full_like(joint, SENTINEL),
    )
    # G_i(c) / sum_c: the per-class factor of q in linear space (p(v|c) = e^{x_v − anchor_c} / sum_c)
    coef = torch.where(
        cm.class_sum > 0, torch.exp(g) / cm.class_sum.clamp_min(1.0), torch.zeros_like(g)
    )

    # ---- per token, linear and non-subtractive: q(v) ∝ Σ_{c ∋ v} e^{x_v − anchor_c}·coef_c.
    # Every exponent is ≤ 0 (the anchor is the class max), so nothing overflows, and a token's
    # q keeps fp32's relative precision instead of the resolution of log q at |log q| ≈ 7.
    q_raw = _token_marginals(logits, tables, cm, coef)
    if _kernel_backend.use_kernels(q_raw, kernels=kernels):
        # T-0077b: the same ops in the same order, fused (cd/kernels/entropy.py); q written into q_raw
        q, entropy = _kernel_entropy.hq_tail(q_raw)
    else:
        total = compensated_sum(q_raw).unsqueeze(-1).clamp_min(torch.finfo(q_raw.dtype).tiny)
        # H = −Σ q·log(max(q, 1e-30)) evaluated on the unnormalised q̃ as log S − Σ q̃·log q̃_c / S, where
        # log q̃_c = max(log q̃, log S + log 1e-30) is exactly log(max(q, 1e-30)) + log S: a rounding δ
        # of S then enters H as δ, not δ·H, and both sums are compensated (criterion 2's bar is ~2 ulp
        # of H at |H| ≈ 7). Forbidden tokens (q̃ = 0) contribute exactly 0.
        # Peak memory (T-0073 D6 item 2): H is taken before q is formed, and the [L, V] term q̃·log q̃_c
        # is built in one buffer, then q̃ is normalised in place: at most two [L, V] tensors are live
        # here. The same elementwise ops on the same values as the out-of-place form, so the same bits.
        log_total = torch.log(total)
        term = q_raw.clamp_min(1e-38)
        term.log_()
        torch.maximum(term, log_total + math.log(1e-30), out=term)  # log q̃_c
        term.mul_(q_raw)  # q̃·log q̃_c
        entropy = log_total.squeeze(-1) - compensated_sum(term) / total.squeeze(-1)
        del term
        q = q_raw.div_(total)
    B = tables.bucket
    return CdMarginals(
        log_z=log_z, q=q, entropy=entropy, log_a=_pad_states(log_a, B), log_b=_pad_states(log_b, B)
    )
