"""Log-space emission: class tables, class log-weights and per-position log transfer matrices.

T-0065 (SPEC §2.3, §4.4). ``build_tables`` runs once on the host when a request is added; it may
use numpy and Python freely. ``class_log_weights`` and ``log_transfer`` are on the per-step path:
no host synchronisation, no global RNG, tensors in the dtype of ``logits``.

**Non-subtractive complement.** A negated class stores its complement ``N_c``; its true member set
is ``V \\ N_c``. Let ``U`` be the union of every class's stored ids. The true set of every class
splits into a part inside ``U`` (listed explicitly: the stored ids of a positive class, ``U \\ N_c``
for a negated one) and, for negated classes only, the whole *outside* ``V \\ U``. So a class mass is
a logsumexp over non-negative terms — the listed ids, plus the outside mass — and never
``log(1 − Σ_{N_c} p)``, which cancels catastrophically once ``N_c`` holds nearly all the mass
(criterion 3).

**Edge multiplicity.** ``log M_i(s, s') = log Σ_{e: s→s'} W[i, class(e)]``: parallel edges add
(SPEC §2.2), including two edges with the same class. ``is_dfa`` records whether the automaton is
deterministic with unit multiplicity (the draw's ``∃`` fast path, T-0066); the transfer matrix sums
parallel edges either way, and skips the per-pair logsumexp when every pair has one edge.
"""

from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Sequence

import numpy as np
import torch

from .semiring import (  # noqa: F401  (module names kept: T-0073's probe swaps them by name)
    DEAD,
    SENTINEL,
    SegmentLayout,
    layout_logsumexp,
    layout_sumexp_split,
    safe_log,
    segment_layout,
    segment_logsumexp,
    segment_sumexp_split,
    sumexp_add,
    two_sum,
)

#: ``distance_to_final`` for states that cannot reach F (padded states included).
UNREACHABLE = 1 << 30


@dataclasses.dataclass(frozen=True)
class CdTables:
    """Device-resident class/edge tables of one automaton (ruling R1)."""

    n_states: int
    bucket: int
    vocab_size: int
    n_classes: int
    is_dfa: bool
    #: int64 ``[bucket]``: min tokens to F by reverse BFS; ``≥ UNREACHABLE`` where F is unreachable.
    distance_to_final: torch.Tensor
    # classes
    class_neg: torch.Tensor  # bool [C]
    union_ids: torch.Tensor  # int64 [|U|]: the stored-id union U, sorted
    in_union: torch.Tensor  # bool [V]
    memb_class: torch.Tensor  # int64 [nnz]: true membership restricted to U, class side
    memb_upos: torch.Tensor  # int64 [nnz]: position in ``union_ids``
    memb_token: torch.Tensor  # int64 [nnz]: token id
    has_outside: bool  # V \ U is non-empty
    # edges
    edge_src: torch.Tensor  # int64 [E]
    edge_dst: torch.Tensor  # int64 [E]
    edge_class: torch.Tensor  # int64 [E]
    edge_pair: torch.Tensor  # int64 [E]: index of (src, dst) among the distinct pairs
    pair_flat: torch.Tensor  # int64 [P]: src * bucket + dst
    n_pairs: int
    single_edge_pairs: bool
    # δ* lookup: membership sorted by token (CSR window of static width)
    tok_sorted: torch.Tensor  # int64 [nnz]
    tok_sorted_class: torch.Tensor  # int64 [nnz]
    max_classes_per_token: int
    # order-fixed segment reductions (T-0073 D4): layouts built here, once, on the host
    memb_layout: SegmentLayout  # memberships by class, slots address the logits column (class_mass)
    upos_layout: SegmentLayout  # memberships by union position, slots address the logits column
    upos_class: torch.Tensor  # int64 [upos_layout slots]: each slot's class (0 at padding)
    edge_class_layout: SegmentLayout  # edges by class
    edge_pair_layout: SegmentLayout  # edges by (src, dst) pair


def build_tables(
    *,
    n_states: int,
    vocab_size: int,
    class_members: Sequence[Sequence[int]],
    class_is_neg: Sequence[bool],
    edge_src: Sequence[int],
    edge_dst: Sequence[int],
    edge_class: Sequence[int],
    finals: Sequence[int],
    bucket: int | None = None,
    device: str | torch.device = "cpu",
) -> CdTables:
    """Host-side construction (once per request). ``class_members[c]`` is the *stored* id list."""
    bucket = n_states if bucket is None else bucket
    if bucket < n_states:
        raise ValueError(f"bucket {bucket} < n_states {n_states}")
    n_classes = len(class_members)
    if len(class_is_neg) != n_classes:
        raise ValueError("class_members and class_is_neg differ in length")
    if not (len(edge_src) == len(edge_dst) == len(edge_class)):
        raise ValueError("edge arrays differ in length")
    V = vocab_size

    stored = [np.unique(np.asarray(list(m), dtype=np.int64)) for m in class_members]
    for s in stored:
        if s.size and (s[0] < 0 or s[-1] >= V):
            raise ValueError("class member id outside the vocabulary")
    union = (
        np.unique(np.concatenate(stored)) if any(s.size for s in stored) else np.zeros(0, np.int64)
    )
    has_outside = union.size < V
    upos_of = np.full(V, -1, dtype=np.int64)
    upos_of[union] = np.arange(union.size, dtype=np.int64)

    # true membership restricted to U, per class
    members_in_u: list[np.ndarray] = []
    for c in range(n_classes):
        if class_is_neg[c]:
            members_in_u.append(np.setdiff1d(union, stored[c], assume_unique=True))
        else:
            members_in_u.append(stored[c])
    nonempty = [
        members_in_u[c].size > 0 or (bool(class_is_neg[c]) and has_outside)
        for c in range(n_classes)
    ]
    memb_class = np.concatenate(
        [np.full(m.size, c, dtype=np.int64) for c, m in enumerate(members_in_u)]
        or [np.zeros(0, np.int64)]
    )
    memb_token = np.concatenate(members_in_u or [np.zeros(0, np.int64)]).astype(np.int64)
    memb_upos = upos_of[memb_token]

    # δ* lookup: membership sorted by token, and the widest per-token class list
    order = np.argsort(memb_token, kind="stable")
    tok_sorted = memb_token[order]
    tok_sorted_class = memb_class[order]
    max_deg = int(np.bincount(memb_token).max()) if memb_token.size else 0

    # edges and (src, dst) pairs
    src = np.asarray(list(edge_src), dtype=np.int64)
    dst = np.asarray(list(edge_dst), dtype=np.int64)
    cls = np.asarray(list(edge_class), dtype=np.int64)
    if src.size and (
        src.min() < 0 or src.max() >= n_states or dst.min() < 0 or dst.max() >= n_states
    ):
        raise ValueError("edge endpoint outside the state range")
    if cls.size and (cls.min() < 0 or cls.max() >= n_classes):
        raise ValueError("edge class outside the class range")
    flat = src * bucket + dst
    pair_flat, edge_pair = np.unique(flat, return_inverse=True)
    edge_pair = edge_pair.reshape(-1).astype(np.int64)
    single = bool(pair_flat.size == flat.size)

    # is_dfa: per source state, the live outgoing labels are pairwise disjoint
    is_dfa = True
    by_src: dict[int, list[int]] = {}
    for e in range(src.size):
        if nonempty[int(cls[e])]:
            by_src.setdefault(int(src[e]), []).append(int(cls[e]))
    for classes in by_src.values():
        if len(classes) < 2:
            continue
        outside = sum(1 for c in classes if class_is_neg[c]) if has_outside else 0
        toks = np.concatenate([members_in_u[c] for c in classes])
        if outside > 1 or np.unique(toks).size != toks.size:
            is_dfa = False
            break

    # d(s): reverse BFS from F over edges with a non-empty label
    dist = np.full(bucket, UNREACHABLE, dtype=np.int64)
    rev: list[list[int]] = [[] for _ in range(n_states)]
    for e in range(src.size):
        if nonempty[int(cls[e])]:
            rev[int(dst[e])].append(int(src[e]))
    queue: deque[int] = deque()
    for f in sorted(set(int(x) for x in finals)):
        if not 0 <= f < n_states:
            raise ValueError("final state outside the state range")
        dist[f] = 0
        queue.append(f)
    while queue:
        t = queue.popleft()
        for s in rev[t]:
            if dist[s] == UNREACHABLE:
                dist[s] = dist[t] + 1
                queue.append(s)

    def dev(x: np.ndarray, dtype: torch.dtype = torch.int64) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(x)).to(device=device, dtype=dtype)

    in_union = np.zeros(V, dtype=bool)
    in_union[union] = True
    # q's per-token sum: the same layout twice, slots addressing the membership j (host, for the
    # slot's class) and the logits column (device)
    by_j = segment_layout(memb_upos, int(union.size))
    upos_j, upos_pad = by_j.src.numpy(), by_j.pad.numpy()
    upos_class = np.zeros(upos_j.size, np.int64)
    if memb_class.size:
        upos_class = np.where(upos_pad, 0, memb_class[upos_j])
    upos_layout = segment_layout(memb_upos, int(union.size), columns=memb_token, device=device)
    return CdTables(
        n_states=n_states,
        bucket=bucket,
        vocab_size=V,
        n_classes=n_classes,
        is_dfa=is_dfa,
        distance_to_final=dev(dist),
        class_neg=dev(np.asarray(list(class_is_neg), dtype=bool), torch.bool),
        union_ids=dev(union),
        in_union=dev(in_union, torch.bool),
        memb_class=dev(memb_class),
        memb_upos=dev(memb_upos),
        memb_token=dev(memb_token),
        has_outside=has_outside,
        edge_src=dev(src),
        edge_dst=dev(dst),
        edge_class=dev(cls),
        edge_pair=dev(edge_pair),
        pair_flat=dev(pair_flat.astype(np.int64)),
        n_pairs=int(pair_flat.size),
        single_edge_pairs=single,
        tok_sorted=dev(tok_sorted),
        tok_sorted_class=dev(tok_sorted_class),
        max_classes_per_token=max_deg,
        memb_layout=segment_layout(memb_class, n_classes, columns=memb_token, device=device),
        upos_layout=upos_layout,
        upos_class=dev(upos_class),
        edge_class_layout=segment_layout(cls, n_classes, device=device),
        edge_pair_layout=segment_layout(edge_pair, int(pair_flat.size), device=device),
    )


@dataclasses.dataclass(frozen=True)
class ClassMass:
    """Split masses of one step: an exact anchor (an input logit) and a **linear** sum beside it.

    ``class_anchor + log(class_sum)`` is ``log Σ_{v ∈ S_c} exp(logits_v)`` (``[L, C]``);
    ``row_max + log(row_sum)`` is the row's logsumexp (``[L]``); ``out_anchor + log(out_sum)`` is
    the mass of ``V \\ U`` (``[L]``). Differences of anchors are exact (or rounded at the size of the
    difference) and the sums keep fp32's *relative* precision, so nothing is rounded at the size of
    a logit (~75 nats at T = 0.4, where the fp32 resolution is 7.6e-6).
    """

    class_anchor: torch.Tensor
    class_sum: torch.Tensor
    row_max: torch.Tensor
    row_sum: torch.Tensor
    out_anchor: torch.Tensor
    out_sum: torch.Tensor


def row_split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(max, Σ exp(x − max))`` over the last axis; a dead row is ``(sentinel, 0)``."""
    mx = x.amax(dim=-1)
    live = mx > DEAD
    safe = torch.where(live, mx, torch.zeros_like(mx))
    s = torch.exp(x - safe.unsqueeze(-1)).sum(dim=-1)
    ok = live & (s > 0)
    return (
        torch.where(ok, mx, torch.full_like(mx, SENTINEL)),
        torch.where(ok, s, torch.zeros_like(s)),
    )


def class_mass(logits: torch.Tensor, tables: CdTables) -> ClassMass:
    """Split class masses: listed ids inside ``U`` plus, for negated classes, the outside mass."""
    L = logits.shape[0]
    row_max, row_sum = row_split(logits)
    m_list, s_list = layout_sumexp_split(logits, tables.memb_layout)
    if tables.has_outside:
        masked = torch.where(
            tables.in_union.unsqueeze(0), torch.full_like(logits[:1], SENTINEL), logits
        )
        out_m, out_s = row_split(masked)
    else:
        out_m = torch.full((L,), SENTINEL, dtype=logits.dtype, device=logits.device)
        out_s = torch.zeros_like(out_m)
    m_neg, s_neg = sumexp_add(m_list, s_list, out_m.unsqueeze(1), out_s.unsqueeze(1))
    neg = tables.class_neg.unsqueeze(0)
    return ClassMass(
        class_anchor=torch.where(neg, m_neg, m_list),
        class_sum=torch.where(neg, s_neg, s_list),
        row_max=row_max,
        row_sum=row_sum,
        out_anchor=out_m,
        out_sum=out_s,
    )


def class_log_weights_split(cm: ClassMass) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(big, k, small)`` with ``log W_c = big + k + small``; only ``small`` carries rounding.

    ``big = fl(anchor_c − row_max)`` (its TwoSum error goes into ``small``), ``k`` is the integer
    nearest ``log(sum_c / row_sum)`` and ``small`` the remainder, a log near 0 (|small| ≲ 0.5). A
    caller that adds an integer gauge to ``big + k`` before ``small`` keeps a near-0 result at fp32
    resolution near 0. Dead classes are not masked here.
    """
    big, big_err = two_sum(cm.class_anchor, -cm.row_max.unsqueeze(1))
    ratio = cm.class_sum / cm.row_sum.clamp_min(1.0).unsqueeze(1)
    # the integer part of log(ratio) joins ``big`` exactly; what is left is a log near 0
    k = torch.round(safe_log(ratio))
    small = safe_log(ratio * torch.exp(-k)) + big_err
    return big, k, small


def class_log_weights_from(cm: ClassMass) -> torch.Tensor:
    """``[L, C]`` normalised class log-weights; dead classes are the sentinel."""
    live = (cm.class_anchor > DEAD) & (cm.class_sum > 0)
    big, k, small = class_log_weights_split(cm)
    out = (big + k) + small
    return torch.where(live, out, torch.full_like(out, SENTINEL)).clamp_min(SENTINEL)


def class_log_weights(logits: torch.Tensor, tables: CdTables) -> torch.Tensor:
    """``[L, C]``: ``log Σ_{v ∈ S_c} softmax(logits_i)_v`` over the true member set.

    Built from non-negative terms only (listed ids inside ``U`` plus the outside mass for negated
    classes), each segment anchored on its own max, anchors subtracted before the log-sums are
    added. Dead classes are the sentinel.
    """
    return class_log_weights_from(class_mass(logits, tables))


def log_transfer(
    logits: torch.Tensor,
    tables: CdTables,
    l_eff: int,
    *,
    class_mass: ClassMass | None = None,
) -> torch.Tensor:
    """``[L, bucket, bucket]``: ``log M_i``; positions ``i ≥ l_eff`` are the identity.

    Zero entries are the finite sentinel, never ``-inf``/NaN. ``class_mass`` (T-0072) is the
    ``class_mass(logits, tables)`` the caller already holds; it is used as is, not recomputed.
    """
    L = logits.shape[0]
    B = tables.bucket
    l_eff = max(0, min(int(l_eff), L))
    if class_mass is None:
        cw = class_log_weights(logits, tables)  # [L, C]
    else:
        cw = class_log_weights_from(class_mass)
    ew = cw.index_select(1, tables.edge_class)  # [L, E]
    if tables.single_edge_pairs:
        # one edge per (src, dst): the pair weight is the edge weight, placed via the edge's pair
        pw = torch.empty((L, tables.n_pairs), dtype=cw.dtype, device=cw.device)
        pw.index_copy_(1, tables.edge_pair, ew)
    else:
        pw = layout_logsumexp(ew, tables.edge_pair_layout)  # parallel edges add
    log_m = torch.full((L, B * B), SENTINEL, dtype=cw.dtype, device=cw.device)
    log_m.index_copy_(1, tables.pair_flat, pw)
    log_m = log_m.view(L, B, B)
    if l_eff < L:
        tail = log_m[l_eff:]
        tail.fill_(SENTINEL)
        tail.diagonal(dim1=-2, dim2=-1).fill_(0.0)
    return log_m
