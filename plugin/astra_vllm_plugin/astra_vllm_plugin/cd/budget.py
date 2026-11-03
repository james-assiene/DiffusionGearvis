"""Budget-aware terminal factor and δ* across canvases (T-0065; SPEC §3.1b, §3.5 trap 1).

- ``L_eff = min(CL, valid_canvas_len, max_tokens − emitted)``: positions ``≥ L_eff`` are identity.
- ``R_after = min(max_tokens − emitted, headroom) − L_eff``: the tokens left **after** this canvas
  (the off-by-one-canvas correction: subtract the canvas being drawn). ``headroom`` already folds in
  the cache's −1 (ruling R6).
- ``b(s) = 1[d(s) ≤ R_after]``: from the canvas's end state, F must stay reachable in what is left.
- ``initial_infeasibility`` is the host check at ``add_request`` (``A_0`` is known on the host).
- ``delta_star`` consumes the committed ids up to and including the first stop id and returns the
  reached state set; an empty set is reported, never replaced by the previous one.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import torch

from .emission import UNREACHABLE, CdTables


@dataclasses.dataclass(frozen=True)
class BudgetInfeasible:
    """``min_{s ∈ A} d(s) > max_tokens − emitted``: no string in budget reaches F."""

    min_distance: int
    remaining: int


@dataclasses.dataclass(frozen=True)
class DeltaStar:
    states: torch.Tensor  # bool [bucket]
    empty: torch.Tensor  # 0-d bool: truthy iff no state survives


def effective_length(
    *, canvas_len: int, valid_canvas_len: int, max_tokens: int, emitted: int
) -> int:
    return min(canvas_len, valid_canvas_len, max_tokens - emitted)


def remaining_after(*, max_tokens: int, emitted: int, headroom: int, l_eff: int) -> int:
    return min(max_tokens - emitted, headroom) - l_eff


def terminal_factor(tables: CdTables, r_after: int) -> torch.Tensor:
    """bool ``[bucket]``: ``d(s) ≤ R_after`` over states that can reach F at all."""
    d = tables.distance_to_final
    return (d <= r_after) & (d < UNREACHABLE)


def initial_infeasibility(
    tables: CdTables, a_start: torch.Tensor, *, max_tokens: int, emitted: int = 0
) -> BudgetInfeasible | None:
    """Host check (syncs; called once at ``add_request``, never on the per-step path)."""
    d = tables.distance_to_final.cpu()
    start = a_start.to(device="cpu", dtype=torch.bool)
    ds = [int(x) for x, on in zip(d.tolist(), start.tolist(), strict=True) if on]
    min_d = min(ds) if ds else UNREACHABLE
    remaining = max_tokens - emitted
    if min_d > remaining:
        return BudgetInfeasible(min_distance=min_d, remaining=remaining)
    return None


#: Stop-id tensors per ``(device, stop-id tuple)`` (T-0072 criterion 3): built once, never from host
#: data on a later call. Keyed by the whole tuple, so a different tuple never gets a stale tensor.
_STOP_CACHE: dict[tuple[torch.device, tuple[int, ...]], torch.Tensor] = {}


#: fp32 leaf bytes per δ* chunk (T-0072 note 1): the scan's transients stay ≲ 1.5× this for any L.
_SCAN_CHUNK_BYTES = 32 << 20


def _stop_tensor(stop_ids: Sequence[int], device: torch.device) -> torch.Tensor:
    key = (device, tuple(int(s) for s in stop_ids))
    t = _STOP_CACHE.get(key)
    if t is None:
        t = torch.as_tensor(list(key[1]), dtype=torch.int64, device=device)
        _STOP_CACHE[key] = t
    return t


def prepare_stop_tensor(stop_ids: Sequence[int], device: torch.device) -> torch.Tensor:
    """Build the stop-id tensor ``delta_star`` reads, at bind time (ruling O5).

    ``torch.as_tensor(list, device=cuda)`` is a host-to-device copy; built lazily on the first constrained
    step it ran inside a sampler call, where sync-debug "error" turned it into ``internal``. ``device`` must be
    the tables' own device (``tables.distance_to_final.device``), the key ``delta_star`` looks up.
    """
    return _stop_tensor(stop_ids, device)


def delta_star(
    tables: CdTables,
    states: torch.Tensor,
    committed_ids: torch.Tensor,
    stop_ids: Sequence[int],
) -> DeltaStar:
    """Advance a state set over the committed ids through the first stop id (inclusive).

    A boolean scan (T-0072): per position a 0/1 transition matrix ``T_t[s, s']`` (some edge
    ``s → s'`` admits ``ids[t]``; the identity at positions after the first stop), composed
    left to right by pairwise batched products, ``ceil(log2 L)`` levels. A float product of 0/1
    matrices thresholded ``> 0`` is exact (a sum of non-negative terms is zero only if every term
    is, under TF32 too), and thresholding after every product keeps each entry ≤ bucket. No host
    synchronisation. Positions are taken in chunks of ``_SCAN_CHUNK_BYTES`` (32 MiB) of fp32 leaves,
    so the live transients are at most the chunk plus its first-level product, ~48 MiB plus
    ``O(L·E)`` edge masks, whatever L (one chunk up to L 1134 at 86 states, 512 at 128).
    """
    device = tables.distance_to_final.device
    B = tables.bucket
    ids = committed_ids.to(device=device, dtype=torch.int64).reshape(-1)
    n = ids.shape[0]
    cur = states.to(device=device, dtype=torch.bool).clone()
    if n == 0:
        return DeltaStar(states=cur, empty=~cur.any())

    # which ids are consumed: everything up to and including the first stop id
    if len(stop_ids):
        stops = _stop_tensor(stop_ids, device)
        is_stop = (ids.unsqueeze(1) == stops.unsqueeze(0)).any(dim=1)
        before = torch.cumsum(is_stop.to(torch.int64), dim=0) - is_stop.to(torch.int64)
        consumed = before == 0
    else:
        consumed = torch.ones(n, dtype=torch.bool, device=device)

    # classes admitting each id: a static-width window into the token-sorted membership
    C = tables.n_classes
    admit = torch.zeros((n, C + 1), dtype=torch.bool, device=device)  # column C is a dump slot
    width = tables.max_classes_per_token
    if width:
        lo = torch.searchsorted(tables.tok_sorted, ids, right=False)
        hi = torch.searchsorted(tables.tok_sorted, ids, right=True)
        pos = lo.unsqueeze(1) + torch.arange(width, device=device).unsqueeze(0)  # [n, w]
        valid = pos < hi.unsqueeze(1)
        nnz = tables.tok_sorted.shape[0]
        cls = tables.tok_sorted_class.index_select(0, pos.clamp_max(nnz - 1).reshape(-1))
        cls = torch.where(valid.reshape(-1), cls, torch.full_like(cls, C)).reshape(n, width)
        admit.scatter_(1, cls, torch.ones_like(cls, dtype=torch.bool))
    # an id outside U is admitted by every negated class
    outside = ~tables.in_union.index_select(0, ids)  # [n]
    admit = admit[:, :C] | (outside.unsqueeze(1) & tables.class_neg.unsqueeze(0))
    # an edge fires at t only if ids[t] is consumed; a position past the first stop is the identity
    edge_ok = admit.index_select(1, tables.edge_class) & consumed.unsqueeze(1)  # [n, E]

    # The scan runs over the S = n_states real states only. A padded state has no edge, and
    # position 0 is always consumed, so it never reaches a state and never survives: its output
    # bit is False, as in the per-position recurrence.
    S = tables.n_states
    pair_flat = tables.pair_flat
    if S != B:
        pair_flat = torch.div(pair_flat, B, rounding_mode="floor") * S + pair_flat.remainder(B)

    # per-position 0/1 transitions: parallel edges of one (src, dst) pair are OR-ed (a count ≥ 1 → 1)
    pair_n = torch.zeros((n, tables.n_pairs), dtype=torch.float32, device=device)
    pair_n.index_add_(1, tables.edge_pair, edge_ok.to(torch.float32))
    pair_n.clamp_(max=1.0)
    ident = (~consumed).to(torch.float32).unsqueeze(1)  # [n, 1]: 1 past the first stop

    # Compose left to right, _SCAN_CHUNK_BYTES of fp32 leaves at a time: each chunk's leaves are
    # reduced pairwise (an odd level folds its first matrix into the state vector), then the
    # vector passes through the chunk's product. Everything stays fp32 0/1 (``gt_`` in place).
    vec = cur[:S].to(torch.float32)
    k_max = max(1, _SCAN_CHUNK_BYTES // (4 * S * S)) if S else n
    for k0 in range(0, n, k_max):
        k = min(n, k0 + k_max) - k0
        trans = torch.zeros((k, S * S), dtype=torch.float32, device=device)
        trans.index_copy_(1, pair_flat, pair_n[k0 : k0 + k])
        trans = trans.view(k, S, S)
        # a non-consumed position has no edge: adding 1 on the diagonal makes it the identity
        trans.diagonal(dim1=-2, dim2=-1).add_(ident[k0 : k0 + k])
        while trans.shape[0] > 1:
            if trans.shape[0] % 2:
                vec = (vec @ trans[0]).gt_(0)
                trans = trans[1:]
            trans = torch.bmm(trans[0::2], trans[1::2]).gt_(0)
        vec = (vec @ trans[0]).gt_(0)
    live = vec > 0
    if S != B:
        cur = torch.zeros_like(cur)
        cur[:S] = live
    else:
        cur = live
    return DeltaStar(states=cur, empty=~cur.any())
