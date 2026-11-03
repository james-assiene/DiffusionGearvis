"""FROZEN ORACLE -- never edit. The pre-change ``budget.delta_star`` (T-0065), copied verbatim.

Source: ``plugin/astra_vllm_plugin/astra_vllm_plugin/cd/budget.py`` at commit ``f9304b1`` (the merge of T-0067
into the T-0062 integration branch; the base of T-0072). The function body below is byte-identical to that
file's ``delta_star``; ``DeltaStar`` is the same dataclass. Only the module docstring and the import of
``CdTables`` (a type annotation, made a string) differ. T-0072 criterion 1 compares the rewritten scan against
this sequential implementation bit for bit; criterion 2's liveness controls use it as the op-heavy baseline.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import torch

CdTables = Any


@dataclasses.dataclass(frozen=True)
class DeltaStar:
    states: torch.Tensor  # bool [bucket]
    empty: torch.Tensor  # 0-d bool: truthy iff no state survives


def delta_star(
    tables: CdTables,
    states: torch.Tensor,
    committed_ids: torch.Tensor,
    stop_ids: Sequence[int],
) -> DeltaStar:
    """Advance a state set over the committed ids through the first stop id (inclusive)."""
    device = tables.distance_to_final.device
    B = tables.bucket
    ids = committed_ids.to(device=device, dtype=torch.int64).reshape(-1)
    n = ids.shape[0]
    cur = states.to(device=device, dtype=torch.bool).clone()
    if n == 0:
        return DeltaStar(states=cur, empty=~cur.any())

    # which ids are consumed: everything up to and including the first stop id
    if len(stop_ids):
        stops = torch.as_tensor(list(stop_ids), dtype=torch.int64, device=device)
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
    edge_ok = admit.index_select(1, tables.edge_class)  # [n, E]

    for t in range(n):
        fire = cur.index_select(0, tables.edge_src) & edge_ok[t]
        nxt = torch.zeros(B, dtype=torch.int32, device=device)
        nxt.index_add_(0, tables.edge_dst, fire.to(torch.int32))
        cur = torch.where(consumed[t], nxt > 0, cur)
    return DeltaStar(states=cur, empty=~cur.any())
