"""Log-semiring primitives for the constrained-decoding core (T-0065; SPEC §2.6, docs/07 §0).

Everything here is in log space and uses the finite sentinel ``SENTINEL = -1e30`` for "log 0",
never ``-inf``. Two primitives:

- ``log_matmul``: ``C[..., i, k] = log Σ_j exp(A[..., i, j] + B[..., j, k])`` with the **pairwise**
  anchor ``max_j (A[i, j] + B[j, k])`` per output entry. A single global shift, or a row-max of A
  plus a column-max of B, loses entries whose dominant term sits far below the foreign anchor (the
  ``live_simple_106-63-0`` two-band regime: tail at 0 nats, the genuine paths ~850 nats below). The
  broadcast ``[..., n, m, k]`` is materialised in chunks of at most ``CHUNK_ELEMS`` elements, so no
  single allocation exceeds a fixed cap whatever the bucket (criterion 8).
- ``segment_logsumexp``: a scatter-logsumexp over the last axis, anchored per segment (two passes:
  segment max, then the shifted sum). This is how class masses are built as sums of non-negative
  terms, never by subtracting two large sums. Every segment reduction runs on a ``SegmentLayout``
  (stable sort by segment, padded dense blocks, ``sum``/``amax`` over the last axis), never on
  ``index_add_``/scatter atomics, so its summation order is fixed and results are bitwise
  repeatable on CUDA as on CPU (T-0073 D4). The layouts are built on the host with the tables.

No host synchronisation: shapes and chunk bounds are Python ints taken from ``.shape``; liveness
is a tensor predicate applied with ``torch.where``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import torch

from .kernels import backend as _kernel_backend

#: Finite "log 0" (docs/07 §0). Anything at or below ``DEAD`` is treated as zero.
SENTINEL = -1e30
DEAD = SENTINEL / 2.0

#: Elements per materialised chunk of a broadcast (16 MiB in fp32, 32 MiB in fp64).
CHUNK_ELEMS = 1 << 22

#: Slots per block of a ``SegmentLayout`` level (a longer segment spans more than one level).
LAYOUT_BLOCK = 64


def _finish(mx: torch.Tensor, s: torch.Tensor, compensated: bool = False) -> torch.Tensor:
    """``mx + log s`` where ``mx`` is live, the sentinel elsewhere (``s ≥ 1`` where live).

    ``compensated``: ``log s`` is corrected by its first-order residual ``s·e^{−ls} − 1`` and
    ``mx + ls`` by its TwoSum error, so the stored value carries (about) one rounding.
    """
    live = mx > DEAD
    safe = torch.where(live, mx, torch.zeros_like(mx))
    ls = torch.log(s)
    if compensated:
        hi, lo = two_sum(safe, ls)
        out = hi + (lo + (s * torch.exp(-ls) - 1.0))
    else:
        out = safe + ls
    return torch.where(live, out, torch.full_like(out, SENTINEL))


def log_matmul(
    a: torch.Tensor, b: torch.Tensor, *, compensated: bool = False, kernels: bool = True
) -> torch.Tensor:
    """Log-semiring product ``[..., n, m] ⊗ [..., m, k] -> [..., n, k]``, batched over leading dims.

    Exact per entry at any dynamic range (pairwise max anchor). All-dead entries return the
    sentinel. Chunked so that no temporary exceeds ``CHUNK_ELEMS`` elements.

    ``compensated=True`` carries the TwoSum rounding error of every ``A_ij + B_jk`` into the
    exponent, so the only rounding left is that of the stored result (measured on the gauged
    pass-2 tree: what an fp64 combine stored to fp32 achieves). About 2× the elementwise work.

    On CUDA, with triton importable and ``kernels`` not False, fp32 operands go to the fused
    Triton kernel (``kernels/logmatmul.py``, T-0077c: one launch, no chunk temporaries, the
    task's spec order). The CPU path (the reference) is the torch code below, unchanged: the
    decision reads only the device already read for ``out``. On that kernel path every call
    runs the **compensated** variant whatever ``compensated`` asks for (T-0077c ruling O9: the
    served path never uses the plain kernel); the torch path keeps plain/compensated as asked.
    """
    n, m = a.shape[-2], a.shape[-1]
    m2, k = b.shape[-2], b.shape[-1]
    if m != m2:
        raise ValueError(f"log_matmul: inner dims differ ({m} vs {m2})")
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    nb = 1
    for d in batch:
        nb *= d
    dtype = torch.promote_types(a.dtype, b.dtype)
    a3 = a.to(dtype).expand(*batch, n, m).reshape(nb, n, m)
    b3 = b.to(dtype).expand(*batch, m, k).reshape(nb, m, k)
    dev = a.device
    if dev.type == "cuda" and _kernel_backend.use_logmatmul(a3, b3, kernels=kernels):
        from .kernels import logmatmul as _kernel_logmatmul

        lm = _kernel_logmatmul.log_matmul(a3, b3, compensated=True)  # O9: always compensated
        return lm.reshape(*batch, n, k)
    out = torch.empty((nb, n, k), dtype=dtype, device=dev)

    kc = min(k, max(1, CHUNK_ELEMS // max(1, m)))
    rc = min(n, max(1, CHUNK_ELEMS // max(1, m * kc)))
    bg = max(1, CHUNK_ELEMS // max(1, n * m * kc)) if rc == n else 1
    for b0 in range(0, nb, bg):
        b1 = min(nb, b0 + bg)
        for r0 in range(0, n, rc):
            r1 = min(n, r0 + rc)
            lhs = a3[b0:b1, r0:r1, :].unsqueeze(-1)  # [g, r, m, 1]
            for c0 in range(0, k, kc):
                c1 = min(k, c0 + kc)
                rhs = b3[b0:b1, :, c0:c1].unsqueeze(-3)
                t = lhs + rhs  # [g, r, m, c]
                mx = t.amax(dim=-2)  # [g, r, c]: the pairwise anchor
                live = mx > DEAD
                safe = torch.where(live, mx, torch.zeros_like(mx))
                if compensated:
                    # TwoSum error of lhs + rhs; dead terms (sentinels) give garbage here but
                    # their exponent is ~−1e30 either way, so they still vanish
                    bb = t - lhs
                    err = (lhs - (t - bb)).add_(rhs - bb)
                    t.sub_(safe.unsqueeze(-2)).add_(err).exp_()
                else:
                    t.sub_(safe.unsqueeze(-2)).exp_()
                out[b0:b1, r0:r1, c0:c1] = _finish(mx, t.sum(dim=-2), compensated)
    return out.reshape(*batch, n, k)


def segment_sumexp_split(
    values: torch.Tensor,
    segment: torch.Tensor,
    n_segments: int,
    columns: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Segment logsumexp over the last axis, returned as ``(anchor, sum)``.

    ``values`` is ``[R, K]``, ``segment`` int64 ``[K]``; both outputs are ``[R, n_segments]``. The
    segment's log-mass is ``anchor + log(sum)`` with ``anchor`` the segment's max input (an input
    value, so exact) and ``sum = Σ exp(v − anchor) ∈ [1, K]`` kept **linear** (fp32 relative
    precision). Keeping the two apart lets callers subtract anchors before anything is rounded at a
    large magnitude. Dead segments: anchor = sentinel, sum = 0.

    With ``columns`` (int64 ``[K]``) the inputs are ``values[:, columns]``.

    Builds the ``SegmentLayout`` on the fly, which reads ``segment`` on the host: **not** for the
    per-step path, which hands the layout its ``CdTables`` precomputed to ``layout_sumexp_split``.
    Bit-identical to ``layout_sumexp_split`` on that layout.
    """
    layout = segment_layout(segment, n_segments, columns=columns, device=values.device)
    return layout_sumexp_split(values, layout)


def safe_log(s: torch.Tensor) -> torch.Tensor:
    """``log s`` where ``s > 0``, 0 elsewhere (callers mask dead entries by their anchor)."""
    ok = s > 0
    return torch.where(ok, torch.log(torch.where(ok, s, torch.ones_like(s))), torch.zeros_like(s))


def segment_logsumexp(values: torch.Tensor, segment: torch.Tensor, n_segments: int) -> torch.Tensor:
    """``out[..., g] = log Σ_{j : segment[j] = g} exp(values[..., j])`` over the last axis.

    ``[R, K] -> [R, n_segments]``; the sentinel for an empty or all-dead segment. Anchored on each
    segment's own max; the shifted terms are non-negative, so nothing cancels. Builds the layout
    on the fly (host read of ``segment``); the per-step path uses ``layout_logsumexp``.
    """
    mx, s = segment_sumexp_split(values, segment, n_segments)
    return torch.where(mx > DEAD, mx + safe_log(s), mx)


# ---------------------------------------------------------------------------- order-fixed segment sums


@dataclasses.dataclass(frozen=True)
class SegmentLevel:
    """An upper level of a ``SegmentLayout``: ``n_blocks`` blocks of ``block`` slots over the level below."""

    idx: torch.Tensor  # int64 [n_blocks * block]: a block of the level below (0 at padding)
    pad: torch.Tensor  # bool [n_blocks * block]
    n_blocks: int
    block: int


@dataclasses.dataclass(frozen=True)
class SegmentLayout:
    """A fixed-order reduction plan for ``width`` inputs grouped into ``n_segments`` (T-0073 D4).

    Built once on the host (``segment_layout``). The inputs are stably sorted by segment and laid out
    densely: level 0 is ``n_blocks`` blocks of ``block`` slots, each block inside one segment and
    padded; each upper level groups the blocks of the level below the same way, and the last level
    has exactly one block per segment, in segment order. A reduction is gathers plus dense
    ``sum``/``amax`` over the last axis only (no ``index_add_``/scatter atomics), so the order of
    every addition is fixed by the layout, on every device.
    """

    n_segments: int
    width: int  # the number of inputs K (0: every segment is empty)
    block: int
    n_blocks: int
    src: torch.Tensor  # int64 [n_blocks * block]: each level-0 slot's input column (0 at padding)
    pad: torch.Tensor  # bool [n_blocks * block]
    block_segment: torch.Tensor  # int64 [n_blocks]: the segment of each level-0 block
    upper: tuple[SegmentLevel, ...]


def _host_int64(x: np.ndarray | torch.Tensor | object) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.int64).reshape(-1)


def segment_layout(
    segment: np.ndarray | torch.Tensor,
    n_segments: int,
    *,
    columns: np.ndarray | torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    block_cap: int = LAYOUT_BLOCK,
) -> SegmentLayout:
    """Host-side: the ``SegmentLayout`` of ``segment`` (int ``[K]``, values in ``[0, n_segments)``).

    With ``columns`` (int ``[K]``) level-0 slot of input ``j`` addresses column ``columns[j]`` (the
    gather of ``segment_sumexp_split``'s ``columns``, folded in). Inside a segment the inputs keep
    their original order. A block holds at most ``block_cap`` slots.
    """
    seg = _host_int64(segment)
    width = int(seg.size)
    if width and (int(seg.min()) < 0 or int(seg.max()) >= n_segments):
        raise ValueError("segment id outside [0, n_segments)")
    cols = None
    if columns is not None:
        cols = _host_int64(columns)
        if cols.size != width:
            raise ValueError("columns and segment differ in length")
    # the current level's items, grouped by segment in order (``items``), and their counts
    items = np.argsort(seg, kind="stable").astype(np.int64)
    if width:
        counts = np.bincount(seg, minlength=n_segments).astype(np.int64)
    else:
        counts = np.zeros(n_segments, np.int64)
    cap = max(2, int(block_cap))  # a block of 1 would never shrink a level

    levels: list[tuple[np.ndarray, np.ndarray, int, int, np.ndarray]] = []
    while True:
        longest = int(counts.max()) if n_segments else 0
        final = longest <= cap
        w = max(1, longest) if final else cap
        nblk = -(-counts // w)
        if final:
            nblk = np.maximum(nblk, 1)  # the last level has one block per segment, empty ones too
        nb = int(nblk.sum())
        first_blk = np.cumsum(nblk) - nblk  # [G]
        first_item = np.cumsum(counts) - counts  # [G]
        owner_item = np.repeat(np.arange(n_segments, dtype=np.int64), counts)  # [items]
        rank = np.arange(items.size, dtype=np.int64) - first_item[owner_item]
        flat = np.full(nb * w, -1, dtype=np.int64)
        flat[(first_blk[owner_item] + rank // w) * w + rank % w] = items
        owner = np.repeat(np.arange(n_segments, dtype=np.int64), nblk)  # [nb]
        pad = flat < 0
        levels.append((np.where(pad, 0, flat), pad, nb, w, owner))
        if final:
            break
        items, counts = np.arange(nb, dtype=np.int64), nblk  # blocks are grouped by segment

    def dev(x: np.ndarray, dtype: torch.dtype = torch.int64) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(x)).to(device=device, dtype=dtype)

    idx0, pad0, nb0, w0, owner0 = levels[0]
    if cols is not None and width:
        idx0 = np.where(pad0, cols[0], cols[idx0])
    return SegmentLayout(
        n_segments=n_segments,
        width=width,
        block=w0,
        n_blocks=nb0,
        src=dev(idx0),
        pad=dev(pad0, torch.bool),
        block_segment=dev(owner0),
        upper=tuple(
            SegmentLevel(idx=dev(i), pad=dev(p, torch.bool), n_blocks=n, block=w)
            for i, p, n, w, _ in levels[1:]
        ),
    )


def layout_reduce_blocks(
    layout: SegmentLayout,
    rows: int,
    slots: Callable[[int, int], torch.Tensor],
    reduce: str,
    like: torch.Tensor,
) -> torch.Tensor:
    """``[rows, n_blocks]``: every level-0 block reduced by ``"sum"`` or ``"amax"`` over its slots.

    ``slots(b0, b1)`` returns ``[rows, b1 − b0, block]``: the values of level-0 blocks ``b0:b1``,
    padding already neutral. Chunked by whole blocks so no temporary exceeds ``CHUNK_ELEMS``; the
    chunks are a function of the shapes only, so the order of every addition is too.
    """
    w, nb = layout.block, layout.n_blocks
    cb = max(1, CHUNK_ELEMS // max(1, rows * w))

    def red(b0: int, b1: int) -> torch.Tensor:
        t = slots(b0, b1)
        return t.sum(dim=-1) if reduce == "sum" else t.amax(dim=-1)

    if nb <= cb:
        return red(0, nb)
    out = torch.empty((rows, nb), dtype=like.dtype, device=like.device)
    for b0 in range(0, nb, cb):
        b1 = min(nb, b0 + cb)
        out[:, b0:b1] = red(b0, b1)
    return out


def layout_reduce_upper(layout: SegmentLayout, x: torch.Tensor, reduce: str) -> torch.Tensor:
    """``[rows, n_blocks] -> [rows, n_segments]`` through the upper levels (fixed order, no atomics).

    Each level is chunked by whole output blocks, like ``layout_reduce_blocks``, so no gathered
    temporary exceeds ``CHUNK_ELEMS`` whatever the segment sizes.
    """
    neutral = torch.full((), 0.0 if reduce == "sum" else SENTINEL, dtype=x.dtype, device=x.device)
    rows = x.shape[0]
    for lvl in layout.upper:
        w, nb = lvl.block, lvl.n_blocks
        cb = max(1, CHUNK_ELEMS // max(1, rows * w))

        def red(b0: int, b1: int, x: torch.Tensor = x, lvl: SegmentLevel = lvl) -> torch.Tensor:
            j0, j1 = b0 * lvl.block, b1 * lvl.block
            g = torch.where(lvl.pad[j0:j1], neutral, x.index_select(1, lvl.idx[j0:j1]))
            g = g.view(rows, b1 - b0, lvl.block)
            return g.sum(dim=-1) if reduce == "sum" else g.amax(dim=-1)

        if nb <= cb:
            x = red(0, nb)
            continue
        out = torch.empty((rows, nb), dtype=x.dtype, device=x.device)
        for b0 in range(0, nb, cb):
            b1 = min(nb, b0 + cb)
            out[:, b0:b1] = red(b0, b1)
        x = out
    return x


def layout_sumexp_split(
    values: torch.Tensor, layout: SegmentLayout
) -> tuple[torch.Tensor, torch.Tensor]:
    """``segment_sumexp_split`` on a precomputed layout: the per-step form (no host sync, no atomics).

    ``values`` is ``[R, ·]``, addressed by ``layout.src``; returns ``(anchor, sum)``, each
    ``[R, n_segments]``, with ``segment_sumexp_split``'s semantics.
    """
    rows = values.shape[0]
    dtype, device = values.dtype, values.device
    if layout.width == 0:
        empty = torch.full((rows, layout.n_segments), SENTINEL, dtype=dtype, device=device)
        return empty, torch.zeros_like(empty)
    w = layout.block
    sent = torch.full((), SENTINEL, dtype=dtype, device=device)

    def gathered(b0: int, b1: int) -> torch.Tensor:
        j0, j1 = b0 * w, b1 * w
        g = torch.where(layout.pad[j0:j1], sent, values.index_select(1, layout.src[j0:j1]))
        return g.view(rows, b1 - b0, w)

    mx = layout_reduce_upper(
        layout, layout_reduce_blocks(layout, rows, gathered, "amax", values), "amax"
    )
    live = mx > DEAD
    safe = torch.where(live, mx, torch.zeros_like(mx))
    anchor = safe.index_select(1, layout.block_segment)  # [R, n_blocks]

    def shifted(b0: int, b1: int) -> torch.Tensor:
        # padding is the sentinel: its exponent is ~−1e30, so it adds exactly 0
        return (gathered(b0, b1) - anchor[:, b0:b1].unsqueeze(-1)).exp_()

    s = layout_reduce_upper(
        layout, layout_reduce_blocks(layout, rows, shifted, "sum", values), "sum"
    )
    ok = live & (s > 0)
    return torch.where(ok, mx, torch.full_like(mx, SENTINEL)), torch.where(
        ok, s, torch.zeros_like(s)
    )


def layout_logsumexp(values: torch.Tensor, layout: SegmentLayout) -> torch.Tensor:
    """``segment_logsumexp`` on a precomputed layout (the per-step form)."""
    mx, s = layout_sumexp_split(values, layout)
    return torch.where(mx > DEAD, mx + safe_log(s), mx)


def sumexp_add(
    m1: torch.Tensor, s1: torch.Tensor, m2: torch.Tensor, s2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of two split masses ``s1·e^{m1} + s2·e^{m2}`` as ``(anchor, sum)``; non-negative terms."""
    m = torch.maximum(m1, m2)
    live = m > DEAD
    safe = torch.where(live, m, torch.zeros_like(m))
    s = s1 * torch.exp(m1 - safe) + s2 * torch.exp(m2 - safe)
    ok = live & (s > 0)
    return torch.where(ok, m, torch.full_like(m, SENTINEL)), torch.where(ok, s, torch.zeros_like(s))


def compensated_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum over the last axis by a pairwise tree carrying every TwoSum error (fp32-safe).

    The rounding of a plain fp32 reduction over V = 262144 is a few ulp of the result, which is
    what criterion 2's H(q) bar is made of; this leaves ~1 ulp. ``log2(n)`` batched levels.
    """
    n = x.shape[-1]
    p = 1
    while p < n:
        p *= 2
    if p != n:
        x = torch.nn.functional.pad(x, (0, p - n))
    err = torch.zeros_like(x[..., : max(1, p // 2)])
    first = True
    while x.shape[-1] > 1:
        a, b = x[..., 0::2], x[..., 1::2]
        s, e = two_sum(a, b)
        err = e if first else (err[..., 0::2] + err[..., 1::2]).add_(e)
        first = False
        x = s
    if first:
        return x.squeeze(-1)
    return (x + err).squeeze(-1)


def two_sum(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Knuth's TwoSum: ``s = fl(x + y)`` and its rounding error ``e``, so ``s + e = x + y`` exactly.

    Meaningful for finite inputs; callers mask dead (sentinel) entries afterwards.
    """
    s = x + y
    bb = s - x
    e = (x - (s - bb)) + (y - bb)
    return s, e


def _integer_shift(mx: torch.Tensor) -> torch.Tensor:
    """``ceil(max)`` where live, 0 where dead: an integer shift, so ``x − shift`` is exact in the
    range that matters and the accumulated log-scales are exact integer sums."""
    live = mx > DEAD
    return torch.where(live, torch.ceil(mx), torch.zeros_like(mx))


def normalise_last2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift each ``[..., S, S]`` matrix so its max lies in (−1, 0]; returns ``(x − shift, shift)``.

    The shift is an integer (``ceil(max)``); dead matrices keep shift 0 and stay at the sentinel.
    """
    shift = _integer_shift(x.amax(dim=(-2, -1)))
    return x - shift[..., None, None], shift


def normalise_last(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift each ``[..., S]`` vector so its max lies in (−1, 0]; integer shift, dead → 0."""
    shift = _integer_shift(x.amax(dim=-1))
    return x - shift[..., None], shift
