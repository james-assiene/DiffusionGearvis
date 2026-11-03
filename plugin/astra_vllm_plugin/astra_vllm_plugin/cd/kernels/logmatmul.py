"""The log-semiring product ``semiring.log_matmul`` in Triton, plain and compensated, in one launch (T-0077c).

``[nb, n, m] ⊗ [nb, m, k] -> [nb, n, k]``, ``C[g, i, k] = log Σ_j exp(A[g, i, j] + B[g, j, k])``. Each program owns a
``[BLOCK_N, BLOCK_K]`` tile of one batch entry and computes, in registers, **the spec order** of
``docs/tasks/T-0077c.md`` (the contract, bit for bit), per output entry, in IEEE fp32 round-to-nearest with no
contraction (``backend.launch_options()``):

1. max pass over j (any order; max is exact): ``mx = max_j t_j``, ``t_j = fl(a_ij + b_jk)``, NaN-propagating
   (``tl.maximum(..., propagate_nan=ALL)``, inline);
2. ``live = mx > DEAD``, ``safe = live ? mx : 0`` (``DEAD`` and ``SENTINEL`` enter as fp32 bit patterns);
3. sum pass, **strictly j = 0, 1, …, m − 1 ascending**, from ``s = c = +0.0``: compensated
   ``x_j = exp(fl(fl(t_j − safe) + err_j))`` with ``bb = fl(t_j − a)``, ``err_j = fl(fl(a − fl(t_j − bb)) + fl(b − bb))``
   (``semiring.log_matmul``'s formulas); plain ``x_j = exp(fl(t_j − safe))``; then TwoSum ``s' = fl(s + x_j)``,
   ``z = fl(s' − s)``, ``e = fl(fl(s − fl(s' − z)) + fl(x_j − z))``, ``c = fl(c + e)``, ``s = s'``;
4. ``S_tot = fl(s + c)``;
5. ``semiring._finish(mx, S_tot, compensated)`` op for op: ``ls = log(S_tot)``; compensated TwoSum(``safe``, ``ls``)
   → ``hi + (lo + (fl(S_tot·exp(−ls)) − 1))``, plain ``safe + ls``; then ``where(live, ·, SENTINEL)``.

The j loop runs over exactly ``m`` terms (no masked j lanes), so the bits do not depend on the tile (criterion 3b);
lanes outside ``n``/``k`` are computed on whatever they load and never stored. No ``tl.dot`` (TF32), no
``tl.sum``/``tl.reduce`` (no promised order), no atomics.

**Elementwise functions.** On CUDA ``exp``/``log`` are libdevice ``__nv_expf``/``__nv_logf`` (``tl.exp``/``tl.log``
are approximate there), compiled with ``enable_reflect_ftz=False`` and ``enable_fp_fusion=False`` per launch. Under
``TRITON_INTERPRET=1`` libdevice cannot run: a ``constexpr`` read from ``triton.knobs.runtime.interpret`` (read,
never set) selects numpy's fp64 ``exp``/``log`` rounded once to fp32 (ruling O2).

**Launches, memory, sync.** One launch per call, whatever the shape (grid ``nb · ⌈n/BLOCK_N⌉ · ⌈k/BLOCK_K⌉``). The
only allocation is the ``[nb, n, k]`` output. Strides and ``n, m, k`` are runtime ints (live-block views have row
stride B > S; a host broadcast has batch stride 0); tiles are ``constexpr`` Python ints from the shapes. Offsets are
int32: the host checks every addressed offset fits. No host sync.

Triton is imported only inside ``_kernels`` (ruling M1): it binds ``triton``, ``tl``, ``libdevice`` and every
``@triton.jit`` function as module globals (ruling K3) and is cached with ``functools.cache``.
"""

from __future__ import annotations

import functools
import struct
from typing import Any

import torch

from . import backend

#: the production tile edge for a matrix-matrix product (``[TILE, TILE]`` outputs per program)
TILE = 32
#: the production tile length along the vector dimension of a vec-mat (n = 1) or mat-vec (k = 1) product: 128 lanes
#: = 4 warps of one output each (the down-sweep's vector products have few programs, so their time is the serial j
#: path of one program, which is shortest with one output per thread)
TILE_VEC = 128

_INT32_MAX = (1 << 31) - 1


def _f32_bits(x: float) -> int:
    """The fp32 rounding of ``x`` as a signed int32 bit pattern (what torch makes of a Python scalar in fp32)."""
    bits: int = struct.unpack("<i", struct.pack("<f", x))[0]
    return bits


#: ``semiring.SENTINEL`` and ``semiring.DEAD`` in fp32 (the torch path compares/fills fp32 tensors with them)
SENTINEL_BITS = _f32_bits(-1e30)
DEAD_BITS = _f32_bits(-1e30 / 2.0)


def _np2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def tiles_for(
    n: int, k: int, block_n: int | None = None, block_k: int | None = None
) -> tuple[int, int]:
    """The output tile ``(BLOCK_N, BLOCK_K)``: powers of two ≥ 1, each clamped to the next power of two of ``n``
    (resp. ``k``); ``None`` is the production choice (``TILE`` square, or ``TILE_VEC`` along the vector of a vec-mat
    or mat-vec). A function of the shapes only; the bits never depend on it."""
    for name, b in (("block_n", block_n), ("block_k", block_k)):
        if b is not None and (b < 1 or b & (b - 1)):
            raise ValueError(f"{name} must be a power of two >= 1, got {b}")
    pn, pk = _np2(n), _np2(k)
    if block_n is None:
        block_n = TILE_VEC if k == 1 else TILE
    if block_k is None:
        block_k = TILE_VEC if n == 1 else TILE
    return min(block_n, pn), min(block_k, pk)


def num_warps_for(bn: int, bk: int) -> int:
    """Warps per program: one thread per output up to 4 warps (a tile of ≥ 128 outputs gets 4), fewer for smaller
    tiles (shape-derived)."""
    return max(1, min(4, (bn * bk) // 32))


def launches(n: int, k: int, block_n: int | None = None, block_k: int | None = None) -> int:
    """Kernel launches per ``log_matmul`` call: one, independent of the shapes (criterion 2)."""
    tiles_for(n, k, block_n, block_k)
    return 1


@functools.cache
def _kernels() -> dict[str, Any]:
    """Import triton and define the kernel (once per process). See the module docstring."""
    global triton, tl, libdevice
    global _lm_f32, _lm_exp, _lm_log, _lm_ld, _lm_term, _lm_add, _lm_kernel
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _lm_f32(BITS: tl.constexpr):
        # an fp32 constant from its bit pattern, shape [1, 1]
        return tl.full((1, 1), BITS, tl.int32).to(tl.float32, bitcast=True)

    @triton.jit
    def _lm_exp(x, INTERP: tl.constexpr):
        if INTERP:
            # ruling O2: numpy's exp in fp64, rounded once to fp32 (never compiled: INTERP is a constexpr)
            y = tl.exp(x.to(tl.float64)).to(tl.float32)
        else:
            y = libdevice.exp(x)
        return y

    @triton.jit
    def _lm_log(x, INTERP: tl.constexpr):
        if INTERP:
            y = tl.log(x.to(tl.float64)).to(tl.float32)
        else:
            y = libdevice.log(x)
        return y

    @triton.jit
    def _lm_ld(a_rows, b_cols, j, sa2, sb1, mask_n, mask_k):
        # a[g, i, j] as [BLOCK_N, 1] and b[g, j, k] as [1, BLOCK_K] (lanes outside n/k load 0.0, never stored)
        av = tl.load(a_rows + j * sa2, mask=mask_n, other=0.0)
        bv = tl.load(b_cols + j * sb1, mask=mask_k, other=0.0)
        return av[:, None], bv[None, :]

    @triton.jit
    def _lm_term(av, bv, safe, COMP: tl.constexpr, INTERP: tl.constexpr):
        # spec step 3's x_j: compensated exp(fl(fl(t - safe) + err)) with semiring's bb/err; plain exp(fl(t - safe))
        t = av + bv
        if COMP:
            bb = t - av
            err = (av - (t - bb)) + (bv - bb)
            x = _lm_exp((t - safe) + err, INTERP)
        else:
            x = _lm_exp(t - safe, INTERP)
        return x

    @triton.jit
    def _lm_add(s, c, x):
        # spec step 3's TwoSum with the carried error: s' = s + x, z = s' - s, e = (s - (s' - z)) + (x - z), c += e
        s2 = s + x
        z = s2 - s
        e = (s - (s2 - z)) + (x - z)
        return s2, c + e

    @triton.jit
    def _lm_kernel(
        a,
        b,
        out,
        n,
        m,
        k,
        sa0,
        sa1,
        sa2,
        sb0,
        sb1,
        sb2,
        so0,
        so1,
        so2,
        tiles_n,
        tiles_k,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        COMP: tl.constexpr,
        INTERP: tl.constexpr,
        DEAD: tl.constexpr,
        SENTINEL: tl.constexpr,
    ):
        # one program per (batch entry, [BLOCK_N, BLOCK_K] output tile): the spec order, steps 1-5
        pid = tl.program_id(0)
        per_g = tiles_n * tiles_k
        g = pid // per_g
        r = pid - g * per_g
        tn = r // tiles_k
        tk = r - tn * tiles_k
        offs_n = tn * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tk * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_n = offs_n < n
        mask_k = offs_k < k
        a_rows = a + g * sa0 + offs_n * sa1  # [BLOCK_N] pointers to a[g, i, 0]
        b_cols = b + g * sb0 + offs_k * sb2  # [BLOCK_K] pointers to b[g, 0, k]

        # Both passes walk j strictly ascending. The bulk runs in groups of 8 consecutive j whose 8 operand loads are
        # issued before the 8 updates that consume them (in order), so one memory latency is paid per group
        # rather than per j; a scalar tail loop does the last m mod 8 (no j is masked). The per-term ops and
        # their order are those of the one-j loop, so the bits are too.

        # 1. the pairwise anchor: max_j fl(a_ij + b_jk), NaN-propagating
        a0, b0 = _lm_ld(a_rows, b_cols, 0, sa2, sb1, mask_n, mask_k)
        mx = a0 + b0
        m1 = 1 + ((m - 1) // 8) * 8
        for j0 in range(1, m1, 8):
            a0, b0 = _lm_ld(a_rows, b_cols, j0, sa2, sb1, mask_n, mask_k)
            a1, b1 = _lm_ld(a_rows, b_cols, j0 + 1, sa2, sb1, mask_n, mask_k)
            a2, b2 = _lm_ld(a_rows, b_cols, j0 + 2, sa2, sb1, mask_n, mask_k)
            a3, b3 = _lm_ld(a_rows, b_cols, j0 + 3, sa2, sb1, mask_n, mask_k)
            a4, b4 = _lm_ld(a_rows, b_cols, j0 + 4, sa2, sb1, mask_n, mask_k)
            a5, b5 = _lm_ld(a_rows, b_cols, j0 + 5, sa2, sb1, mask_n, mask_k)
            a6, b6 = _lm_ld(a_rows, b_cols, j0 + 6, sa2, sb1, mask_n, mask_k)
            a7, b7 = _lm_ld(a_rows, b_cols, j0 + 7, sa2, sb1, mask_n, mask_k)
            mx = tl.maximum(mx, a0 + b0, propagate_nan=tl.PropagateNan.ALL)
            mx = tl.maximum(mx, a1 + b1, propagate_nan=tl.PropagateNan.ALL)
            mx = tl.maximum(mx, a2 + b2, propagate_nan=tl.PropagateNan.ALL)
            mx = tl.maximum(mx, a3 + b3, propagate_nan=tl.PropagateNan.ALL)
            mx = tl.maximum(mx, a4 + b4, propagate_nan=tl.PropagateNan.ALL)
            mx = tl.maximum(mx, a5 + b5, propagate_nan=tl.PropagateNan.ALL)
            mx = tl.maximum(mx, a6 + b6, propagate_nan=tl.PropagateNan.ALL)
            mx = tl.maximum(mx, a7 + b7, propagate_nan=tl.PropagateNan.ALL)
        for j in range(m1, m):
            a0, b0 = _lm_ld(a_rows, b_cols, j, sa2, sb1, mask_n, mask_k)
            mx = tl.maximum(mx, a0 + b0, propagate_nan=tl.PropagateNan.ALL)

        # 2. live / safe
        live = mx > _lm_f32(DEAD)
        safe = tl.where(live, mx, 0.0)

        # 3. j ascending, TwoSum with the carried error
        s = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
        c = tl.zeros((BLOCK_N, BLOCK_K), tl.float32)
        m8 = (m // 8) * 8
        for j0 in range(0, m8, 8):
            a0, b0 = _lm_ld(a_rows, b_cols, j0, sa2, sb1, mask_n, mask_k)
            a1, b1 = _lm_ld(a_rows, b_cols, j0 + 1, sa2, sb1, mask_n, mask_k)
            a2, b2 = _lm_ld(a_rows, b_cols, j0 + 2, sa2, sb1, mask_n, mask_k)
            a3, b3 = _lm_ld(a_rows, b_cols, j0 + 3, sa2, sb1, mask_n, mask_k)
            a4, b4 = _lm_ld(a_rows, b_cols, j0 + 4, sa2, sb1, mask_n, mask_k)
            a5, b5 = _lm_ld(a_rows, b_cols, j0 + 5, sa2, sb1, mask_n, mask_k)
            a6, b6 = _lm_ld(a_rows, b_cols, j0 + 6, sa2, sb1, mask_n, mask_k)
            a7, b7 = _lm_ld(a_rows, b_cols, j0 + 7, sa2, sb1, mask_n, mask_k)
            s, c = _lm_add(s, c, _lm_term(a0, b0, safe, COMP, INTERP))
            s, c = _lm_add(s, c, _lm_term(a1, b1, safe, COMP, INTERP))
            s, c = _lm_add(s, c, _lm_term(a2, b2, safe, COMP, INTERP))
            s, c = _lm_add(s, c, _lm_term(a3, b3, safe, COMP, INTERP))
            s, c = _lm_add(s, c, _lm_term(a4, b4, safe, COMP, INTERP))
            s, c = _lm_add(s, c, _lm_term(a5, b5, safe, COMP, INTERP))
            s, c = _lm_add(s, c, _lm_term(a6, b6, safe, COMP, INTERP))
            s, c = _lm_add(s, c, _lm_term(a7, b7, safe, COMP, INTERP))
        for j in range(m8, m):
            a0, b0 = _lm_ld(a_rows, b_cols, j, sa2, sb1, mask_n, mask_k)
            s, c = _lm_add(s, c, _lm_term(a0, b0, safe, COMP, INTERP))

        # 4. S_tot
        st = s + c

        # 5. semiring._finish, op for op
        ls = _lm_log(st, INTERP)
        if COMP:
            hi = safe + ls
            bh = hi - safe
            lo = (safe - (hi - bh)) + (ls - bh)
            res = hi + (lo + (st * _lm_exp(-ls, INTERP) - 1.0))
        else:
            res = safe + ls
        res = tl.where(live, res, _lm_f32(SENTINEL))

        o = out + g * so0 + offs_n[:, None] * so1 + offs_k[None, :] * so2
        tl.store(o, res, mask=mask_n[:, None] & mask_k[None, :])

    return {
        "kernel": _lm_kernel,
        "interp": bool(triton.knobs.runtime.interpret),
        "version": str(triton.__version__),
    }


def _check(a3: torch.Tensor, b3: torch.Tensor) -> tuple[int, int, int, int]:
    if a3.dtype != torch.float32 or b3.dtype != torch.float32 or a3.dim() != 3 or b3.dim() != 3:
        raise ValueError(
            "cd.kernels.logmatmul takes fp32 [nb, n, m] and [nb, m, k] tensors, got "
            f"{a3.dtype} {tuple(a3.shape)} and {b3.dtype} {tuple(b3.shape)}"
        )
    nb, n, m = (int(d) for d in a3.shape)
    nb2, m2, k = (int(d) for d in b3.shape)
    if nb != nb2 or m != m2:
        raise ValueError(
            f"cd.kernels.logmatmul: shapes do not compose: {tuple(a3.shape)} ⊗ {tuple(b3.shape)}"
        )
    if m < 1:
        raise ValueError("cd.kernels.logmatmul needs an inner dimension m >= 1")
    return nb, n, m, k


def _max_offset(t: torch.Tensor) -> int:
    return sum((int(d) - 1) * int(s) for d, s in zip(t.shape, t.stride(), strict=True))


def log_matmul(
    a3: torch.Tensor,
    b3: torch.Tensor,
    *,
    compensated: bool = False,
    block_n: int | None = None,
    block_k: int | None = None,
) -> torch.Tensor:
    """``[nb, n, m] ⊗ [nb, m, k] -> [nb, n, k]`` (a new fp32 tensor), the spec order's bits whatever the tiles.

    ``a3``/``b3``: fp32, any strides (live-block views, a stride-0 broadcast batch), never written. On CUDA, or on
    CPU under ``TRITON_INTERPRET=1``; the device is not checked here (``backend.use_logmatmul`` decides). Runs
    eagerly under ``torch.compile`` (a per-function ``torch.compiler.disable``, applied on first use: applying it at
    import would import dynamo, and through it triton, on hosts that never dispatch)."""
    return _eager()(a3, b3, compensated, block_n, block_k)


@functools.cache
def _eager() -> Any:
    return torch.compiler.disable(_log_matmul)


def _log_matmul(
    a3: torch.Tensor,
    b3: torch.Tensor,
    compensated: bool,
    block_n: int | None,
    block_k: int | None,
) -> torch.Tensor:
    nb, n, m, k = _check(a3, b3)
    out = torch.empty((nb, n, k), dtype=torch.float32, device=a3.device)
    if out.numel() == 0:
        return out
    bn, bk = tiles_for(n, k, block_n, block_k)
    tiles_n = -(-n // bn)
    tiles_k = -(-k // bk)
    grid = nb * tiles_n * tiles_k
    # int32 offsets: every addressed element and every program id must fit (the core's sizes are far below)
    worst = max(_max_offset(a3), _max_offset(b3), _max_offset(out), grid, out.numel())
    if worst > _INT32_MAX - max(bn, bk):
        raise ValueError(f"cd.kernels.logmatmul: offsets exceed int32 ({worst})")
    kern = _kernels()
    kern["kernel"][(grid,)](
        a3,
        b3,
        out,
        n,
        m,
        k,
        *a3.stride(),
        *b3.stride(),
        *out.stride(),
        tiles_n,
        tiles_k,
        BLOCK_N=bn,
        BLOCK_K=bk,
        COMP=bool(compensated),
        INTERP=kern["interp"],
        DEAD=DEAD_BITS,
        SENTINEL=SENTINEL_BITS,
        num_warps=num_warps_for(bn, bk),
        **backend.launch_options(),
    )
    return out
