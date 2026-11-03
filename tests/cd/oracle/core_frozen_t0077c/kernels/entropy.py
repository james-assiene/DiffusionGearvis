"""The ``[L, V]`` H(q) tail of ``constrained_marginals`` in Triton, bit-identical to the torch path (T-0077b).

The torch tail (``marginals.py`` after ``q_raw = _token_marginals(...)``) is the spec, op by op in fp32:

1. ``total = clamp_min(compensated_sum(q_raw), tiny)``
2. ``log_total = log(total)``
3. ``c = clamp_min(q_raw, 1e-38)`` (the fp32 rounding of 1e-38, a subnormal)
4. ``c = log(c)``
5. ``c = maximum(c, log_total + fp32(log 1e-30))``
6. ``term = c * q_raw``, rounded on its own
7. ``entropy = log_total - compensated_sum(term) / total``
8. ``q = q_raw / total``, written in place into ``q_raw``

**Same tree, same order (criterion 3).** ``semiring.compensated_sum`` pads the row with +0.0 to the next power of
two ``p`` and sums adjacent pairs ``(x[2i], x[2i+1])`` through TwoSum; the error is ``e`` at the first level and
``(err[2i] + err[2i+1]) + e`` above it; the result is ``s + err`` (``p = 1`` returns ``x``). An aligned
power-of-two block of the padded row is a subtree of that tree, so stage 1 reduces each block of ``BLOCK`` lanes
to its subtree's ``(s, err)`` with exactly those formulas (the tree is built explicitly, reshape ``[n/2, 2]`` and
split, one level per ``static_range`` step: ``tl.sum``/``tl.reduce`` promise no order), and the combine stage
reduces the ``[L, p/BLOCK]`` partials with the upper-level formulas. The bits do not depend on ``BLOCK``.

**Elementwise ops.** ``/`` is ``tl.math.div_rn`` (Triton's ``/`` lowers to an approximate divide); ``log`` is
libdevice's ``__nv_logf`` on CUDA (``tl.log`` is approximate there), compiled with ``enable_reflect_ftz=False``
(libdevice flushes subnormals by default; torch does not); the clamps are ``tl.maximum`` with NaN propagation
(``torch.clamp_min``/``torch.maximum`` propagate NaN); every launch passes ``enable_fp_fusion=False``, so the
H-term product is never contracted into the first TwoSum add. The fp32 constants (tiny, fp32(1e-38),
fp32(log 1e-30)) enter as their bit patterns, so no Python-float conversion inside Triton can change them (a
Python float below fp32's normal range would become an fp64 constant). Masked lanes add +0.0, as the torch pad.

**Interpreter.** Under ``TRITON_INTERPRET=1`` libdevice cannot run (``extern_elementwise`` is not implemented
there): a ``constexpr`` read from ``triton.knobs.runtime.interpret`` (read, never set) selects ``tl.log`` (numpy's
log) instead. The row sum and ``q`` use only IEEE add and divide, so they stay bit-identical there; the entropy is
gated at N4 under the interpreter (ruling M2) and bit for bit on CUDA.

**Launches, memory, sync.** Per ``hq_tail`` call: the row sum of ``q_raw`` (stage 1, then the combine that writes
``total``), then the H-term row sum (stage 1, which also writes ``q = q_raw / total`` into ``q_raw``'s storage after
reading it in the same program, then the combine that writes ``entropy``): 4 launches whenever ``p/BLOCK`` ≤
``COMBINE_MAX`` (one more per combine pass beyond that). No ``[L, V]`` temporary: the partials are ``[L, p/BLOCK]``.
No host sync: grids and ``constexpr`` s are Python ints from shapes; ``L`` only sizes the grid, ``BLOCK`` depends on
``V`` only.

Triton is imported only inside ``_kernels`` (ruling M1): it binds ``triton``, ``tl``, ``libdevice`` and every
``@triton.jit`` function as module globals (the interpreter re-executes a jitted function's source in its module's
globals, ruling K3) and is cached with ``functools.cache``.
"""

from __future__ import annotations

import functools
import math
import struct
from typing import Any

import torch

from . import backend

#: the production block: a function of V only (min(BLOCK_DEFAULT, padded V))
BLOCK_DEFAULT = 1024
#: the most partials one combine program reduces
COMBINE_MAX = 1024
#: smallest block a caller may ask for
BLOCK_MIN = 16

_EPI_SUM, _EPI_TOTAL, _EPI_ENTROPY = 0, 1, 2


def _f32_bits(x: float) -> int:
    """The fp32 rounding of ``x`` as a signed int32 bit pattern (what torch makes of a Python scalar in fp32)."""
    bits: int = struct.unpack("<i", struct.pack("<f", x))[0]
    return bits


TINY_BITS = _f32_bits(torch.finfo(torch.float32).tiny)
Q_FLOOR_BITS = _f32_bits(1e-38)
LOG_FLOOR_BITS = _f32_bits(math.log(1e-30))


def _padded(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def block_for(V: int, block: int | None = None) -> int:
    """The per-program block for a row of ``V``: a power of two, clamped to the padded row (``None`` = production)."""
    if block is None:
        block = BLOCK_DEFAULT
    elif block < BLOCK_MIN or block & (block - 1):
        raise ValueError(f"block must be a power of two >= {BLOCK_MIN}, got {block}")
    return min(block, _padded(V))


def _combine_plan(nb: int) -> list[int]:
    """The group size of each combine pass over ``nb`` partials (powers of two; the last pass ends at 1)."""
    plan = []
    while True:
        g = min(nb, COMBINE_MAX)
        plan.append(g)
        nb //= g
        if nb == 1:
            return plan


def launches(V: int, block: int | None = None) -> int:
    """Kernel launches per ``hq_tail`` call on rows of ``V`` (two row sums, each stage 1 + its combine passes)."""
    nb = _padded(V) // block_for(V, block)
    return 2 * (1 + len(_combine_plan(nb)))


@functools.cache
def _kernels() -> dict[str, Any]:
    """Import triton and define the kernels (once per process). See the module docstring."""
    global triton, tl, libdevice
    global _f32, _log, _halve, _two_sum, _upper, _elementwise, _stage1, _combine
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _f32(BITS: tl.constexpr):
        # an fp32 constant from its bit pattern, shape [1]
        return tl.full((1,), BITS, tl.int32).to(tl.float32, bitcast=True)

    @triton.jit
    def _log(x, INTERP: tl.constexpr):
        if INTERP:
            # numpy's log in fp64, rounded once to fp32: numpy's fp32 log is not correctly rounded (0.53 ulp
            # off at fp32(1e-38), where torch's is 0.47); never compiled (INTERP is a constexpr)
            y = tl.log(x.to(tl.float64)).to(tl.float32)
        else:
            y = libdevice.log(x)
        return y

    @triton.jit
    def _halve(v, N: tl.constexpr):
        # [N] -> the adjacent pairs (v[2i], v[2i+1]) as two [N/2] tensors
        return tl.split(tl.reshape(v, (N // 2, 2)))

    @triton.jit
    def _two_sum(a, b):
        s = a + b
        bb = s - a
        e = (a - (s - bb)) + (b - bb)
        return s, e

    @triton.jit
    def _upper(s, err, N: tl.constexpr, LEVELS: tl.constexpr):
        # [N] partial sums and errors of aligned subtrees -> [1]; every level above the first
        for k in tl.static_range(LEVELS):
            a, b = _halve(s, N >> k)
            ea, eb = _halve(err, N >> k)
            s, e = _two_sum(a, b)
            err = (ea + eb) + e
        return s, err

    @triton.jit
    def _elementwise(
        x, t, mask, INTERP: tl.constexpr, Q_FLOOR: tl.constexpr, LOG_FLOOR: tl.constexpr
    ):
        # ops 2-6 of the torch tail: the H term max(log max(x, 1e-38), log t + log 1e-30) * x; masked lanes +0.0
        lt = _log(t, INTERP)
        c = tl.maximum(x, _f32(Q_FLOOR), propagate_nan=tl.PropagateNan.ALL)
        c = _log(c, INTERP)
        c = tl.maximum(c, lt + _f32(LOG_FLOOR), propagate_nan=tl.PropagateNan.ALL)
        return tl.where(mask, c * x, 0.0)

    @triton.jit
    def _stage1(
        src,
        total,
        s_out,
        e_out,
        V,
        NB,
        BLOCK: tl.constexpr,
        LEVELS: tl.constexpr,
        TERM: tl.constexpr,
        INTERP: tl.constexpr,
        Q_FLOOR: tl.constexpr,
        LOG_FLOOR: tl.constexpr,
    ):
        # one program per (row, aligned block): the block's subtree (s, err); TERM: the H term of q_raw, and
        # q = q_raw / total written back into the lanes just read
        row = tl.program_id(0)
        blk = tl.program_id(1)
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        mask = offs < V
        ptr = src + row.to(tl.int64) * V + offs
        x = tl.load(ptr, mask=mask, other=0.0)
        if TERM:
            t = tl.load(total + row)
            v = _elementwise(x, t, mask, INTERP, Q_FLOOR, LOG_FLOOR)
            tl.store(ptr, tl.math.div_rn(x, t), mask=mask)
        else:
            v = x
        if BLOCK == 1:
            s = v
            err = tl.zeros((1,), tl.float32)
        else:
            a, b = _halve(v, BLOCK)
            s, err = _two_sum(a, b)
            s, err = _upper(s, err, BLOCK // 2, LEVELS - 1)
        o = row * NB + blk + tl.arange(0, 1)
        tl.store(s_out + o, s)
        tl.store(e_out + o, err)

    @triton.jit
    def _combine(
        s_in,
        e_in,
        s_out,
        e_out,
        total,
        out,
        NIN,
        NOUT,
        G: tl.constexpr,
        LEVELS: tl.constexpr,
        FINAL: tl.constexpr,
        EPI: tl.constexpr,
        IDENT: tl.constexpr,
        INTERP: tl.constexpr,
        TINY: tl.constexpr,
    ):
        # one program per (row, group of G partials): the upper levels; FINAL: s + err (s alone for a row of one
        # lane) and the epilogue: 0 the sum, 1 total = max(sum, tiny), 2 entropy = log t - sum / t
        row = tl.program_id(0)
        grp = tl.program_id(1)
        offs = row * NIN + grp * G + tl.arange(0, G)
        s, err = _upper(tl.load(s_in + offs), tl.load(e_in + offs), G, LEVELS)
        one = tl.arange(0, 1)
        if FINAL:
            if IDENT:
                r = s
            else:
                r = s + err
            if EPI == 1:
                r = tl.maximum(r, _f32(TINY), propagate_nan=tl.PropagateNan.ALL)
            if EPI == 2:
                t = tl.load(total + row + one)
                r = _log(t, INTERP) - tl.math.div_rn(r, t)
            tl.store(out + row + one, r)
        else:
            o = row * NOUT + grp + one
            tl.store(s_out + o, s)
            tl.store(e_out + o, err)

    return {
        "stage1": _stage1,
        "combine": _combine,
        "interp": bool(triton.knobs.runtime.interpret),
        "version": str(triton.__version__),
    }


def _check(x: torch.Tensor) -> tuple[int, int]:
    if x.dtype != torch.float32 or x.dim() != 2 or not x.is_contiguous():
        raise ValueError(
            "cd.kernels.entropy takes a contiguous fp32 [L, V] tensor, got "
            f"{x.dtype} {tuple(x.shape)} strides {tuple(x.stride())}"
        )
    L, V = x.shape
    if V < 1:
        raise ValueError("cd.kernels.entropy needs V >= 1")
    return int(L), int(V)


def _row_reduce(
    src: torch.Tensor, out: torch.Tensor, block: int | None, epi: int, total: torch.Tensor | None
) -> None:
    """Stage 1 over ``src`` (the H term of it when ``epi`` is the entropy) and the combine passes into ``out``."""
    k = _kernels()
    L, V = src.shape
    B = block_for(V, block)
    nb = _padded(V) // B
    opts = backend.launch_options()
    term = epi == _EPI_ENTROPY
    tot = total if total is not None else src
    s = torch.empty((L, nb), dtype=torch.float32, device=src.device)
    e = torch.empty((L, nb), dtype=torch.float32, device=src.device)
    k["stage1"][(L, nb)](
        src,
        tot,
        s,
        e,
        V,
        nb,
        BLOCK=B,
        LEVELS=B.bit_length() - 1,
        TERM=term,
        INTERP=k["interp"],
        Q_FLOOR=Q_FLOOR_BITS,
        LOG_FLOOR=LOG_FLOOR_BITS,
        **opts,
    )
    n = nb
    for g in _combine_plan(nb):
        n_out = n // g
        final = n_out == 1
        s2 = s if final else torch.empty((L, n_out), dtype=torch.float32, device=src.device)
        e2 = e if final else torch.empty((L, n_out), dtype=torch.float32, device=src.device)
        k["combine"][(L, n_out)](
            s,
            e,
            s2,
            e2,
            tot,
            out,
            n,
            n_out,
            G=g,
            LEVELS=g.bit_length() - 1,
            FINAL=final,
            EPI=epi,
            IDENT=V == 1,
            INTERP=k["interp"],
            TINY=TINY_BITS,
            **opts,
        )
        s, e, n = s2, e2, n_out


def compensated_row_sum(x: torch.Tensor, *, block: int | None = None) -> torch.Tensor:
    """``[L, V] -> [L]``, bit-identical to ``semiring.compensated_sum(x)`` (criterion 3) whatever ``block``."""
    L, _ = _check(x)
    out = torch.empty((L,), dtype=torch.float32, device=x.device)
    if L:
        _row_reduce(x, out, block, _EPI_SUM, None)
    return out


def hq_tail(q_raw: torch.Tensor, *, block: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The tail of ``constrained_marginals`` from ``q_raw``: ``(q, entropy)``, ``q`` written into ``q_raw``."""
    L, _ = _check(q_raw)
    total = torch.empty((L,), dtype=torch.float32, device=q_raw.device)
    entropy = torch.empty((L,), dtype=torch.float32, device=q_raw.device)
    if L:
        _row_reduce(q_raw, total, block, _EPI_TOTAL, None)
        _row_reduce(q_raw, entropy, block, _EPI_ENTROPY, total)
    return q_raw, entropy
