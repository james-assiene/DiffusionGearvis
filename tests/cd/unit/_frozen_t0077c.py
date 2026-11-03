"""Shared helpers for T-0077c's rows (the fused ``log_matmul``, plain and compensated; ``docs/tasks/T-0077c.md``).

API ruling (test-author, 2026-10-05; K1 precedent — to be confirmed by the orchestrator). Module root
``plugin/astra_vllm_plugin/astra_vllm_plugin/cd/``:

``cd/kernels/logmatmul.py`` (host entry; triton imported only inside function bodies, M1/K3)
  ``log_matmul(a3, b3, *, compensated=False, block_n=None, block_k=None) -> Tensor``: ``a3`` fp32 ``[nb, n, m]``,
  ``b3`` fp32 ``[nb, m, k]``, any strides (live-block views with row stride B > S, a broadcast batch with batch stride
  0), on CUDA, or on CPU under ``TRITON_INTERPRET=1``; returns a new fp32 ``[nb, n, k]`` tensor, the spec order's bits
  (``spec_log_matmul`` below). ``block_n``/``block_k``: the output tile, a power of two ≥ 1; a block larger than the
  next power of two of ``n`` (resp. ``k``) is clamped to it; ``None`` is the production choice. The bits never depend
  on the blocks (criterion 3b).
``cd/kernels/backend.py``
  ``logmatmul_layout_ok(a3, b3) -> bool``: pure, reads only dtype/shape/dim: True exactly when both are fp32, both
  3-D, ``a3.shape[0] == b3.shape[0]``, ``a3.shape[2] == b3.shape[1]`` and every dim ≥ 1 (criterion 8's guard).
  ``use_logmatmul(a3, b3, *, kernels: bool) -> bool``: the device-level dispatch of the second family (device first,
  then ``kernels``, triton, the layout guard); a layout refusal is logged once (its own record, not the decision).
  ``use_kernels(t, *, kernels)`` (T-0077b's) stays as it is. The decision log is **one** record per process across
  both families (O7) on ``astra_vllm_plugin.cd.kernels``, naming the family that decided first: the ``log_matmul``
  family's record contains ``log_matmul``, the H(q) family's ``H(q)`` (its T-0077b spelling); CPU → "cpu";
  dispatched → "triton" and "3.7.1"; never ``ASTRA_CD ``-prefixed.
``cd/semiring.py``
  ``log_matmul(a, b, *, compensated=False, kernels: bool = True)``, ``kernels`` keyword-only.

Whether a call dispatched is observed with ``sys.setprofile`` (``entry_calls``): any Python frame whose code lives in
``cd/kernels/logmatmul.py`` counts, whatever the implementation names it.

**The spec order** (task "Implementation notes", transcribed here from the text, not from a kernel):
``spec_log_matmul``. Its exp/log are parameters: torch's on the device for CUDA (criterion 3a), numpy fp64 rounded
once to fp32 for the interpreter (ruling O2, ``np64_exp``/``np64_log``). The **power controls** (``MUTANTS``) are the
realistic wrong orders of the task's list (i)-(ix); ``tests/cd/unit/test_cd_logmatmul_spec.py`` shows each differs
bitwise from the spec on a pinned instance and the precision ones fail a bar.

The frozen path is ``tests/cd/oracle/core_frozen_t0077c/`` (byte-identical copies of ``cd/{semiring,tree,emission,
marginals,mapdecode,draw}.py`` and ``cd/kernels/{__init__,backend,entropy}.py`` at b53457a; ``MANIFEST.md``).

Regime: CPU rows torch 2.13.0+cpu, numpy 2.4.6 (``uv.lock``), fp32; T-0065's operand generator (outer 64, 30 %
sentinels), seeds 0 and 1.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib
import math
import struct
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from tests.cd.unit import _frozen_t0077 as fz
from tests.cd.unit import _frozen_t0077b as kb
from tests.cd.unit._core_support import CD_DIR, PLUGIN_SRC

FROZEN_PKG = "tests.cd.oracle.core_frozen_t0077c"
FROZEN_DIR = Path(__file__).resolve().parents[1] / "oracle" / "core_frozen_t0077c"
FROZEN_COMMIT = "b53457a"
#: frozen module (dotted under FROZEN_PKG) -> sha256 of the blob at b53457a (MANIFEST.md)
FROZEN_SHA256 = {
    "semiring": "0b4ae05f7de105c3ea0642acd89876ad1bd14f725494342aa3a6d1d48577d00c",
    "tree": "45017614e056dbc2be2f42f9dfa683adc65449998f8a3d7971dd9a7a52ac1a51",
    "emission": "aaff33c4aa9f0877d4f2800276e48323304a6de40eef0c7e29347f1f8f95ed27",
    "marginals": "a85c404f054b60f5a552e6b6bffd7c58633a209abac5a91edc8bc71e6b6adc0d",
    "mapdecode": "1d13fb06c90e7760080742892c50d492214be9d9026f08f8f5a7a92890ce2ee5",
    "draw": "e45e8899f59d9fb8db0bf1f78a2f4749cf83f5173b2960a5ffb970b3409b9022",
    "kernels.__init__": "99a754b62d55422b55107fd27b8408014c7e1c9e126275b94f1970ca78e558b0",
    "kernels.backend": "14146a02723c3fac513699103cacdd287649bb1785d9ab31353de1220926346c",
    "kernels.entropy": "db0302a382d8b3cec8578a21d181626682f8be5d97419c2ebfcc9c292ebf0847",
}
KERNELS_DIR = CD_DIR / "kernels"
LOGGER = "astra_vllm_plugin.cd.kernels"
TRITON_VERSION = "3.7.1"
INSTANCES = fz.INSTANCES

#: fp32 bit patterns of semiring's constants (SENTINEL = -1e30, DEAD = SENTINEL / 2)
F32 = torch.float32
SENTINEL32 = struct.unpack("<f", struct.pack("<f", -1e30))[0]
DEAD32 = struct.unpack("<f", struct.pack("<f", -1e30 / 2.0))[0]
ULP1 = 2.0**-23  # fp32 ulp of 1.0
CHUNK_BYTES = (1 << 22) * 4  # one CHUNK_ELEMS fp32 chunk (16 MiB)


def frozen(name: str) -> ModuleType:
    """``tests.cd.oracle.core_frozen_t0077c.<name>`` (``kernels.backend`` etc. for the frozen kernels)."""
    return importlib.import_module(f"{FROZEN_PKG}.{name}")


def frozen_path(name: str) -> Path:
    return FROZEN_DIR / (name.replace(".", "/") + ".py")


def plugin_rel(name: str) -> str:
    return f"plugin/astra_vllm_plugin/astra_vllm_plugin/cd/{name.replace('.', '/')}.py"


def live(name: str) -> ModuleType:
    """``astra_vllm_plugin.cd.<name>``; a missing module is an ``AssertionError`` ("not implemented"), not an
    import error at collection."""
    if str(PLUGIN_SRC) not in sys.path:
        sys.path.insert(0, str(PLUGIN_SRC))
    try:
        return importlib.import_module(f"astra_vllm_plugin.cd.{name}")
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"T-0077c not implemented: astra_vllm_plugin.cd.{name} ({exc})"
        ) from exc


def entry(module: str, name: str) -> Any:
    """``getattr(live(module), name)``; absent → ``AssertionError`` naming this file's API ruling."""
    fn = getattr(live(module), name, None)
    assert fn is not None, (
        f"T-0077c not implemented: cd.{module}.{name} (API ruling, _frozen_t0077c)"
    )
    return fn


def lm_entry() -> Callable[..., torch.Tensor]:
    return entry("kernels.logmatmul", "log_matmul")


def require_kernels_keyword() -> None:
    import inspect

    p = inspect.signature(live("semiring").log_matmul).parameters.get("kernels")
    assert p is not None, (
        "T-0077c not implemented: semiring.log_matmul has no keyword-only `kernels` (criterion 8)"
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


bit_diff = kb.bit_diff
max_ulp = kb.max_ulp
sha256_bits = kb.sha256_bits
TRITON_BLOCKER = kb.TRITON_BLOCKER


# --------------------------------------------------------------------------- exp/log routes


def np64_exp(x: torch.Tensor) -> torch.Tensor:
    """Ruling O2: numpy's fp64 ``exp`` of the fp32 value, rounded once to fp32 (the interpreter's route)."""
    with np.errstate(all="ignore"):
        y = np.exp(x.detach().cpu().numpy().astype(np.float64)).astype(np.float32)
    return torch.from_numpy(y).to(x.device)


def np64_log(x: torch.Tensor) -> torch.Tensor:
    """Ruling O2: numpy's fp64 ``log`` of the fp32 value, rounded once to fp32."""
    with np.errstate(all="ignore"):
        y = np.log(x.detach().cpu().numpy().astype(np.float64)).astype(np.float32)
    return torch.from_numpy(y).to(x.device)


# --------------------------------------------------------------------------- the spec order (and its mutants)


def _const(v: float, like: torch.Tensor) -> torch.Tensor:
    return torch.full((), v, dtype=F32, device=like.device)


def _two_sum(s: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Spec step 3's TwoSum: ``s' = fl(s + x)``, ``z = fl(s' − s)``, ``e = fl(fl(s − fl(s' − z)) + fl(x − z))``."""
    s2 = s + x
    z = s2 - s
    e = (s - (s2 - z)) + (x - z)
    return s2, e


def _finish(
    mx: torch.Tensor,
    live: torch.Tensor,
    safe: torch.Tensor,
    S: torch.Tensor,
    compensated: bool,
    log: Callable[[torch.Tensor], torch.Tensor],
    exp: Callable[[torch.Tensor], torch.Tensor],
    residual: bool = True,
) -> torch.Tensor:
    """Spec step 5: ``ls = log(S_tot)``; compensated: TwoSum(safe, ls) → ``hi + (lo + (fl(S_tot·exp(−ls)) − 1))``;
    plain: ``safe + ls``; then ``where(live, ·, SENTINEL)``."""
    ls = log(S)
    if compensated:
        hi = safe + ls
        bb = hi - safe
        lo = (safe - (hi - bb)) + (ls - bb)
        # residual=False is mutant (v): the residual on log s dropped
        out = hi + (lo + (S * exp(-ls) - _const(1.0, S))) if residual else hi + lo
    else:
        out = safe + ls
    return torch.where(live, out, _const(SENTINEL32, out).expand_as(out))


def _check3(a3: torch.Tensor, b3: torch.Tensor) -> tuple[int, int, int, int]:
    assert a3.dim() == 3 and b3.dim() == 3, (a3.shape, b3.shape)
    assert a3.dtype == F32 and b3.dtype == F32, (a3.dtype, b3.dtype)
    nb, n, m = a3.shape
    nb2, m2, k = b3.shape
    assert nb == nb2 and m == m2, (a3.shape, b3.shape)
    return nb, n, m, k


def _terms(
    a3: torch.Tensor,
    b3: torch.Tensor,
    j: int,
    safe: torch.Tensor,
    compensated: bool,
    ab_err: bool,
    exp: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Spec step 3's ``x_j`` for every output at once: compensated ``exp(fl(fl(t − safe) + err))`` with
    ``bb = fl(t − a)``, ``err = fl(fl(a − fl(t − bb)) + fl(b − bb))`` (semiring's formulas); plain ``exp(fl(t − safe))``."""
    av = a3[:, :, j : j + 1]
    bv = b3[:, j : j + 1, :]
    t = av + bv
    if compensated:
        if ab_err:
            bb = t - av
            err = (av - (t - bb)) + (bv - bb)
            return exp((t - safe) + err)
        return exp((t - safe) + torch.zeros_like(t))  # mutant (iv): the A+B error dropped
    return exp(t - safe)


def _anchor(a3: torch.Tensor, b3: torch.Tensor) -> torch.Tensor:
    """Spec step 1: ``max_j fl(a_ij + b_jk)``, NaN-propagating (``torch.maximum``); any order (max is exact)."""
    mx = a3[:, :, 0:1] + b3[:, 0:1, :]
    for j in range(1, a3.shape[2]):
        mx = torch.maximum(mx, a3[:, :, j : j + 1] + b3[:, j : j + 1, :])
    return mx


def spec_log_matmul(
    a3: torch.Tensor,
    b3: torch.Tensor,
    *,
    compensated: bool,
    exp: Callable[[torch.Tensor], torch.Tensor] = torch.exp,
    log: Callable[[torch.Tensor], torch.Tensor] = torch.log,
) -> torch.Tensor:
    """The spec order of ``docs/tasks/T-0077c.md`` ("Implementation notes", steps 1-5), in torch fp32 elementwise ops
    (no contraction: every op is its own rounding), on ``a3``'s device. ``[nb, n, m] ⊗ [nb, m, k] -> [nb, n, k]``.

    1. ``mx = max_j t_j``, ``t_j = fl(a + b)``, NaN-propagating;
    2. ``live = mx > DEAD``, ``safe = live ? mx : 0`` (DEAD, SENTINEL as fp32);
    3. j = 0, 1, …, m − 1 ascending from ``s = c = +0.0``: ``x_j`` (``_terms``), then TwoSum ``s' = s + x``,
       ``z = s' − s``, ``e = (s − (s' − z)) + (x − z)``, ``c = c + e``, ``s = s'``;
    4. ``S_tot = fl(s + c)``;
    5. ``_finish`` (``semiring._finish`` op for op).
    """
    return emulate(a3, b3, compensated=compensated, exp=exp, log=log)


def emulate(
    a3: torch.Tensor,
    b3: torch.Tensor,
    *,
    compensated: bool,
    exp: Callable[[torch.Tensor], torch.Tensor] = torch.exp,
    log: Callable[[torch.Tensor], torch.Tensor] = torch.log,
    carry: bool = True,
    ab_err: bool = True,
    residual: bool = True,
    descending: bool = False,
    pad_zero_to: int | None = None,
) -> torch.Tensor:
    """``spec_log_matmul`` with one knob per mutant (defaults = the spec). ``carry=False``: (i) plain sequential;
    ``ab_err=False``: (iv); ``residual=False``: (v); ``descending``: (viii); ``pad_zero_to``: (ix) the inner axis padded
    to that length with lanes loaded as 0.0 (in both operands, both passes)."""
    _check3(a3, b3)
    if pad_zero_to is not None and pad_zero_to > a3.shape[2]:
        p = pad_zero_to - a3.shape[2]
        a3 = torch.cat(
            [a3, torch.zeros(a3.shape[0], a3.shape[1], p, dtype=F32, device=a3.device)], 2
        )
        b3 = torch.cat(
            [b3, torch.zeros(b3.shape[0], p, b3.shape[2], dtype=F32, device=b3.device)], 1
        )
    m = a3.shape[2]
    mx = _anchor(a3, b3)
    live = mx > _const(DEAD32, mx)
    safe = torch.where(live, mx, torch.zeros_like(mx))
    s = torch.zeros_like(mx)
    c = torch.zeros_like(mx)
    order = range(m - 1, -1, -1) if descending else range(m)
    for j in order:
        x = _terms(a3, b3, j, safe, compensated, ab_err, exp)
        if carry:
            s, e = _two_sum(s, x)
            c = c + e
        else:
            s = s + x
    S = s + c if carry else s
    return _finish(mx, live, safe, S, compensated, log, exp, residual=residual)


def mutant_torch_sum(
    a3: torch.Tensor, b3: torch.Tensor, *, compensated: bool, **_: Any
) -> torch.Tensor:
    """(ii) torch's ``sum(dim=-2)``: the frozen (branch-base) ``semiring.log_matmul`` itself."""
    return frozen("semiring").log_matmul(a3, b3, compensated=compensated)


def _anchor_safe(
    a3: torch.Tensor, b3: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mx = _anchor(a3, b3)
    live = mx > _const(DEAD32, mx)
    return mx, live, torch.where(live, mx, torch.zeros_like(mx))


def mutant_pair_tree(
    a3: torch.Tensor,
    b3: torch.Tensor,
    *,
    compensated: bool,
    exp: Callable[[torch.Tensor], torch.Tensor] = torch.exp,
    log: Callable[[torch.Tensor], torch.Tensor] = torch.log,
) -> torch.Tensor:
    """(iii) the spec's terms summed by a plain adjacent-pair tree over m (pad to a power of two with +0.0, then
    ``x[2i] + x[2i+1]`` per level; no error carried): what ``tl.sum`` over a j-tile gives."""
    mx, live, safe = _anchor_safe(a3, b3)
    xs = torch.stack(
        [_terms(a3, b3, j, safe, compensated, True, exp) for j in range(a3.shape[2])], -1
    )
    p = _next_pow2(xs.shape[-1])
    xs = torch.nn.functional.pad(xs, (0, p - xs.shape[-1]))
    while xs.shape[-1] > 1:
        xs = xs[..., 0::2] + xs[..., 1::2]
    return _finish(mx, live, safe, xs.squeeze(-1), compensated, log, exp)


def mutant_compensated_pair_tree(
    a3: torch.Tensor,
    b3: torch.Tensor,
    *,
    compensated: bool,
    exp: Callable[[torch.Tensor], torch.Tensor] = torch.exp,
    log: Callable[[torch.Tensor], torch.Tensor] = torch.log,
) -> torch.Tensor:
    """(iii-b, stronger than the task's list) the spec's terms summed by ``semiring.compensated_sum`` (adjacent-pair
    TwoSum tree carrying every error): as precise as the spec, so only near-tie inputs tell it apart."""
    mx, live, safe = _anchor_safe(a3, b3)
    xs = torch.stack(
        [_terms(a3, b3, j, safe, compensated, True, exp) for j in range(a3.shape[2])], -1
    )
    S = frozen("semiring").compensated_sum(xs)
    return _finish(mx, live, safe, S, compensated, log, exp)


def mutant_rowcol_anchor(
    a3: torch.Tensor,
    b3: torch.Tensor,
    *,
    compensated: bool,
    exp: Callable[[torch.Tensor], torch.Tensor] = torch.exp,
    log: Callable[[torch.Tensor], torch.Tensor] = torch.log,
) -> torch.Tensor:
    """(vi) separate anchors: ``rowmax_j a[i, j] + colmax_j b[j, k]`` instead of the pairwise ``max_j (a + b)``; the
    rest is the spec (carried sum, ``_finish``). A dominant term far below the foreign anchor underflows."""
    mx = a3.amax(dim=2, keepdim=True) + b3.amax(dim=1, keepdim=True)
    live = mx > _const(DEAD32, mx)
    safe = torch.where(live, mx, torch.zeros_like(mx))
    s = torch.zeros_like(mx)
    c = torch.zeros_like(mx)
    for j in range(a3.shape[2]):
        s, e = _two_sum(s, _terms(a3, b3, j, safe, compensated, True, exp))
        c = c + e
    return _finish(mx, live, safe, s + c, compensated, log, exp)


def mutant_online(
    a3: torch.Tensor,
    b3: torch.Tensor,
    *,
    compensated: bool,
    exp: Callable[[torch.Tensor], torch.Tensor] = torch.exp,
    log: Callable[[torch.Tensor], torch.Tensor] = torch.log,
) -> torch.Tensor:
    """(vii) single-pass online-rescaled sum: a running anchor; when it grows, ``s`` and ``c`` are scaled by
    ``exp(old − new)``; terms TwoSum-added as in the spec. Same anchor at the end, a different sum."""
    m = a3.shape[2]
    run = None
    s = c = None
    for j in range(m):
        t = a3[:, :, j : j + 1] + b3[:, j : j + 1, :]
        new = t if run is None else torch.maximum(run, t)
        live = new > _const(DEAD32, new)
        safe = torch.where(live, new, torch.zeros_like(new))
        if run is None:
            s = torch.zeros_like(new)
            c = torch.zeros_like(new)
        else:
            old_live = run > _const(DEAD32, run)
            old_safe = torch.where(old_live, run, torch.zeros_like(run))
            r = exp(old_safe - safe)
            s, c = s * r, c * r
        s, e = _two_sum(s, _terms(a3, b3, j, safe, compensated, True, exp))
        c = c + e
        run = new
    assert run is not None
    live = run > _const(DEAD32, run)
    safe = torch.where(live, run, torch.zeros_like(run))
    return _finish(run, live, safe, s + c, compensated, log, exp)


def _next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


#: the task's power controls (i)-(ix): name -> f(a3, b3, *, compensated, exp, log)
MUTANTS: dict[str, Callable[..., torch.Tensor]] = {
    "i_plain_sequential": functools.partial(emulate, carry=False),
    "ii_torch_sum": mutant_torch_sum,
    "iii_pair_tree": mutant_pair_tree,
    "iii_b_compensated_pair_tree": mutant_compensated_pair_tree,
    "iv_ab_error_dropped": functools.partial(emulate, ab_err=False),
    "v_residual_dropped": functools.partial(emulate, residual=False),
    "vi_rowcol_anchors": mutant_rowcol_anchor,
    "vii_online_rescaled": mutant_online,
    "viii_j_descending": functools.partial(emulate, descending=True),
    "ix_masked_lanes_zero": lambda a3, b3, **kw: emulate(
        a3, b3, pad_zero_to=_next_pow2(a3.shape[2]), **kw
    ),
}


# --------------------------------------------------------------------------- oracles and bars


def lse_oracle(a3: torch.Tensor, b3: torch.Tensor) -> np.ndarray:
    """fp64 + ``math.fsum``: ``log Σ_j exp(a + b)`` per entry on the fp32-rounded inputs (the fp64 sum of two fp32
    values is exact); entries with no term above −1e29 → −1e30 (T-0065's ``_lse_oracle``, batched)."""
    a = a3.detach().cpu().double().numpy()
    b = b3.detach().cpu().double().numpy()
    t = a[:, :, :, None] + b[:, None, :, :]  # [nb, n, m, k]
    live = t > -1e29
    with np.errstate(all="ignore"):
        mx = np.where(live, t, -np.inf).max(axis=2)
        s = np.where(live, np.exp(t - mx[:, :, None, :]), 0.0)
    out = np.full(mx.shape, -1e30)
    for g, i, k in zip(*np.nonzero(np.isfinite(mx)), strict=True):
        out[g, i, k] = mx[g, i, k] + math.log(math.fsum(s[g, i, :, k]))
    return out


def ulp32(x: np.ndarray) -> np.ndarray:
    e = torch.tensor(x, dtype=torch.float32).abs()
    return (torch.nextafter(e, torch.tensor(float("inf"))) - e).double().numpy()


def wide_stats(got: torch.Tensor, exp: np.ndarray) -> tuple[float, float]:
    """T-0065's wide metric: (max ulp, share of entries > 0.51 ulp), ulp = fp32 ulp of the oracle value."""
    u = np.abs(got.detach().cpu().double().numpy() - exp) / ulp32(exp)
    return float(u.max()), float(np.mean(u > 0.51))


def gauged_stats(got: torch.Tensor, exp: np.ndarray) -> tuple[float, float]:
    """T-0065's gauged metric: (mean, max) of |got − oracle| in ulp(1.0) = 2^-23."""
    err = np.abs(got.detach().cpu().double().numpy() - exp) / ULP1
    return float(err.mean()), float(err.max())


def wide_ok(got: torch.Tensor, exp: np.ndarray) -> tuple[bool, str]:
    mx, frac = wide_stats(got, exp)
    return mx <= 1.0 and frac <= 0.05, f"max {mx:.3f} ulp, {frac:.2%} > 0.51 ulp (bar 1.0 / 5 %)"


def gauged_ok(got: torch.Tensor, exp: np.ndarray) -> tuple[bool, str]:
    mean, mx = gauged_stats(got, exp)
    return mean <= 0.75 and mx <= 2.5, f"mean {mean:.3f}, max {mx:.3f} ulp(1.0) (bar 0.75 / 2.5)"


# --------------------------------------------------------------------------- instances (CPU, fp32)

REGIMES = {"wide": (-24.0, -16.0), "gauged": (-3.35, -2.35)}
INNERS = (32, 91, 205, 400, 512)
SEEDS = (0, 1)
OUTER = 64


def t0065_operands(
    lo: float, hi: float, inner: int, seed: int, n: int = OUTER
) -> tuple[torch.Tensor, torch.Tensor]:
    """T-0065's generator (``test_core_compensated._operands``, same draws in the same order): ``[n, inner]`` and
    ``[inner, n]`` uniform in [lo, hi], 30 % sentinels each."""
    g = torch.Generator().manual_seed(seed)
    a = (lo + (hi - lo) * torch.rand(n, inner, generator=g)).float()
    b = (lo + (hi - lo) * torch.rand(inner, n, generator=g)).float()
    a[torch.rand(n, inner, generator=g) < 0.3] = -1e30
    b[torch.rand(inner, n, generator=g) < 0.3] = -1e30
    return a, b


@functools.cache
def regime_instance(regime: str, inner: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``[1, 64, inner] ⊗ [1, inner, 64]`` in T-0065's regime ``wide`` or ``gauged``."""
    lo, hi = REGIMES[regime]
    a, b = t0065_operands(lo, hi, inner, seed)
    return a.unsqueeze(0), b.unsqueeze(0)


@functools.cache
def regime_oracle(regime: str, inner: int, seed: int) -> np.ndarray:
    return lse_oracle(*regime_instance(regime, inner, seed))


def two_band(inner: int, seed: int, n: int = 48, k: int = 48) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``live_simple_106-63-0`` regime: a 0-nat tail band and genuine paths ≈ 850 nats below. Half of the inner
    states carry ``a ≈ 0`` and ``b ≈ −850``, the other half ``a ≈ −850`` and ``b ≈ 0``: every term is ≈ −850 while
    ``rowmax(a) + colmax(b) ≈ 0`` (a foreign anchor 850 nats above every term: exp underflows to 0). Column 0 has
    ``b ≈ −20`` on the first half instead (entries ≈ −20 beside terms ≈ −850: both bands in one product; kept away
    from 0, where the wide bar's relative ulp is not the regime's). 20 % sentinels."""
    g = torch.Generator().manual_seed(9000 + seed)
    half = inner // 2
    u = lambda *s: -torch.rand(*s, generator=g)  # noqa: E731  in (-1, 0]
    a = u(n, inner)
    b = u(inner, k)
    a[:, half:] -= 850.0
    b[:half, :] -= 850.0
    b[:half, 0] += 830.0
    a[torch.rand(n, inner, generator=g) < 0.2] = -1e30
    b[torch.rand(inner, k, generator=g) < 0.2] = -1e30
    return a.float().unsqueeze(0), b.float().unsqueeze(0)


def all_sentinel(
    inner: int, seed: int, n: int = 20, k: int = 18
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauged-range operands with all-sentinel rows of ``a`` (1, 7) and columns of ``b`` (0, 5, k − 1), and a row
    (3) whose live inner states never meet column 2's: those outputs must be exactly the sentinel."""
    g = torch.Generator().manual_seed(9100 + seed)
    a = -2.35 - torch.rand(2, n, inner, generator=g)
    b = -2.35 - torch.rand(2, inner, k, generator=g)
    a[:, [1, 7], :] = -1e30
    b[:, :, [0, 5, k - 1]] = -1e30
    half = max(1, inner // 2)
    a[:, 3, half:] = -1e30
    b[:, :half, 2] = -1e30
    return a.float(), b.float()


def nan_inf(inner: int, seed: int, n: int = 9, k: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """NaN, +inf and −inf in both operands at fixed positions (positions compared, not payloads)."""
    g = torch.Generator().manual_seed(9200 + seed)
    a = (-3.0 * torch.rand(1, n, inner, generator=g)).float()
    b = (-3.0 * torch.rand(1, inner, k, generator=g)).float()
    j = inner - 1
    a[0, 0, 0] = float("nan")
    a[0, 1, j] = float("inf")
    a[0, 2, j // 2] = float("-inf")
    b[0, 0, 3 % k] = float("nan")
    b[0, j, 4 % k] = float("inf")
    b[0, j // 2, 5 % k] = float("-inf")
    a[0, 4, :] = float("-inf")  # a whole row at -inf: no live term
    return a, b


def near_dead(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """``m = 2`` products whose anchor is exactly fp32(DEAD), the next fp32 above it and the next below (``mx > DEAD``
    must be the fp32 comparison: only the first is live), plus ordinary live entries."""
    up = float(torch.nextafter(torch.tensor(DEAD32), torch.tensor(0.0)))
    down = float(torch.nextafter(torch.tensor(DEAD32), torch.tensor(-math.inf)))
    a = torch.tensor([[[DEAD32, -1e30], [up, -1e30], [down, -1e30], [-1.0, -2.0]]], dtype=F32)
    b = torch.tensor([[[0.0, -0.5, -1e30], [-3.0, 0.0, -1e30]]], dtype=F32)
    return a, b


def live_block_views(
    h: int, S: int, B: int, seed: int, lo: float = -3.35, hi: float = -2.35
) -> tuple[torch.Tensor, torch.Tensor]:
    """``up_sweep``'s operands: ``big[0:2h:2, :S, :S]`` and ``big[1:2h:2, :S, :S]`` of a ``[2h, B, B]`` block (batch
    stride 2·B·B, row stride B > S), 25 % sentinels."""
    g = torch.Generator().manual_seed(9300 + seed)
    big = (lo + (hi - lo) * torch.rand(2 * h, B, B, generator=g)).float()
    big[torch.rand(2 * h, B, B, generator=g) < 0.25] = -1e30
    return big[0 : 2 * h : 2, :S, :S], big[1 : 2 * h : 2, :S, :S]


def vec_mat(p: int, S: int, B: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``marginals``' vec-mat ``[p, 1, S] ⊗ [p, S, S]`` on live-block views."""
    g = torch.Generator().manual_seed(9400 + seed)
    v = (-3.0 * torch.rand(p, B, generator=g)).float()
    m = (-3.0 * torch.rand(p, B, B, generator=g)).float()
    v[torch.rand(p, B, generator=g) < 0.25] = -1e30
    return v[:, :S].unsqueeze(-2), m[:, :S, :S]


def mat_vec(p: int, S: int, B: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``marginals``' mat-vec ``[p, S, S] ⊗ [p, S, 1]`` on live-block views."""
    g = torch.Generator().manual_seed(9500 + seed)
    m = (-3.0 * torch.rand(p, B, B, generator=g)).float()
    v = (-3.0 * torch.rand(p, B, generator=g)).float()
    m[torch.rand(p, B, B, generator=g) < 0.25] = -1e30
    return m[:, :S, :S], v[:, :S].unsqueeze(-1)


def broadcast_batch(
    nb: int, n: int, m: int, k: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``a`` ``[1, n, m]`` expanded to ``[nb, n, m]`` (batch stride 0), ``b`` ``[nb, m, k]``: what ``semiring``'s host
    broadcast hands the entry for ``[n, m] ⊗ [nb, m, k]``."""
    g = torch.Generator().manual_seed(9600 + seed)
    a = (-3.0 * torch.rand(1, n, m, generator=g)).float().expand(nb, n, m)
    b = (-3.0 * torch.rand(nb, m, k, generator=g)).float()
    return a, b


def near_tie(seed: int, n: int, m: int, live: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """``[1, n, m] ⊗ [1, m, 1]`` (``b = 0``): each row has ``live`` live inner states at random positions (the rest
    the sentinel, which adds exactly nothing), one at ``a = 0`` (the anchor: ``x = 1`` exactly) and the others at
    ``fp32(log v)``, ``v = 2^-e·(1 + k/4)``, e ∈ [18, 52]: terms with two mantissa bits over ~34 binades, so the carried
    error ``c`` itself rounds and exact ties are common. Smooth inputs give 0 differing entries in 82k between the
    spec and j-descending / a compensated pair tree (measured 2026-10-05); these rows are where an order shows."""
    g = torch.Generator().manual_seed(seed)
    e = torch.randint(18, 53, (1, n, m), generator=g).double()
    k = torch.randint(0, 4, (1, n, m), generator=g).double()
    a = torch.log(torch.exp2(-e) * (1 + k / 4)).float()
    keep = torch.argsort(torch.rand(n, m, generator=g), dim=1)[:, :live]
    mask = torch.zeros(n, m, dtype=torch.bool)
    mask.scatter_(1, keep, True)
    a = torch.where(mask[None], a, torch.full_like(a, SENTINEL32))
    a[0, torch.arange(n), keep[:, 0]] = 0.0
    return a, torch.zeros(1, m, 1, dtype=F32)


#: ``near_tie(seed, NEAR_TIE_SEARCH_N, m)[:, rows]``: rows on which j-descending, the compensated pair tree and the
#: online-rescaled sum each give other bits than the spec, with torch's CPU exp/log *and* the O2 route (numpy fp64
#: rounded once), compensated and plain. Seeded search, 2026-10-05, test-author (CPU, torch 2.13.0+cpu, numpy 2.4.6):
#: of 20000 rows, m 31: 1 / 1 / 2 rows (descending / compensated tree / online) differ under all four settings;
#: m 33: 4 / 4 / 2; m 91: 3 / 0 / 4. Up to three rows per control are pinned.
NEAR_TIE_SEARCH_N = 20000
NEAR_TIE_PINS: dict[int, tuple[int, tuple[int, ...]]] = {
    31: (31, (3723, 12210, 18329, 19300)),
    33: (33, (477, 8952, 11872, 18229)),
    91: (91, (1648, 3580, 5932, 8632, 9568, 10043)),
}


def near_tie_pinned(m: int) -> tuple[torch.Tensor, torch.Tensor]:
    seed, rows = NEAR_TIE_PINS[m]
    a, b = near_tie(seed, NEAR_TIE_SEARCH_N, m)
    return a[:, list(rows), :].contiguous(), b


# --------------------------------------------------------------------------- dispatch evidence


@contextlib.contextmanager
def entry_calls(module_file: str = "logmatmul.py") -> Iterator[list[str]]:
    """Every Python call into ``cd/kernels/<module_file>`` while the block runs (this thread), as ``co_name``s. Chains
    to the profiler it replaces, so it nests with other spies in either order."""
    target = str((KERNELS_DIR / module_file).resolve())

    def prof(frame: Any, event: str, arg: Any) -> None:
        if event == "call":
            fn = frame.f_code.co_filename
            if fn.endswith(module_file) and str(Path(fn).resolve()) == target:
                seen.append(frame.f_code.co_name)
        if prev is not None:  # chain to an enclosing profiler (Python keeps one per thread)
            prev(frame, event, arg)

    seen: list[str] = []
    prev = sys.getprofile()
    sys.setprofile(prof)
    try:
        yield seen
    finally:
        sys.setprofile(prev)
