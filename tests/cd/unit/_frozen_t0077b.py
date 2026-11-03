"""Shared helpers for T-0077b's rows (the fused H(q) tail of ``constrained_marginals``; ``docs/tasks/T-0077b.md``).

API ruling T-0077b K1 (test-author, 2026-10-04). The task names "the row-sum entry", "the tail entry" and "the
dispatch decision" without fixing their spelling; the rows fix it as follows (module root
``plugin/astra_vllm_plugin/astra_vllm_plugin/cd/``):

``cd/kernels/entropy.py`` (host entries; triton imported only inside function bodies, ruling M1)
  ``compensated_row_sum(x: Tensor, *, block: int | None = None) -> Tensor``: ``x`` a contiguous fp32 ``[L, V]``
  tensor (CUDA, or CPU under ``TRITON_INTERPRET=1``); returns ``[L]``, bit-identical to
  ``semiring.compensated_sum(x)`` (criterion 3; ``V = 1`` returns ``x[:, 0]``'s bits, ``-0.0`` included).
  ``block`` overrides the per-program block (a power of two ≥ 16; a block larger than the padded row is clamped
  to it); ``None`` is the production choice, a function of ``V`` only. The bits never depend on ``block``.
  ``hq_tail(q_raw: Tensor, *, block: int | None = None) -> tuple[Tensor, Tensor]``: the tail of
  ``constrained_marginals`` after ``q_raw = _token_marginals(...)`` (task "Implementation notes", ops 1-8):
  returns ``(q, entropy)`` with ``q`` written **in place** into ``q_raw``'s storage (``q.data_ptr() ==
  q_raw.data_ptr()``) and ``entropy`` ``[L]``; on CUDA both bit-identical to the torch tail.
``cd/kernels/backend.py``
  ``dispatch_decision(*, on_cuda: bool, triton_available: bool, kernels: bool) -> bool``: pure; True exactly
  when all three hold (criterion 7's truth table). The process-wide decision log is a record on the logger
  ``astra_vllm_plugin.cd.kernels``.
``cd/marginals.py``
  ``constrained_marginals(..., *, class_mass=None, kernels: bool = True)``; the tail still starts from the
  module-level name ``_token_marginals`` (criterion 10: everything before ``q_raw`` unchanged), which the
  adversarial rows replace to feed a chosen ``q_raw`` through the real call (``tail_via``).

Whether a call dispatched is observed with ``sys.setprofile`` (``entry_calls``): any Python frame whose code lives
in ``cd/kernels/entropy.py`` counts, whatever name the implementation binds it under.

Bit comparisons (``bit_diff``) are stricter than ``torch.equal``: they compare the fp32 bit patterns (so ``-0.0``
and ``+0.0`` differ) and NaN *positions* (not payloads, task "Tests to write").

The frozen path is ``tests/cd/oracle/core_frozen_t0077b/`` (byte-identical copies of ``cd/{semiring,tree,
emission,marginals,mapdecode,draw}.py`` at 2d90189; ``MANIFEST.md``).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import math
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from tests.cd.oracle.enumerate import Toy
from tests.cd.unit import _frozen_t0077 as fz
from tests.cd.unit._core_support import CD_DIR, PLUGIN_SRC

FROZEN_PKG = "tests.cd.oracle.core_frozen_t0077b"
FROZEN_DIR = Path(__file__).resolve().parents[1] / "oracle" / "core_frozen_t0077b"
FROZEN_COMMIT = "2d90189"
FROZEN_MODULES = ("semiring", "tree", "emission", "marginals", "mapdecode", "draw")
#: sha256 of each blob at 2d90189 (MANIFEST.md); the frozen files must still hash to these.
FROZEN_SHA256 = {
    "semiring": "0b4ae05f7de105c3ea0642acd89876ad1bd14f725494342aa3a6d1d48577d00c",
    "tree": "45017614e056dbc2be2f42f9dfa683adc65449998f8a3d7971dd9a7a52ac1a51",
    "emission": "aaff33c4aa9f0877d4f2800276e48323304a6de40eef0c7e29347f1f8f95ed27",
    "marginals": "4d9f2c34047fe458cb543f9051d98461aac3c28c7e6247996e460d7db9ec9402",
    "mapdecode": "1d13fb06c90e7760080742892c50d492214be9d9026f08f8f5a7a92890ce2ee5",
    "draw": "e45e8899f59d9fb8db0bf1f78a2f4749cf83f5173b2960a5ffb970b3409b9022",
}
MARGINAL_FIELDS = ("log_z", "q", "entropy", "log_a", "log_b")
KERNELS_DIR = CD_DIR / "kernels"
KERNEL_MODULES = ("__init__", "backend", "entropy")
LOGGER = "astra_vllm_plugin.cd.kernels"
TRITON_VERSION = "3.7.1"
INSTANCES = fz.INSTANCES


def frozen(name: str) -> ModuleType:
    return importlib.import_module(f"{FROZEN_PKG}.{name}")


def live(name: str) -> ModuleType:
    """``astra_vllm_plugin.cd.<name>``; a module that does not exist yet is an ``AssertionError`` (a red row that
    reads "not implemented", not a collection or import error)."""
    if str(PLUGIN_SRC) not in sys.path:
        sys.path.insert(0, str(PLUGIN_SRC))
    try:
        return importlib.import_module(f"astra_vllm_plugin.cd.{name}")
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"T-0077b not implemented: astra_vllm_plugin.cd.{name} ({exc})"
        ) from exc


def entry(module: str, name: str) -> Any:
    """``getattr(live(module), name)``, absent → ``AssertionError`` naming ruling K1's spelling."""
    mod = live(module)
    fn = getattr(mod, name, None)
    assert fn is not None, (
        f"T-0077b not implemented: cd.{module}.{name} (ruling K1, _frozen_t0077b)"
    )
    return fn


def _require_kernels_keyword(mg: ModuleType, kw: dict[str, Any]) -> None:
    if "kernels" in kw:
        import inspect

        params = inspect.signature(mg.constrained_marginals).parameters
        assert "kernels" in params, (
            "T-0077b not implemented: constrained_marginals has no keyword-only `kernels` (criterion 7, M6)"
        )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- bits


def _ordered(t: torch.Tensor) -> torch.Tensor:
    """fp32 bits as a monotone int64 (adjacent floats differ by 1; -0.0 and +0.0 by 1)."""
    i = t.contiguous().view(torch.int32).to(torch.int64)
    return torch.where(i < 0, -(i & 0x7FFFFFFF) - 1, i)


def bit_diff(want: torch.Tensor, got: torch.Tensor) -> str | None:
    """``None`` when ``got`` has ``want``'s shape, dtype, NaN positions and (elsewhere) bit patterns; else why not.

    The reason names the number of differing entries and the max ulp (criterion 2's report)."""
    if want.shape != got.shape or want.dtype != got.dtype:
        return f"shape/dtype {tuple(got.shape)} {got.dtype} != {tuple(want.shape)} {want.dtype}"
    a, b = want.detach().cpu(), got.detach().cpu()
    if not a.is_floating_point():
        n = int((a != b).sum())
        return None if n == 0 else f"{n} differing entries"
    if a.dtype != torch.float32:
        a, b = a.float(), b.float()
    na, nb = torch.isnan(a), torch.isnan(b)
    if not torch.equal(na, nb):
        return f"NaN positions differ ({int((na ^ nb).sum())} entries)"
    keep = ~na
    oa, ob = _ordered(a)[keep], _ordered(b)[keep]
    diff = oa != ob
    n = int(diff.sum())
    if n == 0:
        return None
    ulp = int((oa - ob).abs().max())
    return f"{n} of {int(keep.sum())} entries differ bitwise, max {ulp} ulp"


def max_ulp(want: torch.Tensor, got: torch.Tensor) -> int:
    """Max ulp distance over entries that are not NaN in both (NaN positions are checked separately)."""
    a, b = want.detach().cpu().float(), got.detach().cpu().float()
    keep = ~(torch.isnan(a) | torch.isnan(b))
    if not bool(keep.any()):
        return 0
    return int((_ordered(a)[keep] - _ordered(b)[keep]).abs().max())


def sha256_bits(t: torch.Tensor) -> str:
    """sha256 over the raw bytes of ``t`` (CPU copy, contiguous), plus its shape and dtype."""
    c = t.detach().cpu().contiguous()
    h = hashlib.sha256(f"{tuple(c.shape)} {c.dtype}".encode())
    # reshape(-1) first: a 0-dim tensor (log_z) has no byte view; the shape is in the header
    h.update(c.reshape(-1).view(torch.uint8).numpy().tobytes() if c.numel() else b"")
    return h.hexdigest()


# --------------------------------------------------------------------------- blocking triton in a subprocess

#: Python source run first in a subprocess: triton is unavailable, and every attempt to reach it is recorded.
#: ``sys.modules["triton"] = None`` makes ``importlib.util.find_spec("triton")`` return None (the probe torch's lazy
#: CUDA init makes, ``torch/cuda/__init__.py:_register_triton_kernels``, ruling K9) and ``import triton`` raise
#: ``ModuleNotFoundError``; ``__import__``, ``importlib.import_module`` and ``importlib.util.find_spec`` are wrapped
#: to record ``[name, caller file, how]`` in ``attempts`` and then delegate. ``blocker_check()`` (call it last: it
#: records its own attempts) says whether both halves of the block hold.
TRITON_BLOCKER = """
import builtins as _b, importlib as _il, importlib.util as _ilu, sys as _sys
attempts = []
_sys.modules["triton"] = None
def _rec(name, how, depth):
    if isinstance(name, str) and name.split(".")[0] == "triton":
        attempts.append([name, _sys._getframe(depth).f_code.co_filename, how])
_real_import, _real_im, _real_fs = _b.__import__, _il.import_module, _ilu.find_spec
def _imp(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0:
        _rec(name, "import", 2)
    return _real_import(name, globals, locals, fromlist, level)
def _im(name, package=None):
    _rec(name, "import_module", 2)
    return _real_im(name, package)
def _fs(name, package=None):
    _rec(name, "find_spec", 2)
    return _real_fs(name, package)
_b.__import__, _il.import_module, _ilu.find_spec = _imp, _im, _fs
def blocker_check():
    try:
        import triton  # noqa: F401
        raises = False
    except ImportError:
        raises = True
    return {"find_spec_none": _ilu.find_spec("triton") is None, "import_raises": raises}
"""


# --------------------------------------------------------------------------- dispatch evidence


@contextlib.contextmanager
def entry_calls() -> Iterator[list[str]]:
    """Every Python call into ``cd/kernels/entropy.py`` while the block runs (this thread), as ``co_name``s."""
    target = str((KERNELS_DIR / "entropy.py").resolve())
    seen: list[str] = []

    def prof(frame: Any, event: str, arg: Any) -> None:
        if event == "call":
            fn = frame.f_code.co_filename
            if fn.endswith("entropy.py") and str(Path(fn).resolve()) == target:
                seen.append(frame.f_code.co_name)

    prev = sys.getprofile()
    sys.setprofile(prof)
    try:
        yield seen
    finally:
        sys.setprofile(prev)


# --------------------------------------------------------------------------- the per-path pipeline


def tables_on(em: ModuleType, inst: fz.Inst, device: Any) -> Any:
    toy = inst.toy
    return em.build_tables(
        n_states=toy.n_states,
        vocab_size=toy.vocab_size,
        class_members=[sorted(stored) for stored, _ in toy.classes],
        class_is_neg=[neg for _, neg in toy.classes],
        edge_src=[e[0] for e in toy.edges],
        edge_dst=[e[1] for e in toy.edges],
        edge_class=[e[2] for e in toy.edges],
        finals=sorted(toy.finals),
        bucket=inst.bucket,
        device=device,
    )


def _mask(ids: Any, size: int, device: Any) -> torch.Tensor:
    m = torch.zeros(size, dtype=torch.bool)
    for s in ids:
        m[int(s)] = True
    return m.to(device)


def path_mods(which: str) -> dict[str, ModuleType]:
    get = frozen if which == "frozen" else live
    return {n: get(n) for n in ("emission", "tree", "marginals")}


def prepared(which: str, inst: fz.Inst, device: Any = "cpu") -> Callable[..., Any]:
    """Everything before ``constrained_marginals`` (tables, class mass, tree, masks) built now, on ``device``;
    the returned ``call(**kw)`` runs only ``constrained_marginals(..., **kw)`` (the sync-debug rows wrap just it)."""
    m = path_mods(which)
    em = m["emission"]
    tables = tables_on(em, inst, device)
    B = tables.bucket
    x = inst.logits.clone().to(device)
    a, b = _mask(inst.start, B, device), _mask(inst.end, B, device)
    cm = em.class_mass(x, tables)
    tree = m["tree"].up_sweep(em.log_transfer(x, tables, inst.l_eff, class_mass=cm))
    mg = m["marginals"]

    def call(**kw: Any) -> Any:
        _require_kernels_keyword(mg, kw)
        return mg.constrained_marginals(x, tables, tree, a, b, inst.l_eff, class_mass=cm, **kw)

    return call


def marginals_on(which: str, inst: fz.Inst, device: Any = "cpu", **kw: Any) -> Any:
    """``class_mass`` → ``log_transfer`` → ``up_sweep`` → ``constrained_marginals`` on one path's own modules.

    ``kw`` goes to ``constrained_marginals`` only (``kernels=`` on the live path)."""
    return prepared(which, inst, device)(**kw)


def field_diffs(want: Any, got: Any, fields: tuple[str, ...] = MARGINAL_FIELDS) -> dict[str, str]:
    out = {}
    for f in fields:
        d = bit_diff(getattr(want, f), getattr(got, f))
        if d is not None:
            out[f] = d
    return out


# --------------------------------------------------------------------------- a chosen q_raw through the real call


def tail_toy(V: int) -> Toy:
    """One final state looping on one positive class holding every id: any ``q_raw`` is a legal tail input."""
    return Toy(1, V, ((frozenset(range(V)), False),), ((0, 0, 0),), frozenset({0}))


@contextlib.contextmanager
def _token_marginals_returning(mg: ModuleType, q_raw: torch.Tensor) -> Iterator[None]:
    saved = mg._token_marginals

    def fake(logits: Any, tables: Any, cm: Any, coef: Any) -> torch.Tensor:
        assert tuple(logits.shape) == tuple(q_raw.shape), (logits.shape, q_raw.shape)
        assert logits.device == q_raw.device, (logits.device, q_raw.device)
        return q_raw.clone()

    mg._token_marginals = fake
    try:
        yield
    finally:
        mg._token_marginals = saved


def prepared_tail(which: str, q_raw: torch.Tensor, device: Any = "cpu") -> Callable[..., Any]:
    """``tail_via``'s inputs built now; ``call(**kw)`` patches ``_token_marginals`` and runs only
    ``constrained_marginals`` (``q_raw`` is moved to ``device`` here, so the call itself copies device to device)."""
    L, V = q_raw.shape
    m = path_mods(which)
    em = m["emission"]
    toy = tail_toy(V)
    inst = fz.Inst("tail", toy, 1, torch.zeros(L, V), (0,), (0,), L, 0)
    tables = tables_on(em, inst, device)
    x = torch.zeros(L, V, dtype=torch.float32, device=device)
    a, b = _mask((0,), 1, device), _mask((0,), 1, device)
    cm = em.class_mass(x, tables)
    tree = m["tree"].up_sweep(em.log_transfer(x, tables, L, class_mass=cm))
    mg = m["marginals"]
    src = q_raw.to(device=device, dtype=torch.float32)

    def call(**kw: Any) -> Any:
        _require_kernels_keyword(mg, kw)
        with _token_marginals_returning(mg, src):
            return mg.constrained_marginals(x, tables, tree, a, b, L, class_mass=cm, **kw)

    return call


def tail_via(which: str, q_raw: torch.Tensor, device: Any = "cpu", **kw: Any) -> Any:
    """``constrained_marginals`` on a 1-state, 1-class toy of width ``V``, with ``_token_marginals`` returning a
    copy of ``q_raw`` (``[L, V]`` fp32): the frozen path's ``q``/``entropy`` are then exactly the torch tail
    of the branch base applied to ``q_raw`` (no transcription of it here)."""
    return prepared_tail(which, q_raw, device)(**kw)


# --------------------------------------------------------------------------- adversarial inputs (CPU, fp32)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _log_uniform(L: int, V: int, lo: float, hi: float, seed: int) -> torch.Tensor:
    u = torch.rand(L, V, generator=_gen(seed), dtype=torch.float64)
    return torch.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))).to(torch.float32)


def contraction_row(L: int = 4, V: int = 256, seed: int = 31) -> torch.Tensor:
    """q̃ log-uniform in [1e-6, 50]: products ``c·q̃`` whose exact (FMA) and rounded forms differ in the first
    TwoSum level (``fma_sensitive`` is the control)."""
    return _log_uniform(L, V, 1e-6, 50.0, seed)


def subnormal_row(L: int = 3, V: int = 1000, seed: int = 37) -> torch.Tensor:
    """Every q̃ in (0, 1e-38] (fp32 subnormals and the normal edge), total ≪ 1e-8: catches FTZ."""
    x = _log_uniform(L, V, 2e-45, 1e-38, seed)
    return torch.where(x > 0, x, torch.full_like(x, 1.4e-45))


def tail_cases(wide: bool = False) -> dict[str, torch.Tensor]:
    """The adversarial ``q_raw`` set (task "Tests to write", adversarial): ``[L, V]`` fp32, CPU."""
    cases: dict[str, torch.Tensor] = {}
    # q̃ > 1 and q̃ < 1 in one row: the terms q̃·log q̃ have both signs
    cases["mixed_sign"] = _log_uniform(4, 300, math.exp(-4.0), math.exp(4.0), 11)
    # q̃ = 1 ± δ: terms ±δ, a sum that cancels to a small fraction of its terms
    d = (torch.rand(4, 513, generator=_gen(12), dtype=torch.float64) * 2 - 1) * 1e-3
    cases["cancellation"] = (1.0 + d).to(torch.float32)
    # values spanning 1e-30 .. 1e3
    cases["span_1e-30_1e3"] = _log_uniform(4, 777, 1e-30, 1e3, 13)
    # an all-zero row: total clamps to fp32 tiny
    z = _log_uniform(3, 64, 1e-3, 10.0, 14)
    z[1] = 0.0
    cases["all_zero_row"] = z
    # V not a power of two; an all-forbidden row (q̃ = 0 everywhere) and forbidden tokens elsewhere
    f = _log_uniform(3, 100, 1e-4, 4.0, 15)
    f[0] = 0.0
    f[2, ::3] = 0.0
    cases["npot_forbidden_row"] = f
    # NaN (and +inf) in q_raw: the NaN positions of q and entropy must match the torch path's
    n = _log_uniform(3, 50, 1e-3, 3.0, 16)
    n[1, 7] = float("nan")
    n[2, 0] = float("inf")
    cases["nan_row"] = n
    cases["subnormal"] = subnormal_row()
    cases["contraction"] = contraction_row()
    if wide:
        cases["wide_v262144"] = _log_uniform(4, 262_144, 1e-12, 30.0, 17)
    return cases


def row_sum_cases() -> dict[str, torch.Tensor]:
    """Adversarial inputs of the row sum itself (signed, cancelling, wide-range)."""
    cases: dict[str, torch.Tensor] = {}
    g = _gen(21)
    mag = torch.exp(torch.randn(6, 1000, generator=g, dtype=torch.float64) * 8.0)
    sign = torch.where(torch.rand(6, 1000, generator=g) < 0.5, -1.0, 1.0).to(torch.float64)
    cases["signed_wide"] = (mag * sign).to(torch.float32)
    # large cancellation: x, -x pairs (not adjacent) plus small residues
    base = _log_uniform(4, 512, 1e-2, 1e3, 22)
    perm = torch.randperm(512, generator=_gen(23))
    c = torch.cat([base, -base[:, perm]], dim=1)
    c[:, ::97] += 1e-3
    cases["cancelling_pairs"] = c
    cases["span_1e-30_1e3"] = _log_uniform(3, 999, 1e-30, 1e3, 24)
    # halves pairing *with* the full TwoSum error tree (reviewer A, finding 1): pinned rows where it differs
    for V, (seed, rows) in HALVES_PINS.items():
        cases[f"halves_tie_v{V}"] = near_tie(seed, HALVES_SEARCH_L, V)[list(rows)]
    return cases


#: ``near_tie(seed, 2000, V)[rows]``: rows on which a compensated tree that pairs halves (block 1024 = production,
#: and block 16, i.e. halves inside 16-lane blocks then halves across the partials) gives other bits than
#: ``compensated_sum``. Found by a seeded search (2026-10-05, test-author; CPU, torch 2.13.0+cpu): 30/21 of 2000
#: rows at V 64, 26/23 at V 100, 15/10 at V 1000 differ at block production/16; rows differing at both are
#: pinned. Smooth distributions (``randn·exp(20·randn)``, cancelling pairs, log-uniform) give 0 in 2000: the
#: compensated sum is nearly correctly rounded and only near-ties expose the error tree's order.
HALVES_SEARCH_L = 2000
HALVES_PINS: dict[int, tuple[int, tuple[int, ...]]] = {
    64: (51, (44, 77, 238, 264)),
    100: (52, (83, 211, 294, 475)),
    1000: (53, (585, 636, 950, 1284)),
}


def near_tie(seed: int, L: int, V: int) -> torch.Tensor:
    """Signed values ``±2^e·(1 + k/4)``, e ∈ [-50, 0], k ∈ 0..3 (two mantissa bits: exact ties are common), 20 %
    of them ``±2^20``: TwoSum errors over ~70 binades whose plain-added error tree rounds by order."""
    g = _gen(seed)
    sign = torch.where(torch.rand(L, V, generator=g) < 0.5, -1.0, 1.0).to(torch.float64)
    e = torch.randint(-50, 1, (L, V), generator=g).to(torch.float64)
    m = 1 + torch.randint(0, 4, (L, V), generator=g).to(torch.float64) / 4
    x = sign * torch.exp2(e) * m
    big = torch.rand(L, V, generator=g) < 0.2
    return torch.where(big, sign * 2.0**20, x).to(torch.float32)


# --------------------------------------------------------------------------- controls (pure torch / numpy)


def adjacent_pairs_no_error(x: torch.Tensor) -> torch.Tensor:
    """(i) the same adjacent-pair tree as ``compensated_sum`` with the TwoSum error dropped."""
    n = x.shape[-1]
    p = 1 << max(0, (n - 1).bit_length())
    x = torch.nn.functional.pad(x, (0, p - n))
    while x.shape[-1] > 1:
        x = x[..., 0::2] + x[..., 1::2]
    return x.squeeze(-1)


def compensated_halves(x: torch.Tensor) -> torch.Tensor:
    """(iv) ``compensated_sum``'s TwoSum error tree, but pairing ``(x[i], x[i + n/2])`` at every level (the order
    ``tl.reshape(v, (2, n/2))`` + permute + split gives): only the pairing differs from the spec."""
    n = x.shape[-1]
    p = 1 << max(0, (n - 1).bit_length())
    x = torch.nn.functional.pad(x, (0, p - n))
    if p == 1:
        return x.squeeze(-1)
    sem = frozen("semiring")
    h = p // 2
    s, err = sem.two_sum(x[..., :h], x[..., h:])
    while s.shape[-1] > 1:
        h = s.shape[-1] // 2
        s, e = sem.two_sum(s[..., :h], s[..., h:])
        err = (err[..., :h] + err[..., h:]) + e
    return (s + err).squeeze(-1)


def halves_pairing(x: torch.Tensor) -> torch.Tensor:
    """(iii) ``x[:n/2] + x[n/2:]`` repeatedly (a typical ``tl.reduce`` order), no error term."""
    n = x.shape[-1]
    p = 1 << max(0, (n - 1).bit_length())
    x = torch.nn.functional.pad(x, (0, p - n))
    while x.shape[-1] > 1:
        h = x.shape[-1] // 2
        x = x[..., :h] + x[..., h:]
    return x.squeeze(-1)


def fma_sensitive(q_raw: torch.Tensor) -> int:
    """How many first-level pairs ``(2i, 2i+1)`` of the H-term sum change when the first product is contracted
    into the add (``fma(c, q̃, t)`` vs ``fl(fl(c·q̃) + t)``). Computed in fp64 from the fp32 operands, with the
    branch base's elementwise ops on CPU for ``c``. The fp64 product of two fp32 values is exact; the fp64 add
    may round once more before the fp32 rounding, so this is a sensitivity control, not an FMA oracle."""
    sem = frozen("semiring")
    q = q_raw.to(torch.float32)
    total = sem.compensated_sum(q).unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
    lt = torch.log(total)
    c = torch.maximum(torch.log(q.clamp_min(1e-38)), lt + math.log(1e-30))
    t = (c * q).numpy()
    c64, q64 = c.numpy().astype(np.float64), q.numpy().astype(np.float64)
    n = q.shape[-1] - q.shape[-1] % 2
    exact = c64[:, 0:n:2] * q64[:, 0:n:2]  # exact: 24 + 24 significant bits < 53
    fused = (exact + t[:, 1:n:2].astype(np.float64)).astype(np.float32)
    plain = (t[:, 0:n:2] + t[:, 1:n:2]).astype(np.float32)
    return int((fused.view(np.int32) != plain.view(np.int32)).sum())


def plugin_src() -> Path:
    return PLUGIN_SRC
