"""T-0073 numerics probe: one run collects what ruling D1 needs about the three tier-2 failures of T-0065's core.

Run INSIDE the pinned vLLM container on the RTX 5090 by ``tests/contract/test_vllm_cd_numerics_gpu.py`` (the repo
mounted read-only at ``/repo``, no ``/hostsite``, ``--network none``, no engine boot), and on the host ``cd`` group
by ``tests/cd/unit/test_cd_numerics_probe_cpu.py`` in ``--device cpu --small`` mode (tier 1). Imports: torch,
numpy, the stdlib, the plugin's ``cd`` core (source tree), the numpy/stdlib oracles under ``tests/cd/oracle`` and the
failing rows' instance builders (``tests/cd/unit/_core_support.py``, which imports pytest: the image ships 9.1.1).
Never scipy, never vllm (the vllm version is read from package metadata without importing it).

The three failures (``logs/t0067-tier2-295c13b.log``; rows of ``tests/cd/cuda/test_cd_cuda_repeats.py``):
- (a) ``c2_differential…[0.4]`` — L 256, V 4096, ``random.Random(7)`` toy + ``torch.Generator().manual_seed(3)``.
- (b) ``c5_no_nan…`` — V 6, L 6, S 4 NFA, ``random.Random(5)``, top-3 filter and a −1e20 sentinel.
- (c) compensated ``log_matmul`` — ``np.random.default_rng(0)``, 64×128 ⊗ 128×64 in [−24, −16], 30 % sentinels.
Margin seeds 1-4 are added in full mode.

Every section runs on the target device and on CPU from the same fp32 inputs (``--device cpu`` runs CPU only),
eager and ``torch.compile``d where it applies (compiled arms are skipped in ``--small`` unless ``--compile``).
Each section is wrapped: an exception is recorded as ``{"error": <traceback>}`` and the probe continues. The
payload is printed as one line ``ASTRA_T0073_PROBE_JSON <json>`` (``tests/numerics_probe_schema.py``), capped at
5 MiB (the P6 dumps are dropped first). Exit status: 0 when every section P1-P7 is present (errored or not), 2
when one is missing.

Run with ``PYTHONPATH=<repo>:<repo>/plugin/astra_vllm_plugin`` (the gpu wrapper and the CPU rows set it).

Sections: P1 ulp of exp/log/log1p/expm1 vs float64; P2 denormal / flush-to-zero behaviour; P3 TwoSum
exactness, alone and inside ``log_matmul``'s in-place chain; P4 ``log_matmul`` stage by stage on (c) with each
stage's first-order contribution in output ulp, and candidate arms; P5 the (a) instance: err/bar per position,
stage-swap attribution, fp64-on-device control, precision/TF32 arms; P6 the (b) fixture: every intermediate of
the pipeline with NaN/±inf/sentinel/subnormal counts, CPU-vs-device diffs and the first divergent stage, dumped
in full; P7 timing of ``constrained_marginals`` at CL 256, buckets 64 and 128.

**Staged copies.** P4, P5 and P6 need values *inside* ``semiring.log_matmul`` and
``marginals.constrained_marginals``. Rather than edit plugin code (forbidden before D1), the glue of those two
functions is copied here from the source at commit ``5e180cc`` (the last commit touching ``semiring.py`` /
``marginals.py`` on the T-0073 base 87de2da); the plugin's own helpers are called for everything else. Each of
those sections carries ``staged_equals_plugin``: the staged recomputation reproduces the plugin's outputs bit for
bit on CPU, which the schema requires, so the staging is trustworthy (and flags drift if the plugin changes).
"""

# Every nested ``run`` closure below is invoked at once by ``guarded()`` inside the loop iteration that
# defines it, so late binding of loop variables cannot occur (B023 is a false positive here).
# ruff: noqa: B023
from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import importlib.metadata as md
import itertools
import json
import math
import platform
import random
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PLUGIN_SRC = REPO / "plugin" / "astra_vllm_plugin"
# Import paths come from the environment, never from this script (tests/unit/test_cd_tier2_harness.py forbids
# touching the interpreter's search path in container scripts): PYTHONPATH=<repo>:<repo>/plugin/astra_vllm_plugin.

import numpy as np  # noqa: E402
import torch  # noqa: E402
from astra_vllm_plugin.cd import emission as em  # noqa: E402
from astra_vllm_plugin.cd import mapdecode as mapdec  # noqa: E402
from astra_vllm_plugin.cd import marginals as mg  # noqa: E402
from astra_vllm_plugin.cd import semiring as sr  # noqa: E402
from astra_vllm_plugin.cd import tree as tr  # noqa: E402

from tests.cd.oracle import dense64  # noqa: E402
from tests.cd.oracle import enumerate as en  # noqa: E402
from tests.cd.unit._core_support import random_logits, random_toy  # noqa: E402
from tests.numerics_probe_schema import (  # noqa: E402
    CAP_BYTES,
    FORBIDDEN_MODULES,
    MARKER,
    SECTIONS,
)

SOURCE_COMMIT = "5e180cc"
#: T-0073 D0: the c5 builder's seed 5 (the T-0067 row) has an empty language under its −inf mask; seed 13 keeps the
#: builder and the −inf / −1e20 stress with a non-empty one (378 accepted strings, 189 through the −1e20 id). P6
#: dumps seed 13 and reports seed 5 as ``empty_fixture``.
C5_FIXED_SEED = 13
C5_ORIGINAL_SEED = 5
SENTINEL, DEAD = sr.SENTINEL, sr.DEAD
F32 = torch.float32
TINY32 = float(np.finfo(np.float32).tiny)  # 2^-126
SUB_MIN32 = float(np.float32(2.0**-149))


# ============================================================================ helpers


@dataclasses.dataclass
class Ctx:
    device: str
    small: bool
    compile: bool
    force_error: frozenset[str]

    @property
    def devices(self) -> list[str]:
        return ["cpu"] if self.device == "cpu" else ["cpu", self.device]

    @property
    def target(self) -> str:
        return self.device


def fnum(x: float) -> float | str:
    """JSON-safe float: non-finite values become strings."""
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def tolist(t: torch.Tensor) -> Any:
    t = t.detach().cpu()
    if t.dtype == torch.bool or not t.is_floating_point():
        return t.tolist()
    arr = t.double().numpy()
    return np.vectorize(fnum, otypes=[object])(arr).tolist() if arr.ndim else fnum(float(arr))


def bits(t: torch.Tensor) -> list[str]:
    t = t.detach().cpu().contiguous()
    iv = t.view(torch.int32) if t.dtype == F32 else t.view(torch.int64)
    width = 8 if t.dtype == F32 else 16
    return [f"{int(v) & ((1 << (4 * width)) - 1):0{width}x}" for v in iv.reshape(-1).tolist()]


def sync(dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def move(obj: Any, dev: str) -> Any:
    """Tensors, tuples/lists/dicts and dataclasses of them, moved to ``dev`` (other leaves unchanged)."""
    if isinstance(obj, torch.Tensor):
        return obj.to(dev)
    if isinstance(obj, tuple):
        return tuple(move(o, dev) for o in obj)
    if isinstance(obj, list):
        return [move(o, dev) for o in obj]
    if isinstance(obj, dict):
        return {k: move(v, dev) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(
            obj, **{f.name: move(getattr(obj, f.name), dev) for f in dataclasses.fields(obj)}
        )
    return obj


def flatten(prefix: str, obj: Any) -> dict[str, torch.Tensor]:
    """Named tensors inside ``obj`` (``cm.class_anchor``, ``tree.levels[0]``, …)."""
    if isinstance(obj, torch.Tensor):
        return {prefix: obj}
    out: dict[str, torch.Tensor] = {}
    if isinstance(obj, (tuple, list)):
        for i, o in enumerate(obj):
            out.update(flatten(f"{prefix}[{i}]", o))
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if isinstance(v, (torch.Tensor, tuple, list)):
                out.update(flatten(f"{prefix}.{f.name}", v))
    return out


def tstats(t: torch.Tensor) -> dict[str, Any]:
    t = t.detach().cpu()
    d: dict[str, Any] = {"shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", "")}
    if not t.is_floating_point():
        d["sum"] = int(t.long().sum()) if t.numel() else 0
        return d
    x = t.double()
    fin = torch.isfinite(x)
    sent = fin & (x <= DEAD)
    live = fin & ~sent
    tiny = float(torch.finfo(t.dtype).tiny)
    d.update(
        nan=int(torch.isnan(x).sum()),
        posinf=int(torch.isposinf(x).sum()),
        neginf=int(torch.isneginf(x).sum()),
        sentinel=int(sent.sum()),
        zero=int((x == 0).sum()),
        subnormal=int((live & (x != 0) & (x.abs() < tiny)).sum()),
        live_min=fnum(float(x[live].min())) if live.any() else None,
        live_max=fnum(float(x[live].max())) if live.any() else None,
    )
    return d


def _kind(x: torch.Tensor) -> torch.Tensor:
    """0 live nonzero normal, 1 zero, 2 subnormal, 3 sentinel, 4 nan, 5 +inf, 6 -inf."""
    tiny = float(torch.finfo(x.dtype).tiny) if x.is_floating_point() else 0.0
    x = x.double()
    k = torch.zeros(x.shape, dtype=torch.int8)
    k[(x == 0)] = 1
    k[(x != 0) & (x.abs() < tiny)] = 2
    k[torch.isfinite(x) & (x <= DEAD)] = 3
    k[torch.isnan(x)] = 4
    k[torch.isposinf(x)] = 5
    k[torch.isneginf(x)] = 6
    return k


def tcompare(ref: torch.Tensor, got: torch.Tensor) -> dict[str, Any]:
    """``ref`` vs ``got`` (moved to CPU): bitwise equality, kind-pattern mismatches, live diffs."""
    ref, got = ref.detach().cpu(), got.detach().cpu()
    if ref.shape != got.shape:
        return {"shape_mismatch": [list(ref.shape), list(got.shape)]}
    if not ref.is_floating_point():
        return {"bitwise_equal": bool(torch.equal(ref, got))}
    same_dtype = ref.dtype == got.dtype
    beq = bool(same_dtype and bits(ref) == bits(got)) if ref.numel() <= 4096 else None
    if beq is None:
        iv = torch.int32 if ref.dtype == F32 else torch.int64
        beq = bool(same_dtype and torch.equal(ref.contiguous().view(iv), got.contiguous().view(iv)))
    kr, kg = _kind(ref), _kind(got.to(ref.dtype) if not same_dtype else got)
    both = (kr == 0) & (kg == 0)
    r, g = ref.double(), got.double()
    diff = (r - g).abs()[both]
    rel = (diff / r.abs()[both].clamp_min(1e-300)) if both.any() else diff
    ulp = (
        torch.from_numpy(np.spacing(np.abs(r[both].numpy()).astype(np.float32)).astype(np.float64))
        if both.any()
        else diff
    )
    return {
        "bitwise_equal": beq,
        "kind_mismatch": int((kr != kg).sum()),
        "max_abs_diff": fnum(float(diff.max())) if diff.numel() else 0.0,
        "max_rel_diff": fnum(float(rel.max())) if rel.numel() else 0.0,
        "max_diff_fp32_ulp": fnum(float((diff / ulp).max())) if diff.numel() else 0.0,
    }


def ulp32(ref: np.ndarray) -> np.ndarray:
    return np.spacing(np.abs(ref).astype(np.float32)).astype(np.float64)


def ulp_stats(err_ulp: np.ndarray) -> dict[str, Any]:
    e = np.asarray(err_ulp, dtype=np.float64)
    if e.size == 0:
        return {"n": 0}
    return {
        "n": int(e.size),
        "max": fnum(float(np.max(e))),
        "mean": fnum(float(np.mean(e))),
        "p99": fnum(float(np.quantile(e, 0.99))),
        "frac_gt_0.51": float(np.mean(e > 0.51)),
        "frac_gt_1": float(np.mean(e > 1.0)),
    }


@contextlib.contextmanager
def precision(mode: str, tf32: bool | None) -> Iterator[None]:
    """``set_float32_matmul_precision(mode)`` then (optionally) force ``allow_tf32``; restored after."""
    prev = torch.get_float32_matmul_precision()
    prev_m, prev_c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.set_float32_matmul_precision(mode)
    if tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)
        torch.backends.cuda.matmul.allow_tf32 = prev_m
        torch.backends.cudnn.allow_tf32 = prev_c


PRECISION_ARMS = (
    ("high", None),
    ("highest", None),
    ("high", False),
    ("highest", True),
)


def compiled(fn: Callable[..., Any]) -> Callable[..., Any]:
    return torch.compile(fn, dynamic=False)  # type: ignore[no-any-return]


def guarded(fn: Callable[[], Any]) -> Any:
    """Per-arm error capture: an arm that raises does not take its section down."""
    try:
        return fn()
    except Exception:
        return {"error": traceback.format_exc(limit=8)}


def want_compile(ctx: Ctx, dev: str) -> bool:
    """Compiled arms run in full mode on every device, and in ``--small`` only with ``--compile``."""
    return ctx.compile or not ctx.small


# ============================================================================ P1: ulp of the transcendental ops

P1_RANGES: tuple[tuple[str, str, Callable[[np.random.Generator, int], np.ndarray]], ...] = (
    ("exp", "[-100, 0]", lambda g, n: g.uniform(-100.0, 0.0, n)),
    ("exp", "[-1e-3, 1e-3]", lambda g, n: g.uniform(-1e-3, 1e-3, n)),
    ("exp", "[-log 512, 0] (finish residual e^-ls)", lambda g, n: g.uniform(-math.log(512), 0, n)),
    ("log", "[1, 512]", lambda g, n: g.uniform(1.0, 512.0, n)),
    (
        "log",
        "[FLT_MIN/2^23, 1e-30] log-uniform",
        lambda g, n: np.exp(g.uniform(math.log(2.0**-149), math.log(1e-30), n)),
    ),
    ("log1p", "[-1e-3, 1e-3]", lambda g, n: g.uniform(-1e-3, 1e-3, n)),
    ("log1p", "[0, 1]", lambda g, n: g.uniform(0.0, 1.0, n)),
    ("expm1", "[-1e-3, 1e-3]", lambda g, n: g.uniform(-1e-3, 1e-3, n)),
    ("expm1", "[-1, 0]", lambda g, n: g.uniform(-1.0, 0.0, n)),
)
NP_OPS = {"exp": np.exp, "log": np.log, "log1p": np.log1p, "expm1": np.expm1}


def _op(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    f = getattr(torch, name)

    def op(t: torch.Tensor) -> torch.Tensor:
        return f(t)  # type: ignore[no-any-return]

    return op


def p1(ctx: Ctx) -> dict[str, Any]:
    n = 1 << 14 if ctx.small else 1 << 20
    g = np.random.default_rng(101)
    rows = []
    compiled_ops: dict[tuple[str, str], Callable[..., Any]] = {}
    for op_name, rng_name, draw in P1_RANGES:
        x32 = draw(g, n).astype(np.float32)
        x64 = x32.astype(np.float64)
        with np.errstate(divide="ignore", over="ignore", under="ignore"):
            ref = NP_OPS[op_name](x64)
        ulp = ulp32(ref)
        row: dict[str, Any] = {
            "op": op_name,
            "range": rng_name,
            "n": n,
            "inputs_subnormal": int(np.sum((x32 != 0) & (np.abs(x32) < TINY32))),
        }
        for dev in ctx.devices:
            arms: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {"eager": _op(op_name)}
            if want_compile(ctx, dev):
                key = (op_name, dev)
                if key not in compiled_ops:
                    compiled_ops[key] = compiled(_op(op_name))
                arms["compiled"] = compiled_ops[key]
            for arm, fn in arms.items():

                def run(fn: Callable[..., Any] = fn, dev: str = dev) -> dict[str, Any]:
                    got = fn(torch.from_numpy(x32).to(dev)).cpu().double().numpy()
                    fin = np.isfinite(ref)
                    e = np.where(fin, np.abs(got - ref) / ulp, np.where(got == ref, 0.0, np.inf))
                    worst = int(np.argmax(np.where(np.isnan(e), np.inf, e)))
                    return {
                        **ulp_stats(e),
                        "flushed_to_zero": int(np.sum((got == 0) & (np.abs(ref) >= SUB_MIN32))),
                        "subnormal_outputs": int(
                            np.sum((got != 0) & (np.abs(got) < TINY32) & np.isfinite(got))
                        ),
                        "neginf_outputs": int(np.sum(np.isneginf(got))),
                        "neginf_expected": int(np.sum(np.isneginf(ref))),
                        "worst": {
                            "x": fnum(float(x64[worst])),
                            "x_bits": bits(torch.tensor([x32[worst]]))[0],
                            "got": fnum(float(got[worst])),
                            "ref": fnum(float(ref[worst])),
                        },
                    }

                row[f"{dev}/{arm}"] = guarded(run)
        rows.append(row)
    return {"ops": rows, "seed": 101}


# ============================================================================ P2: denormals and FTZ


def p2(ctx: Ctx) -> dict[str, Any]:
    tiny = torch.finfo(F32).tiny
    items: dict[str, tuple[list[float], Callable[[torch.Tensor], torch.Tensor]]] = {
        "roundtrip_copy": ([1e-38, 1e-40, 1e-45, tiny, 2.0**-149], lambda t: t.clone()),
        "roundtrip_mul1": ([1e-38, 1e-40, 1e-45, tiny, 2.0**-149], lambda t: t * 1.0),
        "roundtrip_add0": ([1e-38, 1e-40, 1e-45, tiny, 2.0**-149], lambda t: t + 0.0),
        "clamp_min_1e-38": ([0.0, 1e-45, 1e-40, 1e-38, 2e-38, 1.0], lambda t: t.clamp_min(1e-38)),
        "clamp_min_tiny (marginals total)": (
            [0.0, 1e-45, 1e-40, 1e-38, 1.0],
            lambda t: t.clamp_min(tiny),
        ),
        "log_clamp_min_tiny": ([0.0, 1e-45, 1e-40, 1.0], lambda t: torch.log(t.clamp_min(tiny))),
        "log_small": ([tiny, 1e-38, 1e-40, 1e-45, 2.0**-149, 0.0], torch.log),
        "exp_edges": (
            [
                -87.3,
                -87.3365,
                -87.34,
                -87.5,
                -88.72,
                -100.0,
                -103.0,
                -103.97,
                -104.0,
                -110.0,
                87.3,
                88.72,
                88.73,
                103.0,
            ],
            torch.exp,
        ),
        "mul_into_subnormal": (
            [1e-20, 1e-30, tiny, 1e-38, 3e-39],
            lambda t: t * torch.tensor([1e-20, 1e-10, 0.5, 0.1, 1.0], dtype=F32, device=t.device),
        ),
        "div_into_subnormal": ([1e-38, tiny, 1e-30], lambda t: t / 10.0),
        # marginals' entropy path on a q~ row with zero, subnormal and tiny entries
        "entropy_lq_raw": (
            [0.0, 1e-40, 1e-38, 1e-30, 0.5],
            lambda t: torch.maximum(
                torch.log(t.clamp_min(1e-38)), torch.log(t.sum().clamp_min(tiny)) + math.log(1e-30)
            ),
        ),
        "entropy_q_times_lq": (
            [0.0, 1e-40, 1e-38, 1e-30, 0.5],
            lambda t: t * torch.log(t.clamp_min(1e-38)),
        ),
    }
    out: dict[str, Any] = {}
    for name, (vals, fn) in items.items():
        x = torch.tensor(vals, dtype=F32)
        row: dict[str, Any] = {"input": [fnum(v) for v in x.tolist()], "input_bits": bits(x)}
        for dev in ctx.devices:
            arms: dict[str, Callable[..., Any]] = {"eager": fn}
            if want_compile(ctx, dev):
                arms["compiled"] = compiled(fn)
            for arm, f in arms.items():

                def run(f: Callable[..., Any] = f, dev: str = dev) -> dict[str, Any]:
                    y = f(x.to(dev)).cpu()
                    return {"out": [fnum(v) for v in y.double().tolist()], "out_bits": bits(y)}

                row[f"{dev}/{arm}"] = guarded(run)
        out[name] = row
    return {
        "items": out,
        "cpu_capability": guarded(lambda: torch.backends.cpu.get_cpu_capability()),
    }


# ============================================================================ P3: TwoSum


def _twosum64_exact(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Whether ``x + y`` (fp32 values) is exact in float64 (fp64 TwoSum error is zero)."""
    s = x + y
    bb = s - x
    return ((x - (s - bb)) + (y - bb)) == 0


def c_operands(seed: int, small: bool) -> tuple[np.ndarray, np.ndarray]:
    """The (c) row's inputs, built exactly as ``test_compensated_log_matmul_on_the_served_path`` does."""
    g = np.random.default_rng(seed)
    a = g.uniform(-24, -16, (64, 128))
    b = g.uniform(-24, -16, (128, 64))
    a[g.random(a.shape) < 0.3] = SENTINEL
    b[g.random(b.shape) < 0.3] = SENTINEL
    return a, b


def _chain(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # copied from semiring.log_matmul (compensated branch) at 5e180cc
    t = lhs + rhs
    bb = t - lhs
    err = (lhs - (t - bb)).add_(rhs - bb)
    return t, err


def p3(ctx: Ctx) -> dict[str, Any]:
    n = 1 << 14 if ctx.small else 1 << 20
    g = np.random.default_rng(303)
    regimes = {
        "c_regime [-24,-16]^2": (g.uniform(-24, -16, n), g.uniform(-24, -16, n)),
        "mixed_scale x·1e-6": (lambda x: (x, x * g.uniform(-1e-6, 1e-6, n)))(g.uniform(-50, 50, n)),
        "near_cancel": (lambda x: (x, -x * (1 + g.uniform(-1e-6, 1e-6, n))))(g.uniform(-50, 50, n)),
        "gauge int + frac": (g.integers(-300, 300, n).astype(np.float64), g.uniform(-1, 1, n)),
        "anchor - row_max [-75,0]": (g.uniform(-75, 0, n), -g.uniform(-75, 0, n)),
    }
    pairs: dict[str, Any] = {}
    for name, (x, y) in regimes.items():
        x32, y32 = x.astype(np.float32), y.astype(np.float32)
        x64, y64 = x32.astype(np.float64), y32.astype(np.float64)
        ok64 = _twosum64_exact(x64, y64)
        exact = x64 + y64
        row: dict[str, Any] = {"n": n, "n_checkable": int(ok64.sum())}
        for dev in ctx.devices:
            arms: dict[str, Callable[..., Any]] = {"eager": sr.two_sum}
            if want_compile(ctx, dev):
                arms["compiled"] = compiled(sr.two_sum)
            for arm, f in arms.items():

                def run(f: Callable[..., Any] = f, dev: str = dev) -> dict[str, Any]:
                    s, e = f(torch.from_numpy(x32).to(dev), torch.from_numpy(y32).to(dev))
                    s64, e64 = s.cpu().double().numpy(), e.cpu().double().numpy()
                    s_ok = s64 == (x32 + y32).astype(
                        np.float64
                    )  # numpy fp32 add is correctly rounded
                    se_ok = (s64 + e64) == exact
                    bad = np.flatnonzero(ok64 & ~(s_ok & se_ok))[:5]
                    return {
                        "s_not_rounded_sum": int(np.sum(ok64 & ~s_ok)),
                        "s_plus_e_inexact": int(np.sum(ok64 & ~se_ok)),
                        "examples": [
                            {
                                "x": fnum(x64[i]),
                                "y": fnum(y64[i]),
                                "s": fnum(s64[i]),
                                "e": fnum(e64[i]),
                                "exact": fnum(exact[i]),
                            }
                            for i in bad
                        ],
                    }

                row[f"{dev}/{arm}"] = guarded(run)
        pairs[name] = row

    a, b = c_operands(0, ctx.small)
    ta, tb = torch.tensor(a, dtype=F32), torch.tensor(b, dtype=F32)
    lhs, rhs = ta.reshape(1, 64, 128).unsqueeze(-1), tb.reshape(1, 128, 64).unsqueeze(-3)
    l64, r64 = lhs.double(), rhs.double()
    live = ((l64 > DEAD) & (r64 > DEAD)).expand(1, 64, 128, 64).numpy()
    ex = (l64 + r64).numpy()
    chain: dict[str, Any] = {"n_live_terms": int(live.sum()), "instance": "(c) seed 0"}
    for dev in ctx.devices:
        arms = {"eager": _chain}
        if want_compile(ctx, dev):
            arms["compiled"] = compiled(_chain)
        for arm, f in arms.items():

            def run_chain(f: Callable[..., Any] = f, dev: str = dev) -> dict[str, Any]:
                t, err = f(lhs.to(dev), rhs.to(dev))
                t64, e64 = t.cpu().double().numpy(), err.cpu().double().numpy()
                t_ok = t64 == ex.astype(np.float32).astype(np.float64)
                e_ok = e64 == (ex - t64)
                d = np.abs(e64 - (ex - t64))[live] / ulp32(t64[live])
                return {
                    "t_not_rounded_sum": int(np.sum(live & ~t_ok)),
                    "err_inexact": int(np.sum(live & ~e_ok)),
                    "err_error_in_ulp_of_t": ulp_stats(d),
                }

            chain[f"{dev}/{arm}"] = guarded(run_chain)
    return {"pairs": pairs, "chain": chain, "seed": 303}


# ============================================================================ P4: log_matmul stage by stage on (c)


def lm_staged(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    exp64: bool = False,
    comp_m: bool = False,
    finish: str = "plugin",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``semiring.log_matmul(a, b, compensated=True)`` for 2-D inputs that fit one chunk, stage by stage.

    Copied from ``semiring.log_matmul`` and ``semiring._finish`` at 5e180cc (the chunk loop collapses to one
    iteration at 64×128×64 < CHUNK_ELEMS). The keyword arms change exactly one stage each: ``exp64`` (exp in
    float64 on the ``[g, r, m, c]`` terms), ``comp_m`` (TwoSum-compensated reduction over m), ``finish``
    ``"fp64_log"`` (log in float64, residual exact), ``"expm1_log1p"`` (residual ``s·expm1(−ls) + (s − 1)`` fed to
    ``log1p``) or ``"fp64"`` (the whole finish ``safe + log s`` in float64 on ``[n, k]``).
    """
    n, m = a.shape
    k = b.shape[1]
    assert n * m * k <= sr.CHUNK_ELEMS
    a3 = a.expand(n, m).reshape(1, n, m)
    b3 = b.expand(m, k).reshape(1, m, k)
    lhs = a3[0:1, 0:n, :].unsqueeze(-1)
    rhs = b3[0:1, :, 0:k].unsqueeze(-3)
    t = lhs + rhs
    mx = t.amax(dim=-2)
    live = mx > DEAD
    safe = torch.where(live, mx, torch.zeros_like(mx))
    bb = t - lhs
    err = (lhs - (t - bb)).add_(rhs - bb)
    arg = t.sub(safe.unsqueeze(-2)).add_(err)
    terms = arg.double().exp().to(arg.dtype) if exp64 else arg.exp()
    s = sr.compensated_sum(terms.transpose(-2, -1).contiguous()) if comp_m else terms.sum(dim=-2)
    ls = torch.log(s)
    hi, lo = sr.two_sum(safe, ls)
    if finish == "plugin":
        resid = s * torch.exp(-ls) - 1.0
        out = hi + (lo + resid)
    elif finish == "fp64_log":
        ls64 = torch.log(s.double())
        ls = ls64.to(s.dtype)
        hi, lo = sr.two_sum(safe, ls)
        resid = (ls64 - ls.double()).to(s.dtype)
        out = hi + (lo + resid)
    elif finish == "expm1_log1p":
        resid = torch.log1p(s * torch.expm1(-ls) + (s - 1.0))
        out = hi + (lo + resid)
    elif finish == "fp64":
        resid = torch.zeros_like(s)
        out = (safe.double() + torch.log(s.double())).to(s.dtype)
    else:
        raise ValueError(finish)
    out = torch.where(live, out, torch.full_like(out, SENTINEL))
    stages = {
        "lhs": lhs,
        "rhs": rhs,
        "t": t,
        "mx": mx,
        "safe": safe,
        "err": err,
        "arg": arg,
        "terms": terms,
        "s": s,
        "ls": ls,
        "hi": hi,
        "lo": lo,
        "resid": resid,
        "out": out,
    }
    return out.reshape(n, k), stages


def lm_oracle(a64: np.ndarray, b64: np.ndarray) -> np.ndarray:
    """float64 ``log Σ_j exp(a_ij + b_jk)``; the row's ``_log_matmul64`` (fed fp64 or fp32-rounded inputs)."""
    s = a64[..., :, :, None] + b64[..., None, :, :]
    m = s.max(axis=-2)
    return m + np.log(np.exp(s - m[..., None, :]).sum(axis=-2))  # type: ignore[no-any-return]


def lm_errors(out: torch.Tensor, want: np.ndarray, live: np.ndarray) -> dict[str, Any]:
    got = out.detach().cpu().double().numpy()
    return ulp_stats((np.abs(got - want) / ulp32(want))[live])


def stage_contributions(st: dict[str, torch.Tensor], want: np.ndarray) -> dict[str, Any]:
    """First-order contribution of each stage's rounding to the output, in output ulp (fp64, from the device's
    own stage tensors). ``total`` should equal the sum of the parts up to second order (``model_residual``)."""
    f = {k: v.detach().cpu().double().numpy() for k, v in st.items()}
    lhs, rhs, t, safe = f["lhs"], f["rhs"], f["t"], f["safe"]
    live_t = (lhs > DEAD) & (rhs > DEAD)  # [1, n, m, k]
    t_ex = lhs + rhs
    err_ex = t_ex - t
    safe_b = safe[..., None, :]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        w = np.where(live_t, np.exp(t_ex - safe_b), 0.0)
        w = w / np.where(w.sum(axis=-2, keepdims=True) > 0, w.sum(axis=-2, keepdims=True), 1.0)
        arg_given = (t - safe_b) + f["err"]
        exp_at_arg = np.exp(f["arg"])
        rel_exp = np.where(
            live_t, (f["terms"] - exp_at_arg) / np.where(exp_at_arg > 0, exp_at_arg, 1), 0
        )
        s_given = np.where(live_t, f["terms"], 0).sum(axis=-2)
        s = f["s"]
        c = {
            "twosum_err": (w * np.where(live_t, f["err"] - err_ex, 0)).sum(axis=-2),
            "arg_rounding": (w * np.where(live_t, f["arg"] - arg_given, 0)).sum(axis=-2),
            "exp": (w * rel_exp).sum(axis=-2),
            "sum_over_m": (s - s_given) / np.where(s > 0, s, 1.0),
            "finish (log, residual, hi+lo)": f["out"] - (safe + np.log(np.where(s > 0, s, 1.0))),
        }
    out = f["out"].reshape(want.shape)
    live_o = (want > -1e20) & (out > DEAD)
    u = ulp32(want)
    res: dict[str, Any] = {}
    total_parts = np.zeros(want.shape)
    for name, v in c.items():
        v = v.reshape(want.shape)
        total_parts += v
        res[name] = {
            **ulp_stats((np.abs(v) / u)[live_o]),
            "signed_mean": fnum(float(np.mean((v / u)[live_o]))),
        }
    total = out - want
    res["total"] = {
        **ulp_stats((np.abs(total) / u)[live_o]),
        "signed_mean": fnum(float(np.mean((total / u)[live_o]))),
    }
    res["model_residual"] = ulp_stats((np.abs(total - total_parts) / u)[live_o])
    ex = np.where(
        live_t, np.abs(f["terms"] - exp_at_arg) / ulp32(np.where(live_t, exp_at_arg, 1.0)), 0.0
    )
    res["exp_terms_ulp"] = ulp_stats(ex[live_t])
    ls, s2 = f["ls"].reshape(want.shape), f["s"].reshape(want.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        lref = np.log(np.where(s2 > 0, s2, 1.0))
    res["log_s_ulp"] = ulp_stats((np.abs(ls - lref) / ulp32(lref))[live_o & (np.abs(lref) > 0)])
    res["twosum_err_inexact_terms"] = int(np.sum(live_t & (f["err"] != err_ex)))
    return res


P4_ARMS: dict[str, dict[str, Any]] = {
    "staged": {},
    "fp64_exp": {"exp64": True},
    "fp64_log": {"finish": "fp64_log"},
    "expm1_log1p_residual": {"finish": "expm1_log1p"},
    "compensated_m_reduction": {"comp_m": True},
    "fp64_finish_nk": {"finish": "fp64"},
    "compensated_m+fp64_finish": {"comp_m": True, "finish": "fp64"},
    "fp64_exp+compensated_m+fp64_finish": {"exp64": True, "comp_m": True, "finish": "fp64"},
}


def _median_ms(fn: Callable[[], Any], dev: str, n: int) -> float:
    fn()
    sync(dev)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def p4(ctx: Ctx) -> dict[str, Any]:
    seeds = [0] if ctx.small else [0, 1, 2, 3, 4]
    instances: dict[str, Any] = {}
    contributions: dict[str, Any] = {}
    arms_out: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    equal: dict[str, Any] = {}
    compiled_plugin: dict[str, Callable[..., Any]] = {}
    compiled_staged: dict[str, Callable[..., Any]] = {}
    for seed in seeds:
        a, b = c_operands(seed, ctx.small)
        ta, tb = torch.tensor(a, dtype=F32), torch.tensor(b, dtype=F32)
        want_row = lm_oracle(a, b)  # exactly the row's oracle: unrounded float64 inputs
        want = lm_oracle(ta.double().numpy(), tb.double().numpy())  # the fp32 inputs actually used
        live_row, live = want_row > -1e20, want > -1e20
        key = f"seed{seed}"
        instances[key] = {
            "shape": [64, 128, 64],
            "live_outputs": int(live.sum()),
            "input_rounding_in_output_ulp": ulp_stats(
                (np.abs(want_row - want) / ulp32(want))[live]
            ),
        }
        for dev in ctx.devices:
            da, db = ta.to(dev), tb.to(dev)
            row: dict[str, Any] = {}
            ctl: dict[str, Any] = {}
            plug = sr.log_matmul(da, db, compensated=True)
            unc = sr.log_matmul(da, db, compensated=False)
            ctl["row_as_written (fp64-input oracle)"] = {
                "compensated": lm_errors(plug, want_row, live_row),
                "uncompensated": lm_errors(unc, want_row, live_row),
            }
            ctl["fp32-rounded-input oracle"] = {
                "compensated": lm_errors(plug, want, live),
                "uncompensated": lm_errors(unc, want, live),
            }
            controls.setdefault(key, {})[dev] = ctl
            row["plugin_eager"] = lm_errors(plug, want, live)
            row["plugin_uncompensated"] = lm_errors(unc, want, live)
            staged_out, staged_st = lm_staged(da, db)
            if seed == 0:
                eqd = bool(
                    torch.equal(staged_out.cpu().view(torch.int32), plug.cpu().view(torch.int32))
                )
                equal[dev] = eqd
                contributions[dev] = stage_contributions(staged_st, want)
                row["timing_ms_plugin_eager"] = _median_ms(
                    lambda da=da, db=db: sr.log_matmul(da, db, compensated=True),
                    dev,
                    5 if ctx.small else 20,
                )
            for arm, kw in P4_ARMS.items():
                row[arm] = guarded(
                    lambda kw=kw, da=da, db=db: lm_errors(lm_staged(da, db, **kw)[0], want, live)
                )
            if want_compile(ctx, dev):
                if dev not in compiled_plugin:
                    compiled_plugin[dev] = compiled(sr.log_matmul)
                    compiled_staged[dev] = compiled(lambda x, y: lm_staged(x, y)[0])
                cp, cs = compiled_plugin[dev], compiled_staged[dev]
                row["plugin_compiled"] = guarded(
                    lambda cp=cp, da=da, db=db: lm_errors(cp(da, db, compensated=True), want, live)
                )
                row["staged_compiled"] = guarded(
                    lambda cs=cs, da=da, db=db: lm_errors(cs(da, db), want, live)
                )
            else:
                row["plugin_compiled"] = {"skipped": "small mode (pass --compile)"}
            for mode, tf32 in PRECISION_ARMS:

                def run_prec(
                    mode: str = mode, tf32: bool | None = tf32, da: Any = da, db: Any = db
                ) -> Any:
                    with precision(mode, tf32):
                        return lm_errors(sr.log_matmul(da, db, compensated=True), want, live)

                row[f"plugin_eager@{mode},tf32={tf32}"] = guarded(run_prec)
            arms_out.setdefault(key, {})[dev] = row
    if ctx.target != "cpu":
        a, b = c_operands(0, ctx.small)
        ta, tb = torch.tensor(a, dtype=F32), torch.tensor(b, dtype=F32)
        c_cpu = lm_staged(ta, tb)[1]
        c_dev = lm_staged(ta.to(ctx.target), tb.to(ctx.target))[1]
        div = {name: tcompare(c_cpu[name], c_dev[name]) for name in c_cpu}
        contributions["cpu_vs_device_per_stage"] = div
        contributions["first_divergent_stage"] = next(
            (n for n, d in div.items() if not d.get("bitwise_equal")), None
        )
    return {
        "instances": instances,
        "stage_contributions": contributions,
        "arms": arms_out,
        "oracle_controls": controls,
        "staged_equals_plugin": equal,
        "bars": "row: compensated max <= 1.0 ulp and <= 5 % > 0.51 ulp; uncompensated > 5 % > 0.51",
    }


# ============================================================================ staged constrained_marginals (P5, P6)


def _st_class_mass(e: dict[str, Any]) -> dict[str, Any]:
    return {"cm": em.class_mass(e["logits"], e["tables"])}


def _st_log_transfer(e: dict[str, Any]) -> dict[str, Any]:
    return {"log_m": em.log_transfer(e["logits"], e["tables"], e["l_eff"], class_mass=e["cm"])}


def _st_up_sweep(e: dict[str, Any]) -> dict[str, Any]:
    return {"tree": tr.up_sweep(e["log_m"])}


def _st_pass1(e: dict[str, Any]) -> dict[str, Any]:
    # copied from marginals.constrained_marginals at 5e180cc (pass 1: down-sweep, ends, log Z, log_a/log_b)
    tree = e["tree"]
    root = tree.levels[-1][0]
    root_scale = tree.log_scales[-1][0]
    la0 = mg._mask_to_log(e["a_start"], root)
    lb_end = mg._mask_to_log(e["b_final"], root)
    a, sa, b, sb = mg._down_sweep(tree, la0, lb_end)
    a_last, a_last_m = sr.normalise_last(mg._vec_mat(la0.unsqueeze(0), root.unsqueeze(0)))
    b_first, b_first_m = sr.normalise_last(mg._mat_vec(root.unsqueeze(0), lb_end.unsqueeze(0)))
    sa_last = root_scale + a_last_m
    sb_first = root_scale + b_first_m
    t0 = la0 + b_first[0]
    log_z = (torch.logsumexp(t0, dim=0) + sb_first[0]).clamp_min(SENTINEL)
    log_z = torch.where(t0.amax() > DEAD, log_z, torch.full_like(log_z, SENTINEL))
    log_a = torch.cat([mg._absolute(a, sa), mg._absolute(a_last, sa_last)], dim=0)
    log_b = torch.cat([mg._absolute(b_first, sb_first), mg._absolute(b, sb)], dim=0)
    return {"la0": la0, "lb_end": lb_end, "log_z": log_z, "log_a": log_a, "log_b": log_b}


def _st_gauge(e: dict[str, Any]) -> dict[str, Any]:
    log_b = e["log_b"]
    viable = log_b > DEAD
    beta = torch.where(viable, torch.round(log_b), torch.zeros_like(log_b))
    return {"viable": viable, "beta": beta}


def _st_w_edge(e: dict[str, Any]) -> dict[str, Any]:
    return {
        "w_edge": mg._gauged_edge_weights(e["cm"], e["tables"], e["beta"], e["viable"], e["l_eff"])
    }


def _st_leaves(e: dict[str, Any]) -> dict[str, Any]:
    return {
        "leaves": mg._gauged_leaves(e["w_edge"], e["tables"], e["beta"], e["viable"], e["l_eff"])
    }


def _st_tree2(e: dict[str, Any]) -> dict[str, Any]:
    return {"tree2": tr.up_sweep(e["leaves"], compensated=True)}


def _st_a2b2(e: dict[str, Any]) -> dict[str, Any]:
    la0, lb_end, beta, viable = e["la0"], e["lb_end"], e["beta"], e["viable"]
    ga0 = torch.where(viable[0] & (la0 > DEAD), la0 + beta[0], torch.full_like(la0, SENTINEL))
    gbl = torch.where(
        viable[-1] & (lb_end > DEAD), lb_end - beta[-1], torch.full_like(lb_end, SENTINEL)
    )
    a2, _, b2, _ = mg._down_sweep(e["tree2"], ga0, gbl, compensated=True)
    return {"ga0": ga0, "gbl": gbl, "a2": a2, "b2": b2}


def _st_joint(e: dict[str, Any]) -> dict[str, Any]:
    t = e["tables"]
    joint_e = (e["a2"].index_select(1, t.edge_src) + e["b2"].index_select(1, t.edge_dst)) + e[
        "w_edge"
    ]
    joint_e = joint_e.clamp_min(SENTINEL)
    c_m, c_s = sr.segment_sumexp_split(joint_e, t.edge_class, t.n_classes)
    joint = torch.where(c_m > DEAD, c_m + sr.safe_log(c_s), c_m)
    return {"joint_e": joint_e, "c_m": c_m, "c_s": c_s, "joint": joint}


def _st_g(e: dict[str, Any]) -> dict[str, Any]:
    joint = e["joint"]
    j_m, j_s = em.row_split(joint)
    g = torch.where(
        (joint > DEAD) & (j_m > DEAD).unsqueeze(1),
        (joint - j_m.unsqueeze(1)) - sr.safe_log(j_s).unsqueeze(1),
        torch.full_like(joint, SENTINEL),
    )
    return {"j_m": j_m, "j_s": j_s, "g": g}


def _st_coef(e: dict[str, Any]) -> dict[str, Any]:
    cm, g = e["cm"], e["g"]
    coef = torch.where(
        cm.class_sum > 0, torch.exp(g) / cm.class_sum.clamp_min(1.0), torch.zeros_like(g)
    )
    return {"coef": coef}


def _st_q_raw(e: dict[str, Any]) -> dict[str, Any]:
    return {"q_raw": mg._token_marginals(e["logits"], e["tables"], e["cm"], e["coef"])}


def _st_total(e: dict[str, Any]) -> dict[str, Any]:
    q_raw = e["q_raw"]
    total_unclamped = sr.compensated_sum(q_raw).unsqueeze(-1)
    total = total_unclamped.clamp_min(torch.finfo(q_raw.dtype).tiny)
    return {"total_unclamped": total_unclamped, "total": total}


def _st_q(e: dict[str, Any]) -> dict[str, Any]:
    return {"q": e["q_raw"] / e["total"]}


def _st_entropy(e: dict[str, Any]) -> dict[str, Any]:
    q_raw, total = e["q_raw"], e["total"]
    log_total = torch.log(total)
    lq_raw = torch.maximum(torch.log(q_raw.clamp_min(1e-38)), log_total + math.log(1e-30))
    qlq = q_raw * lq_raw
    entropy = log_total.squeeze(-1) - sr.compensated_sum(qlq) / total.squeeze(-1)
    return {
        "log_total": log_total,
        "lq_raw": lq_raw,
        "qlq_sum": sr.compensated_sum(qlq),
        "entropy": entropy,
    }


#: (name, inputs, fn) in pipeline order; names are ``numerics_probe_schema.P6_STAGES``
STAGES: tuple[tuple[str, tuple[str, ...], Callable[[dict[str, Any]], dict[str, Any]]], ...] = (
    ("class_mass", ("logits", "tables"), _st_class_mass),
    ("log_transfer", ("logits", "tables", "l_eff", "cm"), _st_log_transfer),
    ("up_sweep", ("log_m",), _st_up_sweep),
    ("pass1", ("tree", "a_start", "b_final"), _st_pass1),
    ("gauge", ("log_b",), _st_gauge),
    ("w_edge", ("cm", "tables", "beta", "viable", "l_eff"), _st_w_edge),
    ("leaves", ("w_edge", "tables", "beta", "viable", "l_eff"), _st_leaves),
    ("tree2", ("leaves",), _st_tree2),
    ("a2b2", ("la0", "lb_end", "beta", "viable", "tree2"), _st_a2b2),
    ("joint", ("a2", "b2", "w_edge", "tables"), _st_joint),
    ("g", ("joint",), _st_g),
    ("coef", ("cm", "g"), _st_coef),
    ("q_raw", ("logits", "tables", "cm", "coef"), _st_q_raw),
    ("total", ("q_raw",), _st_total),
    ("q", ("q_raw", "total"), _st_q),
    ("entropy", ("q_raw", "total"), _st_entropy),
)
STAGE_NAMES = tuple(s[0] for s in STAGES)


def tables_for(toy: en.Toy, dev: str) -> Any:
    """``tests/cd/cuda/test_cd_cuda_repeats.py::tables`` on ``dev``."""
    return em.build_tables(
        n_states=toy.n_states,
        vocab_size=toy.vocab_size,
        class_members=[sorted(s) for s, _ in toy.classes],
        class_is_neg=[n for _, n in toy.classes],
        edge_src=[e[0] for e in toy.edges],
        edge_dst=[e[1] for e in toy.edges],
        edge_class=[e[2] for e in toy.edges],
        finals=sorted(toy.finals),
        device=dev,
    )


def vec(idx: Any, n: int, dev: str) -> torch.Tensor:
    m = torch.zeros(n, dtype=torch.bool, device=dev)
    m[list(idx)] = True
    return m


@dataclasses.dataclass
class Problem:
    toy: en.Toy
    x: torch.Tensor  # CPU, the dtype the row feeds
    start: list[int]
    end: list[int]
    L: int


def run_staged(
    p: Problem, place: dict[str, str], *, dtype: torch.dtype = F32, segments: str = "plugin"
) -> dict[str, dict[str, Any]]:
    """Run the staged pipeline with each stage on ``place[stage]``; returns every stage's outputs on CPU.

    ``segments="deterministic"`` swaps the plugin's ``index_add_`` segment sums for the probe's fixed-order ones
    (``deterministic_segments``) for the duration of the call."""
    with segment_variant(segments):
        return _run_staged(p, place, dtype)


def _run_staged(p: Problem, place: dict[str, str], dtype: torch.dtype) -> dict[str, dict[str, Any]]:
    tabs: dict[str, Any] = {}

    def tables_on(dev: str) -> Any:
        if dev not in tabs:
            tabs[dev] = tables_for(p.toy, dev)
        return tabs[dev]

    B = tables_on("cpu").bucket
    env: dict[str, Any] = {
        "logits": p.x.to(dtype),
        "a_start": vec(p.start, B, "cpu"),
        "b_final": vec(p.end, B, "cpu"),
        "l_eff": max(0, min(int(p.L), p.x.shape[0])),
    }
    out: dict[str, dict[str, Any]] = {}
    for name, inputs, fn in STAGES:
        dev = place[name]
        sub = {k: (tables_on(dev) if k == "tables" else move(env[k], dev)) for k in inputs}
        res = fn(sub)
        sync(dev)
        env.update(res)
        out[name] = move(res, "cpu")
    return out


def run_plugin(p: Problem, dev: str, *, dtype: torch.dtype = F32, segments: str = "plugin") -> Any:
    """The rows' ``pipeline``: log_transfer -> up_sweep -> constrained_marginals, all on ``dev``."""
    t = tables_for(p.toy, dev)
    x = p.x.to(dtype).to(dev)
    with segment_variant(segments):
        tree = tr.up_sweep(em.log_transfer(x, t, p.L))
        return mg.constrained_marginals(
            x, t, tree, vec(p.start, t.bucket, dev), vec(p.end, t.bucket, dev), p.L
        )


# ---------------------------------------------------------------------------- deterministic segment sums (D3)


def det_segment_sum(vals: torch.Tensor, segment: torch.Tensor, n_segments: int) -> torch.Tensor:
    """``out[r, g] = Σ_{j: segment[j] = g} vals[r, j]`` in a fixed order, without atomics: the terms are laid out
    densely as ``[R, n_segments, max_len]`` (stable sort by segment, zero padding) and reduced with ``sum(-1)``.
    Probe-only (the ``max_len`` read is a host sync)."""
    rows, width = vals.shape
    out_shape = (rows, n_segments)
    if width == 0:
        return torch.zeros(out_shape, dtype=vals.dtype, device=vals.device)
    order = torch.argsort(segment, stable=True)
    seg_sorted = segment.index_select(0, order)
    counts = torch.bincount(segment, minlength=n_segments)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(width, device=vals.device) - starts.index_select(0, seg_sorted)
    max_len = int(counts.max())
    dense = torch.zeros((rows, n_segments, max_len), dtype=vals.dtype, device=vals.device)
    dense[:, seg_sorted, pos] = vals.index_select(1, order)
    return dense.sum(dim=-1)


def det_segment_sumexp_split(
    values: torch.Tensor,
    segment: torch.Tensor,
    n_segments: int,
    columns: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``semiring.segment_sumexp_split`` (5e180cc) with the ``index_add_`` sum replaced by ``det_segment_sum``.
    Unchunked (probe sizes). The segment max (``scatter_reduce_`` amax) is order-independent and kept."""
    vals = values if columns is None else values.index_select(1, columns)
    rows, width = vals.shape
    mx = torch.full((rows, n_segments), SENTINEL, dtype=vals.dtype, device=vals.device)
    if width == 0:
        return mx, torch.zeros_like(mx)
    mx.scatter_reduce_(
        1, segment.unsqueeze(0).expand(rows, width), vals, reduce="amax", include_self=True
    )
    live = mx > DEAD
    safe = torch.where(live, mx, torch.zeros_like(mx))
    s = det_segment_sum((vals - safe.index_select(1, segment)).exp(), segment, n_segments)
    ok = live & (s > 0)
    return torch.where(ok, mx, torch.full_like(mx, SENTINEL)), torch.where(
        ok, s, torch.zeros_like(s)
    )


def det_token_marginals(
    logits: torch.Tensor, tables: Any, cm: Any, coef: torch.Tensor
) -> torch.Tensor:
    """``marginals._token_marginals`` (5e180cc) with the ``index_add_`` over ``memb_upos`` replaced by
    ``det_segment_sum``. Unchunked (probe sizes)."""
    L = logits.shape[0]
    zero = torch.zeros((), dtype=logits.dtype, device=logits.device)
    if tables.has_outside:
        out_m = cm.out_anchor.unsqueeze(1)
        k = torch.where(
            tables.class_neg.unsqueeze(0),
            coef * torch.exp((out_m - cm.class_anchor).clamp_max(0.0)),
            zero,
        ).sum(dim=1, keepdim=True)
        q = torch.exp((logits - out_m).clamp_max(0.0)) * k
    else:
        q = torch.zeros_like(logits)
    n_u = tables.union_ids.shape[0]
    cls = tables.memb_class
    x = logits.index_select(1, tables.memb_token)
    e = torch.exp((x - cm.class_anchor.index_select(1, cls)).clamp_max(0.0))
    q_u = det_segment_sum(e * coef.index_select(1, cls), tables.memb_upos, n_u)
    if q_u.shape != (L, n_u):
        raise AssertionError(q_u.shape)
    q.index_copy_(1, tables.union_ids, q_u)
    return q


@contextlib.contextmanager
def segment_variant(kind: str) -> Iterator[None]:
    """``"plugin"``: unchanged. ``"deterministic"``: for the duration of the block only, the plugin modules' names
    ``segment_sumexp_split`` (semiring, emission, marginals) and ``marginals._token_marginals`` point at the probe's
    fixed-order versions; restored in ``finally``. A probe-local diagnostic arm: no torch/CUDA global config is
    touched (``torch.use_deterministic_algorithms`` is never called)."""
    if kind == "plugin":
        yield
        return
    if kind != "deterministic":
        raise ValueError(kind)
    saved = (
        sr.segment_sumexp_split,
        em.segment_sumexp_split,
        mg.segment_sumexp_split,
        mg._token_marginals,
    )
    sr.segment_sumexp_split = det_segment_sumexp_split
    em.segment_sumexp_split = det_segment_sumexp_split
    mg.segment_sumexp_split = det_segment_sumexp_split
    mg._token_marginals = det_token_marginals
    try:
        yield
    finally:
        (
            sr.segment_sumexp_split,
            em.segment_sumexp_split,
            mg.segment_sumexp_split,
            mg._token_marginals,
        ) = saved


def staged_vs_plugin(staged: dict[str, dict[str, Any]], res: Any) -> dict[str, bool]:
    pairs = {
        "log_z": (staged["pass1"]["log_z"], res.log_z),
        "log_a": (staged["pass1"]["log_a"], res.log_a),
        "log_b": (staged["pass1"]["log_b"], res.log_b),
        "q": (staged["q"]["q"], res.q),
        "entropy": (staged["entropy"]["entropy"], res.entropy),
    }
    return {k: bool(tcompare(a, b).get("bitwise_equal")) for k, (a, b) in pairs.items()}


def stage_divergence(
    ref: dict[str, dict[str, Any]], got: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    per: dict[str, Any] = {}
    first_bitwise, first_material = None, None
    for name in STAGE_NAMES:
        fr = {k: t for k2, v in ref[name].items() for k, t in flatten(k2, v).items()}
        fg = {k: t for k2, v in got[name].items() for k, t in flatten(k2, v).items()}
        rows = {k: tcompare(fr[k], fg[k]) for k in fr}
        per[name] = rows
        if first_bitwise is None and any(not r.get("bitwise_equal") for r in rows.values()):
            first_bitwise = name
        material = any(
            r.get("kind_mismatch", 0)
            or (isinstance(r.get("max_rel_diff"), float) and r["max_rel_diff"] > 1e-5)
            for r in rows.values()
        )
        if first_material is None and material:
            first_material = name
    return {
        "per_stage": per,
        "first_bitwise_divergent": first_bitwise,
        "first_material_divergent (kind mismatch or rel > 1e-5)": first_material,
    }


# ============================================================================ P5: the (a) instance (v2, D2/D3)


def a_problem(
    temp: float, *, small: bool, seed_rng: int = 7, seed_gen: int = 3
) -> tuple[Problem, np.ndarray]:
    """``test_t0065_c2_differential_vs_stock_error_on_cuda`` instance (L 256, V 4096; small: L 32, V 512).
    Returns the problem (logits as the row generated them, unrounded float64) and those logits as numpy."""
    rng = random.Random(seed_rng)
    V, L = (512, 32) if small else (4096, 256)
    toy = random_toy(rng, V=V, S=12, nfa=True)
    g = torch.Generator().manual_seed(seed_gen)
    raw = torch.randn(L, V, generator=g, dtype=torch.float64) * 40.0
    x64 = (30.0 * torch.tanh(raw / 30.0)) / temp
    return Problem(toy, x64, [0], sorted(toy.finals), L), x64.numpy()


def a_oracle(p: Problem, x64: np.ndarray) -> np.ndarray:
    start, end = np.zeros(p.toy.n_states, bool), np.zeros(p.toy.n_states, bool)
    start[p.start] = True
    end[p.end] = True
    o = dense64.forward_backward64(x64, p.toy, start, end, p.L)
    if not math.isfinite(o.log_z):
        raise AssertionError("fixture: empty language on this instance")
    return np.asarray(o.entropy)


def stock_h64(x64: torch.Tensor) -> np.ndarray:
    lp64 = torch.log_softmax(x64, dim=-1)
    return -(lp64.exp() * lp64).sum(-1).numpy()  # type: ignore[no-any-return]


def stock_h32(x64: torch.Tensor, dev: str) -> np.ndarray:
    """Stock fp32 H(p) on ``dev`` (as the row computes it), returned as float64."""
    lp = torch.log_softmax(x64.to(F32).to(dev), dim=-1)
    return -(lp.exp() * lp).sum(-1).double().cpu().numpy()  # type: ignore[no-any-return]


def rounded(x64: torch.Tensor) -> torch.Tensor:
    return x64.to(F32).to(torch.float64)


@dataclasses.dataclass
class ARefs:
    """References of one (a) instance: oracle H and stock fp64 H(p), each on unrounded (u) and rounded (r) logits."""

    oracle_u: np.ndarray
    oracle_r: np.ndarray
    hp64_u: np.ndarray
    hp64_r: np.ndarray


def a_refs(p: Problem, x64: np.ndarray) -> ARefs:
    xr = rounded(p.x)
    return ARefs(
        oracle_u=a_oracle(p, x64),
        oracle_r=a_oracle(p, xr.numpy()),
        hp64_u=stock_h64(p.x),
        hp64_r=stock_h64(xr),
    )


def bar_of(hp32: np.ndarray, hp64: np.ndarray) -> np.ndarray:
    return 10.0 * np.abs(hp32 - hp64) + 1e-6  # type: ignore[no-any-return]


def a_bar(x64: torch.Tensor, dev: str) -> np.ndarray:
    """The D2 bar: stock fp32 H(p) on ``dev`` vs fp64 H(p) on the fp32-rounded logits."""
    return bar_of(stock_h32(x64, dev), stock_h64(rounded(x64)))


def err_over_bar(h: torch.Tensor, oracle: np.ndarray, bar: np.ndarray, L: int) -> dict[str, Any]:
    err = np.abs(h.detach().cpu().double().numpy()[:L] - oracle)
    r = err / bar
    return {
        "worst_err_over_bar": fnum(float(np.max(r))),
        "argmax": int(np.argmax(r)),
        "n_over_bar": int(np.sum(r > 1)),
        "max_err": fnum(float(np.max(err))),
        "median_err_over_bar": fnum(float(np.median(r))),
    }


def b64(a: Any, dtype: str) -> dict[str, Any]:
    """Compact array: little-endian ``dtype`` bytes, base64. Decode: ``np.frombuffer(base64.b64decode(b64),
    dtype="<f4"|"<f8").reshape(shape)``."""
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.dtype(dtype).newbyteorder("<")))
    return {
        "dtype": dtype,
        "shape": list(arr.shape),
        "b64": base64.b64encode(arr.tobytes()).decode(),
    }


def scores(h: torch.Tensor, refs: ARefs, hp32: np.ndarray, L: int) -> dict[str, Any]:
    """The four oracle x bar-reference combinations; the first is D2 (the rows' definition after 2026-09-30)."""
    return {
        "oracle_rounded/bar_rounded (D2)": err_over_bar(
            h, refs.oracle_r, bar_of(hp32, refs.hp64_r), L
        ),
        "oracle_rounded/bar_unrounded": err_over_bar(
            h, refs.oracle_r, bar_of(hp32, refs.hp64_u), L
        ),
        "oracle_unrounded/bar_unrounded (pre-D2 row)": err_over_bar(
            h, refs.oracle_u, bar_of(hp32, refs.hp64_u), L
        ),
        "oracle_unrounded/bar_rounded": err_over_bar(
            h, refs.oracle_u, bar_of(hp32, refs.hp64_r), L
        ),
    }


def repeats(
    p: Problem, dev: str, n: int, refs: ARefs, bar: np.ndarray, segments: str
) -> dict[str, Any]:
    """``n`` same-process runs of the staged pipeline on ``dev``; every stage's tensors compared bitwise with run 0."""
    runs = [run_staged(p, {s: dev for s in STAGE_NAMES}, segments=segments) for _ in range(n)]
    base = {
        name: {k: t for k2, v in runs[0][name].items() for k, t in flatten(k2, v).items()}
        for name in STAGE_NAMES
    }
    first_diff: list[str | None] = []
    differing = {name: 0 for name in STAGE_NAMES}
    for r in runs[1:]:
        first = None
        for name in STAGE_NAMES:
            cur = {k: t for k2, v in r[name].items() for k, t in flatten(k2, v).items()}
            if any(not tcompare(base[name][k], cur[k]).get("bitwise_equal") for k in base[name]):
                differing[name] += 1
                first = first or name
        first_diff.append(first)
    hs = np.stack([r["entropy"]["entropy"].double().numpy()[: p.L] for r in runs])
    ratios = [float(np.max(np.abs(h - refs.oracle_r) / bar)) for h in hs]
    return {
        "n": n,
        "segments": segments,
        "all_bitwise_repeatable": all(f is None for f in first_diff),
        "first_differing_stage_per_repeat": first_diff,
        "repeats_differing_per_stage": differing,
        "worst_err_over_bar_D2_per_repeat": [fnum(v) for v in ratios],
        "worst_err_over_bar_D2_min_max": [fnum(min(ratios)), fnum(max(ratios))],
        "entropy_spread_max_abs": fnum(float(np.max(hs.max(0) - hs.min(0)))),
        "entropy_runs": b64(hs.astype(np.float32), "float32"),
    }


def p5(ctx: Ctx) -> dict[str, Any]:
    temps = (0.4, 0.8)
    dev = ctx.target
    instances: dict[str, Any] = {}
    swap: dict[str, Any] = {}
    divergence: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    equal: dict[str, Any] = {}
    reps: dict[str, Any] = {}
    det: dict[str, Any] = {}
    n_rep = {d: (3 if ctx.small else (20 if d != "cpu" else 5)) for d in ctx.devices}
    for temp in temps:
        key = f"T{temp}"
        p, x64 = a_problem(temp, small=ctx.small)
        refs = a_refs(p, x64)
        hp32 = {d: stock_h32(p.x, d) for d in ctx.devices}
        bars = {d: bar_of(hp32[d], refs.hp64_r) for d in ctx.devices}  # D2
        inst: dict[str, Any] = {
            "L": p.L,
            "V": p.toy.vocab_size,
            "S": p.toy.n_states,
            "seeds": "random.Random(7) toy, torch.Generator(3) logits",
            "primary_scoring": "D2: oracle and bar reference on the fp32-rounded logits",
            "bar_ratio_device_over_cpu": ulp_stats(bars[dev] / bars["cpu"]),
            "oracle_shift_rounded_vs_unrounded_max": fnum(
                float(np.max(np.abs(refs.oracle_r - refs.oracle_u)))
            ),
            "arrays": {
                "oracle_H_rounded": b64(refs.oracle_r, "float64"),
                "oracle_H_unrounded": b64(refs.oracle_u, "float64"),
                "stock_Hp64_rounded": b64(refs.hp64_r, "float64"),
                "stock_Hp64_unrounded": b64(refs.hp64_u, "float64"),
            },
        }
        for d in ctx.devices:
            res = run_plugin(p, d)
            res2 = run_plugin(p, d)
            h = res.entropy
            row = err_over_bar(h, refs.oracle_r, bars[d], p.L)
            row["scores"] = scores(h, refs, hp32[d], p.L)
            row["with_cpu_bar"] = err_over_bar(h, refs.oracle_r, bars["cpu"], p.L)
            row["plugin_repeat_bitwise_equal"] = bool(
                tcompare(res.entropy, res2.entropy).get("bitwise_equal")
            )
            row["arrays"] = {
                "entropy_q": b64(h.detach().cpu().numpy()[: p.L], "float32"),
                "stock_Hp32": b64(hp32[d], "float64"),
            }
            inst[d] = row
        instances[key] = inst

        reps[key] = {
            d: guarded(lambda d=d: repeats(p, d, n_rep[d], refs, bars[d], "plugin"))
            for d in ctx.devices
        }
        det[key] = {
            d: guarded(lambda d=d: repeats(p, d, n_rep[d], refs, bars[d], "deterministic"))
            for d in ctx.devices
        }
        det[key]["plugin_level"] = {
            d: guarded(
                lambda d=d: err_over_bar(
                    run_plugin(p, d, segments="deterministic").entropy, refs.oracle_r, bars[d], p.L
                )
            )
            for d in ctx.devices
        }

        all_cpu = {s: "cpu" for s in STAGE_NAMES}
        all_dev = {s: dev for s in STAGE_NAMES}
        st_cpu = run_staged(p, all_cpu)
        st_dev = run_staged(p, all_dev)
        eq_cpu = staged_vs_plugin(st_cpu, run_plugin(p, "cpu"))
        eq_dev = staged_vs_plugin(st_dev, run_plugin(p, dev))
        equal.setdefault("cpu_fields", {})[key] = eq_cpu
        equal.setdefault(f"{dev}_fields", {})[key] = eq_dev
        divergence[key] = stage_divergence(st_cpu, st_dev)

        def h_err(st: dict[str, dict[str, Any]]) -> dict[str, Any]:
            return err_over_bar(st["entropy"]["entropy"], refs.oracle_r, bars[dev], p.L)

        sw: dict[str, Any] = {"all_on_device": h_err(st_dev), "all_on_cpu": h_err(st_cpu)}
        for s in STAGE_NAMES:
            sw[f"only {s} on cpu"] = guarded(
                lambda s=s: h_err(run_staged(p, {**all_dev, s: "cpu"}))
            )
            sw[f"only {s} on device"] = guarded(
                lambda s=s: h_err(run_staged(p, {**all_cpu, s: dev}))
            )
        swap[key] = sw

        ctl: dict[str, Any] = {}
        ctl["fp64_on_device"] = guarded(
            lambda: err_over_bar(
                run_plugin(p, dev, dtype=torch.float64).entropy, refs.oracle_r, bars[dev], p.L
            )
        )
        for mode, tf32 in PRECISION_ARMS:

            def run_prec(mode: str = mode, tf32: bool | None = tf32) -> Any:
                with precision(mode, tf32):
                    return err_over_bar(run_plugin(p, dev).entropy, refs.oracle_r, bars[dev], p.L)

            ctl[f"plugin@{mode},tf32={tf32}"] = guarded(run_prec)
        if want_compile(ctx, dev):

            def run_compiled() -> Any:
                t = tables_for(p.toy, dev)
                x = p.x.to(F32).to(dev)

                def pipe(x: torch.Tensor) -> Any:
                    tree = tr.up_sweep(em.log_transfer(x, t, p.L))
                    return mg.constrained_marginals(
                        x, t, tree, vec(p.start, t.bucket, dev), vec(p.end, t.bucket, dev), p.L
                    ).entropy

                return err_over_bar(compiled(pipe)(x), refs.oracle_r, bars[dev], p.L)

            ctl["plugin_compiled_pipeline"] = guarded(run_compiled)
        else:
            ctl["plugin_compiled_pipeline"] = {"skipped": "small mode (pass --compile)"}
        controls[key] = ctl

    if not ctx.small:
        margins: dict[str, Any] = {}
        for s in (1, 2, 3, 4):
            for temp in temps:

                def run_margin(s: int = s, temp: float = temp) -> Any:
                    p, x64 = a_problem(temp, small=False, seed_rng=s, seed_gen=s)
                    refs = a_refs(p, x64)
                    return {
                        d: scores(run_plugin(p, d).entropy, refs, stock_h32(p.x, d), p.L)
                        for d in ctx.devices
                    }

                margins[f"seed{s}/T{temp}"] = guarded(run_margin)
        controls["margin_seeds"] = margins
    equal["cpu"] = all(all(v.values()) for v in equal["cpu_fields"].values())
    if dev != "cpu":
        equal[dev] = all(all(v.values()) for v in equal[f"{dev}_fields"].values())
    return {
        "instances": instances,
        "repeats": reps,
        "deterministic_segments": det,
        "stage_swap": swap,
        "stage_divergence": divergence,
        "controls": controls,
        "staged_equals_plugin": equal,
        "bar": "D2: err = |H(q) - oracle(rounded logits)| <= 10*|H(p)_fp32 - H(p)_fp64(rounded logits)| + 1e-6 "
        "per position, stock fp32 H(p) on the same device",
    }


# ============================================================================ P6: the (b) fixture


def b_problem(seed: int = 5) -> Problem:
    """``test_t0065_c5_no_nan_with_neg_inf_and_sentinels_on_cuda``'s fixture."""
    rng = random.Random(seed)
    toy = random_toy(rng, V=6, S=4, nfa=True)
    L = 6
    x = torch.tensor(random_logits(rng, L, 6, span=8.0), dtype=F32)
    keep = x.topk(3, dim=-1).indices
    filt = torch.full_like(x, float("-inf")).scatter(1, keep, x.gather(1, keep))
    filt[0, keep[0, 0]] = -1e20
    return Problem(toy, filt, [0], sorted(toy.finals), L)


def b_fixture(p: Problem) -> dict[str, Any]:
    """The row's own gate (enumerator on the ``clamp_min(-1e300)`` logits) and an independent brute-force check
    of whether any accepted string avoids the ``-inf`` (filtered) ids."""
    o = en.enumerate_log(
        en.log_softmax64(p.x.double().clamp_min(-1e300).tolist()), p.toy, p.start, p.end, p.L
    )
    allowed = [
        [v for v in range(p.x.shape[1]) if math.isfinite(float(p.x[i, v]))] for i in range(p.L)
    ]
    accepted = [
        s for s in itertools.product(*allowed) if en.accepts(p.toy, p.start, p.end, list(s))
    ]
    sent = [(i, v) for i in range(p.L) for v in range(p.x.shape[1]) if float(p.x[i, v]) == -1e20]
    exact = en.enumerate_log(en.log_softmax64(p.x.double().tolist()), p.toy, p.start, p.end, p.L)
    return {
        "exact_oracle_log_z (no clamp)": fnum(exact.log_z),
        "row_gate_oracle_log_z": fnum(o.log_z),
        "row_gate_passes (isfinite)": math.isfinite(o.log_z),
        "row_gate_oracle_entropy": [fnum(v) for v in o.entropy],
        "n_strings_over_finite_ids": int(np.prod([len(a) for a in allowed])),
        "n_accepted_over_finite_ids": len(accepted),
        "n_accepted_avoiding_the_-1e20_sentinel": sum(
            1 for s in accepted if all((i, v) not in sent for i, v in enumerate(s))
        ),
        "language_empty_under_the_-inf_mask": len(accepted) == 0,
        "toy": repr(p.toy),
        "logits": tolist(p.x),
    }


def _c5_vs_oracle(p: Problem, res: Any) -> dict[str, Any]:
    """Criterion 5's measurement: q and H(q) against the exact enumerator, and the q row sums."""
    o = en.enumerate_log(en.log_softmax64(p.x.double().tolist()), p.toy, p.start, p.end, p.L)
    if not math.isfinite(o.log_z):
        return {"oracle": "empty language"}
    q = res.q.detach().cpu().double().numpy()[: p.L]
    h = res.entropy.detach().cpu().double().numpy()[: p.L]
    dq = float(np.max(np.abs(q - np.array(o.q))))
    dh = float(np.max(np.abs(h - np.array(o.entropy))))
    ds = float(np.max(np.abs(q.sum(-1) - 1.0)))
    return {
        "dq": fnum(dq),
        "dH": fnum(dh),
        "max_abs_row_sum_minus_1": fnum(ds),
        "meets_criterion5 (dq<=1e-6, dH<=1e-5, rows 1+-1e-6)": dq <= 1e-6
        and dh <= 1e-5
        and ds <= 1e-6,
    }


def _empty_fixture(ctx: Ctx) -> dict[str, Any]:
    p = b_problem(C5_ORIGINAL_SEED)
    out: dict[str, Any] = {"seed": C5_ORIGINAL_SEED, **b_fixture(p)}
    for d in ctx.devices:
        r = run_plugin(p, d)
        t = tables_for(p.toy, d)
        m = mapdec.constrained_map(
            p.x.to(d), t, vec(p.start, t.bucket, d), vec(p.end, t.bucket, d), p.L
        )
        out[d] = {
            "marginals_log_z (unspecified at Z=0, T-0065 R6)": fnum(float(r.log_z)),
            "marginals_entropy (unspecified at Z=0)": tolist(r.entropy),
            "constrained_map_empty (the served z0 signal)": bool(m.empty.cpu()),
            "constrained_map_score": fnum(float(m.score.cpu())),
        }
    return out


def p6(ctx: Ctx) -> dict[str, Any]:
    p = b_problem(C5_FIXED_SEED)
    dev = ctx.target
    runs = {d: run_staged(p, {s: d for s in STAGE_NAMES}) for d in ctx.devices}
    runs["cpu_fp64"] = run_staged(p, {s: "cpu" for s in STAGE_NAMES}, dtype=torch.float64)
    equal: dict[str, Any] = {}
    for d in ctx.devices:
        fields = staged_vs_plugin(runs[d], run_plugin(p, d))
        equal[d] = all(fields.values())
        equal[f"{d}_fields"] = fields
    fields64 = staged_vs_plugin(runs["cpu_fp64"], run_plugin(p, "cpu", dtype=torch.float64))
    equal["cpu_fp64"] = all(fields64.values())
    stages: dict[str, Any] = {}
    dumps: dict[str, Any] = {}
    for rk, st in runs.items():
        stages[rk] = {}
        dumps[rk] = {}
        for name in STAGE_NAMES:
            named = {k: t for k2, v in st[name].items() for k, t in flatten(k2, v).items()}
            stages[rk][name] = {k: tstats(t) for k, t in named.items()}
            dumps[rk][name] = {k: tolist(t) for k, t in named.items()}
    res = {d: run_plugin(p, d) for d in ctx.devices}
    outputs = {
        d: {
            "log_z": fnum(float(r.log_z)),
            "entropy": tolist(r.entropy),
            "q_row_sums": tolist(r.q.sum(-1)),
            "vs_exact_oracle": _c5_vs_oracle(p, r),
        }
        for d, r in res.items()
    }
    margins: dict[str, Any] = {}
    for s in (1, 2, 3, 4):

        def run_margin(s: int = s) -> Any:
            pm = b_problem(s)
            fx = b_fixture(pm)
            return {
                "language_empty_under_the_-inf_mask": fx["language_empty_under_the_-inf_mask"],
                "row_gate_passes (isfinite)": fx["row_gate_passes (isfinite)"],
                **{
                    d: {
                        "log_z": fnum(float(r.log_z)),
                        "entropy": tolist(r.entropy),
                        "q_row_sums": tolist(r.q.sum(-1)),
                    }
                    for d in ctx.devices
                    for r in [run_plugin(pm, d)]
                },
            }

        margins[f"seed{s}"] = guarded(run_margin)
    return {
        "fixture": {"seed": C5_FIXED_SEED, **b_fixture(p)},
        "empty_fixture": guarded(lambda: _empty_fixture(ctx)),
        "stage_order": list(STAGE_NAMES),
        "plugin_outputs": outputs,
        "stages": stages,
        "dumps": dumps,
        "first_divergent_stage": {
            f"cpu_vs_{dev}": stage_divergence(runs["cpu"], runs[dev]),
            "cpu_fp32_vs_cpu_fp64": {
                k: v
                for k, v in stage_divergence(runs["cpu_fp64"], runs["cpu"]).items()
                if k != "per_stage"
            },
        },
        "staged_equals_plugin": equal,
        "margin_seeds": margins,
    }


# ============================================================================ P7: timing


def synthetic_tables(bucket: int, V: int, dev: str, seed: int = 0) -> Any:
    """``tests/container_scripts/astra_cd_perf_probe.py::synthetic_tables`` (T-0072) with ``V`` a parameter."""
    rng = np.random.default_rng(seed + bucket)
    S = bucket - 1
    members, neg, src, dst, cls = [], [], [], [], []
    for s in range(S):
        for _ in range(3):
            k = int(rng.integers(1, 65))
            members.append(np.sort(rng.choice(V, size=k, replace=False)).astype(np.int64))
            neg.append(bool(rng.random() < 0.1))
            src.append(s)
            dst.append(int(rng.integers(0, S)))
            cls.append(len(members) - 1)
    return em.build_tables(
        n_states=S,
        vocab_size=V,
        class_members=members,
        class_is_neg=neg,
        edge_src=np.array(src, dtype=np.int64),
        edge_dst=np.array(dst, dtype=np.int64),
        edge_class=np.array(cls, dtype=np.int64),
        finals=np.arange(0, S, 7, dtype=np.int64),
        bucket=bucket,
        device=dev,
    )


def _timed(fn: Callable[[], Any], dev: str, n: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    sync(dev)
    times = []
    for _ in range(n):
        if dev.startswith("cuda"):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            times.append(float(a.elapsed_time(b)))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    out: dict[str, Any] = {
        "n": n,
        "warmup": warmup,
        "median_ms": times[len(times) // 2],
        "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
        "clock": "cuda events" if dev.startswith("cuda") else "perf_counter",
    }
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        fn()
        torch.cuda.synchronize()
        out["peak_mem_delta_bytes"] = int(torch.cuda.max_memory_allocated() - base)
    return out


def p7(ctx: Ctx) -> dict[str, Any]:
    dev = ctx.target
    if ctx.small:
        CL, V, buckets, n, warmup = 32, 4096, (8, 16), 3, 1
    else:
        CL, V, buckets, n, warmup = 256, 262_144, (64, 128), 200, 20
    out: dict[str, Any] = {}
    with precision("high", None):
        for bucket in buckets:

            def run(bucket: int = bucket) -> dict[str, Any]:
                t = synthetic_tables(bucket, V, dev)
                g = torch.Generator().manual_seed(0)
                logits = (torch.randn(CL, V, generator=g) * 4.0).to(dev)
                a0 = torch.zeros(bucket, dtype=torch.bool, device=dev)
                a0[0] = True
                b = torch.ones(bucket, dtype=torch.bool, device=dev)
                cm = em.class_mass(logits, t)
                tree = tr.up_sweep(em.log_transfer(logits, t, CL, class_mass=cm))
                res = mg.constrained_marginals(logits, t, tree, a0, b, CL, class_mass=cm)
                return {
                    "log_z": fnum(float(res.log_z)),
                    "constrained_marginals": _timed(
                        lambda: mg.constrained_marginals(logits, t, tree, a0, b, CL, class_mass=cm),
                        dev,
                        n,
                        warmup,
                    ),
                    "log_transfer+up_sweep": _timed(
                        lambda: tr.up_sweep(em.log_transfer(logits, t, CL, class_mass=cm)),
                        dev,
                        n,
                        warmup,
                    ),
                }

            out[f"bucket{bucket}"] = guarded(run)
    return {
        "timings": out,
        "regime": {
            "CL": CL,
            "V": V,
            "buckets": list(buckets),
            "device": dev,
            "precision": "high",
            "automaton": "astra_cd_perf_probe.synthetic_tables (seeded), logits randn*4 seed 0",
        },
    }


# ============================================================================ driver

SECTION_FNS: dict[str, Callable[[Ctx], dict[str, Any]]] = {
    "P1": p1,
    "P2": p2,
    "P3": p3,
    "P4": p4,
    "P5": p5,
    "P6": p6,
    "P7": p7,
}


def regime(ctx: Ctx, argv: list[str]) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    try:
        vllm_dist = md.version("vllm")
    except md.PackageNotFoundError:
        vllm_dist = "absent"
    return {
        "probe": "T-0073 astra_cd_numerics_probe v2 (D2 scoring, repeats, deterministic segments)",
        "argv": argv,
        "device": ctx.device,
        "small": ctx.small,
        "compile_arms": "on" if (ctx.compile or not ctx.small) else "off (small)",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "vllm_dist": vllm_dist,
        "cuda_available": cuda,
        "device_name": torch.cuda.get_device_name(0) if cuda and ctx.device != "cpu" else "cpu",
        "device_capability": list(torch.cuda.get_device_capability(0))
        if cuda and ctx.device != "cpu"
        else None,
        "fp32_matmul_precision_at_start": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32_at_start": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32_at_start": torch.backends.cudnn.allow_tf32,
        "torch_num_threads": torch.get_num_threads(),
        "plugin_src": str(PLUGIN_SRC),
        "source_commit_of_staged_copies": SOURCE_COMMIT,
        "dtype": "float32 (fp64 only in named arms and oracles)",
    }


def _bar_c(stats: dict[str, Any]) -> bool:
    return bool(stats["max"] <= 1.0 and stats["frac_gt_0.51"] <= 0.05)


def verdicts(sections: dict[str, Any], devices: list[str]) -> dict[str, Any]:
    """Which tier-2 failures are test (fixture / oracle) defects and which are real, per device, from the data."""
    out: dict[str, Any] = {}

    def c() -> dict[str, Any]:
        ctl = sections["P4"]["oracle_controls"]["seed0"]
        res = {}
        for d in devices:
            row = _bar_c(ctl[d]["row_as_written (fp64-input oracle)"]["compensated"])
            rounded = _bar_c(ctl[d]["fp32-rounded-input oracle"]["compensated"])
            control_red = not _bar_c(ctl[d]["fp32-rounded-input oracle"]["uncompensated"])
            if rounded and not row:
                v = "oracle defect: the row's fp64-input oracle fails, the core meets the bar on rounded inputs"
            elif not rounded:
                v = "real: the core misses the bar against the fp32-rounded-input oracle"
            else:
                v = "pass"
            res[d] = {"verdict": v, "uncompensated_control_red": control_red}
        return res

    def b() -> dict[str, Any]:
        p6s = sections["P6"]
        empty = p6s["empty_fixture"]
        res: dict[str, Any] = {
            f"seed{C5_ORIGINAL_SEED} (T-0067 row)": (
                "fixture defect: empty language under the -inf mask, gate defeated by the -1e300 clamp"
                if empty.get("language_empty_under_the_-inf_mask")
                else "language non-empty"
            )
        }
        for d in devices:
            cmp = p6s["plugin_outputs"][d]["vs_exact_oracle"]
            ok = cmp.get("meets_criterion5 (dq<=1e-6, dH<=1e-5, rows 1+-1e-6)")
            res[f"seed{C5_FIXED_SEED} {d}"] = "pass" if ok else f"real: {cmp}"
            if d in empty:
                res[f"seed{C5_ORIGINAL_SEED} {d} z0 detected"] = empty[d][
                    "constrained_map_empty (the served z0 signal)"
                ]
        return res

    def a() -> dict[str, Any]:
        inst = sections["P5"]["instances"]
        res = {}
        for key, rows in inst.items():
            over = {d: rows[d]["worst_err_over_bar"] > 1.0 for d in devices}
            dev = devices[-1]
            if over["cpu"] and over[dev] and dev != "cpu":
                v = "real on CPU and device"
            elif over["cpu"]:
                v = "real on CPU"
            elif over[dev]:
                v = "real, device-specific"
            else:
                v = "pass"
            res[key] = {"verdict": v, **{d: rows[d]["worst_err_over_bar"] for d in devices}}
        return res

    for name, fn in (("a", a), ("b", b), ("c", c)):
        out[name] = guarded(fn)
    return out


def cap_payload(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True)
    if len(text.encode()) <= CAP_BYTES:
        return text
    p6s = payload["sections"].get("P6", {})
    if isinstance(p6s, dict) and "dumps" in p6s:
        p6s["dumps"] = {"dropped": f"payload exceeded {CAP_BYTES} bytes"}
    for per_temp in (payload["sections"].get("P5", {}).get("repeats") or {}).values():
        for row in per_temp.values() if isinstance(per_temp, dict) else []:
            if isinstance(row, dict) and "entropy_runs" in row:
                row["entropy_runs"] = "dropped (cap)"
    for sec in payload["sections"].values():
        for inst in (sec.get("instances") or {}).values() if isinstance(sec, dict) else []:
            for row in inst.values() if isinstance(inst, dict) else []:
                if isinstance(row, dict) and "per_position" in row:
                    row["per_position"] = "dropped (cap)"
    payload["regime"]["capped"] = True
    text = json.dumps(payload, sort_keys=True)
    if len(text.encode()) > CAP_BYTES:
        for name in SECTIONS:
            payload["sections"][name] = {
                "error": f"payload exceeded {CAP_BYTES} bytes after dropping dumps"
            }
        text = json.dumps(payload, sort_keys=True)
    return text


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda", help="cuda (runs CUDA and in-process CPU) or cpu")
    ap.add_argument("--small", action="store_true", help="tier-1 sizes; compiled arms off")
    ap.add_argument("--compile", action="store_true", help="run compiled arms even in --small")
    ap.add_argument(
        "--sections", default=",".join(SECTIONS), help="comma list; others recorded as skipped"
    )
    ap.add_argument(
        "--force-error", default="", help="(test hook) comma list of sections made to raise"
    )
    ap.add_argument(
        "--force-missing", default="", help="(test hook) comma list of sections left out"
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=0,
        help="torch CPU threads (default: 4 in --small, else torch's own)",
    )
    args = ap.parse_args(argv)
    threads = args.threads or (4 if args.small else 0)
    if threads:
        torch.set_num_threads(
            threads
        )  # tier-1 runs share a busy host; recorded as regime.torch_num_threads
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"{MARKER}-PREFLIGHT FAILED: --device {args.device} but CUDA is not available",
            flush=True,
        )
        return 3
    ctx = Ctx(
        device=args.device,
        small=args.small,
        compile=args.compile,
        force_error=frozenset(s for s in args.force_error.split(",") if s),
    )
    selected = [s for s in args.sections.split(",") if s]
    missing_hook = {s for s in args.force_missing.split(",") if s}
    t_all = time.perf_counter()
    reg = regime(ctx, argv)
    sections: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    for name in SECTIONS:
        if name in missing_hook:
            continue
        if name not in selected:
            sections[name] = {"skipped": "not selected (--sections)"}
            continue
        t0 = time.perf_counter()
        try:
            if name in ctx.force_error:
                raise RuntimeError(f"forced error in {name} (--force-error)")
            sections[name] = SECTION_FNS[name](ctx)
        except Exception:
            sections[name] = {"error": traceback.format_exc(limit=12)}
        seconds[name] = round(time.perf_counter() - t0, 3)
        print(f"[probe] {name} done in {seconds[name]:.1f}s", file=sys.stderr, flush=True)
    reg["section_seconds"] = seconds
    reg["total_seconds"] = round(time.perf_counter() - t_all, 3)
    reg["forbidden_modules_loaded"] = sorted(
        m for m in sys.modules if m.split(".")[0] in FORBIDDEN_MODULES
    )
    payload = {"regime": reg, "sections": sections, "verdicts": verdicts(sections, ctx.devices)}
    print(f"{MARKER} {cap_payload(payload)}", flush=True)
    return 0 if all(s in sections for s in SECTIONS) else 2


if __name__ == "__main__":
    sys.exit(main())
