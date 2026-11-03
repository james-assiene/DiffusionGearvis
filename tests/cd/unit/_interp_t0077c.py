"""T-0077c criterion 3c under ``TRITON_INTERPRET=1`` (CPU): run as a subprocess by ``test_cd_logmatmul_backend.py``.

    TRITON_INTERPRET=1 python -m tests.cd.unit._interp_t0077c      # cwd = the repository root

A separate process, so triton's process-wide kernel cache never mixes interpreted and compiled kernels. Prints one
``ASTRA_T0077C_INTERP_JSON <json>`` line; every check is computed here and asserted row by row in the parent. API:
``_frozen_t0077c`` (``kernels.logmatmul.log_matmul(a3, b3, *, compensated, block_n, block_k)``).

- ``bits``: the entry vs ``spec_log_matmul`` with numpy fp64 exp/log rounded once (ruling O2), bit for bit (fp32
  patterns, NaN positions), plain and compensated, at inner {1, 2, 3, 31, 32, 33, 91} (n, k ≤ 33, nb ≤ 3), strided
  live-block views, vec-mat, mat-vec, broadcast batch, two-band, all-sentinel, NaN/±inf, near-DEAD and the near-tie
  pins; each also checks the output is a new fp32 ``[nb, n, k]`` tensor and the inputs (and the block they view) are
  unchanged.
- ``tiles``: the same inputs under several ``(block_n, block_k)``: every result is the spec's bits (criterion 3b).
- ``bars``: criterion 4 at outer 64 (T-0065's generator): compensated meets the wide bar (inner 32, 91) and the gauged
  bar (inner 32, 91); ``compensated=False`` fails the compensated bar on T-0065's wide inner-64 input (power control).

Regime: CPU, torch 2.13.0+cpu, triton 3.7.1, numpy 2.4.6 (``uv.lock``), fp32; seeds per case.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from collections.abc import Callable
from typing import Any

import torch

from tests.cd.unit import _frozen_t0077c as c

MARKER = "ASTRA_T0077C_INTERP_JSON"
VARIANTS = {"comp": True, "plain": False}
TILE_BLOCKS: tuple[tuple[int | None, int | None], ...] = (
    (None, None),
    (2, 4),
    (16, 16),
    (32, 32),
    (64, 8),
)
SMALL_TILE_BLOCKS: tuple[tuple[int | None, int | None], ...] = ((1, 1), (1, 2), (4, 1))


def _gauged_like(nb: int, n: int, m: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(7000 + 100 * m + seed)
    a = (-2.35 - torch.rand(nb, n, m, generator=g)).float()
    b = (-2.35 - torch.rand(nb, m, k, generator=g)).float()
    a[torch.rand(nb, n, m, generator=g) < 0.3] = c.SENTINEL32
    b[torch.rand(nb, m, k, generator=g) < 0.3] = c.SENTINEL32
    return a, b


def _t65(regime: str, inner: int, n: int = 33) -> tuple[torch.Tensor, torch.Tensor]:
    lo, hi = c.REGIMES[regime]
    a, b = c.t0065_operands(lo, hi, inner, 0, n=n)
    return a.unsqueeze(0), b.unsqueeze(0)


def cases() -> dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]]:
    """name -> a builder of ``(a3, b3)`` (3-D, fp32, CPU). Builders, so the parent can enumerate names cheaply."""
    out: dict[str, Callable[[], tuple[torch.Tensor, torch.Tensor]]] = {}
    for m in (1, 2, 3, 31, 32, 33, 91):
        out[f"inner{m}"] = lambda m=m: _gauged_like(3 if m <= 3 else 2, 33, m, 17, 0)
    for m in (31, 32, 33, 91):
        out[f"wide{m}"] = lambda m=m: _t65("wide", m)
        out[f"gauged{m}"] = lambda m=m: _t65("gauged", m)
    out["live_block"] = lambda: c.live_block_views(3, 33, 40, 0)
    out["live_block_s1"] = lambda: c.live_block_views(2, 1, 8, 1)
    out["vec_mat"] = lambda: c.vec_mat(3, 33, 40, 0)
    out["mat_vec"] = lambda: c.mat_vec(3, 33, 40, 0)
    out["broadcast"] = lambda: c.broadcast_batch(3, 5, 7, 4, 0)
    out["broadcast33"] = lambda: c.broadcast_batch(2, 33, 33, 9, 1)
    out["two_band"] = lambda: c.two_band(32, 0, n=33, k=33)
    out["all_sentinel1"] = lambda: c.all_sentinel(1, 0)
    out["all_sentinel33"] = lambda: c.all_sentinel(33, 0)
    out["nan_inf3"] = lambda: c.nan_inf(3, 0)
    out["nan_inf33"] = lambda: c.nan_inf(33, 0)
    out["near_dead"] = c.near_dead
    for m in c.NEAR_TIE_PINS:
        out[f"near_tie{m}"] = lambda m=m: c.near_tie_pinned(m)
    return out


TILE_CASES = (
    "inner33",
    "inner91",
    "live_block",
    "broadcast33",
    "near_tie33",
    "near_tie91",
    "nan_inf33",
)
SMALL_TILE_CASES = ("broadcast", "near_tie31", "near_dead")
BAR_CELLS = (("wide", 32), ("wide", 91), ("gauged", 32), ("gauged", 91))


def bits_keys() -> list[str]:
    return [f"{n}-{v}" for n in cases() for v in VARIANTS]


def _bname(b: tuple[int | None, int | None]) -> str:
    return f"{b[0]}x{b[1]}"


def tile_keys() -> list[str]:
    return [f"{n}-{v}" for n in (*TILE_CASES, *SMALL_TILE_CASES) for v in VARIANTS]


def bar_keys() -> list[str]:
    return [f"{r}{i}" for r, i in BAR_CELLS] + ["plain_power_wide64"]


def _spec(a3: torch.Tensor, b3: torch.Tensor, compensated: bool) -> torch.Tensor:
    return c.spec_log_matmul(a3, b3, compensated=compensated, exp=c.np64_exp, log=c.np64_log)


def _call(
    lm: Callable[..., torch.Tensor],
    a3: torch.Tensor,
    b3: torch.Tensor,
    compensated: bool,
    block: tuple[int | None, int | None] = (None, None),
) -> torch.Tensor:
    return lm(a3, b3, compensated=compensated, block_n=block[0], block_k=block[1])


def _check(
    lm: Callable[..., torch.Tensor],
    a3: torch.Tensor,
    b3: torch.Tensor,
    compensated: bool,
    block: tuple[int | None, int | None] = (None, None),
) -> dict[str, Any]:
    want = _spec(a3, b3, compensated)
    base_a = a3._base if a3._base is not None else a3
    base_b = b3._base if b3._base is not None else b3
    snap = (base_a.clone(), base_b.clone())
    try:
        got = _call(lm, a3, b3, compensated, block)
    except Exception as exc:  # reported per row
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    problems = []
    if not isinstance(got, torch.Tensor):
        return {"ok": False, "detail": f"returned {type(got).__name__}"}
    if got.dtype != torch.float32 or tuple(got.shape) != tuple(want.shape):
        problems.append(f"shape/dtype {tuple(got.shape)} {got.dtype} != {tuple(want.shape)} fp32")
    if got.data_ptr() in (base_a.data_ptr(), base_b.data_ptr()):
        problems.append("the output aliases an input")
    if not torch.equal(base_a.view(torch.int32), snap[0].view(torch.int32)):
        problems.append("the entry wrote into a3's storage")
    if not torch.equal(base_b.view(torch.int32), snap[1].view(torch.int32)):
        problems.append("the entry wrote into b3's storage")
    if not problems:
        d = c.bit_diff(want, got)
        if d is not None:
            problems.append(d)
    return {"ok": not problems, "detail": "; ".join(problems) or None}


def main() -> int:
    out: dict[str, Any] = {
        "triton_interpret": os.environ.get("TRITON_INTERPRET"),
        "error": None,
        "bits": {},
        "tiles": {},
        "bars": {},
    }
    try:
        if str(c.PLUGIN_SRC) not in sys.path:
            sys.path.insert(0, str(c.PLUGIN_SRC))
        lm = importlib.import_module("astra_vllm_plugin.cd.kernels.logmatmul").log_matmul
        built = {n: f() for n, f in cases().items()}
        for n, (a3, b3) in built.items():
            for v, comp in VARIANTS.items():
                out["bits"][f"{n}-{v}"] = _check(lm, a3, b3, comp)
        for n in TILE_CASES + SMALL_TILE_CASES:
            a3, b3 = built[n]
            blocks = TILE_BLOCKS + (SMALL_TILE_BLOCKS if n in SMALL_TILE_CASES else ())
            for v, comp in VARIANTS.items():
                out["tiles"][f"{n}-{v}"] = {_bname(b): _check(lm, a3, b3, comp, b) for b in blocks}
        for regime, inner in BAR_CELLS:
            a3, b3 = c.regime_instance(regime, inner, 0)
            exp = c.regime_oracle(regime, inner, 0)
            try:
                got = _call(lm, a3, b3, True)
                ok, msg = (c.wide_ok if regime == "wide" else c.gauged_ok)(got, exp)
                out["bars"][f"{regime}{inner}"] = {"ok": ok, "detail": msg}
            except Exception as exc:
                out["bars"][f"{regime}{inner}"] = {
                    "ok": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                }
        lo, hi = c.REGIMES["wide"]
        a, b = c.t0065_operands(lo, hi, 64, 0)
        a3, b3 = a.unsqueeze(0), b.unsqueeze(0)
        try:
            ok, msg = c.wide_ok(_call(lm, a3, b3, False), c.lse_oracle(a3, b3))
            out["bars"]["plain_power_wide64"] = {"ok": not ok, "detail": f"plain: {msg}"}
        except Exception as exc:
            out["bars"]["plain_power_wide64"] = {
                "ok": False,
                "detail": f"{type(exc).__name__}: {exc}",
            }
    except Exception:
        out["error"] = traceback.format_exc()
    print(MARKER + " " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
