"""T-0077b criteria 3-4 under ``TRITON_INTERPRET=1`` (CPU): run as a subprocess by ``test_cd_kernel_backend.py``.

    TRITON_INTERPRET=1 python -m tests.cd.unit._interp_t0077b      # cwd = the repository root

A separate process, so triton's process-wide kernel cache never mixes interpreted and compiled kernels (task
"Tests to write"). Prints one ``ASTRA_T0077B_INTERP_JSON <json>`` line; every check is computed here and asserted
row by row in the parent. Entries and their API: ``_frozen_t0077b`` (ruling K1).

- ``row_sum``: ``entropy.compensated_row_sum(x, block=b)`` vs the branch base's ``semiring.compensated_sum(x)``,
  bit for bit (fp32 bit patterns, ``-0.0`` included), at sizes 1, 2, 3, 5, 2^k ± 1 (k = 3..12), L ∈ {1, 3, 8},
  block ∈ {production, 16} (16: several blocks per row from V = 17 on).
- ``block``: the same input under blocks 16, 32, 64, 256, 1024 and the production one: every result is the
  reference's bits (so the bits do not depend on the block size).
- ``row_sum_adv``: the adversarial row-sum inputs (signed, cancelling pairs, 1e-30..1e3) and the H-term rows
  built from the adversarial ``q_raw`` set.
- ``tail``: ``entropy.hq_tail(q_raw, block=b)`` vs the branch base's ``constrained_marginals`` fed the same
  ``q_raw`` (``tail_via("frozen", ...)``): ``q`` bit for bit and in place; ``entropy`` NaN positions equal and
  max |dH| (ruling M2: N4, 1e-5) with its max ulp reported (the interpreter's log is numpy's).

Regime: CPU, torch and triton per ``uv.lock`` (2.13.0+cpu, 3.7.1), numpy per ``uv.lock``, fp32; seeds per case.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
import traceback
from typing import Any

import torch

from tests.cd.unit import _frozen_t0077b as kb

MARKER = "ASTRA_T0077B_INTERP_JSON"
SIZES = (1, 2, 3, 5, *sorted({v for k in range(3, 13) for v in (2**k - 1, 2**k + 1)}))
LS = (1, 3, 8)
ROW_SUM_BLOCKS: tuple[int | None, ...] = (None, 16)
INVARIANCE_BLOCKS: tuple[int | None, ...] = (16, 32, 64, 256, 1024, None)
INVARIANCE_SHAPES = ((3, 1000), (2, 4097))
TAIL_BLOCKS: tuple[int | None, ...] = (None, 16)


def _l_for(V: int) -> tuple[int, ...]:
    return LS if V <= 1025 else (1,)


def row_sum_keys() -> list[str]:
    return [f"V{V}-L{L}-b{b}" for V in SIZES for L in _l_for(V) for b in ROW_SUM_BLOCKS]


def block_keys() -> list[str]:
    return [f"{L}x{V}" for L, V in INVARIANCE_SHAPES]


def row_sum_adv_keys() -> list[str]:
    names = list(kb.row_sum_cases()) + [f"term:{n}" for n in kb.tail_cases()]
    return [f"{n}-b{b}" for n in names for b in ROW_SUM_BLOCKS]


def tail_keys() -> list[str]:
    return [f"{n}-b{b}" for n in kb.tail_cases() for b in TAIL_BLOCKS]


def _bname(b: int | None) -> str:
    return "prod" if b is None else str(b)


def sized_input(V: int, L: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 * V + L)
    mag = torch.exp(torch.randn(L, V, generator=g, dtype=torch.float64) * 6.0)
    sign = torch.where(torch.rand(L, V, generator=g) < 0.5, -1.0, 1.0).to(torch.float64)
    x = (mag * sign).to(torch.float32)
    if V == 1:
        x[0, 0] = -0.0  # V = 1 returns x: the sign of zero survives
    return x


def _terms(q: torch.Tensor) -> torch.Tensor:
    """The H-term row the torch tail sums (inputs for the row-sum entry; a sum input, not a reference)."""
    sem = kb.frozen("semiring")
    tot = sem.compensated_sum(q).unsqueeze(-1).clamp_min(torch.finfo(torch.float32).tiny)
    lt = torch.log(tot)
    return torch.maximum(torch.log(q.clamp_min(1e-38)), lt + math.log(1e-30)) * q


def _sum_check(ent: Any, x: torch.Tensor, block: int | None) -> dict[str, Any]:
    ref = kb.frozen("semiring").compensated_sum(x)
    try:
        got = ent.compensated_row_sum(x.clone(), block=block)
    except Exception as exc:  # reported per row
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    d = kb.bit_diff(ref, got)
    return {"ok": d is None, "detail": d}


def main() -> int:
    out: dict[str, Any] = {
        "triton_interpret": os.environ.get("TRITON_INTERPRET"),
        "error": None,
        "row_sum": {},
        "block": {},
        "row_sum_adv": {},
        "tail": {},
    }
    try:
        if str(kb.PLUGIN_SRC) not in sys.path:
            sys.path.insert(0, str(kb.PLUGIN_SRC))
        ent = importlib.import_module("astra_vllm_plugin.cd.kernels.entropy")
        for V in SIZES:
            for L in _l_for(V):
                x = sized_input(V, L)
                for b in ROW_SUM_BLOCKS:
                    out["row_sum"][f"V{V}-L{L}-b{b}"] = _sum_check(ent, x, b)
        for L, V in INVARIANCE_SHAPES:
            x = sized_input(V, L)
            out["block"][f"{L}x{V}"] = {_bname(b): _sum_check(ent, x, b) for b in INVARIANCE_BLOCKS}
        adv = {**kb.row_sum_cases(), **{f"term:{n}": _terms(q) for n, q in kb.tail_cases().items()}}
        for n, x in adv.items():
            for b in ROW_SUM_BLOCKS:
                out["row_sum_adv"][f"{n}-b{b}"] = _sum_check(ent, x, b)
        for n, q_raw in kb.tail_cases().items():
            ref = kb.tail_via("frozen", q_raw)
            for b in TAIL_BLOCKS:
                buf = q_raw.clone()
                row: dict[str, Any] = {}
                try:
                    q, h = ent.hq_tail(buf, block=b)
                except Exception as exc:
                    out["tail"][f"{n}-b{b}"] = {"error": f"{type(exc).__name__}: {exc}"}
                    continue
                row["q"] = kb.bit_diff(ref.q, q)
                row["in_place"] = bool(q.data_ptr() == buf.data_ptr())
                row["nan_positions_equal"] = bool(
                    torch.equal(torch.isnan(ref.entropy), torch.isnan(h))
                )
                keep = ~(torch.isnan(ref.entropy) | torch.isnan(h))
                dh = (ref.entropy.double() - h.double()).abs()[keep]
                row["max_abs_dh"] = float(dh.max()) if dh.numel() else 0.0
                row["finite_dh"] = bool(torch.isfinite(dh).all())
                row["max_ulp_h"] = kb.max_ulp(ref.entropy, h)
                row["shape_h"] = list(h.shape)
                out["tail"][f"{n}-b{b}"] = row
    except Exception:
        out["error"] = traceback.format_exc()
    print(MARKER + " " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
