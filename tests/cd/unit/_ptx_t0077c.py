"""T-0077c PTX inspection (CPU, no GPU, no driver): run as a subprocess by ``test_cd_logmatmul_backend.py``.

    python -m tests.cd.unit._ptx_t0077c      # cwd = the repository root; TRITON_INTERPRET unset

Every ``@triton.jit`` launch the entry makes is compiled ahead of time for ``GPUTarget("cuda", 120, 32)`` (the 5090)
instead of being launched: triton's driver is replaced by one that only names that target (and refuses anything else),
and ``JITFunction.run`` is wrapped to compile with ``warmup=True`` (triton's own compile path, the launch's own
arguments and options) and record the PTX. Nothing runs on a device. The entry is called on CPU tensors outside the
interpreter (API ruling: the entry itself does not check the device; ``backend.use_logmatmul`` does).

Prints one ``ASTRA_T0077C_PTX_JSON <json>`` line: per variant (comp/plain) and shape, the captured kernels, with the
set of ``.ftz`` mnemonics, whether ``max.NaN.f32``, ``ex2.approx.f32`` (non-FTZ: ``tl.exp``'s lowering, absent from
libdevice ``__nv_expf``), ``mma`` occur, and the launch options seen.

Regime: CPU, triton 3.7.1 (``uv.lock``), target sm_120 / warp 32.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
import traceback
from typing import Any

import torch

from tests.cd.unit import _frozen_t0077c as c

MARKER = "ASTRA_T0077C_PTX_JSON"
SHAPES = {
    "mm_2x33x91x17": ((2, 33, 91), (2, 91, 17)),
    "vec_mat_3x1x33x33": ((3, 1, 33), (3, 33, 33)),
    "mat_vec_3x33x33x1": ((3, 33, 33), (3, 33, 1)),
    "mm_1x40x205x40": ((1, 40, 205), (1, 205, 40)),
}
OPTION_KEYS = ("enable_fp_fusion", "enable_reflect_ftz")


def _ops(ptx: str) -> set[str]:
    out = set()
    for line in ptx.splitlines():
        m = re.match(r"\s*(?:@%?\w+\s+)?([a-z][\w.]*)\s", line)
        if m:
            out.add(m.group(1))
    return out


_TOY: dict[str, Any] = {}


def _toy_kernel() -> Any:
    """Harness control: a kernel with the spec's ingredients (libdevice or ``tl`` exp, a NaN-propagating max)."""
    if "k" not in _TOY:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice

        _TOY.update(tl=tl, libdevice=libdevice)

        @triton.jit
        def toy(x, y, n, B: tl.constexpr, LIBDEVICE: tl.constexpr):
            o = tl.arange(0, B)
            v = tl.load(x + o, mask=o < n, other=0.0)
            e = libdevice.exp(v) if LIBDEVICE else tl.exp(v)
            w = tl.maximum(v, e * v - 1.0, propagate_nan=tl.PropagateNan.ALL)
            tl.store(y + o, w, mask=o < n)

        @triton.jit
        def toy_log(x, y, n, B: tl.constexpr, LOGMODE: tl.constexpr):
            o = tl.arange(0, B)
            v = tl.load(x + o, mask=o < n, other=1.0)
            if LOGMODE == 0:
                w = libdevice.log(v)
            elif LOGMODE == 1:
                w = tl.log(v)
            else:  # an approximate log: what the lg2 rule must catch
                w = tl.inline_asm_elementwise(
                    "lg2.approx.f32 $0, $1;", "=r,r", [v], dtype=tl.float32, is_pure=True, pack=1
                )
            tl.store(y + o, w, mask=o < n)

        _TOY["k"] = toy
        _TOY["log"] = toy_log
    return _TOY["k"]


def toy_log_entry(x: torch.Tensor, *, logmode: int) -> torch.Tensor:
    _toy_kernel()
    y = torch.empty_like(x)
    _TOY["log"][(1,)](
        x, y, x.numel(), B=64, LOGMODE=logmode, enable_fp_fusion=False, enable_reflect_ftz=False
    )
    return y


def toy_entry(x: torch.Tensor, *, libdevice: bool, reflect_ftz: bool) -> torch.Tensor:
    y = torch.empty_like(x)
    _toy_kernel()[(1,)](
        x,
        y,
        x.numel(),
        B=64,
        LIBDEVICE=libdevice,
        enable_fp_fusion=False,
        enable_reflect_ftz=reflect_ftz,
    )
    return y


TOY_ARMS = {"libdevice": (True, False), "tl_exp": (False, False), "reflect_ftz_on": (True, True)}
#: log arms of the harness control: libdevice ``__nv_logf``, ``tl.log``, and an inline ``lg2.approx.f32`` (triton
#: 3.7.1 lowers neither ``tl.log`` nor ``tl.log2`` to ``lg2.approx`` at sm_120, measured 2026-10-06)
TOY_LOG_ARMS = {"libdevice_log": 0, "tl_log": 1, "lg2_approx_asm": 2}


def _instr_sha(ptx: str) -> str:
    """sha256 of the instruction stream (no comments, directives or labels): equal = the same machine code."""
    lines = [ln.strip() for ln in ptx.splitlines()]
    body = [ln for ln in lines if ln and ln[0].isalpha() and not ln.startswith("//")]
    return hashlib.sha256("\n".join(body).encode()).hexdigest()


def main() -> int:
    toy = "--toy" in sys.argv
    out: dict[str, Any] = {"error": None, "kernels": {}, "toy": toy}
    try:
        from triton.backends.compiler import GPUTarget
        from triton.runtime.driver import driver
        from triton.runtime.jit import JITFunction

        target = GPUTarget("cuda", 120, 32)

        class AotDriver:
            def get_current_device(self) -> int:
                return 0

            def get_current_stream(self, device: Any = None) -> int:
                return 0

            def get_current_target(self) -> Any:
                return target

            def __getattr__(self, name: str) -> Any:
                raise AssertionError(
                    f"AOT driver: {name} needs a GPU (PTX inspection compiles only)"
                )

        driver.set_active(AotDriver())
        captured: list[dict[str, Any]] = []
        real_run = JITFunction.run

        def run(self: Any, *args: Any, grid: Any, warmup: bool, **kwargs: Any) -> Any:
            opts = {k: kwargs.get(k, "<default>") for k in OPTION_KEYS}
            k = real_run(self, *args, grid=grid, warmup=True, **kwargs)
            ptx = k.asm["ptx"]
            ops = _ops(ptx)
            captured.append(
                {
                    "name": self.fn.__name__,
                    "options": opts,
                    "ftz_ops": sorted(o for o in ops if ".ftz" in o),
                    "max_nan": "max.NaN.f32" in ops,
                    "ex2_approx_nonftz": "ex2.approx.f32" in ops,
                    "lg2_approx": sorted(o for o in ops if o.startswith("lg2.approx")),
                    "instr_sha": _instr_sha(ptx),
                    "mma": sorted(o for o in ops if "mma" in o),
                    "n_lines": len(ptx.splitlines()),
                }
            )
            return k

        JITFunction.run = run  # type: ignore[method-assign]
        if toy:
            globals()["triton"] = importlib.import_module("triton")
            for arm, (libdev, ftz) in TOY_ARMS.items():
                captured.clear()
                toy_entry(-torch.rand(10), libdevice=libdev, reflect_ftz=ftz)
                out["kernels"][arm] = list(captured)
            for arm, mode in TOY_LOG_ARMS.items():
                captured.clear()
                toy_log_entry(torch.rand(10) + 0.5, logmode=mode)
                out["kernels"][arm] = list(captured)
            print(MARKER + " " + json.dumps(out), flush=True)
            return 0
        if str(c.PLUGIN_SRC) not in sys.path:
            sys.path.insert(0, str(c.PLUGIN_SRC))
        lm = importlib.import_module("astra_vllm_plugin.cd.kernels.logmatmul").log_matmul
        for variant, comp in (("comp", True), ("plain", False)):
            for shape_name, (sa, sb) in SHAPES.items():
                captured.clear()
                a3 = -torch.rand(*sa)
                b3 = -torch.rand(*sb)
                try:
                    lm(a3, b3, compensated=comp)
                    out["kernels"][f"{variant}-{shape_name}"] = list(captured)
                except Exception as exc:
                    out["kernels"][f"{variant}-{shape_name}"] = {
                        "error": f"{type(exc).__name__}: {exc}",
                        "captured": list(captured),
                    }
    except Exception:
        out["error"] = traceback.format_exc()
    print(MARKER + " " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
