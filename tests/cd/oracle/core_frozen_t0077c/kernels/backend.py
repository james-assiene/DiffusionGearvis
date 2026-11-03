"""Dispatch of the CD kernels: triton availability, the per-call decision and its once-per-process log (T-0077b).

A kernel runs exactly when the tensor is on CUDA, triton can be imported and the caller's ``kernels`` keyword is
not False (``dispatch_decision``, criterion 7). The device is checked first, so a CPU tensor never imports triton
(the CPU path is the reference and stays byte-for-byte the torch code). Whether triton imports is probed once per
process and cached.

The first decision of the process is logged, once, on the ``astra_vllm_plugin.cd.kernels`` logger: the backend
and the triton version, or the reason for the torch path. The message never starts with ``ASTRA_CD `` (the record
and golden captures read those lines). A layout the kernels do not take (not a contiguous fp32 ``[L, V]`` tensor)
goes to the torch path too; that is a layout guard, not a numerics fallback, and its first occurrence is logged
as well.

Compile options are passed per launch only (``launch_options``): FMA contraction off (the H-term product is
rounded on its own before the first TwoSum add) and libdevice's FTZ reflection off (torch on CUDA keeps
subnormals). Nothing here writes ``triton.knobs``, the environment or any torch/CUDA global (N6).
"""

from __future__ import annotations

import functools
import logging
import threading

import torch

LOGGER_NAME = "astra_vllm_plugin.cd.kernels"
log = logging.getLogger(LOGGER_NAME)

_lock = threading.Lock()
_logged = {"decision": False, "layout": False}


def launch_options() -> dict[str, bool]:
    """The compile options every kernel launch passes (per launch; never knobs or environment variables)."""
    return {"enable_fp_fusion": False, "enable_reflect_ftz": False}


def dispatch_decision(*, on_cuda: bool, triton_available: bool, kernels: bool) -> bool:
    """Pure: dispatch exactly when the tensor is on CUDA, triton imports and ``kernels`` is not False."""
    return bool(on_cuda and triton_available and kernels)


@functools.cache
def triton_status() -> tuple[str | None, str]:
    """``(version, "")`` if triton imports in this process, else ``(None, reason)``; probed once, cached."""
    try:
        import triton
    except (
        Exception
    ) as exc:  # an import failure of any kind means the torch path, with the reason logged
        return None, f"{type(exc).__name__}: {exc}"
    return str(triton.__version__), ""


def _log_once(key: str, level: int, msg: str) -> None:
    with _lock:
        if _logged[key]:
            return
        _logged[key] = True
    log.log(level, msg)


def use_kernels(t: torch.Tensor, *, kernels: bool) -> bool:
    """The dispatch for one call on ``t`` (the ``[L, V]`` input of a kernel entry); logs the first decision.

    Order: device (a CPU tensor never imports triton), the ``kernels`` keyword, triton's availability, then the
    layout guard (contiguous fp32 ``[L, V]``)."""
    on_cuda = bool(t.is_cuda)
    if not on_cuda:
        _log_once("decision", logging.INFO, "cd.kernels: torch path for the H(q) tail: cpu tensor")
        return False
    if not kernels:
        _log_once(
            "decision", logging.INFO, "cd.kernels: torch path for the H(q) tail: kernels=False"
        )
        return False
    version, reason = triton_status()
    if not dispatch_decision(
        on_cuda=on_cuda, triton_available=version is not None, kernels=kernels
    ):
        _log_once(
            "decision",
            logging.WARNING,
            f"cd.kernels: torch path for the H(q) tail: triton not importable ({reason})",
        )
        return False
    if t.dtype != torch.float32 or t.dim() != 2 or not t.is_contiguous():
        _log_once(
            "layout",
            logging.WARNING,
            "cd.kernels: torch path for the H(q) tail: not a contiguous fp32 [L, V] tensor "
            f"(dtype {t.dtype}, shape {tuple(t.shape)}, strides {tuple(t.stride())})",
        )
        return False
    opts = ", ".join(f"{k}={v}" for k, v in launch_options().items())
    _log_once(
        "decision",
        logging.INFO,
        f"cd.kernels: H(q) tail on triton {version} (cuda; launch options {opts})",
    )
    return True
