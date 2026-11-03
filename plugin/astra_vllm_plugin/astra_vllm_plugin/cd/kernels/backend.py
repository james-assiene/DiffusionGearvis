"""Dispatch of the CD kernels: triton availability, the per-call decision and its once-per-process log (T-0077b,
T-0077c).

Two families: the H(q) tail (``use_kernels``, T-0077b) and the log-semiring product (``use_logmatmul`` with its
layout guard ``logmatmul_layout_ok``, T-0077c). They share the decision record: one per process, naming the family
that decided first (ruling O7).

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
#: "decision" is shared by both kernel families (one decision record per process, ruling O7); each family logs its
#: own first layout refusal
_logged = {"decision": False, "layout": False, "layout_logmatmul": False}


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


def logmatmul_layout_ok(a3: torch.Tensor, b3: torch.Tensor) -> bool:
    """Pure: the ``log_matmul`` kernel takes the operands (reads only dtype, ``dim()`` and shape; any strides).

    True exactly when both are fp32, both 3-D (``[nb, n, m]`` and ``[nb, m, k]`` after the host broadcast), the batch
    and inner dims agree and every dim is ≥ 1."""
    if a3.dtype != torch.float32 or b3.dtype != torch.float32:
        return False
    if a3.dim() != 3 or b3.dim() != 3:
        return False
    nb, n, m = a3.shape
    nb2, m2, k = b3.shape
    return bool(nb == nb2 and m == m2 and min(nb, n, m, k) >= 1)


def use_logmatmul(a3: torch.Tensor, b3: torch.Tensor, *, kernels: bool) -> bool:
    """The dispatch of the ``log_matmul`` family for one call on ``a3 ⊗ b3``; logs the process's first decision.

    Order: device (CPU operands never import triton), the ``kernels`` keyword, triton's availability, then the layout
    guard (``logmatmul_layout_ok``; a refusal is a layout guard, not a numerics fallback, logged once on its own). The
    decision record is shared with the H(q) family (``use_kernels``): one per process, naming the family that decided
    first (ruling O7)."""
    on_cuda = bool(a3.is_cuda and b3.is_cuda)
    if not on_cuda:
        _log_once("decision", logging.INFO, "cd.kernels: torch path for log_matmul: cpu tensor")
        return False
    if not kernels:
        _log_once("decision", logging.INFO, "cd.kernels: torch path for log_matmul: kernels=False")
        return False
    version, reason = triton_status()
    if not dispatch_decision(
        on_cuda=on_cuda, triton_available=version is not None, kernels=kernels
    ):
        _log_once(
            "decision",
            logging.WARNING,
            f"cd.kernels: torch path for log_matmul: triton not importable ({reason})",
        )
        return False
    if not logmatmul_layout_ok(a3, b3):
        _log_once(
            "layout_logmatmul",
            logging.WARNING,
            "cd.kernels: torch path for log_matmul: not fp32 [nb, n, m] x [nb, m, k] operands "
            f"(dtypes {a3.dtype}/{b3.dtype}, shapes {tuple(a3.shape)}/{tuple(b3.shape)})",
        )
        return False
    if _logged["decision"]:
        # per-call fast path: the process's decision record is already out (flags only ever go False -> True,
        # so an unlocked read can only miss a record being written now, and _log_once re-checks under the lock)
        return True
    opts = ", ".join(f"{k}={v}" for k, v in launch_options().items())
    _log_once(
        "decision",
        logging.INFO,
        f"cd.kernels: log_matmul on triton {version} (cuda; launch options {opts})",
    )
    return True
