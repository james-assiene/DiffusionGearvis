"""CPU stand-ins for the vLLM helpers the vendored v0.29.0 sampler code imports (ours, not vendored).

Each stand-in keeps the image's name, signature and attribute names (read from the image's
``vllm/v1/worker/gpu/buffer_utils.py``, ``sample/output.py``, ``sample/states.py``, ``v1/outputs.py``
on 2026-09-30 by ``docker cp``, image not started) and replaces only the device mechanics:

- ``UvaBackedTensor``: ``.cpu`` / ``.np`` / ``.gpu`` / ``copy_to_uva()`` as upstream; the "UVA" view is a
  CPU copy of ``.np`` taken at ``copy_to_uva()`` time (upstream: a pooled pinned buffer, same values).
- ``async_copy_to_gpu`` / ``async_tensor_h2d``: a plain copy on CPU (no pinning: no CUDA here).
- ``current_platform.mem_get_info()``: a fixed 64 GiB free, so the stock tiling runs the whole batch
  in one tile (upstream says tiling is bit-identical anyway).
- the logprobs, temperature and min-p kernels are not modelled: calling them is an assertion, so a
  test that strays onto those paths fails loudly instead of running something invented.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np
import torch

NO_LOGPROBS = -1
_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max

SamplingParams = Any


class LogprobsTensors(NamedTuple):  # vllm/v1/outputs.py:95
    logprob_token_ids: torch.Tensor
    logprobs: torch.Tensor
    selected_token_ranks: torch.Tensor
    cu_num_generated_tokens: list[int] | None = None
    cu_num_generated_tokens_tensor: torch.Tensor | None = None


@dataclass
class SamplerOutput:  # vllm/v1/worker/gpu/sample/output.py
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
    num_nans: torch.Tensor | None
    num_sampled: torch.Tensor | None
    num_rejected: torch.Tensor | None = None
    sampling_mask_tensors: Any | None = None


class UvaBackedTensor:  # vllm/v1/worker/gpu/buffer_utils.py:91
    def __init__(self, size: Any, dtype: torch.dtype, max_concurrency: int | None = None):
        self.dtype = dtype
        self.cpu = torch.zeros(size, dtype=dtype, device="cpu")
        self.np = self.cpu.numpy()
        self.gpu = torch.from_numpy(self.np.copy())

    def copy_to_uva(self, n: int | None = None) -> torch.Tensor:
        self.gpu = torch.from_numpy((self.np[:n] if n is not None else self.np).copy())
        return self.gpu


def async_copy_to_gpu(x: Any, out: torch.Tensor | None = None, device: Any = None) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    assert x.is_cpu
    if out is None:
        assert device is not None
        out = torch.empty_like(x, device=device)
    return out.copy_(x, non_blocking=True)


def async_tensor_h2d(data: list[int], device: Any = None, **_: Any) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.int64, device=device)


current_platform = SimpleNamespace(
    mem_get_info=lambda: (64 << 30, 64 << 30),
    is_cpu=lambda: True,
    simple_compile_backend="eager",
)


def use_penalty(sampling_params: Any) -> bool:
    return bool(
        getattr(sampling_params, "repetition_penalty", 1.0) != 1.0
        or getattr(sampling_params, "frequency_penalty", 0.0) != 0.0
        or getattr(sampling_params, "presence_penalty", 0.0) != 0.0
    )


def _not_modelled(name: str) -> Any:
    def f(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"fake vLLM: {name} is not modelled on CPU (a test strayed onto it)")

    return f


compute_topk_scores = _not_modelled("compute_topk_scores (logprobs)")
apply_temperature = _not_modelled("apply_temperature")
apply_min_p = _not_modelled("apply_min_p")


class _Logger(logging.LoggerAdapter):  # vllm's logger has warning_once
    def warning_once(self, msg: str, *args: Any) -> None:
        self.warning(msg, *args)


logger = _Logger(logging.getLogger("vllm.model_executor.models.diffusion_gemma"), {})
