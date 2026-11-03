"""Prompts, byte-identical to diffgemma-fa's builders at 27c7d0d (T-0070 criterion 2).

BFCL: ``eval/run.py::build_prompt(rec, fn, style="stock")``; Sudoku: ``compile/tasks/sudoku.py::
build_prompt(rec)``. Both are vendored verbatim (``_vendor/run_prompt.py``, ``_vendor/tasks/sudoku.py``).
Each prompt is sent as one user message with thinking off (``request_body``).
"""

from __future__ import annotations

from typing import Any

from local_astra.eval.cd._vendor import run_prompt as _run_prompt
from local_astra.eval.cd._vendor.tasks import sudoku as _sudoku

__all__ = ["MAX_TOKENS", "build_prompt", "request_body"]

#: The JAX single-canvas regime (T-0070 implementation notes).
MAX_TOKENS = 256


def build_prompt(task: str, record: Any) -> str:
    if task == "bfcl":
        prompt: str = _run_prompt.build_prompt(record, record.function, style="stock")
        return prompt
    if task == "sudoku":
        text: str = _sudoku.build_prompt(record)
        return text
    raise ValueError(f"unknown task {task!r}")


def request_body(model: str, prompt: str, xargs: dict[str, Any] | None = None) -> dict[str, Any]:
    """One user message, ``enable_thinking`` false, ``max_tokens`` 256, generated ids returned; no
    temperature, seed or response_format (the diffusion path's stock sampling)."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": MAX_TOKENS,
        "return_token_ids": True,
    }
    if xargs:
        body["vllm_xargs"] = dict(xargs)
    return body
