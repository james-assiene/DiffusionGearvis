"""``TapSampler``: a transparent wrapper around whatever sampler ``custom_sampler`` returned (T-0011).

The runner feature-detects its sampler by attribute (``model_runner.py:1462-1463``), so the wrapper
answers ``hasattr`` exactly as the inner object does: every name is delegated, its own state is
private (``_inner``, ``_tap``), and it defines nothing the inner lacks -- in particular no
``remove_request`` (``DiffusionSampler`` has none; requests leave through the model state).

``__call__`` delegates to the inner sampler **unguarded**: an exception there is the engine's, and
reaches it as the same object. The tap's work before and after delegating is guarded by the tap
itself (``CanvasTap.before_call``/``after_call``) and never raises into the engine loop.
"""

from __future__ import annotations

from typing import Any


class TapSampler:
    def __init__(self, inner: Any, tap: Any) -> None:
        self._inner = inner
        self._tap = tap

    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> Any:
        """Per-request options arrive here (``vllm_xargs`` -> ``extra_args``); the rest is the inner's."""
        tap = self._tap
        if tap is not None:
            try:
                tap.slots.options(req_idx, getattr(sampling_params, "extra_args", None))
            except Exception as exc:
                tap.record_error(exc, "add_request")
        return self._inner.add_request(req_idx, prompt_len, sampling_params)

    def __call__(self, logits: Any, input_batch: Any, draft_logits: Any = None) -> Any:
        tap = self._tap
        plan = tap.before_call(input_batch) if tap is not None else None
        out = self._inner(logits, input_batch, draft_logits)
        if plan is not None:
            tap.after_call(plan, out)
        return out

    def __getattr__(self, name: str) -> Any:
        try:
            inner = self.__dict__["_inner"]
        except KeyError:
            raise AttributeError(name) from None
        return getattr(inner, name)
