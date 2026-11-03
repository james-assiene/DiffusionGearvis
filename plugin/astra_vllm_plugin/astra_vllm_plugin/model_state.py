"""The model-state factory: the one place the tap meets vLLM's request lifecycle (T-0011, R9).

``make_model_state_cls(base, *, tap)`` subclasses *any* model-state base -- the stock
``DiffusionGemmaModelState``, or a wrapper base such as T-0062's ``make_cd_model_state_cls(...)``
-- and wraps whatever ``super().custom_sampler()`` returned; it never constructs a sampler and
never reads the inputs the stock one is built from. Composition lives in
``model.py::AstraDiffusionGemma.get_model_state_cls``, not here.

``tap`` is a factory ``tap(slots) -> CanvasTap | None``, called once, when ``custom_sampler``
wraps, so the pinned pool is allocated then. ``req_id`` exists only on this side
(``add_request(req_index, new_req_data)``), which is why the slot table lives on the model state
(``state.astra_slots``). Nothing the tap does here raises into the engine loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from astra_vllm_plugin.sampler import TapSampler
from astra_vllm_plugin.tap import SlotTable

log = logging.getLogger("astra_vllm_plugin.model_state")


def make_model_state_cls(base: type, *, tap: Callable[[SlotTable], Any]) -> type:
    class AstraDiffusionGemmaModelState(base):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.astra_slots = SlotTable()
            self._astra_tap: Any = None
            super().__init__(*args, **kwargs)

        def custom_sampler(self, sampler: Any) -> Any:
            got = super().custom_sampler(sampler)
            if not (isinstance(got, tuple) and len(got) == 2):
                return got  # None ("no custom sampler"), or a shape we do not know how to wrap
            inner, second = got
            try:
                canvas_tap = tap(self.astra_slots)
                if canvas_tap is None:
                    return got
                canvas_tap.attach(inner)
            except Exception:
                log.exception("canvas tap could not be set up; the sampler runs unwrapped")
                return got
            self._astra_tap = canvas_tap
            return TapSampler(inner, canvas_tap), second

        def add_request(self, req_index: int, new_req_data: Any, *args: Any, **kwargs: Any) -> Any:
            result = super().add_request(req_index, new_req_data, *args, **kwargs)
            try:
                self.astra_slots.bind(req_index, new_req_data.req_id)
                if self._astra_tap is not None:
                    self._astra_tap.on_bind(req_index)
            except Exception as exc:
                self._astra_error(exc, "add_request")
            return result

        def remove_request(self, req_id: str, *args: Any, **kwargs: Any) -> Any:
            try:
                released = self.astra_slots.release(req_id)
                if released is not None and self._astra_tap is not None:
                    slot, old = released
                    self._astra_tap.on_release(old, slot)
            except Exception as exc:
                self._astra_error(exc, "remove_request")
            return super().remove_request(req_id, *args, **kwargs)

        def _astra_error(self, exc: Exception, where: str) -> None:
            if self._astra_tap is not None:
                self._astra_tap.record_error(exc, where)
            else:
                log.warning("canvas tap: %s raised %r", where, exc)

    return AstraDiffusionGemmaModelState
