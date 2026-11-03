"""``make_cd_model_state_cls(base)``: the model-state subclass that wraps the stock sampler (T-0067 criterion 1).

``custom_sampler`` calls ``super()`` and returns ``(ConstrainedDiffusionSampler(inner), second)`` with
``inner`` and ``second`` exactly the objects super returned; ``None`` stays ``None`` and an exception from
super propagates unchanged. The CD code never constructs the stock sampler. The factory composes with
T-0011's (``make_model_state_cls(make_cd_model_state_cls(DiffusionGemmaModelState), tap=...)``).

``add_request(req_index, new_req_data)`` / ``remove_request(req_id)`` call super first and then keep the
CD registry: the request id is bound to its slot (either order with the sampler's ``add_request``), and
the request's CD state is reset in ``remove_request``.
"""

from __future__ import annotations

from typing import Any

from .sampler import ConstrainedDiffusionSampler

__all__ = ["make_cd_model_state_cls"]


def _resumed(new_req_data: Any) -> bool:
    """A prefill longer than the prompt means the request resumes after preemption (ruling S10)."""
    prefill = getattr(new_req_data, "prefill_token_ids", None)
    prompt = getattr(new_req_data, "prompt_token_ids", None)
    if prefill is None or prompt is None:
        return False
    return len(prefill) > len(prompt)


def make_cd_model_state_cls(base: type) -> type:
    class CdModelState(base):  # type: ignore[misc,valid-type]
        def custom_sampler(self, sampler: Any) -> Any:
            got = super().custom_sampler(sampler)
            if got is None:
                return None
            inner, second = got
            wrapper = ConstrainedDiffusionSampler(inner)
            self._astra_cd_sampler = wrapper
            return wrapper, second

        def add_request(self, req_index: int, new_req_data: Any) -> None:
            super().add_request(req_index, new_req_data)
            slots = self.__dict__.setdefault("_astra_cd_slot_of", {})
            req_id = new_req_data.req_id
            slots[req_id] = req_index
            wrapper = self.__dict__.get("_astra_cd_sampler")
            if wrapper is not None:
                wrapper._bind(req_index, req_id, _resumed(new_req_data))

        def remove_request(self, req_id: str) -> None:
            super().remove_request(req_id)
            slot = self.__dict__.get("_astra_cd_slot_of", {}).pop(req_id, None)
            wrapper = self.__dict__.get("_astra_cd_sampler")
            if wrapper is not None and slot is not None:
                wrapper._release(slot, req_id)

    CdModelState.__name__ = f"Cd{base.__name__}"
    CdModelState.__qualname__ = CdModelState.__name__
    return CdModelState
