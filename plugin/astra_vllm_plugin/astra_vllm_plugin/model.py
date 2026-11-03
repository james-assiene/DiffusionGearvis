"""Wrapper around vLLM's DiffusionGemma that repairs the broken image path.

Why this exists
---------------
``diffusion_gemma.py`` does not subclass ``Gemma4ForConditionalGeneration``; it BORROWS seven
methods by class-level assignment (``diffusion_gemma.py:289-301``), with the comment "the vision
tower, pooler, embed_vision, and their processing logic are architecturally identical to Gemma4.
Delegate to avoid maintaining a duplicate copy."

That borrow is only safe while the borrowed methods touch no state that Gemma4's ``__init__``
sets and DiffusionGemma's does not. On 2026-08-13, ``[LoRA][Gemma4] Support vision tower LoRA``
(#42662, commit 95c914442) added ``self._enable_mm_lora`` to ``Gemma4.__init__`` and read it
inside ``_process_image_input`` -- silently breaking every DiffusionGemma image request:

    AttributeError: 'DiffusionGemmaForConditionalGeneration' object has no attribute
    '_enable_mm_lora'   (gemma4_mm.py:1317)

and because it raises inside the model runner it kills the EngineCore (``EngineDeadError``),
not just the request. Verified on this box 2026-09-13 against v0.29.0; upstream ``main`` is
still identical. DiffusionGemma's image path worked from 2026-06-12 (when #45163 added it) until
2026-08-12.

Filed upstream: https://github.com/vllm-project/vllm/issues/56712
When that issue closes as fixed, delete this shim and the tests that assert it is needed
(tests/contract/test_vllm_shim_gpu.py). See docs/tasks/T-0015.md.

An AST audit of all seven borrowed methods against ``gemma4_mm.py`` found three such attributes,
not one. The other two are audio-only and guarded by an audio input being present
(``gemma4_mm.py:1244`` and ``:1623``), so they are latent for us -- we send images only -- but
they are the same class of bug and would surface as the same confusing AttributeError.

Scope: this file is the compatibility shim plus one seam, ``get_model_state_cls`` (T-0011, R9;
T-0069), which composes the served model state as Tap(CD(stock)): the per-step canvas tap around
the constrained-decoding wrapper around the stock model state. The tap itself lives in
``model_state.py``, ``sampler.py`` and ``tap.py``; constrained decoding lives in ``cd/``. A request
with no ``astra_cd_*`` key takes the stock path inside the CD wrapper, so the tap sees exactly what
it saw over the stock sampler.
"""

from __future__ import annotations

import functools
from typing import Any

# No try/except around this import. The scaffold wrapped it in a bare `except` that rebound the
# base class to `object`, which turned a wrong class path into a silently useless registration
# instead of a loud failure -- exactly the mistake that hid the name being wrong in the first
# place (v0.29.0 has no `DiffusionGemmaForCausalLM`; the class is `...ForConditionalGeneration`,
# registry key `DiffusionGemmaForBlockDiffusion`). If this import fails, the plugin must fail.
from vllm.model_executor.models.diffusion_gemma import (  # type: ignore[import-not-found]
    DiffusionGemmaForConditionalGeneration,
)


class AstraDiffusionGemma(DiffusionGemmaForConditionalGeneration):  # type: ignore[misc,valid-type]
    """Stock DiffusionGemma plus the attributes its borrowed Gemma4 methods expect.

    Everything here is a compatibility shim for an upstream defect. Nothing in this class is
    Astra behaviour, and each attribute should be deleted the moment upstream provides it.
    """

    # Read by the borrowed `_process_image_input` (gemma4_mm.py:1317, :1364). False is what
    # Gemma4's own __init__ computes for us: `bool(lora_config is not None and
    # lora_config.enable_tower_connector_lora)`, and we run no LoRA. Declared at class level
    # rather than in __init__ so we do not have to mirror an upstream constructor signature.
    _enable_mm_lora = False

    # Latent, audio-only siblings of the same defect. Reached only if an audio input is actually
    # present, which our pipeline never sends (speech goes to Kyutai STT, never to this model).
    # Stubs rather than borrowed implementations: DiffusionGemma has no audio tower, so borrowing
    # Gemma4's would fail later and more confusingly. Fail with a sentence that says what happened.
    def _parse_and_validate_audio_input(self, **kwargs: object) -> Any:
        raise NotImplementedError(
            "DiffusionGemma has no audio tower; Gemma4's borrowed multimodal parser reached the "
            "audio branch. Astra sends audio to Kyutai STT, never to the LLM."
        )

    def _process_audio_input(self, audio_input: Any) -> Any:
        raise NotImplementedError(
            "DiffusionGemma has no audio tower; Gemma4's borrowed embed_multimodal reached the "
            "audio branch. Astra sends audio to Kyutai STT, never to the LLM."
        )

    # T-0011 (R9) and T-0069: the single composition expression, Tap(CD(stock)). ``custom_sampler``
    # CONSTRUCTS the stock DiffusionSampler (diffusion_gemma.py:828), so each layer wraps what its
    # base returned instead of replacing it: CD wraps the stock sampler, the tap wraps CD and so
    # reads what CD wrote (the per-step canvas, the MAP in ``argmax_canvas``, the MAP history).
    # Imports are lazy (the plugin's modules stay torch/vLLM-free at import) and the class is built
    # once. No try/except: if the ``cd`` import fails, the boot must fail. A Tap(stock) fallback
    # would ignore ``astra_cd_*`` keys and serve those requests unconstrained, silently.
    @staticmethod
    @functools.cache
    def get_model_state_cls() -> type:
        from astra_vllm_plugin.cd.model_state import make_cd_model_state_cls
        from astra_vllm_plugin.model_state import make_model_state_cls
        from astra_vllm_plugin.tap import tap_from_env

        return make_model_state_cls(
            make_cd_model_state_cls(DiffusionGemmaForConditionalGeneration.get_model_state_cls()),
            tap=tap_from_env,
        )
