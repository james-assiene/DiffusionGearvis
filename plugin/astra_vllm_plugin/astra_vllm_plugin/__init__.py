"""vLLM general plugin entry point. Import of vllm is deferred so the pure parts stay testable."""

from __future__ import annotations

import logging
import os

log = logging.getLogger("astra_vllm_plugin")

# Registry key, and the value services/vllm.sh puts in --hf-overrides architectures. Renamed from
# AstraDiffusionGemmaForCausalLM: v0.29.0 has no DiffusionGemmaForCausalLM at all (the class is
# DiffusionGemmaForConditionalGeneration, registry key DiffusionGemmaForBlockDiffusion), so the
# old name described a class that never existed. Change here and in services/vllm.sh together.
ARCHITECTURE = "AstraDiffusionGemma"


def register() -> None:
    """Called by vLLM at start-up for every process (engine core and API server).

    Registers the wrapped architecture; the tap itself is created lazily per model state in
    the process that runs the sampler.
    """
    if os.environ.get("ASTRA_PLUGIN_DISABLED") == "1":
        log.info("astra plugin disabled by env")
        return
    try:
        from vllm import ModelRegistry  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - only inside the container
        log.warning("vllm not importable (%s); astra plugin inactive", e)
        return
    ModelRegistry.register_model(ARCHITECTURE, "astra_vllm_plugin.model:AstraDiffusionGemma")
    log.info("astra plugin registered %s", ARCHITECTURE)
