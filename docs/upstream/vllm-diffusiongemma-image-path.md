# Upstream bug report — FILED as https://github.com/vllm-project/vllm/issues/56712

Filed 2026-09-13 by the repository owner via `gh`, with the body below verbatim. The URL is
recorded in `docs/tasks/T-0015.md` criterion 6 and in
`plugin/astra_vllm_plugin/astra_vllm_plugin/model.py`. When the issue closes as fixed, delete the
shim and `tests/contract/test_vllm_shim_gpu.py::test_upstream_defect_is_still_present` together —
that test going red is the signal.

---

**Title:**

```
[Bug]: DiffusionGemma image input crashes the EngineCore — borrowed Gemma4 methods read `_enable_mm_lora`, which DiffusionGemma never sets
```

**Labels:** `bug`

---

**Body:**

### Your current environment

- vLLM: `v0.29.0` (container `vllm/vllm-openai:v0.29.0`, digest `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`). Also present on `main` as of 2026-09-13.
- Model: `nvidia/diffusiongemma-26B-A4B-it-NVFP4` (revision `2ea837236295d617ac27f8c17d61228081932c40`)
- GPU: RTX 5090 (sm_120), driver 595.84, CUDA 13.0 container
- Flags: `--attention-backend TRITON_ATTN --diffusion-config '{"canvas_length": 256, "max_denoising_steps": 48}' --max-num-seqs 3 --kv-cache-dtype bfloat16 --kv-cache-memory-bytes 4294967296 --enable-prefix-caching --limit-mm-per-prompt '{"image": 4, "video": 1}'`
- No LoRA is configured.

### 🐛 Describe the bug

**Any image request to DiffusionGemma raises `AttributeError` inside the model runner, which kills the EngineCore rather than just failing the request.** Text-only requests to the same served model work correctly.

```
AttributeError: 'DiffusionGemmaForConditionalGeneration' object has no attribute '_enable_mm_lora'
  File "vllm/model_executor/models/gemma4_mm.py", line 1317, in _process_image_input
    if self._enable_mm_lora:
  File "vllm/model_executor/models/gemma4_mm.py", line 1616, in embed_multimodal
    self._process_image_input(multimodal_input)
  File "vllm/v1/worker/gpu/mm/encoder_runner.py", line 168, in execute_mm_encoder
    else self.model.embed_multimodal(**mm_kwargs_batch)
...
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue.
```

The server then shuts down; subsequent requests get 500s.

### Root cause

`DiffusionGemmaForConditionalGeneration` does **not** subclass `Gemma4ForConditionalGeneration`. It borrows seven methods by class-level assignment (`vllm/model_executor/models/diffusion_gemma.py:289-301`):

```python
    # The vision tower, pooler, embed_vision, and their processing logic
    # are architecturally identical to Gemma4.  Delegate to avoid
    # maintaining a duplicate copy.
    _parse_and_validate_image_input = Gemma4ForConditionalGeneration._parse_and_validate_image_input
    ...
    _process_image_input = Gemma4ForConditionalGeneration._process_image_input
    _process_video_input = Gemma4ForConditionalGeneration._process_video_input
    embed_multimodal = Gemma4ForConditionalGeneration.embed_multimodal
```

That delegation is only safe while the borrowed methods touch no state that `Gemma4ForConditionalGeneration.__init__` sets and `DiffusionGemmaForConditionalGeneration.__init__` does not.

`_enable_mm_lora` is set **only** in Gemma4's `__init__` (`gemma4_mm.py:1061`):

```python
# fmt: off
        lora_config = vllm_config.lora_config
        self._enable_mm_lora = bool(
            lora_config is not None and lora_config.enable_tower_connector_lora
        )
# fmt: on
```

and read in the borrowed `_process_image_input` at `gemma4_mm.py:1317` and `:1364`. Since Gemma4's `__init__` never runs for DiffusionGemma, the attribute does not exist.

### Regression window

- `2026-06-12` — `[Model] Add DiffusionGemma Support` (#45163) added `diffusion_gemma.py` with the borrow. At that time `gemma4_mm.py` contained **no** reference to `_enable_mm_lora`, so the borrow was safe.
- `2026-08-13` — `[LoRA][Gemma4] Support vision tower LoRA` (#42662, commit `95c914442`) introduced `self._enable_mm_lora` in Gemma4's `__init__` and read it inside `_process_image_input`. This is where DiffusionGemma's image path broke.

Verified by fetching `gemma4_mm.py` at commits `70009fb93` (2026-07-25, 0 references) and `95c914442` (3 references, 1 in `__init__`, 2 in `_process_image_input`).

So the image path worked from 2026-06-12 to 2026-08-12. `v0.29.0` is the first release containing DiffusionGemma and it shipped with the path already broken, so there is no release to downgrade to.

### Two latent siblings of the same defect

An AST audit of all seven borrowed methods against the current `gemma4_mm.py` found two further attributes read by borrowed code but never defined on `DiffusionGemmaForConditionalGeneration`:

- `_parse_and_validate_audio_input`, read by the borrowed `_parse_and_validate_multimodal_inputs` (`gemma4_mm.py:1248`)
- `_process_audio_input`, read by the borrowed `embed_multimodal` (`gemma4_mm.py:1626`)

Both are currently unreachable for this model because they sit behind audio-modality guards and `DiffusionGemmaProcessingInfo.get_supported_mm_limits()` declares only `{"image", "video"}` — but they are the same class of latent breakage, and would surface as the same confusing `AttributeError` if the model ever declares audio support.

### Suggested fix

Any of:

1. Set the attribute in `DiffusionGemmaForConditionalGeneration.__init__` alongside the other Gemma4-parity state (smallest change, but leaves the class of bug open — the next attribute Gemma4 adds breaks it again).
2. Read it defensively at the two call sites: `getattr(self, "_enable_mm_lora", False)`.
3. **Structural, and the one I'd suggest:** move the shared vision logic into a mixin that both classes inherit, so the "architecturally identical, delegate to avoid a duplicate copy" intent is expressed by inheritance. The borrow-by-assignment pattern makes every future addition to Gemma4's `__init__` a silent, action-at-a-distance break in DiffusionGemma, discoverable only at inference time and only as an engine crash.

A regression test that sends one image request to DiffusionGemma would have caught this and would catch the audio siblings too.

### Severity note

The failure kills the EngineCore rather than returning an error for the offending request, so a single image request takes the whole server down. That turns a model-specific input bug into an availability issue for any deployment serving DiffusionGemma with multimodal input enabled.

---

*Found and reported by Claude (Anthropic's Claude Code) while bringing up DiffusionGemma on vLLM; the diagnosis, the git-history bisection and this report were written by Claude, and reviewed and filed by the repository owner.*
