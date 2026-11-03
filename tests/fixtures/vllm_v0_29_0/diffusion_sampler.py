# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# fmt: off
#
# VENDORED VERBATIM from vllm/vllm-openai:v0.29.0@sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1
# (see MANIFEST.md / manifest.json in this directory: path, line range and sha256 of every segment).
# Segments sit between "# >>> vendored <name>" / "# <<< vendored <name>" markers; the bytes between the
# markers are exactly the image's (the `@torch.compile(...)` decorator line above each function is the only
# thing left out, for eager CPU use). Everything outside the markers is ours. Do not edit inside them.
"""The stock v0.29.0 ``DiffusionSampler`` / ``DiffusionGemmaRequestStates`` / ``SamplingStates`` and the
pytorch top-k/top-p, verbatim, bound to CPU stand-ins for the vLLM helpers they import (``_shims.py``)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from ._shims import (
    NO_LOGPROBS,
    LogprobsTensors,
    SamplerOutput,
    SamplingParams,
    UvaBackedTensor,
    _NP_INT64_MAX,
    _NP_INT64_MIN,
    apply_min_p,
    apply_temperature,
    async_copy_to_gpu,
    async_tensor_h2d,
    compute_topk_scores,
    current_platform,
    logger,
    use_penalty,
)
from .diffusion_gemma_step import _compiled_sample_step, _compute_num_rejected


# >>> vendored apply_top_k_top_p_pytorch
def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    allow_cpu_sync: bool = False,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits

        if allow_cpu_sync:
            # Avoid sorting vocab for top-k only case.
            return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.size(1) - k.to(torch.long)  # shape: B
        # Get all the top_k values.
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        # Apply top-p.
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    # Re-sort the probabilities.
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)
# <<< vendored apply_top_k_top_p_pytorch


def apply_top_k_top_p(logits: torch.Tensor, k: Any, p: Any) -> torch.Tensor:
    """Ours: the image's dispatcher picks Triton on CUDA; on a CPU host without Triton it is the pytorch
    sort path with ``allow_cpu_sync=False`` (the GPU small-batch branch), which is sync-free."""
    return apply_top_k_top_p_pytorch(logits, k, p)


# >>> vendored SamplingStates
class SamplingStates:
    def __init__(self, max_num_reqs: int, vocab_size: int):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size

        self.temperature = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.top_k = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.top_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.min_p = UvaBackedTensor(max_num_reqs, dtype=torch.float32)
        self.seeds = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        # Tracks whether `seed` was set explicitly by the user, so callers
        # can fall back from RNG paths that don't honor per-request seeds.
        self.seeds_set = np.zeros(max_num_reqs, dtype=bool)

        # Initialize top_k and top_p manually because 0 is an invalid value for them.
        self.top_k.np.fill(self.vocab_size)
        self.top_k.copy_to_uva()
        self.top_p.np.fill(1.0)
        self.top_p.copy_to_uva()

        self.num_logprobs = np.empty(self.max_num_reqs, dtype=np.int32)
        # -1 means no logprobs are requested.
        self.num_logprobs.fill(NO_LOGPROBS)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        self.temperature.np[req_idx] = sampling_params.temperature
        self.top_p.np[req_idx] = sampling_params.top_p
        top_k = sampling_params.top_k
        if top_k <= 0 or top_k > self.vocab_size:
            top_k = self.vocab_size
        self.top_k.np[req_idx] = top_k
        self.min_p.np[req_idx] = sampling_params.min_p

        seed = sampling_params.seed
        self.seeds_set[req_idx] = seed is not None
        if seed is None:
            seed = np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX)
        self.seeds.np[req_idx] = seed

        num_logprobs = sampling_params.logprobs
        if num_logprobs is None:
            num_logprobs = NO_LOGPROBS
        elif num_logprobs == -1:
            num_logprobs = self.vocab_size
        self.num_logprobs[req_idx] = num_logprobs

    def apply_staged_writes(self) -> None:
        self.temperature.copy_to_uva()
        self.top_p.copy_to_uva()
        self.top_k.copy_to_uva()
        self.min_p.copy_to_uva()
        self.seeds.copy_to_uva()

    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        temp_np = self.temperature.np[idx_mapping_np]
        if np.all((temp_np == 0.0) | (temp_np == 1.0)):
            # No request requires temperature. Skip the kernel launch.
            return

        apply_temperature(logits, expanded_idx_mapping, self.temperature.gpu)

    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if np.all(self.min_p.np[idx_mapping_np] == 0.0):
            # No request uses min_p. Skip the kernel launch.
            return
        apply_min_p(logits, expanded_idx_mapping, self.min_p.gpu)

    def get_top_k_top_p(
        self, expanded_idx_mapping: torch.Tensor, idx_mapping_np: np.ndarray
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        do_top_k = np.any(self.top_k.np[idx_mapping_np] != self.vocab_size)
        do_top_p = np.any(self.top_p.np[idx_mapping_np] != 1.0)
        top_k = self.top_k.gpu[expanded_idx_mapping] if do_top_k else None
        top_p = self.top_p.gpu[expanded_idx_mapping] if do_top_p else None
        return top_k, top_p

    def apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> torch.Tensor:
        top_k, top_p = self.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
        if top_k is None and top_p is None:
            return logits
        return apply_top_k_top_p(logits, top_k, top_p)

    def any_greedy(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self.temperature.np[idx_mapping_np] == 0.0))

    def any_explicit_seed(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self.seeds_set[idx_mapping_np]))

    def max_num_logprobs(self, idx_mapping_np: np.ndarray) -> int:
        return int(np.max(self.num_logprobs[idx_mapping_np]))
# <<< vendored SamplingStates


# >>> vendored DiffusionGemmaRequestStates
class DiffusionGemmaRequestStates:
    """Pre-allocated GPU tensors for DiffusionGemma per-request state.

    Follows the indexed-slot pattern used by ``RequestState``.
    """

    def __init__(
        self,
        max_num_reqs: int,
        canvas_length: int,
        vocab_size: int,
        max_denoising_steps: int,
        device: torch.device,
        hidden_size: int,
        stability_threshold: int,
    ):
        self.max_num_reqs = max_num_reqs
        self.canvas_length = canvas_length
        self.vocab_size = vocab_size
        self.max_denoising_steps = max_denoising_steps
        self.stability_threshold = stability_threshold
        self.device = device

        self.is_encoder_phase = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )
        # Canvas tokens [max_num_reqs, canvas_length]
        self.canvas = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int64, device=device
        )
        # Step counter (counts up from 0 to max_denoising_steps)
        self.step = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        # Accepted canvas history for stability check
        self.accepted_canvas_history = torch.zeros(
            max_num_reqs,
            stability_threshold,
            canvas_length,
            dtype=torch.int64,
            device=device,
        )
        self.accepted_canvas_history_len = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        # Latest argmax(processed_logits) per slot — what we COMMIT.
        # NOT `current_canvas` (which is the post-renoise stochastic input for
        # the next denoise step). We keep this separate from `canvas` because
        # canvas gets renoised in-place during denoise, while argmax_canvas is
        # the deterministic best-guess we ultimately emit.
        self.argmax_canvas = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int64, device=device
        )

        # Per-slot prompt length (set by add_request).
        self.prompt_len = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )

        # Per-slot confidence flag, set by the sampler each step.
        self.confident = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)

        # Per-slot self-conditioning soft embedding (probs @ embed_weight) from
        # the previous denoise step. Storing the [.., hidden] soft embed instead
        # of the full [.., vocab] distribution shrinks this buffer by
        # vocab/hidden (~170x) and moves the matmul to denoise time; the result
        # is identical (SC consumes probs @ embed_weight anyway).
        self.self_conditioning_embeds = torch.zeros(
            max_num_reqs, canvas_length, hidden_size, dtype=torch.float32, device=device
        )

    def init_canvas(self, slot_indices: torch.Tensor) -> None:
        """Initialize canvas with random tokens for the given slots.

        `slot_indices` must already be on device to avoid a cpu->gpu sync.
        """
        n = slot_indices.shape[0]
        self.canvas[slot_indices] = torch.randint(
            0,
            self.vocab_size,
            (n, self.canvas_length),
            dtype=torch.int64,
            device=self.device,
        )

    def add_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx].fill_(True)
        self.init_canvas(async_tensor_h2d([slot_idx], device=self.device))
        self.step[slot_idx].fill_(0)
        self.accepted_canvas_history_len[slot_idx].fill_(0)
        self.self_conditioning_embeds[slot_idx] = 0

    def remove_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx].fill_(False)
        self.accepted_canvas_history_len[slot_idx].fill_(0)
        self.self_conditioning_embeds[slot_idx] = 0
# <<< vendored DiffusionGemmaRequestStates


# >>> vendored _NO_PENALTIES_STATE
_NO_PENALTIES_STATE = SimpleNamespace(output_bin_counts=None)
# <<< vendored _NO_PENALTIES_STATE


# >>> vendored DiffusionSampler
class DiffusionSampler:
    """Batched accept/renoise sampler for DiffusionGemma.

    Follows the same structure as ``vllm.v1.worker.gpu.sample.sampler.Sampler``:
    decomposed into named methods, all GPU state in pre-allocated buffers,
    no GPU→CPU syncs on the hot path.
    """

    def __init__(
        self,
        sampler: Any,
        diffusion_config: Any,
        vocab_size: int,
        diffusion_states: DiffusionGemmaRequestStates,
        *,
        confidence_threshold: float,
        t_min: float,
        t_max: float,
        entropy_bound: float,
        embed_weight: torch.Tensor,
        normalizer: torch.Tensor,
        sc_vocab_start: int = 0,
        sc_vocab_end: int | None = None,
        tp_size: int = 1,
        tp_group_name: str = "",
    ):
        self.sampling_states = sampler.sampling_states
        self.req_states = sampler.req_states
        self.logits_mode = sampler.logprobs_mode in ("raw_logits", "processed_logits")
        # Self-conditioning soft embed = probs @ embed_weight * normalizer,
        # computed in the sampler (see _compiled_sample_step). ``embed_weight``
        # is the vocab-parallel shard; [sc_vocab_start, sc_vocab_end) is this
        # rank's slice of the full vocab and tp_* drive the cross-rank
        # all-reduce.
        self.embed_weight = embed_weight
        self.normalizer = normalizer
        self.sc_vocab_start = sc_vocab_start
        self.sc_vocab_end = sc_vocab_end if sc_vocab_end is not None else vocab_size
        self.tp_size = tp_size
        self.tp_group_name = tp_group_name
        self.canvas_length = (
            diffusion_config.canvas_length if diffusion_config is not None else 32
        )
        self.t_min = t_min
        self.t_max = t_max
        self.confidence_threshold = confidence_threshold
        self.vocab_size = vocab_size
        self.diffusion_states = diffusion_states
        self.entropy_bound = entropy_bound

        max_num_reqs = diffusion_states.max_num_reqs
        device = diffusion_states.device
        self._sampled = torch.zeros(
            max_num_reqs,
            self.canvas_length,
            dtype=torch.int32,
            device=device,
        )
        self._num_sampled = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self._decode_slots = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._decode_idx = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._query_lens = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self._num_logits = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

        # Per-slot stash for logprobs computed on the converging denoise step.
        # Populated after the post-sample kernel detects convergence; consumed
        # on the subsequent commit step when num_sampled=CANVAS_LEN.
        self._pending_logprobs: dict[int, LogprobsTensors] = {}

    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> None:
        if use_penalty(sampling_params):
            logger.warning_once(
                "DiffusionGemma does not support repetition/frequency/presence "
                "penalties; ignoring them for this request."
            )
        # Purge any stale logprobs stashed under this slot by a prior request
        # that was aborted between its converging denoise and commit steps.
        self._pending_logprobs.pop(req_idx, None)
        self.sampling_states.add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()

    @property
    def penalties_state(self):
        # Diffusion applies no penalties. The runner reads
        # penalties_state.output_bin_counts, so expose a stub holding None;
        # post_update treats None bin counts as "no penalty bookkeeping".
        return _NO_PENALTIES_STATE

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _finish_prefills(
        self, input_batch: Any, prefill_indices_np: np.ndarray
    ) -> None:
        """Transition requests whose prompt completes this step to denoising.

        Initializes their canvas, seeds draft tokens, and flips
        is_encoder_phase to False. Mid-chunk requests (prompt longer than the
        token budget) are left untouched so is_encoder_phase stays True and
        prepare_attn keeps causal attention for their remaining chunks.
        """
        states = self.diffusion_states
        done_prefill_np = (
            input_batch.num_computed_prefill_tokens_np[prefill_indices_np]
            + input_batch.num_scheduled_tokens[prefill_indices_np]
            >= input_batch.prefill_len_np[prefill_indices_np]
        )
        ps = input_batch.idx_mapping_np[prefill_indices_np[done_prefill_np]]
        if len(ps) == 0:
            return
        # Move the slot indices across once, up front: indexing a device
        # tensor with a numpy array copies them over synchronously each time.
        ps_gpu = async_copy_to_gpu(
            ps.astype(np.int64), device=states.is_encoder_phase.device
        )
        states.init_canvas(ps_gpu)
        self.req_states.draft_tokens[ps_gpu, : self.canvas_length] = states.canvas[
            ps_gpu
        ]
        states.is_encoder_phase.index_fill_(0, ps_gpu, False)

    def _handle_prefill(
        self,
        input_batch: Any,
        device: torch.device,
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        self._finish_prefills(input_batch, np.arange(num_reqs))
        sampled = self._sampled[:num_reqs, :1]
        sampled.zero_()
        num_sampled = self._num_sampled[:num_reqs]
        num_sampled.zero_()
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_sampled,
        )

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    def _build_output(
        self,
        input_batch: Any,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        per_req_nlogits_np: np.ndarray,
        device: torch.device,
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> SamplerOutput:
        """Compute num_rejected and build SamplerOutput."""
        num_reqs = input_batch.num_reqs

        self._query_lens.np[:num_reqs] = np.diff(
            input_batch.query_start_loc_np[: num_reqs + 1]
        )
        self._num_logits.np[:num_reqs] = per_req_nlogits_np
        self._query_lens.copy_to_uva()
        self._num_logits.copy_to_uva()

        num_rejected = _compute_num_rejected(
            self._num_logits.gpu[:num_reqs],
            num_sampled,
            input_batch.query_start_loc[: num_reqs + 1],
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs_tensors,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: Any,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        device = logits.device

        if input_batch.num_draft_tokens == 0:
            return self._handle_prefill(input_batch, device)

        # --- CPU/NumPy setup (outside compile): split decode vs prefill, init
        # canvas for any new prefills, and stage decode slot indices to GPU. ---
        states = self.diffusion_states
        CL = self.canvas_length
        slots_np = input_batch.idx_mapping_np[:num_reqs]
        per_req_nlogits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])

        decode_indices_np = np.where(per_req_nlogits_np > 0)[0]
        prefill_indices_np = np.where(per_req_nlogits_np == 0)[0]
        decode_slots_np = slots_np[decode_indices_np]

        if len(prefill_indices_np) > 0:
            self._finish_prefills(input_batch, prefill_indices_np)

        num_decode = len(decode_indices_np)
        self._decode_slots.np[:num_decode] = decode_slots_np
        self._decode_idx.np[:num_decode] = decode_indices_np
        self._decode_slots.copy_to_uva()
        self._decode_idx.copy_to_uva()
        decode_slots = self._decode_slots.gpu[:num_decode]
        decode_idx = self._decode_idx.gpu[:num_decode]

        # Real canvas length per decode request. Equals CL except when a canvas
        # was truncated near max_model_len, in which case the scheduler gave us
        # fewer than CL logits for that request.
        valid_canvas_len_np = per_req_nlogits_np[per_req_nlogits_np > 0]
        valid_canvas_len = async_copy_to_gpu(
            valid_canvas_len_np.astype(np.int64), device=device
        )

        # Per-request top_k/top_p, mirroring the AR sampler. Masked tokens
        # become -inf and survive the temperature scaling in the compiled
        # step, so Gumbel sampling, probs, and entropy all see the filtered
        # distribution. The committed argmax (always the top-1 token) is
        # unaffected; only the canvas exploration is constrained. Applied
        # before canvas padding so phantom positions stay uniform.
        if num_decode > 0:
            top_k, top_p = self.sampling_states.get_top_k_top_p(
                decode_slots.repeat_interleave(
                    valid_canvas_len, output_size=int(valid_canvas_len_np.sum())
                ),
                decode_slots_np,
            )
            if top_k is not None or top_p is not None:
                logits = apply_top_k_top_p(logits.float(), top_k, top_p)

        # Pad any truncated canvas back to CL so the uniform-CL sampler math
        # holds. Phantom (padded) positions are zeroed → uniform logits → high
        # entropy (no premature convergence) and argmax 0 (stable); they are
        # never committed (num_sampled == real length). masked_fill (not
        # multiply) so -inf entries from top_k/top_p filtering above don't
        # turn phantom rows into NaN.
        if num_decode > 0 and valid_canvas_len_np.min() < CL:
            ar = torch.arange(CL, device=device)
            starts = valid_canvas_len.cumsum(0) - valid_canvas_len  # row offset per req
            valid = ar.unsqueeze(0) < valid_canvas_len.unsqueeze(1)  # [num_decode, CL]
            src = (starts.unsqueeze(1) + ar.unsqueeze(0)).clamp_max(logits.shape[0] - 1)
            logits = logits[src.reshape(-1)].masked_fill_(~valid.reshape(-1, 1), 0)

        # Clear once: the tiled loop below only scatters its own decode slots,
        # so it must not re-clear earlier tiles' writes.
        sampled = self._sampled[:num_reqs]
        num_sampled = self._num_sampled[:num_reqs]
        sampled.zero_()
        num_sampled.zero_()

        all_slots = input_batch.idx_mapping[:num_reqs]

        # Snapshot which slots are committing BEFORE the compiled step runs,
        # since it mutates is_encoder_phase (commit→False, converge→True).
        is_committing = states.is_encoder_phase[decode_slots].clone()

        slots_np = input_batch.idx_mapping_np[:num_reqs]
        is_decode_np = per_req_nlogits_np > 0
        max_num_logprobs = self.sampling_states.max_num_logprobs(slots_np)

        # Sample over the [num_decode * CL, vocab] logits. The fp32 pipeline in
        # _compiled_sample_step keeps several live [group * CL, vocab] copies, so
        # size each tile to a fraction of free memory to bound the transient at
        # high concurrency. Tiling is bit-identical to a single pass.
        group = max(num_decode, 1)
        if num_decode > 0:
            free, _ = current_platform.mem_get_info()
            # ~10 transient fp32 copies of [group * CL, vocab] inside the step
            # (eager peaks at ~8; pad for allocator overhead and small tensors).
            bytes_per_req = CL * self.vocab_size * 4 * 10
            budget = int(free * 0.5) // max(bytes_per_req, 1)
            group = max(1, min(num_decode, budget))

        for start_req in range(0, num_decode, group):
            end_req = min(start_req + group, num_decode)
            tile = slice(start_req, end_req)
            tile_slots = decode_slots[tile]

            scaled = _compiled_sample_step(
                logits[start_req * CL : end_req * CL],
                tile_slots,
                decode_idx[tile],
                all_slots,
                valid_canvas_len[tile],
                # State
                states.canvas,
                states.argmax_canvas,
                states.step,
                states.is_encoder_phase,
                states.confident,
                states.self_conditioning_embeds,
                self.embed_weight,
                self.normalizer,
                states.accepted_canvas_history,
                states.accepted_canvas_history_len,
                # Output
                sampled,
                num_sampled,
                self.req_states.draft_tokens,
                # Config
                max_denoising_steps=float(states.max_denoising_steps),
                t_min=self.t_min,
                t_max=self.t_max,
                confidence_threshold=self.confidence_threshold,
                vocab_size=self.vocab_size,
                CL=CL,
                ST=states.stability_threshold,
                entropy_bound=self.entropy_bound,
                sc_vocab_start=self.sc_vocab_start,
                sc_vocab_end=self.sc_vocab_end,
                tp_size=self.tp_size,
                tp_group_name=self.tp_group_name,
            )

            # Logprobs for denoise steps that just converged (is_encoder_phase
            # flipped False→True), stashed per tile so `scaled` is freed each tile.
            if max_num_logprobs >= 0:
                converged_mask = states.is_encoder_phase[tile_slots]
                just_converged = converged_mask & ~is_committing[tile]
                if just_converged.any():
                    flat_logits = scaled.reshape(-1, scaled.shape[-1])
                    argmax_tokens = scaled.argmax(dim=-1)
                    for local_idx in just_converged.nonzero(as_tuple=True)[0]:
                        li = local_idx.item()
                        slot = tile_slots[local_idx]
                        # Stash only the real canvas positions (== CL unless this
                        # canvas was truncated near max_model_len); padded tail
                        # positions are never emitted.
                        k_i = int(valid_canvas_len_np[start_req + li])
                        pos = li * CL
                        self._pending_logprobs[slot.item()] = compute_topk_scores(
                            flat_logits[pos : pos + k_i],
                            max_num_logprobs,
                            argmax_tokens[local_idx][:k_i],
                            logits_mode=self.logits_mode,
                        )

        # Commit steps: is_committing was True at entry. Reassemble previously
        # stashed logprobs and attach to SamplerOutput.
        logprobs_tensors = None
        if max_num_logprobs >= 0 and is_committing.any() and self._pending_logprobs:
            parts_ids, parts_lp, parts_ranks = [], [], []
            cu_gen: list[int] = []
            flat_offset = 0
            for i in range(num_reqs):
                cu_gen.append(flat_offset)
                slot = int(slots_np[i])
                if is_decode_np[i] and slot in self._pending_logprobs:
                    lp = self._pending_logprobs.pop(slot)
                    parts_ids.append(lp.logprob_token_ids)
                    parts_lp.append(lp.logprobs)
                    parts_ranks.append(lp.selected_token_ranks)
                    flat_offset += lp.logprobs.shape[0]
            if parts_ids:
                logprobs_tensors = LogprobsTensors(
                    logprob_token_ids=torch.cat(parts_ids),
                    logprobs=torch.cat(parts_lp),
                    selected_token_ranks=torch.cat(parts_ranks),
                    cu_num_generated_tokens=cu_gen,
                )

        return self._build_output(
            input_batch,
            sampled,
            num_sampled,
            per_req_nlogits_np,
            device,
            logprobs_tensors=logprobs_tensors,
        )
# <<< vendored DiffusionSampler
