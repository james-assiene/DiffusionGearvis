"""Draft-initialised constrained repair (T-0081): seed a constrained request's **first** canvas from a draft.

The wire side (``astra_cd_draft*`` keys, the context-free checks) is ``cd/xargs.py``; this module holds the rest:

- ``refusal``: the admission checks that need the engine's numbers (more than CL ids -> ``draft_bad_length``,
  ``draft_start_step >= max_denoising_steps`` -> ``draft_start_step_out_of_range``);
- ``build``: at ``add_request`` (host work, allowed to copy to the device) the draft padded to CL, the keep mask
  ``i < len(draft)`` and the start step become device tensors, so the seeding call writes device to device only
  (T-0073 D6: a Python scalar written into a CUDA row is a host sync);
- ``completing_prefill_slots``: the slots whose prefill completes in this call, by stock's own host-numpy rule
  (``_finish_prefills``; pinned against the vendored source by ``tests/cd/unit/test_cd_seed_coupling.py``);
- ``apply``: after the inner (stock) call, which has already run ``init_canvas`` on the global RNG and flipped the
  slot to the denoise phase, overwrite ``canvas[slot]``, ``draft_tokens[slot, :CL]`` and ``step[slot]``.

Seed semantics (ruling SD4): ``canvas[i] = draft[i]`` for ``i < n`` except a noised set of exactly
``round(noise * n)`` positions, drawn without replacement; every other position (the noised set and the tail
``i >= n``) holds a uniform id. Every random number comes from the CD-owned generator, in this order per seeded
slot: ``randint(0, V, (CL,))`` (the uniform ids), then, only when the noised set is non-empty, ``rand(n)`` (the
noised set is the ``k`` positions of smallest key). The global RNG is never touched, so stock's consumption is
unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

from .automaton import R_DRAFT_LENGTH, R_DRAFT_START
from .xargs import CdSpec

__all__ = ["Seed", "apply", "build", "completing_prefill_slots", "refusal"]


@dataclasses.dataclass(eq=False)
class Seed:
    """One request's seed, on the device (built at ``add_request``)."""

    n: int  # draft length (host)
    k: int  # noised positions, round(noise * n) (host)
    ids: torch.Tensor  # int64 [CL]: the draft, zero-padded
    keep: torch.Tensor  # bool [CL]: i < n
    step: torch.Tensor  # int64 0-d: draft_start_step


def refusal(spec: CdSpec, canvas_length: int, max_steps: int | None) -> tuple[str, str] | None:
    """``(reason, detail)`` when the seed cannot be served by this engine, else ``None``."""
    if spec.draft is None:
        return None
    n = int(spec.draft.shape[0])
    if n > canvas_length:
        return R_DRAFT_LENGTH, f"draft holds {n} ids > canvas length {canvas_length}"
    if max_steps is not None and spec.draft_start_step >= max_steps:
        return (
            R_DRAFT_START,
            f"draft_start_step {spec.draft_start_step} >= max_denoising_steps {max_steps}",
        )
    return None


def build(spec: CdSpec, canvas_length: int, device: torch.device) -> Seed:
    """The device side of a seed whose ``refusal`` was ``None`` (host work at ``add_request``)."""
    assert spec.draft is not None
    n = int(spec.draft.shape[0])
    ids = np.zeros(canvas_length, dtype=np.int64)
    ids[:n] = spec.draft
    keep = np.arange(canvas_length) < n
    return Seed(
        n=n,
        k=int(round(spec.draft_noise * n)),
        ids=torch.from_numpy(ids).to(device),
        keep=torch.from_numpy(keep).to(device),
        step=torch.full((), spec.draft_start_step, dtype=torch.int64, device=device),
    )


def completing_prefill_slots(input_batch: Any) -> np.ndarray:
    """The slots (batch order) whose prefill completes in this call, by stock's ``_finish_prefills`` rule.

    Prefill rows are every row on the prefill-only path (``num_draft_tokens == 0``, ``_handle_prefill``) and the
    rows with no logits on a decode call. Pure host numpy.
    """
    n = int(input_batch.num_reqs)
    if input_batch.num_draft_tokens == 0:
        idx = np.arange(n)
    else:
        idx = np.flatnonzero(np.diff(input_batch.cu_num_logits_np[: n + 1]) == 0)
    done = (
        input_batch.num_computed_prefill_tokens_np[idx] + input_batch.num_scheduled_tokens[idx]
        >= input_batch.prefill_len_np[idx]
    )
    out: np.ndarray = input_batch.idx_mapping_np[idx[done]]
    return out


def apply(
    seed: Seed, states: Any, draft_tokens: torch.Tensor, slot: int, generator: torch.Generator
) -> None:
    """Overwrite the slot's first canvas, its draft-token row and its step (device writes only)."""
    CL = int(seed.ids.shape[0])
    dev = seed.ids.device
    uniform = torch.randint(
        0, int(states.vocab_size), (CL,), generator=generator, device=dev, dtype=torch.int64
    )
    keep = seed.keep
    if seed.k:
        keys = torch.rand(seed.n, generator=generator, device=dev)
        kept_head = keys.argsort().argsort() >= seed.k  # exactly k positions have rank < k
        keep = torch.cat([kept_head, seed.keep[seed.n :]])
    row = torch.where(keep, seed.ids, uniform)
    states.canvas[slot] = row
    draft_tokens[slot, :CL] = row
    states.step[slot] = seed.step
