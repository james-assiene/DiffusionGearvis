"""A faithful CPU stand-in for v0.29.0's DiffusionGemma sampling stack (T-0067 tier 1).

Everything that decides a token is the image's own code, vendored verbatim under
``tests/fixtures/vllm_v0_29_0/`` (``DiffusionSampler``, ``DiffusionGemmaRequestStates``,
``SamplingStates``, ``_compiled_sample_step``, the pytorch top-k/top-p). This module only **drives** it
(`.claude/rules/tests.md`, "closes by delegation"): it builds ``InputBatch`` stand-ins with the image's
attribute names, owns the request/slot bookkeeping a scheduler would, and snapshots rows. It never
re-implements a line of the step.

``FakeModelStateBase`` stands in for ``DiffusionGemmaModelState``: its ``custom_sampler`` constructs the
vendored ``DiffusionSampler`` exactly as upstream does (``diffusion_gemma.py:828-860``) and returns
``(sampler, None)``; ``add_request``/``remove_request`` are the upstream bodies (``:865-875``).

The CD code under test is reached only through its public entry points:
``make_cd_model_state_cls(base)`` (``astra_vllm_plugin.cd.model_state``) and the sampler it returns.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

#: ``ASTRA_CD_DEVICE`` (default ``cpu``) and ``ASTRA_CD_STOCK=image`` (T-0067 criterion 12, inside the pinned
#: container: the image's own compiled ``DiffusionSampler`` / ``SamplingStates`` on CUDA instead of the vendored
#: eager copy). The tier-1 default is the vendored copy on CPU.
DEVICE = torch.device(os.environ.get("ASTRA_CD_DEVICE", "cpu"))
if os.environ.get("ASTRA_CD_STOCK") == "image":
    from vllm.model_executor.models import diffusion_gemma as _dg
    from vllm.v1.worker.gpu.sample import states as _states

    stock = SimpleNamespace(
        DiffusionSampler=_dg.DiffusionSampler,
        DiffusionGemmaRequestStates=_dg.DiffusionGemmaRequestStates,
        SamplingStates=_states.SamplingStates,
    )
else:
    from tests.fixtures.vllm_v0_29_0 import diffusion_sampler as stock

REPO = Path(__file__).resolve().parents[3]
PLUGIN_SRC = REPO / "plugin" / "astra_vllm_plugin"

V = 262_144
EOS = 1
STOP_IDS = (1, 50, 106)

#: The state rows (criterion 4) as ``DiffusionGemmaRequestStates`` names them, plus the runner's
#: ``req_states.draft_tokens``.
STATE_ROWS = (
    "canvas",
    "argmax_canvas",
    "step",
    "is_encoder_phase",
    "confident",
    "self_conditioning_embeds",
    "accepted_canvas_history",
    "accepted_canvas_history_len",
)
OUTPUT_ROWS = ("sampled", "num_sampled", "num_rejected")


def cd_module(name: str) -> ModuleType:
    """``astra_vllm_plugin.cd.<name>``; absent means "T-0067 not implemented", as an assertion."""
    if str(PLUGIN_SRC) not in sys.path:
        sys.path.insert(0, str(PLUGIN_SRC))
    mod = f"astra_vllm_plugin.cd.{name}"
    try:
        return importlib.import_module(mod)
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("astra_vllm_plugin.cd"):
            pytest.fail(f"T-0067 not implemented: `{mod}` does not exist ({exc})")
        raise


def make_cd_model_state_cls() -> Callable[[type], type]:
    fn = getattr(cd_module("model_state"), "make_cd_model_state_cls", None)
    if fn is None:
        pytest.fail("T-0067 not implemented: cd.model_state.make_cd_model_state_cls is missing")
    return fn  # type: ignore[no-any-return]


def cd_sampler_cls() -> type:
    cls = getattr(cd_module("sampler"), "ConstrainedDiffusionSampler", None)
    if cls is None:
        pytest.fail("T-0067 not implemented: cd.sampler.ConstrainedDiffusionSampler is missing")
    return cls  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- configuration


@dataclasses.dataclass(frozen=True)
class Cfg:
    CL: int = 4
    max_num_reqs: int = 8
    hidden: int = 4
    max_denoising_steps: int = 48
    #: generation_config ``stability_threshold``; the states hold ``+1`` (upstream :800)
    stability_threshold: int = 1
    t_min: float = 0.4
    t_max: float = 0.8
    entropy_bound: float = 1.0
    confidence_threshold: float = (
        1e9  # always confident: convergence = stability (set 0 to test the cap)
    )
    prompt_len: int = 5


class FakeRunnerSampler:
    """The runner's ``Sampler`` as ``DiffusionSampler.__init__`` reads it: ``sampling_states``,
    ``req_states`` (``.draft_tokens``), ``logprobs_mode`` (image ``sample/sampler.py:47-53``)."""

    def __init__(self, cfg: Cfg) -> None:
        self.sampling_states = stock.SamplingStates(cfg.max_num_reqs, V)
        self.req_states = SimpleNamespace(
            draft_tokens=torch.zeros(cfg.max_num_reqs, cfg.CL, dtype=torch.int64, device=DEVICE)
        )
        self.logprobs_mode = "raw_logprobs"


class FakeModelStateBase:
    """Stands in for ``DiffusionGemmaModelState`` (see the module docstring)."""

    #: the class ``custom_sampler`` constructs; tests may substitute a subclass that counts calls
    sampler_cls: type = stock.DiffusionSampler
    #: ``custom_sampler`` behaviour switch for criterion 1 ("tuple" | "none" | "raise")
    custom_sampler_mode = "tuple"

    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self.diffusion_states = stock.DiffusionGemmaRequestStates(
            max_num_reqs=cfg.max_num_reqs,
            canvas_length=cfg.CL,
            vocab_size=V,
            max_denoising_steps=cfg.max_denoising_steps,
            device=DEVICE,
            hidden_size=cfg.hidden,
            stability_threshold=cfg.stability_threshold + 1,
        )
        self._req_id_to_index: dict[str, int] = {}
        g = torch.Generator().manual_seed(1234)
        self.embed_weight = (torch.randn(V, cfg.hidden, generator=g) * 0.02).to(DEVICE)
        self.normalizer = torch.tensor(cfg.hidden**0.5, device=DEVICE)
        self.constructed: list[Any] = []

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        if self.custom_sampler_mode == "none":
            return None
        if self.custom_sampler_mode == "raise":
            raise ValueError("DiffusionGemma requires an EntropyBound sampler_config")
        cfg = self.cfg
        inner = self.sampler_cls(
            sampler=sampler,
            diffusion_config=SimpleNamespace(canvas_length=cfg.CL),
            vocab_size=V,
            diffusion_states=self.diffusion_states,
            t_min=cfg.t_min,
            t_max=cfg.t_max,
            entropy_bound=cfg.entropy_bound,
            confidence_threshold=cfg.confidence_threshold,
            embed_weight=self.embed_weight,
            normalizer=self.normalizer,
        )
        self.constructed.append(inner)
        return inner, None

    # upstream diffusion_gemma.py:865-875
    def add_request(self, req_index: int, new_req_data: Any) -> None:
        self._req_id_to_index[new_req_data.req_id] = req_index
        self.diffusion_states.add_request(req_index)
        if not new_req_data.req_id.startswith("_warmup_"):
            prompt_len = len(new_req_data.prompt_token_ids)
            self.diffusion_states.prompt_len[req_index].fill_(prompt_len)

    def remove_request(self, req_id: str) -> None:
        idx = self._req_id_to_index.pop(req_id, None)
        if idx is not None:
            self.diffusion_states.remove_request(idx)


def sampling_params(
    *,
    extra_args: Mapping[str, Any] | None = None,
    max_tokens: int = 4096,
    top_k: int = 0,
    top_p: float = 1.0,
    logprobs: int | None = None,
) -> SimpleNamespace:
    """``SamplingParams`` as ``SamplingStates.add_request`` and the CD wrapper read it."""
    return SimpleNamespace(
        temperature=1.0,
        top_p=top_p,
        top_k=top_k,
        min_p=0.0,
        seed=None,
        logprobs=logprobs,
        max_tokens=max_tokens,
        extra_args=None if extra_args is None else dict(extra_args),
        stop_token_ids=list(STOP_IDS),
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )


# --------------------------------------------------------------------------- the driven world


@dataclasses.dataclass
class Req:
    req_id: str
    slot: int
    params: SimpleNamespace
    prefilled: bool = False
    valid_len: int | None = None  # None: CL


@dataclasses.dataclass
class Snap:
    """One sampler call, cloned on the host right after it returned."""

    call: int
    kind: str  # "prefill" | "decode"
    order: tuple[str, ...]  # req ids in batch order
    slots: dict[str, int]
    state: dict[str, torch.Tensor]  # full [max_num_reqs, ...] clones
    draft_tokens: torch.Tensor
    out: dict[str, torch.Tensor]  # [num_reqs, ...] clones
    returned: Any
    rng_after: torch.Tensor
    entered_commit: dict[str, bool]  # is_encoder_phase at entry, per decode request

    def row(self, name: str, req_id: str) -> torch.Tensor:
        return self.state[name][self.slots[req_id]]

    def out_row(self, name: str, req_id: str) -> torch.Tensor:
        return self.out[name][self.order.index(req_id)]

    def committed(self, req_id: str) -> list[int] | None:
        """The ids this call committed for ``req_id`` (valid length), or None on a denoise step."""
        if self.kind != "decode" or req_id not in self.order:
            return None
        n = int(self.out_row("num_sampled", req_id))
        if n == 0:
            return None
        return [int(x) for x in self.out_row("sampled", req_id)[:n]]


LogitsFn = Callable[[int, str, int], torch.Tensor]  # (call, req_id, valid_len) -> [valid_len, V]


class World:
    """One engine: a model state (stock or CD-wrapped), its sampler, and a toy scheduler.

    ``cd=True`` builds ``make_cd_model_state_cls(FakeModelStateBase)``; ``cd=False`` the stock base.
    The global torch RNG is seeded once here; both runs of a comparison must make the same calls.
    """

    def __init__(
        self, cfg: Cfg, *, cd: bool, seed: int = 0, sampler_cls: type | None = None
    ) -> None:
        torch.manual_seed(seed)
        base: type = FakeModelStateBase
        if sampler_cls is not None:
            base = type("Base", (FakeModelStateBase,), {"sampler_cls": sampler_cls})
        cls = make_cd_model_state_cls()(base) if cd else base
        self.cfg = cfg
        self.ms = cls(cfg)
        self.runner_sampler = FakeRunnerSampler(cfg)
        got = self.ms.custom_sampler(self.runner_sampler)
        assert got is not None
        self.sampler, _ = got
        self.inner = self.ms.constructed[-1]
        self.reqs: dict[str, Req] = {}
        self.free_slots: list[int] = list(range(cfg.max_num_reqs - 1, -1, -1))  # pop() -> 0 first
        self.calls = 0
        self.snaps: list[Snap] = []
        #: ``decode_unsynced`` only: per call, the batch's ``is_encoder_phase`` at entry, left on the device
        self.entry_phases: list[torch.Tensor] = []
        #: context entered around **only** the sampler call (instruments must not see the harness's own work)
        self.call_ctx: Callable[[], Any] = contextlib.nullcontext

    # ---- requests
    def add(
        self,
        req_id: str,
        *,
        xargs: Mapping[str, Any] | None = None,
        sampler_first: bool = False,
        slot: int | None = None,
        prefill_token_ids: Sequence[int] | None = None,
        **params: Any,
    ) -> int:
        """Admit a request. Slots are reused last-in first-out (``free_slots`` is a stack).

        ``prefill_token_ids`` is ``NewRequestData.prefill_token_ids`` (image ``output.py:49``); the default
        equals the prompt, as for a fresh request. The V2 scheduler re-sends a resumed request with
        ``prefill_token_ids=req._all_token_ids`` (prompt + outputs so far; ruling S14)."""
        if slot is None:
            slot = self.free_slots.pop()
        else:
            self.free_slots.remove(slot)
        sp = sampling_params(extra_args=xargs, **params)
        prompt = [2] * self.cfg.prompt_len
        new_req_data = SimpleNamespace(
            req_id=req_id,
            prompt_token_ids=prompt,
            prefill_token_ids=list(prompt if prefill_token_ids is None else prefill_token_ids),
        )
        if sampler_first:
            self.sampler.add_request(slot, self.cfg.prompt_len, sp)
            self.ms.add_request(slot, new_req_data)
        else:
            self.ms.add_request(slot, new_req_data)
            self.sampler.add_request(slot, self.cfg.prompt_len, sp)
        self.sampler.apply_staged_writes()
        self.reqs[req_id] = Req(req_id, slot, sp)
        return slot

    def remove(self, req_id: str) -> None:
        req = self.reqs.pop(req_id)
        self.ms.remove_request(req_id)
        self.free_slots.append(req.slot)

    # ---- batches
    def _batch(
        self, reqs: Sequence[Req], *, prefill: bool, non_blocking: bool = False
    ) -> SimpleNamespace:
        n = len(reqs)
        CL = self.cfg.CL
        slots = np.array([r.slot for r in reqs], dtype=np.int64)
        if prefill:
            nlog = np.zeros(n, dtype=np.int64)
            qlens = np.full(n, self.cfg.prompt_len, dtype=np.int64)
        else:
            nlog = np.array([r.valid_len or CL for r in reqs], dtype=np.int64)
            qlens = nlog.copy()
        cu = np.concatenate([[0], np.cumsum(nlog)]).astype(np.int32)
        qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)

        def up(a: np.ndarray) -> torch.Tensor:
            return _upload(torch.from_numpy(a.copy()), non_blocking)

        return SimpleNamespace(
            req_ids=[r.req_id for r in reqs],
            num_reqs=n,
            idx_mapping=up(slots),
            idx_mapping_np=slots,
            num_draft_tokens=0 if prefill else int(nlog.sum()),
            cu_num_logits_np=cu,
            cu_num_logits=up(cu),
            query_start_loc_np=qsl,
            query_start_loc=up(qsl),
            num_scheduled_tokens=qlens.astype(np.int32),
            num_computed_prefill_tokens_np=np.zeros(n, dtype=np.int32),
            prefill_len_np=np.full(n, self.cfg.prompt_len, dtype=np.int32),
        )

    def _snap(self, kind: str, reqs: Sequence[Req], returned: Any, entry: dict[str, bool]) -> Snap:
        st = self.ms.diffusion_states
        out = {}
        if returned is not None:
            out = {
                "sampled": returned.sampled_token_ids.clone().cpu(),
                "num_sampled": returned.num_sampled.clone().cpu(),
                "num_rejected": returned.num_rejected.clone().cpu(),
            }
        snap = Snap(
            call=self.calls,
            kind=kind,
            order=tuple(r.req_id for r in reqs),
            slots={r.req_id: r.slot for r in reqs},
            state={name: getattr(st, name).clone().cpu() for name in STATE_ROWS},
            draft_tokens=self.runner_sampler.req_states.draft_tokens.clone().cpu(),
            out=out,
            returned=returned,
            rng_after=(
                torch.cat([torch.get_rng_state(), torch.cuda.get_rng_state()])
                if DEVICE.type == "cuda"
                else torch.get_rng_state()
            ),
            entered_commit=entry,
        )
        self.snaps.append(snap)
        self.calls += 1
        return snap

    def prefill(self) -> Snap:
        """One prefill-only call for every admitted request that has not prefilled yet."""
        reqs = [r for r in self.reqs.values() if not r.prefilled]
        batch = self._batch(reqs, prefill=True)
        logits = torch.zeros(0, V, device=DEVICE)
        with self.call_ctx():
            ret = self.sampler(logits, batch)
        for r in reqs:
            r.prefilled = True
        return self._snap("prefill", reqs, ret, {})

    def batch_for(self, reqs: Sequence[Req]) -> SimpleNamespace:
        return self._batch(reqs, prefill=False)

    def decode(self, logits_fn: LogitsFn, *, only: Sequence[str] | None = None) -> Snap:
        """One decode call over every prefilled request (in admission order) or ``only``."""
        reqs = [r for r in self.reqs.values() if r.prefilled and (only is None or r.req_id in only)]
        batch = self._batch(reqs, prefill=False)
        parts = [
            logits_fn(self.calls, r.req_id, r.valid_len or self.cfg.CL).to(torch.float32)
            for r in reqs
        ]
        logits = (torch.cat(parts, dim=0) if parts else torch.zeros(0, V)).to(DEVICE)
        st = self.ms.diffusion_states
        entry = {r.req_id: bool(st.is_encoder_phase[r.slot]) for r in reqs}
        with self.call_ctx():
            ret = self.sampler(logits, batch)
        return self._snap("decode", reqs, ret, entry)

    def decode_unsynced(
        self, logits_fn: LogitsFn, *, only: Sequence[str] | None = None
    ) -> list[str]:
        """``decode`` without a single host read of device memory (the CUDA lag row, ruling S14).

        No snapshot is taken; the entry phases are kept on the device (``entry_phases``, a ``[num_reqs]``
        clone per call, enqueued before the call and never read here); the batch tensors and logits are
        uploaded from pinned memory with ``non_blocking=True`` (a pageable ``.to(cuda)`` synchronises the
        stream in PyTorch and would drain the lag). Returns the req ids in batch order."""
        reqs = [r for r in self.reqs.values() if r.prefilled and (only is None or r.req_id in only)]
        batch = self._batch(reqs, prefill=False, non_blocking=True)
        parts = [
            logits_fn(self.calls, r.req_id, r.valid_len or self.cfg.CL).to(torch.float32)
            for r in reqs
        ]
        logits = _upload(torch.cat(parts, dim=0) if parts else torch.zeros(0, V), True)
        st = self.ms.diffusion_states
        self.entry_phases.append(st.is_encoder_phase[batch.idx_mapping].clone())
        with self.call_ctx():
            self.sampler(logits, batch)
        self.calls += 1
        return [r.req_id for r in reqs]


def _upload(t: torch.Tensor, non_blocking: bool) -> torch.Tensor:
    """Host tensor -> DEVICE; ``non_blocking`` pins first so the copy is truly asynchronous on CUDA."""
    if non_blocking and DEVICE.type == "cuda":
        return t.pin_memory().to(DEVICE, non_blocking=True)
    return t.to(DEVICE)


# --------------------------------------------------------------------------- logits


def row_logits(values: Mapping[int, float], *, floor: float = -30.0, CL: int) -> torch.Tensor:
    """``[CL, V]`` at ``floor`` (the softcap minimum) except ``values[token]`` at every position."""
    x = torch.full((CL, V), floor, dtype=torch.float32)
    for t, v in values.items():
        x[:, t] = v
    return x


def random_live_logits(
    seed: int,
    live: Sequence[int],
    CL: int,
    *,
    lo: float = 0.0,
    hi: float = 6.0,
    floor: float = -30.0,
) -> torch.Tensor:
    """``[CL, V]``: ``floor`` everywhere except the ``live`` ids, uniform in ``[lo, hi]`` (fixed seed)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.full((CL, V), floor, dtype=torch.float32)
    x[:, list(live)] = lo + (hi - lo) * torch.rand(CL, len(live), generator=g)
    return x


def bits(t: torch.Tensor) -> torch.Tensor:
    """Bit pattern view for exact comparison (NaN-safe)."""
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype == torch.float64:
        return t.view(torch.int64)
    return t


def rows_differ(a: Snap, b: Snap, req_id: str) -> list[str]:
    """Criterion 4's field list: every state row, draft_tokens[:CL] and every output row that differs."""
    bad = []
    sa, sb = a.slots[req_id], b.slots[req_id]
    for name in STATE_ROWS:
        if not torch.equal(bits(a.state[name][sa]), bits(b.state[name][sb])):
            bad.append(name)
    CL = a.state["canvas"].shape[1]
    if not torch.equal(a.draft_tokens[sa, :CL], b.draft_tokens[sb, :CL]):
        bad.append("draft_tokens")
    if a.out and b.out:
        for name in OUTPUT_ROWS:
            if not torch.equal(a.out_row(name, req_id), b.out_row(name, req_id)):
                bad.append(name)
    return bad
