"""``ConstrainedDiffusionSampler``: the per-request CD wrapper around v0.29.0's stock ``DiffusionSampler``.

Design (T-0067; T-0062 decisions 7 and 10):

- **Transparent.** The wrapper defines ``add_request`` and ``__call__`` and nothing else public; every other
  attribute is the inner sampler's own (``__getattr__``), so the runner's feature detection by attribute
  sees the stock object (no ``remove_request``: ``DiffusionSampler`` has none).
- **Fast path.** When no decode slot of the batch belongs to a CD request, ``__call__`` returns the inner
  ``SamplerOutput`` itself after a host-numpy decision; it issues no torch operation.
- **Whole batch through stock.** Otherwise the inner sampler still runs on the whole, unchanged batch, so
  every unconstrained slot's rows and the global RNG are exactly stock's. The CD work never touches the
  global RNG (``end_projection`` is deterministic) and writes only the constrained slots' rows.
- **end_projection.** Stock trajectory, history, acceptance and convergence. Convergence is decided on the
  device inside the stock step, and the host must not wait for it, so the constrained MAP of the step's
  logits is computed on every denoise call of a constrained slot **before** the inner call (from the
  pristine logits) and selected **after** it with ``torch.where`` on ``converged = is_encoder_phase & ~entry``.
  On the converging call ``argmax_canvas`` and ``canvas`` become the MAP (so the commit pass writes it into
  the KV) and ``draft_tokens[slot, :CL]`` is re-synced. On a commit call A_{k+1} = δ*(A_k, committed) and
  ``emitted += valid length``, both on the device.
- **Per-step arms** (T-0068, ``cd/steps.py``). ``perstep_mf`` / ``perstep_mar`` (+ constrained SC) keep the
  same shape: stock still runs on the whole batch (so unconstrained slots and the global RNG are stock's), the
  heavy work (marginals, joint draw from the CD-owned generator, MAP, entropies, the SC matmul) runs before the
  inner call on the pristine logits, and afterwards the slot's canvas, history, ``argmax_canvas``, phase,
  ``confident`` and ``sc_embeds`` rows are overwritten with CD's own values, selected on the device by the entry
  phase. log Z = −inf is checked at every denoise step (ruling P10). The report carries the canvas's accepted
  count so the record's ``mean_accepted_per_step`` needs no extra read.
- **Budget without a host sync.** ``R_after`` is computed on the device from the device emitted count.
  ``L_eff`` must be a host int (T-0066's API): the host knows the emitted count from the reports it has
  read; for the (at most a few) calls whose reports are still unread it enumerates the possible emitted
  counts, computes one MAP per distinct ``L_eff`` and selects the right one on the device. On CPU the
  reports of call t−1 are always read at call t, so exactly one MAP is computed.
- **Loud, non-fatal failures.** A request refused at ``add_request`` (T-0064 reason, ``unsupported_option``,
  ``infeasible``) and a device-detected failure (``z0`` at the converging step, ``empty_delta`` at a
  commit) end the same way: the slot is forced to commit an all-EOS canvas on the next call
  (``argmax_canvas = canvas = EOS``, ``is_encoder_phase = True``), and one error record is emitted. The
  forcing repeats on every later call, so the request never continues unconstrained.
- **Seeded first canvas** (T-0081, ``cd/seed.py``). A request carrying a draft is flagged on the host at
  ``add_request``; on the call where its prefill completes (stock's own host-numpy rule, on the prefill-only path or
  inside a decode call) its canvas, draft-token row and step are overwritten after the inner call, from device
  tensors and the CD-owned generator. Calls with no flagged slot completing its prefill take the paths above
  unchanged (one host-numpy check, no torch op). A request is seeded at most once and never after preemption.
- **Records without a host sync.** Each constrained call packs a small int64 report per constrained slot on
  the device, copies it ``non_blocking`` into a pre-allocated (pinned on CUDA) host buffer and records an
  event; the host reads it at a later call (≥ 1 call later), in order, and emits the ``ASTRA_CD`` lines.
"""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict, deque
from typing import Any

import numpy as np
import torch

from . import record as _record
from . import seed as _seed
from . import steps as _steps
from .budget import delta_star, terminal_factor
from .mapdecode import class_max_of, constrained_map
from .requests import (
    DEV_EMPTY_DELTA,
    DEV_HOST,
    DEV_Z0,
    E_INTERNAL,
    E_PREEMPTED,
    EOS_ID,
    STOP_IDS,
    CdRequest,
    error_name,
    log_internal,
    new_request,
)

__all__ = ["ConstrainedDiffusionSampler"]

_log = logging.getLogger("vllm.astra_cd.sampler")
_HEADER = 7  # ec, step at entry, err, emitted after, in_lang, map_eq, accepted sum (per-step)
_NO_LIMIT = 1 << 30
_MIN_BUF = 4096
#: request ids CD has served, so a preempted request that comes back is recognised (ruling S10)
_TOMBSTONES = 1 << 16


class _Buf:
    """A host buffer the device reports are copied into (pinned on CUDA)."""

    def __init__(self, n: int, device: torch.device) -> None:
        if device.type == "cuda":
            self.t = torch.empty(n, dtype=torch.int64, pin_memory=True)
            self.np = self.t.numpy()
        else:
            self.np = np.zeros(n, dtype=np.int64)
            self.t = torch.from_numpy(self.np)
        self.n = n


class _Pending:
    """One call's reports: where each constrained slot's report sits in the buffer."""

    def __init__(
        self, buf: _Buf, event: Any, items: list[tuple[CdRequest, int, int, int, bool]]
    ) -> None:
        self.buf = buf
        self.event = event
        self.items = items  # (request, offset, valid length, canvas length, live layout)


class ConstrainedDiffusionSampler:
    """Wraps the stock ``DiffusionSampler`` returned by ``DiffusionGemmaModelState.custom_sampler``."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        ds = getattr(inner, "diffusion_states", None)
        n_slots = int(getattr(ds, "max_num_reqs", 0) or 0)
        self._slot_flag = np.zeros(max(n_slots, 1024), dtype=bool)
        #: T-0081: slots whose request still waits for its seed (first prefill completion)
        self._seed_flag = np.zeros_like(self._slot_flag)
        self._n_seed = 0
        self._by_slot: dict[int, CdRequest] = {}
        self._slot_req_id: dict[int, str] = {}
        self._n_cd = 0
        self._pending: deque[_Pending] = deque()
        self._free: list[_Buf] = []
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._resumed_slot: dict[int, str] = {}
        #: per-step arm state of the live per-step requests (T-0068)
        self._perstep: dict[CdRequest, _steps.PerStep] = {}
        self._generators: dict[torch.device, torch.Generator] = {}

    # ------------------------------------------------------------------ transparency

    def __getattr__(self, name: str) -> Any:
        if name == "_inner" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    # ------------------------------------------------------------------ registry

    def _device(self) -> torch.device:
        dev = getattr(getattr(self._inner, "diffusion_states", None), "device", None)
        return torch.device(dev) if dev is not None else torch.device("cpu")

    def _grow(self, slot: int) -> None:
        if slot >= self._slot_flag.shape[0]:
            flag = np.zeros(max(2 * self._slot_flag.shape[0], slot + 1), dtype=bool)
            flag[: self._slot_flag.shape[0]] = self._slot_flag
            seed_flag = np.zeros_like(flag)
            seed_flag[: self._seed_flag.shape[0]] = self._seed_flag
            self._slot_flag, self._seed_flag = flag, seed_flag

    def _unflag_seed(self, slot: int) -> None:
        if self._seed_flag[slot]:
            self._seed_flag[slot] = False
            self._n_seed -= 1

    def _drop(self, slot: int) -> None:
        self._unflag_seed(slot)
        req = self._by_slot.pop(slot, None)
        if req is None:
            return
        self._slot_flag[slot] = False
        self._n_cd -= 1
        self._perstep.pop(req, None)
        req.removed = True
        self._maybe_final_error(req)

    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> None:
        self._grow(req_idx)
        self._drop(req_idx)  # a new request in this slot: whatever sat here is gone
        ds = getattr(self._inner, "diffusion_states", None)
        cl = int(getattr(self._inner, "canvas_length", 0) or getattr(ds, "canvas_length", 0) or 1)
        max_steps = int(getattr(ds, "max_denoising_steps", 0) or 0) or None
        req = new_request(req_idx, sampling_params, self._device(), cl, max_steps)
        inner_params = sampling_params
        if req is not None and getattr(sampling_params, "logprobs", None) is not None:
            # refused below (unsupported_option); stock must not stash logprobs for it either
            inner_params = copy.copy(sampling_params)
            inner_params.logprobs = None
        self._inner.add_request(req_idx, prompt_len, inner_params)
        if req is None:
            self._slot_req_id.pop(req_idx, None)
            return
        req.req_id = self._slot_req_id.get(req_idx)
        resumed = self._resumed_slot.pop(req_idx, None)
        if req.req_id is not None and resumed == req.req_id:
            self._preempt(req)
        self._by_slot[req_idx] = req
        ps = _steps.new_perstep(req, self._inner, self._device())
        if ps is not None:
            self._perstep[req] = ps
        self._slot_flag[req_idx] = True
        self._n_cd += 1
        if req.seed is not None and req.live:
            self._seed_flag[req_idx] = True
            self._n_seed += 1
        if req.req_id is not None:
            self._remember(req.req_id)
        if req.host_error is not None and req.host_error != E_PREEMPTED:
            _log.warning(
                "astra_cd request refused (tag=%s): %s; it ends with an EOS canvas",
                req.record_tag,
                req.detail or req.host_error,
            )

    def _remember(self, req_id: str) -> None:
        self._seen[req_id] = None
        self._seen.move_to_end(req_id)
        while len(self._seen) > _TOMBSTONES:
            self._seen.popitem(last=False)

    def _preempt(self, req: CdRequest) -> None:
        """Ruling S10: a resumed request is not restarted from A_0; it ends loudly."""
        if req.host_error is None:
            req.host_error = E_PREEMPTED
            req.detail = "request resumed after preemption; CD does not continue it"
            _log.warning(
                "astra_cd request preempted and resumed (tag=%s); it ends with an EOS canvas",
                req.record_tag,
            )

    def _bind(self, slot: int, req_id: str, resumed_hint: bool = False) -> None:
        """Model state ``add_request`` (either order with ``add_request`` above).

        ``resumed_hint``: the model state saw a prefill longer than the prompt (a resume after preemption).
        A request id CD has already served is a resume too (tombstone).
        """
        resumed = resumed_hint or req_id in self._seen
        req = self._by_slot.get(slot)
        if req is not None and req.req_id is not None and req.req_id != req_id:
            self._drop(slot)  # stale occupant never removed
            req = None
        if req is not None:
            req.req_id = req_id
            if resumed:
                self._preempt(req)
            self._remember(req_id)
        elif resumed:
            self._resumed_slot[slot] = req_id
        else:
            self._resumed_slot.pop(slot, None)
        self._slot_req_id[slot] = req_id

    def _release(self, slot: int, req_id: str) -> None:
        """Model state ``remove_request``: the CD state of this request is reset."""
        if self._slot_req_id.get(slot) == req_id:
            del self._slot_req_id[slot]
        self._resumed_slot.pop(slot, None)
        req = self._by_slot.get(slot)
        if req is not None and req.req_id in (req_id, None):
            self._drop(slot)
        if self._pending:
            # ruling S11: the runner has synced by removal time, so this reads without waiting
            self._flush()

    # ------------------------------------------------------------------ call

    def __call__(
        self, logits: torch.Tensor, input_batch: Any, draft_logits: torch.Tensor | None = None
    ) -> Any:
        if self._pending:
            self._flush()
        seeds = self._completing_seeds(input_batch) if self._n_seed else None
        out = self._dispatch(logits, input_batch, draft_logits)
        if seeds:
            self._seed(seeds)
        return out

    def _dispatch(
        self, logits: torch.Tensor, input_batch: Any, draft_logits: torch.Tensor | None
    ) -> Any:
        if not self._n_cd or input_batch.num_draft_tokens == 0:
            return self._inner(logits, input_batch, draft_logits)
        n = input_batch.num_reqs
        slots = input_batch.idx_mapping_np[:n]
        flags = self._slot_flag[slots]
        if not flags.any():
            return self._inner(logits, input_batch, draft_logits)
        nlog = np.diff(input_batch.cu_num_logits_np[: n + 1])
        sel = np.flatnonzero(flags & (nlog > 0))
        if sel.size == 0:
            return self._inner(logits, input_batch, draft_logits)
        return self._constrained(logits, input_batch, draft_logits, slots, nlog, sel)

    def _constrained(
        self,
        logits: torch.Tensor,
        input_batch: Any,
        draft_logits: torch.Tensor | None,
        slots: np.ndarray,
        nlog: np.ndarray,
        sel: np.ndarray,
    ) -> Any:
        states = self._inner.diffusion_states
        CL = int(states.canvas_length)
        cu = input_batch.cu_num_logits_np
        jobs: list[_Job] = []
        for i in sel.tolist():
            slot = int(slots[i])
            req = self._by_slot[slot]
            ps = self._perstep.get(req)
            job = _Job(req, slot, int(nlog[i]), int(cu[i]), CL, ps)
            if ps is not None:
                job.inner, job.generator = self._inner, self._generator(logits.device)
            jobs.append(job)
            try:
                job.pre(states, logits)
            except Exception as exc:  # ruling S9: nothing written yet; end this request loudly
                log_internal(job.req, "preparing a constrained step", exc)
                job.live = False
        out = self._inner(logits, input_batch, draft_logits)  # stock exceptions propagate unchanged
        draft = self._inner.req_states.draft_tokens
        for job in jobs:
            try:
                job.post(states, draft)
            except Exception as exc:  # ruling S9: overwrite what post may have partly written
                log_internal(job.req, "writing a constrained step back", exc)
                job.live = False
                try:
                    job.post(states, draft)
                except Exception as exc2:
                    log_internal(job.req, "forcing the EOS canvas", exc2)
        try:
            self._report([job for job in jobs if job.report is not None])
        except Exception as exc:
            # ruling S9: records are lost for this call; the requests end loudly
            for job in jobs:
                log_internal(job.req, "packing the records", exc)
        return out

    # ------------------------------------------------------------------ seed (T-0081)

    def _completing_seeds(self, input_batch: Any) -> list[int] | None:
        """Flagged slots whose prefill completes in this call (host numpy only; ``None``: nothing to seed)."""
        slots = input_batch.idx_mapping_np[: input_batch.num_reqs]
        if not self._seed_flag[slots].any():
            return None
        done = _seed.completing_prefill_slots(input_batch)
        hit = [int(s) for s in done if self._seed_flag[s]]
        return hit or None

    def _seed(self, slots: list[int]) -> None:
        """After the inner call: stock has initialised and flipped these slots; overwrite their first canvas."""
        states = self._inner.diffusion_states
        draft = self._inner.req_states.draft_tokens
        for slot in slots:
            self._unflag_seed(slot)
            req = self._by_slot.get(slot)
            if req is None or req.seed is None:
                continue
            seed, req.seed = req.seed, None  # at most once: a later completion is never re-seeded
            if not req.live:  # preempted (S10) or failed since admission: stock's canvas, ended loudly
                continue
            try:
                _seed.apply(seed, states, draft, slot, self._generator(seed.ids.device))
            except Exception as exc:  # ruling S9: never raise into the engine; the request ends loudly
                log_internal(req, "seeding the first canvas", exc)

    def _generator(self, device: torch.device) -> torch.Generator:
        """The CD-owned generator of the per-step arms (ruling P6): the global RNG is never touched."""
        g = self._generators.get(device)
        if g is None:
            g = torch.Generator(device=device)
            g.manual_seed(_steps.GENERATOR_SEED)
            self._generators[device] = g
        return g

    # ------------------------------------------------------------------ records

    def _buffer(self, n: int, device: torch.device) -> _Buf:
        for k, buf in enumerate(self._free):
            if buf.n >= n:
                return self._free.pop(k)
        return _Buf(max(n, _MIN_BUF), device)

    def _report(self, jobs: list[_Job]) -> None:
        if not jobs:
            return
        parts = [job.report for job in jobs]
        flat = torch.cat(parts) if len(parts) > 1 else parts[0]
        dev = flat.device
        buf = self._buffer(flat.shape[0], dev)
        buf.t[: flat.shape[0]].copy_(flat, non_blocking=True)
        event = None
        if dev.type == "cuda":
            event = torch.cuda.Event()
            event.record()
        items = []
        off = 0
        for job, part in zip(jobs, parts, strict=True):
            items.append((job.req, off, job.valid, job.CL, job.live))
            job.req.pending_valid.append(job.valid)
            off += part.shape[0]
        self._pending.append(_Pending(buf, event, items))

    def _flush(self) -> None:
        while self._pending:
            p = self._pending[0]
            if p.event is not None and not p.event.query():
                return
            self._pending.popleft()
            for req, off, valid, CL, live in p.items:
                req.pending_valid.popleft()  # first, so a raise below cannot misalign the queues
                try:
                    self._read(req, p.buf.np, off, valid, CL, live)
                except Exception as exc:  # ruling S14: never raise into the engine from a flush
                    log_internal(req, "reading a record", exc)
                    try:
                        self._maybe_final_error(req)
                    except Exception as exc2:
                        log_internal(req, "emitting the final error record", exc2)
            self._free.append(p.buf)

    def _read(
        self, req: CdRequest, data: np.ndarray, off: int, valid: int, CL: int, live: bool
    ) -> None:
        h = data[off : off + _HEADER]
        committed_now = bool(h[0])
        step0, err_code = int(h[1]), int(h[2])
        if committed_now:
            req.confirmed_emitted += valid
        if req.dead_code is None and req.host_error is None and err_code:
            req.dead_code = error_name(err_code) or E_INTERNAL
        code = req.host_error or req.dead_code
        if committed_now and not req.error_emitted:
            ids = data[off + _HEADER : off + _HEADER + valid]
            err_here = code
            state_set: list[int] = []
            in_final = False
            shadow = None
            if live and err_here is None and req.bucket is not None and req.is_final is not None:
                base = off + _HEADER + 2 * CL
                a_after = data[base : base + req.bucket].astype(bool)
                state_set = np.flatnonzero(a_after).tolist()
                in_final = bool((a_after & req.is_final).any())
                shadow = data[off + _HEADER + CL : off + _HEADER + CL + valid].tolist()
            rec = _record.make_record(
                record_tag=req.record_tag,
                arm=req.arm,
                canvas_idx=req.canvas_idx,
                committed_ids=ids.tolist(),
                state_set_after=state_set if err_here is None else [],
                in_final=in_final if err_here is None else False,
                steps=step0,
                bucket=req.bucket,
                error=err_here,
                shadow_argmax_ids=None if err_here is not None else shadow,
                argmax_in_language=None if err_here is not None else bool(h[4]),
                map_equals_argmax=None if err_here is not None else bool(h[5]),
                stop_ids=STOP_IDS,
                mean_accepted_per_step=(
                    int(h[6]) / step0
                    if err_here is None and step0 > 0 and req.arm in _steps.PERSTEP_MODES
                    else None
                ),
            )
            _record.emit(rec)
            req.canvas_idx += 1
            if err_here is not None:
                req.error_emitted = True
        self._maybe_final_error(req)

    def _maybe_final_error(self, req: CdRequest) -> None:
        """A failing request removed before its EOS commit was read still gets its one error record."""
        code = req.host_error or req.dead_code
        if req.removed and not req.pending_valid and code is not None and not req.error_emitted:
            req.error_emitted = True
            _record.emit(
                _record.make_record(
                    record_tag=req.record_tag,
                    arm=req.arm,
                    canvas_idx=req.canvas_idx,
                    committed_ids=(),
                    state_set_after=(),
                    in_final=False,
                    steps=0,
                    bucket=req.bucket,
                    error=code,
                )
            )


class _Job:
    """One constrained slot in one call."""

    def __init__(
        self,
        req: CdRequest,
        slot: int,
        valid: int,
        off: int,
        CL: int,
        ps: _steps.PerStep | None = None,
    ) -> None:
        self.req = req
        self.slot = slot
        self.valid = valid
        self.off = off
        self.CL = CL
        self.ps = ps  # the per-step arm (T-0068); None: end_projection
        self.inner: Any = None
        self.generator: torch.Generator | None = None
        self.prep: _steps.Prepared | None = None
        self.report: torch.Tensor | None = None
        self.live = req.live

    # ---- before the inner call: snapshot, the commit's state update, the MAP and its δ*
    def pre(self, st: Any, logits: torch.Tensor) -> None:
        req, s, valid = self.req, self.slot, self.valid
        self.ec = st.is_encoder_phase[s].clone()
        self.step0 = st.step[s].clone()
        self.committed = st.argmax_canvas[s].clone()
        if not self.live:
            return
        assert req.tables is not None and req.A is not None and req.A_next is not None
        assert req.emitted is not None and req.err is not None
        tables = req.tables
        # a commit call applies A_{k+1} = δ*(A_k, committed), computed on the converging call
        self.A_cur = torch.where(self.ec, req.A_next, req.A)
        self.emitted_new = req.emitted + torch.where(self.ec, valid, 0)

        # the MAP of this step's logits (selected only if this call converges; a commit call never does)
        x = logits[self.off : self.off + valid]
        if x.dtype != torch.float32:
            x = x.float()
        headroom = valid if valid < self.CL else _NO_LIMIT
        rem = req.max_tokens - self.emitted_new  # device
        r_cap = torch.clamp(rem, max=headroom)
        cands = _l_eff_candidates(req, valid, self.CL)
        l_dev = torch.clamp(torch.clamp(rem, max=min(self.CL, valid)), min=0)
        if self.ps is not None:
            assert self.generator is not None
            pr = _steps.prepare(
                self.ps,
                self.inner,
                st,
                s,
                x,
                tables,
                self.A_cur,
                r_cap,
                l_dev,
                cands,
                self.step0,
                self.generator,
            )
            self.prep = pr
            self.map_tokens, self.map_empty, self.l_pick = pr.tokens, pr.map_empty, pr.l_pick
            self.ds = delta_star(tables, self.A_cur, pr.tokens, STOP_IDS)
            return
        tokens = empty = l_pick = None
        cmax = class_max_of(x, tables)  # once per slot per call, shared by every L_eff candidate
        for l_eff in cands:
            b = terminal_factor(tables, r_cap - l_eff)
            m = constrained_map(x, tables, self.A_cur, b, l_eff, class_max=cmax)
            if tokens is None:
                tokens, empty = m.tokens, m.empty
                l_pick = torch.full((), l_eff, dtype=torch.int64, device=x.device)
            else:
                hit = l_dev == l_eff
                tokens = torch.where(hit, m.tokens, tokens)
                empty = torch.where(hit, m.empty, empty)
                l_pick = torch.where(hit, l_eff, l_pick)
        assert tokens is not None and empty is not None and l_pick is not None
        self.map_tokens, self.map_empty, self.l_pick = tokens, empty, l_pick
        # δ* of the canvas this call would commit (checked for emptiness at the converging call)
        self.ds = delta_star(tables, self.A_cur, tokens, STOP_IDS)

    # ---- after the inner call: select, force, write back, pack the report
    def post(self, st: Any, draft: torch.Tensor) -> None:
        req, s, valid, CL = self.req, self.slot, self.valid, self.CL
        eos = torch.full((CL,), EOS_ID, dtype=st.argmax_canvas.dtype, device=self.ec.device)
        self.acc = torch.zeros((), dtype=torch.int64, device=eos.device)
        if self.live and self.ps is not None:
            self._post_perstep(st, eos)
            header_err = req.err
            emitted = self.emitted_new
            zero = torch.zeros((), dtype=torch.bool, device=eos.device)
            flags = (zero, zero)
            tail = [self.committed, req.shadow, req.A.to(torch.int64)]  # type: ignore[union-attr]
        elif self.live:
            assert req.err is not None and req.shadow is not None
            assert req.A is not None and req.A_next is not None and req.emitted is not None
            assert req.in_lang is not None and req.map_eq is not None
            conv = st.is_encoder_phase[s] & ~self.ec
            stock_row = st.argmax_canvas[s].clone()
            new_row = stock_row.clone()
            new_row[:valid] = torch.where(self.map_empty, EOS_ID, self.map_tokens)
            clean = req.err == 0
            z0_now = conv & self.map_empty & clean
            empty_now = conv & ~self.map_empty & self.ds.empty & clean
            err = torch.where(z0_now, DEV_Z0, torch.where(empty_now, DEV_EMPTY_DELTA, req.err))
            pos = torch.arange(valid, device=stock_row.device)
            same = new_row[:valid] == stock_row[:valid]
            map_eq = same.all()
            in_lang = ~self.map_empty & (same | (pos >= self.l_pick)).all()
            req.shadow.copy_(torch.where(conv, stock_row, req.shadow))
            req.in_lang.copy_(torch.where(conv, in_lang, req.in_lang))
            req.map_eq.copy_(torch.where(conv, map_eq, req.map_eq))
            req.A.copy_(self.A_cur)
            req.A_next.copy_(torch.where(conv, self.ds.states, req.A_next))
            req.emitted.copy_(self.emitted_new)
            req.err.copy_(err)
            force = err != 0
            chosen = torch.where(conv, new_row, stock_row)
            final = torch.where(force, eos, chosen)
            st.argmax_canvas[s] = final
            st.canvas[s] = torch.where(conv | force, final, st.canvas[s])
            st.is_encoder_phase[s] = st.is_encoder_phase[s] | force
            header_err = err
            emitted, flags = self.emitted_new, (req.in_lang, req.map_eq)
            tail = [self.committed, req.shadow, req.A.to(torch.int64)]
        else:
            if not hasattr(self, "committed"):  # pre raised before its snapshot: not a commit here
                self.ec = torch.zeros((), dtype=torch.bool, device=eos.device)
                self.step0 = torch.zeros((), dtype=torch.int64, device=eos.device)
                self.committed = torch.zeros(CL, dtype=torch.int64, device=eos.device)
            st.argmax_canvas[s] = eos
            st.canvas[s] = eos
            # a device-resident True (T-0073 D6 item 1): a Python scalar written into a CUDA row is a host sync
            st.is_encoder_phase[s] = torch.ones((), dtype=torch.bool, device=eos.device)
            header_err = torch.full((), DEV_HOST, dtype=torch.int64, device=eos.device)
            zero = torch.zeros((), dtype=torch.bool, device=eos.device)
            emitted, flags = torch.zeros((), dtype=torch.int64, device=eos.device), (zero, zero)
            tail = [self.committed]
        draft[s, :CL] = st.canvas[s]
        header = torch.stack(
            [
                self.ec.to(torch.int64),
                self.step0.to(torch.int64),
                header_err.to(torch.int64),
                emitted.to(torch.int64),
                flags[0].to(torch.int64),
                flags[1].to(torch.int64),
                self.acc,
            ]
        )
        self.report = torch.cat([header, *(t.to(torch.int64) for t in tail)])

    def _post_perstep(self, st: Any, eos: torch.Tensor) -> None:
        """Per-step arm (T-0068): CD's own history, convergence, fed-back canvas and sc rows."""
        req, s, pr, ps = self.req, self.slot, self.prep, self.ps
        assert pr is not None and ps is not None
        assert req.err is not None and req.A is not None and req.A_next is not None
        assert req.emitted is not None
        conv, den, acc = _steps.write(ps, pr, st, s, self.ec)
        clean = req.err == 0
        # ruling P10: log Z = -inf is checked at every denoise step; δ* at the converging one. Once the
        # budget is spent (emitted ≥ max_tokens) the engine commits nothing more, and L_eff = 0 with
        # R_after < 0 makes Z = 0 by construction: that is the end of the request, not a failure.
        in_budget = self.emitted_new < req.max_tokens
        z0_now = den & self.map_empty & clean & in_budget
        empty_now = conv & ~self.map_empty & self.ds.empty & clean
        err = torch.where(z0_now, DEV_Z0, torch.where(empty_now, DEV_EMPTY_DELTA, req.err))
        req.A.copy_(self.A_cur)
        req.A_next.copy_(torch.where(conv, self.ds.states, req.A_next))
        req.emitted.copy_(self.emitted_new)
        req.err.copy_(err)
        _steps.finish(pr, st, s, self.ec, conv, err != 0, eos)
        self.acc = acc


def _l_eff_candidates(req: CdRequest, valid: int, CL: int) -> list[int]:
    """Every ``L_eff`` the device emitted count can imply, given the reports not yet read."""
    cands = {req.confirmed_emitted}
    for v in req.pending_valid:
        cands |= {e + v for e in cands}
    return sorted({max(0, min(CL, valid, req.max_tokens - e)) for e in cands})
