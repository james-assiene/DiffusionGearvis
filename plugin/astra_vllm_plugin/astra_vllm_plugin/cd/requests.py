"""Per-request CD state (T-0067 criterion 5): parsed options, device tables and the per-canvas state.

Built on the engine thread in ``add_request`` (host work, allowed to allocate and to copy to the device);
never raises. A request that cannot be served constrained gets ``host_error`` set to a reason code and is
ended loudly by the sampler (T-0062 decision 7) -- it is never served unconstrained.

Error codes (ruling S3): T-0064's ``CdRequestError.reason`` for refused options (and T-0081's draft reasons,
the two that need CL / ``max_denoising_steps`` checked here at admission); ``unsupported_option``
(logprobs, top_k/top_p on any arm; ``entropy_bound``/``confidence_threshold`` on end_projection, whose
trajectory is stock's; T-0068 ruling P13); ``infeasible`` (no string within
``max_tokens`` reaches F, T-0065 criterion 6); ``z0`` (log Z = −inf at the converging step) and
``empty_delta`` (δ* empty at a commit) are raised on the device by the sampler.

Device state lives **per request**, not per slot, so a slot re-taken by another request can never see
the previous occupant's automaton, state set A_k or emitted count.
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import re
import threading
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from . import seed as _seed
from .budget import initial_infeasibility, prepare_stop_tensor
from .emission import CdTables, build_tables
from .xargs import CD_PREFIX, MODES, CdRequestError, CdSpec, WireAutomaton, parse_cd_xargs

__all__ = [
    "EOS_ID",
    "E_EMPTY_DELTA",
    "E_INFEASIBLE",
    "E_INTERNAL",
    "E_PREEMPTED",
    "E_UNSUPPORTED",
    "E_Z0",
    "STOP_IDS",
    "CdRequest",
    "error_name",
    "log_internal",
    "new_request",
]

#: T-0063's stop augmentation; EOS is the first.
STOP_IDS = (1, 50, 106)
EOS_ID = 1

E_Z0 = "z0"
E_EMPTY_DELTA = "empty_delta"
E_INFEASIBLE = "infeasible"
E_UNSUPPORTED = "unsupported_option"
E_INTERNAL = "internal"
E_PREEMPTED = "preempted"

#: Device-side error codes (int64 in ``CdRequest.err``); 0 = none. Host-side codes never go to the device.
DEVICE_CODES = {1: E_Z0, 2: E_EMPTY_DELTA}
DEV_Z0 = 1
DEV_EMPTY_DELTA = 2
#: A host-known error is reported through the same header slot with this marker.
DEV_HOST = 3

SERVED_MODES = ("end_projection", "perstep_mf", "perstep_mar", "perstep_mar_qargmax")
_TAG = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
_NO_LIMIT = 1 << 30
_log = logging.getLogger("vllm.astra_cd.requests")


def error_name(code: int) -> str | None:
    return DEVICE_CODES.get(code)


@dataclasses.dataclass(eq=False)
class CdRequest:
    """One constrained request (or one that must be ended loudly)."""

    slot: int
    req_id: str | None
    record_tag: str | None
    arm: str | None
    max_tokens: int
    host_error: str | None = None
    detail: str = ""
    spec: CdSpec | None = None
    bucket: int | None = None
    tables: CdTables | None = None
    is_final: np.ndarray | None = None
    # device state (live requests only)
    A: torch.Tensor | None = None  # bool [bucket]: the state set A_k
    A_next: torch.Tensor | None = None  # bool [bucket]: δ*(A_k, MAP) of the last convergence
    emitted: torch.Tensor | None = None  # int64 0-d: tokens committed so far
    err: torch.Tensor | None = None  # int64 0-d: DEVICE_CODES key, 0 = none
    shadow: torch.Tensor | None = None  # int64 [CL]: stock argmax at the last convergence
    in_lang: torch.Tensor | None = None  # bool 0-d
    map_eq: torch.Tensor | None = None  # bool 0-d
    #: T-0081: the first canvas's seed (device tensors); set to None on the host once applied (never re-seeded)
    seed: _seed.Seed | None = None
    # host bookkeeping (lags the device by the unread reports)
    confirmed_emitted: int = 0
    canvas_idx: int = 0
    pending_valid: collections.deque[int] = dataclasses.field(default_factory=collections.deque)
    dead_code: str | None = None
    error_emitted: bool = False
    removed: bool = False

    @property
    def live(self) -> bool:
        return self.host_error is None and self.dead_code is None


# --------------------------------------------------------------------------- tables cache

_TABLES_CACHE = 16
_tables: collections.OrderedDict[tuple[str, str], CdTables] = collections.OrderedDict()
_tables_lock = threading.Lock()


def _tables_for(sha: str, wa: WireAutomaton, device: torch.device) -> CdTables:
    key = (sha, str(device))
    with _tables_lock:
        hit = _tables.get(key)
        if hit is not None:
            _tables.move_to_end(key)
            return hit
    n = wa.n_states
    indptr, indices = wa.sum_indptr, wa.sum_indices
    members = [indices[int(indptr[c]) : int(indptr[c + 1])] for c in range(wa.n_classes)]
    tables = build_tables(
        n_states=n,
        vocab_size=wa.vocab_size,
        class_members=members,
        class_is_neg=[bool(x) for x in wa.sum_is_neg],
        edge_src=wa.edge_src.astype(np.int64),
        edge_dst=wa.edge_dst.astype(np.int64),
        edge_class=wa.edge_class.astype(np.int64),
        finals=np.flatnonzero(wa.is_final[:n]).astype(np.int64),
        bucket=wa.n_states_bucket,
        device=device,
    )
    with _tables_lock:
        _tables[key] = tables
        while len(_tables) > _TABLES_CACHE:
            _tables.popitem(last=False)
    return tables


# --------------------------------------------------------------------------- construction


def _raw_tag_and_arm(extra_args: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    """Best-effort tag and mode for an error record of a request the parser refused (ruling W8/W9)."""
    x = extra_args or {}
    tag = x.get(CD_PREFIX + "record_tag")
    mode = x.get(CD_PREFIX + "mode")
    ok_tag = tag if type(tag) is str and _TAG.fullmatch(tag) else None
    ok_mode = mode if type(mode) is str and mode in MODES else None
    return ok_tag, ok_mode


def _unsupported(sampling_params: Any, spec: CdSpec, vocab_size: int) -> str | None:
    if spec.mode not in SERVED_MODES:
        return f"mode {spec.mode!r} is not served by this sampler"
    if spec.mode == "end_projection" and (
        spec.entropy_bound is not None or spec.confidence_threshold is not None
    ):
        return (
            "entropy_bound/confidence_threshold would change the stock trajectory of end_projection"
        )
    if getattr(sampling_params, "logprobs", None) is not None:
        return "logprobs on a constrained request (stock would report the unconstrained argmax's)"
    top_k = getattr(sampling_params, "top_k", 0)
    if top_k is not None and 0 < int(top_k) < vocab_size:
        return f"top_k={top_k} on a constrained request"
    top_p = getattr(sampling_params, "top_p", 1.0)
    if top_p is not None and float(top_p) < 1.0:
        return f"top_p={top_p} on a constrained request"
    return None


def new_request(
    slot: int,
    sampling_params: Any,
    device: torch.device,
    canvas_length: int,
    max_steps: int | None = None,
) -> CdRequest | None:
    """``None`` for an unconstrained request; otherwise a live or loudly failing ``CdRequest``."""
    extra = getattr(sampling_params, "extra_args", None)
    max_tokens = getattr(sampling_params, "max_tokens", None)
    max_tokens = _NO_LIMIT if max_tokens is None else int(max_tokens)
    try:
        spec = parse_cd_xargs(extra)
    except CdRequestError as err:
        tag, arm = _raw_tag_and_arm(extra)
        return CdRequest(slot, None, tag, arm, max_tokens, host_error=err.reason, detail=str(err))
    if spec is None:
        return None
    wa = spec.automaton
    req = CdRequest(
        slot, None, spec.record_tag, spec.mode, max_tokens, spec=spec, bucket=wa.n_states_bucket
    )
    refused = _seed.refusal(spec, canvas_length, max_steps)
    if refused is not None:
        req.host_error, req.detail = refused
        return req
    why = _unsupported(sampling_params, spec, wa.vocab_size)
    if why is not None:
        req.host_error, req.detail = E_UNSUPPORTED, why
        return req
    try:
        _build_live(req, spec, device, canvas_length)
    except Exception as exc:  # decision 7 / ruling S9: never raise into the engine thread
        log_internal(req, "building the request's device state", exc)
    return req


def _build_live(req: CdRequest, spec: CdSpec, device: torch.device, canvas_length: int) -> None:
    wa = spec.automaton
    tables = _tables_for(spec.automaton_sha256, wa, device)
    a_start = torch.from_numpy(np.array(wa.start_vector, dtype=bool)).to(device)
    bad = initial_infeasibility(tables, a_start, max_tokens=req.max_tokens)
    if bad is not None:
        req.host_error = E_INFEASIBLE
        req.detail = f"min d(A_0) = {bad.min_distance} > max_tokens {bad.remaining}"
        return
    req.tables = tables
    prepare_stop_tensor(STOP_IDS, tables.distance_to_final.device)  # ruling O5: before any call
    req.is_final = np.asarray(wa.is_final, dtype=bool)
    req.A = a_start.clone()
    req.A_next = a_start.clone()
    req.emitted = torch.zeros((), dtype=torch.int64, device=device)
    req.err = torch.zeros((), dtype=torch.int64, device=device)
    req.shadow = torch.zeros(canvas_length, dtype=torch.int64, device=device)
    req.in_lang = torch.zeros((), dtype=torch.bool, device=device)
    req.map_eq = torch.zeros((), dtype=torch.bool, device=device)
    if spec.draft is not None:
        req.seed = _seed.build(spec, canvas_length, device)


def log_internal(req: CdRequest, where: str, exc: BaseException) -> None:
    """Ruling S9: an unexpected exception in the CD path ends that request loudly as ``internal``.

    The traceback is logged once per request; the error record (and the per-process error counter) follow
    through the usual path when the request's EOS canvas is committed or the request is removed.
    """
    first = req.host_error != E_INTERNAL
    req.host_error = E_INTERNAL
    req.detail = f"{where}: {type(exc).__name__}: {exc}"
    if first:
        _log.error(
            "astra_cd internal error (tag=%s) while %s; the request ends with an EOS canvas",
            req.record_tag,
            where,
            exc_info=exc,
        )
