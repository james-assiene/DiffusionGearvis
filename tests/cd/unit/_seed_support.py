"""Shared helpers for the T-0081 rows (draft-initialised constrained repair), and the API those rows fix.

API assumed by these rows (test-author, 2026-10-08; rulings SD1-SD9, to be recorded in docs/tasks/T-0081.md by the
owner of that file). Everything not named here is left to the implementer.

SD1 (pydantic, ``local_astra.contracts.cd.CdRequest``). Three optional fields: ``draft`` (standard base64, RFC 4648
    alphabet with padding, of little-endian int32 ids; non-empty), ``draft_noise`` (finite float in [0, 1)) and
    ``draft_start_step`` (int >= 0). ``draft_noise`` or ``draft_start_step`` given (any value, 0 included) without
    ``draft`` is a ``ValidationError``. ``to_xargs()`` emits no ``astra_cd_draft*`` key when ``draft`` is unset (so the
    pre-existing exact-dict rows of ``tests/unit/test_cd_contract_model.py`` are unchanged); with ``draft`` set it
    emits ``astra_cd_draft`` and, when explicitly set, ``astra_cd_draft_noise`` / ``astra_cd_draft_start_step``.
    Whether an unset noise/start is omitted or emitted as its default (0.0 / 0) is not pinned -- the plugin reads the
    same spec either way. The contract does not know CL or ``max_denoising_steps``.
SD2 (plugin parser, ``cd/xargs.parse_cd_xargs``). The keys are ``"astra_cd_" + field``. ``CdSpec`` gains
    ``draft`` (``None`` when absent, else a 1-D integer numpy array or sequence of the decoded ids), ``draft_noise``
    (``0.0`` when absent) and ``draft_start_step`` (``0`` when absent). The context-free checks run in the parser,
    in W1's "bad values" stage (after unknown key / missing mode / missing required key, before the payload cap, the
    sha and the decode).
SD3 (reason codes; new constants in ``cd/automaton.py`` and members of ``REASONS``):
    ``draft_bad_encoding``  -- not strict standard base64, or a byte length that is not a multiple of 4;
    ``draft_bad_length``    -- 0 ids (parser), or more than CL ids (admission, where CL is known);
    ``draft_id_out_of_vocab`` -- an id < 0 or >= 262 144;
    ``draft_noise_out_of_range`` -- noise < 0, >= 1, NaN or infinite;
    ``draft_start_step_out_of_range`` -- start step < 0 (parser) or >= ``max_denoising_steps`` (admission).
    ``draft_noise`` / ``draft_start_step`` without ``draft`` -> ``missing_key``. A value of the wrong JSON type
    (string noise, float start step, list draft) is refused with the field's reason or ``bad_value``.
    Every refusal ends the request loudly (EOS canvas, one error record carrying the reason), as T-0067's refusals do.
SD4 (seed semantics, observed through the stand-in ``World``): on the call where a seeded request's prefill completes
    (prefill-only path or a mixed decode call) and only then: ``canvas[slot, i] = draft[i]`` for ``i < len(draft)``
    except a noised set of exactly ``round(noise * len(draft))`` positions (drawn without replacement), which hold
    uniform ids; ``canvas[slot, i]`` for ``i >= len(draft)`` is uniform; **every** uniform id (noised positions and the
    tail) comes from the CD-owned generator (``ConstrainedDiffusionSampler._generator(device)``, seeded
    ``steps.GENERATOR_SEED``), never from stock's ``init_canvas`` values; ``draft_tokens[slot, :CL] == canvas[slot]``;
    ``step[slot] = draft_start_step``; history length 0, SC embeddings 0 and ``is_encoder_phase`` False (stock's).
    The exact sequence of generator calls is not pinned; the rows pin counts, membership, determinism given the
    generator state, dependence on it, independence of the global RNG and the distribution.
SD5 (coupling, ``cd/seed.py``): ``completing_prefill_slots(input_batch) -> np.ndarray`` -- the **slots** (not batch
    indices), in batch order, whose prefill completes in this call, by stock's own rule (all rows on the prefill-only
    path ``num_draft_tokens == 0``; the ``nlog == 0`` rows on a decode call; filtered by
    ``num_computed_prefill_tokens_np + num_scheduled_tokens >= prefill_len_np``). Pure host numpy.
SD6 (fast path): a call in which no seeded slot completes its prefill and no constrained slot decodes issues no torch
    op outside the inner call and returns the inner object (T-0067 c3 extended to the seed hook).
SD7 (no record change): ``RECORD_FIELDS`` and ``CdCanvasRecord`` keep their 14 fields; a seeded request's records
    carry nothing about the seed.
SD8 (preemption): a seeded request that comes back after preemption is ended as ``preempted`` (S10) and its resumed
    prefill completion is **not** seeded.
SD9 (step accounting): with ``draft_start_step = k0`` the record's ``steps`` = k0 + executed denoise steps, and the
    step cap binds at ``max_denoising_steps`` (at most max - k0 denoise steps).
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import importlib
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.cd.fakes.diffusion_sampler import DEVICE, PLUGIN_SRC, STOP_IDS, Snap, V, World, _upload
from tests.cd.unit._sampler_support import LIVE

DRAFT_KEYS = {
    "draft": "astra_cd_draft",
    "draft_noise": "astra_cd_draft_noise",
    "draft_start_step": "astra_cd_draft_start_step",
}
R_DRAFT_ENCODING = "draft_bad_encoding"
R_DRAFT_LENGTH = "draft_bad_length"
R_DRAFT_VOCAB = "draft_id_out_of_vocab"
R_DRAFT_NOISE = "draft_noise_out_of_range"
R_DRAFT_START = "draft_start_step_out_of_range"
SEED_REASONS = frozenset(
    {R_DRAFT_ENCODING, R_DRAFT_LENGTH, R_DRAFT_VOCAB, R_DRAFT_NOISE, R_DRAFT_START}
)

#: a full-length draft (CL 8) and a short one over the live ids; every id is a live id (so in any loop language)
DRAFT8 = (1000, 1003, 1005, 1001, 1007, 1002, 1004, 1006)
DRAFT5 = (1001, 1002, 1003, 1004, 1005)
LIVE_OR_STOP = frozenset(LIVE) | frozenset(STOP_IDS)


def b64(ids: Sequence[int]) -> str:
    """The wire format of criterion 1: standard base64 of little-endian int32 ids."""
    return base64.b64encode(np.asarray(list(ids), dtype="<i4").tobytes()).decode("ascii")


def with_seed(
    xargs: Mapping[str, Any],
    ids: Sequence[int] | None = None,
    *,
    noise: float | None = None,
    start: int | None = None,
    raw_draft: Any = None,
) -> dict[str, Any]:
    """``xargs`` plus the draft keys (``raw_draft`` overrides the encoded ``ids``; ``None`` leaves a key out)."""
    out = dict(xargs)
    if raw_draft is not None:
        out[DRAFT_KEYS["draft"]] = raw_draft
    elif ids is not None:
        out[DRAFT_KEYS["draft"]] = b64(ids)
    if noise is not None:
        out[DRAFT_KEYS["draft_noise"]] = noise
    if start is not None:
        out[DRAFT_KEYS["draft_start_step"]] = start
    return out


def seed_module() -> ModuleType:
    """``astra_vllm_plugin.cd.seed`` (new in T-0081); absent reads as an assertion, never a collection error."""
    if str(PLUGIN_SRC) not in sys.path:
        sys.path.insert(0, str(PLUGIN_SRC))
    try:
        return importlib.import_module("astra_vllm_plugin.cd.seed")
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("astra_vllm_plugin.cd"):
            raise AssertionError(
                f"T-0081 not implemented: astra_vllm_plugin.cd.seed does not exist ({exc})"
            ) from None
        raise


def spec_seed(spec: Any) -> tuple[list[int] | None, float, int]:
    """``(draft ids or None, draft_noise, draft_start_step)`` of a ``CdSpec`` (SD2); a missing field is an assertion."""
    missing = [n for n in ("draft", "draft_noise", "draft_start_step") if not hasattr(spec, n)]
    assert not missing, f"T-0081 not implemented: CdSpec has no {missing} (ruling SD2)"
    d = spec.draft
    ids = None if d is None else [int(x) for x in np.asarray(d).reshape(-1).tolist()]
    return ids, spec.draft_noise, spec.draft_start_step


def row(s: Snap, name: str, req: str) -> list[int]:
    return [int(x) for x in s.row(name, req).reshape(-1)]


def draft_row(s: Snap, req: str, CL: int) -> list[int]:
    return [int(x) for x in s.draft_tokens[s.slots[req], :CL]]


def renoised(t: int) -> bool:
    """Not a live or stop id: a uniform id (P(false negative) = 11 / 262 144 per position)."""
    return int(t) not in LIVE_OR_STOP


def cd_generator(w: World) -> torch.Generator:
    """The CD-owned generator (SD4): ``ConstrainedDiffusionSampler._generator(device)``, created lazily."""
    gen = getattr(w.sampler, "_generator", None)
    assert callable(gen), "fixture: the CD sampler has no _generator(device) (T-0068 ruling P6)"
    g = gen(DEVICE)
    assert isinstance(g, torch.Generator), type(g)
    return g


# --------------------------------------------------------------------------- mixed and chunked batches


@dataclasses.dataclass
class Row:
    """One row of a hand-built batch: a decode row (``prefill=False``) or a prefill chunk."""

    req_id: str
    prefill: bool = False
    computed: int = 0  # num_computed_prefill_tokens before this chunk
    scheduled: int | None = None  # default: the rest of the prompt
    prefill_len: int | None = None  # default: the prompt length


def batch_for_rows(w: World, rows: Sequence[Row]) -> tuple[SimpleNamespace, list[Any], bool]:
    """An ``InputBatch`` stand-in with the image's attribute names for ``rows`` (mixed, chunked or prefill-only)."""
    cfg = w.cfg
    reqs = [w.reqs[r.req_id] for r in rows]
    n = len(rows)
    slots = np.array([q.slot for q in reqs], dtype=np.int64)
    nlog = np.array(
        [0 if r.prefill else (q.valid_len or cfg.CL) for r, q in zip(rows, reqs, strict=True)],
        dtype=np.int64,
    )
    plen = np.array(
        [cfg.prompt_len if r.prefill_len is None else r.prefill_len for r in rows], dtype=np.int32
    )
    computed = np.array(
        [r.computed if r.prefill else int(plen[i]) for i, r in enumerate(rows)], dtype=np.int32
    )
    sched = np.array(
        [
            (int(plen[i]) - r.computed if r.scheduled is None else r.scheduled)
            if r.prefill
            else int(nlog[i])
            for i, r in enumerate(rows)
        ],
        dtype=np.int32,
    )
    cu = np.concatenate([[0], np.cumsum(nlog)]).astype(np.int32)
    qsl = np.concatenate([[0], np.cumsum(sched.astype(np.int64))]).astype(np.int32)
    prefill_only = int(nlog.sum()) == 0
    batch = SimpleNamespace(
        req_ids=[r.req_id for r in rows],
        num_reqs=n,
        idx_mapping=_upload(torch.from_numpy(slots.copy()), False),
        idx_mapping_np=slots,
        num_draft_tokens=int(nlog.sum()),
        cu_num_logits_np=cu,
        cu_num_logits=_upload(torch.from_numpy(cu.copy()), False),
        query_start_loc_np=qsl,
        query_start_loc=_upload(torch.from_numpy(qsl.copy()), False),
        num_scheduled_tokens=sched,
        num_computed_prefill_tokens_np=computed,
        prefill_len_np=plen,
    )
    return batch, reqs, prefill_only


def call(
    w: World,
    rows: Sequence[Row],
    logits_fn: Callable[[int, str, int], torch.Tensor] | None = None,
) -> Snap:
    """One sampler call over ``rows`` on ``w`` (CD or stock), snapshotted like ``World.decode``."""
    batch, reqs, prefill_only = batch_for_rows(w, rows)
    parts = []
    for r, q in zip(rows, reqs, strict=True):
        if not r.prefill:
            assert logits_fn is not None, "fixture: a decode row needs logits"
            parts.append(logits_fn(w.calls, r.req_id, q.valid_len or w.cfg.CL).to(torch.float32))
    logits = (torch.cat(parts, dim=0) if parts else torch.zeros(0, V)).to(DEVICE)
    st = w.ms.diffusion_states
    entry = {
        r.req_id: bool(st.is_encoder_phase[q.slot])
        for r, q in zip(rows, reqs, strict=True)
        if not r.prefill
    }
    with w.call_ctx():
        ret = w.sampler(logits, batch)
    for i, r in enumerate(rows):
        if r.prefill:
            done = int(batch.num_computed_prefill_tokens_np[i]) + int(batch.num_scheduled_tokens[i])
            if done >= int(batch.prefill_len_np[i]):
                w.reqs[r.req_id].prefilled = True
    return w._snap("prefill" if prefill_only else "decode", reqs, ret, entry)


# --------------------------------------------------------------------------- logits


def copy_model(
    w: World, *, peak: float = 12.0, floor: float = -30.0, live: Sequence[int] = LIVE
) -> Callable[[int, str, int], torch.Tensor]:
    """A model that echoes its input canvas (``draft_tokens[slot, :n]``, what the engine feeds the model): a confident
    peak on a live or stop id, a flat row over the live ids where the input holds noise. Seeded with an in-language
    draft it is confident from the first step; from uniform noise it is not."""
    known = frozenset(live) | frozenset(STOP_IDS)

    def fn(call_idx: int, req: str, n: int) -> torch.Tensor:
        slot = w.reqs[req].slot
        cur = [int(t) for t in w.runner_sampler.req_states.draft_tokens[slot, :n].cpu()]
        x = torch.full((n, V), floor, dtype=torch.float32)
        for i, t in enumerate(cur):
            if t in known:
                x[i, t] = peak
            else:
                x[i, list(live)] = 0.0
        return x

    return fn


def fixed_logits(
    rows: Sequence[Mapping[int, float]], *, floor: float = -30.0
) -> Callable[[int, str, int], torch.Tensor]:
    """The same ``[n, V]`` logits on every call (row ``i`` from ``rows[i % len(rows)]``)."""

    def fn(call_idx: int, req: str, n: int) -> torch.Tensor:
        x = torch.full((n, V), floor, dtype=torch.float32)
        for i in range(n):
            for t, v in rows[i % len(rows)].items():
                x[i, t] = v
        return x

    return fn


# --------------------------------------------------------------------------- instruments


class OpCounter:
    """Torch ops issued *outside* the delegated inner (stock) call, by function and by dispatch."""

    def __init__(self) -> None:
        self.inside = 0
        self.outside: list[str] = []
        self.returned: list[Any] = []


COUNTER = OpCounter()


def counting_sampler_cls(stock_cls: type) -> type:
    class CountingDiffusionSampler(stock_cls):  # type: ignore[misc,valid-type]
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            COUNTER.inside += 1
            try:
                out = super().__call__(*args, **kwargs)
            finally:
                COUNTER.inside -= 1
            COUNTER.returned.append(out)
            return out

    return CountingDiffusionSampler


class FnMode(torch.overrides.TorchFunctionMode):
    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if not COUNTER.inside:
            COUNTER.outside.append(f"fn:{getattr(func, '__name__', func)}")
        return func(*args, **(kwargs or {}))


class DispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        if not COUNTER.inside:
            COUNTER.outside.append(f"aten:{func}")
        return func(*args, **(kwargs or {}))


@contextlib.contextmanager
def count_ops() -> Any:
    COUNTER.outside.clear()
    COUNTER.returned.clear()
    with DispatchMode(), FnMode():
        yield COUNTER


def _cd_frame() -> str | None:
    for f in reversed(traceback.extract_stack()):
        if "/astra_vllm_plugin/cd/" in f.filename.replace("\\", "/"):
            return f"{f.filename.split('/')[-1]}:{f.lineno} {f.name}"
    return None


class HostWrites(torch.overrides.TorchFunctionMode):
    """T-0073 D6 / T-0069 O5 on CPU: inside a sampler call, a CD frame must not (a) write a Python scalar into a
    tensor (``t[i] = True``: an H2D sync on CUDA) or (b) build a tensor from host data (``torch.tensor(list)``,
    ``torch.as_tensor(ndarray)``, ``torch.from_numpy``)."""

    def __init__(self) -> None:
        super().__init__()
        self.hits: list[str] = []

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        if (
            func is torch.Tensor.__setitem__
            and len(args) == 3
            and not isinstance(args[2], torch.Tensor)
        ):
            where = _cd_frame()
            if where:
                self.hits.append(f"__setitem__({type(args[2]).__name__}) at {where}")
        elif (
            func in (torch.tensor, torch.as_tensor, torch.from_numpy)
            and args
            and not isinstance(args[0], torch.Tensor)
        ):
            where = _cd_frame()
            if where:
                self.hits.append(
                    f"{getattr(func, '__name__', func)}({type(args[0]).__name__}) at {where}"
                )
        return func(*args, **kwargs)


# --------------------------------------------------------------------------- not-yet-implemented as assertions


def parse_ok(xargs: Mapping[str, Any]) -> Any:
    """``parse_cd_xargs`` on a request that must be accepted; a refusal is an assertion naming its reason."""
    from tests.cd import _wire_support as ws

    err_type = ws.plugin("xargs").CdRequestError
    try:
        return ws.parse(xargs)
    except err_type as err:
        raise AssertionError(
            f"a valid seeded request was refused: {getattr(err, 'reason', None)!r} ({err})"
        ) from None


def cd_request(**fields: Any) -> Any:
    """``CdRequest(**fields)`` for a request that must validate; a ``ValidationError`` is an assertion."""
    from pydantic import ValidationError

    from tests.cd import _wire_support as ws

    try:
        return ws.contracts_cd().CdRequest(**fields)
    except ValidationError as exc:
        raise AssertionError(
            f"T-0081 not implemented: CdRequest refuses a valid seeded request ({exc.errors()[:2]})"
        ) from None


def request_xargs(payload: str, **fields: Any) -> dict[str, Any]:
    """``tests/cd/_wire_support.request_xargs`` (``CdRequest.to_xargs()`` through JSON) via ``cd_request``."""
    import json

    from tests.cd import _wire_support as ws

    fields.setdefault("mode", "perstep_mar")
    fields.setdefault("self_conditioning", "model")
    fields.setdefault("record_tag", "t0081-rt")
    req = cd_request(automaton=payload, automaton_sha256=ws.sha(payload), **fields)
    out: dict[str, Any] = json.loads(json.dumps(req.to_xargs()))
    return out
