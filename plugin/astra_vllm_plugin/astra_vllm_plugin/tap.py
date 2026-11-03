"""Per-step canvas tap (T-0011; docs/12 §2, docs/07 §5 milestone 0).

What it publishes: one ``astra.canvas.step`` frame per request per publishing call. One wire
shape per phase (R10): a ``denoise`` frame carries the post-renoise ``canvas``, the ``argmax``
canvas and the exact ``stable`` mask (bit p set iff every row of ``accepted_canvas_history`` agrees
with row 0 at p and the history is full); a ``commit`` frame carries the ``committed_tokens`` and
the step the canvas had reached before the commit reset it (stock resets ``step`` to 0), and none
of the rows. ``accepted`` and ``entropy`` are never sent: v0.29.0 keeps
both as locals of its compiled step, and nothing stands in for them (criterion 6).

How, without a host sync on the sampler thread (criteria 3 and 4):

* everything the tap decides on the sampler thread is host data -- the batch's numpy mirrors
  (``idx_mapping_np``, ``cu_num_logits_np``), the slot table and the rate limiter -- and the rate
  limit runs before any device tensor is touched;
* ``SamplerOutput.num_sampled`` is a device tensor (R4), so whether a call committed is unknown on
  the host. A tap-owned device *latch* keeps, per slot, a commit counter and the last committed
  row (length, step, ids); every decode call updates it with fixed-size device ops, suppressed
  calls included. A publishing call packs the slot's rows, the reduced ``stable`` mask and the
  latch into one int32 row, copies it ``non_blocking`` into a pinned host buffer from a pool
  allocated when the sampler was wrapped, and records an event after the copy;
* the consumer (``drain``, run by the publisher thread) waits on that event, reads the buffer and
  turns counter movements into commit frames. A commit is therefore published exactly once, at the
  request's next publishing call or at its release (S11), whichever comes first; two commits
  between publishing calls cost the first one, counted in ``commit_overrun`` (never silent).

Identity (criterion 9): a staged bundle carries the request id, slot and slot epoch fixed when it
was staged; the consumer never maps a slot back to a request. ``canvas_idx`` is the occupant's
commit counter (carried in the bundle) plus the request's base, which continues across a
preemption from a bounded tombstone table.

Queue (R11): a full queue drops its oldest *step* bundle; release bundles (the S11 flush) are
dropped only when the queue holds nothing else -- an incoming step bundle is then dropped instead,
and an incoming release drops the oldest release and counts in ``commit_overrun`` too. Each release
of a live (or latched-off) request logs one INFO line with its ``req_id`` and the tap's counters
under the ``vllm`` logger hierarchy, so it lands in the engine log (R12).

Failure (criterion 5b): nothing here raises into the engine loop. Every raise is counted in
``errors`` and logged with one traceback per exception type per process; a raise while staging
returns the buffer to the pool and latches the call's slots off until their next occupant.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import msgpack
import numpy as np

log = logging.getLogger("astra_vllm_plugin.tap")
#: R12/R14: the per-release counter line goes through vLLM's own logger hierarchy, whose handler
#: writes the engine log (a plugin logger outside ``vllm`` has no handler in the engine process)
release_log = logging.getLogger("vllm.astra_vllm_plugin.tap")
RELEASE_LINE = "canvas tap release:"
COUNTERS = ("submitted", "published", "dropped", "errors", "commit_overrun")

DEFAULT_HZ = 30.0
DEFAULT_ENDPOINT = "tcp://127.0.0.1:8002"
DEFAULT_QUEUE = 64
DEFAULT_HWM = 64
PHASES = ("denoise", "commit")
WARMUP_PREFIX = "_warmup_"
#: request ids remembered after release so a preempted request continues its canvas_idx (S10)
TOMBSTONES = 4096

_LOOPBACK = re.compile(r"^tcp://127\.0\.0\.1:\d+$")
_OFF = frozenset({"0", "false", "off", "no"})

# packed row layout (int32): [step, num_sampled_now, latch..., canvas, argmax, stable]
# latch: [commit_count, commit_len, commit_step, commit_row(CL)]
_STEP, _NS, _COUNT, _LEN, _LSTEP = 0, 1, 2, 3, 4
_HEAD = 5
_LATCH_HEAD = 3


# --------------------------------------------------------------------------- errors


_logged_types: set[type] = set()
_logged_lock = threading.Lock()


def _log_once(exc: BaseException, where: str) -> None:
    """One traceback per exception type per process; the counter counts every raise."""
    try:
        with _logged_lock:
            first = type(exc) not in _logged_types
            _logged_types.add(type(exc))
        if first:
            log.error(
                "canvas tap: %s raised %s (counted in errors; logged once per type)",
                where,
                type(exc).__name__,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
    except Exception:  # logging must not be the thing that breaks the engine loop
        pass


# --------------------------------------------------------------------------- pure encoding


def stable_from_history(history: Any, history_len: int) -> list[bool]:
    """Criterion 7 on the host: bit p iff ``history[h][p] == history[0][p]`` for all h < ST and
    ``history_len >= ST``. The device path reduces the same way (``CanvasTap``)."""
    h = np.asarray(history)
    if h.ndim != 2:
        raise ValueError("history must be [ST, CL]")
    st = h.shape[0]
    if st == 0 or int(history_len) < st:
        return [False] * h.shape[1]
    return [bool(b) for b in np.all(h == h[0:1], axis=0)]


def stable_mask_hex(bits: Sequence[bool] | np.ndarray) -> str:
    """Bitmask, MSB first, one bit per canvas position, zero-padded to whole bytes."""
    arr = np.asarray(bits, dtype=bool)
    if arr.ndim != 1:
        raise ValueError("the stable mask must be 1-D")
    if arr.size == 0:
        return ""
    return np.packbits(arr, bitorder="big").tobytes().hex()


def encode_step(
    *,
    request_id: str,
    canvas_idx: int,
    step: int,
    max_steps: int,
    phase: str,
    canvas: Sequence[int] | np.ndarray | None = None,
    argmax: Sequence[int] | np.ndarray | None = None,
    stable: Sequence[bool] | np.ndarray | None = None,
    committed_tokens: Sequence[int] | np.ndarray = (),
    segment: str | None = None,
    canvas_length: int | None = None,
) -> dict[str, Any]:
    """The wire dict of one ``astra.canvas.step`` (``local_astra.contracts.protocol``).

    R10: a ``denoise`` frame carries all of ``canvas``, ``argmax`` and ``stable``; a ``commit``
    frame carries non-empty ``committed_tokens``, none of the rows, and ``canvas_length``.
    ``accepted`` and ``entropy`` are never written.
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, not {phase!r}")
    rows = {"canvas": canvas, "argmax": argmax, "stable": stable}
    given = {k: v for k, v in rows.items() if v is not None}
    if given and len(given) != len(rows):
        raise ValueError("canvas, argmax and stable are sent together or not at all")
    lengths = {len(v) for v in given.values()}
    if len(lengths) > 1:
        raise ValueError("canvas, argmax and stable must have the same length")
    n = lengths.pop() if lengths else canvas_length
    if n is None or (canvas_length is not None and canvas_length != n) or n < 1:
        raise ValueError("canvas_length is missing or disagrees with the rows")
    committed = [int(t) for t in committed_tokens]
    if phase == "denoise" and (committed or not given):
        raise ValueError("a denoise frame carries the three rows and no committed_tokens")
    if phase == "commit" and (given or not committed):
        raise ValueError("a commit frame carries committed_tokens and none of the rows")
    # R14/R15: a denoise frame's post-call step may reach the cap; a commit's pre-reset step may be 0
    # (ruling O6: a CD-ended slot scheduled again commits its EOS canvas before any denoise step)
    if not 0 <= int(step) <= int(max_steps):
        raise ValueError(f"a {phase} frame's step must be in 0..max_steps, not {step}")
    if len(committed) > n:
        raise ValueError("committed_tokens cannot exceed the canvas length")
    msg: dict[str, Any] = {
        "type": "astra.canvas.step",
        "request_id": str(request_id),
        "canvas_idx": int(canvas_idx),
        "step": int(step),
        "max_steps": int(max_steps),
        "canvas_length": int(n),
        "phase": phase,
    }
    if canvas is not None and argmax is not None and stable is not None:
        msg["canvas"] = [int(t) for t in canvas]
        msg["argmax"] = [int(t) for t in argmax]
        msg["stable"] = stable_mask_hex(stable)
    if phase == "commit":
        msg["committed_tokens"] = committed
    if segment is not None:
        msg["segment"] = segment
    return msg


# --------------------------------------------------------------------------- rate limiter


class RateLimiter:
    """At most ``hz`` passes per key (per request) on the injected monotonic clock."""

    def __init__(self, hz: float = DEFAULT_HZ, clock: Callable[[], float] = time.monotonic) -> None:
        if hz <= 0:
            raise ValueError("hz must be positive")
        self.hz = float(hz)
        self.clock = clock
        self._period = 1.0 / self.hz
        self._last: dict[str, float] = {}

    def allow(self, key: str, *, force: bool = False) -> bool:
        now = self.clock()
        last = self._last.get(key)
        if not force and last is not None and now - last < self._period:
            return False
        self._last[key] = now
        return True

    def forget(self, key: str) -> None:
        self._last.pop(key, None)


# --------------------------------------------------------------------------- slot table


@dataclass(frozen=True)
class SlotInfo:
    """One slot's current occupant, as of the moment it was read."""

    request_id: str | None
    enabled: bool = True
    canvas_idx: int = 0
    epoch: int = 0
    tripped: bool = False

    @property
    def live(self) -> bool:
        """Publishes: a real request (not a warm-up or dummy run), opted in, not latched off."""
        return (
            self.request_id is not None
            and not self.request_id.startswith(WARMUP_PREFIX)
            and self.enabled
            and not self.tripped
        )


def tap_enabled(extra_args: Any) -> bool:
    """``vllm_xargs={"astra_tap": false}`` opts a request out; anything malformed means on."""
    if not isinstance(extra_args, Mapping):
        return True
    v = extra_args.get("astra_tap", True)
    if v is False or (isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0):
        return False
    return not (isinstance(v, str) and v.strip().lower() in _OFF)


class SlotTable:
    """slot -> occupant. ``req_id`` arrives from the model state, options from the sampler's
    ``add_request``; either may come first. A release moves the slot's epoch on."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots: dict[int, SlotInfo] = {}
        self._by_req: dict[str, int] = {}
        self._epochs: dict[int, int] = {}

    def _get(self, slot: int) -> SlotInfo:
        return self._slots.get(slot) or SlotInfo(None, epoch=self._epochs.get(slot, 0))

    def bind(self, slot: int, request_id: str) -> SlotInfo:
        slot, request_id = int(slot), str(request_id)
        with self._lock:
            cur = self._get(slot)
            if cur.request_id is not None and cur.request_id != request_id:
                self._by_req.pop(cur.request_id, None)
                self._epochs[slot] = cur.epoch + 1
                cur = SlotInfo(None, epoch=self._epochs[slot])
            info = replace(cur, request_id=request_id, canvas_idx=0, tripped=False)
            self._slots[slot] = info
            self._by_req[request_id] = slot
            return info

    def options(self, slot: int, extra_args: Any) -> None:
        slot = int(slot)
        enabled = tap_enabled(extra_args)
        with self._lock:
            self._slots[slot] = replace(self._get(slot), enabled=enabled)

    def release(self, request_id: str) -> tuple[int, SlotInfo] | None:
        """Free the request's slot; returns ``(slot, occupant as it was)``, or ``None`` if unknown."""
        with self._lock:
            slot = self._by_req.pop(str(request_id), None)
            if slot is None:
                return None
            old = self._get(slot)
            self._epochs[slot] = old.epoch + 1
            self._slots.pop(slot, None)
            return slot, old

    def info(self, slot: int) -> SlotInfo | None:
        with self._lock:
            return self._slots.get(int(slot))

    def epoch(self, slot: int) -> int:
        with self._lock:
            return self._get(int(slot)).epoch

    def trip(self, slot: int, epoch: int) -> None:
        """Latch the slot off until its next occupant (criterion 5b)."""
        with self._lock:
            cur = self._slots.get(slot)
            if cur is not None and cur.epoch == epoch:
                self._slots[slot] = replace(cur, tripped=True)

    def advance(self, slot: int, epoch: int, canvas_idx: int) -> None:
        """The consumer reports the occupant's current canvas index (informational)."""
        with self._lock:
            cur = self._slots.get(slot)
            if cur is not None and cur.epoch == epoch and canvas_idx > cur.canvas_idx:
                self._slots[slot] = replace(cur, canvas_idx=canvas_idx)


# --------------------------------------------------------------------------- publisher


class StepPublisher:
    """ZeroMQ PUB on loopback: one two-part message per frame (topic, msgpack payload).

    The socket is opened lazily on the first ``send``, on the thread that sends (the tap's
    consumer). Bounded ``SNDHWM``, ``LINGER 0`` and ``NOBLOCK``: a slow subscriber costs frames,
    never decode time.
    """

    TOPIC = b"astra.canvas.step"

    def __init__(
        self, endpoint: str = DEFAULT_ENDPOINT, *, hwm: int = DEFAULT_HWM, zmq_module: Any = None
    ) -> None:
        if not (_LOOPBACK.match(endpoint) or (endpoint.startswith("ipc://") and len(endpoint) > 6)):
            raise ValueError(
                f"tap endpoint must be tcp://127.0.0.1:<port> or ipc://, not {endpoint!r}"
            )
        if hwm < 1:
            raise ValueError("hwm must be bounded and positive")
        self.endpoint = endpoint
        self.hwm = int(hwm)
        self.published = 0
        self.dropped = 0
        self._zmq = zmq_module
        self._sock: Any = None

    @classmethod
    def from_env(cls) -> StepPublisher:
        return cls(
            os.environ.get("ASTRA_TAP_ENDPOINT", DEFAULT_ENDPOINT),
            hwm=int(os.environ.get("ASTRA_TAP_HWM", DEFAULT_HWM)),
        )

    def _module(self) -> Any:
        if self._zmq is None:
            import zmq

            self._zmq = zmq
        return self._zmq

    def _socket(self) -> Any:
        if self._sock is None:
            zmq = self._module()
            sock = zmq.Context.instance().socket(zmq.PUB)
            sock.setsockopt(zmq.SNDHWM, self.hwm)
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind(self.endpoint)
            self._sock = sock
        return self._sock

    def send(self, msg: Mapping[str, Any]) -> bool:
        """``False`` when the frame was dropped at the high-water mark; never blocks."""
        zmq = self._module()
        payload = msgpack.packb(dict(msg), use_bin_type=True)
        try:
            self._socket().send_multipart([self.TOPIC, payload], flags=zmq.NOBLOCK)
        except zmq.Again:
            self.dropped += 1
            return False
        self.published += 1
        return True

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None


# --------------------------------------------------------------------------- torch backend


class _HostEvent:
    """CPU stand-in for ``torch.cuda.Event``: a CPU copy is complete when ``copy_`` returns."""

    def record(self, *_a: Any, **_k: Any) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def query(self) -> bool:
        return True


class TorchTapBackend:
    """The real seam: pinned host buffers and CUDA events (``_HostEvent`` on a CPU device).

    torch is imported here, not at module scope, so the host suite imports this module without it.
    """

    def __init__(self, device: Any = "cuda") -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self._cuda = self.device.type == "cuda"

    def host_buffer(self, shape: Any, dtype: Any = None) -> Any:
        return self.torch.zeros(shape, dtype=dtype or self.torch.int32, pin_memory=self._cuda)

    def event(self) -> Any:
        # looked up on every call (not cached) so a patched torch.cuda.Event is honoured; blocking
        # so the consumer's synchronize() sleeps on the event instead of spinning a core (R13)
        return self.torch.cuda.Event(blocking=True) if self._cuda else _HostEvent()


# --------------------------------------------------------------------------- the tap


class _Bundle:
    """One pooled pinned buffer, its event and what a staged copy into it means."""

    __slots__ = ("buf", "event", "kind", "meta")

    def __init__(self, buf: Any) -> None:
        self.buf = buf
        self.event: Any = None
        self.kind = ""
        self.meta: tuple[str, int, int] = ("", 0, 0)  # request_id, slot, epoch


@dataclass
class _Plan:
    """Host-only decisions taken before delegating: which batch rows the tap cares about."""

    batch: Any
    pre_step: Any  # device copy of every slot's step before the call (a commit resets it)
    n: int
    live: list[tuple[int, int, SlotInfo]]  # (batch row, slot, occupant)
    publish: list[tuple[int, int, SlotInfo]]


@dataclass
class _Occupant:
    base: int  # the request's canvas index when this occupant started
    seen: int = 0  # commit counter value already turned into frames


class CanvasTap:
    """Sampler-thread staging (``before_call``/``after_call``/``on_bind``/``on_release``) and the
    consumer side (``drain``). The constructor starts no thread; ``start()`` does."""

    def __init__(
        self,
        publisher: Any,
        *,
        slots: SlotTable,
        backend: Any,
        hz: float = DEFAULT_HZ,
        clock: Callable[[], float] = time.monotonic,
        queue_size: int = DEFAULT_QUEUE,
    ) -> None:
        if queue_size < 1:
            raise ValueError("queue_size must be positive")
        self.publisher = publisher
        self.slots = slots
        self.backend = backend
        self.queue_size = int(queue_size)
        self.limiter = RateLimiter(hz, clock)
        self.submitted = 0
        self.published = 0
        self.dropped = 0
        self.errors = 0
        self.commit_overrun = 0
        self._lock = threading.Lock()  # queue + pool, held for O(1) work only
        self._drain_lock = threading.Lock()  # one consumer at a time
        self._queue: deque[_Bundle] = deque()
        self._free: list[_Bundle] = []
        self._states: Any = None
        self._latch: Any = None
        self._release_pad: Any = None
        self._i32: Any = None
        self._i64: Any = None
        self._cl = 0
        self._st = 0
        self._width = 0
        self._max_steps = 0
        self._occupants: OrderedDict[tuple[int, int], _Occupant] = OrderedDict()
        self._next_idx: OrderedDict[str, int] = OrderedDict()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- set-up (when custom_sampler wraps) --------------------------------------------------
    def attach(self, sampler: Any) -> None:
        """Size everything from the sampler's states and allocate the pinned pool now (criterion 4)."""
        states = sampler.diffusion_states
        torch = self.backend.torch
        r, cl = (int(x) for x in states.canvas.shape)
        self._cl = cl
        self._st = int(states.accepted_canvas_history.shape[1])
        self._max_steps = int(states.max_denoising_steps)
        self._width = _HEAD + 4 * cl
        self._i32 = torch.int32
        self._i64 = torch.int64
        self._latch = torch.zeros((r, _LATCH_HEAD + cl), dtype=self._i32, device=states.device)
        # a release row has the step-row layout, its state-row columns zero (never read)
        self._release_pad = (
            torch.zeros((2,), dtype=self._i32, device=states.device),
            torch.zeros((3 * cl,), dtype=self._i32, device=states.device),
        )
        self._free = [
            _Bundle(self.backend.host_buffer((self._width,), self._i32))
            for _ in range(self.queue_size + 2)
        ]
        self._states = states

    # ---- errors --------------------------------------------------------------------------------
    def record_error(self, exc: BaseException, where: str) -> None:
        self.errors += 1
        _log_once(exc, where)

    # ---- request bookkeeping (engine thread) ------------------------------------------------
    def on_bind(self, slot: int) -> None:
        """A new occupant starts its commit counter at zero (a device fill, not a sync)."""
        try:
            if self._latch is not None:
                self._latch[int(slot)].zero_()
        except Exception as exc:
            self.record_error(exc, "bind")

    def on_release(self, old: SlotInfo, slot: int) -> None:
        """Stage the slot's latch so a commit not yet published goes out now (S11), then log the
        release line (R12) for every real, opted-in request -- latched-off ones included."""
        try:
            if old.request_id is not None:
                self.limiter.forget(old.request_id)
            if old.live and self._latch is not None:
                lead, tail = self._release_pad
                row = self.backend.torch.cat([lead, self._latch[slot], tail])
                self._stage("release", row, (old.request_id or "", slot, old.epoch))
        except Exception as exc:
            self.record_error(exc, "release")
            self.slots.trip(slot, old.epoch)
        rid = old.request_id
        if rid is not None and not rid.startswith(WARMUP_PREFIX) and old.enabled:
            try:
                counters = " ".join(f"{k}={int(getattr(self, k))}" for k in COUNTERS)
                release_log.info("%s req_id=%s %s", RELEASE_LINE, rid, counters)
            except Exception:  # logging must not be the thing that breaks the engine loop
                pass

    # ---- the hot path (sampler thread) ----------------------------------------------------------
    def before_call(self, input_batch: Any) -> _Plan | None:
        """Host data only: which rows are live, and which of them the rate limiter lets through."""
        try:
            if self._states is None or input_batch.num_draft_tokens == 0:
                return None  # not attached, or a prefill-only call: nothing commits
            n = int(input_batch.num_reqs)
            slots_np = input_batch.idx_mapping_np[:n]
            nlog = np.diff(input_batch.cu_num_logits_np[: n + 1])
            live: list[tuple[int, int, SlotInfo]] = []
            for i in range(n):
                if nlog[i] <= 0:
                    continue  # still prefilling inside a decode batch
                slot = int(slots_np[i])
                info = self.slots.info(slot)
                if info is not None and info.live:
                    live.append((i, slot, info))
            if not live:
                return None  # dummy/profile/warm-up batches and opted-out requests
            publish = [x for x in live if self.limiter.allow(x[2].request_id or "")]
        except Exception as exc:
            self.record_error(exc, "before the sampler call")
            return None
        try:
            # the step a committing canvas reached: stock resets it inside the call. A device
            # clone (fixed size, no host read), taken after the rate limit like every device op
            pre_step = self._states.step.clone()
        except Exception as exc:
            self.record_error(exc, "before the sampler call")
            for _i, slot, info in live:  # their latch misses this call: latch them off
                self.slots.trip(slot, info.epoch)
            return None
        return _Plan(input_batch, pre_step, n, live, publish)

    def after_call(self, plan: _Plan, out: Any) -> None:
        try:
            ns = self._update_latch(plan, out)
            for i, slot, info in plan.publish:
                row = self._pack(i, slot, ns)
                self._stage("step", row, (info.request_id or "", slot, info.epoch))
        except Exception as exc:
            self.record_error(exc, "after the sampler call")
            for _i, slot, info in plan.live:
                self.slots.trip(slot, info.epoch)

    def _update_latch(self, plan: _Plan, out: Any) -> Any:
        """Fixed-size device ops over the batch: count commits, keep the last committed row."""
        torch, i32, n = self.backend.torch, self._i32, plan.n
        latch = self._latch
        idx = plan.batch.idx_mapping[:n].to(self._i64)
        ns = out.num_sampled[:n].to(i32)
        committed = ns > 0
        cur = latch.index_select(0, idx)
        step = plan.pre_step.index_select(0, idx).to(i32)
        head = torch.stack(
            [
                cur[:, 0] + committed.to(i32),
                torch.where(committed, ns, cur[:, 1]),
                torch.where(committed, step, cur[:, 2]),
            ],
            dim=1,
        )
        row = torch.where(
            committed.unsqueeze(1), out.sampled_token_ids[:n].to(i32), cur[:, _LATCH_HEAD:]
        )
        latch.index_copy_(0, idx, torch.cat([head, row], dim=1))
        return ns

    def _pack(self, i: int, slot: int, ns: Any) -> Any:
        """One request's row: the post-call state rows, the device-reduced stable mask, the latch."""
        st, i32 = self._states, self._i32
        hist = st.accepted_canvas_history[slot]
        full = st.accepted_canvas_history_len[slot : slot + 1] >= self._st
        stable = (hist == hist[0:1]).all(0) & full
        return self.backend.torch.cat(
            [
                st.step[slot : slot + 1].to(i32),
                ns[i : i + 1],
                self._latch[slot],
                st.canvas[slot].to(i32),
                st.argmax_canvas[slot].to(i32),
                stable.to(i32),
            ]
        )

    def _stage(self, kind: str, row: Any, meta: tuple[str, int, int]) -> None:
        with self._lock:
            bundle = self._free.pop() if self._free else None
        if bundle is None:  # cannot happen with the pool sized queue + 2; drop, never allocate
            self.dropped += 1
            return
        try:
            bundle.buf.copy_(row, non_blocking=True)
            if bundle.event is None:
                bundle.event = self.backend.event()
            bundle.event.record()
        except BaseException:
            with self._lock:
                self._free.append(bundle)
            raise
        bundle.kind, bundle.meta = kind, meta
        with self._lock:
            if len(self._queue) >= self.queue_size:
                victim = next((b for b in self._queue if b.kind != "release"), None)
                if victim is None and kind != "release":
                    # R14: only release flushes are queued; they stay. This step bundle goes, and
                    # the occupant's next bundle carries its latch, so no commit is lost by it
                    self._free.append(bundle)
                    self.dropped += 1
                    return
                if victim is None:
                    # R11: nothing but release flushes; the oldest goes, and its commit with it
                    victim = self._queue[0]
                    self.commit_overrun += 1
                self._queue.remove(victim)
                self._free.append(victim)
                self.dropped += 1
            self._queue.append(bundle)
            self.submitted += 1
        self._wake.set()

    # ---- the consumer --------------------------------------------------------------------------
    def drain(self) -> int:
        """Publish everything staged so far; returns the number of frames handed to the publisher.

        Waits on each bundle's event (never on the sampler thread), reads its buffer, returns the
        buffer to the pool, then sends frame by frame with a guard around each.
        """
        sent = 0
        with self._drain_lock:
            while True:
                with self._lock:
                    if not self._queue:
                        break
                    bundle = self._queue.popleft()
                # R13: identity, kind and payload are read before the buffer goes back to the
                # pool -- the sampler thread may re-stage it the moment the lock is free
                kind, meta = bundle.kind, bundle.meta
                data = None
                try:
                    bundle.event.synchronize()
                    data = np.array(bundle.buf.numpy(), dtype=np.int64)
                except Exception as exc:
                    self.record_error(exc, "reading a staged bundle")
                finally:
                    with self._lock:
                        self._free.append(bundle)
                if data is None:
                    continue
                try:
                    frames = self._frames(kind, meta, data)
                except Exception as exc:
                    self.record_error(exc, "decoding a staged bundle")
                    continue
                for frame in frames:
                    try:
                        if self.publisher.send(frame) is not False:
                            self.published += 1
                            sent += 1
                    except Exception as exc:
                        self.record_error(exc, "publishing a frame")
        return sent

    def _occupant(self, request_id: str, slot: int, epoch: int) -> _Occupant:
        key = (slot, epoch)
        occ = self._occupants.get(key)
        if occ is None:
            occ = _Occupant(base=self._next_idx.get(request_id, 0))
            self._occupants[key] = occ
            while len(self._occupants) > TOMBSTONES:
                self._occupants.popitem(last=False)
        return occ

    def _frames(
        self, kind: str, meta: tuple[str, int, int], data: np.ndarray
    ) -> list[dict[str, Any]]:
        request_id, slot, epoch = meta
        cl = self._cl
        occ = self._occupant(request_id, slot, epoch)
        count = int(data[_COUNT])
        delta = count - occ.seen
        committed = data[_HEAD : _HEAD + cl][: int(data[_LEN])]
        frames: list[dict[str, Any]] = []
        common = {"request_id": request_id, "max_steps": self._max_steps}
        if kind == "step":
            rows = {
                "canvas": data[_HEAD + cl : _HEAD + 2 * cl],
                "argmax": data[_HEAD + 2 * cl : _HEAD + 3 * cl],
                "stable": data[_HEAD + 3 * cl : _HEAD + 4 * cl].astype(bool),
            }
            if int(data[_NS]) > 0:  # this call committed: its own frame is the commit
                frames.append(self._commit(common, occ, count, max(delta, 1), data, committed))
            else:
                if delta > 0:
                    frames.append(self._commit(common, occ, count, delta, data, committed))
                frames.append(
                    encode_step(
                        **common,
                        canvas_idx=occ.base + count,
                        step=int(data[_STEP]),
                        phase="denoise",
                        **rows,
                    )
                )
        elif delta > 0:  # release: only a commit nobody has published yet
            frames.append(self._commit(common, occ, count, delta, data, committed))
        occ.seen = max(occ.seen, count)
        idx = occ.base + occ.seen
        self._next_idx[request_id] = idx
        self._next_idx.move_to_end(request_id)
        while len(self._next_idx) > TOMBSTONES:
            self._next_idx.popitem(last=False)
        self.slots.advance(slot, epoch, idx)
        return frames

    def _commit(
        self,
        common: dict[str, Any],
        occ: _Occupant,
        count: int,
        delta: int,
        data: np.ndarray,
        committed: np.ndarray,
    ) -> dict[str, Any]:
        """The commit frame, from the latch: the committed ids and the pre-reset step, none of the
        rows (R10), whether the commit was on this bundle's call or an earlier suppressed one. If
        the counter moved by more than one, the earlier commits' rows were overwritten: overrun."""
        self.commit_overrun += max(delta - 1, 0)
        return encode_step(
            **common,
            canvas_idx=occ.base + max(count - 1, 0),
            step=int(data[_LSTEP]),
            phase="commit",
            committed_tokens=committed,
            canvas_length=self._cl,
        )

    # ---- the publisher thread -----------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="astra-canvas-tap", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(0.05)
            self._wake.clear()
            try:
                self.drain()
            except Exception as exc:  # drain guards itself; this keeps the thread alive regardless
                self.record_error(exc, "the publisher thread")


def tap_from_env(slots: SlotTable) -> CanvasTap | None:
    """The production ``tap=`` factory: on unless ``ASTRA_TAP=0``; publishes on
    ``ASTRA_TAP_ENDPOINT`` at ``ASTRA_TAP_HZ``; starts its publisher thread."""
    if os.environ.get("ASTRA_TAP", "1") == "0":
        log.info("canvas tap disabled by ASTRA_TAP=0")
        return None
    tap = CanvasTap(
        StepPublisher.from_env(),
        slots=slots,
        backend=TorchTapBackend("cuda"),
        hz=float(os.environ.get("ASTRA_TAP_HZ", DEFAULT_HZ)),
        clock=time.monotonic,
        queue_size=int(os.environ.get("ASTRA_TAP_QUEUE", DEFAULT_QUEUE)),
    )
    tap.start()
    return tap
