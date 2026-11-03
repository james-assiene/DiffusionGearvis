"""Fakes for the vLLM plugin's per-step canvas tap (T-0011, tier 1a), with **no torch and no vllm**.

The fast suite has neither package (`.claude/rules/tests.md`, docs/tasks/T-0011.md "Tests to write"), so
everything the tap touches on the hot path is faked here. Tier 1b (``tests/stock/``) runs the same tap
around the *real* vendored v0.29.0 sampler on CPU torch; that is the oracle, this is the fast copy.

Device semantics (re-planned 2026-10-02, rulings R4/R5 and S8)
---------------------------------------------------------------
``SpyTensor``
    a device tensor, backed by a numpy array. It carries real values (so a copy can be checked end
    to end) and a torch-like method surface. It **records and refuses** every host synchronisation
    in criterion 3's list -- ``.item()``, ``.tolist()``, ``.numpy()``, ``np.asarray()``, ``bool()``,
    ``int()``, ``float()``, ``.nonzero()``, ``.cpu()``, iteration, ``__index__`` -- and a
    device→host ``.to``/``copy_`` that blocks. A same-device ``.to`` is not a sync (S8). Every
    hit is appended to ``SPY.hits`` *before* the ``AssertionError`` is raised, because the tap's
    catch-all (criterion 5b) swallows the raise: the log is the witness, not the exception.
    Tensors the tap only *reads* (the state rows, the sampler's outputs, the batch) are read-only:
    an in-place op or ``__setitem__`` on them, or on a view of them, is recorded in ``SPY.writes``
    and refused (criterion 5a). Results of ops are new, tap-owned, writable tensors.
``FakeTorch`` (``backend.torch``)
    the torch module as the tap sees it through the backend seam (criterion 10: "torch is imported
    only lazily or behind the backend seam"). Tensor factories, ``where``/``stack``/``cat`` and
    friends over ``SpyTensor``; names it does not model fall through to numpy's function of the
    same name (closing by delegation), and anything numpy lacks too is recorded in
    ``SPY.unmodelled`` and raised as a named ``AssertionError`` -- report it to the reviewer, the
    fake is wrong, not the implementation.
``FakeHostBuffer`` / ``FakeEvent`` / ``FakeTapBackend``
    the pinned-host-buffer + CUDA-event seam. ``copy_(src, non_blocking=True)`` snapshots the
    device values at enqueue time (stream order). A buffer's values are readable only after a
    ``FakeEvent`` *recorded after that copy* has been ``synchronize()``d -- ``wait()`` is a
    stream-side wait in torch and makes nothing host-readable, ``query()`` is a poll. A copy into
    a buffer whose previous contents were staged but not yet read, followed by a read, is a race
    (``SPY.races``): the frame would carry another step's data.
``FakeDiffusionSampler``
    mirrors v0.29.0's ``DiffusionSampler`` **attribute surface** and its output shape:
    ``SamplerOutput.sampled_token_ids`` (``[num_reqs, CL]`` int32, rows in **batch order**) and
    ``num_sampled`` (``[num_reqs]`` int32) are *device* views of buffers the sampler reuses and
    zeroes on every call (vendored ``diffusion_sampler.py`` ``_sampled``/``_num_sampled``,
    ``:371-375, 450-457, 575-577``). The engine-side state writes of a call (``step``, ``canvas``,
    ``argmax_canvas``) happen *inside* ``__call__``, so a tap that snapshots before delegating
    reads the previous step. A commit call matches stock (vendored ``diffusion_gemma_step.py``
    Phase 5, R10): it resets ``step`` and the history length to 0, re-inits the canvas with noise
    and keeps ``argmax_canvas``; the step the canvas reached is the value *before* the call.
    A ``__setitem__`` of a Python scalar into a device tensor is a host sync (the CUDA landmine:
    a host-to-device copy that ``set_sync_debug_mode("error")`` refuses).
``FakeModelStateBase``
    stands in for ``DiffusionGemmaModelState``: ``custom_sampler`` returns the 2-tuple the runner
    unpacks (``vllm/v1/worker/gpu/model_runner.py:449-452``), or ``None``, or raises.

The API these fakes drive (fixed by the test-author, ruling R1 -- the task names behaviours):

* ``astra_vllm_plugin.model_state.make_model_state_cls(base, *, tap)``: ``tap`` is a factory,
  ``tap(slots) -> CanvasTap``, called when ``custom_sampler`` wraps (so the host pool is
  allocated then); the model state owns the ``SlotTable`` as ``state.astra_slots``.
* ``astra_vllm_plugin.tap.CanvasTap(publisher, *, slots, backend, hz, clock, queue_size)`` with
  ``drain()`` (the consumer side, never started by the constructor) and the counters
  ``submitted``, ``published``, ``dropped``, ``errors``, ``commit_overrun``.
* ``backend``: ``host_buffer(shape, dtype)``, ``event()``, ``torch``.
* ``astra_vllm_plugin.tap.TorchTapBackend(device)``: the real backend (tier 1b and 2).

Provenance of the upstream facts: docs/tasks/T-0011.md and the vendored v0.29.0 source under
``tests/fixtures/vllm_v0_29_0/``. Nothing here is taken from the plugin's implementation; when the
container probe of criteria 11-12 or tier 1b contradicts this file, this file is wrong.
"""

from __future__ import annotations

import copy
import importlib
import operator
import sys
import threading
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = REPO / "plugin" / "astra_vllm_plugin"

HOST_SYNC = "host sync on the sampler thread"
DEFAULT_COMMIT = (91, 92)


# --------------------------------------------------------------------------- guarded loading


def load_plugin_module(name: str, criterion: str) -> Any:
    """Import ``astra_vllm_plugin.<name>``; a missing module is an assertion, not a collect error.

    The plugin lives outside the installed package (it is built into a wheel and installed inside
    the container), so the test suite imports it off the source tree.
    """
    if str(PLUGIN_ROOT) not in sys.path:
        sys.path.insert(0, str(PLUGIN_ROOT))
    mod = f"astra_vllm_plugin.{name}"
    try:
        return importlib.import_module(mod)
    except Exception as exc:
        raise AssertionError(
            f"T-0011 criterion {criterion}: cannot import {mod} ({type(exc).__name__}: {exc})"
        ) from None


def load_module(mod: str, criterion: str) -> Any:
    """Same, for a first-party module (``local_astra.bench.b16_tap``)."""
    try:
        return importlib.import_module(mod)
    except Exception as exc:
        raise AssertionError(
            f"T-0011 criterion {criterion}: cannot import {mod} ({type(exc).__name__}: {exc})"
        ) from None


def attr(obj: Any, name: str, criterion: str) -> Any:
    """Fetch a name the task says must exist, reporting its absence as the criterion's failure."""
    try:
        return getattr(obj, name)
    except AttributeError:
        raise AssertionError(
            f"T-0011 criterion {criterion}: {obj!r} has no attribute {name!r}"
        ) from None


def build_canvas_step(**kwargs: Any) -> Any:
    """Construct an ``AstraCanvasStep`` with the amended field set (criterion 6).

    ``local_astra.contracts.protocol`` is strict (``extra='forbid'``), so before the amendment
    lands every one of these constructions raises ``ValidationError``. Translate that into an
    assertion naming the criterion so the test reports "the contract has not been amended yet"
    instead of erroring.
    """
    protocol = importlib.import_module("local_astra.contracts.protocol")
    try:
        return protocol.AstraCanvasStep(**kwargs)
    except Exception as exc:
        raise AssertionError(
            "T-0011 criterion 6: AstraCanvasStep does not accept the amended fields "
            f"({sorted(kwargs)}): {type(exc).__name__}: {exc}"
        ) from None


def validate_canvas_step(msg: dict[str, Any]) -> Any:
    """Validate a wire dict through the real ``AstraMessageAdapter`` (criterion 6/8)."""
    protocol = importlib.import_module("local_astra.contracts.protocol")
    try:
        return protocol.AstraMessageAdapter.validate_python(msg)
    except Exception as exc:
        raise AssertionError(
            f"T-0011 criterion 6: {msg.get('type')!r} frame does not validate as the amended "
            f"AstraCanvasStep: {type(exc).__name__}: {exc}"
        ) from None


def call_new_api(fn: Any, criterion: str, /, **kwargs: Any) -> Any:
    """Call a function whose *signature* the task changes; a stale signature is the failure.

    ``ValueError`` is left alone: several tests assert the new function rejects bad input.
    """
    try:
        return fn(**kwargs)
    except TypeError as exc:
        raise AssertionError(
            f"T-0011 criterion {criterion}: {getattr(fn, '__name__', fn)} does not accept the "
            f"specified signature ({sorted(kwargs)}): {exc}"
        ) from None


# --------------------------------------------------------------------------- the spy's log


class SpyLog:
    """Everything the device fakes saw that the criteria forbid. Reset by every ``TapHarness``."""

    def __init__(self) -> None:
        self.hits: list[str] = []  # host syncs (criterion 3)
        self.writes: list[str] = []  # writes to tensors the tap only reads (criterion 5a)
        self.allocs: list[str] = []  # host tensors allocated during a call (criterion 4)
        self.races: list[str] = []  # host buffer read after being overwritten (criterion 9)
        self.unmodelled: list[str] = []  # names the fake does not model (a fake defect)
        self.work = 0  # elements produced by device ops (criterion 4: fixed per-call size)
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.hits.clear()
            self.writes.clear()
            self.allocs.clear()
            self.races.clear()
            self.unmodelled.clear()
            self.work = 0

    def _add(self, where: list[str], what: str) -> None:
        with self._lock:
            where.append(f"{what} [thread {threading.get_ident()}]")

    def violations(self) -> dict[str, list[str]]:
        out = {
            "host syncs": self.hits,
            "writes to read-only device tensors": self.writes,
            "host allocations during a call": self.allocs,
            "host buffers read after being overwritten": self.races,
            "names the fake does not model (fake defect: tell the reviewer)": self.unmodelled,
        }
        return {k: list(v) for k, v in out.items() if v}


SPY = SpyLog()


class FaultSwitch:
    """Shared by a group of fake tensors/objects: while ``exc`` is set, tap-side access raises it."""

    def __init__(self) -> None:
        self.exc: type[BaseException] | None = None

    def check(self) -> None:
        if self.exc is not None:
            raise self.exc("injected fault (tests/fakes/diffusion_sampler.py)")


# --------------------------------------------------------------------------- device tensors


class FakeDevice(str):
    """``torch.device`` stand-in: compares equal to its string, has ``.type`` and ``.index``."""

    @property
    def type(self) -> str:
        return self.split(":")[0]

    @property
    def index(self) -> int | None:
        parts = self.split(":")
        return int(parts[1]) if len(parts) > 1 else None


CUDA = FakeDevice("cuda:0")
CPU = FakeDevice("cpu")

_DTYPES = {
    "int64": np.dtype(np.int64),
    "long": np.dtype(np.int64),
    "int32": np.dtype(np.int32),
    "int": np.dtype(np.int32),
    "int16": np.dtype(np.int16),
    "short": np.dtype(np.int16),
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
    "bool": np.dtype(np.bool_),
    "float32": np.dtype(np.float32),
    "float": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
    "double": np.dtype(np.float64),
    "float16": np.dtype(np.float16),
    "half": np.dtype(np.float16),
    "bfloat16": np.dtype(np.float32),  # numpy has none; the tap ships ids and masks, not bf16
}


def _dt(d: Any) -> np.dtype[Any] | None:
    if d is None:
        return None
    if isinstance(d, np.dtype):
        return d
    if isinstance(d, str):
        return _DTYPES.get(d.replace("torch.", ""), np.dtype(d))
    if d is bool:
        return np.dtype(np.bool_)
    if d is int:
        return np.dtype(np.int64)
    if d is float:
        return np.dtype(np.float32)
    return np.dtype(d)


def _dev(d: Any) -> FakeDevice | None:
    if d is None:
        return None
    if isinstance(d, FakeDevice):
        return d
    if isinstance(d, int):
        return FakeDevice(f"cuda:{d}")
    s = str(d)
    if s == "cuda":
        return CUDA
    return FakeDevice(s)


def _is_host(d: Any) -> bool:
    return d is not None and str(d).startswith("cpu")


def _shape_arg(args: Sequence[Any]) -> tuple[int, ...]:
    if len(args) == 1 and isinstance(args[0], (tuple, list)):
        return tuple(int(x) for x in args[0])
    return tuple(int(x) for x in args)


class _MaxResult(NamedTuple):
    values: Any
    indices: Any


class SpyTensor:
    """A device tensor that records and refuses host syncs and writes to what the tap only reads."""

    __array_priority__ = 1000  # numpy scalars on the left defer to our reflected operators

    def __init__(
        self,
        data: Any,
        *,
        dtype: Any = None,
        name: str = "t",
        readonly: bool = False,
        device: Any = CUDA,
        fault: FaultSwitch | None = None,
    ) -> None:
        if isinstance(data, np.ndarray) and dtype is None:
            arr = data
        else:
            arr = np.array(data, dtype=_dt(dtype) if dtype is not None else None)
            if dtype is None and arr.dtype == np.float64 and _all_ints(data):
                arr = arr.astype(np.int64)
        if readonly and arr.flags.writeable:
            arr.flags.writeable = False
        self._arr = arr
        self.name = name
        self._readonly = readonly
        self._device = _dev(device) or CUDA
        self._fault = fault
        #: appended to by ``FakeHostBuffer.copy_`` -- (dst shape, non_blocking)
        self.copied_by: list[tuple[str, bool]] = []

    # -- plumbing ---------------------------------------------------------------------------
    @property
    def _on_device(self) -> bool:
        return not _is_host(self._device)

    def _check(self) -> None:
        if self._fault is not None:
            self._fault.check()

    def _derive(self, arr: Any, name: str, *, view: bool = False) -> SpyTensor:
        a = np.asarray(arr)
        SPY.work += int(a.size)
        return SpyTensor(
            a,
            name=name,
            readonly=self._readonly and view,
            device=self._device,
            fault=self._fault,
        )

    def _sync(self, what: str) -> Any:
        SPY._add(SPY.hits, f"{what} on {self.name}")
        raise AssertionError(f"{HOST_SYNC}: {what} on {self.name}")

    def _refuse_write(self, op: str) -> None:
        self._check()
        if self._readonly:
            SPY._add(SPY.writes, f"{op} on {self.name}")
            raise AssertionError(
                f"the tap wrote to a device tensor it only reads: {op} on {self.name}"
            )

    def _poke(self, idx: Any, value: Any) -> None:
        """The engine's own write (the sampler mutating its state); bypasses faults and guards."""
        base = self._arr.base if self._arr.base is not None else self._arr
        was = base.flags.writeable
        base.flags.writeable = True
        try:
            self._arr[idx] = np.asarray(_raw(value))
        finally:
            base.flags.writeable = was

    def _view(self, idx: Any, name: str) -> SpyTensor:
        """An engine-made view (e.g. ``self._sampled[:n]``): shares memory, read-only, same fault."""
        return SpyTensor(
            self._arr[idx], name=name, readonly=True, device=self._device, fault=self._fault
        )

    def _same_device(self, other: Any, op: str) -> None:
        if (
            isinstance(other, SpyTensor)
            and other._arr.ndim > 0
            and self._arr.ndim > 0
            and _is_host(other._device) != _is_host(self._device)
        ):
            raise AssertionError(
                f"fake torch: {op} mixes {self._device} and {other._device} tensors "
                "(torch raises 'Expected all tensors to be on the same device')"
            )

    # -- host metadata: free, no sync ------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._arr.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._arr.dtype

    @property
    def device(self) -> FakeDevice:
        return self._device

    @property
    def is_cuda(self) -> bool:
        return self._on_device

    @property
    def ndim(self) -> int:
        return int(self._arr.ndim)

    @property
    def T(self) -> SpyTensor:
        self._check()
        return self._derive(self._arr.T, f"{self.name}.T", view=True)

    def dim(self) -> int:
        return int(self._arr.ndim)

    def size(self, dim: int | None = None) -> Any:
        return self.shape if dim is None else self.shape[dim]

    def numel(self) -> int:
        return int(self._arr.size)

    nelement = numel

    def element_size(self) -> int:
        return int(self._arr.itemsize)

    def is_pinned(self) -> bool:
        return False

    def is_contiguous(self) -> bool:
        return True

    def data_ptr(self) -> int:
        return int(self._arr.__array_interface__["data"][0])

    def record_stream(self, *_a: Any, **_k: Any) -> None:
        return None

    def __len__(self) -> int:
        return int(self._arr.shape[0]) if self._arr.ndim else 0

    def __repr__(self) -> str:
        return f"SpyTensor({self.name}, shape={self.shape}, dtype={self.dtype}, {self._device})"

    __hash__ = object.__hash__

    # -- host synchronisation: forbidden ----------------------------------------------------
    def item(self) -> Any:
        if not self._on_device:
            return self._arr.item()
        return self._sync(".item()")

    def tolist(self) -> Any:
        if not self._on_device:
            return self._arr.tolist()
        return self._sync(".tolist()")

    def numpy(self, *_a: Any, **_k: Any) -> Any:
        if not self._on_device:
            return self._arr
        return self._sync(".numpy()")

    def cpu(self) -> Any:
        if not self._on_device:
            return self
        return self._sync(".cpu()")

    def nonzero(self, *_a: Any, **_k: Any) -> Any:
        if not self._on_device:
            return SpyTensor(np.argwhere(self._arr), device=CPU, name=f"nonzero({self.name})")
        return self._sync(".nonzero()")

    def __float__(self) -> float:
        if not self._on_device:
            return float(self._arr)
        return self._sync("float()")

    def __int__(self) -> int:
        if not self._on_device:
            return int(self._arr)
        return self._sync("int()")

    def __index__(self) -> int:
        if not self._on_device:
            return int(self._arr)
        return self._sync("__index__")

    def __bool__(self) -> bool:
        if not self._on_device:
            return bool(self._arr)
        return self._sync("bool()")

    def __array__(self, *_a: Any, **_k: Any) -> Any:
        if not self._on_device:
            return self._arr
        return self._sync("np.asarray()")

    def __iter__(self) -> Any:
        if not self._on_device:
            return iter([self[i] for i in range(len(self))])
        return self._sync("iteration")

    def __contains__(self, _x: Any) -> bool:
        return self._sync("`in`")

    # -- moves -------------------------------------------------------------------------------
    def to(self, *args: Any, **kwargs: Any) -> Any:
        self._check()
        device = _dev(kwargs.get("device"))
        dtype = _dt(kwargs.get("dtype"))
        non_blocking = bool(kwargs.get("non_blocking", False))
        for a in args:
            if isinstance(a, bool):  # before int: bool is an int subclass
                non_blocking = a
            elif isinstance(a, SpyTensor):
                device, dtype = a.device, a.dtype
            elif isinstance(a, np.dtype) or (
                isinstance(a, str) and a.replace("torch.", "") in _DTYPES
            ):
                dtype = _dt(a)
            elif isinstance(a, (str, FakeDevice, int)):
                device = _dev(a)
            elif isinstance(a, type):
                dtype = _dt(a)
        if device is not None and _is_host(device) and self._on_device:
            if not non_blocking:
                return self._sync(".to(cpu) (blocking device->host copy)")
            SPY._add(SPY.allocs, f".to(cpu, non_blocking=True) on {self.name}")
            raise AssertionError(
                f"the tap allocated a host tensor during a call: .to(cpu) on {self.name}; "
                "copy into a pre-allocated pinned buffer instead (criterion 4)"
            )
        if dtype is None or dtype == self._arr.dtype:
            return self
        return self._derive(self._arr.astype(dtype), f"{self.name}.to({dtype})")

    def cuda(self, *_a: Any, **_k: Any) -> SpyTensor:
        return self

    def type(self, dtype: Any = None) -> Any:
        if dtype is None:
            return str(self.dtype)
        return self.to(dtype)

    def type_as(self, other: SpyTensor) -> SpyTensor:
        return self.to(other.dtype)

    def int(self) -> SpyTensor:
        return self.to(np.dtype(np.int32))

    def long(self) -> SpyTensor:
        return self.to(np.dtype(np.int64))

    def short(self) -> SpyTensor:
        return self.to(np.dtype(np.int16))

    def byte(self) -> SpyTensor:
        return self.to(np.dtype(np.uint8))

    def bool(self) -> SpyTensor:
        return self.to(np.dtype(np.bool_))

    def float(self) -> SpyTensor:
        return self.to(np.dtype(np.float32))

    def double(self) -> SpyTensor:
        return self.to(np.dtype(np.float64))

    # -- indexing ----------------------------------------------------------------------------
    @staticmethod
    def _index(idx: Any) -> Any:
        def one(i: Any) -> Any:
            if isinstance(i, SpyTensor):
                i._check()
                return i._arr
            if isinstance(i, FakeHostBuffer):
                return i._read("index")
            return i

        idx = tuple(one(i) for i in idx) if isinstance(idx, tuple) else one(idx)
        # basic integer/slice indexing keeps a view, as torch does (``t[i]`` aliases ``t``)
        basic = (int, slice, type(None), type(Ellipsis))
        parts = idx if isinstance(idx, tuple) else (idx,)
        if all(isinstance(p, basic) and not isinstance(p, bool) for p in parts):
            if Ellipsis not in parts:
                idx = (*parts, Ellipsis)
            return idx, True
        return idx, False

    def __getitem__(self, idx: Any) -> SpyTensor:
        self._check()
        np_idx, is_view = self._index(idx)
        return self._derive(self._arr[np_idx], f"{self.name}[{idx!r}]", view=is_view)

    def __setitem__(self, idx: Any, value: Any) -> None:
        self._refuse_write("__setitem__")
        if self._on_device and isinstance(value, (bool, int, float, np.generic)):
            # the CUDA landmine (PROGRESS 2026-10-01): ``t[i] = <python scalar>`` on a CUDA tensor
            # is a host-to-device copy that synchronises (sync-debug "error" raises on it). Use a
            # device op instead (``fill_``/``zero_`` on a view, ``torch.where``, ``index_copy_``).
            self._sync("__setitem__ with a Python scalar (host-to-device sync)")
        np_idx, _ = self._index(idx)
        self._arr[np_idx] = _raw(value)

    # -- elementwise operators ----------------------------------------------------------------
    def _bin(
        self, other: Any, fn: Callable[[Any, Any], Any], sym: str, *, rev: bool = False
    ) -> SpyTensor:
        self._check()
        self._same_device(other, sym)
        o = _raw(other)
        out = fn(o, self._arr) if rev else fn(self._arr, o)
        fault = self._fault or (other._fault if isinstance(other, SpyTensor) else None)
        a = np.asarray(out)
        SPY.work += int(a.size)
        return SpyTensor(a, name=f"({self.name}{sym})", device=self._device, fault=fault)

    def __eq__(self, other: Any) -> SpyTensor:  # type: ignore[override]
        return self._bin(other, operator.eq, "==")

    def __ne__(self, other: Any) -> SpyTensor:  # type: ignore[override]
        return self._bin(other, operator.ne, "!=")

    def __lt__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.lt, "<")

    def __le__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.le, "<=")

    def __gt__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.gt, ">")

    def __ge__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.ge, ">=")

    def __and__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.and_, "&")

    __rand__ = __and__

    def __or__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.or_, "|")

    __ror__ = __or__

    def __xor__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.xor, "^")

    def __invert__(self) -> SpyTensor:
        self._check()
        return self._derive(~self._arr, f"~{self.name}")

    def __neg__(self) -> SpyTensor:
        self._check()
        return self._derive(-self._arr, f"-{self.name}")

    def __add__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.add, "+")

    def __radd__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.add, "+", rev=True)

    def __sub__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.sub, "-")

    def __rsub__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.sub, "-", rev=True)

    def __mul__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.mul, "*")

    def __rmul__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.mul, "*", rev=True)

    def __truediv__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.truediv, "/")

    def __floordiv__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.floordiv, "//")

    def __mod__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.mod, "%")

    def __pow__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.pow, "**")

    def __lshift__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.lshift, "<<")

    def __rshift__(self, other: Any) -> SpyTensor:
        return self._bin(other, operator.rshift, ">>")

    # in-place operators: Python falls back to the binary ones when these are absent, which
    # would *rebind* instead of writing -- so they are defined, and guarded
    def _inplace(self, other: Any, fn: Callable[[Any, Any], Any], op: str) -> SpyTensor:
        self._refuse_write(op)
        self._arr[...] = fn(self._arr, _raw(other))
        return self

    def __iadd__(self, other: Any) -> SpyTensor:
        return self._inplace(other, operator.add, "+=")

    def __isub__(self, other: Any) -> SpyTensor:
        return self._inplace(other, operator.sub, "-=")

    def __imul__(self, other: Any) -> SpyTensor:
        return self._inplace(other, operator.mul, "*=")

    def __iand__(self, other: Any) -> SpyTensor:
        return self._inplace(other, operator.and_, "&=")

    def __ior__(self, other: Any) -> SpyTensor:
        return self._inplace(other, operator.or_, "|=")

    def eq(self, other: Any) -> SpyTensor:
        return self == other

    def ne(self, other: Any) -> SpyTensor:
        return self != other

    def gt(self, other: Any) -> SpyTensor:
        return self > other

    def ge(self, other: Any) -> SpyTensor:
        return self >= other

    def lt(self, other: Any) -> SpyTensor:
        return self < other

    def le(self, other: Any) -> SpyTensor:
        return self <= other

    def logical_and(self, other: Any) -> SpyTensor:
        return self._bin(other, np.logical_and, "&&")

    def logical_or(self, other: Any) -> SpyTensor:
        return self._bin(other, np.logical_or, "||")

    def logical_xor(self, other: Any) -> SpyTensor:
        return self._bin(other, np.logical_xor, "^^")

    def logical_not(self) -> SpyTensor:
        self._check()
        return self._derive(np.logical_not(self._arr), f"!{self.name}")

    def bitwise_and(self, other: Any) -> SpyTensor:
        return self & other

    def bitwise_or(self, other: Any) -> SpyTensor:
        return self | other

    def bitwise_not(self) -> SpyTensor:
        return ~self

    def abs(self) -> SpyTensor:
        self._check()
        return self._derive(np.abs(self._arr), f"abs({self.name})")

    def add(self, other: Any, alpha: Any = 1) -> SpyTensor:
        return self._bin(other, lambda a, b: a + b * alpha, "+")

    def sub(self, other: Any, alpha: Any = 1) -> SpyTensor:
        return self._bin(other, lambda a, b: a - b * alpha, "-")

    def mul(self, other: Any) -> SpyTensor:
        return self * other

    def clamp(self, min: Any = None, max: Any = None) -> SpyTensor:
        self._check()
        return self._derive(np.clip(self._arr, _raw(min), _raw(max)), f"clamp({self.name})")

    def clamp_min(self, v: Any) -> SpyTensor:
        return self.clamp(min=v)

    def clamp_max(self, v: Any) -> SpyTensor:
        return self.clamp(max=v)

    def where(self, condition: Any, other: Any) -> SpyTensor:
        self._check()
        out = np.where(_raw(condition), self._arr, _raw(other))
        return self._derive(out, f"where({self.name})")

    def masked_fill(self, mask: Any, value: Any) -> SpyTensor:
        self._check()
        return self._derive(
            np.where(_raw(mask), _raw(value), self._arr), f"masked_fill({self.name})"
        )

    # -- reductions -------------------------------------------------------------------------
    def _reduce(
        self, fn: Callable[..., Any], dim: Any, keepdim: bool, label: str, **kw: Any
    ) -> SpyTensor:
        self._check()
        out = (
            fn(self._arr, axis=dim, keepdims=keepdim, **kw)
            if dim is not None
            else fn(self._arr, **kw)
        )
        return self._derive(np.asarray(out), f"{label}({self.name})")

    def all(self, dim: Any = None, keepdim: bool = False) -> SpyTensor:
        return self._reduce(np.all, dim, keepdim, "all")

    def any(self, dim: Any = None, keepdim: bool = False) -> SpyTensor:
        return self._reduce(np.any, dim, keepdim, "any")

    def sum(self, dim: Any = None, keepdim: bool = False, dtype: Any = None) -> SpyTensor:
        out = self._reduce(np.sum, dim, keepdim, "sum")
        if dtype is not None:
            return out.to(dtype)
        if out.dtype == np.bool_ or np.issubdtype(out.dtype, np.integer):
            return out.to(np.dtype(np.int64))
        return out

    def prod(self, dim: Any = None, keepdim: bool = False) -> SpyTensor:
        return self._reduce(np.prod, dim, keepdim, "prod")

    def count_nonzero(self, dim: Any = None) -> SpyTensor:
        self._check()
        return self._derive(np.count_nonzero(self._arr, axis=dim), f"count_nonzero({self.name})")

    def amax(self, dim: Any = None, keepdim: bool = False) -> SpyTensor:
        return self._reduce(np.amax, dim, keepdim, "amax")

    def amin(self, dim: Any = None, keepdim: bool = False) -> SpyTensor:
        return self._reduce(np.amin, dim, keepdim, "amin")

    def max(self, dim: Any = None, keepdim: bool = False) -> Any:
        if dim is None or isinstance(dim, SpyTensor):
            if isinstance(dim, SpyTensor):
                return self._bin(dim, np.maximum, "max")
            return self._reduce(np.amax, None, False, "max")
        return _MaxResult(self.amax(dim, keepdim), self.argmax(dim, keepdim))

    def min(self, dim: Any = None, keepdim: bool = False) -> Any:
        if dim is None or isinstance(dim, SpyTensor):
            if isinstance(dim, SpyTensor):
                return self._bin(dim, np.minimum, "min")
            return self._reduce(np.amin, None, False, "min")
        return _MaxResult(self.amin(dim, keepdim), self.argmin(dim, keepdim))

    def argmax(self, dim: Any = None, keepdim: bool = False) -> SpyTensor:
        return self._reduce(np.argmax, dim, keepdim, "argmax")

    def argmin(self, dim: Any = None, keepdim: bool = False) -> SpyTensor:
        return self._reduce(np.argmin, dim, keepdim, "argmin")

    def cumsum(self, dim: int, dtype: Any = None) -> SpyTensor:
        self._check()
        out = self._derive(np.cumsum(self._arr, axis=dim), f"cumsum({self.name})")
        return out.to(dtype) if dtype is not None else out

    # -- shape ops (views where torch makes views) --------------------------------------------
    def clone(self, *_a: Any, **_k: Any) -> SpyTensor:
        self._check()
        return self._derive(self._arr.copy(), f"clone({self.name})")

    def contiguous(self, *_a: Any, **_k: Any) -> SpyTensor:
        return self

    def detach(self) -> SpyTensor:
        return self

    def view(self, *shape: Any) -> SpyTensor:
        self._check()
        if len(shape) == 1 and not isinstance(shape[0], (int, tuple, list)):
            return self.to(shape[0])  # view(dtype)
        return self._derive(self._arr.reshape(_shape_arg(shape)), f"{self.name}.view", view=True)

    def reshape(self, *shape: Any) -> SpyTensor:
        self._check()
        return self._derive(self._arr.reshape(_shape_arg(shape)), f"{self.name}.reshape", view=True)

    def view_as(self, other: SpyTensor) -> SpyTensor:
        return self.view(other.shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> SpyTensor:
        self._check()
        s = list(self.shape)
        end = end_dim % len(s) if s else 0
        new = [*s[:start_dim], int(np.prod(s[start_dim : end + 1])), *s[end + 1 :]]
        return self._derive(self._arr.reshape(new), f"{self.name}.flatten", view=True)

    def unsqueeze(self, dim: int) -> SpyTensor:
        self._check()
        return self._derive(np.expand_dims(self._arr, dim), f"{self.name}.unsqueeze", view=True)

    def squeeze(self, dim: int | None = None) -> SpyTensor:
        self._check()
        if dim is None:
            return self._derive(np.squeeze(self._arr), f"{self.name}.squeeze", view=True)
        if self._arr.shape[dim] != 1:
            return self
        return self._derive(np.squeeze(self._arr, axis=dim), f"{self.name}.squeeze", view=True)

    def expand(self, *sizes: Any) -> SpyTensor:
        self._check()
        target = list(_shape_arg(sizes))
        lead = len(target) - self._arr.ndim
        for i, s in enumerate(target):
            if s == -1:
                target[i] = self._arr.shape[i - lead]
        return self._derive(np.broadcast_to(self._arr, target), f"{self.name}.expand", view=True)

    def expand_as(self, other: SpyTensor) -> SpyTensor:
        return self.expand(other.shape)

    def repeat(self, *sizes: Any) -> SpyTensor:
        self._check()
        return self._derive(np.tile(self._arr, _shape_arg(sizes)), f"{self.name}.repeat")

    def permute(self, *dims: Any) -> SpyTensor:
        self._check()
        return self._derive(
            np.transpose(self._arr, _shape_arg(dims)), f"{self.name}.permute", view=True
        )

    def transpose(self, d0: int, d1: int) -> SpyTensor:
        self._check()
        return self._derive(np.swapaxes(self._arr, d0, d1), f"{self.name}.transpose", view=True)

    def t(self) -> SpyTensor:
        return self.T

    def flip(self, dims: Any) -> SpyTensor:
        self._check()
        return self._derive(np.flip(self._arr, axis=dims), f"{self.name}.flip")

    def roll(self, shifts: Any, dims: Any = None) -> SpyTensor:
        self._check()
        return self._derive(np.roll(self._arr, shifts, axis=dims), f"{self.name}.roll")

    def narrow(self, dim: int, start: int, length: int) -> SpyTensor:
        idx = [slice(None)] * self._arr.ndim
        idx[dim] = slice(start, start + length)
        return self[tuple(idx)]

    def select(self, dim: int, index: int) -> SpyTensor:
        idx: list[Any] = [slice(None)] * self._arr.ndim
        idx[dim] = index
        return self[tuple(idx)]

    def index_select(self, dim: int, index: Any) -> SpyTensor:
        self._check()
        return self._derive(np.take(self._arr, _raw(index), axis=dim), f"{self.name}.index_select")

    def gather(self, dim: int, index: Any) -> SpyTensor:
        self._check()
        return self._derive(
            np.take_along_axis(self._arr, np.asarray(_raw(index)), axis=dim), f"{self.name}.gather"
        )

    def new_zeros(self, *size: Any, dtype: Any = None, device: Any = None, **_k: Any) -> SpyTensor:
        return FakeTorch().zeros(*size, dtype=dtype or self.dtype, device=device or self.device)

    def new_ones(self, *size: Any, dtype: Any = None, device: Any = None, **_k: Any) -> SpyTensor:
        return FakeTorch().ones(*size, dtype=dtype or self.dtype, device=device or self.device)

    def new_empty(self, *size: Any, dtype: Any = None, device: Any = None, **_k: Any) -> SpyTensor:
        return self.new_zeros(*size, dtype=dtype, device=device)

    def new_full(
        self, size: Any, fill_value: Any, *, dtype: Any = None, device: Any = None, **_k: Any
    ) -> SpyTensor:
        return FakeTorch().full(
            size, fill_value, dtype=dtype or self.dtype, device=device or self.device
        )

    # -- in-place ops: only on tap-owned tensors --------------------------------------------
    def copy_(self, src: Any, non_blocking: bool = False, **_k: Any) -> SpyTensor:
        self._refuse_write("copy_")
        if isinstance(src, FakeHostBuffer):
            self._arr[...] = src._read("copy_ into a device tensor")
        elif isinstance(src, SpyTensor) and src._on_device and not self._on_device:
            # device -> an *unpinned* host tensor: pageable memory, so even non_blocking=True
            # completes synchronously on CUDA. The tap's pinned buffers come from backend.host_buffer.
            src._sync(f"copy_ device->host into an unpinned host tensor {self.name}")
        else:
            self._arr[...] = _raw(src)
        return self

    def zero_(self) -> SpyTensor:
        self._refuse_write("zero_")
        self._arr[...] = 0
        return self

    def fill_(self, value: Any) -> SpyTensor:
        self._refuse_write("fill_")
        self._arr[...] = _raw(value)
        return self

    def add_(self, other: Any, alpha: Any = 1) -> SpyTensor:
        self._refuse_write("add_")
        self._arr[...] = self._arr + _raw(other) * alpha
        return self

    def sub_(self, other: Any, alpha: Any = 1) -> SpyTensor:
        self._refuse_write("sub_")
        self._arr[...] = self._arr - _raw(other) * alpha
        return self

    def mul_(self, other: Any) -> SpyTensor:
        self._refuse_write("mul_")
        self._arr[...] = self._arr * _raw(other)
        return self

    def clamp_(self, min: Any = None, max: Any = None) -> SpyTensor:
        self._refuse_write("clamp_")
        self._arr[...] = np.clip(self._arr, _raw(min), _raw(max))
        return self

    def masked_fill_(self, mask: Any, value: Any) -> SpyTensor:
        self._refuse_write("masked_fill_")
        self._arr[...] = np.where(_raw(mask), _raw(value), self._arr)
        return self

    def index_copy_(self, dim: int, index: Any, source: Any) -> SpyTensor:
        self._refuse_write("index_copy_")
        view = np.moveaxis(self._arr, dim, 0)
        view[np.asarray(_raw(index))] = np.moveaxis(np.asarray(_raw(source)), dim, 0)
        return self

    def index_add_(self, dim: int, index: Any, source: Any, alpha: Any = 1) -> SpyTensor:
        self._refuse_write("index_add_")
        view = np.moveaxis(self._arr, dim, 0)
        np.add.at(
            view, np.asarray(_raw(index)), np.moveaxis(np.asarray(_raw(source)), dim, 0) * alpha
        )
        return self

    def index_fill_(self, dim: int, index: Any, value: Any) -> SpyTensor:
        self._refuse_write("index_fill_")
        view = np.moveaxis(self._arr, dim, 0)
        view[np.asarray(_raw(index))] = _raw(value)
        return self

    def index_put_(self, indices: Any, values: Any, accumulate: bool = False) -> SpyTensor:
        self._refuse_write("index_put_")
        idx = tuple(np.asarray(_raw(i)) for i in indices)
        if accumulate:
            np.add.at(self._arr, idx, _raw(values))
        else:
            self._arr[idx] = _raw(values)
        return self

    def scatter_(self, dim: int, index: Any, src: Any = None, value: Any = None) -> SpyTensor:
        self._refuse_write("scatter_")
        v = _raw(src) if src is not None else _raw(value)
        np.put_along_axis(self._arr, np.asarray(_raw(index)), v, axis=dim)
        return self

    # -- anything else: delegate to numpy, or say plainly that the fake does not model it ----
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name.endswith("_"):
            return _unmodelled(f"Tensor.{name} (in-place)")
        meth = getattr(np.ndarray, name, None)
        if meth is None:
            return _unmodelled(f"Tensor.{name}")

        def call(*args: Any, **kwargs: Any) -> Any:
            self._check()
            kwargs = _np_kwargs(kwargs)
            out = getattr(self._arr, name)(*[_raw(a) for a in args], **kwargs)
            if isinstance(out, (np.ndarray, np.generic)):
                return self._derive(out, f"{self.name}.{name}")
            return out

        return call


def _all_ints(data: Any) -> bool:
    try:
        flat = np.array(data, dtype=object).ravel()
    except Exception:
        return False
    return all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in flat)


def _raw(x: Any) -> Any:
    if isinstance(x, SpyTensor):
        x._check()
        return x._arr
    if isinstance(x, FakeHostBuffer):
        return x._read("use as an operand")
    if isinstance(x, (list, tuple)) and x and any(isinstance(v, SpyTensor) for v in x):
        return [_raw(v) for v in x]
    return x


def _np_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    out = dict(kwargs)
    if "dim" in out:
        out["axis"] = out.pop("dim")
    if "keepdim" in out:
        out["keepdims"] = out.pop("keepdim")
    return out


def _unmodelled(what: str) -> Any:
    SPY._add(SPY.unmodelled, what)
    raise AssertionError(
        f"tests/fakes/diffusion_sampler.py does not model {what}; this is a defect of the fake, "
        "not of the implementation -- report it to the reviewer (the implementer may not edit tests/)"
    )


class FakeTorch:
    """The ``torch`` module as the tap sees it through ``backend.torch`` (fast suite only)."""

    def __init__(self) -> None:
        for k, v in _DTYPES.items():
            setattr(self, k, v)
        self.Tensor = SpyTensor
        self.Size = tuple

    @staticmethod
    def _make(arr: Any, device: Any, name: str) -> SpyTensor:
        a = np.asarray(arr)
        SPY.work += int(a.size)
        return SpyTensor(a, name=name, device=_dev(device) or CPU)

    def device(self, d: Any) -> FakeDevice:
        return _dev(d) or CPU

    def zeros(self, *size: Any, dtype: Any = None, device: Any = None, **_k: Any) -> SpyTensor:
        return self._make(
            np.zeros(_shape_arg(size), dtype=_dt(dtype) or np.float32), device, "zeros"
        )

    def ones(self, *size: Any, dtype: Any = None, device: Any = None, **_k: Any) -> SpyTensor:
        return self._make(np.ones(_shape_arg(size), dtype=_dt(dtype) or np.float32), device, "ones")

    def empty(self, *size: Any, dtype: Any = None, device: Any = None, **_k: Any) -> SpyTensor:
        return self.zeros(*size, dtype=dtype, device=device)

    def full(
        self, size: Any, fill_value: Any, *, dtype: Any = None, device: Any = None, **_k: Any
    ) -> SpyTensor:
        shape = _shape_arg([size]) if not isinstance(size, int) else (size,)
        v = _raw(fill_value)
        dt = _dt(dtype) or np.asarray(v).dtype
        return self._make(np.full(shape, v, dtype=dt), device, "full")

    def zeros_like(
        self, t: SpyTensor, *, dtype: Any = None, device: Any = None, **_k: Any
    ) -> SpyTensor:
        return self.zeros(t.shape, dtype=dtype or t.dtype, device=device or t.device)

    def ones_like(
        self, t: SpyTensor, *, dtype: Any = None, device: Any = None, **_k: Any
    ) -> SpyTensor:
        return self.ones(t.shape, dtype=dtype or t.dtype, device=device or t.device)

    def empty_like(
        self, t: SpyTensor, *, dtype: Any = None, device: Any = None, **_k: Any
    ) -> SpyTensor:
        return self.zeros_like(t, dtype=dtype, device=device)

    def full_like(
        self, t: SpyTensor, fill_value: Any, *, dtype: Any = None, device: Any = None, **_k: Any
    ) -> SpyTensor:
        return self.full(t.shape, fill_value, dtype=dtype or t.dtype, device=device or t.device)

    def arange(
        self,
        start: Any,
        end: Any = None,
        step: Any = 1,
        *,
        dtype: Any = None,
        device: Any = None,
        **_k: Any,
    ) -> SpyTensor:
        if end is None:
            start, end = 0, start
        return self._make(
            np.arange(start, end, step, dtype=_dt(dtype) or np.int64), device, "arange"
        )

    def tensor(self, data: Any, *, dtype: Any = None, device: Any = None, **_k: Any) -> SpyTensor:
        arr = np.array(_raw(data), dtype=_dt(dtype))
        if dtype is None and arr.dtype == np.float64 and _all_ints(data):
            arr = arr.astype(np.int64)
        return self._make(arr, device, "tensor")

    def as_tensor(self, data: Any, *, dtype: Any = None, device: Any = None) -> SpyTensor:
        if isinstance(data, SpyTensor):
            return data.to(dtype) if dtype is not None else data
        return self.tensor(data, dtype=dtype, device=device)

    def from_numpy(self, arr: Any) -> SpyTensor:
        return SpyTensor(arr, name="from_numpy", device=CPU)

    def where(self, cond: Any, a: Any = None, b: Any = None) -> SpyTensor:
        if a is None and b is None and isinstance(cond, SpyTensor) and cond._on_device:
            return cond._sync("torch.where(cond) (nonzero)")
        dev = next((x.device for x in (cond, a, b) if isinstance(x, SpyTensor)), CUDA)
        fault = next(
            (x._fault for x in (cond, a, b) if isinstance(x, SpyTensor) and x._fault), None
        )
        out = self._make(np.where(_raw(cond), _raw(a), _raw(b)), dev, "where")
        out._fault = fault
        return out

    def _seq(self, seq: Iterable[Any], fn: Callable[..., Any], dim: int, label: str) -> SpyTensor:
        items = list(seq)
        dev = next((x.device for x in items if isinstance(x, SpyTensor)), CUDA)
        return self._make(fn([np.asarray(_raw(x)) for x in items], axis=dim), dev, label)

    def stack(self, seq: Iterable[Any], dim: int = 0) -> SpyTensor:
        return self._seq(seq, np.stack, dim, "stack")

    def cat(self, seq: Iterable[Any], dim: int = 0) -> SpyTensor:
        return self._seq(seq, np.concatenate, dim, "cat")

    concat = cat

    def equal(self, a: Any, b: Any) -> Any:
        if isinstance(a, SpyTensor) and a._on_device:
            return a._sync("torch.equal")
        return bool(np.array_equal(_raw(a), _raw(b)))

    def nonzero(self, t: SpyTensor, *a: Any, **k: Any) -> Any:
        return t.nonzero(*a, **k)

    def is_nonzero(self, t: SpyTensor) -> Any:
        return t.__bool__()

    def index_select(self, t: SpyTensor, dim: int, index: Any) -> SpyTensor:
        return t.index_select(dim, index)

    def gather(self, t: SpyTensor, dim: int, index: Any) -> SpyTensor:
        return t.gather(dim, index)

    def clamp(self, t: SpyTensor, min: Any = None, max: Any = None) -> SpyTensor:
        return t.clamp(min, max)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        fn = getattr(np, name, None)
        if fn is None or not callable(fn):
            return _unmodelled(f"torch.{name}")

        def call(*args: Any, **kwargs: Any) -> Any:
            dev = next((a.device for a in args if isinstance(a, SpyTensor)), CUDA)
            kwargs = _np_kwargs(kwargs)
            kwargs.pop("device", None)
            out = fn(*[_raw(a) for a in args], **kwargs)
            if isinstance(out, (np.ndarray, np.generic)):
                return self._make(out, dev, name)
            return out

        return call


# --------------------------------------------------------------------------- the host seam


@dataclass
class CopyRecord:
    src: str
    n: int
    non_blocking: bool
    thread: int


class _Root:
    """Readiness bookkeeping shared by a host buffer and every view of it."""

    def __init__(self) -> None:
        self.gen = 0  # bumped by every copy into the buffer
        self.awaited_gen = 0  # the newest generation an awaited event covered


class FakeHostBuffer:
    """A pinned host buffer. Its values are unreadable until an event recorded after the copy is awaited."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: Any,
        backend: FakeTapBackend,
        *,
        _arr: np.ndarray | None = None,
        _root: _Root | None = None,
        _owner: FakeHostBuffer | None = None,
    ) -> None:
        self._arr = _arr if _arr is not None else np.zeros(shape, dtype=_dt(dtype) or np.int64)
        self.shape = tuple(self._arr.shape)
        self.dtype = self._arr.dtype
        self.device = CPU
        self.pinned = True
        self._backend = backend
        self._root = _root if _root is not None else _Root()
        # never `_owner or self`: that calls FakeHostBuffer.__bool__ -> item() -> _read, which raises
        # on a buffer with a copy pending and silently re-owns a view whose element 0 is zero
        self._owner = _owner if _owner is not None else self
        self.copy_threads: list[int] = []
        self.read_threads: list[int] = []

    def is_pinned(self) -> bool:
        return True

    def numel(self) -> int:
        return int(self._arr.size)

    def __len__(self) -> int:
        return int(self._arr.shape[0]) if self._arr.ndim else 0

    def size(self, dim: int | None = None) -> Any:
        return self.shape if dim is None else self.shape[dim]

    def __getitem__(self, idx: Any) -> FakeHostBuffer:
        np_idx, _ = SpyTensor._index(idx)
        view = self._arr[np_idx]
        return FakeHostBuffer(
            view.shape, view.dtype, self._backend, _arr=view, _root=self._root, _owner=self._owner
        )

    def view(self, *shape: Any) -> FakeHostBuffer:
        v = self._arr.reshape(_shape_arg(shape))
        return FakeHostBuffer(
            v.shape, v.dtype, self._backend, _arr=v, _root=self._root, _owner=self._owner
        )

    reshape = view

    def copy_(self, src: Any, non_blocking: bool = False, **kwargs: Any) -> FakeHostBuffer:
        non_blocking = bool(kwargs.get("non_blocking", non_blocking))
        if isinstance(src, SpyTensor) and src._on_device:
            src._check()
            if not non_blocking:
                src._sync("blocking copy_ device->host (pass non_blocking=True)")
            rec = CopyRecord(
                src=src.name, n=src.numel(), non_blocking=True, thread=threading.get_ident()
            )
            self._backend.copies.append(rec)
            src.copied_by.append((str(self.shape), True))
            self._arr[...] = src._arr
        elif isinstance(src, FakeHostBuffer):
            self._arr[...] = src._read("host-to-host copy")
        else:
            self._arr[...] = np.asarray(_raw(src))
        self._root.gen += 1
        self._owner.copy_threads.append(threading.get_ident())
        self._backend._mark_pending(self._root)
        return self

    def __setitem__(self, idx: Any, value: Any) -> None:
        """A plain host write (synchronous, no event needed)."""
        np_idx, _ = SpyTensor._index(idx)
        self._arr[np_idx] = np.asarray(_raw(value))

    def _read(self, what: str) -> np.ndarray:
        r = self._root
        if r.awaited_gen < r.gen:
            if r.awaited_gen > 0:
                SPY._add(
                    SPY.races,
                    f"{what}: host buffer overwritten by copy generation {r.gen} after the event "
                    f"covering generation {r.awaited_gen} was awaited",
                )
                raise AssertionError(
                    "host buffer read after a newer copy overwrote it: the frame would carry "
                    "another step's data"
                )
            SPY._add(
                SPY.races, f"{what}: host buffer read before any event covering it was awaited"
            )
            raise AssertionError(
                "host buffer read before the CUDA event recorded after its copy was awaited"
            )
        self._owner.read_threads.append(threading.get_ident())
        return self._arr

    def tolist(self) -> Any:
        return copy.deepcopy(self._read(".tolist()").tolist())

    def numpy(self, *_a: Any, **_k: Any) -> np.ndarray:
        return self._read(".numpy()").copy()

    def item(self) -> Any:
        return self._read(".item()").reshape(-1)[0].item()

    def __array__(self, *_a: Any, **_k: Any) -> np.ndarray:
        return self.numpy()

    def __int__(self) -> int:
        return int(self.item())

    def __index__(self) -> int:
        return int(self.item())

    def __float__(self) -> float:
        return float(self.item())

    def __bool__(self) -> bool:
        return bool(self.item())

    def __iter__(self) -> Any:
        return iter(self.tolist())


class FakeEvent:
    """``torch.cuda.Event`` stand-in: covers the copies staged before ``record()``."""

    def __init__(self, backend: FakeTapBackend) -> None:
        self._backend = backend
        self._covered: list[tuple[_Root, int]] = []
        self.record_threads: list[int] = []
        self.wait_threads: list[int] = []

    def record(self, *_a: Any, **_k: Any) -> None:
        b = self._backend
        if b.fail_record > 0:
            b.fail_record -= 1
            b.record_raises += 1
            raise b.fail_record_exc("injected fault: event.record()")
        self._covered = [(r, r.gen) for r in b._pending]
        b._pending = []
        self.record_threads.append(threading.get_ident())
        b.events_recorded += 1

    def synchronize(self) -> None:
        self.wait_threads.append(threading.get_ident())
        self._backend.events_awaited += 1
        for root, gen in self._covered:
            root.awaited_gen = max(root.awaited_gen, gen)

    def wait(self, *_a: Any, **_k: Any) -> None:
        """torch: make a *stream* wait for this event. The host does not wait; nothing becomes readable."""
        self._backend.stream_waits += 1

    def query(self) -> bool:
        return True

    def elapsed_time(self, _other: Any) -> float:
        return 0.0


class FakeTapBackend:
    """Injected in place of the torch/CUDA facilities so the host suite can drive the tap."""

    def __init__(self) -> None:
        self.allocations: list[tuple[tuple[int, ...], str]] = []
        self.buffers: list[FakeHostBuffer] = []
        self.events: list[FakeEvent] = []
        self.copies: list[CopyRecord] = []
        self.events_created = 0
        self.events_recorded = 0
        self.events_awaited = 0
        self.stream_waits = 0
        #: fault injection: the next ``fail_record`` calls to ``FakeEvent.record`` raise
        self.fail_record = 0
        self.fail_record_exc: type[BaseException] = RuntimeError
        self.record_raises = 0
        #: fault injection: while set, ``host_buffer`` raises it (a pinned allocation that fails
        #: while ``custom_sampler`` wraps the sampler)
        self.fail_host_buffer: BaseException | None = None
        self._pending: list[_Root] = []
        self.torch = FakeTorch()
        self.device = CUDA

    def _mark_pending(self, root: _Root) -> None:
        if root not in self._pending:
            self._pending.append(root)

    def host_buffer(self, shape: Any, dtype: Any = "int64") -> FakeHostBuffer:
        if self.fail_host_buffer is not None:
            raise self.fail_host_buffer
        shp = (int(shape),) if isinstance(shape, int) else tuple(int(s) for s in shape)
        self.allocations.append((shp, str(dtype)))
        buf = FakeHostBuffer(shp, dtype, self)
        self.buffers.append(buf)
        return buf

    def event(self) -> FakeEvent:
        self.events_created += 1
        ev = FakeEvent(self)
        self.events.append(ev)
        return ev

    # -- convenience for assertions --------------------------------------------------------
    @property
    def scalars_copied(self) -> int:
        return sum(c.n for c in self.copies)

    def reset_counts(self) -> None:
        self.copies.clear()
        self.events_recorded = 0
        self.events_awaited = 0


# --------------------------------------------------------------------------- sampler + state


class FakeDiffusionStates:
    """``DiffusionGemmaRequestStates`` stand-in (field names: vendored v0.29.0 source)."""

    def __init__(
        self,
        *,
        max_num_reqs: int = 2,
        canvas_length: int = 16,
        max_denoising_steps: int = 48,
        history_steps: int = 8,
        vocab: int = 1000,
    ) -> None:
        self.max_num_reqs = max_num_reqs
        self.canvas_length = canvas_length
        self.max_denoising_steps = max_denoising_steps
        self.stability_threshold = history_steps
        self.vocab_size = vocab
        self.device = CUDA
        self.fault_switch = FaultSwitch()
        cl, r, st = canvas_length, max_num_reqs, history_steps
        f = self.fault_switch

        def ro(data: Any, dtype: str, name: str) -> SpyTensor:
            return SpyTensor(np.array(data, dtype=_dt(dtype)), name=name, readonly=True, fault=f)

        self.canvas = ro(
            [[(i * 7 + j) % vocab for j in range(cl)] for i in range(r)], "int64", "canvas"
        )
        self.argmax_canvas = ro(
            [[(i * 7 + j) % vocab for j in range(cl)] for i in range(r)], "int64", "argmax_canvas"
        )
        self.step = ro([0] * r, "int32", "step")
        self.is_encoder_phase = ro([False] * r, "bool", "is_encoder_phase")
        self.confident = ro([False] * r, "bool", "confident")
        self.prompt_len = ro([0] * r, "int32", "prompt_len")
        self.accepted_canvas_history = ro(
            np.zeros((r, st, cl), dtype=np.int64), "int64", "accepted_canvas_history"
        )
        self.accepted_canvas_history_len = ro([0] * r, "int32", "accepted_canvas_history_len")

    def set_slot(
        self,
        slot: int,
        *,
        canvas: Sequence[int] | None = None,
        argmax: Sequence[int] | None = None,
        step: int | None = None,
        history: Sequence[Sequence[int]] | None = None,
        history_len: int | None = None,
    ) -> None:
        """The engine writing a slot's rows (bypasses the read-only guard and the fault switch)."""
        if canvas is not None:
            self.canvas._poke(slot, list(canvas))
        if argmax is not None:
            self.argmax_canvas._poke(slot, list(argmax))
        if step is not None:
            self.step._poke(slot, step)
        if history is not None:
            self.accepted_canvas_history._poke(slot, [list(row) for row in history])
        if history_len is not None:
            self.accepted_canvas_history_len._poke(slot, history_len)

    def row(self, name: str, slot: int) -> Any:
        """Test-side read of a row (outside any tap code path)."""
        return getattr(self, name)._arr[slot].tolist()


@dataclass
class FakeSamplerOutput:
    """``vllm.v1.worker.gpu.sample.output.SamplerOutput``: device tensors, rows in batch order."""

    sampled_token_ids: Any = None
    logprobs_tensors: Any = None
    num_nans: Any = None
    num_sampled: Any = None
    num_rejected: Any = None


@dataclass
class SamplingParams:
    """Only the channel the plugin is allowed to read (``vllm_xargs`` → ``extra_args``)."""

    extra_args: Any = None


@dataclass
class NewRequestData:
    req_id: str
    prompt_token_ids: list[int] = field(default_factory=lambda: [2] * 7)


class FakeInputBatch:
    """``vllm.v1.worker.gpu.input_batch.InputBatch``, the attributes the sampler path reads.

    ``num_reqs`` and ``num_draft_tokens`` are read by the (fake) inner sampler itself, so they are
    never faulted; every other attribute raises while ``fault.exc`` is set.
    """

    ENGINE_FIELDS = frozenset({"num_reqs", "num_draft_tokens"})

    def __init__(
        self,
        slots: Sequence[int],
        *,
        cl: int,
        prefill: bool = False,
        valid_lens: Sequence[int] | None = None,
        prompt_len: int = 7,
    ) -> None:
        n = len(slots)
        nlog = [0] * n if prefill else list(valid_lens or [cl] * n)
        cu = np.concatenate([[0], np.cumsum(nlog)]).astype(np.int32)
        qlens = [prompt_len] * n if prefill else nlog
        qsl = np.concatenate([[0], np.cumsum(qlens)]).astype(np.int32)
        d = self.__dict__
        d["fault"] = FaultSwitch()
        d["num_reqs"] = n
        d["num_draft_tokens"] = 0 if prefill else int(sum(nlog))
        d["idx_mapping_np"] = np.array(list(slots), dtype=np.int64)
        d["idx_mapping"] = SpyTensor(
            np.array(list(slots), dtype=np.int64), name="idx_mapping", readonly=True
        )
        d["cu_num_logits_np"] = cu
        d["cu_num_logits"] = SpyTensor(cu.copy(), name="cu_num_logits", readonly=True)
        d["query_start_loc_np"] = qsl
        d["query_start_loc"] = SpyTensor(qsl.copy(), name="query_start_loc", readonly=True)
        d["num_scheduled_tokens"] = np.array(qlens, dtype=np.int32)
        d["num_computed_prefill_tokens_np"] = np.zeros(n, dtype=np.int32)
        d["prefill_len_np"] = np.full(n, prompt_len, dtype=np.int32)
        d["req_ids"] = [f"<batch row {i}>" for i in range(n)]

    def __getattribute__(self, name: str) -> Any:
        if (
            not name.startswith("_")
            and name not in FakeInputBatch.ENGINE_FIELDS
            and name != "fault"
        ):
            object.__getattribute__(self, "__dict__")["fault"].check()
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AssertionError(
            f"the tap wrote input_batch.{name}; it only reads the batch (criterion 5a)"
        )


class FakeDiffusionSampler:
    """Attribute surface of v0.29.0's ``DiffusionSampler`` -- nothing more, nothing less."""

    instances = 0

    def __init__(self, states: FakeDiffusionStates | None = None) -> None:
        type(self).instances += 1
        self.diffusion_states = states or FakeDiffusionStates()
        st = self.diffusion_states
        self.sampling_states = object()
        self.req_states = object()
        self.logits_mode = False
        self.canvas_length = st.canvas_length
        self.vocab_size = st.vocab_size
        r, cl = st.max_num_reqs, st.canvas_length
        # the sampler's reused output buffers (vendored diffusion_sampler.py:371-375)
        self._sampled = SpyTensor(
            np.zeros((r, cl), np.int32), name="_sampled", readonly=True, fault=st.fault_switch
        )
        self._num_sampled = SpyTensor(
            np.zeros(r, np.int32), name="_num_sampled", readonly=True, fault=st.fault_switch
        )
        self.calls: list[tuple[Any, Any, Any]] = []
        self.added: list[tuple[int, int, Any]] = []
        self.probes: list[str] = []
        self.next_output: Any = None
        self.raises: BaseException | None = None
        self.last_output: Any = None
        #: engine writes applied *inside* the next ``__call__`` (after any tap pre-work)
        self.pending: list[Callable[[], None]] = []
        #: batch row -> ids committed by the next decode call
        self.commits: dict[int, Sequence[int]] = {}

    # methods DiffusionSampler really has
    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> str:
        self.added.append((req_idx, prompt_len, sampling_params))
        return "inner-add"

    def apply_staged_writes(self) -> str:
        return "inner-staged"

    @property
    def penalties_state(self) -> Any:
        return _NO_PENALTIES

    def _finish_prefills(self, *_a: Any) -> None: ...

    def _handle_prefill(self, *_a: Any) -> None: ...

    def _build_output(self, *_a: Any) -> None: ...

    def __call__(self, logits: Any, input_batch: Any, draft_logits: Any = None) -> Any:
        self.calls.append((logits, input_batch, draft_logits))
        if self.raises is not None:
            raise self.raises
        for write in self.pending:
            write()
        self.pending = []
        if self.next_output is not None:
            self.last_output = self.next_output
            return self.next_output
        st = self.diffusion_states
        n = getattr(input_batch, "num_reqs", st.max_num_reqs)
        prefill = getattr(input_batch, "num_draft_tokens", 1) == 0
        self._sampled._poke(slice(None), 0)
        self._num_sampled._poke(slice(None), 0)
        if prefill:
            sampled = self._sampled._view((slice(0, n), slice(0, 1)), "sampled_token_ids")
        else:
            for row, toks in self.commits.items():
                self._sampled._poke((row, slice(0, len(toks))), list(toks))
                self._num_sampled._poke(row, len(toks))
            sampled = self._sampled._view(slice(0, n), "sampled_token_ids")
        self.commits = {}
        num_sampled = self._num_sampled._view(slice(0, n), "num_sampled")
        out = FakeSamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_sampled,
        )
        self.last_output = out
        return out

    # every other access is recorded, then refused exactly as the real object would
    def __getattr__(self, name: str) -> Any:
        self.__dict__.setdefault("probes", []).append(name)
        raise AttributeError(name)


class _NoPenalties:
    output_bin_counts = None


_NO_PENALTIES = _NoPenalties()


class FakeModelStateBase:
    """``DiffusionGemmaModelState`` stand-in: the ``super()`` whose objects must be preserved."""

    def __init__(
        self,
        *,
        result: Any = "unset",
        raises: BaseException | None = None,
        sampler: FakeDiffusionSampler | None = None,
    ) -> None:
        self.sampler = sampler or FakeDiffusionSampler()
        self.second = object()
        self._result = (self.sampler, self.second) if result == "unset" else result
        self._raises = raises
        self.custom_sampler_calls: list[Any] = []
        self.add_request_calls: list[tuple[int, Any]] = []
        self.remove_request_calls: list[str] = []
        self.probes: list[str] = []

    def custom_sampler(self, sampler: Any) -> Any:
        self.custom_sampler_calls.append(sampler)
        if self._raises is not None:
            raise self._raises
        return self._result

    def add_request(self, req_index: int, new_req_data: Any) -> str:
        self.add_request_calls.append((req_index, new_req_data))
        return "super-add"

    def remove_request(self, req_id: str) -> str:
        self.remove_request_calls.append(req_id)
        return "super-remove"

    def __getattr__(self, name: str) -> Any:
        self.__dict__.setdefault("probes", []).append(name)
        raise AttributeError(name)


def make_two_level_base(base: type) -> type:
    """A wrapper base shaped like ``make_cd_model_state_cls(base)`` (T-0062): a model-state factory
    whose ``custom_sampler`` wraps super's sampler in its own transparent wrapper. T-0069 composes
    ``make_model_state_cls(make_cd_model_state_cls(...), tap=...)``; criterion 1 needs the tap to
    work over that shape without knowing it."""

    class LayerSampler:
        def __init__(self, inner: Any) -> None:
            self._layer_inner = inner
            self.layer_calls = 0

        def __call__(self, logits: Any, input_batch: Any, draft_logits: Any = None) -> Any:
            self.layer_calls += 1
            return self._layer_inner(logits, input_batch, draft_logits)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.__dict__["_layer_inner"], name)

    class TwoLevelModelState(base):  # type: ignore[misc,valid-type]
        layer_sampler_cls = LayerSampler

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.layer_added: list[tuple[int, str]] = []
            self.layer_removed: list[str] = []

        def custom_sampler(self, sampler: Any) -> Any:
            got = super().custom_sampler(sampler)
            if got is None:
                return None
            inner, second = got
            return LayerSampler(inner), second

        def add_request(self, req_index: int, new_req_data: Any) -> Any:
            self.layer_added.append((req_index, new_req_data.req_id))
            return super().add_request(req_index, new_req_data)

        def remove_request(self, req_id: str) -> Any:
            self.layer_removed.append(req_id)
            return super().remove_request(req_id)

    return TwoLevelModelState


# --------------------------------------------------------------------------- publisher seam


class RecordingPublisher:
    """Stands in for ``StepPublisher``: collects the wire dicts the consumer thread emits."""

    def __init__(
        self, fail_with: BaseException | None = None, *, fail_on: Iterable[int] = ()
    ) -> None:
        self.sent: list[dict[str, Any]] = []
        self.threads: list[int] = []
        self.fail_with = fail_with
        #: 1-based indices of ``send`` calls that raise (criterion 5b: one bad frame)
        self.fail_on = set(fail_on)
        self.calls = 0

    def send(self, msg: dict[str, Any]) -> bool:
        self.calls += 1
        if self.fail_with is not None and (not self.fail_on or self.calls in self.fail_on):
            raise self.fail_with
        self.sent.append(msg)
        self.threads.append(threading.get_ident())
        return True

    def close(self) -> None: ...


class BlockingPublisher(RecordingPublisher):
    """A subscriber that stops reading: ``send`` blocks until ``gate`` is set (criterion 3, 5c)."""

    def __init__(self, gate: threading.Event) -> None:
        super().__init__()
        self.gate = gate
        self.entered = threading.Event()

    def send(self, msg: dict[str, Any]) -> bool:
        self.entered.set()
        self.gate.wait()
        return super().send(msg)


class ManualClock:
    """Monotonic seconds under the test's control (no sleeping, no wall clock)."""

    def __init__(self, t0: float = 1000.0) -> None:
        self.t = t0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += dt
        return self.t


# --------------------------------------------------------------------------- harness


def reinit_noise(slot: int, cl: int, vocab: int) -> list[int]:
    """The fake's stand-in for stock's ``torch.randint`` canvas re-init at a commit: deterministic,
    and disjoint in pattern from every canvas a test writes (descending from the top of the vocab)."""
    return [(vocab - 1 - (7 * slot + 3 * j)) % vocab for j in range(cl)]


@dataclass
class Row:
    """One request's row in a decode call: what the engine writes for it during the call.

    A denoise row (``commit is None``) leaves the slot at ``step`` with ``canvas``/``argmax`` (when
    given). A commit row follows stock v0.29.0: ``step`` is the step the canvas had *reached* (the
    pre-call value, R10's "pre-reset step"); during the call the slot's ``step`` and history length
    reset to 0, the canvas is re-initialised (``reinit_noise``) and ``argmax_canvas`` is kept.
    """

    slot: int
    step: int
    commit: Sequence[int] | None = None
    canvas: Sequence[int] | None = None
    argmax: Sequence[int] | None = None
    valid_len: int | None = None


def model_state_cls(base: type, tap: Any, criterion: str) -> type:
    """``make_model_state_cls(base, *, tap=...)``; a stale signature is the criterion's failure."""
    ms = load_plugin_module("model_state", criterion)
    fn = attr(ms, "make_model_state_cls", criterion)

    def make_model_state_cls(**kwargs: Any) -> Any:
        return fn(base, **kwargs)

    return call_new_api(make_model_state_cls, criterion, tap=tap)  # type: ignore[no-any-return]


class TapFactory:
    """The ``tap=`` factory: builds a ``CanvasTap`` over the fakes and remembers every one it built."""

    def __init__(
        self,
        *,
        hz: float = 30.0,
        queue_size: int = 16,
        publisher: Any = None,
        backend: FakeTapBackend | None = None,
        clock: ManualClock | None = None,
        criterion: str = "1",
    ) -> None:
        self.tapmod = load_plugin_module("tap", criterion)
        self.criterion = criterion
        self.hz = hz
        self.queue_size = queue_size
        self.publisher = publisher or RecordingPublisher()
        self.backend = backend or FakeTapBackend()
        self.clock = clock or ManualClock()
        self.built: list[Any] = []
        self.slots_seen: list[Any] = []

    def __call__(self, slots: Any) -> Any:
        cls = attr(self.tapmod, "CanvasTap", self.criterion)
        tap = cls(
            self.publisher,
            slots=slots,
            backend=self.backend,
            hz=self.hz,
            clock=self.clock,
            queue_size=self.queue_size,
        )
        self.built.append(tap)
        self.slots_seen.append(slots)
        return tap


class TapHarness:
    """The whole stack over the fakes: ``make_model_state_cls(base, tap=factory)`` → model state →
    ``custom_sampler`` → ``TapSampler`` around ``FakeDiffusionSampler``, with assertion handles.

    Counter semantics the tests rely on (fixed by the test-author, T-0011):

    ``submitted``
        bundles staged and pushed onto the publish queue;
    ``dropped``
        frames evicted from a full queue (criterion 5c: drop the oldest, never block);
    ``published``
        frames handed to the publisher;
    ``errors``
        exceptions swallowed anywhere in the tap (criterion 5b), one per raise;
    ``commit_overrun``
        commits that could not be published because a later commit of the same slot overwrote
        them before any bundle carried them (criterion 4, never silent).
    """

    def __init__(
        self,
        *,
        hz: float = 30.0,
        queue_size: int = 16,
        canvas_length: int = 16,
        max_num_reqs: int = 2,
        max_denoising_steps: int = 48,
        history_steps: int = 8,
        publisher: Any = None,
        criterion: str = "3",
        base: type = FakeModelStateBase,
    ) -> None:
        SPY.reset()
        self.criterion = criterion
        self.states = FakeDiffusionStates(
            max_num_reqs=max_num_reqs,
            canvas_length=canvas_length,
            max_denoising_steps=max_denoising_steps,
            history_steps=history_steps,
        )
        self.sampler = FakeDiffusionSampler(self.states)
        self.factory = TapFactory(
            hz=hz, queue_size=queue_size, publisher=publisher, criterion=criterion
        )
        self.clock = self.factory.clock
        self.backend = self.factory.backend
        self.publisher = self.factory.publisher
        self.tap_module = self.factory.tapmod
        cls = model_state_cls(base, self.factory, criterion)
        self.ms = cls(sampler=self.sampler)
        got = self.ms.custom_sampler(object())
        assert isinstance(got, tuple) and len(got) == 2, (
            f"T-0011 criterion 1: custom_sampler returned {got!r}, not the 2-tuple"
        )
        self.wrapper, self.second = got
        assert len(self.factory.built) == 1, (
            f"T-0011 criterion 1: wrapping called the tap factory {len(self.factory.built)} times"
        )
        self.tap = self.factory.built[0]
        self.slots = attr(self.ms, "astra_slots", criterion)
        self.allocations_at_wrap = len(self.backend.allocations)

    # -- requests -----------------------------------------------------------------------------
    def bind(
        self,
        slot: int = 0,
        req_id: str = "req-a",
        extra_args: Any = None,
        *,
        sampler_first: bool = False,
    ) -> None:
        params = SamplingParams(extra_args=extra_args)
        if sampler_first:
            self.wrapper.add_request(slot, 7, params)
            self.ms.add_request(slot, NewRequestData(req_id=req_id))
        else:
            self.ms.add_request(slot, NewRequestData(req_id=req_id))
            self.wrapper.add_request(slot, 7, params)

    def release(self, req_id: str) -> Any:
        return self.ms.remove_request(req_id)

    # -- calls --------------------------------------------------------------------------------
    def call(self, rows: Sequence[Row], *, batch: FakeInputBatch | None = None) -> Any:
        st = self.states
        if batch is None:
            batch = FakeInputBatch(
                [r.slot for r in rows],
                cl=st.canvas_length,
                valid_lens=[r.valid_len or st.canvas_length for r in rows],
            )
        for i, r in enumerate(rows):
            if r.commit is not None:
                # stock v0.29.0 (vendored diffusion_gemma_step.py, Phase 5): a commit call does not
                # advance ``step``; the canvas *reached* ``r.step`` on the previous call, so that is
                # the row's value going in. The call then resets ``step`` and the history length to
                # 0, re-inits the canvas with noise and preserves ``argmax_canvas``.
                assert r.canvas is None and r.argmax is None, (
                    "fixture: a commit re-inits the canvas and preserves argmax; pass neither"
                )
                st.set_slot(r.slot, step=r.step)

                def write(r: Row = r) -> None:
                    st.set_slot(
                        r.slot,
                        step=0,
                        canvas=reinit_noise(r.slot, st.canvas_length, st.vocab_size),
                        history_len=0,
                    )

                self.sampler.commits[i] = list(r.commit)
            else:

                def write(r: Row = r) -> None:
                    st.set_slot(r.slot, step=r.step, canvas=r.canvas, argmax=r.argmax)

            self.sampler.pending.append(write)
        logits = SpyTensor(np.zeros((1, 1), np.float32), name="logits", readonly=True)
        return self.wrapper(logits, batch)

    def step(
        self,
        *,
        slot: int = 0,
        step: int,
        commit: bool = False,
        tokens: Sequence[int] | None = None,
        canvas: Sequence[int] | None = None,
        argmax: Sequence[int] | None = None,
    ) -> Any:
        toks = tokens if tokens is not None else (DEFAULT_COMMIT if commit else None)
        return self.call([Row(slot, step, toks, canvas, argmax)])

    def prefill(self, slots: Sequence[int]) -> Any:
        batch = FakeInputBatch(list(slots), cl=self.states.canvas_length, prefill=True)
        logits = SpyTensor(np.zeros((0, 1), np.float32), name="logits", readonly=True)
        return self.wrapper(logits, batch)

    # -- reading what came out ----------------------------------------------------------------
    def frames(self, req_id: str | None = None) -> list[dict[str, Any]]:
        sent = self.publisher.sent
        return [m for m in sent if req_id is None or m.get("request_id") == req_id]

    def commit_frames(self, req_id: str | None = None) -> list[dict[str, Any]]:
        return [m for m in self.frames(req_id) if m.get("phase") == "commit"]


#: R12: the counters the tap logs at each release, in the order the ruling names them
TAP_COUNTERS = ("submitted", "published", "dropped", "errors", "commit_overrun")


class ReleaseLogFault(RuntimeError):
    """The one injected raise of ``release_log_scenario`` (its own type: one traceback per type)."""


#: R12 scenario's counters at the release of ``req-log``, by hand: req-err's only staging raises at
#: ``event.record()`` (errors 1, nothing submitted); req-log stages 5 bundles into a queue of 2
#: (3 dropped, the oldest); the drain publishes the 2 left. ``submitted`` is 5 before the release
#: stages its flush and 6 after, so either is the logged value depending on when the line is written.
RELEASE_LOG_EXPECTED = {"published": 2, "dropped": 3, "errors": 1, "commit_overrun": 0}
RELEASE_LOG_SUBMITTED = (5, 6)


def release_log_scenario() -> TapHarness:
    """R12: one release of a live request (``req-log``) with every counter but ``commit_overrun``
    non-zero and pairwise distinct, so a line that printed the wrong counter cannot match.
    hz 1000, 10 ms apart, queue 2, n = 6 sampler calls, one drain, then the release."""
    h = TapHarness(hz=1000.0, queue_size=2)
    h.bind(1, "req-err")
    h.backend.fail_record = 1
    h.backend.fail_record_exc = ReleaseLogFault
    h.step(slot=1, step=1)
    h.clock.advance(0.01)
    h.bind(0, "req-log")
    for s in range(1, 6):
        h.step(slot=0, step=s)
        h.clock.advance(0.01)
    h.tap.drain()
    h.release("req-log")
    return h


def spy_report() -> str:
    v = SPY.violations()
    return (
        "no spy violations"
        if not v
        else "spy violations: " + "; ".join(f"{k}: {vs[:4]}" for k, vs in v.items())
    )


# --------------------------------------------------------------------------- zeromq stand-in


class FakeZmqSocket:
    def __init__(self, kind: int, ctx: FakeZmqContext) -> None:
        self.kind = kind
        self.ctx = ctx
        self.options: list[tuple[int, Any]] = []
        self.bound: list[str] = []
        self.sent: list[tuple[list[bytes], int]] = []
        self.closed = False
        self.hwm_full = False

    def setsockopt(self, opt: int, value: Any) -> None:
        self.options.append((opt, value))

    def bind(self, endpoint: str) -> None:
        self.bound.append(endpoint)

    def send_multipart(self, parts: list[bytes], flags: int = 0) -> None:
        if self.hwm_full:
            raise self.ctx.zmq.Again("high water mark reached")
        self.sent.append(([bytes(p) for p in parts], flags))

    def close(self, linger: int | None = None) -> None:
        self.closed = True

    def option(self, opt: int) -> Any:
        vals = [v for o, v in self.options if o == opt]
        return vals[-1] if vals else None


class FakeZmqContext:
    def __init__(self, zmq: FakeZmq) -> None:
        self.zmq = zmq
        self.sockets: list[FakeZmqSocket] = []

    def socket(self, kind: int) -> FakeZmqSocket:
        s = FakeZmqSocket(kind, self)
        self.sockets.append(s)
        return s

    def term(self) -> None: ...


class FakeZmq:
    """The subset of ``zmq`` the publisher uses.

    The constants are libzmq's own wire-stable values (``zmq.h``: ZMQ_PUB 1, ZMQ_SNDHWM 23,
    ZMQ_LINGER 17, ZMQ_DONTWAIT/NOBLOCK 1); ``tests/contract/test_canvas_step_wire.py`` checks
    them against the installed pyzmq when it is present, per `.claude/rules/tests.md` (a fake's
    behaviour needs a real-service check in the same PR).
    """

    PUB = 1
    SUB = 2
    SNDHWM = 23
    RCVHWM = 24
    LINGER = 17
    NOBLOCK = 1
    DONTWAIT = 1
    SUBSCRIBE = 6

    class Again(Exception):
        pass

    class ZMQError(Exception):
        pass

    def __init__(self) -> None:
        self.contexts: list[FakeZmqContext] = []
        #: both spellings the real module offers: ``zmq.Context()`` and ``zmq.Context.instance()``
        self.Context = _FakeContextFactory(self)

    def new_context(self) -> FakeZmqContext:
        ctx = FakeZmqContext(self)
        self.contexts.append(ctx)
        return ctx


class _FakeContextFactory:
    """Stands in for the ``zmq.Context`` *class*, which is both callable and has ``instance()``."""

    def __init__(self, zmq: FakeZmq) -> None:
        self._zmq = zmq
        self._instance: FakeZmqContext | None = None

    def __call__(self) -> FakeZmqContext:
        return self._zmq.new_context()

    def instance(self) -> FakeZmqContext:
        if self._instance is None:
            self._instance = self._zmq.new_context()
        return self._instance
