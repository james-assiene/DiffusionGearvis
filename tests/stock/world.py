"""A stock-only driver for the canvas tap around the real v0.29.0 sampler (T-0011 tier 1b, criteria 15-16).

Every token decision is the vendored (``tests/fixtures/vllm_v0_29_0/``) or, under ``ASTRA_STOCK=image``, the
image's own code: ``DiffusionSampler``, ``DiffusionGemmaRequestStates``, ``SamplingStates`` and
``_compiled_sample_step``. This module only drives them -- it builds ``InputBatch`` stand-ins with the
image's attribute names, plays the scheduler (slots, prefill, decode, release) and snapshots rows. It never
re-implements a line of the step. ``FakeModelStateBase.custom_sampler`` constructs the stock sampler as
upstream does (``diffusion_gemma.py:828-860``); ``add_request``/``remove_request`` are upstream's bodies.

Environment:

``ASTRA_STOCK=image``
    the image's compiled classes instead of the vendored eager copy (criterion 16, inside the container);
``ASTRA_STOCK_DEVICE``
    ``cpu`` (default) or ``cuda``.

The tap is reached only through its public entry points: ``make_model_state_cls(base, tap=factory)``
(``astra_vllm_plugin.model_state``), ``CanvasTap`` and ``TorchTapBackend(device)``
(``astra_vllm_plugin.tap``). Shared helpers live here, not in the conftest (the container runs
``--noconftest``).
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

try:
    import numpy as np
    import torch
except ImportError as exc:  # the variable is set but the group is missing: fail, never skip
    raise ImportError(
        "tests/stock needs the `stock` dependency group (torch 2.13.0+cpu): run "
        "`ASTRA_STOCK_TESTS=1 uv run --group stock pytest tests/stock`. "
        f"Import failed: {exc}"
    ) from exc

import pytest

from tests.fakes.diffusion_sampler import ManualClock, RecordingPublisher

REPO = Path(__file__).resolve().parents[2]
PLUGIN_SRC = REPO / "plugin" / "astra_vllm_plugin"

DEVICE = torch.device(os.environ.get("ASTRA_STOCK_DEVICE", "cpu"))
IMAGE = os.environ.get("ASTRA_STOCK") == "image"
if IMAGE:
    from vllm.model_executor.models import diffusion_gemma as _dg
    from vllm.v1.worker.gpu.sample import states as _states

    stock = SimpleNamespace(
        DiffusionSampler=_dg.DiffusionSampler,
        DiffusionGemmaRequestStates=_dg.DiffusionGemmaRequestStates,
        SamplingStates=_states.SamplingStates,
    )
else:
    from tests.fixtures.vllm_v0_29_0 import diffusion_sampler as stock  # type: ignore[no-redef]

V = 64  # vocab of the toy world: every token decision is still stock's
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
OUTPUT_ROWS = ("sampled_token_ids", "num_sampled", "num_rejected")


def regime() -> str:
    return (
        f"torch {torch.__version__}, device {DEVICE}, stock={'image' if IMAGE else 'vendored v0.29.0'}, "
        f"V={V}, python {sys.version.split()[0]}"
    )


# --------------------------------------------------------------------------- the plugin's entry points


def plugin_module(name: str) -> ModuleType:
    if str(PLUGIN_SRC) not in sys.path:
        sys.path.insert(0, str(PLUGIN_SRC))
    mod = f"astra_vllm_plugin.{name}"
    try:
        return importlib.import_module(mod)
    except Exception as exc:
        pytest.fail(f"T-0011 not implemented: cannot import {mod} ({type(exc).__name__}: {exc})")


def plugin_attr(module: str, name: str) -> Any:
    obj = getattr(plugin_module(module), name, None)
    if obj is None:
        pytest.fail(f"T-0011 not implemented: astra_vllm_plugin.{module}.{name} is missing")
    return obj


class TapFactory:
    """``tap=``: builds ``CanvasTap(publisher, slots=..., backend=TorchTapBackend(DEVICE), ...)``."""

    def __init__(self, *, hz: float, queue_size: int, clock: ManualClock) -> None:
        self.hz, self.queue_size, self.clock = hz, queue_size, clock
        self.publisher = RecordingPublisher()
        self.built: list[Any] = []

    def __call__(self, slots: Any) -> Any:
        backend = plugin_attr("tap", "TorchTapBackend")(DEVICE)
        tap = plugin_attr("tap", "CanvasTap")(
            self.publisher,
            slots=slots,
            backend=backend,
            hz=self.hz,
            clock=self.clock,
            queue_size=self.queue_size,
        )
        self.built.append(tap)
        return tap


# --------------------------------------------------------------------------- configuration + stock model state


@dataclasses.dataclass(frozen=True)
class Cfg:
    CL: int = 8
    max_num_reqs: int = 4
    hidden: int = 4
    max_denoising_steps: int = 12
    #: generation_config ``stability_threshold``; the states hold ``+1`` (upstream :800)
    stability_threshold: int = 1
    t_min: float = 0.4
    t_max: float = 0.8
    entropy_bound: float = 1.0
    confidence_threshold: float = 1e9  # always confident: convergence = stability or the step cap
    prompt_len: int = 5


class FakeRunnerSampler:
    """The runner's ``Sampler`` as ``DiffusionSampler.__init__`` reads it."""

    def __init__(self, cfg: Cfg) -> None:
        self.sampling_states = stock.SamplingStates(cfg.max_num_reqs, V)
        self.req_states = SimpleNamespace(
            draft_tokens=torch.zeros(cfg.max_num_reqs, cfg.CL, dtype=torch.int64, device=DEVICE)
        )
        self.logprobs_mode = "raw_logprobs"


class FakeModelStateBase:
    """``DiffusionGemmaModelState`` stand-in whose ``custom_sampler`` *constructs* the stock sampler."""

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
        cfg = self.cfg
        inner = stock.DiffusionSampler(
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
            self.diffusion_states.prompt_len[req_index].fill_(len(new_req_data.prompt_token_ids))

    def remove_request(self, req_id: str) -> None:
        idx = self._req_id_to_index.pop(req_id, None)
        if idx is not None:
            self.diffusion_states.remove_request(idx)


def sampling_params(extra_args: Mapping[str, Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=None,
        logprobs=None,
        max_tokens=4096,
        extra_args=None if extra_args is None else dict(extra_args),
        stop_token_ids=[1],
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )


# --------------------------------------------------------------------------- the driven world


@dataclasses.dataclass
class Req:
    req_id: str
    slot: int
    prefilled: bool = False
    valid_len: int | None = None  # None: CL


@dataclasses.dataclass
class Snap:
    """One sampler call, cloned on the host right after it returned (outside any spy)."""

    call: int
    kind: str  # "prefill" | "decode" | "release"
    order: tuple[str, ...]
    slots: dict[str, int]
    state: dict[str, torch.Tensor]
    draft_tokens: torch.Tensor
    out: dict[str, torch.Tensor]
    rng_after: torch.Tensor
    frames: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    #: the slots' ``step`` row *before* the call: on a commit call it is the step the canvas had
    #: reached, which R10 makes the commit frame's ``step`` (stock resets ``step`` to 0 in the call)
    pre_step: torch.Tensor | None = None

    def row(self, name: str, req_id: str) -> torch.Tensor:
        return self.state[name][self.slots[req_id]]

    def out_row(self, name: str, req_id: str) -> torch.Tensor:
        return self.out[name][self.order.index(req_id)]

    def committed(self, req_id: str) -> list[int] | None:
        if self.kind != "decode" or req_id not in self.order:
            return None
        n = int(self.out_row("num_sampled", req_id))
        if n == 0:
            return None
        return [int(x) for x in self.out_row("sampled_token_ids", req_id)[:n]]


LogitsFn = Callable[[int, str, int], torch.Tensor]  # (call, req_id, valid_len) -> [valid_len, V]


class World:
    """One engine: a model state (stock, or wrapped by the tap), its sampler and a toy scheduler.

    ``tap=True`` builds ``make_model_state_cls(FakeModelStateBase, tap=TapFactory(...))``. The global torch
    RNG is seeded here; two runs of a comparison must make the same calls. ``drain_each_call`` drains the
    tap synchronously after every call (outside the spy) and attributes the new frames to that call.
    """

    def __init__(
        self,
        cfg: Cfg,
        *,
        tap: bool,
        seed: int = 0,
        hz: float = 1e6,
        dt: float = 1.0,
        queue_size: int = 4096,
        drain_each_call: bool = True,
    ) -> None:
        torch.manual_seed(seed)
        self.cfg = cfg
        self.clock = ManualClock()
        self.dt = dt
        self.factory: TapFactory | None = None
        base: type = FakeModelStateBase
        if tap:
            self.factory = TapFactory(hz=hz, queue_size=queue_size, clock=self.clock)
            make = plugin_attr("model_state", "make_model_state_cls")
            base = make(FakeModelStateBase, tap=self.factory)
        self.ms = base(cfg)
        self.runner_sampler = FakeRunnerSampler(cfg)
        got = self.ms.custom_sampler(self.runner_sampler)
        assert got is not None
        self.sampler, _ = got
        self.tap = self.factory.built[-1] if self.factory else None
        self.reqs: dict[str, Req] = {}
        self.free_slots: list[int] = list(range(cfg.max_num_reqs - 1, -1, -1))  # LIFO: reuse
        self.calls = 0
        self.snaps: list[Snap] = []
        self.drain_each_call = drain_each_call
        self._drained = 0
        #: context entered around **only** the sampler call (and the release)
        self.call_ctx: Callable[[], Any] = contextlib.nullcontext
        #: context entered around **only** the model state's ``add_request`` (where the tap binds)
        self.bind_ctx: Callable[[], Any] = contextlib.nullcontext

    # ---- frames
    def _new_frames(self) -> list[dict[str, Any]]:
        if self.tap is None:
            return []
        self.tap.drain()
        sent = self.factory.publisher.sent if self.factory else []
        new = sent[self._drained :]
        self._drained = len(sent)
        return list(new)

    @property
    def frames(self) -> list[dict[str, Any]]:
        return list(self.factory.publisher.sent) if self.factory else []

    # ---- requests
    def add(self, req_id: str, *, valid_len: int | None = None, xargs: Any = None) -> int:
        slot = self.free_slots.pop()
        new_req_data = SimpleNamespace(
            req_id=req_id,
            prompt_token_ids=[2] * self.cfg.prompt_len,
            prefill_token_ids=[2] * self.cfg.prompt_len,
        )
        with self.bind_ctx():
            self.ms.add_request(slot, new_req_data)
        self.sampler.add_request(slot, self.cfg.prompt_len, sampling_params(xargs))
        self.sampler.apply_staged_writes()
        self.reqs[req_id] = Req(req_id, slot, valid_len=valid_len)
        return slot

    def remove(self, req_id: str) -> Snap:
        req = self.reqs.pop(req_id)
        with self.call_ctx():
            self.ms.remove_request(req_id)
        self.free_slots.append(req.slot)
        snap = self._snap("release", [req], None)
        return snap

    # ---- batches
    def _batch(self, reqs: Sequence[Req], *, prefill: bool) -> SimpleNamespace:
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
            t = torch.from_numpy(a.copy())
            if DEVICE.type == "cuda":
                return t.pin_memory().to(DEVICE, non_blocking=True)
            return t

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

    def _snap(
        self, kind: str, reqs: Sequence[Req], returned: Any, pre_step: torch.Tensor | None = None
    ) -> Snap:
        st = self.ms.diffusion_states
        out: dict[str, torch.Tensor] = {}
        if returned is not None:
            out = {name: getattr(returned, name).clone().cpu() for name in OUTPUT_ROWS}
        snap = Snap(
            call=self.calls,
            kind=kind,
            order=tuple(r.req_id for r in reqs),
            slots={r.req_id: r.slot for r in reqs},
            state={name: getattr(st, name).clone().cpu() for name in STATE_ROWS},
            draft_tokens=self.runner_sampler.req_states.draft_tokens.clone().cpu(),
            out=out,
            pre_step=pre_step,
            rng_after=(
                torch.cat([torch.get_rng_state(), torch.cuda.get_rng_state()])
                if DEVICE.type == "cuda"
                else torch.get_rng_state()
            ),
        )
        if self.drain_each_call:
            snap.frames = self._new_frames()
        self.snaps.append(snap)
        self.calls += 1
        self.clock.advance(self.dt)
        return snap

    def prefill(self) -> Snap:
        reqs = [r for r in self.reqs.values() if not r.prefilled]
        batch = self._batch(reqs, prefill=True)
        logits = torch.zeros(0, V, device=DEVICE)
        with self.call_ctx():
            ret = self.sampler(logits, batch)
        for r in reqs:
            r.prefilled = True
        return self._snap("prefill", reqs, ret)

    def _decode_inputs(
        self, logits_fn: LogitsFn, only: Sequence[str] | None
    ) -> tuple[list[Req], Any, Any]:
        reqs = [r for r in self.reqs.values() if r.prefilled and (only is None or r.req_id in only)]
        batch = self._batch(reqs, prefill=False)
        parts = [logits_fn(self.calls, r.req_id, r.valid_len or self.cfg.CL).float() for r in reqs]
        logits = torch.cat(parts, dim=0) if parts else torch.zeros(0, V)
        if DEVICE.type == "cuda":
            logits = logits.pin_memory().to(DEVICE, non_blocking=True)
        return reqs, batch, logits

    def decode(self, logits_fn: LogitsFn, *, only: Sequence[str] | None = None) -> Snap:
        reqs, batch, logits = self._decode_inputs(logits_fn, only)
        pre_step = self.ms.diffusion_states.step.clone().cpu()  # outside the spy and the call
        with self.call_ctx():
            ret = self.sampler(logits, batch)
        return self._snap("decode", reqs, ret, pre_step)

    def decode_unsynced(
        self, logits_fn: LogitsFn, *, only: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """``decode`` with no host read of device memory (criterion 16b): the rows and outputs are
        cloned **on the device** and read only after the run."""
        reqs, batch, logits = self._decode_inputs(logits_fn, only)
        st = self.ms.diffusion_states
        pre_step = st.step.clone()  # on the device: no host read
        with self.call_ctx():
            ret = self.sampler(logits, batch)
        dev = {
            "pre_step": pre_step,
            "order": tuple(r.req_id for r in reqs),
            "slots": {r.req_id: r.slot for r in reqs},
            "state": {name: getattr(st, name).clone() for name in STATE_ROWS},
            "out": {name: getattr(ret, name).clone() for name in OUTPUT_ROWS},
        }
        self.calls += 1
        self.clock.advance(self.dt)
        return dev


# --------------------------------------------------------------------------- logits


def peaked_logits(tokens: Mapping[str, int], *, period: int = 9) -> LogitsFn:
    """``[L, V]`` at -30 except one token per request at +8, the token shifting every ``period``
    calls so successive canvases commit different ids. Deterministic: no generator involved."""

    def fn(call: int, req_id: str, valid_len: int) -> torch.Tensor:
        x = torch.full((valid_len, V), -30.0)
        t = (tokens[req_id] + call // period) % (V - 2) + 2
        x[:, t] = 8.0
        x[::2, (t + 1) % V] = 7.5  # a runner-up on even positions keeps the stock sampler honest
        return x

    return fn


# --------------------------------------------------------------------------- the S8 spy (torch)


def _short(path: str) -> str:
    for marker in ("/site-packages/", "/dist-packages/", "/repo/", str(REPO) + "/"):
        if marker in path:
            return path.split(marker, 1)[1]
    return "/".join(path.split("/")[-3:])


def _owner(stack: Sequence[Any]) -> str:
    """Whose code made the call: walking out from the innermost frame (torch internals skipped),
    the first frame that is the plugin's or stock's decides. The plugin's ``TapSampler`` frame sits
    *outside* every stock frame it delegates to, so a stock sync is never charged to the tap."""
    for f in reversed(stack):
        if "astra_vllm_plugin" in f.filename:
            return "plugin"
        if "/vllm/" in f.filename or "vllm_v0_29_0" in f.filename:
            return "stock"
    return "other"


@dataclasses.dataclass(frozen=True)
class SyncHit:
    """Where a recorded sync came from: the op, the innermost frame that called it, the innermost
    frame in this repository (tests, plugin), and whether ``astra_vllm_plugin`` is on the stack."""

    op: str
    site: str
    repo_site: str
    from_plugin: bool
    device: str = "?"

    @classmethod
    def here(cls, op: str, device: str = "?") -> SyncHit:
        import traceback

        stack = [
            f
            for f in traceback.extract_stack()
            if not (f.filename == __file__ and f.name in ("_hit", "here", "__torch_function__"))
        ]
        inner = stack[-1] if stack else None
        repo = next(
            (
                f
                for f in reversed(stack)
                if "astra_vllm_plugin" in f.filename or "/tests/" in f.filename
            ),
            None,
        )
        fmt = (lambda f: f"{_short(f.filename)}:{f.lineno} {f.name}") if stack else str
        return cls(
            op=op,
            site=fmt(inner) if inner else "?",
            repo_site=fmt(repo) if repo else "?",
            from_plugin=_owner(stack) == "plugin",
            device=device,
        )

    def __str__(self) -> str:
        who = "TAP" if self.from_plugin else "stock"
        return f"{self.op} on {self.device} [{who}] at {self.site} <- {self.repo_site}"


class SyncSpy(torch.overrides.TorchFunctionMode):
    """Records every host-synchronising call made while active (criterion 3's list, ruling S8).

    S8 defines a sync as a device->host read. A value read (``.item``, ``.tolist``, ``.numpy``,
    ``bool``/``int``/``float``/``__index__``, ``np.asarray``, ``.nonzero``) therefore counts only
    on a tensor that lives on the world's device -- ``device`` (default ``DEVICE``). On CUDA a CPU
    tensor's ``.item()`` is a host read, not a sync: inductor's generated wrappers read CPU scalar
    tensors on every call (tier 2 at 5fc032e: 240 such ``.item`` hits). On a CPU run the world's
    device *is* the CPU, so every CPU tensor stands in for a device tensor, as before. The
    device->host move rules (``.cpu``, blocking ``.to``/``copy_``) are unchanged.
    """

    FORBIDDEN = {
        torch.Tensor.item: "item",
        torch.Tensor.tolist: "tolist",
        torch.Tensor.numpy: "numpy",
        torch.Tensor.__bool__: "__bool__",
        torch.Tensor.__int__: "__int__",
        torch.Tensor.__float__: "__float__",
        torch.Tensor.__index__: "__index__",
        torch.Tensor.__array__: "np.asarray",
        torch.Tensor.nonzero: "nonzero",
        torch.nonzero: "nonzero",
    }

    def __init__(self, device: Any = None) -> None:
        super().__init__()
        self.device_type = torch.device(device if device is not None else DEVICE).type
        self.hits: list[str] = []
        #: one per hit: the op, the tensor's device, its innermost call site and whose code it was
        self.records: list[SyncHit] = []

    def _hit(self, name: str, device: Any = "?") -> None:
        self.hits.append(name)
        self.records.append(SyncHit.here(name, str(device)))

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        src = args[0] if args and isinstance(args[0], torch.Tensor) else None
        name = self.FORBIDDEN.get(func)
        if name and src is not None and src.device.type == self.device_type:
            self._hit(name, src.device)
        # S8: a move counts only when it is device -> host and blocking; same-device `.to` is not a sync
        if src is not None and src.device.type != "cpu":
            if func is torch.Tensor.cpu:
                self._hit("cpu", src.device)
            elif func is torch.Tensor.to and not kwargs.get("non_blocking", False):
                dev = kwargs.get("device")
                for a in args[1:]:
                    if isinstance(a, (str, torch.device)):
                        dev = a
                    elif isinstance(a, torch.Tensor):
                        dev = a.device
                if dev is not None and torch.device(dev).type == "cpu":
                    self._hit("to(cpu, blocking)", src.device)
        if func is torch.Tensor.copy_ and len(args) > 1 and isinstance(args[1], torch.Tensor):
            dst, src2 = args[0], args[1]
            if src2.device.type != "cpu" and dst.device.type == "cpu":
                if not kwargs.get("non_blocking", False):
                    self._hit("copy_(D2H, blocking)", src2.device)
                elif not dst.is_pinned():
                    self._hit("copy_(D2H into pageable memory)", src2.device)
        return func(*args, **kwargs)


# --------------------------------------------------------------------------- the oracle for one frame


def stable_hex_oracle(history: torch.Tensor, history_len: int) -> str:
    """Criterion 7 by float64 numpy: bit p iff every history row equals row 0 at p and len >= ST."""
    h = np.asarray(history.cpu().numpy(), dtype=np.float64)
    st = h.shape[0]
    bits = np.all(h == h[0:1], axis=0) & (history_len >= st)
    return np.packbits(bits.astype(np.uint8)).tobytes().hex()  # MSB-first, padded to whole bytes


def frame_mismatches(
    frame: Mapping[str, Any], snap: Snap, req_id: str, *, canvas_idx: int, max_steps: int
) -> list[str]:
    """Every field of ``frame`` that disagrees with the rows of the call it was staged at."""
    bad: list[str] = []

    def chk(name: str, got: Any, want: Any) -> None:
        if got != want:
            bad.append(f"{name}: frame {got!r} != rows {want!r}")

    chk("request_id", frame.get("request_id"), req_id)
    chk("max_steps", frame.get("max_steps"), max_steps)
    committed = snap.committed(req_id)
    chk("phase", frame.get("phase"), "commit" if committed else "denoise")
    if committed:
        # R10: a commit frame carries the committed ids and none of the rows, and its step is the
        # step the canvas had reached -- the pre-call value (stock resets it to 0 in the call)
        chk("committed_tokens", list(frame.get("committed_tokens", ())), committed)
        for name in ("canvas", "argmax", "stable"):
            chk(name, frame.get(name), None)
        assert snap.pre_step is not None, "fixture: a decode snap records the pre-call step"
        chk("step (pre-reset)", frame.get("step"), int(snap.pre_step[snap.slots[req_id]]))
    else:
        chk("canvas", list(frame.get("canvas") or ()), [int(x) for x in snap.row("canvas", req_id)])
        chk(
            "argmax",
            list(frame.get("argmax") or ()),
            [int(x) for x in snap.row("argmax_canvas", req_id)],
        )
        chk("step", frame.get("step"), int(snap.row("step", req_id)))
        chk(
            "stable",
            frame.get("stable"),
            stable_hex_oracle(
                snap.row("accepted_canvas_history", req_id),
                int(snap.row("accepted_canvas_history_len", req_id)),
            ),
        )
        chk("committed_tokens", list(frame.get("committed_tokens") or ()), [])
    chk("canvas_idx", frame.get("canvas_idx"), canvas_idx)
    chk("accepted", frame.get("accepted"), None)
    chk("entropy", frame.get("entropy"), None)
    return bad


def bits(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    if t.dtype == torch.float64:
        return t.view(torch.int64)
    return t


def snaps_differ(a: Snap, b: Snap) -> list[str]:
    """Every output field, state row, ``draft_tokens`` and the RNG state that differ, bit for bit."""
    bad = [n for n in STATE_ROWS if not torch.equal(bits(a.state[n]), bits(b.state[n]))]
    if not torch.equal(a.draft_tokens, b.draft_tokens):
        bad.append("draft_tokens")
    bad += [n for n in a.out if not torch.equal(bits(a.out[n]), bits(b.out[n]))]
    if set(a.out) != set(b.out):
        bad.append("output fields")
    if not torch.equal(a.rng_after, b.rng_after):
        bad.append("rng state")
    return bad
