"""The served composition Tap ∘ CD ∘ stock on the CD stand-in (T-0069 criteria 1-3), shared by the CPU module
``tests/cd/integration/test_cd_tap_composition.py`` and the CUDA module ``tests/cd/cuda/test_cd_tap_composition_cuda.py``.

A module, not a conftest: the container runs pytest with ``--noconftest``.

Every token decision is still stock's: the model state's base is ``tests/cd/fakes/diffusion_sampler.py``'s
``FakeModelStateBase``, whose ``custom_sampler`` constructs the vendored v0.29.0 ``DiffusionSampler`` (or, under
``ASTRA_CD_STOCK=image``, the image's compiled one). The plugin is reached only through its public entry points:

- ``astra_vllm_plugin.model.AstraDiffusionGemma.get_model_state_cls()`` -- the served composition, built over a
  fake ``vllm.model_executor.models.diffusion_gemma`` whose ``DiffusionGemmaForConditionalGeneration.get_model_state_cls()``
  returns the stand-in base (``served_model_state_cls``). The fake vllm module holds that one name and nothing else;
  ``model.py`` imports nothing else from vllm.
- ``astra_vllm_plugin.model_state.make_model_state_cls(base, tap=...)`` (T-0011) and
  ``astra_vllm_plugin.cd.model_state.make_cd_model_state_cls(base)`` (T-0067), composed directly, for the reference
  runs and for the CUDA row (the image's real ``DiffusionGemmaModelState`` cannot be built without an engine).

The tap factory is ``TapFactory``: ``CanvasTap(RecordingPublisher(), slots=..., backend=TorchTapBackend(DEVICE),
hz=..., clock=ManualClock, queue_size=...)``, as ``tests/stock/world.py`` builds it. ``model.py`` names
``tap_from_env``; the served rows replace ``astra_vllm_plugin.tap.tap_from_env`` with a ``TapRouter`` that forwards
to the current world's factory (the composed class is cached, so its ``tap=`` is captured once).
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.cd.fakes import diffusion_sampler as cdf
from tests.cd.fakes.diffusion_sampler import DEVICE, Cfg, FakeModelStateBase, Req, Snap
from tests.fakes.diffusion_sampler import ManualClock, RecordingPublisher

FAKE_VLLM_MODULE = "vllm.model_executor.models.diffusion_gemma"


def plugin_attr(module: str, name: str, task: str = "T-0069") -> Any:
    """``astra_vllm_plugin.<module>.<name>``; absent reads as "<task> not implemented", as an assertion."""
    if str(cdf.PLUGIN_SRC) not in sys.path:
        sys.path.insert(0, str(cdf.PLUGIN_SRC))
    mod = f"astra_vllm_plugin.{module}"
    try:
        m = importlib.import_module(mod)
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("astra_vllm_plugin"):
            pytest.fail(f"{task} not implemented: cannot import {mod} ({exc})")
        raise
    obj = getattr(m, name, None)
    if obj is None:
        pytest.fail(f"{task} not implemented: {mod}.{name} is missing")
    return obj


class TapFactory:
    """``tap=``: one ``CanvasTap`` per ``custom_sampler`` call, publishing into ``self.publisher``."""

    def __init__(self, *, hz: float = 1e6, queue_size: int = 4096) -> None:
        self.hz, self.queue_size = hz, queue_size
        self.clock = ManualClock()
        self.publisher = RecordingPublisher()
        self.built: list[Any] = []

    def __call__(self, slots: Any) -> Any:
        backend = plugin_attr("tap", "TorchTapBackend", "T-0011")(DEVICE)
        tap = plugin_attr("tap", "CanvasTap", "T-0011")(
            self.publisher,
            slots=slots,
            backend=backend,
            hz=self.hz,
            clock=self.clock,
            queue_size=self.queue_size,
        )
        self.built.append(tap)
        return tap


class TapRouter:
    """Stands in for ``tap_from_env`` in the served rows: forwards to ``target`` (set per world)."""

    def __init__(self) -> None:
        self.target: TapFactory | None = None
        self.calls = 0

    def __call__(self, slots: Any) -> Any:
        self.calls += 1
        assert self.target is not None, "fixture: no TapFactory routed for this world"
        return self.target(slots)


class FakeDiffusionGemmaForConditionalGeneration:
    """The stock model class as ``model.py`` uses it: ``get_model_state_cls()`` is a staticmethod returning
    the model-state class (here the CD stand-in's ``FakeModelStateBase``)."""

    stock_model_state: type = FakeModelStateBase

    @staticmethod
    def get_model_state_cls() -> type:
        return FakeDiffusionGemmaForConditionalGeneration.stock_model_state


def served_model(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, TapRouter]:
    """Import ``astra_vllm_plugin.model`` afresh over the fake vllm module; ``tap_from_env`` is a ``TapRouter``.

    Returns ``(AstraDiffusionGemma, router)``. Every ``sys.modules`` entry touched is restored at teardown, so the
    fresh ``astra_vllm_plugin.model`` (built on the fake) never leaks into another row."""
    if str(cdf.PLUGIN_SRC) not in sys.path:
        sys.path.insert(0, str(cdf.PLUGIN_SRC))
    parts = FAKE_VLLM_MODULE.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        mod = types.ModuleType(name)
        if i < len(parts):
            mod.__path__ = []  # a package
        monkeypatch.setitem(sys.modules, name, mod)
    leaf = sys.modules[FAKE_VLLM_MODULE]
    leaf.DiffusionGemmaForConditionalGeneration = FakeDiffusionGemmaForConditionalGeneration  # type: ignore[attr-defined]
    # record the original entry (or its absence) so teardown restores it, then force a fresh import
    monkeypatch.setitem(sys.modules, "astra_vllm_plugin.model", None)
    del sys.modules["astra_vllm_plugin.model"]
    router = TapRouter()
    tap_mod = importlib.import_module("astra_vllm_plugin.tap")
    monkeypatch.setattr(tap_mod, "tap_from_env", router)
    try:
        model = importlib.import_module("astra_vllm_plugin.model")
    except ModuleNotFoundError as exc:
        pytest.fail(f"astra_vllm_plugin.model does not import over the fake vllm module: {exc}")
    return model.AstraDiffusionGemma, router


def chain(sampler: Any) -> list[Any]:
    """The wrapper chain by identity: ``sampler``, its ``_inner``, that one's ``_inner``, ... (T-0011 ``TapSampler``
    and T-0067 ``ConstrainedDiffusionSampler`` both keep the wrapped object as ``_inner``; read from ``vars`` so
    ``__getattr__`` delegation cannot answer)."""
    out = [sampler]
    while True:
        nxt = vars(out[-1]).get("_inner") if hasattr(out[-1], "__dict__") else None
        if nxt is None or len(out) > 5:
            return out
        out.append(nxt)


def chain_names(sampler: Any) -> list[str]:
    return [type(x).__name__ for x in chain(sampler)]


class World(cdf.World):
    """``tests/cd/fakes/diffusion_sampler.World`` with a model-state class chosen by the caller and the tap's frames.

    ``cls`` is the model-state class (served, or composed by hand); ``factory`` the ``TapFactory`` its tap uses
    (``None``: no tap). With ``router`` given, the router is pointed at ``factory`` before ``custom_sampler``.
    After every call (and every release) the tap is drained **outside** ``call_ctx`` and the new frames are
    attached to the snapshot (``snap.frames``) or to ``release_frames[req_id]``; ``snap.pre_step`` holds every
    slot's ``step`` before the call (a commit frame's step, R10). ``drain_each_call=False`` leaves the frames
    staged until ``drain()``.
    """

    def __init__(
        self,
        cfg: Cfg,
        cls: type,
        *,
        factory: TapFactory | None = None,
        router: TapRouter | None = None,
        seed: int = 0,
        drain_each_call: bool = True,
        dt: float = 1.0,
    ) -> None:
        torch.manual_seed(seed)
        self.cfg = cfg
        self.factory = factory
        if router is not None:
            router.target = factory
        self.ms = cls(cfg)
        self.runner_sampler = cdf.FakeRunnerSampler(cfg)
        got = self.ms.custom_sampler(self.runner_sampler)
        assert got is not None, "fixture: custom_sampler returned None"
        self.sampler, self.second = got
        self.inner = self.ms.constructed[-1]
        self.tap = factory.built[-1] if factory is not None and factory.built else None
        self.reqs: dict[str, Req] = {}
        self.free_slots: list[int] = list(range(cfg.max_num_reqs - 1, -1, -1))
        self.calls = 0
        self.snaps: list[Snap] = []
        self.entry_phases: list[torch.Tensor] = []
        self.call_ctx: Callable[[], Any] = contextlib.nullcontext
        #: context entered around **only** the model state's ``add_request`` (where the tap and CD bind)
        self.bind_ctx: Callable[[], Any] = contextlib.nullcontext
        self.drain_each_call = drain_each_call
        self.dt = dt
        self._drained = 0
        self.release_frames: dict[str, list[dict[str, Any]]] = {}

    # ---- frames
    def drain(self) -> list[dict[str, Any]]:
        if self.tap is None or self.factory is None:
            return []
        self.tap.drain()
        sent = self.factory.publisher.sent
        new = sent[self._drained :]
        self._drained = len(sent)
        return list(new)

    @property
    def frames(self) -> list[dict[str, Any]]:
        return list(self.factory.publisher.sent) if self.factory is not None else []

    # ---- requests
    def add(  # type: ignore[override]
        self,
        req_id: str,
        *,
        xargs: Mapping[str, Any] | None = None,
        slot: int | None = None,
        **params: Any,
    ) -> int:
        if slot is None:
            slot = self.free_slots.pop()
        else:
            self.free_slots.remove(slot)
        sp = cdf.sampling_params(extra_args=xargs, **params)
        prompt = [2] * self.cfg.prompt_len
        new_req_data = SimpleNamespace(
            req_id=req_id, prompt_token_ids=prompt, prefill_token_ids=list(prompt)
        )
        with self.bind_ctx():
            self.ms.add_request(slot, new_req_data)
        self.sampler.add_request(slot, self.cfg.prompt_len, sp)
        self.sampler.apply_staged_writes()
        self.reqs[req_id] = Req(req_id, slot, sp)
        return slot

    def remove(self, req_id: str) -> None:
        req = self.reqs.pop(req_id)
        with self.call_ctx():
            self.ms.remove_request(req_id)
        self.free_slots.append(req.slot)
        if self.drain_each_call:
            self.release_frames[req_id] = self.drain()
        if self.factory is not None:
            self.factory.clock.advance(self.dt)

    # ---- calls
    def _snap(self, kind: str, reqs: Sequence[Req], returned: Any, entry: dict[str, bool]) -> Snap:
        snap = super()._snap(kind, reqs, returned, entry)
        snap.frames = self.drain() if self.drain_each_call else []  # type: ignore[attr-defined]
        if self.factory is not None:
            self.factory.clock.advance(self.dt)
        return snap

    def decode(self, logits_fn: Any, *, only: Sequence[str] | None = None) -> Snap:
        pre = self.ms.diffusion_states.step.clone().cpu()  # outside call_ctx
        snap = super().decode(logits_fn, only=only)
        snap.pre_step = pre  # type: ignore[attr-defined]
        return snap

    def prefill(self) -> Snap:
        pre = self.ms.diffusion_states.step.clone().cpu()
        snap = super().prefill()
        snap.pre_step = pre  # type: ignore[attr-defined]
        return snap


def commit_frames(frames: Sequence[Mapping[str, Any]], req_id: str) -> list[Mapping[str, Any]]:
    return [f for f in frames if f.get("request_id") == req_id and f.get("phase") == "commit"]
