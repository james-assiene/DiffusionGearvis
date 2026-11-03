"""Probe for T-0069 criterion 3, run INSIDE the pinned vLLM container (never on the host; no engine boot).

Driven by ``tests/contract/test_vllm_cd_wiring_gpu.py`` with this directory mounted at ``/astra_scripts`` and the plugin
source at ``/plugin`` (both read-only, ``--network none``). Prints one ``ASTRA_PROBE_JSON {...}`` line of *facts*; every
assertion lives on the host. Nothing here decides pass/fail.

Facts:
- ``model_state``: ``AstraDiffusionGemma.get_model_state_cls()`` -- its name, its bases, its base's bases, whether that
  base's base *is* the image's stock model state (``DiffusionGemmaForConditionalGeneration.get_model_state_cls()``),
  the full MRO and whether the call is cached.
- ``chain``: ``custom_sampler`` on an instance of that class. The image's stock model state needs an engine to
  construct, so for this one call its ``__init__`` is a no-op and its ``custom_sampler`` returns
  ``(toy, SENTINEL)``, where ``toy`` is a real image ``DiffusionSampler`` built on CUDA (the shape of
  ``astra_tap_probe.py``'s toy) and ``SENTINEL`` a fresh object. The chain is read by identity through
  ``vars(obj)["_inner"]`` (never through ``__getattr__`` delegation). The tap is the production one
  (``tap_from_env``) on an ``ipc://`` endpoint in the container's ``/tmp``.
- ``no_fallback``: with ``astra_vllm_plugin.cd.model_state`` made unimportable and the cache cleared,
  ``get_model_state_cls()`` must raise (reported: the exception type, or what it returned).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
import traceback
from types import SimpleNamespace
from typing import Any

MARKER = "ASTRA_PROBE_JSON"
DG = "vllm.model_executor.models.diffusion_gemma"


def _q(obj: Any) -> str:
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _regime() -> dict[str, Any]:
    import importlib.metadata as md
    import platform

    def version(dist: str) -> str:
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            return "absent"

    reg: dict[str, Any] = {
        "python": platform.python_version(),
        "python_from": sys.executable,
        "vllm": version("vllm"),
        "pytest": version("pytest"),
        "pyzmq": version("pyzmq"),
    }
    try:
        import torch

        cuda = torch.cuda.is_available()
        reg.update(
            torch=torch.__version__,
            torch_from=torch.__file__,
            torch_cuda=torch.version.cuda,
            cuda_available=cuda,
            device=torch.cuda.get_device_name(0) if cuda else "none",
        )
    except Exception as exc:
        reg["torch"] = f"import failed: {type(exc).__name__}: {exc}"
    return reg


def _toy_sampler() -> tuple[Any, Any]:
    """A real image ``DiffusionSampler`` on CUDA and the runner-sampler stand-in it was built from."""
    import torch

    dg = importlib.import_module(DG)
    states_mod = importlib.import_module("vllm.v1.worker.gpu.sample.states")
    dev = torch.device("cuda")
    R, CL, V, H = 2, 4, 64, 4
    st = dg.DiffusionGemmaRequestStates(
        max_num_reqs=R,
        canvas_length=CL,
        vocab_size=V,
        max_denoising_steps=8,
        device=dev,
        hidden_size=H,
        stability_threshold=2,
    )
    runner = SimpleNamespace(
        sampling_states=states_mod.SamplingStates(R, V),
        req_states=SimpleNamespace(draft_tokens=torch.zeros(R, CL, dtype=torch.int64, device=dev)),
        logprobs_mode="raw_logprobs",
    )
    toy = dg.DiffusionSampler(
        sampler=runner,
        diffusion_config=SimpleNamespace(canvas_length=CL),
        vocab_size=V,
        diffusion_states=st,
        t_min=0.4,
        t_max=0.8,
        entropy_bound=1.0,
        confidence_threshold=1e9,
        embed_weight=torch.zeros(V, H, device=dev),
        normalizer=torch.tensor(2.0, device=dev),
    )
    return toy, runner


def _chain(served: type, stock_ms: type) -> dict[str, Any]:
    toy, runner = _toy_sampler()
    sentinel = object()
    calls: list[Any] = []

    def no_init(self: Any, *a: Any, **k: Any) -> None:
        return None

    def stock_custom_sampler(self: Any, sampler: Any) -> Any:
        calls.append(sampler)
        return toy, sentinel

    saved = {n: stock_ms.__dict__.get(n) for n in ("__init__", "custom_sampler")}
    os.environ.setdefault("ASTRA_TAP_ENDPOINT", "ipc:///tmp/astra-cd-wiring-probe.sock")
    try:
        stock_ms.__init__ = no_init  # type: ignore[misc]
        stock_ms.custom_sampler = stock_custom_sampler  # type: ignore[attr-defined]
        ms = served()
        got = ms.custom_sampler(runner)
    finally:
        for n, v in saved.items():
            if v is None:
                delattr(stock_ms, n)
            else:
                setattr(stock_ms, n, v)
    out: dict[str, Any] = {
        "stock_custom_sampler_calls": len(calls),
        "stock_got_the_runner_sampler": bool(calls) and calls[0] is runner,
    }
    if not (isinstance(got, tuple) and len(got) == 2):
        out["returned"] = repr(got)[:200]
        return out
    outer, second = got
    objs = [outer]
    while len(objs) < 6:
        nxt = vars(objs[-1]).get("_inner") if hasattr(objs[-1], "__dict__") else None
        if nxt is None:
            break
        objs.append(nxt)
    out["types"] = [_q(o) for o in objs]
    out["innermost_is_stocks"] = objs[-1] is toy
    out["second_is_stocks"] = second is sentinel
    tap = vars(ms).get("_astra_tap")
    if tap is not None and hasattr(tap, "stop"):
        with contextlib.suppress(Exception):
            tap.stop()
    return out


def _no_fallback(astra: type) -> dict[str, Any]:
    fn = astra.get_model_state_cls
    clear = getattr(fn, "cache_clear", None)
    if clear is None:
        return {"error": "get_model_state_cls has no cache_clear (not functools.cache?)"}
    saved = sys.modules.get("astra_vllm_plugin.cd.model_state", "<absent>")
    sys.modules["astra_vllm_plugin.cd.model_state"] = None  # type: ignore[assignment]
    clear()
    try:
        got = astra.get_model_state_cls()
        return {"raised": None, "returned": _q(got)}
    except BaseException as exc:
        return {"raised": type(exc).__name__, "message": str(exc)[:200]}
    finally:
        if saved == "<absent>":
            sys.modules.pop("astra_vllm_plugin.cd.model_state", None)
        else:
            sys.modules["astra_vllm_plugin.cd.model_state"] = saved  # type: ignore[assignment]
        clear()


def main() -> int:
    out: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        out["regime"] = _regime()
    except Exception:
        out["regime"] = {"error": traceback.format_exc()}
    try:
        import vllm

        out["versions"] = {"vllm": getattr(vllm, "__version__", None)}
        dg = importlib.import_module(DG)
        stock_ms = dg.DiffusionGemmaForConditionalGeneration.get_model_state_cls()
        out["stock_model_state"] = _q(stock_ms)
        import astra_vllm_plugin.model as pm

        astra = pm.AstraDiffusionGemma
        try:
            served = astra.get_model_state_cls()
            again = astra.get_model_state_cls()
        except Exception:
            out["model_state"] = {"error": traceback.format_exc()}
            served = None
        if served is not None:
            bases = served.__bases__
            cd = bases[0] if len(bases) == 1 else None
            out["model_state"] = {
                "name": served.__name__,
                "qualname": _q(served),
                "bases": [_q(b) for b in bases],
                "base_name": cd.__name__ if cd is not None else None,
                "base_bases": [_q(b) for b in cd.__bases__] if cd is not None else None,
                "base_base_is_stock": cd is not None and cd.__bases__ == (stock_ms,),
                "mro": [_q(c) for c in served.__mro__],
                "cached": served is again,
            }
            try:
                out["chain"] = _chain(served, stock_ms)
            except Exception:
                out["chain"] = {"error": traceback.format_exc()}
            try:
                out["no_fallback"] = _no_fallback(astra)
            except Exception:
                out["no_fallback"] = {"error": traceback.format_exc()}
    except Exception:
        out["error"] = traceback.format_exc()
    print(MARKER, json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
