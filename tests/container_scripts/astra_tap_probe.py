"""Probe for T-0011 criteria 11-12, run INSIDE the pinned vLLM container (never on the host; no engine boot).

Driven by ``tests/contract/test_vllm_tap_fields_gpu.py``: ``argv[1]`` is a JSON object with the vendored
``manifest.json`` (``"manifest"``) and the image-module list of criterion 10 (``"modules"``). The plugin source is
mounted at ``/plugin`` and on ``PYTHONPATH``. Prints one ``ASTRA_PROBE_JSON {...}`` line of *facts*; every assertion
lives on the host, so a failure is a readable pytest message, not a container traceback. Nothing here decides
pass/fail.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import sys
import traceback
from types import SimpleNamespace
from typing import Any

MARKER = "ASTRA_PROBE_JSON"
DG = "vllm.model_executor.models.diffusion_gemma"

#: what the tap reads, keyed by the upstream class's dotted path (criterion 11)
READS = {
    f"{DG}.DiffusionSampler": [
        "diffusion_states",
        "add_request",
        "apply_staged_writes",
        "penalties_state",
    ],
    f"{DG}.DiffusionGemmaRequestStates": [
        "canvas",
        "argmax_canvas",
        "step",
        "accepted_canvas_history",
        "accepted_canvas_history_len",
        "max_denoising_steps",
        "stability_threshold",
        "canvas_length",
        "max_num_reqs",
        "device",
    ],
    "vllm.v1.worker.gpu.sample.output.SamplerOutput": ["sampled_token_ids", "num_sampled"],
    "vllm.v1.worker.gpu.input_batch.InputBatch": [
        "num_reqs",
        "num_draft_tokens",
        "idx_mapping_np",
        "idx_mapping",
        "cu_num_logits_np",
    ],
}


def _resolve(dotted: str) -> Any:
    mod, _, name = dotted.rpartition(".")
    return getattr(importlib.import_module(mod), name)


def _self_stores(cls: type) -> set[str]:
    names: set[str] = set()
    for c in cls.__mro__:
        try:
            tree = ast.parse(_dedent(c))
        except (OSError, TypeError, SyntaxError):
            continue
        names |= {
            n.attr
            for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)
            and isinstance(n.ctx, ast.Store)
            and isinstance(n.value, ast.Name)
            and n.value.id == "self"
        }
    return names


def _dedent(c: type) -> str:
    import textwrap

    return textwrap.dedent(inspect.getsource(c))


def _has(cls: type, name: str) -> bool:
    fields = getattr(cls, "__dataclass_fields__", {}) or {}
    return hasattr(cls, name) or name in fields or name in _self_stores(cls)


def _return_arities(fn: Any) -> list[int | str]:
    tree = ast.parse(_dedent_src(fn))
    out: list[int | str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Return):
            if isinstance(n.value, ast.Tuple):
                out.append(len(n.value.elts))
            elif isinstance(n.value, ast.Constant) and n.value.value is None:
                out.append("None")
            else:
                out.append(ast.dump(n.value)[:80])
    return out


def _dedent_src(fn: Any) -> str:
    import textwrap

    return textwrap.dedent(inspect.getsource(fn))


def _segments(manifest: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, seg in manifest["segments"].items():
        try:
            with open(seg["path"], encoding="utf-8") as f:
                lines = f.readlines()
            a, b = seg["lines"]
            text = "".join(lines[a - 1 : b])
            out[name] = {"sha256": hashlib.sha256(text.encode()).hexdigest(), "path": seg["path"]}
        except Exception as exc:  # reported, decided on the host
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _num_sampled_device() -> dict[str, Any]:
    """Build a toy stock sampler on CUDA, run one prefill call and one decode call, report devices."""
    import numpy as np
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
    s = dg.DiffusionSampler(
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
    sp = SimpleNamespace(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=None,
        logprobs=None,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    st.add_request(0)
    s.add_request(0, 3, sp)
    s.apply_staged_writes()

    def batch(prefill: bool) -> Any:
        slots = np.array([0], dtype=np.int64)
        nlog = np.array([0 if prefill else CL])
        cu = np.concatenate([[0], np.cumsum(nlog)]).astype(np.int32)
        q = np.array([3 if prefill else CL], dtype=np.int32)
        qsl = np.concatenate([[0], np.cumsum(q)]).astype(np.int32)
        return SimpleNamespace(
            num_reqs=1,
            idx_mapping=torch.from_numpy(slots).to(dev),
            idx_mapping_np=slots,
            num_draft_tokens=0 if prefill else int(nlog.sum()),
            cu_num_logits_np=cu,
            query_start_loc_np=qsl,
            query_start_loc=torch.from_numpy(qsl).to(dev),
            num_scheduled_tokens=q,
            num_computed_prefill_tokens_np=np.zeros(1, dtype=np.int32),
            prefill_len_np=np.full(1, 3, dtype=np.int32),
        )

    out: dict[str, Any] = {}
    pre = s(torch.zeros(0, V, device=dev), batch(True))
    out["prefill"] = {
        "num_sampled": str(pre.num_sampled.device),
        "sampled_token_ids": str(pre.sampled_token_ids.device),
    }
    try:
        dec = s(torch.zeros(CL, V, device=dev), batch(False))
        out["decode"] = {
            "num_sampled": str(dec.num_sampled.device),
            "sampled_token_ids": str(dec.sampled_token_ids.device),
        }
    except Exception as exc:
        out["decode"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _regime() -> dict[str, Any]:
    """The container half of the per-row regime line (S15): versions and where each came from."""
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
    try:
        import vllm

        reg["vllm_from"] = vllm.__file__
    except Exception as exc:
        reg["vllm_from"] = f"import failed: {type(exc).__name__}"
    try:
        import pytest

        reg["pytest_from"] = pytest.__file__
    except Exception:
        reg["pytest_from"] = "absent"
    return reg


def main() -> int:
    args = json.loads(sys.argv[1])
    out: dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        out["regime"] = _regime()
    except Exception:
        out["regime"] = {"error": traceback.format_exc()}
    try:
        import vllm

        out["versions"] = {"vllm": getattr(vllm, "__version__", None)}
        dg = importlib.import_module(DG)
        stock_cls = dg.DiffusionGemmaForConditionalGeneration
        out["has_DiffusionGemmaForCausalLM"] = hasattr(dg, "DiffusionGemmaForCausalLM")
        gms = inspect.getattr_static(stock_cls, "get_model_state_cls", None)
        out["stock_get_model_state_cls_is_staticmethod"] = isinstance(gms, staticmethod)
        ms_cls = stock_cls.get_model_state_cls()
        out["model_state_cls"] = f"{ms_cls.__module__}.{ms_cls.__qualname__}"
        out["custom_sampler_return_arities"] = _return_arities(ms_cls.custom_sampler)

        sampler_cls = dg.DiffusionSampler
        sig = inspect.signature(sampler_cls.__call__)
        out["call_signature"] = [
            [p.name, None if p.default is inspect.Parameter.empty else repr(p.default)]
            for p in sig.parameters.values()
        ]
        out["attrs"] = {}
        out["missing_classes"] = {}
        for dotted, names in READS.items():
            try:
                cls = _resolve(dotted)
            except Exception as exc:
                out["missing_classes"][dotted] = f"{type(exc).__name__}: {exc}"
                continue
            out["attrs"][dotted] = {n: _has(cls, n) for n in names}
        states_cls = dg.DiffusionGemmaRequestStates
        out["negatives"] = {
            "DiffusionSampler.states": _has(sampler_cls, "states"),
            "DiffusionSampler.remove_request": _has(sampler_cls, "remove_request"),
            "accept_fields": sorted(
                n
                for c in (sampler_cls, states_cls)
                for n in (set(dir(c)) | _self_stores(c))
                if "accept" in n and not n.startswith("accepted_canvas_history")
            ),
        }
        out["segments"] = _segments(args["manifest"])
        out["modules"] = {}
        for m in args["modules"]:
            try:
                importlib.import_module(m)
                out["modules"][m] = "ok"
            except Exception as exc:
                out["modules"][m] = f"{type(exc).__name__}: {exc}"

        # criterion 12 + R9: the plugin's seam
        import astra_vllm_plugin.model as pm

        astra = pm.AstraDiffusionGemma
        own = sorted(n for n in vars(astra) if not (n.startswith("__") and n.endswith("__")))
        out["astra_own_members"] = own
        # what class creation alone puts on *any* subclass of the same bases (ABCMeta's _abc_impl,
        # typing.Protocol's _is_protocol, ...): the oracle for "members the plugin did not write"
        import types as _types

        try:
            bare = _types.new_class("Bare", astra.__bases__)
            out["bare_subclass_own_members"] = sorted(
                n for n in vars(bare) if not (n.startswith("__") and n.endswith("__"))
            )
        except Exception:
            out["bare_subclass_own_members"] = {"error": traceback.format_exc()}
        out["astra_gms_is_staticmethod"] = isinstance(
            inspect.getattr_static(astra, "get_model_state_cls", None), staticmethod
        )
        donor = importlib.import_module(
            "vllm.model_executor.models.gemma4_mm"
        ).Gemma4ForConditionalGeneration
        out["process_image_input_is_gemma4s"] = getattr(
            astra, "_process_image_input", None
        ) is getattr(donor, "_process_image_input", None)
        try:
            a, b = astra.get_model_state_cls(), astra.get_model_state_cls()
            out["astra_model_state"] = {
                "name": a.__name__,
                "subclasses_stock": issubclass(a, ms_cls),
                "cached": a is b,
            }
        except Exception:
            out["astra_model_state"] = {"error": traceback.format_exc()}
        try:
            out["num_sampled_device"] = _num_sampled_device()
        except Exception:
            out["num_sampled_device"] = {"error": traceback.format_exc()}
    except Exception:
        out["error"] = traceback.format_exc()
    print(MARKER, json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
