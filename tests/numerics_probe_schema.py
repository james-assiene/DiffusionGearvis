"""The output contract of the T-0073 numerics probe (``tests/container_scripts/astra_cd_numerics_probe.py``).

Torch-free and numpy-free on purpose: the host gpu wrapper (``tests/contract/test_vllm_cd_numerics_gpu.py``) runs
in the main venv, which has neither, and must validate what the container printed. The probe imports the marker
and the section list from here, so the producer and both consumers (the host wrapper and the CPU unit rows in
``tests/cd/unit/test_cd_numerics_probe_cpu.py``) cannot drift apart.

The probe prints exactly one line ``ASTRA_T0073_PROBE_JSON <json>``; everything else on stdout is noise.
"""

from __future__ import annotations

import json
from typing import Any

MARKER = "ASTRA_T0073_PROBE_JSON"
SECTIONS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
#: 5 MiB cap on the serialised payload (docs/tasks/T-0073.md, P7)
CAP_BYTES = 5 * 1024 * 1024

#: keys every non-errored section must carry
SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "P1": ("ops",),
    "P2": ("items",),
    "P3": ("pairs", "chain"),
    "P4": ("instances", "stage_contributions", "arms", "oracle_controls", "staged_equals_plugin"),
    "P5": (
        "instances",
        "repeats",
        "deterministic_segments",
        "stage_swap",
        "stage_divergence",
        "controls",
        "staged_equals_plugin",
    ),
    "P6": (
        "fixture",
        "empty_fixture",
        "plugin_outputs",
        "stages",
        "first_divergent_stage",
        "dumps",
        "staged_equals_plugin",
    ),
    "P7": ("timings",),
}

#: the pass-2 dump of P6 must reach every intermediate from log_transfer through q~ and total (task, P6)
P6_STAGES = (
    "class_mass",
    "log_transfer",
    "up_sweep",
    "pass1",
    "gauge",
    "w_edge",
    "leaves",
    "tree2",
    "a2b2",
    "joint",
    "g",
    "coef",
    "q_raw",
    "total",
    "q",
    "entropy",
)

REGIME_KEYS = (
    "probe",
    "argv",
    "device",
    "small",
    "torch",
    "torch_cuda",
    "python",
    "numpy",
    "vllm_dist",
    "cuda_available",
    "device_name",
    "fp32_matmul_precision_at_start",
    "cuda_matmul_allow_tf32_at_start",
    "source_commit_of_staged_copies",
    "forbidden_modules_loaded",
    "section_seconds",
    "total_seconds",
)

#: modules the probe must never import (the image's Python 3.12 cannot load the host's cp311 scipy; vllm is
#: not needed and would boot platform detection)
FORBIDDEN_MODULES = ("scipy", "vllm")


def extract(stdout: str) -> dict[str, Any]:
    """The JSON payload printed under ``MARKER``; ``AssertionError`` when there is none or more than one."""
    lines = [ln for ln in stdout.splitlines() if ln.startswith(MARKER + " ")]
    if len(lines) != 1:
        raise AssertionError(
            f"expected exactly one {MARKER} line, found {len(lines)}\n{stdout[-4000:]}"
        )
    payload = json.loads(lines[0][len(MARKER) + 1 :])
    if not isinstance(payload, dict):
        raise AssertionError(f"{MARKER} payload is not a JSON object")
    return payload


def nested_errors(obj: Any, path: str = "") -> list[str]:
    """Paths of every dict carrying an ``"error"`` key strictly below ``obj`` (per-arm errors a section caught)."""
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            sub = f"{path}/{k}"
            if isinstance(v, dict) and "error" in v:
                out.append(sub)
            out += nested_errors(v, sub)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            sub = f"{path}[{i}]"
            if isinstance(v, dict) and "error" in v:
                out.append(sub)
            out += nested_errors(v, sub)
    return out


def problems(payload: dict[str, Any], *, allow_skipped: bool = False) -> list[str]:
    """Everything that makes a payload unusable as launch evidence; ``[]`` when it is complete.

    A section is a problem when it is missing, recorded ``{"error": ...}``, skipped (unless ``allow_skipped``),
    lacks one of its keys, or holds an ``{"error": ...}`` anywhere below it (an arm that raised and was caught); the staged recomputations must reproduce the plugin bit for bit on CPU.
    """
    out: list[str] = []
    regime = payload.get("regime")
    if not isinstance(regime, dict):
        out.append("regime block missing")
    else:
        out += [f"regime.{k} missing" for k in REGIME_KEYS if k not in regime]
        bad = regime.get("forbidden_modules_loaded")
        if bad:
            out.append(f"forbidden modules imported: {bad}")
    v = payload.get("verdicts")
    if not isinstance(v, dict) or any(k not in v for k in ("a", "b", "c")):
        out.append("verdicts block missing (a, b, c)")
    sections = payload.get("sections")
    if not isinstance(sections, dict):
        return [*out, "sections block missing"]
    for name in SECTIONS:
        sec = sections.get(name)
        if not isinstance(sec, dict):
            out.append(f"{name} missing")
            continue
        if "error" in sec:
            out.append(f"{name} errored: {str(sec['error']).splitlines()[-1:]}")
            continue
        if "skipped" in sec:
            if not allow_skipped:
                out.append(f"{name} skipped: {sec['skipped']}")
            continue
        out += [f"{name}.{k} missing" for k in SECTION_KEYS[name] if k not in sec]
        out += [f"{name} nested error at {p}" for p in nested_errors(sec)]
        if name == "P6" and isinstance(sec.get("stages"), dict):
            for dev, stages in sec["stages"].items():
                missing = [s for s in P6_STAGES if s not in stages]
                if missing:
                    out.append(f"P6.stages[{dev}] lacks {missing}")
        if name in ("P4", "P5", "P6"):
            eq = sec.get("staged_equals_plugin", {})
            cpu = eq.get("cpu") if isinstance(eq, dict) else None
            if cpu is not True:
                out.append(
                    f"{name}: staged recomputation does not reproduce the plugin on CPU: {cpu}"
                )
    return out
