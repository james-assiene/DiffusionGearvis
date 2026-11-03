"""T-0067 criterion 11 (tier 2, ``-m gpu``, ``/launch``, container only, no engine boot).

The CD wrapper is coupled to v0.29.0 internals; this pins the coupling to the image. A version bump reads as
"the CD coupling drifted from vLLM X". The vendored CPU oracle's step must equal the image's (sha256 and
parameter names), every attribute the CD code reads must exist, and ``custom_sampler`` must return a 2-tuple.

Criterion 12 (CUDA repeats: criterion 4 with the real compiled step, T-0065 criteria 1-5 and T-0066 criteria
1, 4, 5 under ``torch.set_float32_matmul_precision("high")``, peak added memory ≤ 1.5 GiB at CL 256) lives in
``tests/contract/test_vllm_cd_cuda_gpu.py``, which runs ``tests/cd/cuda/test_cd_cuda_repeats.py`` and
``tests/cd/unit/test_cd_mixed_batch_identity.py`` inside the same pinned image.
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests import vllm_container as vc

pytestmark = pytest.mark.gpu

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "tests" / "fixtures" / "vllm_v0_29_0"
CD_DIR = REPO / "plugin" / "astra_vllm_plugin" / "astra_vllm_plugin" / "cd"
DG = "vllm.model_executor.models.diffusion_gemma"
#: what the CD code may read on each upstream class, keyed by the class's dotted path in the pinned image
#: (v0.29.0 source: ``DiffusionSampler.req_states`` is the runner's ``vllm.v1.worker.gpu.states.RequestState``
#: -- ``diffusion_gemma.py`` imports it and writes ``self.req_states.draft_tokens`` -- and the sampler's
#: ``input_batch`` is the ``vllm.v1.worker.gpu.input_batch.InputBatch`` dataclass)
READS = {
    f"{DG}.DiffusionSampler": [
        "sampling_states",
        "req_states",
        "diffusion_states",
        "canvas_length",
        "vocab_size",
        "add_request",
        "apply_staged_writes",
        "penalties_state",
    ],
    f"{DG}.DiffusionGemmaRequestStates": [
        "canvas",
        "argmax_canvas",
        "step",
        "is_encoder_phase",
        "confident",
        "self_conditioning_embeds",
        "accepted_canvas_history",
        "accepted_canvas_history_len",
        "max_denoising_steps",
        "stability_threshold",
        "canvas_length",
        "max_num_reqs",
        "device",
    ],
    "vllm.v1.worker.gpu.sample.output.SamplerOutput": [
        "sampled_token_ids",
        "num_sampled",
        "num_rejected",
        "logprobs_tensors",
    ],
    "vllm.v1.worker.gpu.states.RequestState": ["draft_tokens"],
    "vllm.v1.worker.gpu.input_batch.InputBatch": [
        "num_reqs",
        "num_draft_tokens",
        "idx_mapping_np",
        "cu_num_logits_np",
    ],
}


def _run(script: str, arg: str) -> dict[str, Any]:
    image = vc.require_container()
    argv = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--network",
        "none",
        "-v",
        f"{vc.SCRIPTS}:/astra_scripts:ro",
        "--entrypoint",
        "python3",
        image,
        f"/astra_scripts/{script}",
        arg,
    ]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=600, check=False)
    for line in proc.stdout.splitlines():
        if line.startswith(vc.PROBE_MARKER):
            return json.loads(line[len(vc.PROBE_MARKER) :])  # type: ignore[no-any-return]
    raise AssertionError(f"no probe line\n{proc.stdout}\n{proc.stderr}")


@pytest.fixture(scope="module")
def probe() -> dict[str, Any]:
    facts = _run("astra_cd_probe.py", json.dumps(READS))
    if "error" in facts:
        pytest.fail(f"probe failed inside the pinned container:\n{facts['error']}")
    return facts


def _vendored_params() -> list[str]:
    tree = ast.parse((FIXTURE / "diffusion_gemma_step.py").read_text())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_compiled_sample_step"
    )
    return [a.arg for a in fn.args.args] + [a.arg for a in fn.args.kwonlyargs]


def test_step_matches_the_vendored_oracle(probe: dict[str, Any]) -> None:
    man = json.loads((FIXTURE / "manifest.json").read_text())
    v = probe["versions"]["vllm"]
    assert probe["step"]["sha256"] == man["segments"]["_compiled_sample_step"]["sha256"], (
        f"the CD coupling drifted from vLLM {v}: _compiled_sample_step's source changed"
    )
    assert probe["step"]["params"] == _vendored_params(), (
        f"the CD coupling drifted from vLLM {v}: parameters"
    )


def test_every_attribute_cd_reads_exists(probe: dict[str, Any]) -> None:
    v = probe["versions"]["vllm"]
    gone = probe["missing_classes"]
    assert gone == {}, "the CD coupling drifted from vLLM {}: {}".format(
        v, "; ".join(f"class {c} not found in vLLM {v} ({why})" for c, why in sorted(gone.items()))
    )
    assert set(probe["attrs"]) == set(READS), f"probe reported {sorted(probe['attrs'])}"
    missing = {c: [n for n, ok in m.items() if not ok] for c, m in probe["attrs"].items()}
    missing = {c: ns for c, ns in missing.items() if ns}
    assert missing == {}, (
        f"the CD coupling drifted from vLLM {probe['versions']['vllm']}: {missing}"
    )


def test_custom_sampler_returns_a_two_tuple(probe: dict[str, Any]) -> None:
    ar = probe["custom_sampler"]["return_tuple_arities"]
    assert ar and all(a == 2 for a in ar), (
        f"the CD coupling drifted from vLLM {probe['versions']['vllm']}: custom_sampler returns {ar}"
    )
