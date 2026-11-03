"""T-0077b criteria 2, 3, 6 and 7 on the 5090 (tier 2, ``-m gpu``, ``/launch``): ``tests/cd/cuda/test_cd_kernels_cuda.py``
inside the pinned container, **twice** (two container processes), with equal output digests.

T-0073 wrapper pattern: ``docker run --rm --gpus all --network none``, the repository read-only at ``/repo``, no
``/hostsite``, ``PYTHONDONTWRITEBYTECODE=1``, T-0067's runner (5090 preflight, ``ASTRA_TIER2_REGIME`` line),
``--noconftest -o addopts= -p no:cacheprovider``, ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1``, ``ASTRA_CD_REQUIRE_CUDA=1``.
The four dev automata of the A/B are compiled on the host (``scripts/gpu/cd_microbench.write_payloads``, which needs
the ``cd`` group: run this row with ``uv run --group cd pytest ... -m gpu``) and mounted read-only at
``/cdpayloads`` (``ASTRA_CD_PAYLOADS``); the same payload directory serves both runs.

Pass = per run: exit 0, the runner on a 5090, the inner summary exactly ``KERNELS_CUDA_PASSED`` passed and nothing
else; the ``ASTRA_T0077B_DIGEST`` set of each run complete (every case × field of ``DIGEST_CASES``, one digest
each) and the two runs' digests equal; the image's triton is 3.7.1 (reported in the regime line); the host tree
clean.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from tests import vllm_container as vc

REPO = Path(__file__).resolve().parents[2]
MODULE = "tests/cd/cuda/test_cd_kernels_cuda.py"
RUNNER = "/repo/tests/container_scripts/astra_cd_tier2_runner.py"
RUNNER_REGIME = "ASTRA_TIER2_REGIME"
PAYLOADS = "/cdpayloads"
#: rows of ``test_cd_kernels_cuda.py`` (its docstring): criterion 2 11 + 4 + 9, criterion 3 3, criterion 6 4,
#: criterion 7 3
KERNELS_CUDA_PASSED = 34
BUCKETS = (64, 128, 256, 512)
CORE = (
    "s1_b64",
    "s2_b64",
    "s31_b64_dfa",
    "s31_b64_nfa_leff",
    "s32_b32",
    "s91_b128_nfa",
    "s91_b128_dfa_leff",
    "s205_b256",
    "last_state_s31_b64",
    "last_state_s91_b128",
    "empty_s31_b64",
)
TAIL = (
    "mixed_sign",
    "cancellation",
    "span_1e-30_1e3",
    "all_zero_row",
    "npot_forbidden_row",
    "nan_row",
    "subnormal",
    "contraction",
    "wide_v262144",
)
FIELDS = ("log_z", "q", "entropy", "log_a", "log_b")
EXPECTED_DIGESTS = frozenset(
    f"{case}.{f}"
    for case in (
        *(f"core:{n}" for n in CORE),
        *(f"dev:b{b}" for b in BUCKETS),
        *(f"tail:{n}" for n in TAIL),
    )
    for f in FIELDS
)
_DIGEST = re.compile(r"ASTRA_T0077B_DIGEST (\S+) ([0-9a-f]{64})")
_REGIME = re.compile(r"ASTRA_T0077B_REGIME (\{.*\})")


def _host(argv: list[str]) -> str:
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)
    assert proc.returncode == 0, f"{argv} exited {proc.returncode}: {proc.stderr}"
    return proc.stdout.strip()


def digests(stdout: str) -> dict[str, str]:
    """``name -> sha256`` from one run's output; a name printed twice with different digests fails."""
    out: dict[str, str] = {}
    for name, h in _DIGEST.findall(stdout):
        assert out.get(name, h) == h, f"{name} printed with two digests in one run"
        out[name] = h
    return out


def _write_payloads(dest: Path) -> dict[str, Any]:
    gpu = str(REPO / "scripts" / "gpu")
    if gpu not in sys.path:
        sys.path.insert(0, gpu)
    try:
        import cd_microbench as mb

        manifest: dict[str, Any] = mb.write_payloads(dest)
    except (ImportError, SystemExit) as exc:
        pytest.fail(
            f"cannot compile the dev automata on the host ({exc!r}); run this row with "
            "`uv run --group cd pytest tests/contract/test_vllm_cd_kernels_gpu.py -m gpu`"
        )
    return manifest


def _docker_argv(image: str, payloads: Path) -> list[str]:
    argv = ["docker", "run", "--rm", "--gpus", "all", "--network", "none"]
    argv += ["-v", f"{REPO}:/repo:ro", "-v", f"{payloads}:{PAYLOADS}:ro", "-w", "/repo"]
    env = {
        "PYTHONPATH": "/repo:/repo/plugin/astra_vllm_plugin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "ASTRA_CD_TESTS": "1",
        "ASTRA_CD_REQUIRE_CUDA": "1",
        "ASTRA_CD_DEVICE": "cuda",
        "ASTRA_CD_STOCK": "image",
        "ASTRA_CD_PAYLOADS": PAYLOADS,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    for k, v in env.items():
        argv += ["-e", f"{k}={v}"]
    args = ["-rA", "-p", "no:cacheprovider", "--noconftest", "-o", "addopts=", MODULE]
    return [*argv, "--entrypoint", "python3", image, RUNNER, *args]


@pytest.mark.gpu
def test_kernels_module_twice_on_cuda_with_equal_digests() -> None:
    from tests.pytest_summary import require_exact

    image = vc.require_container()
    host = {
        "image": image,
        "driver": _host(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "host_head": _host(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "host_tree_dirty": bool(_host(["git", "-C", str(REPO), "status", "--porcelain"])),
    }
    runs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="t0077b-cdpayloads-") as tmp:
        manifest = _write_payloads(Path(tmp))
        for i in (1, 2):
            proc = vc._run(_docker_argv(image, Path(tmp)), timeout=7200)
            inner = next(
                (
                    json.loads(ln[len(RUNNER_REGIME) + 1 :])
                    for ln in proc.stdout.splitlines()
                    if ln.startswith(RUNNER_REGIME + " ")
                ),
                None,
            )
            reg = _REGIME.search(proc.stdout)
            kern = json.loads(reg.group(1)) if reg else None
            print(
                f"T-0077b run {i} regime: "
                + json.dumps({**host, "inner": inner, "kernels": kern}, sort_keys=True)
            )
            print(f"--- container pytest output, run {i} ({MODULE}) ---\n{proc.stdout[-40000:]}")
            runs.append({"proc": proc, "inner": inner, "kernels": kern})
    automata = {b: (m["id"], m["n_states"]) for b, m in manifest.items()}
    print(f"T-0077b dev automata: {automata}")

    sets = []
    for i, r in enumerate(runs, 1):
        proc, inner = r["proc"], r["inner"]
        assert inner is not None, f"run {i}: no {RUNNER_REGIME} line\n{proc.stdout[-8000:]}"
        assert inner["cuda_available"] and "5090" in inner["device"], f"run {i}: {inner}"
        assert proc.returncode == 0, f"run {i}:\n{proc.stdout[-20000:]}{proc.stderr[-4000:]}"
        require_exact(proc.stdout, KERNELS_CUDA_PASSED)
        assert r["kernels"] is not None and r["kernels"]["triton"] == "3.7.1", (
            f"run {i}: image triton {r['kernels']}"
        )
        d = digests(proc.stdout)
        assert set(d) == EXPECTED_DIGESTS, (
            f"run {i}: digest set missing {sorted(EXPECTED_DIGESTS - set(d))[:10]}, "
            f"extra {sorted(set(d) - EXPECTED_DIGESTS)[:10]}"
        )
        sets.append(d)
    differ = sorted(k for k in EXPECTED_DIGESTS if sets[0][k] != sets[1][k])
    assert differ == [], f"output bits differ between the two processes: {differ}"
    assert not host["host_tree_dirty"], (
        "launch evidence from a dirty tree (CLAUDE.md: never against a dirty tree)"
    )


def test_digest_parser_and_expected_set() -> None:
    """Host-side control (fast suite, no container): the parser reads lines anywhere in a line, refuses two digests
    for one name, and the expected set is 24 cases × 5 fields. The ``gpu`` row above is the only one tier 2 runs
    (host summary exactly "1 passed")."""
    h1, h2 = "a" * 64, "b" * 64
    out = f"x\nASTRA_T0077B_DIGEST core:s1_b64.q {h1}\n.. ASTRA_T0077B_DIGEST tail:nan_row.q {h2}\n"
    assert digests(out) == {"core:s1_b64.q": h1, "tail:nan_row.q": h2}
    with pytest.raises(AssertionError):
        digests(out + f"ASTRA_T0077B_DIGEST core:s1_b64.q {h2}\n")
    assert len(EXPECTED_DIGESTS) == 120
