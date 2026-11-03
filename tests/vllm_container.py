"""Run a probe script inside the pinned vLLM container.

The host in this project has neither ``vllm`` nor ``torch`` -- they live only in the container
pinned by ``services/versions.lock`` and named in ``services/vllm.sh``. Tests that need to look
at real vLLM classes therefore ship a script into the container and read JSON back out. This is
the single place that knows how to do that; change the invocation here, not in the tests.

Nothing here boots an engine or loads weights: the container runs ``python3 <script>`` and exits.
Still ``gpu``-marked, because it needs the image and (for platform detection at import time) the
card.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
VLLM_SH = REPO / "services" / "vllm.sh"
SCRIPTS = Path(__file__).resolve().parent / "container_scripts"
PLUGIN_DIR = REPO / "plugin" / "astra_vllm_plugin"
PROBE_MARKER = "ASTRA_PROBE_JSON"


def pinned_image() -> str:
    """The container tag ``services/vllm.sh`` defaults to, so tests cannot drift off the pin."""
    text = VLLM_SH.read_text()
    match = re.search(r"^CONTAINER=\$\{ASTRA_VLLM_CONTAINER:-([^}]+)\}", text, re.MULTILINE)
    if match is None:
        raise AssertionError(
            f"{VLLM_SH} no longer defines CONTAINER=${{ASTRA_VLLM_CONTAINER:-<image>}}; "
            "tests/vllm_container.py reads the pinned image from there"
        )
    return match.group(1).strip()


@lru_cache(maxsize=1)
def _docker_prefix() -> list[str] | None:
    """``docker`` if the daemon answers, ``sg docker -c`` if only the group grants access."""
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0:
            return ["docker"]
    except (OSError, subprocess.TimeoutExpired):
        return None
    try:
        probe = subprocess.run(
            ["sg", "docker", "-c", "docker info"], capture_output=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return ["sg", "docker", "-c"] if probe.returncode == 0 else None


def _run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    prefix = _docker_prefix()
    assert prefix is not None
    cmd = argv if prefix == ["docker"] else [*prefix, shlex.join(argv)]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def require_container() -> str:
    """Skip (never error) when this machine cannot run the pinned image."""
    image = pinned_image()
    if _docker_prefix() is None:
        pytest.skip("docker daemon not reachable (not the 5090 box)")
    if _run(["docker", "image", "inspect", image], timeout=60).returncode != 0:
        pytest.skip(f"pinned vLLM image {image} is not present locally (no pull in tests)")
    return image


def run_probe(script: str, *, timeout: int = 600) -> dict[str, Any]:
    """Run ``tests/container_scripts/<script>`` in the container; return the JSON it printed.

    Read-only mounts, ``--network none``: the probe imports and inspects, it must not reach the
    network (docs/02 §9 loopback rule) or touch the tree.
    """
    image = require_container()
    argv = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--network",
        "none",
        "-v",
        f"{SCRIPTS}:/astra_scripts:ro",
        "-v",
        f"{PLUGIN_DIR}:/plugin:ro",
        "-e",
        "PYTHONPATH=/plugin",
        "-e",
        "HF_HUB_OFFLINE=1",
        "--entrypoint",
        "python3",
        image,
        f"/astra_scripts/{script}",
    ]
    proc = _run(argv, timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith(PROBE_MARKER):
            return json.loads(line[len(PROBE_MARKER) :])
    raise AssertionError(
        f"{script} printed no {PROBE_MARKER} line in {image}\n"
        f"exit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
