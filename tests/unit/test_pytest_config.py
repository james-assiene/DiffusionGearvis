"""T-0046: ``gpu``-marked rows never run by accident.

On 2026-09-15 a test-author ran ``uv run pytest tests/eval -q`` as a "skip check", the live engine
answered on 8001, and twelve generations ran without a ``/launch`` review (T-0044 ruling A3). The
guard hook keys on an explicit ``-m gpu``; a path-only invocation slipped past it. The fix is in
``pyproject.toml``: ``addopts`` carries ``-m 'not gpu'``, so a path-only run deselects every gpu
row, and a reviewed run states ``-m gpu`` explicitly (a later command-line ``-m`` replaces the
default one — pinned below by running pytest itself, not by reading the option).

Regime: no GPU, no services; the collect-only subprocesses take well under a second each.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _collect(*args: str) -> str:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return out.stdout + out.stderr


def _counts(text: str) -> tuple[int, int]:
    """(selected, deselected) from pytest's collect-only summary line."""
    selected = deselected = 0
    m = re.search(r"(\d+)(?:/\d+)? tests? collected", text)
    if m:
        selected = int(m.group(1))
    m = re.search(r"(\d+) deselected", text)
    if m:
        deselected = int(m.group(1))
    return selected, deselected


def test_addopts_deselects_gpu_rows_by_default() -> None:
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text())
    addopts = cfg["tool"]["pytest"]["ini_options"]["addopts"]
    assert "-m 'not gpu'" in addopts or '-m "not gpu"' in addopts, (
        f"T-0046: pyproject addopts must carry -m 'not gpu' so a path-only run cannot spend GPU; "
        f"addopts is {addopts!r}"
    )


def test_a_path_only_run_of_the_eval_directory_selects_nothing() -> None:
    """The exact invocation that spent GPU on 2026-09-15 must now collect zero rows."""
    eval_dir = REPO / "tests" / "eval"
    if not any(eval_dir.glob("test_*.py")):
        return  # nothing gpu-marked there yet; the addopts row above still pins the default
    selected, deselected = _counts(_collect("tests/eval"))
    assert selected == 0 and deselected > 0, (
        f"T-0046: `pytest tests/eval` selected {selected} row(s) (deselected {deselected}); every "
        "row there is gpu-marked and must be deselected without an explicit -m gpu"
    )


def test_an_explicit_marker_still_selects_the_gpu_rows() -> None:
    """A /launch-reviewed run says ``-m gpu`` and the default must not veto it."""
    text = _collect("tests/contract/test_speech_clients_contract.py", "-m", "gpu")
    selected, _ = _counts(text)
    assert selected > 0, (
        "T-0046: an explicit `-m gpu` on the command line must override the addopts default; "
        f"pytest collected nothing:\n{text[-600:]}"
    )
