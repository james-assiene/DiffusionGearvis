"""T-0081 tier 2's locked script ``scripts/locked/t0081_seed.sh``, hermetically (fast suite, no GPU, no engine).

The script is copied into a throwaway git repository and run with a stub ``PATH`` directory that shadows every
command able to reach hardware, containers or the network (``uv``, ``docker``, ``nvidia-smi``, ``curl``, ``astra``,
``sudo``, ``systemctl``, ``wget``); each stub records its invocation and exits with a code the row chooses.

Pinned (the repository's locked-script conventions, e.g. ``scripts/locked/t0077c_ab.sh``; the task file does not
spell them out -- see the report's ambiguity list):
- a dirty tree is refused with exit 2 before any stubbed command runs;
- the driver's failure is the script's failure: the exit code is not swallowed by ``| tee`` (real exit-code capture);
- a driver that exits 0 but leaves no evidence does not read as success;
- a second run at the same commit refuses (exit 2) instead of overwriting the first run's log.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "locked" / "t0081_seed.sh"
STUBS = ("uv", "docker", "nvidia-smi", "curl", "astra", "sudo", "systemctl", "wget")


def _script() -> Path:
    assert SCRIPT.exists(), f"T-0081 not implemented: {SCRIPT.relative_to(REPO)} does not exist"
    return SCRIPT


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True,
        capture_output=True,
    )


def _sandbox(tmp_path: Path, stub_exit: int) -> tuple[Path, dict[str, str], Path]:
    repo = tmp_path / "repo"
    (repo / "scripts" / "locked").mkdir(parents=True)
    shutil.copy2(_script(), repo / "scripts" / "locked" / "t0081_seed.sh")
    driver = REPO / "scripts" / "gpu" / "cd_seed_ab.py"
    if driver.exists():
        (repo / "scripts" / "gpu").mkdir(parents=True)
        shutil.copy2(driver, repo / "scripts" / "gpu" / "cd_seed_ab.py")
    (repo / ".gitignore").write_text("logs/\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    for name in STUBS:
        stub = bin_dir / name
        stub.write_text(f'#!/usr/bin/env bash\necho "{name} $*" >> "{calls}"\nexit {stub_exit}\n')
        stub.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    return repo, env, calls


def _run(repo: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/locked/t0081_seed.sh"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _called(calls: Path) -> list[str]:
    return calls.read_text().splitlines() if calls.exists() else []


def test_script_parses() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(_script())], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr


def test_a_dirty_tree_is_refused_before_anything_runs(tmp_path: Path) -> None:
    repo, env, calls = _sandbox(tmp_path, 0)
    (repo / "scripts" / "locked" / "t0081_seed.sh").write_text(
        (repo / "scripts" / "locked" / "t0081_seed.sh").read_text() + "\n# dirty\n"
    )
    proc = _run(repo, env)
    assert proc.returncode == 2, (proc.returncode, proc.stdout[-2000:], proc.stderr[-2000:])
    assert "REFUSE" in proc.stdout + proc.stderr
    assert _called(calls) == [], f"commands ran before the refusal: {_called(calls)}"


def test_the_drivers_failure_is_the_scripts_failure(tmp_path: Path) -> None:
    repo, env, calls = _sandbox(tmp_path, 1)
    proc = _run(repo, env)
    assert proc.returncode != 0, (
        "a failing driver was reported as success (exit code lost in a pipe?)"
    )
    assert "PASS" not in proc.stdout.replace("PASSED", ""), proc.stdout[-2000:]
    assert _called(calls), "fixture: the script ran none of the stubbed commands"


def test_exit_0_without_evidence_is_not_success(tmp_path: Path) -> None:
    repo, env, calls = _sandbox(tmp_path, 0)
    proc = _run(repo, env)
    assert proc.returncode != 0, "exit 0 with no evidence written"
    assert _called(calls), "fixture: the script ran none of the stubbed commands"


def test_a_second_run_at_the_same_commit_refuses(tmp_path: Path) -> None:
    repo, env, calls = _sandbox(tmp_path, 1)
    _run(repo, env)
    logs = (
        sorted(p for p in (repo / "logs").rglob("*") if p.is_file())
        if (repo / "logs").exists()
        else []
    )
    assert logs, "the first run left no log under logs/"
    before = {p: p.read_bytes() for p in logs}
    n_calls = len(_called(calls))
    proc = _run(repo, env)
    assert proc.returncode == 2 and "REFUSE" in proc.stdout + proc.stderr, (
        proc.returncode,
        proc.stderr[-1000:],
    )
    assert len(_called(calls)) == n_calls, "the refused second run still ran commands"
    assert all(p.read_bytes() == b for p, b in before.items()), (
        "the first run's log was overwritten"
    )
