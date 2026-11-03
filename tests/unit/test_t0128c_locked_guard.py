"""T-0128c O-C4: the locked smoke script's merge guard must not depend on pipe timing.

The real ``scripts/locked/t0128c_smoke.sh`` is copied into a throwaway git repo (it cds to its own
repo's toplevel) whose ``git log --merges --format='%H %s'`` output is larger than a 64 KB pipe, with
the group A and B merges at the top of the log. ``docker``, ``make``, ``ss``, ``uv``, ``timeout`` and
``sg`` are stubs on PATH that record the call and fail, so nothing real can start.

The repo carries a tag named ``806da12`` (the pin the script diffs ``services/vllm.sh`` against) and a
later commit changing ``services/vllm.sh``: a script that passes the merge guard therefore refuses at
the pin (exit 2) before it reaches ``ss``, ``docker``, ``make`` or the harness. Without the tag
``git diff 806da12..HEAD`` fails inside a command substitution, which ``set -e`` ignores, and the
script would go on to call the ``ss``/``docker`` stubs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "locked" / "t0128c_smoke.sh"
STUBS = ("docker", "make", "ss", "uv", "timeout", "sg")
FILLER_MERGES = 400
FILLER_SUBJECT_PAD = "x" * 200
RUNS = 50

MERGE_A = "Merge T-0128c group A: canvas relay (O-C4 fixture)"
MERGE_B = "Merge T-0128c group B: denoising page (O-C4 fixture)"
# Replacements that must not satisfy the guard (a different group letter).
NOT_A = "Merge T-0128c group C: stands in for the missing group A merge"
NOT_B = "Merge T-0128c group C: stands in for the missing group B merge"

MERGE_REFUSAL = "no 'Merge T-0128c group"
PIN_REFUSAL = "REFUSED: services/vllm.sh changed since 806da12"


def _blob(data: bytes) -> bytes:
    return b"data %d\n" % len(data) + data + b"\n"


def _fast_import_stream(script_bytes: bytes, top_subjects: list[str]) -> bytes:
    """A fast-import stream: base (tagged 806da12), a vllm.sh change, filler merges, then top merges."""
    out: list[bytes] = []
    t = 1_700_000_000
    mark = 0

    def commit(ref: str, msg: str, parents: list[int], files: dict[str, bytes]) -> int:
        nonlocal t, mark
        mark += 1
        t += 1
        out.append(b"commit " + ref.encode() + b"\n")
        out.append(b"mark :%d\n" % mark)
        out.append(b"committer Test <test@example.invalid> %d +0000\n" % t)
        out.append(_blob(msg.encode()))
        if parents:
            out.append(b"from :%d\n" % parents[0])
            for p in parents[1:]:
                out.append(b"merge :%d\n" % p)
        for path, data in files.items():
            out.append(b"M 100644 inline " + path.encode() + b"\n")
            out.append(_blob(data))
        return mark

    base = commit(
        "refs/heads/main",
        "base",
        [],
        {
            "services/vllm.sh": b"# vllm boot at the pin\n",
            "scripts/locked/t0128c_smoke.sh": script_bytes,
        },
    )
    out.append(b"tag 806da12\nfrom :%d\ntagger Test <test@example.invalid> %d +0000\n" % (base, t))
    out.append(_blob(b"the boot-regime pin"))
    head = commit(
        "refs/heads/main",
        "change vllm.sh after the pin",
        [base],
        {"services/vllm.sh": b"# changed\n"},
    )
    subjects = [
        f"Merge filler-{i:04d}: {FILLER_SUBJECT_PAD}" for i in range(FILLER_MERGES)
    ] + top_subjects
    for i, subject in enumerate(subjects):
        side = commit(
            "refs/heads/side", f"side {i}", [head], {f"side/{i % 7}.txt": f"{i}\n".encode()}
        )
        head = commit("refs/heads/main", subject, [head, side], {})
    out.append(b"done\n")
    return b"".join(out)


def _git(repo: Path, *args: str, env: dict[str, str], stdin: bytes | None = None) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], input=stdin, env=env, capture_output=True
    )
    assert proc.returncode == 0, f"git {args[0]} failed: {proc.stderr.decode(errors='replace')}"
    return proc.stdout


def _make_env(tmp: Path) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    calls = tmp / "stub_calls.log"
    calls.touch()
    for name in STUBS:
        stub = bin_dir / name
        stub.write_text(f'#!/bin/sh\necho "{name} $*" >> "{calls}"\nexit 1\n')
        stub.chmod(0o755)
    git = shutil.which("git")
    assert git is not None
    path = os.pathsep.join([str(bin_dir), str(Path(git).parent), "/usr/bin", "/bin"])
    home = tmp / "home"
    home.mkdir()
    env = {
        "PATH": path,
        "HOME": str(home),
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    return env, bin_dir, calls


def _build_repo(tmp: Path, top_subjects: list[str], env: dict[str, str]) -> Path:
    repo = tmp / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main", env=env)
    _git(
        repo,
        "fast-import",
        "--quiet",
        env=env,
        stdin=_fast_import_stream(SCRIPT.read_bytes(), top_subjects),
    )
    _git(repo, "reset", "-q", "--hard", "main", env=env)
    assert _git(repo, "status", "--porcelain", env=env) == b""
    log = _git(repo, "log", "--merges", "--format=%H %s", "HEAD", env=env)
    assert len(log) > 64 * 1024, len(log)  # larger than a Linux pipe buffer
    first = log.decode().splitlines()[: len(top_subjects)]
    assert [line.split(" ", 1)[1] for line in first] == list(reversed(top_subjects))
    return repo


def _run(repo: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(repo / "scripts" / "locked" / "t0128c_smoke.sh")],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_guard_accepts_both_merges_over_many_runs_without_pipe_race(tmp_path: Path) -> None:
    """Row 1: both merges at the top of a > 64 KB merge log: the merge refusal never appears."""
    env, _, calls = _make_env(tmp_path)
    repo = _build_repo(tmp_path, [MERGE_A, MERGE_B], env)
    bad: list[tuple[int, int, str]] = []
    for i in range(RUNS):
        proc = _run(repo, env)
        if MERGE_REFUSAL in proc.stderr or proc.returncode != 2 or PIN_REFUSAL not in proc.stderr:
            bad.append((i, proc.returncode, proc.stderr.strip()))
    assert not bad, (
        f"{len(bad)}/{RUNS} runs did not stop at the services/vllm.sh pin; first: {bad[0]}"
    )
    assert calls.read_text() == "", f"stubs were called: {calls.read_text()!r}"


@pytest.mark.parametrize(
    ("top", "group"),
    [([NOT_A, MERGE_B], "A"), ([MERGE_A, NOT_B], "B")],
    ids=["without-group-A", "without-group-B"],
)
def test_guard_refuses_when_a_group_merge_is_absent(
    tmp_path: Path, top: list[str], group: str
) -> None:
    """Rows 2 and 3: a missing group merge still gets the merge refusal with exit 2."""
    env, _, calls = _make_env(tmp_path)
    repo = _build_repo(tmp_path, top, env)
    for _ in range(5):
        proc = _run(repo, env)
        assert proc.returncode == 2, proc.stderr
        expected = (
            f"REFUSED: no 'Merge T-0128c group {group}' merge commit reachable from HEAD; "
            "the smoke runs after both group merges"
        )
        assert expected in proc.stderr, proc.stderr
        other = "B" if group == "A" else "A"
        assert f"group {other}'" not in proc.stderr, proc.stderr
        assert PIN_REFUSAL not in proc.stderr
    assert calls.read_text() == "", f"stubs were called: {calls.read_text()!r}"
