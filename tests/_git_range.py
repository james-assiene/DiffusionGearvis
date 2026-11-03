"""Closed-range git helpers for scope guards (T-0094).

A finished task's scope guard compares two fixed commits: ``BASE`` = ``git merge-base M^1 M^2`` and ``TIP`` = ``M^2``
of the task's merge commit ``M``. Never ``main``, ``origin/*``, ``HEAD`` or the working tree: those move, so a guard
over them is either vacuous (on main ``merge-base(HEAD, main)`` is ``HEAD``) or goes red when a later task legitimately
edits a guarded file.

``LATER`` (6a8600b, main when T-0094 was planned) is the literal "later" commit of the power rows: the same predicate
over ``BASE..LATER`` must report a change a later task is known to have made, so a guard that cannot see anything
fails there. If a commit is absent (a shallow clone) the guard skips naming the SHA; it never passes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
#: main at T-0094's fork point; every inventory merge commit is its ancestor (T-0094 criterion 3).
LATER = "6a8600b"
#: main when T-0094b was planned (T-0094's merge). The "later" commit for guards whose merge is not an ancestor of
#: ``LATER`` (T-0096) or whose known later change came after it (T-0094's e2fefde edit of a T-0069 fenced file).
LATER_B = "2c584d9"


def _git(*args: str, repo: Path = REPO) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )


def has_commit(sha: str, *, repo: Path = REPO) -> bool:
    return _git("cat-file", "-e", f"{sha}^{{commit}}", repo=repo).returncode == 0


#: ``why`` for rows whose premise is the repository's merge history, absent from the squashed public release (one
#: commit, no history: T-0157 §Recipe; release re-pin, James 2026-11-01).
NO_HISTORY = (
    "the repository's history is absent (a shallow clone or the squashed release, T-0157 §Recipe)"
)


def require_commits(*shas: str, repo: Path = REPO, why: str | None = None) -> None:
    """Skip (never pass) when any of ``shas`` is not in the object store, naming it (and ``why``, if given)."""
    for sha in shas:
        if not has_commit(sha, repo=repo):
            if why is None:
                pytest.skip(
                    f"commit {sha} is absent from this clone (shallow?); the guard cannot run"
                )
            pytest.skip(f"commit {sha} is absent: {why}; the row cannot run")


def is_ancestor(a: str, b: str, *, repo: Path = REPO) -> bool:
    proc = _git("merge-base", "--is-ancestor", a, b, repo=repo)
    assert proc.returncode in (0, 1), f"git merge-base --is-ancestor {a} {b}: {proc.stderr}"
    return proc.returncode == 0


def prove_range(base: str, tip: str, merge: str, *, later: str = LATER) -> None:
    """T-0094 criterion 3: BASE is an ancestor of TIP, TIP of the merge M, and M of ``later`` (``LATER`` unless a
    guard needs ``LATER_B``, T-0094b criterion 4); skip if any is absent."""
    require_commits(base, tip, merge, later)
    assert is_ancestor(base, tip), f"{base} is not an ancestor of {tip}"
    assert is_ancestor(tip, merge), f"{tip} is not an ancestor of the merge {merge}"
    assert is_ancestor(merge, later), f"the merge {merge} is not an ancestor of {later}"
    parents = _git("rev-list", "--parents", "-n", "1", merge).stdout.split()[1:]
    tip_full = _git("rev-parse", "--verify", "--quiet", f"{tip}^{{commit}}").stdout.strip()
    assert len(parents) == 2 and tip_full == parents[1], (
        f"{tip} is not the second parent of the merge {merge}: {parents}"
    )
    mb = _git("merge-base", parents[0], parents[1]).stdout.strip()
    base_full = _git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}").stdout.strip()
    assert mb == base_full, f"{base} is not merge-base({merge}^1, {merge}^2) = {mb[:9]}"


def changed(
    base: str, tip: str, *paths: str, diff_filter: str | None = None, repo: Path = REPO
) -> list[str]:
    """Sorted paths under ``paths`` that differ between the commits ``base`` and ``tip`` (``git diff --name-only``)."""
    assert paths, "a path set is required: an empty one would diff the whole tree"
    flt = [f"--diff-filter={diff_filter}"] if diff_filter else []
    proc = _git("diff", "--name-only", *flt, base, tip, "--", *paths, repo=repo)
    assert proc.returncode == 0, f"git diff {base} {tip}: {proc.stderr}"
    return sorted(proc.stdout.split())
