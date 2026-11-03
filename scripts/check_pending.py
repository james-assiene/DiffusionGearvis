#!/usr/bin/env python3
"""Report pending tests per task, check their task files exist, and check that a task's
``**Status:**`` and its ``pending`` markers agree (T-0019 criterion 5).

Usage: ``uv run python scripts/check_pending.py [--task T-NNNN] [--strict]``

Markers are enumerated by *collecting* the suite (``tests/conftest.py`` writes the list it saw
to ``$ASTRA_PENDING_REPORT``), so aliases, helper modules and ``pytestmark`` all count — nothing
is inferred from source text.

- default: list pending tests grouped by task id; exit 1 if a task file is missing or a task
  marked ``Status: done``/``merged`` still has pending tests;
- ``--task T-NNNN``: exit 1 if *any* pending test remains for that task (the orchestrator runs
  this before the PR of ``/implement T-NNNN``; the id comes from the branch name);
- ``--strict``: also fail for tasks whose file says ``Status: in progress``.

**Warnings (exit 0)** — the two halves of the activate-then-implement mechanism must agree
(docs/09 "The prescribed red window"; the Stop hook reads the same status field through
:func:`parse_status`): a task that is ``in progress`` but still carries markers, and a task that is
``planned`` whose claimed test modules carry none, are each reported with ``WARNING``.

The helpers below — :func:`parse_status`, :func:`claimed_modules`, :func:`strong_claims` (through
:func:`task_claims`) and :func:`module_owner` — are the single source of truth for "what state is
this task in" and "which test modules does it claim"; the Stop hook
(``.claude/hooks/classify_failures.py``) imports them so the hook and this script can never drift.
:func:`owned_paths` is a copy of the grants parser in ``.claude/hooks/_lib.sh`` (``owned-globs``),
kept equal by a parity test rather than called, so ``## Owned files`` means the same thing to the
grants and to the claims (T-0147).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "docs" / "tasks"

# The status field is the first thing after ``**Status:**`` up to the ``·`` separator (or the end
# of the line). Real task files carry parentheticals and dates there
# (``in progress (65 tests landed 2026-09-13, rulings A1–A6)``, ``merged 2026-09-13 (a640f89; …)``,
# ``Python half merged 2026-09-13; criteria 4–6 open``), so classification is by *leading words*
# and keywords, never by exact match.
_STATUS_FIELD = re.compile(r"\*\*Status:\*\*\s*([^·\n]*)")
_TEST_MODULE = re.compile(r"\btests/[A-Za-z0-9_./-]+\.py\b")

IN_PROGRESS = "in progress"
PLANNED = "planned"
DONE = "done"
OTHER = "other"


def parse_status(task_text: str) -> str:
    """Classify a task file's ``**Status:**`` field as one of the four buckets.

    ``in progress`` — the activate-then-implement window is open for this task; ``planned`` — the
    markers should still be on; ``done`` — merged/closed (``merged``, ``done``, ``closed``
    anywhere in the field); ``other`` — anything else (``seeded by the scaffold``, ``blocked``,
    …), which the Stop hook treats like ``done`` (the safe default: a red module is a regression
    unless a task explicitly says it is in flight).
    """
    m = _STATUS_FIELD.search(task_text)
    if not m:
        return OTHER
    field = m.group(1).strip().lower()
    if field.startswith("in progress") or field.startswith("in-progress"):
        return IN_PROGRESS
    if field.startswith("planned"):
        return PLANNED
    if re.search(r"\b(merged|done|closed)\b", field):
        return DONE
    return OTHER


def claimed_modules(task_text: str) -> set[str]:
    """Every ``tests/**.py`` path the task file names, anywhere in the file.

    Ownership is a whole-file scan on purpose (T-0019 ruling A1): merged tasks name their modules
    under *Evidence required*, in the Goal or in prose rather than only under *Owned files* or
    *Tests to write*, so this includes paths in a not-owned list too. A module named by several
    tasks is resolved by :func:`module_owner` by precedence (an in-progress task's structured
    claim, see :func:`strong_claims`; then done > other > in progress > planned), so a merged
    task's module can never be excused by an in-progress task that merely mentions it.
    """
    return set(_TEST_MODULE.findall(task_text))


def task_files(tasks_dir: Path = TASKS) -> dict[str, str]:
    """``{task_id: file_text}`` for every ``docs/tasks/T-NNNN.md``."""
    out: dict[str, str] = {}
    for path in sorted(tasks_dir.glob("T-*.md")):
        out[path.stem] = path.read_text()
    return out


def collect() -> list[dict[str, str]]:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "pending.json"
        env = {**os.environ, "ASTRA_PENDING_REPORT": str(out)}
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "--collect-only",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode not in (0, 5) or not out.exists():
            print(proc.stdout[-2000:], proc.stderr[-2000:], file=sys.stderr)
            raise SystemExit("collection failed")
        data: list[dict[str, str]] = json.loads(out.read_text())
        return data


STATUS_RANK = {DONE: 0, OTHER: 1, IN_PROGRESS: 2, PLANNED: 3}


_SECTION = re.compile(r"^##+ +(.+?)\s*$", re.M)
TESTS_TO_WRITE = "tests to write"

# Copied from owned_globs() in .claude/hooks/_lib.sh (the grants parser); do not reinterpret
# them. T-0147's parity test (tests/unit/test_check_pending_claims.py) keeps the two copies equal.
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_OWNED = re.compile(r"^\s{0,3}##+\s*Owned files\s*$", re.I)
_MARKER = re.compile(
    r"does not own|deliberately does not touch|ownership conflicts|claims no ", re.I
)
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
_SHAPE = re.compile(r"^\.?[A-Za-z0-9_][A-Za-z0-9_.\-/]*(\*\*|/\*|\*)?$")


def owned_paths(task_text: str) -> list[str]:
    """What ``bash .claude/hooks/_lib.sh owned-globs`` prints for this task file, in order.

    A Python port of the grants parser (T-0147 ruling (a)): the first ``## Owned files`` heading
    outside a fenced block; up to the next heading of any level or a non-fenced not-owned marker
    line; fenced lines skipped; only the first token of a ``-``/``*``/``+`` bullet, backticks
    stripped, then ``,;:`` from the right, then a leading ``./``; the token must match SHAPE, have
    no ``..`` segment, and contain ``/`` or ``.`` unless it is the bullet's only token. ``[]``
    where ``owned-globs`` exits 3 (no such section). The result may contain globs.
    """
    lines = task_text.splitlines()
    fenced, start = False, None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and _OWNED.match(line):
            start = i
            break
    if start is None:
        return []
    out: list[str] = []
    fenced = False
    for line in lines[start + 1 :]:
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if _HEADING.match(line) or _MARKER.search(line):
            break
        m = _BULLET.match(line)
        if not m:
            continue
        words = m.group(1).split()
        tok = words[0] if words else ""
        tok = tok.strip("`").rstrip(",;:")
        if tok.startswith("./"):
            tok = tok[2:]
        pathish = "/" in tok or "." in tok or len(words) == 1
        if not tok or ".." in tok.split("/") or not _SHAPE.match(tok) or not pathish:
            continue
        out.append(tok)
    return out


def _before_marker(body: str) -> str:
    """``body`` up to (not including) its first non-fenced not-owned marker line."""
    kept: list[str] = []
    fenced = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and _MARKER.search(line):
            break
        kept.append(line)
    return "\n".join(kept)


def strong_claims(task_text: str) -> set[str]:
    """The ``tests/**.py`` modules a task claims structurally (T-0019 rulings A7/A10, T-0147).

    - ``## Owned files``: the concrete ``tests/**.py`` entries of :func:`owned_paths`, i.e. read
      with the grants parser's rule (first token of a bullet only, up to any heading or a
      not-owned marker line, fences skipped). Globs never claim a module.
    - ``## Tests to write``: every ``tests/**.py`` path in the section (prose included: an
      opening path, a second path in a bullet, a bold heading, ``path::test``), up to the first
      non-fenced not-owned marker line.

    A structured claim is what lets an *in-progress* task take over a module a merged task
    created (T-0026 amends ``test_guard_bash_hook.py``, T-0025 amends
    ``test_hud_constants_parity.py``): the activate-then-implement window applies to those modules
    too. A mention elsewhere, including a not-owned list, never outranks anything.
    """
    out = {p for p in owned_paths(task_text) if _TEST_MODULE.fullmatch(p)}
    pieces = _SECTION.split(task_text)
    # pieces = [preamble, title1, body1, title2, body2, ...]
    for title, body in zip(pieces[1::2], pieces[2::2], strict=False):
        if title.strip().lower().rstrip(":") == TESTS_TO_WRITE:
            out |= claimed_modules(_before_marker(body))
    return out


TaskClaims = tuple[str, set[str], set[str]]  # (status, every module named, strong claims)


def task_claims(task_text: str) -> TaskClaims:
    return parse_status(task_text), claimed_modules(task_text), strong_claims(task_text)


def module_owner(module: str, statuses: dict[str, TaskClaims]) -> str | None:
    """The task a module is attributed to when several name it.

    An in-progress task that claims the module under ``## Owned files``/``## Tests to write``
    wins (its window is open for that module); otherwise done > other > in progress > planned,
    ties by task id. ``None`` when no task names it. The Stop hook uses the same rule, so a merged
    task's module can never be excused by a later task that merely *mentions* it.
    """
    claims = sorted(
        (0 if status == IN_PROGRESS and module in strong else 1 + STATUS_RANK[status], task)
        for task, (status, mods, strong) in statuses.items()
        if module in mods
    )
    return claims[0][1] if claims else None


def status_marker_warnings(by_task: dict[str, list[str]], files: dict[str, str]) -> list[str]:
    """T-0019 criterion 5: the status and the markers must agree, in both directions."""
    warnings: list[str] = []
    statuses = {task: task_claims(text) for task, text in files.items()}
    for task in sorted(files):
        status, mods, _strong = statuses[task]
        nodes = by_task.get(task, [])
        if status == IN_PROGRESS and nodes:
            modules = sorted({n.split("::")[0] for n in nodes})
            warnings.append(
                f"WARNING {task}: in progress but {len(nodes)} pending test(s) are still marked "
                f"in {', '.join(modules)} — the test-author removes the markers when /implement "
                "starts"
            )
        elif status == PLANNED and not nodes:
            # Only modules this task is the attributed owner of: a planned task that merely
            # mentions a merged task's module is not "activated without a status change".
            # ... and only test modules: fakes, conftests and helpers a task file names in
            # its ownership prose carry no tests and can never carry a marker.
            present = sorted(
                m
                for m in mods
                if Path(m).name.startswith("test_")
                and (REPO / m).exists()
                and module_owner(m, statuses) == task
            )
            if present:
                warnings.append(
                    f"WARNING {task}: planned but its test modules carry no pending marker "
                    f"({', '.join(present)}) — markers removed without moving the status; set "
                    "the task to in progress or restore the markers"
                )
    return warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="fail if any pending test remains for this task id")
    ap.add_argument(
        "--strict", action="store_true", help="fail if a task in progress still has pending tests"
    )
    args = ap.parse_args()

    by_task: dict[str, list[str]] = defaultdict(list)
    for entry in collect():
        by_task[entry["task"]].append(entry["nodeid"])
    files = task_files()
    problems: list[str] = []
    if args.task and args.task in by_task:
        nodes = by_task[args.task]
        problems.append(
            f"{args.task}: {len(nodes)} pending test(s) still marked: " + ", ".join(nodes[:5])
        )
    for task, nodes in sorted(by_task.items()):
        if task not in files:
            problems.append(f"{task}: pending tests but no docs/tasks/{task}.md")
            status = "?"
        else:
            status = parse_status(files[task])
            if status == DONE:
                problems.append(f"{task}: status done but {len(nodes)} pending test(s) remain")
            if args.strict and status == IN_PROGRESS:
                problems.append(
                    f"{task}: in progress with {len(nodes)} pending test(s) still marked"
                )
        modules = sorted({n.split("::")[0] for n in nodes})
        print(
            f"{task} [{status}] — {len(nodes)} test(s) in {len(modules)} module(s): {', '.join(modules)}"
        )
    if not by_task:
        print("no pending tests")
    for w in status_marker_warnings(by_task, files):
        print(w)
    for p in problems:
        print("ERROR", p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
