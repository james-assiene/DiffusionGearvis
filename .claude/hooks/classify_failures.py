#!/usr/bin/env python3
"""Classify a fast-suite run for the Stop hook (T-0019).

Reads pytest's output on stdin, groups the ``FAILED``/``ERROR`` lines by test module, looks each
module up in ``docs/tasks/*.md`` (through ``scripts/check_pending.py``'s ``parse_status`` and
``claimed_modules`` — the same parser the backlog script uses, so the two halves of the mechanism
cannot drift), and prints a report:

* **regressions first** — a failing module that no task claims, or that is claimed only by a
  task whose status is ``planned`` (markers removed without moving the status), ``done`` or
  anything other than ``in progress``;
* then the **expected block**, one line per in-progress task:
  ``expected: T-0004 (43 failing in tests/unit/test_speech_gate.py, implementer pending)``.

Exit 0 when every failure is expected, 1 when any regression exists. The whole report goes to
stdout; the hook decides which stream to forward.

**Never reads ``pending`` markers** (T-0019 criterion 7): tolerance comes only from the task
file's status and the modules it names. Precedence when several tasks name one module: an
in-progress task that claims it under ``## Owned files`` / ``## Tests to write`` wins (its window
is open for that module — T-0026 amending ``test_guard_bash_hook.py``); otherwise done/other >
in progress > planned — a merged task's module can never be excused by an in-progress task that
merely *mentions* it (ruling A10).

Usage: ``python3 classify_failures.py <repo-top> < pytest-output``
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

_RESULT = re.compile(r"^(?:FAILED|ERROR)\s+(?P<node>\S+)")


# Both alternatives are line-anchored: the ``Interrupted:`` banner pytest prints when collection
# aborts, and the ``=== N errors during collection ===`` variant. A fixed-width prefix window
# instead of an anchor let a failing test that *quotes* the banner in its assertion message fake
# an abort depending on where the phrase landed on the ``E`` line (ruling A13).
_INTERRUPTED = re.compile(r"^[!\s]*Interrupted:|^=+ .*\berrors? during collection\b", re.I | re.M)


def failures_by_module(pytest_output: str) -> tuple[dict[str, int], list[str]]:
    """``({module: failing tests}, [modules that did not even collect])``.

    A collection-level ``ERROR tests/x.py`` (no ``::``) or an ``Interrupted:`` banner means pytest
    aborted before running the suite, so no status can excuse it (review finding, T-0019).
    """
    counts: dict[str, int] = defaultdict(int)
    broken: list[str] = []
    for line in pytest_output.splitlines():
        # Line start only: pytest never indents its summary lines, and captured test output
        # that quotes "FAILED tests/x.py::t" behind a gutter must not count (ruling A14).
        m = _RESULT.match(line)
        if not m:
            continue
        node = m.group("node")
        if "::" not in node and line.startswith("ERROR"):
            if node not in broken:
                broken.append(node)
            continue
        counts[node.split("::")[0]] += 1
    if _INTERRUPTED.search(pytest_output) and not broken:
        broken.append("<collection>")
    return dict(counts), broken


def main(argv: list[str]) -> int:
    top = Path(argv[1]).resolve() if len(argv) > 1 else Path.cwd()
    sys.path.insert(0, str(top / "scripts"))
    from check_pending import IN_PROGRESS, PLANNED, module_owner, task_claims

    counts, broken = failures_by_module(sys.stdin.read())
    if not counts and not broken:
        return 0

    tasks = {
        path.stem: task_claims(path.read_text())
        for path in sorted((top / "docs" / "tasks").glob("T-*.md"))
    }

    regressions: list[str] = [
        f"regression: {module} did not collect — pytest aborted before the suite ran, so no "
        "task status can excuse it (a stub must exist for every planned module, "
        ".claude/rules/tests.md)"
        for module in broken
    ]
    expected: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for module, n in sorted(counts.items()):
        task = module_owner(module, tasks)
        if task is None:
            regressions.append(f"regression: {n} failing in {module} — claimed by no task")
            continue
        status = tasks[task][0]
        if status == IN_PROGRESS:
            expected[task].append((module, n))
        elif status == PLANNED:
            regressions.append(
                f"regression: {n} failing in {module} — {task} is planned, so its pending "
                "markers were removed without moving the status; set the task to in progress "
                "(or restore the markers)"
            )
        else:
            regressions.append(f"regression: {n} failing in {module} — {task} is {status}")

    for line in regressions:
        print(line)
    for task in sorted(expected):
        for module, n in expected[task]:
            print(f"expected: {task} ({n} failing in {module}, implementer pending)")
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
