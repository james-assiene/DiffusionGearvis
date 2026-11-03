"""Coded hard stops (T-0070 criterion 8). Pure: the runner feeds rows and the elapsed time in.

Each stop is a string ``"<letter>: <text>"``:

- (a) a replay-rejected terminated constrained output, or replay and the plugin's ``in_final`` disagree;
- (b) a loud CD error of class Z0, empty delta or infeasible in a constrained arm (ruling E6: codes
  ``z0``, ``empty_delta``, ``infeasible``; ``bucket_too_large`` and other refusals do not stop);
- (c) a ``join_error`` in the first 20 records;
- (d) unconstrained arm, after 30 records: BFCL parsed < 0.40 (T-0078) or Sudoku solved < 0.50;
- (e) a constrained arm, from 20 records on: more than half of its canvases hit the step cap;
- (f) is **not** a stop: the header-prefix rate of the first 20 unconstrained records is reported;
- (g) wall clock over the budget;
- (h) a dirty tree or a commit mismatch at start (checked by ``check_tree``).
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "HEADER_WINDOW",
    "STOP_B_CODES",
    "Monitor",
    "check_budget",
    "check_tree",
]

STOP_B_CODES = frozenset({"z0", "empty_delta", "infeasible"})
JOIN_WINDOW = 20
STACK_CHECK_AT = 30
STACK_FLOOR = {"bfcl": 0.40, "sudoku": 0.50}
STEP_CAP_AFTER = 20
STEP_CAP_FRACTION = 0.5
HEADER_WINDOW = 20


def check_tree(repo: Path, expected_commit: str) -> str | None:
    """(h): refuse a dirty tree (naming the paths) or a HEAD other than ``expected_commit``."""
    git = ["git", "-C", str(repo)]
    status = subprocess.run(
        [*git, "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        return f"h: git status failed in {repo}: {status.stderr.strip()[:200]}"
    dirty = [line[3:] for line in status.stdout.splitlines() if line.strip()]
    if dirty:
        shown = ", ".join(dirty[:10]) + (" ..." if len(dirty) > 10 else "")
        return f"h: dirty tree in {repo}: {shown}"
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if not head or not expected_commit or head != expected_commit:
        return f"h: commit mismatch: HEAD {head or '?'} != expected {expected_commit or '?'}"
    return None


def check_budget(elapsed_s: float, budget_s: float) -> str | None:
    """(g)."""
    if elapsed_s > budget_s:
        return f"g: wall clock {elapsed_s:.1f} s over the budget of {budget_s:.1f} s"
    return None


class Monitor:
    """Observes each finished row in order and returns the first stop it triggers, if any."""

    def __init__(self, task: str, arm: str, *, constrained: bool, step_cap: int) -> None:
        self.task = task
        self.arm = arm
        self.constrained = constrained
        self.step_cap = step_cap
        self.n = 0
        self.stack_ok = 0
        self.canvases = 0
        self.capped = 0
        self.header_hits = 0
        self.header_seen = 0

    def observe(
        self,
        row: Mapping[str, Any],
        *,
        replay: bool | None = None,
        plugin_in_final: bool | None = None,
        is_terminated: bool = False,
        opens_with_header: bool | None = None,
    ) -> str | None:
        index = self.n
        self.n += 1
        rid = row.get("id")
        error = row.get("error")
        scores = row.get("scores") or {}
        if self.constrained:
            if error in STOP_B_CODES:
                return f"b: loud CD error {error!r} on record {index} ({rid})"
            if replay is not None and plugin_in_final is not None and error is None:
                if replay != plugin_in_final:
                    return (
                        f"a: replay {'accepts' if replay else 'rejects'} record {index} ({rid}) but "
                        f"the plugin says in_final={plugin_in_final}"
                    )
                if is_terminated and not replay:
                    return f"a: terminated output of record {index} ({rid}) rejected by replay"
            for steps in row.get("steps") or ():
                self.canvases += 1
                self.capped += int(int(steps) >= self.step_cap)
        if error == "join_error" and index < JOIN_WINDOW:
            return f"c: join_error on record {index} ({rid}), within the first {JOIN_WINDOW}"
        if not self.constrained:
            # BFCL counts outputs that parsed (T-0078, human decision 2026-10-07): an out-of-schema
            # argument is a model error CD exists to prevent, not a broken stack.
            if self.task == "bfcl":
                key = "parsed"
                self.stack_ok += int(bool(row.get("parse")))
            else:
                key = "solved"
                self.stack_ok += int(bool(scores.get(key)))
            if index < HEADER_WINDOW and opens_with_header is not None:
                # measured from the generated ids (ruling E5); a record without them is not counted
                self.header_seen += 1
                self.header_hits += int(bool(opens_with_header))
            if self.n == STACK_CHECK_AT:
                rate = self.stack_ok / self.n
                floor = STACK_FLOOR[self.task]
                if rate < floor:
                    return (
                        f"d: unconstrained {key} {self.stack_ok}/{self.n} = {rate:.3f} < {floor} "
                        "after 30 records (stack broken?)"
                    )
        elif self.n >= STEP_CAP_AFTER and self.canvases:
            frac = self.capped / self.canvases
            if frac > STEP_CAP_FRACTION:
                return (
                    f"e: {self.capped}/{self.canvases} canvases hit the step cap ({self.step_cap}) "
                    f"after {self.n} records"
                )
        return None

    def header_prefix_rate(self) -> float | None:
        """(f): the fraction of the first 20 unconstrained records whose output opens with
        ``<|channel>``; ``None`` for constrained arms or before any record."""
        if self.constrained or not self.header_seen:
            return None
        return self.header_hits / self.header_seen
