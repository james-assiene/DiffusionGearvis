"""Acceptance tests for ``scripts/check_pending.py``'s status/marker agreement (T-0019, crit. 5).

T-0019 makes the Stop hook tolerate failures confined to modules of tasks whose ``**Status:**`` is
``in progress``. That only works if the status is kept honest, so the same task requires the other
half of the mechanism: ``scripts/check_pending.py`` **warns** when a task is

* ``in progress`` and still has ``pending`` markers (the implementer is nominally at work but the
  acceptance tests are still parked as strict xfail -- nothing is actually being held to account);
* ``planned`` and has none (markers were removed without moving the status -- the state the hook
  reports as a regression in exactly those words, see ``tests/unit/test_stop_hook.py``).

Each case builds a throwaway repo in ``tmp_path``: a copy of ``scripts/check_pending.py``, a copy
of ``tests/conftest.py`` (the ``pending`` mechanism itself is out of scope for T-0019, so it is
used as-is rather than re-implemented), a ``docs/tasks/`` and a handful of tiny test modules. The
script is then run as a subprocess against that repo, so nothing depends on the pending backlog of
this repo on the day the tests run.

Every case pairs the mismatch it asserts on with a *consistent* task that must stay silent, so a
blanket "warn about every task" implementation fails, and so no assertion here can pass today by
accident (today the script never prints a warning at all).

What the criterion does not settle, and what is therefore not asserted here: whether a ``planned``
task whose test modules have not been written yet should warn (it has no markers either), and
whether a warning changes the exit status. Both are left to the implementer.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
# ``ASTRA_CHECK_PENDING`` points the cases at another checkout's copy of the script (an
# implementer worktree, a reviewer probe); the conftest is taken from the same checkout, since the
# two halves of the ``pending`` mechanism have to be read together.
SCRIPT = Path(os.environ.get("ASTRA_CHECK_PENDING") or REPO / "scripts" / "check_pending.py")
CONFTEST = SCRIPT.resolve().parents[1] / "tests" / "conftest.py"

TASK_TEMPLATE = """# {task} — sandbox task

**Status:** {status} · **Blast radius:** low · **Depends on:** none

## Goal
A throwaway task file in the shape of `docs/tasks/README.md`. Modules under "Tests to write" are
structured claims; anything named in this paragraph is a prose mention (ruling A10).
{mentions}

## Acceptance criteria
1. The modules under "Tests to write" go green.

## Owned files
- local_astra/sandbox/{slug}.py

## Tests to write
{tests}
"""


@dataclass
class PendingSandbox:
    top: Path

    def task(
        self,
        task_id: str,
        status: str,
        modules: Sequence[str],
        mentions: Sequence[str] = (),
    ) -> None:
        (self.top / "docs" / "tasks" / f"{task_id}.md").write_text(
            TASK_TEMPLATE.format(
                task=task_id,
                status=status,
                slug=task_id.lower().replace("-", "_"),
                mentions="".join(
                    f"\nThe bug this task chases was first seen in `{m}`, which it does not own.\n"
                    for m in mentions
                ),
                tests="\n".join(
                    f"- `{m}` — unit — asserts what does not exist yet." for m in modules
                ),
            )
        )

    def module(self, rel: str, *, tests: int, pending_task: str | None) -> str:
        """A module of ``tests`` trivially-passing tests, each marked pending or not.

        The bodies pass so that a marker's only effect on the run is the strict-xfail conversion
        recorded in ``$ASTRA_PENDING_REPORT``; nothing here depends on pass/fail counts.
        """
        lines = ["import pytest", "", "pytest  # noqa: B018  -- imported for the marker below", ""]
        for i in range(tests):
            lines += [
                f'@pytest.mark.pending(task="{pending_task}")' if pending_task else "# unmarked",
                f"def test_case_{i}():",
                "    assert 2 + 2 == 4",
                "",
            ]
        path = self.top / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines))
        return rel

    def file(self, rel: str, body: str) -> str:
        """A non-test file under ``tests/`` (a fake, a conftest): named in ownership prose, but it
        carries no tests, so it can never carry a ``pending`` marker either (ruling A8)."""
        path = self.top / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        return rel

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.top / "scripts" / "check_pending.py"), *args],
            capture_output=True,
            text=True,
            cwd=str(self.top),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            timeout=300,
        )


@pytest.fixture
def sandbox(tmp_path: Path) -> PendingSandbox:
    top = tmp_path / "repo"
    (top / "docs" / "tasks").mkdir(parents=True)
    (top / "scripts").mkdir()
    (top / "tests" / "unit").mkdir(parents=True)
    shutil.copy(SCRIPT, top / "scripts" / "check_pending.py")
    shutil.copy(CONFTEST, top / "tests" / "conftest.py")
    (top / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    pending(task): written ahead of its implementation\n"
    )
    return PendingSandbox(top=top)


def warnings_for(res: subprocess.CompletedProcess[str], task: str) -> list[str]:
    """Lines that warn about ``task`` -- the criterion's own word, in either stream."""
    return [
        line.strip()
        for stream in (res.stdout, res.stderr)
        for line in stream.splitlines()
        if task in line and re.search(r"warn", line, re.I)
    ]


def _diag(res: subprocess.CompletedProcess[str]) -> str:
    return f"exit={res.returncode}\n--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}"


def test_warns_when_in_progress_task_still_has_markers(sandbox: PendingSandbox) -> None:
    """``in progress`` + markers present -> warn; ``in progress`` + no markers -> silent."""
    stale = sandbox.module("tests/unit/test_alpha.py", tests=2, pending_task="T-0201")
    live = sandbox.module("tests/unit/test_beta.py", tests=2, pending_task=None)
    sandbox.task("T-0201", "in progress", [stale])
    sandbox.task("T-0202", "in progress", [live])

    res = sandbox.run()

    assert warnings_for(res, "T-0201"), (
        "T-0201 is in progress with its acceptance tests still parked as strict xfail; the two "
        f"halves of the mechanism disagree and nothing says so\n{_diag(res)}"
    )
    assert not warnings_for(res, "T-0202"), (
        f"T-0202 is the consistent case (in progress, markers removed) and must stay silent"
        f"\n{_diag(res)}"
    )


def test_warns_when_planned_task_has_no_markers_left(sandbox: PendingSandbox) -> None:
    """``planned`` + markers gone -> warn; ``planned`` + markers, and a merged task -> silent."""
    activated = sandbox.module("tests/unit/test_gamma.py", tests=2, pending_task=None)
    parked = sandbox.module("tests/unit/test_delta.py", tests=2, pending_task="T-0204")
    merged = sandbox.module("tests/unit/test_epsilon.py", tests=2, pending_task=None)
    sandbox.task("T-0203", "planned", [activated])
    sandbox.task("T-0204", "planned", [parked])
    sandbox.task("T-0206", "merged 2026-09-13 (a640f89; re-review MERGE)", [merged])

    res = sandbox.run()

    assert warnings_for(res, "T-0203"), (
        "T-0203 is planned yet its module carries no markers: they were removed without moving "
        f"the status, which is what the Stop hook reports as a regression\n{_diag(res)}"
    )
    assert not warnings_for(res, "T-0204"), (
        f"T-0204 is the consistent case (planned, markers on) and must stay silent\n{_diag(res)}"
    )
    assert not warnings_for(res, "T-0206"), (
        "a merged task with no markers is the normal end state, not a mismatch; warning there is "
        f"the noise T-0019 exists to remove\n{_diag(res)}"
    )


def test_status_with_a_parenthetical_is_still_in_progress(sandbox: PendingSandbox) -> None:
    """Adversarial: real status lines carry a parenthetical (T-0006 and T-0015 on 2026-09-13).

    ``**Status:** in progress (65 tests landed 2026-09-13, rulings A1-A6) · **Blast radius:** ...``
    is the shape live task files have. The Stop hook reads the same field, so the two halves must
    parse it the same way; a script that gives up and calls the status unknown warns about nothing
    and the mechanism silently stops agreeing.
    """
    stale = sandbox.module("tests/unit/test_alpha.py", tests=2, pending_task="T-0205")
    sandbox.task("T-0205", "in progress (65 tests landed 2026-09-13, rulings A1–A6)", [stale])

    res = sandbox.run()

    assert warnings_for(res, "T-0205"), _diag(res)


def test_planned_task_naming_only_a_fake_is_not_a_missing_marker(sandbox: PendingSandbox) -> None:
    """Ruling A8: the planned-without-markers warning considers only ``test_*.py`` modules.

    T-0207 is planned and names ``tests/fakes/foo.py`` -- a fake, not a test module. It has no
    markers because it can never have one, so "planned with no markers left" does not describe it
    and a warning there is exactly the noise T-0019 exists to remove. T-0208 is the real mismatch
    in the same run, so an implementation that warns about neither fails too.
    """
    fake = sandbox.file(
        "tests/fakes/foo.py", "class FakeThing:\n    def read(self) -> bytes:\n        return b''\n"
    )
    activated = sandbox.module("tests/unit/test_zeta.py", tests=2, pending_task=None)
    sandbox.task("T-0207", "planned", [fake])
    sandbox.task("T-0208", "planned", [activated])

    res = sandbox.run()

    assert not warnings_for(res, "T-0207"), (
        f"{fake} carries no tests, so it can carry no marker; a planned task that names it is not "
        f"a status/marker mismatch\n{_diag(res)}"
    )
    assert warnings_for(res, "T-0208"), (
        "T-0208 is the genuine mismatch in the same run (planned, markers removed from a real "
        f"test module) and must still be warned about\n{_diag(res)}"
    )


def test_planned_task_mentioning_a_merged_module_in_prose_is_not_warned_about(
    sandbox: PendingSandbox,
) -> None:
    """Ruling A10, the ``check_pending`` twin: a prose mention is not "activated without a status".

    T-0210 is planned and its Goal paragraph names ``test_eta.py``, a module merged T-0209 owns and
    which of course carries no markers. Warning about that would tell the planner to move a status
    they have nothing to do with, for a module they do not own -- the kind of misfire that gets a
    warning skimmed. T-0211 is the genuine mismatch in the same run, so an implementation that
    warns about nothing fails too.
    """
    owned_by_merged = sandbox.module("tests/unit/test_eta.py", tests=2, pending_task=None)
    activated = sandbox.module("tests/unit/test_theta.py", tests=2, pending_task=None)
    sandbox.task("T-0209", "merged 2026-09-13 (a640f89; re-review MERGE)", [owned_by_merged])
    sandbox.task("T-0210", "planned", [], mentions=[owned_by_merged])
    sandbox.task("T-0211", "planned", [activated])

    res = sandbox.run()

    assert not warnings_for(res, "T-0210"), (
        f"T-0210 only mentions {owned_by_merged} in prose; it owns no module whose markers could "
        f"have been removed\n{_diag(res)}"
    )
    assert not warnings_for(res, "T-0209"), (
        f"{owned_by_merged} belongs to a merged task with no markers -- the normal end state"
        f"\n{_diag(res)}"
    )
    assert warnings_for(res, "T-0211"), (
        "T-0211 is the genuine mismatch in the same run (planned, markers removed from a module it "
        f"claims) and must still be warned about\n{_diag(res)}"
    )


def test_planned_task_whose_module_is_not_written_yet_is_not_a_missing_marker(
    sandbox: PendingSandbox,
) -> None:
    """Ruling A3: a planned task whose modules do not exist yet is silent.

    T-0212 is the state every task passes through between ``/plan`` and ``/implement``: planned,
    claiming ``tests/unit/test_not_written_yet.py`` under "Tests to write", and the test-author
    has not started. It has no markers because it has no tests, so "planned with the markers
    removed" does not describe it -- and warning about it would fire on every open task in the
    backlog, which is how a warning stops being read. T-0213 is the genuine mismatch in the same
    run: its module is on disk with the markers taken off.

    This is the case the reviewer found survives deleting the ``exists()`` filter.
    """
    unwritten = "tests/unit/test_not_written_yet.py"
    assert not (sandbox.top / unwritten).exists(), "the point of the case is that it is absent"
    activated = sandbox.module("tests/unit/test_iota.py", tests=2, pending_task=None)
    sandbox.task("T-0212", "planned", [unwritten])
    sandbox.task("T-0213", "planned", [activated])

    res = sandbox.run()

    assert not warnings_for(res, "T-0212"), (
        f"{unwritten} has not been written yet, so it carries no markers and can lose none; "
        f"warning here fires on every planned task in the backlog\n{_diag(res)}"
    )
    assert warnings_for(res, "T-0213"), (
        "T-0213 is the genuine mismatch in the same run (planned, module on disk, markers "
        f"removed) and must still be warned about\n{_diag(res)}"
    )
