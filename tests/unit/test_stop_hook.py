"""Acceptance tests for the Stop/SubagentStop hook, ``.claude/hooks/stop_fast_suite.sh`` (T-0019).

The hook's job is to catch a **regression in merged code** before an agent finishes its turn. Its
defect (T-0019, observed eight times in one session on 2026-09-13) is that it also fires during
the one state the protocol prescribes as red: the window between the test-author removing
``pending`` markers and the implementer landing the code. After T-0019 the hook classifies each
failing module against ``docs/tasks/*.md`` and tolerates failures **confined to modules of tasks
whose ``**Status:**`` is ``in progress``**.

Contract pinned here (T-0019 acceptance criteria 1-4 and 7):

* exit 0 = the turn may finish, non-zero = it may not (the hook's whole contract today);
* a tolerated task prints exactly ``expected: T-0004 (43 failing in tests/unit/test_speech_gate.py,
  implementer pending)`` -- one line per in-progress task, so the red stays visible and attributed;
* anything else is a **regression**: an unclaimed module, a module of a ``planned`` task (markers
  removed without moving the status), a module of a ``merged``/``done`` task;
* a regression is listed *first*, before the expected block;
* **negative (criterion 7):** tolerance comes only from the task file. The hook must never read
  ``pending`` markers out of test sources -- otherwise "mark it pending" becomes a way to silence a
  real failure, which CLAUDE.md forbids.

Every case builds its own throwaway git repo in ``tmp_path`` with its own ``docs/tasks/`` and its
own ``tests/unit`` modules, so nothing here depends on the state of this repo.

**The seam.** The hook runs the fast suite as ``uv run pytest tests/unit tests/contract ...`` after
``cd``-ing to the git toplevel of the hook input's ``cwd``. That is enough of a seam already: each
sandbox puts a tiny ``uv`` shim first on ``PATH`` which forwards ``run pytest ...`` to this
interpreter's ``python -m pytest`` (and answers ``run mypy`` with exit 0). The hook's command text
is therefore exercised verbatim, and the tests impose no new interface. The one thing they do
require of the implementer is that the suite keeps being invoked through ``uv run`` from the
worktree root; a hook that shelled out to an absolute interpreter, or that read the failing
modules from anywhere but a pytest run of the sandbox, would fail these tests.

Note on what is deliberately *not* tested here: a fully green sandbox (exit 0, no output) and a
sandbox with no Python changes (exit 0 without running the suite) already behave correctly today,
so a ``pending`` test for them would XPASS.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
# ``ASTRA_STOP_HOOK`` points the cases at another checkout's copy of the hook (an implementer
# worktree, a reviewer probe) without touching this file; unset, they run against this repo's.
HOOK = Path(os.environ.get("ASTRA_STOP_HOOK") or REPO / ".claude" / "hooks" / "stop_fast_suite.sh")
# The repo the hook belongs to: <root>/.claude/hooks/stop_fast_suite.sh. Helpers live under
# ``.claude/hooks/`` (reached by the hook's own ``dirname $0``) or ``scripts/`` (ruling A5), and
# the sandbox borrows the latter from the same checkout as the hook, not from wherever the tests
# happen to live.
HOOK_ROOT = HOOK.resolve().parents[2]

# Git in a hermetic state: no user config, no global hooks, deterministic identity.
GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_AUTHOR_NAME": "stop-hook-test",
    "GIT_AUTHOR_EMAIL": "stop-hook-test@example.invalid",
    "GIT_COMMITTER_NAME": "stop-hook-test",
    "GIT_COMMITTER_EMAIL": "stop-hook-test@example.invalid",
}

# One line per in-progress task, exactly as T-0019 criterion 2 spells it.
EXPECTED_LINE = re.compile(
    r"^expected: (?P<task>T-\d{4}) \((?P<n>\d+) failing in (?P<module>\S+), implementer pending\)$"
)

TASK_TEMPLATE = """# {task} — sandbox task for the Stop hook

**Status:** {status} · **Blast radius:** low · **Depends on:** none

## Goal
A throwaway task file in the shape of `docs/tasks/README.md`, used to exercise the hook's
classifier. Its *structured* claims are the modules listed under "Tests to write"; anything named
here in the Goal paragraph is a prose mention and claims nothing (ruling A10).
{mentions}

## Acceptance criteria
1. The modules under "Tests to write" go green.

## Owned files
- local_astra/sandbox/{slug}.py

## Tests to write
{tests}

## Out of scope
Everything else.
"""


def _module_source(*, failing: int, passing: int, pending_task: str | None) -> str:
    """A test module with a hand-known number of failures.

    ``2 + 2 == 5`` is false and ``2 + 2 == 4`` is true; the counts asserted by the tests below are
    therefore read off the call, not off a pytest run. When ``pending_task`` is given each failing
    test carries a ``pending`` marker; when it is not, the *same line* is a comment naming the same
    task, so the two variants differ only in whether that one character makes the marker real.
    The sandbox registers ``pending`` as an inert marker (no ``conftest.py``), so the set of
    failures is identical either way -- which is exactly what criterion 7 needs to isolate.
    """
    lines = ["import pytest", "", "pytest  # noqa: B018  -- imported for the marker below", ""]
    for i in range(failing):
        mark = f'@pytest.mark.pending(task="{pending_task}")' if pending_task else "# unmarked"
        lines += [
            mark,
            f"def test_not_implemented_{i}():",
            "    assert 2 + 2 == 5",
            "",
        ]
    for i in range(passing):
        lines += [
            "# unmarked",
            f"def test_already_green_{i}():",
            "    assert 2 + 2 == 4",
            "",
        ]
    return "\n".join(lines)


@dataclass
class StopSandbox:
    """A throwaway worktree with ``docs/tasks/``, ``tests/unit`` and a ``uv`` shim on PATH."""

    top: Path
    hook: Path = HOOK

    def task(
        self,
        task_id: str,
        status: str,
        modules: Sequence[str],
        mentions: Sequence[str] = (),
    ) -> None:
        """A task file claiming ``modules`` under ``## Tests to write`` (a *structured* claim) and
        merely naming ``mentions`` in its Goal prose.

        The distinction is ruling A10: a task file quoting another task's module in passing --
        T-0019.md itself names ``tests/unit/test_speech_gate.py`` while recounting the session that
        motivated it -- must not thereby acquire any say over that module.
        """
        body = TASK_TEMPLATE.format(
            task=task_id,
            status=status,
            slug=task_id.lower().replace("-", "_"),
            mentions="".join(
                f"\nThe session that motivated this task saw failures in `{m}`, which it does not "
                "own.\n"
                for m in mentions
            ),
            tests="\n".join(
                f"- `{m}` — unit — asserts the behaviour that does not exist yet." for m in modules
            ),
        )
        (self.top / "docs" / "tasks" / f"{task_id}.md").write_text(body)

    def module(
        self, rel: str, *, failing: int, passing: int = 0, pending_task: str | None = None
    ) -> str:
        path = self.top / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_module_source(failing=failing, passing=passing, pending_task=pending_task))
        return rel

    def raw_module(self, rel: str, source: str) -> str:
        """A test module written verbatim, for payloads ``_module_source`` cannot express."""
        path = self.top / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        return rel

    def uncollectable_module(self, rel: str) -> str:
        """A module pytest cannot import: the session aborts before any test runs (ruling A6)."""
        path = self.top / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "import sandbox_pkg_that_does_not_exist\n\n\ndef test_never_runs():\n    pass\n"
        )
        return rel

    def run(self) -> subprocess.CompletedProcess[str]:
        payload = json.dumps({"cwd": str(self.top), "agent_type": "none"})
        env = {**GIT_ENV, "PATH": f"{self.top / 'bin'}{os.pathsep}{os.environ['PATH']}"}
        return subprocess.run(
            ["bash", str(self.hook)],
            input=payload,
            capture_output=True,
            text=True,
            cwd=str(self.top),
            env=env,
            timeout=300,
        )


# A helper that reads the run, says something unrelated on stderr and exits 0 -- a plausible
# accident (an unparseable task file, a deprecation notice) rather than a verdict.
CHATTY_HELPER = """#!/usr/bin/env python3
import sys

sys.stdin.read()
print("note: docs/tasks/T-0099.md has no **Status:** line; skipping it", file=sys.stderr)
"""


def _make_sandbox(root: Path, *, chatty_helper: bool = False) -> StopSandbox:
    top = root
    (top / "docs" / "tasks").mkdir(parents=True)
    (top / "tests" / "unit").mkdir(parents=True)
    (top / "tests" / "contract").mkdir(parents=True)
    (top / "tests" / "contract" / ".gitkeep").write_text("")
    # Rootdir for the sandbox's pytest, and ``pending`` registered as an *inert* marker: no
    # conftest.py here, so a marker changes no outcome. See _module_source.
    (top / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    pending(task): written ahead of its implementation\n"
    )
    # ``uv`` shim: the hook checks ``command -v uv`` and runs ``uv run pytest ...`` / ``uv run
    # mypy``. Forward to this interpreter so the sandbox's own tests are "the fast suite".
    bin_dir = top / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "uv"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        '[ "$1" = "run" ] && shift\n'
        'case "$1" in\n'
        f'  python|python3) shift; exec "{sys.executable}" "$@";;\n'
        "  mypy) exit 0;;\n"
        f'  *) exec "{sys.executable}" -m "$@";;\n'
        "esac\n"
    )
    shim.chmod(0o755)
    # ``scripts/`` is shared with the real repo so a hook that grew a Python helper there still
    # finds it; a symlink, so git does not see the outer repo's files as sandbox changes.
    (top / "scripts").symlink_to(HOOK_ROOT / "scripts")
    hook = HOOK
    if chatty_helper:
        # Run a private copy of the whole hook directory with every Python helper in it replaced
        # by CHATTY_HELPER. The directory rather than one filename because ruling A5 fixes only
        # *where* a helper may live (``.claude/hooks/`` or ``scripts/``), not what it is called;
        # a helper that moved to ``scripts/`` would make this case fail loudly rather than pass
        # vacuously, which is the right way round.
        hooks_dir = top / ".claude" / "hooks"
        shutil.copytree(HOOK.resolve().parent, hooks_dir)
        stubbed = sorted(hooks_dir.glob("*.py"))
        assert stubbed, f"no Python helper under {HOOK.resolve().parent} to stand in for"
        for helper in stubbed:
            helper.write_text(CHATTY_HELPER)
        hook = hooks_dir / HOOK.name
    subprocess.run(
        ["git", "init", "-q", str(top)], check=True, env=GIT_ENV, capture_output=True, text=True
    )
    return StopSandbox(top=top, hook=hook)


@pytest.fixture
def sandbox(tmp_path: Path) -> StopSandbox:
    return _make_sandbox(tmp_path / "repo")


def _streams(res: subprocess.CompletedProcess[str]) -> list[str]:
    return [res.stderr, res.stdout]


def _report(res: subprocess.CompletedProcess[str]) -> str:
    """The single stream carrying the hook's classification.

    The hook reports through stderr (that is what exit 2 feeds back), but a report split across
    two streams would be unreadable either way, so the tests take whichever stream carries the
    classification and require the whole report to be in it.
    """
    for stream in _streams(res):
        if EXPECTED_LINE.search(stream) or re.search(r"regression", stream, re.I):
            return stream
    return res.stderr + res.stdout


def expected_lines(res: subprocess.CompletedProcess[str]) -> list[str]:
    return [
        line.strip()
        for stream in _streams(res)
        for line in stream.splitlines()
        if line.strip().startswith("expected:")
    ]


def regression_lines(res: subprocess.CompletedProcess[str]) -> list[str]:
    return [
        line.strip()
        for stream in _streams(res)
        for line in stream.splitlines()
        if re.search(r"regression", line, re.I)
    ]


def _diag(res: subprocess.CompletedProcess[str]) -> str:
    return f"exit={res.returncode}\n--- stderr ---\n{res.stderr}\n--- stdout ---\n{res.stdout}"


# ---------------------------------------------------------------------------------------------
# Criterion 2 -- only expected failures: exit 0, one attributed line per in-progress task.
# ---------------------------------------------------------------------------------------------


def test_failures_confined_to_an_in_progress_module_are_tolerated(sandbox: StopSandbox) -> None:
    """One in-progress task, two failing tests and one passing in its module -> exit 0.

    Oracle by hand: ``_module_source(failing=2, passing=1)`` writes two ``assert 2 + 2 == 5`` and
    one ``assert 2 + 2 == 4``, so the count in the expected line is 2, not 3 and not 1 (the hook
    must not keep ``-x``, which stops after the first failure and cannot count).
    """
    module = sandbox.module("tests/unit/test_alpha.py", failing=2, passing=1)
    sandbox.task("T-0101", "in progress", [module])

    res = sandbox.run()

    assert res.returncode == 0, f"prescribed red window blocked the turn\n{_diag(res)}"
    assert expected_lines(res) == [
        f"expected: T-0101 (2 failing in {module}, implementer pending)"
    ], _diag(res)


def test_one_expected_line_per_in_progress_task(sandbox: StopSandbox) -> None:
    """Two tasks in flight, one module each: two lines, each attributed to its own task."""
    alpha = sandbox.module("tests/unit/test_alpha.py", failing=2)
    beta = sandbox.module("tests/unit/test_beta.py", failing=3)
    sandbox.task("T-0101", "in progress", [alpha])
    sandbox.task("T-0102", "in progress", [beta])

    res = sandbox.run()

    assert res.returncode == 0, _diag(res)
    assert sorted(expected_lines(res)) == sorted(
        [
            f"expected: T-0101 (2 failing in {alpha}, implementer pending)",
            f"expected: T-0102 (3 failing in {beta}, implementer pending)",
        ]
    ), _diag(res)


def test_status_with_a_parenthetical_is_still_in_progress(sandbox: StopSandbox) -> None:
    """Adversarial: real status lines carry a parenthetical (T-0006, T-0015 on 2026-09-13).

    ``**Status:** in progress (65 tests landed 2026-09-13, rulings A1-A6) · **Blast radius:** ...``
    must classify as in progress. A parser that only accepts the bare word sends every task in
    flight back to "regression" and the hook is crying wolf again -- the exact defect T-0019 fixes.
    """
    module = sandbox.module("tests/unit/test_alpha.py", failing=2)
    sandbox.task("T-0101", "in progress (65 tests landed 2026-09-13, rulings A1–A6)", [module])

    res = sandbox.run()

    assert res.returncode == 0, _diag(res)
    assert expected_lines(res) == [
        f"expected: T-0101 (2 failing in {module}, implementer pending)"
    ], _diag(res)


# ---------------------------------------------------------------------------------------------
# Criteria 3 and 4 -- everything else is a regression, and it is listed first.
# ---------------------------------------------------------------------------------------------


def test_failure_in_an_unclaimed_module_is_a_regression(sandbox: StopSandbox) -> None:
    """The safe default: no task claims ``test_gamma.py``, so its failure blocks the turn.

    The exit status alone already holds today, so the assertion also pins that the hook *says*
    which module regressed, in the vocabulary of criteria 1/3 ("regression"). Without that, the
    report is the 40-line pytest tail it was before.
    """
    claimed = sandbox.module("tests/unit/test_alpha.py", failing=2)
    unclaimed = sandbox.module("tests/unit/test_gamma.py", failing=1)
    sandbox.task("T-0101", "in progress", [claimed])

    res = sandbox.run()

    assert res.returncode != 0, f"an unclaimed failing module must block the turn\n{_diag(res)}"
    assert f"expected: T-0101 (2 failing in {claimed}, implementer pending)" in _report(res), (
        "the in-progress task's own module must still be attributed, not swept into the "
        f"regression\n{_diag(res)}"
    )
    named = [line for line in regression_lines(res) if unclaimed in line]
    assert named, f"no regression line names {unclaimed}\n{_diag(res)}"
    assert not [line for line in expected_lines(res) if unclaimed in line], (
        f"{unclaimed} is claimed by no task and must not be excused\n{_diag(res)}"
    )


def test_failure_in_a_merged_tasks_module_is_a_regression(sandbox: StopSandbox) -> None:
    """The thing the hook exists for: merged code going red is never tolerated.

    Status wording taken from a real merged task (T-0004: ``merged 2026-09-13 (a640f89; ...)``).
    """
    module = sandbox.module("tests/unit/test_alpha.py", failing=1)
    sandbox.task("T-0104", "merged 2026-09-13 (a640f89; re-review MERGE)", [module])

    res = sandbox.run()

    assert res.returncode != 0, f"a merged module went red and the hook allowed it\n{_diag(res)}"
    assert [line for line in regression_lines(res) if module in line], _diag(res)
    assert expected_lines(res) == [], f"a merged task must never be excused\n{_diag(res)}"


def test_planned_task_with_failing_module_says_markers_were_removed(sandbox: StopSandbox) -> None:
    """Criterion 4, second half: markers removed without moving the status, said in those words."""
    module = sandbox.module("tests/unit/test_delta.py", failing=2)
    sandbox.task("T-0103", "planned", [module])

    res = sandbox.run()

    assert res.returncode != 0, (
        f"a planned task's module failed and the hook allowed it\n{_diag(res)}"
    )
    report = " ".join(_report(res).split()).lower()
    assert "without moving the status" in report, (
        "criterion 4 asks for those words, so the reader knows the fix is to move the status "
        f"rather than to debug the module\n{_diag(res)}"
    )
    assert "marker" in report, _diag(res)
    assert "t-0103" in report and module in _report(res), _diag(res)


def test_regression_is_listed_before_the_expected_block(sandbox: StopSandbox) -> None:
    """Criterion 3: mixed run -- the regression must not be buried under the expected noise."""
    claimed = sandbox.module("tests/unit/test_alpha.py", failing=4)
    unclaimed = sandbox.module("tests/unit/test_gamma.py", failing=1)
    sandbox.task("T-0101", "in progress", [claimed])

    res = sandbox.run()

    assert res.returncode != 0, _diag(res)
    report = _report(res)
    assert f"expected: T-0101 (4 failing in {claimed}, implementer pending)" in report, _diag(res)
    first_regression = min(
        (i for i, line in enumerate(report.splitlines()) if unclaimed in line), default=-1
    )
    first_expected = min(
        (i for i, line in enumerate(report.splitlines()) if line.strip().startswith("expected:")),
        default=-1,
    )
    assert first_regression >= 0 and first_expected >= 0, _diag(res)
    assert first_regression < first_expected, (
        f"the regression in {unclaimed} is buried under the expected block\n{_diag(res)}"
    )


# ---------------------------------------------------------------------------------------------
# Criterion 7 (negative) -- tolerance never comes from a ``pending`` marker in a test source.
# ---------------------------------------------------------------------------------------------


def test_pending_marker_in_the_source_changes_nothing(tmp_path: Path) -> None:
    """Two identical sandboxes but for the marker text: the hook's verdict must be byte-identical.

    Both claim ``tests/unit/test_alpha.py`` from an in-progress task, so both are tolerated. The
    marker is inert in the sandbox (no conftest), the two modules have the same line numbering, so
    any difference in output is the hook having read the marker.
    """
    outputs = []
    for name, marker in (("plain", None), ("marked", "T-0101")):
        sb = _make_sandbox(tmp_path / name)
        module = sb.module("tests/unit/test_alpha.py", failing=2, pending_task=marker)
        sb.task("T-0101", "in progress", [module])
        res = sb.run()
        text = (res.stderr + "\n" + res.stdout).replace(str(sb.top.resolve()), "<REPO>")
        outputs.append((res.returncode, text.replace(str(sb.top), "<REPO>")))

    assert outputs[0][0] == 0, "the marker-free sandbox is the tolerated case"
    assert outputs[0] == outputs[1], (
        "the hook's report changed when a pending marker was added to the sources; tolerance must "
        f"come from docs/tasks only\n--- without ---\n{outputs[0]}\n--- with ---\n{outputs[1]}"
    )


def test_pending_marker_cannot_excuse_an_unclaimed_module(sandbox: StopSandbox) -> None:
    """The failure mode criterion 7 names: "mark it pending" as a way to silence a real failure.

    ``tests/unit/test_gamma.py`` is claimed by no task file, but its failing tests carry
    ``pending(task="T-0101")`` and T-0101 is in progress. A hook that derived ownership from the
    markers in the sources would excuse it. Ownership comes from the task file, so it is a
    regression.
    """
    claimed = sandbox.module("tests/unit/test_alpha.py", failing=2)
    smuggled = sandbox.module("tests/unit/test_gamma.py", failing=1, pending_task="T-0101")
    sandbox.task("T-0101", "in progress", [claimed])

    res = sandbox.run()

    assert res.returncode != 0, (
        "a pending marker in the source excused a module no task claims; that turns 'mark it "
        f"pending' into a mute button\n{_diag(res)}"
    )
    assert expected_lines(res) == [
        f"expected: T-0101 (2 failing in {claimed}, implementer pending)"
    ], (
        "T-0101 owns test_alpha.py and nothing else; the smuggled module must not be excused nor "
        f"counted into T-0101's line\n{_diag(res)}"
    )
    assert [line for line in regression_lines(res) if smuggled in line], _diag(res)


# ---------------------------------------------------------------------------------------------
# Rulings A6 and A7 (2026-09-13 review) -- two ways a status can be made to excuse more than the
# module it owns. Both were holes in the first implementation.
# ---------------------------------------------------------------------------------------------


def test_collection_error_in_an_in_progress_module_must_not_excuse_the_whole_suite(
    sandbox: StopSandbox,
) -> None:
    """Ruling A6: a collection abort is a regression whatever the module's owner.

    ``test_alpha.py`` cannot be imported, so pytest aborts the session: *nothing runs*, including
    the three real failures waiting in the unclaimed ``test_gamma.py``. A classifier that reads
    ``ERROR tests/unit/test_alpha.py`` as "N failing in a module T-0101 owns" therefore hands out
    exit 0 for a suite that never executed, and the three regressions are invisible. The
    pre-T-0019 hook blocked here; no status may make it stop blocking.
    """
    broken = sandbox.uncollectable_module("tests/unit/test_alpha.py")
    unclaimed = sandbox.module("tests/unit/test_gamma.py", failing=3)
    sandbox.task("T-0101", "in progress", [broken])

    res = sandbox.run()

    assert res.returncode != 0, (
        "a module of an in-progress task failed to import, so no test in the suite ran -- "
        f"including the 3 failures in {unclaimed} -- and the hook let the turn finish\n{_diag(res)}"
    )
    assert not [line for line in expected_lines(res) if broken in line], (
        f"nothing ran in {broken}; there is no failure count to excuse\n{_diag(res)}"
    )
    assert [line for line in regression_lines(res) if broken in line], (
        f"the collection abort in {broken} must be reported as the regression it is\n{_diag(res)}"
    )


def test_in_progress_mention_cannot_excuse_a_merged_modules_failure(sandbox: StopSandbox) -> None:
    """Ruling A7: precedence -- a merged owner beats an in-progress mention of the same module.

    Task files name other tasks' modules all the time (T-0019.md itself names
    ``tests/unit/test_speech_gate.py`` while quoting the session that motivated it). Under the
    whole-file ownership scan of ruling A1 that mention would otherwise be enough to excuse a
    regression in merged code -- the one thing the hook exists to catch. Ruling A10 narrows the
    precedence to *structured* claims, so here T-0102 names the merged module in its Goal prose
    only; the sibling test below is the case where it claims it under "Tests to write" instead.
    """
    merged_module = sandbox.module("tests/unit/test_alpha.py", failing=2)
    own_module = sandbox.module("tests/unit/test_beta.py", failing=3)
    sandbox.task("T-0104", "merged 2026-09-13 (a640f89; re-review MERGE)", [merged_module])
    sandbox.task("T-0102", "in progress", [own_module], mentions=[merged_module])

    res = sandbox.run()

    assert res.returncode != 0, (
        f"{merged_module} is merged code gone red; a passing mention of it in an in-progress "
        f"task file excused it\n{_diag(res)}"
    )
    assert not [line for line in expected_lines(res) if merged_module in line], _diag(res)
    assert [line for line in regression_lines(res) if merged_module in line], (
        f"{merged_module} must be named as a regression\n{_diag(res)}"
    )
    assert f"expected: T-0102 (3 failing in {own_module}, implementer pending)" in _report(res), (
        f"T-0102's own module stays tolerated and attributed\n{_diag(res)}"
    )


def test_a_structured_claim_by_an_in_progress_task_takes_over_a_merged_modules_failure(
    sandbox: StopSandbox,
) -> None:
    """Ruling A10: the other half of precedence -- an open window covers amended modules too.

    Merged T-0104 created ``test_alpha.py``; in-progress T-0102 is amending it and says so under
    "Tests to write", exactly as T-0026 amends ``test_guard_bash_hook.py`` (created by merged
    T-0016) and T-0025 amends ``test_hud_constants_parity.py``. That is the prescribed red window
    for those two failures, so the turn may finish and the red is attributed to T-0102 -- the task
    actually holding the pen -- not to the merged creator that cannot fix it.

    The contrast with the prose-only case above is the whole content of A10: same two statuses,
    same module, and the verdict flips on which section names it.
    """
    module = sandbox.module("tests/unit/test_alpha.py", failing=2)
    sandbox.task("T-0104", "merged 2026-09-13 (a640f89; re-review MERGE)", [module])
    sandbox.task("T-0102", "in progress", [module])

    res = sandbox.run()

    assert res.returncode == 0, (
        f"T-0102 claims {module} under '## Tests to write' and is in progress; blocking here is "
        f"the cry-wolf failure T-0019 exists to remove\n{_diag(res)}"
    )
    assert expected_lines(res) == [
        f"expected: T-0102 (2 failing in {module}, implementer pending)"
    ], (
        f"the red belongs to the task holding the pen, not to {module}'s merged creator"
        f"\n{_diag(res)}"
    )


# ---------------------------------------------------------------------------------------------
# Rulings A11 and A12 (2026-09-13 re-review) -- what counts as a collection abort, and which
# stream each verdict is written to.
# ---------------------------------------------------------------------------------------------


# The banner pytest prints when collection aborts, as a failing test might quote it. It is only
# ever *quoted* here, so every line carrying it starts with pytest's own ``E``/``>`` gutter and it
# never begins a line. The decoration width is a parameter because nothing in ruling A11 makes the
# verdict depend on it: pytest's own banner is padded to the terminal width, and a test quoting a
# narrow one is quoting the same thing.
def _banner(bangs: int) -> str:
    return "!" * bangs + " Interrupted: 1 error during collection " + "!" * bangs


def _banner_quoting_module(bangs: int) -> str:
    """A module of three ordinary failures, one of which quotes the abort banner in its message.

    The literal lives in a module constant, never in the ``assert`` statement, so the source line
    pytest echoes does not contain it either -- the only lines carrying it are ``E``-prefixed.
    """
    return f"""BANNER = {_banner(bangs)!r}


def test_quotes_the_banner_in_its_message():
    assert "" == BANNER, "the model replied: " + BANNER


def test_ordinary_failure_a():
    assert 2 + 2 == 5


def test_ordinary_failure_b():
    assert 2 + 2 == 5
"""


@pytest.mark.parametrize("bangs", [20, 8, 4])
def test_an_interrupted_banner_quoted_mid_line_is_not_a_collection_abort(
    tmp_path: Path, bangs: int
) -> None:
    """Ruling A11: the banner counts only at line start, so test output cannot fake an abort.

    ``test_quotes_the_banner_in_its_message`` fails with the collection-abort banner inside its
    assertion message; pytest prints it behind its own ``E``/``>`` gutter, never at the start of a
    line. All three failures are ordinary failures in a module T-0101 owns, so the turn may finish
    -- a classifier that looks for the banner anywhere, or in a fixed-width prefix of a line,
    blocks on a string a test happened to mention, which is the cry-wolf failure again.

    The second sandbox is the genuine article (a module that cannot be imported, so pytest really
    does abort) and must reach the opposite verdict on the same hook. The only difference between
    the two payloads is whether the banner begins a line.
    """
    quoting = _make_sandbox(tmp_path / f"quoting{bangs}")
    module = quoting.raw_module("tests/unit/test_alpha.py", _banner_quoting_module(bangs))
    quoting.task("T-0101", "in progress", [module])
    quoted = quoting.run()

    genuine = _make_sandbox(tmp_path / f"genuine{bangs}")
    broken = genuine.uncollectable_module("tests/unit/test_alpha.py")
    genuine.task("T-0101", "in progress", [broken])
    aborted = genuine.run()

    assert quoted.returncode == 0, (
        "a failing test quoted the collection-abort banner in its message and the hook treated the "
        f"run as an abort\n{_diag(quoted)}"
    )
    assert expected_lines(quoted) == [
        f"expected: T-0101 (3 failing in {module}, implementer pending)"
    ], f"all three are ordinary failures and must be counted as such\n{_diag(quoted)}"
    assert aborted.returncode != 0, (
        f"the genuine collection abort must still block\n{_diag(aborted)}"
    )
    assert [line for line in regression_lines(aborted) if broken in line], _diag(aborted)


def test_the_tolerated_report_goes_to_stdout_and_stderr_stays_empty(sandbox: StopSandbox) -> None:
    """Ruling A12: the tolerated report is transcript-visible, so it belongs on stdout.

    A Stop hook's stderr is what gets fed back to the agent on a block; writing the tolerated
    block there would read as a failure report on a turn that is allowed to finish.
    """
    module = sandbox.module("tests/unit/test_alpha.py", failing=2)
    sandbox.task("T-0101", "in progress", [module])

    res = sandbox.run()

    assert res.returncode == 0, _diag(res)
    assert f"expected: T-0101 (2 failing in {module}, implementer pending)" in res.stdout, (
        f"the expected block must be on stdout\n{_diag(res)}"
    )
    assert res.stderr.strip() == "", f"nothing was blocked, so stderr must stay empty\n{_diag(res)}"


def test_the_blocking_report_goes_to_stderr(sandbox: StopSandbox) -> None:
    """Ruling A12, the other half: what blocks the turn is fed back, so it belongs on stderr."""
    claimed = sandbox.module("tests/unit/test_alpha.py", failing=2)
    unclaimed = sandbox.module("tests/unit/test_gamma.py", failing=1)
    sandbox.task("T-0101", "in progress", [claimed])

    res = sandbox.run()

    assert res.returncode != 0, _diag(res)
    assert [
        line
        for line in res.stderr.splitlines()
        if unclaimed in line and re.search(r"regression", line, re.I)
    ], f"the regression that blocked the turn must be on stderr\n{_diag(res)}"
    assert not [
        line
        for line in res.stdout.splitlines()
        if line.strip().startswith("expected:") or re.search(r"regression", line, re.I)
    ], f"the blocking report must not be split across both streams\n{_diag(res)}"


# ---------------------------------------------------------------------------------------------
# Ruling A14 (2026-09-14, third review) -- what counts as a result line, and what counts as a
# report. Both are ways for text that is not a verdict to be read as one.
# ---------------------------------------------------------------------------------------------

# A test that quotes pytest's own short-summary format in its captured output, indented the way
# an agent transcript or a nested pytest run would print it. ``tests/unit/test_ghost.py`` does
# not exist and no task claims it, so counting this line invents an unclaimed failing module out
# of thin air -- and inventing one is a block.
GHOST_QUOTING_MODULE = """def test_prints_a_summary_line_it_read_somewhere():
    print("  FAILED tests/unit/test_ghost.py::test_x - AssertionError: assert 1 == 2")
    assert 2 + 2 == 5


def test_ordinary_failure():
    assert 2 + 2 == 5
"""


def test_a_failed_line_quoted_in_captured_output_is_not_counted(sandbox: StopSandbox) -> None:
    """Ruling A14: ``FAILED``/``ERROR`` count at line start only.

    pytest reproduces a failing test's captured stdout verbatim, indentation and all, so a test
    that quotes a summary line -- an agent transcript under assertion, a nested pytest run, a
    fixture echoing a log -- injects text that looks exactly like a result. Counting it conjures
    ``tests/unit/test_ghost.py``, which exists nowhere and which no task claims, so the safe
    default (unclaimed module = regression) turns a prescribed red window into a blocked turn.
    """
    module = sandbox.raw_module("tests/unit/test_alpha.py", GHOST_QUOTING_MODULE)
    sandbox.task("T-0101", "in progress", [module])

    res = sandbox.run()

    assert res.returncode == 0, (
        "a failing test quoted a FAILED summary line in its captured output and the hook counted "
        f"it as a real result\n{_diag(res)}"
    )
    assert expected_lines(res) == [
        f"expected: T-0101 (2 failing in {module}, implementer pending)"
    ], f"two tests failed, both in {module}\n{_diag(res)}"
    assert "test_ghost" not in _report(res), (
        f"a module that does not exist was reported as failing\n{_diag(res)}"
    )


def test_a_report_without_an_attributed_line_blocks_the_turn(tmp_path: Path) -> None:
    """Ruling A14: a helper that exits 0 with chatter has not classified anything.

    The tolerated path is entered on two conditions -- the classifier exited 0 *and* it produced
    a report. If "a report" means "any output", then a helper that writes an unrelated note to
    stderr and exits 0 (an unparseable task file, a deprecation warning) silently converts every
    red suite into a finished turn: the exact hole the hook exists to close, opened by accident.
    Here the suite is one an in-progress task would otherwise excuse, so nothing but the missing
    attribution can be what blocks.
    """
    sandbox = _make_sandbox(tmp_path / "repo", chatty_helper=True)
    module = sandbox.module("tests/unit/test_alpha.py", failing=2)
    sandbox.task("T-0101", "in progress", [module])

    res = sandbox.run()

    assert res.returncode != 0, (
        "the classifier attributed nothing -- it only chattered on stderr -- and the hook let a "
        f"red suite finish the turn\n{_diag(res)}"
    )
    assert expected_lines(res) == [], (
        f"nothing was classified, so nothing may be presented as expected\n{_diag(res)}"
    )
    assert module in res.stderr, (
        f"with no usable report the hook must fall back to showing the run itself\n{_diag(res)}"
    )


# ---------------------------------------------------------------------------------------------
# T-0031 -- the same gate, run in parallel: the command line, the timeout, and the two shapes of
# xdist output the classifier must survive.
# ---------------------------------------------------------------------------------------------

CLASSIFIER = HOOK.resolve().parent / "classify_failures.py"


def hook_pytest_args() -> list[str]:
    """The argument vector the hook hands to pytest, as tokens."""
    for line in HOOK.read_text().splitlines():
        if re.search(r"\bpytest\b", line) and "tests/unit" in line:
            fragment = line[line.index("pytest") + len("pytest") :].split("2>&1")[0]
            return shlex.split(fragment.rstrip(") \t;"))
    raise AssertionError(f"{HOOK} no longer invokes pytest on tests/unit")


def hook_timeout_seconds() -> int:
    """The ``timeout N`` the hook wraps the suite in."""
    for line in HOOK.read_text().splitlines():
        if "pytest" in line and "tests/unit" in line:
            m = re.search(r"\btimeout\s+(\d+)\b", line)
            assert m, f"the hook no longer wraps the fast suite in a timeout: {line.strip()!r}"
            return int(m.group(1))
    raise AssertionError(f"{HOOK} no longer invokes pytest on tests/unit")


def test_the_hook_runs_the_suite_in_parallel_with_file_level_distribution() -> None:
    """T-0031 criterion 1: the per-turn run is parallel, and nothing else about it moved.

    ``--dist loadfile`` rather than the default ``load`` because a module's tests must not
    interleave across workers (the audio modules assert against elapsed time), and a *fixed*
    worker count rather than ``-n auto`` because the box also hosts vLLM. The selection tokens
    are asserted too: parallelism is the only thing this task is allowed to change about the
    command.
    """
    args = hook_pytest_args()
    for token in ("tests/unit", "tests/contract", "-q", "-rfE"):
        assert token in args, f"the hook's pytest command lost {token!r}: {args}"
    assert args[args.index("-m") + 1] == "not gpu", f"the marker expression changed: {args}"

    workers = [a for a in args if a == "-n" or a.startswith(("-n", "--numprocesses"))]
    assert workers, (
        "the Stop hook still runs the fast suite in one process; T-0031 makes it "
        f"`-n <N> --dist loadfile` with a fixed 4 <= N <= 12: {args}"
    )
    if args[args.index(workers[0])] in ("-n", "--numprocesses"):
        value = args[args.index(workers[0]) + 1]
    else:
        value = re.split(r"^-n|^--numprocesses=?", workers[0])[-1]
    assert value.isdigit(), (
        f"the worker count is {value!r}; it must be a fixed integer, not 'auto' or 'logical' -- "
        "the box hosts vLLM and a 32-way run is not what was measured"
    )
    assert 4 <= int(value) <= 12, f"worker count {value} is outside the measured range 4..12"

    dist = [args[i + 1] for i, a in enumerate(args) if a == "--dist"] + [
        a.split("=", 1)[1] for a in args if a.startswith("--dist=")
    ]
    assert dist == ["loadfile"], (
        f"distribution is {dist or 'default (load)'}; T-0031 requires --dist loadfile so a "
        "module's tests never interleave across workers"
    )


def test_the_hook_timeout_is_at_most_two_minutes() -> None:
    """Criterion 6: the timeout matches the new scale.

    280 s was a ceiling for a 135 s serial run. Once the run is ~30 s, a timeout that long means
    a hung worker costs every agent nearly five minutes before the turn is blocked.
    """
    seconds = hook_timeout_seconds()
    assert seconds <= 120, (
        f"the hook still allows {seconds} s for a run budgeted at 60 s (criterion 6)"
    )
    assert seconds >= 60, (
        f"the timeout ({seconds} s) is at or below the run's own budget: a slow-but-healthy run "
        "would be killed and reported as a crash"
    )


# Both payloads below carry the *same* failure set: two tests in test_alpha.py, one in
# test_beta.py. The serial one is pytest's ordinary ``-q -rfE`` output; the xdist one was
# captured from a real ``pytest -q -rfE -n 2 --dist loadfile`` run (pytest 8, pytest-xdist 3.8.0)
# and its paths rewritten -- hence the worker-completion ordering (test_beta first), the
# ``bringing up nodes...`` banner and the ``[gwN] linux -- Python ...`` line under each failure.
SERIAL_OUTPUT = """\
FFF..                                                                    [100%]
=================================== FAILURES ===================================
___________________________________ test_a1 ____________________________________

    def test_a1():
>       assert 2 + 2 == 5
E       assert (2 + 2) == 5

tests/unit/test_alpha.py:2: AssertionError
___________________________________ test_a2 ____________________________________

    def test_a2():
>       assert 2 + 2 == 5
E       assert (2 + 2) == 5

tests/unit/test_alpha.py:6: AssertionError
___________________________________ test_b1 ____________________________________

    def test_b1():
>       assert 2 + 2 == 5
E       assert (2 + 2) == 5

tests/unit/test_beta.py:2: AssertionError
=========================== short test summary info ============================
FAILED tests/unit/test_alpha.py::test_a1 - assert (2 + 2) == 5
FAILED tests/unit/test_alpha.py::test_a2 - assert (2 + 2) == 5
FAILED tests/unit/test_beta.py::test_b1 - assert (2 + 2) == 5
3 failed, 2 passed in 0.21s
"""

XDIST_OUTPUT = """\
bringing up nodes...

FF.F.                                                                    [100%]
=================================== FAILURES ===================================
___________________________________ test_b1 ____________________________________
[gw1] linux -- Python 3.11.16 /home/user/demos/local_astra/.venv/bin/python3

    def test_b1():
>       assert 2 + 2 == 5
E       assert (2 + 2) == 5

tests/unit/test_beta.py:2: AssertionError
___________________________________ test_a1 ____________________________________
[gw0] linux -- Python 3.11.16 /home/user/demos/local_astra/.venv/bin/python3

    def test_a1():
>       assert 2 + 2 == 5
E       assert (2 + 2) == 5

tests/unit/test_alpha.py:2: AssertionError
___________________________________ test_a2 ____________________________________
[gw0] linux -- Python 3.11.16 /home/user/demos/local_astra/.venv/bin/python3

    def test_a2():
>       assert 2 + 2 == 5
E       assert (2 + 2) == 5

tests/unit/test_alpha.py:6: AssertionError
=========================== short test summary info ============================
FAILED tests/unit/test_beta.py::test_b1 - assert (2 + 2) == 5
FAILED tests/unit/test_alpha.py::test_a1 - assert (2 + 2) == 5
FAILED tests/unit/test_alpha.py::test_a2 - assert (2 + 2) == 5
3 failed, 2 passed in 0.31s
"""


def _classify(sandbox: StopSandbox, output: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLASSIFIER), str(sandbox.top)],
        input=output,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("status", "expect_exit"), [("in progress", 0), ("merged 2026-09-01", 1)], ids=["window", "red"]
)
def test_an_xdist_summary_is_classified_exactly_like_the_serial_one(
    sandbox: StopSandbox, status: str, expect_exit: int
) -> None:
    """T-0031 criterion 4 (**adversarial**): parallelism must not change a single verdict.

    xdist reorders the short-summary lines (worker completion order, not collection order) and
    puts ``[gwN]`` markers in the failure bodies, but it leaves the ``FAILED`` lines themselves
    unprefixed and at line start -- which is the only thing the classifier reads (ruling A14).
    The two runs therefore have to produce byte-identical reports *and* the same exit code, once
    the lines are compared as sets: an ordering the scheduler chose must not decide whether a
    turn is blocked. Both statuses are exercised so a classifier that became order-blind by
    ignoring everything would still be caught by the regression row.
    """
    sandbox.module("tests/unit/test_alpha.py", failing=2)
    sandbox.module("tests/unit/test_beta.py", failing=1)
    sandbox.task("T-0101", status, ["tests/unit/test_alpha.py", "tests/unit/test_beta.py"])

    serial = _classify(sandbox, SERIAL_OUTPUT)
    parallel = _classify(sandbox, XDIST_OUTPUT)

    assert serial.returncode == expect_exit, (
        f"the serial fixture is the control and must exit {expect_exit}\n{serial.stdout}"
        f"{serial.stderr}"
    )
    assert parallel.returncode == serial.returncode, (
        f"xdist output exits {parallel.returncode}, serial {serial.returncode}, for the same "
        f"failure set\n--- parallel ---\n{parallel.stdout}{parallel.stderr}"
        f"\n--- serial ---\n{serial.stdout}"
    )
    assert sorted(parallel.stdout.splitlines()) == sorted(serial.stdout.splitlines()), (
        "the verdict text differs between the parallel and the serial run of the same failures"
        f"\n--- parallel ---\n{parallel.stdout}\n--- serial ---\n{serial.stdout}"
    )
    assert "3 failing in tests/unit/test_alpha.py" not in parallel.stdout, (
        "the [gw0] body lines were counted as results: test_alpha.py has two failures, not three"
        f"\n{parallel.stdout}"
    )


# A real capture (pytest 8, pytest-xdist 3.8.0, ``-q -rfE -n 2 --dist loadfile``) of a worker that
# died at import time: every node goes down, pytest exits 5, and **not one FAILED or ERROR line is
# printed**. The ``replacing crashed worker gwN`` line is xdist's other crash banner
# (xdist/dsession.py:264) and appears when a worker dies mid-run. Nothing here is attributable, so
# nothing here may be excused.
CRASHED_WORKER_OUTPUT = """\
bringing up nodes...
[gw0] node down: Not properly terminated

replacing crashed worker gw0
[gw1] node down: Not properly terminated

replacing crashed worker gw1
[gw2] node down: Not properly terminated
[gw3] node down: Not properly terminated

no tests ran in 0.50s
"""

TIMEOUT_OUTPUT = """\
bringing up nodes...

.....F..                                                                 [ 42%]
"""


def _canned_sandbox(root: Path, output: str, rc: int) -> StopSandbox:
    """A sandbox whose ``uv run pytest`` replays a canned run instead of running one.

    The crash shapes this exercises (a worker dying at import time, ``timeout`` killing the run)
    cannot be produced reliably from inside a test, and faking them at the classifier's input is
    exactly where the hook's decision is made.
    """
    sandbox = _make_sandbox(root)
    canned = sandbox.top / "canned_output.txt"
    canned.write_text(output)
    (sandbox.top / "bin" / "uv").write_text(
        "#!/usr/bin/env bash\n"
        '[ "$1" = "run" ] && shift\n'
        'case "$1" in\n'
        f'  python|python3) shift; exec "{sys.executable}" "$@";;\n'
        "  mypy) exit 0;;\n"
        f'  pytest) cat "{canned}"; exit {rc};;\n'
        f'  *) exec "{sys.executable}" -m "$@";;\n'
        "esac\n"
    )
    (sandbox.top / "bin" / "uv").chmod(0o755)
    return sandbox


@pytest.mark.parametrize(
    ("output", "rc", "evidence"),
    [
        (CRASHED_WORKER_OUTPUT, 5, "node down"),
        (TIMEOUT_OUTPUT, 124, "42%"),
    ],
    ids=["worker-crash", "timeout"],
)
def test_a_crashed_worker_blocks_the_turn(
    tmp_path: Path, output: str, rc: int, evidence: str
) -> None:
    """T-0031 criterion 5 (**adversarial**): un-attributable red must never read as green.

    A parallel run has a failure mode the serial one does not: a worker can die, taking its share
    of the suite with it and printing no result line at all. The classifier then attributes
    nothing and exits 0 -- and an in-progress task is present here, so the *only* thing that can
    block is the hook noticing that a non-zero pytest was never explained (ruling A14). If this
    ever passed, every crashed worker would silently finish the turn with a third of the suite
    unrun.
    """
    sandbox = _canned_sandbox(tmp_path / "repo", output, rc)
    module = sandbox.module("tests/unit/test_alpha.py", failing=1)
    sandbox.task("T-0101", "in progress", [module])

    res = sandbox.run()

    assert res.returncode == 2, (
        "pytest failed with nothing attributable in its output -- a worker crash or a timeout -- "
        f"and the hook let the turn finish\n{_diag(res)}"
    )
    assert expected_lines(res) == [], (
        f"nothing was classified, so nothing may be presented as expected\n{_diag(res)}"
    )
    assert evidence in res.stderr, (
        f"the hook must show the tail of a run it could not classify; {evidence!r} is not in "
        f"stderr\n{_diag(res)}"
    )
