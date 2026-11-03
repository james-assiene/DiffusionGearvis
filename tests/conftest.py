"""Shared fixtures and the ``pending`` convention.

``@pytest.mark.pending(task="T-NNNN")`` marks a test written *ahead* of its implementation
(the acceptance test the task must make green). Until the task's test-author activates it
(removes the marker in ``/implement`` step 3), it runs as ``xfail(strict=True)``:

- it still executes, so a pending test that accidentally passes (a tautology, or a stub that
  already satisfies it) is reported as XPASS(strict) and fails the suite;
- it does not count as a failure while the implementation does not exist;
- ``scripts/check_pending.py`` lists them per task and checks the task file exists.

Run ``pytest --pending`` to execute pending tests as ordinary tests (see them fail for real).
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

from tests import _home_jail

# T-0175: the session home jail and the spawn guard, installed at the import of this module. Pytest imports it
# before any sub-conftest or test module, so ``tests/unit/conftest.py``'s imports, ``memory.store.DEFAULT_DIR`` and the
# hook modules' import-time ``GIT_ENV`` copies already see the jail, and so does every child (old BASE trees, CLI
# rigs, ``uv run``, nested pytest, browsers). xdist workers and nested runs inherit the marker and reuse the jail.
# The real home holds the Brave key file; see tests/_home_jail.py and docs/tasks/T-0175.md.
HOME_JAIL = _home_jail.install()

# T-0186 criterion 1: strip git's repo selectors (GIT_DIR, GIT_WORK_TREE, GIT_INDEX_FILE, ...) before any test or
# fixture runs, so no test's git child names a repo it did not pass on its own command line. On 2026-10-28 an exported
# GIT_DIR made a sandbox's `git init <tmp>` re-initialise the main repo and its commit landed on `main`. The spawn
# guard's git clause (tests/_home_jail.py) refuses a git child that still carries GIT_DIR or GIT_WORK_TREE.
GIT_SELECTORS_STRIPPED = _home_jail.strip_git_selectors()

# Task ids are ``T-NNNN`` with an optional letter suffix for a split task
# (``docs/tasks/T-0007a.md`` is T-0007 group A as its own task).
TASK_ID = re.compile(r"^T-\d{4}[a-z]?$")
REPO = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


# The fast suite's two budgets (docs/tasks/T-0031.md). The Stop hook runs
# ``tests/unit tests/contract -m "not gpu"`` in parallel on every agent finish and must come back
# under ``STOP_HOOK_BUDGET_S``; the canonical *serial* run (``make fast``, what the verifier and
# ``make check`` run) keeps CLAUDE.md's two-minute ceiling. Both numbers are informational here:
# the terminal-summary line below reports the wall beside them and never changes the exit code,
# because a loaded box is not a red suite. The acceptance measurement is taken on a clean tree.
STOP_HOOK_BUDGET_S = 60.0
SERIAL_CEILING_S = 120.0
_SESSION_START = time.monotonic()


# T-0146: inside an xdist worker (the Stop hook, `make fast-parallel`) Hypothesis runs every phase
# but shrink/explain, so a broken invariant is reported at once, by name, with its raw counterexample
# instead of shrinking past the hook's `timeout 110`. Generation is untouched (same examples per seed),
# so detection is unchanged; the serial canonical run still shrinks. Loaded in pytest_configure, before
# any test module is imported, because a settings object takes the profile active when it is built.
# trylast: compose on top of hypothesis's own configure (verbosity/explain options), and an explicit
# --hypothesis-profile always wins (H-T146-1: trigger on being a worker, no hook change).
_GATE_PROFILE = "astra-gate-no-shrink"


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    # T-0175: the first config of the process removes the jail it created. A pytest process with no jail in
    # force fails loudly here (review F1); library imports of this module never reach pytest_configure.
    _home_jail.claim_owner(config)
    _home_jail.state()
    if not hasattr(config, "workerinput") or config.getoption("hypothesis_profile", None):
        return
    from hypothesis import Phase, settings

    parent = settings.get_profile(settings.get_current_profile_name())
    settings.register_profile(
        _GATE_PROFILE,
        parent=parent,
        phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.target),
    )
    settings.load_profile(_GATE_PROFILE)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--pending",
        action="store_true",
        default=False,
        help="run tests marked pending as ordinary tests instead of strict xfail",
    )


def pytest_unconfigure(config: pytest.Config) -> None:
    """T-0175 criterion 9: the process that created the jail removes it (symlinks unlinked, never followed)."""
    _home_jail.cleanup(config)


#: T-0175 criterion 7: why ``real_home_env`` refused.
REAL_HOME_ENV_REFUSED = (
    "T-0175: real_home_env hands a child the real home, where the Brave Search API key file lives; it runs only "
    "under ASTRA_LIVE_SEARCH=1 (make search-live, H-T163-5). A row in the fast suite that requests it is a bug."
)


@pytest.fixture
def real_home_env() -> Iterator[dict[str, str]]:
    """T-0175 criterion 7: the one escape from the session jail. An env whose ``HOME`` is the pre-switch home (and
    no ``XDG_CONFIG_HOME``), for the opt-in live Brave rows; while the row runs, this process's spawn guard permits
    that home. Fails (never skips) unless ``ASTRA_LIVE_SEARCH=1``."""
    if os.environ.get("ASTRA_LIVE_SEARCH") != "1":
        pytest.fail(REAL_HOME_ENV_REFUSED, pytrace=False)
    with _home_jail.escape() as real:
        env = {k: v for k, v in os.environ.items() if k != "XDG_CONFIG_HOME"}
        env["HOME"] = real
        yield env


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_pending = config.getoption("--pending")
    report: list[dict[str, str]] = []
    for item in items:
        mark = item.get_closest_marker("pending")
        if mark is None:
            continue
        task = mark.kwargs.get("task")
        if not isinstance(task, str) or not TASK_ID.match(task):
            raise pytest.UsageError(f"{item.nodeid}: pending marker needs task='T-NNNN'")
        if not (REPO / "docs" / "tasks" / f"{task}.md").exists():
            raise pytest.UsageError(
                f"{item.nodeid}: pending task {task} has no docs/tasks/{task}.md"
            )
        report.append({"nodeid": item.nodeid, "task": task})
        if not run_pending:
            # Only the two legitimate "not implemented yet" outcomes are tolerated. A wrong
            # attribute name, a bad call signature or an import problem in a pending test is a
            # bug in the test and must surface as a failure, not hide as XFAIL.
            item.add_marker(
                pytest.mark.xfail(
                    strict=True,
                    raises=(NotImplementedError, AssertionError),
                    reason=f"pending {task}: activate by removing the marker (docs/tasks/{task}.md)",
                )
            )
    out = os.environ.get("ASTRA_PENDING_REPORT")
    if out:
        Path(out).write_text(json.dumps(report, indent=1))


# ---------------------------------------------------------------------------------------------
# T-0057 criteria 1, 2 and 4 -- a thread that outlives its test fails that test, by name.
# ---------------------------------------------------------------------------------------------
#
# ``PipeWireAudioIO`` starts ``astra-audio-writer`` on the first ``play()`` and, until T-0057,
# had no way to stop it: the thread's only exit was 2.0 s of idleness, so 48 of the 66 rows in
# four audio modules handed the *next* row a live thread reading the device ~4 000 times a
# second (docs/tasks/T-0057.md §Regime; reproduced on this tree: 7 960 reads in 1.99 s). The only
# signal was a ``PytestUnhandledThreadExceptionWarning`` that showed up in one run out of three.
# A warning nobody reads is the current behaviour; this is the failure that replaces it.
#
# Two halves, because a guard needs a row for being too wide as well as one for being absent
# (.claude/rules/tests.md):
#
# * ``astra_thread_leak_guard`` attributes to a row only threads that were **not** alive when the
#   row's fixtures started and whose name begins ``astra-``. The prefix is the whole tolerance
#   rule -- ``MainThread``, ``asyncio_0``, ``ThreadPoolExecutor-*`` and the xdist workers are
#   never named, and never enumerated either. A thread an earlier row leaked is in the next row's
#   *before* set, so it is reported once, against its creator.
# * ``astra_audio_modules_are_stopped`` calls the stop seam of every ``PipeWireAudioIO`` the row
#   built, *before* the guard looks (it depends on the guard, so it is torn down first). That is
#   what lets the ~60 existing rows stop leaking without a single edit to them (criterion 6).
#
# Both are autouse and run for every row in the suite. Neither ever looks at ``daemon``: a daemon
# thread still runs, still polls the device and still burns the CPU the planner measured.

#: The only names ever attributed to a row. A prefix rule, not a list of tolerated names.
ASTRA_THREAD_PREFIX = "astra-"


def live_astra_threads() -> set[threading.Thread]:
    """Threads this project started that are alive *now*. Identity, not name: a name can repeat."""
    return {
        t for t in threading.enumerate() if t.is_alive() and t.name.startswith(ASTRA_THREAD_PREFIX)
    }


@pytest.fixture(autouse=True)
def astra_thread_leak_guard(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail the row that left an ``astra-`` thread running, naming the row and the thread."""
    before = live_astra_threads()
    yield
    leaked = sorted(
        f"{t.name} (daemon={t.daemon}, alive={t.is_alive()})" for t in live_astra_threads() - before
    )
    if leaked:
        raise AssertionError(
            f"{request.node.nodeid} finished but left {len(leaked)} thread(s) running: "
            + ", ".join(leaked)
            + ". A thread that outlives its test keeps reading the device and whatever the next "
            "row patched, so the next row measures this one (docs/tasks/T-0057.md criterion 1). "
            "Daemonising it does not help: it is still running. Stop it before the row ends."
        )


def _audio_device_fakes() -> ModuleType | None:
    """``tests.fakes.audio_device`` if this row ever imported it; never imports it itself.

    A dict lookup, so a row that has nothing to do with audio pays nothing -- and the tracking
    that this reads is installed by that module *at import*, which is collection time for every
    module that builds a playback module.
    """
    return sys.modules.get("tests.fakes.audio_device")


@pytest.fixture(autouse=True)
def astra_audio_modules_are_stopped(astra_thread_leak_guard: None) -> Iterator[None]:
    """Stop every ``PipeWireAudioIO`` this row built (T-0057 criteria 3 and 4).

    Depends on the guard on purpose: fixtures are finalised last-in-first-out, so this teardown
    runs *before* the guard looks at the thread set. Errors are not swallowed -- a stop seam that
    raises is a failed row, not a silent leak.
    """
    yield
    fakes = _audio_device_fakes()
    if fakes is not None:
        fakes.stop_tracked_audio_modules()


@dataclass
class VirtualClock:
    """Deterministic session clock in ms. Tests advance it explicitly; nothing sleeps."""

    now_ms: int = 0
    _log: list[int] = field(default_factory=list)

    def advance(self, ms: int) -> int:
        if ms < 0:
            raise ValueError("time does not go backwards")
        self.now_ms += ms
        self._log.append(self.now_ms)
        return self.now_ms

    def ticks(self) -> list[int]:
        return list(self._log)


@pytest.fixture
def clock() -> VirtualClock:
    return VirtualClock()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def quiz_example_text(fixtures_dir: Path) -> str:
    return (fixtures_dir / "a2ui" / "quiz_example.jsonl").read_text()


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Report this run's wall beside the two budgets. Informational only (T-0031).

    ``ASTRA_SUITE_WALL_S`` substitutes a wall, so the line itself can be tested without waiting
    for one.
    """
    override = os.environ.get("ASTRA_SUITE_WALL_S")
    wall = float(override) if override else time.monotonic() - _SESSION_START
    terminalreporter.write_line(
        f"fast suite: {wall:.1f} s wall "
        f"(Stop-hook budget {STOP_HOOK_BUDGET_S:.0f} s parallel, "
        f"serial ceiling {SERIAL_CEILING_S:.0f} s) — informational, see docs/tasks/T-0031.md"
    )
