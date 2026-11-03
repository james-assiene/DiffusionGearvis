"""The fast suite's budget, its selection and the places that state it (T-0031).

The Stop hook (``.claude/hooks/stop_fast_suite.sh``) runs the fast suite on **every** agent
finish, so its wall is a tax on every turn. T-0031 makes that run parallel (`pytest-xdist`,
fixed worker count, file-level distribution) **without moving a single test out of the
selection**. The cases here pin the parts of that which are mechanically checkable:

* the parallelism lives in the hook's command line only — the project default stays
  single-process, because ``scripts/check_pending.py`` collects in a subprocess and every
  process writes ``$ASTRA_PENDING_REPORT`` from ``pytest_collection_modifyitems``; under xdist
  each worker would overwrite it and the pending backlog would silently truncate (criterion 7);
* the node-id set the hook collects is exactly the set the canonical serial command collects
  (criterion 2) — no marker, no ``--ignore``, no ``deselect``;
* the three documents that state the budget agree, and none of them claims the fast suite is run
  with ``-m "not slow"`` when no invocation does (criterion 9);
* the verifier's pre-merge run is still serial (criterion 10);
* ``pytest-xdist`` is declared in the ``dev`` extra and importable (criterion 8).

The wall itself (≤ 60 s over 5 consecutive runs) is deliberately *not* a test: it is a
measurement taken on a clean tree by the implementer and re-taken by the verifier, with the
regime quoted (docs/tasks/T-0031.md §Regime). What is a test is that the number is *stated* in
the terminal summary and in the three documents.
"""

from __future__ import annotations

import importlib
import os
import re
import shlex
import subprocess
import sys
import tomllib
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / ".claude" / "hooks" / "stop_fast_suite.sh"
PYPROJECT = REPO / "pyproject.toml"
CLAUDE_MD = REPO / "CLAUDE.md"
RULES = REPO / ".claude" / "rules" / "tests.md"
WORKFLOW = REPO / "docs" / "09-agent-workflow.md"
VERIFIER = REPO / ".claude" / "agents" / "verifier.md"
MAKEFILE = REPO / "Makefile"

# T-0143: the Stop-hook gate's xdist worker count. 12 is the top of the 4..12 range T-0031
# measured and that ``test_stop_hook.py``'s bound row pins; going higher is H-T143-1.
STOP_HOOK_WORKERS = 12

sys.path.insert(0, str(REPO / "tests"))
# The two numbers live in tests/conftest.py, which also prints them in the terminal summary, so
# the documents and the summary line are checked against one definition.
_conftest = importlib.import_module("conftest")
STOP_HOOK_BUDGET_S: float = _conftest.STOP_HOOK_BUDGET_S
SERIAL_CEILING_S: float = _conftest.SERIAL_CEILING_S

# The canonical fast-suite selection, quoted from CLAUDE.md's Commands block. Everything in this
# module compares against *these* tokens, never against whatever the hook happens to say.
CANONICAL_PATHS = ("tests/unit", "tests/contract")
CANONICAL_MARKER = "not gpu"

# Flags that only change *how* the run is scheduled, never *what* it selects.
SCHEDULING_FLAGS = {"-n", "--numprocesses", "--dist", "--maxprocesses", "-p", "--durations"}


def _hook_pytest_line() -> str:
    for line in HOOK.read_text().splitlines():
        if re.search(r"\bpytest\b", line) and "tests/unit" in line:
            return line
    raise AssertionError(f"{HOOK} no longer invokes pytest on tests/unit")


def hook_pytest_args() -> list[str]:
    """The argument vector the Stop hook passes to pytest, as tokens."""
    line = _hook_pytest_line()
    fragment = line[line.index("pytest") + len("pytest") :].split("2>&1")[0]
    return shlex.split(fragment.rstrip(") \t;"))


def selection_of(args: list[str]) -> tuple[frozenset[str], str | None]:
    """``(selected paths, -m expression)`` — the part of an argv that decides *what* runs."""
    paths: set[str] = set()
    marker: str | None = None
    for i, tok in enumerate(args):
        if tok == "-m":
            marker = args[i + 1]
        elif tok.startswith("tests/"):
            paths.add(tok)
    return frozenset(paths), marker


def _collect(args: list[str]) -> set[str]:
    # ``-q`` twice means *quieter* and stops pytest listing node ids at all, so the run's own
    # reporting flags are dropped here; they change no selection.
    args = [a for a in args if a not in {"-q", "-v", "-rfE", "--tb=short"}]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"collection failed for {args}\n--- stdout ---\n{proc.stdout[-3000:]}\n"
        f"--- stderr ---\n{proc.stderr[-2000:]}"
    )
    return {
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip().startswith("tests/") and "::" in line
    }


# ---------------------------------------------------------------------------------------------
# Criterion 7 — parallelism lives in the hook's command line, never in the project default.
# ---------------------------------------------------------------------------------------------


def test_the_project_default_is_single_process() -> None:
    """``addopts`` (and the scripts that shell out to pytest) must not turn xdist on.

    ``scripts/check_pending.py`` runs ``pytest --collect-only`` in a subprocess and the report it
    reads back is written by ``tests/conftest.py`` from ``pytest_collection_modifyitems`` to a
    single path. Under xdist every worker runs that hook and the last writer wins, so a default
    ``-n`` would truncate the pending backlog silently — the worst kind of breakage, because the
    script would still exit 0.
    """
    data = tomllib.loads(PYPROJECT.read_text())
    addopts = data["tool"]["pytest"]["ini_options"].get("addopts", "")
    tokens = shlex.split(addopts) if isinstance(addopts, str) else list(addopts)
    offenders = [
        t
        for t in tokens
        if t in {"-n", "--numprocesses", "--dist"}
        or t.startswith(("-n", "--numprocesses=", "--dist="))
        or "xdist" in t
    ]
    assert offenders == [], (
        f"addopts activates xdist for every pytest run ({offenders}); the pending report is "
        "written once per process and the last worker would win. Parallelism belongs in the "
        "Stop hook's and make fast-parallel's explicit command lines (T-0031 criterion 7)"
    )
    for script in (REPO / "scripts" / "check_pending.py", REPO / "scripts" / "bootstrap.sh"):
        if not script.exists():
            continue
        text = script.read_text()
        for line in text.splitlines():
            if "pytest" not in line:
                continue
            assert not re.search(r"(?<![\w-])(-n\b|--numprocesses|--dist)", line), (
                f"{script.name} shells out to pytest with a parallel flag: {line.strip()!r} — "
                "the pending report cannot survive it (T-0031 criterion 7)"
            )


def test_the_pending_report_is_complete() -> None:
    """``check_pending.collect()`` must see every marker, checked against two oracles.

    Oracle 1 (source): a regex walk of ``tests/`` for ``pending(task="T-NNNN")`` gives the set of
    (module, task) pairs that must appear. Oracle 2 (collection): for the modules whose marker is
    a module-level ``pytestmark``, *every* test in the module is pending, so the count is the
    number of node ids a plain collection of that module reports — computed by a separate pytest
    process that knows nothing about the report file.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    import check_pending

    entries = check_pending.collect()
    got_pairs = {(e["nodeid"].split("::")[0], e["task"]) for e in entries}
    got_counts = Counter(e["nodeid"].split("::")[0] for e in entries)

    # Only real markers count: a decorator or a module-level ``pytestmark`` at line start. Prose
    # that *quotes* ``pending(task="T-0016")`` in a docstring, and the marker objects some modules
    # build inside ``parametrize`` data, are deliberately not part of this oracle — the first is
    # not a marker, the second is not visible in the source text.
    declared = re.compile(
        r"^\s*(?:@pytest\.mark\.pending|pytestmark\s*=\s*pytest\.mark\.pending)"
        r"\(task=[\"'](T-\d{4}[a-z]?)[\"']\)"
    )
    want_pairs: set[tuple[str, str]] = set()
    module_level: set[str] = set()
    for path in sorted((REPO / "tests").rglob("test_*.py")):
        rel = path.relative_to(REPO).as_posix()
        for line in path.read_text().splitlines():
            m = declared.match(line)
            if not m:
                continue
            want_pairs.add((rel, m.group(1)))
            if line.lstrip().startswith("pytestmark"):
                module_level.add(rel)

    assert want_pairs <= got_pairs, (
        "the pending report is missing markers that are in the sources: "
        f"{sorted(want_pairs - got_pairs)}. A truncated report is how a parallel collection "
        "would fail — every worker writes $ASTRA_PENDING_REPORT and the last one wins "
        "(T-0031 criterion 7)"
    )
    for module, task in sorted(got_pairs - want_pairs):
        assert "pending(task=" in (REPO / module).read_text(), (
            f"the report attributes a pending test in {module} to {task}, but that module "
            "carries no pending marker at all"
        )
    if module_level:
        ids = _collect(sorted(module_level))
        want_counts = Counter(nid.split("::")[0] for nid in ids)
        for module in sorted(module_level):
            assert got_counts[module] == want_counts[module], (
                f"{module} carries a module-level pending marker, so all "
                f"{want_counts[module]} of its tests are pending, but the report lists "
                f"{got_counts[module]} — the report was truncated"
            )


# ---------------------------------------------------------------------------------------------
# Criteria 2 and 8 — the dependency, and a selection that does not move.
# ---------------------------------------------------------------------------------------------


def test_xdist_is_a_declared_dev_dependency_and_importable() -> None:
    """Criterion 8: declared with a lower bound in the ``dev`` extra, and installed."""
    import importlib.util

    dev = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]["dev"]
    pins = [d for d in dev if re.match(r"^pytest-xdist\b", d)]
    assert pins, (
        f"pytest-xdist is not in pyproject.toml's dev extra (it has {dev}); the Stop hook's "
        "parallel run depends on it (T-0031 criterion 8)"
    )
    assert any(re.search(r">=\s*\d", p) for p in pins), (
        f"pytest-xdist is declared without a lower bound ({pins}); pin one (criterion 8)"
    )
    assert importlib.util.find_spec("xdist") is not None, (
        "pytest-xdist is not importable in this environment — `uv sync --all-extras` must "
        "install it, otherwise the Stop hook's `-n` flag turns every turn into a usage error "
        "(T-0031 criterion 8)"
    )


def test_the_parallel_and_serial_selections_are_the_same_set() -> None:
    """Criterion 2: the hook collects exactly what the canonical serial command collects.

    Two ``--collect-only -q`` subprocesses on this commit, compared as sets of node ids. The
    canonical side is the literal command from CLAUDE.md's Commands block, not something read
    back out of the hook, so a hook that quietly grew an ``--ignore`` or a narrower ``-m`` fails
    here rather than shrinking the gate.
    """
    args = hook_pytest_args()
    paths, marker = selection_of(args)
    assert paths == frozenset(CANONICAL_PATHS), (
        f"the Stop hook selects {sorted(paths)}, the canonical fast suite is "
        f"{list(CANONICAL_PATHS)} (criterion 2)"
    )
    assert marker == CANONICAL_MARKER, (
        f"the Stop hook's marker expression is {marker!r}, not {CANONICAL_MARKER!r} (criterion 2)"
    )
    forbidden = [t for t in args if t.startswith(("--ignore", "--deselect", "-k"))]
    assert forbidden == [], f"the Stop hook narrows the selection with {forbidden} (criterion 2)"

    canonical = [*CANONICAL_PATHS, "-m", CANONICAL_MARKER]
    assert _collect(args) == _collect(canonical), (
        "the hook's command and the canonical serial command collect different node ids; "
        "T-0031 moves no test out of the Stop-hook gate (criterion 2)"
    )


# ---------------------------------------------------------------------------------------------
# Criteria 9 and 10 — the documents.
# ---------------------------------------------------------------------------------------------


def _fast_suite_commands(text: str) -> list[list[str]]:
    """Every fast-suite pytest invocation quoted in a document, as argv token lists."""
    out: list[list[str]] = []
    for line in text.splitlines():
        if "pytest" not in line or "tests/unit" not in line:
            continue
        fragment = line[line.index("pytest") + len("pytest") :]
        fragment = fragment.split("#")[0].split("`")[0].strip().rstrip("`. ")
        try:
            out.append(shlex.split(fragment))
        except ValueError:  # an unbalanced quote in prose
            continue
    return out


def _section(text: str, number: int) -> str:
    m = re.search(rf"^## {number}\..*?(?=^## |\Z)", text, re.M | re.S)
    assert m, f"§{number} not found"
    return m.group(0)


def test_the_budget_is_stated_identically_in_every_place_that_states_it() -> None:
    """Criterion 9: three documents, one command and two numbers.

    Compared as parsed selections and as numbers, not as strings, so wording may differ. The
    ``not slow`` clause is the landmine: ``.claude/rules/tests.md`` describes a marker no
    invocation uses, and the fix is to describe reality — never to add ``-m "not slow"`` to the
    command.
    """
    docs = {
        "CLAUDE.md": CLAUDE_MD.read_text(),
        ".claude/rules/tests.md": RULES.read_text(),
        "docs/09-agent-workflow.md §3": _section(WORKFLOW.read_text(), 3),
    }
    selections: dict[str, tuple[frozenset[str], str | None]] = {}
    for name, text in docs.items():
        commands = _fast_suite_commands(text)
        assert commands, f"{name} states the fast-suite budget but quotes no command (criterion 9)"
        for cmd in commands:
            selections[name] = selection_of(cmd)
            assert selection_of(cmd) == (frozenset(CANONICAL_PATHS), CANONICAL_MARKER), (
                f"{name} quotes {' '.join(cmd)!r}, which is not the canonical fast-suite "
                f"selection {list(CANONICAL_PATHS)} -m {CANONICAL_MARKER!r} (criterion 9)"
            )
        assert "not slow" not in text, (
            f"{name} claims the fast suite is run with -m 'not slow'; no invocation does. "
            "Describe reality, do not add the flag (T-0031 landmine)"
        )
        budget = re.search(rf"\b{int(STOP_HOOK_BUDGET_S)}\s*s\b", text)
        assert budget, (
            f"{name} does not state the Stop-hook budget ({int(STOP_HOOK_BUDGET_S)} s); the "
            "three statements of the budget must agree (criterion 9)"
        )
        ceiling = re.search(rf"\b{int(SERIAL_CEILING_S)}\s*s\b|two[- ]minute|2 min", text)
        assert ceiling, (
            f"{name} does not state the serial ceiling ({int(SERIAL_CEILING_S)} s / two "
            "minutes) (criterion 9)"
        )
    assert len(set(selections.values())) == 1, f"the documents disagree: {selections}"


def test_the_verifier_still_names_a_serial_run() -> None:
    """Criterion 10: the pre-merge run is the serial one, and docs/09 §2 says so.

    The parallel run is a per-turn convenience; the number that goes in a PR, and the run whose
    ordering is comparable with the baselines, is serial.
    """
    verifier = VERIFIER.read_text()
    assert re.search(r"`?make fast`?(?!-)", verifier), (
        "verifier.md does not name the serial fast-suite command (`make fast`); 'run the fast "
        "suite' is ambiguous once a parallel command exists (T-0031 criterion 10)"
    )
    for line in verifier.splitlines():
        if "pytest" in line or "make fast" in line:
            assert not re.search(r"(?<![\w-])(-n \d|--numprocesses|--dist)", line), (
                f"the verifier's pre-merge run is parallel: {line.strip()!r} (criterion 10)"
            )
    section2 = _section(WORKFLOW.read_text(), 2)
    assert "canonical" in section2 and re.search(r"make fast|serial", section2), (
        "docs/09 §2 does not say which fast-suite run is canonical (criterion 10)"
    )


# ---------------------------------------------------------------------------------------------
# The terminal-summary line: the wall is reported, never enforced.
# ---------------------------------------------------------------------------------------------


def test_the_terminal_summary_names_the_wall_and_the_budget() -> None:
    """A run prints its own wall beside the budget, and that line never changes the exit code.

    ``ASTRA_SUITE_WALL_S`` fabricates a wall far over the budget: the line must report it and the
    run must still exit 0. Enforcing the budget from inside the suite would turn a loaded box
    into a red suite; the 60 s number is evidence for a human, measured on a clean tree.
    """
    env = {**os.environ, "ASTRA_SUITE_WALL_S": "999.0", "ASTRA_PENDING_REPORT": ""}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/unit/test_suite_budget.py",
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
    assert proc.returncode == 0, (
        f"the summary line must not change the exit code\n{proc.stdout[-2000:]}\n{proc.stderr}"
    )
    lines = [ln for ln in proc.stdout.splitlines() if "999" in ln]
    assert lines, (
        "no terminal-summary line reported the wall (fabricated as 999.0 s); the suite must "
        f"state its own wall (T-0031)\n{proc.stdout[-2000:]}"
    )
    line = lines[0]
    assert re.search(rf"\b{int(STOP_HOOK_BUDGET_S)}\b", line), (
        f"the summary line {line!r} reports a wall but not the {int(STOP_HOOK_BUDGET_S)} s "
        "Stop-hook budget it should be compared against"
    )


# ---------------------------------------------------------------------------------------------
# T-0143 — one worker count, stated in five places, and a command that moved only by it.
# ---------------------------------------------------------------------------------------------

# A worker-count flag in a command line or in prose: ``-n 12``, ``-n12``, ``-n=12``,
# ``--numprocesses 12``, ``--numprocesses=12``. The value is captured raw, so ``-n auto`` is
# reported as a non-integer rather than skipped. ``(?<![\w-])`` keeps ``--dist-n`` or ``x-n`` out.
_WORKERS_RE = re.compile(r"(?<![\w-])(?:-n|--numprocesses)(?:=|[ \t]+|(?=\d))([^\s`'\",;)]+)")


def worker_counts_in(text: str) -> list[str]:
    """Every worker-count value stated in ``text``, raw, in order of appearance."""
    return _WORKERS_RE.findall(text)


def worker_count_problems(places: Mapping[str, str]) -> tuple[int | None, list[str]]:
    """``(agreed value or None, problems)`` for a ``{place name: text}`` mapping.

    A place is a problem if it states no value (absent), a non-integer value, or more than one
    distinct value; and every place whose value differs from the most common one is named. Pure:
    the tests below call it on in-memory texts as well as on the five files.
    """
    problems: list[str] = []
    values: dict[str, int] = {}
    for name, text in places.items():
        raw = worker_counts_in(text)
        if not raw:
            problems.append(f"{name}: the worker count is absent")
            continue
        bad = [v for v in raw if not v.isdigit()]
        if bad:
            problems.append(f"{name}: the worker count is not an integer ({bad})")
            continue
        distinct = sorted({int(v) for v in raw})
        if len(distinct) > 1:
            problems.append(f"{name}: states several worker counts {distinct}")
            continue
        values[name] = distinct[0]
    if not values:
        return None, problems
    counts = Counter(values.values())
    top = max(counts.values())
    majority = sorted(v for v, c in counts.items() if c == top)
    if len(counts) > 1:
        for name, value in values.items():
            if len(majority) > 1 or value != majority[0]:
                others = {n: v for n, v in values.items() if v != value}
                problems.append(f"{name} says -n {value} while {others}")
    agreed = majority[0] if len(counts) == 1 and not problems else None
    return agreed, problems


def _makefile_fast_parallel_recipe(text: str) -> str:
    """The ``fast-parallel`` rule's line, without its trailing ``#`` comment."""
    for line in text.splitlines():
        if line.startswith("fast-parallel:"):
            return line.split("#", 1)[0].rstrip()
    raise AssertionError("Makefile has no `fast-parallel:` rule")


def _claude_md_fast_parallel_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.lstrip().startswith("make fast-parallel")]
    assert len(lines) == 1, f"CLAUDE.md has {len(lines)} `make fast-parallel` Commands lines"
    return lines[0]


def _rules_stop_hook_lines(text: str) -> str:
    """The lines of ``.claude/rules/tests.md`` that describe the Stop hook's run (line 8 today)."""
    lines = [ln for ln in text.splitlines() if "Stop hook" in ln]
    assert lines, ".claude/rules/tests.md no longer describes the Stop hook's run"
    return "\n".join(lines)


def _the_five_places() -> dict[str, str]:
    return {
        "stop_fast_suite.sh pytest line": shlex.join(hook_pytest_args()),
        "Makefile fast-parallel recipe": _makefile_fast_parallel_recipe(MAKEFILE.read_text()),
        "CLAUDE.md make fast-parallel line": _claude_md_fast_parallel_line(CLAUDE_MD.read_text()),
        ".claude/rules/tests.md": _rules_stop_hook_lines(RULES.read_text()),
        "docs/09-agent-workflow.md §3": _section(WORKFLOW.read_text(), 3),
    }


def test_the_worker_count_is_stated_once_everywhere() -> None:
    """T-0143 criterion 2: the hook, the Makefile and three documents state one ``-n`` value.

    T-0031's parity row drops scheduling flags on purpose, so without this a Makefile left at
    ``-n 8`` beside a hook at ``-n 12`` stays green. Green at BASE (all five say 8) and after.
    """
    agreed, problems = worker_count_problems(_the_five_places())
    assert problems == [], (
        "the places that state the Stop hook's worker count disagree (T-0143 criterion 2):\n"
        + "\n".join(problems)
    )
    assert agreed is not None


# In-memory texts shaped like the five places at TIP. Each adversarial row edits one of them.
_AT_TIP = {
    "hook": 'tests/unit tests/contract -q -m "not gpu" -rfE -n 12 --dist loadfile',
    "Makefile": 'fast-parallel: ; uv run pytest tests/unit tests/contract -q -m "not gpu" '
    "-n 12 --dist loadfile",
    "CLAUDE.md": "make fast-parallel   # the same selection under the Stop hook: -n 12 "
    "--dist loadfile, budget 60 s",
    "rules": "the Stop hook runs the identical selection with `-n 12 --dist loadfile` under a "
    "60 s budget (T-0031)",
    "docs/09": "which runs the fast suite (`uv run pytest tests/unit tests/contract -q -m "
    '"not gpu"`, with `-n 12 --dist loadfile` so the gate answers in ≤ 60 s',
}


def test_worker_helper_accepts_five_places_that_agree() -> None:
    """Control for the rows below: the unedited in-memory texts agree on 12."""
    assert worker_count_problems(_AT_TIP) == (12, [])


def test_worker_helper_names_a_makefile_left_behind() -> None:
    """Reviewer mutation 1 in memory: the Makefile at ``-n 8`` beside a hook at ``-n 12``."""
    places = {**_AT_TIP, "Makefile": _AT_TIP["Makefile"].replace("-n 12", "-n 8")}
    agreed, problems = worker_count_problems(places)
    assert agreed is None
    assert len(problems) == 1 and problems[0].startswith("Makefile says -n 8"), problems


def test_worker_helper_names_a_document_left_behind_when_the_split_is_even() -> None:
    """Reviewer mutation 4 in memory, two places only: neither side is the 'majority'."""
    places = {"CLAUDE.md": _AT_TIP["CLAUDE.md"], "docs/09": _AT_TIP["docs/09"].replace("12", "8")}
    agreed, problems = worker_count_problems(places)
    assert agreed is None
    assert sorted(p.split(" ")[0] for p in problems) == ["CLAUDE.md", "docs/09"], problems


def test_worker_helper_reports_a_dropped_value_as_absent() -> None:
    """A document that stops stating the value must fail as absent, not pass vacuously."""
    places = {
        **_AT_TIP,
        "rules": "the Stop hook runs the identical selection in parallel under a 60 s budget",
    }
    agreed, problems = worker_count_problems(places)
    assert agreed is None
    assert problems == ["rules: the worker count is absent"], problems


def test_worker_helper_refuses_auto_and_two_values_in_one_place() -> None:
    places = {**_AT_TIP, "hook": _AT_TIP["hook"].replace("-n 12", "-n auto")}
    assert worker_count_problems(places)[1] == [
        "hook: the worker count is not an integer (['auto'])"
    ]
    places = {**_AT_TIP, "docs/09": _AT_TIP["docs/09"] + "; formerly `-n 8`"}
    assert worker_count_problems(places)[1] == ["docs/09: states several worker counts [8, 12]"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("pytest -q --numprocesses=12 --dist loadfile", ["12"]),
        ("pytest -q -n12 --dist loadfile", ["12"]),
        ("pytest -q -n 12 --dist loadfile", ["12"]),
        ("pytest -q --numprocesses 12", ["12"]),
        ("with `-n 12 --dist loadfile` so", ["12"]),
        # Not worker counts: a flag that merely ends in -n, and a word that contains it.
        ("grep --dist-n 3 and re-n 4", []),
    ],
)
def test_worker_count_parser_reads_every_spelling(text: str, expected: list[str]) -> None:
    assert worker_counts_in(text) == expected


def test_the_gate_runs_on_twelve_workers() -> None:
    """T-0143 criteria 1 and 2: the agreed value is 12, still distributed by file."""
    agreed, problems = worker_count_problems(_the_five_places())
    assert problems == [], problems
    assert agreed == STOP_HOOK_WORKERS, (
        f"the five places agree on -n {agreed}, T-0143 sets the gate to -n {STOP_HOOK_WORKERS}"
    )
    hook = hook_pytest_args()
    recipe = shlex.split(_makefile_fast_parallel_recipe(MAKEFILE.read_text()))
    for name, argv in (("hook", hook), ("Makefile fast-parallel", recipe)):
        pairs = [(argv[i], argv[i + 1]) for i in range(len(argv) - 1)]
        assert ("--dist", "loadfile") in pairs, f"{name} lost `--dist loadfile`: {argv}"


# BASE (9b75c78) as literals, copied by hand from the files at that commit. Not `git show main:`:
# a moving ref goes vacuous after merge (.claude/rules/tests.md, T-0089).
BASE_HOOK_PYTEST_LINE = (
    'if ! out=$(timeout 110 uv run pytest tests/unit tests/contract -q -m "not gpu" -rfE '
    "-n 8 --dist loadfile 2>&1); then"
)
BASE_HOOK_ARGV = [
    "tests/unit",
    "tests/contract",
    "-q",
    "-m",
    "not gpu",
    "-rfE",
    "-n",
    "8",
    "--dist",
    "loadfile",
]
BASE_MAKEFILE_RECIPE = (
    'fast-parallel: ; uv run pytest tests/unit tests/contract -q -m "not gpu" -n 8 --dist loadfile'
)


def mask_worker_count(argv: list[str]) -> list[str]:
    """``argv`` with the value after each separate ``-n`` token replaced by ``<N>``.

    Only the spelling BASE uses is masked: criterion 1 keeps every other token byte-identical, so
    a respelling (``-n12``, ``--numprocesses=12``) is a difference, not a mask.
    """
    out = list(argv)
    for i, tok in enumerate(out[:-1]):
        if tok == "-n":
            out[i + 1] = "<N>"
    return out


def _mask_line(line: str) -> str:
    return re.sub(r"(?<=\s-n )\S+", "<N>", line.strip())


def test_the_parallel_command_differs_from_base_only_by_the_worker_count() -> None:
    """T-0143 criteria 1 and 7: BASE's command with the ``-n`` value masked, token for token.

    Green before and after. The bound on the value itself (4..12, an integer) is
    ``test_stop_hook.py``'s row; this one pins that nothing *else* moved: no ``-p`` plugin, no
    ``--ignore``, no ``--dist`` change, ``timeout 110`` and ``2>&1`` kept.
    """
    assert mask_worker_count(hook_pytest_args()) == mask_worker_count(BASE_HOOK_ARGV), (
        f"the hook's pytest argv moved by more than the worker count: {hook_pytest_args()} "
        f"against BASE {BASE_HOOK_ARGV} (T-0143 criterion 1)"
    )
    line = _hook_pytest_line()
    assert re.search(r"\btimeout 110 uv run pytest\b", line), (
        f"the hook's timeout is no longer 110: {line.strip()!r} (T-0143 criterion 7)"
    )
    assert _mask_line(line) == _mask_line(BASE_HOOK_PYTEST_LINE), (
        f"the hook's pytest line moved by more than the worker count:\n  {line.strip()!r}\n"
        f"  BASE {BASE_HOOK_PYTEST_LINE!r} (T-0143 criterion 1)"
    )
    recipe = _makefile_fast_parallel_recipe(MAKEFILE.read_text())
    assert _mask_line(recipe) == _mask_line(BASE_MAKEFILE_RECIPE), (
        f"Makefile's fast-parallel recipe moved by more than the worker count:\n  {recipe!r}\n"
        f"  BASE {BASE_MAKEFILE_RECIPE!r} (T-0143 criterion 1)"
    )


@pytest.mark.parametrize(
    "mutant",
    [
        pytest.param([*BASE_HOOK_ARGV, "-p", "no:randomly"], id="added -p plugin"),
        pytest.param([*BASE_HOOK_ARGV, "--ignore=tests/unit/test_audio_clock.py"], id="--ignore"),
        pytest.param(
            [*BASE_HOOK_ARGV[:-1], "worksteal"], id="--dist worksteal (reviewer mutation 3)"
        ),
        pytest.param([*BASE_HOOK_ARGV[:6], "-n12", *BASE_HOOK_ARGV[8:]], id="respelled -n12"),
        pytest.param([a for a in BASE_HOOK_ARGV if a != "-rfE"], id="dropped -rfE"),
    ],
)
def test_the_base_argv_comparison_catches_anything_but_the_count(mutant: list[str]) -> None:
    """Adversarial rows for the comparison above, on in-memory argvs (none touches the hook)."""
    assert mask_worker_count(mutant) != mask_worker_count(BASE_HOOK_ARGV)
    assert _mask_line(shlex.join(mutant)) != _mask_line(shlex.join(BASE_HOOK_ARGV))


def test_the_base_argv_comparison_accepts_the_count_alone() -> None:
    """Control: BASE's argv at ``-n 12`` is equal under the mask, as is the line."""
    at_twelve = [*BASE_HOOK_ARGV[:7], "12", *BASE_HOOK_ARGV[8:]]
    assert at_twelve != BASE_HOOK_ARGV
    assert mask_worker_count(at_twelve) == mask_worker_count(BASE_HOOK_ARGV)
    assert _mask_line(BASE_HOOK_PYTEST_LINE.replace("-n 8", "-n 12")) == _mask_line(
        BASE_HOOK_PYTEST_LINE
    )
