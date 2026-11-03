"""Acceptance tests for T-0147: ``strong_claims`` reads ``## Owned files`` with the grants rule.

At BASE (bd5f661) ``scripts/check_pending.py::strong_claims`` takes every ``tests/**.py`` path
anywhere under ``## Owned files`` (and ``## Tests to write``) as a structured claim, including the
paths of a "does not own" list. Through ``module_owner`` such a claim outranks the done task that
owns the module, so the Stop hook reports a done task's red module as ``expected:`` (the T-0146
shape: a failing turn-machine invariant passed the gate as ``expected: T-0146``).

After T-0147 the Owned-files part of the claims comes from ``check_pending.owned_paths(text)``,
which must print exactly what ``bash .claude/hooks/_lib.sh owned-globs <file>`` prints (first token
of a bullet only, the section ends at any heading or a not-owned marker line, fences skipped), and
``## Tests to write`` keeps its whole-section scan but stops at the same marker.

Every crafted text below says in its docstring whether it *discriminates* (the BASE parser and the
T-0147 parser give different results; the row was ``pending`` until activation) or is a
*too-narrow guard* (both give the same result, and an over-fix such as M5/M6/M7 would break it;
the row was never marked, because a strict XPASS would have failed the suite).

Expected values are written by hand. The only oracle that is not a literal is the hook parser
itself (``owned-globs``), which is the thing criterion 2 says ``owned_paths`` must agree with, and
``test_corpus_hand_expectations_agree_with_the_hook_parser`` checks that oracle against the hand
values so a corpus row cannot silently mean something other than its docstring says.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]
# Same override as tests/unit/test_check_pending.py: point the rows at another checkout's script.
SCRIPT = Path(os.environ.get("ASTRA_CHECK_PENDING") or REPO / "scripts" / "check_pending.py")
LIB = REPO / ".claude" / "hooks" / "_lib.sh"
CLASSIFIER = REPO / ".claude" / "hooks" / "classify_failures.py"
TASKS_DIR = REPO / "docs" / "tasks"

# Criterion 2's "a concrete tests/.../*.py path", written from the criterion, not copied from the
# script: no glob characters, ends in .py, under tests/.
CONCRETE_TEST_PATH = re.compile(r"tests/[A-Za-z0-9_./-]+\.py")

GIT_FREE_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_pending_under_test_t0147", SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load {SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cp = _load_script()


def owned_paths(text: str) -> list[str]:
    """``check_pending.owned_paths``, looked up at call time so collection never errors at BASE."""
    fn: Callable[[str], list[str]] | None = getattr(cp, "owned_paths", None)
    assert fn is not None, (
        f"{SCRIPT} has no owned_paths(task_text) -> list[str] (T-0147 criterion 2)"
    )
    return list(fn(text))


def strong(text: str) -> set[str]:
    return set(cp.strong_claims(text))


def owned_globs(task_file: Path) -> subprocess.CompletedProcess[str]:
    """The grants parser, run the way the protect-paths hook module runs it (stdin closed)."""
    return subprocess.run(
        ["bash", str(LIB), "owned-globs", str(task_file)],
        input="",
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env=GIT_FREE_ENV,
        timeout=30,
        check=False,
    )


def hook_owned_list(task_file: Path) -> list[str]:
    """What ``owned-globs`` prints, with exit 3 + "no '## Owned files' section" read as ``[]``."""
    res = owned_globs(task_file)
    if res.returncode == 3 and "no '## Owned files' section" in res.stderr:
        return []
    assert res.returncode == 0, (
        f"owned-globs failed on {task_file} (rc {res.returncode}): {res.stderr.strip()}"
    )
    return [line for line in res.stdout.splitlines() if line]


# ---------------------------------------------------------------------------------------------
# Builders. Every task text is assembled inline from these; there are no shared fixtures.
# ---------------------------------------------------------------------------------------------


def status_line(status: str) -> str:
    return f"**Status:** {status} · **Blast radius:** low · **Depends on:** none\n"


def task_text(
    status: str = "in progress 2026-10-23",
    *,
    goal: str = "A sandbox task.",
    owned: str | None = "- `local_astra/sandbox/x.py`",
    owned_heading: str = "## Owned files",
    tests: str | None = None,
    tail: str = "",
) -> str:
    parts = [f"# T-0999 — sandbox\n\n{status_line(status)}\n## Goal\n\n{goal}\n"]
    if owned is not None:
        parts.append(f"\n{owned_heading}\n\n{owned}\n")
    if tests is not None:
        parts.append(f"\n## Tests to write\n\n{tests}\n")
    parts.append(f"\n## Implementation notes\n\nNothing.\n{tail}")
    return "".join(parts)


# The four MARKER phrases of `_lib.sh` owned_globs(), an upper-case variant, and T-0023's wording.
MARKER_LINES = {
    "does-not-own": "This task does not own the following (not owned):",
    "deliberately-does-not-touch": "It deliberately does not touch these:",
    "ownership-conflicts": "Ownership conflicts (resolved by the orchestrator):",
    "claims-no": "This task claims no other files; for reference:",
    "upper-case": "THIS TASK DOES NOT OWN THE FOLLOWING:",
    "t0023-wording": "**Files this task needs and does not own:**",
}

MINE = "tests/unit/test_mine.py"
FOREIGN = "tests/unit/test_foreign.py"


def marker_owned(marker_line: str) -> str:
    return f"- `{MINE}`\n- `local_astra/sandbox/x.py`\n\n{marker_line}\n- `{FOREIGN}`"


# T-0146's ``## Owned files`` as of 41093ec^ (verbatim), the pre-workaround shape of the bug.
T0146_OWNED = """\
Main session (implementation):
- `tests/conftest.py`
- `.claude/rules/tests.md`

Test-author:
- `tests/unit/test_t0146_gate_profile.py`

Planner and orchestrator:
- `docs/tasks/T-0146.md`
- `PROGRESS.md`

This task does not own the following (not owned); stop and report if any needs a change:
- `.claude/hooks/stop_fast_suite.sh`
- `.claude/hooks/classify_failures.py`
- `Makefile`
- `CLAUDE.md`
- `docs/09-agent-workflow.md`
- `tests/unit/test_turn_invariants.py`
- `tests/unit/test_suite_budget.py`
- `tests/unit/test_stop_hook.py`
- `local_astra/**`
- `pyproject.toml`"""

T0146_FOREIGN = (
    "tests/unit/test_turn_invariants.py",
    "tests/unit/test_suite_budget.py",
    "tests/unit/test_stop_hook.py",
)

FIRST_TOKEN_OWNED = """\
- `local_astra/sandbox/x.py` *(new; mirrors tests/unit/test_annotation.py)*
- `tests/unit/test_first_of_two.py`, `tests/unit/test_second_of_two.py`
- `./tests/unit/test_dot_slash.py`

The test-author writes tests/unit/test_paragraph.py as a new module (the T-0060 shape)."""

FENCED_OWNED = """\
- `local_astra/sandbox/x.py`

```text
- `tests/unit/test_fenced.py`
This task does not own anything below this line.
```

- `tests/unit/test_after_fence.py`"""

TOO_WIDE_GOAL = "The module tests/unit/test_turn_invariants.py gains one row."
TOO_WIDE_OWNED = """\
- `tests/unit/test_turn_invariants.py`
- `local_astra/sandbox/x.py`

This task does not own the following (not owned):
- `tests/unit/test_turn_invariants.py` (existing rows stay unedited)
- `tests/unit/test_stop_hook.py`"""

# The task's two examples plus a trailing-star glob. `tests/unit/test_x_*.py` (star mid-token) is
# not even a grant: the hook's SHAPE regex allows a star only at the end (see the report).
GLOB_OWNED = """\
- `tests/**`
- `tests/unit/test_x_*.py`
- `tests/unit/test_x_*`"""

TESTS_TO_WRITE_PROSE = """\
`tests/integration/test_para_open.py` (new module; the T-0023 shape) covers the acoustic gate.

- `tests/unit/test_multi_a.py` and its wire twin `tests/unit/test_multi_b.py` (T-0043 shape).

**`tests/unit/test_bold_head.py`** (pending T-0999; the T-0108 shape):
- `tests/unit/test_node.py::test_name` must go green."""

# The first marker line carries a path itself (reviewer A, X1: keeping the marker line before the
# cut brings the false excuse back for the most natural way to write a reference).
TESTS_TO_WRITE_MARKER = """\
- `tests/unit/test_mine.py`: the new rows.
- `tests/unit/test_on_marker_line.py` is referenced; this task does not own it.

Referenced, unchanged; this task does not own these:
- `tests/unit/test_reference.py` must stay green untouched."""


# ---------------------------------------------------------------------------------------------
# Unit rows over strong_claims / owned_paths.
# ---------------------------------------------------------------------------------------------


def test_not_owned_list_under_owned_files_is_not_a_claim() -> None:
    """Criterion 1 at function level. Discriminates: BASE claims all five test paths.

    T-0146's own Owned files (41093ec^), sub-labels and all, with the not-owned bullets directly
    under the marker and no heading in between.
    """
    text = task_text(owned=T0146_OWNED)
    claims = strong(text)
    assert "tests/unit/test_t0146_gate_profile.py" in claims, claims
    for foreign in T0146_FOREIGN:
        assert foreign not in claims, (
            f"{foreign} is listed after 'This task does not own' and must not be a structured "
            f"claim: {sorted(claims)}"
        )
    assert claims == {"tests/conftest.py", "tests/unit/test_t0146_gate_profile.py"}, claims
    assert owned_paths(text) == [
        "tests/conftest.py",
        ".claude/rules/tests.md",
        "tests/unit/test_t0146_gate_profile.py",
        "docs/tasks/T-0146.md",
        "PROGRESS.md",
    ]


@pytest.mark.parametrize("marker", list(MARKER_LINES.values()), ids=list(MARKER_LINES))
def test_each_marker_phrase_ends_the_owned_section(marker: str) -> None:
    """Criterion 3 (M1, M3 via the upper-case row). Discriminates: BASE claims FOREIGN too."""
    text = task_text(owned=marker_owned(marker))
    claims = strong(text)
    assert MINE in claims, f"the path before the marker line must stay claimed: {claims}"
    assert FOREIGN not in claims, f"the path after {marker!r} must not be claimed: {claims}"
    assert owned_paths(text) == [MINE, "local_astra/sandbox/x.py"]


@pytest.mark.parametrize(
    "heading",
    [
        pytest.param("## Not owned", id="##"),
        pytest.param("### Not owned", id="###"),
        pytest.param("# Not owned", id="#"),
    ],
)
def test_any_heading_level_ends_the_owned_section(heading: str) -> None:
    """Criterion 3, the heading half (M4 is caught by the ``###`` row once the parser is new).

    ``##`` and ``###`` are too-narrow guards: BASE's section split already ends there, and the
    new parser must keep it. ``#`` discriminates: BASE's split needs two hashes, so FOREIGN after
    a level-1 heading is claimed at BASE; the grants parser ends at any level (``#{1,6}``).
    """
    text = task_text(owned=f"- `{MINE}`\n\n{heading}\n\n- `{FOREIGN}`")
    claims = strong(text)
    assert MINE in claims, claims
    assert FOREIGN not in claims, f"{FOREIGN} sits after {heading!r}: {claims}"


def test_only_the_first_token_of_a_bullet_is_a_claim() -> None:
    """Criterion 4 (M2). Discriminates: BASE claims the annotation, the second path and the prose.

    ``./tests/...`` as a first token is stripped to ``tests/...`` and claimed (the same at BASE,
    where the regex matches inside it; it pins the strip in the new parser).
    """
    text = task_text(owned=FIRST_TOKEN_OWNED)
    claims = strong(text)
    assert "tests/unit/test_annotation.py" not in claims, (
        f"a path in a bullet's annotation is not a claim: {claims}"
    )
    assert "tests/unit/test_second_of_two.py" not in claims, (
        f"the second path of a bullet is not a claim: {claims}"
    )
    assert "tests/unit/test_paragraph.py" not in claims, (
        f"a path in a non-bullet paragraph under ## Owned files is not a claim: {claims}"
    )
    assert "tests/unit/test_dot_slash.py" in claims, (
        f"a ./tests/... first token is a claim once ./ is stripped: {claims}"
    )
    assert "./tests/unit/test_dot_slash.py" not in claims, claims


def test_a_fenced_block_neither_claims_nor_ends_the_section() -> None:
    """Criterion 4, both halves (M9).

    Discriminates on the first half (BASE claims the fenced path). The second half (the path after
    the fence, behind a fenced marker, stays claimed) is a too-narrow guard against a parser that
    lets a fenced marker end the section.
    """
    text = task_text(owned=FENCED_OWNED)
    claims = strong(text)
    assert "tests/unit/test_fenced.py" not in claims, f"a fenced path is not a claim: {claims}"
    assert "tests/unit/test_after_fence.py" in claims, (
        f"a marker inside a fenced block must not end the Owned files section: {claims}"
    )
    assert owned_paths(text) == ["local_astra/sandbox/x.py", "tests/unit/test_after_fence.py"]


def test_too_wide_owned_bullet_survives_a_later_not_owned_mention() -> None:
    """Criterion 5 at function level (M7: subtracting later mentions instead of truncating).

    Too-narrow guard: BASE claims the module and the new parser must too. ``test_stop_hook.py``
    after the marker is not asserted here (it discriminates, and this row stays unmarked).
    """
    text = task_text(goal=TOO_WIDE_GOAL, owned=TOO_WIDE_OWNED)
    assert "tests/unit/test_turn_invariants.py" in strong(text), (
        "a module that is an Owned-files bullet before the marker stays a structured claim even "
        "though the not-owned list and the Goal name it again"
    )


def test_globs_claim_no_module() -> None:
    """Criterion 6 (M8: claiming through fnmatch). Too-narrow guard on the other side: BASE
    claims no module from a glob, and neither may the new parser.

    Checked through ``module_owner``, the interface the Stop hook uses: the in-progress task names
    ``tests/unit/test_x_a.py`` in its Goal (so it is a candidate owner), owns two globs that cover
    it, and loses to the done task because a glob is not a structured claim.
    """
    module = "tests/unit/test_x_a.py"
    in_progress = task_text(goal=f"Touches {module} indirectly.", owned=GLOB_OWNED)
    done = task_text("merged 2026-09-01", goal=f"Created {module}.")
    claims = strong(in_progress)
    assert module not in claims, f"a glob grant is not a claim on {module}: {claims}"
    assert not any("*" in c for c in claims), f"a glob leaked into the claims: {claims}"
    statuses = {"T-0001": cp.task_claims(done), "T-0999": cp.task_claims(in_progress)}
    assert cp.module_owner(module, statuses) == "T-0001"


def test_tests_to_write_prose_claims_survive() -> None:
    """Criterion 7, the too-narrow guard (M5: the first-token rule applied to Tests to write).

    BASE claims all four and the new parser must too: a paragraph-opening path, the second path
    of a bullet, a bold heading line, and ``path::test_name``.
    """
    text = task_text(tests=TESTS_TO_WRITE_PROSE)
    claims = strong(text)
    for module in (
        "tests/integration/test_para_open.py",
        "tests/unit/test_multi_b.py",
        "tests/unit/test_bold_head.py",
        "tests/unit/test_node.py",
    ):
        assert module in claims, f"{module} is a ## Tests to write claim (ruling A10): {claims}"


def test_marker_ends_tests_to_write_claims() -> None:
    """Criterion 7, second half (M12). Discriminates: BASE claims the reference after the marker."""
    text = task_text(tests=TESTS_TO_WRITE_MARKER)
    claims = strong(text)
    assert MINE in claims, claims
    assert "tests/unit/test_reference.py" not in claims, (
        f"a path after a not-owned marker in ## Tests to write is not a claim: {claims}"
    )
    assert "tests/unit/test_on_marker_line.py" not in claims, (
        "a path on the not-owned marker line itself is not a claim (the marker line is cut, "
        f"not kept; reviewer A mutant X1): {claims}"
    )


def test_claimed_modules_still_sees_the_not_owned_paths() -> None:
    """Criterion 8a (negative, M11). Too-narrow guard: the whole-file mention scan (A1) is
    unchanged, so the not-owned paths are still mentions."""
    mods = set(cp.claimed_modules(task_text(owned=T0146_OWNED)))
    assert set(T0146_FOREIGN) | {"tests/unit/test_t0146_gate_profile.py"} <= mods, mods


# ---------------------------------------------------------------------------------------------
# Contract rows: parity with the hook's grants parser.
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """A crafted task file and what ``owned-globs`` prints for it, worked out by hand."""

    text: str
    owned: tuple[str, ...]


CORPUS: dict[str, Case] = {
    "t0146-shape": Case(
        task_text(owned=T0146_OWNED),
        (
            "tests/conftest.py",
            ".claude/rules/tests.md",
            "tests/unit/test_t0146_gate_profile.py",
            "docs/tasks/T-0146.md",
            "PROGRESS.md",
        ),
    ),
    **{
        f"marker-{key}": Case(
            task_text(owned=marker_owned(line)), (MINE, "local_astra/sandbox/x.py")
        )
        for key, line in MARKER_LINES.items()
    },
    **{
        f"heading-{n}": Case(
            task_text(owned=f"- `{MINE}`\n\n{'#' * n} Not owned\n\n- `{FOREIGN}`"), (MINE,)
        )
        for n in (1, 2, 3)
    },
    # The first token of "`a`, `b`" keeps its closing backtick under strip("`").rstrip(",;:"),
    # so the hook parser drops the whole bullet (see the report: criterion 4 only says the second
    # path is not a claim).
    "first-token": Case(
        task_text(owned=FIRST_TOKEN_OWNED),
        ("local_astra/sandbox/x.py", "tests/unit/test_dot_slash.py"),
    ),
    "fenced": Case(
        task_text(owned=FENCED_OWNED),
        ("local_astra/sandbox/x.py", "tests/unit/test_after_fence.py"),
    ),
    "too-wide": Case(
        task_text(goal=TOO_WIDE_GOAL, owned=TOO_WIDE_OWNED),
        ("tests/unit/test_turn_invariants.py", "local_astra/sandbox/x.py"),
    ),
    "globs": Case(task_text(owned=GLOB_OWNED), ("tests/**", "tests/unit/test_x_*")),
    "tests-to-write-prose": Case(
        task_text(tests=TESTS_TO_WRITE_PROSE), ("local_astra/sandbox/x.py",)
    ),
    "tests-to-write-marker": Case(
        task_text(tests=TESTS_TO_WRITE_MARKER), ("local_astra/sandbox/x.py",)
    ),
    # T-0029 shape: only the first ## Owned files section counts (M10). BASE claims both.
    "two-owned-sections": Case(
        task_text(
            owned="- `tests/unit/test_first_section.py`",
            tail="\n## Owned files\n\n- `tests/unit/test_second_section.py`\n",
        ),
        ("tests/unit/test_first_section.py",),
    ),
    # The hook parser finds no section (exit 3): no claims. BASE claims through rstrip(":").
    "owned-heading-colon": Case(
        task_text(owned="- `tests/unit/test_colon.py`", owned_heading="## Owned files:"), ()
    ),
    # No section either way: pins parity.
    "owned-heading-draft": Case(
        task_text(owned="- `tests/unit/test_draft.py`", owned_heading="## Owned files (draft)"), ()
    ),
    # The hook parser accepts these two; BASE's `^##+ +` split does not (it claims nothing).
    "owned-heading-no-space": Case(
        task_text(owned="- `tests/unit/test_nospace.py`", owned_heading="##Owned files"),
        ("tests/unit/test_nospace.py",),
    ),
    "owned-heading-indented": Case(
        task_text(owned="- `tests/unit/test_indented.py`", owned_heading="  ## Owned files"),
        ("tests/unit/test_indented.py",),
    ),
    # A bullet whose only token is Makefile is a grant, not a test claim.
    "makefile-only-token": Case(
        task_text(owned="- Makefile\n- `tests/unit/test_beside_makefile.py`"),
        ("Makefile", "tests/unit/test_beside_makefile.py"),
    ),
    # A ".." segment is never a grant. BASE claims the path.
    "dotdot-segment": Case(
        task_text(owned="- `local_astra/sandbox/x.py`\n- `tests/unit/../unit/test_dots.py`"),
        ("local_astra/sandbox/x.py",),
    ),
    # Bold markup fails the SHAPE regex. BASE claims the path inside it.
    "bold-token": Case(
        task_text(owned="- `local_astra/sandbox/x.py`\n- **tests/unit/test_bold.py**"),
        ("local_astra/sandbox/x.py",),
    ),
    "no-owned-section": Case(task_text(owned=None, tests=f"- `{MINE}`"), ()),
    # A bullet that itself carries a marker phrase ends the section before its own token: the
    # hook breaks on the marker line before reading the bullet (reviewer A, X3). BASE claims all
    # three paths.
    "marker-on-bullet": Case(
        task_text(
            owned=(
                f"- `{MINE}`\n"
                "- `tests/unit/test_ref.py` (this task does not own it)\n"
                "- `tests/unit/test_after_ref.py`"
            )
        ),
        (MINE,),
    ),
}


def _write_corpus(tmp_path: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for name, case in CORPUS.items():
        path = tmp_path / f"{name}.md"
        path.write_text(case.text, encoding="utf-8")
        out[name] = path
    return out


def test_corpus_hand_expectations_agree_with_the_hook_parser(tmp_path: Path) -> None:
    """Oracle check, not a criterion row: the hand-written ``owned`` of every corpus case is what
    ``owned-globs`` prints, so the parity rows below compare against what the docstrings claim.
    True at BASE (it never touches ``check_pending.py``)."""
    wrong = {
        name: (list(CORPUS[name].owned), hook_owned_list(path))
        for name, path in _write_corpus(tmp_path).items()
        if hook_owned_list(path) != list(CORPUS[name].owned)
    }
    assert not wrong, f"hand expectation vs owned-globs: {wrong}"


def test_owned_paths_matches_owned_globs_on_every_task_file() -> None:
    """Criterion 2 over every real ``docs/tasks/T-*.md`` (T-0129 and T-0130 are live exit-3 cases)."""
    files = sorted(TASKS_DIR.glob("T-*.md"))
    assert len(files) > 100, f"expected the real task corpus under {TASKS_DIR}, got {len(files)}"
    mismatches: dict[str, dict[str, list[str]]] = {}
    for path in files:
        hook = hook_owned_list(path)
        mine = owned_paths(path.read_text(encoding="utf-8"))
        if mine != hook:
            mismatches[path.name] = {
                "only_owned_paths": sorted(set(mine) - set(hook)),
                "only_owned_globs": sorted(set(hook) - set(mine)),
                "order_or_dupes": [] if set(mine) ^ set(hook) else mine,
            }
    assert not mismatches, f"owned_paths disagrees with owned-globs: {mismatches}"


def test_owned_paths_matches_owned_globs_on_the_adversarial_corpus(tmp_path: Path) -> None:
    """Criterion 2 over every crafted text (M10 via the two-section and colon rows)."""
    mismatches = {}
    for name, path in _write_corpus(tmp_path).items():
        hook = hook_owned_list(path)
        mine = owned_paths(CORPUS[name].text)
        if mine != hook:
            mismatches[name] = {"owned_paths": mine, "owned_globs": hook}
    assert not mismatches, f"owned_paths disagrees with owned-globs: {mismatches}"


@pytest.mark.parametrize(
    "name", [n for n, c in CORPUS.items() if "## Tests to write" not in c.text]
)
def test_strong_owned_part_is_the_test_paths_of_owned_paths(name: str) -> None:
    """Criterion 2's identity, for corpus files with no ``## Tests to write`` section: the
    structured claims are exactly the concrete test paths among the Owned-files grants (the hand
    list, which the oracle row above ties to owned-globs)."""
    case = CORPUS[name]
    expected = {p for p in case.owned if CONCRETE_TEST_PATH.fullmatch(p)}
    assert strong(case.text) == expected
    assert {p for p in owned_paths(case.text) if CONCRETE_TEST_PATH.fullmatch(p)} == expected


# ---------------------------------------------------------------------------------------------
# Integration, CPU: the Stop hook's classifier end to end on a sandbox top.
# ---------------------------------------------------------------------------------------------

INVARIANTS = "tests/unit/test_turn_invariants.py"

T0003_DONE = (
    "# T-0003 — turn machine\n\n"
    + status_line("merged 2026-09-13 (3a553ea)")
    + "\n## Goal\n\nThe invariants live in `tests/unit/test_turn_invariants.py`.\n"
    + "It also created `tests/unit/test_mine.py`.\n"
)


def _sandbox(tmp_path: Path, t0146: str) -> Path:
    top = tmp_path / "top"
    (top / "scripts").mkdir(parents=True)
    (top / "docs" / "tasks").mkdir(parents=True)
    (top / "scripts" / "check_pending.py").write_text(SCRIPT.read_text())
    (top / "docs" / "tasks" / "T-0003.md").write_text(T0003_DONE)
    (top / "docs" / "tasks" / "T-0146.md").write_text(t0146)
    return top


def _classify(top: Path, *modules: str) -> subprocess.CompletedProcess[str]:
    stdin = "".join(f"FAILED {m}::test_x - boom\n" for m in modules)
    return subprocess.run(
        [sys.executable, str(CLASSIFIER), str(top)],
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env=GIT_FREE_ENV,
    )


def _lines(res: subprocess.CompletedProcess[str]) -> list[str]:
    return res.stdout.splitlines()


def _diag(res: subprocess.CompletedProcess[str]) -> str:
    return f"rc={res.returncode}\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"


def test_t0146_shape_stays_a_regression(tmp_path: Path) -> None:
    """Criterion 1, adversarial: the bug itself. At BASE this prints ``expected: T-0146`` and
    exits 0 (reproduced with docs/tasks from 41093ec^)."""
    top = _sandbox(tmp_path, task_text(owned=T0146_OWNED))
    res = _classify(top, INVARIANTS)
    regressions = [ln for ln in _lines(res) if ln.startswith("regression:")]
    assert any(INVARIANTS in ln and "T-0003" in ln for ln in regressions), _diag(res)
    assert not any(ln.startswith("expected: T-0146") for ln in _lines(res)), _diag(res)
    assert res.returncode == 1, _diag(res)


def test_structured_owned_bullet_still_takes_over_a_done_module(tmp_path: Path) -> None:
    """Over-fix guard (A10, M6): an Owned-files bullet before the marker still takes a done
    task's module over. Too-narrow guard: true at BASE, must stay true."""
    owned = (
        f"- `{INVARIANTS}`\n- `local_astra/sandbox/x.py`\n\n"
        "This task does not own the following (not owned):\n- `tests/unit/test_stop_hook.py`"
    )
    top = _sandbox(tmp_path, task_text(owned=owned))
    res = _classify(top, INVARIANTS)
    assert f"expected: T-0146 (1 failing in {INVARIANTS}, implementer pending)" in _lines(res), (
        _diag(res)
    )
    assert res.returncode == 0, _diag(res)


def test_too_wide_row_end_to_end(tmp_path: Path) -> None:
    """Criterion 5 through the classifier (M7). Too-narrow guard: true at BASE, must stay true."""
    top = _sandbox(tmp_path, task_text(goal=TOO_WIDE_GOAL, owned=TOO_WIDE_OWNED))
    res = _classify(top, INVARIANTS)
    assert f"expected: T-0146 (1 failing in {INVARIANTS}, implementer pending)" in _lines(res), (
        _diag(res)
    )
    assert res.returncode == 0, _diag(res)


def test_tests_to_write_claim_after_marker_is_a_regression(tmp_path: Path) -> None:
    """Criterion 7 through the classifier (M12). Discriminates: BASE reports both as expected.

    Both modules are also named by the done T-0003, so the claim before the marker is what keeps
    ``test_mine.py`` expected (a parser that dropped the whole section would turn it red too).
    """
    tests = (
        f"- `{MINE}`: the new rows.\n\n"
        "Referenced, unchanged; this task does not own these:\n"
        f"- `{INVARIANTS}` must stay green untouched."
    )
    top = _sandbox(tmp_path, task_text(tests=tests))
    res = _classify(top, MINE, INVARIANTS)
    lines = _lines(res)
    assert f"expected: T-0146 (1 failing in {MINE}, implementer pending)" in lines, _diag(res)
    assert any(
        ln.startswith("regression:") and INVARIANTS in ln and "T-0003" in ln for ln in lines
    ), _diag(res)
    assert res.returncode == 1, _diag(res)
