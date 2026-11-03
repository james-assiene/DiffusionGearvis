"""T-0057 criteria 4, 5 and 6: what the fix is gated on, and what must not move to achieve it.

The gate itself is load-independent and lives in the fast suite: **zero leaked ``astra-`` threads**,
row by row, from ``tests/conftest.py``'s guard. That is what this task is judged on. What lives
here is the *regression fence* beside it (ruling A12) and the two negatives.

None of these rows needs the measurement itself (that is ``tests/bench/test_audio_switch_budget.py``,
about a minute for two arms, run on request), and none of them may be satisfied by making the suite
*faster*, because the planner measured that there is no wall to recover:

* the counterfactual halves the suite's voluntary context switches (1 577 775 -> 798 218) and
  recovers **no wall at all** -- the matched four-module comparison gives 3.0 s of 34.05 s and the
  full-suite arms give none;
* only 54 rows of 2 393 leak a writer, 48 of them in the four audio modules, which is why the
  subset number does not scale;
* the pytest process's own CPU is 26.3 s of 211.4 s of row wall, so ~91 s of the reported CPU is
  child processes, not pytest.

A serial suite still at ~212 s after this task is therefore the **expected** result. These rows
exist so that the next reader cannot quietly turn the wall into the gate, buy the switch count by
deleting assertions from the audio modules, or -- ruling A12 -- compare this session's run against
a number somebody wrote down on a differently loaded box.
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import replace
from pathlib import Path

from tests.bench.audio_switch_budget import (
    ARM_A_SUBSET_SWITCHES,
    ARM_B_SUBSET_SWITCHES,
    AUDIO_MODULES,
    HISTORICAL_SUBSET_WALL_S,
    MEASURED_SUBSET_RATIOS,
    SESSION,
    SUBSET_MAX_RATIO,
    SUITE_MAX_RATIO,
    SUITE_SELECTION,
    Measurement,
    max_ratio_for,
    report_line,
    verdict,
)
from tests.conftest import SERIAL_CEILING_S, STOP_HOOK_BUDGET_S

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "docs" / "tasks" / "T-0057.md"
HARNESS = REPO / "tests" / "bench" / "audio_switch_budget.py"

#: Criterion 6's fence, as (``assert`` statements, ``test_`` functions) counted by an AST walk of
#: each module on d12ddca, before any implementation exists. Every one of these rows must stay
#: green **without being edited**; the guard and the stop seam are supposed to be invisible to
#: them. A premise rewrite that quietly drops an assertion and a fixture adjustment look identical
#: in a normal diff and are one command apart (.claude/rules/tests.md), so the count is pinned.
AUDIO_MODULE_ASSERTIONS = {
    "tests/unit/test_audio_clock.py": (69, 16),
    "tests/unit/test_audio_played_text.py": (14, 8),
    "tests/contract/test_audio_io_conformance.py": (16, 5),
    "tests/unit/test_null_audio.py": (40, 13),
    "tests/unit/test_audio_capture_clock.py": (87, 16),
    "tests/unit/test_audio_stt_framing.py": (22, 5),
    "tests/contract/test_session_audio_seam.py": (15, 3),
}


def _arm(
    arm: str,
    switches: int,
    wall_s: float,
    *,
    failed: int = 0,
    rows: int = 66,
    session: str = SESSION,
    selection: tuple[str, ...] = AUDIO_MODULES,
) -> Measurement:
    """One arm of a pair, with the load average a real run would have recorded."""
    return Measurement(
        arm=arm,
        switches=switches,
        wall_s=wall_s,
        rows=rows,
        failed=failed,
        selection=selection,
        load_avg=7.5,
        session=session,
    )


def _pair(
    before_switches: int,
    after_switches: int,
    *,
    before_wall_s: float = HISTORICAL_SUBSET_WALL_S,
    after_wall_s: float = HISTORICAL_SUBSET_WALL_S,
) -> tuple[Measurement, Measurement]:
    return (
        _arm("before", before_switches, before_wall_s),
        _arm("after", after_switches, after_wall_s),
    )


# ---------------------------------------------------------------------------------------------
# Criterion 4 / ruling A12 -- the fence is a ratio against a same-session before.
# ---------------------------------------------------------------------------------------------


def test_a_suite_that_did_not_get_faster_still_passes_the_fence() -> None:
    """The case the planner measured: switches down by a factor of six, wall slightly *worse*.

    Arm B of the four-module counterfactual: 152 009 switches against 986 449, and a wall that
    moved 34.05 -> 31.06 s in the subset and *up* over the whole suite. If the wall were in the
    fence, a fix that works would be reported as a failure.
    """
    before, after = _pair(ARM_A_SUBSET_SWITCHES, ARM_B_SUBSET_SWITCHES, after_wall_s=35.0)
    ok, why = verdict(after, before, max_ratio=max_ratio_for(AUDIO_MODULES))
    assert ok, (
        "a run with 85 % fewer voluntary context switches and a second *more* wall was rejected: "
        f"{why}. T-0057 criterion 4 fences the switches and only reports the wall"
    )


def test_a_faster_suite_that_still_polls_fails_the_fence() -> None:
    """The mirror: the wall improved and the thread is still there. That is not this fix.

    A wall win with the switch count intact is somebody else's change (the subprocess-launching
    rows, the 0.2 ms rendezvous in the fakes) landing at the same time.
    """
    before, after = _pair(ARM_A_SUBSET_SWITCHES, ARM_A_SUBSET_SWITCHES, after_wall_s=12.0)
    ok, why = verdict(after, before, max_ratio=max_ratio_for(AUDIO_MODULES))
    assert not ok, (
        "a run that kept every context switch but cut the wall to a third was accepted; the "
        f"fence must read the switch count, not the clock ({why})"
    )
    assert "context switches" in why, f"the reason given is not about the fence: {why!r}"


def test_a_red_run_is_never_a_measurement() -> None:
    """Arm B of the planner's probe had 7 failures; its numbers are not comparable.

    The cheap wrong fix (``WRITER_IDLE_EXIT_S`` -> 0.05 s) buys the switch count by breaking five
    drain/underrun rows. A fence that read only the counter would call that a pass. Both arms are
    checked: a before arm that could not run is as useless as a red after.
    """
    before, after = _pair(ARM_A_SUBSET_SWITCHES, ARM_B_SUBSET_SWITCHES)
    fence = max_ratio_for(AUDIO_MODULES)

    ok, why = verdict(replace(after, failed=5), before, max_ratio=fence)
    assert not ok, f"a run with 5 failing rows was accepted as a measurement ({why})"
    ok, why = verdict(after, replace(before, failed=5), max_ratio=fence)
    assert not ok, f"a before arm with 5 failing rows was accepted as a baseline ({why})"


def test_a_before_from_another_session_is_not_a_measurement() -> None:
    """Ruling A12's whole point: the before is re-measured beside the after, or it is not a before.

    The switch count is rate x writer-alive time, and in-row alive time grows with row wall, so
    the *same* fix on the *same* tree gave 0.74, 0.62 and 0.65 at loads 6 to 13. A before from
    another run is a number from another box-load, and comparing to it is how a fence turns back
    into the absolute threshold ruling A12 removed.
    """
    before, after = _pair(ARM_A_SUBSET_SWITCHES, ARM_B_SUBSET_SWITCHES)
    stale = replace(before, session="measured-on-main-at-998aa85")

    ok, why = verdict(after, stale, max_ratio=max_ratio_for(AUDIO_MODULES))
    assert not ok, (
        "an after was compared against a before from another session and accepted: the ratio it "
        f"computed is between two differently loaded boxes ({why})"
    )
    assert "session" in why, f"the reason given does not name the stale before: {why!r}"


def test_the_fence_is_a_ratio_and_so_says_the_same_thing_at_every_scale() -> None:
    """Load-independence, which is the property an absolute number did not have.

    The same before/after *ratio* must get the same verdict whether the pair was taken on an idle
    box or under three other agents. Hand-computed: 0.90 is over the 0.85 subset fence and 0.60
    is under it, at a twentieth of the size and at twice it.
    """
    fence = max_ratio_for(AUDIO_MODULES)
    for scale in (0.02, 1.0, 2.0):
        over = _pair(int(1_000_000 * scale), int(900_000 * scale))
        under = _pair(int(1_000_000 * scale), int(600_000 * scale))
        ok_over, why_over = verdict(over[1], over[0], max_ratio=fence)
        ok_under, why_under = verdict(under[1], under[0], max_ratio=fence)
        assert not ok_over, (
            f"at scale {scale} a run that removed a tenth of the switches passed the "
            f"{fence:.2f} fence: {why_over}"
        )
        assert ok_under, (
            f"at scale {scale} a run that removed two fifths of the switches failed the "
            f"{fence:.2f} fence: {why_under}"
        )


def test_the_fence_sits_where_the_task_says_and_the_measured_fix_clears_it() -> None:
    """The boundary, hand-computed against a round before of a million.

    0.84 of a million passes and 0.86 fails, so a harness quietly loosened to 0.95 is red here as
    well as against the task file. The three ratios ruling A12 measured for the real fix (0.74,
    0.62, 0.65) all clear it -- the fence is a fence, not a target, and not a squeaker either.
    """
    fence = max_ratio_for(AUDIO_MODULES)
    million = 1_000_000

    under = _pair(million, 840_000)
    ok, why = verdict(under[1], under[0], max_ratio=fence)
    assert ok, f"0.84 x before was rejected by a {fence:.2f} fence: {why}"

    over = _pair(million, 860_000)
    ok, why = verdict(over[1], over[0], max_ratio=fence)
    assert not ok, (
        f"0.86 x before passed a fence of {fence:.2f} x. Ruling A12 puts the subset fence at "
        f"{SUBSET_MAX_RATIO:.2f}; a looser one lets a half-fixed lifecycle through"
    )

    for measured in MEASURED_SUBSET_RATIOS:
        pair = _pair(million, int(measured * million))
        ok, why = verdict(pair[1], pair[0], max_ratio=fence)
        assert ok, (
            f"the fix's own measured ratio of {measured} was rejected by the fence: {why}. A "
            "fence the fix does not clear is a target, and ruling A12 says it is not one"
        )


def test_the_suite_fence_is_looser_than_the_subset_one() -> None:
    """Two numbers, not one: 0.85 on the four modules, 0.90 suite-wide.

    The four modules are 63 % of the suite's blocking, so the same fix shows up diluted over
    ``tests/unit tests/contract``. Collapsing the two into one number is a half-move in the other
    direction -- a suite-wide fence at 0.85 would fail a correct fix, and a subset fence at 0.90
    would pass a half-fixed one.
    """
    pair = _pair(1_000_000, 880_000)
    suite = replace(pair[0], selection=SUITE_SELECTION), replace(pair[1], selection=SUITE_SELECTION)

    ok, why = verdict(suite[1], suite[0], max_ratio=max_ratio_for(SUITE_SELECTION))
    assert ok, f"0.88 x before was rejected by the suite-wide fence of {SUITE_MAX_RATIO:.2f}: {why}"
    ok, why = verdict(pair[1], pair[0], max_ratio=max_ratio_for(AUDIO_MODULES))
    assert not ok, (
        f"0.88 x before passed the subset fence of {SUBSET_MAX_RATIO:.2f}; the subset is where "
        "the leak is and its fence is the tighter of the two"
    )


def test_a_count_bought_by_running_fewer_rows_is_not_a_reduction() -> None:
    """The adversarial arithmetic: half the rows collect half the switches.

    A deselect, a collection error that eats a module, or a ``-x`` that stops early all produce a
    beautiful ratio. The arms have to have run the same rows over the same selection.
    """
    fence = max_ratio_for(AUDIO_MODULES)
    before, after = _pair(ARM_A_SUBSET_SWITCHES, ARM_B_SUBSET_SWITCHES)

    ok, why = verdict(replace(after, rows=33), before, max_ratio=fence)
    assert not ok, f"an after that ran half the rows was accepted as a reduction ({why})"
    ok, why = verdict(
        replace(after, selection=SUITE_SELECTION),
        replace(before, selection=SUITE_SELECTION[:1]),
        max_ratio=fence,
    )
    assert not ok, f"two arms over different selections were compared ({why})"


def test_the_report_line_carries_both_numbers_both_load_averages_and_the_ratio() -> None:
    """Ruling A12: the line has to carry enough to be compared to the next one.

    A load-dependent counter without its load average is the "quoting a number without its
    regime" mistake in CLAUDE.md, and here it is not a style point: the same fix ranged over
    0.62-0.74 with the load alone.
    """
    before = _arm("before", 1_000_000, 34.0)
    after = replace(_arm("after", 650_000, 33.0), load_avg=11.25)
    line = report_line(after, before, max_ratio=max_ratio_for(AUDIO_MODULES))

    for needed, what in (
        ("1,000,000", "the before switch count"),
        ("650,000", "the after switch count"),
        ("7.50", "the before load average"),
        ("11.25", "the after load average"),
        ("0.650", "the ratio (650 000 / 1 000 000, computed by hand)"),
        ("0.85", "the fence it was judged against"),
        ("66", "the row count both arms ran"),
    ):
        assert needed in line, f"the report line does not carry {what}: {line}"
    assert "not gated" in line and "wall" in line, (
        f"the report line does not say the wall is reported rather than gated: {line}"
    )


def _module_level_numbers(source: str) -> set[str]:
    """Names bound at module level to an int or a tuple of numbers -- i.e. stored measurements."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        value = getattr(node, "value", None)
        if value is None:
            continue
        literal = (isinstance(value, ast.Constant) and isinstance(value.value, int | float)) or (
            isinstance(value, ast.Tuple)
            and all(
                isinstance(e, ast.Constant) and isinstance(e.value, int | float) for e in value.elts
            )
        )
        if literal:
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def test_the_fence_reads_no_stored_number() -> None:
    """Structural, because the behavioural rows above cannot see every way back to an absolute.

    ``verdict`` must judge on its two arguments alone. Reaching for ``ARM_A_SUBSET_SWITCHES`` --
    or clamping against it, or falling back to it when the before is missing -- reinstates the
    threshold ruling A12 removed, and would do it in a way that still passes a pair whose ratio
    happens to be right. The fence's *own* two ratios are allowed: they are the policy, not a
    measurement.
    """
    source = HARNESS.read_text()
    stored = _module_level_numbers(source) - {"SUBSET_MAX_RATIO", "SUITE_MAX_RATIO"}
    assert {"ARM_A_SUBSET_SWITCHES", "HISTORICAL_SUITE_SWITCHES"} <= stored, (
        "this row can no longer see the harness's stored measurements, so it is checking "
        f"nothing; it found {sorted(stored)}"
    )

    fn = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "verdict"
    )
    read = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert read & stored == set(), (
        f"verdict() reads the stored measurement(s) {sorted(read & stored)}. Ruling A12: the "
        "fence is a ratio against a before measured in the same session, never an absolute "
        "number -- the count is load-dependent and the same fix measured 0.74, 0.62 and 0.65"
    )

    sig = inspect.signature(verdict)
    assert list(sig.parameters) == ["after", "before", "max_ratio"], (
        f"verdict{sig} no longer takes an explicit before and fence: a default for either is a "
        "place for a stale number to hide"
    )
    assert sig.parameters["before"].default is inspect.Parameter.empty, (
        "verdict()'s before has a default: the caller must hand it a measurement it just took"
    )
    assert sig.parameters["max_ratio"].default is inspect.Parameter.empty, (
        "verdict()'s max_ratio has a default: the subset and the suite have different fences "
        "(0.85 and 0.90) and picking one silently is a half-move"
    )


#: The shapes the pin accepts for a ratio in the task file: ``0.85 x before`` and ``85 % of the
#: before``, with the multiplication sign written any of the three usual ways.
_RATIO_FORMS = (
    re.compile(r"([01]\.\d+)\s*(?:×|x|\*)\s*(?:the\s+)?before", re.IGNORECASE),
    re.compile(r"(\d{2})\s*%\s*(?:×|x|\*)?\s*(?:of\s+)?(?:the\s+)?before", re.IGNORECASE),
)
#: An absolute switch threshold: a six- or seven-digit count introduced as a bound. This is the
#: shape ruling A12 struck out ("the after value is <= 946 000"), and the shape criterion 4 must
#: not grow back.
_ABSOLUTE_GATE = re.compile(
    r"(?:≤|<=|at most|under|no more than|gate of)\s*\**\s*(\d{3}[\s ,]\d{3})", re.IGNORECASE
)


def _criteria_section(text: str) -> str:
    """The acceptance criteria only. The rulings below them are history and may quote the old gate."""
    start = text.index("## Acceptance criteria")
    end = text.index("\n## ", start + 1)
    return text[start:end]


def _stated_ratios(section: str) -> dict[str, float]:
    """``{"subset": r, "suite": r}`` as the task file states them, by the nearer of the two words."""
    found: dict[str, float] = {}
    for form in _RATIO_FORMS:
        for m in form.finditer(section):
            value = float(m.group(1))
            if value > 1:  # the percent form
                value /= 100
            trailing = section[m.end() : m.end() + 80].lower()
            near = [(trailing.index(w), w) for w in ("subset", "suite") if w in trailing]
            if near:
                found.setdefault(min(near)[1], value)
    return found


def test_the_task_and_the_harness_state_the_same_numbers() -> None:
    """The fence in ``docs/tasks/T-0057.md`` and the fence in the harness are the same two numbers.

    The ``test_suite_budget.py`` precedent: a threshold that lives in two places drifts, and the
    verifier quotes whichever one it read. Ruling A12 moves three places at once -- the task file,
    ``tests/bench/audio_switch_budget.py`` and this row -- so this row is what catches a half-move
    in either direction: a harness loosened to 0.95 against a task file that still says 0.85, or a
    task file relaxed without the harness.
    """
    text = TASK.read_text()
    section = _criteria_section(text)
    stated = _stated_ratios(section)

    assert stated.get("subset") == SUBSET_MAX_RATIO, (
        f"the acceptance criteria state a subset fence of {stated.get('subset')!r} and the "
        f"harness uses {SUBSET_MAX_RATIO}. Ruling A12: criterion 4 must read as a ratio against "
        f"a same-session before -- 'at most {SUBSET_MAX_RATIO} x before on the four-module "
        f"subset' -- and the two files must agree"
    )
    assert stated.get("suite") == SUITE_MAX_RATIO, (
        f"the acceptance criteria state a suite-wide fence of {stated.get('suite')!r} and the "
        f"harness uses {SUITE_MAX_RATIO}. Ruling A12: 'at most {SUITE_MAX_RATIO} x before "
        f"suite-wide'"
    )

    absolute = _ABSOLUTE_GATE.findall(section)
    assert absolute == [], (
        f"the acceptance criteria still bound the switch count absolutely ({absolute}). The "
        "count is load-dependent -- the same fix measured 0.74, 0.62 and 0.65 at loads 6-13 -- "
        "so ruling A12 replaced the absolute gate with a ratio against a same-session before. "
        "The historical numbers belong in the Regime section and the rulings, not in a criterion"
    )
    assert re.search(r"same session", section, re.IGNORECASE), (
        "the acceptance criteria no longer require the before to be re-measured in the same "
        "session as the after; without that the ratio is between two different box-loads"
    )
    assert re.search(r"load average", section, re.IGNORECASE), (
        "the acceptance criteria no longer require the load average to be recorded beside both "
        "numbers, which is the part of the regime this counter is sensitive to"
    )
    assert re.search(r"zero leaked|leak.{0,20}zero", section, re.IGNORECASE), (
        "the acceptance criteria no longer state the real gate: zero leaked astra- threads, "
        "which is the load-independent thing the task is judged on (ruling A12)"
    )
    assert re.search(r"[Rr]eported,? not gated", section), (
        "the acceptance criteria no longer say the wall is reported and not gated; that sentence "
        "is the whole reason a suite still at 212 s after this task is the expected result"
    )
    for wall in (212.47, 214.02, 212.37):
        assert f"{wall:.2f}" in text, (
            f"the task no longer records the before wall {wall:.2f} s, which criterion 4 reports "
            "rather than gates"
        )


# ---------------------------------------------------------------------------------------------
# Criterion 5 -- the ceiling is not moved.
# ---------------------------------------------------------------------------------------------


def test_the_two_suite_budgets_are_not_raised_to_accommodate_the_leak() -> None:
    """A ceiling raised to accommodate a leak is a ceiling that hides it.

    The decision about which run is canonical is T-0050's, and it has to move five places at once.
    This task moves none of them.
    """
    assert SERIAL_CEILING_S == 120.0, (
        f"tests/conftest.py's SERIAL_CEILING_S is {SERIAL_CEILING_S}, not 120.0 (T-0057 "
        "criterion 5: the ceiling belongs to T-0050)"
    )
    assert STOP_HOOK_BUDGET_S == 60.0, (
        f"tests/conftest.py's STOP_HOOK_BUDGET_S is {STOP_HOOK_BUDGET_S}, not 60.0 (criterion 5)"
    )
    claude_md = (REPO / "CLAUDE.md").read_text()
    assert re.search(r"120 s|two minutes|two-minute", claude_md), (
        "CLAUDE.md no longer states the two-minute serial ceiling (criterion 5)"
    )


# ---------------------------------------------------------------------------------------------
# Criterion 6 -- nothing else about audio moves.
# ---------------------------------------------------------------------------------------------


def test_the_audio_modules_keep_every_assertion_they_had() -> None:
    """The audio path is T-0021's, T-0023's and T-0035's; this task transfers one file only.

    Counted by an AST walk, so a row moved or renamed is fine and a check deleted is not. If one
    of these modules has to change to stay green, that is a criterion-6 violation and a finding
    for the reviewer -- not a fix.
    """
    drift = []
    for module, (asserts, rows) in AUDIO_MODULE_ASSERTIONS.items():
        tree = ast.parse((REPO / module).read_text())
        got_asserts = sum(isinstance(n, ast.Assert) for n in ast.walk(tree))
        got_rows = sum(
            1
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name.startswith("test_")
        )
        if (got_asserts, got_rows) != (asserts, rows):
            drift.append(f"{module}: {asserts} asserts/{rows} rows -> {got_asserts}/{got_rows}")
    assert drift == [], (
        "the audio modules lost or gained checks while a lifecycle seam was being added: "
        + "; ".join(drift)
        + ". T-0057 criterion 6: every existing row of these modules stays green *without being "
        "edited*, and `git diff main...HEAD -- tests/ | grep -E '^-[^-]' | grep assert` must "
        "list nothing from them"
    )
