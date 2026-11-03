"""T-0146: inside an xdist worker (the Stop-hook gate) Hypothesis runs every phase but ``shrink`` and
``explain``; the serial canonical run keeps the default phases; nothing but ``phases`` differs.

How the rows see the settings (docs/tasks/T-0146.md, "Tests to write"):

* ``test_probe_effective_hypothesis_settings`` is the probe. It reads the four stateful ``TestCase``
  classes of ``tests/unit/test_turn_invariants.py`` -- imported **at this module's import**, so their
  ``.settings`` are built at collection time exactly as they are in the suite -- and builds two
  settings objects in its body: ``settings(max_examples=1)`` (criterion 1's call-time object) and a
  bare ``settings()``. When ``ASTRA_T0146_PROBE_OUT`` names a path it writes every object's effective
  fields there as JSON and asserts nothing (the driver mode used by the rows below). Otherwise it
  asserts in-process: the worker branch in a worker, the serial branch otherwise.
* Every other row drives the probe in a child pytest (``sys.executable -m pytest``, ``cwd=REPO``,
  ``-p no:cacheprovider``) once per arm, all arms launched together by one module-scoped fixture,
  and reads the JSON. Each arm's premise (is the child a worker; is xdist loaded) is recorded by the
  probe and asserted, so an arm that silently ran in the wrong mode cannot pass.

Why the bare ``settings()``: every object criterion 1 names passes ``max_examples`` explicitly
(200, 20, 200, 300 and 1), and an explicit argument beats the active profile. A gate profile that
also set ``max_examples`` (reviewer mutation P1) would therefore leave all five equal between the
worker and the serial run. The bare object inherits every field from the active profile, which is
where such a widening shows up.

The gate profile must inherit from the profile that is active, not from ``default``: under ``CI=1``
Hypothesis loads its ``ci`` profile (``derandomize=True, database=None``) and ``--hypothesis-verbosity``
loads a ``<profile>-with-<v>-verbosity`` profile. ``test_worker_inherits_the_active_profile`` runs a
worker and a serial arm under each and asserts that every non-phase field is equal between them
(reviewer A, 2026-10-23: a profile built with ``parent=settings.get_profile("default")`` passed
every other row).

No row is pending any more: rows 1 and 5 and the probe's in-process worker branch were pending until
T-0146's activation commit (2194df1); the rest held at BASE d8515d8 and were never marked.

Criteria 5-9 are measurements (reviewers and verifier), not rows.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import settings

# Imported at module level on purpose: the stateful classes' settings must be built at collection
# time, before any test body runs, as they are in the suite. A probe that imported them lazily from
# its body would see objects built *after* a too-late profile load and miss reviewer mutation P2.
from tests.unit import test_turn_invariants as tti

REPO = Path(__file__).resolve().parents[2]
THIS = Path(__file__).resolve().relative_to(REPO).as_posix()
PROBE = f"{THIS}::test_probe_effective_hypothesis_settings"
OUT_ENV = "ASTRA_T0146_PROBE_OUT"
INVARIANTS = REPO / "tests" / "unit" / "test_turn_invariants.py"

#: Criterion 1, copied by hand from the task file: every phase but shrink and explain.
GATE_PHASES = ["explicit", "reuse", "generate", "target"]
#: Hypothesis's default phases (6.168.0), copied by hand; the serial run must still carry all six.
DEFAULT_PHASES = ["explicit", "reuse", "generate", "target", "shrink", "explain"]

#: Criterion 3, copied by hand from tests/unit/test_turn_invariants.py (max_examples, step count).
DECLARED = {
    "TestTurnMachineInvariants": (200, 40),
    "TestTurnMachineBoundedState": (20, 200),
    "TestToolTurnInvariants": (200, 40),
    "TestToolLoopInvariants": (300, 80),
}
IMPORT_TIME = tuple(DECLARED)
CALL_TIME = ("call_time_max_examples_1", "call_time_bare")
OBJECTS = IMPORT_TIME + CALL_TIME
NON_PHASE_FIELDS = (
    "max_examples",
    "stateful_step_count",
    "deadline",
    "derandomize",
    "database",
    "suppress_health_check",
    "verbosity",
    "report_multiple_bugs",
    "print_blob",
)

_IN_WORKER = bool(os.environ.get("PYTEST_XDIST_WORKER"))
_DRIVER = bool(os.environ.get(OUT_ENV))
#: Only the probe's in-process worker branch needs the new behaviour (see the module docstring).
_probe_marks: list[pytest.MarkDecorator] = []  # activated for T-0146 (was pending in workers)


def _fields(s: settings) -> dict[str, Any]:
    db = s.database
    return {
        "phases": [p.name for p in s.phases],
        "max_examples": s.max_examples,
        "stateful_step_count": s.stateful_step_count,
        "deadline": None if s.deadline is None else s.deadline.total_seconds(),
        "derandomize": s.derandomize,
        # Criterion 3: compare the database by type and path.
        "database": None if db is None else [type(db).__name__, str(getattr(db, "path", None))],
        "suppress_health_check": sorted(h.name for h in s.suppress_health_check),
        "verbosity": s.verbosity.name,
        "report_multiple_bugs": s.report_multiple_bugs,
        "print_blob": s.print_blob,
    }


def _probe_objects() -> dict[str, settings]:
    objs: dict[str, settings] = {name: getattr(tti, name).settings for name in IMPORT_TIME}
    objs["call_time_max_examples_1"] = settings(max_examples=1)
    objs["call_time_bare"] = settings()
    return objs


def _probe_mark(fn: Any) -> Any:
    for mark in _probe_marks:
        fn = mark(fn)
    return fn


@_probe_mark
def test_probe_effective_hypothesis_settings(request: pytest.FixtureRequest) -> None:
    objs = _probe_objects()
    fields = {name: _fields(s) for name, s in objs.items()}
    out = os.environ.get(OUT_ENV)
    if out:
        Path(out).write_text(
            json.dumps(
                {
                    "worker": os.environ.get("PYTEST_XDIST_WORKER"),
                    "has_workerinput": hasattr(request.config, "workerinput"),
                    "xdist_loaded": request.config.pluginmanager.hasplugin("xdist"),
                    "explicit_profile": request.config.getoption("hypothesis_profile", None),
                    "current_profile": settings.get_current_profile_name(),
                    "objects": fields,
                },
                indent=1,
            )
        )
        return
    if request.config.getoption("hypothesis_profile", None):
        pytest.skip("an explicit --hypothesis-profile decides the phases; row 4 covers it")
    if _IN_WORKER:
        bad = {n: f["phases"] for n, f in fields.items() if f["phases"] != GATE_PHASES}
        assert not bad, f"in an xdist worker these objects still carry other phases: {bad}"
    else:
        bad = {
            n: f["phases"]
            for n, f in fields.items()
            if not {"shrink", "explain"} <= set(f["phases"])
        }
        assert not bad, f"in a serial run these objects lost shrink/explain: {bad}"


# ------------------------------------------------------------------------------------------------
# The driver: one child pytest per arm, all launched together, each writing the probe's JSON.
# ------------------------------------------------------------------------------------------------

ARMS: dict[str, list[str]] = {
    # Row 1: an xdist worker (the gate's mode).
    "worker": ["-n", "1"],
    # Row 2: plain serial run.
    "serial": [],
    # Row 4: explicit profile in a worker.
    "explicit_default_in_worker": ["-n", "1", "--hypothesis-profile=default"],
    # Row 6: xdist loaded explicitly, no -n.
    "xdist_loaded_no_n": ["-p", "xdist"],
    # Row 6: -n 0 runs in-process (no worker).
    "n0": ["-n", "0"],
    # test_worker_inherits_the_active_profile: Hypothesis's ci profile (CI=1) and the
    # --hypothesis-verbosity profile, each in a worker and serially.
    "ci_worker": ["-n", "1"],
    "ci_serial": [],
    "verbose_worker": ["-n", "1", "--hypothesis-verbosity=verbose"],
    "verbose_serial": ["--hypothesis-verbosity=verbose"],
}
#: Extra child environment per arm (on top of :func:`_child_env`).
ARM_ENV: dict[str, dict[str, str]] = {"ci_worker": {"CI": "1"}, "ci_serial": {"CI": "1"}}


def _child_env(out: Path) -> dict[str, str]:
    env = dict(os.environ)
    # When this module itself runs in a gate worker, the worker's identity must not leak into a
    # child that is meant to be serial: a trigger keyed on PYTEST_XDIST_WORKER would otherwise see
    # every arm as a worker. PYTEST_ADDOPTS could add -n to the serial arm.
    for key in list(env):
        if key.startswith("PYTEST_XDIST_") or key in {"PYTEST_CURRENT_TEST", "PYTEST_ADDOPTS"}:
            del env[key]
    env[OUT_ENV] = str(out)
    return env


class Arm:
    def __init__(self, name: str, out: Path, proc: subprocess.CompletedProcess[str]) -> None:
        self.name = name
        self.out = out
        self.proc = proc

    @property
    def tail(self) -> str:
        text = (self.proc.stdout or "") + (self.proc.stderr or "")
        return (
            f"\n--- arm {self.name}: argv {self.proc.args} rc={self.proc.returncode}\n"
            + "\n".join(text.splitlines()[-25:])
        )

    def data(self) -> dict[str, Any]:
        assert self.out.exists(), f"the probe wrote no JSON in arm {self.name}{self.tail}"
        assert self.proc.returncode == 0, f"the probe's child pytest failed{self.tail}"
        return json.loads(self.out.read_text())  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def arms(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Arm]:
    base = tmp_path_factory.mktemp("t0146")
    procs: dict[str, tuple[Path, list[str], subprocess.Popen[str]]] = {}
    for name, extra in ARMS.items():
        out = base / f"{name}.json"
        argv = [sys.executable, "-m", "pytest", *extra, "-p", "no:cacheprovider", "-q", PROBE]
        procs[name] = (
            out,
            argv,
            subprocess.Popen(
                argv,
                cwd=REPO,
                env={**_child_env(out), **ARM_ENV.get(name, {})},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ),
        )
    result: dict[str, Arm] = {}
    for name, (out, argv, p) in procs.items():
        try:
            stdout, stderr = p.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
            stdout, stderr = p.communicate()
        result[name] = Arm(
            name, out, subprocess.CompletedProcess(argv, p.returncode, stdout, stderr)
        )
    return result


def _assert_worker(arm: Arm, data: dict[str, Any]) -> None:
    assert data["worker"] and data["has_workerinput"], (
        f"premise: arm {arm.name} must run the probe in an xdist worker{arm.tail}"
    )


def _assert_not_worker(arm: Arm, data: dict[str, Any]) -> None:
    assert not data["worker"] and not data["has_workerinput"], (
        f"premise: arm {arm.name} must run the probe in-process, not in a worker{arm.tail}"
    )


def _assert_shrinks(arm: Arm, data: dict[str, Any]) -> None:
    objs = data["objects"]
    assert set(objs) == set(OBJECTS), f"probe objects {sorted(objs)}{arm.tail}"
    lost = {
        name: f["phases"] for name, f in objs.items() if not set(DEFAULT_PHASES) <= set(f["phases"])
    }
    assert not lost, (
        f"arm {arm.name}: these settings objects lost a default phase (shrink/explain must stay "
        f"outside xdist workers or under an explicit profile): {lost}{arm.tail}"
    )


# ------------------------------------------------------------------------------------------------
# Rows
# ------------------------------------------------------------------------------------------------


def test_xdist_worker_runs_every_phase_but_shrink(arms: dict[str, Arm]) -> None:
    """Row 1 / criterion 1: in a worker every object has exactly explicit, reuse, generate, target."""
    arm = arms["worker"]
    data = arm.data()
    _assert_worker(arm, data)
    objs = data["objects"]
    assert set(objs) == set(OBJECTS), f"probe objects {sorted(objs)}{arm.tail}"
    wrong = {name: f["phases"] for name, f in objs.items() if f["phases"] != GATE_PHASES}
    assert not wrong, (
        f"in an xdist worker every settings object must have phases {GATE_PHASES}; these do not: "
        f"{wrong}{arm.tail}"
    )


def test_serial_run_keeps_the_default_phases(arms: dict[str, Arm]) -> None:
    """Row 2 / criterion 2: outside xdist workers shrink and explain are still on, for every object."""
    arm = arms["serial"]
    data = arm.data()
    _assert_not_worker(arm, data)
    _assert_shrinks(arm, data)


def test_only_phases_differs_between_worker_and_serial(arms: dict[str, Arm]) -> None:
    """Row 3 / criterion 3: every non-phases field is equal worker vs serial; the declared literals hold."""
    w_arm, s_arm = arms["worker"], arms["serial"]
    worker, serial = w_arm.data()["objects"], s_arm.data()["objects"]
    tails = w_arm.tail + s_arm.tail
    assert set(worker) == set(serial) == set(OBJECTS), f"{sorted(worker)} {sorted(serial)}{tails}"
    diffs = {
        (name, field): (worker[name][field], serial[name][field])
        for name in OBJECTS
        for field in NON_PHASE_FIELDS
        if worker[name][field] != serial[name][field]
    }
    assert not diffs, (
        f"the gate profile may change phases only; (object, field): (worker, serial) {diffs}{tails}"
    )
    for name, (examples, steps) in DECLARED.items():
        for which, objs in (("worker", worker), ("serial", serial)):
            got = (objs[name]["max_examples"], objs[name]["stateful_step_count"])
            assert got == (examples, steps), (
                f"{name}.settings in the {which} run: (max_examples, stateful_step_count) = {got}, "
                f"test_turn_invariants.py declares {(examples, steps)}{tails}"
            )
            assert objs[name]["deadline"] is None, f"{name} deadline in {which}{tails}"
    for which, objs in (("worker", worker), ("serial", serial)):
        assert objs["call_time_max_examples_1"]["max_examples"] == 1, f"{which}{tails}"


def test_explicit_hypothesis_profile_wins_in_a_worker(arms: dict[str, Arm]) -> None:
    """Row 4 / criterion 4: ``-n 1 --hypothesis-profile=default`` gives the default phases, shrink on."""
    arm = arms["explicit_default_in_worker"]
    data = arm.data()
    _assert_worker(arm, data)
    assert data["explicit_profile"] == "default", f"premise: profile option{arm.tail}"
    _assert_shrinks(arm, data)


def test_import_time_settings_follow_the_gate_profile(arms: dict[str, Arm]) -> None:
    """Row 5, adversarial (a profile loaded after the test modules import).

    Reads the classes' ``.settings`` (built at import), never ``settings.default``: on 6.168.0 an
    object built before ``load_profile`` keeps shrink while ``settings.default`` does not.
    """
    arm = arms["worker"]
    data = arm.data()
    _assert_worker(arm, data)
    objs = data["objects"]
    call_time = objs["call_time_max_examples_1"]["phases"]
    for name in IMPORT_TIME:
        assert objs[name]["phases"] == GATE_PHASES, (
            f"{name}.settings (built at import) has phases {objs[name]['phases']} in a worker; the "
            f"gate profile must be active before test modules import{arm.tail}"
        )
        assert objs[name]["phases"] == call_time, (
            f"{name}.settings (import time) {objs[name]['phases']} != settings(max_examples=1) "
            f"(call time) {call_time}{arm.tail}"
        )


#: Premise per mode, by hand from Hypothesis 6.168.0: what the *serial* arm must show for the mode to
#: be active at all (else equality between the arms would be vacuous). ``ci`` is Hypothesis's built-in
#: profile under ``CI=1`` (``derandomize=True, database=None``); ``verbose`` is the
#: ``--hypothesis-verbosity=verbose`` option.
INHERITED_MODES = {
    "ci": {"derandomize": True, "database": None},
    "verbose": {"verbosity": "verbose"},
}


@pytest.mark.parametrize("mode", sorted(INHERITED_MODES))
def test_worker_inherits_the_active_profile(arms: dict[str, Arm], mode: str) -> None:
    """Criterion 3 under a non-default active profile (reviewer A, 2026-10-23).

    The gate profile changes phases only, so in a worker it must sit on top of whatever profile is
    active (``ci`` under ``CI=1``, the verbosity profile under ``--hypothesis-verbosity``), not on top
    of ``default``. A gate profile built with ``parent=settings.get_profile("default")`` makes a CI
    worker drop ``derandomize=True`` and ``database=None`` while the serial run keeps them.
    """
    w_arm, s_arm = arms[f"{mode}_worker"], arms[f"{mode}_serial"]
    worker, serial = w_arm.data(), s_arm.data()
    tails = w_arm.tail + s_arm.tail
    _assert_worker(w_arm, worker)
    _assert_not_worker(s_arm, serial)
    for name in OBJECTS:
        # None of the probed objects sets derandomize, database or verbosity explicitly.
        for field, value in INHERITED_MODES[mode].items():
            assert serial["objects"][name][field] == value, (
                f"premise: the serial {mode} arm must show {field}={value!r} on {name}, got "
                f"{serial['objects'][name][field]!r} (profile {serial['current_profile']}){tails}"
            )
        assert worker["objects"][name]["phases"] == GATE_PHASES, (
            f"the {mode} worker arm must still run the gate phases on {name}: "
            f"{worker['objects'][name]['phases']}{tails}"
        )
    diffs = {
        (name, field): (worker["objects"][name][field], serial["objects"][name][field])
        for name in OBJECTS
        for field in NON_PHASE_FIELDS
        if worker["objects"][name][field] != serial["objects"][name][field]
    }
    assert not diffs, (
        f"under the {mode} profile the gate must change phases only; (object, field): "
        f"(worker, serial) {diffs} (worker profile {worker['current_profile']}, serial profile "
        f"{serial['current_profile']}){tails}"
    )


@pytest.mark.parametrize("arm_name", ["xdist_loaded_no_n", "n0"])
def test_a_non_worker_run_with_xdist_installed_still_shrinks(
    arms: dict[str, Arm], arm_name: str
) -> None:
    """Row 6, adversarial (too wide): xdist loaded or ``-n 0`` is not a worker and must still shrink."""
    arm = arms[arm_name]
    data = arm.data()
    _assert_not_worker(arm, data)
    assert data["xdist_loaded"], f"premise: xdist must be loaded in arm {arm_name}{arm.tail}"
    _assert_shrinks(arm, data)


_BLOCKING_MARKS = {"xfail", "skip", "skipif", "pending"}


def _mark_names(decorators: list[ast.expr]) -> set[str]:
    names: set[str] = set()
    for d in decorators:
        node = d.func if isinstance(d, ast.Call) else d
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


def test_the_m4_shape_has_a_deterministic_row() -> None:
    """Row 7: criterion 6(c) names this row, so a rename or a skip/xfail/pending must fail here first."""
    tree = ast.parse(INVARIANTS.read_text())
    table = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TestTable"]
    assert len(table) == 1, "tests/unit/test_turn_invariants.py must define exactly one TestTable"
    cls = table[0]
    assert not (_mark_names(cls.decorator_list) & _BLOCKING_MARKS), (
        f"TestTable carries {_mark_names(cls.decorator_list) & _BLOCKING_MARKS}"
    )
    rows = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        and n.name == "test_thought_commit_and_unclosed_thought_never_speak"
    ]
    assert len(rows) == 1, (
        "TestTable.test_thought_commit_and_unclosed_thought_never_speak is the row criterion 6(c) "
        "names; it is missing or defined twice"
    )
    blocking = _mark_names(rows[0].decorator_list) & _BLOCKING_MARKS
    assert not blocking, f"the M4 row is marked {blocking}; it must run and be able to fail"
    # A ``pytestmark`` at module level, in TestTable's body or assigned as ``X.pytestmark`` reaches
    # the row as well as a decorator does.
    for node in [*tree.body, *cls.body]:
        if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(
            (isinstance(t, ast.Name) and t.id == "pytestmark")
            or (isinstance(t, ast.Attribute) and t.attr == "pytestmark")
            for t in targets
        ):
            continue
        found = {n.attr for n in ast.walk(node.value) if isinstance(n, ast.Attribute)} | {
            n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)
        }
        assert not (found & _BLOCKING_MARKS), (
            f"a pytestmark in test_turn_invariants.py (line {node.lineno}) carries "
            f"{found & _BLOCKING_MARKS}; it would reach the M4 row"
        )
