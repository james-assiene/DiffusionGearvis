"""T-0081 tier 2's driver ``scripts/gpu/cd_seed_ab.py``: its pure functions, hermetically (fast suite, no engine).

The driver is loaded by path with ``importlib`` (``scripts/gpu`` is not a package); nothing here starts it, posts a
request or touches a GPU. Reviewer B's follow-up:

(a) **copy rate** -- ``is_copy(committed, draft) -> bool`` (name proposed here): the constrained commit is a copy of
    the draft iff the two agree up to and including the draft's first stop id (stops 1, 50, 106). A committed list
    that runs on to the full canvas (256) after that stop is a copy; any differing token before it, or a commit that
    stops earlier, is not.
(b) **``reading(by_arm)``** (criterion 9): the three conditions are evaluated in exact ``Fraction`` arithmetic, each
    at its boundary (exact 1/2 vs 11/20 - 1/20; mean steps 6 vs 3/5 x 10; p50 wall 800 vs 4/5 x 1000 -> all True,
    "worth wiring" YES), each one tipped just past its boundary (-> NO), and a pair float arithmetic misreads
    (exact 2/100 vs 7/100 - 1/20: equal, but 0.07 - 0.05 > 0.02 in binary floating point).
(c) **knob load** -- ``load_knobs(path) -> dict`` (name proposed here): reads a T-0079 calibration file
    (``docs/cd/calibration-bfcl*.json``) and returns its ``entropy_bound`` / ``confidence_threshold``; a file for
    another arm (or another task) is refused with exit 2 (``SystemExit(2)``), before any request.

Expected values are hand-written (the BFCL ``perstep_mar`` knobs are T-0079's 1.0 / 0.005, as the task file states).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "scripts" / "gpu" / "cd_seed_ab.py"
CALIB = REPO / "docs" / "cd"


def driver() -> ModuleType:
    assert DRIVER.exists(), f"{DRIVER} does not exist"
    spec = importlib.util.spec_from_file_location("astra_t0081_cd_seed_ab", DRIVER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module through sys.modules
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        del sys.modules[spec.name]
        raise
    return mod


def attr(name: str) -> Any:
    fn = getattr(driver(), name, None)
    assert callable(fn), (
        f"scripts/gpu/cd_seed_ab.py has no {name}() (reviewer B follow-up; name proposed here)"
    )
    return fn


# --------------------------------------------------------------------------- (a) copy rate

DRAFT = [
    3724,
    106,
    1,
]  # ends at a stop id (106 = <turn|>, then EOS): the unconstrained output cut at its end
DRAFT_EOS = [2717, 3723, 107, 1]


@pytest.mark.parametrize("draft", [DRAFT, DRAFT_EOS], ids=["turn_close", "eos"])
def test_a_commit_padded_to_the_canvas_after_the_stop_is_a_copy(draft: list[int]) -> None:
    is_copy = attr("is_copy")
    stop = next(i for i, t in enumerate(draft) if t in (1, 50, 106))
    head = draft[: stop + 1]
    padded = head + [(7 * i) % 262_144 for i in range(256 - len(head))]
    assert len(padded) == 256
    assert is_copy(padded, draft) is True
    assert is_copy(list(draft), draft) is True


@pytest.mark.parametrize("draft", [DRAFT, DRAFT_EOS], ids=["turn_close", "eos"])
def test_a_differing_token_or_an_earlier_stop_is_not_a_copy(draft: list[int]) -> None:
    is_copy = attr("is_copy")
    stop = next(i for i, t in enumerate(draft) if t in (1, 50, 106))
    pad = [0] * (256 - stop - 1)
    for i in range(stop):  # every position before the stop
        changed = list(draft[: stop + 1])
        changed[i] = 4242
        assert is_copy(changed + pad, draft) is False, i
    early = [*draft[:stop], 50] if stop > 0 else [50]  # ends one stop earlier / with another id
    if early != draft[: stop + 1]:
        assert is_copy(early + [0] * (256 - len(early)), draft) is False
    assert is_copy([1] + [0] * 255, draft) is False


# --------------------------------------------------------------------------- (b) reading()


def _rows(
    n: int, exact: int, executed: list[int], wall: float, *, start_id: int = 0
) -> list[dict[str, Any]]:
    """``n`` complete rows: ``exact`` of them exact_call; per-row ``executed`` steps; every wall ``wall`` ms."""
    return [
        {
            "id": f"r{start_id + i}",
            "exact_call": i < exact,
            "schema_valid": True,
            "cs": True,
            "executed": list(executed),
            "wall_ms": wall,
            "copy": False,
            "draft_schema_valid": True,
        }
        for i in range(n)
    ]


def _by_arm(
    base: list[dict[str, Any]], seeded: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    d = driver()
    out: dict[str, list[dict[str, Any]]] = {a.name: [] for a in d.ARMS}
    out[d.SCRATCH_MAR.name] = base
    out[d.SEEDED_MAR[0].name] = seeded
    return out


def _verdicts(lines: list[str], arm: str) -> list[bool]:
    line = next(ln for ln in lines if ln.startswith(arm + ":") and "->" in ln)
    return [v == "True" for v in re.findall(r"-> (True|False)", line)]


def _worth(lines: list[str]) -> str:
    line = next(ln for ln in lines if ln.startswith("worth wiring"))
    return line.rsplit(" ", 1)[-1]


def test_reading_holds_exactly_at_all_three_boundaries() -> None:
    """exact 10/20 = 1/2 vs 11/20 - 1/20; mean steps 6 vs 3/5 x 10; p50 wall 800 vs 4/5 x 1000: all three True."""
    d = driver()
    base = _rows(20, 11, [10], 1000.0)
    seeded = _rows(20, 10, [6], 800.0)
    lines = d.reading(_by_arm(base, seeded))
    assert _verdicts(lines, d.SEEDED_MAR[0].name) == [True, True, True], lines
    assert _worth(lines) == "YES", lines


@pytest.mark.parametrize(
    ("which", "seeded_kw"),
    [
        ("exact", {"exact": 9, "executed": [6], "wall": 800.0}),
        ("steps", {"exact": 10, "executed": [6, 7] * 1 + [6] * 8, "wall": 800.0}),
        ("wall", {"exact": 10, "executed": [6], "wall": 800.001}),
    ],
)
def test_reading_fails_just_past_each_boundary(which: str, seeded_kw: dict[str, Any]) -> None:
    d = driver()
    base = _rows(20, 11, [10], 1000.0)
    seeded = _rows(20, seeded_kw["exact"], seeded_kw["executed"], seeded_kw["wall"])
    lines = d.reading(_by_arm(base, seeded))
    want = {
        "exact": [False, True, True],
        "steps": [True, False, True],
        "wall": [True, True, False],
    }[which]
    assert _verdicts(lines, d.SEEDED_MAR[0].name) == want, lines
    assert _worth(lines) == "NO", lines


def test_reading_uses_exact_arithmetic_where_floats_misread() -> None:
    """Base exact 7/100, seeded 2/100: 2/100 >= 7/100 - 1/20 holds exactly, but 0.07 - 0.05 = 0.020000000000000004."""
    assert not (0.02 >= 0.07 - 0.05), "premise: binary floating point misreads this pair"
    assert Fraction(2, 100) >= Fraction(7, 100) - Fraction(1, 20)
    d = driver()
    base = _rows(100, 7, [10], 1000.0)
    seeded = _rows(100, 2, [5], 700.0)
    lines = d.reading(_by_arm(base, seeded))
    assert _verdicts(lines, d.SEEDED_MAR[0].name) == [True, True, True], lines
    assert _worth(lines) == "YES", lines


# --------------------------------------------------------------------------- (c) knob load


def test_the_bfcl_perstep_mar_calibration_loads_t0079s_knobs() -> None:
    knobs = attr("load_knobs")(CALIB / "calibration-bfcl.json")
    assert (knobs["entropy_bound"], knobs["confidence_threshold"]) == (1.0, 0.005)


@pytest.mark.parametrize("name", ["calibration-bfcl-perstep_mar_qargmax.json"])
def test_a_calibration_file_for_another_arm_is_refused_with_exit_2(name: str) -> None:
    load = attr("load_knobs")
    with pytest.raises(SystemExit) as info:
        load(CALIB / name)
    assert info.value.code == 2


@pytest.mark.parametrize("arm", ["perstep_mf", "perstep_mar_qargmax", "end_projection"])
def test_a_crafted_wrong_arm_is_refused_and_the_control_loads(arm: str, tmp_path: Path) -> None:
    load = attr("load_knobs")
    data = json.loads((CALIB / "calibration-bfcl.json").read_text())
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps(data))
    assert load(ok)["entropy_bound"] == 1.0  # control: the unmodified copy loads
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**data, "arm": arm}))
    with pytest.raises(SystemExit) as info:
        load(bad)
    assert info.value.code == 2


def test_a_sudoku_calibration_is_refused_for_the_bfcl_run() -> None:
    """Same arm, wrong task: the Sudoku ``perstep_mar`` knobs must not be used for a BFCL run."""
    with pytest.raises(SystemExit) as info:
        attr("load_knobs")(CALIB / "calibration-sudoku.json")
    assert info.value.code == 2
