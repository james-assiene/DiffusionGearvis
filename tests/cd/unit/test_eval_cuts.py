"""T-0070 criterion 1: the four cuts, their disjointness, and the Sudoku test puzzles being the JAX ones.

Oracles: the ids of diffgemma-fa's preserved ``exp_e5_grammar130_map`` (BFCL test), the single-function
live_simple records 130-229 of the pinned gorilla file (BFCL dev), and ``sudoku_puzzles.json`` generated
by diffgemma-fa's own ``generate`` at 27c7d0d (tests/fixtures/cd/dgfa_preserved/MANIFEST.md). Regime:
CPU, no model.
"""

from __future__ import annotations

import json

import pytest

from tests.cd import _compile_support as cs
from tests.cd import _eval_support as es


def _ids(task: str, cut: str) -> list[str]:
    return [r.id for r in es.cuts().load(task, cut)]


def test_bfcl_test_cut_is_exactly_the_130_ids_of_the_preserved_map_arm() -> None:
    expected = [r["id"] for r in cs.preserved("exp_e5_grammar130_map")["rows"]]
    assert len(expected) == 130
    assert _ids("bfcl", "test") == expected


def test_bfcl_dev_cut_is_single_function_records_130_to_229() -> None:
    single = [r["id"] for r in cs.bfcl_records() if len(r.get("function") or ()) == 1]
    expected = single[130:230]
    assert len(expected) == 100
    got = es.cuts().load("bfcl", "dev")
    assert [r.id for r in got] == expected
    by_id = cs.bfcl_by_id()
    for r in got:
        assert r.function == by_id[r.id]["function"][0], r.id
        assert r.question == by_id[r.id]["question"], r.id


@pytest.mark.parametrize(("cut", "n", "seed"), [("test", 250, 0), ("dev", 100, 1)])
def test_sudoku_cuts_are_generate_n_seed_six_blanks(cut: str, n: int, seed: int) -> None:
    fixture = json.loads((cs.PRESERVED / "sudoku_puzzles.json").read_text())["cuts"][cut]
    assert (fixture["n"], fixture["seed"], fixture["n_blanks"]) == (n, seed, 6)
    got = es.cuts().load("sudoku", cut)
    assert len(got) == n
    for rec, (rid, puzzle, solution) in zip(got, cs.sudoku_cut(cut), strict=True):
        assert tuple(tuple(r) for r in rec.puzzle) == puzzle, rid
        assert tuple(tuple(r) for r in rec.solution) == solution, rid
        assert sum(c == 0 for row in rec.puzzle for c in row) == 6


def test_sudoku_test_ids_are_the_jax_ids() -> None:
    """The test cut joins to the preserved JAX rows by id (sudoku4_0 ... sudoku4_249, in order)."""
    assert _ids("sudoku", "test") == [
        r["id"] for r in cs.preserved("task_sudoku_unconstrained")["rows"]
    ]


@pytest.mark.parametrize("task", ["bfcl", "sudoku"])
def test_test_and_dev_are_disjoint_by_id(task: str) -> None:
    """Adversarial for Sudoku: generate() names both cuts sudoku4_0..; the harness's ids must not collide
    (T-0063's 239/250 trap came from joining the two cuts on id)."""
    test, dev = _ids(task, "test"), _ids(task, "dev")
    assert len(set(test)) == len(test) and len(set(dev)) == len(dev)
    assert not set(test) & set(dev)


def test_sudoku_test_and_dev_are_disjoint_by_puzzle() -> None:
    test = {tuple(map(tuple, r.puzzle)) for r in es.cuts().load("sudoku", "test")}
    dev = {tuple(map(tuple, r.puzzle)) for r in es.cuts().load("sudoku", "dev")}
    assert len(test) == 250 and len(dev) == 100
    assert not test & dev


def test_bfcl_test_and_dev_are_disjoint_by_function_record() -> None:
    test = {
        json.dumps(r.function, sort_keys=True) + json.dumps(r.question)
        for r in es.cuts().load("bfcl", "test")
    }
    dev = {
        json.dumps(r.function, sort_keys=True) + json.dumps(r.question)
        for r in es.cuts().load("bfcl", "dev")
    }
    assert not test & dev


def test_sudoku_test_puzzles_reproduce_every_recorded_ok_flag() -> None:
    """Re-scoring the 250 preserved unconstrained texts against the harness's test cut reproduces every
    recorded ``ok`` (216 true). A mis-seeded cut or a dev/test mix-up fails row by row."""
    recs = {r.id: r for r in es.cuts().load("sudoku", "test")}
    rows = cs.preserved("task_sudoku_unconstrained")["rows"]
    mismatches = [
        (row["id"], row["ok"])
        for row in rows
        if bool(es.cuts().score("sudoku", recs[row["id"]], row["text"])["solved"])
        != bool(row["ok"])
    ]
    assert mismatches == []
    assert sum(bool(r["ok"]) for r in rows) == 216


def test_sudoku_dev_puzzles_do_not_reproduce_the_ok_flags() -> None:
    """Companion: the same texts scored against the dev puzzles disagree somewhere (the check has teeth)."""
    dev = es.cuts().load("sudoku", "dev")
    rows = cs.preserved("task_sudoku_unconstrained")["rows"][:100]
    agree = [
        bool(es.cuts().score("sudoku", rec, row["text"])["solved"]) == bool(row["ok"])
        for rec, row in zip(dev, rows, strict=True)
    ]
    assert not all(agree)
