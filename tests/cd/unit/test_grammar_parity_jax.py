"""T-0063 criteria 5-6: grammar parity with the JAX baseline, by replay of preserved emissions.

Each preserved row's ``text`` is re-tokenised with the pinned HF tokenizer (``replay_ids``: tokens + EOS,
see tests/fixtures/cd/dgfa_preserved/MANIFEST.md for why) and walked through the host-compiled automaton
with an independent set-of-states walker. Acceptance must equal the recorded ``accepted`` flag on every
row; a mismatch lists the rows and is never waived by count.

Grammar options are the baseline artifact's (``exp_e5_grammar130_map``): whitespace ``pretty``, optional
fence, case-insensitive enums, expanded wildcards, nonempty required strings, channel header
``100 . Σ_name{1,8} . 107 . 101``. Sudoku is diffgemma-fa ``eval/run_tasks.py``'s
``sudoku_regex(puzzle, row_separator="\\n")`` with the default 8-token header.

API these rows assume: ``grammars.compile_bfcl(function)``, ``grammars.compile_sudoku(puzzle)``,
``grammars.BFCL_BASELINE_OPTIONS`` (mapping of option -> value), and the criterion-6 function
``report.sudoku_answer_acceptance(rows, puzzles) -> Mapping[id, bool]``.

Regime: CPU only; outlines-core 0.2.14; tokenizer.json sha256 cc8d3a0c...; model rev 2ea8372...;
diffgemma-fa 27c7d0d. The BFCL rows need tests/fixtures/cd/bfcl/ (human-approved gorilla data) and
fail with a message naming it while it is absent.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from tests.cd._compile_support import (
    EOS,
    PHASE0_HEADER,
    answer_region_oracle,
    bfcl_function,
    bfcl_test_cut_ids,
    compiled_bfcl,
    compiled_sudoku,
    encode,
    grammars,
    preserved,
    replay_ids,
    report,
    sudoku_cut,
    sudoku_regex_oracle,
    walker,
    warm_bfcl,
    warm_sudoku,
)

#: Criterion 6's number, from the independent oracle below over the 250 preserved unconstrained texts
#: (measured 2026-09-29 with diffgemma-fa's own answer_region + sudoku_regex; re-derived in-row).
ORACLE_SUDOKU_UNCONSTRAINED_ACCEPTED = 239


def _replay_mismatches(rows: Sequence[Mapping[str, Any]], automaton_for: Any) -> list[str]:
    bad = []
    for row in rows:
        got = walker(automaton_for(row)).accepts(replay_ids(row["text"]))
        if got != bool(row["accepted"]):
            bad.append(
                f"{row['id']}: recorded accepted={row['accepted']}, replay={got}, text={row['text'][:80]!r}"
            )
    return bad


def _sudoku_puzzles() -> dict[str, tuple[tuple[int, ...], ...]]:
    return {rid: puzzle for rid, puzzle, _ in sudoku_cut("test")}


# --------------------------------------------------------------------------- options


def test_bfcl_baseline_options_equal_the_baseline_artifact() -> None:
    art = preserved("exp_e5_grammar130_map")
    opts = dict(grammars().BFCL_BASELINE_OPTIONS)
    expected = {
        "whitespace": art["whitespace"],
        "fence": art["fence"],
        "ci_enums": art["ci_enums"],
        "expand_wildcard": art["expand_wildcard"],
        "nonempty_strings": art["nonempty_strings"],
    }
    assert expected == {
        "whitespace": "pretty",
        "fence": True,
        "ci_enums": True,
        "expand_wildcard": True,
        "nonempty_strings": True,
    }
    assert {k: opts.get(k) for k in expected} == expected
    assert opts.get("channel_header") is True


# --------------------------------------------------------------------------- criterion 5, BFCL


def test_bfcl_test_cut_function_names_match_the_preserved_rows() -> None:
    """Fixture identity: the approved BFCL data is the data the JAX arm ran on (by id and function)."""
    rows = preserved("exp_e5_grammar130_map")["rows"]
    assert len(rows) == 130
    wrong = [
        (r["id"], r["fn"], bfcl_function(r["id"])["name"])
        for r in rows
        if bfcl_function(r["id"])["name"] != r["fn"]
    ]
    assert not wrong, wrong


def test_all_130_rows_of_the_grammar130_map_arm_are_accepted() -> None:
    rows = preserved("exp_e5_grammar130_map")["rows"]
    assert len(rows) == 130 and all(r["accepted"] for r in rows)
    assert [r["id"] for r in rows] == bfcl_test_cut_ids()
    warm_bfcl([r["id"] for r in rows])
    bad = _replay_mismatches(rows, lambda r: compiled_bfcl(r["id"]))
    assert not bad, f"{len(bad)} of 130 grammar130_map rows rejected by replay:\n" + "\n".join(bad)


def test_unconstrained_bfcl_acceptance_equals_the_recorded_flag() -> None:
    rows = preserved("eval_bfcl_live_simple_unconstrained_sample")["rows"]
    assert len(rows) == 130
    warm_bfcl([r["id"] for r in rows])
    bad = _replay_mismatches(rows, lambda r: compiled_bfcl(r["id"]))
    assert not bad, (
        f"{len(bad)} unconstrained BFCL rows disagree with the recorded flag:\n" + "\n".join(bad)
    )


# --------------------------------------------------------------------------- criterion 5, Sudoku


@pytest.mark.parametrize("artifact", ["task_sudoku_unconstrained", "task_sudoku_j0map"])
def test_sudoku_acceptance_equals_the_recorded_flag(artifact: str) -> None:
    rows = preserved(artifact)["rows"]
    assert len(rows) == 250
    puzzles = _sudoku_puzzles()
    warm_sudoku("test")
    bad = _replay_mismatches(rows, lambda r: compiled_sudoku(puzzles[r["id"]]))
    assert not bad, (
        f"{len(bad)} of 250 {artifact} rows disagree with the recorded flag:\n" + "\n".join(bad)
    )


def test_sudoku_givens_are_pinned() -> None:
    """Discriminating row for the j0map parity above: overwriting one given is rejected, the true
    solution is accepted, so "all 250 accepted" cannot come from a format-only grammar."""
    rid, puzzle, solution = sudoku_cut("test")[0]
    a = compiled_sudoku(puzzle)
    good = "\n".join("".join(str(c) for c in row) for row in solution)
    assert walker(a).accepts([*PHASE0_HEADER, *encode(good), EOS])
    r, c = next((r, c) for r in range(4) for c in range(4) if puzzle[r][c])
    grid = [list(row) for row in solution]
    grid[r][c] = grid[r][c] % 4 + 1
    bad = "\n".join("".join(str(x) for x in row) for row in grid)
    assert not walker(a).accepts([*PHASE0_HEADER, *encode(bad), EOS]), (
        f"{rid}: given ({r},{c}) overwritten and accepted"
    )


# --------------------------------------------------------------------------- criterion 6


def test_sudoku_oracle_self_check() -> None:
    """The oracle reproduces the number measured with diffgemma-fa's own functions (so it is not a
    re-statement with a bug of its own)."""
    rows = preserved("task_sudoku_unconstrained")["rows"]
    puzzles = _sudoku_puzzles()
    n = sum(
        bool(re.fullmatch(sudoku_regex_oracle(puzzles[r["id"]]), answer_region_oracle(r["text"])))
        for r in rows
    )
    assert n == ORACLE_SUDOKU_UNCONSTRAINED_ACCEPTED


def test_unconstrained_sudoku_answer_acceptance_equals_the_regex_oracle() -> None:
    rows = preserved("task_sudoku_unconstrained")["rows"]
    assert len(rows) == 250
    puzzles = _sudoku_puzzles()
    result = report().sudoku_answer_acceptance(rows, puzzles)
    assert set(result) == {r["id"] for r in rows}, "one verdict per record, none dropped"
    oracle = {
        r["id"]: bool(
            re.fullmatch(sudoku_regex_oracle(puzzles[r["id"]]), answer_region_oracle(r["text"]))
        )
        for r in rows
    }
    bad = [
        f"{rid}: function={bool(result[rid])}, oracle={oracle[rid]}"
        for rid in oracle
        if bool(result[rid]) != oracle[rid]
    ]
    assert not bad, (
        f"{len(bad)} rows disagree with re.fullmatch(sudoku_regex, answer_region):\n"
        + "\n".join(bad)
    )
    assert sum(bool(v) for v in result.values()) == ORACLE_SUDOKU_UNCONSTRAINED_ACCEPTED
