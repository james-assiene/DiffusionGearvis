"""Test-of-the-test for the T-0070 rows: the scripted Sudoku texts mean what the rows assume.

``solved``/``wrong`` texts are accepted by the puzzle's compiled automaton (replayed as
``encode(text) + [EOS]`` through the independent set-of-states walker of ``_compile_support``), and
``rejected`` is not; ``solved`` solves the puzzle and ``wrong`` does not (checked by rows/columns/boxes by
hand, not by the harness). The header is Phase 0's ids. Regime: CPU, first 10 Sudoku test puzzles.
"""

from __future__ import annotations

import pytest

from tests.cd import _compile_support as cs
from tests.cd import _eval_support as es


def _is_solution(grid: list[list[int]]) -> bool:
    want = {1, 2, 3, 4}
    rows = all(set(r) == want for r in grid)
    cols = all({grid[r][c] for r in range(4)} == want for c in range(4))
    boxes = all(
        {grid[r][c] for r in (br, br + 1) for c in (bc, bc + 1)} == want
        for br in (0, 2)
        for bc in (0, 2)
    )
    return rows and cols and boxes


def _grid(text: str) -> list[list[int]]:
    return [[int(ch) for ch in line] for line in text.split("<channel|>", 1)[1].splitlines()]


def test_header_is_the_phase0_ids() -> None:
    assert cs.encode(es.HEADER) == cs.PHASE0_HEADER


@pytest.mark.parametrize("k", range(10))
def test_scripted_texts_replay_as_intended(k: int) -> None:
    rid, puzzle, solution = cs.sudoku_cut("test")[k]
    w = cs.walker(cs.compiled_sudoku(puzzle))
    solved, wrong, rejected = (
        es.solved_text(solution),
        es.wrong_text(puzzle, solution),
        es.rejected_text(puzzle, solution),
    )
    assert w.accepts(cs.replay_ids(solved)), rid
    assert w.accepts(cs.replay_ids(wrong)), rid
    assert not w.accepts(cs.replay_ids(rejected)), rid
    assert _is_solution(_grid(solved))
    assert not _is_solution(_grid(wrong))
