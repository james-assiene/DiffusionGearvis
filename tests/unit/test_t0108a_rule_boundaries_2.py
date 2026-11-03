"""T-0108a criterion 6, the second half of the boundary rows (unit; CPU; no engine). The rows and the check live in
``tests/unit/test_t0108a_rule_boundaries.py``; this module only runs the rows that one does not, so that a third
``--dist loadfile`` worker shares the cost.

Regime: CPU, pure, hand-built data, the real ``UiGate`` and vendored A2UI v0.9.1 schema at the working tree; no
measurement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.test_t0108a_rule_boundaries import BOUNDARY_ROWS, FIRST_HALF, check_boundary_row

SECOND_HALF = [row for row in BOUNDARY_ROWS if row not in FIRST_HALF]


def test_the_two_halves_are_every_row_once() -> None:
    """Every boundary row runs exactly once across the two modules (32 rows: 16 + 16)."""
    assert len(BOUNDARY_ROWS) == 32
    assert sorted(FIRST_HALF + SECOND_HALF) == sorted(BOUNDARY_ROWS)
    assert len(FIRST_HALF) == len(SECOND_HALF) == 16


@pytest.mark.parametrize("row", SECOND_HALF)
def test_t0108a_rule_boundaries_2(tmp_path: Path, row: str) -> None:
    """The second half of criterion 6's rows; the check is ``check_boundary_row``."""
    check_boundary_row(tmp_path, row)
