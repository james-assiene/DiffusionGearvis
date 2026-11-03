"""T-0108a criterion 6: every boundary row of the decision rule H-T108a on hand-counted 1320-line runs (unit; CPU; no
engine). Split from ``tests/unit/test_t0108a_ui_fields_ab.py`` (whose docstring states the assumed surface, the baseline
run and the thresholds) so that two workers share the cost under ``--dist loadfile``; every helper is imported from there.

Every row needs the full run: an incomplete run is inconclusive, so a smaller fixture could not assert the outcome.

Regime: CPU, pure, hand-built data, the real ``UiGate`` and vendored A2UI v0.9.1 schema at the working tree; no
measurement.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_t0098_prompt_ab import ANOMALY
from tests.unit.test_t0098c_ui_prompt_ab import ERROR_C, OPEN_SURFACE, Q12_TOOL, WITH_SURFACE
from tests.unit.test_t0103_c3_confirm import (
    NEGATIVE_SLOTS,
    NO_BELONG_SLOTS,
    PLACEMENT_SLOTS,
    PLAIN_SLOTS,
    TOOL_SLOTS,
)
from tests.unit.test_t0108a_ui_fields_ab import (
    A0_63,
    A0_64,
    K1_BASE,
    K_90,
    Plan,
    _get_path,
    _kn,
    _pos,
    _put,
    _seq,
    _type_silent,
    failing,
    four,
    kind_of,
    named,
    report_108a,
)

#: (edit, K1 failing, K2 failing, outcome kind, winner, reason fragment, {arm: {key: value}} premises)
Row = tuple[
    Callable[[Plan], None],
    set[str],
    set[str],
    str,
    str | None,
    str | None,
    dict[str, dict[str, Any]],
]
Q1 = {"Q1"}
BOUNDARY_ROWS: dict[str, Row] = {
    # Q1: K complete >= A0 + 24 of 160 (63 + 24 = 87); no case protected (A0 <= 16 per case)
    "q1-63-to-87-passes": (
        _pos(A0_63, ((22, 22, 22, 21), (14, 14, 14, 15))), set(), Q1, "selected", "K1", None,
        {"A0": {"q12_positive_complete": _kn(63, 160)}, "K1": {"q12_positive_complete": _kn(87, 160),
                                                               "q12_positive_last_ok": _kn(144, 160)}},
    ),
    "q1-63-to-86-fails": (
        _pos(A0_63, ((22, 22, 21, 21), (14, 14, 15, 15))), Q1, Q1, "no change", None, None,
        {"K1": {"q12_positive_complete": _kn(86, 160)}},
    ),
    # Q2: flight_booking protected (A0 27 >= 24); K 24 passes, 23 fails though Q1 passes (131 >= 57 + 24)
    "q2-protected-27-to-24-passes": (
        _pos(((27, 10, 10, 10), (9, 26, 26, 26)), ((24, 36, 36, 36), (12, 0, 0, 0))), set(), Q1, "selected", "K1",
        None, {"A0": {"q12_per_case_complete": {"flight_booking": _kn(27, 40)}}},
    ),
    "q2-protected-27-to-23-fails-with-q1-passing": (
        _pos(((27, 10, 10, 10), (9, 26, 26, 26)), ((23, 36, 36, 36), (13, 0, 0, 0))), {"Q2"}, Q1, "no change",
        None, None, {"K1": {"q12_per_case_complete": {"flight_booking": _kn(23, 40)},
                            "q12_positive_complete": _kn(131, 160)}},
    ),
    "q2-unprotected-23-to-0-holds": (
        _pos(((23, 10, 10, 10), (13, 26, 26, 26)), ((0, 36, 36, 36), (36, 0, 0, 0))), set(), Q1, "selected", "K1",
        None, {"K1": {"q12_per_case_complete": {"flight_booking": _kn(0, 40)}}},
    ),
    # adversarial: the protected set is A0's (the flights), not K's (primes, omelette); Q1 passes (80 >= 48 + 24)
    "q2-adv-protected-set-from-a0": (
        _pos(((24, 24, 0, 0), (12, 12, 36, 36)), ((0, 0, 40, 40), (36, 36, 0, 0))), {"Q2"}, Q1, "no change", None,
        None, {"K1": {"q12_positive_complete": _kn(80, 160)}},
    ),
    # U1: last-OK K >= A0 - 16 (149 - 16 = 133); 132 fails with shown still 133 (U2 holds)
    "u1-149-to-133-passes": (
        _pos((four(20), (18, 18, 17, 16)), (four(26), (8, 7, 7, 7))), set(), Q1, "selected", "K1", None,
        {"A0": {"q12_positive_last_ok": _kn(149, 160)}, "K1": {"q12_positive_last_ok": _kn(133, 160)}},
    ),
    "u1-149-to-132-fails": (
        _pos((four(20), (18, 18, 17, 16)), (four(26), four(7), (1, 0, 0, 0))), {"U1"}, Q1, "no change", None,
        None, {"K1": {"q12_positive_last_ok": _kn(132, 160), "q12_positive_shown": _kn(133, 160)}},
    ),
    # U1 per case: A0 38 protects flight_booking, K 23 there fails (total 143 >= 130; shown 40, so U2 holds)
    "u1-per-case-38-to-23-fails": (
        _pos((four(20), (18, 16, 16, 16)), ((23, 36, 36, 36), (0, 4, 4, 4), (17, 0, 0, 0))), {"U1"}, Q1,
        "no change", None, None,
        {"K1": {"q12_per_case_last_ok": {"flight_booking": _kn(23, 40)}, "q12_positive_last_ok": _kn(143, 160)}},
    ),
    # U2: shown K >= A0 - 16 (129 - 16 = 113); last-OK 112 vs A0 112 (U1 holds)
    "u2-129-to-113-passes": (
        _pos((four(20), four(8), (5, 4, 4, 4)), (four(26), four(2), (1, 0, 0, 0))), set(), Q1, "selected", "K1",
        None, {"A0": {"q12_positive_shown": _kn(129, 160), "q12_positive_last_ok": _kn(112, 160)},
               "K1": {"q12_positive_shown": _kn(113, 160)}},
    ),
    "u2-129-to-112-fails": (
        _pos((four(20), four(8), (5, 4, 4, 4)), (four(26), four(2))), {"U2"}, Q1, "no change", None, None,
        {"K1": {"q12_positive_shown": _kn(112, 160)}},
    ),
    # P9 verbatim
    "p9-no-belong-0-to-1of80-passes": (
        _put("K1", NO_BELONG_SLOTS[:1], WITH_SURFACE), set(), Q1, "selected", "K1", None,
        {"K1": {"surfaces_no_belong": _kn(1, 80)}},
    ),
    "p9-no-belong-0-to-2of80-fails": (
        _put("K1", NO_BELONG_SLOTS[:2], WITH_SURFACE), {"P9"}, Q1, "no change", None, None,
        {"K1": {"surfaces_no_belong": _kn(2, 80)}},
    ),
    "p9-tool-0-to-2of200-passes": (
        _put("K1", TOOL_SLOTS[:2], WITH_SURFACE), set(), Q1, "selected", "K1", None,
        {"K1": {"surfaces_tool": _kn(2, 200)}},
    ),
    "p9-tool-0-to-3of200-fails": (
        _put("K1", TOOL_SLOTS[:3], WITH_SURFACE), {"P9"}, Q1, "no change", None, None,
        {"K1": {"surfaces_tool": _kn(3, 200)}},
    ),
    # P10 (reading S): placement on positives <= A0 + 8
    "p10-2-to-10-passes": (
        _seq(_put("A0", PLACEMENT_SLOTS[:2], OPEN_SURFACE), _put("K1", PLACEMENT_SLOTS[:10], OPEN_SURFACE)),
        set(), Q1, "selected", "K1", None,
        {"A0": {"placement_failures_q12_positive": _kn(2, 160)},
         "K1": {"placement_failures_q12_positive": _kn(10, 160)}},
    ),
    "p10-2-to-11-fails": (
        _seq(_put("A0", PLACEMENT_SLOTS[:2], OPEN_SURFACE), _put("K1", PLACEMENT_SLOTS[:11], OPEN_SURFACE)),
        {"P10"}, Q1, "no change", None, None, {"K1": {"placement_failures_q12_positive": _kn(11, 160)}},
    ),
    # P11 (reading S): Q12 calls <= A0 + 10 of 175
    "p11-0-to-10of175-passes": (
        _put("K1", NEGATIVE_SLOTS[:10], Q12_TOOL), set(), Q1, "selected", "K1", None,
        {"K1": {"q12_tool_calls": _kn(10, 175)}, "A0": {"q12_tool_calls": _kn(0, 175)}},
    ),
    "p11-0-to-11of175-fails": (
        _put("K1", NEGATIVE_SLOTS[:11], Q12_TOOL), {"P11"}, Q1, "no change", None, None,
        {"K1": {"q12_tool_calls": _kn(11, 175)}},
    ),
    # P6 over all 440, here on an extra-repeat trial (K1's omelette r39 is speech only in the baseline)
    "p6-one-anomaly-fails": (
        _put("K1", [("omelette_steps", 39)], ANOMALY), {"P6"}, Q1, "no change", None, None, {},
    ),
    # P1/P2 dropped (D16): K1 silent on all 50 type_at trials still selects
    "p1-p2-k-speaks-0of50-type-at-no-effect": (
        _type_silent("K1"), set(), Q1, "selected", "K1", None,
        {"K1": {"tools.type_at.spoken": _kn(0, 50)}},
    ),
    # R3: A0 shown >= 112/160
    "r3-112-holds": (
        _pos((four(20), four(8)), K1_BASE), set(), Q1, "selected", "K1", None,
        {"A0": {"q12_positive_shown": _kn(112, 160)}},
    ),
    "r3-111-inconclusive-though-k1-passes": (
        _pos((four(20), (8, 8, 8, 7)), K1_BASE), set(), Q1, "inconclusive", None, "R3",
        {"A0": {"q12_positive_shown": _kn(111, 160)}},
    ),
    # R4: A0 complete <= 96/160 (every case protected at 24; K1 36 there)
    "r4-96-holds": (
        _pos((four(24), four(12)), K1_BASE), set(), Q1, "selected", "K1", None,
        {"A0": {"q12_positive_complete": _kn(96, 160)}},
    ),
    "r4-97-inconclusive-though-k1-passes": (
        _pos(((25, 24, 24, 24), (11, 12, 12, 12)), K1_BASE), set(), Q1, "inconclusive", None, "R4",
        {"A0": {"q12_positive_complete": _kn(97, 160)}},
    ),
    # void: >= 9 of 440 errored (over 300, 8 would already be > 2 %)
    "void-a0-8of440-not-void": (
        _put("A0", PLAIN_SLOTS[:8], ERROR_C), set(), Q1, "selected", "K1", None,
        {"A0": {"errored": _kn(8, 440)}},
    ),
    "void-a0-9of440-inconclusive": (
        _put("A0", PLAIN_SLOTS[:9], ERROR_C), set(), Q1, "inconclusive", None, "void",
        {"A0": {"errored": _kn(9, 440)}},
    ),
    "void-k1-9of440-inconclusive": (
        _put("K1", PLAIN_SLOTS[:9], ERROR_C), set(), Q1, "inconclusive", None, "void",
        {"K1": {"errored": _kn(9, 440)}},
    ),
    # both pass: the higher complete count wins; a tie goes to K1 (A0 64, so Q1 needs 88)
    "both-pass-k1-90-k2-95-k2-selected": (
        _pos(A0_64, K_90, ((24, 24, 24, 23), (12, 12, 12, 13))), set(), set(), "selected", "K2", None,
        {"K1": {"q12_positive_complete": _kn(90, 160)}, "K2": {"q12_positive_complete": _kn(95, 160)}},
    ),
    "both-pass-tie-k1-selected": (
        _pos(A0_64, K_90, K_90), set(), set(), "selected", "K1", None,
        {"K2": {"q12_positive_complete": _kn(90, 160)}},
    ),
    # adversarial: extra-repeat trials stay out of P5's 65 and P7's 200 (K1 r36-39 speech only in the baseline)
    "adv-q12-calls-at-r36-38-stay-out-of-p5": (
        _put("K1", [("primes_table", r) for r in (36, 37, 38)], Q12_TOOL), set(), Q1, "selected", "K1", None,
        {"K1": {"false_tools": _kn(0, 65), "q12_tool_calls": _kn(3, 175), "q12_positive_shown": _kn(144, 160)}},
    ),
    "adv-open-thoughts-at-r5-9-stay-out-of-p7": (
        _put("K1", [("flight_booking", r) for r in range(5, 10)], OPEN_SURFACE), set(), Q1, "selected", "K1",
        None, {"K1": {"placement_failures_q12_positive": _kn(5, 160), "thought_open": _kn(0, 200)}},
    ),
}  # fmt: skip


#: The rows run here; the rest run in ``tests/unit/test_t0108a_rule_boundaries_2.py`` (a third ``--dist loadfile``
#: worker). The union is every row, each once (asserted there).
FIRST_HALF = list(BOUNDARY_ROWS)[:16]


@pytest.mark.parametrize("row", FIRST_HALF)
def test_t0108a_rule_boundaries(tmp_path: Path, row: str) -> None:
    """The first half of criterion 6's rows; the check is ``check_boundary_row``."""
    check_boundary_row(tmp_path, row)


def check_boundary_row(tmp_path: Path, row: str) -> None:
    """Criterion 6, every row on hand-counted trial fixtures (the module docstring's baseline, one condition moved; K2
    copies A0's positives unless the row says otherwise, so it fails exactly Q1). K1's and K2's failing conditions are
    exactly the expected sets, the outcome has the expected form (a "no change" names every failing condition and
    nothing else; an inconclusive one its reason), ``winner`` is the selected K or no change, and the counts the row
    moves read as counted by hand."""
    edit, k1_fail, k2_fail, kind, winner, reason, premises = BOUNDARY_ROWS[row]
    report = report_108a(tmp_path, edit)
    for arm, keys in premises.items():
        for key, value in keys.items():
            got = _get_path(report["arms"][arm], key)
            if isinstance(value, Mapping) and key.startswith("q12_per_case"):
                got = {cid: got[cid] for cid in value}
            assert got == value, (row, arm, key, got)
    assert failing(report, "K1") == k1_fail, f"{row}: K1 fails {sorted(failing(report, 'K1'))}"
    assert failing(report, "K2") == k2_fail, f"{row}: K2 fails {sorted(failing(report, 'K2'))}"
    assert kind_of(report) == kind, f"{row}: outcome {report['outcome']!r}"
    if kind == "selected":
        assert (
            str(report["outcome"]).startswith(f"{winner} selected") and report["winner"] == winner
        )
    else:
        assert report["winner"] in (None, "no change"), report["winner"]
    if kind == "no change":
        assert named(report) == k1_fail | k2_fail, f"{row}: outcome {report['outcome']!r}"
    if reason is not None:
        assert reason in str(report["outcome"]), f"{row}: outcome {report['outcome']!r}"
    if reason == "void":
        arm = "A0" if row.startswith("void-a0") else "K1"
        assert list(report["void_arms"]) == [arm] and arm in str(report["outcome"])
    else:
        assert list(report["void_arms"]) == []
    assert report["readings"] == {"R3": reason != "R3", "R4": reason != "R4"}
