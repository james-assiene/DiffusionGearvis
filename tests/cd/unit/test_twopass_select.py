"""T-0080b criterion 5: ``twopass.select(rows, rule)`` -- the pre-registered selection rules of T-0080c, on synthetic rows.

Rules (T-0080c criteria 1 and 3, T-0080 Q7 tolerances, and the rulings at the end of docs/tasks/T-0080b.md), as
these rows read them:

``rule="stage1"`` (dev pilot). Candidates are the cells ``thought x named x perstep_mar`` (K-D2: ``union`` is not a
  stage-1 cell -- it may be reported, never chosen; ``draft_only`` is reported, never chosen in stage 1).
  Eligible = schema_valid rate 1 (over the cell's rows with ``called``). Keep exact >= best eligible - 1/10
  (exact ``Fraction`` arithmetic; exact over the cell's ``tool_case`` rows). Choose the lowest median
  ``pass2_wall_ms``; among cells whose median is <= 1.05 x that minimum (inclusive), fewer mean ``steps``; then the
  order T4, T2, T1.
``rule="stage2"`` (test). Candidates ``draft_only``, ``cd_always:<arm>``, ``cd_on_invalid:<arm>``. Eligible =
  schema_valid rate 1 and every constrained row (``cs`` not None) has ``cs`` True; ``draft_only`` is eligible only
  through its own schema_valid rate. Keep exact >= best eligible - 1/20. Choose the lowest p50 ``e2e_wall_ms`` over
  **tool cases** (no-tool rows are reported separately, never in the latency); among policies whose p50 is <= 1.05 x
  that minimum, the lower p90; then fewer mean ``steps``; then ``draft_only`` first, ``cd_on_invalid`` before
  ``cd_always``, and ``end_projection`` < ``perstep_mar`` < ``perstep_mar_qargmax``.

API (proposed by the test-author at 02df09d, confirmed by the orchestrator's rulings): ``select(rows, rule) -> str``,
the chosen policy name; ``ValueError`` when no policy is eligible or the rule is unknown. A row is a mapping
``{"policy", "case", "tool_case", "called", "schema_valid", "exact", "cs", "pass2_wall_ms", "e2e_wall_ms", "steps"}``;
stage-1 policy names are cells ``"<thought>/<scope>/<arm>"`` (e.g. ``"T4/named/perstep_mar"``) plus
``"draft_only"``. A row that made no call has ``schema_valid`` None.

Every wall list is built so that its median and p90 agree under every common percentile definition (nearest rank,
linear, exclusive, numpy lower/higher): the two middle values are equal and the top values around rank 0.9 n are
equal. Expected choices are worked out by hand in each docstring. CPU, pure; n per policy stated per row.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import pytest


def twopass() -> ModuleType:
    """The module under test; its absence fails the row as an assertion ("T-0080b not implemented")."""
    name = "local_astra.eval.cd.twopass"
    try:
        m = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise  # something the module imports is missing: a real error, not "not implemented"
        raise AssertionError(f"T-0080b not implemented: {name} does not exist") from None
    if not hasattr(m, "select"):
        raise AssertionError("T-0080b not implemented: twopass.select is missing")
    return m


def select(rows: Sequence[dict[str, Any]], rule: str) -> str:
    return str(twopass().select(rows, rule))


def policy(
    name: str,
    n: int,
    *,
    exact: int,
    valid: int | None = None,
    walls: Sequence[float] | float = 100.0,
    e2e: Sequence[float] | float | None = None,
    steps: float = 10.0,
    cs_bad: int = 0,
    constrained: bool = True,
) -> list[dict[str, Any]]:
    """``n`` tool-case rows of ``name``: the first ``exact`` exact, the first ``valid`` (default n) schema-valid,
    the last ``cs_bad`` with ``cs`` False. ``walls`` is pass-2 wall per row (scalar = constant); ``e2e`` defaults to
    1000 + pass-2 wall. ``constrained=False`` is a row with no pass 2 (``draft_only``, or ``cd_on_invalid`` on a valid
    draft): ``cs`` None, pass-2 wall None, 0 steps."""
    valid = n if valid is None else valid
    pw = [float(walls)] * n if isinstance(walls, int | float) else [float(w) for w in walls]
    assert len(pw) == n
    if e2e is None:
        ew = [1000.0 + w for w in pw]
    elif isinstance(e2e, int | float):
        ew = [float(e2e)] * n
    else:
        ew = [float(w) for w in e2e]
    assert len(ew) == n
    return [
        {
            "policy": name,
            "case": f"c{i:03d}",
            "tool_case": True,
            "called": True,
            "schema_valid": i < valid,
            "exact": i < exact and i < valid,
            "cs": (i < n - cs_bad) if constrained else None,
            "pass2_wall_ms": pw[i] if constrained else None,
            "e2e_wall_ms": ew[i],
            "steps": float(steps) if constrained else 0.0,
        }
        for i in range(n)
    ]


def no_tool(name: str, n: int, *, e2e: float) -> list[dict[str, Any]]:
    """``n`` no-tool rows (answered without a tag: no call, no pass 2)."""
    return [
        {
            "policy": name,
            "case": f"chat{i:03d}",
            "tool_case": False,
            "called": False,
            "schema_valid": None,
            "exact": False,
            "cs": None,
            "pass2_wall_ms": None,
            "e2e_wall_ms": float(e2e),
            "steps": 0.0,
        }
        for i in range(n)
    ]


def flat(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for g in groups for r in g]


N1 = 40  # dev tool cases
N2 = 80  # test tool cases
MAR = "perstep_mar"

# =========================================================================================== stage 1


def test_stage1_lowest_median_pass2_wall_among_eligible() -> None:
    """n=40 each. T4/named: exact 30/40, median 300. T2/named: exact 31/40, median 200. best = 31/40, floor 27/40:
    both kept; T2 has the lower median (300 > 1.05 x 200 = 210) -> T2/named."""
    rows = flat(
        policy(f"T4/named/{MAR}", N1, exact=30, walls=300),
        policy(f"T2/named/{MAR}", N1, exact=31, walls=200),
        policy("draft_only", N1, exact=20, constrained=False),
    )
    assert select(rows, "stage1") == f"T2/named/{MAR}"


def test_stage1_best_accuracy_with_an_invalid_row_is_never_chosen() -> None:
    """Adversarial: T1/named is the most accurate (39/40 exact) and the fastest (median 50) but one row is not
    schema-valid (39/40) -> ineligible. Of the rest, T4 (36/40, 120) and T2 (36/40, 110): 120 > 1.05 x 110 = 115.5
    -> T2 by median."""
    rows = flat(
        policy(f"T1/named/{MAR}", N1, exact=39, valid=39, walls=50),
        policy(f"T4/named/{MAR}", N1, exact=36, walls=120),
        policy(f"T2/named/{MAR}", N1, exact=36, walls=110),
    )
    assert select(rows, "stage1") == f"T2/named/{MAR}"


def test_stage1_an_ineligible_cell_does_not_raise_the_accuracy_bar() -> None:
    """ "best" is the best *eligible* cell. T1 (39/40 exact, 39/40 valid) is ineligible; best eligible is T4 at
    34/40, floor 30/40, so T2 (30/40, median 100) is kept and faster than T4 (median 300). Were the bar taken
    from T1 (39/40 - 4/40 = 35/40), both eligible cells would fail it."""
    rows = flat(
        policy(f"T1/named/{MAR}", N1, exact=39, valid=39, walls=500),
        policy(f"T4/named/{MAR}", N1, exact=34, walls=300),
        policy(f"T2/named/{MAR}", N1, exact=30, walls=100),
    )
    assert select(rows, "stage1") == f"T2/named/{MAR}"


def test_stage1_draft_only_is_never_the_stage1_choice() -> None:
    """``draft_only`` has no pass 2 and a perfect, most accurate draft here; stage 1 chooses a thought, so it is
    reported and never selected. Only cell: T4/named -> chosen."""
    rows = flat(
        policy("draft_only", N1, exact=40, constrained=False),
        policy(f"T4/named/{MAR}", N1, exact=35, walls=400),
    )
    assert select(rows, "stage1") == f"T4/named/{MAR}"


def test_stage1_union_is_reported_never_chosen() -> None:
    """K-D2: ``union`` is not a stage-1 cell. T2/union is valid, the most accurate and the fastest; T4/named
    (34/40, median 400) is chosen anyway. The union cell does not set the accuracy bar either: from it the floor
    would be 40/40 - 4/40 = 36/40 and T4/named would be dropped."""
    rows = flat(
        policy(f"T2/union/{MAR}", N1, exact=40, walls=50, steps=2),
        policy(f"T4/named/{MAR}", N1, exact=34, walls=400),
    )
    assert select(rows, "stage1") == f"T4/named/{MAR}"


def test_stage1_only_union_and_draft_only_is_no_choice() -> None:
    rows = flat(
        policy(f"T4/union/{MAR}", N1, exact=35, walls=100),
        policy("draft_only", N1, exact=30, constrained=False),
    )
    with pytest.raises(ValueError):
        select(rows, "stage1")


@pytest.mark.parametrize(
    ("fast_exact", "expected"),
    [
        (33, f"T2/named/{MAR}"),  # 33/40 = 37/40 - 1/10 exactly: kept, and it is faster
        (32, f"T4/named/{MAR}"),  # 32/40 < 33/40: dropped
    ],
)
def test_stage1_accuracy_tolerance_boundary_is_exact(fast_exact: int, expected: str) -> None:
    """Adversarial (Fraction, as T-0079): best = 37/40. The fast cell T2 (median 100) has ``fast_exact``/40; the
    accurate cell T4 has 37/40 and median 300. In float, ``33/40 >= 37/40 - 0.10`` is False (0.825 vs
    0.8250000000000001); exactly it is True. Both directions are pinned: at the boundary T2 is kept and wins on
    wall; one case below, T2 is dropped and T4 wins."""
    assert not (33 / 40 >= 37 / 40 - 0.10), (
        "premise: the float comparison is wrong at this boundary"
    )
    rows = flat(
        policy(f"T4/named/{MAR}", N1, exact=37, walls=300),
        policy(f"T2/named/{MAR}", N1, exact=fast_exact, walls=100),
    )
    assert select(rows, "stage1") == expected


@pytest.mark.parametrize(
    ("other_median", "expected"),
    [
        (105, f"T2/named/{MAR}"),  # 105 <= 1.05 x 100: within, fewer steps wins
        (106, f"T4/named/{MAR}"),  # 106 > 105: outside, the lowest median wins
    ],
)
def test_stage1_wall_tolerance_boundary_is_inclusive(other_median: int, expected: str) -> None:
    """Adversarial: T4 median 100 with 20 mean steps; T2 median ``other_median`` with 8 mean steps; equal accuracy.
    At exactly 1.05 x the minimum, T2 is within the tolerance and wins on steps; 1 ms above, it is not.
    (1.05 * 100 is 105.0 in float, so this boundary does not depend on the arithmetic.)"""
    rows = flat(
        policy(f"T4/named/{MAR}", N1, exact=30, walls=100, steps=20),
        policy(f"T2/named/{MAR}", N1, exact=30, walls=other_median, steps=8),
    )
    assert select(rows, "stage1") == expected


def test_stage1_median_not_mean() -> None:
    """T4 walls: 21 rows at 100 and 19 at 5000 -> median 100 (the middle pair both 100), mean 2422.5. T2: all 400
    (mean = median = 400). By median T4 wins (100 < 400); by mean T2 would."""
    rows = flat(
        policy(f"T4/named/{MAR}", N1, exact=30, walls=[100.0] * 21 + [5000.0] * 19),
        policy(f"T2/named/{MAR}", N1, exact=30, walls=400),
    )
    assert select(rows, "stage1") == f"T4/named/{MAR}"


def test_stage1_steps_decide_within_the_tolerance_not_the_lowest_median() -> None:
    """T4 median 100, 30 steps; T2 median 104, 10 steps; T1 median 103, 20 steps. All within 105: the fewest
    steps wins (T2), not the lowest median (T4) nor the thought order."""
    rows = flat(
        policy(f"T4/named/{MAR}", N1, exact=30, walls=100, steps=30),
        policy(f"T2/named/{MAR}", N1, exact=30, walls=104, steps=10),
        policy(f"T1/named/{MAR}", N1, exact=30, walls=103, steps=20),
    )
    assert select(rows, "stage1") == f"T2/named/{MAR}"


def test_stage1_full_tie_goes_t4_then_t2_then_t1() -> None:
    """Equal accuracy, medians and steps: the cheapest prompt first."""
    for present, expected in (
        (("T1", "T2", "T4"), "T4"),
        (("T4", "T1"), "T4"),
        (("T1", "T2"), "T2"),
        (("T2", "T1"), "T2"),
    ):
        rows = flat(*[policy(f"{t}/named/{MAR}", N1, exact=30, walls=200) for t in present])
        assert select(rows, "stage1") == f"{expected}/named/{MAR}", present


def test_stage1_no_eligible_cell_raises() -> None:
    rows = flat(
        policy(f"T4/named/{MAR}", N1, exact=30, valid=39),
        policy(f"T2/named/{MAR}", N1, exact=30, valid=38),
    )
    with pytest.raises(ValueError):
        select(rows, "stage1")


def _walls(p50: float, p90: float) -> list[float]:
    """80 walls whose sorted 40th/41st values are ``p50`` and whose 72nd-80th values are ``p90``."""
    return [p50 - 10] * 39 + [p50] * 2 + [(p50 + p90) / 2] * 30 + [p90] * 9


# =========================================================================================== stage 2


def test_stage2_lowest_e2e_p50() -> None:
    """n=80. cd_on_invalid:perstep_mar p50 e2e 1400 / p90 1500, cd_always:perstep_mar p50 1100 / p90 5000, both 70/80
    exact, 80/80 valid: 1400 > 1.05 x 1100 -> cd_always:perstep_mar. The winner loses every later key (higher p90,
    and the policy order puts cd_on_invalid first), so only p50 can choose it."""
    rows = flat(
        policy("cd_on_invalid:perstep_mar", N2, exact=70, e2e=_walls(1400, 1500)),
        policy("cd_always:perstep_mar", N2, exact=70, e2e=_walls(1100, 5000)),
    )
    assert select(rows, "stage2") == "cd_always:perstep_mar"


def test_stage2_cs_violation_is_ineligible() -> None:
    """Adversarial: cd_always:end_projection is the fastest (p50 1050) and the most accurate, but one constrained
    row has cs False -> ineligible; cd_always:perstep_mar (p50 1300) is chosen."""
    rows = flat(
        policy("cd_always:end_projection", N2, exact=78, e2e=1050, cs_bad=1),
        policy("cd_always:perstep_mar", N2, exact=75, e2e=1300),
    )
    assert select(rows, "stage2") == "cd_always:perstep_mar"


@pytest.mark.parametrize(("valid", "expected"), [(80, "draft_only"), (79, "cd_always:perstep_mar")])
def test_stage2_draft_only_eligible_only_on_its_own_schema_rate(valid: int, expected: str) -> None:
    """``draft_only`` (p50 e2e 1000, no constrained row) is chosen when all 80 drafts are schema-valid and never when
    one is not -- a fast, accurate draft_only with 79/80 valid must not win."""
    rows = flat(
        policy("draft_only", N2, exact=70, valid=valid, e2e=1000, constrained=False),
        policy("cd_always:perstep_mar", N2, exact=70, e2e=1500),
    )
    assert select(rows, "stage2") == expected


def test_stage2_best_is_taken_among_eligible_only() -> None:
    """An ineligible policy (79/80 valid) with 78/80 exact must not raise the bar: best eligible = 70/80, floor 66/80;
    cd_on_invalid:perstep_mar (67/80, p50 1100) is kept and faster than cd_always:perstep_mar (70/80, p50 1500).
    Were the bar 78/80 - 4/80 = 74/80, neither eligible policy would pass."""
    rows = flat(
        policy("cd_always:end_projection", N2, exact=78, valid=79, e2e=900),
        policy("cd_always:perstep_mar", N2, exact=70, e2e=1500),
        policy("cd_on_invalid:perstep_mar", N2, exact=67, e2e=1100),
    )
    assert select(rows, "stage2") == "cd_on_invalid:perstep_mar"


@pytest.mark.parametrize(
    ("fast_exact", "expected"),
    [(33, "cd_on_invalid:perstep_mar"), (32, "cd_always:perstep_mar")],
)
def test_stage2_accuracy_tolerance_boundary_is_exact(fast_exact: int, expected: str) -> None:
    """Adversarial (Fraction): n=80, best 37/80, tolerance 1/20 = 4/80 -> floor 33/80. Float says
    ``33/80 >= 37/80 - 0.05`` is False; exactly it is True. The fast policy (p50 1000) is kept at 33/80, dropped
    at 32/80."""
    assert not (33 / 80 >= 37 / 80 - 0.05), (
        "premise: the float comparison is wrong at this boundary"
    )
    rows = flat(
        policy("cd_always:perstep_mar", N2, exact=37, e2e=2000),
        policy("cd_on_invalid:perstep_mar", N2, exact=fast_exact, e2e=1000),
    )
    assert select(rows, "stage2") == expected


@pytest.mark.parametrize(("other_p50", "expected"), [(1050, "B"), (1051, "A")])
def test_stage2_within_five_percent_the_lower_p90_wins(other_p50: int, expected: str) -> None:
    """A = cd_always:end_projection: p50 1000, p90 3000. B = cd_always:perstep_mar: p50 ``other_p50``, p90 1200; equal
    accuracy and steps. At 1050 (= 1.05 x 1000, inclusive) B is within and has the lower p90; at 1051 it is outside
    and A's lower p50 wins. B loses every later key (equal steps; end_projection precedes perstep_mar in the policy
    order), so at 1050 only p90 can choose it (review of 528d385: the earlier row let the policy order decide)."""
    names = {"A": "cd_always:end_projection", "B": "cd_always:perstep_mar"}
    rows = flat(
        policy(names["A"], N2, exact=70, e2e=_walls(1000, 3000)),
        policy(names["B"], N2, exact=70, e2e=_walls(other_p50, 1200)),
    )
    assert select(rows, "stage2") == names[expected]


def test_stage2_latency_is_over_tool_cases_only() -> None:
    """Ruling: end-to-end p50/p90 are computed over tool cases; no-tool cases are reported separately.
    A = cd_always:perstep_mar: 80 tool rows, 41 at 1000 and 39 at 5000 (p50 1000), plus 12 no-tool rows at 50000.
    B = cd_on_invalid:perstep_mar: every row (80 tool + 12 no-tool) at 1200. Over tool cases A wins (1200 > 1050).
    Were the no-tool rows counted, A's p50 over 92 rows would be 5000 (the 46th/47th values) and B would win."""
    rows = flat(
        policy("cd_always:perstep_mar", N2, exact=70, e2e=[1000.0] * 41 + [5000.0] * 39),
        no_tool("cd_always:perstep_mar", 12, e2e=50000.0),
        policy("cd_on_invalid:perstep_mar", N2, exact=70, e2e=1200),
        no_tool("cd_on_invalid:perstep_mar", 12, e2e=1200.0),
    )
    assert select(rows, "stage2") == "cd_always:perstep_mar"


def test_stage2_no_call_rows_do_not_break_the_schema_rate() -> None:
    """A no-tool case answered without a tag makes no call: it neither counts against schema_valid nor makes a
    policy ineligible. Both policies carry 12 such rows; the faster (cd_on_invalid, 1100) is chosen."""
    rows = flat(
        policy("cd_always:perstep_mar", N2, exact=70, e2e=1400),
        no_tool("cd_always:perstep_mar", 12, e2e=900.0),
        policy("cd_on_invalid:perstep_mar", N2, exact=70, e2e=1100),
        no_tool("cd_on_invalid:perstep_mar", 12, e2e=900.0),
    )
    assert select(rows, "stage2") == "cd_on_invalid:perstep_mar"


def test_stage2_steps_break_a_wall_tie() -> None:
    """Equal accuracy and walls: fewer mean steps wins (12 < 30), against the policy order."""
    rows = flat(
        policy("cd_always:perstep_mar", N2, exact=70, e2e=1200, steps=12),
        policy("cd_always:end_projection", N2, exact=70, e2e=1200, steps=30),
    )
    assert select(rows, "stage2") == "cd_always:perstep_mar"


def test_stage2_draft_only_comes_first_in_a_full_tie() -> None:
    """Ruling: in the final tie order ``draft_only`` comes first (cheapest). With every draft valid,
    ``cd_on_invalid:<arm>`` never sends pass 2, so its rows equal draft_only's (same walls, 0 steps): the natural tie.
    draft_only wins it, also with a cd_always policy at the same walls and 0 steps."""
    rows = flat(
        policy("cd_on_invalid:end_projection", N2, exact=70, e2e=1000, constrained=False),
        policy("draft_only", N2, exact=70, e2e=1000, constrained=False),
    )
    assert select(rows, "stage2") == "draft_only"
    rows = flat(
        policy("cd_always:end_projection", N2, exact=70, e2e=1000, steps=0),
        policy("cd_on_invalid:perstep_mar", N2, exact=70, e2e=1000, constrained=False),
        policy("draft_only", N2, exact=70, e2e=1000, constrained=False),
    )
    assert select(rows, "stage2") == "draft_only"


def test_stage2_policy_order_breaks_the_remaining_ties() -> None:
    """Equal accuracy, walls and steps: ``cd_on_invalid`` before ``cd_always``, then end_projection < perstep_mar <
    perstep_mar_qargmax. Removing the winner each time walks the whole order (inputs listed in reverse)."""
    expected_order = [
        "cd_on_invalid:end_projection",
        "cd_on_invalid:perstep_mar",
        "cd_on_invalid:perstep_mar_qargmax",
        "cd_always:end_projection",
        "cd_always:perstep_mar",
        "cd_always:perstep_mar_qargmax",
    ]
    remaining = list(reversed(expected_order))
    for want in expected_order:
        rows = flat(*[policy(p, N2, exact=70, e2e=1200, steps=10) for p in remaining])
        assert select(rows, "stage2") == want, remaining
        remaining.remove(want)


def test_stage2_no_eligible_policy_raises() -> None:
    rows = flat(
        policy("draft_only", N2, exact=70, valid=70, constrained=False),
        policy("cd_always:perstep_mar", N2, exact=70, cs_bad=2),
    )
    with pytest.raises(ValueError):
        select(rows, "stage2")


def test_unknown_rule_raises() -> None:
    rows = policy("cd_always:perstep_mar", N2, exact=70)
    with pytest.raises(ValueError):
        select(rows, "stage3")


# =========================================================================================== T-0081b criterion 8
# The seed arm ``perstep_mar+seed`` through the existing rules (negative: no tolerance, ordering or ``_STAGE1`` change).
# These rows hold before T-0081b by construction; they pin that the seed arm is handled by the existing code.

SEED = "perstep_mar+seed"
QARGMAX = "perstep_mar_qargmax"


def test_stage2_exact_tie_qargmax_before_the_seed_arm() -> None:
    """Criterion 8: an exact tie (accuracy, walls, steps) between ``cd_always:T4/named/perstep_mar_qargmax`` and
    ``cd_always:T4/named/perstep_mar+seed`` chooses qargmax: the seed arm ranks after it through the unlisted-arm
    fallback. Inputs in both orders (the winner is not the first or last listed)."""
    a = f"cd_always:T4/named/{QARGMAX}"
    b = f"cd_always:T4/named/{SEED}"
    for names in ((a, b), (b, a)):
        rows = flat(*[policy(p, N2, exact=70, e2e=1200, steps=10) for p in names])
        assert select(rows, "stage2") == a, names


def test_stage2_full_tie_walks_the_order_with_the_seed_arm_last_per_kind() -> None:
    """Criterion 8: the whole stage-2 tie order with the seed arm: ``cd_on_invalid`` before ``cd_always`` (kind first),
    and within a kind end_projection < perstep_mar < perstep_mar_qargmax < perstep_mar+seed. So
    ``cd_on_invalid:…+seed`` precedes ``cd_always:…end_projection``. Removing each winner walks it (inputs reversed)."""
    arms = ("end_projection", "perstep_mar", QARGMAX, SEED)
    expected_order = [f"{k}:T4/named/{a}" for k in ("cd_on_invalid", "cd_always") for a in arms]
    remaining = list(reversed(expected_order))
    for want in expected_order:
        rows = flat(*[policy(p, N2, exact=70, e2e=1200, steps=10) for p in remaining])
        assert select(rows, "stage2") == want, remaining
        remaining.remove(want)


def test_stage2_the_seed_arm_wins_when_it_is_faster() -> None:
    """The other side of the tie row (the seed arm is a candidate, not excluded): ``cd_always:…+seed`` with p50 e2e
    1000 against qargmax's 1500, equal accuracy -> the seed arm (1500 > 1.05 x 1000)."""
    rows = flat(
        policy(f"cd_always:T4/named/{QARGMAX}", N2, exact=70, e2e=1500),
        policy(f"cd_always:T4/named/{SEED}", N2, exact=70, e2e=1000),
    )
    assert select(rows, "stage2") == f"cd_always:T4/named/{SEED}"


def test_stage2_cd_on_invalid_seed_policy_is_a_candidate() -> None:
    """Criterion 8: ``cd_on_invalid:…+seed`` is handled like any ``cd_on_invalid`` policy: fastest p50 wins."""
    rows = flat(
        policy(f"cd_always:T4/named/{SEED}", N2, exact=70, e2e=1500),
        policy(f"cd_on_invalid:T4/named/{SEED}", N2, exact=70, e2e=1100),
        policy("draft_only", N2, exact=70, valid=79, e2e=1000, constrained=False),
    )
    assert select(rows, "stage2") == f"cd_on_invalid:T4/named/{SEED}"


def test_stage1_ignores_a_seed_cell_in_the_rows() -> None:
    """Criterion 7/8: ``select("stage1")`` candidates are unchanged -- a ``T4/named/perstep_mar+seed`` cell that is the
    most accurate and fastest is never chosen and does not raise the accuracy bar: from it (40/40) the floor would be
    36/40 and T2/named (33/40) would be dropped."""
    rows = flat(
        policy(f"T4/named/{SEED}", N1, exact=40, walls=50, steps=2),
        policy(f"T2/named/{MAR}", N1, exact=33, walls=300),
    )
    assert select(rows, "stage1") == f"T2/named/{MAR}"


# =========================================================================================== T-0082b criterion 5
# ``select("stage2")`` on a replay artifact (pass-1 rows carrying ``replayed_from``, a regime naming the source) gives
# the same choice as on the equivalent live rows: no code path specific to replay. Negative pins: they hold before
# T-0082b by construction (``select``'s rule, tolerances and order are unchanged, criterion 6) and guard the diff.

EP_T4 = "T4/named/end_projection"
MAR_T4 = f"T4/named/{MAR}"
REPLAY_SHA = "ab" * 32


def _raw_rows(replayed: bool) -> list[dict[str, Any]]:
    """Five tool cases; pass-1 wall 100 each; drafts t01-t03 valid and exact, t04-t05 unparseable. Pass 2 on every
    case for both cells: valid, exact, CS 1, 5 steps; wall 50 (end_projection) and 30 (perstep_mar).

    Hand derivation: ``draft_only`` schema 3/5 -> ineligible. e2e: cd_always end_projection 150 x5, perstep_mar 130 x5;
    cd_on_invalid end_projection [100, 100, 100, 150, 150] (p50 100, p90 150), perstep_mar [100, 100, 100, 130, 130]
    (p50 100, p90 130). All CD policies exact 5/5. Min p50 100; within 1.05 x: both cd_on_invalid; lower p90 ->
    ``cd_on_invalid:T4/named/perstep_mar``. (Middle and top values repeat, so every percentile definition agrees.)"""
    rows: list[dict[str, Any]] = []
    for i in range(5):
        case = f"t{i + 1:02d}"
        valid = i < 3
        call = {"name": "scroll", "arguments": {"direction": "up"}} if valid else None
        p1: dict[str, Any] = {
            "pass": 1,
            "case": case,
            "kind": "tool",
            "tool": "scroll",
            "error": None,
            "wall_ms": 100.0,
            "prefix_hit_tokens": 10,
            "prefix_query_tokens": 20,
            "route": {"kind": "toolcall", "tag": "cc", "draft_name": "scroll", "spoken_prefix": ""},
            "draft_close": "",
            "draft_parse": "ok" if valid else "not JSON",
            "draft_call": call,
            "draft_schema_valid": valid,
            "draft_exact": valid,
        }
        if replayed:
            p1["replayed_from"] = REPLAY_SHA
        rows.append(p1)
        for cell, wall in ((EP_T4, 50.0), (MAR_T4, 30.0)):
            rows.append(
                {
                    "pass": 2,
                    "case": case,
                    "cell": cell,
                    "requests": 1,
                    "parse": "ok",
                    "call": {"name": "scroll", "arguments": {"direction": "up"}},
                    "schema_valid": True,
                    "exact": True,
                    "cs": True,
                    "error": None,
                    "wall_ms": wall,
                    "steps": [5],
                    "prefix_hit_tokens": None,
                    "prefix_query_tokens": None,
                    "close_101_pos": None,
                }
            )
    return rows


def _derived(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tp = twopass()
    names = [
        "draft_only",
        *(f"{k}:{c}" for k in ("cd_always", "cd_on_invalid") for c in (EP_T4, MAR_T4)),
    ]
    return [d for p in names for d in tp.derive(rows, p)]


def test_stage2_on_a_replay_artifact_matches_the_live_rows() -> None:
    """T-0082b criterion 5: ``derive`` + ``select("stage2")`` on the replay rows = on the live rows = the hand choice
    (``_raw_rows`` docstring)."""
    want = f"cd_on_invalid:{MAR_T4}"
    assert select(_derived(_raw_rows(replayed=False)), "stage2") == want
    assert select(_derived(_raw_rows(replayed=True)), "stage2") == want


def test_report_selection_on_a_replay_artifact_matches_the_live_artifact(tmp_path: Any) -> None:
    """T-0082b criterion 5 through ``report``: the same rows written as a test-stage artifact, once live and once as a
    replay (regime naming ``pass1_from``/``pass1_sha256``), give the same stage-2 selection, the hand choice."""
    import json

    tp = twopass()
    sel = {}
    for name, replayed in (("live", False), ("replay", True)):
        regime: dict[str, Any] = {
            "kind": "regime",
            "task": "toolcall",
            "stage": "test",
            "cells": [EP_T4, MAR_T4],
        }
        if replayed:
            regime.update(pass1_from="logs/src.jsonl", pass1_sha256=REPLAY_SHA)
        lines = [regime, *_raw_rows(replayed), {"kind": "trailer", "stopped": None}]
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
        sel[name] = tp.report([path])["selection"]
    assert sel["live"] == sel["replay"] == {"rule": "stage2", "choice": f"cd_on_invalid:{MAR_T4}"}


# =========================================================================================== T-0084d criterion 2
# Negative: ``select(rows, "stage1"/"stage2")``, its tolerances, orders and ``_STAGE1`` are unchanged by the format
# comparison, and on T-0080c's stored artifacts it returns the recorded choices (docs/tasks/T-0080c.md: stage 1
# ``T1/named/perstep_mar``, stage 2 ``cd_on_invalid:T1/named/perstep_mar+seed``). Green before the implementation by
# construction; they guard the diff.


def _stored_policy_rows(stage: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(stage-1 rows, stage-2 rows) derived from the stored T-0080c artifact: stage 1 = ``draft_only`` plus each cell's
    ``cd_always`` rows named by the bare cell (the stage-1 candidates, T-0080c criterion 1); stage 2 = ``draft_only``
    and both kinds over every cell."""
    import json as _json

    from tests.cd import _native_artifacts as na

    tp = twopass()
    path = na.stored(f"t0080c-{stage}")
    lines = [_json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    cells = lines[0]["cells"]
    rows = [x for x in lines[1:] if x.get("pass") in (1, 2)]
    s1 = list(tp.derive(rows, "draft_only"))
    s2 = list(s1)
    for c in cells:
        s1 += [{**r, "policy": c} for r in tp.derive(rows, f"cd_always:{c}")]
        s2 += list(tp.derive(rows, f"cd_always:{c}")) + list(tp.derive(rows, f"cd_on_invalid:{c}"))
    return s1, s2


def test_t0084d_select_reproduces_t0080cs_recorded_choices() -> None:
    """Criterion 2 on golden rows: stage 1 on the dev artifact -> ``T1/named/perstep_mar`` (also the stored
    ``selection-dev.json``'s choice); stage 2 on the test artifact -> ``cd_on_invalid:T1/named/perstep_mar+seed``."""
    s1, _ = _stored_policy_rows("dev")
    assert select(s1, "stage1") == "T1/named/perstep_mar"
    _, s2 = _stored_policy_rows("test")
    assert select(s2, "stage2") == "cd_on_invalid:T1/named/perstep_mar+seed"


#: The select constants of T-0084d criterion 2, compared by value (``_STAGE1`` by its pattern).
SELECT_CONSTANTS = (
    "STAGE1_TOLERANCE",
    "STAGE2_TOLERANCE",
    "WALL_TOLERANCE",
    "_THOUGHT_ORDER",
    "_KIND_ORDER",
    "_ARM_ORDER",
)


def _select_constant_diffs(
    base: Any, tip: Any, names: Sequence[str] = SELECT_CONSTANTS
) -> list[str]:
    """Criterion 2's predicate over two ``twopass`` modules: the names whose values differ (``_STAGE1`` by pattern);
    a name absent on either side counts as a difference."""
    out = [
        n
        for n in names
        if not (hasattr(base, n) and hasattr(tip, n)) or getattr(tip, n) != getattr(base, n)
    ]
    if tip._STAGE1.pattern != base._STAGE1.pattern:
        out.append("_STAGE1")
    return out


def test_t0084d_select_constants_are_main_s() -> None:
    """Criterion 2: the tolerances (1/10, 1/20), the 5 % wall tolerance, the thought/kind/arm orders and the
    ``_STAGE1`` pattern of T-0084d's TIP blob (9abdb3b) equal those of ``main`` at the fork point (its BASE 9e84f21;
    T-0094b: the task's own two blobs, never BASE against the live module). The tolerances' hand values are asserted
    on the current module."""
    from fractions import Fraction

    from tests import _git_range as gr
    from tests.cd import _native_artifacts as na

    gr.prove_range(na.T0084D_BASE, na.T0084D_TIP, na.T0084D_MERGE)
    base_rev, tip_rev = na.T0084D_BASE, na.T0084D_TIP
    tp, base, tip = twopass(), na.main_twopass(base_rev), na.main_twopass(tip_rev)
    blob = {
        r: gr._git("show", f"{r}:local_astra/eval/cd/twopass.py").stdout
        for r in (base_rev, tip_rev)
    }
    assert blob[base_rev] != blob[tip_rev], "premise: two blobs"
    assert _select_constant_diffs(base, tip) == []
    hand = (Fraction(1, 10), Fraction(1, 20), Fraction(105, 100))
    assert hand == (tp.STAGE1_TOLERANCE, tp.STAGE2_TOLERANCE, tp.WALL_TOLERANCE)


def test_the_select_constants_predicate_has_power() -> None:
    """T-0094b criterion 5 (adversarial): no revision of ``twopass.py`` in history changes a select constant, so the
    same predicate runs on a copy of the TIP module with ``STAGE1_TOLERANCE`` and the ``_STAGE1`` pattern changed (a
    scratch stand-in, ruling P2) and must name both; a predicate that is always empty fails."""
    import re
    import types
    from fractions import Fraction

    from tests.cd import _native_artifacts as na

    tip = na.main_twopass(na.T0084D_TIP)
    edited = types.SimpleNamespace(**{n: getattr(tip, n) for n in SELECT_CONSTANTS})
    edited.STAGE1_TOLERANCE = Fraction(1, 5)
    edited._STAGE1 = re.compile(tip._STAGE1.pattern + "x")
    assert _select_constant_diffs(tip, edited) == ["STAGE1_TOLERANCE", "_STAGE1"]


def test_the_select_constants_row_compares_two_different_blobs() -> None:
    """T-0094b too-strict twin: with ``FORMAT_RULE`` (T-0084d's own addition) added to the compared names, the predicate
    over T-0084d's pair reports it; a pair reduced to TIP vs TIP fails here."""
    from tests import _git_range as gr
    from tests.cd import _native_artifacts as na

    gr.prove_range(na.T0084D_BASE, na.T0084D_TIP, na.T0084D_MERGE)
    base, tip = na.main_twopass(), na.main_twopass(na.T0084D_TIP)
    names = (*SELECT_CONSTANTS, "FORMAT_RULE")
    assert _select_constant_diffs(base, tip, names) == ["FORMAT_RULE"]


def test_t0084d_select_treats_native_rows_as_main_does() -> None:
    """Criterion 2: ``select`` itself is unchanged on rows carrying native policies -- it gives ``main``'s answer, worked
    out by hand from main's rule. Stage 2 (n=80): ``draft_only:native`` (80/80, e2e 100) is not a stage-2 candidate
    (only ``draft_only`` and the ``cd_*`` prefixes are), else it would win; ``cd_on_invalid:native/T4/named/perstep_mar``
    (80/80, e2e 500) is a ``cd_on_invalid`` candidate by prefix and wins (the cc policy, 70/80, is below the 76/80 bar).
    Stage 1: a native perstep_mar cell does not match ``_STAGE1`` -> T2/named/perstep_mar."""
    from tests.cd import _native_artifacts as na

    nat = "native/T4/named/perstep_mar"
    rows = flat(
        policy("cd_always:T4/named/perstep_mar", N2, exact=70, e2e=2000),
        policy("draft_only:native", N2, exact=80, e2e=100, constrained=False),
        policy(f"cd_on_invalid:{nat}", N2, exact=80, e2e=500),
    )
    assert select(rows, "stage2") == f"cd_on_invalid:{nat}"
    assert na.main_twopass().select(rows, "stage2") == f"cd_on_invalid:{nat}"
    rows = flat(
        policy(f"T2/named/{MAR}", N1, exact=30, walls=300),
        policy(nat, N1, exact=40, walls=10),
    )
    assert select(rows, "stage1") == f"T2/named/{MAR}"
    assert na.main_twopass().select(rows, "stage1") == f"T2/named/{MAR}"
