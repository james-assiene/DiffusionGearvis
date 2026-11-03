"""T-0084d criterion 1: ``twopass.select_format(rows, stage, native_thought=None) -> FormatChoice``, on synthetic
policy rows (hand-computed; CPU, pure).

The rule (docs/tasks/T-0084d.md criterion 1; QN2 of T-0084.md; the T-0080c rule of docs/tasks/T-0080c.md):

- **Step 1 (dev only).** Over ``cd_always:native/<T1|T2|T4>/named/perstep_mar``: eligible = schema_valid 1.000 over
  called rows; keep exact >= best - 1/10; the lowest median pass-2 wall; within 5 %, fewer mean steps; then T4, T2,
  T1. The result is θ_n.
- **Step 2.** Family ``draft_only``, ``draft_only:native`` and ``cd_always:``/``cd_on_invalid:`` over cc
  ``T1/named/{4 arms}`` and native ``native/θ_n/named/{4 arms}``. Eligible = schema 1.000 and CS 1.000 on every
  constrained row. Keep exact >= best eligible - tol (1/10 dev, 1/20 test). The lowest e2e p50 over tool cases; within
  5 %, the lower p90; then fewer mean steps; then kind order; then arm order (seed arm last); then cc before native.
- ``winner`` is the chosen policy's format. The test stage takes θ_n as given (from the dev selection file) and runs
  step 2 only.

API pins (test-author proposal, for the orchestrator's ruling; the task names the function and the four fields):

- ``rows`` are **policy rows** as ``select`` takes them (``derive`` output: ``policy``, ``case``, ``tool_case``,
  ``called``, ``schema_valid``, ``exact``, ``cs``, ``pass2_wall_ms``, ``e2e_wall_ms``, ``steps``), with full policy
  names: ``draft_only``, ``draft_only:native``, ``cd_always:<cell>``, ``cd_on_invalid:<cell>``.
- ``stage`` is ``"dev"`` or ``"test"``; ``native_thought`` (keyword) is required at ``test`` and gives θ_n.
- ``FormatChoice`` has attributes ``native_thought`` (``"T1"|"T2"|"T4"``), ``choice`` (a policy name), ``winner``
  (``"cc"|"native"``) and ``tables`` (a mapping; only its mention of the candidates is pinned).
- ``ValueError``: an unknown stage, ``test`` without ``native_thought``, no eligible policy, or a θ_n whose seed cell
  (``native/θ_n/named/perstep_mar+seed``) has no rows.

Every wall list is constant per policy (median = p90 = mean under every definition) unless a docstring says so.
Default rows (``family``) are slow (e2e 9000) and exact 0/n so that the overridden policies decide; ``draft_only`` and
``draft_only:native`` default to schema 0/n (ineligible). n per policy is stated per row.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from fractions import Fraction
from types import ModuleType
from typing import Any

import pytest

ARMS = ("end_projection", "perstep_mar", "perstep_mar_qargmax", "perstep_mar+seed")
KINDS = ("cd_on_invalid", "cd_always")
N_DEV = 40
N_TEST = 80


def twopass() -> ModuleType:
    m = importlib.import_module("local_astra.eval.cd.twopass")
    if not hasattr(m, "select_format"):
        raise AssertionError("T-0084d not implemented: twopass.select_format is missing")
    return m


def select_format(rows: Sequence[Mapping[str, Any]], stage: str, **kw: Any) -> Any:
    return twopass().select_format(list(rows), stage, **kw)


def policy(
    name: str,
    n: int,
    *,
    exact: int = 0,
    valid: int | None = None,
    walls: float = 100.0,
    e2e: Sequence[float] | float | None = None,
    steps: float = 10.0,
    cs_bad: int = 0,
    constrained: bool = True,
) -> list[dict[str, Any]]:
    """``n`` tool-case rows of ``name``: the first ``exact`` exact (and valid), the first ``valid`` (default n)
    schema-valid, the last ``cs_bad`` with ``cs`` False. ``e2e`` defaults to 1000 + pass-2 wall."""
    valid = n if valid is None else valid
    if e2e is None:
        ew = [1000.0 + walls] * n
    elif isinstance(e2e, int | float):
        ew = [float(e2e)] * n
    else:
        ew = [float(w) for w in e2e]
    assert len(ew) == n
    return [
        {
            "policy": name,
            "case": f"c{i:03d}",
            "tool": "click_at",
            "tool_case": True,
            "called": True,
            "schema_valid": i < valid,
            "exact": i < exact and i < valid,
            "parsed": i < valid,
            "cs": (i < n - cs_bad) if constrained else None,
            "pass2_wall_ms": float(walls) if constrained else None,
            "e2e_wall_ms": ew[i],
            "steps": float(steps) if constrained else 0.0,
            "pass2_sent": constrained,
            "missing": False,
        }
        for i in range(n)
    ]


def family(
    n: int = N_DEV,
    *,
    thoughts: Sequence[str] = ("T1", "T2", "T4"),
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    drop: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Every policy a dev artifact derives: ``draft_only``, ``draft_only:native``, and both kinds over cc
    ``T1/named/{4}`` and native ``<T>/named/{4}`` for each of ``thoughts``. Defaults: exact 0, e2e 9000, pass-2 wall
    100, 10 steps; the draft-only policies schema 0/n. ``overrides`` replaces a policy's keyword arguments."""
    overrides = dict(overrides or {})
    names = ["draft_only", "draft_only:native"]
    names += [f"{k}:T1/named/{a}" for k in KINDS for a in ARMS]
    names += [f"{k}:native/{t}/named/{a}" for t in thoughts for k in KINDS for a in ARMS]
    for o in overrides:
        assert o in names, f"fixture: {o} is not a family policy"
    rows: list[dict[str, Any]] = []
    for name in names:
        if name in drop:
            continue
        if name.startswith("draft_only"):
            kw: dict[str, Any] = {"valid": 0, "e2e": 9000.0, "constrained": False}
        else:
            kw = {"e2e": 9000.0}
        kw.update(overrides.get(name, {}))
        rows += policy(name, n, **kw)
    return rows


def _choice(fc: Any) -> tuple[str, str, str]:
    return (str(fc.native_thought), str(fc.choice), str(fc.winner))


S1 = "cd_always:native/{}/named/perstep_mar"
NAT_EP = "cd_on_invalid:native/{}/named/end_projection"
CC_EP = "cd_on_invalid:T1/named/end_projection"


# =========================================================================== step 1


def test_step1_picks_theta_n_by_the_t0080c_stage1_rule() -> None:
    """n=40. Step-1 cells: native T1 exact 30, median pass-2 300; T2 exact 31, median 200; T4 exact 20, median 50.
    best 31/40, bar 27/40: T4 is dropped (20 < 27) though fastest; T1 and T2 kept; T2 has the lower median
    (300 > 1.05 x 200) -> θ_n = T2. Step 2 then runs over native T2 only: ``cd_on_invalid:native/T2/named/
    end_projection`` (exact 37, e2e 1000) wins; best eligible 37/40, bar 33/40 drops T2's perstep_mar (31). The
    native T4 ``end_projection`` policy (exact 40, e2e 500) would win -- and raise the bar -- were it in the family;
    it is not (θ_n = T2)."""
    rows = family(
        overrides={
            S1.format("T1"): {"exact": 30, "walls": 300},
            S1.format("T2"): {"exact": 31, "walls": 200},
            S1.format("T4"): {"exact": 20, "walls": 50},
            NAT_EP.format("T2"): {"exact": 37, "e2e": 1000.0},
            NAT_EP.format("T4"): {"exact": 40, "e2e": 500.0},
        }
    )
    assert _choice(select_format(rows, "dev")) == ("T2", NAT_EP.format("T2"), "native")


@pytest.mark.parametrize(("fast_exact", "theta"), [(33, "T2"), (32, "T4")])
def test_step1_accuracy_tolerance_boundary_is_exact(fast_exact: int, theta: str) -> None:
    """Adversarial (Fraction): native T4 perstep_mar exact 37/40, median 300; T2 ``fast_exact``/40, median 100. Float
    says ``33/40 >= 37/40 - 0.10`` is False; exactly it is True: T2 is kept and wins on wall; at 32/40 it is dropped
    and T4 is θ_n. (The rest of the family is default; θ_n is all this row reads.)"""
    assert not (33 / 40 >= 37 / 40 - 0.10), "premise: float is wrong at this boundary"
    rows = family(
        overrides={
            S1.format("T4"): {"exact": 37, "walls": 300},
            S1.format("T2"): {"exact": fast_exact, "walls": 100},
        }
    )
    assert select_format(rows, "dev").native_thought == theta


def test_step1_ties_go_t4_then_t2_then_t1() -> None:
    """Equal accuracy, medians and steps over the three step-1 cells -> T4."""
    assert select_format(family(), "dev").native_thought == "T4"


def test_step1_within_five_percent_fewer_steps_wins() -> None:
    """T4 median 100 / 20 steps, T1 median 105 (= 1.05 x 100, inclusive) / 8 steps, T2 median 400: T1 is within and
    has fewer steps -> T1 (against the thought order)."""
    rows = family(
        overrides={
            S1.format("T4"): {"walls": 100, "steps": 20},
            S1.format("T1"): {"walls": 105, "steps": 8},
            S1.format("T2"): {"walls": 400, "steps": 8},
        }
    )
    assert select_format(rows, "dev").native_thought == "T1"


def test_step1_reads_only_native_named_perstep_mar_cells() -> None:
    """Step 1's candidates are native ``perstep_mar`` cells only: the cc ``cd_always:T1/named/perstep_mar`` (exact 40,
    median 10) and native T1's seed and qargmax cells (exact 40, median 10) neither become θ_n nor raise the bar.
    The native perstep_mar cells tie -> T4."""
    rows = family(
        overrides={
            "cd_always:T1/named/perstep_mar": {"exact": 40, "walls": 10},
            "cd_always:native/T1/named/perstep_mar+seed": {"exact": 40, "walls": 10},
            "cd_always:native/T1/named/perstep_mar_qargmax": {"exact": 40, "walls": 10},
        }
    )
    assert select_format(rows, "dev").native_thought == "T4"


def test_step1_schema_below_one_is_ineligible() -> None:
    """Adversarial: native T2 perstep_mar exact 39/40 at median 10 but schema 39/40 -> ineligible, and its 39 does not
    raise the bar (from it, 35/40; T4's 30 would be dropped). T4 (30/40, median 300) vs T1 (30/40, median 500) -> T4."""
    rows = family(
        overrides={
            S1.format("T2"): {"exact": 39, "valid": 39, "walls": 10},
            S1.format("T4"): {"exact": 30, "walls": 300},
            S1.format("T1"): {"exact": 30, "walls": 500},
        }
    )
    assert select_format(rows, "dev").native_thought == "T4"


# =========================================================================== step 2: tolerances


@pytest.mark.parametrize(("native_exact", "winner"), [(33, "native"), (32, "cc")])
def test_step2_dev_tolerance_is_one_tenth_exactly(native_exact: int, winner: str) -> None:
    """Adversarial (Fraction), n=40, dev: cc ``cd_always:T1/named/perstep_mar`` exact 37/40, e2e 2000; native
    ``cd_on_invalid:native/T4/named/end_projection`` ``native_exact``/40, e2e 1000. Bar 37/40 - 1/10 = 33/40, which
    float gets wrong. At 33 native is kept and faster; at 32 it is dropped and cc wins."""
    assert not (33 / 40 >= 37 / 40 - 0.10)
    rows = family(
        overrides={
            "cd_always:T1/named/perstep_mar": {"exact": 37, "e2e": 2000.0},
            NAT_EP.format("T4"): {"exact": native_exact, "e2e": 1000.0},
        }
    )
    fc = select_format(rows, "dev")
    want = NAT_EP.format("T4") if winner == "native" else "cd_always:T1/named/perstep_mar"
    assert _choice(fc) == ("T4", want, winner)


@pytest.mark.parametrize(("native_exact", "winner"), [(33, "native"), (32, "cc")])
def test_step2_test_tolerance_is_one_twentieth_exactly(native_exact: int, winner: str) -> None:
    """Adversarial (Fraction), n=80, test stage, θ_n = T4 given: cc exact 37/80 (e2e 2000) vs native ``native_exact``/80
    (e2e 1000). Bar 37/80 - 1/20 = 33/80 (float: False at 33)."""
    assert not (33 / 80 >= 37 / 80 - 0.05)
    rows = family(
        N_TEST,
        thoughts=("T4",),
        overrides={
            "cd_always:T1/named/perstep_mar": {"exact": 37, "e2e": 2000.0},
            NAT_EP.format("T4"): {"exact": native_exact, "e2e": 1000.0},
        },
    )
    fc = select_format(rows, "test", native_thought="T4")
    want = NAT_EP.format("T4") if winner == "native" else "cd_always:T1/named/perstep_mar"
    assert _choice(fc) == ("T4", want, winner)


@pytest.mark.parametrize(("stage", "winner"), [("dev", "native"), ("test", "cc")])
def test_step2_the_tolerance_follows_the_stage(stage: str, winner: str) -> None:
    """The same rows at both stages, n=80: cc 37/80 (e2e 2000), native 29/80 (e2e 1000). 29/80 = 37/80 - 1/10: kept on
    dev (1/10), dropped on test (1/20). A rule with one tolerance for both stages fails one side."""
    rows = family(
        N_TEST,
        thoughts=("T4",),
        overrides={
            "cd_always:T1/named/perstep_mar": {"exact": 37, "e2e": 2000.0},
            NAT_EP.format("T4"): {"exact": 29, "e2e": 1000.0},
        },
    )
    kw = {"native_thought": "T4"} if stage == "test" else {}
    assert select_format(rows, stage, **kw).winner == winner


def test_step2_native_wins_on_latency_within_the_accuracy_tolerance() -> None:
    """n=40 dev: cc ``cd_on_invalid:T1/named/end_projection`` is the most accurate (38/40, e2e 1500); native
    ``cd_on_invalid:native/T4/named/perstep_mar`` 35/40 (>= 38/40 - 1/10 = 34/40) at e2e 1000 (1500 > 1.05 x 1000)
    -> native, though less accurate."""
    rows = family(
        overrides={
            CC_EP: {"exact": 38, "e2e": 1500.0},
            "cd_on_invalid:native/T4/named/perstep_mar": {"exact": 35, "e2e": 1000.0},
        }
    )
    assert _choice(select_format(rows, "dev")) == (
        "T4",
        "cd_on_invalid:native/T4/named/perstep_mar",
        "native",
    )


# =========================================================================== step 2: the tie to cc


def test_a_full_cc_native_tie_goes_to_cc() -> None:
    """QN2 "final-level ties go to cc": the same kind and arm in both formats, equal exact (37/40), walls, p90 and
    steps -> the cc policy, whichever order the rows arrive in."""
    over = {CC_EP: {"exact": 37, "e2e": 1000.0}, NAT_EP.format("T4"): {"exact": 37, "e2e": 1000.0}}
    rows = family(overrides=over)
    assert _choice(select_format(rows, "dev")) == ("T4", CC_EP, "cc")
    assert _choice(select_format(list(reversed(rows)), "dev")) == ("T4", CC_EP, "cc")


def test_draft_only_ties_go_to_cc_and_draft_only_native_can_win() -> None:
    """``winner`` for ``draft_only:native``: every draft valid in both formats (schema 40/40), exact 37/40, e2e 600 --
    faster than every CD policy (default 9000). Alone it is chosen, winner native; with ``draft_only`` tied, cc's
    ``draft_only`` wins (same kind, format cc first). ``select("stage2")`` ignores ``draft_only:native`` -- a rule
    reusing it unchanged fails the first assertion."""
    draft = {"valid": 40, "exact": 37, "e2e": 600.0, "constrained": False}
    rows = family(overrides={"draft_only:native": draft})
    assert _choice(select_format(rows, "dev")) == ("T4", "draft_only:native", "native")
    rows = family(overrides={"draft_only:native": draft, "draft_only": draft})
    assert _choice(select_format(rows, "dev")) == ("T4", "draft_only", "cc")


@pytest.mark.parametrize(
    ("cc", "native", "why"),
    [
        ("cd_always:T1/named/end_projection", NAT_EP.format("T4"), "kind before format"),
        ("cd_on_invalid:T1/named/perstep_mar", NAT_EP.format("T4"), "arm before format"),
        (
            "cd_on_invalid:T1/named/perstep_mar+seed",
            "cd_on_invalid:native/T4/named/perstep_mar_qargmax",
            "seed arm last",
        ),
    ],
)
def test_format_is_the_last_tie_break(cc: str, native: str, why: str) -> None:
    """Equal exact, walls and steps; the native policy precedes the cc one on an earlier key (``why``) -> native. A
    rule that ranks format before kind or arm gives cc."""
    rows = family(
        overrides={cc: {"exact": 37, "e2e": 1000.0}, native: {"exact": 37, "e2e": 1000.0}}
    )
    assert _choice(select_format(rows, "dev")) == ("T4", native, "native"), why


def test_fewer_steps_beat_the_format_order() -> None:
    """Equal accuracy and walls; native 6 mean steps vs cc's 10 -> native."""
    rows = family(
        overrides={
            CC_EP: {"exact": 37, "e2e": 1000.0, "steps": 10},
            NAT_EP.format("T4"): {"exact": 37, "e2e": 1000.0, "steps": 6},
        }
    )
    assert select_format(rows, "dev").winner == "native"


# =========================================================================== step 2: latency keys (review F1)
# p90 is nearest rank (the ceil(0.9 n)-th value): with n=40 it is the 36th, so 5 outliers move it and 4 do not.


def test_within_five_percent_the_lower_p90_wins_across_formats() -> None:
    """Review F1: cc ``cd_on_invalid:T1/named/end_projection`` e2e 35 x 1000 + 5 x 3000 (p50 1000, p90 3000) against
    native ``cd_on_invalid:native/T4/named/end_projection`` 35 x 1040 + 5 x 1100 (p50 1040 <= 1.05 x 1000, p90 1100),
    both 37/40 -> native on p90. cc wins every later key (equal steps, kind and arm; then format), so only p90 can
    choose native; without the 5 % window only cc (the lowest p50) would remain."""
    rows = family(
        overrides={
            CC_EP: {"exact": 37, "e2e": [1000.0] * 35 + [3000.0] * 5},
            NAT_EP.format("T4"): {"exact": 37, "e2e": [1040.0] * 35 + [1100.0] * 5},
        }
    )
    assert _choice(select_format(rows, "dev")) == ("T4", NAT_EP.format("T4"), "native")


@pytest.mark.parametrize(("native_e2e", "winner"), [(1050.0, "native"), (1051.0, "cc")])
def test_the_five_percent_window_is_inclusive(native_e2e: float, winner: str) -> None:
    """Review F1: cc 35 x 1000 + 5 x 5000 (p50 1000, p90 5000) against native constant ``native_e2e``, both 37/40. At
    1050 (= 1.05 x 1000) native is inside the window and its p90 (1050) beats cc's -> native; at 1051 it is outside and
    cc's lower p50 decides -> cc. A strict bound gives cc at 1050; no window gives native at 1051."""
    rows = family(
        overrides={
            CC_EP: {"exact": 37, "e2e": [1000.0] * 35 + [5000.0] * 5},
            NAT_EP.format("T4"): {"exact": 37, "e2e": native_e2e},
        }
    )
    assert select_format(rows, "dev").winner == winner


def test_latency_is_over_tool_cases_only() -> None:
    """Review F1: native ``cd_on_invalid:native/T4/named/end_projection`` 40 tool rows at e2e 1000 plus 60 no-tool rows
    (no call) at 90000; cc ``cd_on_invalid:T1/named/end_projection`` 40 tool rows at 1200; both 37/40. Over tool cases
    native wins (1200 > 1.05 x 1000); over all 100 rows native's p50 would be 90000 and cc would win."""
    rows = family(
        overrides={
            CC_EP: {"exact": 37, "e2e": 1200.0},
            NAT_EP.format("T4"): {"exact": 37, "e2e": 1000.0},
        }
    )
    rows += [
        {
            "policy": NAT_EP.format("T4"),
            "case": f"chat{i:03d}",
            "tool": None,
            "tool_case": False,
            "called": False,
            "schema_valid": None,
            "exact": False,
            "parsed": False,
            "cs": None,
            "pass2_wall_ms": None,
            "e2e_wall_ms": 90000.0,
            "steps": 0.0,
            "pass2_sent": False,
            "missing": False,
        }
        for i in range(60)
    ]
    assert _choice(select_format(rows, "dev")) == ("T4", NAT_EP.format("T4"), "native")


# =========================================================================== adversarial ineligibility


def test_a_native_policy_with_schema_75_of_76_is_never_chosen_and_sets_no_bar() -> None:
    """Adversarial, n=76 dev: native ``cd_always:native/T4/named/end_projection`` has the best exact (70/76) and the
    lowest e2e (500), but schema 75/76 -> ineligible. cc ``cd_on_invalid:T1/named/perstep_mar`` 62/76 at e2e 1500 is
    chosen. Were the bar taken from the ineligible 70/76 it would be 70/76 - 1/10 > 62/76 and cc would be dropped
    (then no kept policy but defaults at 0/76 -- also dropped)."""
    assert Fraction(62, 76) < Fraction(70, 76) - Fraction(1, 10)
    rows = family(
        76,
        overrides={
            "cd_always:native/T4/named/end_projection": {"exact": 70, "valid": 75, "e2e": 500.0},
            "cd_on_invalid:T1/named/perstep_mar": {"exact": 62, "e2e": 1500.0},
        },
    )
    assert _choice(select_format(rows, "dev")) == ("T4", "cd_on_invalid:T1/named/perstep_mar", "cc")


def test_a_single_cs_false_row_makes_a_policy_ineligible() -> None:
    """Adversarial, n=40 dev: native ``cd_always:native/T4/named/perstep_mar_qargmax`` exact 40/40, e2e 500, with one
    constrained row ``cs`` False -> ineligible; cc ``cd_on_invalid:T1/named/end_projection`` (38/40, e2e 1500) wins.
    The other side (guard not too wide): with ``cs_bad`` 0 the native policy wins."""
    over: dict[str, dict[str, Any]] = {
        "cd_always:native/T4/named/perstep_mar_qargmax": {"exact": 40, "e2e": 500.0, "cs_bad": 1},
        CC_EP: {"exact": 38, "e2e": 1500.0},
    }
    assert _choice(select_format(family(overrides=over), "dev")) == ("T4", CC_EP, "cc")
    over["cd_always:native/T4/named/perstep_mar_qargmax"]["cs_bad"] = 0
    assert select_format(family(overrides=over), "dev").winner == "native"


@pytest.mark.parametrize("stage", ["dev", "test"])
def test_a_theta_n_whose_seed_cell_is_missing_raises(stage: str) -> None:
    """Adversarial: θ_n = T4 (the step-1 tie), and no row of ``native/T4/named/perstep_mar+seed`` under either kind ->
    ``ValueError`` (the family is incomplete; choosing without the seed arm would be a different rule)."""
    seed = [f"{k}:native/T4/named/perstep_mar+seed" for k in KINDS]
    rows = family(drop=seed, overrides={CC_EP: {"exact": 37, "e2e": 1000.0}})
    kw = {"native_thought": "T4"} if stage == "test" else {}
    with pytest.raises(ValueError):
        select_format(rows, stage, **kw)


def test_a_missing_seed_cell_of_another_thought_does_not_raise() -> None:
    """The other side of the refusal: native T1's seed cell missing while θ_n = T4 -> no error, cc's fast policy wins."""
    seed = [f"{k}:native/T1/named/perstep_mar+seed" for k in KINDS]
    rows = family(drop=seed, overrides={CC_EP: {"exact": 37, "e2e": 1000.0}})
    assert _choice(select_format(rows, "dev")) == ("T4", CC_EP, "cc")


# =========================================================================== the test stage


def test_the_test_stage_takes_theta_n_as_given_and_runs_step_2_only() -> None:
    """n=80, test, ``native_thought="T1"``: only native T1 cells (plus cc) exist, and the native step-1 cell
    ``cd_always:native/T1/named/perstep_mar`` has schema 79/80 -- step 1 would have no eligible cell and raise. Step 2
    alone chooses ``cd_on_invalid:native/T1/named/end_projection`` (exact 70/80, e2e 1000)."""
    rows = family(
        N_TEST,
        thoughts=("T1",),
        overrides={
            S1.format("T1"): {"exact": 70, "valid": 79},
            NAT_EP.format("T1"): {"exact": 70, "e2e": 1000.0},
        },
    )
    assert _choice(select_format(rows, "test", native_thought="T1")) == (
        "T1",
        NAT_EP.format("T1"),
        "native",
    )


def test_the_test_stage_family_is_the_given_thought_only() -> None:
    """Test stage with native T2 and T4 rows present, ``native_thought="T2"``: the faster native T4 policy (e2e 500) is
    outside the family; native T2's (e2e 1000) wins."""
    rows = family(
        N_TEST,
        thoughts=("T2", "T4"),
        overrides={
            NAT_EP.format("T4"): {"exact": 70, "e2e": 500.0},
            NAT_EP.format("T2"): {"exact": 70, "e2e": 1000.0},
        },
    )
    assert _choice(select_format(rows, "test", native_thought="T2")) == (
        "T2",
        NAT_EP.format("T2"),
        "native",
    )


def test_the_test_stage_without_a_native_thought_raises() -> None:
    with pytest.raises(ValueError):
        select_format(family(N_TEST, thoughts=("T4",)), "test")


def test_an_unknown_stage_raises() -> None:
    with pytest.raises(ValueError):
        select_format(family(), "stage2")


def test_no_eligible_policy_raises() -> None:
    """Every family CD policy has one CS-false row and both draft-only policies have schema 0/40 -> ``ValueError``.
    (Step 1 is still decided: CS is not part of the step-1 rule.)"""
    names = [f"{k}:T1/named/{a}" for k in KINDS for a in ARMS]
    names += [f"{k}:native/{t}/named/{a}" for t in ("T1", "T2", "T4") for k in KINDS for a in ARMS]
    with pytest.raises(ValueError):
        select_format(family(overrides={n: {"cs_bad": 1} for n in names}), "dev")


# =========================================================================== shape


def test_the_result_carries_the_tables_of_both_steps_on_dev() -> None:
    """Criterion 1/3: ``tables`` holds the step-1 and step-2 tables; each step-1 cell and each family policy is named in
    it (only the mention is pinned; the layout is the implementer's)."""
    fc = select_format(family(overrides={CC_EP: {"exact": 37, "e2e": 1000.0}}), "dev")
    assert isinstance(fc.tables, Mapping) and fc.tables
    text = repr(fc.tables)
    for t in ("T1", "T2", "T4"):
        assert f"native/{t}/named/perstep_mar" in text, t
    for a in ARMS:
        assert f"native/T4/named/{a}" in text and f"T1/named/{a}" in text, a
    assert "draft_only:native" in text
