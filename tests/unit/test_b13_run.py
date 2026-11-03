"""T-0071a rulings R1-R4 (review round 1): ``local_astra/bench/b13_run.py``, the a2 driver's pure logic (fast suite,
torch-free, no I/O).

Rulings (``docs/tasks/T-0071a.md`` at ea78cd4):
- R1: each gated stratum's verdict is computed at F̂, at F̂ = 0 and at F̂⁺ = F̂ + 2·SE(F̂) (OLS standard error of the
  intercept). PASS only if all three pass; FAIL if F̂ and F̂ = 0 both fail; otherwise ``not_resolvable``.
- R2: F̂ and SE(F̂) per gated stratum, on that stratum's A and A′ only; too-uniform A/A′ steps → ``not_resolvable``,
  never a pooled substitute; the pooled fit (all gated A/A′) is a sensitivity value only.
- R3: ``schedule``, ``check_rows``, ``evaluate_b13`` live here (pure); the driver only calls them.
- R4: a crash after the gated strata still evaluates the strata that completed (the crash is named); a report-only
  stratum that fails to build is a run problem.
- T2: D (or E in a report-only stratum) at slot ``r % 4`` of the rotated A/B/A′ order. T3: a commit frame's ``step``
  equals the CD record's ``steps`` exactly; a mismatch is a finding (run problem, exit 1).
- Round dropping (implementation log, accepted by the review): a round with a problem in A, B or A′ is dropped; a
  problem in D drops D only.

Assumed API (test-author ruling, 2026-10-03; to be recorded in docs/tasks/T-0071a.md by its owner). Rows, frames and
the engine log are shaped like ``scripts/gpu/cd_b13.py``'s evidence (``requests.jsonl`` rows, tap wire dicts, the
engine log text with ``canvas tap release:`` lines and ``ASTRA_CD <json>`` records).

``schedule(n_rounds: int, extra: str) -> list[list[str]]``
    Round ``r`` = ``b13_steps.rotation(n)[r]`` with ``extra`` inserted at index ``r % 4``.
``fit_fixed_cost_se(steps: Sequence[int], wall: Sequence[float]) -> tuple[float, float]``
    ``(F̂, SE(F̂))``: OLS intercept and its standard error ``s·sqrt(1/n + x̄²/Sxx)``, ``s² = RSS/(n − 2)``.
    ``ValueError`` on unequal lengths, fewer than 3 points, or all steps equal.
``combine_verdicts(at_fit: str, at_zero: str, at_upper: str) -> str``  -- R1's rule.
``gate(rows: Iterable[b13_steps.RequestTiming], *, threshold_pct: float) -> GateResult``
    One gated stratum (R1 + R2): F̂/SE from its own A and A′ rows, E10 (``b13_cd.evaluate``) on the A/B/A′ per-step
    times paired by round at F̂, 0 and F̂ + 2·SE. ``GateResult`` has ``verdict``, ``fixed``, ``se``, ``at_fit``,
    ``at_zero``, ``at_upper`` (``B13Result`` or ``None`` when not computable). Never raises on data: an
    unidentifiable fit or a non-positive per-step time makes the verdict ``not_resolvable``.
``check_rows(rows, frames, engine_log: str) -> RowCheck``
    Per measured row (``warmup`` false, arm in A/A′/B/D/E): Σsteps from its commit frames (joined by
    ``response.id`` + ``-<8 hex>``), commit count = ceil(completion_tokens/256), a release line with
    ``commit_overrun == 0``, and for a row with a ``record_tag`` the records' checks and T3. ``RowCheck`` has
    ``problems: dict[int, list[str]]`` (row index → problems), ``steps: dict[int, int]`` (row index → Σsteps),
    ``t3_mismatches: list[str]`` and ``run_problems: list[str]`` (every T3 mismatch is one).
``StratumSpec(bucket: int, gated: bool, rounds: int, build_error: str | None = None)``
``evaluate_b13(rows, problems, strata, thresholds, *, crash: str | None = None) -> B13Evaluation``
    ``rows`` carry ``stratum``, ``round``, ``arm``, ``wall_ms``, ``steps``, ``warmup``; ``problems`` is
    ``RowCheck.problems``; ``thresholds`` maps bucket → limit %. ``B13Evaluation`` has ``gates: dict[int,
    GateResult]`` (gated buckets only), ``pooled_fixed: float | None``, ``exposure: dict[int, B13Result]`` (D vs A
    per step at the stratum's F̂, A/A′ control), ``dropped: dict[int, list[int]]`` (rounds dropped per bucket),
    ``run_problems: list[str]`` and ``lines: list[str]``.
``exit_code(run_problems: Sequence[str]) -> int``  -- 0 when empty, else 1 (a run defect; D-H1: never the verdict).

Every expected value is worked by hand from the generating model ``wall = F + steps × per_step`` or comes from an
already-tested oracle (``b13_steps.fit_fixed_cost``, ``b13_cd.evaluate``). n = 30 rounds; no random numbers.
"""

from __future__ import annotations

import collections
import importlib
import json
import math
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

import pytest

from local_astra.bench import b13_cd, b13_steps

N = 30
F = 47.0
A_STEP = 18.5
ARMS = ("A", "B", "A'")
RT = b13_steps.RequestTiming


def mod() -> ModuleType:
    try:
        return importlib.import_module("local_astra.bench.b13_run")
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("local_astra.bench.b13_run"):
            pytest.fail(
                f"T-0071a R3 not implemented: local_astra.bench.b13_run does not exist ({exc})"
            )
        raise


def cycle(*vals: int) -> Callable[[int], int]:
    return lambda r: vals[r % len(vals)]


def timings(
    *,
    a_steps: Callable[[int], int],
    b_steps: int,
    b_step_ms: float,
    fixed: float = F,
    a_step_ms: float = A_STEP,
    a_extra: Callable[[int, str], float] = lambda r, arm: 0.0,
    n: int = N,
) -> list[Any]:
    """A, B, A′ per round from ``wall = fixed + steps × per_step`` (+ ``a_extra(r, arm)`` on A and A′)."""
    out = []
    for r in range(n):
        s = a_steps(r)
        out.append(RT(round_idx=r, arm="A", wall=fixed + s * a_step_ms + a_extra(r, "A"), steps=s))
        out.append(RT(round_idx=r, arm="B", wall=fixed + b_steps * b_step_ms, steps=b_steps))
        out.append(
            RT(round_idx=r, arm="A'", wall=fixed + s * a_step_ms + a_extra(r, "A'"), steps=s)
        )
    return out


# --------------------------------------------------------------------------- T2: the schedule


def test_schedule_puts_d_at_slot_r_mod_4_of_the_rotated_order() -> None:
    got = [list(x) for x in mod().schedule(5, "D")]
    assert got == [
        ["D", "A", "B", "A'"],
        ["B", "D", "A'", "A"],
        ["A'", "A", "D", "B"],
        ["A", "B", "A'", "D"],
        ["D", "B", "A'", "A"],
    ]


def test_schedule_over_twelve_rounds_meets_every_rotation_and_d_slot_once() -> None:
    """lcm(3, 4) = 12: each of the 3 gated orders × 4 D slots exactly once; over 30 rounds the gated order (D
    removed) is still 10/10/10. Mutant killed: D at a fixed slot, or D's slot tied to the rotation."""
    got = mod().schedule(12, "D")
    pairs = {(tuple(a for a in o if a != "D"), list(o).index("D")) for o in got}
    assert len(pairs) == 12
    thirty = [tuple(a for a in o if a != "D") for o in mod().schedule(N, "D")]
    assert collections.Counter(thirty) == {ARMS: 10, ("B", "A'", "A"): 10, ("A'", "A", "B"): 10}
    assert [len(o) for o in mod().schedule(N, "E")] == [4] * N


# --------------------------------------------------------------------------- R1: F̂, SE(F̂) and the three-way rule


def test_intercept_standard_error_by_hand() -> None:
    """x = 0, 1, 2; y = 1, 2, 4: intercept 5/6; residuals 1/6, −1/3, 1/6, RSS 1/6, s² = RSS/(n − 2) = 1/6,
    x̄ = 1, Sxx = 2 → SE² = 1/6 · (1/3 + 1/2) = 5/36, SE = √5/6."""
    f, se = mod().fit_fixed_cost_se([0, 1, 2], [1.0, 2.0, 4.0])
    assert f == pytest.approx(5 / 6, abs=1e-12)
    assert se == pytest.approx(math.sqrt(5) / 6, abs=1e-12)


def test_intercept_standard_error_is_zero_on_a_noise_free_line() -> None:
    f, se = mod().fit_fixed_cost_se([3, 4, 5, 9], [F + s * A_STEP for s in (3, 4, 5, 9)])
    assert f == pytest.approx(F, abs=1e-9)
    assert se == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("steps", "wall"),
    [([4, 4, 4, 4], [121.0, 122.0, 120.0, 121.0]), ([3, 4], [102.5, 121.0]), ([3, 4, 5], [1.0])],
    ids=["all-steps-equal", "two-points", "unequal-lengths"],
)
def test_intercept_standard_error_refuses_an_unidentifiable_fit(
    steps: list[int], wall: list[float]
) -> None:
    with pytest.raises(ValueError):
        mod().fit_fixed_cost_se(steps, wall)


@pytest.mark.parametrize(
    ("at_fit", "at_zero", "at_upper", "want"),
    [
        ("pass", "pass", "pass", "pass"),
        ("pass", "pass", "fail", "not_resolvable"),
        ("pass", "pass", "not_resolvable", "not_resolvable"),
        ("pass", "fail", "pass", "not_resolvable"),
        ("fail", "pass", "fail", "not_resolvable"),
        ("fail", "fail", "fail", "fail"),
        ("fail", "fail", "pass", "fail"),
        ("fail", "fail", "not_resolvable", "fail"),
        ("not_resolvable", "fail", "fail", "not_resolvable"),
        ("fail", "not_resolvable", "fail", "not_resolvable"),
        ("not_resolvable", "not_resolvable", "pass", "not_resolvable"),
        ("pass", "not_resolvable", "pass", "not_resolvable"),
    ],
)
def test_r1_rule(at_fit: str, at_zero: str, at_upper: str, want: str) -> None:
    assert mod().combine_verdicts(at_fit, at_zero, at_upper) == want


def test_gate_fails_a_true_290_pct_overhead() -> None:
    """Noise-free: F̂ = 47 exactly, SE 0, +290 % at F̂ (and at F̂⁺ = F̂), +142 % at 0: FAIL."""
    g = mod().gate(
        timings(a_steps=cycle(3, 4, 5), b_steps=42, b_step_ms=3.9 * A_STEP), threshold_pct=10.0
    )
    assert g.fixed == pytest.approx(F, abs=1e-9) and g.se == pytest.approx(0.0, abs=1e-9)
    assert g.at_fit.delta_pct == pytest.approx(290.0, abs=1e-6)
    assert g.at_upper.delta_pct == pytest.approx(290.0, abs=1e-6)
    assert g.verdict == "fail"


def test_gate_passes_a_clean_two_pct_overhead() -> None:
    g = mod().gate(
        timings(a_steps=cycle(3, 4, 5), b_steps=42, b_step_ms=1.02 * A_STEP), threshold_pct=10.0
    )
    assert (g.at_fit.verdict, g.at_zero.verdict, g.at_upper.verdict) == ("pass", "pass", "pass")
    assert g.verdict == "pass"


def test_reviewer_b_false_pass_shape_is_not_resolvable() -> None:
    """True +10 % at a 10 % limit (B 20.35 ms per step, 42 steps), A/A′ at 10 or 11 steps (alternating rounds, a
    narrow spread far from 0 so F̂ is poorly identified). The A/A′ walls carry a residual −0.2 ms at 10 steps and
    +0.2 ms at 11 (so the line through the two group means has intercept F − 21 × 0.2 = 42.8: F̂ under-reads F
    by 4.2 ms), plus +2 ms on A and −2 ms on A′ (residual scatter, so SE > 0).

    By hand: RSS = 60 × 2² = 240, s² = 240/58, x̄ = 10.5, Sxx = 60 × 0.25 = 15, SE = √(240/58) · √(1/60 + 110.25/15)
    ≈ 5.52, F̂⁺ ≈ 53.84.
    - at F̂ = 42.8: A per step 18.5 + 6/10 = 19.10 (s 10) or 18.5 + 6.4/11 = 19.08 (s 11); B 20.35 + 4.2/42 = 20.45:
      +7.1 % → pass (control A′/A ≈ 2.1 %);
    - at 0: A ≈ 23.4, B 21.47: ≈ −8 % → pass;
    - at F̂⁺: A 18.5 + (−6.84 − 0.2 + 2)/10 = 18.00 or 18.5 + (−6.84 + 0.2 + 2)/11 = 18.08, B 20.35 − 6.84/42 = 20.19:
      +11.7..+12.2 % → fail.
    A point-F̂ verdict says PASS; R1 says ``not_resolvable``. Mutant killed: F̂⁺ ignored (or SE taken as 0)."""

    def extra(r: int, arm: str) -> float:
        resid = -0.2 if r % 2 == 0 else 0.2
        return resid + (2.0 if arm == "A" else -2.0)

    g = mod().gate(
        timings(a_steps=cycle(10, 11), b_steps=42, b_step_ms=1.10 * A_STEP, a_extra=extra),
        threshold_pct=10.0,
    )
    se = math.sqrt(240 / 58) * math.sqrt(1 / 60 + 10.5**2 / 15)
    assert g.fixed == pytest.approx(F - 21 * 0.2, abs=1e-9)
    assert g.se == pytest.approx(se, rel=1e-9)
    assert g.at_fit.verdict == "pass" and g.at_zero.verdict == "pass", (g.at_fit, g.at_zero)
    assert g.at_upper.verdict == "fail", g.at_upper
    want_upper = b13_cd.evaluate(
        [A_STEP + (F - (F - 4.2 + 2 * se) + extra(r, "A")) / (10, 11)[r % 2] for r in range(N)],
        [1.10 * A_STEP + (F - (F - 4.2 + 2 * se)) / 42] * N,
        [A_STEP + (F - (F - 4.2 + 2 * se) + extra(r, "A'")) / (10, 11)[r % 2] for r in range(N)],
        threshold_pct=10.0,
    )
    assert g.at_upper.delta_pct == pytest.approx(want_upper.delta_pct, abs=1e-6)
    assert g.verdict == "not_resolvable"


def test_gate_with_uniform_a_steps_is_not_resolvable_and_does_not_raise() -> None:
    """R2: every A/A′ at 4 steps, so no F̂ exists; a clean +2 % still cannot pass."""
    g = mod().gate(
        timings(a_steps=lambda r: 4, b_steps=42, b_step_ms=1.02 * A_STEP), threshold_pct=10.0
    )
    assert g.verdict == "not_resolvable"


def test_gate_with_a_huge_se_is_not_resolvable_and_does_not_raise() -> None:
    """A/A′ at 20 steps except one round at 21, ±20 ms scatter: SE(F̂) is in the hundreds of ms, F̂⁺ exceeds every A
    wall (47 + 21 × 18.5 + 20 = 455.5 ms at most), so A's per-step times at F̂⁺ are negative and E10 refuses them.
    The verdict is ``not_resolvable``, not an exception. Mutant killed: letting E10's ``ValueError`` escape."""
    g = mod().gate(
        timings(
            a_steps=lambda r: 21 if r == 7 else 20,
            b_steps=42,
            b_step_ms=1.02 * A_STEP,
            a_extra=lambda r, arm: 20.0 if (r + (arm == "A'")) % 2 else -20.0,
        ),
        threshold_pct=10.0,
    )
    assert g.se > 200.0
    assert g.fixed + 2 * g.se > F + 21 * A_STEP + 20.0, "fixture: F̂⁺ must exceed every A wall"
    assert g.verdict == "not_resolvable"


# --------------------------------------------------------------------------- driver-shaped rows for evaluate_b13


def stratum_rows(
    bucket: int,
    *,
    a_steps: Callable[[int], int],
    b_steps: int,
    b_step_ms: float,
    fixed: float = F,
    extra: str = "D",
    rounds: int = N,
) -> list[dict[str, Any]]:
    """``requests.jsonl``-shaped rows (with ``steps`` filled, as after ``check_rows``) for one stratum, in schedule
    order, plus one warm-up round that must be ignored. D/E: 96 % of an A wall at A's steps."""
    out: list[dict[str, Any]] = []
    plan = [(-1, ["A", "B", "A'", extra], True)]
    plan += [(r, list(o), False) for r, o in enumerate(mod().schedule(rounds, extra))]
    for r, order, warm in plan:
        for slot, arm in enumerate(order):
            s = b_steps if arm in ("B", "E") else a_steps(max(r, 0))
            if arm in ("A", "A'"):
                wall = fixed + s * A_STEP
            elif arm in ("B", "E"):
                wall = fixed + s * b_step_ms
            else:
                wall = 0.96 * (fixed + s * A_STEP)
            out.append(
                {
                    "name": f"b{bucket}:{'w' if warm else 'r'}{abs(r)}:{arm}",
                    "stratum": bucket,
                    "round": r,
                    "slot": slot,
                    "arm": arm,
                    "warmup": warm,
                    "wall_ms": 9999.0 if warm else wall,
                    "steps": s,
                }
            )
    return out


def index_of(rows: Sequence[dict[str, Any]], bucket: int, rnd: int, arm: str) -> int:
    return next(
        i
        for i, r in enumerate(rows)
        if r["stratum"] == bucket and r["round"] == rnd and r["arm"] == arm and not r["warmup"]
    )


GATED = (64, 128)
LIMITS = {64: 10.0, 128: 15.0}


def specs(*, report_only: bool = False, build_error: str | None = None) -> list[Any]:
    m = mod()
    out = [
        m.StratumSpec(bucket=64, gated=True, rounds=N),
        m.StratumSpec(bucket=128, gated=True, rounds=N),
    ]
    if report_only:
        out.append(m.StratumSpec(bucket=256, gated=False, rounds=5, build_error=build_error))
        out.append(m.StratumSpec(bucket=512, gated=False, rounds=5))
    return out


def two_strata(**b64: Any) -> list[dict[str, Any]]:
    kw: dict[str, Any] = {"a_steps": cycle(3, 4, 5), "b_steps": 42, "b_step_ms": 1.02 * A_STEP}
    kw.update(b64)
    rows = stratum_rows(64, **kw)
    rows += stratum_rows(
        128, a_steps=cycle(8, 10, 12), b_steps=60, b_step_ms=1.05 * A_STEP, fixed=300.0
    )
    return rows


# --------------------------------------------------------------------------- R2: per-stratum F̂


def test_each_gated_stratum_uses_its_own_fixed_cost() -> None:
    """b64 has F = 47, b128 F = 300 (a longer prompt): each gate's F̂ is its own, exactly; the pooled F̂ is the fit
    over every gated A/A′ (oracle: ``b13_steps.fit_fixed_cost``) and is reported, not used. Mutant killed: the
    implementation's pre-R2 single pooled F̂ for both strata."""
    rows = two_strata()
    res = mod().evaluate_b13(rows, {}, specs(), LIMITS)
    assert set(res.gates) == set(GATED)
    assert res.gates[64].fixed == pytest.approx(F, abs=1e-6)
    assert res.gates[128].fixed == pytest.approx(300.0, abs=1e-6)
    assert res.gates[64].verdict == "pass" and res.gates[128].verdict == "pass"
    ref = [r for r in rows if not r["warmup"] and r["arm"] in ("A", "A'")]
    pooled = b13_steps.fit_fixed_cost([r["steps"] for r in ref], [r["wall_ms"] for r in ref])
    assert res.pooled_fixed == pytest.approx(pooled, abs=1e-6)
    assert abs(pooled - F) > 1.0 and abs(pooled - 300.0) > 1.0, (
        "fixture: pooled must differ from both"
    )


def test_a_stratum_with_uniform_a_steps_is_not_resolvable_never_pooled() -> None:
    """b64's A/A′ all at 4 steps (no own F̂), both strata on the same line (F = 47, 18.5 ms per step), so the pooled
    F̂ is exactly 47 and would pass b64's clean +2 %. R2: b64 is ``not_resolvable``; b128 still gets its verdict.
    Mutant killed: falling back to the pooled F̂."""
    rows = stratum_rows(64, a_steps=lambda r: 4, b_steps=42, b_step_ms=1.02 * A_STEP)
    rows += stratum_rows(128, a_steps=cycle(8, 10, 12), b_steps=60, b_step_ms=1.05 * A_STEP)
    res = mod().evaluate_b13(rows, {}, specs(), LIMITS)
    assert res.pooled_fixed == pytest.approx(F, abs=1e-6)
    assert res.gates[64].verdict == "not_resolvable"
    assert res.gates[128].verdict == "pass"


# --------------------------------------------------------------------------- dropping rounds


def test_a_problem_in_b_drops_the_whole_round() -> None:
    rows = two_strata()
    problems = {index_of(rows, 64, 3, "B"): ["HTTP 500: boom"]}
    res = mod().evaluate_b13(rows, problems, specs(), LIMITS)
    assert sorted(res.dropped[64]) == [3]
    assert res.gates[64].at_fit.n == N - 1
    assert res.gates[128].at_fit.n == N and sorted(res.dropped[128]) == []
    assert any("3" in p and "64" in p for p in res.run_problems), res.run_problems


@pytest.mark.parametrize("arm", ["A", "A'"])
def test_a_problem_in_a_or_a_prime_drops_the_whole_round(arm: str) -> None:
    rows = two_strata()
    res = mod().evaluate_b13(
        rows, {index_of(rows, 64, 11, arm): ["no tap release line"]}, specs(), LIMITS
    )
    assert sorted(res.dropped[64]) == [11]
    assert res.gates[64].at_fit.n == N - 1


def test_a_problem_in_d_drops_d_only() -> None:
    """D is informational: its problem removes it from the exposure (n 29), not the round from the gate (n 30).
    Mutants killed: dropping the round on a D problem; ignoring D problems (exposure n 30)."""
    rows = two_strata()
    res = mod().evaluate_b13(
        rows, {index_of(rows, 64, 5, "D"): ["Σsteps: no commit frames"]}, specs(), LIMITS
    )
    assert sorted(res.dropped[64]) == []
    assert res.gates[64].at_fit.n == N
    assert res.exposure[64].n == N - 1
    assert res.exposure[128].n == N


D_DEFECTS = [
    "Σsteps: no commit frames",
    "no tap release line for chatcmpl-0b5",
    "release line chatcmpl-0b5-0a1b2c3d has commit_overrun 3",
    "1 commit frames, want ceil(300/256) = 2",
]


@pytest.mark.parametrize(
    "why", D_DEFECTS, ids=["no-commit", "no-release", "overrun", "short-count"]
)
def test_a_problem_in_d_keeps_the_round_but_is_a_run_problem(why: str) -> None:
    """Reviewer A's N1 (R7 as clarified at 262495c): the defect behind any dropped row is a run problem. D's problem
    (a tap defect on the no-xargs path) leaves the gate's n at 30, drops D from the exposure, and makes the run exit
    1. Mutant killed: "a D/E problem only drops that row from the figures" (exit 0 with a tap defect)."""
    rows = two_strata()
    res = mod().evaluate_b13(rows, {index_of(rows, 128, 5, "D"): [why]}, specs(), LIMITS)
    assert res.gates[128].at_fit.n == N and sorted(res.dropped[128]) == []
    assert res.exposure[128].n == N - 1
    assert any(why in p for p in res.run_problems), res.run_problems
    assert mod().exit_code(res.run_problems) == 1


def test_a_problem_in_e_of_a_report_only_stratum_is_a_run_problem() -> None:
    """The same for E (``end_projection``) in a report-only stratum: a CD error record on E is a defect, whatever the
    stratum's figures do with the row. The gated verdicts are unchanged. Mutant killed: as above, for E."""
    rows = two_strata() + stratum_rows(
        256, a_steps=cycle(5, 6, 7), b_steps=80, b_step_ms=2.0 * A_STEP, extra="E", rounds=5
    )
    rows += stratum_rows(
        512, a_steps=cycle(5, 6, 7), b_steps=90, b_step_ms=3.0 * A_STEP, extra="E", rounds=5
    )
    why = "canvas 0 has error internal"
    res = mod().evaluate_b13(
        rows, {index_of(rows, 256, 2, "E"): [why]}, specs(report_only=True), LIMITS
    )
    assert any(why in p for p in res.run_problems), res.run_problems
    assert mod().exit_code(res.run_problems) == 1
    assert res.gates[64].verdict == "pass" and res.gates[128].verdict == "pass"
    assert res.gates[64].at_fit.n == N and res.gates[128].at_fit.n == N


def test_a_round_missing_an_arm_after_a_crash_is_dropped_and_the_gate_still_computed() -> None:
    """Reviewer B's survivor: the run died mid-round, so b64 round 3 has A (and D) but no B row at all (not a row with
    a problem: no row). The round is dropped and named, the gate is computed on the 29 complete rounds and still
    passes the clean +2 %. Mutants killed: an incomplete round reaching ``pair_by_round`` (``ValueError`` → no verdict
    or an exception), and the round silently kept with a B borrowed from elsewhere (n 30)."""
    rows = two_strata()
    del rows[index_of(rows, 64, 3, "B")]
    res = mod().evaluate_b13(
        rows, {}, specs(), LIMITS, crash="RuntimeError: engine died in b64 round 3"
    )
    assert sorted(res.dropped[64]) == [3]
    assert res.gates[64].at_fit is not None and res.gates[64].at_fit.n == N - 1
    assert res.gates[64].verdict == "pass"
    assert res.gates[128].at_fit.n == N


def test_the_xargs_exposure_is_d_vs_a_per_step_at_the_strata_fixed_cost() -> None:
    """D's wall is 96 % of A's at A's steps: at F̂ = 47 its per step is (0.96 (47 + 18.5 s) − 47)/s, below A's
    18.5: the exposure delta is negative, worked by hand for s = 3, 4, 5 and paired by round."""
    rows = two_strata()
    res = mod().evaluate_b13(rows, {}, specs(), LIMITS)
    s_of = cycle(3, 4, 5)
    d = [(0.96 * (F + s_of(r) * A_STEP) - F) / s_of(r) for r in range(N)]
    want = b13_cd.evaluate([A_STEP] * N, d, [A_STEP] * N, threshold_pct=10.0)
    assert res.exposure[64].delta_pct == pytest.approx(want.delta_pct, abs=1e-6)
    assert res.exposure[64].delta_pct < 0


# --------------------------------------------------------------------------- R4: crashes and build failures


def test_a_crash_after_the_gated_strata_still_evaluates_them() -> None:
    """The run died during b256: no report-only rows at all. Both gated verdicts are still computed; the crash is a
    named run problem (exit 1). Mutant killed: evaluation skipped (or raising) when a stratum has no rows."""
    rows = two_strata()
    crash = "RuntimeError: engine died during b256 round 2"
    res = mod().evaluate_b13(rows, {}, specs(report_only=True), LIMITS, crash=crash)
    assert res.gates[64].verdict == "pass" and res.gates[128].verdict == "pass"
    assert any(crash in p for p in res.run_problems), res.run_problems
    assert mod().exit_code(res.run_problems) == 1


def test_a_report_only_stratum_that_failed_to_build_is_a_run_problem() -> None:
    why = "live_simple_130-84-0 compiled to bucket 128, not 256"
    rows = two_strata() + stratum_rows(
        512, a_steps=cycle(5, 6, 7), b_steps=80, b_step_ms=2.0 * A_STEP, extra="E", rounds=5
    )
    res = mod().evaluate_b13(rows, {}, specs(report_only=True, build_error=why), LIMITS)
    assert any(why in p for p in res.run_problems), res.run_problems
    assert mod().exit_code(res.run_problems) == 1
    assert set(res.gates) == set(GATED), "report-only strata never get a gate"


def test_a_clean_run_has_no_run_problems() -> None:
    """The negative companion of the two rows above: nothing dropped, nothing crashed → exit 0."""
    rows = two_strata() + stratum_rows(
        256, a_steps=cycle(5, 6, 7), b_steps=80, b_step_ms=2.0 * A_STEP, extra="E", rounds=5
    )
    rows += stratum_rows(
        512, a_steps=cycle(5, 6, 7), b_steps=90, b_step_ms=3.0 * A_STEP, extra="E", rounds=5
    )
    res = mod().evaluate_b13(rows, {}, specs(report_only=True), LIMITS)
    assert res.run_problems == []
    assert mod().exit_code(res.run_problems) == 0


# --------------------------------------------------------------------------- check_rows on driver-shaped evidence

SUFFIX = "-0a1b2c3d"
STEPS = 9


def evidence(
    *, b_record_steps: int = STEPS, release_for_a: bool = True, a_tokens: int = 40
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """One measured round of A, B, A′, D (plus a warm-up row and a flush row, both ignored): each response 40
    tokens = one canvas, one commit frame at step 9, one tap release line; B's ``ASTRA_CD`` record."""
    rows: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    log: list[str] = ["INFO 10-03 12:00:00 [core.py:1] engine up"]
    arms = [("warmup:sub", "A", True), ("r0:A", "A", False), ("r0:B", "B", False)]
    arms += [("r0:A'", "A'", False), ("r0:D", "D", False), ("flush:1", "flush", True)]
    for k, (name, arm, warm) in enumerate(arms):
        rid = f"chatcmpl-{k:04x}"
        ct = a_tokens if arm == "A" and not warm else 40
        tag = "t71a.abcd1234.b64.0.B" if arm == "B" else None
        ids = [1000 + (j % 7) for j in range(39)] + [1]
        rows.append(
            {
                "name": name,
                "stratum": 64,
                "round": -1 if warm else 0,
                "arm": arm,
                "warmup": warm,
                "record_tag": tag,
                "status": 200,
                "wall_ms": 200.0,
                "response": {"id": rid, "usage": {"completion_tokens": ct}},
            }
        )
        frames.append(
            {
                "type": "astra.canvas.step",
                "request_id": rid + SUFFIX,
                "canvas_idx": 0,
                "step": STEPS,
                "max_steps": 48,
                "canvas_length": 256,
                "phase": "commit",
                "committed_tokens": ids,
            }
        )
        if arm != "A" or warm or release_for_a:
            log.append(
                f"INFO 10-03 12:00:01 [tap.py:9] canvas tap release: req_id={rid}{SUFFIX} submitted=10 "
                "published=10 dropped=0 errors=0 commit_overrun=0"
            )
        if tag:
            rec = {
                "record_tag": tag,
                "arm": "perstep_mar",
                "canvas_idx": 0,
                "committed_ids": ids,
                "state_set_after": [3],
                "in_final": True,
                "steps": b_record_steps,
                "bucket": 64,
                "stop_pos": 39,
                "error": None,
                "mean_accepted_per_step": 4.5,
                "shadow_argmax_ids": None,
                "argmax_in_language": None,
                "map_equals_argmax": None,
            }
            log.append(f"INFO 10-03 12:00:01 [record.py:3] ASTRA_CD {json.dumps(rec)}")
    return rows, frames, "\n".join(log) + "\n"


def test_check_rows_reads_steps_from_commit_frames_and_finds_nothing_on_clean_evidence() -> None:
    rows, frames, log = evidence()
    chk = mod().check_rows(rows, frames, log)
    measured = [i for i, r in enumerate(rows) if not r["warmup"]]
    assert chk.problems == {}, chk.problems
    assert {i: chk.steps[i] for i in measured} == {i: STEPS for i in measured}
    assert not set(chk.steps) - set(measured), "warm-up and flush rows are not measured"
    assert chk.t3_mismatches == [] and chk.run_problems == []
    assert mod().exit_code(chk.run_problems) == 0


def test_a_frame_step_record_steps_mismatch_is_a_run_problem() -> None:
    """T3: B's commit frame says step 9, its record says 10. A finding, not a tolerance: the row has a problem, the
    mismatch is listed, and it is a run problem (exit 1). Mutant killed: T3 check removed or only printed."""
    rows, frames, log = evidence(b_record_steps=STEPS + 1)
    chk = mod().check_rows(rows, frames, log)
    b = next(i for i, r in enumerate(rows) if r["arm"] == "B")
    assert chk.problems.get(b), chk.problems
    assert len(chk.t3_mismatches) == 1
    assert chk.run_problems and mod().exit_code(chk.run_problems) == 1
    assert set(chk.problems) == {b}, "only B carries a record"


def test_check_rows_flags_a_missing_release_line_and_a_short_commit_count() -> None:
    """A without its tap release line, then A with 300 completion tokens (2 canvases) but one commit frame: each is
    a problem on A's row only."""
    a = 1
    rows, frames, log = evidence(release_for_a=False)
    assert set(mod().check_rows(rows, frames, log).problems) == {a}
    rows, frames, log = evidence(a_tokens=300)
    assert set(mod().check_rows(rows, frames, log).problems) == {a}
