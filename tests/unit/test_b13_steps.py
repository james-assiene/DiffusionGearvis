"""T-0071a criterion 3's pure half: ``local_astra/bench/b13_steps.py`` (fast suite, stdlib only, no torch).

Step time = (wall − F̂)/Σsteps (T-0071 criterion 3): Σsteps from the request's tap commit frames, F̂ the fixed
per-request cost fitted by least squares (wall on steps, with an intercept) over every A and A′ request of the run,
the verdict computed by ``b13_cd.evaluate`` (T-0070 E10) at F̂ and at F̂ = 0, and ``not_resolvable`` when the two
disagree. Rounds are paired by round index and the arm order rotates per round over A, B, A′ (T-0011 R17).

Assumed API (test-author ruling for T-0071a, 2026-10-03; to be recorded in docs/tasks/T-0071a.md by its owner):

``sum_steps(frames: Iterable[Mapping[str, Any]]) -> int``
    ``frames`` are one request's tap wire dicts (``astra.canvas.step``: ``request_id``, ``canvas_idx``, ``step``,
    ``phase``). Returns Σ ``step`` over the ``phase == "commit"`` frames (a commit's ``step`` is the step the canvas
    had reached, ``AstraCanvasStep``; step 0 is legal, T-0069 O6); denoise frames are ignored. ``ValueError`` when
    there is no commit frame (the T-0069 O7 slow joiner: no frames must never read as zero steps), when two commit
    frames carry the same ``canvas_idx`` (double counting), or when the commit frames carry more than one
    ``request_id`` (frames are joined per request, never by time window).
``fit_fixed_cost(steps: Sequence[int], wall: Sequence[float]) -> float``
    The least-squares intercept of ``wall`` on ``steps`` (ordinary least squares with an intercept, same unit as
    ``wall``). ``ValueError`` on unequal lengths, fewer than two points, or all ``steps`` equal (no fit exists).
``rotation(n_rounds: int) -> list[tuple[str, str, str]]``
    The arm order of each round: round ``i`` is ``("A", "B", "A'")`` rotated left by ``i % 3`` -- ABA′, BA′A, A′AB,
    ABA′, ... (the gated arms; where D goes in a round is the driver's choice and is not fixed here).
``pair_by_round(rows: Iterable[tuple[int, str, float]]) -> tuple[list[float], list[float], list[float]]``
    ``(round_idx, arm, value)`` rows in any order → the A, B and A′ values, each ordered by ascending round index,
    so index ``k`` of the three lists is the same round. Rows of any other arm (D) are ignored. ``ValueError`` when a
    round lacks one of A, B, A′ or has one twice (no silent shrinking of the pairs).
``RequestTiming(round_idx: int, arm: str, wall: float, steps: int)``
    One measured request (arm in ``"A"``, ``"B"``, ``"A'"``, ``"D"``); constructed with keywords.
``evaluate_steps(rows: Iterable[RequestTiming], *, threshold_pct: float) -> result``
    ``result.fixed`` = F̂ fitted on the A and A′ rows only; ``result.at_fit`` / ``result.at_zero`` =
    ``b13_cd.evaluate(A, B, A′, threshold_pct=...)`` (``B13Result``) on per-step times (wall − F̂)/steps resp.
    wall/steps, paired by round index; ``result.verdict`` = the common verdict when the two agree, else
    ``"not_resolvable"``.

Expected values are hand-computed from the generating model ``wall = F + steps × per_step`` (noise-free rows make
the fit exact), with the T-0071 evidence numbers (n = 1, not measurements): stock ≈ 18.5 ms per step, ≈ 47 ms fixed
per request, Sudoku ``perstep_mar`` at bucket 64 ≈ 72 ms per step. n = 30 rounds (the a2 regime), seed 0 where noise
is drawn.
"""

from __future__ import annotations

import collections
import importlib
import random
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

import pytest

from local_astra.bench import b13_cd

N = 30
F = 47.0  # ms, fixed per request
A_STEP = 18.5  # ms per step, unconstrained
ARMS = ("A", "B", "A'")


def mod() -> ModuleType:
    try:
        return importlib.import_module("local_astra.bench.b13_steps")
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("local_astra.bench.b13_steps"):
            pytest.fail(
                f"T-0071a not implemented: local_astra.bench.b13_steps does not exist ({exc})"
            )
        raise


def frame(
    canvas_idx: int, step: int, phase: str = "commit", request_id: str = "chatcmpl-1-0a1b2c3d"
) -> dict[str, Any]:
    f: dict[str, Any] = {
        "type": "astra.canvas.step",
        "request_id": request_id,
        "canvas_idx": canvas_idx,
        "step": step,
        "max_steps": 48,
        "canvas_length": 256,
        "phase": phase,
    }
    if phase == "commit":
        f["committed_tokens"] = [1000 + step, 1]
    else:
        f["canvas"] = [0] * 256
        f["argmax"] = [0] * 256
        f["stable"] = "00" * 32
    return f


def rows_from(
    *,
    a_steps: Callable[[int], int],
    b_steps: Callable[[int], int],
    a_step_ms: float,
    b_step_ms: float,
    b_extra_fixed: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
    with_d: bool = False,
) -> list[Any]:
    """30 rounds of A, B, A′ (and D) generated as ``wall = F + steps × per_step`` (B: + ``b_extra_fixed``),
    each wall times ``1 + U(-noise, noise)``. A′ is A's twin (same steps, same cost)."""
    m = mod()
    rng = random.Random(seed)
    out = []
    for r in range(N):

        def j() -> float:
            return 1.0 + rng.uniform(-noise, noise)

        sa, sb = a_steps(r), b_steps(r)
        out.append(m.RequestTiming(round_idx=r, arm="A", wall=(F + sa * a_step_ms) * j(), steps=sa))
        out.append(
            m.RequestTiming(
                round_idx=r, arm="B", wall=(F + b_extra_fixed + sb * b_step_ms) * j(), steps=sb
            )
        )
        out.append(
            m.RequestTiming(round_idx=r, arm="A'", wall=(F + sa * a_step_ms) * j(), steps=sa)
        )
        if with_d:
            out.append(m.RequestTiming(round_idx=r, arm="D", wall=0.9 * sa * a_step_ms, steps=sa))
    return out


def cycle(*vals: int) -> Callable[[int], int]:
    return lambda r: vals[r % len(vals)]


# --------------------------------------------------------------------------- Σsteps from commit frames


def test_sum_steps_adds_the_commit_frames_steps_and_ignores_denoise_frames() -> None:
    frames = [frame(0, s, "denoise") for s in range(1, 13)] + [frame(0, 12)]
    frames += [frame(1, s, "denoise") for s in range(1, 8)] + [frame(1, 7)]
    frames += [frame(2, 5)]  # a canvas whose denoise frames the 30 Hz limiter dropped
    assert mod().sum_steps(frames) == 12 + 7 + 5


def test_sum_steps_counts_a_step_zero_commit_as_zero_not_as_an_error() -> None:
    """T-0069 O6: a commit may carry step 0 (a CD-ended slot); it adds nothing and is not refused."""
    assert mod().sum_steps([frame(0, 9), frame(1, 0)]) == 9


def test_sum_steps_does_not_depend_on_frame_order() -> None:
    frames = [frame(2, 4), frame(0, 11), frame(1, 6, "denoise"), frame(1, 6)]
    assert mod().sum_steps(frames) == mod().sum_steps(list(reversed(frames))) == 21


def test_sum_steps_refuses_a_request_without_commit_frames() -> None:
    """The O7 slow joiner: a subscriber that lost the frames must not report 0 steps (a division by zero, or a
    request silently dropped from the fit)."""
    with pytest.raises(ValueError):
        mod().sum_steps([frame(0, s, "denoise") for s in range(1, 6)])
    with pytest.raises(ValueError):
        mod().sum_steps([])


def test_sum_steps_refuses_a_duplicated_commit_frame() -> None:
    with pytest.raises(ValueError):
        mod().sum_steps([frame(0, 12), frame(1, 7), frame(1, 7)])


def test_sum_steps_refuses_commit_frames_of_two_requests() -> None:
    with pytest.raises(ValueError):
        mod().sum_steps([frame(0, 12), frame(1, 7, request_id="chatcmpl-2-ffffffff")])


# --------------------------------------------------------------------------- F̂


def test_fixed_cost_fit_recovers_a_known_intercept_exactly_on_a_noise_free_line() -> None:
    steps = [3, 4, 5, 9, 17, 30, 48]
    wall = [F + s * A_STEP for s in steps]
    assert mod().fit_fixed_cost(steps, wall) == pytest.approx(F, abs=1e-9)


def test_fixed_cost_fit_is_an_intercept_not_a_ratio_or_a_minimum() -> None:
    """Mutants: a fit through the origin (F̂ = 0), min(wall) (150 here), min(wall/steps) ... A line with intercept
    120 and slope 3 matches none of those."""
    steps = [10, 20, 40]
    wall = [150.0, 180.0, 240.0]  # 120 + 3 s
    assert mod().fit_fixed_cost(steps, wall) == pytest.approx(120.0, abs=1e-9)


def test_fixed_cost_fit_is_least_squares_on_noisy_points() -> None:
    """Three points off a line: the OLS intercept, worked by hand. x = 0, 1, 2; y = 1, 2, 4:
    x̄ = 1, ȳ = 7/3, Sxy = 3, Sxx = 2 → slope 1.5, intercept 7/3 − 1.5 = 5/6."""
    assert mod().fit_fixed_cost([0, 1, 2], [1.0, 2.0, 4.0]) == pytest.approx(5 / 6, abs=1e-12)


def test_fixed_cost_fit_recovers_the_intercept_under_noise() -> None:
    rng = random.Random(0)
    steps = [rng.randint(2, 48) for _ in range(60)]
    wall = [F + s * A_STEP + rng.uniform(-1.0, 1.0) for s in steps]
    assert mod().fit_fixed_cost(steps, wall) == pytest.approx(F, abs=0.75)


@pytest.mark.parametrize(
    ("steps", "wall"),
    [([4, 4, 4], [121.0, 122.0, 120.0]), ([4], [121.0]), ([], []), ([3, 4], [100.0])],
    ids=["all-steps-equal", "one-point", "empty", "unequal-lengths"],
)
def test_fixed_cost_fit_refuses_an_unidentifiable_fit(steps: list[int], wall: list[float]) -> None:
    with pytest.raises(ValueError):
        mod().fit_fixed_cost(steps, wall)


# --------------------------------------------------------------------------- rotation and pairing


def test_rotation_starts_aba_and_rotates_left_each_round() -> None:
    got = mod().rotation(6)
    want = [ARMS, ("B", "A'", "A"), ("A'", "A", "B")] * 2
    assert [tuple(x) for x in got] == want


def test_rotation_covers_the_three_orders_equally_over_thirty_rounds() -> None:
    got = [tuple(x) for x in mod().rotation(N)]
    assert len(got) == N
    assert collections.Counter(got) == {ARMS: 10, ("B", "A'", "A"): 10, ("A'", "A", "B"): 10}
    for pos in range(3):
        per_pos = collections.Counter(order[pos] for order in got)
        assert per_pos == {"A": 10, "B": 10, "A'": 10}, f"position {pos}: {per_pos}"
    assert all(got[i] != got[i + 1] for i in range(N - 1)), "consecutive rounds share an order"


def test_rotation_for_the_report_only_strata() -> None:
    """Criterion 4: n = 5 rounds at buckets 256 and 512 -- the schedule's prefix, not a different one."""
    assert [tuple(x) for x in mod().rotation(5)] == [tuple(x) for x in mod().rotation(N)[:5]]


def test_pairing_is_by_round_index_not_by_arrival_order() -> None:
    rows = [
        (r, arm, r + off) for r in range(N) for arm, off in zip(ARMS, (0.1, 0.2, 0.3), strict=True)
    ]
    rows += [(r, "D", -1.0) for r in range(N)]
    random.Random(0).shuffle(rows)
    a, b, ap = mod().pair_by_round(rows)
    assert list(a) == pytest.approx([r + 0.1 for r in range(N)])
    assert list(b) == pytest.approx([r + 0.2 for r in range(N)])
    assert list(ap) == pytest.approx([r + 0.3 for r in range(N)])


def test_pairing_follows_round_index_even_when_rounds_are_not_contiguous() -> None:
    rows = [
        (7, "B", 2.0),
        (3, "A", 10.0),
        (7, "A", 1.0),
        (3, "A'", 30.0),
        (3, "B", 20.0),
        (7, "A'", 3.0),
    ]
    assert tuple(list(x) for x in mod().pair_by_round(rows)) == (
        [10.0, 1.0],
        [20.0, 2.0],
        [30.0, 3.0],
    )


@pytest.mark.parametrize("problem", ["missing-B", "duplicate-A"])
def test_pairing_refuses_an_incomplete_or_duplicated_round(problem: str) -> None:
    rows = [(r, arm, 1.0) for r in range(4) for arm in ARMS]
    if problem == "missing-B":
        rows.remove((2, "B", 1.0))
    else:
        rows.append((1, "A", 1.5))
    with pytest.raises(ValueError):
        mod().pair_by_round(rows)


# --------------------------------------------------------------------------- the verdict at F̂ and at F̂ = 0


def test_the_fit_uses_the_a_and_a_prime_requests_only() -> None:
    """B carries a 453 ms extra fixed cost and D a different line altogether: F̂ must still be 47 exactly."""
    rows = rows_from(
        a_steps=cycle(3, 4, 5),
        b_steps=cycle(30, 42, 47),
        a_step_ms=A_STEP,
        b_step_ms=72.15,
        b_extra_fixed=453.0,
        with_d=True,
    )
    res = mod().evaluate_steps(rows, threshold_pct=10.0)
    assert res.fixed == pytest.approx(F, abs=1e-9)


def test_true_290_pct_per_step_overhead_reads_3_9x_not_the_naive_2_4x() -> None:
    """Adversarial (task file): A at ≈ 4 steps (3/4/5, so F̂ is identifiable), B at 42, 47 ms fixed, B's true per
    step 3.9 × 18.5 = 72.15 ms (+290 %). Naive wall/steps: B (47 + 42·72.15)/42 = 73.27, A at 4 steps
    (47 + 74)/4 = 30.25 → ×2.42 (+142 %, the median round). The module must read +290 % at F̂ and report the naive
    +142 % at F̂ = 0; both are far above 10 %, so the verdict is ``fail``."""
    rows = rows_from(
        a_steps=cycle(3, 4, 5), b_steps=lambda r: 42, a_step_ms=A_STEP, b_step_ms=3.9 * A_STEP
    )
    res = mod().evaluate_steps(rows, threshold_pct=10.0)
    assert res.fixed == pytest.approx(F, abs=1e-9)
    assert res.at_fit.delta_pct == pytest.approx(290.0, abs=1e-6), res.at_fit
    naive_b = (F + 42 * 3.9 * A_STEP) / 42
    naive_a4 = (F + 4 * A_STEP) / 4
    assert res.at_zero.delta_pct == pytest.approx(100 * (naive_b / naive_a4 - 1), abs=1e-6)
    assert 140 < res.at_zero.delta_pct < 145
    assert res.at_fit.verdict == res.at_zero.verdict == res.verdict == "fail"


def test_the_per_step_times_handed_to_e10_are_wall_minus_f_hat_over_steps() -> None:
    """``at_fit`` is ``b13_cd.evaluate`` on the per-step times worked by hand: A and A′ 18.5 ms, B 72.15 ms."""
    rows = rows_from(
        a_steps=cycle(3, 4, 5), b_steps=lambda r: 42, a_step_ms=A_STEP, b_step_ms=3.9 * A_STEP
    )
    res = mod().evaluate_steps(rows, threshold_pct=10.0)
    want = b13_cd.evaluate([A_STEP] * N, [3.9 * A_STEP] * N, [A_STEP] * N, threshold_pct=10.0)
    assert isinstance(res.at_fit, b13_cd.B13Result)
    assert res.at_fit.n == N and res.at_fit.threshold_pct == 10.0
    for field in ("delta_pct", "ci_low_pct", "ci_high_pct", "control_pct"):
        assert getattr(res.at_fit, field) == pytest.approx(getattr(want, field), abs=1e-6), field


def test_the_true_290_pct_overhead_survives_one_pct_noise() -> None:
    rows = rows_from(
        a_steps=cycle(3, 4, 5),
        b_steps=lambda r: 42,
        a_step_ms=A_STEP,
        b_step_ms=3.9 * A_STEP,
        noise=0.01,
        seed=0,
    )
    res = mod().evaluate_steps(rows, threshold_pct=10.0)
    assert 270 < res.at_fit.delta_pct < 310, res.at_fit
    assert res.verdict == "fail"


@pytest.mark.parametrize(
    ("a_steps", "b_steps", "true_pct", "threshold", "at_fit", "at_zero"),
    [
        # B has more steps: the naive reading dilutes B's fixed cost more than A's, so it under-reads.
        # +20 % true (fail at 15); naive: B 22.2 + 47/48 = 23.18 vs A 18.5 + 47/s, s ∈ {16, 20, 24}: +8.1..+13.3 %.
        ((16, 20, 24), 48, 20.0, 15.0, "fail", "pass"),
        # B has fewer steps: the naive reading over-reads. +5 % true (pass at 15); naive: B 19.425 + 47/8 = 25.30
        # vs A 18.5 + 47/s, s ∈ {36, 40, 44}: +28.3..+28.9 %.
        ((36, 40, 44), 8, 5.0, 15.0, "pass", "fail"),
    ],
    ids=["fit-fails-naive-passes", "fit-passes-naive-fails"],
)
def test_f_hat_and_f_zero_on_opposite_sides_of_the_threshold_is_not_resolvable(
    a_steps: Sequence[int],
    b_steps: int,
    true_pct: float,
    threshold: float,
    at_fit: str,
    at_zero: str,
) -> None:
    rows = rows_from(
        a_steps=cycle(*a_steps),
        b_steps=lambda r: b_steps,
        a_step_ms=A_STEP,
        b_step_ms=(1 + true_pct / 100) * A_STEP,
    )
    res = mod().evaluate_steps(rows, threshold_pct=threshold)
    assert res.at_fit.delta_pct == pytest.approx(true_pct, abs=1e-6)
    assert (res.at_fit.verdict, res.at_zero.verdict) == (at_fit, at_zero), (res.at_fit, res.at_zero)
    assert res.verdict == "not_resolvable"


def test_agreeing_pass_at_both_f_hat_and_zero_is_a_pass() -> None:
    """+2 % true at bucket 64 (10 %): pass at F̂; at F̂ = 0 the naive reading is negative (B's fixed cost is spread
    over 42 steps) and passes too. A module that always answers ``not_resolvable`` fails here."""
    rows = rows_from(
        a_steps=cycle(3, 4, 5), b_steps=lambda r: 42, a_step_ms=A_STEP, b_step_ms=1.02 * A_STEP
    )
    res = mod().evaluate_steps(rows, threshold_pct=10.0)
    assert res.at_fit.delta_pct == pytest.approx(2.0, abs=1e-6)
    assert res.at_fit.verdict == res.at_zero.verdict == res.verdict == "pass"


def test_a_four_pct_per_request_xargs_cost_on_b_only_is_not_per_step_overhead() -> None:
    """Adversarial (task file; T-0011 R17: ``vllm_xargs`` ≈ 4 % end to end). B's per step equals A's (0 % true
    overhead); B alone pays a per-request cost of 4 % of an A request's wall, 0.04 × (47 + 4 × 18.5) = 4.84 ms.
    After subtraction it is spread over B's 42 steps: 4.84/42/18.5 = +0.62 %, not +4 %. A wall-ratio metric reads
    B/A ≈ ×6.9, a naive wall/steps one ≈ −35 %."""
    extra = 0.04 * (F + 4 * A_STEP)
    rows = rows_from(
        a_steps=cycle(3, 4, 5),
        b_steps=lambda r: 42,
        a_step_ms=A_STEP,
        b_step_ms=A_STEP,
        b_extra_fixed=extra,
    )
    res = mod().evaluate_steps(rows, threshold_pct=10.0)
    assert res.at_fit.delta_pct == pytest.approx(100 * extra / 42 / A_STEP, abs=1e-6)
    assert abs(res.at_fit.delta_pct) < 1.0
    assert res.verdict == "pass"


def test_the_verdict_does_not_depend_on_the_order_rows_arrive_in() -> None:
    """Rounds drift (per-step cost moves ±20 % between rounds, shared by the round's three arms); B is +5 %
    per step in its own round. Paired by round index, every ratio is close to +5 %; zipped by arrival order after
    a shuffle, the ratios would mix rounds."""
    m = mod()
    rows = []
    for r in range(N):
        drift = 1.0 + 0.2 * ((r * 7) % 11 - 5) / 5
        sa = (3, 4, 5)[r % 3]
        rows.append(m.RequestTiming(round_idx=r, arm="A", wall=F + sa * A_STEP * drift, steps=sa))
        rows.append(
            m.RequestTiming(round_idx=r, arm="B", wall=F + 42 * 1.05 * A_STEP * drift, steps=42)
        )
        rows.append(m.RequestTiming(round_idx=r, arm="A'", wall=F + sa * A_STEP * drift, steps=sa))
    ordered = m.evaluate_steps(rows, threshold_pct=10.0)
    shuffled_rows = list(rows)
    random.Random(0).shuffle(shuffled_rows)
    shuffled = m.evaluate_steps(shuffled_rows, threshold_pct=10.0)
    assert shuffled.fixed == pytest.approx(ordered.fixed, abs=1e-9)
    for name in ("at_fit", "at_zero"):
        x, y = getattr(ordered, name), getattr(shuffled, name)
        assert (x.verdict, x.n) == (y.verdict, y.n)
        assert x.delta_pct == pytest.approx(y.delta_pct, abs=1e-9)
        assert x.ci_low_pct == pytest.approx(y.ci_low_pct, abs=1e-9)
        assert x.ci_high_pct == pytest.approx(y.ci_high_pct, abs=1e-9)
    assert shuffled.verdict == ordered.verdict


# --------------------------------------------------------------------------- review round 1 (rulings R1-R6)


def test_a_given_fixed_cost_is_used_as_is_and_never_refitted() -> None:
    """``evaluate_steps(..., fixed=30)`` on rows whose own fit is 47 (noise-free +290 % rows): ``fixed`` is 30 and
    ``at_fit`` is E10 on (wall − 30)/steps, worked from the generating model: A and A′ 18.5 + 17/s (s = 3, 4, 5 by
    round), B 72.15 + 17/42. Mutant killed: "ignore ``fixed`` and refit" (reads 47 and +290 % exactly)."""
    rows = rows_from(
        a_steps=cycle(3, 4, 5), b_steps=lambda r: 42, a_step_ms=A_STEP, b_step_ms=3.9 * A_STEP
    )
    res = mod().evaluate_steps(rows, threshold_pct=10.0, fixed=30.0)
    assert res.fixed == 30.0
    a = [A_STEP + (F - 30.0) / (3, 4, 5)[r % 3] for r in range(N)]
    b = [3.9 * A_STEP + (F - 30.0) / 42 for _ in range(N)]
    want = b13_cd.evaluate(a, b, list(a), threshold_pct=10.0)
    for field in ("delta_pct", "ci_low_pct", "ci_high_pct", "control_pct"):
        assert getattr(res.at_fit, field) == pytest.approx(getattr(want, field), abs=1e-6), field
    assert res.at_fit.delta_pct != pytest.approx(290.0, abs=1.0), "the rows' own F̂ was used"
    zero = b13_cd.evaluate(
        [A_STEP + F / (3, 4, 5)[r % 3] for r in range(N)],
        [3.9 * A_STEP + F / 42] * N,
        [A_STEP + F / (3, 4, 5)[r % 3] for r in range(N)],
        threshold_pct=10.0,
    )
    assert res.at_zero.delta_pct == pytest.approx(zero.delta_pct, abs=1e-6)


def test_per_step_times_refuses_a_request_with_zero_steps() -> None:
    """Σsteps 0 has no per-step time (a division by zero or an infinite per-step cost, never a number). Mutant
    killed: dropping the guard (ZeroDivisionError) or skipping the row silently."""
    m = mod()
    rows = [
        m.RequestTiming(round_idx=0, arm="A", wall=121.0, steps=4),
        m.RequestTiming(round_idx=0, arm="B", wall=100.0, steps=0),
    ]
    with pytest.raises(ValueError):
        m.per_step_times(rows, 47.0)
    assert m.per_step_times(rows[:1], 47.0) == [(0, "A", pytest.approx((121.0 - 47.0) / 4))]
