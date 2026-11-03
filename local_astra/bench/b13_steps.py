"""B13 per-step times from tap commit frames and request walls (T-0071a criterion 3). Pure, stdlib only.

Step time = (wall − F̂)/Σsteps:

- ``sum_steps``: Σsteps of one request from its ``astra.canvas.step`` commit frames. A commit frame's ``step``
  is the step the canvas had reached before the commit reset it (T-0011 R14, T-0069 O6: 0 is legal);
  denoise frames are ignored (the 30 Hz limiter drops some, never a commit).
- ``fit_fixed_cost``: F̂, the ordinary-least-squares intercept of wall on steps.
- ``rotation``: the gated arms' order per round, ``("A", "B", "A'")`` rotated left by ``round % 3``
  (T-0011 R17); where the informational D arm goes is the driver's choice (ruling T2).
- ``pair_by_round``: A, B and A′ values ordered by round index, so index ``k`` is the same round.
- ``evaluate_steps``: F̂ fitted on the A and A′ requests (or given by the caller), then
  ``b13_cd.evaluate`` (T-0070 E10) on per-step times at F̂ and at F̂ = 0; the verdict is the common one,
  else ``not_resolvable``.

The API is the test-author's ruling for T-0071a (``tests/unit/test_b13_steps.py``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

from local_astra.bench.b13_cd import B13Result, evaluate

__all__ = [
    "GATED_ARMS",
    "B13Steps",
    "RequestTiming",
    "evaluate_steps",
    "fit_fixed_cost",
    "pair_by_round",
    "per_step_times",
    "rotation",
    "sum_steps",
]

#: The gated arms in round 0's order: A and A′ unconstrained, B constrained, identical tap xargs.
GATED_ARMS: tuple[str, str, str] = ("A", "B", "A'")
COMMIT = "commit"

Verdict = Literal["pass", "fail", "not_resolvable"]


def sum_steps(frames: Iterable[Mapping[str, Any]]) -> int:
    """Σ ``step`` over one request's commit frames. ``ValueError`` when there is none (frames lost must
    never read as zero steps), when a ``canvas_idx`` is committed twice, or when the commit frames name
    more than one ``request_id``."""
    seen: set[int] = set()
    request_ids: set[str] = set()
    total = 0
    for f in frames:
        if f.get("phase") != COMMIT:
            continue
        idx = int(f["canvas_idx"])
        if idx in seen:
            raise ValueError(f"canvas {idx} has more than one commit frame")
        seen.add(idx)
        request_ids.add(str(f.get("request_id")))
        step = int(f["step"])
        if step < 0:
            raise ValueError(f"canvas {idx}: negative commit step {step}")
        total += step
    if not seen:
        raise ValueError("no commit frame: the request's steps are unknown, not zero")
    if len(request_ids) > 1:
        raise ValueError(f"commit frames of more than one request: {sorted(request_ids)}")
    return total


def fit_fixed_cost(steps: Sequence[int], wall: Sequence[float]) -> float:
    """The OLS intercept of ``wall`` on ``steps`` (same unit as ``wall``)."""
    n = len(steps)
    if n != len(wall):
        raise ValueError(f"{n} step counts for {len(wall)} walls")
    if n < 2:
        raise ValueError(f"{n} point(s): a line needs at least two")
    xs = [float(s) for s in steps]
    ys = [float(w) for w in wall]
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    sxx = sum((x - x_bar) ** 2 for x in xs)
    if sxx == 0.0:
        raise ValueError("every request has the same step count: the intercept is not identifiable")
    sxy = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys, strict=True))
    return y_bar - (sxy / sxx) * x_bar


def rotation(n_rounds: int) -> list[tuple[str, str, str]]:
    """Round ``i`` runs ``GATED_ARMS`` rotated left by ``i % 3``: ABA′, BA′A, A′AB, ..."""
    out: list[tuple[str, str, str]] = []
    for i in range(n_rounds):
        k = i % 3
        a, b, c = GATED_ARMS[k:] + GATED_ARMS[:k]
        out.append((a, b, c))
    return out


def pair_by_round(
    rows: Iterable[tuple[int, str, float]],
) -> tuple[list[float], list[float], list[float]]:
    """``(round_idx, arm, value)`` rows → the A, B and A′ values ordered by round index. Other arms are
    ignored; a round missing one of the three, or holding one twice, is a ``ValueError``."""
    by_round: dict[int, dict[str, float]] = {}
    for round_idx, arm, value in rows:
        if arm not in GATED_ARMS:
            continue
        slot = by_round.setdefault(int(round_idx), {})
        if arm in slot:
            raise ValueError(f"round {round_idx} has arm {arm} twice")
        slot[arm] = float(value)
    out: tuple[list[float], list[float], list[float]] = ([], [], [])
    for round_idx in sorted(by_round):
        slot = by_round[round_idx]
        missing = [arm for arm in GATED_ARMS if arm not in slot]
        if missing:
            raise ValueError(f"round {round_idx} lacks {missing}")
        for k, arm in enumerate(GATED_ARMS):
            out[k].append(slot[arm])
    return out


@dataclasses.dataclass(frozen=True)
class RequestTiming:
    """One measured request: its round, arm (``A``, ``B``, ``A'``, ``D``), wall and Σsteps."""

    round_idx: int
    arm: str
    wall: float
    steps: int


@dataclasses.dataclass(frozen=True)
class B13Steps:
    """``fixed`` = F̂; ``at_fit``/``at_zero`` = E10 on (wall − F̂)/steps resp. wall/steps."""

    verdict: Verdict
    fixed: float
    at_fit: B13Result
    at_zero: B13Result


def per_step_times(rows: Iterable[RequestTiming], fixed: float) -> list[tuple[int, str, float]]:
    """``(round_idx, arm, (wall − fixed)/steps)`` per row; a row with no steps is a ``ValueError``."""
    out: list[tuple[int, str, float]] = []
    for r in rows:
        if r.steps <= 0:
            raise ValueError(f"round {r.round_idx} arm {r.arm}: {r.steps} steps, no per-step time")
        out.append((r.round_idx, r.arm, (r.wall - fixed) / r.steps))
    return out


def evaluate_steps(
    rows: Iterable[RequestTiming], *, threshold_pct: float, fixed: float | None = None
) -> B13Steps:
    """F̂ (fitted on the A and A′ rows unless ``fixed`` is given, e.g. a fit over a whole run), then
    E10 at F̂ and at F̂ = 0, paired by round index."""
    rows = list(rows)
    if fixed is None:
        ref = [r for r in rows if r.arm in ("A", "A'")]
        fixed = fit_fixed_cost([r.steps for r in ref], [r.wall for r in ref])
    gated = [r for r in rows if r.arm in GATED_ARMS]

    def at(f: float) -> B13Result:
        a, b, ap = pair_by_round(per_step_times(gated, f))
        return evaluate(a, b, ap, threshold_pct=threshold_pct)

    at_fit, at_zero = at(fixed), at(0.0)
    verdict: Verdict = at_fit.verdict if at_fit.verdict == at_zero.verdict else "not_resolvable"
    return B13Steps(verdict=verdict, fixed=fixed, at_fit=at_fit, at_zero=at_zero)
