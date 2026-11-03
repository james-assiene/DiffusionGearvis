"""The a2 B13 driver's pure logic (T-0071a rulings R1-R4, R7): schedule, row checks, the gate, the evaluation.

``scripts/gpu/cd_b13.py`` only sends requests and writes evidence; every decision is taken here, on its evidence
(``requests.jsonl``-shaped rows, tap wire frames, the engine-log text). No I/O, no clocks, no torch.

- ``schedule``: round ``r`` = ``b13_steps.rotation`` order with D (or E) at index ``r % 4`` (ruling T2).
- ``fit_fixed_cost_se``: F̂ and its OLS standard error ``s·sqrt(1/n + x̄²/Sxx)``, ``s² = RSS/(n − 2)``.
- ``gate`` (R1 + R2): one gated stratum, F̂/SE from its own A and A′; E10 at F̂, at 0 and at F̂⁺ = F̂ + 2·SE;
  ``combine_verdicts``: PASS only if all three pass, FAIL if F̂ and 0 both fail, else ``not_resolvable``. Never
  raises on data.
- ``check_rows``: Σsteps from commit frames, commit count, release line, records and ruling T3 per measured row.
- ``evaluate_b13``: rounds with a problem in A, B or A′ are dropped (named); a D/E problem drops D/E only; one gate
  per gated stratum; the pooled F̂ is a sensitivity value only; a crash or a stratum build error is a run problem.
- ``exit_code``: 0 without run problems, else 1 (D-H1: never the verdict).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from local_astra.bench import b13_cd, b13_steps
from local_astra.bench.b16_tap import parse_tap_release_lines
from local_astra.contracts.cd import CdCanvasRecord
from local_astra.eval.cd import smoke

__all__ = [
    "CANVAS",
    "MEASURED_ARMS",
    "B13Evaluation",
    "GateResult",
    "RowCheck",
    "StratumSpec",
    "check_rows",
    "combine_verdicts",
    "evaluate_b13",
    "exit_code",
    "fit_fixed_cost_se",
    "gate",
    "parse_records",
    "schedule",
]

CANVAS = 256
MEASURED_ARMS = ("A", "A'", "B", "D", "E")
GATED_ARMS = b13_steps.GATED_ARMS
RECORD_MARK = "ASTRA_CD "
#: the CD mode a constrained arm's records name
ARM_MODE = {"B": "perstep_mar", "E": "end_projection"}


# --------------------------------------------------------------------------- schedule and F̂


def schedule(n_rounds: int, extra: str) -> list[list[str]]:
    """Round ``r``: the gated rotation with ``extra`` (D or E) inserted at index ``r % 4`` (ruling T2)."""
    out: list[list[str]] = []
    for r, order in enumerate(b13_steps.rotation(n_rounds)):
        o = list(order)
        o.insert(r % 4, extra)
        out.append(o)
    return out


def fit_fixed_cost_se(steps: Sequence[int], wall: Sequence[float]) -> tuple[float, float]:
    """``(F̂, SE(F̂))``: the OLS intercept of ``wall`` on ``steps`` and its standard error."""
    n = len(steps)
    if n != len(wall):
        raise ValueError(f"{n} step counts for {len(wall)} walls")
    if n < 3:
        raise ValueError(f"{n} point(s): an intercept's standard error needs at least three")
    xs = [float(s) for s in steps]
    ys = [float(w) for w in wall]
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    sxx = sum((x - x_bar) ** 2 for x in xs)
    if sxx == 0.0:
        raise ValueError("every request has the same step count: the intercept is not identifiable")
    slope = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys, strict=True)) / sxx
    intercept = y_bar - slope * x_bar
    rss = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys, strict=True))
    s2 = rss / (n - 2)
    return intercept, math.sqrt(s2 * (1.0 / n + x_bar**2 / sxx))


# --------------------------------------------------------------------------- R1 + R2: the gate


def combine_verdicts(at_fit: str, at_zero: str, at_upper: str) -> str:
    """R1: PASS only if all three pass; FAIL if F̂ and F̂ = 0 both fail; otherwise ``not_resolvable``."""
    if at_fit == at_zero == at_upper == "pass":
        return "pass"
    if at_fit == at_zero == "fail":
        return "fail"
    return "not_resolvable"


@dataclasses.dataclass(frozen=True)
class GateResult:
    """One gated stratum. ``at_*`` is ``None`` when E10 cannot be computed there (no F̂, a non-positive A)."""

    verdict: str
    fixed: float | None
    se: float | None
    at_fit: b13_cd.B13Result | None
    at_zero: b13_cd.B13Result | None
    at_upper: b13_cd.B13Result | None
    why: str = ""


def _e10(
    rows: Sequence[b13_steps.RequestTiming], fixed: float | None, threshold_pct: float
) -> tuple[b13_cd.B13Result | None, str]:
    if fixed is None:
        return None, "no F̂"
    try:
        a, b, ap = b13_steps.pair_by_round(b13_steps.per_step_times(rows, fixed))
        return b13_cd.evaluate(a, b, ap, threshold_pct=threshold_pct), ""
    except ValueError as exc:
        return None, str(exc)


def gate(rows: Iterable[b13_steps.RequestTiming], *, threshold_pct: float) -> GateResult:
    """F̂ and SE(F̂) from this stratum's own A and A′ (R2); E10 at F̂, 0 and F̂ + 2·SE (R1)."""
    gated = [r for r in rows if r.arm in GATED_ARMS]
    ref = [r for r in gated if r.arm in ("A", "A'")]
    fixed: float | None
    se: float | None
    why: list[str] = []
    try:
        fixed, se = fit_fixed_cost_se([r.steps for r in ref], [r.wall for r in ref])
    except ValueError as exc:
        fixed = se = None
        why.append(f"F̂: {exc}")
    at_fit, w1 = _e10(gated, fixed, threshold_pct)
    at_zero, w0 = _e10(gated, 0.0, threshold_pct)
    upper = None if fixed is None or se is None else fixed + 2.0 * se
    at_upper, wu = _e10(gated, upper, threshold_pct)
    for label, w in (("at F̂", w1), ("at 0", w0), ("at F̂⁺", wu)):
        if w and w != "no F̂":
            why.append(f"{label}: {w}")

    def v(x: b13_cd.B13Result | None) -> str:
        return x.verdict if x is not None else "not_resolvable"

    verdict = combine_verdicts(v(at_fit), v(at_zero), v(at_upper))
    return GateResult(verdict, fixed, se, at_fit, at_zero, at_upper, "; ".join(why))


# --------------------------------------------------------------------------- evidence checks


@dataclasses.dataclass(frozen=True)
class RowCheck:
    problems: dict[int, list[str]]
    steps: dict[int, int]
    t3_mismatches: list[str]
    run_problems: list[str]
    t3_canvases: int = 0
    malformed_records: int = 0


def parse_records(engine_log: str) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """``ASTRA_CD <json>`` lines validated as ``CdCanvasRecord``, by ``record_tag``; and the malformed count."""
    by_tag: dict[str, list[dict[str, Any]]] = {}
    malformed = 0
    for line in engine_log.splitlines():
        if RECORD_MARK not in line:
            continue
        try:
            rec = CdCanvasRecord.model_validate_json(line.split(RECORD_MARK, 1)[1])
        except ValidationError:
            malformed += 1
            continue
        if rec.record_tag is not None:
            by_tag.setdefault(rec.record_tag, []).append(rec.model_dump(mode="json"))
    return by_tag, malformed


def _response_id(row: Mapping[str, Any]) -> str | None:
    body = row.get("response")
    if row.get("status", 200) == 200 and isinstance(body, Mapping) and body.get("id"):
        return str(body["id"])
    return None


def _completion_tokens(row: Mapping[str, Any]) -> int | None:
    try:
        return int(row["response"]["usage"]["completion_tokens"])
    except (KeyError, TypeError, ValueError):
        return None


def check_rows(
    rows: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
    engine_log: str,
) -> RowCheck:
    """Per measured row: Σsteps from its commit frames, commit count = ceil(completion_tokens/256), a tap release
    line with ``commit_overrun == 0``, and for a row with a ``record_tag`` the records' checks and ruling T3
    (frame ``step`` == record ``steps`` per canvas; every mismatch is a run problem)."""
    by_tag, malformed = parse_records(engine_log)
    try:
        releases: list[dict[str, str | int]] = parse_tap_release_lines(engine_log)
        release_error = ""
    except ValueError as exc:
        releases, release_error = [], str(exc)
    problems: dict[int, list[str]] = {}
    steps: dict[int, int] = {}
    mismatches: list[str] = []
    run_problems: list[str] = []
    if release_error:
        run_problems.append(f"unreadable tap release line: {release_error}")
    if malformed:
        run_problems.append(f"{malformed} ASTRA_CD lines did not validate as CdCanvasRecord")
    compared = 0
    for i, row in enumerate(rows):
        if row.get("warmup") or row.get("arm") not in MEASURED_ARMS:
            continue
        name = str(row.get("name", i))
        rid, ct = _response_id(row), _completion_tokens(row)
        if rid is None or ct is None:
            problems[i] = [f"HTTP {row.get('status')}: {str(row.get('response'))[:200]}"]
            continue
        p: list[str] = []
        mine = smoke.frames_for(frames, rid)
        commits = sorted(
            (f for f in mine if f.get("phase") == "commit"), key=lambda f: int(f["canvas_idx"])
        )
        want = math.ceil(ct / CANVAS)
        if len(commits) != want:
            p.append(f"{len(commits)} commit frames, want ceil({ct}/{CANVAS}) = {want}")
        try:
            total = b13_steps.sum_steps(mine)
            steps[i] = total
            if total <= 0:
                p.append("Σsteps is 0: no per-step time")
        except ValueError as exc:
            p.append(f"Σsteps: {exc}")
        p += smoke.release_problems(releases, rid)
        p += smoke.frame_problems(mine)
        tag = row.get("record_tag")
        if tag:
            recs = by_tag.get(str(tag), [])
            p += smoke.record_problems(
                recs, record_tag=str(tag), arm=ARM_MODE.get(str(row["arm"]), "perstep_mar")
            )
            p += smoke.commit_frame_problems(commits, recs)
            frame_step = {int(f["canvas_idx"]): int(f["step"]) for f in commits}
            for r in recs:
                idx = int(r["canvas_idx"])
                if idx not in frame_step:
                    continue
                compared += 1
                if frame_step[idx] != int(r["steps"]):
                    m = f"{name} canvas {idx}: frame step {frame_step[idx]} != record steps {r['steps']}"
                    mismatches.append(m)
                    run_problems.append(f"T3: {m}")
                    p.append(f"T3: {m}")
        if p:
            problems[i] = p
    return RowCheck(problems, steps, mismatches, run_problems, compared, malformed)


# --------------------------------------------------------------------------- the evaluation


@dataclasses.dataclass(frozen=True)
class StratumSpec:
    bucket: int
    gated: bool
    rounds: int
    build_error: str | None = None
    #: what the report names the stratum by (task, record id, |S|); free text
    label: str = ""
    #: the automaton payload size B (and E) carry; A and A′ carry none (R5)
    payload_chars: int | None = None


@dataclasses.dataclass(frozen=True)
class B13Evaluation:
    gates: dict[int, GateResult]
    pooled_fixed: float | None
    exposure: dict[int, b13_cd.B13Result]
    dropped: dict[int, list[int]]
    run_problems: list[str]
    lines: list[str]
    #: R2/R4: each gated stratum evaluated at the pooled F̂ (sensitivity only, never the verdict)
    pooled_sensitivity: dict[int, str] = dataclasses.field(default_factory=dict)


def _fmt(res: b13_cd.B13Result | None) -> str:
    if res is None:
        return "not computable"
    return (
        f"{res.verdict} (delta {res.delta_pct:+.2f} %, 95 % CI [{res.ci_low_pct:+.2f}, "
        f"{res.ci_high_pct:+.2f}] %, control A′/A {res.control_pct:.2f} %, n {res.n}, threshold "
        f"{res.threshold_pct:g} %)"
    )


def _timing(row: Mapping[str, Any], arm: str | None = None) -> b13_steps.RequestTiming:
    return b13_steps.RequestTiming(
        round_idx=int(row["round"]),
        arm=str(arm or row["arm"]),
        wall=float(row["wall_ms"]),
        steps=int(row["steps"]),
    )


def _side_vs_a(
    gated_rows: Sequence[b13_steps.RequestTiming],
    side: Mapping[int, b13_steps.RequestTiming],
    fixed: float,
    threshold_pct: float,
) -> b13_cd.B13Result:
    """D (or E) vs A per step at ``fixed``, paired by round, with the A/A′ control (E10's statistics)."""
    a = {t.round_idx: t for t in gated_rows if t.arm == "A"}
    ap = {t.round_idx: t for t in gated_rows if t.arm == "A'"}
    rounds = sorted(set(a) & set(ap) & set(side))
    return b13_cd.evaluate(
        [(a[r].wall - fixed) / a[r].steps for r in rounds],
        [(side[r].wall - fixed) / side[r].steps for r in rounds],
        [(ap[r].wall - fixed) / ap[r].steps for r in rounds],
        threshold_pct=threshold_pct,
    )


def evaluate_b13(
    rows: Sequence[Mapping[str, Any]],
    problems: Mapping[int, Sequence[str]],
    strata: Sequence[StratumSpec],
    thresholds: Mapping[int, float],
    *,
    crash: str | None = None,
) -> B13Evaluation:
    """Gate every gated stratum (R1/R2), report D's xargs exposure and the report-only strata; ``rows`` carry
    ``steps`` (``RowCheck.steps`` merged by the caller)."""
    run_problems: list[str] = []
    if crash:
        run_problems.append(f"crash: {crash}")
    gates: dict[int, GateResult] = {}
    exposure: dict[int, b13_cd.B13Result] = {}
    dropped: dict[int, list[int]] = {}
    timings: dict[int, list[b13_steps.RequestTiming]] = {}
    sides: dict[int, dict[int, b13_steps.RequestTiming]] = {}
    lines: list[str] = []

    for s in strata:
        name = s.label or f"b{s.bucket}"
        if s.build_error:
            run_problems.append(f"stratum b{s.bucket} ({name}) not run: {s.build_error}")
        mine = [
            (i, r)
            for i, r in enumerate(rows)
            if r.get("stratum") == s.bucket
            and not r.get("warmup")
            and r.get("arm") in MEASURED_ARMS
        ]
        by_round: dict[int, dict[str, int]] = {}
        for i, r in mine:
            by_round.setdefault(int(r["round"]), {})[str(r["arm"])] = i
        bad: dict[int, list[str]] = {}
        for rnd, arms in sorted(by_round.items()):
            for arm in GATED_ARMS:
                if arm not in arms:
                    bad.setdefault(rnd, []).append(f"{arm} missing")
                elif arms[arm] in problems:
                    bad.setdefault(rnd, []).append(f"{arm}: {problems[arms[arm]][0]}")
        dropped[s.bucket] = sorted(bad)
        for rnd, why in sorted(bad.items()):
            run_problems.append(f"b{s.bucket} round {rnd} dropped: {'; '.join(why)}")
        # R7 as clarified (N1): a D/E problem is a run problem too; it only removes that row from the figures
        for rnd, arms in sorted(by_round.items()):
            for arm, idx in sorted(arms.items()):
                if arm not in GATED_ARMS and idx in problems:
                    run_problems.append(
                        f"b{s.bucket} round {rnd} {arm} excluded from the figures: "
                        + "; ".join(problems[idx])
                    )
        timings[s.bucket] = [
            _timing(rows[idx])
            for rnd, arms in sorted(by_round.items())
            if rnd not in bad
            for arm, idx in arms.items()
            if arm in GATED_ARMS
        ]
        extra = "D" if s.gated else "E"
        sides[s.bucket] = {
            rnd: _timing(rows[arms[extra]])
            for rnd, arms in sorted(by_round.items())
            if rnd not in bad and extra in arms and arms[extra] not in problems
        }

    gated = [s for s in strata if s.gated]
    ref = [t for s in gated for t in timings[s.bucket] if t.arm in ("A", "A'")]
    pooled: float | None
    try:
        pooled = b13_steps.fit_fixed_cost([t.steps for t in ref], [t.wall for t in ref])
    except ValueError:
        pooled = None

    for s in strata:
        name = s.label or f"b{s.bucket}"
        payload = f"{s.payload_chars} chars" if s.payload_chars is not None else "unknown"
        lines.append(
            f"stratum {name}: {'gated' if s.gated else 'report-only'}, {s.rounds} rounds planned, "
            f"{len({t.round_idx for t in timings[s.bucket]})} kept, dropped {dropped[s.bucket] or 'none'}; "
            f"automaton payload on B{'/E' if not s.gated else ''}: {payload} (A and A′ carry none)"
        )

    lines.append("")
    lines.append(
        "xargs exposure (read before the gate): D (no xargs) vs A (xargs {astra_tap: true}) per step at the "
        "stratum's F̂, paired by round, against the A/A′ control (informational, not gated):"
    )
    for s in gated:
        g = gate(timings[s.bucket], threshold_pct=float(thresholds[s.bucket]))
        gates[s.bucket] = g
        name = s.label or f"b{s.bucket}"
        if g.fixed is None:
            lines.append(f"  {name}: not computed (no F̂: {g.why})")
            continue
        try:
            res = _side_vs_a(
                timings[s.bucket], sides[s.bucket], g.fixed, float(thresholds[s.bucket])
            )
            exposure[s.bucket] = res
            lines.append(
                f"  {name}: D vs A delta {res.delta_pct:+.2f} % (95 % CI [{res.ci_low_pct:+.2f}, "
                f"{res.ci_high_pct:+.2f}] %), control {res.control_pct:.2f} %, n {res.n}"
            )
        except ValueError as exc:
            lines.append(f"  {name}: not computed ({exc})")

    lines.append("")
    lines.append(
        "B13 gate (criterion 3; T-0062 decision 8; R1: PASS only if F̂, F̂ = 0 and F̂⁺ = F̂ + 2·SE all pass, FAIL if "
        "F̂ and F̂ = 0 both fail; R2: F̂ per stratum):"
    )
    sensitivity: dict[int, str] = {}
    for s in gated:
        g = gates[s.bucket]
        name = s.label or f"b{s.bucket}"
        lines.append(f"  b{s.bucket} [{name}]: VERDICT {g.verdict.upper()}")
        if g.fixed is not None and g.se is not None:
            lines.append(
                f"    F̂ = {g.fixed:.2f} ms, SE {g.se:.2f} ms, F̂⁺ = {g.fixed + 2 * g.se:.2f} ms"
            )
        lines.append(f"    at F̂:  {_fmt(g.at_fit)}")
        lines.append(f"    at 0:  {_fmt(g.at_zero)}")
        lines.append(f"    at F̂⁺: {_fmt(g.at_upper)}")
        if g.why:
            lines.append(f"    why: {g.why}")
        if pooled is not None:
            res_p, w = _e10(timings[s.bucket], pooled, float(thresholds[s.bucket]))
            sensitivity[s.bucket] = res_p.verdict if res_p is not None else f"not computable ({w})"
            lines.append(
                f"    sensitivity (never the verdict): at the pooled F̂ {pooled:.2f} ms -> {_fmt(res_p)}"
            )
    lines.append(
        "pooled F̂ over every gated A and A′ (sensitivity only): "
        + (f"{pooled:.2f} ms" if pooled is not None else "not identifiable")
    )

    lines.append("")
    lines.append("Report-only strata (criterion 4; never a verdict):")
    for s in strata:
        if s.gated:
            continue
        name = s.label or f"b{s.bucket}"
        ts = timings[s.bucket]
        if not ts:
            lines.append(f"  {name}: no complete round")
            continue
        try:
            own: float | None = b13_steps.fit_fixed_cost(
                [t.steps for t in ts if t.arm in ("A", "A'")],
                [t.wall for t in ts if t.arm in ("A", "A'")],
            )
        except ValueError:
            own = None
        for label, f in (("at its own F̂", own), ("at F̂ = 0", 0.0)):
            if f is None:
                lines.append(f"  {name} {label}: not computed (F̂ not identifiable)")
                continue
            try:
                res_b, w = _e10(ts, f, 0.0)
                res_e = _side_vs_a(ts, sides[s.bucket], f, 0.0)
                lines.append(
                    f"  {name} {label} ({f:.2f} ms): perstep_mar vs A "
                    + (
                        f"{res_b.delta_pct:+.1f} %"
                        if res_b is not None
                        else f"not computable ({w})"
                    )
                    + f", end_projection vs A {res_e.delta_pct:+.1f} %, control {res_e.control_pct:.1f} %, "
                    f"n {res_e.n}"
                )
            except ValueError as exc:
                lines.append(f"  {name} {label}: not computed ({exc})")

    return B13Evaluation(gates, pooled, exposure, dropped, run_problems, lines, sensitivity)


def exit_code(run_problems: Sequence[str]) -> int:
    """0 without run problems, else 1 (a run defect). Never the B13 verdict (D-H1)."""
    return 1 if run_problems else 0
