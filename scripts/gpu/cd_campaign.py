"""T-0071b -- the CD campaign's stack check and Mar calibration on the dev cuts. `/launch` ONLY.

    uv run --group cd python scripts/gpu/cd_campaign.py --task sudoku --cut dev \
        --base-url http://127.0.0.1:8001/v1 --engine-log logs/vllm.log --boot-log logs/vllm.log \
        --out-dir logs/t0071b-calib-sudoku-<sha> --repo . --expected-commit $(git rev-parse HEAD) \
        --freeze-path logs/t0071b-calib-sudoku-<sha>/calibration-sudoku.json \
        [--arm perstep_mar] [--arm perstep_mar_qargmax]

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never
boots one; ``scripts/locked/t0071b_calib.sh`` does (engine only). Importing this file opens no socket and
starts no run; the decision logic is in ``run_grid`` and ``choose`` (pure, injected cell runner and clock).

One launch = one task, concurrency 1, dev cut only (``--cut`` accepts ``dev`` and nothing else):

1. Pre-flight on the CPU, before any request: refusals (an existing freeze file, no boot in ``--boot-log``),
   then the dev cut's bucket histogram (JSON whitespace), printed and kept in the report. Then the engine
   health probe (``GET <base-url>/models``, loopback; C9) and, with ``--probe-diffusion``, the 400 probe.
2. Stack check (umbrella criterion 7): ``unconstrained`` on the whole dev cut (n = 100, ``--limit`` does not
   apply) through T-0070's runner, hard stops armed; (d) -- or any other stop -- ends the launch; (f), the
   header-prefix rate of the first 20 records, is reported.
3. Per calibrated arm (``--arm``, repeatable, one of ``calibration.GRID_ARMS``, default ``perstep_mar``;
   rulings C3/C4 -- an SC twin is refused, exit 2, before any request): the 18 cells of
   ``calibration.grid()`` on the dev cut (``--limit`` records each), every hard stop armed. An (e)-stopped cell
   is ineligible, its partial score (over the rows it wrote) kept, and the grid continues (D-H3); any other stop,
   or an ``internal`` CD error row, ends the launch with nothing frozen for that arm or any later one (C2). Per-cell
   wall time and an ETA are printed after each cell.
4. Freeze (criterion 3): ``calibration.select`` over the eligible cells, written by ``calibration.freeze``
   (read-only, T-0070) with ``task``, ``cut == "dev"``, ``arm``, the stock-knob reference row (the grid's
   (0.1, 0.005) cell) and the chosen cell's stats (mean steps per canvas from the rows; mean accepted per step
   from the engine log's CD records, joined on the rows' ``record_tag``, step-weighted). ``perstep_mar`` freezes
   at ``--freeze-path``; any other arm at ``<stem>-<arm><suffix>`` beside it. The file is read back with
   ``load_frozen(path, task=, arm=)``.
5. Each calibrated arm's SC twin (``REPORTED_AT_CHOICE``, C5) at that arm's chosen values on the same dev
   records (reported under the twin's name, never selected on).

Engine health (C9): the engine is probed before the stack check, before every cell and before every SC-twin
run, and any ``http_error`` or ``timeout`` row in the stack check, a cell or an SC-twin run ends the launch
with a stop starting "engine", the report written and nothing frozen for the arm in progress (exit 1). A dead
engine turns every request into such a row, which no hard stop (a)-(h) catches.

Budget: ``--budget-s`` (default ``BUDGET_S[task]``) per calibrated arm. Arm ``i`` (0-based) must finish by
``launch start + (i + 1) * budget``; the stack check counts against the first arm, and time an arm leaves
unused carries over. The budget running out ends the launch with a stop starting "g" and nothing frozen for
that arm (a partial grid selects nothing); the report is still written.

``<out-dir>/campaign-<task>.json`` is written in every outcome, before the exit code. Exit 0 iff every
requested arm was frozen; 1 otherwise (a stop, no eligible cell, a crash); 2 refused (argparse included).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from local_astra.eval.cd import calibration as _calibration
from local_astra.eval.cd import records as _records
from local_astra.eval.cd import runner as _runner

REPO = Path(__file__).resolve().parents[2]

#: The launch budget per task and per calibrated arm (D-H3: BFCL 6 h; Sudoku the runner's 4 h default,
#: against an estimated 0.3-1.8 h).
BUDGET_S: Mapping[str, float] = {"bfcl": 21_600.0, "sudoku": 14_400.0}
#: Stock DiffusionGemma knobs (generation_config.json): a grid cell, and the reference row (C2).
STOCK_KNOBS: Mapping[str, float] = {"entropy_bound": 0.1, "confidence_threshold": 0.005}
#: Calibrated arms -> the arm reported (never selected on) at the calibrated arm's chosen values.
REPORTED_AT_CHOICE: Mapping[str, str] = {
    "perstep_mar": "perstep_mar_sc",
    "perstep_mar_qargmax": "perstep_mar_qargmax_sc",
}
DEFAULT_ARMS = ("perstep_mar",)
SCORE_KEY = {"bfcl": "exact_call", "sudoku": "solved"}
EXIT_FROZEN, EXIT_NOT_FROZEN, EXIT_REFUSED = 0, 1, 2
INTERNAL = "internal"
#: Row errors that mean the engine did not answer (C9): never a score of 0, always the end of the launch.
ENGINE_ERRORS = ("http_error", "timeout")
RULE = "max score; ties: fewer mean steps, then larger entropy_bound; (e)-stopped cells ineligible (D-H3)"


def calibrated_arms() -> tuple[str, ...]:
    """Arms the grid can calibrate: exactly ``calibration.GRID_ARMS`` (C4; an SC twin has no frozen file of
    its own, T-0076 A2)."""
    return tuple(_calibration.GRID_ARMS)


# =========================================================================== pure decision logic


@dataclasses.dataclass
class GridResult:
    """One entry per cell run, in order; ``stopped`` is ``None`` iff the grid ran to its end."""

    cells: list[dict[str, Any]]
    stopped: str | None
    arm: str = "perstep_mar"


def _score(task: str, rows: Sequence[Mapping[str, Any]]) -> float:
    key = SCORE_KEY[task]
    return sum(bool((r.get("scores") or {}).get(key)) for r in rows) / len(rows) if rows else 0.0


def _mean_steps(rows: Sequence[Mapping[str, Any]]) -> float:
    """T-0070's tie-break quantity: mean over records of the record's total steps (inf with none)."""
    totals = [sum(int(s) for s in r["steps"]) for r in rows if r.get("steps")]
    return sum(totals) / len(totals) if totals else math.inf


def steps_per_canvas(rows: Sequence[Mapping[str, Any]]) -> float | None:
    canvases = [int(s) for r in rows for s in (r.get("steps") or ())]
    return sum(canvases) / len(canvases) if canvases else None


def _internal(rows: Sequence[Mapping[str, Any]], point: Mapping[str, Any]) -> str | None:
    for i, r in enumerate(rows):
        if r.get("error") == INTERNAL:
            return (
                f"internal: CD error 'internal' on record {i} ({r.get('id')}) of cell "
                f"eb={float(point['entropy_bound']):g} ct={float(point['confidence_threshold']):g}"
            )
    return None


def engine_error(rows: Sequence[Mapping[str, Any]], what: str) -> str | None:
    """C9: the stop for the first ``http_error``/``timeout`` row of ``rows`` (``None`` without one)."""
    bad = [(i, r) for i, r in enumerate(rows) if r.get("error") in ENGINE_ERRORS]
    if not bad:
        return None
    i, r = bad[0]
    return (
        f"engine: {len(bad)} of {len(rows)} rows of {what} are {sorted({str(x['error']) for _, x in bad})} "
        f"(first: {r.get('error')} on record {i} ({r.get('id')}))"
    )


def _cell_name(arm: str, point: Mapping[str, Any]) -> str:
    return (
        f"{arm} cell eb={float(point['entropy_bound']):g} "
        f"ct={float(point['confidence_threshold']):g}"
    )


def run_grid(
    cells: Sequence[Mapping[str, Any]],
    run_cell: Callable[[Mapping[str, Any], float], Any],
    *,
    task: str,
    budget_s: float,
    clock: Callable[[], float] = time.monotonic,
    arm: str = "perstep_mar",
    log: Callable[[str], None] = lambda _msg: None,
    probe: Callable[[], str | None] | None = None,
) -> GridResult:
    """Run every cell in order (D-H3 / C2). ``run_cell(point, left_s)`` returns a ``runner.RunResult``.

    - an (e)-stopped cell is recorded ineligible with its partial score and the grid continues;
    - any ``http_error``/``timeout`` row ends the grid at that cell (C9, ``stopped`` starts with "engine"),
      ahead of the cell's own stop (a dead engine also trips (d) or (e));
    - any other stop, or an ``internal`` CD error row, ends the grid at that cell (``stopped`` says why);
    - no cell starts once ``clock() - start >= budget_s`` (``stopped`` starts with "g");
    - ``probe()`` (the engine health probe; ``None`` = healthy) runs before every cell, and a problem ends
      the grid before that cell (``stopped`` starts with "engine").
    """
    if task not in SCORE_KEY:
        raise ValueError(f"unknown task {task!r}")
    start = clock()
    out: list[dict[str, Any]] = []
    stopped: str | None = None
    for i, point in enumerate(cells):
        elapsed = clock() - start
        left = budget_s - elapsed
        if left <= 0:
            stopped = (
                f"g: launch budget of {budget_s:.1f} s spent ({elapsed:.1f} s) after {i} of "
                f"{len(cells)} cells"
            )
            break
        if probe is not None:
            problem = probe()
            if problem is not None:
                stopped = (
                    f"engine: {problem} (before {_cell_name(arm, point)}, {i} of {len(cells)} run)"
                )
                break
        t0 = clock()
        res = run_cell(point, left)
        wall = clock() - t0
        regime, rows, _other = _runner.read_artifact(res.artifact)
        cell_stop = res.stopped
        e_stopped = cell_stop is not None and str(cell_stop).startswith("e")
        cell: dict[str, Any] = {
            "entropy_bound": float(point["entropy_bound"]),
            "confidence_threshold": float(point["confidence_threshold"]),
            "score": _score(task, rows),
            "n": len(rows),
            "eligible": not e_stopped,
            "stopped": cell_stop,
            "mean_steps": _mean_steps(rows),
            "mean_steps_per_canvas": steps_per_canvas(rows),
            "errors": dict(Counter(str(r["error"]) for r in rows if r.get("error"))),
            "wall_s": wall,
            "artifact": str(res.artifact),
            "run_id": regime.get("run_id"),
        }
        out.append(cell)
        done = i + 1
        eta = (clock() - start) / done * (len(cells) - done)
        log(
            f"[{arm} {done}/{len(cells)}] eb={cell['entropy_bound']:g} ct={cell['confidence_threshold']:g} "
            f"score {cell['score']:.3f} n {cell['n']} stopped {cell_stop} wall {wall:.1f} s; "
            f"ETA {eta:.0f} s (budget left {budget_s - (clock() - start):.0f} s)"
        )
        engine = engine_error(rows, _cell_name(arm, point))
        if engine is not None:
            stopped = engine
            break
        if cell_stop is not None and not e_stopped:
            stopped = str(cell_stop)
            break
        internal = _internal(rows, point)
        if internal is not None:
            stopped = internal
            break
    return GridResult(cells=out, stopped=stopped, arm=arm)


def choose(grid: GridResult) -> Mapping[str, Any]:
    """``calibration.select`` over the eligible cells; a stopped (partial) grid or no eligible cell raises."""
    if grid.stopped:
        raise ValueError(f"the grid was stopped ({grid.stopped}); a partial grid selects nothing")
    eligible = [c for c in grid.cells if c["eligible"]]
    if not eligible:
        raise ValueError("no eligible cell: every cell was (e)-stopped")
    return _calibration.select(eligible)


def freeze_path_for(base: Path, arm: str) -> Path:
    """``perstep_mar`` freezes at ``base``; any other calibrated arm beside it, named by the arm (C3)."""
    if arm == "perstep_mar":
        return base
    return base.with_name(f"{base.stem}-{arm}{base.suffix}")


# =========================================================================== I/O helpers


def _say(msg: str) -> None:
    print(f"cd_campaign: {msg}", flush=True)


def _jsonable(x: Any) -> Any:
    """Non-finite floats become ``None`` (strict JSON)."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def accepted_per_step(log: _records.EngineLog, rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Step-weighted mean of the CD records' ``mean_accepted_per_step`` over the rows' canvases, joined on
    each row's ``record_tag`` (the rows do not carry it; C2: read from the engine log, read-only)."""
    num = den = 0.0
    plain: list[float] = []
    for r in rows:
        tag = (r.get("knobs") or {}).get("record_tag")
        if not isinstance(tag, str):
            continue
        for rec in _records.join(log.records(tag)).records:
            if rec.mean_accepted_per_step is None:
                continue
            plain.append(float(rec.mean_accepted_per_step))
            num += float(rec.mean_accepted_per_step) * int(rec.steps)
            den += int(rec.steps)
    if den > 0:
        return num / den
    return sum(plain) / len(plain) if plain else None


def cell_stats(log: _records.EngineLog, cell: Mapping[str, Any]) -> dict[str, Any]:
    rows = _runner.read_rows(cell["artifact"])
    return {
        "mean_steps_per_canvas": steps_per_canvas(rows),
        "mean_accepted_per_step": accepted_per_step(log, rows),
        "n": len(rows),
        "score": cell["score"],
        "stopped": cell["stopped"],
    }


def bucket_histogram(task: str) -> dict[str, int]:
    """The dev cut's automaton buckets (JSON whitespace), on the CPU, before the first request."""
    from local_astra.eval.cd import cuts, replay

    hist: Counter[int] = Counter()
    for rec in cuts.load(task, "dev"):
        hist[int(replay.compile_automaton(task, rec, whitespace="json").n_states_bucket)] += 1
    return {str(b): hist[b] for b in sorted(hist)}


def health_probe(base_url: str, timeout_s: float = 10.0) -> str | None:
    """C9: ``GET <base_url>/models`` on loopback must answer 200; ``None`` = healthy, else the problem.
    Reading the engine's status spends nothing (gpu-services rule)."""
    _runner.check_loopback(base_url)
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            status = resp.status
            resp.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
    except (OSError, ValueError) as exc:
        return f"health probe {url} failed ({exc})"
    return None if status == 200 else f"health probe {url} returned {status}"


def diffusion_probe(base_url: str) -> str | None:
    """A ``temperature`` request must be a 400 on the diffusion path (gpu-services rule); ``None`` = ok."""
    _runner.check_loopback(base_url)
    body = {
        "model": _runner.DEFAULT_MODEL,
        "max_tokens": 8,
        "temperature": 0.7,
        "messages": [{"role": "user", "content": "hi"}],
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            status = resp.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    except (OSError, ValueError) as exc:
        return f"no engine at {base_url} ({exc})"
    return (
        None if status == 400 else f"temperature returned {status}, not 400: not the diffusion path"
    )


# =========================================================================== main


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--task", choices=sorted(SCORE_KEY), required=True)
    # Criterion 4: the test cut is refused by construction.
    ap.add_argument("--cut", choices=["dev"], required=True)
    ap.add_argument("--arm", action="append", choices=calibrated_arms(), default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--engine-log", type=Path, required=True)
    ap.add_argument("--boot-log", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--expected-commit", required=True)
    ap.add_argument("--freeze-path", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget-s", type=float, default=None)
    ap.add_argument("--request-timeout-s", type=float, default=600.0)
    ap.add_argument("--probe-diffusion", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    t_launch = time.monotonic()
    task: str = args.task
    arm_list: list[str] = list(dict.fromkeys(args.arm or DEFAULT_ARMS))
    budget = float(args.budget_s if args.budget_s is not None else BUDGET_S[task])
    out: Path = args.out_dir
    report_path = out / f"campaign-{task}.json"
    freeze_paths = {a: freeze_path_for(args.freeze_path, a) for a in arm_list}

    # ---- refusals: nothing is sent before these pass
    for a, p in freeze_paths.items():
        if p.exists():
            _say(f"refusing: {p} exists (a frozen calibration for {a} is never overwritten)")
            return EXIT_REFUSED
    if args.boot_log is not None:
        from local_astra.bench.b16_tap import last_boot

        try:
            last_boot(args.boot_log.read_text(errors="replace"))
        except (OSError, ValueError) as exc:
            _say(f"refusing: no boot in {args.boot_log}: {exc}")
            return EXIT_REFUSED
    try:
        _runner.check_loopback(args.base_url)
    except ValueError as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED

    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "task": task,
        "cut": "dev",
        "arms": arm_list,
        "expected_commit": args.expected_commit,
        "limit": args.limit,
        "budget_s_per_arm": budget,
        "bucket_histogram": None,
        "stack_check": None,
        "cells": [],
        "grids": {},
        "stopped": None,
        "frozen": None,
        "frozen_by_arm": {},
        # C5: one entry per requested arm's SC twin, None until that twin has run
        **{REPORTED_AT_CHOICE[a]: None for a in arm_list if a in REPORTED_AT_CHOICE},
        "regime": (
            f"T-0071b: task {task}, dev cut, arms {arm_list}, grid {len(_calibration.grid())} cells x "
            f"limit {args.limit}, stack check unconstrained n = all dev, budget {budget:.0f} s per arm, "
            "engine only, concurrency 1, no seed, enable_thinking=false, max_tokens 256, JSON whitespace"
        ),
    }

    def write_report() -> None:
        report["elapsed_s"] = time.monotonic() - t_launch
        report_path.write_text(json.dumps(_jsonable(report), indent=2) + "\n")

    def cfg(arm: str, *, limit: int | None, budget_s: float, knobs: Any = None) -> Any:
        return _runner.RunConfig(
            task=task,
            cut="dev",
            arm=arm,
            base_url=args.base_url,
            engine_log=args.engine_log,
            out_dir=out,
            repo=args.repo,
            expected_commit=args.expected_commit,
            limit=limit,
            budget_s=budget_s,
            request_timeout_s=args.request_timeout_s,
            mar_knobs=knobs,
        )

    def left_for(arm_index: int) -> float:
        return t_launch + (arm_index + 1) * budget - time.monotonic()

    def probe() -> str | None:
        return health_probe(args.base_url)

    def engine_stop(stop: str, arm: str | None = None) -> int:
        """C9: the launch ends; the report is written, nothing more is frozen; exit 1."""
        report["stopped"] = stop
        if arm is not None:
            report["stopped_arm"] = arm
        write_report()
        _say(f"FAIL: {stop}")
        return EXIT_NOT_FROZEN

    try:
        hist = bucket_histogram(task)
        report["bucket_histogram"] = hist
        _say(f"{task} dev bucket histogram (JSON whitespace): {hist}")

        problem = probe()
        if problem is not None:
            return engine_stop(f"engine: {problem} (before the stack check)")

        if args.probe_diffusion:
            problem = diffusion_probe(args.base_url)
            if problem is not None:
                report["stopped"] = f"refused: {problem}"
                write_report()
                _say(f"refusing: {problem}")
                return EXIT_REFUSED

        # ---- 1. stack check: unconstrained, the whole dev cut, first, same boot
        res = _runner.run(cfg("unconstrained", limit=None, budget_s=left_for(0)))
        _regime, _rows, other = _runner.read_artifact(res.artifact)
        trailer = next((x for x in reversed(other) if x.get("kind") == "trailer"), {})
        report["stack_check"] = {
            "n": len(_rows),
            "stopped": res.stopped,
            "header_prefix_rate": trailer.get("header_prefix_rate"),
            "summary": trailer.get("summary"),
            "artifact": str(res.artifact),
        }
        _say(
            f"stack check: n {len(_rows)}, stopped {res.stopped}, (f) header-prefix rate "
            f"{trailer.get('header_prefix_rate')}"
        )
        engine = engine_error(_rows, "the stack check")
        if engine is not None:
            return engine_stop(engine)
        if res.stopped:
            report["stopped"] = str(res.stopped)
            write_report()
            return EXIT_NOT_FROZEN

        # ---- 2-4. per calibrated arm: grid, choose, freeze
        for ai, arm in enumerate(arm_list):

            def run_cell(point: Mapping[str, Any], left: float, _arm: str = arm) -> Any:
                return _runner.run(cfg(_arm, limit=args.limit, budget_s=left, knobs=dict(point)))

            grid = run_grid(
                _calibration.grid(),
                run_cell,
                task=task,
                budget_s=left_for(ai),
                arm=arm,
                log=_say,
                probe=probe,
            )
            report["grids"][arm] = {"cells": grid.cells, "stopped": grid.stopped}
            if arm == arm_list[0]:
                report["cells"] = grid.cells
            if grid.stopped:
                report["stopped"] = grid.stopped
                report["stopped_arm"] = arm
                write_report()
                return EXIT_NOT_FROZEN
            try:
                best = choose(grid)
            except ValueError as exc:
                report["stopped"] = f"no choice for {arm}: {exc}"
                write_report()
                return EXIT_NOT_FROZEN

            log = _records.EngineLog(args.engine_log, from_start=True)
            log.poll()
            stock_cell = next(
                c
                for c in grid.cells
                if (c["entropy_bound"], c["confidence_threshold"])
                == (STOCK_KNOBS["entropy_bound"], STOCK_KNOBS["confidence_threshold"])
            )
            stock_ref = {**STOCK_KNOBS, **cell_stats(log, stock_cell)}
            chosen = cell_stats(log, best)
            path = freeze_paths[arm]
            sha = _calibration.freeze(
                {
                    "entropy_bound": best["entropy_bound"],
                    "confidence_threshold": best["confidence_threshold"],
                    "task": task,
                    "cut": "dev",
                    "arm": arm,
                    "expected_commit": args.expected_commit,
                    "limit": args.limit,
                    "rule": RULE,
                    "stock_reference": _jsonable(stock_ref),
                    "chosen_stats": _jsonable(chosen),
                    "grid_results": _jsonable(
                        [{**c, "artifact": Path(c["artifact"]).name} for c in grid.cells]
                    ),
                },
                path,
            )
            _calibration.load_frozen(path, task=task, arm=arm)  # reads back as arm's file (C4)
            frozen = {"path": str(path), "sha256": sha}
            report["frozen_by_arm"][arm] = {
                **frozen,
                "entropy_bound": best["entropy_bound"],
                "confidence_threshold": best["confidence_threshold"],
                "stock_reference": stock_ref,
                "chosen_stats": chosen,
            }
            if arm == "perstep_mar":
                report["frozen"] = frozen
            _say(
                f"{arm} frozen at {path} sha256 {sha}: eb={best['entropy_bound']:g} "
                f"ct={best['confidence_threshold']:g} score {best['score']:.3f}; chosen {chosen}; "
                f"stock {stock_ref}"
            )
            write_report()

            # ---- 5. the SC twin at the chosen values: reported, never a selection input
            companion = REPORTED_AT_CHOICE.get(arm)
            if companion is not None:
                problem = probe()
                if problem is not None:
                    return engine_stop(f"engine: {problem} (before {companion})", arm)
                knobs = {
                    "entropy_bound": float(best["entropy_bound"]),
                    "confidence_threshold": float(best["confidence_threshold"]),
                }
                sres = _runner.run(
                    cfg(
                        companion,
                        limit=args.limit,
                        budget_s=max(left_for(ai), 1.0),
                        knobs=knobs,
                    )
                )
                srows = _runner.read_rows(sres.artifact)
                report[companion] = {
                    **knobs,
                    "arm": companion,
                    "calibrated_arm": arm,
                    "score": _score(task, srows),
                    "n": len(srows),
                    "stopped": sres.stopped,
                    "artifact": str(sres.artifact),
                    "note": f"reported only; selection used {arm}'s score",
                }
                engine = engine_error(srows, companion)
                if engine is not None:
                    return engine_stop(engine, arm)
                if sres.stopped:
                    _say(f"WARNING: {companion} at the chosen values stopped: {sres.stopped}")
                _say(f"{companion} at the chosen values: {report[companion]}")
                write_report()
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        report["stopped"] = f"crash: {tb.strip().splitlines()[-1]}"
        report["traceback"] = tb
        write_report()
        return EXIT_NOT_FROZEN
    finally:
        if args.boot_log is not None:
            from local_astra.bench.b16_tap import last_boot

            try:
                (out / "engine.log").write_text(
                    last_boot(args.boot_log.read_text(errors="replace"))
                )
            except (OSError, ValueError) as exc:
                _say(f"engine log slice not written: {exc}")

    write_report()
    return EXIT_FROZEN if len(report["frozen_by_arm"]) == len(arm_list) else EXIT_NOT_FROZEN


if __name__ == "__main__":
    raise SystemExit(main())
