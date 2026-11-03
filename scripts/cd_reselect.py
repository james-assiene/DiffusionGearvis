#!/usr/bin/env python3
"""T-0079 -- re-freeze the BFCL calibration under the human's H2 speed-aware rule. CPU only, no network.

    uv run --group cd python scripts/cd_reselect.py \
        --evidence-root .claude/worktrees/t0071b-launch2 \
        --campaign logs/t0071b-calib-bfcl-067ee04/campaign-bfcl.json \
        --engine-log logs/t0071b-calib-bfcl-067ee04/engine.log \
        --out-dir logs/t0079-reselect-<sha> --task bfcl --decision-date 2026-10-08

Relative ``--campaign`` / ``--engine-log`` paths resolve against ``--evidence-root`` (the launch worktree; the
campaign's artifact paths are relative to it too). Every input is opened read-only.

Rule (H2, docs/tasks/T-0071b.md): per grid arm, the pool is the cells with ``eligible`` true, ``stopped`` null
and ``n`` equal to the dev cut size; no other cell is chosen or sets "best". A pool cell qualifies iff
``score >= best - 0.05``, compared exactly on integer counts (``Fraction(k, n) >= Fraction(k_best, n) - 1/20``).
The choice is the qualifying cell with the fewest ``mean_steps``, then the larger ``entropy_bound``; a tie left
after both keys is refused.

Each arm's file is written with ``calibration.freeze`` at ``calibration.default_path(out_dir, task, arm)``,
keeping every field of the T-0071b frozen file and adding the provenance (``rule``, ``decision``,
``source_campaign``, ``superseded``, ``grid_results``, ``reselected_by``). No wall-clock time is written, so the
same inputs, decision date and commit give the same bytes.

Exit 0: both files written. Exit 2: refused (``--task`` other than bfcl, an existing output, a campaign whose
cut is not dev, a non-integer count, an empty pool, a tie, a superseded file whose bytes do not match the
campaign); nothing is written on a refusal.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

from local_astra.eval.cd import calibration as _calibration
from local_astra.eval.cd import records as _records

EXIT_OK = 0
EXIT_REFUSED = 2

TOOL_REL = "scripts/cd_reselect.py"
REPO = Path(__file__).resolve().parents[1]
#: The BFCL dev cut (records 130-229).
CUT_SIZE = {"bfcl": 100}
MARGIN = Fraction(1, 20)
INTEGER_TOL = 1e-9
RULE = (
    "H2 speed-aware: pool = cells with eligible true, stopped null and n equal to the dev cut size (100); "
    "a pool cell qualifies iff score >= best - 0.05 (exact integer counts, inclusive); choose the qualifying "
    "cell with the fewest mean steps, then the larger entropy_bound; a remaining tie is refused"
)
DECISION_REF = "docs/tasks/T-0071b.md#H2"


class Refused(Exception):
    """A refusal: exit 2, nothing written."""


def _say(msg: str) -> None:
    print(f"cd_reselect: {msg}", flush=True)


def _load_campaign_module() -> ModuleType:
    """``scripts/gpu/cd_campaign.py`` (read-only reuse of ``cell_stats`` and ``_jsonable``); importing it opens
    no socket and starts no run."""
    path = REPO / "scripts" / "gpu" / "cd_campaign.py"
    spec = importlib.util.spec_from_file_location("_t0079_cd_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as fh:
        return fh.read()


def _resolve(root: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else root / p


def _rel(root: Path, p: Path) -> str:
    """``p`` relative to the evidence root when inside it (so the bytes do not depend on where it lives)."""
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(p)


def _git_head() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    )
    if out.returncode != 0 or not out.stdout.strip():
        raise Refused(f"cannot read the tool's git commit: {out.stderr.strip()}")
    return out.stdout.strip()


# =========================================================================== the rule (pure)


def _count(cell: Mapping[str, Any]) -> int:
    """``score * n`` as an integer; refuses a score that is not a count over ``n``."""
    n = int(cell["n"])
    raw = float(cell["score"]) * n
    k = round(raw)
    if abs(raw - k) > INTEGER_TOL:
        raise Refused(
            f"cell eb={cell['entropy_bound']} ct={cell['confidence_threshold']}: score {cell['score']} "
            f"x n {n} = {raw!r} is not an integer count"
        )
    return k


def in_pool(cell: Mapping[str, Any], cut_size: int) -> bool:
    return (
        cell.get("eligible") is True and cell.get("stopped") is None and cell.get("n") == cut_size
    )


def reselect(cells: Sequence[Mapping[str, Any]], cut_size: int) -> tuple[int, list[dict[str, Any]]]:
    """``(index of the chosen cell, one table row per cell)``; raises ``Refused``."""
    pool = [i for i, c in enumerate(cells) if in_pool(c, cut_size)]
    if not pool:
        raise Refused("no cell in the pool (eligible, not stopped, n = cut size)")
    for i in pool:
        for key in ("mean_steps", "mean_steps_per_canvas"):
            if cells[i].get(key) is None:
                raise Refused(f"pool cell {i} has no {key!r}")
    counts = {i: _count(cells[i]) for i in pool}  # R5: only pool cells need an integer count
    k_best = max(counts.values())
    threshold = Fraction(k_best, cut_size) - MARGIN
    qualifying = [i for i in pool if Fraction(counts[i], cut_size) >= threshold]
    fewest = min(float(cells[i]["mean_steps"]) for i in qualifying)
    by_steps = [i for i in qualifying if float(cells[i]["mean_steps"]) == fewest]
    largest_eb = max(float(cells[i]["entropy_bound"]) for i in by_steps)
    chosen = [i for i in by_steps if float(cells[i]["entropy_bound"]) == largest_eb]
    table = [
        {
            "idx": i,
            "pool": i in counts,
            "k": counts.get(i),
            "qualifies": i in qualifying,
            "chosen": len(chosen) == 1 and i == chosen[0],
        }
        for i in range(len(cells))
    ]
    _print_table(cells, table, k_best, cut_size)
    if len(chosen) != 1:
        raise Refused(
            f"tie after mean steps and entropy_bound between cells {chosen}; no third key is invented"
        )
    return chosen[0], table


def _print_table(
    cells: Sequence[Mapping[str, Any]],
    table: Sequence[Mapping[str, Any]],
    k_best: int,
    cut_size: int,
) -> None:
    thr = Fraction(k_best, cut_size) - MARGIN
    print(f"  best = {k_best}/{cut_size}; threshold = {thr} = {float(thr):.2f} (inclusive)")
    print(
        f"  {'idx':>3} {'eb':>6} {'ct':>7} {'score':>6} {'n':>4} {'elig':>5} {'stopped':>8} "
        f"{'steps':>6} {'st/cv':>6} {'pool':>5} {'qual':>5} chosen"
    )
    for row in table:
        c = cells[int(row["idx"])]
        print(
            f"  {row['idx']:>3} {c['entropy_bound']:>6g} {c['confidence_threshold']:>7g} "
            f"{c['score']:>6g} {c['n']!s:>4} {c.get('eligible')!s:>5} "
            f"{'-' if c.get('stopped') is None else 'yes':>8} "
            f"{c.get('mean_steps')!s:>6} {c.get('mean_steps_per_canvas')!s:>6} "
            f"{row['pool']!s:>5} {row['qualifies']!s:>5} {'<==' if row['chosen'] else ''}"
        )


# =========================================================================== I/O


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    p.add_argument("--evidence-root", required=True, type=Path)
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--engine-log", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--task", required=True)
    p.add_argument("--decision-date", required=True)
    return p


def _plan(args: argparse.Namespace) -> list[tuple[Path, dict[str, Any]]]:
    """Every check and every computation; returns ``(path, content)`` per file. Writes nothing."""
    if args.task != "bfcl":
        raise Refused(
            f"--task {args.task!r}: H2 re-selects BFCL only (Sudoku keeps its pre-registered picks)"
        )
    try:
        _dt.date.fromisoformat(args.decision_date)
    except ValueError:
        raise Refused(f"--decision-date {args.decision_date!r} is not YYYY-MM-DD") from None
    task = args.task
    root: Path = args.evidence_root
    out_dir: Path = args.out_dir
    arms = _calibration.GRID_ARMS
    outs = {arm: _calibration.default_path(out_dir, task, arm) for arm in arms}
    existing = [str(p) for p in outs.values() if p.exists()]
    if existing:
        raise Refused(f"output exists, never overwritten: {existing}")

    campaign_path = _resolve(root, args.campaign)
    raw = _read_bytes(campaign_path)
    campaign_sha = _sha256(raw)
    campaign = json.loads(raw)
    if campaign.get("task") != task:
        raise Refused(f"the campaign calibrates {campaign.get('task')!r}, not {task!r}")
    if campaign.get("cut") != "dev":
        raise Refused(f"the campaign's cut is {campaign.get('cut')!r}, not 'dev'")
    if campaign.get("stopped") is not None:
        raise Refused(f"the campaign was stopped: {campaign['stopped']!r}")
    cut_size = CUT_SIZE[task]
    head = _git_head()

    cd_campaign = _load_campaign_module()
    log = _records.EngineLog(_resolve(root, args.engine_log), from_start=True)
    log.poll()

    plans: list[tuple[Path, dict[str, Any]]] = []
    for arm in arms:
        grid = (campaign.get("grids") or {}).get(arm)
        if grid is None:
            raise Refused(f"the campaign has no grid for {arm}")
        if grid.get("stopped") is not None:
            raise Refused(f"the {arm} grid was stopped: {grid['stopped']!r}")
        frozen = (campaign.get("frozen_by_arm") or {}).get(arm)
        if frozen is None:
            raise Refused(f"the campaign froze nothing for {arm}; nothing to supersede")
        cells: list[dict[str, Any]] = grid["cells"]
        _say(f"{arm}: {len(cells)} cells")
        idx, _ = reselect(cells, cut_size)
        best = cells[idx]

        old_path = _resolve(root, frozen["path"])
        old_raw = _read_bytes(old_path)
        old_sha = _sha256(old_raw)
        if old_sha != frozen["sha256"]:
            raise Refused(f"{old_path} sha256 {old_sha} != the campaign's {frozen['sha256']}")
        old = json.loads(old_raw)
        old_cell = [
            c
            for c in cells
            if (float(c["entropy_bound"]), float(c["confidence_threshold"]))
            == (float(old["entropy_bound"]), float(old["confidence_threshold"]))
        ]
        if len(old_cell) != 1:
            raise Refused(f"the superseded knobs of {arm} do not name exactly one grid cell")

        chosen_stats = cd_campaign.cell_stats(
            log, {**best, "artifact": str(_resolve(root, best["artifact"]))}
        )
        content: dict[str, Any] = {
            "entropy_bound": best["entropy_bound"],
            "confidence_threshold": best["confidence_threshold"],
            "task": task,
            "cut": "dev",
            "arm": arm,
            "expected_commit": campaign["expected_commit"],
            "limit": campaign.get("limit"),
            "rule": RULE,
            "stock_reference": cd_campaign._jsonable(frozen["stock_reference"]),
            "chosen_stats": cd_campaign._jsonable(chosen_stats),
            "grid_results": cd_campaign._jsonable(cells),
            "decision": {"by": "human", "date": args.decision_date, "ref": DECISION_REF},
            "source_campaign": {"path": _rel(root, campaign_path), "sha256": campaign_sha},
            "superseded": {
                "path": _rel(root, old_path),
                "sha256": old_sha,
                "entropy_bound": old["entropy_bound"],
                "confidence_threshold": old["confidence_threshold"],
                "score": old_cell[0]["score"],
                "mean_steps": old_cell[0]["mean_steps"],
            },
            "reselected_by": {"tool": TOOL_REL, "commit": head},
        }
        _say(
            f"{arm}: chose eb={best['entropy_bound']:g} ct={best['confidence_threshold']:g} score "
            f"{best['score']} mean_steps {best['mean_steps']} mean_steps_per_canvas "
            f"{best['mean_steps_per_canvas']}; chosen_stats {chosen_stats}; supersedes eb="
            f"{old['entropy_bound']:g} ct={old['confidence_threshold']:g} ({old_sha[:8]})"
        )
        plans.append((outs[arm], content))
    return plans


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plans = _plan(args)
    except Refused as exc:
        _say(f"REFUSED: {exc}")
        return EXIT_REFUSED
    for path, content in plans:
        sha = _calibration.freeze(content, path)
        knobs, _ = _calibration.load_frozen(path, task=content["task"], arm=content["arm"])
        _say(f"wrote {path} sha256 {sha} knobs {knobs}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
