"""Calibration of the Mar knobs on the dev cuts, frozen before any test-cut run (T-0070 criterion 10).

- ``grid()``: ``entropy_bound`` in {0.003, 0.01, 0.03, 0.1, 0.3, 1.0} x ``confidence_threshold`` in
  {5e-5, 5e-4, 5e-3} (diffgemma-fa SPEC §3.4), 18 points, arm ``perstep_mar`` or (T-0076)
  ``perstep_mar_qargmax``; each SC twin runs on its base arm's frozen file (``CALIBRATION_SOURCE``).
- ``select(results)``: maximise the score (BFCL exact call, Sudoku solved); ties go to fewer mean steps,
  then to the larger ``entropy_bound``.
- ``freeze(choice, path)`` writes ``calibration.json`` and returns the sha256 of its bytes. The file
  carries ``content_sha256`` over its canonical content, so an edit after freezing is detected
  (ruling E7: tamper-evident); ``load_frozen`` refuses such a file.
- ``calibrate(...)`` runs the grid on the dev cut through the runner and freezes the winner.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "CALIBRATED_ARMS",
    "CALIBRATION_SOURCE",
    "CONFIDENCE_THRESHOLDS",
    "ENTROPY_BOUNDS",
    "GRID_ARMS",
    "calibrate",
    "default_path",
    "freeze",
    "grid",
    "load_frozen",
    "select",
]

ENTROPY_BOUNDS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
CONFIDENCE_THRESHOLDS = (5e-5, 5e-4, 5e-3)
KNOBS = ("entropy_bound", "confidence_threshold")
#: The grid runs ``perstep_mar`` or ``perstep_mar_qargmax`` (``GRID_ARMS``); each SC twin shares its base
#: arm's knobs (T-0076 Q3).
CALIBRATED_ARMS = ("perstep_mar", "perstep_mar_sc", "perstep_mar_qargmax", "perstep_mar_qargmax_sc")
#: The arms ``calibrate`` runs the grid for; a frozen file a run accepts names one of these, never an SC
#: twin (T-0076 ruling A2).
GRID_ARMS = ("perstep_mar", "perstep_mar_qargmax")
#: The arm whose frozen file each calibrated arm runs on (T-0076 criterion 8).
CALIBRATION_SOURCE = {
    "perstep_mar": "perstep_mar",
    "perstep_mar_sc": "perstep_mar",
    "perstep_mar_qargmax": "perstep_mar_qargmax",
    "perstep_mar_qargmax_sc": "perstep_mar_qargmax",
}
_SELF_HASH = "content_sha256"


def grid() -> list[dict[str, float]]:
    return [
        {"entropy_bound": eb, "confidence_threshold": ct}
        for eb, ct in itertools.product(ENTROPY_BOUNDS, CONFIDENCE_THRESHOLDS)
    ]


def select(results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The pre-registered rule: max ``score``, then min ``mean_steps``, then max ``entropy_bound``."""
    if not results:
        raise ValueError("no calibration results to select from")
    return max(
        results,
        key=lambda r: (
            float(r["score"]),
            -float(r["mean_steps"]),
            float(r["entropy_bound"]),
        ),
    )


def _canonical(content: Mapping[str, Any]) -> str:
    return json.dumps(dict(content), sort_keys=True, separators=(",", ":"))


def freeze(choice: Mapping[str, Any], path: Path | str) -> str:
    """Write the frozen calibration and return the sha256 of the file's bytes."""
    content = {k: v for k, v in choice.items() if k != _SELF_HASH}
    for k in KNOBS:
        if k not in content:
            raise ValueError(f"calibration needs {k!r}")
        content[k] = float(content[k])
    content[_SELF_HASH] = hashlib.sha256(_canonical(content).encode()).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(content, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def load_frozen(
    path: Path | str, *, task: str | None = None, arm: str | None = None
) -> tuple[dict[str, float], str]:
    """``(knobs, file sha256)``. Raises ``ValueError`` for a missing, unfrozen, edited or other-task
    calibration, and -- when ``arm`` is given -- for a file that is not ``arm``'s calibration source
    (T-0076 criterion 8; an SC twin's file is never a source, ruling A2)."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"no frozen calibration at {path}")
    raw = path.read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not JSON: {exc}") from None
    if not isinstance(data, dict) or _SELF_HASH not in data:
        raise ValueError(f"{path} was not written by calibration.freeze (no {_SELF_HASH})")
    content = {k: v for k, v in data.items() if k != _SELF_HASH}
    if hashlib.sha256(_canonical(content).encode()).hexdigest() != data[_SELF_HASH]:
        raise ValueError(f"{path} was edited after freezing (content hash mismatch)")
    # Ruling E11: the file names what it calibrated -- a task, the dev cut, a Mar arm.
    if content.get("task") not in ("bfcl", "sudoku"):
        raise ValueError(f"{path} does not name the task it calibrated")
    if task is not None and content["task"] != task:
        raise ValueError(f"{path} calibrates {content['task']!r}, not {task!r}")
    if content.get("cut") != "dev":
        raise ValueError(f"{path} was not calibrated on the dev cut (cut={content.get('cut')!r})")
    if content.get("arm") not in CALIBRATED_ARMS:
        raise ValueError(f"{path} does not name a calibrated arm {CALIBRATED_ARMS}")
    if arm is not None:
        source = CALIBRATION_SOURCE.get(arm)
        if source is None:
            raise ValueError(f"{arm!r} is not a calibrated arm {CALIBRATED_ARMS}")
        if content["arm"] != source:
            raise ValueError(
                f"{path} calibrated {content['arm']!r}; {arm} runs only on {source!r}'s frozen file"
            )
    knobs = {k: float(content[k]) for k in KNOBS}
    return knobs, hashlib.sha256(raw).hexdigest()


def _score_rows(task: str, rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    key = "exact_call" if task == "bfcl" else "solved"
    n = len(rows)
    ok = sum(bool((r.get("scores") or {}).get(key)) for r in rows)
    steps = [sum(r["steps"]) for r in rows if r.get("steps")]
    mean_steps = sum(steps) / len(steps) if steps else float("inf")
    return (ok / n if n else 0.0), mean_steps


def default_path(out_dir: Path | str, task: str, arm: str) -> Path:
    """``calibration-<task>.json`` for ``perstep_mar``, else ``calibration-<task>-<arm>.json`` (ruling A1)."""
    name = f"calibration-{task}.json" if arm == "perstep_mar" else f"calibration-{task}-{arm}.json"
    return Path(out_dir) / name


def calibrate(
    *,
    task: str,
    base_url: str,
    engine_log: Path | str,
    out_dir: Path | str,
    repo: Path | str,
    expected_commit: str,
    limit: int | None = None,
    path: Path | str | None = None,
    arm: str = "perstep_mar",
    **run_kwargs: Any,
) -> Path:
    """Run ``arm`` (one of ``GRID_ARMS``) on the dev cut at every grid point, select, freeze; return the
    file's path (``default_path`` unless ``path`` is given).

    Any hard stop aborts the calibration (``RuntimeError``): a partial grid selects nothing.
    """
    import local_astra.eval.cd.runner as runner

    if arm not in GRID_ARMS:
        raise ValueError(
            f"calibrate runs one of {GRID_ARMS}, not {arm!r} (an SC twin reuses its base arm's file)"
        )
    results: list[dict[str, Any]] = []
    for point in grid():
        cfg = runner.RunConfig(
            task=task,
            cut="dev",
            arm=arm,
            base_url=base_url,
            engine_log=Path(engine_log),
            out_dir=Path(out_dir),
            repo=Path(repo),
            expected_commit=expected_commit,
            limit=limit,
            mar_knobs=point,
            **run_kwargs,
        )
        res = runner.run(cfg)
        if res.exit_code != 0:
            raise RuntimeError(f"calibration run {point} stopped: {res.stopped}")
        rows = runner.read_rows(res.artifact)
        score, mean_steps = _score_rows(task, rows)
        results.append(
            {
                **point,
                "score": score,
                "mean_steps": mean_steps,
                "n": len(rows),
                "artifact": Path(res.artifact).name,
            }
        )
    best = select(results)
    out = Path(path) if path is not None else default_path(out_dir, task, arm)
    freeze(
        {
            "entropy_bound": best["entropy_bound"],
            "confidence_threshold": best["confidence_threshold"],
            "task": task,
            "cut": "dev",
            "arm": arm,
            "expected_commit": expected_commit,
            "rule": "max score; ties: fewer mean steps, then larger entropy_bound",
            "grid_results": results,
        },
        out,
    )
    return out
