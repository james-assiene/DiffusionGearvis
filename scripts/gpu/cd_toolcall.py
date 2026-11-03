"""T-0080b/c -- the two-pass tool-call driver (pass 1 once per case, every pass-2 cell paired). `/launch` ONLY.

    uv run --group cd python scripts/gpu/cd_toolcall.py --stage dev \
        --base-url http://127.0.0.1:8001/v1 --engine-log logs/vllm.log --boot-log logs/vllm.log \
        --out-dir logs/t0080-toolcall-dev-<sha> --repo . --expected-commit $(git rev-parse HEAD)
    uv run --group cd python scripts/gpu/cd_toolcall.py --stage test \
        --selection logs/t0080-toolcall-dev-<sha>/selection-dev.json ...

GPU-spending by definition (scripts/gpu/, T-0016): it sends inference requests to a booted engine. It never
boots one; ``scripts/locked/t0080_toolcall.sh`` does (engine only). Importing this file opens no socket.

- ``--stage dev`` (T-0080c stage 1): cells ``T1|T2|T4 / named / perstep_mar`` (K-D2) unless ``--cell`` is given;
  the stage-1 rule (``twopass.select(rows, "stage1")``) is applied to the artifact and its choice is written to
  ``<out-dir>/selection-dev.json`` with the commit, the artifact's sha256 and the full report.
- ``--stage test`` (stage 2) needs ``--selection``: a ``selection-dev.json`` whose ``commit`` is
  ``--expected-commit`` (the same SHA's dev stage) and whose ``choice`` names the (thought, scope); the cells
  are that (thought, scope) under ``end_projection``, ``perstep_mar``, ``perstep_mar_qargmax`` and
  ``perstep_mar+seed`` (T-0081b: the pass-1 draft seeds the first canvas; noise and start step are the
  driver's constants, not options).
- Replay (T-0082b): ``--pass1-from PATH --pass1-sha256 HEX`` sends no pass 1; the pass-1 rows are that earlier
  artifact's, re-scored under the current router, and only the cells' pass-2 requests are sent
  (``twopass.TwoPassConfig.pass1_from``). The cells are ``REPLAY_CELLS`` unless ``--cell`` is given (ruling R2), at
  either stage; ``--selection`` is refused and no ``selection-dev.json`` is written (ruling R6).
- Format comparison (T-0084d): ``--compare-native --stage dev`` runs ``COMPARE_NATIVE_DEV_CELLS`` (cc
  ``T1/named/{4 arms}`` plus native ``{T1,T2,T4}/named/{4 arms}``, 16 cells) unless ``--cell`` is given;
  ``--compare-native --stage test --selection <selection-native-dev.json>`` runs cc ``T1/named/{4}`` plus
  ``native/θ_n/named/{4}`` (8 cells), θ_n being the selection's ``native_thought``. The rule is
  ``twopass.select_format`` (read from the report's ``format_selection``) and its result is written to
  ``<out-dir>/selection-native-<stage>.json`` (stage, rule, native_thought, choice, winner, commit, artifact sha256,
  cells). The selection's commit and winner are the locked script's to check (ruling D6), not this CLI's.
- The knobs come from the frozen ``docs/cd/calibration-bfcl*.json`` (``--calibration ARM=PATH`` overrides),
  read by ``calibration.load_frozen(path, task="bfcl", arm=arm)``. The seed arm uses ``perstep_mar``'s file;
  ``--calibration perstep_mar+seed=...`` is refused (exit 2).

``<out-dir>/report-<stage>.json`` and ``.md`` are written from the artifact alone, in every outcome where an
artifact exists. Exit codes: the driver's (0 ok; 1 an engine, CS, join or loud-error stop, a crash, or no
eligible stage-1 cell; 2 refused, argparse included).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from local_astra.eval.cd import twopass

REPO = Path(__file__).resolve().parents[2]
SET_DIR = REPO / "tests" / "eval" / "toolcall"
DEFAULT_CALIBRATION: Mapping[str, Path] = {
    "perstep_mar": REPO / "docs" / "cd" / "calibration-bfcl.json",
    "perstep_mar_qargmax": REPO / "docs" / "cd" / "calibration-bfcl-perstep_mar_qargmax.json",
}
STAGE1_CELLS = ("T1/named/perstep_mar", "T2/named/perstep_mar", "T4/named/perstep_mar")
STAGE2_ARMS = ("end_projection", "perstep_mar", "perstep_mar_qargmax", twopass.SEED_ARM)
#: T-0082b ruling R2: the replay's cells when no ``--cell`` is given (T-0082c's: T1/named x the four stage-2 arms)
REPLAY_CELLS = (
    "T1/named/end_projection",
    "T1/named/perstep_mar",
    "T1/named/perstep_mar_qargmax",
    "T1/named/perstep_mar+seed",
)
#: T-0084d criterion 4: the format comparison's cells (cc at T1, as T-0080c chose; native at every thought on dev)
COMPARE_NATIVE_ARMS = STAGE2_ARMS
COMPARE_NATIVE_DEV_CELLS = (
    *(f"T1/named/{arm}" for arm in COMPARE_NATIVE_ARMS),
    *(f"native/{t}/named/{arm}" for t in ("T1", "T2", "T4") for arm in COMPARE_NATIVE_ARMS),
)
EXIT_OK, EXIT_STOPPED, EXIT_REFUSED = 0, 1, 2


def _say(msg: str) -> None:
    print(f"cd_toolcall: {msg}", flush=True)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--stage", choices=list(twopass.STAGES), required=True)
    ap.add_argument("--cell", action="append", default=None)
    ap.add_argument("--selection", type=Path, default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--engine-log", type=Path, required=True)
    ap.add_argument("--boot-log", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--expected-commit", required=True)
    ap.add_argument("--calibration", action="append", default=None, metavar="ARM=PATH")
    ap.add_argument("--set-dir", type=Path, default=SET_DIR)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--request-timeout-s", type=float, default=600.0)
    ap.add_argument("--pass1-from", type=Path, default=None)
    ap.add_argument("--pass1-sha256", default=None)
    ap.add_argument("--compare-native", action="store_true")
    return ap


def stage2_cells(selection: Path, expected_commit: str) -> list[str]:
    """The stage-2 cells from a dev-stage selection of the same commit; ``ValueError`` otherwise."""
    data = json.loads(selection.read_text())
    if data.get("stage") != "dev" or data.get("rule") != "stage1":
        raise ValueError(f"{selection} is not a stage-1 (dev) selection")
    if data.get("commit") != expected_commit:
        raise ValueError(
            f"{selection} was written at {data.get('commit')}, not at {expected_commit} "
            "(stage 2 runs only on this SHA's dev selection)"
        )
    choice = data.get("choice")
    if not isinstance(choice, str):
        raise ValueError(f"{selection} has no choice")
    cell = twopass.parse_cell(choice)
    return [f"{cell.thought}/{cell.scope}/{arm}" for arm in STAGE2_ARMS]


def native_thought_of(selection: Path) -> str:
    """θ_n from a ``selection-native-dev.json`` (T-0084d criterion 4; ruling D4); ``ValueError`` without one."""
    data = json.loads(selection.read_text())
    thought = data.get("native_thought") if isinstance(data, dict) else None
    if thought not in twopass.THOUGHTS:
        raise ValueError(
            f"{selection} has no native_thought in {twopass.THOUGHTS} (got {thought!r})"
        )
    return str(thought)


def compare_native_test_cells(thought: str) -> list[str]:
    return [f"T1/named/{arm}" for arm in COMPARE_NATIVE_ARMS] + [
        f"native/{thought}/named/{arm}" for arm in COMPARE_NATIVE_ARMS
    ]


def _calibration(items: Sequence[str] | None) -> dict[str, Path]:
    out = dict(DEFAULT_CALIBRATION)
    for item in items or ():
        arm, sep, path = item.partition("=")
        if not sep or not arm or not path:
            raise ValueError(f"--calibration takes ARM=PATH, not {item!r}")
        if arm == twopass.SEED_ARM:
            raise ValueError(
                f"{arm} has no calibration of its own (it uses {twopass.SEED_BASE_ARM}'s frozen file)"
            )
        out[arm] = Path(path)
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_native_selection(
    out: Path,
    stage: str,
    rep: Mapping[str, Any],
    native_thought: str | None,
    commit: str,
    artifact: Path,
    cells: Sequence[str],
) -> int:
    """T-0084d criterion 4: ``<out>/selection-native-<stage>.json`` from the report's ``format_selection``; exit 1
    (nothing written) when the rule chose nothing or, on test, chose at another θ_n than the selection's."""
    fs = rep.get("format_selection") or {}
    choice = fs.get("choice")
    if choice is None:
        _say(f"no format choice: {fs.get('reason')}")
        return EXIT_STOPPED
    if native_thought is not None and fs.get("native_thought") != native_thought:
        _say(
            f"the report's θ_n {fs.get('native_thought')} is not the selection's {native_thought}; "
            "no selection written"
        )
        return EXIT_STOPPED
    selection: dict[str, Any] = {
        "stage": stage,
        "rule": twopass.FORMAT_RULE,
        "native_thought": fs.get("native_thought"),
        "choice": choice,
        "winner": fs.get("winner"),
        "commit": commit,
        "head": rep["regime"][0].get("commit"),
        "artifact": str(artifact),
        "artifact_sha256": _sha256(Path(artifact)),
        "cells": list(cells),
        "tables": fs.get("tables"),
    }
    path = out / f"selection-native-{stage}.json"
    path.write_text(json.dumps(selection, indent=2) + "\n")
    _say(
        f"{path.name} written: {choice}, winner {selection['winner']}, θ_n {selection['native_thought']} "
        f"(sha256 {_sha256(path)})"
    )
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    out: Path = args.out_dir
    replay = args.pass1_from is not None or args.pass1_sha256 is not None
    compare = bool(args.compare_native)
    native_thought: str | None = None
    try:
        if compare:
            if replay:
                raise ValueError(
                    "--compare-native takes no --pass1-from (a replay re-uses cc pass-1 rows only)"
                )
            if args.stage == "test":
                if args.selection is None:
                    raise ValueError(
                        "--compare-native --stage test needs --selection <selection-native-dev.json>"
                    )
                if args.cell:
                    raise ValueError(
                        "--cell is not accepted with --stage test (the selection fixes them)"
                    )
                native_thought = native_thought_of(args.selection)
                cells = compare_native_test_cells(native_thought)
            else:
                if args.selection is not None:
                    raise ValueError("--compare-native --stage dev takes no --selection")
                cells = list(args.cell or COMPARE_NATIVE_DEV_CELLS)
        elif replay:
            if args.pass1_from is None or not args.pass1_sha256:
                raise ValueError("--pass1-from and --pass1-sha256 go together")
            if args.selection is not None:
                raise ValueError("a replay takes no --selection (no stage-1 selection on a replay)")
            cells = list(args.cell or REPLAY_CELLS)
        elif args.stage == "test":
            if args.selection is None:
                raise ValueError("--stage test needs --selection <selection-dev.json>")
            cells = stage2_cells(args.selection, args.expected_commit)
            if args.cell:
                raise ValueError(
                    "--cell is not accepted with --stage test (the selection fixes them)"
                )
        else:
            cells = list(args.cell or STAGE1_CELLS)
        cfg = twopass.TwoPassConfig(
            stage=args.stage,
            cells=cells,
            base_url=args.base_url,
            engine_log=args.engine_log,
            out_dir=out,
            repo=args.repo,
            expected_commit=args.expected_commit,
            calibration=_calibration(args.calibration),
            set_dir=args.set_dir,
            limit=args.limit,
            request_timeout_s=args.request_timeout_s,
            pass1_from=args.pass1_from,
            pass1_sha256=args.pass1_sha256,
        )
    except (ValueError, OSError) as exc:
        _say(f"refusing: {exc}")
        return EXIT_REFUSED
    out.mkdir(parents=True, exist_ok=True)
    try:
        try:
            result = twopass.run(cfg)
        except ValueError as exc:
            _say(f"refusing: {exc}")
            return EXIT_REFUSED
        _say(f"artifact {result.artifact} exit {result.exit_code} stopped {result.stopped}")
        rep = twopass.report([result.artifact])
        (out / f"report-{args.stage}.json").write_text(json.dumps(rep, indent=2) + "\n")
        (out / f"report-{args.stage}.md").write_text(twopass.render_markdown(rep))
        if compare:
            _say(f"format selection: {rep.get('format_selection')}")
        else:
            _say(f"selection: {rep['selection']}")
        if result.exit_code != EXIT_OK:
            return int(result.exit_code)
        if compare:
            return _write_native_selection(
                out, args.stage, rep, native_thought, args.expected_commit, result.artifact, cells
            )
        if replay:
            _say("replay: no selection file is written")
            return EXIT_OK
        if args.stage == "dev":
            choice = rep["selection"].get("choice")
            if choice is None:
                _say(f"no stage-1 choice: {rep['selection'].get('reason')}")
                return EXIT_STOPPED
            selection: dict[str, Any] = {
                "stage": "dev",
                "rule": "stage1",
                "choice": choice,
                "commit": args.expected_commit,
                "expected_commit": args.expected_commit,
                "head": rep["regime"][0].get("commit"),
                "artifact": str(result.artifact),
                "artifact_sha256": _sha256(Path(result.artifact)),
                "cells": cells,
                "policies": rep["policies"],
            }
            path = out / "selection-dev.json"
            path.write_text(json.dumps(selection, indent=2) + "\n")
            _say(f"selection-dev.json written: {choice} (sha256 {_sha256(path)})")
        return EXIT_OK
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return EXIT_STOPPED
    finally:
        if args.boot_log is not None:
            from local_astra.bench.b16_tap import last_boot

            try:
                (out / "engine.log").write_text(
                    last_boot(args.boot_log.read_text(errors="replace"))
                )
            except (OSError, ValueError) as exc:
                _say(f"engine log slice not written: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
