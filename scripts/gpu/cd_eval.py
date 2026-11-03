"""CLI wrapper for the T-0070 CD evaluation harness. Running it against the engine is GPU time: /launch.

Lives under ``scripts/gpu/`` so the guard gates it (T-0016). Everything it does is in
``local_astra.eval.cd``; this file only parses arguments.

    uv run --group cd python scripts/gpu/cd_eval.py run --task bfcl --cut test --arm perstep_mar \
        --calibration artifacts/cd/calibration-bfcl.json --expected-commit $(git rev-parse HEAD)
    uv run --group cd python scripts/gpu/cd_eval.py calibrate --task sudoku --expected-commit ...
    uv run --group cd python scripts/gpu/cd_eval.py report artifacts/cd/*.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--task", choices=["bfcl", "sudoku"], required=True)
        p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
        p.add_argument("--engine-log", type=Path, required=True)
        p.add_argument("--out-dir", type=Path, default=REPO / "artifacts" / "cd")
        p.add_argument("--repo", type=Path, default=REPO)
        p.add_argument("--expected-commit", required=True)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--request-timeout-s", type=float, default=600.0)

    r = sub.add_parser("run")
    common(r)
    r.add_argument("--cut", choices=["test", "dev"], required=True)
    r.add_argument(
        "--arm",
        choices=[
            "unconstrained",
            "end_projection",
            "perstep_mf",
            "perstep_mar",
            "perstep_mar_sc",
            "perstep_mar_qargmax",
            "perstep_mar_qargmax_sc",
        ],
        required=True,
    )
    r.add_argument("--calibration", type=Path, default=None)
    r.add_argument("--whitespace", choices=["json", "pretty"], default="json")
    r.add_argument("--budget-s", type=float, default=14_400.0)
    r.add_argument("--max-states", type=int, default=512)
    r.add_argument("--step-cap", type=int, default=48)

    c = sub.add_parser("calibrate")
    common(c)

    rep = sub.add_parser("report")
    rep.add_argument("artifacts", nargs="+", type=Path)

    args = ap.parse_args(argv)
    if args.cmd == "report":
        from local_astra.eval.cd import report

        sys.stdout.write(report.build(args.artifacts))
        return 0
    if args.cmd == "calibrate":
        from local_astra.eval.cd import calibration

        path = calibration.calibrate(
            task=args.task,
            base_url=args.base_url,
            engine_log=args.engine_log,
            out_dir=args.out_dir,
            repo=args.repo,
            expected_commit=args.expected_commit,
            limit=args.limit,
            request_timeout_s=args.request_timeout_s,
        )
        print(path)
        return 0
    from local_astra.eval.cd import runner

    res = runner.run(
        runner.RunConfig(
            task=args.task,
            cut=args.cut,
            arm=args.arm,
            base_url=args.base_url,
            engine_log=args.engine_log,
            out_dir=args.out_dir,
            repo=args.repo,
            expected_commit=args.expected_commit,
            calibration=args.calibration,
            limit=args.limit,
            budget_s=args.budget_s,
            request_timeout_s=args.request_timeout_s,
            whitespace=args.whitespace,
            max_states=args.max_states,
            step_cap=args.step_cap,
        )
    )
    print(f"{res.artifact}\nstopped: {res.stopped}\nsummary: {dict(res.summary)}")
    return res.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
