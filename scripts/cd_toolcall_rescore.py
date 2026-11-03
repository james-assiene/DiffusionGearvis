#!/usr/bin/env python3
"""T-0082a -- re-score a stored two-pass tool-call artifact under the stripping router. CPU only, no network.

    uv run python scripts/cd_toolcall_rescore.py \
        --evidence-root logs \
        --artifact t0080-toolcall-test-0dc9134/20261009T152205.274601Z-toolcall-test.jsonl \
        --sha256 1c69a9c2... --out-dir <new dir>

``--artifact`` resolves against ``--evidence-root`` (an absolute path is taken as is). Every input is opened
read-only; nothing is written under the evidence root unless ``--out-dir`` points there.

Each pass-1 row is re-routed and re-scored from its stored ``text`` by ``twopass.rescore`` (the current router
and scorer; ground truth from ``tests/eval/toolcall/cases.jsonl`` at the MANIFEST-pinned sha, which must also be
the artifact's). Pass-2 rows are reused as stored. The report (``report-rescore.json`` / ``.md`` in a new
``--out-dir``) has ``draft_only`` and, per unseeded cell, ``cd_on_invalid:<cell>`` and ``cd_always:<cell>``,
built with ``twopass.derive``; seeded cells are listed as stale (their seeds were built from the unstripped
draft) and get no policy. ``select`` is never called: this is a counterfactual on data already seen, not a
choice.

Exit 0: both files written. Exit 2: refused (sha mismatch, an existing out-dir, a stopped artifact, cases or
tools that are not the artifact's, or a ``rescore`` refusal); nothing is written on a refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from local_astra.eval.cd import twopass as _twopass
from local_astra.toolcall import tools as _tools

EXIT_OK = 0
EXIT_REFUSED = 2

TOOL_REL = "scripts/cd_toolcall_rescore.py"
REPO = Path(__file__).resolve().parents[1]
SET_DIR = REPO / "tests" / "eval" / "toolcall"
STALE = "stale (seeds built before the strip)"
CAVEAT = (
    "Pre-registration caveat: the </cc> strip was designed after seeing the T-0080c test split, so this "
    "re-score is a counterfactual on data already seen, not a pre-registered measurement. Pass-2 rows are "
    "reused as stored; no constrained decoding was run."
)


class Refused(Exception):
    """A refusal: exit 2, nothing written."""


def _say(msg: str) -> None:
    print(f"cd_toolcall_rescore: {msg}", flush=True)


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as fh:
        return fh.read()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_head() -> str | None:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    )
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _cases(expected_sha: Any) -> list[dict[str, Any]]:
    raw = _read_bytes(SET_DIR / "cases.jsonl")
    sha = _sha256(raw)
    pinned = json.loads(_read_bytes(SET_DIR / "MANIFEST.json")).get("cases_sha256")
    if sha != pinned:
        raise Refused(f"cases.jsonl sha256 {sha} != MANIFEST's {pinned}")
    if sha != expected_sha:
        raise Refused(f"cases.jsonl sha256 {sha} != the artifact's {expected_sha}")
    return [json.loads(x) for x in raw.decode().splitlines() if x.strip()]


def _tool_set(expected_sha: Any) -> _tools.ToolSet:
    tools = _tools.load()
    if tools.version != expected_sha:
        raise Refused(f"tools_v1.json sha256 {tools.version} != the artifact's {expected_sha}")
    return tools


def _changes(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Per pass-1 row, what the re-score changed: stripped closes, newly schema-valid / exact drafts (and any
    lost, which the strip should never cause)."""
    b1 = [r for r in before if r.get("pass") == 1]
    a1 = [r for r in after if r.get("pass") == 1]
    out: dict[str, list[str]] = {
        "stripped": [],
        "newly_schema_valid": [],
        "newly_exact": [],
        "lost_schema_valid": [],
        "lost_exact": [],
    }
    for b, a in zip(b1, a1, strict=True):
        case = str(a["case"])
        if a.get("draft_close"):
            out["stripped"].append(case)
        bv, av = b.get("draft_schema_valid") is True, a.get("draft_schema_valid") is True
        be, ae = bool(b.get("draft_exact")), bool(a.get("draft_exact"))
        if av and not bv:
            out["newly_schema_valid"].append(case)
        if bv and not av:
            out["lost_schema_valid"].append(case)
        if ae and not be:
            out["newly_exact"].append(case)
        if be and not ae:
            out["lost_exact"].append(case)
    return {k: {"n": len(v), "cases": v} for k, v in out.items()}


def _policy_table(rows: Sequence[Mapping[str, Any]], policies: Sequence[str]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for p in policies:
        drows = _twopass.derive(rows, p)
        tools = sorted({str(r.get("tool")) for r in drows if r.get("tool_case")})
        entry = {"all": _twopass._summary(drows)}
        for t in tools:
            entry[t] = _twopass._summary([r for r in drows if r.get("tool") == t])
        table[p] = entry
    return table


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    p.add_argument("--evidence-root", required=True, type=Path)
    p.add_argument("--artifact", required=True, type=Path)
    p.add_argument("--sha256", required=True)
    p.add_argument("--out-dir", required=True, type=Path)
    return p


def _plan(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    """Every check and every computation; returns ``(json report, markdown)``. Writes nothing."""
    out_dir: Path = args.out_dir
    if out_dir.exists():
        raise Refused(f"--out-dir {out_dir} exists, never overwritten")
    root: Path = args.evidence_root
    path = args.artifact if args.artifact.is_absolute() else root / args.artifact
    raw = _read_bytes(path)
    sha = _sha256(raw)
    if sha != args.sha256:
        raise Refused(f"{path} sha256 {sha} != --sha256 {args.sha256}")
    lines = [json.loads(x) for x in raw.decode().splitlines() if x.strip()]
    if not lines:
        raise Refused(f"{path} is empty")
    regime = lines[0]
    stopped = [x.get("stopped") for x in lines if x.get("pass") not in (1, 2) and x.get("stopped")]
    if stopped:
        raise Refused(f"the artifact was stopped: {stopped}")
    rows = [x for x in lines[1:] if x.get("pass") in (1, 2)]
    cases = _cases(regime.get("cases_sha256"))
    tools = _tool_set(regime.get("tools_sha256"))
    try:
        rescored = _twopass.rescore(rows, tools, cases)
    except ValueError as exc:
        raise Refused(f"rescore refused: {exc}") from None

    cells = [str(c) for c in regime.get("cells") or []]
    seeded = [c for c in cells if _twopass.parse_cell(c).arm == _twopass.SEED_ARM]
    unseeded = [c for c in cells if c not in seeded]
    policies = ["draft_only"]
    policies += [f"cd_on_invalid:{c}" for c in unseeded]
    policies += [f"cd_always:{c}" for c in unseeded]
    table = _policy_table(rescored, policies)
    stored_draft_only = _policy_table(rows, ["draft_only"])["draft_only"]
    changes = _changes(rows, rescored)
    routed = sum(
        1
        for r in rescored
        if r.get("pass") == 1
        and isinstance(r.get("route"), Mapping)
        and r["route"]["kind"] == "toolcall"
    )
    rep: dict[str, Any] = {
        "tool": TOOL_REL,
        "tool_commit": _git_head(),
        "task": "T-0082a",
        "caveat": CAVEAT,
        "artifact": str(args.artifact),
        "artifact_sha256": sha,
        "regime": {k: v for k, v in regime.items() if k != "system_prompt"},
        "n_cases": sum(1 for r in rescored if r.get("pass") == 1),
        "routed": routed,
        "changes": changes,
        "stripped_close": {"k": changes["stripped"]["n"], "n": routed},
        "draft_only_stored": stored_draft_only,
        "cells": cells,
        "stale_cells": [{"cell": c, "status": STALE} for c in seeded],
        "policies": table,
    }
    return rep, _markdown(rep)


def _rate(r: Mapping[str, Any]) -> str:
    return f"{r['k']}/{r['n']}" if r["n"] else "n/a"


def _markdown(rep: Mapping[str, Any]) -> str:
    ch = rep["changes"]
    stored = rep["draft_only_stored"]["all"]
    lines = [
        "# Two-pass tool-call re-score under the `</cc>` strip (T-0082a)",
        "",
        f"> {rep['caveat']}",
        "",
        f"- artifact {rep['artifact']} (sha256 {rep['artifact_sha256']}); stage {rep['regime'].get('stage')}, "
        f"commit {rep['regime'].get('commit')}, cells {rep['cells']}, n {rep['n_cases']}, "
        f"tools sha256 {rep['regime'].get('tools_sha256')}, cases sha256 {rep['regime'].get('cases_sha256')}; "
        f"re-scored by {rep['tool']} at {rep['tool_commit']}",
        f"- drafts with a stripped `</cc>` close: {rep['stripped_close']['k']}/{rep['stripped_close']['n']} "
        "routed",
        f"- newly schema-valid drafts: {ch['newly_schema_valid']['n']}; newly exact: {ch['newly_exact']['n']}; "
        f"lost schema-valid: {ch['lost_schema_valid']['n']}; lost exact: {ch['lost_exact']['n']}",
        f"- draft_only (stored, before the strip): schema_valid {_rate(stored['schema_valid'])}, "
        f"exact {_rate(stored['exact'])}",
    ]
    for s in rep["stale_cells"]:
        lines.append(f"- {s['cell']}: {s['status']}; excluded from the policy table")
    lines += [
        "",
        "| policy | schema_valid | exact | CS | pass 2 sent |",
        "|---|---|---|---|---|",
    ]
    for p, entry in rep["policies"].items():
        s = entry["all"]
        name = "draft_only (re-scored)" if p == "draft_only" else p
        lines.append(
            f"| {name} | {_rate(s['schema_valid'])} | {_rate(s['exact'])} | {_rate(s['cs'])} "
            f"| {s['pass2_sent']} |"
        )
    lines += ["", "| policy | tool | schema_valid | exact |", "|---|---|---|---|"]
    for p, entry in rep["policies"].items():
        for t, s in entry.items():
            if t != "all":
                lines.append(f"| {p} | {t} | {_rate(s['schema_valid'])} | {_rate(s['exact'])} |")
    lines += ["", f"Stripped cases: {', '.join(ch['stripped']['cases']) or 'none'}"]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        rep, md = _plan(args)
    except Refused as exc:
        _say(f"REFUSED: {exc}")
        return EXIT_REFUSED
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "report-rescore.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
    )
    (out_dir / "report-rescore.md").write_text(md)
    _say(f"wrote {out_dir}/report-rescore.json and .md")
    sys.stdout.write(md)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
