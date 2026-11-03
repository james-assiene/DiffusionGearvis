#!/usr/bin/env python3
"""T-0086 -- re-score a stored two-pass tool-call artifact under the plausible-query ``exact`` rule. CPU only, no
network.

    uv run python scripts/cd_toolcall_rescore_exact.py \
        --evidence-root logs \
        --artifact t0084-native-test-794161d/20261011T031521.606364Z-toolcall-test.jsonl \
        --sha256 14728634... --out-dir <new dir>

``--artifact`` resolves against ``--evidence-root`` (an absolute path is taken as is). Every input is opened
read-only; the only writes are the two report files in the new ``--out-dir``.

- **Pass 1:** each row is re-routed and re-scored from its stored ``text`` by ``twopass.rescore`` (current router
  and scorer). Its "old" flag is the 4c7dfdc rule on the re-scored draft call (ruling S1: the router held fixed, so
  the T-0080c artifacts, which predate the ``</cc>`` strip, get their strip-rescored counts); rows without a ``text``
  keep their stored flag as "old".
- **Pass 2:** "old" is the stored ``exact``; "new" is ``score.exact`` recomputed from the stored ``call`` (``None``
  is not exact).
- **Consistency (criterion 7):** for a google_search call against a google_search case, new == (old and
  ``query_problem(query) is None``); for every other row, new == old. Any violation refuses, naming case, cell and
  pass.
- **Report** (``report-rescore-exact.{json,md}``): every ``derive`` policy (draft policies, and ``cd_always`` /
  ``cd_on_invalid`` over every cell, seeded cells included, ruling S4), old against new exact overall and per tool;
  the flipped rows (pass, case, cell, reason) and a reason histogram; the counterfactual selection of the
  artifact's own rule, old beside new (ruling S3: cc-only dev ``select("stage1")``, cc-only test and the T-0082c
  replays ``select("stage2")``, mixed ``select_format`` -- on test with the recorded θ_n T4 on both lines). The
  selection is labelled "counterfactual; not a decision"; no selection file is written.

Ground truth is ``tests/eval/toolcall/cases.jsonl`` at the MANIFEST-pinned sha, which must also be the artifact's.

Exit 0: both files written. Exit 2: refused (``REFUSED:``; nothing written): sha mismatch, an existing out-dir, a
stopped artifact, cases or tools that are not the artifact's, a ``rescore`` refusal, or a consistency violation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from local_astra.eval.cd import twopass as _twopass
from local_astra.toolcall import score as _score
from local_astra.toolcall import tools as _tools

EXIT_OK = 0
EXIT_REFUSED = 2

TOOL_REL = "scripts/cd_toolcall_rescore_exact.py"
REPO = Path(__file__).resolve().parents[1]
SET_DIR = REPO / "tests" / "eval" / "toolcall"
JSON_NAME = "report-rescore-exact.json"
MD_NAME = "report-rescore-exact.md"
LABEL = "counterfactual; not a decision"
#: θ_n recorded by the T-0084e dev selection (selection-native-dev.json, H-T84-1); ruling S3 uses it on both lines
RECORDED_NATIVE_THOUGHT = "T4"
CAVEAT = (
    "Pre-registration caveat: the plausible-query rule (T-0086) was written after reading the stored queries, so "
    "this re-score is a counterfactual on data already seen, not a pre-registered measurement. No model was run: "
    "pass-1 drafts are re-scored from their stored text and pass-2 calls from their stored call. Decisions stand "
    "as recorded unless the human rules otherwise."
)


class Refused(Exception):
    """A refusal: exit 2, nothing written."""


def _say(msg: str) -> None:
    print(f"cd_toolcall_rescore_exact: {msg}", flush=True)


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


# --------------------------------------------------------------------------- old and new flags


def _old_exact(call: Mapping[str, Any], gt: Mapping[str, Any]) -> bool:
    """The 4c7dfdc rule: for google_search the keyword rule alone (the first conjunct of the new rule, unchanged);
    every other tool is scored by ``score.exact`` itself (criterion 4: unchanged off google_search)."""
    args = call.get("arguments")
    if call.get("name") == gt.get("name") == "google_search" and isinstance(args, Mapping):
        return _score._keywords_present(args, gt.get("keywords"))
    return _score.exact(call, gt)


def _flag(rule: Any, call: Any, gt: Any) -> bool:
    return isinstance(call, Mapping) and bool(gt) and bool(rule(call, gt))


def _search_query(call: Any, gt: Any) -> tuple[bool, Any]:
    """``(is a google_search call against a google_search case, its query)``."""
    if not (isinstance(call, Mapping) and isinstance(gt, Mapping)):
        return False, None
    if call.get("name") != "google_search" or gt.get("name") != "google_search":
        return False, None
    args = call.get("arguments")
    return True, (args.get("query") if isinstance(args, Mapping) else None)


def _check(where: str, old: bool, new: bool, call: Any, gt: Any) -> str | None:
    """Criterion 7; returns the flip reason (``query_problem``) when old is True and new False."""
    is_search, query = _search_query(call, gt)
    if is_search:
        plausible = isinstance(query, str) and _score.query_problem(query) is None
        expected = old and plausible
    else:
        expected = old
    if new != expected:
        raise Refused(
            f"{where}: stored/old exact {old}, recomputed {new}, expected {expected} (criterion 7)"
        )
    if old and not new:
        return _score.query_problem(query) if isinstance(query, str) else "not a string"
    return None


def _old_new(
    rows: Sequence[Mapping[str, Any]],
    rescored: Sequence[Mapping[str, Any]],
    gts: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """``(old rows, new rows, flipped)``: the rescored rows with the old and the new exact flag, and every row whose
    flag flipped."""
    old_rows: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    flipped: list[dict[str, Any]] = []
    for orig, r in zip(rows, rescored, strict=True):
        case = str(r["case"])
        if case not in gts:
            raise Refused(f"case {case}: not in cases.jsonl")
        gt = gts[case]
        if r.get("pass") == 1:
            call, key, cell = r.get("draft_call"), "draft_exact", None
            where = f"case {case}, format {_twopass.row_format(r)}, pass 1"
            new = _flag(_score.exact, call, gt)
            if "text" in orig:
                old = _flag(_old_exact, call, gt)
                if new != bool(r.get("draft_exact")):
                    raise Refused(f"{where}: rescore's draft_exact is not the recomputed {new}")
            else:
                old = bool(r.get("draft_exact"))
        else:
            call, key, cell = r.get("call"), "exact", str(r.get("cell"))
            where = f"case {case}, cell {cell}, pass 2"
            old, new = bool(r.get("exact")), _flag(_score.exact, call, gt)
        reason = _check(where, old, new, call, gt)
        if reason is not None:
            flipped.append(
                {
                    "pass": r["pass"],
                    "case": case,
                    "cell": cell,
                    "format": _twopass.row_format(r),
                    "reason": reason,
                }
            )
        old_rows.append({**r, key: old})
        new_rows.append({**r, key: new})
    return old_rows, new_rows, flipped


# --------------------------------------------------------------------------- policies and selection


def _policies(cells: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], bool]:
    """Every ``derive`` policy of the artifact (as ``twopass.report``), and whether it is mixed."""
    present = {_twopass.cell_format(c) for c in cells} | {_twopass.row_format(r) for r in rows}
    mixed = "native" in present
    drafts = ["draft_only"]
    if mixed:
        drafts = [
            "draft_only" if f == "cc" else _twopass.DRAFT_ONLY_NATIVE
            for f in _twopass.FORMATS
            if f in present
        ]
    return [
        *drafts,
        *[f"cd_always:{c}" for c in cells],
        *[f"cd_on_invalid:{c}" for c in cells],
    ], mixed


def _table(derived: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p, drows in derived.items():
        tools = sorted({str(r.get("tool")) for r in drows if r.get("tool_case")})
        entry = {"all": _twopass._summary(drows)}
        for t in tools:
            entry[t] = _twopass._summary([r for r in drows if r.get("tool") == t])
        out[p] = entry
    return out


def _cc_rule(regime: Mapping[str, Any]) -> str:
    """Ruling S3: a dev artifact's own rule is stage 1; test and the T-0082c replays (``pass1_from``) stage 2."""
    if regime.get("stage") == "dev" and not regime.get("pass1_from"):
        return "stage1"
    return "stage2"


def _cc_choice(
    derived: Mapping[str, Sequence[Mapping[str, Any]]], cells: Sequence[str], rule: str
) -> dict[str, Any]:
    if rule == "stage1":
        rows = [
            *derived["draft_only"],
            *[{**r, "policy": c} for c in cells for r in derived[f"cd_always:{c}"]],
        ]
    else:
        rows = [r for rs in derived.values() for r in rs]
    try:
        return {"choice": _twopass.select(rows, rule)}
    except ValueError as exc:
        return {"choice": None, "status": f"not computable ({exc})"}


def _family_cells(theta: str) -> list[str]:
    fam = _twopass.format_family(theta)
    return list(dict.fromkeys(p.split(":", 1)[1] for p in fam if p.startswith("cd_")))


def _format_choice(
    derived: Mapping[str, Sequence[Mapping[str, Any]]], cells: Sequence[str], stage: str
) -> dict[str, Any]:
    """``select_format`` (T-0084d) on one side's policy rows. Dev runs step 1 to find θ_n first (as ``select_format``
    does), so a family whose cells were not run is reported as such rather than chosen."""
    rows = [r for rs in derived.values() for r in rs]
    try:
        if stage == "dev":
            step1 = [
                f"cd_always:{_twopass.NATIVE_PREFIX}{t}/named/perstep_mar"
                for t in _twopass.THOUGHTS
            ]
            step1_rows = [
                {**r, "policy": _twopass._bare(p).removeprefix("cd_always:")}
                for p in step1
                for r in derived.get(p, [])
            ]
            theta = _twopass.select(step1_rows, "stage1").split("/")[0]
        else:
            theta = RECORDED_NATIVE_THOUGHT
        missing = [c for c in _family_cells(theta) if c not in cells]
        if missing:
            return {
                "native_thought": theta,
                "choice": None,
                "status": f"not computable (cells not run: {', '.join(missing)})",
            }
        if stage == "dev":
            fc = _twopass.select_format(rows, "dev")
        else:
            fc = _twopass.select_format(rows, "test", native_thought=theta)
    except ValueError as exc:
        return {"choice": None, "status": f"not computable ({exc})"}
    return {"native_thought": fc.native_thought, "choice": fc.choice, "winner": fc.winner}


def _selection(
    sides: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    regime: Mapping[str, Any],
    cells: Sequence[str],
    mixed: bool,
) -> dict[str, Any]:
    stage = str(regime.get("stage"))
    sel: dict[str, Any] = {"label": LABEL}
    if mixed:
        sel["rule"] = f"select_format ({_twopass.FORMAT_RULE}), stage {stage}"
        if stage != "dev":
            sel["note"] = (
                f"θ_n is the recorded {RECORDED_NATIVE_THOUGHT} (T-0084e dev selection) on both lines (ruling S3); "
                "the dev counterfactual θ_n is in the dev artifact's report"
            )
        for side, derived in sides.items():
            sel[side] = _format_choice(derived, cells, stage)
    else:
        rule = _cc_rule(regime)
        sel["rule"] = f'select ("{rule}")'
        for side, derived in sides.items():
            sel[side] = _cc_choice(derived, cells, rule)
    return sel


# --------------------------------------------------------------------------- the plan


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
    gts = {str(c["id"]): c.get("gt") for c in cases}
    old_rows, new_rows, flipped = _old_new(rows, rescored, gts)

    cells = [str(c) for c in regime.get("cells") or []]
    policies, mixed = _policies(cells, rows)
    sides = {
        "old": {p: _twopass.derive(old_rows, p) for p in policies},
        "new": {p: _twopass.derive(new_rows, p) for p in policies},
    }
    old_t, new_t = _table(sides["old"]), _table(sides["new"])
    per_cell = Counter(f["cell"] if f["pass"] == 2 else f"pass 1 ({f['format']})" for f in flipped)
    rep: dict[str, Any] = {
        "tool": TOOL_REL,
        "tool_commit": _git_head(),
        "task": "T-0086",
        "caveat": CAVEAT,
        "artifact": str(args.artifact),
        "artifact_sha256": sha,
        "regime": {k: v for k, v in regime.items() if k != "system_prompt"},
        "n_cases": sum(1 for r in rows if r.get("pass") == 1),
        "cells": cells,
        "mixed": mixed,
        "policies": {p: {"old": old_t[p], "new": new_t[p]} for p in policies},
        "flipped": flipped,
        "flipped_per_cell": dict(per_cell),
        "reasons": dict(Counter(f["reason"] for f in flipped)),
        "selection": _selection(sides, regime, cells, mixed),
    }
    return rep, _markdown(rep)


def _rate(r: Mapping[str, Any]) -> str:
    return f"{r['k']}/{r['n']}" if r["n"] else "n/a"


def _sel_line(s: Mapping[str, Any]) -> str:
    if s.get("choice") is None:
        head = f"θ_n {s['native_thought']}: " if s.get("native_thought") else ""
        return head + str(s.get("status"))
    parts = [f"choice `{s['choice']}`"]
    if s.get("native_thought"):
        parts.insert(0, f"θ_n {s['native_thought']}")
    if s.get("winner"):
        parts.append(f"winner {s['winner']}")
    return ", ".join(parts)


def _markdown(rep: Mapping[str, Any]) -> str:
    reg = rep["regime"]
    lines = [
        "# Two-pass tool-call re-score under the plausible-query exact rule (T-0086)",
        "",
        f"> {rep['caveat']}",
        "",
        f"- artifact {rep['artifact']} (sha256 {rep['artifact_sha256']}); stage {reg.get('stage')}, "
        f"commit {reg.get('commit')}, n {rep['n_cases']} pass-1 rows, cells {rep['cells']}, "
        f"tools sha256 {reg.get('tools_sha256')}, cases sha256 {reg.get('cases_sha256')}; "
        f"re-scored by {rep['tool']} at {rep['tool_commit']}",
        f"- flipped rows (old exact, new not): {len(rep['flipped'])}",
        "",
        "## Exact, old (4c7dfdc rule) against new (T-0086 rule)",
        "",
        "| policy | old exact | new exact | old google_search | new google_search |",
        "|---|---|---|---|---|",
    ]
    for p, e in rep["policies"].items():
        og = e["old"].get("google_search", {}).get("exact", {"k": 0, "n": 0})
        ng = e["new"].get("google_search", {}).get("exact", {"k": 0, "n": 0})
        lines.append(
            f"| {p} | {_rate(e['old']['all']['exact'])} | {_rate(e['new']['all']['exact'])} "
            f"| {_rate(og)} | {_rate(ng)} |"
        )
    lines += ["", "## Flipped rows per cell and pass", "", "| cell / pass | flipped |", "|---|---|"]
    for c, n in rep["flipped_per_cell"].items():
        lines.append(f"| {c} | {n} |")
    lines += [
        "",
        "## query_problem reasons over the flipped rows",
        "",
        "| reason | rows |",
        "|---|---|",
    ]
    for r, n in sorted(rep["reasons"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {r} | {n} |")
    sel = rep["selection"]
    lines += [
        "",
        f"## Selection ({sel['label']})",
        "",
        f"Rule: {sel['rule']}.",
    ]
    if sel.get("note"):
        lines.append(f"Note: {sel['note']}.")
    lines += [
        "",
        f"- old: {_sel_line(sel['old'])}",
        f"- new: {_sel_line(sel['new'])}",
        "",
        "## Flipped rows",
        "",
        "| pass | case | cell | reason |",
        "|---|---|---|---|",
    ]
    for f in rep["flipped"]:
        lines.append(f"| {f['pass']} | {f['case']} | {f['cell'] or f['format']} | {f['reason']} |")
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
    (out_dir / JSON_NAME).write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    (out_dir / MD_NAME).write_text(md)
    _say(f"wrote {out_dir}/{JSON_NAME} and .md")
    sys.stdout.write(md)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
