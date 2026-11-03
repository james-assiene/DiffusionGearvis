"""Regenerate ``prompts.json``: diffgemma-fa's own prompt builders at 27c7d0d, run on this repo's fixtures.

Usage (host, no model, no GPU)::

    python3 tests/fixtures/cd/prompts/generate.py <diffgemma-fa clone>

The two builders are read with ``git show 27c7d0d:<path>`` and executed from their source text, so the
fixture is produced by the upstream code, not by anything in ``local_astra``:

- BFCL: ``diffgemma_fa/eval/run.py::build_prompt(rec, fn, style="stock")`` (with ``PROMPT_STYLES``);
- Sudoku: ``diffgemma_fa/compile/tasks/sudoku.py::build_prompt(rec)``.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CD = HERE.parent
COMMIT = "27c7d0d"


def _source(repo: Path, path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{COMMIT}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _extract(src: str, names: set[str]) -> str:
    tree = ast.parse(src)
    keep = [
        n
        for n in tree.body
        if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))
    ]
    lines = src.splitlines()
    out = []
    for n in keep:  # decorators included (ast.get_source_segment would drop them)
        first = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
        out.append("\n".join(lines[first - 1 : n.end_lineno]))
    return "\n\n".join(out)


def main(repo: Path) -> None:
    ns_bfcl: dict = {}
    exec(
        _extract(_source(repo, "diffgemma_fa/eval/run.py"), {"PROMPT_STYLES", "build_prompt"}),
        ns_bfcl,
    )
    ns_sud: dict = {"dataclasses": dataclasses}
    exec(
        _extract(
            _source(repo, "diffgemma_fa/compile/tasks/sudoku.py"), {"SudokuRecord", "build_prompt"}
        ),
        ns_sud,
    )

    records = [
        json.loads(line)
        for line in (CD / "bfcl" / "BFCL_v4_live_simple.json").read_text().splitlines()
        if line.strip()
    ]
    by_id = {r["id"]: r for r in records}
    single = [r["id"] for r in records if len(r.get("function") or ()) == 1]
    test_ids = [
        r["id"]
        for r in json.loads((CD / "dgfa_preserved" / "exp_e5_grammar130_map.json").read_text())[
            "rows"
        ]
    ]
    dev_ids = single[130:230]

    class _Rec:
        def __init__(self, raw: dict) -> None:
            self.id = raw["id"]
            self.question = raw["question"]

    def bfcl(ids: list[str]) -> list[dict]:
        out = []
        for rid in ids:
            raw = by_id[rid]
            out.append(
                {
                    "id": rid,
                    "prompt": ns_bfcl["build_prompt"](_Rec(raw), raw["function"][0], "stock"),
                }
            )
        return out

    puzzles = json.loads((CD / "dgfa_preserved" / "sudoku_puzzles.json").read_text())["cuts"]

    def sudoku(cut: str) -> list[dict]:
        out = []
        for r in puzzles[cut]["records"]:
            rec = ns_sud["SudokuRecord"](
                r["id"], tuple(map(tuple, r["puzzle"])), tuple(map(tuple, r["solution"]))
            )
            out.append(
                {"id": r["id"], "puzzle": r["puzzle"], "prompt": ns_sud["build_prompt"](rec)}
            )
        return out

    data = {
        "generator": f"tests/fixtures/cd/prompts/generate.py over diffgemma-fa {COMMIT}",
        "bfcl": {"test": bfcl(test_ids), "dev": bfcl(dev_ids)},
        "sudoku": {"test": sudoku("test"), "dev": sudoku("dev")},
    }
    (HERE / "prompts.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
