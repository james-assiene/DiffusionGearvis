"""The four cuts of the CD evaluation and their scorers (T-0070 criteria 1, 3; ruling E1, E2, E4).

Cuts:

- **BFCL test** = the 130 ids of diffgemma-fa's ``exp_e5_grammar130_map``: the first 130 single-function
  records of ``BFCL_v4_live_simple.json`` in file order (``eval/run.py`` with offset 0, n 130). The ids are
  pinned by the sha256 of their newline-joined list, so a changed data file cannot silently move the cut.
- **BFCL dev** = single-function records 130-229 (n = 100).
- **Sudoku test** = ``generate(250, seed=0, n_blanks=6)``, ids ``sudoku4_0`` ... ``sudoku4_249`` (they join
  the preserved JAX rows).
- **Sudoku dev** = ``generate(100, seed=1, n_blanks=6)``. ``generate`` names both cuts ``sudoku4_<k>``;
  dev ids are cut-qualified (``dev:sudoku4_<k>``, ruling E4) so the two cuts can never join on id.

The BFCL files are the pinned gorilla ``6ea57973c7a6`` copies (T-0062 decision 3); the directory defaults
to ``tests/fixtures/cd/bfcl`` of this checkout and is overridden with ``ASTRA_CD_BFCL_DIR``. Their sha256
is checked on load.

Scoring (``score``) is diffgemma-fa's, vendored unmodified (``_vendor/metrics.py``, ``_vendor/tasks``):
BFCL arguments with BFCL's own normalisation, schema validity against the normalised schema, Sudoku by
``score_solution``. Ruling E2: ``ci_enums`` and ``expand_wildcard`` default **on** (the grammar's own
language); the published unconstrained figures are reproduced with both off.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from local_astra.eval.cd._vendor import metrics as _metrics
from local_astra.eval.cd._vendor.tasks import produced_an_answer
from local_astra.eval.cd._vendor.tasks import sudoku as _sudoku

__all__ = [
    "BFCL_DATA_SHA256",
    "BfclCase",
    "SudokuCase",
    "bfcl_data_dir",
    "case_insensitive_enums",
    "load",
    "score",
]

TASKS = ("bfcl", "sudoku")
CUTS = ("test", "dev")

_REPO = Path(__file__).resolve().parents[3]
BFCL_ENV = "ASTRA_CD_BFCL_DIR"
BFCL_FILE = "BFCL_v4_live_simple.json"
#: tests/fixtures/cd/bfcl/MANIFEST.md (gorilla 6ea57973c7a6).
BFCL_DATA_SHA256 = {
    BFCL_FILE: "1af2ac87dca47556db7b7e37e51e28b459a38b594e3c7b3c792b4903598ca0c4",
    f"possible_answer/{BFCL_FILE}": (
        "fec9cfa9744a936f9126981e85a2023da1e63e273eafebc81923a1162fad70ce"
    ),
}
#: sha256 of "\n".join(ids of diffgemma-fa preserved/exp_e5_grammar130_map.json), in its order.
BFCL_TEST_IDS_SHA256 = "8bc6f0866776aa329ecd92fb0c5c4fea43a7c96048a9379c3143e8d27c12f766"
BFCL_TEST_SLICE = (0, 130)
BFCL_DEV_SLICE = (130, 230)
#: (n, seed, n_blanks) of each Sudoku cut.
SUDOKU_CUTS = {"test": (250, 0, 6), "dev": (100, 1, 6)}

SudokuCase = _sudoku.SudokuRecord


@dataclasses.dataclass(frozen=True)
class BfclCase:
    """One single-function BFCL-Live simple record with its ground truth."""

    id: str
    function: Mapping[str, Any]
    question: list[Any]
    #: ``possible_answer``'s ground truth (list of ``{fn_name: {arg: [acceptable...]}}``); may be empty.
    ground_truth: list[Any]


def bfcl_data_dir() -> Path:
    env = os.environ.get(BFCL_ENV)
    return Path(env) if env else _REPO / "tests" / "fixtures" / "cd" / "bfcl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@functools.cache
def _bfcl_single() -> tuple[BfclCase, ...]:
    root = bfcl_data_dir()
    for rel, want in BFCL_DATA_SHA256.items():
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(
                f"BFCL data missing: {path} (T-0062 decision 3; set {BFCL_ENV} or fetch the pinned files)"
            )
        got = _sha256(path)
        if got != want:
            raise ValueError(f"{path} has sha256 {got}, not the pinned {want}")
    truth = {
        str(r["id"]): list(r.get("ground_truth") or [])
        for r in _json_lines(root / "possible_answer" / BFCL_FILE)
        if "id" in r
    }
    out = []
    for rec in _json_lines(root / BFCL_FILE):
        fns = rec.get("function") or []
        if len(fns) != 1:
            continue
        rid = str(rec["id"])
        out.append(
            BfclCase(
                id=rid,
                function=fns[0],
                question=list(rec.get("question") or []),
                ground_truth=truth.get(rid, []),
            )
        )
    return tuple(out)


@functools.cache
def _load(task: str, cut: str) -> tuple[Any, ...]:
    if task == "bfcl":
        lo, hi = BFCL_TEST_SLICE if cut == "test" else BFCL_DEV_SLICE
        recs = _bfcl_single()[lo:hi]
        if len(recs) != hi - lo:
            raise ValueError(f"BFCL {cut} cut: expected {hi - lo} records, found {len(recs)}")
        if cut == "test":
            digest = hashlib.sha256("\n".join(r.id for r in recs).encode()).hexdigest()
            if digest != BFCL_TEST_IDS_SHA256:
                raise ValueError("BFCL test cut is not the 130 ids of exp_e5_grammar130_map")
        return recs
    n, seed, blanks = SUDOKU_CUTS[cut]
    puzzles = _sudoku.generate(n, seed=seed, n_blanks=blanks)
    if cut == "dev":
        puzzles = [dataclasses.replace(r, id=f"dev:{r.id}") for r in puzzles]
    return tuple(puzzles)


def load(task: str, cut: str) -> Sequence[Any]:
    """The records of ``cut`` for ``task``, in order (``BfclCase`` or the vendored ``SudokuRecord``)."""
    if task not in TASKS or cut not in CUTS:
        raise ValueError(f"unknown task/cut {task!r}/{cut!r}")
    return _load(task, cut)


def case_insensitive_enums(schema: Any) -> Any:
    """diffgemma-fa ``eval/run.py::_case_insensitive_enums`` (27c7d0d), as ported by T-0063."""
    from local_astra.decoding.grammars import case_insensitive_enums as ci

    return ci(schema)


def _normalised_schema(
    function: Mapping[str, Any], *, ci_enums: bool, expand_wildcard: bool
) -> Any:
    from local_astra.decoding._vendor.dgfa_compile import schema as _schema

    params = function["parameters"]
    if ci_enums:
        params = case_insensitive_enums(params)
    return _schema.normalize_bfcl_schema(params, expand_wildcard=expand_wildcard)


def _want(case: BfclCase) -> dict[str, Any] | None:
    from local_astra.decoding._vendor.dgfa_compile import bfcl_data

    if not case.ground_truth:
        return None
    block = case.ground_truth[0].get(case.function.get("name"), {})
    want: dict[str, Any] = bfcl_data.materialize_ground_truth(block, case.function["parameters"])
    return want


def score(
    task: str,
    record: Any,
    text: str,
    *,
    ci_enums: bool = True,
    expand_wildcard: bool = True,
) -> dict[str, Any]:
    """Score one emission of ``record``.

    BFCL -> ``{"parsed", "schema_valid", "arg_correct", "arg_total", "exact_call"}``; ``arg_total`` is
    ``len(want)`` whatever the emission ([AUDIT-B]). Sudoku -> ``{"solved", "why", "parsed"}``, where
    ``parsed`` is diffgemma-fa's ``produced_an_answer``.
    """
    if task == "sudoku":
        ok, why = _sudoku.score_solution(text, record)
        return {"solved": bool(ok), "why": why, "parsed": produced_an_answer(why)}
    if task != "bfcl":
        raise ValueError(f"unknown task {task!r}")
    parsed = _metrics.extract_json(text)
    norm = _normalised_schema(record.function, ci_enums=ci_enums, expand_wildcard=expand_wildcard)
    valid = bool(_metrics.schema_valid(parsed, norm))
    want = _want(record)
    if want is None:
        n_ok, n_tot = 0, 0
    else:
        n_ok, n_tot, _ = _metrics.score_arguments(parsed or {}, want)
    return {
        "parsed": parsed is not None,
        "schema_valid": valid,
        "arg_correct": int(n_ok),
        "arg_total": int(n_tot),
        "exact_call": bool(n_tot > 0 and n_ok == n_tot),
    }
