"""Shared helpers for the T-0070 rows (evaluation harness). Not a test module.

The API these rows assume (the test-author's proposal; the orchestrator's ruling fixes it):

- ``local_astra.eval.cd.cuts.load(task, cut)`` -> sequence of records with ``.id``; BFCL records also have
  ``.function`` (the single function dict) and ``.question``; Sudoku records ``.puzzle`` and ``.solution``.
  ``task in {"bfcl", "sudoku"}``, ``cut in {"test", "dev"}``.
- ``local_astra.eval.cd.prompts.build_prompt(task, record) -> str``.
- ``local_astra.eval.cd.runner.RunConfig(task, cut, arm, base_url, engine_log, out_dir, repo, expected_commit,
  calibration=None, limit=None, budget_s=14400.0, request_timeout_s=..., whitespace="json", max_states=512,
  step_cap=48)`` and ``runner.run(cfg) -> RunResult`` with ``exit_code``, ``artifact`` (Path) and ``stopped``
  (``None`` or ``"<letter>: <text>"``, letter a-h of criterion 8). A non-loopback ``base_url`` raises
  ``ValueError`` before any socket is opened.
- Arms: ``unconstrained``, ``end_projection``, ``perstep_mf``, ``perstep_mar``, ``perstep_mar_sc``; the
  pretty-whitespace parity arm is ``whitespace="pretty"`` on a constrained arm (T-0062 decision 11).
- Artifact rows (criterion 9): ``id, text, committed_ids, parse, scores, cs, steps, wall_ms, bucket, error,
  knobs`` and, for ``end_projection``, ``shadow_argmax_ids, argmax_in_language, map_equals_argmax``.
  ``scores`` is ``{"solved", "why"}`` (Sudoku) or ``{"schema_valid", "arg_correct", "arg_total",
  "exact_call"}`` (BFCL). ``error`` is ``None`` or a code (``join_error``, ``too_large``, ``timeout``,
  ``http_error``, or the record's CD error code).
"""

from __future__ import annotations

import functools
import importlib
import json
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.cd import _compile_support as cs
from tests.cd.fakes.vllm_server import FakeVllmServer, Reply, user_prompt

PROMPTS = cs.FIXTURES / "prompts" / "prompts.json"
HEADER = "<|channel>thought\n<channel|>"

ARMS = ("unconstrained", "end_projection", "perstep_mf", "perstep_mar", "perstep_mar_sc")
CONSTRAINED = ARMS[1:]


def need(modname: str) -> ModuleType:
    """Import a T-0070 module; a missing one fails the row with "T-0070 not implemented"."""
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (modname == missing or modname.startswith(missing + ".")):
            pytest.fail(f"T-0070 not implemented: `{modname}` does not exist ({exc})")
        raise


def cuts() -> ModuleType:
    return need("local_astra.eval.cd.cuts")


def prompts() -> ModuleType:
    return need("local_astra.eval.cd.prompts")


def runner() -> ModuleType:
    cs.grammars()  # install the test-side compile cache before the runner compiles through grammars.*
    return need("local_astra.eval.cd.runner")


def stats() -> ModuleType:
    return need("local_astra.eval.cd.stats")


def report() -> ModuleType:
    return need("local_astra.eval.cd.report")


def calibration() -> ModuleType:
    return need("local_astra.eval.cd.calibration")


# --------------------------------------------------------------------------- fixtures


@functools.cache
def prompt_fixture() -> Mapping[str, Any]:
    data: Mapping[str, Any] = json.loads(PROMPTS.read_text())
    return data


def prompt_to_id(task: str, cut: str) -> dict[str, str]:
    """Prompt -> fixture id, for cuts whose prompts are unique (both Sudoku cuts, BFCL dev). The BFCL
    test cut has three pairs of records with identical prompts; use ``BfclResolver`` there."""
    rows = prompt_fixture()[task][cut]
    out = {r["prompt"]: r["id"] for r in rows}
    assert len(out) == len(rows), f"{task}/{cut} has colliding prompts: use BfclResolver"
    return out


def unqualified(record_id: str) -> str:
    """Ruling E4: dev ids are cut-qualified (``dev:sudoku4_<k>``); strip the qualifier."""
    return record_id.split(":", 1)[1] if ":" in record_id else record_id


class BfclResolver:
    """Maps each served BFCL request to its record id, collisions included.

    Three pairs of BFCL test-cut records share a prompt (32-9-0/33-10-0, 36-13-0/37-14-0,
    58-27-0/59-28-0) with *different* functions. A constrained request is disambiguated by its
    automaton (the candidate whose compiled grammar -- JSON or pretty whitespace -- encodes to the sent
    payload); an unconstrained one by occurrence order (the k-th request with a shared prompt is the
    k-th record with it in cut order). Call it once per request, from the fake's script; the ids are
    kept in ``resolved`` in arrival order.
    """

    def __init__(self, cut: str) -> None:
        self.by_prompt: dict[str, list[str]] = {}
        for r in prompt_fixture()["bfcl"][cut]:
            self.by_prompt.setdefault(r["prompt"], []).append(r["id"])
        self.occurrences: dict[str, int] = {}
        self.resolved: list[str] = []
        self._lock = threading.Lock()

    @staticmethod
    def _payloads(rid: str) -> set[str]:
        from tests.cd import _wire_support as ws

        fn = cs.bfcl_function(rid)
        return {
            ws.encode(cs.compiled_bfcl(rid)),
            ws.encode(cs.grammars().compile_bfcl(fn, whitespace="json")),
        }

    def __call__(self, body: Mapping[str, Any]) -> str:
        prompt = user_prompt(body)
        cands = self.by_prompt[prompt]
        payload = (body.get("vllm_xargs") or {}).get("astra_cd_automaton")
        with self._lock:
            if len(cands) == 1:
                rid = cands[0]
            elif payload is not None:
                hits = [c for c in cands if payload in self._payloads(c)]
                assert len(hits) == 1, f"payload matches {hits} among {cands}"
                rid = hits[0]
            else:
                k = self.occurrences.get(prompt, 0)
                self.occurrences[prompt] = k + 1
                rid = cands[min(k, len(cands) - 1)]
            self.resolved.append(rid)
        return rid


def sudoku_by_id(cut: str) -> dict[str, tuple[Any, Any]]:
    return {rid: (p, s) for rid, p, s in cs.sudoku_cut(cut)}


def grid_text(grid: Sequence[Sequence[int]]) -> str:
    return "\n".join("".join(str(c) for c in row) for row in grid)


def solved_text(solution: Sequence[Sequence[int]]) -> str:
    """Accepted by the puzzle's automaton and solves it."""
    return HEADER + grid_text(solution)


def wrong_text(puzzle: Sequence[Sequence[int]], solution: Sequence[Sequence[int]]) -> str:
    """Accepted (givens kept, every blank a digit) but not a solution: the first blank is changed."""
    g = [list(r) for r in solution]
    for r in range(4):
        for c in range(4):
            if puzzle[r][c] == 0:
                g[r][c] = g[r][c] % 4 + 1
                return HEADER + grid_text(g)
    raise AssertionError("puzzle has no blank")


def rejected_text(puzzle: Sequence[Sequence[int]], solution: Sequence[Sequence[int]]) -> str:
    """Rejected by the puzzle's automaton: the first given is overwritten."""
    g = [list(r) for r in solution]
    for r in range(4):
        for c in range(4):
            if puzzle[r][c] != 0:
                g[r][c] = g[r][c] % 4 + 1
                return HEADER + grid_text(g)
    raise AssertionError("puzzle has no given")


def sudoku_script(
    cut: str, decide: Callable[[int, str, Mapping[str, Any]], Reply | str]
) -> Callable[[int, Mapping[str, Any]], Reply]:
    """``decide(index, record_id, body)`` returns a Reply, or one of ``"solved" | "wrong" | "rejected"``."""
    ids = prompt_to_id("sudoku", cut)
    by_id = sudoku_by_id(cut)

    def script(index: int, body: Mapping[str, Any]) -> Reply:
        rid = ids[user_prompt(body)]
        choice = decide(index, rid, body)
        if isinstance(choice, Reply):
            return choice
        puzzle, solution = by_id[rid]
        text = {
            "solved": solved_text,
            "wrong": lambda s: wrong_text(puzzle, s),
            "rejected": lambda s: rejected_text(puzzle, s),
        }[choice](solution)
        return Reply(text=text, in_final=choice != "rejected")

    return script


def fake(
    script: Callable[[int, Mapping[str, Any]], Reply], workdir: Path, **kw: Any
) -> FakeVllmServer:
    from tests.cd import _wire_support as ws

    return FakeVllmServer(script, workdir, encode=cs.encode, parse_xargs=ws.parse, **kw)


# --------------------------------------------------------------------------- runs


def clean_repo(tmp_path: Path) -> tuple[Path, str]:
    """A throwaway clean git repository and its HEAD, so the runs never depend on this worktree's state."""
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], check=True)
    (repo / "README").write_text("x\n")
    subprocess.run([*git, "add", "README"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "init"], check=True)
    head = subprocess.run(
        [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return repo, head


def run(srv: FakeVllmServer, tmp_path: Path, **cfg: Any) -> Any:
    r = runner()
    repo, head = cfg.pop("repo", None), cfg.pop("expected_commit", None)
    if repo is None:
        repo, head = clean_repo(tmp_path)
    out = tmp_path / "artifacts"
    out.mkdir(exist_ok=True)
    config = r.RunConfig(
        base_url=srv.url,
        engine_log=srv.engine_log,
        out_dir=out,
        repo=repo,
        expected_commit=head,
        **cfg,
    )
    return r.run(config)


def read_artifact(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """``(regime, record rows, other lines)``; record rows are the lines carrying an ``id``."""
    lines = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    assert lines, f"{path} is empty"
    regime, rest = lines[0], lines[1:]
    rows = [x for x in rest if "id" in x]
    other = [x for x in rest if "id" not in x]
    return regime, rows, other


def stopped_marker(path: Path) -> str | None:
    """The ``stopped`` value written anywhere in the artifact (regime or a trailer line)."""
    regime, _, other = read_artifact(path)
    for line in [regime, *other]:
        if line.get("stopped"):
            return str(line["stopped"])
    return None


def freeze_calibration(
    tmp_path: Path,
    entropy_bound: float = 0.03,
    confidence_threshold: float = 5e-4,
    *,
    drop: tuple[str, ...] = (),
    **provenance: Any,
) -> Path:
    """A frozen calibration.json written through the harness's own freezer.

    E11: the file names ``task``, ``cut == "dev"`` and ``arm`` (defaults: sudoku / dev / perstep_mar;
    ``perstep_mar_sc`` reuses Mar's values). ``drop`` omits provenance keys, ``provenance`` overrides
    them -- for the refusal rows."""
    path = tmp_path / "calibration.json"
    choice: dict[str, Any] = {
        "entropy_bound": entropy_bound,
        "confidence_threshold": confidence_threshold,
        "task": "sudoku",
        "cut": "dev",
        "arm": "perstep_mar",
        **provenance,
    }
    for key in drop:
        choice.pop(key, None)
    calibration().freeze(choice, path)
    return path


def table_lines(md: str, arm: str) -> list[str]:
    """The markdown table rows (``|``-prefixed) that mention ``arm``."""
    return [line for line in md.splitlines() if line.lstrip().startswith("|") and arm in line]
