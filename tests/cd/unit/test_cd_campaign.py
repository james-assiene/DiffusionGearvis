"""T-0071b criteria 1-4 (umbrella T-0071 criteria 7-9, D-H3): the CD campaign driver ``scripts/gpu/cd_campaign.py``.

Stack check first (``unconstrained``, dev, n = 100, hard stops (d) and (f) armed), then the 18-cell ``perstep_mar``
grid on the dev cut where an (e)-stopped cell is **ineligible** (partial score kept) and the grid continues while
(a), (b), ``internal``, (c), (g) -- and every other stop -- end the launch with nothing frozen; then the freeze
through T-0070's ``calibration.freeze`` (read-only), the stock-knob reference row and the reported-only
``perstep_mar_sc`` rate at the chosen values. The driver refuses ``--cut test`` and never sends a test-cut request.

Regime: CPU, no GPU. The pure rows drive ``run_grid`` with a scripted cell runner (artifacts written in the
runner's own format) and an injected clock; the launch rows run ``main`` end to end against the loopback fake
server of ``tests/cd/fakes/`` (Sudoku dev cut, small ``--limit``). Expected values are computed by hand from the
scripted replies. The script is loaded by file path; importing it must not open a socket or start a run.

Assumed API (test-author proposal for T-0071b, 2026-10-03; to be recorded as a ruling in docs/tasks/T-0071b.md):

``BUDGET_S: Mapping[str, float]``
    The launch budget per task; ``BUDGET_S["bfcl"] == 21600.0`` (D-H3, 6 h).
``run_grid(cells, run_cell, *, task, budget_s, clock=time.monotonic) -> GridResult``
    Pure decision logic. ``cells`` is a sequence of ``{"entropy_bound", "confidence_threshold"}`` points (the driver
    passes ``calibration.grid()``); ``run_cell(point, budget_s) -> runner.RunResult`` runs one cell (``budget_s`` is
    what is left of the launch budget, handed to ``RunConfig.budget_s``). The rows of ``RunResult.artifact`` are read
    with ``runner.read_rows``. No cell starts once ``clock() - start >= budget_s``.
    ``GridResult.cells``: one mapping per cell run, in order, with at least ``entropy_bound``,
    ``confidence_threshold``, ``score`` (BFCL ``exact_call`` / Sudoku ``solved`` over the rows the cell wrote),
    ``n`` (rows written), ``eligible`` (``False`` iff the cell was (e)-stopped) and ``stopped`` (the cell's stop or
    ``None``). ``GridResult.stopped``: ``None`` when the grid ran to the end (with or without (e) cells), else the
    launch-ending reason: the cell's stop string (``"a: ..."``, ``"b: ..."``, ``"c: ..."``, ``"g: ..."``,
    ``"h: ..."``), a string containing ``"internal"`` for a cell with an ``internal`` CD error row, or a string
    starting ``"g"`` when the launch budget ran out.
``choose(grid) -> Mapping[str, Any]``
    ``calibration.select`` over the eligible cells only; ``ValueError`` when the grid was stopped (a partial grid
    selects nothing) or has no eligible cell.
``main(argv) -> int``
    Flags ``--task {bfcl,sudoku} --cut (dev only) --base-url --engine-log --out-dir --repo --expected-commit
    --freeze-path --limit N --budget-s S``. ``--limit`` bounds every grid cell and the ``perstep_mar_sc`` row; the
    stack check always runs the whole dev cut (n = 100). Writes ``<out-dir>/campaign-<task>.json`` in every outcome:
    ``{"task", "cut", "stack_check": {"n", "stopped", "header_prefix_rate", ...}, "cells": [...as above],
    "stopped", "frozen": {"path", "sha256"} | None, "perstep_mar_sc": {"entropy_bound", "confidence_threshold",
    "score", "n"} | None}``. Exit 0 iff the calibration was frozen. The frozen file (written by
    ``calibration.freeze``, so ``load_frozen`` accepts it) carries ``task``, ``cut == "dev"``,
    ``arm == "perstep_mar"``, the chosen ``entropy_bound``/``confidence_threshold``, ``stock_reference``
    (``entropy_bound`` 0.1, ``confidence_threshold`` 0.005, ``mean_steps_per_canvas``, ``mean_accepted_per_step``)
    and ``chosen_stats`` (``mean_steps_per_canvas``, ``mean_accepted_per_step``). Accepted per step comes from the
    CD records' ``mean_accepted_per_step`` (the rows do not carry it).

Rows added after reviews A and B (rulings C4-C13, 2026-10-07; test-author, branch T-0071b-tests-2):

- C4: ``--arm`` accepts exactly ``calibration.GRID_ARMS``; ``perstep_mar`` freezes at ``--freeze-path``
  (``calibration-<task>.json``), ``perstep_mar_qargmax`` beside it as ``calibration-<task>-perstep_mar_qargmax.json``
  (T-0076 A1); each file loads under ``load_frozen(arm=)`` for its own arm only. SC twins are refused: exit 2,
  nothing sent. The driver's *default* arm list stays ``perstep_mar`` (the C1 full-launch row pins it); the locked
  script supplies both arms (C6).
- C5: the report carries, per calibrated arm, its SC twin's run at that arm's chosen knobs under the twin's name
  (``"perstep_mar_sc"`` as in C1, ``"perstep_mar_qargmax_sc"`` alike: ``entropy_bound``, ``confidence_threshold``,
  ``score``, ``n``).
- C9/C12/C13: the engine is probed before the stack check, every cell and every SC-twin run; a failed probe, or
  any ``http_error``/``timeout`` row in any of those runs -- even in a run that also tripped (d)/(e) (C12) --
  ends the launch: ``stopped`` starts with ``"engine"``, nothing is frozen for the arm in progress, exit 1. In
  an SC twin (C13) the base arm's file stays and is listed in the report; later arms do not run. The fake's
  dead engine drops every connection (``FakeVllmServer.die``/``Reply.die_after``), so a driver that does not
  probe sends requests to it (counted in ``posts_while_dead``).
- C7 is withdrawn (C7'): no kernels-record row.
- C6/C11: the locked script is run on the CPU from a throwaway copy with every external command that could
  reach an engine, a container or the GPU stubbed (``run_locked``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.cd import _compile_support as cs
from tests.cd import _eval_support as es
from tests.cd.fakes.vllm_server import FakeVllmServer, Reply, user_prompt

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "gpu" / "cd_campaign.py"
LOCKED = REPO / "scripts" / "locked" / "t0071b_calib.sh"

EB = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
CT = (5e-5, 5e-4, 5e-3)
GRID = [{"entropy_bound": eb, "confidence_threshold": ct} for eb, ct in itertools.product(EB, CT)]
STOCK = (0.1, 5e-3)
STEP_CAP = 48


def campaign() -> ModuleType:
    if not SCRIPT.is_file():
        pytest.fail(f"T-0071b not implemented: {SCRIPT.relative_to(REPO)} does not exist")
    name = "_t0071b_cd_campaign"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    try:
        spec.loader.exec_module(m)
    finally:
        sys.modules.pop(name, None)
    for attr in ("run_grid", "choose", "main", "BUDGET_S"):
        if not hasattr(m, attr):
            pytest.fail(f"T-0071b not implemented: cd_campaign.{attr} is missing")
    return m


def knobs_of(x: Mapping[str, Any]) -> tuple[float, float]:
    return float(x["entropy_bound"]), float(x["confidence_threshold"])


# =========================================================================== pure rows: run_grid / choose


class Cells:
    """A scripted ``run_cell``: ``behave(call_index, knobs, budget_s) -> (rows, stopped, cost_s)``.

    Writes an artifact in the runner's format (regime line, rows, trailer) and returns a ``runner.RunResult``;
    advances the injected clock by ``cost_s``."""

    def __init__(
        self,
        tmp_path: Path,
        task: str,
        behave: Callable[
            [int, tuple[float, float], float], tuple[list[dict[str, Any]], str | None, float]
        ],
    ) -> None:
        self.dir = tmp_path / "cells"
        self.dir.mkdir(exist_ok=True)
        self.task = task
        self.behave = behave
        self.calls: list[tuple[tuple[float, float], float]] = []
        self.now = 0.0

    def clock(self) -> float:
        return self.now

    def __call__(self, point: Mapping[str, Any], budget_s: float) -> Any:
        k = knobs_of(point)
        index = len(self.calls)
        self.calls.append((k, float(budget_s)))
        rows, stopped, cost = self.behave(index, k, float(budget_s))
        self.now += cost
        path = self.dir / f"{index:02d}-{self.task}-dev-perstep_mar.jsonl"
        lines: list[dict[str, Any]] = [
            {
                "kind": "regime",
                "task": self.task,
                "cut": "dev",
                "arm": "perstep_mar",
                "knobs": {
                    "mode": "perstep_mar",
                    "entropy_bound": k[0],
                    "confidence_threshold": k[1],
                },
                "run_id": f"fake{index:02d}",
                "n": len(rows),
            },
            *rows,
            {"kind": "trailer", "stopped": stopped, "header_prefix_rate": None},
        ]
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
        r = es.runner()
        return r.RunResult(3 if stopped else 0, path, stopped, len(rows), {})


def sudoku_rows(
    n: int, solved: int, *, steps: int = 12, error: str | None = None
) -> list[dict[str, Any]]:
    return [
        {
            "id": f"dev:sudoku4_{i}",
            "text": "",
            "committed_ids": [1],
            "parse": True,
            "scores": {"solved": i < solved, "why": ""},
            "cs": True,
            "steps": [steps],
            "wall_ms": 1.0,
            "bucket": 64,
            "error": error,
            "knobs": {},
            "in_final": True,
        }
        for i in range(n)
    ]


def bfcl_rows(n: int, exact: int, *, steps: int = 12) -> list[dict[str, Any]]:
    return [
        {
            "id": f"live_simple_{130 + i}",
            "text": "",
            "committed_ids": [1],
            "parse": True,
            "scores": {
                "schema_valid": True,
                "arg_correct": 1,
                "arg_total": 1,
                "exact_call": i < exact,
            },
            "cs": True,
            "steps": [steps],
            "wall_ms": 1.0,
            "bucket": 64,
            "error": None,
            "knobs": {},
            "in_final": True,
        }
        for i in range(n)
    ]


def grid_cell(grid: Any, k: tuple[float, float]) -> Mapping[str, Any]:
    hits = [c for c in grid.cells if knobs_of(c) == k]
    assert len(hits) == 1, f"cell {k}: {len(hits)} entries in {grid.cells}"
    return hits[0]


def test_bfcl_launch_budget_is_six_hours() -> None:
    """D-H3: BFCL calibration gets a 6 h launch budget (not the runner's 4 h default)."""
    assert float(campaign().BUDGET_S["bfcl"]) == 6 * 3600.0


def test_e_stopped_cell_is_ineligible_keeps_its_partial_score_and_the_grid_continues(
    tmp_path: Path,
) -> None:
    """D-H3. K is (e)-stopped after 20 of 30 rows, all 20 solved: partial score 20/20 = 1.0 (the best score in the
    grid). It must be recorded, ineligible, and not chosen; L (21/30 = 0.7) wins over the rest (3/30 = 0.1)."""
    K, L = (0.01, 5e-4), (0.3, 5e-4)
    e_stop = f"e: 15/20 canvases hit the step cap ({STEP_CAP}) after 20 records"

    def behave(
        i: int, k: tuple[float, float], budget: float
    ) -> tuple[list[dict[str, Any]], str | None, float]:
        if k == K:
            return sudoku_rows(20, 20, steps=STEP_CAP), e_stop, 10.0
        if k == L:
            return sudoku_rows(30, 21), None, 10.0
        return sudoku_rows(30, 3), None, 10.0

    cells = Cells(tmp_path, "sudoku", behave)
    m = campaign()
    grid = m.run_grid(GRID, cells, task="sudoku", budget_s=21600.0, clock=cells.clock)
    assert [k for k, _ in cells.calls] and {k for k, _ in cells.calls} == {
        knobs_of(p) for p in GRID
    }
    assert len(cells.calls) == 18, "the grid continues past an (e)-stopped cell"
    assert grid.stopped is None, grid.stopped
    k_cell = grid_cell(grid, K)
    assert k_cell["eligible"] is False
    assert str(k_cell["stopped"]).startswith("e")
    assert int(k_cell["n"]) == 20 and float(k_cell["score"]) == 1.0, "the partial score is kept"
    assert all(c["eligible"] is True for c in grid.cells if knobs_of(c) != K)
    assert float(grid_cell(grid, L)["score"]) == pytest.approx(0.7)
    assert knobs_of(m.choose(grid)) == L


ABORTS = {
    "a": "a: terminated output of record 3 (dev:sudoku4_3) rejected by replay",
    "b": "b: loud CD error 'z0' on record 0 (dev:sudoku4_0)",
    "c": "c: join_error on record 2 (dev:sudoku4_2), within the first 20",
    "g": "g: wall clock 14401.0 s over the budget of 14400.0 s",
    "h": "h: dirty tree in /repo: scripts/gpu/cd_campaign.py",
}


@pytest.mark.parametrize("kind", [*ABORTS, "internal"])
def test_any_other_stop_ends_the_launch_and_selects_nothing(tmp_path: Path, kind: str) -> None:
    """(a), (b), (c), (g) -- and (h), "every other stop" -- end the launch at the cell that stopped; an ``internal``
    CD error row (the runner does not stop on it) does too. Nothing after that cell runs and ``choose`` refuses."""

    def behave(
        i: int, k: tuple[float, float], budget: float
    ) -> tuple[list[dict[str, Any]], str | None, float]:
        if i == 2:
            if kind == "internal":
                return sudoku_rows(1, 0, error="internal") + sudoku_rows(29, 29), None, 10.0
            return sudoku_rows(4, 4), ABORTS[kind], 10.0
        return sudoku_rows(30, 10), None, 10.0

    cells = Cells(tmp_path, "sudoku", behave)
    m = campaign()
    grid = m.run_grid(GRID, cells, task="sudoku", budget_s=21600.0, clock=cells.clock)
    assert len(cells.calls) == 3, f"the launch must end at the stopped cell, ran {len(cells.calls)}"
    assert grid.stopped, "a launch-ending stop must be reported"
    if kind == "internal":
        assert "internal" in str(grid.stopped)
    else:
        assert str(grid.stopped).startswith(kind)
    with pytest.raises(ValueError):
        m.choose(grid)


def test_launch_budget_stops_between_cells_and_keeps_the_completed_cells(tmp_path: Path) -> None:
    """Budget 2500 s, each cell needs 1000 s. Cells start at t = 0, 1000, 2000; the third is handed what is left
    (<= 500 s), runs out mid-cell as the runner would (a (g) stop with 7 partial rows), and no fourth cell
    starts. The two completed cells keep their scores (10/30); a partial grid selects nothing."""
    budget = 2500.0

    def behave(
        i: int, k: tuple[float, float], left: float
    ) -> tuple[list[dict[str, Any]], str | None, float]:
        if left < 1000.0:
            return (
                sudoku_rows(7, 7),
                f"g: wall clock {left + 1:.1f} s over the budget of {left:.1f} s",
                left + 1,
            )
        return sudoku_rows(30, 10), None, 1000.0

    cells = Cells(tmp_path, "sudoku", behave)
    m = campaign()
    grid = m.run_grid(GRID, cells, task="sudoku", budget_s=budget, clock=cells.clock)
    assert len(cells.calls) == 3, f"ran {len(cells.calls)} cells"
    for idx, (_, handed) in enumerate(cells.calls):
        assert handed <= budget - 1000.0 * idx + 1e-6, (
            f"cell {idx} handed {handed} s, more than what is left"
        )
    assert str(grid.stopped).startswith("g"), grid.stopped
    done = [c for c in grid.cells if c["stopped"] is None]
    assert len(done) == 2
    assert all(int(c["n"]) == 30 and float(c["score"]) == pytest.approx(1 / 3) for c in done)
    with pytest.raises(ValueError):
        m.choose(grid)


def test_bfcl_cells_are_scored_on_exact_call(tmp_path: Path) -> None:
    """BFCL's calibration score is exact call (T-0070 criterion 10), not schema valid: every row here is schema-valid,
    only X has exact calls (15/30). X has the most steps and the smallest knobs, so no tie-break picks it."""
    X = (0.003, 5e-5)

    def behave(
        i: int, k: tuple[float, float], budget: float
    ) -> tuple[list[dict[str, Any]], str | None, float]:
        if k == X:
            return bfcl_rows(30, 15, steps=40), None, 10.0
        return bfcl_rows(30, 0, steps=10), None, 10.0

    cells = Cells(tmp_path, "bfcl", behave)
    m = campaign()
    grid = m.run_grid(GRID, cells, task="bfcl", budget_s=21600.0, clock=cells.clock)
    assert grid.stopped is None
    assert float(grid_cell(grid, X)["score"]) == pytest.approx(0.5)
    assert knobs_of(m.choose(grid)) == X


# =========================================================================== launch rows: main + fake server

DEV = cs.sudoku_cut("dev")
DEV_IDS = [rid for rid, _, _ in DEV]


def dev_prompts() -> dict[str, str]:
    return es.prompt_to_id("sudoku", "dev")


def sudoku_reply(rid: str, kind: str, **kw: Any) -> Reply:
    puzzle, solution = es.sudoku_by_id("dev")[rid]
    if kind == "solved":
        text = es.solved_text(solution)
    elif kind == "wrong":
        text = es.wrong_text(puzzle, solution)
    elif kind == "solved_no_header":
        text = es.grid_text(solution)
    else:
        raise AssertionError(kind)
    return Reply(text=text, **kw)


Decide = Callable[[str, tuple[float, float]], Reply]


def script(
    unconstrained: Callable[[str], Reply], mar: Decide, sc: Decide | None = None
) -> Callable[[int, Mapping[str, Any]], Reply]:
    ids = dev_prompts()

    def s(index: int, body: Mapping[str, Any]) -> Reply:
        prompt = user_prompt(body)
        if (
            prompt not in ids
        ):  # a test-cut (or unknown) prompt: answered, and caught by assert_dev_only
            return Reply(text="")
        rid = ids[prompt]
        x = body.get("vllm_xargs") or {}
        if "astra_cd_mode" not in x:
            return unconstrained(rid)
        k = (float(x["astra_cd_entropy_bound"]), float(x["astra_cd_confidence_threshold"]))
        if x.get("astra_cd_self_conditioning") == "constrained":
            assert sc is not None, "unexpected perstep_mar_sc request"
            return sc(rid, k)
        return mar(rid, k)

    return s


def assert_dev_only(srv: FakeVllmServer) -> None:
    dev = set(dev_prompts())
    test = set(es.prompt_to_id("sudoku", "test"))
    prompts = [user_prompt(b) for b in srv.requests]
    assert not set(prompts) & test, "a test-cut request was sent"
    assert set(prompts) <= dev, "a request outside the dev cut was sent"


def mar_requests(srv: FakeVllmServer) -> list[tuple[float, float]]:
    out = []
    for b in srv.requests:
        x = b.get("vllm_xargs") or {}
        if (
            x.get("astra_cd_mode") == "perstep_mar"
            and x.get("astra_cd_self_conditioning") != "constrained"
        ):
            out.append(
                (float(x["astra_cd_entropy_bound"]), float(x["astra_cd_confidence_threshold"]))
            )
    return out


def sc_requests(srv: FakeVllmServer) -> list[tuple[float, float]]:
    out = []
    for b in srv.requests:
        x = b.get("vllm_xargs") or {}
        if x.get("astra_cd_self_conditioning") == "constrained":
            out.append(
                (float(x["astra_cd_entropy_bound"]), float(x["astra_cd_confidence_threshold"]))
            )
    return out


def other_cd_requests(srv: FakeVllmServer) -> list[str]:
    return [
        str(x["astra_cd_mode"])
        for b in srv.requests
        if (x := b.get("vllm_xargs") or {}).get("astra_cd_mode") not in (None, "perstep_mar")
    ]


def launch(
    tmp_path: Path,
    srv: FakeVllmServer,
    *,
    cut: str = "dev",
    limit: int = 4,
    budget_s: float | None = None,
) -> tuple[int, Path, Path]:
    repo, head = es.clean_repo(tmp_path)
    out = tmp_path / "evidence"
    freeze = tmp_path / "frozen" / "calibration-sudoku.json"
    argv = [
        "--task", "sudoku",
        "--cut", cut,
        "--base-url", srv.url,
        "--engine-log", str(srv.engine_log),
        "--out-dir", str(out),
        "--repo", str(repo),
        "--expected-commit", head,
        "--freeze-path", str(freeze),
        "--limit", str(limit),
    ]  # fmt: skip
    if budget_s is not None:
        argv += ["--budget-s", str(budget_s)]
    try:
        rc = campaign().main(argv)
    except SystemExit as exc:  # an argparse refusal
        rc = exc.code if isinstance(exc.code, int) else 1
    return int(rc), out, freeze


def report_of(out: Path) -> dict[str, Any]:
    path = out / "campaign-sudoku.json"
    assert path.is_file(), f"no campaign report at {path}"
    data: dict[str, Any] = json.loads(path.read_text())
    return data


@pytest.fixture(scope="module", autouse=True)
def _warm() -> None:
    cs.warm_sudoku("dev")


def test_driver_refuses_cut_test_and_sends_nothing(tmp_path: Path) -> None:
    """Criterion 4: ``--cut test`` is refused before any request (and before any flush); nothing is frozen."""
    sudoku = script(
        lambda rid: Reply(text=""), lambda rid, k: Reply(text=""), lambda rid, k: Reply(text="")
    )
    with es.fake(sudoku, tmp_path) as srv:
        rc, _out, freeze = launch(tmp_path, srv, cut="test")
        sent, flushed = list(srv.requests), list(srv.flushes)
    assert rc != 0
    assert sent == [] and flushed == []
    assert not freeze.exists()


def test_stack_check_d_ends_the_launch_before_the_grid(tmp_path: Path) -> None:
    """Criterion 1: the unconstrained stack check runs first; every answer is wrong, so (d) fires at record 30
    (0/30 < 0.50). The launch ends there: 30 unconstrained requests, no constrained request, nothing frozen."""
    sudoku = script(
        lambda rid: sudoku_reply(rid, "wrong"),
        lambda rid, k: sudoku_reply(rid, "solved"),
        lambda rid, k: sudoku_reply(rid, "solved"),
    )
    with es.fake(sudoku, tmp_path) as srv:
        rc, out, freeze = launch(tmp_path, srv)
        bodies = list(srv.requests)
        cd = srv.cd_requests()
        assert_dev_only(srv)
    assert rc != 0
    assert cd == [], f"{len(cd)} constrained requests after (d)"
    assert len(bodies) == 30
    assert not freeze.exists()
    rep = report_of(out)
    assert str(rep["stopped"]).startswith("d"), rep["stopped"]
    assert str(rep["stack_check"]["stopped"]).startswith("d")
    assert rep["cells"] == []


CHOSEN = (0.03, 5e-4)
SC_ONLY = (1.0, 5e-5)


def test_full_launch_freezes_the_dev_choice_with_provenance_stock_row_and_sc_rate(
    tmp_path: Path,
) -> None:
    """The whole launch, hand-computed expectations (limit 4 per cell):

    - stack check: all 100 dev records, first; 4 of the first 20 (dev 0, 2, 4, 6) do not open with the header, so the
      (f) header-prefix rate is 16/20 = 0.8;
    - grid: Mar solves only at CHOSEN (4/4); CHOSEN answers in 2 canvases of 7 steps each with 4.0 accepted per step,
      stock (0.1, 5e-3) in 1 canvas of 20 steps with 2.5 accepted per step, the rest 30 steps and 1.0;
    - ``perstep_mar_sc`` at CHOSEN solves dev 0 only (1/4 = 0.25); at SC_ONLY it would solve everything -- a driver
      that selected on SC would freeze SC_ONLY (the SC rate is reported, never used for selection).
    """
    first20 = DEV_IDS[:20]
    no_header = {first20[i] for i in (0, 2, 4, 6)}

    def unconstrained(rid: str) -> Reply:
        return sudoku_reply(rid, "solved_no_header" if rid in no_header else "solved")

    def mar(rid: str, k: tuple[float, float]) -> Reply:
        if k == CHOSEN:
            return sudoku_reply(rid, "solved", steps=7, canvases=2, mean_accepted_per_step=4.0)
        if k == STOCK:
            return sudoku_reply(rid, "wrong", steps=20, mean_accepted_per_step=2.5)
        return sudoku_reply(rid, "wrong", steps=30, mean_accepted_per_step=1.0)

    def sc(rid: str, k: tuple[float, float]) -> Reply:
        if k == SC_ONLY:
            return sudoku_reply(rid, "solved")
        return sudoku_reply(rid, "solved" if rid == DEV_IDS[0] else "wrong")

    with es.fake(script(unconstrained, mar, sc), tmp_path) as srv:
        rc, out, freeze = launch(tmp_path, srv, limit=4)
        bodies = list(srv.requests)
        mars, scs, others = mar_requests(srv), sc_requests(srv), other_cd_requests(srv)
        assert_dev_only(srv)

    # criterion 1: stack check first, the whole dev cut
    stack = bodies[:100]
    assert all("astra_cd_mode" not in (b.get("vllm_xargs") or {}) for b in stack), (
        "stack check is not first"
    )
    assert {user_prompt(b) for b in stack} == set(dev_prompts())
    rep = report_of(out)
    assert rep["stack_check"]["header_prefix_rate"] == pytest.approx(0.8)

    # criterion 2: the 18 cells on the dev cut, perstep_mar only
    assert set(mars) == {knobs_of(p) for p in GRID}
    assert all(mars.count(knobs_of(p)) >= 4 for p in GRID)
    assert others == []
    assert rc == 0, rep.get("stopped")
    assert rep["stopped"] is None
    assert len(rep["cells"]) == 18 and all(c["eligible"] is True for c in rep["cells"])

    # criterion 3: the frozen file
    assert freeze.is_file()
    raw = freeze.read_bytes()
    assert rep["frozen"]["sha256"] == hashlib.sha256(raw).hexdigest(), (
        "the frozen file's sha256 is recorded"
    )
    knobs, sha = es.calibration().load_frozen(freeze, task="sudoku")
    assert sha == hashlib.sha256(raw).hexdigest()
    assert knobs_of(knobs) == CHOSEN, (
        "selection must follow perstep_mar's score, not perstep_mar_sc's"
    )
    data = json.loads(raw)
    assert (data["task"], data["cut"], data["arm"]) == ("sudoku", "dev", "perstep_mar")
    stock = data["stock_reference"]
    assert knobs_of(stock) == STOCK
    assert float(stock["mean_steps_per_canvas"]) == pytest.approx(20.0)
    assert float(stock["mean_accepted_per_step"]) == pytest.approx(2.5)
    chosen = data["chosen_stats"]
    assert float(chosen["mean_steps_per_canvas"]) == pytest.approx(7.0), (
        "steps per canvas, not per record (14)"
    )
    assert float(chosen["mean_accepted_per_step"]) == pytest.approx(4.0)

    # perstep_mar_sc: reported at the chosen values only, never a selection input
    assert scs and set(scs) == {CHOSEN}, f"perstep_mar_sc ran at {sorted(set(scs))}"
    sc_rep = rep["perstep_mar_sc"]
    assert knobs_of(sc_rep) == CHOSEN
    assert int(sc_rep["n"]) == 4 and float(sc_rep["score"]) == pytest.approx(0.25)


ABORT_REPLIES: dict[str, Callable[[str], Reply]] = {
    # replay accepts the (wrong but well-formed) grid, the plugin says not in final: disagreement
    "a": lambda rid: sudoku_reply(rid, "wrong", in_final=False),
    "b": lambda rid: Reply(text="", error="z0"),
    "internal": lambda rid: Reply(text="", error="internal"),
    # a duplicated canvas_idx: the join is invalid -> join_error within the first 20
    "c": lambda rid: sudoku_reply(rid, "wrong", duplicate_last_canvas=True),
}


@pytest.mark.parametrize("kind", list(ABORT_REPLIES))
def test_launch_aborts_on_a_b_internal_c_and_freezes_nothing(tmp_path: Path, kind: str) -> None:
    """The third distinct cell the driver starts misbehaves on every record. The launch ends with that cell: exactly
    three cells requested, no ``perstep_mar_sc`` row, no frozen file, a non-zero exit, the stop in the report."""
    seen: list[tuple[float, float]] = []

    def mar(rid: str, k: tuple[float, float]) -> Reply:
        if k not in seen:
            seen.append(k)
        if len(seen) >= 3 and k == seen[2]:
            return ABORT_REPLIES[kind](rid)
        return sudoku_reply(rid, "wrong")

    sudoku = script(
        lambda rid: sudoku_reply(rid, "solved"), mar, lambda rid, k: sudoku_reply(rid, "wrong")
    )
    with es.fake(sudoku, tmp_path) as srv:
        rc, out, freeze = launch(tmp_path, srv, limit=4)
        mars, scs = mar_requests(srv), sc_requests(srv)
        assert_dev_only(srv)
    assert len(seen) >= 3, "the grid never reached its third cell"
    assert len(set(mars)) == 3, (
        f"cells started after the stop: {len(set(mars))} distinct cells requested"
    )
    assert mars[-1] == seen[2]
    assert scs == []
    assert rc != 0
    assert not freeze.exists()
    rep = report_of(out)
    if kind == "internal":
        assert "internal" in str(rep["stopped"])
    else:
        assert str(rep["stopped"]).startswith(kind), rep["stopped"]
    assert rep["frozen"] is None


def test_launch_e_stopped_cell_is_ineligible_and_the_grid_continues(tmp_path: Path) -> None:
    """End to end, limit 24: K hits the step cap on every canvas while solving every puzzle, so (e) stops it after 20
    records with a partial 20/20 = 1.0; L solves dev 0-17 (18/24 = 0.75); the rest solve nothing. All 18 cells run,
    K is ineligible with its partial score, and L is frozen."""
    K, L = (0.01, 5e-4), (0.3, 5e-4)
    l_solves = set(DEV_IDS[:18])

    def mar(rid: str, k: tuple[float, float]) -> Reply:
        if k == K:
            return sudoku_reply(rid, "solved", steps=STEP_CAP)
        if k == L:
            return sudoku_reply(rid, "solved" if rid in l_solves else "wrong")
        return sudoku_reply(rid, "wrong")

    sudoku = script(
        lambda rid: sudoku_reply(rid, "solved"), mar, lambda rid, k: sudoku_reply(rid, "wrong")
    )
    with es.fake(sudoku, tmp_path) as srv:
        rc, out, freeze = launch(tmp_path, srv, limit=24)
        mars = mar_requests(srv)
        assert_dev_only(srv)
    assert set(mars) == {knobs_of(p) for p in GRID}, "the grid continues past an (e)-stopped cell"
    assert mars.count(K) == 20
    rep = report_of(out)
    assert rc == 0, rep.get("stopped")
    k_cell = [c for c in rep["cells"] if knobs_of(c) == K]
    assert len(k_cell) == 1
    assert k_cell[0]["eligible"] is False and str(k_cell[0]["stopped"]).startswith("e")
    assert int(k_cell[0]["n"]) == 20 and float(k_cell[0]["score"]) == 1.0
    knobs, _ = es.calibration().load_frozen(freeze, task="sudoku")
    assert knobs_of(knobs) == L


def test_launch_budget_stops_cleanly_with_the_report_and_partial_results(tmp_path: Path) -> None:
    """Budget 8 s; every constrained request takes 0.25 s, so a 4-record cell takes >= 1 s and the 18 cells could
    not fit. The driver stops itself (not killed): non-zero exit, nothing frozen, the report written with the
    completed cells' scores, their artifacts kept, and it returns within a few seconds of the budget."""
    budget = 8.0

    def mar(rid: str, k: tuple[float, float]) -> Reply:
        return sudoku_reply(rid, "wrong", hang_s=0.25)

    sudoku = script(
        lambda rid: sudoku_reply(rid, "solved"), mar, lambda rid, k: sudoku_reply(rid, "wrong")
    )
    t0 = time.monotonic()
    with es.fake(sudoku, tmp_path) as srv:
        rc, out, freeze = launch(tmp_path, srv, limit=4, budget_s=budget)
        mars = mar_requests(srv)
        assert_dev_only(srv)
    elapsed = time.monotonic() - t0
    assert rc != 0
    assert not freeze.exists()
    assert len(set(mars)) < 18
    rep = report_of(out)
    assert str(rep["stopped"]).startswith("g"), rep["stopped"]
    done = [c for c in rep["cells"] if c["stopped"] is None]
    assert done, "the completed cells are kept"
    assert all(int(c["n"]) == 4 and float(c["score"]) == 0.0 for c in done)
    artifacts = list(out.rglob("*-sudoku-dev-perstep_mar.jsonl"))
    assert len(artifacts) >= len(done)
    assert elapsed < budget + 4.0, f"returned {elapsed:.1f} s after start (budget {budget} s)"


# =========================================================================== review A/B rows (C4-C13)

QA = "perstep_mar_qargmax"
CM = (0.03, 5e-4)  # perstep_mar's winner in the two-arm rows
CQ = (0.3, 5e-3)  # perstep_mar_qargmax's winner: a different cell, so a shared freeze is visible
FREEZE_M = "calibration-sudoku.json"
FREEZE_Q = f"calibration-sudoku-{QA}.json"

Decide4 = Callable[[str, str, bool, tuple[float, float]], Reply]


def script_arms(
    unconstrained: Callable[[str], Reply], decide: Decide4
) -> Callable[[int, Mapping[str, Any]], Reply]:
    """Like ``script`` but the decision sees the wire mode and whether self-conditioning is constrained."""
    ids = dev_prompts()

    def s(index: int, body: Mapping[str, Any]) -> Reply:
        prompt = user_prompt(body)
        if prompt not in ids:
            return Reply(text="")
        rid = ids[prompt]
        x = body.get("vllm_xargs") or {}
        if "astra_cd_mode" not in x:
            return unconstrained(rid)
        k = (float(x["astra_cd_entropy_bound"]), float(x["astra_cd_confidence_threshold"]))
        sc = x.get("astra_cd_self_conditioning") == "constrained"
        return decide(rid, str(x["astra_cd_mode"]), sc, k)

    return s


def cd_calls(srv: FakeVllmServer) -> list[tuple[str, bool, tuple[float, float]]]:
    out = []
    for b in srv.requests:
        x = b.get("vllm_xargs") or {}
        if "astra_cd_mode" in x:
            k = (float(x["astra_cd_entropy_bound"]), float(x["astra_cd_confidence_threshold"]))
            out.append(
                (str(x["astra_cd_mode"]), x.get("astra_cd_self_conditioning") == "constrained", k)
            )
    return out


def grid_calls(srv: FakeVllmServer, mode: str) -> list[tuple[float, float]]:
    return [k for m, sc, k in cd_calls(srv) if m == mode and not sc]


def launch_arms(
    tmp_path: Path,
    srv: FakeVllmServer,
    arms: list[str],
    *,
    limit: int = 4,
    extra: tuple[str, ...] = (),
) -> tuple[int, Path, Path]:
    repo, head = es.clean_repo(tmp_path)
    out = tmp_path / "evidence"
    freeze = tmp_path / "frozen" / FREEZE_M
    argv = [
        "--task", "sudoku",
        "--cut", "dev",
        "--base-url", srv.url,
        "--engine-log", str(srv.engine_log),
        "--out-dir", str(out),
        "--repo", str(repo),
        "--expected-commit", head,
        "--freeze-path", str(freeze),
        "--limit", str(limit),
        *extra,
    ]  # fmt: skip
    for a in arms:
        argv += ["--arm", a]
    try:
        rc = campaign().main(argv)
    except SystemExit as exc:  # an argparse refusal
        rc = exc.code if isinstance(exc.code, int) else 1
    return int(rc), out, freeze


def frozen_names(freeze: Path) -> set[str]:
    d = freeze.parent
    return {p.name for p in d.iterdir()} if d.is_dir() else set()


def two_arm_decide(rid: str, mode: str, sc: bool, k: tuple[float, float]) -> Reply:
    if sc:  # the twins: perstep_mar_sc solves dev 0 only, perstep_mar_qargmax_sc dev 0 and 1
        solves = DEV_IDS[:1] if mode == "perstep_mar" else DEV_IDS[:2]
        return sudoku_reply(rid, "solved" if rid in solves else "wrong")
    win = CM if mode == "perstep_mar" else CQ
    return sudoku_reply(rid, "solved" if k == win else "wrong")


def solved_unconstrained(rid: str) -> Reply:
    return sudoku_reply(rid, "solved")


def test_c4_two_arms_freeze_two_files_each_loadable_only_for_its_own_arm(tmp_path: Path) -> None:
    """C4 / C8 / T-0076 A1-A2. Both grid arms in one launch (limit 4): Mar wins at CM, qargmax at CQ. Exactly two
    files are frozen, ``calibration-sudoku.json`` (arm ``perstep_mar``, CM) and
    ``calibration-sudoku-perstep_mar_qargmax.json`` (arm ``perstep_mar_qargmax``, CQ); ``load_frozen(arm=)``
    accepts each for its own arm (and its SC twin, which reuses it) and refuses it for the other grid arm.
    Kills a freeze path that ignores the arm (one file, overwritten) and a frozen ``arm`` field that ignores it."""
    with es.fake(script_arms(solved_unconstrained, two_arm_decide), tmp_path) as srv:
        rc, out, freeze = launch_arms(tmp_path, srv, ["perstep_mar", QA])
        modes = {m for m, _sc, _k in cd_calls(srv)}
        assert_dev_only(srv)
    rep = report_of(out)
    assert rc == 0, rep.get("stopped")
    assert modes == {"perstep_mar", QA}, modes
    assert frozen_names(freeze) == {FREEZE_M, FREEZE_Q}
    cal = es.calibration()
    pm, pq = freeze.parent / FREEZE_M, freeze.parent / FREEZE_Q
    knobs, sha_m = cal.load_frozen(pm, task="sudoku", arm="perstep_mar")
    assert knobs_of(knobs) == CM
    knobs, sha_q = cal.load_frozen(pq, task="sudoku", arm=QA)
    assert knobs_of(knobs) == CQ
    assert json.loads(pm.read_bytes())["arm"] == "perstep_mar"
    assert json.loads(pq.read_bytes())["arm"] == QA
    with pytest.raises(ValueError):
        cal.load_frozen(pm, task="sudoku", arm=QA)
    with pytest.raises(ValueError):
        cal.load_frozen(pq, task="sudoku", arm="perstep_mar")
    cal.load_frozen(pm, task="sudoku", arm="perstep_mar_sc")  # the twin runs on its base arm's file
    cal.load_frozen(pq, task="sudoku", arm=f"{QA}_sc")
    dump = json.dumps(rep)
    assert sha_m in dump and sha_q in dump, (
        "each frozen file's sha256 is in the report (H2 compares)"
    )


def test_c4_qargmax_alone_is_accepted_and_freezes_only_its_own_file(tmp_path: Path) -> None:
    """C4, the other side of the guard: ``--arm perstep_mar_qargmax`` alone is a grid arm, accepted; it freezes
    ``calibration-sudoku-perstep_mar_qargmax.json`` and nothing at the ``perstep_mar`` path."""
    with es.fake(script_arms(solved_unconstrained, two_arm_decide), tmp_path) as srv:
        rc, out, freeze = launch_arms(tmp_path, srv, [QA])
        modes = {m for m, _sc, _k in cd_calls(srv)}
    rep = report_of(out)
    assert rc == 0, rep.get("stopped")
    assert modes == {QA}
    assert frozen_names(freeze) == {FREEZE_Q}
    knobs, _ = es.calibration().load_frozen(freeze.parent / FREEZE_Q, task="sudoku", arm=QA)
    assert knobs_of(knobs) == CQ


@pytest.mark.parametrize("twin", ["perstep_mar_sc", f"{QA}_sc"])
def test_c4_sc_twins_are_refused_before_any_request(tmp_path: Path, twin: str) -> None:
    """C4 / T-0076 A2: an SC twin is never a grid arm. ``--arm <twin>`` exits 2 with no request of any kind sent
    (no POST, no flush, no GET) and nothing frozen."""
    with es.fake(script_arms(solved_unconstrained, two_arm_decide), tmp_path) as srv:
        rc, _out, freeze = launch_arms(tmp_path, srv, [twin])
        sent, flushed, gets = list(srv.requests), list(srv.flushes), list(srv.gets)
    assert rc == 2
    assert sent == [] and flushed == [] and gets == [], (len(sent), len(flushed), gets)
    assert frozen_names(freeze) == set()


def test_c5_each_arms_sc_twin_is_reported_at_that_arms_choice(tmp_path: Path) -> None:
    """C5 / T-0076 Q3. Two arms, limit 4. The twins run only at their base arm's chosen knobs -- ``perstep_mar_sc``
    (mode ``perstep_mar``, SC constrained) at CM, ``perstep_mar_qargmax_sc`` (mode ``perstep_mar_qargmax``, SC
    constrained) at CQ -- and the report gives each twin's score under the twin's name: 1/4 = 0.25 and
    2/4 = 0.5 by construction."""
    with es.fake(script_arms(solved_unconstrained, two_arm_decide), tmp_path) as srv:
        rc, out, _freeze = launch_arms(tmp_path, srv, ["perstep_mar", QA])
        twins = [(m, k) for m, sc, k in cd_calls(srv) if sc]
    rep = report_of(out)
    assert rc == 0, rep.get("stopped")
    assert set(twins) == {("perstep_mar", CM), (QA, CQ)}, sorted(set(twins))
    assert twins.count(("perstep_mar", CM)) == 4 and twins.count((QA, CQ)) == 4
    m_sc, q_sc = rep["perstep_mar_sc"], rep.get(f"{QA}_sc")
    assert q_sc is not None, f"no {QA}_sc entry in the report"
    assert knobs_of(m_sc) == CM and int(m_sc["n"]) == 4
    assert float(m_sc["score"]) == pytest.approx(0.25)
    assert knobs_of(q_sc) == CQ and int(q_sc["n"]) == 4
    assert float(q_sc["score"]) == pytest.approx(0.5)


def test_tie_break_through_run_grid_and_choose(tmp_path: Path) -> None:
    """T-0070's rule through the driver: max score, then fewer mean steps, then the larger entropy bound.
    Three cells tie on 15/30: P1 (eb 1.0, 20 steps), P2 (eb 0.01, 10 steps), P3 (eb 0.03, 10 steps); every other
    cell scores 3/30 in 5 steps. The winner is P3. A score-only choice takes the first maximum in grid order (P2);
    ignoring steps takes P1; a smaller-eb tie-break takes P2; steps before score takes a 5-step cell."""
    P1, P2, P3 = (1.0, 5e-5), (0.01, 5e-4), (0.03, 5e-5)
    steps = {P1: 20, P2: 10, P3: 10}

    def behave(
        i: int, k: tuple[float, float], budget: float
    ) -> tuple[list[dict[str, Any]], str | None, float]:
        if k in steps:
            return sudoku_rows(30, 15, steps=steps[k]), None, 10.0
        return sudoku_rows(30, 3, steps=5), None, 10.0

    cells = Cells(tmp_path, "sudoku", behave)
    m = campaign()
    grid = m.run_grid(GRID, cells, task="sudoku", budget_s=21600.0, clock=cells.clock)
    assert grid.stopped is None
    assert knobs_of(m.choose(grid)) == P3


def test_second_arm_stopping_after_the_first_froze_exits_1_with_only_the_first_file(
    tmp_path: Path,
) -> None:
    """Two arms; ``perstep_mar`` freezes (CM); the third qargmax cell answers a loud CD error on every record, so
    (b) ends the launch. Exit 1 (not every requested arm frozen), the report names the stopped arm, the (b) stop
    is kept, only ``calibration-sudoku.json`` exists and no qargmax twin ran."""
    seen: list[tuple[float, float]] = []

    def decide(rid: str, mode: str, sc: bool, k: tuple[float, float]) -> Reply:
        if mode == QA and not sc:
            if k not in seen:
                seen.append(k)
            if len(seen) >= 3 and k == seen[2]:
                return Reply(text="", error="z0")
        return two_arm_decide(rid, mode, sc, k)

    with es.fake(script_arms(solved_unconstrained, decide), tmp_path) as srv:
        rc, out, freeze = launch_arms(tmp_path, srv, ["perstep_mar", QA])
        qa_cells = set(grid_calls(srv, QA))
        qa_twin = [k for m, sc, k in cd_calls(srv) if m == QA and sc]
    rep = report_of(out)
    assert rc == 1
    assert len(qa_cells) == 3, f"{len(qa_cells)} qargmax cells requested"
    assert qa_twin == []
    assert frozen_names(freeze) == {FREEZE_M}
    knobs, _ = es.calibration().load_frozen(freeze, task="sudoku", arm="perstep_mar")
    assert knobs_of(knobs) == CM
    assert str(rep["stopped"]).startswith("b"), rep["stopped"]
    assert QA in json.dumps({k: rep.get(k) for k in ("stopped", "stopped_arm")}), (
        "the report must name the arm that stopped"
    )


# --------------------------------------------------------------------------- C9 / C12 / C13: engine health


def _assert_engine_stop(rc: int, out: Path) -> dict[str, Any]:
    rep = report_of(out)
    assert rc == 1, (rc, rep.get("stopped"))
    assert str(rep["stopped"]).startswith("engine"), rep["stopped"]
    return rep


C9_CELL = {
    "cell_all_http_error": "every record of the third cell is an HTTP 500",
    "cell_one_http_error": "only the first record of the third cell is an HTTP 500",
    "cell_one_timeout": "only the first record of the third cell outlives --request-timeout-s",
}


@pytest.mark.parametrize("case", list(C9_CELL))
def test_c9_an_http_error_or_timeout_row_in_a_cell_ends_the_launch(
    tmp_path: Path, case: str
) -> None:
    """C9 (review B's HIGH defect). Rows of a dead or failing engine are not wrong answers: a cell holding any
    ``http_error`` or ``timeout`` row ends the launch at that cell -- ``stopped`` starts with ``"engine"``, no
    later cell and no twin runs, nothing is frozen, exit 1. Without the rule the cell scores 0, stays eligible
    and the launch freezes and exits 0 (every other cell runs, the twin runs)."""
    seen: list[tuple[float, float]] = []
    hit: list[str] = []

    def mar(rid: str, k: tuple[float, float]) -> Reply:
        if k not in seen:
            seen.append(k)
        if len(seen) >= 3 and k == seen[2]:
            if case == "cell_all_http_error":
                return sudoku_reply(rid, "wrong", http_status=500)
            if not hit:
                hit.append(rid)
                if case == "cell_one_http_error":
                    return sudoku_reply(rid, "wrong", http_status=500)
                return sudoku_reply(rid, "wrong", hang_s=1.5)
        return sudoku_reply(rid, "solved" if k == CM else "wrong")

    extra = ("--request-timeout-s", "0.5") if case == "cell_one_timeout" else ()
    sudoku = script(solved_unconstrained, mar, lambda rid, k: sudoku_reply(rid, "wrong"))
    with es.fake(sudoku, tmp_path) as srv:
        rc, out, freeze = launch_arms(tmp_path, srv, ["perstep_mar"], extra=extra)
        mars, scs = mar_requests(srv), sc_requests(srv)
    assert len(seen) >= 3, "the grid never reached its third cell"
    assert len(set(mars)) == 3, (
        f"{len(set(mars))} distinct cells requested; the launch must end at the third"
    )
    assert scs == []
    assert frozen_names(freeze) == set()
    rep = _assert_engine_stop(rc, out)
    assert rep["frozen"] is None


def test_c12_an_http_error_row_in_an_e_stopped_cell_is_an_engine_stop(tmp_path: Path) -> None:
    """C12: engine stops take precedence. The third cell (limit 24) answers record 0 with an HTTP 500 and hits the
    step cap on every other canvas, so the runner's (e) also stops it after 20 records. Read as (e) the cell is
    merely ineligible and the grid goes on; read as C12 rules, the launch ends with an engine stop."""
    seen: list[tuple[float, float]] = []
    hit: list[str] = []

    def mar(rid: str, k: tuple[float, float]) -> Reply:
        if k not in seen:
            seen.append(k)
        if len(seen) >= 3 and k == seen[2]:
            if not hit:
                hit.append(rid)
                return sudoku_reply(rid, "wrong", http_status=500)
            return sudoku_reply(rid, "solved", steps=STEP_CAP)
        return sudoku_reply(rid, "solved" if k == CM else "wrong")

    sudoku = script(solved_unconstrained, mar, lambda rid, k: sudoku_reply(rid, "wrong"))
    with es.fake(sudoku, tmp_path) as srv:
        rc, out, freeze = launch_arms(tmp_path, srv, ["perstep_mar"], limit=24)
        mars = mar_requests(srv)
    assert len(seen) >= 3
    assert mars.count(seen[2]) == 20, "premise: (e) stops the third cell after 20 records"
    assert len(set(mars)) == 3, f"{len(set(mars))} distinct cells requested"
    assert frozen_names(freeze) == set()
    _assert_engine_stop(rc, out)


@pytest.mark.parametrize("case", ["all", "one"])
def test_c9_an_http_error_row_in_the_stack_check_ends_the_launch(tmp_path: Path, case: str) -> None:
    """C9 / C12 in the stack check. ``all``: every unconstrained request is an HTTP 500 -- the stop is the
    engine's, not (d)'s (a dead engine is not a model that cannot solve Sudoku). ``one``: dev 5 alone is an HTTP
    500 and the other 99 are solved, so no hard stop (a)-(h) fires; the launch still ends before any constrained
    request."""
    bad = set(DEV_IDS) if case == "all" else {DEV_IDS[5]}

    def unconstrained(rid: str) -> Reply:
        return sudoku_reply(rid, "solved", http_status=500 if rid in bad else 200)

    sudoku = script(unconstrained, lambda rid, k: sudoku_reply(rid, "solved"))
    with es.fake(sudoku, tmp_path) as srv:
        rc, out, freeze = launch_arms(tmp_path, srv, ["perstep_mar"])
        cd = srv.cd_requests()
    assert cd == [], f"{len(cd)} constrained requests after an engine failure in the stack check"
    assert frozen_names(freeze) == set()
    _assert_engine_stop(rc, out)


def test_c13_an_http_error_row_in_an_sc_twin_ends_the_launch_and_keeps_the_base_file(
    tmp_path: Path,
) -> None:
    """C9 / C13 in an SC twin run. Two arms; ``perstep_mar`` freezes (CM) from a healthy grid, then its twin's
    requests are all HTTP 500. The launch ends with an engine stop and exit 1 -- not the WARNING-and-go-on of a
    stopped twin -- the base arm's file stays and is listed in the report (its sha256), and the later arm does
    not run (no qargmax request, no qargmax file)."""

    def decide(rid: str, mode: str, sc: bool, k: tuple[float, float]) -> Reply:
        if sc and mode == "perstep_mar":
            return sudoku_reply(rid, "solved", http_status=500)
        return two_arm_decide(rid, mode, sc, k)

    with es.fake(script_arms(solved_unconstrained, decide), tmp_path) as srv:
        rc, out, freeze = launch_arms(tmp_path, srv, ["perstep_mar", QA])
        twins = [k for m, sc, k in cd_calls(srv) if sc and m == "perstep_mar"]
        qa = [k for m, _sc, k in cd_calls(srv) if m == QA]
    assert twins, "the twin never ran"
    assert qa == [], f"{len(qa)} qargmax requests after the engine stop"
    rep = _assert_engine_stop(rc, out)
    assert frozen_names(freeze) == {FREEZE_M}
    knobs, sha = es.calibration().load_frozen(freeze, task="sudoku", arm="perstep_mar")
    assert knobs_of(knobs) == CM
    assert sha in json.dumps(rep), "the kept base file is listed in the report"


C9_DEAD = ["before_stack_check", "before_third_cell", "before_companion"]


@pytest.mark.parametrize("when", C9_DEAD)
def test_c9_a_dead_engine_at_probe_time_ends_the_launch_before_any_request(
    tmp_path: Path, when: str
) -> None:
    """C9's probe. The engine dies (every later connection dropped, as a crashed server) right after the last
    reply before the stack check / the third cell / the twin (limit 2). A driver that probes before each run
    ends the launch there with an engine stop and sends no request to the dead engine
    (``posts_while_dead == 0``). The runner's own ``GET /v1/models`` at run start does not save a driver that
    does not probe: it treats a failure as "model unknown" and posts anyway. Before the twin, C13: the base
    file (frozen from a healthy grid) stays."""
    limit = 2
    count = [0]
    last_cell_request = {"before_third_cell": 2 * limit, "before_companion": 18 * limit}

    def mar(rid: str, k: tuple[float, float]) -> Reply:
        count[0] += 1
        die = count[0] == last_cell_request.get(when, -1)
        return sudoku_reply(rid, "solved" if k == CM else "wrong", die_after=die)

    sudoku = script(solved_unconstrained, mar, lambda rid, k: sudoku_reply(rid, "solved"))
    with es.fake(sudoku, tmp_path) as srv:
        if when == "before_stack_check":
            srv.die()
        rc, out, freeze = launch_arms(tmp_path, srv, ["perstep_mar"], limit=limit)
        dead_posts, mars, scs = srv.posts_while_dead, mar_requests(srv), sc_requests(srv)
        n_requests = len(srv.requests)
    assert dead_posts == 0, f"{dead_posts} requests sent to a dead engine: no probe before the run"
    if when == "before_stack_check":
        assert n_requests == 0
    elif when == "before_third_cell":
        assert len(set(mars)) == 2
    else:
        assert len(set(mars)) == 18 and scs == []
    if when == "before_companion":
        assert frozen_names(freeze) == {FREEZE_M}
    else:
        assert frozen_names(freeze) == set()
    _assert_engine_stop(rc, out)


# --------------------------------------------------------------------------- C6 / C11: the locked script

#: commands the harness replaces; anything reaching a real engine, container or GPU is among them
STUBBED = (
    "git",
    "ss",
    "docker",
    "sg",
    "setsid",
    "sleep",
    "uptime",
    "make",
    "curl",
    "nvidia-smi",
    "uv",
)
#: commands the script must not call at all in the harness (they fail loudly)
FORBIDDEN = ("sudo", "systemctl", "podman", "vllm")


def _stub(bin_dir: Path, name: str, body: str) -> None:
    p = bin_dir / name
    p.write_text(f'#!/bin/bash\necho "{name} $*" >> "$T_CALLS"\n{body}\n')
    p.chmod(0o755)


def run_locked(
    tmp_path: Path,
    args: list[str],
    *,
    driver_exit: int = 0,
    smi_fails: bool = False,
    dirty: bool = False,
    make_exit: int = 0,
    curl_exit: int = 0,
) -> tuple[int, str, list[list[str]]]:
    """Run a copy of the locked script from a throwaway top-level directory with every external command that
    could reach an engine, a container or the GPU stubbed (``STUBBED``; ``FORBIDDEN`` fail loudly). The copy has
    no ``services/`` directory, so even an unstubbed engine start has nothing to run; a script that names one of
    those commands by absolute path (bypassing the stubs) fails the row before it runs. CPU only, no request.
    Returns the exit code, the console output and the argv of every ``uv`` (driver) call."""
    text = LOCKED.read_text()
    names = "|".join(re.escape(n) for n in STUBBED + FORBIDDEN)
    absolute = re.findall(r"/(?:usr/)?(?:local/)?s?bin/(" + names + r")\b", text)
    if absolute:
        pytest.fail(
            f"the locked script calls {sorted(set(absolute))} by absolute path; "
            "the harness cannot stub it"
        )
    top = tmp_path / "top"
    (top / "scripts" / "locked").mkdir(parents=True)
    shutil.copy(LOCKED, top / "scripts" / "locked" / LOCKED.name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv_log = tmp_path / "uv.argv"
    _stub(
        bin_dir,
        "git",
        'case "$*" in\n'
        '  *--show-toplevel*) echo "$T_TOP" ;;\n'
        '  *"rev-parse --short"*) echo abc1234 ;;\n'
        '  *"rev-parse HEAD"*) echo abc1234abc1234abc1234abc1234abc1234abc1 ;;\n'
        '  *"status --porcelain"*) if [ -n "$T_DIRTY" ]; then echo " M x.py"; fi ;;\n'
        "  *) ;;\nesac\nexit 0",
    )
    for quiet in ("ss", "docker", "sg", "setsid", "sleep", "uptime"):
        _stub(bin_dir, quiet, "exit 0")
    _stub(bin_dir, "make", 'exit "$T_MAKE_EXIT"')
    _stub(bin_dir, "curl", 'exit "$T_CURL_EXIT"')
    _stub(
        bin_dir,
        "nvidia-smi",
        'if [ -n "$T_SMI_FAIL" ]; then echo "NVIDIA-SMI has failed" >&2; exit 9; fi\n'
        'echo "NVIDIA GeForce RTX 5090, 595.84, 1000 MiB, 31000 MiB"',
    )
    _stub(
        bin_dir,
        "uv",
        'printf "%s\\n" "$@" >> "$T_UV"; echo "--end--" >> "$T_UV"; exit "$T_DRIVER_EXIT"',
    )
    for loud in FORBIDDEN:
        _stub(bin_dir, loud, 'echo "unexpected command in the locked-script harness" >&2; exit 97')
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "T_CALLS": str(tmp_path / "calls.log"),
        "T_TOP": str(top),
        "T_UV": str(uv_log),
        "T_DIRTY": "1" if dirty else "",
        "T_SMI_FAIL": "1" if smi_fails else "",
        "T_MAKE_EXIT": str(make_exit),
        "T_CURL_EXIT": str(curl_exit),
        "T_DRIVER_EXIT": str(driver_exit),
    }
    proc = subprocess.run(
        ["bash", str(top / "scripts" / "locked" / LOCKED.name), *args],
        cwd=top,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    calls: list[list[str]] = []
    if uv_log.is_file():
        cur: list[str] = []
        for line in uv_log.read_text().splitlines():
            if line == "--end--":
                calls.append(cur)
                cur = []
            else:
                cur.append(line)
    return proc.returncode, proc.stdout + proc.stderr, calls


def _arms_of(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "--arm"]


def test_c6_locked_script_defaults_to_both_grid_arms(tmp_path: Path) -> None:
    """C6: ``t0071b_calib.sh sudoku`` with no arm runs the driver once with ``--arm perstep_mar --arm
    perstep_mar_qargmax`` (one boot, both arms), on the dev cut."""
    rc, output, calls = run_locked(tmp_path, ["sudoku"])
    assert rc == 0, output
    assert len(calls) == 1, f"driver calls: {calls}\n{output}"
    argv = calls[0]
    assert any(a.endswith("cd_campaign.py") for a in argv), argv
    assert _arms_of(argv) == ["perstep_mar", QA], argv
    assert argv[argv.index("--cut") + 1] == "dev"


@pytest.mark.parametrize("driver_exit", [0, 1])
def test_c11_a_gpu_fault_in_smi_does_not_mask_the_driver_exit(
    tmp_path: Path, driver_exit: int
) -> None:
    """C11: ``nvidia-smi`` failing (a GPU fault; exit 9, no output) must neither stop the launch before the driver
    nor replace its exit code: the driver runs, the script exits with the driver's code and prints
    ``driver exit <code>``."""
    rc, output, calls = run_locked(tmp_path, ["sudoku"], smi_fails=True, driver_exit=driver_exit)
    assert len(calls) == 1, f"the driver did not run:\n{output}"
    assert rc == driver_exit, output
    assert f"driver exit {driver_exit}" in output


EARLY_EXITS: dict[str, tuple[list[str], dict[str, Any]]] = {
    "usage": (["test"], {}),  # an unknown task
    "sc_twin": (["sudoku", "perstep_mar_sc"], {}),  # C4: a twin is refused before the boot
    "dirty": (["sudoku"], {"dirty": True}),  # a dirty tree
    "make": (["sudoku"], {"make_exit": 2}),  # the plugin wheel build fails (set -e would be silent)
    "unhealthy": (["sudoku"], {"curl_exit": 7}),  # the engine never answers /health
}
RECOGNISABLE = re.compile(r"(?m)^.*\b(driver exit \d+|REFUSED|FAIL)\b")


@pytest.mark.parametrize("case", list(EARLY_EXITS))
def test_c11_every_early_exit_prints_a_recognisable_line(tmp_path: Path, case: str) -> None:
    """C11: an early exit -- a refusal or a failure before the driver -- exits non-zero, never runs the driver,
    and prints a ``REFUSED``/``FAIL``/``driver exit N`` line, including exits that ``set -e`` would otherwise
    make silent (a failed ``make``)."""
    args, kw = EARLY_EXITS[case]
    rc, output, calls = run_locked(tmp_path, args, **kw)
    assert rc != 0, output
    assert calls == [], "the driver ran"
    assert RECOGNISABLE.search(output), f"no recognisable exit line in:\n{output}"


def _code_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def test_c11_script_text_smi_pipelines_and_documented_launch_commands() -> None:
    """C11, on the script's text (comment lines are skipped for the first check, so prose cannot trip it):
    - every ``nvidia-smi ... | ...`` pipeline line ends with ``|| true``;
    - the documented launch commands in the header comment write per-SHA console logs
      (``logs/t0071b-calib-<task>-<sha>-run.log``) and none reports the backgrounding status (a header line
      that is only ``echo "exit $?"``, run after the ``&``)."""
    text = LOCKED.read_text()
    piped = [ln for ln in _code_lines(text) if "nvidia-smi" in ln and re.search(r"[^|]\|[^|]", ln)]
    assert piped, "no nvidia-smi pipeline found (the row's premise changed)"
    bad = [ln.strip() for ln in piped if not ln.rstrip().endswith("|| true")]
    assert bad == [], f"nvidia-smi pipelines without '|| true': {bad}"
    header = [ln for ln in text.splitlines() if ln.startswith("#")]
    launches = [ln for ln in header if re.search(r"t0071b_calib\.sh (sudoku|bfcl)\b.*>", ln)]
    assert launches, "no documented launch command (with a console log) in the script header"
    for ln in launches:
        assert re.search(r"logs/t0071b-calib-(sudoku|bfcl)-\S+-run\.log", ln), (
            f"console log not per SHA: {ln}"
        )
    bare = [ln for ln in header if re.match(r"#\s*echo\s+\"?exit \$\?", ln)]
    assert bare == [], f"documented exit capture reports the backgrounding status: {bare}"
