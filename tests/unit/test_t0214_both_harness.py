"""T-0214 group A: the one-arm ("both": ReGround crop 280 + the action-outcome clause) three.js harness
``scripts/gpu/t0214_threejs_both.py`` (unit; CPU; no engine, no socket). docs/tasks/T-0214.md A1-A10, C2, N1, N2 and
"Tests to write" (``tests/unit/test_t0214_both_harness.py``).

BASE 457b86b3 (main; ``Merge T-0212:`` 83f7609c is an ancestor). The rows that need the harness carried the T-0214
pending marker until ``/implement`` (all activated 2026-10-31 at 576703ed); a missing file is a named assertion
("does not exist yet") from :func:`harness`, never an import error. Rows that do not need it (the N1 byte pins, the
eval's import fence) are active and green at BASE by design (negative criteria).

**The harness surface these rows use** (the orchestrator's interface ruling of 2026-10-31; a thin adapter here may
reshape a call, never compute a metric)

Modules and constants
    ``R`` (T-0212's harness, loaded by path), ``M``/``D``/``G`` (``R.M``/``R.D``/``R.G`` by identity); ``ARMS``;
    ``ORDER_SEED``; ``HARD_LIMIT_S``, ``RUN_BUDGET_S``, ``TURN_WORST_S``, ``REPORT_S``; ``EXIT_*`` 0..3;
    ``PROCESS_START_S`` and a ``time`` name the budget clock reads (the rows patch both on the harness and on ``R``);
    ``ANCHOR``, ``ANCHOR_BASE_KEYWORDS_SHA256``.
A2/A3
    ``both_context()``; ``both_keywords(profile)``; ``both_grounding(profile)`` (a dict of ``cli.grounding_of``'s
    fields, as ``R.arm_grounding``). P-A has no public function in the ruling, so the P-A mutants run through
    ``main`` (exit 2, zero sockets), with ``both_context`` read by its global name at call time.
A5/A6 and the report
    ``order(seed) -> [(task, "both", r)]``; ``async _run_all(run, turns)``; ``finish_turn(out, task, arm, repeat, trial,
    steps)``; ``build_report(trials, steps, records_by_key, probes, stopped, manifest)`` (keys ``label``,
    ``violations``, ``per_arm["both"]``, ``per_task["both"]``, ``regime``, ``regime_none``, ``regime_drift``,
    ``anchor``, ``reading = {reading, mark, line, rule}``); ``write_report(out, probes, manifest, stopped)``;
    ``main(argv)`` with ``--base-url`` and ``--out-dir``. The harness defines its own ``reading_mark`` (ruling), so
    that name is not in the A1 no-redefinition list.

Readings (reported to the orchestrator; not resolved silently): see the eval module's (A10 success as an exact rate
``Fraction(k, n)``; the drift pin with json's default separators). A6's ``RUN_BUDGET_S = 1080`` can never bind
before the hard-limit check (a turn may start only at ``elapsed <= 1500 - 540 = 960``), so no row can tell it
apart from an absent run-budget check; ``test_budget_constants_and_check`` pins the constant and the hard-limit
boundary only.

Regime: CPU, pure, hand-built records and stubbed ``_one_turn``/``main``; no engine; n = 1.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import copy
import dataclasses
import functools
import hashlib
import importlib.util
import inspect
import json
import random
import re
import shutil
import socket
import subprocess
import sys
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests import _git_range as gr
from tests.unit import test_t0131_ab_harness as t31
from tests.unit import test_t0205_demo_harness as t205
from tests.unit import test_t0210_threejs_harness as t210
from tests.unit import test_t0212_arms_harness as t212

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0214_threejs_both.py"
EVAL = REPO / "tests" / "eval" / "test_t0214_threejs_both.py"
T0212_HARNESS_REL = "scripts/gpu/t0212_threejs_arms.py"
#: ``git log --merges``: 83f7609c "Merge T-0212: three-arm three.js A/B harness ..." (A1/N1's pin; a literal)
MERGE_T0212_SHA = "83f7609ce33fd5ec2888ba07b1a124ba561fc5d4"
MERGE_T0212 = (
    f"{MERGE_T0212_SHA} Merge T-0212: three-arm three.js A/B harness (base, action outcome, ReGround; 45 turns, "
    "report-only), eval, locked wrapper"
)
#: the T-0212 GPU run (a local log, not in the repo; A10's anchor)
ANCHOR_REL = Path("logs") / "t0212-threejs-arms-5b87b72"

TASK_IDS = ("M1", "M2", "M3", "M4", "M5")
ARM = "both"
LEVERS = ("LIVE_DEFAULT_ACTION_OUTCOME", "LIVE_DEFAULT_REGROUND")
BOTH_DIFF = {
    "added": {"action_outcome": True},
    "removed": ["overview_frame"],
    "changed": {"reground": [False, True]},
}
WORTH = "worth a paired follow-up"
NO_SIGN = "no sign beyond ReGround alone"
MIXED = "mixed"
VERDICTS = (WORTH, NO_SIGN, MIXED)
SUPERSEDED = "superseded by a safety event"
DRIFT_MARK = "provisional (regime drift)"


# =====================================================================================================
# helpers
# =====================================================================================================


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0214 A1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0214_threejs_both_unit", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{HARNESS.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(spec.name, None)
    return module


def api(name: str) -> Any:
    h = harness()
    assert hasattr(h, name), f"T-0214: {HARNESS.name} has no {name!r} (see this module's docstring)"
    return getattr(h, name)


def _r() -> ModuleType:
    r = api("R")
    assert isinstance(r, ModuleType) and str(r.__file__).endswith("t0212_threejs_arms.py"), r
    return r


def _m() -> ModuleType:
    return _r().M


@functools.cache
def _eval() -> ModuleType:
    from importlib import import_module

    return import_module("tests.eval.test_t0214_threejs_both")


def _held(h: ModuleType) -> list[ModuleType]:
    """The harness and every ``scripts/gpu`` module it holds by name, four levels down (R, M, D, D's chain)."""
    seen: dict[int, ModuleType] = {id(h): h}
    frontier = [h]
    for _ in range(4):
        nxt = []
        for mod in frontier:
            for v in vars(mod).values():
                if (
                    isinstance(v, ModuleType)
                    and id(v) not in seen
                    and "/scripts/gpu/" in str(getattr(v, "__file__", "") or "")
                ):
                    seen[id(v)] = v
                    nxt.append(v)
        frontier = nxt
    return list(seen.values())


def _patch_name(mp: pytest.MonkeyPatch, h: ModuleType, name: str, value: Any) -> None:
    mp.setattr(h, name, value, raising=False)
    for mod in _held(h):
        if hasattr(mod, name):
            mp.setattr(mod, name, value)


def _lean() -> Any:
    from local_astra.config import load

    return load().profiles["lean"]


def _levers() -> tuple[Any, Any]:
    from local_astra import cli

    return cli.LIVE_DEFAULT_ACTION_OUTCOME, cli.LIVE_DEFAULT_REGROUND


def _line(report: Any) -> str:
    line = _eval()._reading_line(report)
    assert isinstance(line, str) and line, ("T-0214: the report has no reading line", report.keys())
    return line


@contextlib.contextmanager
def _only(name: str) -> Iterator[None]:
    """A mutant ``both_context`` that rebinds one lever only."""
    from local_astra import cli

    held = getattr(cli, name)
    setattr(cli, name, True)
    try:
        yield
    finally:
        setattr(cli, name, held)


def _manifest(stopped: str | None = None, **regime_over: Any) -> dict[str, Any]:
    """The manifest in ``_measure``'s shape (T-0212's), the arm record from the harness's ``both_keywords`` and
    ``both_grounding`` (base: ``D.session_keywords`` outside any context), through ``R``'s ``_jsonable`` and
    ``keyword_diff``."""
    from local_astra.pages import threejs_tasks

    h, m, r = harness(), _m(), _r()
    lean = _lean()
    kw = {"base": dict(r.D.session_keywords(lean)), "both": dict(h.both_keywords(lean))}
    return {
        "regime": {**REGIME, **regime_over},
        "live_default": dict(m.live_default(lean)),
        "arm_keywords": {a: r._jsonable(v) for a, v in kw.items()},
        "arm_diffs": {ARM: r._jsonable(r.keyword_diff(kw["base"], kw["both"]))},
        "arm_grounding": {"base": r.arm_grounding(lean, "base"), "both": h.both_grounding(lean)},
        "tool_guidance_sha256": {
            a: hashlib.sha256(str(v.get("tool_guidance")).encode("utf-8")).hexdigest()
            for a, v in kw.items()
        },
        "search": "off",
        "task_table_sha256": threejs_tasks.task_table_sha(),
        "provenance_sha256": m.provenance_sha256(),
        "cli_before": _r().cli_defaults(),
        "stopped": stopped,
        "cli_after": _r().cli_defaults(),
    }


def _report(
    turns: list[dict[str, Any]],
    stopped: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``build_report`` over hand turns ``{"trial", "records"}`` (arm "both"; good probes)."""
    trials, by_key = [], {}
    for t in turns:
        trial = {**t["trial"], "arm": ARM}
        trials.append(trial)
        by_key[(str(trial["task"]), ARM, int(trial.get("repeat") or 0))] = t["records"]
    mf = manifest if manifest is not None else _manifest(stopped)
    return dict(api("build_report")(trials, [], by_key, t210._probes(), stopped, mf))


#: A9's regime with distinctive values (the synthetic manifest's ``regime``)
REGIME: dict[str, Any] = {
    "commit": "c0ffee1" + "4" * 33,
    "profile": "lean",
    "canvas_length": 256,
    "max_denoising_steps": 48,
    "thinking": True,
    "engine_version": "vllm-T0214-0.29.0",
    "model": "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
    "model_revision": "2ea83723",
    "kv_cache": "4 GiB bf16 T0214",
    "max_num_seqs": 3,
    "prefix_caching": "on (T0214)",
    "n_per_arm": 15,
    "turns": 15,
    "order_seed": 0,
    "turn_timeout_s": 300,
    "run_budget_s": 1080,
    "hard_limit_s": 1500,
    "turn_worst_s": 480,
    "turn_worst_note": "T0214-WORST-CASE-NOTE pass 2 is bounded by REGROUND_TIMEOUT_MS",
    "report_s": 60,
    "crop_budget": 280,
    "crop_source": "default",
    "regrounding_timeout_ms": 4000,
    "chromium": "chromium-T0214-1187",
    "playwright": "playwright-T0214-1.55.0",
    "versions_lock_sha256": "e" * 63 + "f",
    "search": "off",
    "input": "typed (UserText) T0214; fake STT/TTS",
}


# =====================================================================================================
# 1. reuse and arm isolation (A1, A2, A3, N2)
# =====================================================================================================


def test_reuse_runner_objects() -> None:
    """A1: ``R`` is T-0212's harness under a fixed ``sys.modules`` name; ``M``/``D``/``G`` are ``R``'s module objects;
    ``ARMS == ("both",)``; every metric function A1 lists, where the harness names it, is ``R``'s own object."""
    h, r = harness(), _r()
    assert sys.modules.get(r.__name__) is r and r.__name__ != "__main__", r.__name__
    assert (h.M, h.D, h.G) == (r.M, r.D, r.G)
    assert h.M is r.M and h.D is r.D and h.G is r.G
    assert tuple(h.ARMS) == ("both",)
    for name in REUSED:
        if hasattr(h, name):
            assert getattr(h, name) is getattr(r, name), f"A1: {name} is not R's own"
    text = HARNESS.read_text(encoding="utf-8")
    assert "t0212_threejs_arms.py" in text, "A1: T-0212's harness is not loaded by path"
    assert "_one_turn" in text, "A1: turns do not go through M._one_turn"


#: A1's reused metric functions (``turn_row`` and ``pa_problems`` are reused only in part: a T-0214 equivalent may
#: exist, implementation notes; ``reading_mark`` is the harness's own by the 2026-10-31 interface ruling, since
#: ``R.reading_mark`` has no drift mark)
REUSED = (
    "step_labels",
    "click_misses",
    "reaims",
    "reground_stats",
    "completion_outcomes",
    "completion_asked",
    "repeat_guard_refusals",
    "outcome_kinds",
    "turn_extras",
    "arm_summary",
    "report_label",
    "records_by_key",
    "keyword_diff",
    "_jsonable",
)


def test_harness_never_writes_into_or_redefines_a_runner() -> None:
    """A1 (AST): no assignment to, or ``setattr`` on, an attribute of ``R``/``M``/``D``/``G``/``T``/``P``; no ``def``
    of a reused metric function."""
    harness()
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    runners = {"R", "M", "D", "G", "T", "P"}
    bad = [
        ast.unparse(t)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign)
        for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(t, ast.Attribute)
        and isinstance(t.value, ast.Name | ast.Attribute)
        and (t.value.id if isinstance(t.value, ast.Name) else t.value.attr) in runners
    ]
    assert bad == [], f"A1: the harness writes into an imported runner: {bad}"
    setattrs = [
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in ("setattr", "delattr")
        and n.args
        and isinstance(n.args[0], ast.Name | ast.Attribute)
        and (n.args[0].id if isinstance(n.args[0], ast.Name) else n.args[0].attr) in runners
    ]
    assert setattrs == [], setattrs
    redefined = sorted(
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name in REUSED
    )
    assert redefined == [], f"A1: the harness redefines R's metric functions: {redefined}"


def test_both_keywords_exact_diff() -> None:
    """A3: ``keyword_diff(base, both)`` is exactly A3's dict; ``tool_guidance`` byte-equal; base is
    ``D.session_keywords(lean)`` outside any context (T-0212's base arm too); both levers False afterwards."""
    h, r = harness(), _r()
    lean = _lean()
    base = dict(r.D.session_keywords(lean))
    assert base == dict(r.arm_keywords(lean, "base"))
    both = dict(h.both_keywords(lean))
    assert r.keyword_diff(base, both) == BOTH_DIFF, r.keyword_diff(base, both)
    assert isinstance(both["tool_guidance"], str)
    assert both["tool_guidance"].encode() == base["tool_guidance"].encode()
    assert dict(r.D.session_keywords(lean)) == base, "A3: building both moved base"
    assert _levers() == (False, False), "N2: both_keywords left a lever rebound"


def test_both_is_reground_plus_outcome() -> None:
    """A3 cross-check: both = T-0212's reground arm + ``action_outcome: True``; against T-0212's outcome arm it
    differs exactly by ``reground`` flipped and ``overview_frame`` removed."""
    h, r = harness(), _r()
    lean = _lean()
    both = dict(h.both_keywords(lean))
    assert both == dict(r.arm_keywords(lean, "reground")) | {"action_outcome": True}
    assert r.keyword_diff(dict(r.arm_keywords(lean, "outcome")), both) == {
        "added": {},
        "removed": ["overview_frame"],
        "changed": {"reground": [False, True]},
    }
    assert _levers() == (False, False)


def test_both_grounding_280_default() -> None:
    """A3: ``cli.grounding_of(profile, D._live_args(), "live")`` under ``both_context()``: ReGround on, crop 280 from
    ``"default"``, main budget 1120; P-A records the same, and base's ReGround off."""
    from local_astra import cli

    h = harness()
    lean = _lean()
    with h.both_context():
        g = cli.grounding_of(lean, _r().D._live_args(), "live")
    assert (
        g.reground,
        g.reground_visual_tokens,
        g.reground_visual_tokens_from,
        g.visual_tokens,
    ) == (
        True,
        280,
        "default",
        1120,
    ), g
    gb = dict(h.both_grounding(lean))
    assert (gb["reground"], gb["reground_visual_tokens"], gb["reground_visual_tokens_from"]) == (
        True,
        280,
        "default",
    ), gb
    assert gb["visual_tokens"] == 1120, gb
    assert cli.grounding_of(lean, _r().D._live_args(), "live").reground is False
    assert _levers() == (False, False)


class _FlakyCli(types.ModuleType):
    """``local_astra.cli`` whose second ``setattr`` to ``True`` raises; every other name delegates to the real one."""

    def __init__(self, real: ModuleType) -> None:
        super().__init__(real.__name__)
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_sets", [])

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        sets = object.__getattribute__(self, "_sets")
        if value is True:
            sets.append(name)
            if len(sets) == 2:
                raise RuntimeError("T-0214 rig: the second setattr failed")
        setattr(object.__getattribute__(self, "_real"), name, value)


def test_both_context_restores_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """A2/N2 (adversarial): inside the context both levers are True; a raise in the body leaves both False; when the
    second ``setattr`` itself raises, the first lever is restored too (both False)."""
    import local_astra
    from local_astra import cli

    h = harness()
    seen: list[tuple[Any, Any]] = []
    with pytest.raises(RuntimeError, match="T-0214 rig: body"), h.both_context():
        seen.append(_levers())
        raise RuntimeError("T-0214 rig: body raised")
    assert seen == [(True, True)], seen
    assert _levers() == (False, False), f"A2: a raise left {_levers()}"

    flaky = _FlakyCli(cli)
    monkeypatch.setitem(sys.modules, "local_astra.cli", flaky)
    monkeypatch.setattr(local_astra, "cli", flaky)
    if hasattr(h, "cli"):
        monkeypatch.setattr(h, "cli", flaky)
    with pytest.raises(RuntimeError, match="second setattr"), h.both_context():
        pass
    assert object.__getattribute__(flaky, "_sets"), "T-0214 rig: both_context never set a lever"
    monkeypatch.undo()
    assert _levers() == (False, False), f"A2: a failed second setattr left {_levers()}"


def test_both_context_refuses_nesting() -> None:
    """A2: ``both_context`` inside itself is refused (``RuntimeError``) and the outer rebinding survives; inside
    T-0212's ``arm_context("outcome")`` it is refused, rebinds nothing and never appends to ``R._ACTIVE``."""
    h, r = harness(), _r()
    with h.both_context():
        with pytest.raises(RuntimeError), h.both_context():
            pass
        assert _levers() == (True, True)
    assert _levers() == (False, False)
    with r.arm_context("outcome"):
        assert list(r._ACTIVE) == ["outcome"]
        with pytest.raises(RuntimeError), h.both_context():
            pass
        assert list(r._ACTIVE) == ["outcome"], r._ACTIVE
        assert _levers() == (True, False)
    assert list(r._ACTIVE) == [] and _levers() == (False, False)


def test_run_all_turn_under_both_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A2/A5: each turn runs through ``M._one_turn(run, task, r)`` with both levers on across the whole await (read
    after the stub yields to the loop); both False after the run; every trial carries arm ``both``."""
    start = 1000.0
    h, run, clock = _run(tmp_path, monkeypatch, start)
    seen = _stub_turns(monkeypatch, clock, start, [10.0 * k for k in range(1, 16)])
    turns = [tuple(x) for x in h.order(0)][:4]
    asyncio.run(h._run_all(run, turns))
    assert [(t, rp, lv) for t, rp, lv in seen] == [(t, b, (True, True)) for t, _a, b in turns]
    assert _levers() == (False, False)
    assert [(x["task"], x.get("arm")) for x in _trials(run.out)] == [
        (t, ARM) for t, _a, _b in turns
    ]


def test_never_assigns_runner_attributes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A1: ``vars(R)``, ``vars(R.M)``, ``vars(R.D)``, ``vars(R.G)`` identical (by identity) before and after a
    ``both_context``, ``both_keywords``, ``both_grounding`` and a stub turn through ``_run_all``; ``R._ACTIVE``
    unchanged."""
    h, r = harness(), _r()
    start = 50.0
    _h, run, clock = _run(tmp_path, monkeypatch, start)
    _stub_turns(monkeypatch, clock, start, [5.0, 10.0])
    mods = (r, r.M, r.D, r.G)
    before = [dict(vars(x)) for x in mods]
    active = list(r._ACTIVE)
    with h.both_context():
        pass
    h.both_keywords(_lean())
    h.both_grounding(_lean())
    asyncio.run(h._run_all(run, [tuple(x) for x in h.order(0)][:1]))
    for x, snap in zip(mods, before, strict=True):
        now = vars(x)
        changed = sorted(k for k in set(snap) | set(now) if snap.get(k, ...) is not now.get(k, ...))
        assert changed == [], f"A1: {x.__name__} attributes changed: {changed}"
    assert list(r._ACTIVE) == active


# ---- P-A mutants (A3) -------------------------------------------------------------------------------------------


PA_MUTANTS = ("only-reground", "only-outcome", "crop-560", "crop-1120", "overview-frame-kept")


def _neuter_t0212_pa(mp: pytest.MonkeyPatch) -> None:
    """T-0212's own P-A (``R.pa_check`` inside ``R._refusal``) reports no problem; its keywords and groundings are
    unchanged. A T-0214 refusal then has to be T-0214's own."""
    r = _r()
    real = r.pa_check
    mp.setattr(r, "pa_check", lambda p: {**real(p), "problems": []})


@pytest.mark.parametrize("case", PA_MUTANTS)
def test_pa_kills_mutants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """A3 mutants P-A must kill, through ``main`` (exit 2, zero sockets, no out-dir): (a) only ReGround rebound,
    (b) only the outcome clause rebound, (c) the crop not 280, (d) ``overview_frame`` kept. T-0212's P-A would also
    refuse (c) and (d), so it is neutered here; ``test_pa_neutering_premise`` is the too-wide side."""
    from local_astra import cli

    rig = Rig(tmp_path, monkeypatch)
    h, r = rig.h, _r()
    _neuter_t0212_pa(monkeypatch)
    lean = _lean()
    if case == "only-reground":
        monkeypatch.setattr(h, "both_context", lambda: _only("LIVE_DEFAULT_REGROUND"))
    elif case == "only-outcome":
        monkeypatch.setattr(h, "both_context", lambda: _only("LIVE_DEFAULT_ACTION_OUTCOME"))
    elif case.startswith("crop-"):
        crop = int(case.split("-")[1])
        monkeypatch.setattr(cli, "LIVE_DEFAULT_REGROUND_VISUAL_TOKENS", crop)
        # the crop is not a session keyword: the diff is still A3's; only the grounding clause sees it
        diff = r.keyword_diff(dict(r.D.session_keywords(lean)), dict(h.both_keywords(lean)))
        assert diff == BOTH_DIFF, diff
        assert dict(h.both_grounding(lean))["reground_visual_tokens"] == crop
    elif case == "overview-frame-kept":
        monkeypatch.setattr(cli, "_overview_frame_others", lambda *a, **k: ())
        assert "overview_frame" in dict(h.both_keywords(lean)), "T-0214 rig: the yield did not move"
    else:
        raise KeyError(case)
    rc = h.main(rig.argv())
    err = capsys.readouterr().err
    assert rc == 2, f"A3: P-A passed mutant {case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [], (case, rig.sockets)
    assert not rig.out.exists(), f"{case}: the refusal made the out-dir"
    assert _levers() == (False, False)


def test_pa_neutering_premise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The too-wide side of ``test_pa_kills_mutants``: T-0212's P-A neutered and no mutant, ``main`` refuses
    nothing (exit 1 after trying the engine)."""
    rig = Rig(tmp_path, monkeypatch)
    _neuter_t0212_pa(monkeypatch)
    rc = rig.h.main(rig.argv())
    err = capsys.readouterr().err
    assert rc == 1 and rig.sockets, (rc, rig.sockets, err[-2000:])


# =====================================================================================================
# 2. refusals (A4), the frozen files (A1/N1), the state read
# =====================================================================================================


class Rig:
    """Every real socket refused and recorded; a clean tree and the merges through ``_git`` (patched on the harness
    and every runner it holds; ``show`` reads the real blobs); the kit cli resolves (repair on); Chromium present.

    ``show`` needs the real commits ``Merge T-0210-c:`` (fec43cc3, T-0212's runner pins) and ``Merge T-0212:``
    (83f7609c, T-0214's own pin): without them every ``main`` refuses on "differ" and a refusal row would pass for the
    wrong reason. Release re-pin (James 2026-11-01): the squashed release has no history, so every rig row skips
    naming the commit; on main it runs."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        gr.require_commits(t212.MERGE_T0210C_SHA, MERGE_T0212_SHA, why=gr.NO_HISTORY)
        from local_astra import cli

        self.h = h = harness()
        self.cli = cli
        self.sockets: list[str] = []
        self.porcelain = ""
        self.merges: tuple[str, ...] = (
            *t210.OTHERS,
            t210.MERGE_A,
            t210.MERGE_B,
            t212.MERGE_C,
            MERGE_T0212,
        )
        self.out = tmp_path / "out"
        self.altered: set[str] = set()

        def refuse(what: str) -> Callable[..., Any]:
            def fn(*a: Any, **k: Any) -> Any:
                self.sockets.append(what)
                raise ConnectionRefusedError(f"T-0214 refusal rig: no {what}")

            return fn

        for name in ("connect", "connect_ex", "bind"):
            mp.setattr(socket.socket, name, refuse(name))
        mp.setattr(socket, "create_connection", refuse("create_connection"))
        mp.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))

        def git(*args: str) -> str:
            if args[:1] == ("status",):
                return self.porcelain
            if "log" in args:
                if any("%H" in a for a in args):
                    return "\n".join(self.merges) + "\n"
                return "\n".join(m.split(" ", 1)[1] for m in self.merges) + "\n"
            if "merge-base" in args:
                return ""
            if args[:1] == ("show",):
                blob = subprocess.run(
                    ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
                ).stdout
                if any(a.endswith(":" + rel) for a in args for rel in self.altered):
                    blob += "# T-0214 rig: the merge's bytes differ from the working file\n"
                return blob
            return t31.GIT + "\n"

        _patch_name(mp, h, "_git", git)

        def load(
            profile: Any,
            *,
            live_browser: Any = None,
            page_overview: bool = False,
            wait_tool: bool = False,
            **k: Any,
        ) -> Any:
            v = t205._set_sha(page_overview, wait_tool)
            return (t210._StubKit(tools_version=v, repair_off=None), object())

        mp.setattr(cli, "load_tool_kit", load)
        mp.setattr(cli, "live_prerequisites", lambda *a, **k: None, raising=False)

    def argv(self) -> list[str]:
        return ["--base-url", t31.VALID_URL, "--out-dir", str(self.out)]


REFUSALS = (
    "no-merge-t0212",
    "branch-merge-only",
    "t0212-harness-differs",
    "head-action-outcome-on",
    "head-reground-on",
    "pa-only-reground",
    "pa-only-outcome",
    "dirty-tree",
    "out-dir-exists",
)


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """A4: each refusal exits 2 with a stderr line before any socket, leaving the out-dir as it was. The T-0212
    merge, pin and P-A cases are T-0214's own (T-0212's ``_refusal`` passes them), so their exit 2 is this harness's."""
    rig = Rig(tmp_path, monkeypatch)
    h = rig.h
    base = (*t210.OTHERS, t210.MERGE_A, t210.MERGE_B, t212.MERGE_C)
    if case == "no-merge-t0212":
        rig.merges = base
    elif case == "branch-merge-only":
        rig.merges = (
            *base,
            "0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e0e Merge branch 'main' into T-0212-x",
        )
    elif case == "t0212-harness-differs":
        if _git_show(MERGE_T0212_SHA, T0212_HARNESS_REL) is None:
            pytest.skip(f"{MERGE_T0212_SHA[:8]} is not in this checkout")
        rig.altered = {T0212_HARNESS_REL}
    elif case == "head-action-outcome-on":
        monkeypatch.setattr(rig.cli, "LIVE_DEFAULT_ACTION_OUTCOME", True)
    elif case == "head-reground-on":
        monkeypatch.setattr(rig.cli, "LIVE_DEFAULT_REGROUND", True)
    elif case == "pa-only-reground":
        monkeypatch.setattr(h, "both_context", lambda: _only("LIVE_DEFAULT_REGROUND"))
    elif case == "pa-only-outcome":
        monkeypatch.setattr(h, "both_context", lambda: _only("LIVE_DEFAULT_ACTION_OUTCOME"))
    elif case == "dirty-tree":
        rig.porcelain = " M local_astra/conductor/session.py\n"
    elif case == "out-dir-exists":
        rig.out.mkdir()
        (rig.out / "keep.txt").write_text("x", encoding="utf-8")
    else:
        raise KeyError(case)
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = h.main(rig.argv())
    err = capsys.readouterr().err
    assert rc == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [], (case, rig.sockets)
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert err.strip(), f"{case}: the refusal says nothing"
    if case in ("no-merge-t0212", "branch-merge-only"):
        assert "T-0212" in err, err[-2000:]
    if case == "t0212-harness-differs":
        assert "t0212_threejs_arms.py" in err, err[-2000:]


def test_rig_at_head_goes_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A4, the too-wide direction: at HEAD (both merges reachable, levers off, P-A passing, pins intact) ``main``
    refuses nothing; it asks for the engine and, finding none, exits 1."""
    rig = Rig(tmp_path, monkeypatch)
    rc = rig.h.main(rig.argv())
    err = capsys.readouterr().err
    assert rc == 1, f"exit {rc}\n{err[-2000:]}"
    assert rig.sockets, "main never tried the engine"


def _git_show(rev: str, rel: str) -> bytes | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "show", f"{rev}:{rel}"], check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


@pytest.mark.parametrize(
    "rel",
    [
        T0212_HARNESS_REL,
        "scripts/locked/t0212_threejs_arms.sh",
        "scripts/gpu/t0210_threejs_run.py",
        "scripts/gpu/t0205_demo_run.py",
    ],
)
def test_t0212_harness_unchanged(rel: str) -> None:
    """N1 (negative criterion, green at BASE by design): the read-only runner files' bytes equal their blob at
    ``Merge T-0212:`` 83f7609c (a literal, never a moving ref). Skipped when that commit is absent."""
    pinned = _git_show(MERGE_T0212_SHA, rel)
    if pinned is None:
        pytest.skip(f"{MERGE_T0212_SHA[:8]} is not in this checkout")
    assert (REPO / rel).read_bytes() == pinned, f"N1: {rel} differs from Merge T-0212"


def test_state_read_literal() -> None:
    """A8: the harness names no state-read expression of its own: every string constant mentioning ``window.`` is
    ``M.STATE_READ_JS`` exactly; no ``JSON.stringify``, no ``.evaluate(`` call."""
    m = _m()
    assert m.STATE_READ_JS == "() => window.__state"
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    consts = [
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    assert [c for c in consts if "window." in c and c != m.STATE_READ_JS] == []
    assert not any("JSON.stringify" in c for c in consts)
    calls = [
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("evaluate", "evaluate_handle")
    ]
    assert calls == [], calls


def test_eval_imports_no_harness() -> None:
    """C1 (green at BASE by design): the eval imports nothing under ``scripts/`` and loads no harness by path; the
    only file it loads by path is T-0212's eval (docstrings may name the harnesses)."""
    tree = ast.parse(EVAL.read_text(encoding="utf-8"))
    mods = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names] + [
        n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
    ]
    assert not [x for x in mods if "scripts" in x or "t0212_threejs_arms" in x or "t0214" in x], (
        mods
    )
    docs = {
        id(b[0].value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Module | ast.FunctionDef | ast.ClassDef)
        for b in [n.body]
        if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)
    }
    paths = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and id(n) not in docs
        and n.value.endswith(".py")
    ]
    assert paths == ["test_t0212_threejs_arms.py"], paths


# =====================================================================================================
# 3. order, budget, not-run (A5, A6)
# =====================================================================================================


def test_order_blocks() -> None:
    """A5: 15 entries ``(task, "both", r)``; entries 5r..5r+4 are block r, M1-M5 once; one ``random.Random(0)``
    shuffles block 0, then 1, then 2 (task order before the shuffle); deterministic; seed 1 differs."""
    o = [tuple(x) for x in api("order")(0)]
    assert len(o) == 15
    for b in range(3):
        block = o[5 * b : 5 * b + 5]
        assert {x[2] for x in block} == {b} and {x[1] for x in block} == {ARM}, block
        assert sorted(x[0] for x in block) == list(TASK_IDS), block
    rng = random.Random(0)
    want = []
    for b in range(3):
        cells = list(TASK_IDS)
        rng.shuffle(cells)
        want += [(t, ARM, b) for t in cells]
    assert o == want
    assert [tuple(x) for x in api("order")(0)] == o
    assert [tuple(x) for x in api("order")(1)] != o
    assert api("ORDER_SEED") == 0


def _run(tmp_path: Path, mp: pytest.MonkeyPatch, start: float) -> tuple[Any, Any, Any]:
    h, r, m = harness(), _r(), _m()
    clock = t210._Clock(start)
    for mod in (h, r):
        mp.setattr(mod, "time", clock, raising=False)
        mp.setattr(mod, "PROCESS_START_S", start, raising=False)
    out = tmp_path / "out"
    (out / "sessions").mkdir(parents=True)
    known = {
        "base_url": "http://127.0.0.1:8001/v1",
        "out": out,
        "profile": _lean(),
        "sentinel": b"T0214-BUDGET-SENTINEL",
        "manifest": {"stopped": None},
        "vram": t205._FakeVram(),
    }
    names = {f.name for f in dataclasses.fields(m._Run)}
    return h, m._Run(**{k: v for k, v in known.items() if k in names}), clock


def _stub_turns(
    mp: pytest.MonkeyPatch,
    clock: Any,
    start: float,
    after: list[float],
    raise_on: Callable[[int], bool] = lambda k: False,
) -> list[tuple[str, int, tuple[Any, Any]]]:
    """``M._one_turn`` stubbed: records (task, repeat, levers after yielding to the loop) and sets the clock to
    ``start + after[k]``; turn k raises when ``raise_on(k)``."""
    m = _m()
    seen: list[tuple[str, int, tuple[Any, Any]]] = []

    async def one_turn(run: Any, tid: str, repeat: int, *a: Any, **k: Any) -> Any:
        await asyncio.sleep(0)
        seen.append((tid, repeat, _levers()))
        n = len(seen)
        clock.now = start + after[min(n, len(after)) - 1]
        if raise_on(n):
            raise RuntimeError(f"T-0214 rig: turn {n} raised")
        trial = t210._trial(tid, repeat, t212._st(tid, True), wall_s=1.0)
        trial["success"] = True
        return trial, [], None

    mp.setattr(m, "_one_turn", one_turn)
    return seen


def _trials(out: Path) -> list[dict[str, Any]]:
    p = out / "trials.jsonl"
    return (
        [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        if p.is_file()
        else []
    )


@pytest.mark.parametrize(("second", "want"), [(960.0, 2), (961.0, 1)])
def test_budget_constants_and_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second: float, want: int
) -> None:
    """A6: the constants (1500, 1080, 480, 60); turn 1 ends at elapsed ``second``: turn 2 starts iff
    ``second + 480 + 60 <= 1500`` (960 starts, 961 does not: T-0212's ``R.budget_stop`` at 3600 lets 961 start, a
    dropped check lets both start). A stop sets ``stopped`` and the report written after it is ``incomplete``."""
    h, m = harness(), _m()
    assert (h.HARD_LIMIT_S, h.RUN_BUDGET_S, h.TURN_WORST_S, h.REPORT_S) == (1500, 1080, 480, 60)
    assert m.TURN_TIMEOUT_S + m.READY_TIMEOUT_S + 2 * m.CLOSE_S <= h.TURN_WORST_S
    assert (h.EXIT_OK, h.EXIT_FAILED, h.EXIT_REFUSED, h.EXIT_PROBE) == (0, 1, 2, 3)
    start = 9000.0
    h, run, clock = _run(tmp_path, monkeypatch, start)
    seen = _stub_turns(monkeypatch, clock, start, [second, second + 1.0, second + 2.0])
    asyncio.run(h._run_all(run, [tuple(x) for x in h.order(0)][:3]))
    assert len(seen) == want, (second, len(seen))
    if want == 1:
        assert isinstance(run.stopped, str) and run.stopped
        h.write_report(run.out, t210._probes(), _manifest(run.stopped), run.stopped)
        report = json.loads((run.out / "report.json").read_text(encoding="utf-8"))
        assert report.get("label") == "incomplete", report.get("label")
        assert "provisional (run incomplete)" in _line(report)
        assert (run.out / "report.md").is_file()


def test_errored_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A6: after ``G.ERRORED_STOP`` consecutive errored turns the run stops (nothing more is started)."""
    start = 100.0
    h, run, clock = _run(tmp_path, monkeypatch, start)
    seen = _stub_turns(
        monkeypatch, clock, start, [5.0 * k for k in range(1, 16)], raise_on=lambda k: True
    )
    asyncio.run(h._run_all(run, [tuple(x) for x in h.order(0)]))
    assert len(seen) == _m().G.ERRORED_STOP
    assert isinstance(run.stopped, str) and run.stopped
    assert _levers() == (False, False)


def test_not_run_is_errored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A6/N2 (adversarial): ``_one_turn`` raises at turns 2 and 3: both are ``M._not_run`` lines of arm ``both``
    (success False, ``state_error`` True), never dropped; the run goes on; the levers are False; the report counts
    2 errored with success 2/4 and the reading carries "provisional (errored 2)"."""
    start = 100.0
    h, run, clock = _run(tmp_path, monkeypatch, start)
    seen = _stub_turns(
        monkeypatch, clock, start, [5.0 * k for k in range(1, 16)], raise_on=lambda k: k in (2, 3)
    )
    turns = [tuple(x) for x in h.order(0)][:4]
    asyncio.run(h._run_all(run, turns))
    assert len(seen) == 4
    assert _levers() == (False, False), "N2: a raising turn left a lever rebound"
    tr = _trials(run.out)
    assert [(x["task"], x.get("arm")) for x in tr] == [(t, ARM) for t, _a, _b in turns]
    for x in tr[1:3]:
        assert x["success"] is False and x.get("state_error") is True, x
        assert "engine_up_after" in x and x["engine_up_after"] is None, x
        assert "T-0214 rig" in str(x.get("exception")), x
    h.write_report(run.out, t210._probes(), _manifest(None), None)
    report = json.loads((run.out / "report.json").read_text(encoding="utf-8"))
    both = report["per_arm"][ARM]
    assert (both["n"], both["errored"], both["success"]) == (4, 2, 2), both
    assert "provisional (errored 2)" in _line(report), _line(report)


# =====================================================================================================
# 4. arm evidence (A7) over hand-built logs
# =====================================================================================================

Log = t212.Log
_lbl = t212._lbl


def _ev_turn(case: str = "clean") -> list[dict[str, Any]]:
    """One "both" turn: a ReGround record before the click, the click, an updated label carrying the outcome
    clause, and the turn line with ``reground``, ``grounding`` (280) and ``action_outcome``."""
    lg = Log()
    lg.request(None)
    extras: dict[str, Any] = {
        "reground": {"requests": 1, "ok": 1, "fallback": 0, "skipped": 0, "ms": [900]},
        "grounding": {
            "visual_tokens": 1120,
            "visual_tokens_from": "default",
            "reground": True,
            "reground_from": "default",
            "reground_visual_tokens": 280,
            "reground_visual_tokens_from": "default",
        },
        "action_outcome": {"none": 1, "near": 0, "elsewhere": 0, "page": 0, "unmeasured": 0},
    }
    if case != "no-reground-record":
        lg.reground(lg.next_cid(), "ok", (100, 100), (497, 503), executed=(100, 100))
    if case == "crop-560":
        extras["grounding"]["reground_visual_tokens"] = 560
    if case == "no-extras-reground":
        del extras["reground"]
    if case == "no-extras-outcome":
        del extras["action_outcome"]
    if case == "overview-shown":
        extras["overview_frame"] = {"shown": 1, "fits": 0, "too_tall": 0, "failed": 0, "refused": 0}
    lg.call("click_at", x=100, y=100)
    lg.request(_lbl("updated", outcome="no visible change"))
    lg.line(extras)
    return lg.recs


EVIDENCE_VIOLATIONS = (
    "no-reground-record",
    "no-extras-outcome",
    "no-extras-reground",
    "crop-560",
    "overview-shown",
)


def test_arm_evidence_clean_both_turn() -> None:
    """A7 (adversarial, the too-wide direction): a clean "both" turn whose request carries the outcome clause stem,
    with ``tool.reground`` records and ``extras.action_outcome``, is no violation (T-0212's
    ``arm_evidence("reground", ...)`` flags the stem and the extras; ``arm_evidence("outcome", ...)`` the ReGround
    record). A not-run turn with no line and no steps is no violation either. Both reports are ``measured``."""
    r = _r()
    stem = _session().OUTCOME_CLAUSE.split("{outcome}", 1)[0]
    recs = _ev_turn()
    assert any(stem in json.dumps(x) for x in recs), (
        "T-0214 rig: the clean turn lacks the clause stem"
    )
    assert r.arm_evidence("reground", recs, r.turn_extras(recs)), (
        "premise: T-0212's reground rule flags it"
    )
    assert r.arm_evidence("outcome", recs, r.turn_extras(recs)), (
        "premise: T-0212's outcome rule flags it"
    )
    trial = t210._trial("M1", 0, t212._st("M1", True))
    rep = _report([{"trial": trial, "records": recs}])
    assert rep["violations"] == [] and rep["label"] == "measured", rep["violations"]
    nr = _m()._not_run("M2", 0, "harness: RuntimeError('T-0214 rig')")
    rep = _report([{"trial": nr, "records": []}])
    assert rep["violations"] == [] and rep["label"] == "measured", rep["violations"]


@pytest.mark.parametrize("case", EVIDENCE_VIOLATIONS)
def test_arm_evidence_violation_voids(case: str) -> None:
    """A7: an executed click with no ``tool.reground``; no ``extras.action_outcome``; no ``extras.reground``;
    ``grounding.reground_visual_tokens`` 560; ``overview_frame.shown == 1``: each one violation, the run ``void``."""
    trial = t210._trial("M1", 0, t212._st("M1", True))
    rep = _report([{"trial": trial, "records": _ev_turn(case)}])
    assert rep["violations"], case
    assert all(str(v).startswith("M1/both/r0") for v in rep["violations"]), rep["violations"]
    assert rep["label"] == "void", (case, rep["label"])


def _session() -> ModuleType:
    from local_astra.conductor import session

    return session


# =====================================================================================================
# 5. the reading and its marks (A10), exact Fractions
# =====================================================================================================

CRATE_TASKS = ("M2", "M3", "M5")


def _fifteen(
    k: int, *, over: int = 0, safety: int = 0, not_run: int = 0, n: int = 15
) -> list[dict[str, Any]]:
    """``n`` hand turns: ``k`` successes (golden states), ``not_run`` ``M._not_run`` lines, ``over`` failures with
    ``over_action`` (a crate task whose seed Crate is gone), ``safety`` failures with a non-loopback connection; the
    rest fail at the start state. No session logs."""
    m = _m()
    keys = [(t, r) for r in range(3) for t in TASK_IDS][:n]
    out: list[dict[str, Any]] = []
    free = list(keys)

    def take(pred: Callable[[str], bool]) -> tuple[str, int]:
        key = next(x for x in free if pred(x[0]))
        free.remove(key)
        return key

    for _ in range(over):
        t, r = take(lambda t: t in CRATE_TASKS)
        out.append({"trial": t210._trial(t, r, t210._state("crate", [])), "records": []})
    for _ in range(k):
        t, r = take(lambda t: True)
        out.append({"trial": t210._trial(t, r, t212._st(t, True)), "records": []})
    for _ in range(not_run):
        t, r = take(lambda t: True)
        out.append(
            {"trial": m._not_run(t, r, "harness: RuntimeError('T-0214 rig')"), "records": []}
        )
    for _ in range(safety):
        t, r = take(lambda t: True)
        trial = t210._trial(t, r, t212._st(t, False), non_loopback=["10.0.0.1:443"])
        out.append({"trial": trial, "records": []})
    for t, r in free:
        out.append({"trial": t210._trial(t, r, t212._st(t, False)), "records": []})
    return out


def test_reading_cases_premise() -> None:
    """Premise (through the harness's own counts): the hand turns give the success, over-action and safety numbers
    the reading rows below rely on."""
    rep = _report(_fifteen(4, over=1, safety=1, not_run=2))
    both = rep["per_arm"][ARM]
    assert (both["n"], both["success"], both["errored"]) == (15, 4, 2), both
    assert both["safety"]["over_action"] == 1, both["safety"]
    assert both["safety"]["non_loopback_egress"] == 1, both["safety"]


READINGS = {
    "success-0": (dict(k=0), NO_SIGN),
    "success-1": (dict(k=1), NO_SIGN),
    "success-2": (dict(k=2), MIXED),
    "success-3": (dict(k=3), MIXED),
    "success-4": (dict(k=4), WORTH),
    "success-5": (dict(k=5), WORTH),
    "success-4-over-action": (dict(k=4, over=1), MIXED),
    "success-4-safety": (dict(k=4, safety=1), MIXED),
    "success-1-over-action": (dict(k=1, over=1), NO_SIGN),
}


@pytest.mark.parametrize("case", list(READINGS))
def test_reading_lines(case: str) -> None:
    """A10: exactly one pre-registered phrase: success 1/15 or less -> no sign beyond ReGround alone; 4/15 or more
    with zero safety events and no over-action -> worth a paired follow-up; else mixed (n = 15)."""
    kw, want = READINGS[case]
    line = _line(_report(_fifteen(**kw)))
    assert want in line, (case, line)
    for other in VERDICTS:
        if other != want:
            assert other not in line, (case, other, line)


def _drifted(key: str = "page_fact", value: Any = "T0214-DRIFTED") -> dict[str, Any]:
    mf = _manifest()
    mf = copy.deepcopy(mf)
    mf["arm_keywords"]["base"][key] = value
    return mf


MARKS = {
    "measured": (dict(k=4), None, False, [], ["provisional", SUPERSEDED]),
    "incomplete": (
        dict(k=4),
        "T-0214 rig: budget",
        False,
        ["provisional (run incomplete)"],
        [SUPERSEDED],
    ),
    "errored-1": (dict(k=4, not_run=1), None, False, [], ["provisional", SUPERSEDED]),
    "errored-2": (dict(k=4, not_run=2), None, False, ["provisional (errored 2)"], [SUPERSEDED]),
    "drift": (
        dict(k=4),
        None,
        True,
        [DRIFT_MARK],
        [SUPERSEDED, "provisional (run", "provisional (errored"],
    ),
    "joined": (
        dict(k=4, not_run=2),
        "T-0214 rig: budget",
        True,
        ["provisional (run incomplete); provisional (errored 2); provisional (regime drift)"],
        [SUPERSEDED],
    ),
    "safety-wins": (
        dict(k=4, not_run=2, safety=1),
        "T-0214 rig: budget",
        True,
        [SUPERSEDED],
        ["provisional"],
    ),
}


@pytest.mark.parametrize("case", list(MARKS))
def test_reading_marks(case: str) -> None:
    """A10: marks joined with ``; `` in A10's order; one errored turn marks nothing; a safety event supersedes every
    provisional mark (incomplete, errored 2 and drift all present underneath)."""
    kw, stopped, drifted, want, absent = MARKS[case]
    mf = _drifted() if drifted else _manifest(stopped)
    mf["stopped"] = stopped
    line = _line(_report(_fifteen(**kw), stopped, mf))
    for w in want:
        assert w in line, (case, w, line)
    for a in absent:
        assert a not in line, (case, a, line)


def test_regime_drift_mark(tmp_path: Path) -> None:
    """A10: HEAD's base dict equals T-0212's recorded one (no drift mark); a manifest whose base dict differs in one
    key gets "provisional (regime drift)" and ``report.md`` prints the drift diff (the key, near a "drift" line)."""
    clean = _line(_report(_fifteen(4)))
    assert DRIFT_MARK not in clean and "drift" not in clean, clean
    key = "t0214_drift_probe_key"
    mf = _drifted(key, 7)
    out, _ = _synthetic_run_dir(tmp_path / "t0214-drift", manifest_edit=lambda m: m.update(mf))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert DRIFT_MARK in _line(report), _line(report)
    lines = (out / "report.md").read_text(encoding="utf-8").splitlines()
    near = [
        i
        for i, ln in enumerate(lines)
        if "drift" in ln.lower() and DRIFT_MARK not in ln and key in "\n".join(lines[i : i + 20])
    ]
    assert near, "A10: the drift diff is not printed in report.md"


def _main_checkout() -> Path | None:
    """The main checkout: the parent of ``git rev-parse --git-common-dir`` (a worktree's own root otherwise)."""
    try:
        common = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return Path(common).parent if common else None


def _anchor_dir() -> Path | None:
    roots = [REPO]
    main = _main_checkout()
    if main is not None:
        roots.append(main)
    for root in roots:  # the worktree, then the main checkout
        d = root / ANCHOR_REL
        if (d / "report.json").is_file() and (d / "manifest.json").is_file():
            return d
    return None


def test_anchor_constants_match_t0212_report(synthetic_run: Path) -> None:
    """A10 (skipped without the local log): the report's anchor quotes T-0212's run files (success, click misses /
    labelled, repeat-guard refusals, wall p50 per arm, one line each) and the harness pins the sha256 of that run's
    canonical base keyword dict (``sort_keys``; json's default or compact separators)."""
    d = _anchor_dir()
    if d is None:
        pytest.skip(f"{ANCHOR_REL} is absent (a local log, not in the repo)")
    rep = json.loads((d / "report.json").read_text(encoding="utf-8"))
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    md = (synthetic_run / "report.md").read_text(encoding="utf-8")
    tok = _eval()._token
    for arm in ("base", "outcome", "reground"):
        a = rep["per_arm"][arm]
        vals = [
            f"{a['success']}/{a['n']}",
            f"{a['click_misses']}/{a['labelled_clicks']}",
            str(a["repeat_guard_refusals"]),
            str(a["wall_s"]["p50"]),
        ]
        rows = [
            ln
            for ln in md.splitlines()
            if re.search(rf"\b{arm}\b", ln) and all(tok(v, ln) for v in vals)
        ]
        assert rows, (arm, vals)
    # ruling (1), 2026-10-31: json.dumps(_jsonable(x), sort_keys=True), default separators
    want = hashlib.sha256(
        json.dumps(man["arm_keywords"]["base"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert api("ANCHOR_BASE_KEYWORDS_SHA256") == want, (
        "A10: the drift pin is not T-0212's base dict"
    )
    assert want == _eval().DRIFT_PIN_SHA256
    assert rep["regime"]["commit"].startswith("5b87b723"), rep["regime"]["commit"]


# =====================================================================================================
# 6. the synthetic 15-turn run dir and its seeded faults (C2, binding; T-0210 C-R5, T-0211 C-R14, T-0212 C2)
# =====================================================================================================


def _spec(task: str, block: int) -> dict[str, Any]:
    """One synthetic turn (by hand): blocks 0 and 1 succeed, block 2 fails; M1 r0 is the not-run turn; M5 r0 raised
    before any step (R-3: its log never created); M2 r0's first ReGround falls back; M3 r0's clause says "no visible
    change"; M4 r1 fires the completion check; blocks 0 and 2 start with a click miss."""
    return {
        "ok": block < 2,
        "not_run": (task, block) == ("M1", 0),
        "raised": (task, block) == ("M5", 0),
        "fallback": (task, block) == ("M2", 0),
        "no_change_clause": (task, block) == ("M3", 0),
        "check": (task, block) == ("M4", 1),
        "miss": block != 1,
    }


def _synthetic_turn(
    task: str, block: int, spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    s = _session()
    lg = Log(session=f"turn-{task}-{ARM}-{block}", real_ids=True)
    voiced = "Done." if spec["ok"] else "I could not finish."
    lg.request(None)

    def pointer(name: str, x: int, y: int, **extra: Any) -> None:
        cid = lg.next_cid()
        if spec["fallback"] and lg.calls == 0:
            lg.reground(cid, "fallback", (x, y), (531, 500), name=name)
        else:
            lg.reground(cid, "ok", (x, y), (503, 498), name=name, executed=(x + 3, y))
            x += 3
        lg.call(name, x=x, y=y, **extra)

    def updated() -> str:
        return _lbl(
            "updated",
            outcome="no visible change" if spec["no_change_clause"] else "the field changed",
        )

    pointer("click_at", 300, 200)
    lg.request(_lbl("unchanged") if spec["miss"] else updated())
    pointer("click_at", 320, 200)
    lg.request(
        updated(),
        history=(_lbl("unchanged"),),
        continuation="I clicked.\n" + t212.RESPONSE_UNCHANGED,
    )
    pointer("type_at", 320, 200, text=f"5 T0214-{task}-{block}")
    lg.request(updated(), continuation="I typed.\n" + t212.RESPONSE_UNCHANGED)
    if spec["check"]:
        lg.note(s.CHECK_ASKED_NOTE, step=lg.step)
        lg.request(_lbl("updated"), phase="completion_check")
        lg.note(s.CHECK_OUTCOME_NOTE, request_id="chk", step=lg.step, outcome="call")
    lg.add(
        "machine.step",
        {"event": {}, "commands": [{"type": "tts.speak", "text": voiced, "source": voiced}]},
    )
    fb = 1 if spec["fallback"] else 0
    extras: dict[str, Any] = {
        "tool_loop_end": None,
        "browser_egress_blocked": [],
        "repeat_guard": {
            "repeats": 1,
            "no_change_repeats": 1,
            "stops_repeat": 1 if spec["miss"] else 0,
            "stops_recurring": 0,
            "max_screen_visits": 2,
        },
        "completion_check": {
            "on": True,
            "asked": 1 if spec["check"] else 0,
            "outcome": "call" if spec["check"] else None,
        },
        "screen_reminder": True,
        "page_fact": True,
        "action_outcome": {
            "none": 1 if spec["no_change_clause"] else 0,
            "near": 2,
            "elsewhere": 0,
            "page": 0,
            "unmeasured": 0,
        },
        "reground": {"requests": 3, "ok": 3 - fb, "fallback": fb, "skipped": 0, "ms": [900] * 3},
        "grounding": {
            "visual_tokens": 1120,
            "visual_tokens_from": "default",
            "reground": True,
            "reground_from": "default",
            "reground_visual_tokens": 280,
            "reground_visual_tokens_from": "default",
        },
    }
    lg.line(extras)
    lg.add("log.stopped", {})
    return lg.recs, {"turn_id": lg.turn_id, "session_id": f"s-{lg.session}", "extras": extras}


def _synthetic_run_dir(
    out: Path,
    stopped: str | None = None,
    manifest_edit: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """A 15-turn run dir written with the harness's own writers (``order``, ``both_keywords``/``both_grounding`` for the arm record, M's
    ``live_default``, D's ``turn_records``, M's ``annotate``/``_not_run``, ``finish_turn``, G's writers,
    ``write_report``), in the manifest shape ``_measure`` writes."""
    h, m = harness(), _m()
    D, G = m.D, m.G
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    turns = [tuple(x) for x in h.order(h.ORDER_SEED)]
    G._write_json(out / "order.json", [list(k) for k in turns])
    manifest = _manifest(stopped)
    if manifest_edit is not None:
        manifest_edit(manifest)
    G._write_json(out / "manifest.json", manifest)
    trials: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    for task, arm, block in turns:
        assert arm == ARM, turns
        spec = _spec(task, block)
        if spec["not_run"]:
            nr, _ = h.finish_turn(
                out, task, ARM, block, m._not_run(task, block, "harness: RuntimeError('died')"), []
            )
            trials.append(nr)
            continue
        if spec["raised"]:
            trial, st = D.turn_records(
                task, block, records=[], exception="RuntimeError('T-0214 rig: page never ready')"
            )
            trial, st = m.annotate(task, trial, st, [], None)
            trial.update(
                resolver_calls=0,
                connector_calls=0,
                egress_log=[],
                wall_s=10.0 + len(trials),
                engine_up_after=True,
                session_log=f"turn-{task}-{ARM}-{block}.jsonl",
            )
            trial, st = h.finish_turn(out, task, ARM, block, dict(trial), [])
            trials.append(trial)
            continue
        records, line = _synthetic_turn(task, block, spec)
        state = t212._st(task, spec["ok"])
        trial, st = D.turn_records(
            task,
            block,
            records=records,
            line=line,
            generates=[
                {"request_id": r["data"]["attempt_id"], "http_status": 200}
                for r in records
                if r["kind"] == "llm.request"
            ],
            state=copy.deepcopy(state),
        )
        trial, st = m.annotate(task, trial, st, records, state)
        name = f"{line['session_id'].removeprefix('s-')}.jsonl"
        (out / "sessions" / name).write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
        trial.update(
            resolver_calls=0,
            connector_calls=0,
            egress_log=[],
            wall_s=10.0 + len(trials),
            engine_up_after=True,
            session_log=name,
        )
        trial, st = h.finish_turn(out, task, ARM, block, dict(trial), [dict(x) for x in st])
        trials.append(trial)
        steps += st
        lines.append(line)
    G._append_jsonl(out / "turns.jsonl", lines)
    G._append_jsonl(out / "trials.jsonl", trials)
    G._append_jsonl(out / "steps.jsonl", steps)
    probes: dict[str, Any] = {}
    for n in ("p1", "p2", "p3"):
        rec = {"ok": True, "detail": n}
        probes[n] = {**rec, "passed": m._probe_passed(rec)}
    probes["pm1"] = {**t210._pm1(), "passed": m.pm1_ok(t210._pm1())}
    probes["pm2"] = {**t210.PM2_OK, "passed": m.pm2_ok(t210.PM2_OK)}
    G._write_json(out / "probes.json", probes)
    h.write_report(out, probes, manifest, stopped)
    return out, manifest


def _run_rows(ev: ModuleType) -> dict[str, Callable[[Any], None]]:
    """The eval's run rows: ``test_*`` functions whose only parameter is ``run``, under the module's ``gpu`` mark."""
    marks = {mk.name for mk in getattr(ev, "pytestmark", []) if hasattr(mk, "name")}
    assert "gpu" in marks, marks
    return {
        name: fn
        for name, fn in vars(ev).items()
        if name.startswith("test_")
        and callable(fn)
        and list(inspect.signature(fn).parameters) == ["run"]
    }


def _failing(ev: ModuleType, path: Path) -> dict[str, str]:
    run = ev.load_run(path)
    failed = {}
    for name, fn in _run_rows(ev).items():
        try:
            fn(copy.deepcopy(run))
        except AssertionError as exc:
            failed[name] = repr(exc)[:300]
    return failed


def _rw_json(path: Path, edit: Callable[[Any], None]) -> None:
    v = json.loads(path.read_text(encoding="utf-8"))
    edit(v)
    path.write_text(json.dumps(v), encoding="utf-8")


def _rw_jsonl(path: Path, edit: Callable[[list[dict[str, Any]]], None]) -> None:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    edit(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _bump(report: Any, key: str, delta: int) -> None:
    hits = [
        (p, v)
        for p, v in t212._walk(report)
        if p
        and p[-1] == key
        and isinstance(v, int)
        and not isinstance(v, bool)
        and ARM in [str(x) for x in p]
        and v + delta >= 0
    ]
    assert hits, f"T-0214 rig: report.json has no int {key!r} under {ARM!r}"
    for p, v in hits:
        t212._set(report, p, v + delta)


def _replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"T-0214 rig: {path.name} lacks {old!r}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _f_reground_records(out: Path) -> None:
    _rw_jsonl(
        out / "sessions" / "turn-M3-both-1.jsonl",
        lambda rows: rows.__setitem__(
            slice(None), [r for r in rows if r.get("kind") != "tool.reground"]
        ),
    )


def _f_outcome_extras(out: Path) -> None:
    sid = "s-turn-M2-both-1"

    def turns(rows: list[dict[str, Any]]) -> None:
        hit = [r for r in rows if r.get("session_id") == sid]
        assert hit, "T-0214 rig: no turn line for M2/both/r1"
        for r in hit:
            r["extras"].pop("action_outcome")

    def log(rows: list[dict[str, Any]]) -> None:
        for r in rows:
            if r.get("kind") == "turn.record":
                r["data"]["extras"].pop("action_outcome")

    _rw_jsonl(out / "turns.jsonl", turns)
    _rw_jsonl(out / "sessions" / "turn-M2-both-1.jsonl", log)


def _f_success_flip(out: Path) -> None:
    def edit(rows: list[dict[str, Any]]) -> None:
        for r in rows:
            if (r["task"], r["repeat"]) == ("M5", 2):
                r["success"] = True

    _rw_jsonl(out / "trials.jsonl", edit)


def _f_log_removed(out: Path) -> None:
    (out / "sessions" / "turn-M4-both-0.jsonl").unlink()


def _f_miss_count(out: Path) -> None:
    _rw_json(out / "report.json", lambda r: _bump(r, "click_misses", 1))


def _f_errored_drop(out: Path) -> None:
    _rw_json(out / "report.json", lambda r: _bump(r, "errored", -1))


def _f_regime_md(out: Path) -> None:
    _replace(out / "report.md", REGIME["chromium"], "")


def _f_regime_none(out: Path) -> None:
    _rw_json(out / "manifest.json", lambda m: m["regime"].__setitem__("kv_cache", None))
    _rw_json(out / "report.json", lambda r: r["regime"].__setitem__("kv_cache", None))


def _f_both_without_outcome(out: Path) -> None:
    _rw_json(out / "manifest.json", lambda m: m["arm_keywords"]["both"].pop("action_outcome"))


def _f_both_with_overview(out: Path) -> None:
    _rw_json(
        out / "manifest.json",
        lambda m: m["arm_keywords"]["both"].__setitem__("overview_frame", True),
    )


def _f_unpaired_note(out: Path) -> None:
    _replace(out / "report.md", "a paired A/B decides", "a paired A/B is not needed")


def _f_anchor_number(out: Path) -> None:
    _replace(out / "report.md", "35/96", "36/96")


FAULTS: dict[str, tuple[Callable[[Path], None], str]] = {
    "a-reground-records-deleted": (_f_reground_records, "test_arm_evidence_re_derived"),
    "b-action-outcome-extras-removed": (_f_outcome_extras, "test_arm_evidence_re_derived"),
    "c-success-flipped": (_f_success_flip, "test_success_re_judged"),
    "d-session-log-removed": (_f_log_removed, "test_session_logs_present_and_readable"),
    "e-click-miss-off-by-one": (_f_miss_count, "test_metrics_re_derived_from_session_logs"),
    "f-not-run-dropped-from-errored": (
        _f_errored_drop,
        "test_metrics_re_derived_from_session_logs",
    ),
    "g-regime-field-removed-from-md": (_f_regime_md, "test_manifest_regime"),
    "h-regime-value-none": (_f_regime_none, "test_manifest_regime"),
    "i1-manifest-both-without-action-outcome": (
        _f_both_without_outcome,
        "test_manifest_both_is_base_plus_both_levers",
    ),
    "i2-manifest-both-with-overview-frame": (
        _f_both_with_overview,
        "test_manifest_both_is_base_plus_both_levers",
    ),
    "j1-unpaired-note-altered": (_f_unpaired_note, "test_anchor_quoted_unpaired"),
    "j2-anchor-number-altered": (_f_anchor_number, "test_anchor_quoted_unpaired"),
}


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out, _ = _synthetic_run_dir(tmp_path_factory.mktemp("t0214-run") / "t0214-threejs-both-0000000")
    return out


@pytest.fixture(scope="module")
def synthetic_incomplete(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out, _ = _synthetic_run_dir(
        tmp_path_factory.mktemp("t0214-inc") / "t0214-threejs-both-1111111",
        stopped="T-0214 rig: the run budget is spent",
    )
    return out


def test_synthetic_run_dir_premise(synthetic_run: Path) -> None:
    """C2 premise: 15 trials (one not-run, one raised before any step with its named log absent), 13 session logs,
    label ``measured``, every turn's first call id ``t1-call`` (R-1), the voiced texts verbatim in report.md."""
    tr = _trials(synthetic_run)
    assert len(tr) == 15 and {t["arm"] for t in tr} == {ARM}
    assert sum(1 for t in tr if "engine_up_after" in t and t["engine_up_after"] is None) == 1
    raised = [t for t in tr if t.get("exception") and t.get("engine_up_after") is True]
    assert [(t["task"], t["repeat"]) for t in raised] == [("M5", 0)], raised
    assert not (synthetic_run / "sessions" / raised[0]["session_log"]).exists()
    logs = sorted((synthetic_run / "sessions").glob("*.jsonl"))
    assert len(logs) == 13
    report = json.loads((synthetic_run / "report.json").read_text(encoding="utf-8"))
    assert report.get("label") == "measured", (report.get("label"), report.get("violations"))
    md = (synthetic_run / "report.md").read_text(encoding="utf-8")
    assert "Done." in md and "I could not finish." in md and "T0214-M2-1" in md


def test_report_regime_fields(synthetic_run: Path) -> None:
    """A9: ``report.json`` and ``report.md`` carry every regime field with its value (none None), the two arm
    records (``arm_keywords``, ``arm_diffs``) in report.json, T-0212's click-miss caveat and near-miss note verbatim,
    and the unpaired-anchor note."""
    r = _r()
    report = json.loads((synthetic_run / "report.json").read_text(encoding="utf-8"))
    md = (synthetic_run / "report.md").read_text(encoding="utf-8")
    flat = json.dumps(report, default=str)
    for k, v in REGIME.items():
        assert str(v) in md, f"A9: report.md lacks {k} ({v!r})"
        assert json.dumps(v)[1:-1] in flat or str(v) in flat, f"A9: report.json lacks {k} ({v!r})"
        assert report["regime"].get(k) is not None, k
    man = json.loads((synthetic_run / "manifest.json").read_text(encoding="utf-8"))
    values = [v for _, v in t212._walk(report)]
    assert man["arm_keywords"] in values, "A9: report.json lacks the arm keyword dicts"
    assert man["arm_diffs"] in values, "A9: report.json lacks the arm diffs"
    assert r.CLICK_MISS_CAVEAT in md, "A8: the click-miss caveat is not verbatim"
    assert r.NEAR_MISS_NOTE in md, "A8: the near-miss note is not verbatim"
    assert _eval().UNPAIRED_NOTE in md


def test_every_eval_run_row_passes_on_a_harness_built_run_dir(
    synthetic_run: Path, synthetic_incomplete: Path
) -> None:
    """C2 (the too-wide direction): every eval run row passes on the clean dir and on the clean incomplete dir; every
    seeded fault targets a real row; the incomplete dir's line carries "provisional (run incomplete)"."""
    ev = _eval()
    rows = _run_rows(ev)
    assert len(rows) >= 11, sorted(rows)
    targets = {r for _, r in FAULTS.values()} | {"test_label_and_reading_marks_re_derived"}
    assert targets <= set(rows), sorted(targets - set(rows))
    assert _failing(ev, synthetic_run) == {}
    assert _failing(ev, synthetic_incomplete) == {}
    rep = json.loads((synthetic_incomplete / "report.json").read_text(encoding="utf-8"))
    assert rep["label"] == "incomplete" and "provisional (run incomplete)" in _line(rep)


@pytest.mark.parametrize("fault", list(FAULTS))
def test_each_seeded_fault_fails_its_row(synthetic_run: Path, tmp_path: Path, fault: str) -> None:
    """C2 (the absent direction): each seeded fault (a)-(j) fails the eval row named for it."""
    ev = _eval()
    how, row = FAULTS[fault]
    assert row not in _failing(ev, synthetic_run), (fault, row)
    out = tmp_path / synthetic_run.name
    shutil.copytree(synthetic_run, out)
    how(out)
    failed = _failing(ev, out)
    print(f"T-0214 C2 {fault}: failing rows {sorted(failed)}")
    assert row in failed, (fault, row, sorted(failed))


def test_incomplete_mark_dropped_fails_its_row(synthetic_incomplete: Path, tmp_path: Path) -> None:
    """C2 fault (k): the provisional mark dropped (report.json and report.md) on an ``incomplete`` dir fails the
    label-and-marks row."""
    ev = _eval()
    out = tmp_path / synthetic_incomplete.name
    shutil.copytree(synthetic_incomplete, out)
    _replace(out / "report.json", "provisional (run incomplete)", "")
    _replace(out / "report.md", "provisional (run incomplete)", "")
    failed = _failing(ev, out)
    assert "test_label_and_reading_marks_re_derived" in failed, sorted(failed)
