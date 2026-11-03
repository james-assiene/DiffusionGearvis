"""T-0212 group A: the three-arm three.js A/B harness ``scripts/gpu/t0212_threejs_arms.py`` (unit; CPU; no engine,
no socket). docs/tasks/T-0212.md A1-A11, N1, N2 and "Tests to write" (``tests/unit/test_t0212_arms_harness.py``).

BASE d29b1555 (main; ``Merge T-0210-c:`` fec43cc3 is an ancestor). Every row that needs the harness carried
``pending(task="T-0212")`` until ``/implement`` (all activated 2026-10-31); a missing file is a named assertion
("does not exist yet") from :func:`harness`, never an import error. Premise rows (cli's constants, T-0205/T-0210 helpers, the hand states under the real judge, the frozen
T-0210 C / T-0205 C files) are active and green at BASE.

**The harness surface (reconciled 2026-10-31 with ``scripts/gpu/t0212_threejs_arms.py`` at dfa98a73).** The rows
were first written against an assumed surface; the names below are the harness's own. Where a row needs a shape the
harness spells differently, a thin adapter in this module reshapes the call (``_completion_counts``, ``_summarise``,
``_reading``, ``_mark``, ...). An adapter only reshapes data or composes harness calls; it never computes a metric.

Constants
    ``ARMS``; ``ORDER_SEED = 0``; ``HARD_LIMIT_S``, ``RUN_BUDGET_S``, ``TURN_WORST_S``, ``REPORT_S``; ``EXIT_OK,
    EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3`` (``EXIT_PROBE`` is "void"); ``PROCESS_START_S``; ``M``.

Arms (A2, A3)
    ``arm_context(arm)``; ``arm_keywords(profile, arm)``; ``pa_check(profile) -> {"keywords", "diffs", "grounding",
    "tool_guidance_sha256", "problems"}`` (P-A holds iff ``problems`` is empty; it is also the manifest's record).

Order and run (A5, A6)
    ``order(seed) -> [(task, arm, r)]``; ``async _run_all(run, turns)``; ``finish_turn(out, task, arm, repeat, trial,
    steps) -> (trial, steps)`` (the lines as ``_run_all`` writes them; it reads the turn's session log from
    ``out/sessions``). The trial's ``repeat`` is the block ``r``.

Metric functions (A7, A8), over one turn's session-log records
    ``step_labels(records) -> [{"seq", "step", "key"}]``; ``unchanged_labels``; ``click_misses -> {"clicks",
    "labelled", "misses", "unlabelled"}``; ``reaims``; ``reground_stats -> {"outcomes", "requests", "shifts", "ms",
    "split": {"moved", "not_moved"}}``; ``completion_asked(extras)``, ``completion_outcomes(records)``;
    ``repeat_guard_refusals(extras)``; ``turn_extras(records)`` (the session log's ``turn.record`` extras);
    ``arm_evidence(arm, records, extras)``.

Report (A8-A10)
    ``build_report(trials, steps, records_by_key, probes, stopped, manifest)`` (keys ``label``, ``per_arm``,
    ``per_task``, ``violations``, ``readings[lever] = {reading, mark, confound, line}``,
    ``overview_frame_base_outcome_turns``, ``regime``); ``arm_summary``; ``reading(lever, arm, base, arm_tasks,
    base_tasks)``; ``reading_mark(label, errored, safety)``, ``reading_line``; ``report_label(probe_void, violations,
    stopped)``; ``write_report(out, probes, manifest, stopped)``; ``REGIME_FIELDS``.

Seams ``main`` reads at call time
    ``_git`` (the harness's, M's and D's; the rig answers ``status`` and ``log`` and delegates ``show`` to the real
    repo, so the A1 byte pin reads the real blobs), ``cli.*`` resolvers and constants, ``cli.load_tool_kit``,
    ``cli.live_prerequisites``, ``PROVENANCE_PATH`` and ``threejs_tasks.TASKS``.

**Readings** (reported to the orchestrator; not resolved silently)
    1. A8's labelled request: an ``llm.request`` with ``phase == "tool_step"`` that places its own label
       (``live_label``); the report step's request is one (it is a ``tool_step`` carrying ``report_fact``).
       ``generate`` (step 1), ``reground``, ``completion_check``, ``announce_reask`` and ``tool_repair`` requests are
       not. One label per step (A8 "exact definition, one per step"): a resend of the same step (an engine-empty or
       degenerate retry logs the step's request again) is still that step's one label, and a ``tool_step`` request
       that places no label of its own has none, even when its history repeats an earlier label ("text repeated from
       earlier steps ... is never counted again"). The last two are (c) rows against the harness.
    2. A8's re-aim: "next executed pointer action" counts ``click_at`` and ``type_at`` (both carry a point); a
       ``type_at`` between a miss and a near click breaks the pair. ``scroll`` is not a pointer action here.
    3. A8's reground split: "pass 2 moved the point" means an ``ok`` record whose shift is > 0; ``fallback``,
       ``pass2_timeout``, ``skipped_no_point``, ``error`` and an ``ok`` with shift 0 are "did not".
    4. ``order``'s pre-shuffle cell list is task-major then ``ARMS`` order; A5 fixes only "shuffled in turn with one
       ``random.Random(0)``".
    5. A9's marks: safety beats every provisional mark (the task says so for incomplete); a run-label mark and an
       errored mark may both appear. "When any safety count is non-zero" is read over the whole run (any arm), and
       A10 (iv) "zero safety events" on the lever arm: the task is silent on whose, so the harness's reading.
    6. The pass-2 ms p50/max population is every record with an ``ms`` (harness and eval agree).
    7. Errored turn (A6/A8; the task fixes only that a not-run turn is one): not run, an engine error
       (``M.engine_error``) or a state error (``M.state_error``). Task silent beyond that: the harness's reading;
       the eval counted a raised exception instead and now follows the harness.
    8. Safety events: ``D.SAFETY``'s seven sums plus ``secret_typed``, ``checkout_pressed``, ``submit_pressed`` and
       ``file_ops`` (harness ``SAFETY_EVENTS``; the eval's reading 6 is the same set). ``over_action`` is A10
       (iii)'s; traps, malformed states and state errors are reported, not safety events.
    9. ``order.json`` is written right after the out-dir is made, before the manifest and the probes (A5,
       task-literal; the harness does this).

Regime: CPU, pure, hand-built records and stubbed ``_one_turn``/``main``; no engine; n = 1.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import copy
import dataclasses
import functools
import importlib.util
import inspect
import json
import random
import re
import shutil
import socket
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests import _git_range as gr
from tests.unit import test_t0131_ab_harness as t31
from tests.unit import test_t0205_demo_harness as t205
from tests.unit import test_t0210_threejs_harness as t210

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0212_threejs_arms.py"
EVAL = REPO / "tests" / "eval" / "test_t0212_threejs_arms.py"
T0210_HARNESS = REPO / "scripts" / "gpu" / "t0210_threejs_run.py"
T0205_HARNESS = REPO / "scripts" / "gpu" / "t0205_demo_run.py"
#: ``git log --merges``: fec43cc3 "Merge T-0210-c: three.js editor M1-M5 GPU harness ..." (the pin for A1/N1)
MERGE_T0210C_SHA = "fec43cc32a9fbb8e6edaf0235fdd9dd604a72f04"
MERGE_C = (
    f"{MERGE_T0210C_SHA} Merge T-0210-c: three.js editor M1-M5 GPU harness (report-only, 15 turns)"
)
#: the T-0210 C run dir (a local log, not in the repo; A11)
ANCHOR_REL = Path("logs") / "t0210-threejs-47c2610"

TASK_IDS = ("M1", "M2", "M3", "M4", "M5")
ARMS = ("base", "outcome", "reground")


# =====================================================================================================
# helpers
# =====================================================================================================


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0212 A1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0212_threejs_arms_unit", HARNESS)
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
    assert hasattr(h, name), f"T-0212: {HARNESS.name} has no {name!r} (see this module's docstring)"
    return getattr(h, name)


def _m() -> ModuleType:
    m = api("M")
    assert isinstance(m, ModuleType) and str(m.__file__).endswith("t0210_threejs_run.py"), m
    return m


def _held(h: ModuleType) -> list[ModuleType]:
    """The harness and every ``scripts/gpu`` module it holds by name, three levels down (M, D and D's chain)."""
    seen: dict[int, ModuleType] = {id(h): h}
    frontier = [h]
    for _ in range(3):
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


# ---- adapters to the harness's surface (reshape only; every count is the harness's own) ------------------------


def _request_labels(records: list[dict[str, Any]]) -> list[str | None]:
    return [row["key"] for row in api("step_labels")(records)]


def _evidence(arm: str, records: list[dict[str, Any]]) -> list[str]:
    return list(api("arm_evidence")(arm, records, api("turn_extras")(records)))


def _completion_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    extras = api("turn_extras")(records) or {}
    outcomes = list(api("completion_outcomes")(records))
    return {
        "asked": int(api("completion_asked")(extras)),
        "call": outcomes.count("call"),
        "no_call": outcomes.count("no_call"),
    }


def _report_over(turns: list[dict[str, Any]], stopped: str | None = None) -> dict[str, Any]:
    """``build_report`` over hand turns ``{"arm", "trial", "records"}`` (each trial gets its arm; good probes)."""
    trials, by_key = [], {}
    for t in turns:
        trial = {**t["trial"], "arm": t["arm"]}
        trials.append(trial)
        by_key[(str(trial["task"]), t["arm"], int(trial.get("repeat") or 0))] = t["records"]
    return dict(api("build_report")(trials, [], by_key, t210._probes(), stopped, None))


def _summarise(turns: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    per_arm = _report_over(turns)["per_arm"]
    return {
        arm: {
            **a,
            "completion_fired": a["completion_check"]["fired_turns"],
            "success_after_fire": a["completion_check"]["success_after_fire"],
        }
        for arm, a in per_arm.items()
    }


def _mark(line: str, label: str, errored: int, safety: int) -> str:
    lever, text = line.split(": ", 1)
    return str(api("reading_line")(lever, text, api("reading_mark")(label, errored, safety)))


def _summary(flat: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """A hand metrics dict (the rows' flat keys) in ``arm_summary``'s shape, and its per-task map."""
    h = harness()
    miss = Fraction(flat["misses"], flat["labelled"]) if flat["labelled"] else None
    ok = Fraction(flat["pass2_ok"], flat["pass2_requests"]) if flat["pass2_requests"] else None
    safety = {name: 0 for name in h.SAFETY_EVENTS}
    safety[h.SAFETY_EVENTS[0]] = int(flat["safety"])
    safety["over_action"] = int(flat["over_action"])
    summary = {
        "success": flat["success"],
        "miss_rate": None if miss is None else f"{miss.numerator}/{miss.denominator}",
        "repeat_guard_refusals": flat["repeat_guard"],
        "safety": safety,
        "wall_s": {"p50": flat["wall_p50"]},
        "reground": {"ok_rate": None if ok is None else f"{ok.numerator}/{ok.denominator}"},
    }
    return summary, {t: {"success": k} for t, k in flat["per_task"].items()}


def _reading(lever: str, arm: dict[str, Any], base: dict[str, Any]) -> str:
    a, a_tasks = _summary(arm)
    b, b_tasks = _summary(base)
    return str(api("reading")(lever, a, b, a_tasks, b_tasks))


def test_heads_levers_off_premise() -> None:
    """A4/N2 premise: at BASE both cli constants are ``False`` (the arms would collapse otherwise), and rebinding
    them on cli moves ``D.session_keywords`` (cli's resolvers read them at call time), so A3's diffs are reachable."""
    from local_astra import cli

    assert _levers() == (False, False)
    m = t210.harness()  # T-0210 C's harness: a premise, read-only
    lean = _lean()
    base = m.D.session_keywords(lean)
    assert base.get("reground") is False and base.get("overview_frame") is True, base
    assert "action_outcome" not in base
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(cli, "LIVE_DEFAULT_ACTION_OUTCOME", True)
        assert m.D.session_keywords(lean).get("action_outcome") is True
    finally:
        mp.undo()
    assert _levers() == (False, False)


# =====================================================================================================
# 1. arm isolation (A2, A3, N2)
# =====================================================================================================


def test_arms_constant() -> None:
    """A2: ``ARMS`` exactly."""
    assert tuple(api("ARMS")) == ARMS


def test_arm_keywords_diffs() -> None:
    """A3 (hand diffs): base equals ``D.session_keywords(profile)`` outside any context; outcome adds only
    ``action_outcome: True``; reground flips ``reground`` False -> True, drops ``overview_frame`` (cli's H-T152-1
    yield) and leaves every other key equal, ``tool_guidance`` byte-equal. The cli constants are back to False."""
    h = harness()
    lean = _lean()
    outside = _m().D.session_keywords(lean)
    kw = api("arm_keywords")
    b, o, r = dict(kw(lean, "base")), dict(kw(lean, "outcome")), dict(kw(lean, "reground"))
    assert b == outside, sorted(set(b) ^ set(outside))
    assert set(o) - set(b) == {"action_outcome"} and set(b) <= set(o), sorted(set(o) ^ set(b))
    assert o["action_outcome"] is True
    assert {k: o[k] for k in b} == b
    assert set(b) - set(r) == {"overview_frame"}, sorted(set(b) ^ set(r))
    assert set(r) - set(b) == set(), sorted(set(r) - set(b))
    assert (b["reground"], r["reground"]) == (False, True)
    assert [k for k in r if k != "reground" and r[k] != b[k]] == []
    assert (
        isinstance(r["tool_guidance"], str)
        and r["tool_guidance"].encode() == b["tool_guidance"].encode()
    )
    assert "action_outcome" not in r
    assert _levers() == (False, False), "A2: arm_keywords left a lever rebound"
    del h


def test_reground_crop_is_280_default() -> None:
    """A3: ``cli.grounding_of(profile, D._live_args(), "live")`` under the reground arm: ReGround on, pass-2 budget
    280 from ``"default"`` (T-0133 A5), main budget 1120; under base and outcome ReGround is off."""
    from local_astra import cli

    ctx = api("arm_context")
    lean = _lean()
    ns = _m().D._live_args()
    with ctx("reground"):
        g = cli.grounding_of(lean, ns, "live")
    assert (g.reground, g.reground_visual_tokens, g.reground_visual_tokens_from) == (
        True,
        280,
        "default",
    ), g
    assert g.visual_tokens == 1120, g
    for arm in ("base", "outcome"):
        with ctx(arm):
            g = cli.grounding_of(lean, ns, "live")
        assert g.reground is False, (arm, g)
    assert _levers() == (False, False)


@pytest.mark.parametrize("arm", ARMS)
def test_arm_context_restores_on_exception(arm: str) -> None:
    """A2/N2 (adversarial): inside the context exactly the arm's lever is True; a raise inside it leaves both cli
    constants False afterwards; a nested ``arm_context`` is refused and the outer rebinding survives the refusal."""
    ctx = api("arm_context")
    want = {"base": (False, False), "outcome": (True, False), "reground": (False, True)}[arm]
    seen: list[tuple[Any, Any]] = []
    with pytest.raises(RuntimeError, match="T-0212 rig"), ctx(arm):
        seen.append(_levers())
        raise RuntimeError("T-0212 rig: the turn raised")
    assert seen == [want], seen
    assert _levers() == (False, False), f"A2: {arm} left {_levers()} after a raise"
    with ctx(arm):
        for inner in ARMS:
            with pytest.raises(Exception), ctx(inner):  # noqa: B017 - the task names no type
                pass
            assert _levers() == want, (arm, inner, _levers())
    assert _levers() == (False, False)


def test_arm_context_refuses_an_unknown_arm() -> None:
    """A2 (the too-wide direction's partner): an arm that is not in ``ARMS`` is refused and rebinds nothing."""
    ctx = api("arm_context")
    with pytest.raises(Exception), ctx("both"):  # noqa: B017
        pass
    assert _levers() == (False, False)


def test_arm_context_never_assigns_runner_attributes() -> None:
    """A1: ``vars(M)``, ``vars(D)`` and ``vars(D.G)`` are identical (by identity) before and after each arm's
    context and after ``arm_keywords``."""
    m = _m()
    mods = (m, m.D, m.D.G)
    before = [dict(vars(x)) for x in mods]
    for arm in ARMS:
        with api("arm_context")(arm):
            pass
        api("arm_keywords")(_lean(), arm)
    for x, snap in zip(mods, before, strict=True):
        now = vars(x)
        changed = sorted(k for k in set(snap) | set(now) if snap.get(k, ...) is not now.get(k, ...))
        assert changed == [], f"A1: {x.__name__} attributes changed: {changed}"


def test_harness_never_writes_into_an_imported_runner() -> None:
    """A1 (AST): the harness loads T-0210 C's harness by path and never assigns an attribute on a name bound to a
    loaded runner (``M``, ``D``, ``G``, ``T``, ``F``, ``P``)."""
    harness()
    text = HARNESS.read_text(encoding="utf-8")
    assert "t0210_threejs_run.py" in text, (
        "A1: the harness does not load T-0210 C's harness by path"
    )
    tree = ast.parse(text)
    bad = [
        ast.unparse(t)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign)
        for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(t, ast.Attribute)
        and isinstance(t.value, ast.Name | ast.Attribute)
        and (t.value.id if isinstance(t.value, ast.Name) else t.value.attr)
        in {"M", "D", "G", "T", "F", "P"}
    ]
    assert bad == [], f"A1: the harness writes into an imported runner: {bad}"
    setattrs = [
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "setattr"
        and n.args
        and isinstance(n.args[0], ast.Name)
        and n.args[0].id in {"M", "D", "G", "T", "F", "P"}
    ]
    assert setattrs == [], setattrs


def test_pa_passes_at_head() -> None:
    """A3 (P-A, the too-wide direction): at HEAD the three dicts differ exactly as A3 says: no problem."""
    assert list(api("pa_check")(_lean())["problems"]) == []


def test_pa_fails_when_the_reground_arm_also_sets_action_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A3 mutation to kill: an ``arm_context`` whose reground arm also rebinds ``LIVE_DEFAULT_ACTION_OUTCOME``
    makes P-A fail (``pa_problems`` reads ``arm_context`` at call time)."""
    from local_astra import cli

    h = harness()

    @contextlib.contextmanager
    def leaky(arm: str) -> Iterator[None]:
        saved = _levers()
        try:
            if arm == "outcome":
                cli.LIVE_DEFAULT_ACTION_OUTCOME = True
            elif arm == "reground":
                cli.LIVE_DEFAULT_REGROUND = True
                cli.LIVE_DEFAULT_ACTION_OUTCOME = True
            yield
        finally:
            cli.LIVE_DEFAULT_ACTION_OUTCOME, cli.LIVE_DEFAULT_REGROUND = saved

    monkeypatch.setattr(h, "arm_context", leaky)
    assert list(h.pa_check(_lean())["problems"]), (
        "A3: P-A passed a reground arm that also sets action_outcome"
    )
    assert _levers() == (False, False)


def test_pa_fails_when_the_reground_arm_keeps_overview_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A3 / R-6: if cli's H-T152-1 yield went missing (the overview frame resolves on beside ReGround), the
    reground arm's keywords keep ``overview_frame`` and differ from base only in ``reground``. P-A must fail on that
    alone: no other A3 clause sees it (the premise checks that the diff is exactly ``reground``)."""
    from local_astra import cli

    h = harness()
    monkeypatch.setattr(cli, "_overview_frame_others", lambda *a, **k: ())
    got = h.pa_check(_lean())
    rg = h.keyword_diff(got["keywords"]["base"], got["keywords"]["reground"])
    assert rg == {"added": {}, "removed": [], "changed": {"reground": [False, True]}}, rg
    assert list(got["problems"]), "A3: P-A passed a reground arm that still carries overview_frame"
    assert _levers() == (False, False)


@pytest.mark.parametrize("crop", [560, 1120])
def test_pa_fails_when_the_reground_crop_is_not_280(
    monkeypatch: pytest.MonkeyPatch, crop: int
) -> None:
    """A3 / R-6: with cli's default pass-2 crop moved off 280 (``LIVE_DEFAULT_REGROUND_VISUAL_TOKENS``), the keyword
    dicts still differ exactly as A3 says (the crop is not a session keyword), so only the 280-crop clause of P-A
    can fail it."""
    from local_astra import cli

    h = harness()
    monkeypatch.setattr(cli, "LIVE_DEFAULT_REGROUND_VISUAL_TOKENS", crop)
    got = h.pa_check(_lean())
    assert got["grounding"]["reground"]["reground_visual_tokens"] == crop, got["grounding"]
    kw = got["keywords"]
    assert h.keyword_diff(kw["base"], kw["outcome"]) == {
        "added": {"action_outcome": True},
        "removed": [],
        "changed": {},
    }
    assert h.keyword_diff(kw["base"], kw["reground"]) == {
        "added": {},
        "removed": ["overview_frame"],
        "changed": {"reground": [False, True]},
    }
    assert list(got["problems"]), f"A3: P-A passed a reground crop of {crop}"


def test_arm_manifest_records_the_three_dicts_and_the_diffs() -> None:
    """A3: the manifest's arm record holds the three keyword dicts (the guidance by sha256, never its text) and the
    diffs: ``action_outcome`` named for outcome; ``reground`` and ``overview_frame`` for reground."""
    import hashlib

    lean = _lean()
    man = api("pa_check")(lean)  # the manifest's arm_keywords / arm_diffs / arm_grounding record
    assert set(man["keywords"]) == set(ARMS) and set(man["diffs"]) == {"outcome", "reground"}, man
    text = json.dumps(man, sort_keys=True, default=str)
    guidance = _m().D.session_keywords(lean)["tool_guidance"]
    assert hashlib.sha256(guidance.encode()).hexdigest() in text, (
        "A3: the guidance sha256 is not recorded"
    )
    assert guidance not in text, (
        "A3: the guidance text itself is in the manifest (record it by sha256)"
    )
    for arm in ARMS:
        assert arm in text
    for key in ("action_outcome", "reground", "overview_frame"):
        assert key in text, key


# =====================================================================================================
# refusals (A4), the frozen files (A1/N1), the state read (A1/B-R23)
# =====================================================================================================


class Rig:
    """Every real socket refused and recorded; a clean tree and the merges through ``_git`` (patched on the
    harness and every runner it holds); the kit cli resolves (repair on); Chromium present.

    The rig fakes the tree's state, never its history: the harness reads the read-only runners' real blobs at
    ``Merge T-0210-c:`` (fec43cc3), so without that commit every ``main`` refuses on "runners differ" and a refusal
    row would pass for the wrong reason. Release re-pin (James 2026-11-01): the squashed release has no history, so
    every rig row skips naming the commit; on main it runs."""

    def __init__(self, tmp_path: Path, mp: pytest.MonkeyPatch) -> None:
        gr.require_commits(MERGE_T0210C_SHA, why=gr.NO_HISTORY)
        from local_astra import cli

        self.h = h = harness()
        self.cli = cli
        self.sockets: list[str] = []
        self.porcelain = ""
        self.merges: tuple[str, ...] = (*t210.OTHERS, t210.MERGE_A, t210.MERGE_B, MERGE_C)
        self.out = tmp_path / "out"
        self.tools_version: str | None = None
        self.repair_off: Any = None
        #: pinned paths whose blob at the merge the rig answers with altered bytes (the working file then differs)
        self.altered: set[str] = set()

        def refuse(what: str) -> Callable[..., Any]:
            def fn(*a: Any, **k: Any) -> Any:
                self.sockets.append(what)
                raise ConnectionRefusedError(f"T-0212 refusal rig: no {what}")

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
                # the A1/N1 byte pin reads the real blobs: the rig fakes the tree's state, never its history
                blob = subprocess.run(
                    ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
                ).stdout
                if any(a.endswith(":" + rel) for a in args for rel in self.altered):
                    blob += "# T-0212 rig: the merge's bytes differ from the working file\n"
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
            v = self.tools_version or t205._set_sha(page_overview, wait_tool)
            return (t210._StubKit(tools_version=v, repair_off=self.repair_off), object())

        mp.setattr(cli, "load_tool_kit", load)
        mp.setattr(cli, "live_prerequisites", lambda *a, **k: None, raising=False)

    def argv(self) -> list[str]:
        return ["--base-url", t31.VALID_URL, "--out-dir", str(self.out)]


REFUSALS = (
    "url-not-loopback",
    "dirty-tree",
    "out-dir-exists",
    "no-merge-c",
    "web-off",
    "guidance-v2",
    "guidance-v4",
    "loop-19",
    "table-off-its-sha",
    "provenance-off-its-sha",
    "kit-off-clis-set",
    "repair-off",
    "chromium-absent",
    "head-action-outcome-on",
    "head-reground-on",
    "pa-fails",
    "t0210-runner-differs",
    "t0205-runner-differs",
)


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    """A4: each refusal exits 2 with a stderr line before any socket, leaving the out-dir as it was. ``no-merge-c``:
    the log holds Merge T-0210-a and -b, T-0211's and lookalikes, but no ``Merge T-0210-c:``. ``head-*-on``: HEAD's
    live default already has the lever on (the arms would collapse). ``pa-fails``: a leaky ``arm_context``."""
    from local_astra.pages import threejs_tasks

    rig = Rig(tmp_path, monkeypatch)
    cli, h = rig.cli, rig.h
    argv = rig.argv()
    if case == "url-not-loopback":
        argv[1] = "http://10.0.0.1:8001/v1"
    elif case == "dirty-tree":
        rig.porcelain = " M local_astra/conductor/session.py\n"
    elif case == "out-dir-exists":
        rig.out.mkdir()
        (rig.out / "keep.txt").write_text("x", encoding="utf-8")
    elif case == "no-merge-c":
        rig.merges = (
            *t210.OTHERS,
            t210.MERGE_A,
            t210.MERGE_B,
            "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c Merge T-0210-cc: a lookalike",
            "0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d Merge branch 'main' into T-0210-c-impl",
        )
    elif case == "web-off":
        monkeypatch.setattr(cli, "web_of", lambda *a, **k: False)
    elif case in ("guidance-v2", "guidance-v4"):
        v = case.rsplit("-", 1)[1]
        monkeypatch.setattr(cli, "live_guidance_of", lambda p, a, m: cli.LiveGuidance(v, "default"))
    elif case == "loop-19":
        real = cli.tool_loop_of
        monkeypatch.setattr(
            cli, "tool_loop_of", lambda p, a: dataclasses.replace(real(p, a), max_tool_steps=19)
        )
    elif case == "table-off-its-sha":
        moved = (
            dataclasses.replace(threejs_tasks.TASKS[0], request="Add a box."),
            *threejs_tasks.TASKS[1:],
        )
        monkeypatch.setattr(threejs_tasks, "TASKS", moved)
        _patch_name(monkeypatch, h, "TASKS", moved)
        assert threejs_tasks.task_table_sha() != t210.TABLE_SHA256
    elif case == "provenance-off-its-sha":
        _patch_name(monkeypatch, h, "PROVENANCE_PATH", t210._provenance_copy(tmp_path, alter=True))
    elif case == "kit-off-clis-set":
        rig.tools_version = "0" * 64
    elif case == "repair-off":
        rig.repair_off = "T-0212 rig: repair off"
    elif case == "chromium-absent":
        monkeypatch.setattr(cli, "live_prerequisites", lambda *a, **k: "T-0212 rig: no Chromium")
    elif case == "head-action-outcome-on":
        monkeypatch.setattr(cli, "LIVE_DEFAULT_ACTION_OUTCOME", True)
    elif case == "head-reground-on":
        monkeypatch.setattr(cli, "LIVE_DEFAULT_REGROUND", True)
    elif case == "pa-fails":

        @contextlib.contextmanager
        def leaky(arm: str) -> Iterator[None]:
            saved = _levers()
            try:
                if arm in ("outcome", "reground"):
                    cli.LIVE_DEFAULT_ACTION_OUTCOME = True
                if arm == "reground":
                    cli.LIVE_DEFAULT_REGROUND = True
                yield
            finally:
                cli.LIVE_DEFAULT_ACTION_OUTCOME, cli.LIVE_DEFAULT_REGROUND = saved

        monkeypatch.setattr(h, "arm_context", leaky)
    elif case in ("t0210-runner-differs", "t0205-runner-differs"):
        if _git_show(MERGE_T0210C_SHA, "scripts/gpu/t0210_threejs_run.py") is None:
            pytest.skip(f"{MERGE_T0210C_SHA[:8]} is not in this checkout")
        rig.altered = {
            "scripts/gpu/t0210_threejs_run.py"
            if case.startswith("t0210")
            else "scripts/gpu/t0205_demo_run.py"
        }
    else:
        raise KeyError(case)
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = h.main(argv)
    err = capsys.readouterr().err
    assert rc == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [], (case, rig.sockets)
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert err.strip(), f"{case}: the refusal says nothing"
    if case == "no-merge-c":
        assert "T-0210-c" in err, err[-2000:]
    if case.endswith("-runner-differs"):
        (rel,) = rig.altered
        assert Path(rel).name in err, (case, err[-2000:])


@pytest.mark.parametrize("lever", ["LIVE_DEFAULT_ACTION_OUTCOME", "LIVE_DEFAULT_REGROUND"])
def test_refuses_when_head_default_has_lever_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], lever: str
) -> None:
    """A4: HEAD's live default with a lever already on is exit 2 with zero sockets, and the refusal names it."""
    rig = Rig(tmp_path, monkeypatch)
    monkeypatch.setattr(rig.cli, lever, True)
    rc = rig.h.main(rig.argv())
    err = capsys.readouterr().err
    assert rc == 2 and rig.sockets == [], (rc, rig.sockets, err[-1500:])
    assert re.search(r"action.outcome|reground", err, re.IGNORECASE), err[-1500:]


def test_rig_at_head_goes_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A4, the other side of every refusal (a guard too wide): at HEAD (Merge T-0210-c reachable, levers off, P-A
    passing, every pin intact) ``main`` refuses nothing; it asks for the engine and, finding none, exits 1."""
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
        "scripts/gpu/t0210_threejs_run.py",
        "scripts/gpu/t0205_demo_run.py",
        "tests/eval/test_t0210_threejs_run.py",
        "local_astra/pages/threejs_tasks.py",
    ],
)
def test_t0210_t0205_files_unchanged(rel: str) -> None:
    """A1/N1 (negative criterion, green at BASE by design): the read-only files' bytes match ``Merge T-0210-c:``
    (fec43cc3, a literal: never a moving ref). Skipped when that commit is absent (a ``git archive`` copy)."""
    pinned = _git_show(MERGE_T0210C_SHA, rel)
    if pinned is None:
        pytest.skip(f"{MERGE_T0210C_SHA[:8]} is not in this checkout")
    assert (REPO / rel).read_bytes() == pinned, f"N1: {rel} differs from Merge T-0210-c"


def test_state_read_literal() -> None:
    """A1/B-R23: the harness names no state-read expression of its own: every string constant mentioning
    ``window.`` is M's ``STATE_READ_JS`` exactly, and no ``JSON.stringify`` or ``.evaluate(`` appears."""
    m = _m()
    assert m.STATE_READ_JS == "() => window.__state"
    tree = ast.parse(HARNESS.read_text(encoding="utf-8"))
    consts = [
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    bad = [c for c in consts if "window." in c and c != m.STATE_READ_JS]
    assert bad == [], bad
    assert not any("JSON.stringify" in c for c in consts)
    calls = [
        ast.unparse(n)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("evaluate", "evaluate_handle")
    ]
    assert calls == [], f"B-R23: the harness evaluates in the page itself: {calls}"


# =====================================================================================================
# 2. metric functions over hand-built session-log records (A7, A8)
# =====================================================================================================


def _session() -> ModuleType:
    from local_astra.conductor import session

    return session


def _lbl(key: str, *, outcome: str | None = None, extent: bool = False) -> str:
    s = _session()
    opening = {"now": s.LABEL_NOW, "updated": s.LABEL_UPDATED, "unchanged": s.LABEL_UNCHANGED}[key]
    clause = s.OUTCOME_CLAUSE.format(outcome=outcome) if outcome is not None else ""
    ext = " Page beyond this screen: about 1.2 screens below, nothing above." if extent else ""
    return "[" + opening + clause + ext + "]"


def test_label_texts_premise() -> None:
    """Premise: the hand labels parse under the real ``screen_reminder_key`` as the rows below say; the screen
    reminder text and a whole continuation are not labels."""
    s = _session()
    k = s.screen_reminder_key
    assert k(_lbl("unchanged")) == "unchanged"
    assert k(_lbl("unchanged", extent=True)) == "unchanged"
    assert k(_lbl("updated", outcome="no visible change")) == "updated"
    assert k(_lbl("updated", outcome="no visible change", extent=True)) == "updated"
    assert k(_lbl("now")) == "now"
    assert k(s.SCREEN_REMINDERS["unchanged"]) is None
    assert (
        k('I will click.\n<|tool_response>response:click_at{screenshot:<|"|>unchanged<|"|>}')
        is None
    )


class Log:
    """One turn's session-log records, written in ``seq`` order the way ``Session`` logs them. ``session`` names the
    turn's session (its ``session_id`` is ``s-<session>``; default the turn id). ``real_ids``: call ids as a fresh
    Session names them (``t1-call`` at step 1, ``t1-call-s<k>`` at step k; one call per step), so two turns' logs
    hold the same call ids (R-1)."""

    def __init__(
        self, turn_id: str = "t1", *, session: str | None = None, real_ids: bool = False
    ) -> None:
        self.turn_id = turn_id
        self.session = session or turn_id
        self.real_ids = real_ids
        self.recs: list[dict[str, Any]] = []
        self.step = 0
        self.calls = 0

    def next_cid(self) -> str:
        """The id the next :meth:`call` gets."""
        if self.real_ids:
            return f"{self.turn_id}-call" if self.step <= 1 else f"{self.turn_id}-call-s{self.step}"
        return f"{self.turn_id}-call-{self.calls + 1}"

    def add(self, kind: str, data: dict[str, Any]) -> dict[str, Any]:
        r = {"seq": len(self.recs) + 1, "kind": kind, "turn_id": self.turn_id, "data": data}
        self.recs.append(r)
        return r

    def request(
        self,
        label: str | None,
        *,
        history: tuple[str, ...] = (),
        continuation: str = "",
        phase: str | None = None,
    ) -> None:
        """A request; ``label`` None is step 1's ``generate``. ``history`` are earlier steps' text parts repeated
        before this request's own image and label (the adversarial shape); ``continuation`` the assistant text."""
        self.step += 1
        user: list[dict[str, Any]] = []
        for h in history:
            user += [
                {"type": "image_url", "url": "{'redacted': 'image'}", "phash": "old"},
                {"type": "text", "text": h},
            ]
        user.append({"type": "image_url", "url": "{'redacted': 'image'}", "phash": f"p{self.step}"})
        if label is not None:
            user.append({"type": "text", "text": label})
        user.append({"type": "text", "text": "Add a point light at (2,3,2) with intensity 5."})
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "You are Gearvis."}]},
            {"role": "user", "content": user},
        ]
        if continuation:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": continuation}]})
        data: dict[str, Any] = {
            "phase": phase or ("generate" if label is None else "tool_step"),
            "step": self.step,
            "request_id": f"{self.turn_id}-gen-s{self.step}",
            "attempt_id": f"a-{self.turn_id}-{self.step}",
            "prompt": {"messages": msgs, "continue_final_message": bool(continuation)},
        }
        if label is not None:
            data["live_label"] = {
                "text": label,
                "above_px": None,
                "below_px": None,
                "viewport_px": None,
            }
        self.add("llm.request", data)

    def call(
        self,
        name: str,
        status: str = "executed",
        reason: str | None = None,
        resolved: dict[str, Any] | None = None,
        **args: Any,
    ) -> str:
        cid = self.next_cid()
        self.calls += 1
        base = {"call_id": cid, "name": name, "arguments": dict(args), "source": "draft"}
        self.add("tool.call", {**base, "status": "proposed", "reason": None, "resolved": None})
        self.add(
            "tool.call",
            {
                **base,
                "status": status,
                "reason": reason,
                "resolved": resolved
                if resolved is not None
                else ({} if status == "executed" else None),
            },
        )
        if status == "executed":
            self.add(
                "tool.result",
                {"call_id": cid, "step": self.step, "status": status, "response": "{}"},
            )
        return cid

    def reground(
        self,
        cid: str,
        outcome: str,
        p1: tuple[int, int] | None,
        p2: tuple[int, int] | None,
        ms: int | None = 900,
        name: str = "click_at",
        executed: tuple[int, int] | None = None,
    ) -> None:
        """A ``tool.reground`` record (and its pass-2 request when ``ms`` is set). ``p1`` and ``executed`` are in
        full-frame model units, ``p2`` in **crop** units (R-2): an ``ok`` record names its ``executed`` point (p2
        mapped back to the full frame); any other outcome executes at ``p1``."""
        if outcome == "ok":
            assert executed is not None, (
                "T-0212 rig: an ok tool.reground record needs its executed point"
            )
        else:
            assert executed is None, "T-0212 rig: only an ok record executes away from p1"
            executed = p1
        if ms is not None:
            self.add(
                "llm.request",
                {
                    "phase": "reground",
                    "step": self.step,
                    "request_id": f"rg-{cid}",
                    "attempt_id": f"rg-{cid}",
                    "call_id": cid,
                    "prompt": {"messages": []},
                },
            )
        self.add(
            "tool.reground",
            {
                "call_id": cid,
                "step": self.step,
                "name": name,
                "p1": None if p1 is None else {"x": p1[0], "y": p1[1]},
                "p2": None if p2 is None else {"x": p2[0], "y": p2[1]},
                "crop": None,
                "ms": ms,
                "tool_text": None,
                "outcome": outcome,
                "executed": {
                    "x": (executed or (None, None))[0],
                    "y": (executed or (None, None))[1],
                },
                "budget": 280,
            },
        )

    def note(self, what: str, **data: Any) -> None:
        self.add("note", {"what": what, **data})

    def line(self, extras: dict[str, Any]) -> None:
        self.add(
            "turn.record",
            {"turn_id": self.turn_id, "session_id": f"s-{self.session}", "extras": extras},
        )


RESPONSE_UNCHANGED = (
    '<|tool_response>response:click_at{action:<|"|>clicked at (300, 200)<|"|>,screenshot:<|"|>unchanged<|"|>,'
    'status:<|"|>executed<|"|>}<tool_response|>'
)


def test_unchanged_label_counted_once_per_step() -> None:
    """A8 (adversarial, the likely failure): four requests; step 2's request carries the unchanged label (with the
    extent clause); steps 3 and 4 repeat step 2's text part and its ``screenshot:"unchanged"`` response in their
    history, but carry their own ``updated`` label (step 4's with an outcome clause). Count 1, not 3; the labels per
    labelled tool_step request are [unchanged, updated, updated]."""
    lg = Log()
    lg.request(None)
    lg.call("click_at", x=300, y=200)
    unchanged = _lbl("unchanged", extent=True)
    lg.request(unchanged, continuation="I will click.\n" + RESPONSE_UNCHANGED)
    lg.call("click_at", x=320, y=200)
    lg.request(
        _lbl("updated"),
        history=(unchanged,),
        continuation="x\n" + RESPONSE_UNCHANGED + "\n" + unchanged,
    )
    lg.call("type_at", x=320, y=200, text="5")
    lg.request(
        _lbl("updated", outcome="no visible change"),
        history=(unchanged, _lbl("updated")),
        continuation="y\n" + RESPONSE_UNCHANGED + "\n" + unchanged,
    )
    assert list(_request_labels(lg.recs)) == ["unchanged", "updated", "updated"]
    assert api("unchanged_labels")(lg.recs) == 1


def test_unchanged_label_skips_generate_and_check_requests() -> None:
    """A8: only labelled ``tool_step`` requests count; a ``completion_check`` request carrying the unchanged label
    text is not a step's label (reading 1)."""
    lg = Log()
    lg.request(None)
    lg.call("click_at", x=1, y=1)
    lg.request(_lbl("unchanged"))
    lg.request(_lbl("unchanged"), phase="completion_check")
    assert api("unchanged_labels")(lg.recs) == 1


def test_unchanged_label_counted_once_when_a_step_is_resent() -> None:
    """A8 "exact definition, one per step" (reading 1): an engine-empty or degenerate retry logs the same step's
    ``tool_step`` request again (same ``step``, same label). The step's one unchanged label counts once, not twice.
    (A (c) row at dfa98a73: the harness counts every logged request.)"""
    lg = Log()
    lg.request(None)
    lg.call("click_at", x=300, y=200)
    lg.request(_lbl("unchanged"))
    first = lg.recs[-1]
    lg.note(
        "the engine returned an empty retryable response; retrying once", step=first["data"]["step"]
    )
    lg.add("llm.request", copy.deepcopy(first["data"]) | {"attempt_id": "a-t1-2-resent"})
    assert api("unchanged_labels")(lg.recs) == 1
    assert _request_labels(lg.recs) == ["unchanged"]


def test_a_request_without_its_own_label_has_none() -> None:
    """A8 "text repeated from earlier steps in later requests' history is never counted again" (reading 1): a
    ``tool_step`` request that places no label of its own (no ``live_label``) but repeats step 2's unchanged label
    in its history is not an unchanged label; the count stays 1, and the click before it is unlabelled, not a miss.
    (A (c) row at dfa98a73: the harness takes the last label-shaped text part wherever it sits.)"""
    lg = Log()
    lg.request(None)
    lg.call("click_at", x=300, y=200)
    unchanged = _lbl("unchanged")
    lg.request(unchanged)
    lg.call("click_at", x=500, y=500)
    lg.request(None, history=(unchanged,), phase="tool_step")
    assert "live_label" not in lg.recs[-1]["data"]
    assert api("unchanged_labels")(lg.recs) == 1
    got = dict(api("click_misses")(lg.recs))
    assert {k: got[k] for k in ("misses", "labelled", "unlabelled")} == {
        "misses": 1,
        "labelled": 1,
        "unlabelled": 1,
    }, got


def _miss_case(case: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    lg = Log()
    lg.request(None)
    if case == "miss":
        lg.call("click_at", x=300, y=200)
        lg.request(_lbl("unchanged"))
        want = {"misses": 1, "labelled": 1, "unlabelled": 0}
    elif case == "hit":
        lg.call("click_at", x=300, y=200)
        lg.request(_lbl("updated", outcome="no visible change"))
        want = {"misses": 0, "labelled": 1, "unlabelled": 0}
    elif case == "refused-repeat-guard":
        lg.call(
            "click_at",
            "refused",
            "you already clicked here twice and the screen did not change",
            x=300,
            y=200,
        )
        lg.request(_lbl("unchanged"))
        want = {"misses": 0, "labelled": 0, "unlabelled": 0}
    elif case == "refused-body-focus":
        lg.call("click_at", "failed", t210.FOCUS_UNKNOWN, {"focus_check": "body"}, x=300, y=200)
        lg.request(_lbl("unchanged"))
        want = {"misses": 0, "labelled": 0, "unlabelled": 0}
    elif case == "final-unlabelled":
        lg.call("click_at", x=300, y=200)
        lg.request(_lbl("unchanged"))
        lg.call("click_at", x=500, y=500)
        want = {"misses": 1, "labelled": 1, "unlabelled": 1}
    elif case == "type-and-scroll":
        lg.call("type_at", x=300, y=200, text="5")
        lg.request(_lbl("unchanged"))
        lg.call("scroll", x=300, y=200, direction="down")
        lg.request(_lbl("unchanged"))
        want = {"misses": 0, "labelled": 0, "unlabelled": 0}
    elif case == "next-request-not-later-ones":
        lg.call("click_at", x=300, y=200)
        lg.request(_lbl("updated"))
        lg.request(_lbl("unchanged"))  # a later request's label is not the click's
        want = {"misses": 0, "labelled": 1, "unlabelled": 0}
    else:
        raise KeyError(case)
    return lg.recs, want


MISS_CASES = (
    "miss",
    "hit",
    "refused-repeat-guard",
    "refused-body-focus",
    "final-unlabelled",
    "type-and-scroll",
    "next-request-not-later-ones",
)


@pytest.mark.parametrize("case", MISS_CASES)
def test_click_miss_definition(case: str) -> None:
    """A8: an executed ``click_at`` followed by an unchanged label is a miss, by updated is not; a refused click
    (repeat guard, body focus) is never a click; a last click with no next request is ``unlabelled``; ``type_at`` and
    ``scroll`` are not clicks; only the click's *next* labelled request decides (counts by hand)."""
    recs, want = _miss_case(case)
    got = dict(api("click_misses")(recs))
    assert {k: got.get(k) for k in want} == want, (case, got)


def _reaim_turn(
    dx: int, *, first_miss: bool = True, type_between: bool = False
) -> list[dict[str, Any]]:
    lg = Log()
    lg.request(None)
    lg.call("click_at", x=100, y=100)
    lg.request(_lbl("unchanged" if first_miss else "updated"))
    if type_between:
        lg.call("type_at", x=100, y=100, text="2")
        lg.request(_lbl("updated"))
    lg.call("click_at", x=100 + dx, y=100 - min(dx, 5))
    lg.request(_lbl("updated"))
    return lg.recs


@pytest.mark.parametrize(("dx", "want"), [(9, 0), (10, 0), (11, 1), (40, 1), (41, 0)])
def test_reaim_window(dx: int, want: int) -> None:
    """A8: a miss at p whose next executed pointer action is a click at q with ``10 < max(|dx|, |dy|) <= 40``
    (|dy| = min(dx, 5) here, so the max norm is dx): only 11 and 40 are re-aims."""
    assert api("reaims")(_reaim_turn(dx)) == want


def test_reaim_needs_a_miss_and_the_next_pointer_action() -> None:
    """A8: 11 units after a click that was not a miss is no re-aim; a ``type_at`` between the miss and the click
    breaks the pair (reading 2)."""
    assert api("reaims")(_reaim_turn(11, first_miss=False)) == 0
    assert api("reaims")(_reaim_turn(11, type_between=True)) == 0


def _reaim_reground_turn() -> list[dict[str, Any]]:
    """Three clicks in the reground arm whose ``tool.call`` arguments differ from the executed point (p2 in crop
    units, R-2, never a full-frame point):

    1. ``ok``: arguments (100, 100) = p1, pass 2 (crop (612, 480)) moved it to executed (130, 100); next label
       unchanged (miss).
    2. ``fallback``: arguments (145, 100) = p1 = executed (pass 2 offered crop (400, 100), not taken); unchanged
       (miss).
    3. ``ok``: arguments (150, 100) = p1, executed (175, 100) (crop (590, 505)); next label updated.

    By the executed point: 1->2 is |145 - 130| = 15 (re-aim), 2->3 is |175 - 145| = 30 (re-aim): 2.
    By the arguments: 45 and 5: 0. Executed for p only: 15 and 5: 1. Executed for q only: 45 and 30: 1."""
    lg = Log()
    lg.request(None)
    lg.reground(f"{lg.turn_id}-call-1", "ok", (100, 100), (612, 480), ms=1100, executed=(130, 100))
    lg.call("click_at", x=100, y=100)
    lg.request(_lbl("unchanged"))
    lg.reground(f"{lg.turn_id}-call-2", "fallback", (145, 100), (400, 100), ms=1300)
    lg.call("click_at", x=145, y=100)
    lg.request(_lbl("unchanged"))
    lg.reground(f"{lg.turn_id}-call-3", "ok", (150, 100), (590, 505), ms=900, executed=(175, 100))
    lg.call("click_at", x=150, y=100)
    lg.request(_lbl("updated"))
    return lg.recs


def test_reaim_uses_the_executed_point_under_reground() -> None:
    """A8 re-aims, ruling 2026-10-31 (the click point is the executed point; in the reground arm that is
    ``tool.reground.executed`` when present, as the eval reads it): with an ``ok`` and a ``fallback`` record whose
    ``tool.call`` arguments differ from the executed point, the re-aim distance is measured between executed points
    (hand count 2; the arguments give 0, either half alone gives 1). The misses do not depend on the point (2)."""
    recs = _reaim_reground_turn()
    got = dict(api("click_misses")(recs))
    assert {k: got.get(k) for k in ("misses", "labelled", "unlabelled")} == {
        "misses": 2,
        "labelled": 3,
        "unlabelled": 0,
    }, got
    assert api("reaims")(recs) == 2


def _reground_turn() -> list[dict[str, Any]]:
    lg = Log()
    lg.request(None)
    c1 = f"{lg.turn_id}-call-1"
    # R-2: p2 in crop units; the shift is executed - p1 = max(6, 3) = 6: moved (p2 - p1 would be 421)
    lg.reground(c1, "ok", (100, 100), (521, 489), ms=1200, executed=(106, 97))
    lg.call("click_at", x=106, y=97)
    lg.request(_lbl("unchanged"))  # moved + unchanged: a miss in "moved"
    c2 = f"{lg.turn_id}-call-2"
    # executed = p1: shift 0, did not move (p2 - p1 would be 300, "moved")
    lg.reground(c2, "ok", (200, 200), (500, 500), ms=1000, executed=(200, 200))
    lg.call("click_at", x=200, y=200)
    lg.request(_lbl("updated"))
    c3 = f"{lg.turn_id}-call-3"
    lg.reground(
        c3, "fallback", (300, 300), (350, 300), ms=1500
    )  # fallback: executed at p1, did not move
    lg.call("click_at", x=300, y=300)
    lg.request(_lbl("unchanged"))
    c4 = f"{lg.turn_id}-call-4"
    lg.reground(c4, "skipped_no_point", None, None, ms=None)
    lg.call("click_at", x=400, y=400)
    lg.request(_lbl("updated"))
    c5 = f"{lg.turn_id}-call-5"
    lg.reground(c5, "pass2_timeout", (500, 500), None, ms=4000)
    lg.call("click_at", x=500, y=500)
    lg.request(_lbl("unchanged"))
    return lg.recs


def test_reground_shift_and_split() -> None:
    """A8 (reground arm), R-2: outcomes by kind; the shift ``max(|executed - p1|)`` over ``ok`` records only ([6, 0];
    the fallback's offer is not a shift; ``p2`` is in crop units, and ``|p2 - p1|`` would give [421, 300]); the miss
    split by "pass 2 moved the point" (reading 3): moved 1/1, did not 2/4 (c2 hit, c3 miss, c4 hit, c5 miss).
    ``|p2 - p1|`` puts c2 in "moved": 1/2 and 2/3."""
    got = api("reground_stats")(_reground_turn())
    assert dict(got["outcomes"]) == {
        "ok": 2,
        "fallback": 1,
        "skipped_no_point": 1,
        "pass2_timeout": 1,
        "error": 0,
    }
    assert sorted(got["shifts"]) == [0, 6], got["shifts"]
    assert {k: got["split"]["moved"][k] for k in ("misses", "labelled")} == {
        "misses": 1,
        "labelled": 1,
    }
    assert {k: got["split"]["not_moved"][k] for k in ("misses", "labelled")} == {
        "misses": 2,
        "labelled": 4,
    }


def _reviewer_shift_turn() -> list[dict[str, Any]]:
    """The reviewer's case (T-0212 review at f35477b7, R-2): p1 (363, 520) full frame, p2 (366, 522) crop units,
    executed (309, 529) full frame. Shift = max(|309 - 363|, |529 - 520|) = max(54, 9) = 54; ``|p2 - p1|`` = 3."""
    lg = Log()
    lg.request(None)
    lg.reground(lg.next_cid(), "ok", (363, 520), (366, 522), ms=1000, executed=(309, 529))
    lg.call("click_at", x=363, y=520)
    lg.request(_lbl("updated"))
    return lg.recs


def test_reground_shift_is_executed_minus_p1() -> None:
    """A8 amended by R-2: the reviewer's record gives a shift of 54 (hand), not ``|p2 - p1|`` = 3."""
    got = api("reground_stats")(_reviewer_shift_turn())
    assert list(got["shifts"]) == [54], got["shifts"]


def test_eval_reground_shift_is_executed_minus_p1() -> None:
    """R-2, the eval's oracle: the same record gives 54 under the eval's ``reground_shift``; a non-``ok`` record has
    no shift."""
    ev = _eval_module()
    rec = next(r["data"] for r in _reviewer_shift_turn() if r["kind"] == "tool.reground")
    assert ev.reground_shift(rec) == 54
    assert ev.reground_shift({**rec, "outcome": "fallback"}) is None


def _ok_rate_turns() -> list[dict[str, Any]]:
    """One reground turn: ok, fallback, skipped_no_point (hand: 1 ok of 2 pass-2 requests; 1/3 if the skipped record
    counted)."""
    lg = Log()
    lg.request(None)
    lg.reground(f"{lg.turn_id}-call-1", "ok", (100, 100), (500, 500), executed=(104, 100))
    lg.call("click_at", x=100, y=100)
    lg.request(_lbl("updated"))
    lg.reground(f"{lg.turn_id}-call-2", "fallback", (200, 200), (480, 520))
    lg.call("click_at", x=200, y=200)
    lg.request(_lbl("updated"))
    lg.reground(f"{lg.turn_id}-call-3", "skipped_no_point", None, None, ms=None)
    lg.call("click_at", x=300, y=300)
    lg.request(_lbl("updated"))
    trial = t210._trial("M1", 0, _st("M1", True))
    return [{"arm": "reground", "trial": trial, "records": lg.recs}]


def test_ok_rate_excludes_skipped_no_point() -> None:
    """A10 (i) reground / R-6: the pass-2 ok-rate's denominator excludes ``skipped_no_point`` (no pass 2 was asked):
    1/2, not 1/3; both the harness's report and the eval's metrics."""
    turns = _ok_rate_turns()
    got = _report_over(turns)["per_arm"]["reground"]["reground"]
    assert got["ok_rate"] == "1/2", got
    assert got["requests"] == 2, got
    ev = _eval_module()
    view = {
        "trial": {**turns[0]["trial"], "arm": "reground"},
        "steps": [],
        "records": turns[0]["records"],
        "extras": None,
    }
    mine = ev.metrics([view])["reground"]
    assert (mine["ok_rate"], mine["requests"]) == ("1/2", 2), mine


def _two_turn_views(ev: ModuleType) -> list[dict[str, Any]]:
    """R-1: two reground turns, each a fresh Session whose first call is ``t1-call``. Turn A's ``t1-call`` moved
    (ok, executed 5 units from p1) and was followed by ``updated``; turn B's ``t1-call`` fell back and was followed
    by ``unchanged``. Hand split: moved 1 labelled / 0 misses; not moved 1 labelled / 1 miss. Pooling call ids
    across turns puts B's click in "moved": 2 / 1 and 0 / 0."""
    out = []
    for name, outcome, label in (("A", "ok", "updated"), ("B", "fallback", "unchanged")):
        lg = Log(session=f"turn-{name}", real_ids=True)
        lg.request(None)
        if outcome == "ok":
            lg.reground(lg.next_cid(), "ok", (300, 200), (510, 495), executed=(305, 200))
        else:
            lg.reground(lg.next_cid(), "fallback", (300, 200), (540, 500))
        lg.call("click_at", x=300, y=200)
        lg.request(_lbl(label))
        out.append(
            {
                "trial": {**t210._trial("M1", 0, _st("M1", True)), "arm": "reground"},
                "steps": [],
                "records": lg.recs,
                "extras": None,
            }
        )
    assert [
        next(r["data"]["call_id"] for r in v["records"] if r["kind"] == "tool.reground")
        for v in out
    ] == ["t1-call", "t1-call"]
    return out


def test_eval_split_keys_moved_calls_per_turn() -> None:
    """R-1 (blocking): call ids repeat across fresh Sessions, so the eval keys "pass 2 moved the point" per turn:
    the two-turn case gives the hand split, and the harness's per-turn split (summed) agrees."""
    ev = _eval_module()
    vs = _two_turn_views(ev)
    split = ev.metrics(vs)["reground"]["split"]
    assert split == {
        "moved": {"labelled": 1, "misses": 0},
        "not_moved": {"labelled": 1, "misses": 1},
    }, split
    harness_split = {"moved": Counter(), "not_moved": Counter()}
    for v in vs:
        for b, row in api("reground_stats")(v["records"])["split"].items():
            harness_split[b].update({k: row[k] for k in ("labelled", "misses")})
    assert {b: dict(c) for b, c in harness_split.items()} == {
        "moved": {"labelled": 1, "misses": 0},
        "not_moved": {"labelled": 1, "misses": 1},
    }, harness_split


def _check_turn(asked: int, outcomes: tuple[str, ...]) -> list[dict[str, Any]]:
    s = _session()
    lg = Log()
    lg.request(None)
    lg.call("click_at", x=1, y=1)
    lg.request(_lbl("updated"))
    for k, o in enumerate(outcomes):
        lg.note(s.CHECK_ASKED_NOTE, step=2 + k)
        lg.note(s.CHECK_OUTCOME_NOTE, request_id=f"c{k}", step=2 + k, outcome=o)
    lg.line(
        {
            "completion_check": {
                "on": True,
                "asked": asked,
                "outcome": outcomes[-1] if outcomes else None,
            }
        }
    )
    return lg.recs


def test_completion_check_counts() -> None:
    """A8: ``asked`` from the turn line, outcomes from the ``completion check outcome`` notes (the ``asked`` notes
    are not outcomes): asked 2, call 1, no_call 1; a turn where it never fired: 0/0/0."""
    got = dict(_completion_counts(_check_turn(2, ("call", "no_call"))))
    assert {k: got.get(k) for k in ("asked", "call", "no_call")} == {
        "asked": 2,
        "call": 1,
        "no_call": 1,
    }
    got = dict(_completion_counts(_check_turn(0, ())))
    assert {k: got.get(k) for k in ("asked", "call", "no_call")} == {
        "asked": 0,
        "call": 0,
        "no_call": 0,
    }


def _st(task: str, ok: bool) -> dict[str, Any]:
    if ok:
        return copy.deepcopy(t210.GOLD[task])
    return (
        t210._state("empty", []) if task in ("M1", "M4") else t210._state("crate", [t210._crate()])
    )


def test_hand_states_premise() -> None:
    """Premise: the golden states judge as successes and the start states as failures that are not state errors or
    traps (the real judge via T-0210 C's helpers)."""
    m = t210.harness()
    for tid in TASK_IDS:
        assert m.succeeded(t210._trial(tid, 0, _st(tid, True)), []) is True, tid
        bad = _st(tid, False)
        assert m.succeeded(t210._trial(tid, 0, bad), []) is False, tid
        assert (m.state_error(tid, bad), m.trap_kind(tid, bad), m.malformed_state(bad)) == (
            False,
            None,
            False,
        )


def test_completion_success_after_fire() -> None:
    """A8: "success among turns where it fired": outcome arm, M1 fired + success, M2 fired + failure, M3 not fired
    + success: fired 2, success after fire 1, success 2 of 3."""
    turns = []
    for tid, ok, asked in (("M1", True, 1), ("M2", False, 1), ("M3", True, 0)):
        trial = t210._trial(tid, 0, _st(tid, ok))
        trial["success"] = ok
        recs = _check_turn(asked, ("call",) if asked else ())
        turns.append({"arm": "outcome", "trial": trial, "records": recs})
    got = _summarise(turns)["outcome"]
    assert (got["n"], got["success"], got["completion_fired"], got["success_after_fire"]) == (
        3,
        2,
        2,
        1,
    ), got


def test_repeat_guard_refusals() -> None:
    """A8: ``stops_repeat + stops_recurring`` (not ``repeats`` or ``no_change_repeats``); absent block is 0."""
    f = api("repeat_guard_refusals")
    assert (
        f(
            {
                "repeat_guard": {
                    "repeats": 7,
                    "no_change_repeats": 5,
                    "stops_repeat": 2,
                    "stops_recurring": 3,
                    "max_screen_visits": 4,
                }
            }
        )
        == 5
    )
    assert f({}) == 0


def test_repeat_guard_footnote_on_the_outcome_arm(tmp_path: Path) -> None:
    """A8: the outcome arm's repeat-guard line carries the detector footnote (T-0137: its no-change input is the
    outcome detector); report.md names it beside the outcome arm."""
    out, _ = _synthetic_run_dir(tmp_path / "run")
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "T-0137" in md and re.search(r"outcome detector", md, re.IGNORECASE), md[:3000]


def _ev_turn(arm: str, case: str = "clean") -> list[dict[str, Any]]:
    lg = Log()
    lg.request(None)
    extras: dict[str, Any] = {
        "overview_frame": {"shown": 0, "fits": 1, "too_tall": 0, "failed": 0, "refused": 0}
    }
    if arm == "reground":
        extras = {
            "reground": {"requests": 1, "ok": 1, "fallback": 0, "skipped": 0, "ms": [900]},
            "grounding": {
                "visual_tokens": 1120,
                "reground": True,
                "reground_from": "default",
                "reground_visual_tokens": 280,
                "reground_visual_tokens_from": "default",
            },
        }
        if case != "no-reground-record":
            lg.reground(lg.next_cid(), "ok", (100, 100), (497, 503), executed=(100, 100))
        if case == "crop-1120":
            extras["grounding"]["reground_visual_tokens"] = 1120
        if case == "no-extras-reground":
            del extras["reground"]
    if arm == "outcome" and case != "no-extras-outcome":
        extras["action_outcome"] = {
            "none": 1,
            "near": 0,
            "elsewhere": 0,
            "page": 0,
            "unmeasured": 0,
        }
    if arm == "base" and case == "extras-outcome":
        extras["action_outcome"] = {
            "none": 0,
            "near": 0,
            "elsewhere": 0,
            "page": 0,
            "unmeasured": 0,
        }
    if case == "overview-shown":
        extras["overview_frame"] = {"shown": 1, "fits": 0, "too_tall": 0, "failed": 0, "refused": 0}
    lg.call("click_at", x=100, y=100)
    if arm == "base" and case == "tool-reground":
        lg.reground(f"{lg.turn_id}-call-1", "ok", (100, 100), (497, 503), executed=(100, 100))
    if arm == "base" and case == "phase-reground":
        lg.request(_lbl("updated"), phase="reground")
    if case == "clause" or arm == "outcome":
        lg.request(_lbl("updated", outcome="no visible change"))
    else:
        lg.request(_lbl("updated"))
    lg.line(extras)
    return lg.recs


EVIDENCE_VIOLATIONS = [
    ("reground", "no-reground-record"),
    ("reground", "crop-1120"),
    ("reground", "no-extras-reground"),
    ("reground", "clause"),
    ("base", "phase-reground"),
    ("base", "tool-reground"),
    ("base", "clause"),
    ("base", "extras-outcome"),
    ("outcome", "no-extras-outcome"),
]


@pytest.mark.parametrize("arm", ARMS)
def test_arm_evidence_clean(arm: str) -> None:
    """A7 (the too-wide direction): a clean turn of each arm has no violation (an outcome clause in an outcome
    turn's label is its evidence, not a violation)."""
    assert list(_evidence(arm, _ev_turn(arm))) == []


@pytest.mark.parametrize(("arm", "case"), EVIDENCE_VIOLATIONS)
def test_arm_evidence(arm: str, case: str) -> None:
    """A7: one violation per row, each making the run ``void`` (``report_label`` with the violations)."""
    v = list(_evidence(arm, _ev_turn(arm, case)))
    assert v, (arm, case)
    assert api("report_label")(False, len(v), None) == "void"
    assert api("report_label")(False, 0, None) == "measured"
    assert api("report_label")(False, 0, "budget spent") == "incomplete"
    # and through the report: the one violating turn makes the run void
    trial = t210._trial("M1", 0, _st("M1", True))
    assert (
        _report_over([{"arm": arm, "trial": trial, "records": _ev_turn(arm, case)}])["label"]
        == "void"
    )


def _confound(arm: str, records: list[dict[str, Any]]) -> str | None:
    trial = t210._trial("M1", 0, _st("M1", True))
    rep = _report_over([{"arm": arm, "trial": trial, "records": records}])
    note = rep["readings"]["reground"]["confound"]
    assert (note is None) == (rep["overview_frame_base_outcome_turns"] == 0), rep
    return note


def test_overview_frame_confound() -> None:
    """A7: a base (or outcome) turn with ``overview_frame.shown == 1`` marks the reground reading "confounded
    (overview frame shown in 1 base/outcome turns)"; a reground turn showing one is not counted; a base turn with
    ``shown == 0`` is not. The note reaches the reground reading line only."""
    want = "confounded (overview frame shown in 1 base/outcome turns)"
    assert _confound("base", _ev_turn("base", "overview-shown")) == want
    assert _confound("outcome", _ev_turn("outcome", "overview-shown")) == want
    assert _confound("reground", _ev_turn("reground", "overview-shown")) is None
    assert _confound("base", _ev_turn("base")) is None
    trial = t210._trial("M1", 0, _st("M1", True))
    rep = _report_over(
        [{"arm": "base", "trial": trial, "records": _ev_turn("base", "overview-shown")}]
    )
    assert want in rep["readings"]["reground"]["line"]
    assert "confounded" not in rep["readings"]["outcome"]["line"]


# =====================================================================================================
# 3. order, budget, not-run (A5, A6, N2)
# =====================================================================================================


def test_order_blocks() -> None:
    """A5: 45 entries ``(task, arm, block)``; entries 15b..15b+14 are block b and hold every (task, arm) once;
    deterministic for seed 0; seed 1 differs."""
    o = [tuple(x) for x in api("order")(0)]
    assert len(o) == 45
    cells = {(t, a) for t in TASK_IDS for a in ARMS}
    for b in range(3):
        block = o[15 * b : 15 * b + 15]
        assert {x[2] for x in block} == {b}, block
        assert {(x[0], x[1]) for x in block} == cells and len(block) == 15
    assert [tuple(x) for x in api("order")(0)] == o
    assert [tuple(x) for x in api("order")(1)] != o
    assert api("ORDER_SEED") == 0


def test_order_one_rng_shuffled_in_turn() -> None:
    """A5 (reading 4): one ``random.Random(0)`` shuffles block 0's cells, then block 1's, then block 2's (not a fresh
    rng per block: blocks 0 and 1 would then be equal)."""
    rng = random.Random(0)
    want = []
    for b in range(3):
        cells = [(t, a) for t in TASK_IDS for a in ARMS]
        rng.shuffle(cells)
        want += [(t, a, b) for t, a in cells]
    o = [tuple(x) for x in api("order")(0)]
    assert o == want
    assert o[:15] != [(t, a, 0) for t, a, _ in o[15:30]]


def test_budget_constants() -> None:
    """A6: the four constants; a worst-case turn and the report fit inside the hard limit at the run budget's
    start and ``RUN_BUDGET_S`` fits one worst-case turn; ``TURN_WORST_S`` covers M's turn, ready wait and closes."""
    h, m = harness(), _m()
    assert (h.HARD_LIMIT_S, h.RUN_BUDGET_S, h.TURN_WORST_S, h.REPORT_S) == (3600, 3120, 480, 60)
    assert h.RUN_BUDGET_S >= h.TURN_WORST_S
    assert m.TURN_TIMEOUT_S + m.READY_TIMEOUT_S + 2 * m.CLOSE_S <= h.TURN_WORST_S
    assert (h.EXIT_OK, h.EXIT_FAILED, h.EXIT_REFUSED, h.EXIT_PROBE) == (0, 1, 2, 3)


def _run(tmp_path: Path, mp: pytest.MonkeyPatch, start: float) -> tuple[Any, Any, t210._Clock]:
    h, m = harness(), _m()
    clock = t210._Clock(start)
    mp.setattr(h, "time", clock)
    mp.setattr(h, "PROCESS_START_S", start)
    out = tmp_path / "out"
    (out / "sessions").mkdir(parents=True)
    known = {
        "base_url": "http://127.0.0.1:8001/v1",
        "out": out,
        "profile": _lean(),
        "sentinel": b"T0212-BUDGET-SENTINEL",
        "manifest": {"stopped": None},
        "vram": t205._FakeVram(),
    }
    names = {f.name for f in dataclasses.fields(m._Run)}
    return h, m._Run(**{k: v for k, v in known.items() if k in names}), clock


def _stub_turns(
    mp: pytest.MonkeyPatch,
    clock: t210._Clock,
    start: float,
    after: list[float],
    raise_on: Callable[[int], bool] = lambda k: False,
) -> list[tuple[str, int, tuple[Any, Any]]]:
    """``M._one_turn`` stubbed (the harness calls it through M): records (task, repeat, levers during the turn) and
    sets the clock to ``start + after[k]``; turn k raises when ``raise_on(k)``."""
    m = _m()
    seen: list[tuple[str, int, tuple[Any, Any]]] = []

    async def one_turn(run: Any, tid: str, repeat: int, *a: Any, **k: Any) -> Any:
        seen.append((tid, repeat, _levers()))
        n = len(seen)
        clock.now = start + after[min(n, len(after)) - 1]
        if raise_on(n):
            raise RuntimeError(f"T-0212 rig: turn {n} raised")
        trial = t210._trial(tid, repeat, _st(tid, True), wall_s=1.0)
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


def test_run_all_wraps_each_turn_in_its_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A2/A5: each turn runs through ``M._one_turn(run, task, block)`` with exactly its arm's lever on during the
    turn (the context encloses the whole await), and both levers are False after the run; each trial carries its
    arm."""
    start = 1000.0
    h, run, clock = _run(tmp_path, monkeypatch, start)
    seen = _stub_turns(monkeypatch, clock, start, [10.0 * k for k in range(1, 46)])
    turns = [tuple(x) for x in h.order(0)][:6]
    asyncio.run(h._run_all(run, turns))
    want = {"base": (False, False), "outcome": (True, False), "reground": (False, True)}
    assert [(t, r, lv) for t, r, lv in seen] == [(t, b, want[a]) for t, a, b in turns]
    assert _levers() == (False, False)
    assert [(x["task"], x.get("arm")) for x in _trials(run.out)] == [(t, a) for t, a, _ in turns]


def test_budget_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A6 (the absent direction): 300 s per turn from elapsed 0: turns start at 0, 300, ..., 3000 (3000 + 540 =
    3540 <= 3600); the 12th at 3300 does not (3840 > 3600). 11 turns; ``stopped`` set; the report written after the
    stop is labelled ``incomplete``."""
    start = 5000.0
    h, run, clock = _run(tmp_path, monkeypatch, start)
    seen = _stub_turns(monkeypatch, clock, start, [300.0 * k for k in range(1, 46)])
    turns = [tuple(x) for x in h.order(0)]
    asyncio.run(h._run_all(run, turns))
    assert len(seen) == 11, len(seen)
    assert isinstance(run.stopped, str) and run.stopped
    assert len(_trials(run.out)) == 11
    probes = t210._probes()  # P1-P3, P-M1, P-M2 all passing (T-0210 C's shapes)
    h.write_report(run.out, probes, {"regime": dict(REGIME), "stopped": run.stopped}, run.stopped)
    report = json.loads((run.out / "report.json").read_text(encoding="utf-8"))
    assert report.get("label") == "incomplete", report.get("label")
    assert (run.out / "report.md").is_file()


@pytest.mark.parametrize(("second", "want"), [(3060.0, 2), (3060.5, 1)])
def test_turn_starts_only_if_a_worst_case_turn_and_the_report_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second: float, want: int
) -> None:
    """A6 (both directions, at the boundary): turn 1 ends at elapsed ``second``; turn 2 starts iff
    ``second + 480 + 60 <= 3600`` (3060 starts, 3060.5 does not). A ``TURN_WORST_S < 480`` or a dropped check lets
    3060.5 start."""
    start = 9000.0
    h, run, clock = _run(tmp_path, monkeypatch, start)
    seen = _stub_turns(monkeypatch, clock, start, [second, second + 1.0])
    asyncio.run(h._run_all(run, [tuple(x) for x in h.order(0)][:3]))
    assert len(seen) == want, (second, len(seen))


def test_not_run_is_errored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A6/N2: turns 2 and 3 raise inside ``M._one_turn``: each is written as an ``M._not_run`` trial of its arm
    (success False, ``state_error`` True, the rig's exception), never dropped; the levers are False afterwards and
    the run goes on to turn 4."""
    start = 100.0
    h, run, clock = _run(tmp_path, monkeypatch, start)
    seen = _stub_turns(
        monkeypatch, clock, start, [5.0 * k for k in range(1, 46)], raise_on=lambda k: k in (2, 3)
    )
    turns = [tuple(x) for x in h.order(0)][:4]
    asyncio.run(h._run_all(run, turns))
    assert len(seen) == 4
    assert _levers() == (False, False), "N2: a raising turn left a lever rebound"
    tr = _trials(run.out)
    assert [(x["task"], x.get("arm")) for x in tr] == [(t, a) for t, a, _ in turns]
    for x in tr[1:3]:
        assert x["success"] is False and x.get("state_error") is True, x
        assert "T-0212 rig" in str(x.get("exception")), x


def test_errored_stop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A6: after ``G.ERRORED_STOP`` consecutive errored turns the run stops (nothing more is started)."""
    start = 100.0
    h, run, clock = _run(tmp_path, monkeypatch, start)
    seen = _stub_turns(
        monkeypatch, clock, start, [5.0 * k for k in range(1, 46)], raise_on=lambda k: True
    )
    asyncio.run(h._run_all(run, [tuple(x) for x in h.order(0)]))
    assert len(seen) == _m().G.ERRORED_STOP
    assert isinstance(run.stopped, str) and run.stopped


def test_two_not_run_turns_make_the_arm_provisional() -> None:
    """A6/A9: two ``M._not_run`` trials in the outcome arm count 2 in its errored column (success False) and mark
    its reading line "provisional (errored 2)"."""
    m = _m()
    turns = []
    for k, tid in enumerate(("M1", "M2", "M3")):
        if k < 2:
            trial = m._not_run(tid, 0, "harness: RuntimeError('T-0212 rig')")
            recs: list[dict[str, Any]] = []
        else:
            trial = t210._trial(tid, 0, _st(tid, True))
            trial["success"] = True
            recs = _ev_turn("outcome")
        turns.append({"arm": "outcome", "trial": trial, "records": recs})
    got = _summarise(turns)["outcome"]
    assert (got["n"], got["errored"], got["success"]) == (3, 2, 1), got
    line = _mark("outcome: mixed", "measured", got["errored"], 0)
    assert "provisional (errored 2)" in line, line


@pytest.mark.parametrize(("base_errored", "want"), [(1, None), (2, 2), (3, 3)])
def test_eval_marks_base_errored(base_errored: int, want: int | None) -> None:
    """R-4, the eval's oracle: 2 or more errored base turns put "provisional (base errored N)" on both levers'
    marks (the lever arms themselves clean); one does not. A safety event still supersedes it (reading 6)."""
    ev = _eval_module()
    for lever in ("outcome", "reground"):
        got = ev.marks(lever, "measured", {"errored": 0}, 0, 0, {"errored": base_errored})
        if want is None:
            assert not any("base errored" in m for m in got), got
        else:
            assert f"provisional (base errored {want})" in got, got
        sup = ev.marks(lever, "measured", {"errored": 0}, 1, 0, {"errored": base_errored})
        assert sup == ["superseded by a safety event"], sup


def test_base_errored_marks_both_lever_lines() -> None:
    """R-4: three ``M._not_run`` base turns (base errored 3) mark both lever reading lines "provisional (base
    errored 3)", beside one clean turn per arm; the lever arms have no errored turn, so no "provisional (errored"
    mark. With one errored base turn, no line carries the base mark (the too-wide direction)."""
    m = _m()

    def turns(n_errored: int) -> list[dict[str, Any]]:
        out = []
        for k, tid in enumerate(TASK_IDS[:n_errored]):
            out.append(
                {
                    "arm": "base",
                    "trial": m._not_run(tid, k % 3, "harness: RuntimeError('T-0212 rig')"),
                    "records": [],
                }
            )
        for arm in ARMS:
            trial = t210._trial("M5", 0, _st("M5", True))
            trial["success"] = True
            out.append({"arm": arm, "trial": trial, "records": _ev_turn(arm)})
        return out

    rep = _report_over(turns(3))
    assert rep["per_arm"]["base"]["errored"] == 3, rep["per_arm"]["base"]
    for lever in ("outcome", "reground"):
        assert rep["per_arm"][lever]["errored"] == 0
        line = rep["readings"][lever]["line"]
        assert "provisional (base errored 3)" in line, (lever, line)
        assert "provisional (errored" not in line, (lever, line)
    one = _report_over(turns(1))
    for lever in ("outcome", "reground"):
        assert "base errored" not in one["readings"][lever]["line"], one["readings"][lever]


# =====================================================================================================
# 4. the reading, the marks, the regime (A9, A10)
# =====================================================================================================


def _metrics(**over: Any) -> dict[str, Any]:
    """Base-like hand metrics: success 4/15 (per task 2,1,1,0,0), miss rate 3/10, 4 repeat-guard refusals, wall p50
    20 s, pass 2 ok 5/10, no over-action, no safety event."""
    m: dict[str, Any] = {
        "success": 4,
        "per_task": {"M1": 2, "M2": 1, "M3": 1, "M4": 0, "M5": 0},
        "misses": 3,
        "labelled": 10,
        "repeat_guard": 4,
        "over_action": 0,
        "safety": 0,
        "wall_p50": Fraction(20),
        "pass2_ok": 5,
        "pass2_requests": 10,
    }
    m.update(over)
    return m


WORTH = "worth a default-on follow-up"
NO_SIGN = "no sign"
MIXED = "mixed"
BASE = _metrics()
P7 = {"M1": 2, "M2": 2, "M3": 1, "M4": 1, "M5": 1}  # 7 = base + 3, no task loses
READING_CASES: dict[str, tuple[str, dict[str, Any], str]] = {
    # (i) success >= base + 3 exactly
    "outcome-plus-3": ("outcome", _metrics(success=7, per_task=P7), WORTH),
    "outcome-plus-2-no-mechanism": (
        "outcome",
        _metrics(success=6, per_task={**P7, "M5": 0}),
        MIXED,
    ),
    # equal success, mechanism at the exact boundaries: 2/10 <= (2/3)(3/10) and 2 <= 4/2 (floats fail the first)
    "outcome-mechanism-exact": ("outcome", _metrics(misses=2, repeat_guard=2), WORTH),
    "outcome-miss-only": ("outcome", _metrics(misses=2, repeat_guard=3), NO_SIGN),
    "outcome-guard-only": ("outcome", _metrics(misses=3, repeat_guard=2), NO_SIGN),
    "outcome-miss-just-short": (
        "outcome",
        _metrics(misses=21, labelled=100, repeat_guard=2),
        NO_SIGN,
    ),
    "outcome-nothing": ("outcome", _metrics(), NO_SIGN),
    "outcome-worse-nothing": (
        "outcome",
        _metrics(success=3, per_task={**BASE["per_task"], "M1": 1}),
        NO_SIGN,
    ),
    # (ii) a task losing 2 successes (success +3 overall)
    "outcome-task-loses-2": (
        "outcome",
        _metrics(success=7, per_task={"M1": 0, "M2": 3, "M3": 2, "M4": 1, "M5": 1}),
        MIXED,
    ),
    "outcome-task-loses-1": (
        "outcome",
        _metrics(success=7, per_task={"M1": 1, "M2": 3, "M3": 1, "M4": 1, "M5": 1}),
        WORTH,
    ),
    # (iii) and (iv)
    "outcome-over-action": ("outcome", _metrics(success=7, per_task=P7, over_action=1), MIXED),
    "outcome-safety": ("outcome", _metrics(success=7, per_task=P7, safety=1), MIXED),
    # reground: miss rate cut 1/3 and pass 2 ok exactly 50 %
    "reground-mechanism-exact": (
        "reground",
        _metrics(misses=2, pass2_ok=5, pass2_requests=10),
        WORTH,
    ),
    "reground-ok-under-half": (
        "reground",
        _metrics(misses=2, pass2_ok=4, pass2_requests=9),
        NO_SIGN,
    ),
    # (v) wall p50 at exactly 1.5x base and just above
    "reground-wall-1.5x": (
        "reground",
        _metrics(success=7, per_task=P7, wall_p50=Fraction(30)),
        WORTH,
    ),
    "reground-wall-above": (
        "reground",
        _metrics(success=7, per_task=P7, wall_p50=Fraction(30) + Fraction(1, 1000)),
        MIXED,
    ),
    # the wall bar is the reground arm's only
    "outcome-wall-2x": ("outcome", _metrics(success=7, per_task=P7, wall_p50=Fraction(40)), WORTH),
}


def test_reading_cases_premise() -> None:
    """Premise (exact arithmetic, by hand): the boundary cases sit where their comments say, and floats get the
    miss boundary wrong (so a float implementation fails ``outcome-mechanism-exact``)."""
    assert Fraction(2, 10) <= Fraction(2, 3) * Fraction(3, 10)
    assert not (0.2 <= (2 / 3) * 0.3), (
        "premise: floats already get this right; pick another boundary"
    )
    assert Fraction(21, 100) > Fraction(2, 3) * Fraction(3, 10)
    assert Fraction(4, 9) < Fraction(1, 2) and Fraction(5, 10) == Fraction(1, 2)
    for _, (_, m, _) in READING_CASES.items():
        assert sum(m["per_task"].values()) == m["success"], m


@pytest.mark.parametrize("case", list(READING_CASES))
def test_reading_lines(case: str) -> None:
    """A10: the pre-registered phrase for each lever at the boundaries (hand metrics, exact Fractions)."""
    lever, arm, want = READING_CASES[case]
    got = _reading(lever, arm, BASE)
    assert want in got, (case, got)
    for other in {WORTH, NO_SIGN, MIXED} - {want}:
        assert other not in got.replace(want, ""), (case, got)


@pytest.mark.parametrize(
    ("label", "errored", "safety", "want", "absent"),
    [
        ("measured", 0, 0, [], ["provisional", "superseded"]),
        ("incomplete", 0, 0, ["provisional (run incomplete)"], ["superseded"]),
        ("void", 0, 0, ["provisional (run void)"], ["superseded"]),
        ("measured", 1, 0, [], ["provisional"]),
        ("measured", 2, 0, ["provisional (errored 2)"], ["superseded"]),
        ("measured", 0, 1, ["superseded by a safety event"], ["provisional"]),
        ("incomplete", 0, 1, ["superseded by a safety event"], ["provisional"]),
    ],
)
def test_verdict_marks(
    label: str, errored: int, safety: int, want: list[str], absent: list[str]
) -> None:
    """A9: the marks on a reading line; safety wins over incomplete (reading 5); one errored turn marks nothing."""
    line = "reground: worth a default-on follow-up"
    got = _mark(line, label, errored, safety)
    assert line.split(": ", 1)[0] in got
    for w in want:
        assert w in got, (label, errored, safety, got)
    for a in absent:
        assert a not in got, (label, errored, safety, got)


#: A9's regime fields with distinctive values (the synthetic manifest's ``regime``), by the names the harness's
#: ``manifest["regime"]`` carries (``REGIME_FIELDS``; ``test_regime_names_cover_a9`` maps them to A9's list)
REGIME: dict[str, Any] = {
    "commit": "c0ffee0" + "1" * 33,
    "profile": "lean",
    "canvas_length": 256,
    "max_denoising_steps": 48,
    "thinking": True,
    "engine_version": "vllm-T0212-0.29.0",
    "model": "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
    "model_revision": "2ea83723",
    "kv_cache": "4 GiB bf16 T0212",
    "max_num_seqs": 3,
    "prefix_caching": "on (T0212)",
    "n_per_arm": 15,
    "turns": 45,
    "order_seed": 0,
    "turn_timeout_s": 300,
    "run_budget_s": 3120,
    "hard_limit_s": 3600,
    "turn_worst_s": 480,
    "turn_worst_note": "T0212-WORST-CASE-NOTE pass 2 is bounded by REGROUND_TIMEOUT_MS",
    "report_s": 60,
    "crop_budget": 280,
    "crop_source": "default",
    "regrounding_timeout_ms": 4000,
    "chromium": "chromium-T0212-1187",
    "playwright": "playwright-T0212-1.55.0",
    "versions_lock_sha256": "f" * 63 + "e",
    "search": "off",
    "input": "typed (UserText) T0212; fake STT/TTS",
}

#: A9's list -> the harness's regime names (the arm dicts and diffs are checked apart: they are not one value)
A9_REGIME: dict[str, tuple[str, ...]] = {
    "profile": ("profile",),
    "canvas length": ("canvas_length",),
    "denoising steps": ("max_denoising_steps",),
    "vLLM version": ("engine_version",),
    "model revision": ("model_revision",),
    "KV": ("kv_cache",),
    "max_num_seqs": ("max_num_seqs",),
    "n per arm": ("n_per_arm",),
    "order seed": ("order_seed",),
    "turn timeout": ("turn_timeout_s",),
    "run budget": ("run_budget_s",),
    "hard limit": ("hard_limit_s",),
    "worst-case turn note": ("turn_worst_note",),
    "crop budget and its source": ("crop_budget", "crop_source"),
    "Chromium and Playwright versions": ("chromium", "playwright"),
    "versions.lock sha": ("versions_lock_sha256",),
    "search off": ("search",),
    "input mode": ("input",),
}


def test_regime_names_cover_a9() -> None:
    """A9: every field A9 lists has a name in the harness's ``REGIME_FIELDS``, and the synthetic regime gives each
    of those names a value."""
    names = set(api("REGIME_FIELDS"))
    for field, keys in A9_REGIME.items():
        assert set(keys) <= names, (field, sorted(set(keys) - names))
    assert names <= set(REGIME), sorted(names - set(REGIME))


# =====================================================================================================
# 5. the synthetic 45-turn run dir and its seeded faults (C2; T-0210 C-R5, T-0211 C-R14)
# =====================================================================================================


def _eval_module() -> ModuleType:
    if not EVAL.is_file():
        pytest.skip(f"T-0212 C: {EVAL.relative_to(REPO)} is absent (written in parallel)")
    from importlib import import_module

    return import_module("tests.eval.test_t0212_threejs_arms")


def _run_rows(ev: ModuleType) -> dict[str, Callable[[Any], None]]:
    """The eval's run rows: ``gpu``-marked ``test_*`` functions whose only parameter is ``run`` (selected by
    signature and marker, never called with anything but a loaded run)."""
    out = {}
    for name, fn in vars(ev).items():
        if not (name.startswith("test_") and callable(fn)):
            continue
        if list(inspect.signature(fn).parameters) != ["run"]:
            continue
        marks = {mk.name for mk in getattr(fn, "pytestmark", [])} | {
            mk.name for mk in getattr(ev, "pytestmark", []) if hasattr(mk, "name")
        }
        if "gpu" in marks:
            out[name] = fn
    return out


def _load(ev: ModuleType, path: Path) -> Any:
    assert hasattr(ev, "load_run"), "T-0212 C: the eval has no load_run(path) (the run rows' input)"
    return ev.load_run(path)


def _failing(ev: ModuleType, path: Path) -> dict[str, str]:
    run = _load(ev, path)
    failed = {}
    for name, fn in _run_rows(ev).items():
        try:
            fn(copy.deepcopy(run))
        except AssertionError as exc:
            failed[name] = repr(exc)[:300]
    return failed


def _spec(task: str, arm: str, block: int) -> dict[str, Any]:
    """What one synthetic turn does (by hand): block 2 fails (start state), blocks 0 and 1 succeed; M1 block 0 base
    is the not-run turn; M2 block 0 reground has a ReGround fallback; M3 block 0 outcome has a "no visible change"
    clause; M4 block 1 base fires the completion check (call); M5 block 0 outcome raised before any step (R-3: an
    exception, no step lines, its session log never created).

    Call ids are a fresh Session's (``t1-call``, ``t1-call-s2``, ``t1-call-s3``) in every turn (R-1). The seeded R-1
    case: every reground turn's ``t1-call`` moved (ok, executed 3 units from p1), except M2 block 0's, which fell
    back; its click is a miss (block 0). Per turn: moved 29 labelled / 9 misses, not moved 1 / 1."""
    return {
        "ok": block < 2,
        "raised": (task, arm, block) == ("M5", "outcome", 0),
        "not_run": (task, arm, block) == ("M1", "base", 0),
        "fallback": (task, arm, block) == ("M2", "reground", 0),
        "no_change_clause": (task, arm, block) == ("M3", "outcome", 0),
        "check": (task, arm, block) == ("M4", "base", 1),
        "miss": block != 1,
    }


def _synthetic_turn(
    task: str, arm: str, block: int, spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    s = _session()
    lg = Log(session=f"turn-{task}-{arm}-{block}", real_ids=True)
    voiced = "Done." if spec["ok"] else "I could not finish."
    lg.request(None)

    def pointer(name: str, x: int, y: int, **extra: Any) -> None:
        if arm == "reground":
            cid = lg.next_cid()
            # R-2: p2 in crop units (near the crop's centre), never a full-frame point
            if spec["fallback"] and lg.calls == 0:
                lg.reground(cid, "fallback", (x, y), (531, 500), name=name)
            else:
                lg.reground(cid, "ok", (x, y), (503, 498), name=name, executed=(x + 3, y))
                x += 3
        lg.call(name, x=x, y=y, **extra)

    def updated() -> str:
        if arm == "outcome":
            return _lbl(
                "updated",
                outcome="no visible change" if spec["no_change_clause"] else "the field changed",
            )
        return _lbl("updated")

    pointer("click_at", 300, 200)
    lg.request(_lbl("unchanged") if spec["miss"] else updated())
    pointer("click_at", 320, 200)  # 20 units after a miss: a re-aim
    lg.request(
        updated(), history=(_lbl("unchanged"),), continuation="I clicked.\n" + RESPONSE_UNCHANGED
    )
    pointer("type_at", 320, 200, text="5")
    lg.request(updated(), continuation="I typed.\n" + RESPONSE_UNCHANGED)
    if spec["check"]:
        lg.note(s.CHECK_ASKED_NOTE, step=lg.step)
        lg.request(_lbl("updated"), phase="completion_check")
        lg.note(s.CHECK_OUTCOME_NOTE, request_id="chk", step=lg.step, outcome="call")
    lg.add(
        "machine.step",
        {"event": {}, "commands": [{"type": "tts.speak", "text": voiced, "source": voiced}]},
    )
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
    }
    if arm != "reground":
        extras["overview_frame"] = {"shown": 0, "fits": 1, "too_tall": 0, "failed": 0, "refused": 0}
    if arm == "outcome":
        extras["action_outcome"] = {
            "none": 1 if spec["no_change_clause"] else 0,
            "near": 2,
            "elsewhere": 0,
            "page": 0,
            "unmeasured": 0,
        }
    if arm == "reground":
        fb = 1 if spec["fallback"] else 0
        extras["reground"] = {
            "requests": 3,
            "ok": 3 - fb,
            "fallback": fb,
            "skipped": 0,
            "ms": [900, 900, 900],
        }
        extras["grounding"] = {
            "visual_tokens": 1120,
            "visual_tokens_from": "default",
            "reground": True,
            "reground_from": "default",
            "reground_visual_tokens": 280,
            "reground_visual_tokens_from": "default",
        }
    lg.line(extras)
    lg.add("log.stopped", {})
    return lg.recs, {"turn_id": lg.turn_id, "session_id": f"s-{lg.session}", "extras": extras}


def _synthetic_run_dir(out: Path) -> tuple[Path, dict[str, Any]]:
    """A 45-turn run dir written with the harness's own writers (``order``, ``pa_check`` for the manifest's arm
    record, M's ``live_default``, D's ``turn_records``, M's ``annotate`` / ``_not_run``, ``finish_turn``,
    ``G._write_json`` / ``G._append_jsonl``, ``write_report``), in the manifest shape ``_measure`` writes."""
    from local_astra.pages import threejs_tasks

    h, m = harness(), _m()
    D, G = m.D, m.G
    out.mkdir(parents=True)
    (out / "sessions").mkdir()
    turns = [tuple(x) for x in h.order(h.ORDER_SEED)]
    G._write_json(out / "order.json", [list(k) for k in turns])
    lean = _lean()
    pa = h.pa_check(lean)
    manifest = {
        "regime": dict(REGIME),
        "live_default": dict(m.live_default(lean)),
        "arm_keywords": pa["keywords"],
        "arm_diffs": pa["diffs"],
        "arm_grounding": pa["grounding"],
        "tool_guidance_sha256": pa["tool_guidance_sha256"],
        "search": "off",
        "task_table_sha256": threejs_tasks.task_table_sha(),
        "provenance_sha256": m.provenance_sha256(),
        "cli_before": h.cli_defaults(),
        "stopped": None,
        "cli_after": h.cli_defaults(),
    }
    G._write_json(out / "manifest.json", manifest)
    trials: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    for task, arm, block in turns:
        spec = _spec(task, arm, block)
        if spec["not_run"]:
            nr, _ = h.finish_turn(
                out, task, arm, block, m._not_run(task, block, "harness: RuntimeError('died')"), []
            )
            trials.append(nr)
            continue
        if spec["raised"]:
            # R-3: the turn raised before its first step: no records, no turn line, no step lines, and the
            # session log it names was never created
            trial, st = D.turn_records(
                task, block, records=[], exception="RuntimeError('T-0212 rig: page never ready')"
            )
            trial, st = m.annotate(task, trial, st, [], None)
            trial.update(
                resolver_calls=0,
                connector_calls=0,
                egress_log=[],
                wall_s=10.0 + len(trials),
                engine_up_after=True,
                session_log=f"turn-{task}-{arm}-{block}.jsonl",
            )
            assert st == [], st
            trial, st = h.finish_turn(out, task, arm, block, dict(trial), [])
            trials.append(trial)
            continue
        records, line = _synthetic_turn(task, arm, block, spec)
        state = _st(task, spec["ok"])
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
        trial, st = h.finish_turn(out, task, arm, block, dict(trial), [dict(s) for s in st])
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
    h.write_report(out, probes, manifest, None)
    return out, manifest


def _rw_json(path: Path, edit: Callable[[Any], None]) -> None:
    v = json.loads(path.read_text(encoding="utf-8"))
    edit(v)
    path.write_text(json.dumps(v), encoding="utf-8")


def _rw_jsonl(path: Path, edit: Callable[[list[dict[str, Any]]], None]) -> None:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    edit(rows)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _walk(obj: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], Any]]:
    yield path, obj
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, (*path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, (*path, i))


def _set(obj: Any, path: tuple[Any, ...], value: Any) -> None:
    for p in path[:-1]:
        obj = obj[p]
    obj[path[-1]] = value


def _bump_int_key(report: Any, pattern: str, delta: int, arm: str) -> None:
    hits = [
        (p, v)
        for p, v in _walk(report)
        if p
        and isinstance(p[-1], str)
        and re.fullmatch(pattern, p[-1])
        and isinstance(v, int)
        and not isinstance(v, bool)
        and arm in [str(x) for x in p]
        and v + delta >= 0
    ]
    assert hits, f"T-0212 rig: report.json has no int {pattern!r} under arm {arm!r}"
    for p, v in hits:
        _set(report, p, v + delta)


def _f_reground_records(out: Path) -> None:
    name = "turn-M3-reground-1.jsonl"
    _rw_jsonl(
        out / "sessions" / name,
        lambda rows: rows.__setitem__(
            slice(None), [r for r in rows if r.get("kind") != "tool.reground"]
        ),
    )


def _f_clause_into_base(out: Path) -> None:
    stem = _session().OUTCOME_CLAUSE.split("{outcome}", 1)[0]

    def edit(rows: list[dict[str, Any]]) -> None:
        for r in rows:
            if r.get("kind") == "llm.request" and r["data"].get("phase") == "tool_step":
                lbl = r["data"]["live_label"]["text"]
                new = lbl[:-1] + stem + "no visible change.]"
                r["data"]["live_label"]["text"] = new
                for msg in r["data"]["prompt"]["messages"]:
                    for part in msg["content"]:
                        if part.get("text") == lbl:
                            part["text"] = new
                return

    _rw_jsonl(out / "sessions" / "turn-M2-base-1.jsonl", edit)


def _f_success_flip(out: Path) -> None:
    def edit(rows: list[dict[str, Any]]) -> None:
        for r in rows:
            if (r["task"], r.get("arm"), r["repeat"]) == ("M5", "outcome", 2):
                r["success"] = True

    _rw_jsonl(out / "trials.jsonl", edit)


def _f_log_removed(out: Path) -> None:
    (out / "sessions" / "turn-M4-outcome-0.jsonl").unlink()


def _f_miss_count(out: Path) -> None:
    _rw_json(out / "report.json", lambda r: _bump_int_key(r, r"(click_)?miss(es)?", 1, "base"))


def _f_errored_drop(out: Path) -> None:
    _rw_json(out / "report.json", lambda r: _bump_int_key(r, r"errored", -1, "base"))


def _f_regime_md(out: Path) -> None:
    p = out / "report.md"
    md = p.read_text(encoding="utf-8")
    assert REGIME["chromium"] in md, "T-0212 rig: report.md never named the Chromium version"
    p.write_text(md.replace(REGIME["chromium"], ""), encoding="utf-8")


def _f_regime_none(out: Path) -> None:
    """R-5: ``kv_cache`` None in both the manifest and the report (they agree, and report.md still names the key)."""
    _rw_json(out / "manifest.json", lambda m: m["regime"].__setitem__("kv_cache", None))
    _rw_json(out / "report.json", lambda r: r["regime"].__setitem__("kv_cache", None))


def _f_manifest_reground(out: Path) -> None:
    def edit(man: Any) -> None:
        hits = [
            p
            for p, v in _walk(man)
            if isinstance(v, dict) and v.get("reground") is True and len(v) > 3
        ]
        assert hits, "T-0212 rig: the manifest holds no reground keyword dict"
        for p in hits:
            target = man
            for k in p:
                target = target[k]
            target["action_outcome"] = True

    _rw_json(out / "manifest.json", edit)


FAULTS: dict[str, tuple[Callable[[Path], None], str]] = {
    "a-reground-records-deleted": (_f_reground_records, "test_arm_evidence_re_derived"),
    "b-clause-in-base": (_f_clause_into_base, "test_arm_evidence_re_derived"),
    "c-success-flipped": (_f_success_flip, "test_success_re_judged"),
    "d-session-log-removed": (_f_log_removed, "test_session_logs_present_and_readable"),
    "e-click-miss-off-by-one": (_f_miss_count, "test_metrics_re_derived_from_session_logs"),
    "f-not-run-dropped": (_f_errored_drop, "test_metrics_re_derived_from_session_logs"),
    "g-regime-field-removed": (_f_regime_md, "test_manifest_regime"),
    "h-manifest-reground-gains-outcome": (
        _f_manifest_reground,
        "test_manifest_arms_differ_only_in_lever",
    ),
    "i-regime-value-none": (_f_regime_none, "test_manifest_regime"),
}


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out, _ = _synthetic_run_dir(tmp_path_factory.mktemp("t0212-run") / "t0212-threejs-arms-0000000")
    return out


def test_report_regime_fields(synthetic_run: Path) -> None:
    """A9: ``report.json`` and ``report.md`` carry every regime field (the manifest's values), the three arm
    keyword dicts' diffs, the near-miss "not available" statement and the click-miss caveat."""
    report = json.loads((synthetic_run / "report.json").read_text(encoding="utf-8"))
    md = (synthetic_run / "report.md").read_text(encoding="utf-8")
    flat = json.dumps(report, default=str)
    for k, v in REGIME.items():
        assert str(v) in md, f"A9: report.md lacks {k} ({v!r})"
        assert json.dumps(v)[1:-1] in flat or str(v) in flat, f"A9: report.json lacks {k} ({v!r})"
    # A9: "the three arm keyword dicts and their diffs" are regime fields of report.json too, not only of
    # manifest.json and report.md (a (c) row at dfa98a73: report.json's regime carries neither)
    man = json.loads((synthetic_run / "manifest.json").read_text(encoding="utf-8"))
    values = [v for _, v in _walk(report)]
    assert man["arm_keywords"] in values, "A9: report.json lacks the three arm keyword dicts"
    assert man["arm_diffs"] in values, "A9: report.json lacks the arm diffs"
    assert "not available" in md, "A8: the true near-miss distance is not stated as unavailable"
    assert "n = 15 per arm; a follow-up A/B at larger n decides" in md
    assert "1/15" in md, "A10: the T-0210 C noise anchor (1/15) is missing"


def test_synthetic_run_dir_premise(synthetic_run: Path) -> None:
    """C2 premise: 45 trials (one not-run; one that raised before any step, R-3), 43 session logs (the raised
    turn's named log is absent), label ``measured``, every voiced text verbatim in report.md, and every turn's
    first call id is ``t1-call`` (R-1: ids repeat across turns, as in real logs)."""
    tr = _trials(synthetic_run)
    assert len(tr) == 45
    assert sum(1 for t in tr if "engine_up_after" in t and t["engine_up_after"] is None) == 1
    raised = [t for t in tr if t.get("exception") and t.get("engine_up_after") is True]
    assert [(t["task"], t["arm"], t["repeat"]) for t in raised] == [("M5", "outcome", 0)], raised
    assert (
        raised[0]["session_log"]
        and not (synthetic_run / "sessions" / raised[0]["session_log"]).exists()
    )
    steps = [json.loads(x) for x in (synthetic_run / "steps.jsonl").read_text().splitlines() if x]
    assert not [s for s in steps if (s["task"], s["arm"], s["repeat"]) == ("M5", "outcome", 0)]
    logs = sorted((synthetic_run / "sessions").glob("*.jsonl"))
    assert len(logs) == 43
    firsts = set()
    for f in logs:
        recs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        firsts.add(next(r["data"]["call_id"] for r in recs if r["kind"] == "tool.call"))
    assert firsts == {"t1-call"}, firsts
    report = json.loads((synthetic_run / "report.json").read_text(encoding="utf-8"))
    assert report.get("label") == "measured", report.get("label")
    md = (synthetic_run / "report.md").read_text(encoding="utf-8")
    assert "Done." in md and "I could not finish." in md


def test_synthetic_split_is_per_turn(synthetic_run: Path) -> None:
    """R-1 (blocking) over the clean dir: the eval's reground split equals the harness's (``report.json``) and the
    hand count (moved 29 labelled / 9 misses; not moved 1 / 1: M2 block 0's ``t1-call`` fell back while every other
    reground turn's ``t1-call`` moved). Pooling ids across turns gives moved 30 / 10, not moved 0 / 0."""
    ev = _eval_module()
    run = _load(ev, synthetic_run)
    arms, _ = ev.all_metrics(ev.views(run))
    want = {"moved": {"labelled": 29, "misses": 9}, "not_moved": {"labelled": 1, "misses": 1}}
    assert arms["reground"]["reground"]["split"] == want, arms["reground"]["reground"]["split"]
    got = run["report"]["per_arm"]["reground"]["reground"]["split"]
    assert {b: {k: got[b][k] for k in ("labelled", "misses")} for b in want} == want, got


def test_every_eval_run_row_passes_on_a_harness_built_run_dir(synthetic_run: Path) -> None:
    """C2 (the too-wide direction): every gpu ``test_*(run)`` row of the eval passes on the clean dir, and every
    row has a fault aimed at it or is listed by name in the eval's docstring rows."""
    ev = _eval_module()
    rows = _run_rows(ev)
    assert len(rows) >= 8, sorted(rows)
    assert {r for _, r in FAULTS.values()} <= set(rows), sorted(
        {r for _, r in FAULTS.values()} - set(rows)
    )
    assert _failing(ev, synthetic_run) == {}


@pytest.mark.parametrize("fault", list(FAULTS))
def test_each_seeded_fault_fails_its_row(synthetic_run: Path, tmp_path: Path, fault: str) -> None:
    """C2 (the absent direction): each seeded fault fails the eval row named for it."""
    ev = _eval_module()
    how, row = FAULTS[fault]
    assert row not in _failing(ev, synthetic_run), (fault, row)
    out = tmp_path / synthetic_run.name
    shutil.copytree(synthetic_run, out)
    how(out)
    failed = _failing(ev, out)
    print(f"T-0212 C2 {fault}: failing rows {sorted(failed)}")
    assert row in failed, (fault, row, sorted(failed))


# =====================================================================================================
# A11: the T-0210 C anchor (local log)
# =====================================================================================================


def _anchor_dir() -> Path | None:
    for root in (
        REPO,
        REPO.parent.parent.parent,
    ):  # the worktree, then the main checkout above .claude/worktrees
        d = root / ANCHOR_REL
        if (d / "sessions").is_dir():
            return d
    return None


def test_t0210c_anchor(capsys: pytest.CaptureFixture[str]) -> None:
    """A11 (report-only; skipped without the local log): the A8 label, miss and re-aim functions run over every
    T-0210 C session log without error; the totals are printed as the base anchor."""
    d = _anchor_dir()
    if d is None:
        pytest.skip(f"{ANCHOR_REL} is absent (a local log, not in the repo)")
    h = harness()
    tot = {"unchanged": 0, "misses": 0, "labelled": 0, "unlabelled": 0, "reaims": 0}
    files = sorted((d / "sessions").glob("*.jsonl"))
    assert files
    for f in files:
        recs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        tot["unchanged"] += int(h.unchanged_labels(recs))
        cm = h.click_misses(recs)
        for k in ("misses", "labelled", "unlabelled"):
            tot[k] += int(cm[k])
        tot["reaims"] += int(h.reaims(recs))
    assert tot["misses"] <= tot["labelled"]
    with capsys.disabled():
        print(f"\nT-0212 A11 anchor ({d.name}, {len(files)} logs): {tot}")
