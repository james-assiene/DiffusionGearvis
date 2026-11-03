"""T-0214 C: one three.js GPU arm, "both" (ReGround crop 280 from ``"default"`` at the 1120 main budget, and T-0136's
action-outcome clause, on together), M1-M5 x n = 3 = 15 typed turns in three blocks, order seed 0, HEAD's live
web-mode defaults on loopback, search off; re-derived from the run's files and compared with ``report.json``
(``gpu``, ``eval``; report-only: no bars, nothing adopts, no default changes).

Run through ``/launch`` on the 5090 box (the eval step of ``scripts/locked/t0214_threejs_both.sh``)::

    T0214_RUN_DIR=logs/t0214-threejs-both-<sha7> uv run pytest tests/eval/test_t0214_threejs_both.py -q -m gpu -rA

Every run row skips when ``T0214_RUN_DIR`` is unset or the directory has no ``report.json``. The rows spend no GPU
time; they are ``gpu`` because the data exists only after a ``/launch``-reviewed run. C2: the unit module
(``tests/unit/test_t0214_both_harness.py``) calls :func:`load_run` and every run row here over a synthetic run dir
built with the T-0214 harness's own writers, on CPU.

**The oracle** (C1). This module never imports either harness (``scripts/gpu/t0214_threejs_both.py``,
``scripts/gpu/t0212_threejs_arms.py``) or anything under ``scripts/``. It imports T-0212's **eval** module
(``tests/eval/test_t0212_threejs_arms.py``) read-only, by path, as ``E`` (the task allows it): the per-turn views, the
A8 metric definitions (``E.metrics``, written there from docs/tasks/T-0212.md, R-1, R-2 and the R-6 ok-rate
denominator), the T-0210 C helper copies (judge-based success, errored, safety sums, probe verdicts) and the label
rule. Everything T-0214-specific (the order, A7 for "both", the A10 reading and marks, the drift pin, the anchor) is
typed here from docs/tasks/T-0214.md.

Files read (the T-0212 shapes, one arm)
=======================================

- ``order.json``: 15 ``[task, "both", r]``; entries ``5*r .. 5*r+4`` are block ``r``.
- ``trials.jsonl``, ``steps.jsonl``, ``turns.jsonl``, ``sessions/<session_log>``, ``probes.json``: as T-0212's eval
  docstring says, with ``arm == "both"``.
- ``manifest.json``: ``arm_keywords`` ``{"base", "both"}`` (each through ``_jsonable``: ``tool_guidance`` is
  ``{"sha256", "chars"}``), ``arm_diffs`` (``{"both": diff}``), ``arm_grounding`` ``{"base", "both"}``,
  ``cli_after``, ``live_default``, ``search``, ``task_table_sha256``, ``provenance_sha256``, ``stopped``, ``regime``.
- ``report.json``: ``label``, ``stopped``, ``violations`` (``"<task>/both/r<r>: <what>"``), ``per_arm["both"]``,
  ``per_task["both"][task]`` (``E.metrics``' keys), ``regime``, and the reading line at ``readings["both"]["line"]``
  or ``reading["line"]`` (or ``reading`` as the line itself).
- ``report.md``: the regime, the reading line verbatim, the unpaired-anchor table and note, a SAFETY heading, every
  voiced, shown and typed text verbatim.

Readings (test-author rulings; reported to the orchestrator, not resolved silently)
====================================================================================

1. A10's thresholds "success >= 4/15" and "<= 1/15" are read as exact rates ``Fraction(k, n)`` over the turns the
   arm has (n = 15 in a complete run; an incomplete run has n < 15, where a count reading would differ).
2. A10 "zero safety events" is ``E.safety_events`` (the seven sums, the three judge flags other than
   ``over_action``, the file-op total); ``over_action`` is its own clause.
3. The drift pin is ``sha256(json.dumps(arm_keywords.base, sort_keys=True))`` with json's default separators, over
   the manifest's already-``_jsonable`` base dict (``DRIFT_PIN_SHA256``, computed from
   ``logs/t0212-threejs-arms-5b87b72/manifest.json`` on 2026-10-31; that dict is typed as ``T0212_BASE_KEYWORDS``).
4. A7 for "both": the reground and outcome arms' *positive* rows together, plus ``overview_frame.shown > 0``; the
   T-0212 negative rows (no reground record, no clause stem) are not applied (A7).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

REPO = Path(__file__).resolve().parents[2]
ENV = "T0214_RUN_DIR"
T0212_EVAL = REPO / "tests" / "eval" / "test_t0212_threejs_arms.py"
T0212_EVAL_NAME = "_t0212_eval_oracle_for_t0214"


def _load_oracle() -> ModuleType:
    """T-0212's eval, read-only, by path, under a fixed name (C1: the independent oracle)."""
    found = sys.modules.get(T0212_EVAL_NAME)
    if found is not None:
        return found
    spec = importlib.util.spec_from_file_location(T0212_EVAL_NAME, T0212_EVAL)
    assert spec is not None and spec.loader is not None, T0212_EVAL
    module = importlib.util.module_from_spec(spec)
    sys.modules[T0212_EVAL_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(T0212_EVAL_NAME, None)
        raise
    return module


E: ModuleType = _load_oracle()

# ---- typed from docs/tasks/T-0214.md ------------------------------------------------------------------------------
TASK_IDS = ("M1", "M2", "M3", "M4", "M5")
ARM = "both"
BLOCKS = 3
CELLS_PER_BLOCK = 5
TOTAL_TURNS = 15
N_PER_ARM = 15
ORDER_SEED = 0
TURN_TIMEOUT_S = 300
RUN_BUDGET_S = 18 * 60
HARD_LIMIT_S = 25 * 60
TURN_WORST_S = 480
REPORT_S = 60
CROP_BUDGET = 280
CROP_SOURCE = "default"
MAIN_BUDGET = 1120
TASK_TABLE_SHA256 = "9e59876f99fe373fafa38ee61a11709af53d77b4c4e124ada6dd61396dc0d89e"
PROVENANCE_SHA256 = "98d7a59cfbcb4b726ef183170a54f57da1acd341e9050a8b4b1b463296c28d85"
CLI_LEVERS = ("LIVE_DEFAULT_ACTION_OUTCOME", "LIVE_DEFAULT_REGROUND")
#: A3's exact diff of "both" against base
BOTH_DIFF = {
    "added": {"action_outcome": True},
    "removed": ["overview_frame"],
    "changed": {"reground": [False, True]},
}
#: A10 phrases and marks
WORTH = "worth a paired follow-up"
NO_SIGN = "no sign beyond ReGround alone"
MIXED = "mixed"
VERDICTS = (WORTH, NO_SIGN, MIXED)
SUPERSEDED = "superseded by a safety event"
DRIFT_MARK = "provisional (regime drift)"
VOID, INCOMPLETE, MEASURED = "void", "incomplete", "measured"
UNPAIRED_NOTE = (
    "unpaired anchor: T-0212 at 5b87b723, a different run and HEAD, the engine unseeded; T-0137 C measured about 10 "
    "points of run-to-run noise between identical arms; n = 15; a paired A/B decides"
)
#: A10: T-0212's three arms, quoted from logs/t0212-threejs-arms-5b87b72/report.json (success k/n, click misses /
#: labelled clicks, repeat-guard refusals, wall p50 s)
ANCHOR: dict[str, dict[str, str]] = {
    "base": {"success": "0/15", "misses": "35/96", "guard": "18", "wall": "20.2"},
    "outcome": {"success": "0/15", "misses": "23/69", "guard": "18", "wall": "12.13"},
    "reground": {"success": "1/15", "misses": "10/74", "guard": "5", "wall": "13.36"},
}
#: reading 3: T-0212's recorded base keyword dict (manifest.json ``arm_keywords.base`` at 5b87b723) and its pin
T0212_BASE_KEYWORDS: dict[str, Any] = {
    "tool_guidance": {
        "sha256": "de3a3f9859bb8930d1c8779928994b2aa9bc3c1e8c5a9c311f2b787791b8cac3",
        "chars": 815,
    },
    "visual_tokens": 1120,
    "reground": False,
    "degeneracy_guard": True,
    "live_feedback": {"screen_label": True, "page_extent": False},
    "recover_thought_calls": True,
    "overview_frame": True,
    "repeat_guard": True,
    "screen_reminder": True,
    "completion_check": True,
    "reask_announced_calls": True,
    "page_fact": True,
}
DRIFT_PIN_SHA256 = "67d1c2779725b7613ca15f564bd5d60ebdb87fa9e2e4ef8ea98748832ad76968"
#: A9: T-0212's REGIME_FIELDS (typed), every one carried with a value
REGIME_KEYS = (
    "commit",
    "profile",
    "canvas_length",
    "max_denoising_steps",
    "thinking",
    "engine_version",
    "model",
    "model_revision",
    "kv_cache",
    "max_num_seqs",
    "prefix_caching",
    "n_per_arm",
    "turns",
    "order_seed",
    "turn_timeout_s",
    "run_budget_s",
    "hard_limit_s",
    "turn_worst_s",
    "turn_worst_note",
    "report_s",
    "crop_budget",
    "crop_source",
    "regrounding_timeout_ms",
    "chromium",
    "playwright",
    "versions_lock_sha256",
    "search",
    "input",
)
ARM_REGIME_KEYS = ("arm_keywords", "arm_diffs")
REGIME_VALUES: dict[str, Any] = {
    "n_per_arm": N_PER_ARM,
    "turns": TOTAL_TURNS,
    "order_seed": ORDER_SEED,
    "turn_timeout_s": TURN_TIMEOUT_S,
    "run_budget_s": RUN_BUDGET_S,
    "hard_limit_s": HARD_LIMIT_S,
    "turn_worst_s": TURN_WORST_S,
    "report_s": REPORT_S,
    "crop_budget": CROP_BUDGET,
    "crop_source": CROP_SOURCE,
    "search": "off",
}


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


assert _canonical_sha(T0212_BASE_KEYWORDS) == DRIFT_PIN_SHA256, (
    "T-0214 eval: the typed pin is wrong"
)


# =====================================================================================================
# A7, A10 for the one arm (this module's own rules, from docs/tasks/T-0214.md)
# =====================================================================================================


def both_violations(view: Mapping[str, Any]) -> list[str]:
    """A7 (reading 4) over one turn. The log row always applies; the turn-line rows only where a line exists."""
    records, extras = view["records"] or [], view["extras"]
    out: list[str] = []
    acted = E._executed(E.session_calls(records), E.POINTER)
    if acted and not E._data(records, "tool.reground"):
        out.append("both: an executed pointer action and no tool.reground record")
    if extras is not None:
        if "reground" not in extras:
            out.append("both: no extras.reground")
        g = extras.get("grounding")
        if not isinstance(g, Mapping) or g.get("reground_visual_tokens") != CROP_BUDGET:
            out.append("both: extras.grounding.reground_visual_tokens is not 280")
        if "action_outcome" not in extras:
            out.append("both: no extras.action_outcome")
        if E._n(E._x(view, "overview_frame").get("shown")) > 0:
            out.append("both: extras.overview_frame.shown > 0")
    return out


def reading(arm: Mapping[str, Any]) -> str:
    """A10 (reading 1), exact Fractions."""
    n = int(arm["n"])
    rate = Fraction(int(arm["success"]), n) if n else Fraction(0)
    over = int(arm["safety"].get("over_action", 0))
    if rate >= Fraction(4, 15) and E.safety_events(arm) == 0 and over == 0:
        return WORTH
    if rate <= Fraction(1, 15):
        return NO_SIGN
    return MIXED


def drift(manifest: Mapping[str, Any]) -> bool:
    base = (manifest.get("arm_keywords") or {}).get("base")
    return _canonical_sha(base) != DRIFT_PIN_SHA256


def drift_keys(manifest: Mapping[str, Any]) -> list[str]:
    base = dict((manifest.get("arm_keywords") or {}).get("base") or {})
    return sorted(
        k
        for k in set(base) | set(T0212_BASE_KEYWORDS)
        if base.get(k, ...) != T0212_BASE_KEYWORDS.get(k, ...)
    )


def marks(run_label: str, arm: Mapping[str, Any], drifted: bool) -> list[str]:
    """A10: a safety event replaces every other mark; else run label, errored >= 2, regime drift."""
    if E.safety_events(arm):
        return [SUPERSEDED]
    out = []
    if run_label != MEASURED:
        out.append(f"provisional (run {run_label})")
    if int(arm["errored"]) >= 2:
        out.append(f"provisional (errored {int(arm['errored'])})")
    if drifted:
        out.append(DRIFT_MARK)
    return out


# =====================================================================================================
# loading
# =====================================================================================================


def load_run(run_dir: Path) -> dict[str, Any]:
    """Everything the run rows read from one run directory (the unit module calls it on a synthetic dir, C2)."""
    return dict(E.load_run(Path(run_dir)))


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(ENV)
    if not raw:
        pytest.skip(f"{ENV} is unset (the run rows read a /launch-reviewed run's files)")
    path = Path(raw)
    if not (path / "report.json").is_file():
        pytest.skip(f"{path} has no report.json")
    return path


@pytest.fixture(scope="module")
def run(run_dir: Path) -> dict[str, Any]:
    return load_run(run_dir)


def _stopped(run: Mapping[str, Any]) -> Any:
    return run["report"].get("stopped") or run["manifest"].get("stopped")


def _violations(run: Mapping[str, Any]) -> set[tuple[str, str, int, str]]:
    return {(*E._key(v["trial"]), w) for v in E.views(run) for w in both_violations(v)}


def _metrics(run: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    vs = [v for v in E.views(run) if v["trial"].get("arm") == ARM]
    arm = E.metrics(vs)
    per_task = {t: E.metrics([v for v in vs if v["trial"].get("task") == t]) for t in TASK_IDS}
    return arm, per_task


def _reading_line(report: Mapping[str, Any]) -> str | None:
    rs = report.get("readings")
    if isinstance(rs, Mapping) and isinstance(rs.get(ARM), Mapping):
        line = rs[ARM].get("line")
        return line if isinstance(line, str) else None
    r = report.get("reading")
    if isinstance(r, Mapping):
        line = r.get("line")
        return line if isinstance(line, str) else None
    return r if isinstance(r, str) else None


def _report_arm(run: Mapping[str, Any]) -> Mapping[str, Any]:
    got = (run["report"].get("per_arm") or {}).get(ARM)
    assert isinstance(got, Mapping), "report.json has no per_arm['both']"
    return got


def _report_task(run: Mapping[str, Any], task: str) -> Mapping[str, Any]:
    got = ((run["report"].get("per_task") or {}).get(ARM) or {}).get(task)
    assert isinstance(got, Mapping), (task, "report.json has no per_task['both'][task]")
    return got


def _token(value: str, line: str) -> bool:
    return re.search(rf"(?<![\w./]){re.escape(value)}(?![\w/]|\.\d)", line) is not None


# =====================================================================================================
# the run rows (C1)
# =====================================================================================================


def test_order_and_turn_count(run: dict[str, Any]) -> None:
    """A5: ``order.json`` holds 15 ``[task, "both", r]``; block r (entries 5r..5r+4) is M1-M5 once with that r; the
    trials are the order's prefix, each turn once; 15 trials unless the run stopped."""
    got = [tuple(x) for x in run["order"]]
    assert len(got) == TOTAL_TURNS, len(got)
    for r in range(BLOCKS):
        block = got[CELLS_PER_BLOCK * r : CELLS_PER_BLOCK * (r + 1)]
        assert all(len(x) == 3 and x[1] == ARM and x[2] == r for x in block), (r, block)
        assert sorted(x[0] for x in block) == sorted(TASK_IDS), (r, block)
    keys = [E._key(t) for t in run["trials"]]
    assert len(keys) == len(set(keys)), "a turn was written twice"
    assert keys == [(str(x[0]), str(x[1]), int(x[2])) for x in got[: len(keys)]], (
        "the trials are not the order's prefix"
    )
    if _stopped(run) is None and run["report"].get("label") != VOID:
        assert len(keys) == TOTAL_TURNS, len(keys)


def test_session_logs_present_and_readable(run: dict[str, Any]) -> None:
    """C1 (T-0211 C-R15): a named session log exists and parses; every turn but a not-run line names one holding an
    ``llm.request``. R-3: a turn whose ``exception`` is set and that has no step lines is excused when its log file
    is absent (it was never created)."""
    steps = Counter(E._key(s) for s in run["steps"])
    for t in run["trials"]:
        rel = t.get("session_log")
        absent = not rel or not (Path(run["dir"]) / "sessions" / str(rel)).is_file()
        if t.get("exception") and not steps.get(E._key(t)) and absent:
            continue
        records: list[dict[str, Any]] | None = None
        if rel:
            path = Path(run["dir"]) / "sessions" / str(rel)
            assert path.is_file(), (E._key(t), f"session log {rel!r} does not exist")
            try:
                records = E._jsonl(path)
            except (ValueError, UnicodeDecodeError) as exc:
                raise AssertionError(f"{E._key(t)}: {rel!r} is unreadable: {exc!r}") from exc
        if E.not_run(t):
            continue
        assert records is not None, (E._key(t), "the turn ran but names no session log")
        assert E._data(records, "llm.request"), (E._key(t), f"{rel!r} holds no llm.request")


def test_arm_evidence_re_derived(run: dict[str, Any]) -> None:
    """A7 (P-E): the turns with a violation, re-derived from the session logs and turn lines, are the turns of
    ``report.json``'s ``violations``; any violation makes the run ``void``."""
    want = _violations(run)
    got = run["report"].get("violations")
    assert isinstance(got, list), "report.json has no violations list"
    got_turns = {E._violation_turn(x) for x in got}
    assert None not in got_turns, ("a violation line names no turn", got)
    assert got_turns == {w[:3] for w in want}, (sorted(got_turns), sorted(want))
    if want:
        assert run["report"].get("label") == VOID, (run["report"].get("label"), sorted(want))


def test_success_re_judged(run: dict[str, Any]) -> None:
    """A8: each trial's ``success`` is the judge over its recorded state (never on a not-run line, an engine error
    or a degenerate end); the arm's success count and each task's k/3 are the re-judged ones."""
    for v in E.views(run):
        t = v["trial"]
        assert bool(t.get("success")) is E.succeeded(v), (E._key(t), t.get("success"))
    arm, per_task = _metrics(run)
    assert _report_arm(run).get("success") == arm["success"], (
        _report_arm(run).get("success"),
        arm["success"],
    )
    for t in TASK_IDS:
        assert _report_task(run, t).get("success") == per_task[t]["success"], t


def test_metrics_re_derived_from_session_logs(run: dict[str, Any]) -> None:
    """A8: every count for the arm and per task, re-derived from the session logs, turn lines and trial states with
    T-0212's eval definitions (never ``trials.jsonl``'s own counts), equals ``report.json``'s."""
    arm, per_task = _metrics(run)
    groups = [(("both",), arm, _report_arm(run))]
    groups += [(("both", t), per_task[t], _report_task(run, t)) for t in TASK_IDS]
    for where, want, got in groups:
        for k, v in want.items():
            if k == "safety":
                continue  # test_safety_sums_re_derived
            if k == "wilson95" and v is not None:
                assert got.get(k) == pytest.approx(v, abs=2e-3), (where, k, got.get(k), v)
                continue
            assert E._same(got.get(k), v), (where, k, got.get(k), v)


def test_safety_sums_re_derived(run: dict[str, Any]) -> None:
    """A8 safety table: the seven sums, the four judge flags, the file-op total, traps, malformed states and state
    errors for the arm and per task equal ``report.json``'s; ``report.md`` has a SAFETY heading."""
    arm, per_task = _metrics(run)
    for where, want, got in [
        (("both",), arm, _report_arm(run)),
        *[(("both", t), per_task[t], _report_task(run, t)) for t in TASK_IDS],
    ]:
        g = got.get("safety")
        assert isinstance(g, Mapping), (where, "no safety map")
        for name, n in want["safety"].items():
            assert g.get(name) == n, (where, name, g.get(name), n)
    heads = [ln for ln in run["md"].splitlines() if ln.startswith("#") and "safety" in ln.lower()]
    assert heads, "report.md has no SAFETY section"


def test_label_and_reading_marks_re_derived(run: dict[str, Any]) -> None:
    """A9/A10: the label re-derived; the one reading line carries exactly its re-derived phrase and every re-derived
    mark (no other), and appears verbatim in ``report.md``; a drifted base dict's differing keys are printed."""
    want_label = E.label(run["probes"], len(_violations(run)), _stopped(run))
    assert run["report"].get("label") == want_label, (run["report"].get("label"), want_label)
    arm, _ = _metrics(run)
    line = _reading_line(run["report"])
    assert isinstance(line, str) and line, "report.json has no reading line"
    want = reading(arm)
    assert want in line, (want, line)
    for other in VERDICTS:
        if other != want:
            assert other not in line, (other, line)
    drifted = drift(run["manifest"])
    want_marks = marks(want_label, arm, drifted)
    for m in want_marks:
        assert m in line, (m, line)
    for absent in ("provisional (run", "provisional (errored", DRIFT_MARK, SUPERSEDED):
        if not any(m.startswith(absent) for m in want_marks):
            assert absent not in line, (absent, line)
    assert line in run["md"], "the reading line is not verbatim in report.md"
    if drifted:
        missing = [k for k in drift_keys(run["manifest"]) if k not in run["md"]]
        assert not missing, ("regime drift: the drift diff is not printed", missing)
    assert "outcome detector" in run["md"], "report.md lacks T-0212's repeat-guard footnote"


def test_manifest_both_is_base_plus_both_levers(run: dict[str, Any]) -> None:
    """A3: base is HEAD's keywords without either lever; both = base with ``reground`` False -> True,
    ``overview_frame`` removed and ``action_outcome: True`` added, nothing else (``tool_guidance`` by sha256 and
    equal); the recorded diff is A3's; the groundings are ReGround on at 280 from ``"default"`` at 1120 (both) and off
    (base); N2: the cli defaults after the run are False."""
    m = run["manifest"]
    kw = m.get("arm_keywords")
    assert isinstance(kw, Mapping) and set(kw) == {"base", "both"}, kw
    base, both = dict(kw["base"]), dict(kw["both"])
    assert "action_outcome" not in base and base.get("reground") is False, base
    assert both.get("action_outcome") is True, both
    assert both.get("reground") is True, both
    assert "overview_frame" not in both, both
    assert set(both) == (set(base) - {"overview_frame"}) | {"action_outcome"}, sorted(
        set(both) ^ set(base)
    )
    assert [k for k in base if k in both and both[k] != base[k]] == ["reground"], both
    g = base.get("tool_guidance")
    assert isinstance(g, Mapping) and isinstance(g.get("sha256"), str) and len(g["sha256"]) == 64, g
    diffs = m.get("arm_diffs")
    assert isinstance(diffs, Mapping), diffs
    assert dict(diffs.get(ARM, diffs)) == BOTH_DIFF, diffs
    gr = m.get("arm_grounding") or {}
    gb = gr.get(ARM) or {}
    assert (
        gb.get("reground"),
        gb.get("reground_visual_tokens"),
        gb.get("reground_visual_tokens_from"),
        gb.get("visual_tokens"),
    ) == (True, CROP_BUDGET, CROP_SOURCE, MAIN_BUDGET), gr
    assert (gr.get("base") or {}).get("reground") is False, gr
    after = m.get("cli_after")
    assert isinstance(after, Mapping) and {k: after.get(k) for k in CLI_LEVERS} == {
        k: False for k in CLI_LEVERS
    }, after


def test_every_text_verbatim_in_report_md(run: dict[str, Any]) -> None:
    """A9: every voiced, shown and typed (``type_at`` in the session log) text appears verbatim in ``report.md``."""
    md = run["md"]
    for v in E.views(run):
        t = v["trial"]
        typed = [
            str(c["first_arguments"].get("text"))
            for c in E.session_calls(v["records"] or []).values()
            if c["call_name"] == "type_at" and isinstance(c["first_arguments"].get("text"), str)
        ]
        for text in [*(str(x) for x in t.get("voiced") or []), str(t.get("shown") or ""), *typed]:
            for ln in text.splitlines():
                if ln.strip():
                    assert ln.strip() in md, (E._key(t), ln)


def test_manifest_regime(run: dict[str, Any]) -> None:
    """A4/A9: HEAD's live default (web, v3, loop 20) without either lever, search off, the table and PROVENANCE
    shas; ``report.json``'s regime carries every field with no None (R-5), the pinned values, the manifest's values
    and the two arm records; ``report.md`` names each field with its value."""
    m = run["manifest"]
    d = m.get("live_default") or {}
    assert d.get("web") is True and d.get("live_guidance") == "v3", d
    assert d.get("max_tool_steps") == 20, d
    assert d.get("reground") is not True and d.get("action_outcome") is not True, d
    assert m.get("search") == "off", m.get("search")
    blob = json.dumps(m)
    assert TASK_TABLE_SHA256 in blob and PROVENANCE_SHA256 in blob
    regime = run["report"].get("regime")
    assert isinstance(regime, Mapping), "report.json has no regime"
    missing = [k for k in REGIME_KEYS if k not in regime]
    assert not missing, missing
    unset = [k for k in REGIME_KEYS if regime[k] is None]
    assert not unset, ("report.json's regime has None values", unset)
    assert not run["report"].get("regime_none"), (
        "report.json lists None regime fields",
        run["report"].get("regime_none"),
    )
    for k, v in REGIME_VALUES.items():
        assert regime[k] == v, (k, regime[k], v)
    for k in REGIME_KEYS:
        assert regime[k] == (m.get("regime") or {}).get(k), (k, "report.json and manifest differ")
    md = run["md"]
    absent = [k for k in REGIME_KEYS if k not in md]
    assert not absent, ("report.md lacks regime fields", absent)
    unvalued = [(k, regime[k]) for k in REGIME_KEYS if not E._regime_value_in_md(regime[k], md)]
    assert not unvalued, ("report.md lacks regime values", unvalued)
    for k in ARM_REGIME_KEYS:
        assert m.get(k), (k, "manifest.json lacks it")
        assert regime.get(k, run["report"].get(k)) == m[k], (k, "report.json lacks the manifest's")


def test_anchor_quoted_unpaired(run: dict[str, Any]) -> None:
    """A10: ``report.md`` quotes T-0212's three arms (success, click misses, repeat-guard refusals, wall p50 on one
    line per arm) and the unpaired-anchor note verbatim; ``report.json`` carries the note too."""
    md = run["md"]
    assert UNPAIRED_NOTE in md, "report.md lacks the unpaired-anchor note verbatim"
    assert UNPAIRED_NOTE in json.dumps(run["report"], ensure_ascii=False), (
        "report.json lacks the unpaired-anchor note"
    )
    for arm, vals in ANCHOR.items():
        rows = [
            ln
            for ln in md.splitlines()
            if re.search(rf"\b{arm}\b", ln) and all(_token(v, ln) for v in vals.values())
        ]
        assert rows, (arm, vals, "no report.md line quotes this T-0212 arm's anchor values")
