"""T-0140 group B: the locked 2-arm page-overview A/B harness's pure parts, its ``main`` on CPU and its locked script
(unit; CPU; no engine, no browser). docs/tasks/T-0140.md B1-B7, H-T140-1 (approved as proposed 2026-10-23), BASE_B
d8c3035 (A1 and A2 merged).

Rows that need the harness (``scripts/gpu/t0140_overview_ab.py``) or the locked script
(``scripts/locked/t0140_overview_ab.sh``) were activated 2026-10-24 (rulings O-B, B-1..B-9) and are red on a named assertion in
``harness()``/``api()``/``_locked()`` ("does not exist yet" / "has no ..."), never on an import error: every harness
symbol is looked up inside the row. Rows parametrised on ``which`` run once against the harness and once
against the eval oracle (``tests/eval/test_t0140_ab.py``, active and green now), so a wrong oracle is caught on CPU.

**The assumed harness surface** (the task names the behaviour; these names and shapes are the test-author's, stated
once so the implementer can implement to them or say why not):

Imported, never edited (B1)
    T-0134's harness (``F``) and through it T-0132's (``P``), T-0131's (``T``) and T-0111g's (``G``): ``TASKS``,
    ``LoopMeter``, ``sentinel_hits``, ``url_leaks``, ``log_complete``, ``non_loopback_sockets``, ``turn_records``
    (T-0111g's objects); ``probe_p1``, ``run_probes``, ``false_claim``, ``click_score``, ``config_evidence``
    (T-0131's); ``wilson``, ``newcombe`` (T-0132's); ``score_v2``, ``over_click``, ``no_see``, ``coord_mismatch``
    (T-0134's). Module attributes whose objects come from those files (a ``def``/``class``/literal of those names is
    refused).

Seams ``main`` reads at call time (as T-0135's)
    ``_git`` (the tree), ``order``, ``run_probes`` (P1-P3), ``probe_p6`` and ``probe_p7`` (synchronous callables
    returning the ``probes.json`` records of P6/P7, shapes in the oracle's ``p6_ok``/``p7_ok``; the harness adds
    ``passed``), ``PROCESS_START_S``; ``cli.grounding_of``, ``cli.feedback_of``, ``cli.recovery_of`` and
    ``cli.page_overview_of`` (through ``live_default``, read once per ``main``); ``cli.load_tool_kit`` (the arm-B kit
    is ``cli.load_tool_kit(profile, ..., page_overview=True)``, arm A's without it or with ``False``).

The arms (B1; never hard-coded)
    ``live_default(profile) -> {"visual_tokens", "reground", "live_feedback": None | {"screen_label",
    "page_extent"}, "echo", "recover_thought_calls", "page_overview"}``: HEAD's live default as ``cli`` gives it for
    ``Namespace()`` in ``live`` (``grounding_of``, ``feedback_of`` with ``FEEDBACK_DICT``'s mapping,
    ``recovery_of``'s ``thought_call_recovery``, ``page_overview_of``'s ``page_overview``). ``arm_rows(default) ->
    [row_A, row_B]``, each ``{"label", "visual_tokens", "reground", "live_feedback", "echo", "recover_thought_calls",
    "degeneracy_guard": True, "page_overview", "tools_sha256"}``: A ``False`` on tools v1 (bbfc53cc...), B ``True``
    on ``tools_v1_overview.json`` (8048bdb7...). ``arm_table_sha(rows)`` = sha256 of ``json.dumps(rows,
    sort_keys=True, separators=(",", ":"))``. ``main`` passes every setting explicitly in both arms
    (``Session(..., visual_tokens=, reground=, live_feedback=, recover_thought_calls=, degeneracy_guard=True)``,
    ``BrowserToolExecutor(echo=)``) and records ``live_default``, ``regime.arms``, ``regime.commit`` and
    ``arm_table_sha256`` in the manifest.

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3``; ``REPEATS = 20``; ``ORDER_SEED = 0``;
    ``RUN_BUDGET_S = 62 * 60`` (B6); ``PROCESS_START_S`` (``time.monotonic()`` at import).

Pure functions (each also in the eval oracle, same name and shape; the harness's ``arm_metrics`` is the oracle's
``derive_arm``)
    ``order(seed)``; ``p6_ok(p6)``, ``p7_ok(p7)``; ``configuration_findings(trials, steps, regrounds, bodies,
    arms)`` (B4); ``arm_metrics(trials, steps)`` (B3); ``decision_inputs(trials, steps, regrounds)`` (one arm's B5
    inputs ``{"accepted", "safety", "engine_errors", "success", "s4", "false_claims_non_s4", "over_clicks",
    "s4_overview_call_turns", "first_failing": {"acceptance", "safety", "engine"}}``); ``decide(...)`` and
    ``decide_with_reason(inputs, *, p1_passed, void=False, stopped=None, configuration_findings=())`` with labels
    ``"overview"``, ``"overview unused"``, ``"no change"``, ``"inconclusive"``, ``"void"``, ``"incomplete"`` and
    reasons ``"P1 failed"``, ``"configuration: <first finding>"``, ``"arm <X> ineligible: <clause> (first failing
    trial <task>#<repeat>)"``; ``wilson``/``newcombe``; ``build_report(trials, steps, regrounds, bodies, probes, *,
    arms, stopped=None)`` -> report.json (shape in the eval's docstring; ``arms[X]`` holds T-0131's section with
    T-0134's two renames plus ``arm_metrics``; ``differences`` keys ``"s4 B-A"``, ``"success B-A"``).

Records ``main`` writes (the eval's docstring has every field)
    ``trials.jsonl``, ``steps.jsonl``, ``regrounds.jsonl``, ``bodies.jsonl``, ``probes.json`` (P1-P3, P6, P7),
    ``manifest.json``, ``report.json``, ``report.md``. P6 and P7 run before any arm turn; a failed probe is exit 3 with
    a report and no arm turn (P1 ``inconclusive``; P2, P3, P6, P7 ``void``).

Refusals (exit 2, before any socket, leaving no out-dir)
    T-0134's (dirty tree, non-loopback URL, out-dir present, ``LIVE_TOOL_GUIDANCE`` off its sha, a kit with repair
    off, ...), the per-arm tools check (arm A's kit not v1; arm B's kit, ``load_tool_kit(..., page_overview=True)``,
    not the overview set), a ``cli`` default with the overview already on, a ``cli`` without ``page_overview_of``, a
    ``load_tool_kit`` without the ``page_overview`` keyword, and a ``Session`` without the ``degeneracy_guard`` and
    ``recover_thought_calls`` keywords.

**Readings** (ambiguities reported to the orchestrator; the oracle's docstring states the record-level ones)
    1. The engine clause (B5): an *engine error* is a trial that timed out, whose engine was down after it, or with
       a non-200 step attempt; a ``degenerate`` end is never one. Engine-error and ``degenerate`` trials are left out
       of the acceptance clause, which otherwise keeps T-0134's per-trial rule: without that the relaxation would be
       void (both T-0134's and T-0135's ``inconclusive`` were decided on ``acceptance`` by one timeout). The first
       failing trial of the engine clause is the arm's first engine-error trial.
    2. "Counts as a failed trial in every metric": never a success; judged as not succeeded by the false-claim proxy.
    3. B4's placement: ``[overview image, its OVERVIEW_LABEL, current frame]`` in the final user message of the
       ``tool_step`` request one past the call's step; the label text is judged in user messages only (O-17).
    4. ``overview unused`` reads arm B's S4 overview-call turns (a step whose ``call_name`` is ``page_overview``).
    5. The ``Merge T-0140`` refusal: the group merges "Merge T-0140 group A1"/"group A2" already on main at BASE_B do
       not count (or the refusal would be vacuous); "Merge T-0140:" and "Merge T-0140 group B" do.

Regime: CPU, pure, hand-built records; the dress rows run ``main`` on CPU with T-0131's engine rig and the
``FakeLiveBrowser`` (fake LLM, one S4 turn per arm); no engine, no browser, no measurement.
"""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import functools
import hashlib
import importlib.util
import inspect
import itertools
import json
import socket
import sys
import time
import types
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.eval import test_t0140_ab as oracle
from tests.unit import test_t0131_ab_harness as t31

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0140_overview_ab.py"
T0111G_HARNESS = REPO / "scripts" / "gpu" / "t0111g_live_smoke.py"
T0131_HARNESS = REPO / "scripts" / "gpu" / "t0131_grounding_ab.py"
T0132_HARNESS = REPO / "scripts" / "gpu" / "t0132_crop_budget_ab.py"
T0134_HARNESS = REPO / "scripts" / "gpu" / "t0134_feedback_ab.py"
LOCKED = REPO / "scripts" / "locked" / "t0140_overview_ab.sh"

IMPLS = [pytest.param("harness"), pytest.param("eval-oracle")]

#: the full literals (A1; ``tests/unit/test_t0140_tools.py`` pins the same two)
V1 = "bbfc53ccdd80a390e55a7550805ee5aa78309287f7a1171c091f67d9db9cd707"
OV_SHA = "8048bdb7aabec47810bc28b9393755ac7437f74ecaaab3439f1a8c8c67b3efcc"
V1_TOOLS = ["google_search", "click_at", "type_at", "scroll", "press_key"]
OV_TOOLS = [*V1_TOOLS, "page_overview"]
LABEL_FB = {"screen_label": True, "page_extent": False}
#: two hand-chosen live defaults (inputs, not HEAD's): HEAD's at BASE_B as cli gave it on 2026-10-24 (1120, ReGround
#: off, label, model echo, recovery on), and the settings before T-0134c/T-0135b
D_HEAD = {
    "visual_tokens": 1120,
    "reground": False,
    "live_feedback": dict(LABEL_FB),
    "echo": "model",
    "recover_thought_calls": True,
}
D_OLD = {
    "visual_tokens": 1120,
    "reground": False,
    "live_feedback": None,
    "echo": "pixels",
    "recover_thought_calls": False,
}
#: ``arm_table_sha`` of each default's two rows (written out in ``test_arm_rows_and_sha``), by the test-author
#: 2026-10-24 (hashlib over the canonical JSON, in a scratch script)
ARM_PINS = {
    "head": (D_HEAD, "f6249f1d2185693a3b952c9013bf97f6f2140286e1d9103cf26110678d502886"),
    "old": (D_OLD, "81f447637786ee5b456f00db1a6b3290962ce1cd5bc3ffb854970d35b27bdb91"),
}
#: the arms the hand-built records ran at (B1's rows over ``D_HEAD``, written out)
HAND_ARMS = [
    {
        "label": "A",
        **D_HEAD,
        "degeneracy_guard": True,
        "page_overview": False,
        "tools_sha256": V1,
    },
    {
        "label": "B",
        **D_HEAD,
        "degeneracy_guard": True,
        "page_overview": True,
        "tools_sha256": OV_SHA,
    },
]
TASK_IDS = ("S1", "S2", "S3", "S4", "S5")
FROM = {
    T0111G_HARNESS: (
        "LoopMeter",
        "sentinel_hits",
        "url_leaks",
        "log_complete",
        "non_loopback_sockets",
        "turn_records",
    ),
    T0131_HARNESS: ("probe_p1", "run_probes", "false_claim", "click_score", "config_evidence"),
    T0132_HARNESS: ("wilson", "newcombe"),
    T0134_HARNESS: ("score_v2", "over_click", "no_see", "coord_mismatch"),
}
DATA = ("TASKS",)
OV_CALL = "<|tool_call>call:page_overview{}<tool_call|>"
#: A3's strings and A7's label, typed from the criteria
FITS = "the whole page fits on the browser screen; no overview was taken"
A10_KEYS = ("calls", "shown", "fits", "refused", "failed")


def ov_action(screens: float, clipped: bool) -> str:
    c = ", top 4.0 screens shown" if clipped else ""
    return f"showed the whole page (about {screens:.1f} screens tall{c}) as a picture before the screen"


def ov_label(screens: float, clipped: bool) -> str:
    c = ", top 4.0 screens shown" if clipped else ""
    return (
        f"[Whole page, scaled to fit one screen: about {screens:.1f} screens tall{c}. "
        "The next image is the browser screen.]"
    )


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0140 B1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0140_overview_ab_unit", HARNESS)
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
    assert hasattr(h, name), f"T-0140: {HARNESS.name} has no {name!r} (see this module's docstring)"
    return getattr(h, name)


def impl(which: str, name: str) -> Any:
    if which == "harness":
        return api("arm_metrics" if name == "derive_arm" else name)
    return getattr(oracle, name)


# =====================================================================================================
# B1: imports, the arm table, the order
# =====================================================================================================


def test_imports_never_copied() -> None:
    """B1: importing the harness opens no socket; the T-0111g, T-0131, T-0132 and T-0134 names are those files'
    objects, none is defined in the T-0140 harness."""
    with t31.no_sockets() as attempts:
        harness.cache_clear()
        h = harness()
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"
    _, defined = t31._top_level(ast.parse(HARNESS.read_text(encoding="utf-8")))
    names = {n for ns in FROM.values() for n in ns} | set(DATA)
    assert sorted(defined & names) == [], (
        f"T-0140 B1: defined, not imported: {sorted(defined & names)}"
    )
    for origin, ns in FROM.items():
        for name in ns:
            src = Path(inspect.getsourcefile(api(name)) or "").resolve()
            assert src == origin.resolve(), f"T-0140 B1: {name} comes from {src}, not {origin.name}"
    assert [t.task_id for t in api("TASKS")] == list(TASK_IDS)


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("pin", list(ARM_PINS))
def test_arm_rows_and_sha(which: str, pin: str) -> None:
    """B1: over a live default (its ``page_overview`` off), the two rows carry the default's five settings and the
    guard on in both arms, A without the overview on tools v1, B with it on the overview set (written out here), and
    ``arm_table_sha`` of them is the pin; swapping the rows, or B on v1's sha, gives another sha (reviewer mutation
    6)."""
    default, want_sha = ARM_PINS[pin]
    rows = impl(which, "arm_rows")({**default, "page_overview": False})
    want = [
        {
            "label": "A",
            **default,
            "degeneracy_guard": True,
            "page_overview": False,
            "tools_sha256": V1,
        },
        {
            "label": "B",
            **default,
            "degeneracy_guard": True,
            "page_overview": True,
            "tools_sha256": OV_SHA,
        },
    ]
    assert [dict(r) for r in rows] == want, rows
    sha = impl(which, "arm_table_sha")
    assert sha(rows) == want_sha
    assert sha([rows[1], rows[0]]) != want_sha
    assert sha([rows[0], {**rows[1], "tools_sha256": V1}]) != want_sha


def test_constants() -> None:
    """B1/B6: the constants (``RUN_BUDGET_S = 62*60`` from process start)."""
    assert api("REPEATS") == 20 and api("ORDER_SEED") == 0
    assert api("RUN_BUDGET_S") == 62 * 60, "T-0140 B6: RUN_BUDGET_S = 62*60"
    assert isinstance(api("PROCESS_START_S"), float), "B6: the budget runs from process start"
    assert api("PROCESS_START_S") <= time.monotonic()
    assert (api("EXIT_OK"), api("EXIT_FAILED"), api("EXIT_REFUSED"), api("EXIT_PROBE")) == (
        0,
        1,
        2,
        3,
    )


#: ``cli.Feedback.live_feedback``'s choices as ``live_default`` records them (written out by hand, as T-0135 O-16)
FEEDBACK_DICT = {
    "off": {"screen_label": False, "page_extent": False},
    "label": {"screen_label": True, "page_extent": False},
    "label+extent": {"screen_label": True, "page_extent": True},
}


def _heads_default(cli: Any, lean: Any) -> dict[str, Any]:
    """HEAD's live default as ``cli`` gives it now (never a literal, so the rows survive the next default change)."""
    ns = argparse.Namespace()
    g = cli.grounding_of(lean, ns, "live")
    f = cli.feedback_of(lean, ns, "live")
    r = cli.recovery_of(lean, ns, "live")
    o = cli.page_overview_of(lean, ns, "live")
    return {
        "visual_tokens": g.visual_tokens,
        "reground": g.reground,
        "live_feedback": FEEDBACK_DICT[f.live_feedback],
        "echo": f.action_echo,
        "recover_thought_calls": r.thought_call_recovery,
        "page_overview": o.page_overview,
    }


def test_live_default_follows_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1 (the T-0135 B pattern): ``live_default`` is read from ``cli`` at call time and follows a monkeypatched
    default in each of its four resolvers (560 with ReGround on; ``label+extent`` with pixel echo; recovery off),
    with the overview off (A5's default, ``cli.page_overview_of``'s)."""
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    fn = api("live_default")
    want = _heads_default(cli, lean)
    assert want["page_overview"] is False, "premise: the overview is off by default at HEAD (A5)"
    assert dict(fn(lean)) == want, (fn(lean), want)
    assert oracle.live_default_findings(fn(lean)) == []
    real_g = cli.grounding_of
    monkeypatch.setattr(
        cli,
        "grounding_of",
        lambda p, a, m: dataclasses.replace(real_g(p, a, m), visual_tokens=560, reground=True),
    )
    monkeypatch.setattr(
        cli, "feedback_of", lambda p, a, m: cli.Feedback("label+extent", "flag", "pixels", "flag")
    )
    monkeypatch.setattr(cli, "recovery_of", lambda p, a, m: cli.Recovery(False, "flag"))
    moved = dict(fn(lean))
    assert moved == {
        "visual_tokens": 560,
        "reground": True,
        "live_feedback": FEEDBACK_DICT["label+extent"],
        "echo": "pixels",
        "recover_thought_calls": False,
        "page_overview": False,
    }, moved


@pytest.mark.parametrize("which", IMPLS)
def test_order_200_joint_seed0(which: str) -> None:
    """B1: 200 turns, each arm 100 times, each (arm, task) 20 times with repeats 0-19; deterministic;
    seed-dependent; the arms interleave (a per-arm shuffle would run them in blocks)."""
    order = impl(which, "order")
    turns = [tuple(t) for t in order(0)]
    assert turns == [tuple(t) for t in order(0)] and turns != [tuple(t) for t in order(1)]
    assert len(turns) == 200
    assert Counter(t[0] for t in turns) == {"A": 100, "B": 100}
    per = Counter((t[0], t[1]) for t in turns)
    assert set(per) == {(a, k) for a in "AB" for k in TASK_IDS} and set(per.values()) == {20}
    assert {t[2] for t in turns} == set(range(20))
    switches = sum(1 for x, y in itertools.pairwise(turns) if x[0] != y[0])
    assert switches > 60, f"the arms run in blocks ({switches} switches)"


def test_harness_order_equals_oracle() -> None:
    """B1/B7: the harness's seed-0 order is the oracle's (the eval checks the run against it)."""
    assert [tuple(t) for t in api("order")(0)] == oracle.order(0)


# =====================================================================================================
# B1: refusals before any socket (per-arm tools check, the overview already on)
# =====================================================================================================


def _stand_in_kits() -> dict[bool, Any]:
    return {
        False: types.SimpleNamespace(repair_off=None, tools_version=V1, mode="live"),
        True: types.SimpleNamespace(repair_off=None, tools_version=OV_SHA, mode="live"),
    }


def _rig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Any:
    """Every socket refused and recorded; a clean tree through the harness's ``_git``; stand-in kits (arm A's on
    v1, arm B's -- ``page_overview=True`` -- on the overview set, repair on); Chromium reported present."""
    from local_astra import cli

    h = harness()
    sockets: list[str] = []

    def refuse(what: str) -> Any:
        def fn(*a: Any, **k: Any) -> Any:
            sockets.append(what)
            raise ConnectionRefusedError(f"T-0140 refusal rig: no {what}")

        return fn

    for name in ("connect", "connect_ex", "bind"):
        monkeypatch.setattr(socket.socket, name, refuse(name))
    monkeypatch.setattr(socket, "create_connection", refuse("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuse("getaddrinfo"))
    status = {"porcelain": ""}

    def git(*args: str) -> str:
        return status["porcelain"] if args[:1] == ("status",) else t31.GIT + "\n"

    assert callable(getattr(h, "_git", None)), (
        "the harness has no module-level _git (T-0111g's seam)"
    )
    monkeypatch.setattr(h, "_git", git)
    kits = _stand_in_kits()

    def load(profile: Any, *, live_browser: Any = None, page_overview: bool = False) -> Any:
        return (kits[bool(page_overview)], None)

    monkeypatch.setattr(cli, "load_tool_kit", load)
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    return types.SimpleNamespace(
        h=h,
        cli=cli,
        kits=kits,
        kit=kits[False],
        sockets=sockets,
        status=status,
        out=tmp_path / "out",
        capsys=capsys,
        mp=monkeypatch,
    )


REFUSALS = (
    "dirty-tree",
    "url-not-loopback",
    "out-dir-exists",
    "kit-repair-off",
    "guidance-off-its-sha",
    "kit-a-not-v1",
    "kit-b-not-the-overview-set",
    "overview-already-the-default",
    "cli-without-page-overview-of",
    "load-tool-kit-without-page-overview",
    "session-without-guard-keywords",
)


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, case: str
) -> None:
    """B1: T-0134's refusals stand (dirty tree, non-loopback URL, an existing out-dir, the tool repair off, the
    guidance off its sha); the tools check is made per arm (arm A's kit not v1; arm B's kit -- ``load_tool_kit(...,
    page_overview=True)`` -- not the overview set, as a ``cli`` that ignored the keyword would give: reviewer
    mutation 6's shape); a ``cli`` default with the overview already on is refused (reviewer mutation 5); so are a
    ``cli`` without ``page_overview_of``, a ``load_tool_kit`` without the keyword and a ``Session`` without the
    guard's keywords. Each exits 2 before any socket, leaving the out-dir as it was."""
    from local_astra.conductor import prompt_builder
    from local_astra.conductor import session as session_mod

    rig = _rig(tmp_path, monkeypatch, capsys)
    h = rig.h
    argv = ["--base-url", t31.VALID_URL, "--out-dir", str(rig.out)]
    if case == "dirty-tree":
        rig.status["porcelain"] = " M local_astra/conductor/session.py\n"
    elif case == "url-not-loopback":
        argv[1] = "http://10.0.0.1:8001/v1"
    elif case == "out-dir-exists":
        rig.out.mkdir()
    elif case == "kit-repair-off":
        rig.kits[False].repair_off = "payload stale"
        rig.kits[True].repair_off = "payload stale"
    elif case == "guidance-off-its-sha":
        rig.mp.setattr(
            prompt_builder, "LIVE_TOOL_GUIDANCE", prompt_builder.LIVE_TOOL_GUIDANCE + " "
        )
    elif case == "kit-a-not-v1":
        rig.kits[False].tools_version = OV_SHA
    elif case == "kit-b-not-the-overview-set":
        rig.kits[True].tools_version = V1
    elif case == "overview-already-the-default":
        rig.mp.setattr(
            rig.cli, "page_overview_of", lambda p, a, m: rig.cli.PageOverview(True, "default")
        )
    elif case == "cli-without-page-overview-of":
        rig.mp.delattr(rig.cli, "page_overview_of")
    elif case == "load-tool-kit-without-page-overview":

        def old_load(profile: Any, *, live_browser: Any = None) -> Any:
            return (rig.kits[False], None)

        rig.mp.setattr(rig.cli, "load_tool_kit", old_load)
    else:

        class Old(session_mod.Session):  # type: ignore[misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)

        rig.mp.setattr(session_mod, "Session", Old)
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = h.main(argv)
    err = rig.capsys.readouterr().err
    assert rc == h.EXIT_REFUSED == 2, f"{case}: exit {rc}\n{err[-2000:]}"
    assert rig.sockets == [], rig.sockets
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert err.strip(), f"{case}: the refusal says nothing"
    if case == "cli-without-page-overview-of":
        assert "page_overview_of" in err, f"the refusal does not name cli.page_overview_of: {err!r}"
    if case == "overview-already-the-default":
        assert "overview" in err, f"the refusal does not name the overview: {err!r}"


def test_rig_at_passing_values_goes_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """The other side of every refusal row (a guard too wide): at HEAD's live default, a clean tree and both kits
    at their pins, ``main`` does not refuse; it looks for the engine (a socket attempt) and, finding none, exits 1."""
    rig = _rig(tmp_path, monkeypatch, capsys)
    h = rig.h
    rc = h.main(["--base-url", t31.VALID_URL, "--out-dir", str(rig.out)])
    err = rig.capsys.readouterr().err
    assert rc == h.EXIT_FAILED == 1, f"exit {rc}\n{err[-2000:]}"
    assert rig.sockets, "main never tried the engine"


def test_probes_p1_to_p3_are_t0131s() -> None:
    """B2: P1-P3 are T-0131's own objects; P6 and P7 are the harness's seams."""
    for name in ("probe_p1", "run_probes"):
        src = Path(inspect.getsourcefile(api(name)) or "").resolve()
        assert src == T0131_HARNESS.resolve(), (name, src)
    assert callable(api("probe_p6")) and callable(api("probe_p7"))


# =====================================================================================================
# B2: P6 and P7 from hand records
# =====================================================================================================

#: deals.html at 1280x800 (O-A1-11: a 838x798 overview), 1.52 screens; a fitting page; T-0134's hostile fixture
#: (2800 CSS px = 3.5 screens)
P6_OK: dict[str, Any] = {
    "viewport": [1280, 800],
    "deals": {
        "outcome": "shown",
        "action": ov_action(1.52, False),
        "screens": 1.52,
        "clipped": False,
        "width": 838,
        "height": 798,
        "scroll_y_before": 0.0,
        "scroll_y_after": 0.0,
    },
    "fits": {"outcome": "fits", "action": FITS},
    "hostile": {
        "outcome": "shown",
        "action": ov_action(3.5, False),
        "screens": 3.5,
        "clipped": False,
    },
}


def _p6(**over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(P6_OK)
    for part, fields in over.items():
        out[part] = {**out[part], **fields}
    return out


#: (record, passes), by hand from B2
P6_ROWS: dict[str, tuple[dict[str, Any], bool]] = {
    "ok": (P6_OK, True),
    "deals-exactly-1.3": (_p6(deals={"screens": 1.3, "action": ov_action(1.3, False)}), True),
    "deals-1.29": (_p6(deals={"screens": 1.29, "action": ov_action(1.29, False)}), False),
    "deals-fits": (_p6(deals={"outcome": "fits", "action": FITS}), False),
    "deals-image-too-wide": (_p6(deals={"width": 1281}), False),
    "deals-image-too-tall": (_p6(deals={"height": 801}), False),
    "deals-scrolled": (_p6(deals={"scroll_y_after": 2.0}), False),
    "deals-action-off-a3": (_p6(deals={"action": ov_action(1.6, False)}), False),
    "fits-showed-an-overview": (
        _p6(fits={"outcome": "shown", "action": ov_action(1.0, False)}),
        False,
    ),
    "hostile-spoofed-height": (
        _p6(hostile={"screens": 5.3, "action": ov_action(5.3, True), "clipped": True}),
        False,
    ),
    "hostile-off-by-2px": (
        _p6(hostile={"screens": 2802 / 800, "action": ov_action(2802 / 800, False)}),
        False,
    ),
    "hostile-within-1px": (
        _p6(hostile={"screens": 2800.5 / 800, "action": ov_action(2800.5 / 800, False)}),
        True,
    ),
    "wrong-viewport": ({**P6_OK, "viewport": [1280, 720]}, False),
    "no-hostile": ({k: v for k, v in P6_OK.items() if k != "hostile"}, False),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(P6_ROWS))
def test_p6_rule(which: str, row: str) -> None:
    """B2 P6 (browser-only): deals.html gives an overview of at least 1.3 screens within 1280x800 with A3's action
    string and ``scroll_y`` unchanged; a fitting page gives ``fits``; T-0134's hostile fixture gives its true height
    (1 px tolerance, as T-0134 P5). Each clause failing alone fails P6; a missing part fails, never raises."""
    record, want = P6_ROWS[row]
    assert impl(which, "p6_ok")(copy.deepcopy(record)) is want, row


P7_OK: dict[str, Any] = {
    "tokens": 10_480,
    "limit": 16_128,
    "http_status": 200,
    "healthy_after": True,
    "images": 3,
    "overview_images": 1,
    "overview_tokens": 1_088,
}
P7_ROWS: dict[str, tuple[dict[str, Any], bool]] = {
    "ok": (P7_OK, True),
    "exactly-16128": ({**P7_OK, "tokens": 16_128}, True),
    "16129": ({**P7_OK, "tokens": 16_129}, False),
    "no-count": ({**P7_OK, "tokens": None}, False),
    "http-500": ({**P7_OK, "http_status": 500}, False),
    "engine-down-after": ({**P7_OK, "healthy_after": False}, False),
    "no-overview-in-the-request": ({**P7_OK, "overview_images": 0}, False),
    "five-images": ({**P7_OK, "images": 5}, False),
    "another-limit": ({**P7_OK, "limit": 16_384, "tokens": 16_200}, False),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(P7_ROWS))
def test_p7_rule(which: str, row: str) -> None:
    """B2 P7 (T-0132 P4's rule): a ``/tokenize`` count of the step-2 request carrying one overview (at most 4 images)
    within 16128 tokens, the generate answering 200 and the engine healthy after."""
    record, want = P7_ROWS[row]
    assert impl(which, "p7_ok")(dict(record)) is want, row


# =====================================================================================================
# Hand-built records: four turns per arm
# =====================================================================================================

REQUEST = "Add the cheapest item on the deals page to the cart."
SCREEN_NOW = "[This screenshot is the browser screen now.]"
SCREEN_AFTER = "[This screenshot is the browser screen now, after your last action.]"
SCREEN_SAME = "[This screenshot is the browser screen now; your last action did not change it.]"
TOWEL_SAID = "The cheapest is the cotton tea towel at $4.75."
MUG_SAID = "The cheapest is the enamel mug at $11.50."
NARRATED = "The whole page shows the enamel mug is the cheapest."
DEALS_EMPTY = {"page": "deals", "cart": []}


def _h(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()[:16]


def _i(tag: str) -> dict[str, str]:
    return {"image": _h(tag)}


def _t(text: str) -> dict[str, str]:
    return {"text": text}


def _sys() -> dict[str, Any]:
    return {"role": "system", "parts": [_t("You are Astra.")]}


def _user(*parts: dict[str, str]) -> dict[str, Any]:
    return {"role": "user", "parts": list(parts)}


def _asst(text: str) -> dict[str, Any]:
    return {"role": "assistant", "parts": [_t(text)]}


def _req(
    label: str, phase: str, step: int | None, tools: list[str] | None, *messages: dict[str, Any]
) -> dict[str, Any]:
    return {
        "label": label,
        "phase": phase,
        "step": step,
        "tools": None if tools is None else list(tools),
        "messages": [copy.deepcopy(m) for m in messages],
    }


def _counts(**kw: int) -> dict[str, int]:
    return {k: int(kw.get(k, 0)) for k in A10_KEYS}


def _tr(
    arm: str, task: str, repeat: int, state: Any, voiced: list[str], **kw: Any
) -> dict[str, Any]:
    p = f"{arm}{task}{repeat}"
    base: dict[str, Any] = {
        "arm": arm,
        "voiced": list(voiced),
        "visual_tokens": 1120,
        "reground": False,
        "reground_visual_tokens": None,
        "live_feedback": dict(LABEL_FB),
        "echo": "model",
        "recover_thought_calls": True,
        "degeneracy_guard": True,
        "page_overview": arm == "B",
        "tools_version": V1 if arm == "A" else OV_SHA,
        "page_overview_logged": None if arm == "A" else _counts(),
        "degeneracy_guard_logged": {"on": True, "tripped": []},
        # review F3: the turn line as the session logs it at D_HEAD (T-0134 A5 adds the browser's echo)
        "live_feedback_logged": {**LABEL_FB, "echo": "model"},
        "recover_thought_calls_logged": True,
        "visual_tokens_logged": 1120,
        "budgets_sent": [1120],
        "guidance_sent": True,
        "scanned_requests": [f"prefill:{p}-0", f"generate:{p}-1"],
        "tool_loop_end": "final",
        "session_id": f"s-{p}",
    }
    return t31._trial(task, repeat, state, **(base | kw))


def _st(
    arm: str,
    task: str,
    repeat: int,
    step: int,
    status: str | None,
    done: float | None,
    name: str | None = None,
    *,
    voiced: tuple[str, ...] = (),
    shown: bool = False,
    **kw: Any,
) -> dict[str, Any]:
    p = f"{arm}{task}{repeat}"
    return t31._step(
        task,
        repeat,
        step,
        status,
        None if done is None else done - 100.0,
        done,
        arm=arm,
        visual_tokens=1120,
        reground=False,
        reground_visual_tokens=None,
        call_name=name,
        executed_name=name if status == "executed" else None,
        voiced_sources=list(voiced),
        attempt_id=f"{p}-{step}",
        call_id=f"{p}-call-{step}" if name else None,
        overview_shown=shown,
        **kw,
    )


def _plain_requests(arm: str, task: str, repeat: int, n: int) -> list[dict[str, Any]]:
    """A prefill, the generate (step 1) and ``n - 1`` tool steps, each on its arm's tools, one current frame each."""
    p = f"{arm}{task}{repeat}"
    tools = V1_TOOLS if arm == "A" else OV_TOOLS
    out = [
        _req(f"prefill:{p}-0", "prefill", None, tools, _sys()),
        _req(
            f"generate:{p}-1",
            "generate",
            1,
            tools,
            _sys(),
            _user(_t(REQUEST), _i(f"{p}-f0"), _t(SCREEN_NOW)),
        ),
    ]
    for k in range(2, n + 1):
        out.append(
            _req(
                f"generate:{p}-{k}",
                "tool_step",
                k,
                tools,
                _sys(),
                _user(_t(REQUEST), _i(f"{p}-f{k - 1}"), _t(SCREEN_AFTER)),
                _asst(f"Working. step {k - 1} done."),
            )
        )
    return out


def _line(
    arm: str, task: str, repeat: int, requests: list[Any], overviews: list[Any]
) -> dict[str, Any]:
    return {
        "arm": arm,
        "task": task,
        "repeat": repeat,
        "requests": requests,
        "overviews": overviews,
    }


#: B1's overview (1.9 screens, not clipped) and B2's (9.3 screens, clipped at 4)
OV1 = {
    "step": 1,
    "sha": _h("BS40-ov"),
    "screens": 1.9,
    "clipped": False,
    "width": 838,
    "height": 798,
}
OV2 = {
    "step": 1,
    "sha": _h("BS41-ov"),
    "screens": 9.3,
    "clipped": True,
    "width": 344,
    "height": 800,
}


def _hand_records(arm: str) -> tuple[list[Any], list[Any], list[Any]]:
    """Four turns per arm; every count is written out in ``HAND``'s comment.

    Arm A (overview off): A1 S4#0 a scroll, then the towel and $4.75 (success); A2 S4#1 names the mug (false claim);
    A3 S2#0 two blue mugs, "I have added the blue mug." (over-click, non-S4 false claim); A4 S1#0 success.

    Arm B (overview on): B1 S4#0 ``page_overview`` (1.9 screens) at step 1, a scroll at step 2 (whose request carries
    the overview), the towel at step 3 (success); B2 S4#1 ``page_overview`` (9.3 screens, clipped) at step 1 with two
    history images, step 2 (four images with the overview) narrates the whole page and names the mug (false claim);
    B3 S5#0 ``page_overview`` on a page that fits, a click, success; B4 S2#0 a ``degenerate`` end with the right cart
    (a forced failure; its aborted step 2 answered HTTP 0, not an engine error)."""
    if arm == "A":
        trials = [
            _tr("A", "S4", 0, DEALS_EMPTY, [TOWEL_SAID]),
            _tr("A", "S4", 1, DEALS_EMPTY, [MUG_SAID]),
            _tr(
                "A",
                "S2",
                0,
                {"page": "shop", "cart": ["blue mug", "blue mug"]},
                ["I have added the blue mug."],
            ),
            _tr("A", "S1", 0, {"page": "shop", "cart": ["blue mug"]}, ["Added."]),
        ]
        steps = [
            _st("A", "S4", 0, 1, "executed", 300.0, "scroll"),
            _st("A", "S4", 0, 2, None, 350.0, voiced=(TOWEL_SAID,)),
            _st("A", "S4", 1, 1, None, 360.0, voiced=(MUG_SAID,)),
            _st("A", "S2", 0, 1, "executed", 310.0, "click_at"),
            _st("A", "S2", 0, 2, "executed", 320.0, "click_at"),
            _st("A", "S2", 0, 3, None, 330.0, voiced=("I have added the blue mug.",)),
            _st("A", "S1", 0, 1, "executed", 340.0, "click_at"),
            _st("A", "S1", 0, 2, None, 370.0, voiced=("Added.",)),
        ]
        bodies = [
            _line("A", "S4", 0, _plain_requests("A", "S4", 0, 2), []),
            _line("A", "S4", 1, _plain_requests("A", "S4", 1, 1), []),
            _line("A", "S2", 0, _plain_requests("A", "S2", 0, 3), []),
            _line("A", "S1", 0, _plain_requests("A", "S1", 0, 2), []),
        ]
        return trials, steps, bodies
    trials = [
        _tr(
            "B", "S4", 0, DEALS_EMPTY, [TOWEL_SAID], page_overview_logged=_counts(calls=1, shown=1)
        ),
        _tr("B", "S4", 1, DEALS_EMPTY, [NARRATED], page_overview_logged=_counts(calls=1, shown=1)),
        _tr(
            "B",
            "S5",
            0,
            {"page": "deals", "cart": ["cotton tea towel"]},
            ["Added the cotton tea towel."],
            page_overview_logged=_counts(calls=1, fits=1),
        ),
        _tr(
            "B",
            "S2",
            0,
            {"page": "shop", "cart": ["blue mug"]},
            ["Adding the blue mug."],
            tool_loop_end="degenerate",
        ),
    ]
    steps = [
        _st("B", "S4", 0, 1, "executed", 300.0, "page_overview"),
        _st("B", "S4", 0, 2, "executed", 900.0, "scroll", shown=True),
        _st("B", "S4", 0, 3, None, 305.0, voiced=(TOWEL_SAID,)),
        _st("B", "S4", 1, 1, "executed", 320.0, "page_overview"),
        _st("B", "S4", 1, 2, None, 1100.0, voiced=(NARRATED,), shown=True),
        _st("B", "S5", 0, 1, "executed", 310.0, "page_overview"),
        _st("B", "S5", 0, 2, "executed", 400.0, "click_at"),
        _st("B", "S5", 0, 3, None, 410.0, voiced=("Added the cotton tea towel.",)),
        _st("B", "S2", 0, 1, "executed", 330.0, "click_at", voiced=("Adding the blue mug.",)),
        _st("B", "S2", 0, 2, None, None, http_status=0, error="cancelled"),
    ]
    b1 = _plain_requests("B", "S4", 0, 1)
    b1 += [
        _req(
            "generate:BS40-2",
            "tool_step",
            2,
            OV_TOOLS,
            _sys(),
            _user(
                _t(REQUEST),
                {"image": OV1["sha"]},
                _t(ov_label(1.9, False)),
                _i("BS40-f0"),
                _t(SCREEN_SAME),
            ),
            _asst("Let me look at the whole page. " + OV_CALL),
        ),
        _req(
            "generate:BS40-3",
            "tool_step",
            3,
            OV_TOOLS,
            _sys(),
            _user(_t(REQUEST), _i("BS40-f1"), _t(SCREEN_AFTER)),
            # the action string reaches later bodies in the continuation by design (A9 is about speech and storage)
            _asst("Let me look at the whole page. " + OV_CALL + ov_action(1.9, False)),
        ),
    ]
    history = (
        _user(_t("What is on sale?"), _i("BS41-h1")),
        _asst("Six deals."),
        _user(_t("And the mugs?"), _i("BS41-h2")),
        _asst("Two mugs."),
    )
    b2 = [
        _req("prefill:BS41-0", "prefill", None, OV_TOOLS, _sys(), *history),
        _req(
            "generate:BS41-1",
            "generate",
            1,
            OV_TOOLS,
            _sys(),
            *history,
            _user(_t(REQUEST), _i("BS41-f0"), _t(SCREEN_NOW)),
        ),
        _req(
            "generate:BS41-2",
            "tool_step",
            2,
            OV_TOOLS,
            _sys(),
            *history,
            _user(
                _t(REQUEST),
                {"image": OV2["sha"]},
                _t(ov_label(9.3, True)),
                _i("BS41-f0"),
                _t(SCREEN_SAME),
            ),
            _asst(OV_CALL),
        ),
    ]
    bodies = [
        _line("B", "S4", 0, b1, [dict(OV1)]),
        _line("B", "S4", 1, b2, [dict(OV2)]),
        _line("B", "S5", 0, _plain_requests("B", "S5", 0, 3), []),
        _line("B", "S2", 0, _plain_requests("B", "S2", 0, 2), []),
    ]
    return trials, steps, bodies


def _clean() -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    ta, sa, ba = _hand_records("A")
    tb, sb, bb = _hand_records("B")
    return ta + tb, sa + sb, [], ba + bb


def _find(rows: list[dict[str, Any]], arm: str, task: str, repeat: int) -> dict[str, Any]:
    (r,) = [x for x in rows if (x["arm"], x["task"], x["repeat"]) == (arm, task, repeat)]
    return r


def _request(
    bodies: list[dict[str, Any]], arm: str, task: str, repeat: int, label: str
) -> dict[str, Any]:
    (r,) = [r for r in _find(bodies, arm, task, repeat)["requests"] if r["label"] == label]
    return r


def _final_user(request: dict[str, Any]) -> dict[str, Any]:
    return [m for m in request["messages"] if m["role"] == "user"][-1]


# =====================================================================================================
# B4: configuration findings, one hand trial per finding
# =====================================================================================================


#: review F1: one overview sha for two calls on an unchanged page (real Chromium gave one sha for 3 calls on deals)
REPEAT_OV = {
    "step": 1,
    "sha": _h("BS42-ov"),
    "screens": 1.9,
    "clipped": False,
    "width": 838,
    "height": 798,
}


def _placed_request(p: str, step: int, ov: dict[str, Any], frame: str) -> dict[str, Any]:
    """A ``tool_step`` body at ``step`` with ``ov`` right before the current frame, labelled as A7 labels it."""
    return _req(
        f"generate:{p}-{step}",
        "tool_step",
        step,
        OV_TOOLS,
        _sys(),
        _user(
            _t(REQUEST),
            {"image": ov["sha"]},
            _t(ov_label(float(ov["screens"]), bool(ov["clipped"]))),
            _i(frame),
            _t(SCREEN_SAME),
        ),
        _asst(OV_CALL),
    )


def _with_repeated_calls(trials: list[Any], bodies: list[Any]) -> tuple[list[Any], list[Any]]:
    """Arm B S4#2: ``page_overview`` at steps 1 and 2 on an unchanged page (one sha), each overview right before the
    current frame of the next step's body (steps 2 and 3), then the towel."""
    p = "BS42"
    trials = [
        *trials,
        _tr(
            "B", "S4", 2, DEALS_EMPTY, [TOWEL_SAID], page_overview_logged=_counts(calls=2, shown=2)
        ),
    ]
    requests = _plain_requests("B", "S4", 2, 1)
    requests += [
        _placed_request(p, 2, REPEAT_OV, f"{p}-f0"),
        _placed_request(p, 3, REPEAT_OV, f"{p}-f0"),
    ]
    overviews = [dict(REPEAT_OV), {**REPEAT_OV, "step": 2}]
    return trials, [*bodies, _line("B", "S4", 2, requests, overviews)]


def _repair_of(request: dict[str, Any], label: str, step: int) -> dict[str, Any]:
    """A ``tool_repair`` body of the step-``step`` call: its pass-1 prompt continued (the overview, if any, included,
    as ``session._run_repair`` sends it), the final assistant message the draft."""
    r = copy.deepcopy(request)
    r.update(label=label, phase="tool_repair", step=step)
    r["messages"][-1] = _asst("Let me look at the whole page. <|tool_call>call:scroll{direction:")
    return r


def _fault(name: str) -> tuple[Any, ...]:
    """One B4 fault applied to the clean records; returns (trials, steps, regrounds, bodies, arms, the name a
    finding must carry)."""
    trials, steps, regrounds, bodies = _clean()
    arms: list[dict[str, Any]] = [copy.deepcopy(r) for r in HAND_ARMS]
    if name == "non-1120-body":
        _find(trials, "A", "S4", 0)["budgets_sent"] = [1120, 560]
        return trials, steps, regrounds, bodies, arms, "A S4#0"
    if name == "reground-line":
        regrounds = [{"arm": "B", "task": "S4", "repeat": 0, "step": 2, "outcome": "ok"}]
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "arm-a-body-declares-overview":
        _request(bodies, "A", "S4", 1, "generate:AS41-1")["tools"] = list(OV_TOOLS)
        return trials, steps, regrounds, bodies, arms, "A S4#1"
    if name == "arm-b-body-on-v1":
        _request(bodies, "B", "S5", 0, "prefill:BS50-0")["tools"] = list(V1_TOOLS)
        return trials, steps, regrounds, bodies, arms, "B S5#0"
    if name == "arm-b-step-body-without-tools":
        _request(bodies, "B", "S5", 0, "generate:BS50-2")["tools"] = None
        return trials, steps, regrounds, bodies, arms, "B S5#0"
    if name == "overview-after-current-frame":
        # reviewer mutation 1: [frame, screen label, overview, its label]
        user = _final_user(_request(bodies, "B", "S4", 0, "generate:BS40-2"))
        user["parts"] = [
            _t(REQUEST),
            _i("BS40-f0"),
            _t(SCREEN_SAME),
            {"image": OV1["sha"]},
            _t(ov_label(1.9, False)),
        ]
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-before-an-older-image":
        # the overview sits before an image that is not the current frame (another image follows it)
        user = _final_user(_request(bodies, "B", "S4", 0, "generate:BS40-2"))
        user["parts"] = [
            _t(REQUEST),
            {"image": OV1["sha"]},
            _t(ov_label(1.9, False)),
            _i("BS40-older"),
            _i("BS40-f0"),
            _t(SCREEN_SAME),
        ]
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-kept-a-later-step":
        # reviewer mutation 2: the step-3 request still carries it, placed as A7 places it
        user = _final_user(_request(bodies, "B", "S4", 0, "generate:BS40-3"))
        user["parts"] = [
            _t(REQUEST),
            {"image": OV1["sha"]},
            _t(ov_label(1.9, False)),
            _i("BS40-f1"),
            _t(SCREEN_AFTER),
        ]
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-placed-in-a-history-message-of-the-right-step":
        # review minor (``last_user``): the right request, the right [overview, label, frame] order, but in an earlier
        # user message, not the final one
        r = _request(bodies, "B", "S4", 0, "generate:BS40-2")
        r["messages"] = [
            _sys(),
            _user(_t("Earlier."), {"image": OV1["sha"]}, _t(ov_label(1.9, False)), _i("BS40-h")),
            _asst("Earlier."),
            _user(_t(REQUEST), _i("BS40-f0"), _t(SCREEN_SAME)),
            _asst("Let me look at the whole page. " + OV_CALL),
        ]
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "repeated-call-overview-out-of-step":
        # review F1: two calls share a sha; the sha in a step-4 body matches neither call's step + 1
        trials, bodies = _with_repeated_calls(trials, bodies)
        line = _find(bodies, "B", "S4", 2)
        line["requests"].append(_placed_request("BS42", 4, REPEAT_OV, "BS42-f0"))
        return trials, steps, regrounds, bodies, arms, "B S4#2"
    if name == "repair-of-the-wrong-step":
        # review F2: a repair of the step-3 call carries the step-1 overview (only step 2's may)
        r = _repair_of(_request(bodies, "B", "S4", 0, "generate:BS40-2"), "repair:BS40-2r", 3)
        _find(bodies, "B", "S4", 0)["requests"].append(r)
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-in-prefill":
        r = _request(bodies, "B", "S4", 0, "prefill:BS40-0")
        r["messages"].append(_user(_t("x"), {"image": OV1["sha"]}))
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-in-a-history-message":
        r = _request(bodies, "B", "S4", 0, "generate:BS40-3")
        r["messages"][1:1] = [_user(_t("Earlier."), {"image": OV1["sha"]}), _asst("Earlier.")]
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "two-overview-images":
        user = _final_user(_request(bodies, "B", "S4", 0, "generate:BS40-2"))
        user["parts"].insert(0, {"image": OV1["sha"]})
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "five-images":
        r = _request(bodies, "B", "S4", 1, "generate:BS41-2")
        r["messages"][1:1] = [_user(_t("Before."), _i("BS41-h0")), _asst("Before.")]
        return trials, steps, regrounds, bodies, arms, "B S4#1"
    if name == "overview-without-its-label":
        user = _final_user(_request(bodies, "B", "S4", 0, "generate:BS40-2"))
        user["parts"] = [_t(REQUEST), {"image": OV1["sha"]}, _i("BS40-f0"), _t(SCREEN_SAME)]
        return trials, steps, regrounds, bodies, arms, "B S4#0"
    if name == "overview-label-of-another-height":
        user = _final_user(_request(bodies, "B", "S4", 1, "generate:BS41-2"))
        user["parts"][2] = _t(ov_label(9.3, False))
        return trials, steps, regrounds, bodies, arms, "B S4#1"
    if name == "overview-label-in-a-history-user-message":
        r = _request(bodies, "B", "S4", 1, "generate:BS41-2")
        r["messages"][1]["parts"].append(_t(ov_label(9.3, True)))
        return trials, steps, regrounds, bodies, arms, "B S4#1"
    if name == "feedback-off-its-row":
        _find(trials, "A", "S1", 0)["live_feedback"] = None
        return trials, steps, regrounds, bodies, arms, "A S1#0"
    if name == "echo-off-its-row":
        _find(trials, "B", "S5", 0)["echo"] = "pixels"
        return trials, steps, regrounds, bodies, arms, "B S5#0"
    if name == "guard-off-its-row":
        _find(trials, "B", "S2", 0)["degeneracy_guard"] = False
        return trials, steps, regrounds, bodies, arms, "B S2#0"
    if name == "arms-differ-in-budget":
        arms[1]["visual_tokens"] = 560
        return trials, steps, regrounds, bodies, arms, "visual_tokens"
    if name == "arms-differ-in-echo":
        arms[1]["echo"] = "pixels"
        return trials, steps, regrounds, bodies, arms, "echo"
    if name == "arm-b-row-without-the-overview":
        arms[1]["page_overview"] = False
        return trials, steps, regrounds, bodies, arms, "arm B"
    if name == "arm-a-row-on-the-overview-set":
        arms[0]["tools_sha256"] = OV_SHA
        return trials, steps, regrounds, bodies, arms, "arm A"
    if name == "arm-rows-guard-off":
        for r in arms:
            r["degeneracy_guard"] = False
        return trials, steps, regrounds, bodies, arms, "degeneracy_guard"
    raise KeyError(name)


FAULTS = (
    "non-1120-body",
    "reground-line",
    "arm-a-body-declares-overview",
    "arm-b-body-on-v1",
    "arm-b-step-body-without-tools",
    "overview-after-current-frame",
    "overview-before-an-older-image",
    "overview-kept-a-later-step",
    "overview-in-prefill",
    "overview-in-a-history-message",
    "two-overview-images",
    "five-images",
    "overview-without-its-label",
    "overview-label-of-another-height",
    "overview-label-in-a-history-user-message",
    "feedback-off-its-row",
    "echo-off-its-row",
    "guard-off-its-row",
    "arms-differ-in-budget",
    "arms-differ-in-echo",
    "arm-b-row-without-the-overview",
    "arm-a-row-on-the-overview-set",
    "arm-rows-guard-off",
    "overview-placed-in-a-history-message-of-the-right-step",
    "repeated-call-overview-out-of-step",
    "repair-of-the-wrong-step",
)


def _cf(which: str) -> Any:
    fn = impl(which, "configuration_findings")
    return lambda trials, steps, regrounds, bodies, arms: fn(trials, steps, regrounds, bodies, arms)


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("fault", FAULTS)
def test_configuration_findings(which: str, fault: str) -> None:
    """B4, one hand trial per finding (reviewer mutations 1 and 2 among them): the clean records have none; each
    fault gives at least one finding naming the trial or the setting -- a non-1120 body, a reground line, an arm-A
    body declaring ``page_overview``, an arm-B body not declaring it (or a step body with no tools), the overview after
    the current frame, kept a later step, in a prefill or a history message, twice in one request, a fifth image, the
    overview without its label or with another height's label, the label in a history user message, a trial's
    feedback/echo/guard off its row, the arm rows differing or off B1."""
    cf = _cf(which)
    trials, steps, regrounds, bodies = _clean()
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == [], (
        "premise: the clean run has no finding"
    )
    trials, steps, regrounds, bodies, arms, named = _fault(fault)
    found = cf(trials, steps, regrounds, bodies, arms)
    assert found and any(named in f for f in found), (fault, found)


@pytest.mark.parametrize("which", IMPLS)
def test_configuration_findings_not_too_wide(which: str) -> None:
    """B4 companions, the guard-too-wide side: the clean records hold an overview right before the current frame
    with its label (B1 S4#0 step 2), a clipped one in a four-image request with two history images (B2 S4#1), arm-B
    prefills declaring the tool, the overview's action string in a later continuation, T-0134's screen label after
    the current frame, a fitting overview that kept no image (B3) and a ``degenerate`` end (B4) -- none is a
    finding. A trial that sent nothing is not judged (O-11)."""
    cf = _cf(which)
    trials, steps, regrounds, bodies = _clean()
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == []
    raised = _find(trials, "A", "S2", 0)
    raised.update(scanned_requests=[], exception="RuntimeError('no engine')", budgets_sent=None)
    _request(bodies, "A", "S2", 0, "generate:AS20-1")["tools"] = list(OV_TOOLS)
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == [], "O-11: a trial that sent nothing"


@pytest.mark.parametrize("which", IMPLS)
def test_spoken_overview_words_are_not_a_void(which: str) -> None:
    """O-17's lesson for B4 (mutant: the label judged in every message): the model speaks the label's own words and
    they reach the next request as assistant text -- no finding; the same text in a history *user* message (the
    session storing the label) is one."""
    cf = _cf(which)
    trials, steps, regrounds, bodies = _clean()
    spoken = "Whole page, scaled to fit one screen: the towel is at the bottom."
    _find(trials, "B", "S4", 0)["voiced"] = [spoken, TOWEL_SAID]
    r = _request(bodies, "B", "S4", 0, "generate:BS40-3")
    r["messages"][-1]["parts"] = [_t(spoken + " " + OV_CALL + ov_action(1.9, False))]
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == []
    r["messages"][1:1] = [_user(_t(spoken)), _asst("Earlier.")]
    found = cf(trials, steps, regrounds, bodies, HAND_ARMS)
    assert found and all("B S4#0" in f for f in found), found


@pytest.mark.parametrize("which", IMPLS)
def test_repeated_calls_on_an_unchanged_page_are_not_a_void(which: str) -> None:
    """Review F1 (mutant: overviews keyed by sha alone, the last record winning): two ``page_overview`` calls in one
    turn on an unchanged page give one sha; each overview sits right before the current frame of the step after its
    own call -- no finding. The out-of-step companion is ``repeated-call-overview-out-of-step``."""
    cf = _cf(which)
    trials, steps, regrounds, bodies = _clean()
    trials, bodies = _with_repeated_calls(trials, bodies)
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == [], cf(
        trials, steps, regrounds, bodies, HAND_ARMS
    )


@pytest.mark.parametrize("which", IMPLS)
def test_repair_right_after_an_overview_step_is_not_a_void(which: str) -> None:
    """Review F2 (mutant: only ``tool_step`` bodies may carry the overview): the step-2 call of B1 S4#0 is repaired;
    the repair continues the step-2 pass-1 prompt, overview included (``step`` 2 on its body line) -- no finding. The
    wrong-step companion is ``repair-of-the-wrong-step``."""
    cf = _cf(which)
    trials, steps, regrounds, bodies = _clean()
    r = _repair_of(_request(bodies, "B", "S4", 0, "generate:BS40-2"), "repair:BS40-2r", 2)
    _find(bodies, "B", "S4", 0)["requests"].append(r)
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == [], cf(
        trials, steps, regrounds, bodies, HAND_ARMS
    )


def _wire_image(data: bytes) -> dict[str, Any]:
    import base64

    url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def _wire_body(*, overview: bool) -> dict[str, Any]:
    """A forwarded body on the wire: the final user message ``[request, (overview, its label,) frame, screen label]``,
    the tools declaring ``page_overview``."""
    content: list[dict[str, Any]] = [{"type": "text", "text": REQUEST}]
    if overview:
        content += [_wire_image(b"OVERVIEW-JPEG"), {"type": "text", "text": ov_label(1.9, False)}]
    content += [_wire_image(b"FRAME-JPEG"), {"type": "text", "text": SCREEN_SAME}]
    return {
        "messages": [
            {"role": "system", "content": "You are Astra."},
            {"role": "user", "content": content},
            {"role": "assistant", "content": "Let me look. " + OV_CALL},
        ],
        "tools": [{"type": "function", "function": {"name": n}} for n in OV_TOOLS],
    }


def _request_record(
    phase: str, request_id: str, attempt_id: str, step: int | None = None
) -> dict[str, Any]:
    data: dict[str, Any] = {"phase": phase, "request_id": request_id, "attempt_id": attempt_id}
    if step is not None:
        data["step"] = step
    return {"kind": "llm.request", "data": data}


#: (llm.request records in log order, the repair body's (phase, step)), by hand from the turn machine's ids (pass 1
#: ``{turn}-gen`` / ``{turn}-gen-s{k}``, the turn's one repair ``{turn}-repair``) and review F2
REPAIR_STEPS: dict[str, tuple[list[dict[str, Any]], tuple[str, int | None]]] = {
    "repair-of-step-2": (
        [
            _request_record("generate", "t1-gen", "e-gen-1"),
            _request_record("tool_step", "t1-gen-s2", "e-gen-s2-2", 2),
            _request_record("tool_repair", "t1-repair", "e-rep-3"),
        ],
        ("tool_repair", 2),
    ),
    "repair-of-step-1": (
        [
            _request_record("generate", "t1-gen", "e-gen-1"),
            _request_record("tool_repair", "t1-repair", "e-rep-3"),
        ],
        ("tool_repair", 1),
    ),
    # another turn's pass 1 never gives this turn's repair its step
    "repair-of-another-turns-step": (
        [
            _request_record("generate", "t1-gen", "e-gen-1"),
            _request_record("tool_step", "t1-gen-s2", "e-gen-s2-2", 2),
            _request_record("tool_repair", "t2-repair", "e-rep-3"),
        ],
        ("tool_repair", None),
    ),
    "oddly-named-repair": (
        [
            _request_record("generate", "t1-gen", "e-gen-1"),
            _request_record("tool_step", "t1-gen-s2", "e-gen-s2-2", 2),
            _request_record("tool_repair", "t1-fix", "e-rep-3"),
        ],
        ("tool_repair", None),
    ),
}


@pytest.mark.parametrize("case", list(REPAIR_STEPS))
def test_body_line_gives_a_repair_its_pass1_step(case: str) -> None:
    """Review F2, the record side (mutant: repair steps left ``None``): ``body_line`` over session records gives a
    ``tool_repair`` body the step of the pass-1 request it repairs -- after ``t1-gen`` and ``t1-gen-s2``,
    ``t1-repair`` is step 2; after ``t1-gen`` alone, step 1 -- and ``None`` for a repair it cannot place (another
    turn's id, an odd name). End to end through the oracle: the step-2 repair carrying the step-1 overview right
    before the current frame is no finding; an unplaced repair carrying it is an out-of-place finding, never a false
    pass."""
    records, want = REPAIR_STEPS[case]
    raw = {
        "generate:e-gen-1": _wire_body(overview=False),
        "generate:e-gen-s2-2": _wire_body(overview=True),
        "repair:e-rep-3": _wire_body(overview=True),
    }
    if not any(r["data"]["phase"] == "tool_step" for r in records):
        del raw["generate:e-gen-s2-2"]
    ov = {
        "step": 1,
        "sha": hashlib.sha256(b"OVERVIEW-JPEG").hexdigest()[:16],
        "screens": 1.9,
        "clipped": False,
        "width": 838,
        "height": 798,
    }
    line = api("body_line")("B", "S4", 0, raw, records, [ov])
    (repair,) = [r for r in line["requests"] if r["label"] == "repair:e-rep-3"]
    assert (repair["phase"], repair["step"]) == want, (case, repair["phase"], repair["step"])
    trial = _tr("B", "S4", 0, DEALS_EMPTY, [TOWEL_SAID])
    found = oracle.configuration_findings([trial], [], [], [line], HAND_ARMS)
    if want[1] == 2:
        assert found == [], found
    elif want[1] is None:
        assert any("e-rep-3" in f and "out of place" in f for f in found), found


#: review F3: one turn-line key off its arm row per case (the reviewer's surviving mutants: live_feedback=None, echo
#: pixels, recovery flipped, 560 visual tokens passed to the Session); (trial, field, value)
LOGGED_FAULTS: dict[str, tuple[tuple[str, str, int], str, Any]] = {
    "live-feedback-not-passed": (("B", "S4", 0), "live_feedback_logged", None),
    "echo-pixels": (("A", "S1", 0), "live_feedback_logged", {**LABEL_FB, "echo": "pixels"}),
    "extent-on": (
        ("B", "S5", 0),
        "live_feedback_logged",
        {"screen_label": True, "page_extent": True, "echo": "model"},
    ),
    "recovery-flipped": (("A", "S2", 0), "recover_thought_calls_logged", False),
    "recovery-not-passed": (("B", "S4", 1), "recover_thought_calls_logged", None),
    "visual-tokens-560": (("B", "S2", 0), "visual_tokens_logged", 560),
    "guard-not-logged": (("A", "S4", 1), "degeneracy_guard_logged", None),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("case", list(LOGGED_FAULTS))
def test_logged_settings_off_their_rows(which: str, case: str) -> None:
    """Review F3 (B4's "feedback, echo or guard settings off their rows", judged on the turn line, not on the trial's
    copy of the row): each key the session logs -- ``live_feedback`` with the browser's echo, ``recover_thought_calls``,
    ``visual_tokens``, ``degeneracy_guard`` -- off its arm row is a finding naming the trial; the same value on a
    trial that wrote no turn line (timed out, T-0135b A7) is not judged (its keys are ``None``)."""
    cf = _cf(which)
    (arm, task, repeat), field, value = LOGGED_FAULTS[case]
    trials, steps, regrounds, bodies = _clean()
    _find(trials, arm, task, repeat)[field] = value
    found = cf(trials, steps, regrounds, bodies, HAND_ARMS)
    tid = f"{arm} {task}#{repeat}"
    assert found and all(tid in f for f in found) and any(field in f for f in found), (case, found)
    t = _find(trials, arm, task, repeat)
    t.update(timed_out=True, session_id=None)
    for k in (
        "live_feedback_logged",
        "recover_thought_calls_logged",
        "visual_tokens_logged",
        "degeneracy_guard_logged",
    ):
        t[k] = None
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == [], (
        "T-0135b A7: no turn line, nothing to judge"
    )


PROBES_OK: dict[str, Any] = {
    **{p: {"passed": True} for p in ("p1", "p2", "p3")},
    "p6": {**P6_OK, "passed": True},
    "p7": {**P7_OK, "passed": True},
}
#: a turn that raised before any body went out (O-11): nothing scanned
RAISED = {"scanned_requests": [], "exception": "RuntimeError('harness')", "budgets_sent": None}


@pytest.mark.parametrize("which", IMPLS)
def test_errored_turn_is_acceptance_not_configuration(which: str) -> None:
    """O-11 (T-0134 O-11): a trial that raised before sending any body is not judged by B4, even with an arm-A body
    declaring the tool in its record (no finding, so not ``void``); it is an acceptance failure, so the run is
    ``inconclusive`` naming ``acceptance`` and the trial. Companion: the same body on a trial that did send is a
    finding."""
    cf = _cf(which)
    trials, steps, regrounds, bodies = _clean()
    _find(trials, "A", "S1", 0).update(RAISED)
    _request(bodies, "A", "S1", 0, "generate:AS10-1")["tools"] = list(OV_TOOLS)
    assert cf(trials, steps, regrounds, bodies, HAND_ARMS) == []
    sent = _clean()
    _request(sent[3], "A", "S1", 0, "generate:AS10-1")["tools"] = list(OV_TOOLS)
    assert cf(*sent, HAND_ARMS), "a sending trial is judged"
    if which == "harness":
        report = api("build_report")(trials, steps, regrounds, bodies, PROBES_OK, arms=HAND_ARMS)
        label, reason = report["decision"]["label"], report["decision"]["reason"]
        assert report["configuration_findings"] == [], report["configuration_findings"]
    else:
        inputs = {
            a: oracle.decision_inputs(
                [t for t in trials if t["arm"] == a], [s for s in steps if s["arm"] == a], []
            )
            for a in "AB"
        }
        label, reason = oracle.decide_with_reason(
            inputs,
            p1_passed=True,
            configuration_findings=cf(trials, steps, regrounds, bodies, HAND_ARMS),
        )
    assert (label, reason) == (
        "inconclusive",
        "arm A ineligible: acceptance (first failing trial S1#0)",
    ), (
        label,
        reason,
    )


# =====================================================================================================
# A10/B1 turn-line evidence (the eval's ``arm_setting_findings``; T-0135b A7's exemption)
# =====================================================================================================


def test_arm_setting_findings() -> None:
    """B1/A10 evidence (eval oracle): the clean records have none and no exemption. A trial with a turn line but no
    A10 key in arm B, an A10 key in arm A, or the guard not logged on, is a finding whatever its success; a timed-out
    trial (no turn line, T-0135b A7) is exempted from the turn-line keys only -- its arm's ``page_overview`` and
    ``tools_version`` are still judged."""
    trials, *_ = _clean()
    assert oracle.arm_setting_findings(trials) == ([], [])
    for success in (False, True):
        t = {**_find(trials, "B", "S4", 0), "page_overview_logged": None, "success": success}
        got, ex = oracle.arm_setting_findings([t])
        assert got == ["B S4#0: page_overview_logged None"] and ex == [], (success, got, ex)
    t = {**_find(trials, "A", "S4", 0), "page_overview_logged": _counts()}
    assert (
        oracle.arm_setting_findings([t])[0] and "A S4#0" in oracle.arm_setting_findings([t])[0][0]
    )
    t = {**_find(trials, "A", "S4", 0), "degeneracy_guard_logged": None}
    assert oracle.arm_setting_findings([t])[0] == ["A S4#0: degeneracy_guard_logged None"]
    timed_out = {
        **_find(trials, "B", "S4", 1),
        "timed_out": True,
        "session_id": None,
        "page_overview_logged": None,
        "degeneracy_guard_logged": None,
    }
    assert oracle.arm_setting_findings([timed_out]) == ([], ["B S4#1"])
    got, ex = oracle.arm_setting_findings([{**timed_out, "tools_version": V1}])
    assert got == [f"B S4#1: tools_version {V1!r}"] and ex == ["B S4#1"], (got, ex)
    got, ex = oracle.arm_setting_findings([{**timed_out, "page_overview": False}])
    assert got == ["B S4#1: page_overview False"] and ex == ["B S4#1"], (got, ex)


# =====================================================================================================
# B3: per-arm counts by hand
# =====================================================================================================

ZERO_CALLS = dict.fromkeys(TASK_IDS, 0)
#: by hand from ``_hand_records`` (its docstring):
#: A -- success A1 (S4) and A4 (S1); false claims A2 (S4) and A3 (S2: non-S4); over-click A3; S4 scrolls A1; no
#:      overview anywhere; latency: every step "other", done [300, 350, 360, 310, 320, 330, 340, 370] -> nearest-rank
#:      p50 330 (4th of 8), p95 370 (8th).
#: B -- success B1 (S4) and B3 (S5); B4 has the right cart but ended ``degenerate`` (a failure; not an engine error
#:      though its aborted step answered HTTP 0); false claim B2 (S4); no over-click; S4 overview-call turns B1, B2;
#:      S4 scrolls B1 (step 2); overview calls S4 2, S5 1; A10 sums calls 3, shown 2, fits 1; narration B2 ("whole
#:      page"); overview steps B1#2 900 and B2#2 1100 (p50 900, p95 1100); others [300, 305, 320, 310, 400, 410, 330]
#:      (p50 320, 4th of 7; p95 410, 7th); B4's step 2 has no done_ms.
HAND: dict[str, dict[str, Any]] = {
    "A": {
        "success": 2,
        "per_task": {"S1": 1, "S2": 0, "S3": 0, "S4": 1, "S5": 0},
        "s4": 1,
        "false_claims": 2,
        "false_claims_non_s4": 1,
        "over_clicks": 1,
        "s4_overview_call_turns": 0,
        "s4_scrolls": 1,
        "overview_calls": dict(ZERO_CALLS),
        "overview_counts": _counts(),
        "narration": 0,
        "degenerate": 0,
        "engine_errors": 0,
        "latency": {
            "overview": {"n": 0, "p50": None, "p95": None},
            "other": {"n": 8, "p50": 330.0, "p95": 370.0},
        },
    },
    "B": {
        "success": 2,
        "per_task": {"S1": 0, "S2": 0, "S3": 0, "S4": 1, "S5": 1},
        "s4": 1,
        "false_claims": 1,
        "false_claims_non_s4": 0,
        "over_clicks": 0,
        "s4_overview_call_turns": 2,
        "s4_scrolls": 1,
        "overview_calls": {**ZERO_CALLS, "S4": 2, "S5": 1},
        "overview_counts": _counts(calls=3, shown=2, fits=1),
        "narration": 1,
        "degenerate": 1,
        "engine_errors": 0,
        "latency": {
            "overview": {"n": 2, "p50": 900.0, "p95": 1100.0},
            "other": {"n": 7, "p50": 320.0, "p95": 410.0},
        },
    },
}
#: Wilson 95 % by the closed form (z = 1.959964, computed in a scratch script): 2/4 -> [0.15004, 0.84996];
#: 1/2 -> [0.09453, 0.90547]
HAND_CI = {"success_ci": (0.15004, 0.84996), "s4_ci": (0.09453, 0.90547)}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["A", "B"])
def test_arm_metrics_hand_counts(which: str, arm: str) -> None:
    """B3: every count of four hand-built turns per arm (S4 by ``score_v2``; success and per task; false claims,
    total and non-S4; over-clicks; S4 overview-call turns and scrolls; overview calls per task; A10's sums;
    narration; ``degenerate`` ends; engine errors; the latency of overview-bearing steps vs others) and the Wilson
    intervals of success and S4."""
    trials, steps, _ = _hand_records(arm)
    got = impl(which, "derive_arm")(trials, steps)
    for key, want in HAND[arm].items():
        assert got[key] == want, (arm, key, got[key], want)
    for key, (want_lo, want_hi) in HAND_CI.items():
        lo, hi = got[key]
        assert abs(lo - want_lo) < 5e-5 and abs(hi - want_hi) < 5e-5, (key, got[key])


def _s4(arm: str, repeat: int, **kw: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One S4 turn that names the towel with an empty cart (a ``score_v2`` success unless forced)."""
    t = _tr(arm, "S4", repeat, DEALS_EMPTY, [TOWEL_SAID], **kw)
    return t, [_st(arm, "S4", repeat, 1, None, 300.0, voiced=(TOWEL_SAID,))]


FORCED: dict[str, tuple[dict[str, Any], dict[str, Any], int, int]] = {
    # (trial fields, step fields, success, engine errors), by hand from B5
    "plain": ({}, {}, 1, 0),
    "degenerate": ({"tool_loop_end": "degenerate"}, {}, 0, 0),
    "degenerate-with-an-aborted-attempt": (
        {"tool_loop_end": "degenerate"},
        {"http_status": 0},
        0,
        0,
    ),
    "timed-out": ({"timed_out": True}, {}, 0, 1),
    "http-500": ({}, {"http_status": 500}, 0, 1),
    "a-503-retry": ({}, {"retries": [{"http_status": 503}]}, 0, 1),
    "engine-down-after": ({"engine_up_after": False}, {}, 0, 1),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(FORCED))
def test_forced_failures(which: str, row: str) -> None:
    """B5 (H-T140-1): an engine-error trial (timeout, HTTP, transport, a non-200 retry, the engine down after) and a
    T-0142 ``degenerate`` end count as failed trials in every metric -- an S4 turn whose cart and words are right is
    not a success -- and only the first kind is an engine error (reading 1)."""
    t_over, s_over, success, errors = FORCED[row]
    t, steps = _s4("B", 0, **t_over)
    steps = [{**steps[0], **s_over}]
    got = impl(which, "derive_arm")([t], steps)
    assert (got["success"], got["s4"], got["engine_errors"]) == (success, success, errors), (
        row,
        got,
    )
    assert got["degenerate"] == (1 if "degenerate" in row else 0), got
    # B-2's false-claim half (review, minor): the turn's last step voiced the right answer, which ``score_v2`` passes;
    # a forced failure is judged as not succeeded, so the claim is false -- except a timed-out turn, which T-0131's
    # proxy never judges
    assert score_v2_passes(t), "premise: score_v2 passes the trial on its own"
    want_fc = 0 if row in ("plain", "timed-out") else 1
    assert got["false_claims"] == want_fc, (row, got["false_claims"])


def score_v2_passes(t: dict[str, Any]) -> bool:
    return bool(oracle.score_v2(str(t["task"]), t.get("state"), " ".join(t.get("voiced") or [])))


@pytest.mark.parametrize("which", IMPLS)
def test_empty_arm_has_no_interval(which: str) -> None:
    """O-11: an arm with no trials (a probe-void run writes none) has ``None`` intervals and zero counts."""
    if which == "harness":
        probes = {**PROBES_OK, "p6": {**P6_OK, "outcome": None, "passed": False}}
        report = api("build_report")(
            [], [], [], [], probes, arms=HAND_ARMS, stopped="not run: a pre-flight probe failed"
        )
        arm = report["arms"]["A"]
        assert report["void"] is True and report["decision"]["label"] == "void", report["decision"]
    else:
        arm = oracle.derive_arm([], [])
    assert arm["success_ci"] is None and arm["s4_ci"] is None, arm
    assert arm["success"] == 0 and arm["engine_errors"] == 0, arm


#: Newcombe (1998) published values (as T-0132's rows) and Wilson's
WILSON = {(81, 263): (0.2553, 0.3662), (0, 20): (0.0, 0.1611), (1, 29): (0.0061, 0.1718)}
NEWCOMBE = {(56, 70, 48, 80): (0.0524, 0.3339), (9, 10, 3, 10): (0.1705, 0.8090)}


@pytest.mark.parametrize("which", IMPLS)
def test_wilson_and_newcombe_hand_values(which: str) -> None:
    """B3: Wilson 95 % and Newcombe's method 10 equal the published values to four decimals."""
    w, nc = impl(which, "wilson"), impl(which, "newcombe")
    for args, want in WILSON.items():
        assert all(abs(a - b) < 5e-5 for a, b in zip(w(*args), want, strict=True)), (args, w(*args))
    for args, want in NEWCOMBE.items():
        got = nc(*args)
        assert all(abs(a - b) < 5e-5 for a, b in zip(got, want, strict=True)), (args, got)


# =====================================================================================================
# B5: the decision rule
# =====================================================================================================


def _arm(**over: Any) -> dict[str, Any]:
    base = {
        "accepted": True,
        "safety": 0,
        "engine_errors": 0,
        "success": 65,
        "s4": 0,
        "false_claims_non_s4": 5,
        "over_clicks": 3,
        "s4_overview_call_turns": 10,
        "first_failing": {"acceptance": None, "safety": None, "engine": None},
    }
    return base | over


def _inputs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {"A": _arm(**a), "B": _arm(**b)}


#: (inputs, label) by hand from B5 (H-T140-1): every margin exactly (adopt) and one past (not)
DECISIONS: dict[str, tuple[dict[str, Any], str]] = {
    "s4-exactly-plus-5": (_inputs({}, {"s4": 5}), "overview"),
    "s4-plus-4": (_inputs({}, {"s4": 4}), "no change"),
    "s4-margin-is-relative": (_inputs({"s4": 3}, {"s4": 8}), "overview"),
    "s4-margin-relative-minus-1": (_inputs({"s4": 3}, {"s4": 7}), "no change"),
    "success-exactly-minus-3": (_inputs({}, {"s4": 5, "success": 62}), "overview"),
    "success-minus-4": (_inputs({}, {"s4": 5, "success": 61}), "no change"),
    "false-claims-exactly-plus-2": (_inputs({}, {"s4": 5, "false_claims_non_s4": 7}), "overview"),
    "false-claims-plus-3": (_inputs({}, {"s4": 5, "false_claims_non_s4": 8}), "no change"),
    "over-clicks-exactly-plus-2": (_inputs({}, {"s4": 5, "over_clicks": 5}), "overview"),
    "over-clicks-plus-3": (_inputs({}, {"s4": 5, "over_clicks": 6}), "no change"),
    "every-margin-at-its-boundary": (
        _inputs({}, {"s4": 5, "success": 62, "false_claims_non_s4": 7, "over_clicks": 5}),
        "overview",
    ),
    # the margins are B's allowance over A: A worse than B never blocks
    "a-worse-on-every-count": (
        _inputs({"success": 40, "false_claims_non_s4": 20, "over_clicks": 15}, {"s4": 5}),
        "overview",
    ),
    "unused-at-4-call-turns": (_inputs({}, {"s4_overview_call_turns": 4}), "overview unused"),
    "used-at-5-call-turns": (_inputs({}, {"s4_overview_call_turns": 5}), "no change"),
    "unused-though-s4-plus-4": (
        _inputs({}, {"s4": 4, "s4_overview_call_turns": 4}),
        "overview unused",
    ),
    # the adoption clauses are checked first: a gain with few calls is still adopted
    "adopted-with-4-call-turns": (_inputs({}, {"s4": 5, "s4_overview_call_turns": 4}), "overview"),
    # arm A's call count is never read (it cannot call the tool)
    "arm-a-calls-not-read": (
        _inputs({"s4_overview_call_turns": 0}, {"s4_overview_call_turns": 5}),
        "no change",
    ),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(DECISIONS))
def test_decision_rule(which: str, row: str) -> None:
    """B5 (H-T140-1; reviewer mutation 8, a margin off by one): hand tables at every boundary -- S4 +5/20, success
    -3/100, non-S4 false claims +2/80, over-clicks +2/60 -- each exactly (adopt) and one past (not); ``overview
    unused`` at 4 vs 5 S4 call-turns in B."""
    inputs, want = DECISIONS[row]
    assert impl(which, "decide_with_reason")(inputs, p1_passed=True) == (want, None), row
    assert impl(which, "decide")(inputs, p1_passed=True) == want


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["A", "B"])
def test_engine_eligibility_edge(which: str, arm: str) -> None:
    """B5's changed engine clause (reviewer mutation 9, back to "any error"): 2 engine errors in an arm are eligible
    (the rule decides: ``overview``), 3 are ``inconclusive`` naming the arm, ``engine`` and the first failing
    trial."""
    why = impl(which, "decide_with_reason")
    inputs = json.loads(json.dumps(DECISIONS["s4-exactly-plus-5"][0]))
    inputs[arm] |= {
        "engine_errors": 2,
        "first_failing": {"acceptance": None, "safety": None, "engine": "S2#4"},
    }
    assert why(inputs, p1_passed=True) == ("overview", None)
    inputs[arm]["engine_errors"] = 3
    assert why(inputs, p1_passed=True) == (
        "inconclusive",
        f"arm {arm} ineligible: engine (first failing trial S2#4)",
    )


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["A", "B"])
@pytest.mark.parametrize("clause", ["acceptance", "safety", "engine"])
def test_ineligible_arm(which: str, arm: str, clause: str) -> None:
    """B5 (T-0132 C6b): any arm ineligible -> ``inconclusive``, the reason naming the arm, the clause and the first
    failing trial, exactly; every margin met does not save it."""
    flag = {
        "acceptance": {"accepted": False},
        "safety": {"safety": 1},
        "engine": {"engine_errors": 3},
    }[clause]
    bad = json.loads(json.dumps(DECISIONS["every-margin-at-its-boundary"][0]))
    bad[arm] |= flag | {
        "first_failing": {
            c: ("S4#7" if c == clause else None) for c in ("acceptance", "safety", "engine")
        }
    }
    got = impl(which, "decide_with_reason")(bad, p1_passed=True)
    assert got == ("inconclusive", f"arm {arm} ineligible: {clause} (first failing trial S4#7)"), (
        got
    )


@pytest.mark.parametrize("which", IMPLS)
def test_decision_order(which: str) -> None:
    """B5's order, each step against every later one: P1 failed -> ``inconclusive`` (``P1 failed``) over everything;
    a configuration finding -> ``void`` (``configuration: <first>``); a probe void (P2, P3, P6 or P7) -> ``void``
    over a stop and an ineligible arm; stopped -> ``incomplete`` (its reason) over an ineligible arm; an ineligible
    arm over the rule."""
    why = impl(which, "decide_with_reason")
    good = DECISIONS["s4-exactly-plus-5"][0]
    bad = json.loads(json.dumps(good))
    bad["B"] |= {
        "accepted": False,
        "first_failing": {"acceptance": "S1#0", "safety": None, "engine": None},
    }
    everything = {"void": True, "stopped": "RUN_BUDGET_S", "configuration_findings": ["X", "Y"]}
    assert why(bad, p1_passed=False, **everything) == ("inconclusive", "P1 failed")
    assert why(bad, p1_passed=True, **everything) == ("void", "configuration: X")
    assert why(bad, p1_passed=True, void=True, stopped="RUN_BUDGET_S")[0] == "void"
    assert why(bad, p1_passed=True, stopped="RUN_BUDGET_S") == ("incomplete", "RUN_BUDGET_S")
    assert why(bad, p1_passed=True)[0] == "inconclusive"
    assert why(good, p1_passed=True, configuration_findings=[]) == ("overview", None)


def _engine_run(errors: int) -> tuple[list[Any], list[Any]]:
    """Arm B: five S4 successes, then ``errors`` engine-error trials (a timeout, an HTTP 500, the engine down after,
    another timeout ...), then a ``degenerate`` end whose aborted attempt answered HTTP 0, then a trial whose session
    log is incomplete when ``errors`` is negative (an acceptance failure)."""
    trials: list[Any] = []
    steps: list[Any] = []
    for r in range(5):
        t, s = _s4("B", r)
        trials.append(t)
        steps += s
    kinds = [
        ({"timed_out": True}, {}),
        ({}, {"http_status": 500}),
        ({"engine_up_after": False}, {}),
    ]
    for k in range(max(errors, 0)):
        t_over, s_over = kinds[k % len(kinds)]
        t, s = _s4("B", 10 + k, **t_over)
        trials.append(t)
        steps += [{**s[0], **s_over}]
    t, s = _s4("B", 19, tool_loop_end="degenerate")
    trials.append(t)
    steps += [{**s[0], "http_status": 0}]
    if errors < 0:
        t, s = _s4("B", 18, log_complete=False)
        trials.append(t)
        steps += s
    return trials, steps


def _di(which: str, trials: list[Any], steps: list[Any]) -> dict[str, Any]:
    if which == "harness":
        return dict(api("decision_inputs")(trials, steps, []))
    return oracle.decision_inputs(trials, steps, [])


@pytest.mark.parametrize("which", IMPLS)
def test_decision_inputs_engine_errors_are_not_acceptance(which: str) -> None:
    """B5 from records (reading 1): two engine errors (a timeout, an HTTP 500) and a ``degenerate`` end leave the arm
    accepted, with ``engine_errors`` 2 and the first engine-error trial named; a third (the engine down after) makes
    3; an incomplete session log is still an acceptance failure, named. The forced failures count as failures:
    success 5 of the 5 plain turns only."""
    two = _di(which, *_engine_run(2))
    assert (two["accepted"], two["engine_errors"], two["success"], two["s4"]) == (True, 2, 5, 5), (
        two
    )
    assert two["first_failing"] == {"acceptance": None, "safety": None, "engine": "S4#10"}, two
    three = _di(which, *_engine_run(3))
    assert (three["accepted"], three["engine_errors"]) == (True, 3), three
    log = _di(which, *_engine_run(-1))
    assert log["accepted"] is False and log["first_failing"]["acceptance"] == "S4#18", log
    assert log["engine_errors"] == 0, log


# =====================================================================================================
# build_report wiring (harness only)
# =====================================================================================================


@pytest.mark.parametrize("fault", [False, True])
def test_build_report_wiring(fault: bool) -> None:
    """``build_report`` (mutants: findings dropped; the engine clause back to "any error"; inputs from T-0111g's
    ``score``): the hand records plus three engine-error S4 trials in arm B are ``inconclusive`` with exactly ``arm B
    ineligible: engine (first failing trial S4#10)``; with an arm-A body declaring the tool the run is ``void`` with a
    reason ``configuration: A ...``. The differences are S4 and success B-A."""
    trials, steps, regrounds, bodies = _clean()
    more_t, more_s = _engine_run(3)
    trials += more_t[5:-1]
    steps += more_s[5:-1]
    if fault:
        _request(bodies, "A", "S4", 1, "generate:AS41-1")["tools"] = list(OV_TOOLS)
    report = api("build_report")(trials, steps, regrounds, bodies, PROBES_OK, arms=HAND_ARMS)
    d = report["decision"]
    if fault:
        assert d["label"] == "void" and str(d["reason"]).startswith("configuration: A S4#1"), d
    else:
        assert (d["label"], d["reason"]) == (
            "inconclusive",
            "arm B ineligible: engine (first failing trial S4#10)",
        ), d
        assert report["configuration_findings"] == []
    assert d["inputs"]["B"]["engine_errors"] == 3, d["inputs"]["B"]
    assert d["inputs"]["A"]["first_failing"] == {"acceptance": None, "safety": None, "engine": None}
    assert set(report["differences"]) == {"s4 B-A", "success B-A"}, report["differences"]


def test_build_report_counts_equal_the_oracle() -> None:
    """``build_report`` over the hand records of both arms: each arm's B3 counts are ``HAND``'s, the decision inputs
    are the oracle's re-derivation, the label is the oracle's, and the differences are the oracle's Newcombe
    intervals at B3's fixed denominators."""
    trials, steps, regrounds, bodies = _clean()
    report = api("build_report")(trials, steps, regrounds, bodies, PROBES_OK, arms=HAND_ARMS)
    inputs = {}
    for arm in ("A", "B"):
        for key, want in HAND[arm].items():
            assert report["arms"][arm][key] == want, (arm, key, report["arms"][arm][key], want)
        mine = [[r for r in rows if r["arm"] == arm] for rows in (trials, steps)]
        inputs[arm] = oracle.decision_inputs(mine[0], mine[1], [])
        assert report["decision"]["inputs"][arm] == inputs[arm], (
            arm,
            report["decision"]["inputs"][arm],
        )
    assert (report["decision"]["label"], report["decision"]["reason"]) == oracle.decide_with_reason(
        inputs, p1_passed=True
    )
    want_diff = oracle.differences({a: HAND[a] for a in ("A", "B")})
    for key, (lo, hi) in want_diff.items():
        got = report["differences"][key]
        assert abs(got[0] - lo) < 1e-9 and abs(got[1] - hi) < 1e-9, (key, got, (lo, hi))


# =====================================================================================================
# main on CPU: the dress (T-0131's engine rig, the FakeLiveBrowser with a scripted overview, a fake LLM)
# =====================================================================================================

#: deals-like metrics: 1520 CSS px over an 800 px viewport, 1.9 screens, not clipped
DRESS_METRICS = {"scroll_y": 0.0, "viewport_height": 800.0, "content_height": 1520.0}


def _dress_overview() -> Any:
    from local_astra.contracts.perception import Frame, Source
    from tests.unit._live_browser_support import jpeg_frame

    return Frame(
        source=Source.WINDOW,
        ts_ms=0,
        width=838,
        height=798,
        jpeg=jpeg_frame("dress-overview").jpeg,
        phash="0d0d0d0d0d0d0d0d",
    )


def _dress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
    *,
    turns: tuple[tuple[str, str, int], ...] = (("A", "S4", 0), ("B", "S4", 0)),
    probes: Any = None,
    p6: Any = None,
    p7: Any = None,
    past_budget: bool = False,
) -> tuple[int, Path, Any]:
    """A CPU run of ``main`` (T-0135's dress, adapted): S4 in each arm of ``turns``; step 1 says "Let me look." and
    calls ``page_overview`` (v1's judge refuses it in arm A; arm B's kit executes it over a 1.9-screen page); step 2
    names the towel and $4.75. Returns (exit code, out-dir, the engine rig's state)."""
    from local_astra import cli
    from local_astra.computer import executor as executor_mod
    from local_astra.pages import server as server_mod
    from tests.unit._live_browser_support import FakeLiveBrowser, jpeg_frame, live_kit
    from tests.unit._tool_turn_support import ToolLlm, answer, tool
    from tests.unit.test_t0140_session import overview_kit

    h = harness()

    def git(*args: str) -> str:
        return "" if args[:1] == ("status",) else t31.GIT + "\n"

    monkeypatch.setattr(h, "_git", git)
    for value in list(vars(h).values()):
        if isinstance(value, ModuleType) and hasattr(value, "_git") and value is not h:
            path = Path(getattr(value, "__file__", "") or "").resolve()
            if path.parent == HARNESS.parent.resolve():
                monkeypatch.setattr(value, "_git", git)
    kits = _stand_in_kits()
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    out = tmp_path / "dress"
    rig = types.SimpleNamespace(
        h=h, cli=cli, kit=kits[False], mp=monkeypatch, out=out, capsys=capsys
    )
    st = t31._engine(rig)
    overview = _dress_overview()

    class Pages:
        origin = "http://127.0.0.1:9"

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class Page:
        async def evaluate(self, script: str, arg: Any = None) -> Any:
            if arg is None:
                return {"page": "deals", "cart": []}
            return {"boxes": [], "element": None}

    class Browser(FakeLiveBrowser):
        def __init__(
            self,
            *,
            now_ms: Any = None,
            viewport: Any = (1280, 800),
            echo: str = "pixels",
            **kw: Any,
        ):
            n = len(st.browsers)
            super().__init__(
                [jpeg_frame(f"onset-{n}")],
                post_frames=[jpeg_frame(f"after-{n}-{k}") for k in (1, 2, 3)],
                now_ms=now_ms or (lambda: 0),
                echo=echo,
            )
            self.overview_image = overview
            self.page_metrics = dict(DRESS_METRICS)
            self.driver = types.SimpleNamespace(viewport=tuple(viewport), page=Page())
            st.browsers.append(self)
            st.pending_browser = self

    def turn_llm(browser: Any) -> Any:
        llm = ToolLlm([answer("Let me look. "), tool(OV_CALL)])
        llm.script_steps([answer(TOWEL_SAID)])
        return llm

    def load(profile: Any, *, live_browser: Any = None, page_overview: bool = False) -> Any:
        if live_browser is None:
            return (kits[bool(page_overview)], None)
        return (overview_kit(live_browser) if page_overview else live_kit(live_browser), None)

    monkeypatch.setattr(server_mod, "StartPageServer", Pages)
    monkeypatch.setattr(executor_mod, "BrowserToolExecutor", Browser)
    monkeypatch.setattr(cli, "load_tool_kit", load)
    monkeypatch.setattr(t31, "_turn_llm", turn_llm)
    monkeypatch.setattr(h, "run_probes", lambda base_url, model: probes or t31._probes())
    monkeypatch.setattr(
        h, "probe_p6", lambda *a, **k: copy.deepcopy(p6 if p6 is not None else P6_OK)
    )
    monkeypatch.setattr(h, "probe_p7", lambda *a, **k: dict(p7 if p7 is not None else P7_OK))
    monkeypatch.setattr(h, "order", lambda seed: list(turns))
    if past_budget:
        monkeypatch.setattr(h, "PROCESS_START_S", time.monotonic() - h.RUN_BUDGET_S - 1)
    rc = h.main(["--base-url", t31.VALID_URL, "--out-dir", str(out)])
    return rc, out, st


def _read(out: Path) -> dict[str, Any]:
    return {
        "report": json.loads((out / "report.json").read_text()),
        "manifest": json.loads((out / "manifest.json").read_text()),
        "probes": json.loads((out / "probes.json").read_text()),
        **{
            k: oracle._jsonl(out / f"{k}.jsonl") for k in ("trials", "steps", "regrounds", "bodies")
        },
    }


def _dress_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> dict[str, Any]:
    rc, out, _ = _dress(tmp_path, monkeypatch, capsys)
    err = capsys.readouterr().err
    assert rc == 0, f"exit {rc}\n{err[-3000:]}"
    return _read(out)


def _dress_default_is_heads(manifest: dict[str, Any]) -> None:
    """The manifest's live default is HEAD's as ``cli`` gives it, with the overview off; its arm rows and sha are the
    oracle's over it."""
    from local_astra import cli
    from local_astra.config import load

    d = manifest["live_default"]
    assert d == _heads_default(cli, load().profiles["lean"]), d
    assert oracle.live_default_findings(d) == [], oracle.live_default_findings(d)
    rows = oracle.arm_rows(d)
    assert manifest["regime"]["arms"] == rows, manifest["regime"]["arms"]
    assert manifest["arm_table_sha256"] == oracle.arm_table_sha(rows)


def test_eval_oracle_re_derives_the_dress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """B7 on CPU: over the files a dress of ``main`` wrote, the eval oracle's re-derivations agree with
    ``report.json`` -- per arm the B3 counts, T-0131's sections, the safety counts (all 0) and the decision inputs;
    the findings (none: the dress is clean); the probes and ``void``; the label and reason; the differences; the
    turn-line evidence."""
    run = _dress_run(tmp_path, monkeypatch, capsys)
    assert len(run["trials"]) == 2, "premise: two turns ran"
    for arm in ("A", "B"):
        oracle.test_arm_metrics_re_derived(run, arm)
        oracle.test_t0131_sections_re_derived(run, arm)
        oracle.test_safety_counts_zero(run, arm)
        oracle.test_decision_inputs_re_derived(run, arm)
    oracle.test_configuration_findings_re_derived(run)
    assert run["report"]["configuration_findings"] == [], run["report"]["configuration_findings"]
    oracle.test_probes_re_derived(run)
    oracle.test_decision_re_derived(run)
    oracle.test_differences_re_derived(run)
    oracle.test_live_default_in_both_arms(run)
    oracle.test_arm_setting_reached_every_turn(run)
    _dress_default_is_heads(run["manifest"])


def test_main_dress_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    """B1/B3/B4 record fields from a real ``main`` (mutants: the overview offered in arm A; the guard not passed;
    ``overview_shown`` read from the arm; the bodies' image identities not the JPEG's): arm A's kit is v1 and no body
    declares the tool, its turn line has no A10 key; arm B's step 1 executes ``page_overview``, its step 2 request
    carries the overview (sha16 of the overview frame's JPEG) right before the current frame with A7's label, its turn
    line counts one call and one shown; the guard is on and logged in both; every arm setting is HEAD's."""
    run = _dress_run(tmp_path, monkeypatch, capsys)
    by = {t["arm"]: t for t in run["trials"]}
    steps = {
        a: sorted((s for s in run["steps"] if s["arm"] == a), key=lambda s: int(s["step"]))
        for a in "AB"
    }
    bodies = {b["arm"]: b for b in run["bodies"]}
    assert (by["A"]["tools_version"], by["B"]["tools_version"]) == (V1, OV_SHA), by
    assert (by["A"]["page_overview"], by["B"]["page_overview"]) == (False, True)
    assert by["A"]["page_overview_logged"] is None, by["A"]["page_overview_logged"]
    assert by["B"]["page_overview_logged"] == _counts(calls=1, shown=1), by["B"][
        "page_overview_logged"
    ]
    for arm in "AB":
        assert by[arm]["degeneracy_guard"] is True, by[arm]
        assert (by[arm]["degeneracy_guard_logged"] or {}).get("on") is True, by[arm]
        for k in oracle.DEFAULT_KEYS:
            assert by[arm][k] == run["manifest"]["live_default"][k], (arm, k)
        # review F3: what the Session itself logged on the turn line, against HEAD's default as cli gives it (never
        # the trial's copy of the row)
        d = run["manifest"]["live_default"]
        lf = d["live_feedback"]
        want_lf = None if lf is None else {**lf, "echo": d["echo"]}
        assert by[arm]["live_feedback_logged"] == want_lf, (
            arm,
            by[arm]["live_feedback_logged"],
            want_lf,
        )
        assert by[arm]["recover_thought_calls_logged"] is d["recover_thought_calls"], (arm, by[arm])
        assert by[arm]["visual_tokens_logged"] == d["visual_tokens"], (
            arm,
            by[arm].get("visual_tokens_logged"),
        )
    one = steps["B"][0]
    assert (one["call_name"], one["status"]) == ("page_overview", "executed"), one
    assert [s["overview_shown"] for s in steps["B"]] == [False, True], steps["B"]
    assert not any(s["overview_shown"] for s in steps["A"]), steps["A"]
    want_sha = hashlib.sha256(_dress_overview().jpeg).hexdigest()[:16]
    (ov,) = bodies["B"]["overviews"]
    assert ov["step"] == 1 and ov["sha"] == want_sha and abs(float(ov["screens"]) - 1.9) < 1e-9, ov
    assert ov["clipped"] is False and (ov["width"], ov["height"]) == (838, 798), ov
    (two,) = [r for r in bodies["B"]["requests"] if r["phase"] == "tool_step" and r["step"] == 2]
    parts = _final_user(two)["parts"]
    k = parts.index({"image": want_sha})
    assert parts[k + 1] == {"text": ov_label(1.9, False)}, parts
    assert "image" in parts[k + 2] and not any("image" in p for p in parts[k + 3 :]), parts
    for r in bodies["B"]["requests"]:
        assert r["tools"] is None or "page_overview" in r["tools"], r["label"]
    assert all(
        r["tools"] is None or "page_overview" not in r["tools"] for r in bodies["A"]["requests"]
    )
    assert any(r["phase"] == "generate" and r["tools"] == V1_TOOLS for r in bodies["A"]["requests"])
    assert bodies["A"]["overviews"] == []
    _dress_default_is_heads(run["manifest"])


def test_run_budget_from_process_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """B6 (``RUN_BUDGET_S`` from process start, probes included): with the process started more than
    ``RUN_BUDGET_S`` ago, ``main`` runs no turn, still writes ``report.json``, and the run is ``incomplete``
    (exit 1)."""
    rc, out, st = _dress(tmp_path, monkeypatch, capsys, past_budget=True)
    capsys.readouterr()
    report = json.loads((out / "report.json").read_text())
    assert report.get("stopped") and report["decision"]["label"] == "incomplete", report["decision"]
    assert st.browsers == [], f"a turn ran past the budget: {len(st.browsers)} browser(s)"
    assert rc == harness().EXIT_FAILED, rc


PROBE_FAILURES: dict[str, tuple[dict[str, Any], str]] = {
    "p1": ({"probes": t31._probes_with(p1_560=9999)}, "inconclusive"),
    "p2": ({"probes": t31._probes_with(p2=False)}, "void"),
    "p3": ({"probes": t31._probes_with(p3=False)}, "void"),
    "p6": ({"p6": _p6(deals={"scroll_y_after": 2.0})}, "void"),
    "p7": ({"p7": {**P7_OK, "tokens": 16_129}}, "void"),
}


@pytest.mark.parametrize("probe", list(PROBE_FAILURES))
def test_probe_failure_writes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, probe: str
) -> None:
    """B2: a failed probe runs no turn, exits 3 and still writes ``report.json``: P1 -> ``inconclusive``; P2, P3, P6
    (the page scrolled) or P7 (one token over the window) -> ``void``; the recorded ``passed`` is the oracle's."""
    over, want = PROBE_FAILURES[probe]
    rc, out, st = _dress(tmp_path, monkeypatch, capsys, **over)
    capsys.readouterr()
    report = json.loads((out / "report.json").read_text())
    probes = json.loads((out / "probes.json").read_text())
    assert report["decision"]["label"] == want, report["decision"]
    assert probes[probe]["passed"] is False and oracle.probe_verdicts(probes)[probe] is False, (
        probes[probe]
    )
    assert st.browsers == [], "a turn ran after a failed probe"
    assert rc == harness().EXIT_PROBE == 3, rc


# =====================================================================================================
# B6: the locked script (T-0131's rig, pointed at this script)
# =====================================================================================================

L_OUT = f"logs/t0140-overview-ab-{t31.L_SHORT}"
L_HARNESS = "scripts/gpu/t0140_overview_ab.py"
L_EVAL = "tests/eval/test_t0140_ab.py"
#: the merge subjects that satisfy the check (reading 5)
MERGED = {
    "task": "fedcba9876fedcba9876fedcba9876fedcba9876 Merge T-0140: page_overview tool + locked 2-arm A/B",
    "group-b": "fedcba9876fedcba9876fedcba9876fedcba9876 Merge T-0140 group B: the locked 2-arm page-overview A/B",
}
#: merges that do not: the group A merges already on main at BASE_B (verbatim subjects), T-0140b, branch merges
OTHERS = (
    "68d34d0a68d34d0a68d34d0a68d34d0a68d34d0a Merge T-0140 group A2: --page-overview (off by default) offers the "
    "overview in live sessions; placed before the current frame for one step; A8 refusals; A10 logging; "
    "byte-identical when off",
    "2dd92582dd92582dd92582dd92582dd92582dd92 Merge T-0140 group A1: page_overview tool file, judge tool_set, "
    "executor full-page capture (fit/clip at 4 screens/moved), driver full_page_screenshot; not yet offered by the "
    "session (A2)",
    "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c Merge T-0140b: adopt the overview default",
    "0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d Merge branch 'T-0140-b-ab' into T-0140-page-overview",
    "8547fc28547fc28547fc28547fc28547fc28547f Merge main into T-0140-a1-overview-capture (task statuses, T-0147 plan)",
    "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b Merge T-0135: thought-call recovery + 2-arm A/B",
)
#: the only repository paths the locked script may require to exist (its imports, its eval and the overview set)
L_PLACEHOLDERS = (
    L_HARNESS,
    L_EVAL,
    "scripts/gpu/t0135_thought_call_ab.py",
    "scripts/gpu/t0134_feedback_ab.py",
    "scripts/gpu/t0132_crop_budget_ab.py",
    "scripts/gpu/t0131_grounding_ab.py",
    "scripts/gpu/t0111g_live_smoke.py",
    "local_astra/toolcall/tools_v1_overview.json",
    "local_astra/toolcall/thought_call.py",
)


def _locked(
    monkeypatch: pytest.MonkeyPatch, merged: str = MERGED["group-b"], filler: int = 0
) -> Any:
    """T-0131's ``run_locked`` with its module constants pointed at the T-0140 script, harness, eval and merge;
    ``filler`` other merge lines after the merge (a log past the 64 KiB pipe buffer)."""
    assert LOCKED.is_file(), f"T-0140 B6: {LOCKED.relative_to(REPO)} does not exist yet"
    git_stub = t31.GIT_STUB.replace(t31.MERGED_LINE, merged)
    for other in t31.OTHER_MERGES:
        git_stub = git_stub.replace(f'echo "{other}"; ', "")
    git_stub = git_stub.replace(
        "*--merges*) ", "*--merges*) " + "".join(f'echo "{m}"; ' for m in OTHERS)
    )
    if filler:
        fill = (
            f'for i in $(seq 1 {filler}); do echo "9999999999999999999999999999999999999999 Merge T-0999: '
            'filler line $i of the merge log, long enough to pass the pipe buffer"; done'
        )
        before = f'echo "{merged}"; fi ;;'
        assert before in git_stub, "premise: T-0131's git stub prints the merge line this way"
        git_stub = git_stub.replace(before, f'echo "{merged}"; fi; {fill} ;;')
    monkeypatch.setattr(t31, "LOCKED", LOCKED)
    monkeypatch.setattr(t31, "L_HARNESS", L_HARNESS)
    monkeypatch.setattr(t31, "L_EVAL", L_EVAL)
    monkeypatch.setattr(t31, "L_OUT", L_OUT)
    monkeypatch.setattr(t31, "L_PLACEHOLDERS", L_PLACEHOLDERS)
    monkeypatch.setattr(t31, "GIT_STUB", git_stub)
    uv = t31.UV_STUB.replace("T0131_RUN_DIR", "T0131_RUN_DIR T0140_RUN_DIR")
    uv = uv.replace("probes.json; do", "probes.json bodies.jsonl; do")
    monkeypatch.setattr(t31, "UV_STUB", uv)
    return t31.run_locked


LOCKED_REFUSALS: dict[str, dict[str, object]] = {
    "an-argument": {"args": ("--n", "1")},
    "dirty-tree": {"dirty": True},
    "port-bound-stt-up": {"port_bound": True},
    "containers": {"containers": True},
    "vllm-sh-changed-since-806da12": {"vllm_drift": True},
    "out-dir-exists": {"pre_dirs": (L_OUT,)},
    "not-merged-only-group-a-merges": {"not_merged": True},
}


@pytest.mark.parametrize("case", list(LOCKED_REFUSALS))
def test_locked_script_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str) -> None:
    """B6: ``REFUSED`` and exit 2 before the wheel build, the boot or any step: an argument, a dirty tree, an Astra
    port bound (STT must be down), running containers, ``services/vllm.sh`` changed since 806da12, this SHA's out-dir
    present, and no ``Merge T-0140`` merge reachable from HEAD -- adversarial: the log holds the group A1/A2 merges
    that are on main at BASE_B (verbatim subjects), ``Merge T-0140b``, a ``Merge branch 'T-0140-...'`` and a ``Merge
    main into T-0140-...`` commit (reading 5)."""
    res = _locked(monkeypatch)(tmp_path, **LOCKED_REFUSALS[case])
    assert res.rc == 2 and "REFUSED" in res.output, (res.rc, res.output[-2000:])
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"


@pytest.mark.parametrize("merged", list(MERGED))
def test_locked_script_steps_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, merged: str
) -> None:
    """B6, the other side of every refusal (``Merge T-0140:`` or ``Merge T-0140 group B``): the harness under
    ``timeout -k 60s 75m`` with a loopback ``--base-url`` and ``--out-dir logs/t0140-overview-ab-<sha7>``, then
    ``tests/eval/test_t0140_ab.py -m gpu`` with ``T0140_RUN_DIR`` set to it; every ``uv run`` uses ``--locked``;
    exit 0."""
    from tests.unit import test_t0111g_smoke_harness as g_tests

    res = _locked(monkeypatch, MERGED[merged])(tmp_path)
    assert res.rc == 0, res.output[-3000:]
    steps = res.steps()
    runs = [c for c in steps if c.names(L_HARNESS)]
    evals = [c for c in steps if c.names(L_EVAL)]
    assert runs and evals and steps.index(runs[0]) < steps.index(evals[0]), [c.argv for c in steps]
    out = runs[0].flag("--out-dir")
    assert out is not None and out.rstrip("/").endswith(L_OUT), runs[0].argv
    assert (runs[0].flag("--base-url") or "").startswith(("http://127.0.0.1", "http://[::1]")), (
        runs[0].argv
    )
    assert "gpu" in (evals[0].flag("-m") or ""), evals[0].argv
    assert evals[0].env.get("T0140_RUN_DIR", "").rstrip("/").endswith(L_OUT), evals[0].env
    assert all("--locked" in c.argv for c in res.uv), [c.argv for c in res.uv]
    assert len(res.uv) >= 2, [c.argv for c in res.uv]
    assert g_tests._timeout_of(res.calls, L_HARNESS) == 75 * 60
    timeout_lines = [c for c in res.calls if c.startswith("timeout ") and L_HARNESS in c]
    assert timeout_lines and all(" -k 60s " in f" {c} " for c in timeout_lines), timeout_lines


def test_locked_script_long_merge_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """B6 (the T-0135 here-string safeguard; mutant ``printf '%s' "$MERGES" | grep -q`` under pipefail): a merged HEAD
    whose merge log runs past the 64 KiB pipe buffer after the ``Merge T-0140`` line (3000 filler merges, about 380
    KB) still passes the check and runs both steps."""
    res = _locked(monkeypatch, MERGED["task"], filler=3000)(tmp_path)
    assert res.rc == 0 and "REFUSED" not in res.output, (res.rc, res.output[-2000:])
    assert [c for c in res.steps() if c.names(L_EVAL)], "the eval did not run"


def test_locked_script_eval_after_a_failed_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """B6: the harness failing after it wrote ``report.json`` (a late failure: exit 1) still runs the eval over that
    report, and the script exits 1."""
    res = _locked(monkeypatch)(tmp_path, fail_match=L_HARNESS, fail_code=1)
    evals = [c for c in res.steps() if c.names(L_EVAL)]
    assert evals, f"the eval did not run after the harness failed: {res.output[-2000:]}"
    assert res.rc == 1, (res.rc, res.output[-2000:])


def test_locked_script_shape() -> None:
    """B6 by text: the T-0134 boot regime (``services/vllm.sh`` against 806da12), the ``Merge T-0140`` check through a
    here-string (never a pipe into ``grep``), the ``timeout -k 60s 75m`` bound, ``--locked`` on both ``uv run`` steps,
    the out-dir and the eval only if ``report.json`` exists; ``bash -n`` clean."""
    import subprocess

    assert LOCKED.is_file(), f"T-0140 B6: {LOCKED.relative_to(REPO)} does not exist yet"
    proc = subprocess.run(["bash", "-n", str(LOCKED)], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    text = LOCKED.read_text(encoding="utf-8")
    code = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    body = "\n".join(code)
    assert "timeout -k 60s 75m" in body, body[:600]
    checks = [ln for ln in code if "Merge T-0140" in ln and "grep" in ln]
    assert checks and all("<<<" in ln for ln in checks), checks
    assert not [
        ln
        for ln in code
        if "grep" in ln and ("MERGES" in ln or "--merges" in ln) and "|" in ln.split("grep")[0]
    ], "the merge log is piped into grep"
    runs = [ln for ln in code if "uv run" in ln]
    assert len(runs) >= 2 and all("--locked" in ln for ln in runs), runs
    assert "t0140-overview-ab-" in body and "tests/eval/test_t0140_ab.py" in body
    assert "-m gpu" in body and "report.json" in body
    assert "806da12" in body and "services/vllm.sh" in body, "the T-0134 boot regime"
