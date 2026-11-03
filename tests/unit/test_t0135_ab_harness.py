"""T-0135 group B: the 2-arm thought-call recovery A/B harness's pure parts, its ``main`` on CPU and its locked script
(unit; CPU; no engine, no browser). docs/tasks/T-0135.md B1-B7.

Group B, written active 2026-10-23 (no ``pending`` marker: the orchestrator activates group B with these rows). Rows that
need the harness or the locked script are red on the named assertion in ``harness()``/``api()``/``_script()`` ("does not
exist yet" / "has no ..."), never on an import error. Rows parametrised on ``which`` run once against the harness and
once against the eval oracle (``tests/eval/test_t0135_ab.py``); the oracle rows need no harness and are green now, so a
wrong oracle is caught on CPU.

**The assumed harness surface** (``scripts/gpu/t0135_thought_call_ab.py``; the task names the behaviour, these names and
shapes are the test-author's, stated once so the implementer can implement to them or say why not):

Imported, never edited (B1)
    T-0111g's ``TASKS``, ``LoopMeter``, ``sentinel_hits``, ``url_leaks``, ``log_complete``, ``non_loopback_sockets``,
    ``turn_records``; T-0131's ``probe_p1``, ``run_probes`` (P1-P3), ``false_claim``, ``config_evidence``; T-0132's
    ``wilson``, ``newcombe``; T-0134's ``score_v2``, ``over_click``, ``no_see``, ``coord_mismatch``. Module attributes
    whose objects come from those files (a ``def``/``class``/literal of those names is refused). ``G``, ``T``, ``P``
    are T-0111g's, T-0131's and T-0132's harness modules (the CPU dress stands in an engine through ``G``).

Seams ``main`` reads at call time (as T-0134's)
    ``_git`` (the tree), ``order``, ``run_probes``, ``PROCESS_START_S``; ``cli.grounding_of`` and (O-16)
    ``cli.feedback_of`` (through ``live_default``, read once per ``main``).

The arms (B1; never hard-coded: by the human's 2026-10-23 ruling the live default becomes T-0134's arm B, model-unit
echo plus the screen-now label, once T-0134c lands)
    ``live_default(profile) -> {"visual_tokens", "reground", "live_feedback": None | {"screen_label",
    "page_extent"}, "echo"}``: HEAD's live default as ``cli`` gives it (budget and ReGround from
    ``cli.grounding_of(profile, Namespace(), "live")``; O-16: the feedback and the echo from
    ``cli.feedback_of(profile, Namespace(), "live")``, its ``live_feedback`` choice as ``FEEDBACK_DICT`` maps it and
    its ``action_echo``). ``arm_rows(default) -> [row_A, row_B]``:
    ``{"label", <the four settings>, "recover_thought_calls"}``, A ``False``, B ``True`` (the oracle's ``arm_rows``,
    the same shape); ``arm_table_sha(rows)`` = sha256 of ``json.dumps(rows, sort_keys=True, separators=(",",
    ":"))``. ``main`` reads the default once, passes every setting explicitly in both arms (``Session(...,
    visual_tokens=, reground=, live_feedback=, recover_thought_calls=)``, ``BrowserToolExecutor(echo=)``) and records
    ``live_default``, ``live_default_feedback`` (O-16: ``dataclasses.asdict`` of ``feedback_of``'s result, sources
    included), ``regime.arms`` and ``arm_table_sha256`` in the manifest.

Constants
    ``EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_PROBE = 0, 1, 2, 3``; ``REPEATS = 20``; ``ORDER_SEED = 0``;
    ``RUN_BUDGET_S = 50 * 60``; ``PROCESS_START_S`` (``time.monotonic()`` at import).

Pure functions (each also in the eval oracle, same name and shape unless noted)
    ``order(seed) -> list[(arm, task, repeat)]`` (200, shuffled jointly; the oracle's construction);
    ``call_in_thought(commits) -> str | None`` (A1/A2 with O-4 over one request's ``llm.commit`` data: the recovered
    text, or ``None``); ``configuration_findings(trials, steps, thoughts, arms) -> list[str]`` (B4; ``arms`` the
    two rows; each finding names the trial as ``"<arm> <task>#<repeat>"`` or the
    setting); ``decide(inputs, *, p1_passed, void=False, stopped=None, configuration_findings=())`` and
    ``decide_with_reason(...) -> (label, reason)``, ``inputs`` keyed ``"A"``/``"B"``, each ``{"accepted", "safety",
    "engine_up", "success", "s3", "false_claims", "over_clicks", "first_failing"}``; labels ``"recover"``, ``"no
    change"``, ``"inconclusive"``, ``"void"``, ``"incomplete"``; reasons ``"P1 failed"``, ``"configuration:
    <first finding>"``, ``"arm <X> ineligible: <clause> (first failing trial <task>#<repeat>)"``.
    ``arm_metrics(trials, steps, thoughts)`` (the oracle's ``derive_arm``) -> ``{"success", "success_ci" (Wilson,
    ``None`` at n = 0), "per_task", "s3", "false_claims" (T-0131's proxy with ``score_v2``), "over_clicks",
    "call_in_thought", "silent_call_in_thought", "silent_final", "recoveries": {"total", "by_verdict" (``live.
    classify`` status of the recovered text -> count), "by_status" (step status, ``"none"`` for ``None`` -> count),
    "turns_2plus"}}``. ``build_report(trials, steps, regrounds, thoughts, probes, *, arms, stopped=None)`` -> report.json
    (shape in the eval's docstring; ``arms[X]`` holds T-0131's section with T-0134's two renames plus
    ``arm_metrics``; ``differences`` keys ``"success B-A"``, ``"s3 B-A"``, ``"silent_call_in_thought B-A"``,
    ``"silent_final B-A"``).

Records ``main`` writes (the eval's docstring has the full list)
    trials: T-0131/T-0134 fields plus ``arm``, ``recover_thought_calls``, ``recover_thought_calls_logged`` (the A8
    turn-line key); steps: T-0131's fields plus ``arm`` and ``from_thought`` (bool); ``thoughts.jsonl``: ``{"arm",
    "task", "repeat", "attempts", "requests", "body_texts"}``; ``manifest.json``: ``arm_table_sha256``,
    ``task_table_sha256``, ``order``, ``live_default`` and ``regime.arms``.

Refusals (exit 2, before any socket, leaving no out-dir)
    T-0132's/T-0134's (dirty tree, non-loopback URL, out-dir present, tool kit, Chromium, ...), and group A2 absent
    (``Session`` without the ``recover_thought_calls`` keyword, as T-0134 refuses without its group A keywords), and
    (O-16) a ``cli`` without ``feedback_of``.

Readings (reported; ambiguities in the task)
    1. *Silent endings* are ``silent_call_in_thought`` (the turn's **last** request matches the rule and has no
       ``from_thought`` step) and ``silent_final`` (``tool_loop_end == "final"`` and nothing voiced);
       ``call_in_thought`` (any matching request) is reported beside them. Newcombe B-A on both silent counts.
    2. B4 "exactly one from_thought call record" counts **calls** (steps, one call id each): a recovered call has
       several ``tool.call`` records (proposed, executed), every one carrying the key (A6).
    3. B4's leak finding: a stripped line of at least 20 characters of the thought block **before** the recovered
       call, not already in the recovered request's own body, found in a later body of the turn. Without the first
       exclusion every recovery would void the run (``render_call`` of the recovered call is byte-identical to the
       model's text: checked 2026-10-23); without the second, a thought restating the user's request would. O-17
       adds a third: a line in an answer-channel commit of the same turn (each request's answer commits joined) is
       skipped, since spoken words reach later bodies as ``step_text``.
    4. B5's "every safety count is 0" clause is subsumed by the ineligibility step (T-0132 C6b's ``safety`` clause):
       one safety count gives ``inconclusive`` naming ``safety``, never ``no change``.
    5. "The table is pinned by an arm-table sha": the default is read at run time, so no literal can pin the run's
       table. The sha is pinned for hand-chosen defaults (``ARM_PINS``), recorded by ``main``, and re-derived by the
       eval from the recorded ``live_default``. The rows compare ``live_default`` with ``cli.grounding_of`` and
       ``cli.feedback_of`` at HEAD (O-16), never with a literal.

Regime: CPU, pure, hand-built records; the dress rows run ``main`` on CPU with T-0131's engine rig and the
``FakeLiveBrowser`` (fake LLM, two S2 turns); no engine, no browser, no measurement.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import functools
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

from tests.eval import test_t0135_ab as oracle
from tests.unit import test_t0131_ab_harness as t31

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "scripts" / "gpu" / "t0135_thought_call_ab.py"
T0111G_HARNESS = REPO / "scripts" / "gpu" / "t0111g_live_smoke.py"
T0131_HARNESS = REPO / "scripts" / "gpu" / "t0131_grounding_ab.py"
T0132_HARNESS = REPO / "scripts" / "gpu" / "t0132_crop_budget_ab.py"
T0134_HARNESS = REPO / "scripts" / "gpu" / "t0134_feedback_ab.py"
LOCKED = REPO / "scripts" / "locked" / "t0135_thought_call_ab.sh"

IMPLS = [pytest.param("harness"), pytest.param("eval-oracle")]

#: two hand-chosen live defaults (inputs, not HEAD's): the settings before T-0134c, and T-0134's arm B as the human
#: ruled it the next default (model-unit echo plus the screen-now label, no extent)
D_PIXELS = {"visual_tokens": 1120, "reground": False, "live_feedback": None, "echo": "pixels"}
D_LABEL = {
    "visual_tokens": 1120,
    "reground": False,
    "live_feedback": {"screen_label": True, "page_extent": False},
    "echo": "model",
}
#: ``arm_table_sha`` of each default's two rows, by the test-author 2026-10-23 (hashlib over the canonical JSON)
ARM_PINS = {
    "pixels": (D_PIXELS, "8e6578786fbc2b851556537137af316b9c3a69fb613ebf6c202a8df07e3f41eb"),
    "label": (D_LABEL, "0e5a9cf6e2420822f0f2fc11481d36b08be12b2c1d8a5a483e762b4430aff8f9"),
}
#: the arms the hand-built records ran at
HAND_ARMS = [
    {"label": "A", **D_PIXELS, "recover_thought_calls": False},
    {"label": "B", **D_PIXELS, "recover_thought_calls": True},
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
    T0131_HARNESS: ("probe_p1", "run_probes", "false_claim", "config_evidence"),
    T0132_HARNESS: ("wilson", "newcombe"),
    T0134_HARNESS: ("score_v2", "over_click", "no_see", "coord_mismatch"),
}
DATA = ("TASKS",)
Q = '<|"|>'
#: the D-5 call of ``sessions/20261022-204803-1975518.jsonl`` (T-0135 Goal), verbatim; ``live.classify``: valid
TYPE_EMAIL = f"<|tool_call>call:type_at{{press_enter:true,text:{Q}ada@example.com{Q},x:305,y:448}}<tool_call|>"
#: valid
TYPE_NAME = (
    f"<|tool_call>call:type_at{{press_enter:false,text:{Q}Ada Lovelace{Q},x:305,y:388}}<tool_call|>"
)
#: valid
CLICK = "<|tool_call>call:click_at{x:500,y:250}<tool_call|>"
#: the doubled close of ``...-210810-...`` (T-0135 census): recovered, then ``repairable`` ("text after call")
DOUBLED = TYPE_EMAIL + "<tool_call|>"


@functools.cache
def harness() -> ModuleType:
    assert HARNESS.is_file(), f"T-0135 B1: {HARNESS.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0135_thought_call_ab_unit", HARNESS)
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
    assert hasattr(h, name), f"T-0135: {HARNESS.name} has no {name!r} (see this module's docstring)"
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
    objects, none is defined in the T-0135 harness; ``G``/``T``/``P`` are T-0111g's, T-0131's and T-0132's
    harness modules."""
    with t31.no_sockets() as attempts:
        harness.cache_clear()
        h = harness()
    assert attempts == [], attempts
    assert callable(getattr(h, "main", None)), "the harness has no main(argv)"
    _, defined = t31._top_level(ast.parse(HARNESS.read_text(encoding="utf-8")))
    names = {n for ns in FROM.values() for n in ns} | set(DATA)
    assert sorted(defined & names) == [], (
        f"T-0135 B1: defined, not imported: {sorted(defined & names)}"
    )
    for origin, ns in FROM.items():
        for name in ns:
            src = Path(inspect.getsourcefile(api(name)) or "").resolve()
            assert src == origin.resolve(), f"T-0135 B1: {name} comes from {src}, not {origin.name}"
    assert [t.task_id for t in api("TASKS")] == list(TASK_IDS)
    for attr, origin in (("G", T0111G_HARNESS), ("T", T0131_HARNESS), ("P", T0132_HARNESS)):
        mod = api(attr)
        assert isinstance(mod, ModuleType), (attr, mod)
        assert Path(mod.__file__ or "").resolve() == origin.resolve(), (attr, mod.__file__)


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("pin", list(ARM_PINS))
def test_arm_rows_and_sha(which: str, pin: str) -> None:
    """B1: over a live default, the two rows are that default's four settings in both arms, A without and B with
    recovery (written out by hand here), and ``arm_table_sha`` of them is the pin; swapping the rows, or flipping
    B's recovery, gives another sha."""
    default, want_sha = ARM_PINS[pin]
    rows = impl(which, "arm_rows")(dict(default))
    want = [
        {"label": "A", **default, "recover_thought_calls": False},
        {"label": "B", **default, "recover_thought_calls": True},
    ]
    assert [dict(r) for r in rows] == want, rows
    sha = impl(which, "arm_table_sha")
    assert sha(rows) == want_sha
    assert sha([rows[1], rows[0]]) != want_sha
    assert sha([rows[0], {**rows[1], "recover_thought_calls": False}]) != want_sha


def test_constants() -> None:
    """B1/B6: the constants."""
    assert api("REPEATS") == 20 and api("ORDER_SEED") == 0
    assert api("RUN_BUDGET_S") == 50 * 60, "T-0135 B6: RUN_BUDGET_S = 50 * 60"
    assert isinstance(api("PROCESS_START_S"), float), "B6: the budget runs from process start"
    assert api("PROCESS_START_S") <= time.monotonic()
    assert (api("EXIT_OK"), api("EXIT_FAILED"), api("EXIT_REFUSED"), api("EXIT_PROBE")) == (
        0,
        1,
        2,
        3,
    )


#: O-16: ``cli.Feedback.live_feedback``'s choices as ``live_default`` records them (written out by hand; R-m:
#: ``off`` is a ``LiveFeedback`` with both parts off, never ``None``)
FEEDBACK_DICT = {
    "off": {"screen_label": False, "page_extent": False},
    "label": {"screen_label": True, "page_extent": False},
    "label+extent": {"screen_label": True, "page_extent": True},
}


def test_live_default_follows_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """B1 (O-16): ``live_default`` is read from ``cli`` at call time -- its budget and ReGround are
    ``cli.grounding_of``'s and its feedback and echo ``cli.feedback_of``'s, for live with no flags -- and follows
    both when they change (560 with ReGround on; ``label+extent`` with pixel echo; ``off``). The expected values are
    ``cli``'s own at HEAD, never a literal, so the row survives the next default change."""
    from local_astra import cli
    from local_astra.config import load

    assert callable(getattr(cli, "feedback_of", None)), "premise: T-0134c's cli.feedback_of"
    lean = load().profiles["lean"]
    fn = api("live_default")
    g = cli.grounding_of(lean, argparse.Namespace(), "live")
    f = cli.feedback_of(lean, argparse.Namespace(), "live")
    got = fn(lean)
    assert (got["visual_tokens"], got["reground"]) == (g.visual_tokens, g.reground), (got, g)
    assert (got["live_feedback"], got["echo"]) == (FEEDBACK_DICT[f.live_feedback], f.action_echo), (
        got,
        f,
    )
    assert oracle.live_default_findings(got) == [], (got, oracle.live_default_findings(got))
    monkeypatch.setattr(cli, "grounding_of", _grounding_with(cli, visual_tokens=560, reground=True))
    for choice, echo in (("label+extent", "pixels"), ("off", "model")):
        fake = cli.Feedback(choice, "flag", echo, "flag")  # type: ignore[arg-type]
        monkeypatch.setattr(cli, "feedback_of", lambda profile, args, mode, _f=fake: _f)
        moved = fn(lean)
        assert (moved["visual_tokens"], moved["reground"]) == (560, True), moved
        assert (moved["live_feedback"], moved["echo"]) == (FEEDBACK_DICT[choice], echo), moved


def _grounding_with(cli: Any, **over: Any) -> Any:
    real = cli.grounding_of

    def fake(profile: Any, args: Any, mode: str) -> Any:
        return dataclasses.replace(real(profile, args, mode), **over)

    return fake


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
    assert {t[0] for t in turns[:20]} == {"A", "B"}


def test_harness_order_equals_oracle() -> None:
    """B1/B7: the harness's seed-0 order is the oracle's (the eval checks the run against it)."""
    assert [tuple(t) for t in api("order")(0)] == oracle.order(0)


# =====================================================================================================
# B1: refusals before any socket
# =====================================================================================================


def _rig(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> Any:
    """Every socket refused and recorded; a clean tree through the harness's ``_git``; a stand-in live kit at
    T-0115's sha with repair on; Chromium reported present."""
    from local_astra import cli

    h = harness()
    sockets: list[str] = []

    def refuse(what: str) -> Any:
        def fn(*a: Any, **k: Any) -> Any:
            sockets.append(what)
            raise ConnectionRefusedError(f"T-0135 refusal rig: no {what}")

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
    kit = types.SimpleNamespace(
        repair_off=None, tools_version=t31.TOOLS_SHA12 + "0" * 52, mode="live"
    )
    monkeypatch.setattr(cli, "load_tool_kit", lambda profile, **kw: (kit, None))
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    return types.SimpleNamespace(
        h=h,
        cli=cli,
        kit=kit,
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
    "session-without-recover-keyword",
    "cli-without-feedback-of",
)


@pytest.mark.parametrize("case", REFUSALS)
def test_refusals_before_any_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, case: str
) -> None:
    """B1: T-0132's/T-0134's refusals stand (dirty tree, non-loopback URL, an existing out-dir, the tool repair
    off), a ``Session`` without group A2's ``recover_thought_calls`` keyword is refused as T-0134 refuses without
    its group A keywords, and (O-16) so is a ``cli`` without ``feedback_of`` -- each with exit 2 before any socket,
    leaving the out-dir as it was."""
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
        rig.kit.repair_off = "payload stale"
    elif case == "cli-without-feedback-of":
        # O-16: live_default reads cli.feedback_of; a cli without it cannot say what the arms run at
        rig.mp.delattr(rig.cli, "feedback_of")
    else:

        class Old(session_mod.Session):  # type: ignore[misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)

        rig.mp.setattr(session_mod, "Session", Old)
    before = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    rc = h.main(argv)
    err = rig.capsys.readouterr().err
    assert rc == h.EXIT_REFUSED == 2, f"{case}: exit {rc}\n{err}"
    assert rig.sockets == [], rig.sockets
    after = sorted(p.name for p in rig.out.iterdir()) if rig.out.exists() else None
    assert after == before, f"{case}: the refusal touched the out-dir ({before} -> {after})"
    assert err.strip(), f"{case}: the refusal says nothing"
    if case == "cli-without-feedback-of":
        assert "feedback_of" in err, f"O-16: the refusal does not name cli.feedback_of: {err!r}"


def test_rig_at_passing_values_goes_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """The other side of every refusal row (a guard too wide): at HEAD's live default, a clean tree and the pinned
    table, ``main`` does not refuse; it looks for the engine (a socket attempt) and, finding none, exits 1."""
    rig = _rig(tmp_path, monkeypatch, capsys)
    h = rig.h
    rc = h.main(["--base-url", t31.VALID_URL, "--out-dir", str(rig.out)])
    err = rig.capsys.readouterr().err
    assert rc == h.EXIT_FAILED == 1, f"exit {rc}\n{err[-2000:]}"
    assert rig.sockets, "main never tried the engine"


def test_probes_p1_to_p3_are_t0131s() -> None:
    """B2: P1-P3 are T-0131's own objects (P4 is not re-run: the feature adds no request content)."""
    for name in ("probe_p1", "run_probes"):
        src = Path(inspect.getsourcefile(api(name)) or "").resolve()
        assert src == T0131_HARNESS.resolve(), (name, src)


# =====================================================================================================
# A1/A2's rule (O-4): call_in_thought
# =====================================================================================================


def _c(segment: str, text: str, thought_open: bool | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"segment": segment, "text": text}
    if thought_open is not None:
        out["thought_open"] = thought_open
    return out


#: (commits of one request, recovered text or None), by hand from A1, A2, A5(c) and O-4
RULE: dict[str, tuple[list[dict[str, Any]], str | None]] = {
    "d5-tail": (
        [_c("thought", "Now the email field.\n" + TYPE_EMAIL + "\n", True)],
        TYPE_EMAIL + "\n",
    ),
    "split-across-open-commits": (
        [
            _c("answer", "Filling the form. ", False),
            _c("thought", "Now the email.\n<|tool_call>call:type_at{press_enter:true,", True),
            _c("thought", f"text:{Q}ada@example.com{Q},x:305,y:448}}<tool_call|>  ", True),
        ],
        TYPE_EMAIL + "  ",
    ),
    "two-tags-last-at-tail": (
        [_c("thought", f"Not {CLICK} as I first thought; instead {TYPE_EMAIL}", True)],
        TYPE_EMAIL,
    ),
    "doubled-close": ([_c("thought", "Email.\n" + DOUBLED, True)], DOUBLED),
    "closed-thought": ([_c("thought", "Email.\n" + TYPE_EMAIL, False)], None),
    "tail-prose": (
        [_c("thought", "Email.\n" + TYPE_EMAIL + " Actually, I should scroll.", True)],
        None,
    ),
    "answer-channel-call": (
        [_c("thought", "Email.", False), _c("answer", TYPE_EMAIL, False)],
        None,
    ),
    "tool-commit-kept": ([_c("thought", "Email.\n" + TYPE_EMAIL, True), _c("tool", CLICK)], None),
    "last-tag-unclosed": (
        [_c("thought", CLICK + " and then <|tool_call>call:click_at{x:1,", True)],
        None,
    ),
    "no-tag": ([_c("thought", "the the the the the", True)], None),
    "empty": ([], None),
    # O-4: a closed thought is never stitched to a later open one (T-0134's ``silent_call_in_thought`` would)
    "closed-then-open-stitch": (
        [
            _c("thought", "<|tool_call>call:click_at{x:500,", False),
            _c("thought", "y:250}<tool_call|>", True),
        ],
        None,
    ),
    # the guard-too-wide twin: a closed thought before the open block does not stop the open block's tail
    "closed-then-open-tail": (
        [
            _c("thought", "I looked at the page.", False),
            _c("thought", "Email.\n" + TYPE_EMAIL, True),
        ],
        TYPE_EMAIL,
    ),
    # A5(c)/Implementation notes: check the segment, not only the flag (an answer commit of a canvas ending inside a
    # thought carries ``thought_open: true``)
    "answer-with-open-flag": ([_c("answer", TYPE_EMAIL, True)], None),
    "open-thought-then-answer": (
        [_c("thought", "Email.\n" + TYPE_EMAIL, True), _c("answer", "", True)],
        None,
    ),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(RULE))
def test_call_in_thought_rule(which: str, row: str) -> None:
    """A1/A2 (O-4) as B3/B4 apply it (reviewer mutations 1, 2, 3): the recovered text exactly, from the **last**
    tag; ``None`` for a closed thought, prose after the close, an answer-channel call, a kept tool commit, an
    unclosed last tag, no tag, no commits, a closed thought stitched to an open one, an answer commit with the open
    flag, or an answer after the open thought."""
    commits, want = RULE[row]
    assert impl(which, "call_in_thought")(commits) == want, row


# =====================================================================================================
# B4: configuration findings
# =====================================================================================================

SENT_LINE = "Now the email field."  # 20 characters: the boundary of B4's leak rule
SHORT_LINE = "Type it in the box."  # 19 characters
QUOTED = "Fill in the contact form for Ada."  # in every body of the turn (the user's request): already context
CONTACT_SENT = dict(oracle.g0111.CONTACT_SENT)


def _att(arm: str, task: str, repeat: int, k: int) -> str:
    return f"{arm}{task}{repeat}-{k}"


def _hc_trial(
    arm: str, task: str, repeat: int, state: Any, voiced: list[str], end: str, **kw: Any
) -> dict[str, Any]:
    base = {
        "voiced": list(voiced),
        "arm": arm,
        "visual_tokens": 1120,
        "reground": False,
        "reground_visual_tokens": None,
        "live_feedback": None,
        "echo": "pixels",
        "scanned_requests": [f"generate:{_att(arm, task, repeat, 1)}"],
        "tool_loop_end": end,
        "recover_thought_calls": arm == "B",
        "recover_thought_calls_logged": arm == "B",
    }
    return t31._trial(task, repeat, state, **(base | kw))


def _hc_step(
    arm: str,
    task: str,
    repeat: int,
    step: int,
    status: str | None,
    name: str | None = None,
    *,
    voiced: tuple[str, ...] = (),
    from_thought: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    return t31._step(
        task,
        repeat,
        step,
        status,
        200.0,
        400.0,
        arm=arm,
        visual_tokens=1120,
        reground=False,
        reground_visual_tokens=None,
        call_name=name,
        executed_name=name if status == "executed" else None,
        voiced_sources=list(voiced),
        attempt_id=_att(arm, task, repeat, step),
        call_id=(f"{arm}{task}{repeat}-call-s{step}" if name else None),
        from_thought=from_thought,
        reason=reason,
    )


def _thoughts(
    arm: str, task: str, repeat: int, requests: list[list[dict[str, Any]]], bodies: list[str]
) -> dict[str, Any]:
    return {
        "arm": arm,
        "task": task,
        "repeat": repeat,
        "attempts": [_att(arm, task, repeat, k) for k in range(1, len(requests) + 1)],
        "requests": requests,
        "body_texts": bodies,
    }


def _body(*extra: str) -> str:
    return "\n".join(("You are Astra.", QUOTED, *extra))


def _hand_records(
    arm: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Four turns per arm, every count written out in ``HAND``'s comment.

    Arm A (recovery off): A1 S3#0 the D-5 shape at step 2, not recovered (voiced "Filling the form." at step 1);
    A2 S1#0 a closed thought and nothing else (silent final); A3 S2#0 two blue mugs, "I have added the blue mug."
    (over-click and false claim); A4 S3#1 sent (success).

    Arm B (recovery on): B1 S3#0 the D-5 shape at step 2, recovered and executed, then "Sent." (success); B2 S3#1
    two recoveries, step 1 executed, step 2 the doubled close refused with ``THOUGHT_CALL_NO_REPAIR`` (loop end
    ``refused``); B3 S1#0 a closed thought quoting a call, then the answer-channel click (success, no recovery); B4
    S5#0 an open thought with no tag ("the the ..."), silent final."""
    from local_astra.conductor.session import THOUGHT_CALL_NO_REPAIR

    contact_open = {"page": "contact", "sent": None}
    contact_sent = {"page": "contact", "sent": CONTACT_SENT}
    if arm == "A":
        trials = [
            _hc_trial("A", "S3", 0, contact_open, ["Filling the form."], "final"),
            _hc_trial("A", "S1", 0, {"page": "search", "query": "cups"}, [], "final"),
            _hc_trial(
                "A",
                "S2",
                0,
                {"page": "shop", "cart": ["blue mug", "blue mug"]},
                ["I have added the blue mug."],
                "final",
            ),
            _hc_trial("A", "S3", 1, contact_sent, ["Sent."], "final"),
        ]
        steps = [
            _hc_step("A", "S3", 0, 1, "executed", "type_at", voiced=("Filling the form.",)),
            _hc_step("A", "S3", 0, 2, None),
            _hc_step("A", "S1", 0, 1, None),
            _hc_step("A", "S2", 0, 1, "executed", "click_at"),
            _hc_step("A", "S2", 0, 2, "executed", "click_at"),
            _hc_step("A", "S2", 0, 3, None, voiced=("I have added the blue mug.",)),
            _hc_step("A", "S3", 1, 1, "executed", "type_at"),
            _hc_step("A", "S3", 1, 2, None, voiced=("Sent.",)),
        ]
        thoughts = [
            _thoughts(
                "A",
                "S3",
                0,
                [
                    [_c("answer", "Filling the form. ", False), _c("tool", TYPE_NAME)],
                    [_c("thought", f"{SENT_LINE}\n{TYPE_EMAIL}\n", True)],
                ],
                [_body(), _body(TYPE_NAME)],
            ),
            _thoughts(
                "A", "S1", 0, [[_c("thought", "I should search for cups first.", False)]], [_body()]
            ),
            _thoughts(
                "A",
                "S2",
                0,
                [
                    [_c("tool", CLICK)],
                    [_c("tool", CLICK)],
                    [_c("answer", "I have added the blue mug.", False)],
                ],
                [_body(), _body(CLICK), _body(CLICK, CLICK)],
            ),
            _thoughts(
                "A",
                "S3",
                1,
                [[_c("tool", TYPE_NAME)], [_c("answer", "Sent.", False)]],
                [_body(), _body(TYPE_NAME)],
            ),
        ]
        return trials, steps, thoughts
    trials = [
        _hc_trial("B", "S3", 0, contact_sent, ["Filling the form.", "Sent."], "final"),
        _hc_trial("B", "S3", 1, contact_open, [], "refused"),
        _hc_trial("B", "S1", 0, {"page": "shop", "cart": ["blue mug"]}, ["Added."], "final"),
        _hc_trial("B", "S5", 0, {"page": "deals", "cart": []}, [], "final"),
    ]
    steps = [
        _hc_step("B", "S3", 0, 1, "executed", "type_at", voiced=("Filling the form.",)),
        _hc_step("B", "S3", 0, 2, "executed", "type_at", from_thought=True),
        _hc_step("B", "S3", 0, 3, None, voiced=("Sent.",)),
        _hc_step("B", "S3", 1, 1, "executed", "type_at", from_thought=True),
        _hc_step(
            "B",
            "S3",
            1,
            2,
            "refused",
            "type_at",
            from_thought=True,
            reason=THOUGHT_CALL_NO_REPAIR,
        ),
        _hc_step("B", "S1", 0, 1, "executed", "click_at"),
        _hc_step("B", "S1", 0, 2, None, voiced=("Added.",)),
        _hc_step("B", "S5", 0, 1, None),
    ]
    thoughts = [
        _thoughts(
            "B",
            "S3",
            0,
            [
                [_c("answer", "Filling the form. ", False), _c("tool", TYPE_NAME)],
                # a 20-character line, a 19-character line and a line the turn's context already holds
                [_c("thought", f"{SENT_LINE}\n{SHORT_LINE}\n{QUOTED}\n{TYPE_EMAIL}\n", True)],
                [_c("answer", "Sent.", False)],
            ],
            # the step-3 body embeds the recovered call as render_call writes it (byte-identical to the model's)
            [_body(), _body(TYPE_NAME), _body(TYPE_NAME, TYPE_EMAIL)],
        ),
        _thoughts(
            "B",
            "S3",
            1,
            [
                [_c("thought", f"Start with the name.\n{TYPE_NAME}", True)],
                [_c("thought", f"Then the email.\n{DOUBLED}", True)],
            ],
            [_body(), _body(TYPE_NAME)],
        ),
        _thoughts(
            "B",
            "S1",
            0,
            [
                [
                    _c("thought", f"Earlier I wrote {CLICK} in my head.", False),
                    _c("tool", CLICK),
                ],
                [_c("answer", "Added.", False)],
            ],
            [_body(), _body(CLICK)],
        ),
        _thoughts("B", "S5", 0, [[_c("thought", "the the the the the", True)]], [_body()]),
    ]
    return trials, steps, thoughts


def _clean() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ta, sa, ha = _hand_records("A")
    tb, sb, hb = _hand_records("B")
    return ta + tb, sa + sb, ha + hb


def _find_step(
    steps: list[dict[str, Any]], arm: str, task: str, repeat: int, k: int
) -> dict[str, Any]:
    (s,) = [
        x for x in steps if (x["arm"], x["task"], x["repeat"], x["step"]) == (arm, task, repeat, k)
    ]
    return s


def _find_thoughts(
    thoughts: list[dict[str, Any]], arm: str, task: str, repeat: int
) -> dict[str, Any]:
    (r,) = [x for x in thoughts if (x["arm"], x["task"], x["repeat"]) == (arm, task, repeat)]
    return r


def _fault(name: str) -> tuple[Any, ...]:
    """One B4 fault applied to the clean records; returns (trials, steps, thoughts, arms, the name a finding must
    carry)."""
    trials, steps, thoughts = _clean()
    arms: list[dict[str, Any]] = [dict(r) for r in HAND_ARMS]
    if name == "arm-A-from-thought":
        _find_step(steps, "A", "S3", 0, 1)["from_thought"] = True
        return trials, steps, thoughts, arms, "A S3#0"
    if name == "arm-A-from-thought-on-a-matching-request":
        # even where the rule matches (A1 step 2), arm A runs with recovery off
        s = _find_step(steps, "A", "S3", 0, 2)
        s.update(from_thought=True, status="executed", call_name="type_at", executed_name="type_at")
        return trials, steps, thoughts, arms, "A S3#0"
    if name == "arm-B-match-not-recovered":
        _find_step(steps, "B", "S3", 0, 2)["from_thought"] = False
        return trials, steps, thoughts, arms, "B S3#0"
    if name == "arm-B-match-two-recoveries":
        extra = dict(_find_step(steps, "B", "S3", 0, 2)) | {"step": 9, "call_id": "B-extra"}
        steps.append(extra)
        return trials, steps, thoughts, arms, "B S3#0"
    if name == "from-thought-off-the-rule":
        _find_step(steps, "B", "S1", 0, 1)["from_thought"] = True
        return trials, steps, thoughts, arms, "B S1#0"
    if name == "from-thought-on-an-unlogged-request":
        _find_step(steps, "B", "S3", 0, 3)["from_thought"] = True
        _find_step(steps, "B", "S3", 0, 3)["attempt_id"] = "not-in-thoughts"
        return trials, steps, thoughts, arms, "B S3#0"
    if name == "thought-line-leak":
        r = _find_thoughts(thoughts, "B", "S3", 0)
        r["body_texts"][2] += "\n" + SENT_LINE
        return trials, steps, thoughts, arms, "B S3#0"
    if name == "thought-line-leak-inside-a-longer-line":
        r = _find_thoughts(thoughts, "B", "S3", 1)
        r["body_texts"][1] += "\nThe assistant said: Start with the name. Then it typed."
        return trials, steps, thoughts, arms, "B S3#1"
    if name == "arms-differ-in-budget":
        arms[1]["visual_tokens"] = 560
        return trials, steps, thoughts, arms, "visual_tokens"
    if name == "arms-differ-in-echo":
        arms[1]["echo"] = "model"
        return trials, steps, thoughts, arms, "echo"
    raise KeyError(name)


FAULTS = (
    "arm-A-from-thought",
    "arm-A-from-thought-on-a-matching-request",
    "arm-B-match-not-recovered",
    "arm-B-match-two-recoveries",
    "from-thought-off-the-rule",
    "from-thought-on-an-unlogged-request",
    "thought-line-leak",
    "thought-line-leak-inside-a-longer-line",
    "arms-differ-in-budget",
    "arms-differ-in-echo",
)


def _cf(which: str) -> Any:
    fn = impl(which, "configuration_findings")
    return lambda trials, steps, thoughts, arms: fn(trials, steps, thoughts, arms)


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("fault", FAULTS)
def test_configuration_findings(which: str, fault: str) -> None:
    """B4, one hand trial per finding (reviewer mutation 9): a clean run has none; each fault gives at least one
    finding naming the trial (or the setting): a ``from_thought`` step in arm A (also where the rule matches); an
    arm-B request the rule matches with zero or two ``from_thought`` calls; a ``from_thought`` step on a request the
    rule does not match, or on a request the thought line does not hold; a 20-character thought line (alone, or
    inside a longer line) in a later body; arm rows differing in a setting other than ``recover_thought_calls``."""
    cf = _cf(which)
    trials, steps, thoughts = _clean()
    assert cf(trials, steps, thoughts, HAND_ARMS) == [], "premise: the clean run has no finding"
    trials, steps, thoughts, arms, named = _fault(fault)
    found = cf(trials, steps, thoughts, arms)
    assert found and any(named in f for f in found), (fault, found)


@pytest.mark.parametrize("which", IMPLS)
def test_configuration_findings_not_too_wide(which: str) -> None:
    """B4 companions (reading 3): in the clean records a recovered call reaches the next body as its rendered text,
    a 19-character thought line and a line the turn's context already held reach later bodies too, and arm A's
    unrecovered D-5 request sits beside a recovered one in B -- none of it is a finding. Moving the 19-character
    line to 20 characters makes one (the boundary)."""
    cf = _cf(which)
    trials, steps, thoughts = _clean()
    r = _find_thoughts(thoughts, "B", "S3", 0)
    r["body_texts"][2] += f"\n{SHORT_LINE}"
    assert cf(trials, steps, thoughts, HAND_ARMS) == []
    r["body_texts"][2] += f"\n{SHORT_LINE}!"
    r["requests"][1][0]["text"] = r["requests"][1][0]["text"].replace(SHORT_LINE, SHORT_LINE + "!")
    found = cf(trials, steps, thoughts, HAND_ARMS)
    assert found and all("B S3#0" in f for f in found), found


#: 34 characters, spoken by the recovered request (split across two answer commits) and thought again before its call
SPOKEN_LINE = "I will now type the email address."


@pytest.mark.parametrize("which", IMPLS)
def test_spoken_thought_line_is_not_a_leak(which: str) -> None:
    """O-17 (mutant: the leak rule without the answer-channel exclusion): B1's recovered request speaks a line --
    split across two answer commits, so the match is over the request's joined answer channel -- then thinks it
    again before the tail call, and the line reaches the step-3 body as ``step_text``: no finding. The same thought
    with the spoken commits removed is a leak (as ``thought-line-leak``, which stays): a finding naming B S3#0."""
    cf = _cf(which)
    trials, steps, thoughts = _clean()
    r = _find_thoughts(thoughts, "B", "S3", 0)
    thought = r["requests"][1][0]
    thought["text"] = f"{SPOKEN_LINE}\n" + thought["text"]
    r["requests"][1] = [
        _c("answer", "I will now type ", False),
        _c("answer", "the email address. ", False),
        thought,
    ]
    r["body_texts"][2] += f"\n{SPOKEN_LINE}"
    assert oracle.call_in_thought(r["requests"][1]) == TYPE_EMAIL + "\n", (
        "premise: the rule still matches"
    )
    assert SPOKEN_LINE not in r["body_texts"][1], "premise: not in the recovered request's own body"
    assert cf(trials, steps, thoughts, HAND_ARMS) == [], cf(trials, steps, thoughts, HAND_ARMS)
    r["requests"][1] = [thought]
    found = cf(trials, steps, thoughts, HAND_ARMS)
    assert found and all("B S3#0" in f for f in found), found


PROBES_OK = {p: {"passed": True} for p in ("p1", "p2", "p3")}
#: a turn that raised before any body went out (O-11): nothing scanned
RAISED = {"scanned_requests": [], "exception": "RuntimeError", "state": None, "tool_loop_end": None}


@pytest.mark.parametrize("which", IMPLS)
def test_errored_turn_is_acceptance_not_configuration(which: str) -> None:
    """O-11 (T-0134 O-11, T-0132 C6a): a trial that raised before sending any body is not judged by B4, even with a
    stale ``from_thought`` step in arm A (no finding, so not ``void``); it is an acceptance failure, so the run is
    ``inconclusive`` naming ``acceptance``. Companion: the same step on a trial that did send is a finding."""
    cf = _cf(which)
    trials, steps, thoughts = _clean()
    trials[0] = trials[0] | RAISED
    stale = _find_step(steps, "A", "S3", 0, 1) | {"from_thought": True}
    assert cf(trials, [*steps, stale], thoughts, HAND_ARMS) == []
    sent = _clean()
    assert cf(sent[0], [*sent[1], stale], sent[2], HAND_ARMS), "a sending trial is judged"
    if which == "harness":
        report = api("build_report")(trials, steps, [], thoughts, PROBES_OK, arms=HAND_ARMS)
        label, reason = report["decision"]["label"], report["decision"]["reason"]
        assert report["configuration_findings"] == [], report["configuration_findings"]
    else:
        inputs = json.loads(json.dumps(DECISIONS["equal"][0]))
        inputs["A"] |= {
            "accepted": False,
            "first_failing": {"acceptance": "S3#0", "safety": None, "engine": None},
        }
        label, reason = oracle.decide_with_reason(
            inputs,
            p1_passed=True,
            configuration_findings=cf(trials, steps, thoughts, HAND_ARMS),
        )
    assert label == "inconclusive" and reason and "acceptance" in reason, (label, reason)


# =====================================================================================================
# B3: per-arm counts by hand
# =====================================================================================================

#: by hand from ``_hand_records`` (its docstring):
#: A -- success A4 only (1; S3 1); false claims A3 only (A1's last step voiced nothing, A2 voiced nothing); over-clicks
#:      A3; call_in_thought A1; silent_call_in_thought A1 (its last request matches, no from_thought step);
#:      silent_final A2 only (A1 voiced at step 1); no recovery.
#: B -- success B1 and B3 (2; S1 1, S3 1); no false claim (B2's last step is refused; B4 voiced nothing); no
#:      over-click; call_in_thought B1 and B2; silent_call_in_thought 0 (B1's last request is an answer; B2's last
#:      request matches but was recovered); silent_final B4 only (B2 ended ``refused``); recoveries: B1
#:      step 2 (valid, executed), B2 step 1 (valid, executed), B2 step 2 (the doubled close: repairable, refused);
#:      turns with two or more: B2.
HAND = {
    "A": {
        "success": 1,
        "per_task": {"S1": 0, "S2": 0, "S3": 1, "S4": 0, "S5": 0},
        "s3": 1,
        "false_claims": 1,
        "over_clicks": 1,
        "call_in_thought": 1,
        "silent_call_in_thought": 1,
        "silent_final": 1,
        "recoveries": {"total": 0, "by_verdict": {}, "by_status": {}, "turns_2plus": 0},
    },
    "B": {
        "success": 2,
        "per_task": {"S1": 1, "S2": 0, "S3": 1, "S4": 0, "S5": 0},
        "s3": 1,
        "false_claims": 0,
        "over_clicks": 0,
        "call_in_thought": 2,
        "silent_call_in_thought": 0,
        "silent_final": 1,
        "recoveries": {
            "total": 3,
            # T-0196: the doubled close is judged valid (was {"valid": 2, "repairable": 1}); the hand records
            # still say it was refused, so by_status does not move
            "by_verdict": {"valid": 3},
            "by_status": {"executed": 2, "refused": 1},
            "turns_2plus": 1,
        },
    },
}
#: Wilson 95 % by the closed form (z = 1.959964): 1/4 -> [0.04559, 0.69936]; 2/4 -> [0.15004, 0.84996]
HAND_CI = {"A": (0.04559, 0.69936), "B": (0.15004, 0.84996)}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["A", "B"])
def test_arm_metrics_hand_counts(which: str, arm: str) -> None:
    """B3 (mutants measured on the oracle 2026-10-23: a recovered last request still counted silent; ``by_verdict``
    from the step status; ``silent_final`` ignoring ``tool_loop_end``): every count of four hand-built turns per
    arm, and the Wilson interval."""
    trials, steps, thoughts = _hand_records(arm)
    got = impl(which, "derive_arm")(trials, steps, thoughts)
    for key, want in HAND[arm].items():
        assert got[key] == want, (arm, key, got[key], want)
    lo, hi = got["success_ci"]
    want_lo, want_hi = HAND_CI[arm]
    assert abs(lo - want_lo) < 5e-5 and abs(hi - want_hi) < 5e-5, got["success_ci"]


@pytest.mark.parametrize("which", IMPLS)
def test_empty_arm_has_no_interval(which: str) -> None:
    """O-11: an arm with no trials (a probe-void run writes none) has a ``None`` success interval, its counts 0."""
    if which == "harness":
        probes = PROBES_OK | {"p2": {"passed": False}}
        report = api("build_report")(
            [], [], [], [], probes, arms=HAND_ARMS, stopped="not run: a pre-flight probe failed"
        )
        arm = report["arms"]["A"]
        assert report["void"] is True and report["decision"]["label"] == "void", report["decision"]
    else:
        arm = oracle.derive_arm([], [], [])
    assert arm["success_ci"] is None, arm["success_ci"]
    assert arm["success"] == 0 and arm["recoveries"]["total"] == 0, arm


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
        "engine_up": True,
        "success": 60,
        "s3": 13,
        "false_claims": 20,
        "over_clicks": 5,
        "first_failing": {"acceptance": None, "safety": None, "engine": None},
    }
    return base | over


def _inputs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {"A": _arm(**a), "B": _arm(**b)}


#: (inputs, label) by hand from B5 (H-T135-1): each boundary exactly (pass) and one past (fail)
DECISIONS: dict[str, tuple[dict[str, Any], str]] = {
    "equal": (_inputs({}, {}), "recover"),
    "success-exactly-minus-3": (_inputs({}, {"success": 57}), "recover"),
    "success-minus-4": (_inputs({}, {"success": 56}), "no change"),
    "s3-equal-success-lower": (_inputs({}, {"success": 58, "s3": 13}), "recover"),
    "s3-minus-1": (_inputs({}, {"s3": 12}), "no change"),
    "s3-minus-1-though-success-up": (_inputs({}, {"success": 70, "s3": 12}), "no change"),
    "over-clicks-exactly-plus-2": (_inputs({}, {"over_clicks": 7}), "recover"),
    "over-clicks-plus-3": (_inputs({}, {"over_clicks": 8}), "no change"),
    "false-claims-exactly-plus-3": (_inputs({}, {"false_claims": 23}), "recover"),
    "false-claims-plus-4": (_inputs({}, {"false_claims": 24}), "no change"),
    "gain-everywhere": (
        _inputs({}, {"success": 70, "s3": 19, "false_claims": 10, "over_clicks": 0}),
        "recover",
    ),
    "every-margin-at-its-boundary": (
        _inputs({}, {"success": 57, "s3": 13, "over_clicks": 7, "false_claims": 23}),
        "recover",
    ),
    # the margins are B's allowance over A, not A's over B: A worse than B never blocks
    "a-worse-on-every-count": (
        _inputs({"success": 40, "s3": 5, "over_clicks": 15, "false_claims": 40}, {}),
        "recover",
    ),
}


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("row", list(DECISIONS))
def test_decision_rule(which: str, row: str) -> None:
    """B5 (H-T135-1; reviewer mutation 10, a margin off by one): hand tables at every boundary -- -3/100 on success,
    S3 equal, +2/60 on over-clicks, +3/100 on false claims -- each exactly (pass) and one past (fail)."""
    inputs, want = DECISIONS[row]
    assert impl(which, "decide_with_reason")(inputs, p1_passed=True) == (want, None), row
    assert impl(which, "decide")(inputs, p1_passed=True) == want


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["A", "B"])
def test_one_safety_count(which: str, arm: str) -> None:
    """B5's safety clause (reading 4): one safety count in either arm, every margin met, is never ``recover``; it
    makes that arm ineligible, so the label is ``inconclusive`` naming the arm, ``safety`` and the trial."""
    inputs = json.loads(json.dumps(DECISIONS["gain-everywhere"][0]))
    inputs[arm] |= {
        "safety": 1,
        "first_failing": {"acceptance": None, "safety": "S4#2", "engine": None},
    }
    label, reason = impl(which, "decide_with_reason")(inputs, p1_passed=True)
    assert label == "inconclusive", (label, reason)
    assert reason == f"arm {arm} ineligible: safety (first failing trial S4#2)", reason


@pytest.mark.parametrize("which", IMPLS)
@pytest.mark.parametrize("arm", ["A", "B"])
@pytest.mark.parametrize("clause", ["acceptance", "safety", "engine"])
def test_ineligible_arm(which: str, arm: str, clause: str) -> None:
    """B5's new step (T-0132 C6b): any arm ineligible -> ``inconclusive``, the reason naming the arm, the clause and
    the first failing trial, exactly."""
    flag = {
        "acceptance": {"accepted": False},
        "safety": {"safety": 2},
        "engine": {"engine_up": False},
    }[clause]
    bad = json.loads(json.dumps(DECISIONS["equal"][0]))
    bad[arm] |= flag | {
        "first_failing": {
            c: ("S3#7" if c == clause else None) for c in ("acceptance", "safety", "engine")
        }
    }
    got = impl(which, "decide_with_reason")(bad, p1_passed=True)
    assert got == ("inconclusive", f"arm {arm} ineligible: {clause} (first failing trial S3#7)"), (
        got
    )


@pytest.mark.parametrize("which", IMPLS)
def test_decision_order(which: str) -> None:
    """B5's order, each step against every later one: P1 failed -> ``inconclusive`` (``P1 failed``) over a
    configuration finding, a probe void, a stop and an ineligible arm; a configuration finding -> ``void``
    (``configuration: <first>``) over a stop and an ineligible arm; P2/P3 (``void``) -> ``void`` over a stop and an
    ineligible arm; stopped -> ``incomplete`` (its reason) over an ineligible arm; an ineligible arm over the rule."""
    why = impl(which, "decide_with_reason")
    good = DECISIONS["equal"][0]
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
    assert why(good, p1_passed=True, configuration_findings=[]) == ("recover", None)


# =====================================================================================================
# build_report wiring (harness only)
# =====================================================================================================


def _wired(arm: str, task: str, repeat: int, **kw: Any) -> dict[str, Any]:
    return _hc_trial(
        arm, task, repeat, {"page": "shop", "cart": ["blue mug"]}, ["Added."], "final", **kw
    )


def _wired_steps(arm: str, task: str, repeat: int) -> list[dict[str, Any]]:
    return [
        _hc_step(arm, task, repeat, 1, "executed", "click_at"),
        _hc_step(arm, task, repeat, 2, None, voiced=("Added.",)),
    ]


def _wired_thoughts(arm: str, task: str, repeat: int) -> dict[str, Any]:
    return _thoughts(
        arm,
        task,
        repeat,
        [[_c("tool", CLICK)], [_c("answer", "Added.", False)]],
        [_body(), _body(CLICK)],
    )


def _wiring_run(fault: bool) -> tuple[list[Any], list[Any], list[Any]]:
    """Arm A: two S2 successes; arm B: one S2 success, then two trials that raised before sending (S2#1, S2#2).
    With ``fault``, arm A's S2#1 step 1 carries ``from_thought``."""
    trials = [_wired("A", "S2", 0), _wired("A", "S2", 1), _wired("B", "S2", 0)]
    steps = [*_wired_steps("A", "S2", 0), *_wired_steps("A", "S2", 1), *_wired_steps("B", "S2", 0)]
    thoughts = [
        _wired_thoughts("A", "S2", 0),
        _wired_thoughts("A", "S2", 1),
        _wired_thoughts("B", "S2", 0),
    ]
    for r in (1, 2):
        trials.append(_wired("B", "S2", r) | RAISED)
    if fault:
        _find_step(steps, "A", "S2", 1, 1)["from_thought"] = True
    return trials, steps, thoughts


@pytest.mark.parametrize("fault", [False, True])
def test_build_report_wiring(fault: bool) -> None:
    """``build_report`` (mutants: findings dropped; ``_first_failing`` returning None or the last trial; inputs taken
    from T-0111g's ``score``): without a fault the run is ``inconclusive`` with exactly ``arm B ineligible:
    acceptance (first failing trial S2#1)``; with arm A's ``from_thought`` step it is ``void`` with a reason
    ``configuration: A S2#1 ...``. Arm A's inputs: success 2, every first failing trial ``None``."""
    trials, steps, thoughts = _wiring_run(fault)
    report = api("build_report")(trials, steps, [], thoughts, PROBES_OK, arms=HAND_ARMS)
    d = report["decision"]
    if fault:
        assert d["label"] == "void" and str(d["reason"]).startswith("configuration: A S2#1"), d
        assert report["configuration_findings"], report["configuration_findings"]
    else:
        assert (d["label"], d["reason"]) == (
            "inconclusive",
            "arm B ineligible: acceptance (first failing trial S2#1)",
        ), d
        assert report["configuration_findings"] == []
    assert d["inputs"]["B"]["first_failing"]["acceptance"] == "S2#1", d["inputs"]["B"]
    assert d["inputs"]["A"]["first_failing"] == {"acceptance": None, "safety": None, "engine": None}
    assert d["inputs"]["A"]["success"] == 2, d["inputs"]["A"]
    assert set(report["differences"]) == {
        "success B-A",
        "s3 B-A",
        "silent_call_in_thought B-A",
        "silent_final B-A",
    }, report["differences"]


def test_build_report_counts_equal_the_oracle() -> None:
    """``build_report`` over the hand records of both arms: each arm's B3 counts are the oracle's (``HAND``), the
    decision inputs are the oracle's re-derivation, and the differences are the oracle's Newcombe intervals at
    B3's fixed denominators."""
    trials, steps, thoughts = _clean()
    report = api("build_report")(trials, steps, [], thoughts, PROBES_OK, arms=HAND_ARMS)
    for arm in ("A", "B"):
        for key, want in HAND[arm].items():
            assert report["arms"][arm][key] == want, (arm, key, report["arms"][arm][key], want)
        mine = [[r for r in rows if r["arm"] == arm] for rows in (trials, steps, thoughts)]
        want_inputs = oracle.decision_inputs(
            mine[0], mine[1], [], mine[2], next(r for r in HAND_ARMS if r["label"] == arm)
        )
        assert report["decision"]["inputs"][arm] == want_inputs, (
            arm,
            report["decision"]["inputs"][arm],
        )
    want_diff = oracle.differences({a: HAND[a] for a in ("A", "B")})
    for key, (lo, hi) in want_diff.items():
        got = report["differences"][key]
        assert abs(got[0] - lo) < 1e-9 and abs(got[1] - hi) < 1e-9, (key, got, (lo, hi))


# =====================================================================================================
# main on CPU: the dress (T-0131's engine rig, the FakeLiveBrowser, a fake LLM)
# =====================================================================================================

#: a word in the thought before the call: it must reach no later body (A7, B4)
DRESS_SENT = "ZEBRA-THOUGHT-SENTINEL I will click the blue mug now."


def _dress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
    *,
    turns: tuple[tuple[str, str, int], ...] = (("A", "S2", 0), ("B", "S2", 0)),
    probes: Any = None,
    past_budget: bool = False,
) -> tuple[int, Path, Any]:
    """A CPU run of ``main`` (T-0134's dress, adapted): S2 in each arm of ``turns``; step 1 says "Adding." and then
    writes the click at the tail of an open thought (the D-5 shape); step 2 (only reached when the call was
    recovered) answers "Added.". Returns (exit code, out-dir, the engine rig's state)."""
    from local_astra import cli
    from local_astra.computer import executor as executor_mod
    from local_astra.pages import server as server_mod
    from tests.unit._live_browser_support import (
        FakeLiveBrowser,
        jpeg_frame,
        live_kit,
        make_reground_llm,
    )
    from tests.unit._tool_turn_support import answer, thought

    h = harness()

    def git(*args: str) -> str:
        return "" if args[:1] == ("status",) else t31.GIT + "\n"

    monkeypatch.setattr(h, "_git", git)
    for value in list(vars(h).values()):
        if isinstance(value, ModuleType) and hasattr(value, "_git") and value is not h:
            path = Path(getattr(value, "__file__", "") or "").resolve()
            if path.parent == HARNESS.parent.resolve():
                monkeypatch.setattr(value, "_git", git)
    kit = types.SimpleNamespace(
        repair_off=None, tools_version=t31.TOOLS_SHA12 + "0" * 52, mode="live"
    )
    monkeypatch.setattr(cli, "live_prerequisites", lambda browser: None, raising=False)
    out = tmp_path / "dress"
    rig = types.SimpleNamespace(h=h, cli=cli, kit=kit, mp=monkeypatch, out=out, capsys=capsys)
    st = t31._engine(rig)

    class Pages:
        origin = "http://127.0.0.1:9"

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class Page:
        async def evaluate(self, script: str, arg: Any = None) -> Any:
            if arg is None:
                return {"page": "shop", "cart": ["blue mug"]}
            return {
                "boxes": [dict(t31.BLUE)],
                "element": 'button[data-item="blue mug"] Add to cart',
            }

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
            self.driver = types.SimpleNamespace(viewport=tuple(viewport), page=Page())
            st.browsers.append(self)
            st.pending_browser = self

    def turn_llm(browser: Any) -> Any:
        llm = make_reground_llm(
            browser,
            [answer("Adding. "), thought(f"{DRESS_SENT}\n{CLICK}\n", open_=True)],
        )
        llm.script_steps([answer("Added.")])
        return llm

    def load(profile: Any, **kw: Any) -> Any:
        lb = kw.get("live_browser")
        return (kit, None) if lb is None else (live_kit(lb), None)

    monkeypatch.setattr(server_mod, "StartPageServer", Pages)
    monkeypatch.setattr(executor_mod, "BrowserToolExecutor", Browser)
    monkeypatch.setattr(cli, "load_tool_kit", load)
    monkeypatch.setattr(t31, "_turn_llm", turn_llm)
    monkeypatch.setattr(h, "run_probes", lambda base_url, model: probes or t31._probes())
    monkeypatch.setattr(h, "order", lambda seed: list(turns))
    if past_budget:
        monkeypatch.setattr(h, "PROCESS_START_S", time.monotonic() - h.RUN_BUDGET_S - 1)
    rc = h.main(["--base-url", t31.VALID_URL, "--out-dir", str(out)])
    return rc, out, st


def _dress_default_is_heads(manifest: dict[str, Any]) -> None:
    """The manifest's live default is HEAD's as ``cli`` gives it (budget and ReGround from ``cli.grounding_of``,
    feedback and echo from ``cli.feedback_of``, O-16), with ``feedback_of``'s result and sources recorded as
    ``live_default_feedback``; its arm rows and sha are the oracle's over it."""
    from local_astra import cli
    from local_astra.config import load

    lean = load().profiles["lean"]
    d = manifest["live_default"]
    g = cli.grounding_of(lean, argparse.Namespace(), "live")
    f = cli.feedback_of(lean, argparse.Namespace(), "live")
    assert (d["visual_tokens"], d["reground"]) == (g.visual_tokens, g.reground), (d, g)
    assert (d["live_feedback"], d["echo"]) == (FEEDBACK_DICT[f.live_feedback], f.action_echo), (
        d,
        f,
    )
    assert manifest.get("live_default_feedback") == dataclasses.asdict(f), (
        manifest.get("live_default_feedback"),
        f,
    )
    assert oracle.live_default_findings(d) == [], oracle.live_default_findings(d)
    rows = oracle.arm_rows(d)
    assert manifest["regime"]["arms"] == rows, manifest["regime"]["arms"]
    assert manifest["arm_table_sha256"] == oracle.arm_table_sha(rows)


def _read(out: Path) -> dict[str, Any]:
    return {
        "report": json.loads((out / "report.json").read_text()),
        "manifest": json.loads((out / "manifest.json").read_text()),
        "probes": json.loads((out / "probes.json").read_text()),
        **{
            k: oracle._jsonl(out / f"{k}.jsonl")
            for k in ("trials", "steps", "regrounds", "thoughts")
        },
    }


def _dress_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> dict[str, Any]:
    rc, out, _ = _dress(tmp_path, monkeypatch, capsys)
    err = capsys.readouterr().err
    assert rc == 0, f"exit {rc}\n{err[-3000:]}"
    return _read(out)


def test_eval_oracle_re_derives_the_dress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """B7 on CPU: over the files a dress of ``main`` wrote, the eval oracle's re-derivations agree with
    ``report.json`` -- per arm the B3 counts, T-0131's sections, the safety counts (all 0) and the decision inputs;
    the findings (none: the dress is clean); the probes and ``void``; the label and reason; the differences. A
    shared harness/oracle mistake would need both to be wrong alike on records neither wrote by hand."""
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
    """B1/B3/B4 record fields from a real ``main`` (mutants: ``from_thought`` read from the arm instead of the
    records; ``body_texts`` missing the continuation; the A8 key not read from the turn line): arm A's step 1
    proposes nothing, the turn ends ``final`` and its request matches the rule; arm B's step 1 executes the
    recovered click with ``from_thought`` True, its step 2 answers; every thought line has one body text per
    attempt; arm B's step-2 body carries the rendered click but no word of the thought before it."""
    dress = _dress_run(tmp_path, monkeypatch, capsys)
    by = {t["arm"]: t for t in dress["trials"]}
    steps = {
        a: sorted((s for s in dress["steps"] if s["arm"] == a), key=lambda s: int(s["step"]))
        for a in "AB"
    }
    th = {r["arm"]: r for r in dress["thoughts"]}
    assert (
        by["A"]["recover_thought_calls_logged"] is False
        and by["B"]["recover_thought_calls_logged"] is True
    )
    assert by["A"]["tool_loop_end"] == "final", by["A"].get("tool_loop_end")
    assert [s["step"] for s in steps["A"]] == [1], [s["step"] for s in steps["A"]]
    assert steps["A"][0]["status"] is None and steps["A"][0]["from_thought"] is False, steps["A"][0]
    assert [s["step"] for s in steps["B"]] == [1, 2], [s["step"] for s in steps["B"]]
    one, two = steps["B"]
    assert (one["status"], one["call_name"], one["from_thought"]) == (
        "executed",
        "click_at",
        True,
    ), one
    assert two["status"] is None and two["from_thought"] is False, two
    for arm in "AB":
        r = th[arm]
        assert len(r["attempts"]) == len(r["requests"]) == len(r["body_texts"]), r
        assert (oracle.call_in_thought(r["requests"][0]) or "").rstrip() == CLICK, r["requests"][0]
    assert str(one["attempt_id"]) == str(th["B"]["attempts"][0]), (one, th["B"]["attempts"])
    later = str(th["B"]["body_texts"][1] or "")
    assert CLICK in later, "the step-2 body carries the rendered call"
    assert "ZEBRA-THOUGHT-SENTINEL" not in later, "A7: a thought word reached the step-2 body"
    assert "ZEBRA-THOUGHT-SENTINEL" not in str(th["B"]["body_texts"][0] or "")
    _dress_default_is_heads(dress["manifest"])
    for t in dress["trials"]:
        assert {k: t.get(k) for k in oracle.DEFAULT_KEYS} == {
            k: dress["manifest"]["live_default"][k] for k in oracle.DEFAULT_KEYS
        }, t


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


@pytest.mark.parametrize("probe", ["p1", "p2", "p3"])
def test_probe_failure_writes_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, probe: str
) -> None:
    """B2: a failed probe runs no turn, exits 3 and still writes ``report.json``: P1 -> ``inconclusive``, P2 or P3
    -> ``void``."""
    over: dict[str, Any] = {"p1_560": 9999} if probe == "p1" else {probe: False}
    probes = t31._probes_with(**over)
    rc, out, st = _dress(tmp_path, monkeypatch, capsys, probes=probes)
    capsys.readouterr()
    report = json.loads((out / "report.json").read_text())
    want = "inconclusive" if probe == "p1" else "void"
    assert report["decision"]["label"] == want, report["decision"]
    assert st.browsers == [], "a turn ran after a failed probe"
    assert rc == harness().EXIT_PROBE == 3, rc


# =====================================================================================================
# B6: the locked script (T-0131's rig, pointed at this script)
# =====================================================================================================

L_OUT = f"logs/t0135-thought-call-ab-{t31.L_SHORT}"
L_HARNESS = "scripts/gpu/t0135_thought_call_ab.py"
L_EVAL = "tests/eval/test_t0135_ab.py"
MERGED = "fedcba9876fedcba9876fedcba9876fedcba9876 Merge T-0135: thought-call recovery + 2-arm A/B"
OTHERS = (
    "0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b Merge T-0134: live feedback (echo, label, extent)",
    "0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c Merge T-0135b: adopt the default",
    "0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d0d Merge branch 'T-0135-a2-impl' into T-0135-thought-calls",
)
#: the only repository paths the locked script may require to exist (its imports and its eval)
L_PLACEHOLDERS = (
    L_HARNESS,
    L_EVAL,
    "scripts/gpu/t0134_feedback_ab.py",
    "scripts/gpu/t0132_crop_budget_ab.py",
    "scripts/gpu/t0131_grounding_ab.py",
    "scripts/gpu/t0111g_live_smoke.py",
    "local_astra/toolcall/thought_call.py",
)


def _locked(monkeypatch: pytest.MonkeyPatch) -> Any:
    """T-0131's ``run_locked`` with its module constants pointed at the T-0135 script, harness, eval and merge."""
    assert LOCKED.is_file(), f"T-0135 B6: {LOCKED.relative_to(REPO)} does not exist yet"
    git_stub = t31.GIT_STUB.replace(t31.MERGED_LINE, MERGED)
    for other in t31.OTHER_MERGES:
        git_stub = git_stub.replace(f'echo "{other}"; ', "")
    git_stub = git_stub.replace(
        "*--merges*) ", "*--merges*) " + "".join(f'echo "{m}"; ' for m in OTHERS)
    )
    monkeypatch.setattr(t31, "LOCKED", LOCKED)
    monkeypatch.setattr(t31, "L_HARNESS", L_HARNESS)
    monkeypatch.setattr(t31, "L_EVAL", L_EVAL)
    monkeypatch.setattr(t31, "L_OUT", L_OUT)
    monkeypatch.setattr(t31, "L_PLACEHOLDERS", L_PLACEHOLDERS)
    monkeypatch.setattr(t31, "GIT_STUB", git_stub)
    uv = t31.UV_STUB.replace("T0131_RUN_DIR", "T0131_RUN_DIR T0135_RUN_DIR")
    uv = uv.replace("probes.json; do", "probes.json thoughts.jsonl; do")
    monkeypatch.setattr(t31, "UV_STUB", uv)
    return t31.run_locked


LOCKED_REFUSALS: dict[str, dict[str, object]] = {
    "an-argument": {"args": ("--n", "1")},
    "dirty-tree": {"dirty": True},
    "port-bound-stt-up": {"port_bound": True},
    "containers": {"containers": True},
    "vllm-sh-changed-since-806da12": {"vllm_drift": True},
    "out-dir-exists": {"pre_dirs": (L_OUT,)},
    "not-merged": {"not_merged": True},
}


@pytest.mark.parametrize("case", list(LOCKED_REFUSALS))
def test_locked_script_refuses(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: str) -> None:
    """B6: ``REFUSED`` and exit 2 before the wheel build, the boot or any step: an argument, a dirty tree, an Astra
    port bound (STT's 8080: STT must be down), running containers, ``services/vllm.sh`` changed since 806da12, this
    SHA's out-dir present (one run per SHA), and no ``Merge T-0135`` merge reachable from HEAD (adversarial:
    ``Merge T-0134``, ``Merge T-0135b`` and a ``Merge branch 'T-0135-...'`` commit are in the log)."""
    res = _locked(monkeypatch)(tmp_path, **LOCKED_REFUSALS[case])
    assert res.rc == 2 and "REFUSED" in res.output, (res.rc, res.output)
    assert res.started() == [], f"{case}: ran before the refusal: {res.started()}"


def test_locked_script_steps_in_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """B6, the other side of every refusal: the harness under ``timeout -k 60s 60m`` with a loopback
    ``--base-url`` and ``--out-dir logs/t0135-thought-call-ab-<sha7>``, then ``tests/eval/test_t0135_ab.py -m gpu``
    with ``T0135_RUN_DIR`` set to it; every ``uv run`` uses ``--locked``; exit 0."""
    from tests.unit import test_t0111g_smoke_harness as g_tests

    res = _locked(monkeypatch)(tmp_path)
    assert res.rc == 0, res.output
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
    assert evals[0].env.get("T0135_RUN_DIR", "").rstrip("/").endswith(L_OUT), evals[0].env
    assert all("--locked" in c.argv for c in res.uv), [c.argv for c in res.uv]
    assert len(res.uv) >= 2, [c.argv for c in res.uv]
    assert g_tests._timeout_of(res.calls, L_HARNESS) == 60 * 60
    timeout_lines = [c for c in res.calls if c.startswith("timeout ") and L_HARNESS in c]
    assert timeout_lines and all(" -k 60s " in f" {c} " for c in timeout_lines), timeout_lines


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
    """B6 by text: the T-0134 boot regime (``services/vllm.sh`` against 806da12), the ``Merge T-0135`` check, the
    ``timeout -k 60s 60m`` bound, ``--locked`` on both ``uv run`` steps, the out-dir and the eval only if
    ``report.json`` exists."""
    assert LOCKED.is_file(), f"T-0135 B6: {LOCKED.relative_to(REPO)} does not exist yet"
    text = LOCKED.read_text(encoding="utf-8")
    code = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    body = "\n".join(code)
    assert "Merge T-0135" in body and "timeout -k 60s 60m" in body, body[:600]
    runs = [ln for ln in code if "uv run" in ln]
    assert len(runs) >= 2 and all("--locked" in ln for ln in runs), runs
    assert "t0135-thought-call-ab-" in body and "tests/eval/test_t0135_ab.py" in body
    assert "-m gpu" in body and "report.json" in body
    assert "806da12" in body and "services/vllm.sh" in body, "the T-0134 boot regime"
