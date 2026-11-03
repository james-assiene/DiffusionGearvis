"""T-0191 group B, B2 and B3: the navigation smoke's analysis, a CPU function over the run directory.

**The interface this module pins (the test-author's choice; the implementer provides it):**

``scripts/gpu/t0191_nav_smoke_analysis.py`` (loaded by path) defines ``analyse(run_dir: pathlib.Path) -> dict``,
pure apart from reading ``run_dir``. Run directory (what ``scripts/gpu/t0191_nav_smoke.sh`` writes, see
``test_t0191_nav_smoke_script.py``): ``order.json`` holds ``[scenario, rep]`` pairs (N1, N2, N3; rep 0-4); each
trial's files are under ``turns/<scenario>-<rep>/``: ``sessions/**/*.jsonl`` (the session log), ``turns.jsonl``,
``session.out`` (startup output), ``aux/connect.log`` (strace) and the two probes' outputs (names not pinned).

Returned report (keys pinned here; more keys are allowed)::

    {
      "decision": "PASS" | "FAIL" | "INCONCLUSIVE",          # B3, over N1 and N2 only
      "scenarios": {
        "N1": {"setup_failed": int, "scored": int, "navigated": int},   # scored = valid trials (B3's denominator)
        "N2": {...},
        "N3": {"in_site": int, "over_applied": int},          # report-only, never in the decision
      },
      "safety": {name: int},                                  # every counter; PASS and FAIL need all of them 0
      "void": [str],                                          # one finding per void trial, naming <scenario>-<rep>
      "trials": [{"scenario", "repeat", "void": bool, "setup_failed": bool, "navigated": bool,
                  "turn2_first_call": str | None,             # the name of turn 2's first tool call
                  "turn2_start_page": bool,                   # turn 2 executed Ctrl+L or Ctrl+T
                  "turn2_typed": [{"text": str, "on_web_html": bool}],   # every executed turn-2 type_at
                  "n3": "in_site" | "over_applied" | "other" | None}],   # None outside N3
    }

**Readings (the test-author's; listed in the report as places the criteria leave open):**

- Turns (ruling B-RV1, which replaces the test-author's first reading "the first two turn ids that appear"): a
  session-log record's ``turn_id`` says which turn it belongs to, but real logs open with ``log.started`` and
  ``wire.in`` records with no turn and IDLE ``machine.step`` records under ``turn_id "t0"``, and the steps that open
  turn k carry turn k-1's id. Turn 1 and turn 2 are the first two turn ids that carry an ``llm.request`` (or a
  ``turn.record``), never ``t0``. The builder below writes that preamble, and every tool call twice (``proposed``,
  then its final status), as real logs do; ``test_real_log_shape`` copies a real log's record structure.
- B-RV2: a trial whose turn 2 never started is named in the report's ``turn2_missing`` list (a list of findings, each
  naming ``<scenario>-<rep>``, like ``void``) and dropped from its scenario's denominator, like a void. B-RV4: such a
  trial whose turn 1 also missed the setup site still counts as a setup failure.
- B-RV4: the report-only ``turn2_steps`` and ``turn2_steps_to_arrive`` (per trial) count executed calls only.
- B2 verbatim: **setup failed** iff no *executed* turn-1 ``tool.call`` has a ``resolved.hosts`` entry matching the
  setup site (N1 and N3 ``(^|\\.)amazon\\.[a-z.]+$``, N2 ``(^|\\.)duckduckgo\\.com$``); **navigated** iff some
  *executed* turn-2 ``tool.call`` has a ``resolved.hosts`` entry matching ``(^|\\.)photopea\\.com$`` (N1) or
  ``(^|\\.)wikipedia\\.org$`` (N2).
- "Typed on ``/web.html``": the executor puts the browser on ``/web.html`` with an executed start-page action
  (``press_key`` ``Ctrl+L``/``Ctrl+T``, action ``"opened the start page"``); a ``type_at`` is on ``/web.html`` iff
  such an action came earlier in the session with no executed action carrying ``resolved.navigated == True``
  between them. Every row below starts turn 2 off ``/web.html`` (turn 1 ends with a navigated action), except
  ``test_n3_over_applied_by_typing_on_web_html``, whose turn 1 ends on ``/web.html``.
- N3: ``in_site`` iff turn 2 executed no start-page action, some executed turn-2 action has an ``amazon.*`` host,
  and some executed turn-2 ``type_at`` whose text contains "towel" was not on ``/web.html``; ``over_applied`` iff
  turn 2 executed ``Ctrl+L``/``Ctrl+T`` or typed on ``/web.html``; ``other`` otherwise.
- B3: a scenario is complete iff all five of its trials ran (a turn dir with a session log) and at most one is a
  setup failure; a run with a trial of ``order.json`` that never ran (the B4 budget stop) is incomplete. Ruled
  (task file "Group B readings", 6f78c5c2): a void trial never counts as navigated and is dropped from its
  scenario's denominator (``scored`` = valid trials: ran, setup ok, not void); fewer than four scored -> INCONCLUSIVE.
- Safety (ruled, 6f78c5c2): the copied T-0174 analysis's ``voiced_uncommitted`` and ``non_public_connects``, plus
  the script's hard stops; no new counters. The rows inject those two and assert on the whole ``safety`` dict.

Expected values are worked by hand. Regime: CPU, synthetic run directories in tmp; no GPU, no network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE = REPO / "scripts" / "gpu" / "t0191_nav_smoke_analysis.py"
SCENARIOS = ("N1", "N2", "N3")
N = 5
START = "opened the start page"
GUIDANCE_LINE = "astra session: live guidance: v3 (flag)\n"


# --------------------------------------------------------------------------------------------- the module


def analysis() -> ModuleType:
    assert MODULE.is_file(), f"T-0191 B2: {MODULE.relative_to(REPO)} does not exist yet"
    spec = importlib.util.spec_from_file_location("_t0191_nav_smoke_analysis_unit", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise AssertionError(
            f"{MODULE.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sys.modules.pop(spec.name, None)
    return module


def analyse(run_dir: Path) -> dict[str, Any]:
    fn = getattr(analysis(), "analyse", None)
    assert callable(fn), f"T-0191 B2: {MODULE.name} has no analyse(run_dir)"
    report = fn(run_dir)
    assert isinstance(report, dict), report
    return report


# --------------------------------------------------------------------------------------------- actions


@dataclass(frozen=True)
class Act:
    name: str
    args: dict[str, Any]
    status: str = "executed"
    resolved: dict[str, Any] = field(default_factory=dict)
    action: str = ""
    reason: str | None = None


def go(text: str, *hosts: str, navigated: bool = True) -> Act:
    """``type_at`` with Enter that lands on a page whose proxy hosts are ``hosts``."""
    return Act(
        "type_at",
        {"x": 500, "y": 300, "text": text, "press_enter": True},
        resolved={"hosts": list(hosts), "navigated": navigated},
        action=f"typed {text!r} and pressed Enter",
    )


def key(combo: str, status: str = "executed") -> Act:
    start = combo in ("Ctrl+L", "Ctrl+T") and status == "executed"
    return Act(
        "press_key",
        {"combo": combo},
        status=status,
        resolved={"combo": combo, "hosts": [], "navigated": True} if start else {},
        action=START if start else "",
        reason=None if status == "executed" else "not available in the browser",
    )


def click(*hosts: str, navigated: bool = True) -> Act:
    return Act(
        "click_at", {"x": 400, "y": 200}, resolved={"hosts": list(hosts), "navigated": navigated}
    )


#: turn 1, setup succeeded (lands on the setup site); turn 1, setup failed (never reaches it)
T1_AMAZON = [go("amazon.com", "www.amazon.com", "m.media-amazon.com")]
T1_DDG = [go("duckduckgo.com/html/?q=photo+editor", "html.duckduckgo.com")]
T1_FAIL = [go("amazon.com", "www.mojeek.com")]

#: turn 2 for N1: Ctrl+L, then the address on /web.html (good); the reported bug (typed into amazon's search box)
N1_NAV = [key("Ctrl+L"), go("photopea.com", "www.photopea.com")]
N1_BUG = [go("photopea.com", "www.amazon.com", "m.media-amazon.com")]
N2_NAV = [key("Ctrl+L"), go("en.wikipedia.org", "en.wikipedia.org", "upload.wikimedia.org")]
N2_BUG = [go("Wikipedia", "html.duckduckgo.com")]
#: turn 2 for N3: the site's own box (good); Ctrl+L and the address on /web.html (over-applied)
N3_IN = [go("bath towels", "www.amazon.com", "m.media-amazon.com")]
N3_OVER = [key("Ctrl+L"), go("amazon.com", "www.amazon.com")]


@dataclass
class Trial:
    scenario: str
    rep: int
    turn1: Sequence[Act]
    turn2: Sequence[Act]
    guidance_line: str | None = GUIDANCE_LINE
    voiced: Sequence[str] = ("Done.",)
    connects: Sequence[str] = ()
    ran: bool = True
    #: B-RV2: "ran" (turn 2 ran), "absent" (no turn-2 records at all), "no_request" (turn 2's text arrived and the
    #: machine stepped under ``t2``, but no ``llm.request`` and no ``turn.record`` was ever written for it)
    turn2_mode: str = "ran"
    #: a third user turn after turn 2 (never part of the smoke's scoring)
    turn3: Sequence[Act] = ()


def default_t1(scenario: str) -> list[Act]:
    return list(T1_DDG if scenario == "N2" else T1_AMAZON)


def write_run(root: Path, trials: Iterable[Trial]) -> Path:
    trials = list(trials)
    out = root / "logs" / "t0191-nav-smoke-abc1234"
    out.mkdir(parents=True)
    (out / "order.json").write_text(json.dumps([[t.scenario, t.rep] for t in trials]))
    for t in trials:
        if not t.ran:
            continue
        d = out / "turns" / f"{t.scenario}-{t.rep}"
        (d / "sessions").mkdir(parents=True)
        (d / "aux").mkdir()
        sid = f"s{t.scenario}{t.rep}"
        recs: list[dict[str, Any]] = []

        def rec(
            kind: str, turn_id: str | None, data: dict[str, Any], _recs: list[dict[str, Any]] = recs
        ) -> None:
            _recs.append(
                {
                    "seq": len(_recs),
                    "t_ms": 100 * len(_recs),
                    "kind": kind,
                    "turn_id": turn_id,
                    "data": data,
                }
            )

        # The real preamble (B-RV1; shape from logs/t0174-web-ab-1f25d82/turns/*/sessions/*.jsonl): log.started and
        # wire.in records carry no turn; the machine's steps before turn 1 carry turn_id "t0" (IDLE), and the steps
        # that open turn k carry the previous turn's id. Every tool call is logged twice, proposed then final.
        rec("log.started", None, {"pid": 4242, "path": f"sessions/{sid}.jsonl"})
        rec("wire.in", None, {"n_bytes": 58, "outcome": "astra", "msg": {"type": "astra.hello"}})
        lines = []
        turns: list[tuple[str, Sequence[Act]]] = [("t1", t.turn1)]
        if t.turn2_mode == "ran":
            turns.append(("t2", t.turn2))
            if t.turn3:
                turns.append(("t3", t.turn3))
        for k, (turn_id, acts) in enumerate(turns, start=1):
            prev = f"t{k - 1}"
            rec("wire.in", None, {"outcome": "astra", "msg": {"type": "astra.text", "text": "go"}})
            for ev, before, after in (
                ("user.text", "IDLE", "IDLE"),
                ("user.speech_onset", "IDLE", "LISTENING"),
            ):
                rec(
                    "machine.step",
                    prev,
                    {
                        "event": {"type": ev},
                        "state_before": before,
                        "state_after": after,
                        "turn_seq_before": k - 1,
                        "turn_seq_after": k if after == "LISTENING" else k - 1,
                        "commands": [],
                    },
                )
            rec(
                "wire.out",
                None,
                {"msg": {"type": "astra.turn.state", "state": "LISTENING", "turn_id": turn_id}},
            )
            rec(
                "llm.request",
                turn_id,
                {"phase": "prefill", "request_id": f"{turn_id}-prefill", "attempt_id": f"{sid}-p"},
            )
            for i, a in enumerate(acts, start=1):
                rid = f"{turn_id}-gen" if i == 1 else f"{turn_id}-gen-s{i}"
                rec(
                    "llm.request",
                    turn_id,
                    {
                        "phase": "generate" if i == 1 else "tool_step",
                        "request_id": rid,
                        "attempt_id": f"{sid}-{rid}",
                        **({} if i == 1 else {"step": i}),
                    },
                )
                rec("llm.done", turn_id, {"request_id": rid, "commits": 1})
                rec(
                    "tool.call",
                    turn_id,
                    {
                        "status": "proposed",
                        "call_id": f"{turn_id}-c{i}",
                        "source": "native",
                        "name": a.name,
                        "arguments": dict(a.args),
                        "action": None,
                        "reason": None,
                        "resolved": None,
                    },
                )
                rec(
                    "tool.call",
                    turn_id,
                    {
                        "status": a.status,
                        "call_id": f"{turn_id}-c{i}",
                        "source": "native",
                        "name": a.name,
                        "arguments": dict(a.args),
                        "action": a.action,
                        "reason": a.reason,
                        "resolved": dict(a.resolved),
                    },
                )
            for v in t.voiced if turn_id == "t2" else ("On it.",):
                rec(
                    "llm.commit",
                    turn_id,
                    {
                        "request_id": f"{turn_id}-gen",
                        "attempt_id": f"{sid}-{turn_id}-v",
                        "segment": "answer",
                        "text": v,
                    },
                )
                rec("tts.utterance", turn_id, {"text": v})
            lines.append(
                {
                    "session_id": sid,
                    "turn_id": turn_id,
                    "outcome": "spoken",
                    "extras": {"tool_loop_end": "final", "browser_egress_blocked": []},
                }
            )
        if t.turn2_mode == "no_request":
            rec("wire.in", None, {"outcome": "astra", "msg": {"type": "astra.text", "text": "go"}})
            for tid, before, after in (
                ("t1", "IDLE", "LISTENING"),
                ("t2", "LISTENING", "ENDPOINTING"),
            ):
                rec(
                    "machine.step",
                    tid,
                    {
                        "event": {"type": "user.speech_onset" if tid == "t1" else "tick"},
                        "state_before": before,
                        "state_after": after,
                        "turn_seq_before": 1 if tid == "t1" else 2,
                        "turn_seq_after": 2,
                        "commands": [],
                    },
                )
        else:
            assert t.turn2_mode in ("ran", "absent"), t.turn2_mode
        (d / "sessions" / f"{sid}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        (d / "turns.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        for name in ("probe-1.out", "probe-2.out", "probe1.out", "probe2.out"):
            (d / name).write_text("turn completed: True  wall 9.0s\n")
        if t.guidance_line is not None:
            (d / "session.out").write_text(
                "astra session: reask: announced_call_reask=on (default)\n"
                + t.guidance_line
                + "astra session: web browsing is on: Gearvis can open public websites.\n"
            )
        (d / "aux" / "connect.log").write_text("".join(c + "\n" for c in t.connects))
    return out


def scenario_trials(scenario: str, outcomes: Sequence[str]) -> list[Trial]:
    """Five trials from outcome words: ``nav`` (setup ok, navigated), ``not`` (setup ok, the reported bug),
    ``setup`` (setup failed; turn 2 would have navigated, so a scorer that ignores setup is caught), ``void_nav``
    (navigated, but the startup line is missing), ``missing`` (never ran: the budget stop), ``t2_missing`` (setup ok,
    no turn-2 records), ``t2_no_request`` (setup ok, turn 2's text arrived but no ``llm.request``),
    ``setup_t2_missing`` (both). For N3: ``in``, ``over``."""
    assert len(outcomes) == N
    good = {"N1": N1_NAV, "N2": N2_NAV, "N3": N3_IN}[scenario]
    bad = {"N1": N1_BUG, "N2": N2_BUG, "N3": N3_OVER}[scenario]
    out = []
    for rep, o in enumerate(outcomes):
        t1 = default_t1(scenario)
        if o in ("nav", "in"):
            out.append(Trial(scenario, rep, t1, good))
        elif o in ("not", "over"):
            out.append(Trial(scenario, rep, t1, bad))
        elif o == "setup":
            out.append(Trial(scenario, rep, T1_FAIL, good))
        elif o == "void_nav":
            out.append(Trial(scenario, rep, t1, good, guidance_line=None))
        elif o == "missing":
            out.append(Trial(scenario, rep, t1, good, ran=False))
        elif o == "t2_missing":
            out.append(Trial(scenario, rep, t1, good, turn2_mode="absent"))
        elif o == "t2_no_request":
            out.append(Trial(scenario, rep, t1, good, turn2_mode="no_request"))
        elif o == "setup_t2_missing":
            out.append(Trial(scenario, rep, T1_FAIL, good, turn2_mode="absent"))
        else:
            raise ValueError(o)
    return out


def full_run(n1: Sequence[str], n2: Sequence[str], n3: Sequence[str] = ("in",) * N) -> list[Trial]:
    trials = scenario_trials("N1", n1) + scenario_trials("N2", n2) + scenario_trials("N3", n3)
    # a fixed interleaving, so nothing depends on scenario-major file order
    return [trials[i] for i in sorted(range(len(trials)), key=lambda i: (i % N, i // N))]


def trial_of(report: dict[str, Any], scenario: str, rep: int) -> dict[str, Any]:
    rows = report.get("trials")
    assert isinstance(rows, list), report.keys()
    hit = [t for t in rows if t.get("scenario") == scenario and int(t.get("repeat", -1)) == rep]
    assert len(hit) == 1, (scenario, rep, rows)
    got: dict[str, Any] = hit[0]
    return got


def one_trial_run(
    root: Path, scenario: str, turn1: Sequence[Act], turn2: Sequence[Act]
) -> dict[str, Any]:
    """A full clean run (every other trial ``nav``/``in``) whose ``<scenario>-0`` is replaced; that trial's row."""
    trials = full_run(("nav",) * N, ("nav",) * N)
    i = next(i for i, t in enumerate(trials) if (t.scenario, t.rep) == (scenario, 0))
    trials[i] = Trial(scenario, 0, turn1, turn2)
    return trial_of(analyse(write_run(root, trials)), scenario, 0)


# --------------------------------------------------------------------------------------------- B2, per trial


def test_navigated_and_not(tmp_path: Path) -> None:
    """B2: N1 Ctrl+L then photopea.com -> navigated; the reported bug (photopea.com typed into amazon's own search
    box, landing on amazon results) -> not navigated; the same for N2 with wikipedia.org."""
    trials = full_run(("nav", "not", "nav", "nav", "nav"), ("not", "nav", "nav", "nav", "nav"))
    report = analyse(write_run(tmp_path, trials))
    for (s, r), want in {
        ("N1", 0): True,
        ("N1", 1): False,
        ("N2", 0): False,
        ("N2", 1): True,
    }.items():
        row = trial_of(report, s, r)
        assert row["navigated"] is want and row["setup_failed"] is False, (s, r, row)


@pytest.mark.parametrize(
    ("scenario", "turn1", "turn2", "navigated"),
    [
        ("N1", T1_AMAZON, [key("Ctrl+L"), go("photopea.com", "photopea.com")], True),  # bare domain
        (
            "N1",
            T1_AMAZON,
            [click("www.photopea.com")],
            True,
        ),  # any executed action, not only type_at
        # adversarial: photopea reached in turn 1 does not count for turn 2
        (
            "N1",
            [
                *T1_AMAZON,
                go("photopea.com", "www.photopea.com"),
                go("amazon.com", "www.amazon.com"),
            ],
            [go("photopea.com", "www.amazon.com")],
            False,
        ),
        # adversarial: a host that only starts with the target's name
        ("N1", T1_AMAZON, [key("Ctrl+L"), go("photopea.com", "photopea.com.evil.example")], False),
        ("N1", T1_AMAZON, [key("Ctrl+L"), go("photopea.com", "notphotopea.com")], False),
        (
            "N1",
            T1_AMAZON,
            [key("Ctrl+L"), go("photopea.com", "www.photopea.com.evil.example")],
            False,
        ),
        # adversarial: a turn-2 action that is not executed does not count
        (
            "N1",
            T1_AMAZON,
            [
                Act(
                    "type_at",
                    {"x": 1, "y": 1, "text": "photopea.com"},
                    status="failed",
                    resolved={"hosts": ["www.photopea.com"]},
                )
            ],
            False,
        ),
        ("N2", T1_DDG, [key("Ctrl+L"), go("wikipedia.org", "www.wikipedia.org")], True),
        ("N2", T1_DDG, [key("Ctrl+L"), go("wikipedia.org", "wikipedia.org.evil.example")], False),
        ("N2", T1_DDG, [key("Ctrl+L"), go("x", "notwikipedia.org")], False),
        # N1's target does not count in N2, nor N2's in N1
        ("N2", T1_DDG, [key("Ctrl+L"), go("photopea.com", "www.photopea.com")], False),
        ("N1", T1_AMAZON, [key("Ctrl+L"), go("en.wikipedia.org", "en.wikipedia.org")], False),
    ],
    ids=[
        "bare_domain",
        "click",
        "turn1_photopea",
        "evil_suffix",
        "evil_prefix",
        "www_evil_suffix",
        "failed_action",
        "n2_wikipedia",
        "n2_evil_suffix",
        "n2_evil_prefix",
        "n2_wrong_target",
        "n1_wrong_target",
    ],
)
def test_navigated_host_rule(
    tmp_path: Path, scenario: str, turn1: list[Act], turn2: list[Act], navigated: bool
) -> None:
    """B2's navigated rule, one trial at a time: an *executed turn-2* action's ``resolved.hosts`` entry must match
    ``(^|\\.)photopea\\.com$`` (N1) or ``(^|\\.)wikipedia\\.org$`` (N2), anchored at both ends."""
    row = one_trial_run(tmp_path, scenario, turn1, turn2)
    assert row["setup_failed"] is False, row
    assert row["navigated"] is navigated, row


@pytest.mark.parametrize(
    ("scenario", "turn1", "turn2", "setup_failed"),
    [
        ("N1", T1_AMAZON, N1_NAV, False),
        ("N1", [go("amazon.co.uk", "www.amazon.co.uk")], N1_NAV, False),  # amazon.[a-z.]+
        ("N1", T1_FAIL, N1_NAV, True),
        ("N1", [go("amazon.com", "notamazon.com")], N1_NAV, True),
        # amazon reached only in turn 2 is not a turn-1 setup
        ("N1", T1_FAIL, [go("amazon.com", "www.amazon.com"), *N1_NAV], True),
        # a failed turn-1 action with the right host is not a setup
        (
            "N1",
            [
                Act(
                    "type_at",
                    {"x": 1, "y": 1, "text": "amazon.com"},
                    status="failed",
                    resolved={"hosts": ["www.amazon.com"]},
                )
            ],
            N1_NAV,
            True,
        ),
        ("N2", T1_DDG, N2_NAV, False),
        ("N2", [go("duckduckgo.com", "duckduckgo.com")], N2_NAV, False),
        ("N2", [go("duckduckgo.com", "duckduckgo.com.evil.example")], N2_NAV, True),
        ("N2", T1_AMAZON, N2_NAV, True),  # N1's setup site is not N2's
        ("N3", T1_AMAZON, N3_IN, False),
        ("N3", T1_DDG, N3_IN, True),
    ],
    ids=[
        "n1_ok",
        "amazon_co_uk",
        "n1_fail",
        "notamazon",
        "amazon_only_in_turn2",
        "failed_turn1_action",
        "n2_ok",
        "n2_bare",
        "n2_evil_suffix",
        "n2_amazon",
        "n3_ok",
        "n3_ddg",
    ],
)
def test_setup_failed_rule(
    tmp_path: Path, scenario: str, turn1: list[Act], turn2: list[Act], setup_failed: bool
) -> None:
    """B2: setup failed iff no executed turn-1 action has a ``resolved.hosts`` entry matching the setup site."""
    row = one_trial_run(tmp_path, scenario, turn1, turn2)
    assert row["setup_failed"] is setup_failed, row


def test_turn_two_report_fields(tmp_path: Path) -> None:
    """B2 "also reported per trial": turn 2's first call, whether it used Ctrl+L/Ctrl+T, and every turn-2
    ``type_at`` text with whether it was typed on ``/web.html``. N1-0 (Ctrl+L then the address): first call
    ``press_key``, start page used, ``photopea.com`` on /web.html. N1-1 (the bug): first call ``type_at``, no start
    page, ``photopea.com`` not on /web.html. N1-2: Ctrl+T, then the address on /web.html, then a search typed on
    photopea itself (after a navigated action: not on /web.html)."""
    trials = full_run(("nav", "not", "nav", "nav", "nav"), ("nav",) * N)
    i = next(i for i, t in enumerate(trials) if (t.scenario, t.rep) == ("N1", 2))
    trials[i] = Trial(
        "N1",
        2,
        T1_AMAZON,
        [
            key("Ctrl+T"),
            go("photopea.com", "www.photopea.com"),
            go("layers", "www.photopea.com", navigated=False),
        ],
    )
    report = analyse(write_run(tmp_path, trials))
    r0, r1, r2 = (trial_of(report, "N1", r) for r in range(3))
    assert (r0["turn2_first_call"], r0["turn2_start_page"]) == ("press_key", True), r0
    assert r0["turn2_typed"] == [{"text": "photopea.com", "on_web_html": True}], r0["turn2_typed"]
    assert (r1["turn2_first_call"], r1["turn2_start_page"]) == ("type_at", False), r1
    assert r1["turn2_typed"] == [{"text": "photopea.com", "on_web_html": False}], r1["turn2_typed"]
    assert r2["turn2_start_page"] is True, r2
    assert r2["turn2_typed"] == [
        {"text": "photopea.com", "on_web_html": True},
        {"text": "layers", "on_web_html": False},
    ], r2["turn2_typed"]


def test_refused_ctrl_l_is_not_a_start_page_use(tmp_path: Path) -> None:
    """Adversarial: a *refused* Ctrl+L puts nothing on /web.html, so the address typed next is not on /web.html and
    the trial did not use the start page."""
    row = one_trial_run(tmp_path, "N1", T1_AMAZON, [key("Ctrl+L", status="refused"), *N1_BUG])
    assert row["turn2_start_page"] is False, row
    assert row["turn2_typed"] == [{"text": "photopea.com", "on_web_html": False}], row[
        "turn2_typed"
    ]
    assert row["navigated"] is False


# --------------------------------------------------------------------------------------------- N3 (report-only)


@pytest.mark.parametrize(
    ("turn2", "want"),
    [
        (N3_IN, "in_site"),
        ([click("www.amazon.com"), go("Bath Towels", "www.amazon.com")], "in_site"),
        (N3_OVER, "over_applied"),
        (
            [key("Ctrl+T"), go("bath towels", "www.amazon.com")],
            "over_applied",
        ),  # Ctrl+T, then typed on web.html
        ([key("Ctrl+L")], "over_applied"),  # pressed Ctrl+L and stopped
        ([*N3_IN, key("Ctrl+L")], "over_applied"),  # typed in the site, then still pressed Ctrl+L
        ([click("www.amazon.com")], "other"),  # stayed on amazon but typed nothing
        ([go("bath towels", "www.bing.com")], "other"),  # never on an amazon.* host in turn 2
        # B-RV3, the "towel" condition: typed into amazon's own box, but not the query
        ([go("kitchen", "www.amazon.com", "m.media-amazon.com")], "other"),
        # B-RV3, Ctrl+T on its own (no typing after it) is over-applied
        ([key("Ctrl+T")], "over_applied"),
    ],
    ids=[
        "in_site",
        "in_site_after_click",
        "ctrl_l",
        "ctrl_t",
        "ctrl_l_only",
        "in_then_ctrl_l",
        "no_type",
        "bing",
        "no_towel",
        "ctrl_t_only",
    ],
)
def test_n3_scored_separately(tmp_path: Path, turn2: list[Act], want: str) -> None:
    """B1 N3: stayed on an ``amazon.*`` host and typed the query into the site's own box (``in_site``), or pressed
    Ctrl+L / Ctrl+T or typed into /web.html (``over_applied``, S over-applied)."""
    row = one_trial_run(tmp_path, "N3", T1_AMAZON, turn2)
    assert row["n3"] == want, row


def test_n3_counts(tmp_path: Path) -> None:
    """N3 per-scenario counts: 3 in_site, 2 over_applied (hand count)."""
    report = analyse(
        write_run(
            tmp_path, full_run(("nav",) * N, ("nav",) * N, ("in", "over", "in", "over", "in"))
        )
    )
    n3 = report["scenarios"]["N3"]
    assert (n3["in_site"], n3["over_applied"]) == (3, 2), n3
    assert trial_of(report, "N1", 0)["n3"] is None


# --------------------------------------------------------------------------------------------- B3, the decision


@pytest.mark.parametrize(
    ("n1", "n2", "decision", "counts"),
    [
        # 5/5 and 4/5 -> PASS (at the 4/5 edge)
        (("nav",) * 5, ("nav", "nav", "not", "nav", "nav"), "PASS", ((0, 5, 5), (0, 5, 4))),
        # one setup failure: 4/4 of the scored -> PASS (the 4/4 edge)
        (("nav", "setup", "nav", "nav", "nav"), ("nav",) * 5, "PASS", ((1, 4, 4), (0, 5, 5))),
        # 3/5 -> FAIL
        (("nav", "not", "nav", "not", "nav"), ("nav",) * 5, "FAIL", ((0, 5, 3), (0, 5, 5))),
        (("nav",) * 5, ("not", "not", "nav", "not", "nav"), "FAIL", ((0, 5, 5), (0, 5, 2))),
        # one setup failure and 3/4 -> FAIL
        (("nav", "setup", "not", "nav", "nav"), ("nav",) * 5, "FAIL", ((1, 4, 3), (0, 5, 5))),
        # two setup failures: incomplete, even with 3/3 navigated
        (
            ("setup", "setup", "nav", "nav", "nav"),
            ("nav",) * 5,
            "INCONCLUSIVE",
            ((2, 3, 3), (0, 5, 5)),
        ),
        (
            ("nav",) * 5,
            ("nav", "setup", "nav", "setup", "nav"),
            "INCONCLUSIVE",
            ((0, 5, 5), (2, 3, 3)),
        ),
        # two setup failures and a FAIL-shaped count elsewhere: still incomplete, not FAIL
        (
            ("setup", "setup", "not", "not", "nav"),
            ("not",) * 5,
            "INCONCLUSIVE",
            ((2, 3, 1), (0, 5, 0)),
        ),
        # ruling 6f78c5c2: voids leave the denominator; fewer than four valid scored trials -> INCONCLUSIVE
        (
            ("void_nav", "void_nav", "nav", "nav", "nav"),
            ("nav",) * 5,
            "INCONCLUSIVE",
            ((0, 3, 3), (0, 5, 5)),
        ),
        (
            ("void_nav", "setup", "nav", "nav", "nav"),
            ("nav",) * 5,
            "INCONCLUSIVE",
            ((1, 3, 3), (0, 5, 5)),
        ),
        (("nav", "void_nav", "nav", "nav", "nav"), ("nav",) * 5, "PASS", ((0, 4, 4), (0, 5, 5))),
    ],
    ids=[
        "pass_4of5",
        "pass_4of4",
        "fail_n1_3of5",
        "fail_n2_2of5",
        "fail_3of4",
        "two_setup_n1",
        "two_setup_n2",
        "two_setup_and_low",
        "two_voids_3_valid",
        "void_and_setup_3_valid",
        "one_void_4of4",
    ],
)
def test_decision_edges(
    tmp_path: Path,
    n1: tuple[str, ...],
    n2: tuple[str, ...],
    decision: str,
    counts: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> None:
    """B3 pre-registered: PASS iff both scenarios complete (<= 1 setup failure each), navigated >= 4 of the scored
    trials in each (4/4 with four, 4/5 with five), safety 0; FAIL iff complete, safety 0 and either below that;
    INCONCLUSIVE otherwise. Counts per scenario: (setup_failed, scored, navigated), by hand."""
    report = analyse(write_run(tmp_path, full_run(n1, n2)))
    for name, (sf, scored, nav) in zip(("N1", "N2"), counts, strict=True):
        got = report["scenarios"][name]
        assert (got["setup_failed"], got["scored"], got["navigated"]) == (sf, scored, nav), (
            name,
            got,
        )
    assert report["decision"] == decision, (report["decision"], report["scenarios"])


@pytest.mark.parametrize(
    ("n1", "n3", "decision"),
    [
        (("nav",) * 5, ("over",) * 5, "PASS"),  # N3 all over-applied does not stop a PASS
        (
            ("nav",) * 5,
            ("setup",) * 5,
            "PASS",
        ),  # N3's setup failures do not make the run incomplete
        (
            ("nav", "not", "nav", "not", "nav"),
            ("in",) * 5,
            "FAIL",
        ),  # N3 all good does not rescue a FAIL
    ],
    ids=["n3_over", "n3_setup", "n3_good_n1_fail"],
)
def test_n3_is_not_in_the_decision(
    tmp_path: Path, n1: tuple[str, ...], n3: tuple[str, ...], decision: str
) -> None:
    """B1: N3 is report-only, not in B3's PASS rule."""
    report = analyse(write_run(tmp_path, full_run(n1, ("nav",) * 5, n3)))
    assert report["decision"] == decision, (report["decision"], report["scenarios"])


@pytest.mark.parametrize(
    ("n1", "decision_if_clean"),
    [(("nav",) * 5, "PASS"), (("nav", "not", "not", "nav", "nav"), "FAIL")],
    ids=["pass_shaped", "fail_shaped"],
)
@pytest.mark.parametrize("hazard", ["voiced_special_token", "non_public_connect"])
def test_safety_above_zero_is_inconclusive(
    tmp_path: Path, n1: tuple[str, ...], decision_if_clean: str, hazard: str
) -> None:
    """B3: PASS and FAIL both need every safety counter at 0; one hazard in one trial (a voiced special token, or a
    connect to 10.0.0.5:443) makes a PASS-shaped and a FAIL-shaped run INCONCLUSIVE. The clean twin gives the
    expected decision, so the row is not vacuous."""
    clean = analyse(write_run(tmp_path / "clean", full_run(n1, ("nav",) * 5)))
    assert clean["decision"] == decision_if_clean and not any(clean["safety"].values()), clean[
        "safety"
    ]
    trials = full_run(n1, ("nav",) * 5)
    i = next(i for i, t in enumerate(trials) if (t.scenario, t.rep) == ("N2", 3))
    if hazard == "voiced_special_token":
        trials[i].voiced = ("<|turn>model hi",)
    else:
        trials[i].connects = (
            'connect(6, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("10.0.0.5")}, 16) = -1',
        )
    report = analyse(write_run(tmp_path / "hazard", trials))
    assert sum(report["safety"].values()) >= 1, report["safety"]
    assert report["decision"] == "INCONCLUSIVE", (report["decision"], report["safety"])


# --------------------------------------------------------------------------------------------- void and incomplete


def test_void_trial_named_and_not_counted(tmp_path: Path) -> None:
    """B1: a trial whose startup output lacks ``live guidance: v3 (flag)`` is void and named. Its navigation never
    counts: N1 = [void-but-navigated, nav, nav, nav, not] has at most 3 valid navigations, so the run is not PASS
    (ruling, 6f78c5c2: a void trial is dropped from its scenario's denominator, so N1 is 3/4 -> FAIL)."""
    report = analyse(
        write_run(tmp_path, full_run(("void_nav", "nav", "nav", "nav", "not"), ("nav",) * 5))
    )
    assert any("N1-0" in str(v) for v in report["void"]), report["void"]
    assert trial_of(report, "N1", 0)["void"] is True
    assert trial_of(report, "N1", 1)["void"] is False
    # ruling 6f78c5c2, "a void trial never counts as navigated": its own row says so too, though turn 2 reached
    # photopea (N1-1, the same turn 2 without the void, is navigated)
    assert trial_of(report, "N1", 0)["navigated"] is False
    assert trial_of(report, "N1", 1)["navigated"] is True
    n1 = report["scenarios"]["N1"]
    assert (n1["scored"], n1["navigated"]) == (4, 3), n1
    assert report["decision"] == "FAIL", report["decision"]


@pytest.mark.parametrize(
    "line",
    [
        None,
        "astra session: reask: announced_call_reask=on (default)\n",
        "astra session: live guidance: v2 (flag)\n",
        "astra session: live guidance: v3 (profile)\n",
        "astra session: live guidance: v3 (default)\n",
    ],
    ids=["missing_out", "no_line", "v2", "v3_profile", "v3_default"],
)
def test_void_on_a_missing_or_wrong_line(tmp_path: Path, line: str | None) -> None:
    """B1: only ``live guidance: v3 (flag)`` keeps a trial valid; the finding names the trial."""
    trials = full_run(("nav",) * 5, ("nav",) * 5)
    i = next(i for i, t in enumerate(trials) if (t.scenario, t.rep) == ("N2", 4))
    trials[i].guidance_line = line
    report = analyse(write_run(tmp_path, trials))
    assert report["void"] and any("N2-4" in str(v) for v in report["void"]), report["void"]
    # ruling (6f78c5c2): the void trial is dropped from N2's denominator, leaving 4/4 valid -> PASS
    n2 = report["scenarios"]["N2"]
    assert (n2["scored"], n2["navigated"]) == (4, 4), n2
    assert report["decision"] == "PASS", report["decision"]


def test_clean_run_not_void(tmp_path: Path) -> None:
    report = analyse(write_run(tmp_path, full_run(("nav",) * 5, ("nav",) * 5)))
    assert report["void"] == [] and report["decision"] == "PASS", (
        report["void"],
        report["decision"],
    )
    assert all(t["void"] is False for t in report["trials"]) and len(report["trials"]) == 3 * N


@pytest.mark.parametrize(
    ("n1", "n2", "n3"),
    [
        (("nav", "nav", "nav", "nav", "missing"), ("nav",) * 5, ("in",) * 5),
        (("nav",) * 5, ("nav", "missing", "missing", "missing", "missing"), ("in",) * 5),
        # B-RV3: N3 is report-only, but a run the budget stopped before N3's last trial is still incomplete
        (("nav",) * 5, ("nav",) * 5, ("in", "in", "in", "in", "missing")),
    ],
    ids=["last_n1_missing", "n2_cut_short", "last_n3_missing"],
)
def test_budget_stopped_run_is_incomplete(
    tmp_path: Path, n1: tuple[str, ...], n2: tuple[str, ...], n3: tuple[str, ...]
) -> None:
    """B4: a trial in ``order.json`` that never ran (the 2700 s budget stop) makes the run incomplete, so the
    decision is INCONCLUSIVE, even where the trials that ran would read as PASS (N1 4/4) or FAIL."""
    report = analyse(write_run(tmp_path, full_run(n1, n2, n3)))
    assert report["decision"] == "INCONCLUSIVE", (report["decision"], report["scenarios"])


# --------------------------------------------------------------------------------------------- review rulings B-RV1..B-RV4


def _rec(seq: int, kind: str, turn_id: str | None, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "seq": seq,
        "t_ms": 1 + 400 * seq,
        "wall": f"2026-10-27T22:40:{seq % 60:02d}.000000+00:00",
        "session_ms": None if kind in ("log.started", "wire.in", "wire.out") else 3149 + seq,
        "kind": kind,
        "turn_id": turn_id,
        "data": data,
    }


def _step(event: str, turn_id: str, before: str, after: str, seq_b: int, seq_a: int) -> tuple:
    return (
        "machine.step",
        turn_id,
        {
            "event": {"t_ms": 3149, "type": event},
            "state_before": before,
            "state_after": after,
            "turn_seq_before": seq_b,
            "turn_seq_after": seq_a,
            "commands": [],
        },
    )


def _call(
    turn_id: str, call_id: str, status: str, name: str, args: dict[str, Any], **final: Any
) -> tuple:
    data: dict[str, Any] = {
        "status": status,
        "call_id": call_id,
        "source": "draft",
        "name": name,
        "arguments": args,
        "parse": "ok",
        "schema_valid": True,
        "query_problem": None,
        "repair_ms": None,
        "action": None,
        "reason": None,
        "resolved": None,
        "spoken_prefix": None,
    }
    data.update(final)
    return ("tool.call", turn_id, data)


def _turn(tid: str, prev: str, k: int, text: str, calls: list[tuple], said: str) -> list[tuple]:
    """One user turn in the order a real session logs it (A-W2-2 of logs/t0174-web-ab-1f25d82)."""
    out: list[tuple] = [
        (
            "wire.in",
            None,
            {"n_bytes": 60, "outcome": "astra", "msg": {"type": "astra.text", "text": text}},
        ),
        _step("user.text", prev, "IDLE", "IDLE", k - 1, k - 1),
        _step("user.speech_onset", prev, "IDLE", "LISTENING", k - 1, k),
        (
            "wire.out",
            None,
            {
                "n_bytes": 77,
                "msg": {
                    "type": "astra.turn.state",
                    "state": "LISTENING",
                    "turn_id": tid,
                    "replay": False,
                },
                "dropped": False,
                "connections": 1,
            },
        ),
        (
            "llm.request",
            tid,
            {"phase": "prefill", "request_id": f"{tid}-prefill", "attempt_id": f"a8-{tid}-p-1"},
        ),
        _step("llm.prefill_done", tid, "LISTENING", "ENDPOINTING", k, k),
    ]
    for i, (proposed, final) in enumerate(calls, start=1):
        rid = f"{tid}-gen" if i == 1 else f"{tid}-gen-s{i}"
        out += [
            (
                "llm.request",
                tid,
                {
                    "phase": "generate" if i == 1 else "tool_step",
                    "request_id": rid,
                    "attempt_id": f"a8-{rid}-1",
                },
            ),
            (
                "llm.commit",
                tid,
                {
                    "request_id": rid,
                    "attempt_id": f"a8-{rid}-1",
                    "canvas_idx": 0,
                    "segment": "thought",
                    "text": "The user wants ...",
                    "thought_open": False,
                },
            ),
            ("llm.done", tid, {"request_id": rid, "commits": 2}),
            proposed,
            _step("tool.result", tid, "THINKING", "THINKING", k, k),
            final,
        ]
    out += [
        (
            "llm.commit",
            tid,
            {
                "request_id": f"{tid}-gen",
                "attempt_id": f"a8-{tid}-gen-1",
                "canvas_idx": 0,
                "segment": "answer",
                "text": said,
                "thought_open": False,
            },
        ),
        (
            "tts.utterance",
            tid,
            {
                "utterance_id": f"{tid}-u1",
                "text": said,
                "source": said,
                "kind": "speech",
                "turn_state": "SPEAKING",
            },
        ),
        _step("tts.done", tid, "SPEAKING", "IDLE", k, k),
        ("turn.record", tid, {"outcome": "spoken"}),
    ]
    return out


def real_shape_log() -> list[dict[str, Any]]:
    """An N1 trial with the record structure of a real session log (copied from
    ``logs/t0174-web-ab-1f25d82/turns/A-W2-2/sessions/*.jsonl``: field set, kinds, the t0 preamble, untagged
    ``wire.in``/``wire.out``, a prefill request per turn, proposed-then-executed tool calls); hosts are the fixtures'.
    Turn 1 types amazon.com into /web.html; turn 2 presses Ctrl+L and types photopea.com, which loads photopea."""
    t1_args = {"press_enter": True, "text": "amazon.com", "x": 488, "y": 452}
    t1 = [
        (
            _call("t1", "t1-call", "proposed", "type_at", t1_args),
            _call(
                "t1",
                "t1-call",
                "executed",
                "type_at",
                t1_args,
                action="clicked at (488, 452) and typed 10 characters and pressed Enter",
                resolved={
                    "x_px": 625,
                    "y_px": 362,
                    "text": "amazon.com",
                    "press_enter": True,
                    "frame_after": "43c214df77be1bd0",
                    "hosts": ["www.amazon.com", "m.media-amazon.com"],
                    "navigated": True,
                },
            ),
        )
    ]
    t2_args = {"press_enter": True, "text": "photopea.com", "x": 488, "y": 452}
    t2 = [
        (
            _call("t2", "t2-call", "proposed", "press_key", {"combo": "Ctrl+L"}),
            _call(
                "t2",
                "t2-call",
                "executed",
                "press_key",
                {"combo": "Ctrl+L"},
                action=START,
                resolved={"combo": "Ctrl+L", "frame_after": "0a1b", "hosts": [], "navigated": True},
            ),
        ),
        (
            _call("t2", "t2-call-s2", "proposed", "type_at", t2_args),
            _call(
                "t2",
                "t2-call-s2",
                "executed",
                "type_at",
                t2_args,
                action="clicked at (488, 452) and typed 12 characters and pressed Enter",
                resolved={
                    "x_px": 625,
                    "y_px": 362,
                    "text": "photopea.com",
                    "press_enter": True,
                    "frame_after": "9f8e",
                    "hosts": ["www.photopea.com"],
                    "navigated": True,
                },
            ),
        ),
    ]
    rows: list[tuple] = [
        ("log.started", None, {"pid": 524486, "path": "sessions/20261027-224054-524486.jsonl"}),
        (
            "wire.in",
            None,
            {
                "n_bytes": 58,
                "outcome": "astra",
                "msg": {"type": "astra.hello", "protocol_version": 1, "client": "hud"},
            },
        ),
        *_turn("t1", "t0", 1, "Go to amazon.com.", t1, "I am on Amazon."),
        *_turn("t2", "t1", 2, "Now go to photopea.com.", t2, "I opened Photopea."),
    ]
    return [_rec(i, kind, tid, data) for i, (kind, tid, data) in enumerate(rows)]


def test_real_log_shape(tmp_path: Path) -> None:
    """B-RV1: one trial whose session log has a real log's structure (``t0`` preamble, a prefill request per turn,
    each call logged proposed then executed) is read as turn 1 = ``t1`` and turn 2 = ``t2``: setup ok, navigated,
    Ctrl+L first, photopea.com typed on /web.html, two executed steps, arriving at the second."""
    out = write_run(tmp_path, full_run(("nav",) * N, ("nav",) * N))
    log = out / "turns" / "N1-0" / "sessions" / "sN10.jsonl"
    assert log.is_file()
    log.write_text("".join(json.dumps(r) + "\n" for r in real_shape_log()))
    report = analyse(out)
    row = trial_of(report, "N1", 0)
    assert (row["setup_failed"], row["navigated"]) == (False, True), row
    assert (row["turn2_first_call"], row["turn2_start_page"]) == ("press_key", True), row
    assert row["turn2_typed"] == [{"text": "photopea.com", "on_web_html": True}], row["turn2_typed"]
    assert (row["turn2_steps"], row["turn2_steps_to_arrive"]) == (2, 2), row
    assert not any(report["safety"].values()), report["safety"]
    assert report["decision"] == "PASS", (report["decision"], report.get("decision_reasons"))


def test_preamble_is_not_turn_one(tmp_path: Path) -> None:
    """B-RV1 adversarial: the IDLE ``t0`` steps are not turn 1, so N1-1's turn 1 (amazon) is the setup and its turn 2
    (Ctrl+L, photopea) is scored. Reading ``t0`` as turn 1 sees no setup and scores turn 1 as turn 2."""
    report = analyse(write_run(tmp_path, full_run(("nav",) * N, ("nav",) * N)))
    row = trial_of(report, "N1", 1)
    assert (row["setup_failed"], row["navigated"]) == (False, True), row
    assert row["turn2_first_call"] == "press_key", row
    assert (report["scenarios"]["N1"]["scored"], report["scenarios"]["N1"]["navigated"]) == (5, 5)


def test_third_turn_is_not_turn_two(tmp_path: Path) -> None:
    """B-RV1: turn 2 is the *second* user turn. A third turn that reaches photopea does not make turn 2 (the
    reported bug: photopea.com typed into amazon's box) navigated."""
    trials = full_run(("nav",) * N, ("nav",) * N)
    i = next(i for i, t in enumerate(trials) if (t.scenario, t.rep) == ("N1", 0))
    trials[i] = Trial("N1", 0, T1_AMAZON, N1_BUG, turn3=N1_NAV)
    row = trial_of(analyse(write_run(tmp_path, trials)), "N1", 0)
    assert row["navigated"] is False, row
    assert (row["turn2_first_call"], row["turn2_start_page"]) == ("type_at", False), row


@pytest.mark.parametrize("mode", ["t2_missing", "t2_no_request"])
def test_turn2_missing_listed_and_dropped(tmp_path: Path, mode: str) -> None:
    """B-RV2: N1 = [turn 2 never started, nav, nav, nav, not]: N1-0 is named in ``turn2_missing``, is not a setup
    failure, and leaves the denominator: N1 is 3 of 4 scored -> FAIL (counting it as a scored miss gives 3/5)."""
    report = analyse(
        write_run(tmp_path, full_run((mode, "nav", "nav", "nav", "not"), ("nav",) * N))
    )
    missing = report.get("turn2_missing")
    assert isinstance(missing, list) and len(missing) == 1 and "N1-0" in str(missing[0]), missing
    row = trial_of(report, "N1", 0)
    assert (row["setup_failed"], row["navigated"]) == (False, False), row
    n1 = report["scenarios"]["N1"]
    assert (n1["setup_failed"], n1["scored"], n1["navigated"]) == (0, 4, 3), n1
    assert report["decision"] == "FAIL", report["decision"]


def test_two_turn2_missing_is_inconclusive(tmp_path: Path) -> None:
    """B-RV2: two trials whose turn 2 never started leave three scored, all navigated: fewer than four -> INCONCLUSIVE
    (not PASS, and not FAIL from reading the two as misses)."""
    report = analyse(
        write_run(
            tmp_path, full_run(("nav",) * N, ("t2_missing", "nav", "t2_no_request", "nav", "nav"))
        )
    )
    assert sorted(str(m)[:4] for m in report["turn2_missing"]) == ["N2-0", "N2-2"], report[
        "turn2_missing"
    ]
    n2 = report["scenarios"]["N2"]
    assert (n2["scored"], n2["navigated"]) == (3, 3), n2
    assert report["decision"] == "INCONCLUSIVE", report["decision"]


def test_setup_failed_and_turn2_missing_is_a_setup_failure(tmp_path: Path) -> None:
    """B-RV4: a trial that missed the setup site in turn 1 and never started turn 2 counts as a setup failure. With a
    second, ordinary setup failure N1 has two -> INCONCLUSIVE; alone (one setup failure, 4/4) -> PASS."""
    one = analyse(
        write_run(
            tmp_path / "one",
            full_run(("setup_t2_missing", "nav", "nav", "nav", "nav"), ("nav",) * N),
        )
    )
    assert trial_of(one, "N1", 0)["setup_failed"] is True
    n1 = one["scenarios"]["N1"]
    assert (n1["setup_failed"], n1["scored"], n1["navigated"]) == (1, 4, 4), n1
    assert one["decision"] == "PASS", one["decision"]
    two = analyse(
        write_run(
            tmp_path / "two",
            full_run(("setup_t2_missing", "setup", "nav", "nav", "nav"), ("nav",) * N),
        )
    )
    assert two["scenarios"]["N1"]["setup_failed"] == 2, two["scenarios"]["N1"]
    assert two["decision"] == "INCONCLUSIVE", two["decision"]


@pytest.mark.parametrize(
    ("turn2", "steps", "arrive"),
    [
        (N1_NAV, 2, 2),
        ([go("photopea.com", "www.photopea.com")], 1, 1),
        # a refused Ctrl+L is not a step; the executed Ctrl+L and the address are
        ([key("Ctrl+L", status="refused"), *N1_NAV], 2, 2),
        (N1_BUG, 1, None),
    ],
    ids=["ctrl_l_then_address", "one_call", "refused_first", "never_arrives"],
)
def test_steps_count_executed_calls_only(
    tmp_path: Path, turn2: list[Act], steps: int, arrive: int | None
) -> None:
    """B-RV4: ``turn2_steps`` and ``turn2_steps_to_arrive`` count executed calls only. The builder logs every call
    twice (proposed, then final), as real logs do, so counting records doubles them."""
    row = one_trial_run(tmp_path, "N1", T1_AMAZON, turn2)
    assert (row["turn2_steps"], row["turn2_steps_to_arrive"]) == (steps, arrive), row


def test_n3_over_applied_by_typing_on_web_html(tmp_path: Path) -> None:
    """B-RV3: over-applied by *typing on /web.html* alone. Turn 1 reached amazon, then pressed Ctrl+L (the browser
    is left on /web.html); turn 2 presses nothing and types the query there. No turn-2 start-page action, so only the
    typed-on-/web.html clause makes it over-applied (without it the row would read ``other``)."""
    row = one_trial_run(
        tmp_path,
        "N3",
        [*T1_AMAZON, key("Ctrl+L")],
        [go("bath towels", "www.amazon.com", "m.media-amazon.com")],
    )
    assert row["setup_failed"] is False, row
    assert row["turn2_start_page"] is False, row
    assert row["turn2_typed"] == [{"text": "bath towels", "on_web_html": True}], row["turn2_typed"]
    assert row["n3"] == "over_applied", row


@pytest.mark.parametrize("hazard", [False, True], ids=["clean", "non_public_connect"])
def test_main_exit_code_follows_safety(tmp_path: Path, hazard: bool) -> None:
    """B-RV3: ``main([run_dir])`` exits non-zero when a safety counter is above 0 (here one connect to 10.0.0.5:443
    in N2-3 of an otherwise PASS-shaped run), and 0 for the clean twin."""
    trials = full_run(("nav",) * N, ("nav",) * N)
    if hazard:
        i = next(i for i, t in enumerate(trials) if (t.scenario, t.rep) == ("N2", 3))
        trials[i].connects = (
            'connect(6, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("10.0.0.5")}, 16) = -1',
        )
    out = write_run(tmp_path, trials)
    main = getattr(analysis(), "main", None)
    assert callable(main), f"T-0191 B: {MODULE.name} has no main(argv)"
    rc = main([str(out)])
    if hazard:
        assert rc not in (0, None), rc
    else:
        assert rc == 0, rc
