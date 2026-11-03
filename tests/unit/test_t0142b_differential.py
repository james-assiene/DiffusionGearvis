"""T-0142b A7, A8 (negative) and A6: with ``degeneracy_guard=False`` every session equals BASE 0248cfd's, and with the
guard on and nothing tripping on either segment every session equals BASE's with the guard on (A5's existing key
included); the frozen paths are byte-identical over a closed range; docs/02 gains one §5 paragraph
(docs/tasks/T-0142b.md).

- **The differential** is T-0142's harness, imported, not copied (``test_t0142_differential._capture`` and
  ``_compare``: T-0135's capture, itself the T-0134/T-0131/T-0132/T-0111g lineage, plus T-0142's thought near misses),
  plus this task's answer near misses (``_answer_near_miss``: tools off, an answer with 5 ``ha``, a 3 + 2 straddle of
  ``the.`` over answer canvases, a 63-character dot run, the hand table's conversational prose, a thought with 7
  ``the`` (A8: thought stays at 8); live, a thought ending in 7 ``the`` followed by an answer opening with 5 (12 if
  shared), and the longest clean logged answer of the fixture) and the out-of-scope paths (``_out_of_scope``: a
  ReGround pass 2 and a ``tool_repair`` whose answer text collapses). The base run is a subprocess over ``git archive
  0248cfd`` (T-0114's ``_base_tree``, ``BASE_SHA`` monkeypatched; origin checked), once per guard value. The guard
  value is put on every session of a capture by wrapping ``Session.__init__`` (``setdefault``), in the base run and in
  this process alike, so no harness is edited.
- Both differential rows are green at BASE and must stay green; they kill a guard too wide (``MIN_REPEATS_ANSWER``
  below 6, a shared thought/answer accumulator, thought judged at 6, an out-of-scope path judged) and a guard that
  ignores ``False``.
- **Byte identity** (A6, A8 and "This task does not own"): ``BASE_SHA..TIP_IMPL``, two literals, never the working
  tree. ``TIP_IMPL`` starts at BASE (no implementation yet), so the row holds trivially until it lands. **TIP moves**:
  whoever commits implementation bumps it; at the merge it closes to the last implementation commit.
- ``test_doc02_delta_t0142b`` (A7) reads docs/02 at ``TIP_IMPL``: red until the paragraph is committed and TIP bumped.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, 200 ms per word, tools off / ``dry_run`` / ``live``, loop on
(8 steps) where a scenario says so, n = 1, deterministic.
"""

from __future__ import annotations

import functools
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import _git_range as gr
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0107a_differential as t0107a
from tests.unit import test_t0107a_docs as t7d
from tests.unit import test_t0111g_differential as t0111g
from tests.unit import test_t0142_differential as t42d
from tests.unit import test_t0142_session as t42
from tests.unit._live_browser_support import FakeLiveBrowser, live_frame, live_rig
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins
from tests.unit._tool_turn_support import (
    REPAIRABLE_CLICK,
    REPAIRED_CLICK,
    VALID_CLICK,
    answer,
    thought,
    tool,
    tool_rig,
)
from tests.unit.test_session_reground import click
from tests.unit.test_t0117_differential import _PolicyMachine
from tests.unit.test_t0121_differential import _assert_equal, _with_history

#: T-0142b's BASE: main at 0248cfd (after ``Merge T-0142`` c80554e), the branch point. A literal.
BASE_SHA = "0248cfdaa56792836f2b2e7e7c72a9c5d9c68af3"
#: The implementation tip: a20baab on T-0142b-impl (code and the A7 docs/02 paragraph; was BASE). Bumped by
#: whoever commits implementation, closed to the last implementation commit at the merge.
TIP_IMPL = "a20baab2dfb5e38f1d5fb8e91ba3636d5f1b64d9"
HEADER = "**Spec delta (T-0142b"
T0142_HEADER = "**Spec delta (T-0142)"
GUARD = "degeneracy_guard"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "t0142b" / "answers.jsonl"

#: A6 and A8's list, written out from the criteria
FROZEN_A8 = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/conductor/tool_executor.py",
    "local_astra/cli.py",
    "local_astra/config.py",
    "local_astra/contracts",
    "tests/fakes",
    "local_astra/toolcall",
    "local_astra/computer",
    "hud",
    "services",
)
#: "This task does not own", beyond A8 (the T-0142 and T-0135 rows "stay green unedited")
NOT_OWNED = (
    "scripts/gpu",
    "scripts/locked",
    "tests/unit/test_t0142_degeneracy.py",
    "tests/unit/test_t0142_differential.py",
    # tests/unit/test_t0142_session.py: exempt (R-11 edits test_drafts_never_judged's step-3 answer)
    "tests/eval/test_t0142_replay_logs.py",
    "tests/fixtures/t0142",
    "tests/unit/test_t0135_ab_harness.py",
    "tests/unit/test_t0135_differential.py",
    "tests/unit/test_t0135_live_recovery.py",
    "tests/unit/test_t0135_replay.py",
    "tests/unit/test_t0135_thought_call.py",
    "tests/unit/_live_browser_support.py",
    "docs/tasks/T-0142.md",
    "docs/tasks/T-0135.md",
)

#: the hand table of conversational prose (test_t0142b_answer_detector), as one tools-off answer
CHAT_TURN = (
    "red, blue, green. No, no, no, no, no. That is very, very, very good. bye bye. "
    "1. Open the page. 2. Click add. 3. Check the cart. Well... I think so."
)


def _longest_clean_answer() -> list[Any]:
    rows = [json.loads(x) for x in FIXTURE.read_text().splitlines() if x.strip()]
    row = max((r for r in rows if r["class"] == "clean"), key=lambda r: r["n_tokens"])
    return [answer(c["text"], c["canvas_idx"]) for c in row["answers"]]


async def _answer_near_miss(tmp: Path) -> dict[str, Any]:
    """Answer text one short of the answer rule, and thought text one short of the thought rule (by hand)."""
    tmp.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    async with tool_rig(tmp / "off", kit=None, machine_cls=_PolicyMachine) as rig:
        rig.llm.script = [
            thought("Let me see" + " the" * 7 + " screen."),
            answer("Ha ha ha ha ha, that is a good one. ", 0),
            answer("Let me see. the. the. the.", 1),
            answer(" the. the. Done. Loading" + "." * 63 + " ok.", 2),
        ]
        await rig.ask("what is on this screen")
        await t0107a._until_turns(rig, 1, "the answer near-miss tools-off turn")
        rig.llm.script = [answer(CHAT_TURN, 0)]
        await rig.ask("tell me about it")
        await t0107a._until_turns(rig, 2, "the chat tools-off turn")
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    out["off"] = _with_history(rig, tmp / "off", session_id, log_path)
    browser = FakeLiveBrowser(
        [live_frame("onset")], post_frames=[live_frame(f"after-{i}") for i in range(1, 6)]
    )
    async with live_rig(tmp / "live", browser, machine_cls=_PolicyMachine) as rig:
        rig.llm.script = [answer(tsl.S1), tool(VALID_CLICK)]
        rig.llm.script_steps(
            [
                thought("I look and" + " the" * 7, 0, open_=True),
                answer(" the" * 5 + " page moved.", 1),
                tool(tsl.CLICK2, canvas_idx=1),
            ],
            _longest_clean_answer(),
        )
        await rig.ask(tsl.QUESTION)
        await t0107a._until_turns(rig, 1, "the answer near-miss live turn")
        session_id, log_path = rig.session.session_id, Path(rig.log.path)
    out["live"] = _with_history(rig, tmp / "live", session_id, log_path)
    out["live"]["executed"] = [[i.call_id, dict(i.arguments)] for i in browser.invocations]
    return out


async def _out_of_scope(tmp: Path) -> dict[str, Any]:
    """A ReGround pass 2 and a ``tool_repair`` whose answer text collapses (the guard value comes from the wrapper)."""
    tmp.mkdir(parents=True, exist_ok=True)
    pass2 = [
        thought("The button is at the top.", 0),
        answer("Here." + " the." * 20, 0),
        tool(click(900, 100), canvas_idx=1),
    ]
    rig, extras = await t42._pass2_turn(tmp / "pass2", t42.ABSENT, pass2)
    out: dict[str, Any] = {"pass2": t42._reground_view(rig, extras)}
    out["pass2"]["notes"] = [d.get("what") for d in rig.kind("note")]
    out["pass2"]["guard"] = extras.get(GUARD)
    async with t42.guard_rig(
        tmp / "repair",
        mode="dry_run",
        guard=t42.ABSENT,
        script=[answer(tsl.S1), tool(REPAIRABLE_CLICK)],
    ) as rig:
        rig.llm.script_continuation(answer("Fixed." + " the." * 20, 0), tool(REPAIRED_CLICK))
        await t42.run(rig, [answer(tsl.S3)])
        extras = dict(rig.record("t1").extras)
    out["repair"] = {
        "requests": [(d.get("phase"), d.get("request_id")) for d in rig.kind("llm.request")],
        "tool_calls": [(r.get("call_id"), r.get("status")) for r in rig.tool_records()],
        "executed": [(i.call_id, dict(i.arguments)) for i in rig.executor.invocations],
        "spoken": [c.source for c in rig.machine.commands(t42.C.TtsSpeak)],
        "errors": [(e.request_id, e.message) for e in rig.machine.events(t42.E.LlmError)],
        "aborted": [a.rsplit("-", 2)[-2:] for a in rig.llm.aborted],
        "notes": [d.get("what") for d in rig.kind("note")],
        "extras": {k: v for k, v in extras.items() if k != "ms" and not k.endswith("_ms")},
    }
    return out


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every scenario. Runs in-process on the working tree and in a subprocess on the 0248cfd tree."""
    return {
        "t0142": await t42d._capture(tmp / "t0142"),
        "answer_near_miss": await _answer_near_miss(tmp / "answer_near_miss"),
        "out_of_scope": await _out_of_scope(tmp / "out_of_scope"),
    }


async def _capture_guard(tmp: Path, value: bool) -> dict[str, Any]:
    """``_capture`` with ``degeneracy_guard=value`` put on every session (``setdefault``), then JSON-normalised."""
    cls = importlib.import_module("local_astra.conductor.session").Session
    real_init = cls.__init__

    @functools.wraps(real_init)  # keeps the signature visible to the rigs' seam checks
    def init(self: Any, *a: Any, **k: Any) -> None:
        k.setdefault(GUARD, value)
        real_init(self, *a, **k)

    cls.__init__ = init
    try:
        sections = await _capture(tmp)
    finally:
        cls.__init__ = real_init
    normalised: dict[str, Any] = json.loads(json.dumps(sections))
    return normalised


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0142b_differential as m
out = pathlib.Path(sys.argv[1])
value = sys.argv[2] == "true"
sections = asyncio.run(m._capture_guard(out.parent / "run", value))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[bool, dict[str, Any]]:
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0142b_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_SHA)
        base_dir = tsl._base_tree(tmp / "base_tree")
    got: dict[bool, dict[str, Any]] = {}
    for value in (False, True):
        out = tmp / f"base_out_{value}" / "sections.json"
        out.parent.mkdir(parents=True)
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(out), "true" if value else "false"],
            cwd=base_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 0, (
            f"the base run ({GUARD}={value}) failed:\n{proc.stderr[-3000:]}"
        )
        data = json.loads(out.read_text())
        assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
            f"origin proof: the base run imported local_astra from {data['origin']}, not the {BASE_SHA} tree"
        )
        got[value] = data["sections"]
    return got


def _same(expected: Any, got: Any, what: str) -> None:
    if not isinstance(expected, list) or not isinstance(got, list):
        expected, got = [expected], [got]  # T-0121's helper compares lists
    _assert_equal(expected, got, f"T-0142b A8 {what} (base {BASE_SHA[:7]}; T-0121's helper):")


def _compare(base: dict[str, Any], now: dict[str, Any], label: str) -> None:
    t42d._compare(
        base["t0142"], now["t0142"], f"T-0142b {label}"
    )  # normalises its own observed side
    # T-0155 O-8: the persona sentence only, on the rest of the observed side
    now = observed({k: v for k, v in now.items() if k != "t0142"})
    for name in ("off", "live"):
        for section in t0107a.SECTIONS:
            _same(
                base["answer_near_miss"][name][section],
                now["answer_near_miss"][name][section],
                f"{label} answer near miss {name} {section!r}",
            )
    _same(
        base["answer_near_miss"]["live"]["executed"],
        now["answer_near_miss"]["live"]["executed"],
        f"{label} answer near miss executed",
    )
    for name in ("pass2", "repair"):
        for key in base["out_of_scope"][name]:
            _same(
                base["out_of_scope"][name][key],
                now["out_of_scope"][name][key],
                f"{label} out of scope {name} {key!r}",
            )


def _guard_values(tree: Any) -> list[Any]:
    found: list[Any] = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k == GUARD:
                found.append(v)
            else:
                found += _guard_values(v)
    elif isinstance(tree, list):
        for v in tree:
            found += _guard_values(v)
    return found


async def test_false_equal_base(base: dict[bool, dict[str, Any]], tmp_path: Path) -> None:
    """A8 (a guard that ignores ``False``): ``degeneracy_guard=False`` on every session gives exactly BASE's
    sections with ``False`` (machine steps, HUD messages, prompts, request bodies, session-log records, turn lines,
    history, ReGround and repair views), and no record names the guard. Green at BASE."""
    now = await _capture_guard(tmp_path, False)
    # premises: the near-miss turns ran the paths they name
    assert [c for c, _ in now["answer_near_miss"]["live"]["executed"]] == ["t1-call", "t1-call-s2"]
    assert [r["outcome"] for r in now["out_of_scope"]["pass2"]["regrounds"]] == ["ok"]
    assert ["tool_repair", "t1-repair"] in now["out_of_scope"]["repair"]["requests"]
    _compare(base[False], now, "False")
    assert GUARD not in json.dumps(now), "T-0142b A5: no key with the guard off"


async def test_on_without_trip_equal_base(base: dict[bool, dict[str, Any]], tmp_path: Path) -> None:
    """A8 (a guard too wide: ``MIN_REPEATS_ANSWER`` below 6, a shared accumulator, thought at 6, pass 2 or repair
    judged): with the guard on and nothing tripping on either segment, every section equals BASE's with the guard on,
    the A5 key included, and that key is ``{"on": true, "tripped": 0}`` wherever it appears. Green at BASE."""
    now = await _capture_guard(tmp_path, True)
    found = _guard_values(now)
    assert len(found) >= 10, f"premise: the scenarios wrote turn lines with the key; {len(found)}"
    assert all(v == {"on": True, "tripped": 0} for v in found), [
        v for v in found if v != {"on": True, "tripped": 0}
    ][:3]
    notes = [
        d for d in re.findall(r"degenerate (?:answer|thought): request aborted", json.dumps(now))
    ]
    assert notes == [], f"T-0142b A8: a near miss tripped: {notes[:3]}"
    _compare(base[True], now, "on, no trip")


def test_frozen_paths_over_closed_range() -> None:
    """A6/A8: the machine, the speech gate, ``llm_client.py``, the prompt builder, ``tool_executor.py``, ``cli.py``,
    ``config.py``, the contracts, the fakes, ``toolcall/``, ``computer/``, the HUD, ``services/`` and every not-owned
    path (``scripts/gpu``, ``scripts/locked``, the T-0142 and T-0135 test modules, T-0142's eval and fixtures,
    ``_live_browser_support.py``, the T-0142 and T-0135 task files) are byte-identical over ``BASE_SHA..TIP_IMPL``
    (never the working tree), with no file added or removed under them. Power: the same predicate sees
    ``82da380..02ccff5``'s lock change."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    assert gr.is_ancestor(BASE_SHA, TIP_IMPL), (
        f"{BASE_SHA[:7]} is not an ancestor of {TIP_IMPL[:7]}"
    )
    frozen = FROZEN_A8 + NOT_OWNED
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_SHA, "--", *frozen).stdout.split()
    for p in frozen:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    assert gr.changed(BASE_SHA, TIP_IMPL, *frozen, diff_filter="ADR") == []
    changed = gr.changed(BASE_SHA, TIP_IMPL, *frozen)
    assert changed == [], (
        f"T-0142b A6/A8: frozen paths differ in {BASE_SHA[:7]}..{TIP_IMPL[:7]}: {changed}"
    )
    gr.require_commits(t0111g.P2_PARENT, t0111g.P2)
    assert gr.changed(t0111g.P2_PARENT, t0111g.P2, *frozen) == ["services/versions.lock"], "power"


def _doc02_at(rev: str) -> str:
    proc = gr._git("show", f"{rev}:docs/02-architecture.md")
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_doc02_delta_t0142b() -> None:
    """A7: docs/02 §5 gains exactly one paragraph starting ``**Spec delta (T-0142b``, recording
    ``MIN_REPEATS_ANSWER`` = 6 and the voiced-cost reason, ``DEGENERATE_ANSWER_MESSAGE`` on the HUD, that answers are
    judged *before* submission, that pass 2, repair, prefill and the captioner are out of scope, that there is no
    retry, the evidence counts (1,666 / 1 / 1,665) and that it supersedes T-0142's R-o; T-0142's paragraph stays
    verbatim and removing the new one leaves docs/02 equal to BASE's paragraph for paragraph. Keywords, never
    wording. Read at ``TIP_IMPL``."""
    gr.require_commits(BASE_SHA, TIP_IMPL)
    base_doc = _doc02_at(BASE_SHA)
    base_paras = t7d.paragraphs(base_doc)
    assert [p for p in base_paras if p.lstrip().startswith(HEADER)] == [], "premise"
    t0142 = [p for p in base_paras if p.lstrip().startswith(T0142_HEADER)]
    assert len(t0142) == 1, f"premise: BASE has T-0142's paragraph; found {len(t0142)}"
    doc = _doc02_at(TIP_IMPL)
    paras = [p for p in t7d.paragraphs(t7d.section(doc, "## 5. ")) if p.lstrip().startswith(HEADER)]
    assert len(paras) == 1, (
        f"T-0142b A7: one §5 paragraph starting {HEADER!r} in {TIP_IMPL[:7]}; found {len(paras)}"
    )
    para = paras[0]
    low = para.lower()
    need = {
        "MIN_REPEATS_ANSWER / 6": "MIN_REPEATS_ANSWER" in para
        and re.search(r"\b6\b", para) is not None,
        "DEGENERATE_ANSWER_MESSAGE": "DEGENERATE_ANSWER_MESSAGE" in para,
        "before (submission)": "before" in low,
        "pass 2": "pass 2" in low or "pass-2" in low,
        "R-o (supersedes)": "R-o" in para and "supersed" in low,
        "HUD": "hud" in low,
        "repair": "repair" in low,
        "prefill": "prefill" in low,
        "captioner": "caption" in low,
        "no retry": "retry" in low,
        "voiced cost": "voic" in low,
        "evidence 1,666": "1,666" in para or "1666" in para,
        "evidence 1,665": "1,665" in para or "1665" in para,
    }
    missing = [k for k, ok in need.items() if not ok]
    assert not missing, f"T-0142b A7: the docs/02 delta does not record {missing}"
    rest = t7d.paragraphs(doc)
    assert t0142[0] in rest, "T-0142b A7: T-0142's paragraph stays verbatim"
    rest.remove(para)
    assert rest == base_paras, "T-0142b A7: docs/02 changes only by the one delta paragraph"
