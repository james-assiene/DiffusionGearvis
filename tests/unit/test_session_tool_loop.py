"""T-0114: the tool-result loop wired into the session (docs/tasks/T-0114.md criteria 1-13; T-0111a §1, §3-§5).

Every row runs over the session fakes (``tests/unit/_tool_turn_support.tool_rig``): ``FakeLlm`` with one script per
step (``script_steps``), the recording executor (hanging/raising modes), ``FakeSearch`` used as is, the
``ScriptedClock``. Expected step texts are computed **in the row** with ``loop.step_prompt``/``render_response``
and the payload builders from the fixture rows (T-0113 proved them against the template oracle); nothing is
copied from the implementation.

Rows that need the implementation carry ``T0114_PENDING`` (strict xfail until ``/implement T-0114`` removes it).
Rows that already hold on the base (the loop-off differential, the inert ``llm.continue``, the content pins, the
fixture premises) are not pending: they guard the base behaviour the task must keep.

The surface the rows assume where the task is silent (listed in the test-author's report):

- step machine ids ``{turn}-gen-s{k}``/``{turn}-call-s{k}`` (T-0111a §2), engine ids ``{session}-{machine id}-{n}``
  (``_RequestIds``, unchanged);
- the turn line's three keys live in ``TurnRecord.extras`` (T-0111a §5 "extras gain"); ``tool_steps`` is the number
  of steps the turn ran (generates, the repair not counted);
- ``tool.result`` session-log records carry ``call_id``, ``step`` (the step of the executed call), ``status`` and
  ``response`` (either the rendered block or the payload mapping);
- the withheld-text note is ``note {what: "step text withheld", step, chars}``;
- a dropped terminal of criterion 7(b) is a ``note`` whose ``what`` contains ``"terminal"``; a late result of
  criterion 12 a ``note`` whose ``what`` contains ``"stale"``.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word unless a row says otherwise,
tools ``dry_run``, n = 1 and deterministic per row. No measurement.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_astra.conductor import session as session_mod
from local_astra.conductor.turn_machine import TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit, Prompt
from local_astra.contracts.toolcall import SearchResult, SearchResults, ToolOutcome
from local_astra.contracts.turn import TurnState
from local_astra.telemetry import session_log
from local_astra.toolcall import live, loop
from tests import _git_range as gr
from tests.fakes.llm import http_error
from tests.fakes.search import DEFAULT_ROWS, FakeSearch
from tests.fakes.tool_executor import RecordingToolExecutor
from tests.unit import _tools_switch_support as T
from tests.unit._t0162c_pins import observed  # T-0155 O-8 + T-0162 C pins
from tests.unit._tool_turn_support import (
    CALL_ID,
    REPAIRABLE_CLICK,
    VALID_CLICK,
    ToolRig,
    WatchingExecutor,
    answer,
    final_message,
    thought,
    tool,
    tool_rig,
)
from tests.unit.test_session_loop import TURN_MACHINE_SHA256
from tests.unit.test_session_tools_off_golden import StateMachine, capture_sections, normalise

REPO = Path(__file__).resolve().parents[2]

#: T-0114 branched from main at d3bc6ce (docs/tasks/T-0114.md, "Branch"). A literal, never a moving ref (tests.md).
BASE_SHA = "d3bc6ce"
#: T-0114's closed merge range (T-0131 R1), fixed literals: M = its merge into main, TIP = M^2, the range base =
#: merge-base(M^1, M^2) (the branch merged main mid-way), and a later commit M is an ancestor of (T-0131's BASE).
T0114_MERGE = "a54e5f8"
T0114_TIP = "528f41e"
T0114_RANGE_BASE = "09b3ad2"
T0114_LATER = "f30365d"
#: T-0162 group A's base (main at the branch point, a literal) and the three files its reset row edits.
T0162_BASE = "3162579a"
T0162_MOVED = frozenset(
    {
        "local_astra/conductor/turn_machine.py",
        "local_astra/contracts/events.py",
        "local_astra/contracts/commands.py",
    }
)

#: T-0182 B re-pin (closed range, the T-0162 way): B4 adds one ``MachineState`` field to ``contracts/turn.py``.
#: T-0114's claim about it is that it kept d3bc6ce's content until T-0182 branched, so it is checked at T-0182's
#: literal base; its T-0182 content is pinned as a literal (sha256 at the implementation commit b8d26ae7).
T0182_BASE = "7646b8b6409a30fcc3794c2453f4b77e080cbdb6"
T0182_MOVED = {
    "local_astra/contracts/turn.py": "9a8f98131bd905f856a36ab8a972ad1f89b6d4f4a7da3ef15558a291da2e3d58",
}

#: T-0181 A0-7 re-pin (base-then-literal, the T-0162/T-0182 way): A0 edits ``turn_machine.py`` (the ``MicMuted`` row)
#: and ``contracts/events.py`` (``MicMuted``). At T-0181's literal base 021c2ef6 they are the T-0182 B machine
#: (afc57335's content, the previous TURN_MACHINE_SHA256) and the post-T-0162 events file; their only allowed change is
#: A0's exact blobs at the implementation commit 9dbddf96 (sha256 of each, computed with sha256sum 2026-10-28).
T0181_BASE = "021c2ef6d33a916727fbf83ed554a2c4a5fbeef2"
T0181_PRIOR = {
    "local_astra/conductor/turn_machine.py": "507c51ebc9ffc7c225e6a4b0c4607e25f9e4d52c524476feacbe8f0e37af5ec6",
    "local_astra/contracts/events.py": "71f430714720e5bb098c3627a4cc6b324ac053348f96b814751e2f5d8164c086",
}
T0181_MOVED = {
    "local_astra/conductor/turn_machine.py": "634c9260161c14935f70965166ebf9e876a8974543b543df7dfc51735635dcc7",
    "local_astra/contracts/events.py": "76aabfefc8faad5b7ee772f8a3ca6e5024a735aaa4a1d45fb42386acb4996934",
}

#: sha256 of each file criterion 2 / criterion 13 keeps byte-identical, computed from ``git show d3bc6ce:<path>``
#: (an invariant pins content; ``test_the_content_pins_are_the_base_blobs`` re-derives them from the base).
PINNED = {
    "local_astra/conductor/turn_machine.py": "be46b60bf87d9f6c6db572c11986908fbcf35a4a4b58c2e037353e5962fb65ac",
    "local_astra/contracts/turn.py": "4b450c209e925267f7e811ced05715988191620d1e8d198baa31de1e6a927d8d",
    "local_astra/contracts/events.py": "86bec370b760d1ece4c773748d0bcb05afd0d9f4688b81fe27ba74a3e25098f0",
    "local_astra/contracts/commands.py": "5f599614afe7aff7716d6eec17f1e718f1a9a3983dbf8c390e0e4aca19d5bec6",
    "local_astra/telemetry/spans.py": "01eedb33aff8fe8055c246a3c649b77a5dc0c57d2a9b5463610568492a1da297",
}

#: H-T114-4 defaults, written out from the task.
DEFAULT_BOUNDS = {
    "TOOL_EXEC_TIMEOUT_MS": 10_000,
    "MAX_STEP_TEXT": 1000,
    "MAX_ARG_TEXT": 500,
    "MAX_CONTINUATION": 14_000,
}
#: H-T114-5
LOOP_ENDS = frozenset(
    {
        "final",
        "budget",
        "refused",
        "interrupted",
        "takeover",
        "error",
        "tool_timeout",
        "over_budget",
        "disconnected",
        "stopped",  # T-0193 A-3: a loop turn the HUD's stop ended
    }
)
LOOP_KEYS = ("tool_steps", "tool_steps_executed", "tool_loop_end")
#: The keys of the step-1 ``llm.request`` record at d3bc6ce (``Session._run_generate``), written out by hand.
MAIN_REQUEST_KEYS = frozenset(
    {
        "phase",
        "request_id",
        "attempt_id",
        "options",
        "xargs",
        "prompt",
        "drop_history_turns",
        "prompt_tokens",
    }
)

LOOP = TurnPolicy(thinking=True, tool_loop=True)


def loop_policy(max_steps: int = 8) -> TurnPolicy:
    return TurnPolicy(thinking=True, tool_loop=True, max_tool_steps=max_steps)


# --- fixture calls and texts --------------------------------------------------------------------------------

Q = '<|"|>'
QUESTION = "what is the weather in paris"
QUERY = "weather in Paris"
SEARCH_CALL = f"<|tool_call>call:google_search{{query:{Q}{QUERY}{Q}}}<tool_call|>"
SEARCH_ARGS = {"query": QUERY}
CLICK_ARGS = {"x": 500, "y": 250}
CLICK2 = "<|tool_call>call:click_at{x:120,y:640}<tool_call|>"
CLICK2_ARGS = {"x": 120, "y": 640}
CLICK3 = "<|tool_call>call:click_at{x:900,y:40}<tool_call|>"
CLICK3_ARGS = {"x": 900, "y": 40}
TYPE_TEXT = "T" * 5000
TYPE_CALL = (
    f"<|tool_call>call:type_at{{press_enter:false,text:{Q}{TYPE_TEXT}{Q},x:500,y:250}}<tool_call|>"
)
TYPE_ARGS = {"press_enter": False, "text": TYPE_TEXT, "x": 500, "y": 250}

PREFIX = "Let me look that up."
REPORT = "It is fourteen degrees in Paris."
S1 = "I will open the menu."
S2 = "Now the settings item."
S3 = "Done, the settings are open."
CANARY = "ZQX-CANARY"
SECRET = "SECRET-THOUGHT"
#: what ``RecordingToolExecutor``/``WatchingExecutor`` answer for a click (their default outcome's action)
RECORDED_CLICK = loop.action_payload("recorded click_at")


def results_of(rows: Sequence[Any], query: str) -> SearchResults:
    """The ``SearchResults`` ``FakeSearch`` answers for ``rows`` -- built from the fixture rows, not from the fake."""
    return SearchResults(
        query=query,
        results=tuple(
            SearchResult(title=r["title"], url=r["url"], snippet=r["snippet"]) for r in rows
        ),
    )


def open_answer(text: str) -> Commit:
    return Commit(request_id="?", canvas_idx=0, segment="answer", text=text, thought_open=True)


class RaisingSearch(FakeSearch):
    """A backend that logs the query and raises (criterion 9, "the backend raises")."""

    async def search(self, query: str) -> Any:
        self.queries.append(query)
        raise RuntimeError("the search backend is down")


# --- reading the rig ------------------------------------------------------------------------------------------


def _engine(machine_id: str) -> re.Pattern[str]:
    return re.compile(rf"-{re.escape(machine_id)}-\d+$")


def step_id(k: int, turn: str = "t1") -> str:
    return f"{turn}-gen" if k == 1 else f"{turn}-gen-s{k}"


def call_id(k: int, turn: str = "t1") -> str:
    return f"{turn}-call" if k == 1 else f"{turn}-call-s{k}"


def generates_for(rig: ToolRig, machine_id: str) -> list[tuple[str, Prompt]]:
    pat = _engine(machine_id)
    return [(rid, p) for rid, p, _ in rig.llm.generates if pat.search(rid)]


def sent(rig: ToolRig, machine_id: str) -> Prompt:
    """The prompt of ``machine_id``'s last attempt, as the backend was handed it."""
    found = generates_for(rig, machine_id)
    assert found, (
        f"no generate was sent for {machine_id}; engine ids {[r for r, _, _ in rig.llm.generates]}"
    )
    return found[-1][1]


def steps_of(rig: ToolRig, turn: str = "t1") -> list[tuple[str, Prompt]]:
    pat = re.compile(rf"-{re.escape(turn)}-gen-s\d+-\d+$")
    return [(rid, p) for rid, p in rig.llm.step_prompts if pat.search(rid)]


def step_request(rig: ToolRig, k: int, turn: str = "t1") -> Prompt:
    found = [p for rid, p in rig.llm.step_prompts if _engine(step_id(k, turn)).search(rid)]
    assert len(found) == 1, (
        f"T-0114: exactly one step request for {step_id(k, turn)}; got {len(found)} "
        f"(step requests {[r for r, _ in rig.llm.step_prompts]})"
    )
    return found[0]


def tail(prompt: Prompt) -> str:
    return prompt.messages[-1].text()


def oracle(pass1: Prompt, steps: Sequence[tuple[str, str, Any, Any]]) -> Prompt:
    return loop.step_prompt(pass1, [loop.LoopStep(*s) for s in steps])


def dones(rig: ToolRig, machine_id: str) -> list[Any]:
    found = rig.dones(machine_id)
    assert len(found) == 1, f"one LlmDone({machine_id!r}) reached the machine; got {found}"
    return found


def results_for(rig: ToolRig, cid: str) -> list[Any]:
    return [e for e in rig.machine.events(E.ToolResult) if e.call_id == cid]


def notes(rig: ToolRig, what: str | None = None) -> list[dict[str, Any]]:
    return [d for d in rig.kind("note") if what is None or d.get("what") == what]


def published_json(rig: ToolRig, *types: str) -> str:
    return json.dumps(
        [
            m.model_dump(mode="json")
            for m in rig.published.messages
            if not types or getattr(m, "type", None) in types
        ]
    )


def history_text(rig: ToolRig) -> str:
    return json.dumps([h.model_dump(mode="json") for h in rig.state.history])


def loop_constants() -> dict[str, int]:
    """The four module constants of the implementation notes; their absence is a named failure.

    Called first by every row that hangs the executor: a session that awaited the executor inline (today's loop-off
    path) would block the pump, and the row must fail on this assertion instead of on a settle timeout."""
    missing = [n for n in DEFAULT_BOUNDS if not hasattr(session_mod, n)]
    assert not missing, (
        f"T-0114 implementation notes: session.py has no module constant(s) {missing} (H-T114-4 defaults "
        f"{DEFAULT_BOUNDS})"
    )
    return {n: int(getattr(session_mod, n)) for n in DEFAULT_BOUNDS}


async def until_steps(
    rig: ToolRig, n: int, *, turn: str = "t1", limit_ms: int = 6_000, step: int = 50
) -> None:
    await rig.until(
        lambda: len(steps_of(rig, turn)) >= n,
        f"T-0114: {n} step request(s) of {turn} were never sent (step requests "
        f"{[r for r, _ in rig.llm.step_prompts]}; LlmContinue {rig.commands(C.LlmContinue)}; "
        f"ToolResult {rig.machine.events(E.ToolResult)})",
        limit_ms=limit_ms,
        step=step,
    )


async def loop_turn(
    rig: ToolRig, steps: int, question: str = QUESTION, *, limit_ms: int = 20_000
) -> None:
    """Ask, wait for ``steps - 1`` step requests (fails fast on the base), then for the turn's end."""
    await rig.ask(question)
    if steps > 1:
        await until_steps(rig, steps - 1)
    await rig.turn_done(f"T-0114: the {steps}-step turn never ended", limit_ms=limit_ms)


THREE_STEP = dict(
    script=[answer(S1), tool(VALID_CLICK)],
    steps=([answer(S2), tool(CLICK2)], [answer(S3)]),
)


def three_step_oracle(rig: ToolRig) -> tuple[Prompt, Prompt]:
    pass1 = sent(rig, "t1-gen")
    s1 = (S1, "click_at", CLICK_ARGS, RECORDED_CLICK)
    s2 = (S2, "click_at", CLICK2_ARGS, RECORDED_CLICK)
    return oracle(pass1, [s1]), oracle(pass1, [s1, s2])


# =====================================================================================================
# Fixture premises (not pending: they hold on the base)
# =====================================================================================================


def test_the_fixture_calls_classify_as_the_rows_assume():
    """The rows' calls mean what they say: valid with these arguments, or repairable (``live.classify``)."""
    for text, name, args in (
        (SEARCH_CALL, "google_search", SEARCH_ARGS),
        (VALID_CLICK, "click_at", CLICK_ARGS),
        (CLICK2, "click_at", CLICK2_ARGS),
        (CLICK3, "click_at", CLICK3_ARGS),
        (TYPE_CALL, "type_at", TYPE_ARGS),
    ):
        verdict = live.classify(text)
        assert (verdict.status, verdict.name, verdict.arguments) == ("valid", name, args), (
            text[:80],
            verdict,
        )
    assert live.classify(REPAIRABLE_CLICK).status == "repairable"


# =====================================================================================================
# Criterion 1: loop off is unchanged
# =====================================================================================================


class _LoopOffMachine(StateMachine):
    """The golden's full-state machine (default ``POLICY``: the loop off), taking ``tool_rig``'s policy argument."""

    def __init__(self, policy: Any = None) -> None:
        super().__init__()


#: A loop-off tool turn, a toolless turn and a call with no spoken sentence.
LOOP_OFF_TURNS: tuple[tuple[str, tuple[Commit, ...]], ...] = (
    ("click the sign in button", (answer(S1), tool(VALID_CLICK))),
    ("and what does it say now", (answer("It says welcome back."),)),
    ("search the weather", (tool(SEARCH_CALL),)),
)


async def _loop_off_capture(tmp: Path) -> dict[str, Any]:
    """The scenario of ``test_loop_off_records_equal_base``, through the tools-off golden's section capture.

    Runs in-process on the working tree, and in a subprocess on the d3bc6ce tree; it uses only names both trees
    have (``tool_rig`` without ``policy``/``search``)."""
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, machine_cls=_LoopOffMachine) as rig:
        for i, (question, script) in enumerate(LOOP_OFF_TURNS):
            rig.llm.script = list(script)
            await rig.ask(question)
            await rig.until(
                lambda n=i + 1: rig.state.state is TurnState.IDLE and len(rig.lines()) == n,
                f"loop-off turn {i + 1} never ended",
            )
            await rig.advance(300)
        session_id = rig.session.session_id
        log_path = Path(rig.log.path)
    sections = capture_sections(rig, rig.log_records)
    # every session-log record, not only the golden's llm./machine.step/tts. kinds (tool.call, note, turn.record)
    # Per-process values, normalised here and nowhere else: ``log.started`` names the writer's pid, and
    # ``log.stopped`` counts the raw bytes written, which carry the pid, the tmp paths and the git commit.
    # (``records`` in ``log.stopped`` stays compared.)
    per_process = {"log.started": {"pid": "<pid>"}, "log.stopped": {"bytes": "<bytes>"}}
    sections["log"] = [
        {"kind": r["kind"], "data": {**r["data"], **per_process.get(r["kind"], {})}}
        for r in rig.log_records
    ]
    out: dict[str, Any] = normalise(sections, session_id, tmp, log_path)
    # ``turn.record`` repeats the line's regime, so its ``git_commit`` is normalised as the golden does the line's
    for r in out["log"]:
        regime = r["data"].get("regime") if isinstance(r["data"], dict) else None
        if isinstance(regime, dict) and "git_commit" in regime:
            regime["git_commit"] = "<git>"
    return out


_BASE_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_session_tool_loop as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._loop_off_capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


def _base_tree(dest: Path) -> Path:
    """``local_astra``, ``services`` and the calibration file at d3bc6ce, extracted under ``dest`` (a tmp dir)."""
    archive = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "archive",
            "--format=tar",
            BASE_SHA,
            "local_astra",
            "services",
            "docs/cd/calibration-bfcl.json",
        ],
        capture_output=True,
        check=True,
    ).stdout
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return dest


async def test_loop_off_records_equal_base(tmp_path):
    """Criterion 1 (T-0111a 14; L8 at session level): with ``tool_loop`` off, a tool turn, a toolless turn and a
    prefix-less call produce the same machine steps, HUD messages, prompts, request bodies, session-log records
    (``llm.request``/``llm.commit``/``tool.call`` and every other kind) and ``turns.jsonl`` lines as the session at
    the literal base d3bc6ce. The base runs in a subprocess over ``git archive d3bc6ce`` (its ``local_astra`` first
    on ``sys.path``, origin printed and checked); the harness is the tools-off golden's ``capture_sections`` and
    ``normalise`` (session id, tmp dir, log name, git commit)."""
    gr.require_commits(BASE_SHA)
    base_dir = _base_tree(tmp_path / "base_tree")
    out = tmp_path / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _BASE_RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    base = json.loads(out.read_text())
    assert Path(base["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {base['origin']}, not from the d3bc6ce tree"
    )
    now = observed(
        await _loop_off_capture(tmp_path / "now")
    )  # T-0155 O-8: the persona sentence only
    for section in ("machine", "hud", "prompts", "requests", "log", "records"):
        expected, got = base["sections"][section], now[section]
        first = next(
            (i for i, (a, b) in enumerate(zip(expected, got, strict=False)) if a != b), None
        )
        assert got == expected, (
            f"T-0114 criterion 1: with tool_loop off the {section!r} section differs from {BASE_SHA} "
            f"(lengths {len(expected)} vs {len(got)}; first difference at {first}: "
            f"base {json.dumps(expected[first] if first is not None else None)[:800]} / "
            f"now {json.dumps(got[first] if first is not None else None)[:800]})"
        )
    for record in now["records"]:
        assert not set(LOOP_KEYS) & set(record["extras"]), (
            f"T-0114 criterion 1: no loop-off turn line carries {LOOP_KEYS}; got {sorted(record['extras'])}"
        )
        assert record["regime"]["tools"] == T.REGIME_ON, (
            f"T-0114 criterion 1: Regime.tools is unchanged with the loop off; got {record['regime']['tools']!r}"
        )


@pytest.mark.parametrize("case", ["no_kit_loop_on", "kit_loop_off"])
async def test_llm_continue_inert_without_kit_or_loop(tmp_path, case):
    """Criterion 1, last item: ``LlmContinue`` with ``tool_kit is None`` (loop on) or with the loop off (kit on) is
    inert: one ``note``, and no request, speech, playback, publish, execution, summary or machine step."""
    kit, policy = (
        (None, LOOP) if case == "no_kit_loop_on" else ("default", TurnPolicy(thinking=True))
    )
    async with tool_rig(tmp_path, kit=kit, policy=policy) as rig:
        await rig.pump()

        def snapshot() -> tuple[Any, ...]:
            return (
                list(rig.llm.calls),
                list(rig.tts.requests),
                list(rig.audio.plays),
                list(rig.audio.calls),
                len(rig.published.messages),
                len(rig.executor.invocations),
                rig.session.summaries_scheduled,
                rig.machine.steps,
            )

        before = snapshot()
        await rig.session.executor.execute(
            C.LlmContinue(request_id="t1-gen-s2", step=2, call_id="t1-call")
        )
        await rig.pump()
        assert snapshot() == before, f"T-0114 criterion 1 ({case}): LlmContinue had an effect"
    found = notes(rig, "llm.continue")
    assert len(found) == 1 and found[0].get("inert") is True, (
        f"T-0114 criterion 1 ({case}): exactly one inert llm.continue note; got {found}"
    )


# =====================================================================================================
# Criterion 2: the machine is untouched
# =====================================================================================================


# T-0178 B6a: re-pinned through TURN_MACHINE_SHA256 (f8e26760, then 52be36b0 for B3a); commands.py is already checked at T-0162's base
def test_machine_and_contracts_byte_identical_to_base():
    """Criterion 2 (negative): the machine and contract files hash to the d3bc6ce content, and the session-loop pin
    ``TURN_MACHINE_SHA256`` is that same hash (it is not moved by this task).

    T-0131 R1 (docs/tasks/T-0131.md B9: ``telemetry/spans.py`` gains ``tool.reground``): ``spans.py`` left this
    working-tree comparison. Criterion 13's ``spans.py`` claim is about T-0114's own range, so it is now checked over
    T-0114's closed merge range (``gr.prove_range``; T-0114's branch merged main mid-way, so its range base is
    ``merge-base(M^1, M^2)`` = ``T0114_RANGE_BASE``, not d3bc6ce): ``spans.py`` does not change over it, and the file
    at the tip still hashes to the d3bc6ce pin. Power: the same ``gr.changed`` over the range sees ``session.py``."""
    spans_path = "local_astra/telemetry/spans.py"
    gr.prove_range(T0114_RANGE_BASE, T0114_TIP, T0114_MERGE, later=T0114_LATER)
    assert gr.changed(T0114_RANGE_BASE, T0114_TIP, spans_path) == [], (
        f"T-0114 criterion 13: {spans_path} changed over {T0114_RANGE_BASE}..{T0114_TIP}"
    )
    assert gr.changed(T0114_RANGE_BASE, T0114_TIP, "local_astra/conductor/session.py") == [
        "local_astra/conductor/session.py"
    ], "power: the range predicate sees T-0114's session.py change"
    tip_blob = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{T0114_TIP}:{spans_path}"],
        capture_output=True,
        check=True,
    ).stdout
    assert hashlib.sha256(tip_blob).hexdigest() == PINNED[spans_path], (
        f"T-0114 criterion 13: {spans_path} changed between {BASE_SHA} and T-0114's tip {T0114_TIP}"
    )
    # T-0162 A-2 (pre-authorised pin move, amendment 2026-10-27): the reset row edits these three
    # files. T-0114's claim about them is that they kept d3bc6ce's content until T-0162 branched, so
    # they are checked at T-0162's literal base; T-0162's own additions to them are pinned exactly by
    # test_turn_frames (classes), test_no_new_event_or_command_types (tags) and TURN_MACHINE_SHA256.
    gr.require_commits(T0162_BASE)
    for path in T0162_MOVED:
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{T0162_BASE}:{path}"],
            capture_output=True,
            check=True,
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == PINNED[path], (
            f"T-0114 criterion 2: {path} differs from {BASE_SHA}'s content at T-0162's base {T0162_BASE}"
        )
    gr.require_commits(T0182_BASE)
    for path, sha in T0182_MOVED.items():
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{T0182_BASE}:{path}"],
            capture_output=True,
            check=True,
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == PINNED[path], (
            f"T-0114 criterion 2: {path} differs from {BASE_SHA}'s content at T-0182's base {T0182_BASE}"
        )
        # T-0193 A0 re-pin (A0-9, the T-0192 B way): b8d26ae7's content is the file at T-0193 A0's literal base
        # fdb9fc01 (the base end), and the working tree is exactly the merged 373df531 blob (A0-R5, the tree end)
        from tests.unit.test_t0193_stop_row import (
            T0193_BASE,
            T0193_BASE_SHA256,
            T0193_MOVED_SHA256,
            t0193_end_sha256,
        )

        assert T0193_BASE_SHA256[path] == sha, f"T-0193 A0: {path}'s base end is b8d26ae7's content"
        assert t0193_end_sha256(path, T0193_BASE) == sha, (
            f"T-0182 B / T-0193 A0: {path} at T-0193 A0's base {T0193_BASE[:8]} is not b8d26ae7's content"
        )
        assert t0193_end_sha256(path) == T0193_MOVED_SHA256[path], (
            f"T-0193 A0-R5: {path} is not the merged 373df531 blob"
        )
    working_tree = {
        p: s
        for p, s in PINNED.items()
        if p != spans_path and p not in T0162_MOVED and p not in T0182_MOVED
    }
    changed = {
        path: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
        for path, sha in working_tree.items()
        if hashlib.sha256((REPO / path).read_bytes()).hexdigest() != sha
    }
    assert not changed, (
        f"T-0114 criteria 2/13: these files differ from {BASE_SHA}'s content: {changed}"
    )
    # T-0181 A0-7: the machine and events.py are their T-0181 base content at 021c2ef6 and, in the working tree,
    # exactly A0's blobs at 9dbddf96; any other edit to either file fails here.
    # T-0192 B re-pin (B1 ``ToolResult.hand_back``, B2 the hand-back row; B-RV6, both ends pinned): A0's 9dbddf96
    # blobs are the two files at T-0192 B's literal base da29ddbe (the base end), and in the working tree each is
    # exactly its merged blob at the implementation commit 97409910 (the tree end)
    from tests.unit.test_t0192_hand_back_machine import (
        T0192_BASE,
        T0192_BASE_SHA256,
        T0192_MOVED_SHA256,
        t0192_end_sha256,
    )

    gr.require_commits(T0181_BASE)
    for path, sha in T0181_PRIOR.items():
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{T0181_BASE}:{path}"],
            capture_output=True,
            check=True,
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == sha, (
            f"T-0181 A0-7: {path} at T-0181's base {T0181_BASE[:8]} is not its pre-A0 content"
        )
        assert T0192_BASE_SHA256[path] == T0181_MOVED[path], (
            f"T-0192 B: {path} at T-0192 B's base is A0's 9dbddf96 content"
        )
        assert t0192_end_sha256(path, T0192_BASE) == T0181_MOVED[path], (
            f"T-0192 B: {path} at T-0192 B's base {T0192_BASE[:8]} is not A0's 9dbddf96 content"
        )
        # T-0193 A0 re-pin (A0-9; A0-R5, both ends pinned): the T-0192 B tree end is the file at T-0193 A0's literal
        # base fdb9fc01 (the base end), and the working tree is exactly the merged 373df531 blob (the tree end)
        from tests.unit.test_t0193_stop_row import T0193_BASE, T0193_BASE_SHA256, T0193_MOVED_SHA256

        assert T0193_BASE_SHA256[path] == T0192_MOVED_SHA256[path]
        assert t0192_end_sha256(path, T0193_BASE) == T0192_MOVED_SHA256[path], (
            f"T-0181 A0-7 / T-0192 B: {path} at T-0193 A0's base {T0193_BASE[:8]} is not the merged 97409910 blob"
        )
        assert t0192_end_sha256(path) == T0193_MOVED_SHA256[path], (
            f"T-0193 A0-R5: {path} is not the merged 373df531 blob"
        )
    assert T0193_MOVED_SHA256["local_astra/conductor/turn_machine.py"] == TURN_MACHINE_SHA256, (
        "T-0162/T-0178/T-0182/T-0181/T-0192/T-0193: TURN_MACHINE_SHA256 is the re-pinned tree end (T-0193 A0: "
        "373df531's content)"
    )


def test_the_content_pins_are_the_base_blobs():
    """Provenance of ``PINNED``: each literal is the sha256 of the d3bc6ce blob (fixed blob against fixed literal)."""
    gr.require_commits(BASE_SHA)
    for path, sha in PINNED.items():
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{BASE_SHA}:{path}"], capture_output=True, check=True
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == sha, path


def test_tool_result_is_built_text_free():
    """Criterion 2, second item (T-0112 criterion 11): the session builds ``ToolResult`` with exactly ``t_ms``,
    ``call_id`` and ``status`` as keywords, and nothing else; the contract has no other field. Code only (an AST
    walk; prose cannot trip it).

    T-0192 B1/B4 re-pin (before implementation): the contract gains exactly ``hand_back`` (a closed
    ``Literal["checkout"] | None``, so still no text), and each build passes exactly ``t_ms``, ``call_id`` and
    ``status``, or those three plus ``hand_back``; never another keyword, never ``**``."""
    assert set(E.ToolResult.model_fields) == {"type", "t_ms", "call_id", "status", "hand_back"}
    built: list[set[str]] = []
    for name in ("session.py", "executor.py"):
        tree = ast.parse((REPO / "local_astra" / "conductor" / name).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if called != "ToolResult":
                continue
            assert not node.args, (
                f"{name}:{node.lineno}: ToolResult built with positional arguments"
            )
            built.append({k.arg or "**" for k in node.keywords})
    assert built, (
        "T-0114: the session never builds a ToolResult (it must submit one per executed call)"
    )
    allowed = ({"t_ms", "call_id", "status"}, {"t_ms", "call_id", "status", "hand_back"})
    bad = [k for k in built if k not in allowed]
    assert not bad, (
        f"T-0114 criterion 2 / T-0192 B4: ToolResult carries only t_ms, call_id and status (and hand_back); got {bad}"
    )


# =====================================================================================================
# Criterion 3: a two-step search turn
# =====================================================================================================


async def test_two_step_search_turn(tmp_path):
    """Criterion 3 (T-0111a 15): prefix voiced, FakeSearch asked exactly the query, one step-2 generate whose body
    continues the final message, the step-2 prompt equal to ``loop.step_prompt`` over the step-1 prompt as sent and
    the fixture rows, the report voiced, history ``"<prefix> <report>"``, one executed step, ended ``final``."""
    search = FakeSearch()
    async with tool_rig(
        tmp_path, policy=LOOP, search=search, script=[answer(PREFIX), tool(SEARCH_CALL)]
    ) as rig:
        rig.llm.script_steps([answer(REPORT)])
        await loop_turn(rig, 2)
        assert rig.spoken_sources() == PREFIX + REPORT, (
            f"T-0114 criterion 3: the prefix, then the report, are voiced; sources {rig.spoken_sources()!r}"
        )
        assert search.queries == [QUERY], f"FakeSearch logs exactly the query; got {search.queries}"
        s2 = generates_for(rig, "t1-gen-s2")
        assert len(s2) == 1 and s2[0][1].continue_final_message is True, (
            f"T-0114 criterion 3: one step-2 generate with continue_final_message=True; got {len(s2)}"
        )
        assert len(rig.llm.step_bodies) == 1
        body = rig.llm.step_bodies[0]
        assert (
            body.get("continue_final_message") is True
            and body.get("add_generation_prompt") is False
        ), {k: body.get(k) for k in ("continue_final_message", "add_generation_prompt")}
        pass1 = sent(rig, "t1-gen")
        pass1_body = next(
            b for b in rig.llm.pass1_bodies() if b["messages"] == body["messages"][:-1]
        )
        assert body["tools"] == pass1_body["tools"], "T-0114 criterion 3: tools as at step 1"
        assert body["chat_template_kwargs"] == pass1_body["chat_template_kwargs"], (
            "T-0114 criterion 3 (CLAUDE.md, prefix cache): thinking/template kwargs as at step 1"
        )
        step = step_request(rig, 2)
        assert step.messages[:-1] == pass1.messages, "messages[:-1] is the step-1 prompt as sent"
        assert tail(step).endswith("<tool_response|>"), tail(step)[-80:]
        want = oracle(
            pass1,
            [
                (
                    PREFIX,
                    "google_search",
                    SEARCH_ARGS,
                    loop.search_payload(results_of(DEFAULT_ROWS, QUERY)),
                )
            ],
        )
        assert tail(step) == tail(want), (
            f"T-0114 criterion 3: the final assistant text is form G over the fixture rows\n"
            f"  want {tail(want)!r}\n  got  {tail(step)!r}"
        )
        assert step == want, "T-0114 criterion 3: every other prompt field is the step-1 prompt's"
        assert rig.state.history[-1].assistant_spoken == f"{PREFIX} {REPORT}", rig.state.history[-1]
        extras = rig.record("t1").extras
    assert (
        extras.get("tool_steps_executed"),
        extras.get("tool_loop_end"),
        extras.get("tool_steps"),
    ) == (
        1,
        "final",
        2,
    ), (
        f"T-0114 criterion 3: the turn line; extras {{{', '.join(f'{k}={extras.get(k)!r}' for k in LOOP_KEYS)}}}"
    )


async def test_step_prompt_base_is_the_over_long_retry_as_sent(tmp_path):
    """Criterion 3 ("the step-1 prompt as sent") and reviewer mutation 6: after an over-long retry of step 1, the
    step-2 prompt continues the *retry's* prompt (one history turn dropped), never a rebuilt one."""
    async with tool_rig(tmp_path, policy=LOOP, script=[answer("A first answer here.")]) as rig:
        await rig.ask("tell me something first")
        await rig.turn_done()
        await rig.advance(300)
        rig.llm.script = [answer(S1), tool(VALID_CLICK)]
        rig.llm.script_steps([answer(S3)])
        rig.llm.fail_generates(http_error(413), None)
        await rig.ask("click the sign in button")
        await until_steps(rig, 1, turn="t2")
        await rig.until(lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) == 2, "t2")
        attempts = generates_for(rig, "t2-gen")
        assert len(attempts) == 2 and attempts[0][1].messages != attempts[1][1].messages, (
            "premise: the retry sent a different (shorter) prompt"
        )
        step = step_request(rig, 2, turn="t2")
        assert step.messages[:-1] == attempts[1][1].messages, (
            "T-0114: the step-2 prompt continues the step-1 prompt as sent after the over-long retry"
        )
        assert step == oracle(attempts[1][1], [(S1, "click_at", CLICK_ARGS, RECORDED_CLICK)])


# =====================================================================================================
# Criterion 4: the canary
# =====================================================================================================


async def test_canary_never_voiced_stored_or_published(tmp_path):
    """Criterion 4 (adversarial; T-0111a 16): a search snippet carrying ``ZQX-CANARY`` reaches the step-2 request
    and the ``tool.result`` record, and nothing that is voiced, published, stored or written to the turn line."""
    rows = [
        {"title": "Paris now", "url": "https://w.example.org/p", "snippet": f"{CANARY} says 14C"}
    ]
    search = FakeSearch(default=rows)
    async with tool_rig(
        tmp_path, policy=LOOP, search=search, script=[answer(PREFIX), tool(SEARCH_CALL)]
    ) as rig:
        rig.llm.script_steps([answer(REPORT)])
        await loop_turn(rig, 2)
        step = step_request(rig, 2)
        history = history_text(rig)
        episodes = repr(rig.memory.episodes)
    assert CANARY in tail(step), (
        "T-0114 criterion 4: the canary is inside the step-2 request's final text"
    )
    for c in rig.speaks():
        assert CANARY not in c.source and CANARY not in c.text, c
    leaks = {
        "astra.spoken/astra.spoken.played": published_json(
            rig, "astra.spoken", "astra.spoken.played"
        ),
        "astra.tool.call": published_json(rig, "astra.tool.call"),
        "history": history,
        "memory episodes": episodes,
        "turns.jsonl": "\n".join(rig.lines()),
    }
    bad = [where for where, text in leaks.items() if CANARY in text]
    assert not bad, f"T-0114 criterion 4: the canary leaked into {bad}"
    found = [r for r in rig.kind("tool.result") if r.get("call_id") == CALL_ID]
    assert len(found) == 1 and CANARY in json.dumps(found[0].get("response")), (
        f"T-0114 criterion 4: the canary is in the session log's tool.result.response; records {found}"
    )


# =====================================================================================================
# Criterion 5: thoughts are never replayed
# =====================================================================================================

THOUGHT_CASES: dict[str, tuple[list[Commit], tuple[list[Commit], ...], str]] = {
    # (a) a thought commit at step 1 and at step 2
    "a": (
        [thought(f"{SECRET} one"), answer(PREFIX), tool(SEARCH_CALL)],
        ([thought(f"{SECRET} two"), answer(S2), tool(CLICK2)], [answer(REPORT)]),
        S2,
    ),
    # (b) a thought_open=True answer commit (at step 1 and at step 2)
    "b": (
        [answer(PREFIX), open_answer(f"{SECRET} draft"), tool(SEARCH_CALL)],
        ([open_answer(f"{SECRET} again"), answer(S2), tool(CLICK2)], [answer(REPORT)]),
        S2,
    ),
    # (c) an answer commit carrying a whole thought block (no correct client sends it): at step 2, a withheld step,
    # so the guard is the session's own stripping before embedding (T-0113 note)
    "c": (
        [answer(PREFIX), tool(SEARCH_CALL)],
        ([answer(f"{S2} <|channel>thought {SECRET}<channel|>"), tool(CLICK2)], [answer(REPORT)]),
        S2,
    ),
}


@pytest.mark.parametrize("case", sorted(THOUGHT_CASES))
async def test_thought_never_replayed(tmp_path, case):
    """Criterion 5 (adversarial; H-T111a-5): no step request's final text carries ``SECRET-THOUGHT`` or a channel
    marker, and no ``TtsSpeak`` carries it. Three steps, so step 2's text is embedded too (in the step-3 request);
    the step text that is embedded still holds the visible words (not a vacuous drop of the whole step)."""
    script, steps, visible = THOUGHT_CASES[case]
    async with tool_rig(tmp_path, policy=LOOP, script=script) as rig:
        rig.llm.script_steps(*steps)
        await loop_turn(rig, 3)
        requests = [step_request(rig, 2), step_request(rig, 3)]
    for k, prompt in zip((2, 3), requests, strict=True):
        text = tail(prompt)
        bad = [s for s in (SECRET, "<|channel>", "<channel|>") if s in text]
        assert not bad, f"T-0114 criterion 5({case}): the step-{k} request replays {bad}: {text!r}"
    assert visible in tail(requests[1]), (
        f"T-0114 criterion 5({case}): step 2's visible text is embedded; got {tail(requests[1])!r}"
    )
    for c in rig.speaks():
        assert SECRET not in c.source and SECRET not in c.text, c


# =====================================================================================================
# Criterion 6: drain across steps
# =====================================================================================================


async def test_three_step_drain_marked_per_step(tmp_path):
    """Criterion 6 (adversarial; T-0111a 17): a turn that speaks at step 1 and step 3 marks each run's drain once
    (the step-3 drain is marked, so the turn reaches IDLE), stores ``"<s1> <s3>"``, and step 2's withheld text is in
    one ``note`` and in nothing voiced, published, stored or written to the turn line."""
    async with tool_rig(tmp_path, policy=LOOP, script=THREE_STEP["script"]) as rig:
        rig.llm.script_steps(*THREE_STEP["steps"])
        await loop_turn(rig, 3)
        assert rig.audio.calls.count("mark_drain") == 2, (
            f"T-0114 criterion 6: mark_drain once per spoken run (2); calls {rig.audio.calls}"
        )
        drained = rig.machine.events(E.PlaybackDrained)
        assert [d.played_text for d in drained] == [S1, S3], drained
        assert rig.state.history[-1].assistant_spoken == f"{S1} {S3}", rig.state.history[-1]
        history = history_text(rig)
    held = notes(rig, "step text withheld")
    assert [(n.get("step"), n.get("chars")) for n in held] == [(2, len(S2))], (
        f"T-0114 criterion 6: note {{what: 'step text withheld', step: 2, chars: {len(S2)}}}; got {held}"
    )
    elsewhere = {
        "TtsSpeak": json.dumps([c.model_dump(mode="json") for c in rig.speaks()]),
        "HUD": published_json(rig),
        "history": history,
        "turns.jsonl": "\n".join(rig.lines()),
        "the note": json.dumps(held),
    }
    bad = [where for where, text in elsewhere.items() if S2 in text]
    assert not bad, f"T-0114 criterion 6: step 2's withheld text is in {bad}"


# =====================================================================================================
# Criterion 7: a resume after a paused drain, and duplicate terminals
# =====================================================================================================

LONG_PREFIX = "Let me look that up for you in just a moment please."


async def test_paused_drain_resume_joins_prefix_once(tmp_path):
    """Criterion 7(a) (T-0112 carry-over): the drain-marked step-1 run is paused by an onset, its
    ``playback.drained`` arrives in PAUSED, a backchannel resumes; R11 runs once, step 2 runs, and history holds the
    prefix exactly once."""
    async with tool_rig(
        tmp_path, policy=LOOP, script=[answer(LONG_PREFIX), tool(SEARCH_CALL)]
    ) as rig:
        rig.llm.script_steps([answer(REPORT)])
        await rig.ask(QUESTION)
        await rig.until(
            lambda: (
                rig.state.llm_done
                and rig.state.state is TurnState.SPEAKING
                and "mark_drain" in rig.audio.calls
            ),
            "premise: the step-1 run is drain-marked while it plays",
            limit_ms=4_000,
        )
        rig.audio.hold_events()
        await rig.until(
            lambda: any(isinstance(e, E.PlaybackDrained) for e in rig.audio.pending_events()),
            "premise: the step-1 drain was emitted (held)",
            limit_ms=6_000,
        )
        assert rig.state.state is TurnState.SPEAKING
        now = rig.clock.now_ms
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        await rig.pump()
        assert rig.state.state is TurnState.PAUSED, rig.state.state
        rig.audio.release_events()
        await rig.pump()
        assert any(
            b is TurnState.PAUSED and isinstance(ev, E.PlaybackDrained)
            for b, ev, _ in rig.machine.transitions()
        ), "premise: the drain reached the machine in PAUSED"
        await rig.advance(100)
        rig.submit(E.UserBackchannel(t_ms=rig.clock.now_ms, duration_ms=200))
        await rig.pump()
        await until_steps(rig, 1)
        await rig.turn_done()
        executes = rig.commands(C.ToolExecute)
        assert [c.call_id for c in executes] == [CALL_ID], f"R11 runs once; ToolExecute {executes}"
        assert rig.state.history[-1].assistant_spoken == f"{LONG_PREFIX} {REPORT}", (
            f"T-0114 criterion 7(a): the prefix once, then the report; got {rig.state.history[-1]}"
        )


async def test_duplicate_drained_while_awaiting_is_dropped(tmp_path):
    """Criterion 7(b) (adversarial: prefix doubling): an unattributed ``playback.drained`` carrying the step-1 text,
    injected while the machine awaits the result and again while step 2 generates, never reaches ``machine.step``;
    each is noted once; history holds the prefix once."""
    loop_constants()
    executor = RecordingToolExecutor(hang=True)
    async with tool_rig(
        tmp_path, policy=LOOP, executor=executor, script=[answer(PREFIX), tool(SEARCH_CALL)]
    ) as rig:
        rig.llm.script_steps([answer(REPORT)])
        await rig.ask(QUESTION)
        await rig.until(
            lambda: bool(executor.invocations) and rig.state.awaiting_result == CALL_ID,
            "premise: the machine awaits the step-1 result",
            limit_ms=5_000,
        )
        dup1 = E.PlaybackDrained(t_ms=rig.clock.now_ms, played_ms=800, played_text=PREFIX)
        rig.submit(dup1)
        await rig.pump()
        rig.llm.hold_steps()
        executor.release()
        await until_steps(rig, 1)
        assert rig.state.active_generate == "t1-gen-s2", "premise: step 2 is generating"
        dup2 = E.PlaybackDrained(t_ms=rig.clock.now_ms, played_ms=800, played_text=PREFIX)
        rig.submit(dup2)
        await rig.pump()
        rig.llm.release_steps()
        await rig.turn_done()
        delivered = rig.machine.events(E.PlaybackDrained)
        assert not any(e is dup1 or e is dup2 for e in delivered), (
            "T-0114 criterion 7(b): an unattributed drained in the awaiting/step-2 window reached the machine"
        )
        assert rig.state.history[-1].assistant_spoken == f"{PREFIX} {REPORT}", rig.state.history[-1]
    steps = [
        r["data"]
        for r in rig.log_records
        if r["kind"] == "machine.step" and r["data"]["event"].get("type") == "playback.drained"
    ]
    assert [s["event"]["played_text"] for s in steps] == [PREFIX, REPORT], (
        f"T-0114 criterion 7(b): machine.step records for drains are the two real runs only; got {steps}"
    )
    dropped = [n for n in notes(rig) if "terminal" in str(n.get("what", ""))]
    assert len(dropped) == 2, (
        f"T-0114 criterion 7(b): each injected drained is noted once; notes {dropped}"
    )


async def test_own_step_drain_is_delivered(tmp_path):
    """Criterion 7(c) (the guard-too-wide twin): the step-3 run's own drain reaches the machine in SPEAKING and
    ends the turn; no drain of this turn is noted as dropped."""
    async with tool_rig(tmp_path, policy=LOOP, script=THREE_STEP["script"]) as rig:
        rig.llm.script_steps(*THREE_STEP["steps"])
        await loop_turn(rig, 3)
        last = [
            (b, ev, a)
            for b, ev, a in rig.machine.transitions()
            if isinstance(ev, E.PlaybackDrained)
        ]
        assert [(b, ev.played_text, a) for b, ev, a in last] == [
            (TurnState.SPEAKING, S1, TurnState.THINKING),
            (TurnState.SPEAKING, S3, TurnState.IDLE),
        ], f"T-0114 criterion 7(c): the two runs' drains are delivered; got {last}"
    dropped = [n for n in notes(rig) if "terminal" in str(n.get("what", ""))]
    assert dropped == [], f"T-0114 criterion 7(c): nothing of this turn is dropped; notes {dropped}"


async def test_report_joining_the_ack_clip_run_is_delivered(tmp_path):
    """Criterion 7 implementation note (interim filler; the guard-too-wide twin): a step 1 with no spoken sentence
    leaves the acknowledgement clip's run open (opened in THINKING at step 1); the step-2 report joins that run, and
    its drain, attributed to a run *not* started in step 2, still reaches the machine and ends the turn."""
    async with tool_rig(tmp_path, policy=LOOP, script=[tool(SEARCH_CALL)]) as rig:
        rig.llm.script_steps([answer(REPORT)])
        await rig.ask(QUESTION)
        await rig.until(
            lambda: bool(rig.executor.invocations), "premise: step 1 executed", limit_ms=4_000
        )
        assert rig.audio.clip_plays(), (
            "premise: the acknowledgement clip played at step 1 (a run opened)"
        )
        await until_steps(rig, 1)
        await rig.turn_done()
        drained = rig.machine.events(E.PlaybackDrained)
        assert [d.played_text for d in drained] == [REPORT], drained
        assert rig.state.history[-1].assistant_spoken == REPORT


# =====================================================================================================
# Criterion 8: per-step judging
# =====================================================================================================


async def test_step_call_ids_and_no_rejudge_of_step1_commit(tmp_path):
    """Criterion 8 (adversarial: re-judging a kept commit): a step-2 stream with no ``tool`` commit is judged as no
    call (step 1's kept click is not judged again), and step k >= 2 is judged under ``{turn}-call-s{k}``."""
    async with tool_rig(
        tmp_path / "two", policy=LOOP, script=[answer(PREFIX), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps([answer(REPORT)])
        await loop_turn(rig, 2)
        (done2,) = dones(rig, "t1-gen-s2")
        assert done2.tool is None, (
            f"T-0114 criterion 8: a toolless step 2 carries no call; got {done2.tool}"
        )
        assert [i.call_id for i in rig.executor.invocations] == [CALL_ID], rig.executor.invocations
        assert rig.spoken_sources() == PREFIX + REPORT
    async with tool_rig(tmp_path / "three", policy=LOOP, script=THREE_STEP["script"]) as rig:
        rig.llm.script_steps(*THREE_STEP["steps"])
        await loop_turn(rig, 3)
        (done2,) = dones(rig, "t1-gen-s2")
        (done3,) = dones(rig, "t1-gen-s3")
        assert done2.tool is not None and (done2.tool.call_id, done2.tool.status) == (
            "t1-call-s2",
            "valid",
        ), done2.tool
        assert done3.tool is None, (
            f"step 3 has no tool commit; step 2's is not judged again: {done3.tool}"
        )
        assert [(i.call_id, i.arguments) for i in rig.executor.invocations] == [
            (CALL_ID, CLICK_ARGS),
            ("t1-call-s2", CLICK2_ARGS),
        ], rig.executor.invocations
        assert final_message(rig, "t1-call-s2").status == "dry_run"


async def test_repairable_step2_refused_no_repair_request(tmp_path):
    """Criterion 8 (P5', H-T111a-8): a repairable step-2 draft reaches the machine ``refused`` with
    ``"repair: step 1 only"``; no ``tool.repair_start``, no tokenize, no repair request; the card ends refused with
    that reason; the loop ends ``refused``. T-0178 B3 re-scope: the step's text is withheld (only the step-1 prefix
    is voiced)."""
    said = "I could not find it."
    async with tool_rig(tmp_path, policy=LOOP, script=[answer(PREFIX), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(said), tool(REPAIRABLE_CLICK)])
        await loop_turn(rig, 2)
        (done2,) = dones(rig, "t1-gen-s2")
        assert done2.tool is not None and (
            done2.tool.call_id,
            done2.tool.status,
            done2.tool.reason,
        ) == (
            "t1-call-s2",
            "refused",
            "repair: step 1 only",
        ), f"T-0114 criterion 8: the step-2 draft as the machine sees it: {done2.tool}"
        assert rig.commands(C.ToolRepairStart) == []
        repairs = [
            p
            for _, p in rig.llm.gen_prompts
            if p.continue_final_message and tail(p).endswith("<|tool_call>")
        ]
        assert repairs == [] and rig.llm.tokenized == [], "no repair request and no tokenize"
        card = final_message(rig, "t1-call-s2")
        assert (card.status, card.reason) == ("refused", "repair: step 1 only"), card
        assert rig.spoken_sources() == PREFIX, rig.spoken_sources()  # T-0178 B3 (was PREFIX + said)
        assert said not in rig.spoken_sources()
        extras = rig.record("t1").extras
    assert extras.get("tool_loop_end") == "refused", extras


async def test_withheld_text_noted_not_voiced(tmp_path):
    """Criterion 8, last item: a valid step-2 call with budget left writes the withheld-text note, and no
    ``TtsSpeak`` comes from step 2 (none is emitted between the step-2 request and the step-3 request)."""
    async with tool_rig(tmp_path, policy=LOOP, script=THREE_STEP["script"]) as rig:
        rig.llm.script_steps(*THREE_STEP["steps"])
        await loop_turn(rig, 3)
        log = rig.machine.log
        start = next(
            i
            for i, (_, _, cmds) in enumerate(log)
            if any(isinstance(c, C.LlmContinue) for c in cmds)
        )
        later = [
            i
            for i, (_, _, cmds) in enumerate(log)
            if any(isinstance(c, C.LlmContinue) for c in cmds)
        ][1]
        between = [
            c for _, _, cmds in log[start : later + 1] for c in cmds if isinstance(c, C.TtsSpeak)
        ]
        assert between == [], f"T-0114 criterion 8: step 2 voices nothing; got {between}"
        assert all(S2 not in c.source for c in rig.speaks())
    held = notes(rig, "step text withheld")
    assert [(n.get("step"), n.get("chars")) for n in held] == [(2, len(S2))], held


# =====================================================================================================
# Criterion 9: results on every path
# =====================================================================================================


def _refusing_outcome() -> dict[str, Any]:
    return {
        CALL_ID: ToolOutcome(
            call_id=CALL_ID, status="refused", action="", reason="no frame", resolved={}
        )
    }


#: case -> (call, name, args, executor factory, search factory, ToolResult status, payload or None = read reason)
RESULT_CASES: dict[str, tuple[str, str, Any, Any, Any, str, Any]] = {
    "dry_run_click": (VALID_CLICK, "click_at", CLICK_ARGS, RecordingToolExecutor, None, "dry_run", RECORDED_CLICK),
    "search_none": (SEARCH_CALL, "google_search", SEARCH_ARGS, RecordingToolExecutor, None, "dry_run",
                    loop.search_payload(None)),
    "search_backend": (SEARCH_CALL, "google_search", SEARCH_ARGS, RecordingToolExecutor, FakeSearch, "executed",
                       loop.search_payload(results_of(DEFAULT_ROWS, QUERY))),
    "backend_raises": (SEARCH_CALL, "google_search", SEARCH_ARGS, RecordingToolExecutor, RaisingSearch, "failed",
                       None),
    "executor_raises": (VALID_CLICK, "click_at", CLICK_ARGS,
                        lambda: RecordingToolExecutor(raise_=RuntimeError("the executor blew up")), None, "refused",
                        loop.status_payload("refused", "executor error")),
    "executor_refuses": (VALID_CLICK, "click_at", CLICK_ARGS, lambda: RecordingToolExecutor(_refusing_outcome()),
                         None, "refused", loop.status_payload("refused", "no frame")),
}  # fmt: skip


def _reason_in(text: str, name: str, status: str) -> str:
    """The reason of a ``status_payload(status, reason)`` block ending ``text`` (the task leaves it unspecified)."""
    m = re.search(
        rf'<\|tool_response>response:{name}\{{reason:<\|"\|>(.*?)<\|"\|>,status:<\|"\|>{status}<\|"\|>\}}'
        r"<tool_response\|>$",
        text,
        flags=re.S,
    )
    assert m is not None and m.group(1), (
        f"T-0114 criterion 9/10: the step text ends in status_payload({status!r}, <reason>); tail {text[-200:]!r}"
    )
    return m.group(1)


@pytest.mark.parametrize("case", list(RESULT_CASES))
async def test_result_payload_per_path(tmp_path, case):
    """Criterion 9 (T-0111a 19, §4): exactly one ``ToolResult`` per executed call with the path's status, one
    ``tool.result {call_id, step, status, response}`` record, and the next step's request carrying the matching
    rendered block (the whole step-2 prompt equals the oracle)."""
    call, name, args, make_executor, make_search, status, payload = RESULT_CASES[case]
    search = make_search() if make_search is not None else None
    async with tool_rig(
        tmp_path,
        policy=LOOP,
        executor=make_executor(),
        search=search,
        script=[answer(PREFIX), tool(call)],
    ) as rig:
        rig.llm.script_steps([answer(REPORT)])
        await loop_turn(rig, 2)
        got = results_for(rig, CALL_ID)
        assert [e.status for e in got] == [status], (
            f"T-0114 criterion 9 ({case}): one ToolResult({status!r}); got {got}"
        )
        step = step_request(rig, 2)
        if payload is None:
            payload = loop.status_payload("failed", _reason_in(tail(step), name, "failed"))
        want = oracle(sent(rig, "t1-gen"), [(PREFIX, name, args, payload)])
        assert tail(step) == tail(want), (
            f"T-0114 criterion 9 ({case}): the step-2 text\n  want {tail(want)!r}\n  got  {tail(step)!r}"
        )
    records = [r for r in rig.kind("tool.result") if r.get("call_id") == CALL_ID]
    assert len(records) == 1, f"T-0114 criterion 9 ({case}): one tool.result record; got {records}"
    rec = records[0]
    assert (rec.get("step"), rec.get("status")) == (1, status), rec
    response = rec.get("response")
    as_json = json.loads(json.dumps(payload))
    assert response == loop.render_response(name, payload) or response == as_json, (
        f"T-0114 criterion 9 ({case}): tool.result.response is the rendered block or the payload; got {response!r}"
    )


# =====================================================================================================
# Criterion 10: timeouts and errors
# =====================================================================================================


async def test_hanging_executor_times_out_on_session_clock(tmp_path):
    """Criterion 10: a hanging executor is cancelled ``TOOL_EXEC_TIMEOUT_MS`` after it started, on the session clock
    (a wall-time wait would never end inside the virtual budget below); ``ToolResult(timeout)`` follows; step 2's
    response is ``status_payload("timeout", ...)``; R18 never fires (no ``hud.error``)."""
    consts = loop_constants()
    assert consts == DEFAULT_BOUNDS, f"H-T114-4 defaults: {consts}"
    inner = RecordingToolExecutor(hang=True)
    executor = WatchingExecutor(inner=inner)
    async with tool_rig(
        tmp_path, policy=LOOP, executor=executor, script=[answer(PREFIX), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps([answer(REPORT)])
        await rig.ask(QUESTION)
        await rig.until(lambda: bool(executor.times), "premise: the call executed", limit_ms=5_000)
        await until_steps(rig, 1, limit_ms=12_000, step=250)
        await rig.turn_done()
        (result,) = results_for(rig, CALL_ID)
        waited = result.t_ms - executor.times[0]
        assert result.status == "timeout" and 10_000 <= waited <= 10_050, (
            f"T-0114 criterion 10: ToolResult(timeout) {consts['TOOL_EXEC_TIMEOUT_MS']} ms after the start on the "
            f"session clock; got status {result.status!r} after {waited} ms"
        )
        assert inner.cancelled == [CALL_ID], f"the hanging call is cancelled; {inner.cancelled}"
        assert rig.commands(C.HudError) == [], "R18 never fires (and no other error row)"
        reason = _reason_in(tail(step_request(rig, 2)), "click_at", "timeout")
        want = oracle(
            sent(rig, "t1-gen"),
            [(PREFIX, "click_at", CLICK_ARGS, loop.status_payload("timeout", reason))],
        )
        assert step_request(rig, 2) == want
        assert rig.state.history[-1].assistant_spoken == f"{PREFIX} {REPORT}"


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(lambda: http_error(413), id="step_413_not_retried"),
        pytest.param(lambda: httpx.ConnectError("connection refused"), id="step_transport_error"),
    ],
)
async def test_step_request_error_ends_the_turn(tmp_path, error):
    """Criterion 10: a 413 on a step request is not retried (one generate for that step id) and a transport error
    likewise; the turn ends with ``hud.error``, history holds the played run (T-0112 F3), the loop ends ``error``.
    (The two rows the task names ``test_step_413_not_retried`` and ``test_step_transport_error``.)"""
    async with tool_rig(tmp_path, policy=LOOP, script=[answer(PREFIX), tool(VALID_CLICK)]) as rig:
        rig.llm.script_steps([answer(REPORT)])
        rig.llm.fail_steps(error())
        await rig.ask(QUESTION)
        await rig.until(
            lambda: bool(generates_for(rig, "t1-gen-s2")), "the step-2 request was never sent"
        )
        await rig.turn_done()
        assert len(generates_for(rig, "t1-gen-s2")) == 1, "T-0114 criterion 10: not retried"
        assert len(rig.llm.step_prompts) == 1
        assert rig.commands(C.HudError), "T-0114 criterion 10: the turn ends with hud.error"
        assert rig.state.history[-1].assistant_spoken == PREFIX, rig.state.history[-1]
        assert REPORT not in rig.spoken_sources()
        extras = rig.record("t1").extras
    assert extras.get("tool_loop_end") == "error", extras


# =====================================================================================================
# Criterion 11: the continuation bound
# =====================================================================================================

LONG_S1 = "word " * 240  # 1200 chars, voiced
LONG_STEP = "w" * 1200  # withheld
WORST_STEPS = 8
_SEGMENT = re.compile(
    r"(<\|tool_call>.*?<tool_call\|>)(<\|tool_response>.*?<tool_response\|>)", flags=re.S
)


def _worst_outcomes() -> dict[str, Any]:
    return {
        call_id(k): ToolOutcome(
            call_id=call_id(k), status="dry_run", action="A" * 3000, reason=None, resolved={}
        )
        for k in range(1, WORST_STEPS + 2)
    }


async def _worst_case(tmp_path: Path) -> ToolRig:
    """8 maximal steps: a 1 200-char step text each (the voiced step 1 too), a type_at with a 5 000-char text, a
    3 000-char action (rendered to a 1 500-char response); ms_per_word 1 so the long prefix plays in 240 ms."""
    rig_cm = tool_rig(
        tmp_path,
        policy=loop_policy(WORST_STEPS),
        executor=RecordingToolExecutor(_worst_outcomes()),
        ms_per_word=1,
        script=[answer(LONG_S1), tool(TYPE_CALL)],
    )
    async with rig_cm as rig:
        rig.llm.script_steps(*[[answer(LONG_STEP), tool(TYPE_CALL)] for _ in range(WORST_STEPS)])
        await rig.ask("type it all")
        await until_steps(rig, 1)
        await rig.turn_done(limit_ms=30_000)
        rig.cards = final_message(rig, CALL_ID)  # type: ignore[attr-defined]
        rig.sources = rig.spoken_sources()  # type: ignore[attr-defined]
    return rig


def _bounded_size(steps: int, consts: dict[str, int]) -> int:
    """The continuation of ``steps`` worst-case steps with every text and argument cut to its bound (``[:N]``)."""
    cut_text = LONG_STEP[: consts["MAX_STEP_TEXT"]]
    args = {"press_enter": False, "text": TYPE_TEXT[: consts["MAX_ARG_TEXT"]], "x": 500, "y": 250}
    one = (
        cut_text
        + loop.render_call("type_at", args)
        + loop.render_response("type_at", loop.action_payload("A" * 3000))
    )
    return len(one) * steps


async def test_continuation_bounds_worst_case(tmp_path):
    """Criterion 11: every step text is cut to ``MAX_STEP_TEXT`` and every string argument to ``MAX_ARG_TEXT`` before
    it is embedded, no request carries a continuation over ``MAX_CONTINUATION``, and the voiced prefix and the card
    keep the full text. (On the arithmetic of the task's "8 maximal steps fit", see the report.)"""
    consts = loop_constants()
    assert consts == DEFAULT_BOUNDS, f"H-T114-4 defaults: {consts}"
    rig = await _worst_case(tmp_path)
    sent_steps = steps_of(rig)
    assert sent_steps, "at least the step-2 request was sent"
    for rid, prompt in sent_steps:
        text = tail(prompt)
        assert len(text) <= consts["MAX_CONTINUATION"], (rid, len(text))
        parts = _SEGMENT.split(text)
        texts, calls, responses = parts[0::3], parts[1::3], parts[2::3]
        assert texts[-1] == "" and len(calls) == len(responses) >= 1, (rid, len(calls))
        for t in texts[:-1]:
            assert 0 < len(t) <= consts["MAX_STEP_TEXT"], (rid, len(t))
        for c in calls:
            typed = re.search(r'text:<\|"\|>(.*?)<\|"\|>', c, flags=re.S)
            assert typed is not None and 0 < len(typed.group(1)) <= consts["MAX_ARG_TEXT"], (
                rid,
                c[:80],
            )
        for r in responses:
            assert len(r) <= loop.MAX_RESPONSE, (rid, len(r))
    first = _SEGMENT.split(tail(step_request(rig, 2)))[0]
    assert (
        LONG_S1.startswith(first.rstrip("…"))
        and len(first.rstrip("…")) >= consts["MAX_STEP_TEXT"] // 2
    ), f"the step-1 text is cut, not rewritten: {first[:60]!r}…"
    assert rig.sources.startswith(LONG_S1.strip()), "the voiced prefix is the full text"  # type: ignore[attr-defined]
    assert rig.cards.arguments.get("text") == TYPE_TEXT, "the card shows the full type_at text"  # type: ignore[attr-defined]
    assert rig.record("t1").extras.get("tool_loop_end") in ("budget", "over_budget")


#: T-0120b criterion 6: the old-note status stub a type_at response becomes, hand-typed (16 + 17 + 5 + 5 + 22 + 5 + 1
#: + 7 + 5 + 7 + 5 + 1 + 16 = 112 chars)
TYPE_AT_STUB = (
    '<|tool_response>response:type_at{note:<|"|>earlier result omitted<|"|>,status:<|"|>dry_run<|"|>}'
    "<tool_response|>"
)


async def test_over_budget_ends_loop_and_publishes(tmp_path):
    """Criterion 11 (``.claude/rules/conductor.md``, a self-disabling subsystem publishes), re-scoped by T-0120b
    criterion 6: the session compacts before the size check, and the turn ends ``over_budget`` at the first step whose
    *compacted* continuation does not fit.

    Premise, from the bounds and the rendered lengths: a bounded worst step is ``full`` chars, an elided one ``el =
    full - 1 500 + 112``; ``keep_last`` 2, target 14 000 * 2 // 3. After 4 executed steps 4 * full fits; after 5,
    two elisions are still over the target, so candidates 0-2 go (3 el + 2 full fits: s6 sent, compaction 1); after 6,
    3 el + 3 full is over, candidate 3 goes (4 el + 2 full fits: s7 sent, compaction 2); after 7, 4 el + 3 full is
    over and eliding candidate 4 leaves 5 el + 2 full still over (compaction 3): s8 is not sent and
    ``LlmError("t1-gen-s8", "tool loop: continuation over budget")`` reaches the machine; a note is written, the HUD
    is told, the line says ``over_budget`` and ``tool_compactions == 3``. Every inequality holds by at least 100
    chars, so the voiced step-1 text's exact cut does not move it."""
    consts = loop_constants()
    big = consts["MAX_CONTINUATION"]
    assert _bounded_size(4, consts) <= big < _bounded_size(5, consts), (
        "premise: 4 bounded steps fit, 5 do not",
        _bounded_size(4, consts),
        _bounded_size(5, consts),
    )
    response = loop.render_response("type_at", loop.action_payload("A" * 3000))
    assert len(response) == loop.MAX_RESPONSE == 1500 and len(TYPE_AT_STUB) == 112
    full = _bounded_size(1, consts)
    el = full - len(response) + len(TYPE_AT_STUB)
    target = big * 2 // 3
    fits = [  # (size, must fit) in the order the premise names them
        (4 * full, True),
        (5 * full - 2 * (full - el), False),  # vs the target, below
        (3 * el + 2 * full, True),
        (3 * el + 3 * full, False),
        (4 * el + 2 * full, True),
        (4 * el + 3 * full, False),
        (5 * el + 2 * full, False),
    ]
    assert 5 * full - 2 * (full - el) - target >= 100, (full, el)
    for size, ok in fits[:1] + fits[2:]:
        assert (big - size >= 100) if ok else (size - big >= 100), (size, ok, full, el)
    rig = await _worst_case(tmp_path)
    assert [rid for rid, _ in steps_of(rig)] and [
        k for k in range(2, 10) if generates_for(rig, step_id(k))
    ] == [2, 3, 4, 5, 6, 7], f"step requests sent: {[rid for rid, _ in rig.llm.step_prompts]}"
    assert rig.record("t1").extras.get("tool_compactions") == 3, rig.record("t1").extras
    errors = [e for e in rig.machine.events(E.LlmError) if e.request_id == "t1-gen-s8"]
    assert [e.message for e in errors] == ["tool loop: continuation over budget"], errors
    over = [n for n in notes(rig) if "budget" in json.dumps(n)]
    assert over, (
        f"T-0114 criterion 11: a note says the continuation was over budget; notes {notes(rig)}"
    )
    assert any(getattr(m, "type", "") == "astra.error" for m in rig.published.messages), (
        "T-0114 criterion 11: the loop publishes that it switched itself off"
    )
    assert rig.record("t1").extras.get("tool_loop_end") == "over_budget"


# =====================================================================================================
# Criterion 12: cuts
# =====================================================================================================

P2 = "Clicking it now."
R2 = "Clicked."


async def test_takeover_while_awaiting_cancels_and_ignores_late_result(tmp_path):
    """Criterion 12 (adversarial: a late result leaking into the next turn): a takeover while awaiting cancels the
    execute task; the result that arrives later (a misbehaving executor) is noted stale and never submitted; the
    new turn's ``llm.start`` lacks the first request's words; its step prompt holds nothing of turn 1's steps."""
    loop_constants()
    executor = RecordingToolExecutor(hang={CALL_ID}, shrug_cancel=True)
    async with tool_rig(
        tmp_path, policy=LOOP, executor=executor, script=[answer(S1), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps([answer(R2)])
        await rig.ask(QUESTION)
        await rig.until(
            lambda: bool(executor.invocations) and rig.state.awaiting_result == CALL_ID,
            "premise: awaiting the step-1 result",
            limit_ms=5_000,
        )
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.pump()
        assert rig.state.turn_id == "t2", "premise: R17 opened turn 2"
        assert CALL_ID in executor.cancelled, "T-0114 criterion 12: the execute task is cancelled"
        executor.release()
        await rig.pump()
        assert CALL_ID in executor.returned, "premise: the late result arrived"
        rig.llm.script = [answer(P2), tool(CLICK2)]
        await rig.words("and then")
        await rig.end_of_turn()
        await until_steps(rig, 1, turn="t2")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= 2, "t2 ended"
        )
        assert results_for(rig, CALL_ID) == [], "the late result is never submitted"
        starts = [c.transcript for c in rig.commands(C.LlmStart)]
        assert starts == [QUESTION, "and then"], (
            f"T-0114 criterion 12: llm.start transcripts {starts}"
        )
        want = oracle(sent(rig, "t2-gen"), [(P2, "click_at", CLICK2_ARGS, RECORDED_CLICK)])
        assert step_request(rig, 2, turn="t2") == want, (
            "turn 2's step prompt holds only turn 2's step"
        )
        assert rig.record("t1").extras.get("tool_loop_end") == "takeover"
    stale = [
        n for n in notes(rig) if "stale" in str(n.get("what", "")) and CALL_ID in json.dumps(n)
    ]
    assert stale, f"T-0114 criterion 12: the late t1-call result is noted stale; notes {notes(rig)}"


async def test_interrupt_in_report_step(tmp_path):
    """Criterion 12: an interrupt during the step-3 report ends the turn with no further generate for it."""
    long_report = "Done, the settings are open now and you can change the display options there."
    async with tool_rig(tmp_path, policy=LOOP, script=THREE_STEP["script"]) as rig:
        rig.llm.script_steps(THREE_STEP["steps"][0], [answer(long_report)])
        await rig.ask(QUESTION)
        await until_steps(rig, 2)
        await rig.until(
            lambda: rig.state.state is TurnState.SPEAKING and rig.state.tool_step == 3,
            "premise: the step-3 report is playing",
            limit_ms=4_000,
        )
        await rig.advance(400)
        now = rig.clock.now_ms
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        rig.submit(E.UserTakeover(t_ms=now + 1, onset_ms=now))
        await rig.until(lambda: rig.state.turn_id == "t2", "the interrupted turn never closed")
        await rig.advance(2_000)
        t1 = [rid for rid, _, _ in rig.llm.generates if re.search(r"-t1-(gen|repair)", rid)]
        assert len(t1) == 3, (
            f"T-0114 criterion 12: no generate after the interrupt; t1 generates {t1}"
        )
        assert rig.state.history[-1].assistant_spoken.endswith("[interrupted]"), rig.state.history[
            -1
        ]
        assert rig.record("t1").extras.get("tool_loop_end") == "interrupted"


async def test_disconnect_while_awaiting(tmp_path):
    """Criterion 12: a disconnect while awaiting a result cancels the execute task and ends the turn with no step
    request; history keeps the played run (T-0112 F3)."""
    loop_constants()
    executor = RecordingToolExecutor(hang=True)
    async with tool_rig(
        tmp_path, policy=LOOP, executor=executor, script=[answer(PREFIX), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps([answer(REPORT)])
        await rig.ask(QUESTION)
        await rig.until(
            lambda: bool(executor.invocations) and rig.state.awaiting_result == CALL_ID,
            "premise: awaiting the step-1 result",
            limit_ms=5_000,
        )
        rig.submit(E.ClientDisconnected(t_ms=rig.clock.now_ms))
        await rig.turn_done()
        await rig.advance(12_000, step=500)
        assert generates_for(rig, "t1-gen-s2") == [] and rig.llm.step_prompts == []
        assert executor.cancelled == [CALL_ID], (
            f"the execute task is cancelled; {executor.cancelled}"
        )
        assert rig.state.history[-1].assistant_spoken == PREFIX, rig.state.history[-1]
        assert rig.record("t1").extras.get("tool_loop_end") == "disconnected"


# =====================================================================================================
# Criterion 13: per-step bookkeeping
# =====================================================================================================


async def test_turn_line_and_spans_keep_step1(tmp_path):
    """Criterion 13: ``TurnRecord.tool`` is the step-1 call's last record; ``llm.done``, ``tool.execute`` and
    ``llm.first_canvas`` keep step 1's values; no unknown span; one ``llm.request {phase: tool_step}`` per step >= 2
    with the prompt sent; the step-1 record keeps its keys exactly; the three turn-line keys."""
    executor = WatchingExecutor()
    async with tool_rig(
        tmp_path, policy=LOOP, executor=executor, script=THREE_STEP["script"]
    ) as rig:
        rig.llm.script_steps(*THREE_STEP["steps"])
        await loop_turn(rig, 3)
        (done1,) = dones(rig, "t1-gen")
        first = next(e for e in rig.machine.events(E.LlmCommit) if e.request_id == "t1-gen")
        prompts = {k: step_request(rig, k) for k in (2, 3)}
        engine = {k: generates_for(rig, step_id(k))[0][0] for k in (2, 3)}
        rec = rig.record("t1")
    assert rec.tool is not None and (rec.tool.get("call_id"), rec.tool.get("status")) == (
        CALL_ID,
        "dry_run",
    ), f"T-0114 criterion 13: TurnRecord.tool stays the step-1 call's; got {rec.tool}"
    assert rec.spans["llm.done"][1] == done1.t_ms, (rec.spans["llm.done"], done1.t_ms)
    assert rec.spans["llm.first_canvas"][1] == first.t_ms, (
        rec.spans["llm.first_canvas"],
        first.t_ms,
    )
    assert rec.spans["tool.execute"][0] == executor.times[0], (
        rec.spans["tool.execute"],
        executor.times,
    )
    assert rec.unknown_spans() == [], rec.unknown_spans()
    requests = rig.kind("llm.request")
    main = [r for r in requests if r.get("request_id") == "t1-gen"]
    assert len(main) == 1 and set(main[0]) == MAIN_REQUEST_KEYS, [sorted(r) for r in main]
    steps = [r for r in requests if r.get("phase") == "tool_step"]
    assert [(r.get("step"), r.get("request_id")) for r in steps] == [
        (2, "t1-gen-s2"),
        (3, "t1-gen-s3"),
    ], steps
    for r in steps:
        k = r["step"]
        assert {"phase", "step", "request_id", "attempt_id", "prompt"} <= set(r), sorted(r)
        # T-0114 re-scope: SessionLog.record always redacts image parts (T-0043 criterion 5) and the rig attaches
        # a frame, so the logged prompt is the sent prompt through the log's own ``redact``.
        assert r["attempt_id"] == engine[k] and r["prompt"] == session_log.redact(
            prompts[k].model_dump(mode="json")
        ), k
    assert (
        rec.extras.get("tool_steps"),
        rec.extras.get("tool_steps_executed"),
        rec.extras.get("tool_loop_end"),
    ) == (
        3,
        2,
        "final",
    ), {k: rec.extras.get(k) for k in LOOP_KEYS}
    assert rec.extras["tool_loop_end"] in LOOP_ENDS


# T-0178 B2 re-scope of R16: the third step's text is withheld (was: voiced)
async def test_budget_two_calls_every_step(tmp_path):
    """Criterion 13 (R16 at session level): ``max_tool_steps=2`` and a model that calls on every step give 3
    generates, two executions, the third call cancelled ``step_budget``, ``budget``. T-0178 B2 re-scope: the third
    step's text is withheld (only S1 is voiced)."""
    async with tool_rig(
        tmp_path, policy=loop_policy(2), script=[answer(S1), tool(VALID_CLICK)]
    ) as rig:
        rig.llm.script_steps([answer(S2), tool(CLICK2)], [answer(S3), tool(CLICK3)])
        await loop_turn(rig, 3)
        t1 = [rid for rid, _, _ in rig.llm.generates if re.search(r"-t1-(gen|repair)", rid)]
        assert len(t1) == 3, t1
        assert [i.call_id for i in rig.executor.invocations] == [CALL_ID, "t1-call-s2"]
        cancels = [(c.call_id, c.reason) for c in rig.commands(C.ToolCancel)]
        assert cancels == [("t1-call-s3", "step_budget")], cancels
        assert rig.spoken_sources() == S1, rig.spoken_sources()  # T-0178 B2 (was S1 + S3)
        card = final_message(rig, "t1-call-s3")
        assert (card.status, card.reason) == ("cancelled", "step_budget"), card
        extras = rig.record("t1").extras
    assert (extras.get("tool_steps_executed"), extras.get("tool_loop_end")) == (2, "budget"), extras
