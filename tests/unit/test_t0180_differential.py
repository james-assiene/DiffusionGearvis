"""T-0180 B8 (negative): publishing ``astra.thought`` changes nothing the model sees or hears
(docs/tasks/T-0180.md, group B).

**Method.** A differential over ``BASE..TIP``: ``BASE`` is the literal main commit group B merged from (c03fdfda, T-0177
B+C merged, B-R9; it holds the group B rulings and T-0180 A, publishes ``astra.vision.frame`` and no thoughts), run
in a subprocess over ``git archive BASE`` (T-0114's ``_base_tree``, ``BASE_SHA`` monkeypatched; the origin of ``local_astra`` is checked);
``TIP`` is the working tree, in-process. The only thing ignored is ``astra.thought`` itself: the published messages of
that type, and session-log ``wire.out`` rows whose ``type`` is ``astra.thought`` (the rigs publish to a ``Recorder``,
so no ``wire.out`` row is written at all; the filter is there for the day a rig routes through ``HudServer``).

**Regimes** (B8's four, each a session over the fakes):

- ``tools_off``: two tools-off turns that think and answer (the second with history);
- ``dry_run_loop``: T-0172's re-ask rig, ``dry_run`` loop: step 1 thinks and calls search, step 2 thinks and only
  announces, the re-ask thinks and answers (two steps and a re-ask);
- ``live_retry``: T-0179's live t10 shape: steps 2 and 3 think, step 4's thought trips the T-0142 guard and is retried
  (T-0179), the retry thinks and clicks, the last step answers;
- ``barge_in``: tools off, the stream held after one thought commit, a barge-in (onset, takeover), then a second turn
  that thinks and answers.

**Byte-identical sections** (B8's list): the machine's events, states and commands (so ``astra.spoken``'s source,
``TtsSpeak`` and ``CommitHistory``); every published message but ``astra.thought`` (``astra.spoken``,
``astra.spoken.played``, tool calls, transcripts, turn state, canvases); every prompt handed to the backend (the
system prompt bytes are its first message) and every request body the real ``VllmClient`` builds; the session log
(every record but ``wire.out`` for ``astra.thought``); ``turns.jsonl``; history; the TTS requests; the memory store.

**Anti-vacuity.** On the TIP side each regime must have published ``astra.thought`` with the kind/end the regime
exercises (``tools_off``: ``step``/``done`` under ``t1`` and ``t2``; ``dry_run_loop``: a ``reask`` block;
``live_retry``: ``stopped_repeating`` and a ``retry`` block; ``barge_in``: ``interrupted``), and on the BASE side none,
so the rows are red until the session publishes, and a TIP that published nothing cannot pass vacuously.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, n = 1 per regime,
deterministic. No measurement.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import events as E
from local_astra.contracts.turn import TurnState
from tests import _git_range as gr
from tests.fakes.search import FakeSearch
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0142_session as t42
from tests.unit import test_t0172_session_reask as t72
from tests.unit import test_t0179_degenerate_retry as t79
from tests.unit._tool_turn_support import answer, thought, tool, tool_rig
from tests.unit.test_session_tools_off_golden import normalise

#: B8's BASE (B-R9): main at c03fdfda (T-0177 B+C and T-0181 A0, on top of 18048239's group B rulings and the
#: T-0028/T-0037/T-0039 amendments), merged into ``T-0180-b-impl`` at b891ee5a. It already publishes
#: ``astra.vision.frame``, so B8 still isolates ``astra.thought``. Was 18048239 (merged at bf06407e). A literal, never
#: a moving ref (tests.md).
BASE_SHA = "c03fdfdaa96c6d87a2e0b552578bc5d4000613f5"

THOUGHT = "astra.thought"
REGIMES = ("tools_off", "dry_run_loop", "live_retry", "barge_in")
SECTIONS = (
    "machine",
    "hud",
    "prompts",
    "bodies",
    "log",
    "records",
    "history",
    "tts",
    "memory",
)


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _is_thought_row(record: dict[str, Any]) -> bool:
    data = record.get("data") or {}
    return (
        record.get("kind") == "wire.out" and isinstance(data, dict) and data.get("type") == THOUGHT
    )


def _capture_rig(rig: Any, tmp: Path) -> dict[str, Any]:
    """Every B8 section of one finished rig (after ``aclose``), normalised for session id, tmp dir and log name.
    ``thoughts`` (the ignored messages, summarised) is kept apart for the premises."""
    session_id, log_path = rig.session.session_id, Path(rig.log.path)
    messages = [_dump(m) for m in rig.published.messages]
    per_process = {"log.started": {"pid": "<pid>"}, "log.stopped": {"bytes": "<bytes>"}}
    sections: dict[str, Any] = {
        "machine": [
            {"event": _dump(ev), "state": str(state), "commands": [_dump(c) for c in cmds]}
            for ev, state, cmds in rig.machine.log
        ],
        "hud": [m for m in messages if m.get("type") != THOUGHT],
        "prompts": [
            {"phase": "prefill", "request_id": r, "prompt": _dump(p)} for r, p in rig.llm.prefills
        ]
        + [
            {"phase": "generate", "request_id": r, "prompt": _dump(p), "options": _dump(o)}
            for r, p, o in rig.llm.generates
        ],
        "bodies": [_dump(b) for b in rig.llm.bodies],
        "log": [
            {"kind": r["kind"], "data": {**r["data"], **per_process.get(r["kind"], {})}}
            for r in rig.log_records
            if not _is_thought_row(r)
        ],
        "records": [json.loads(line) for line in rig.lines()],
        "history": [_dump(h) for h in rig.state.history],
        "tts": [list(r) for r in rig.tts.requests],
        "memory": [
            {"id": k, "episode": _dump(v)} for k, v in sorted(rig.session.memory.episodes.items())
        ],
        "thoughts": [
            [m["turn_id"], m["step"], m["kind"], m["end"], m["text"]]
            for m in messages
            if m.get("type") == THOUGHT
        ],
    }
    out: dict[str, Any] = normalise(sections, session_id, tmp, log_path)
    for r in out["log"]:
        regime = r["data"].get("regime") if isinstance(r["data"], dict) else None
        if isinstance(regime, dict) and "git_commit" in regime:
            regime["git_commit"] = "<git>"
    return out


async def _turn_over(rig: Any, n: int, what: str) -> None:
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= n,
        f"T-0180 B8: {what} never ended",
        limit_ms=40_000,
    )
    await rig.advance(300)


async def _tools_off(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, kit=None) as rig:
        rig.llm.script = [
            thought("ZQX-B8-OFF-1 the page has a cart. ", 0),
            thought("It holds two items.", 1, open_=True),
            answer("It is a shopping cart with two items. ", 2),
        ]
        await rig.ask("what is on this screen")
        await _turn_over(rig, 1, "the first tools-off turn")
        rig.llm.script = [thought("ZQX-B8-OFF-2 the total. ", 0), answer("The total is ten. ", 1)]
        await rig.ask("and the total")
        await _turn_over(rig, 2, "the second tools-off turn")
    return _capture_rig(rig, tmp)


async def _dry_run_loop(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with t72.reask_rig(
        tmp,
        search=FakeSearch(),
        script=[thought("ZQX-B8-L1 search first. ", 0), tool(t72.FIRST_CALL)],
    ) as rig:
        await t72.run(
            rig,
            [thought("ZQX-B8-L2 the search failed. ", 0), answer(t72.ANN_PYTHON, 1)],
            reasks=[[thought("ZQX-B8-L3 say it plainly.", 0), answer(t72.R_STOPPED, 1)]],
        )
    return _capture_rig(rig, tmp)


async def _live_retry(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    trip = [
        thought(t79.TRIP_THOUGHT, 0, open_=True),
        answer(t79.LATE, 1),
        tool(t79.CLICK_X, canvas_idx=1),
    ]
    clean = [thought(t79.CLEAN_THOUGHT, 0), tool(t79.CLICK4)]
    async with t79.retry_rig(tmp, script=t42.STEP1) as rig:
        await t79.run(rig, *t79.live_turn_steps(), trip, clean, [answer(t79.FINAL)])
    out = _capture_rig(rig, tmp)
    out["executed"] = [[i.call_id, dict(i.arguments)] for i in rig.browser.invocations]
    return out


async def _barge_in(tmp: Path) -> dict[str, Any]:
    tmp.mkdir(parents=True, exist_ok=True)
    async with tool_rig(tmp, kit=None) as rig:
        rig.llm.script = [
            thought("ZQX-B8-BARGE I start. ", 0, open_=True),
            thought("ZQX-B8-BARGE-LATE more.", 1, open_=True),
            answer("Never voiced.", 2),
        ]
        rig.llm.hold_after(1)
        await rig.ask("what is on this screen")
        await rig.until(
            lambda: any(e.segment == "thought" for e in rig.machine.events(E.LlmCommit)),
            "T-0180 B8: the first thought commit never reached the machine",
        )
        now = rig.clock.now_ms
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        rig.submit(E.UserTakeover(t_ms=now + 1, onset_ms=now))
        await rig.advance(500)
        rig.llm.release()
        await rig.advance(3_000)
        rig.llm.script = [thought("ZQX-B8-AFTER fresh. ", 0), answer("Here it is again. ", 1)]
        await rig.ask("say it again")
        await rig.until(
            lambda: (
                rig.state.state is TurnState.IDLE
                and any(
                    m.type == "astra.spoken" and m.turn_id == "t2"
                    for m in rig.published.messages
                    if hasattr(m, "turn_id")
                )
            ),
            "T-0180 B8: the turn after the barge-in never ended",
            limit_ms=40_000,
        )
        await rig.advance(300)
    return _capture_rig(rig, tmp)


async def _capture(tmp: Path) -> dict[str, Any]:
    """Every regime. Runs in-process on the working tree and in a subprocess on the BASE tree."""
    t42.degenerate_message()  # premise: the T-0142 guard exists at both ends
    return {
        "tools_off": await _tools_off(tmp / "tools_off"),
        "dry_run_loop": await _dry_run_loop(tmp / "dry_run_loop"),
        "live_retry": await _live_retry(tmp / "live_retry"),
        "barge_in": await _barge_in(tmp / "barge_in"),
    }


_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0180_differential as m
out = pathlib.Path(sys.argv[1])
sections = asyncio.run(m._capture(out.parent / "run"))
out.write_text(json.dumps({"origin": local_astra.__file__, "sections": sections}))
"""


@pytest.fixture(scope="module")
def base(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The BASE run of every regime, once per module."""
    gr.require_commits(BASE_SHA)
    tmp = tmp_path_factory.mktemp("t0180_base")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_SHA)
        base_dir = tsl._base_tree(tmp / "base_tree")
    out = tmp / "base_out" / "sections.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    data = json.loads(out.read_text())
    assert Path(data["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {data['origin']}, not the {BASE_SHA[:8]} tree"
    )
    sections: dict[str, Any] = data["sections"]
    return sections


@pytest.fixture(scope="module")
def now(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The working tree's run of every regime, once per module, through the same JSON round trip as BASE's."""
    import asyncio

    tmp = tmp_path_factory.mktemp("t0180_now")
    return json.loads(json.dumps(asyncio.run(_capture(tmp / "run"))))


def _same(expected: list[Any], got: list[Any], what: str) -> None:
    first = next((i for i, (a, b) in enumerate(zip(expected, got, strict=False)) if a != b), None)
    assert got == expected, (
        f"T-0180 B8 {what} differs from {BASE_SHA[:8]} (lengths {len(expected)} vs {len(got)}; first difference "
        f"at {first}: base {json.dumps(expected[first] if first is not None else None)[:800]} / "
        f"now {json.dumps(got[first] if first is not None else None)[:800]})"
    )


#: what each regime's TIP run must have published (anti-vacuity), written by hand from the scripts:
#: ``(turn_id, step, kind, end)`` of each attempt's last message, in first-seen order
WANT_BLOCKS: dict[str, list[tuple[str, int, str, str]]] = {
    "tools_off": [("t1", 1, "step", "done"), ("t2", 1, "step", "done")],
    "dry_run_loop": [
        ("t1", 1, "step", "done"),
        ("t1", 2, "step", "done"),
        ("t1", 2, "reask", "done"),
    ],
    "live_retry": [
        ("t1", 2, "step", "done"),
        ("t1", 3, "step", "done"),
        ("t1", 4, "step", "stopped_repeating"),
        ("t1", 4, "retry", "done"),
    ],
    "barge_in": [("t1", 1, "step", "interrupted"), ("t2", 1, "step", "done")],
}


def _final_blocks(thoughts: list[list[Any]]) -> list[tuple[str, int, str, str]]:
    """The last message of each (turn, step, kind) run, in first-seen order: consecutive messages of one attempt
    share turn, step and kind, and each attempt ends with its final message."""
    out: list[tuple[str, int, str, str]] = []
    for turn_id, step, kind, end, _text in thoughts:
        key = (turn_id, step, kind)
        if out and out[-1][:3] == key and out[-1][3] == "open":
            out[-1] = (*key, end)
        elif not out or out[-1][:3] != key:
            out.append((*key, end))
    return out


@pytest.mark.parametrize("regime", REGIMES)
def test_b8_nothing_but_astra_thought_differs(
    base: dict[str, Any], now: dict[str, Any], regime: str
) -> None:
    """B8: in each regime, every section (machine, published messages other than ``astra.thought``, prompts and
    system prompt bytes, request bodies, the session log, ``turns.jsonl``, history, TTS, memory) equals BASE's; and
    the regime really published thoughts on the TIP side (and none at BASE)."""
    assert base[regime]["thoughts"] == [], (
        f"premise: BASE {BASE_SHA[:8]} publishes no astra.thought; got {base[regime]['thoughts'][:2]}"
    )
    got = _final_blocks(now[regime]["thoughts"])
    assert got == WANT_BLOCKS[regime], (
        f"T-0180 B8 anti-vacuity ({regime}): the TIP run must publish the thoughts its script commits; want "
        f"{WANT_BLOCKS[regime]}, got {got}"
    )
    for section in SECTIONS:
        _same(base[regime][section], now[regime][section], f"{regime} {section!r}")
    if regime == "live_retry":
        _same(base[regime]["executed"], now[regime]["executed"], "live_retry executed calls")


def test_b8_sentinels_reach_only_astra_thought(now: dict[str, Any]) -> None:
    """B8/B2 (adversarial): every ``ZQX-B8`` thought sentinel the scripts commit is absent from every compared
    section (history, prompts, bodies, TTS, ``turns.jsonl``, memory, every other message), except the session log's
    own ``llm.commit`` records (R-1: thought canvases go into the session log verbatim); the barge-in's late sentinel
    is in no published thought either (B4). Red at BASE only through the anti-vacuity half: the sentinels must be in
    ``astra.thought``."""
    for regime in REGIMES:
        sections = now[regime]
        for section in SECTIONS:
            if section == "machine":
                continue  # the pure machine receives thought commits by design (they never enter history)
            values = sections[section]
            if section == "log":
                # R-1: thought canvases are logged verbatim (``llm.commit``, and the machine's own steps)
                values = [r for r in values if r["kind"] not in ("llm.commit", "machine.step")]
            blob = json.dumps(values)
            assert "ZQX-B8" not in blob, (
                f"T-0180 B8: a thought sentinel reached {regime} {section!r}"
            )
    shown = json.dumps(now["tools_off"]["thoughts"] + now["dry_run_loop"]["thoughts"])
    for sentinel in ("ZQX-B8-OFF-1", "ZQX-B8-OFF-2", "ZQX-B8-L1", "ZQX-B8-L2", "ZQX-B8-L3"):
        assert sentinel in shown, f"T-0180 B8 anti-vacuity: {sentinel} is in astra.thought"
    barge = json.dumps(now["barge_in"]["thoughts"])
    assert "ZQX-B8-BARGE I start." in barge and "ZQX-B8-AFTER" in barge, barge[:400]
    assert "ZQX-B8-BARGE-LATE" not in barge, "T-0180 B4: a commit after the barge-in was published"
