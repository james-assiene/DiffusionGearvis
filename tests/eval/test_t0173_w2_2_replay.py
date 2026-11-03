"""T-0173 criterion 8: the W2-2 incident replayed through the real ``TurnMachine`` (eval, CPU).

(a) The fixture's logged ``machine.step`` events, as logged, reproduce ``hud.error "tool timeout"`` at the seq-720
tick, with every replayed step's commands equal to the logged ones. (b) With each non-Tick event's ``t_ms`` replaced
by its record's ``session_ms`` (what the T-0173 restamp does at the session boundary), seq 720 emits no tool timeout
and the ``tool_result`` deadline is the execute step's ``session_ms + 15 000``.

Both rows hold on the base: the machine is pure and unchanged by T-0173, so this is the incident's evidence and the
fix's rationale as a test, not a pending row. Fixture provenance, extraction and replay rules:
``tests/fixtures/t0173/MANIFEST.md``.

Regime: CPU, pure ``TurnMachine.step``, ``TurnPolicy(thinking=True, tool_loop=True, max_tool_steps=8)`` (the source's
``loop8`` regime), ``SpeechGate`` segmenter, n = 1.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.speech_gate import SpeechGate
from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.contracts.turn import TurnState, Word

#: ``gpu`` only gates selection (T-0046), as the other CPU replays in tests/eval.
#: Run: ``uv run pytest tests/eval/test_t0173_w2_2_replay.py -q -m gpu -rA``
pytestmark = [pytest.mark.gpu, pytest.mark.eval]

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "t0173" / "w2_2_t1_steps.jsonl"
FIXTURE_SHA256 = "0c179bfcadbb326b6c468f9ec26412a9ea13615146a4203abf00bba5af06899d"
LIVE_POLICY = TurnPolicy(thinking=True, tool_loop=True, max_tool_steps=8)
R18_MS = 15_000


class _GateSegmenter:
    def __init__(self) -> None:
        self.gate = SpeechGate()

    def __call__(self, text: str, *, final: bool) -> tuple[list[tuple[str, str]], str]:
        held = self.gate.pending
        suffix = text[len(held) :] if text.startswith(held) else text
        commit = Commit(request_id="r", canvas_idx=0, segment="answer", text=suffix)
        out = list(self.gate.feed(commit)) if suffix else []
        if final:
            out.extend(self.gate.close())
        return [(s.source, s.text) for s in out], self.gate.pending


def records() -> list[dict[str, Any]]:
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256, (
        "the fixture is not the one MANIFEST.md describes"
    )
    return [json.loads(x) for x in raw.decode().splitlines() if x.strip()]


def replay(restamp: bool) -> list[tuple[dict[str, Any], list[Any], str]]:
    recs = records()
    m = TurnMachine(LIVE_POLICY, segment=_GateSegmenter())
    s = m.initial()
    out = []
    for r in recs:
        ev = E.EventAdapter.validate_python(r["data"]["event"])
        if restamp and ev.type != "tick":
            ev = ev.model_copy(update={"t_ms": r["session_ms"]})
        assert s.state.value == r["data"]["state_before"], (
            f"seq {r['seq']}: replay diverged before the step"
        )
        s, cmds = m.step(s, ev)
        out.append((r, list(cmds), s.state.value))
        if isinstance(ev, E.UserSpeechOnset) and s.state is TurnState.LISTENING:
            # The typed-turn seed (MANIFEST.md, "Replay rules"): not a machine.step in the log.
            text = recs[0]["data"]["event"]["text"]
            s = s.model_copy(
                update={
                    "words": (Word(text=text, start_ms=ev.onset_ms, end_ms=ev.onset_ms),),
                    "state": TurnState.ENDPOINTING,
                    "t_eot_ms": ev.onset_ms,
                    "flush_t_ms": r["session_ms"],
                }
            )
    return out


def test_fixture_reproduces_the_spurious_timeout() -> None:
    """(a): as logged, every step's commands and state match the log, and seq 720 fails the turn with R18."""
    out = replay(restamp=False)
    for r, cmds, after in out:
        assert [c.model_dump(mode="json") for c in cmds] == r["data"]["commands"], (
            f"seq {r['seq']} commands"
        )
        assert after == r["data"]["state_after"], f"seq {r['seq']} state"
    r, cmds, after = out[-1]
    assert r["seq"] == 720 and [c.message for c in cmds if isinstance(c, C.HudError)] == [
        "tool timeout"
    ]
    assert after == "IDLE"


def test_session_clock_restamp_removes_it() -> None:
    """(b): restamped, no tool timeout at seq 720, and tool_result is armed at the execute step's session_ms + 15 000."""
    out = replay(restamp=True)
    execute = [(r, cmds) for r, cmds, _ in out if any(isinstance(c, C.ToolExecute) for c in cmds)]
    assert len(execute) == 1, "one execute step"
    r, cmds = execute[0]
    arms = [c.at_ms for c in cmds if isinstance(c, C.ArmTimer) and c.name == "tool_result"]
    assert arms == [r["session_ms"] + R18_MS], (
        f"tool_result armed at {arms}, want [{r['session_ms'] + R18_MS}]"
    )
    last, cmds, after = out[-1]
    assert last["seq"] == 720
    assert not [c for c in cmds if isinstance(c, C.HudError)] and after == "THINKING", (
        f"restamped: seq 720 keeps waiting for the result; got {[c.type for c in cmds]}, {after}"
    )
