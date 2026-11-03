"""The T-0024/T-0010 seam closure: a surface goes out, a click comes back, a turn opens.

T-0028 criterion 7, end to end over the fakes and a **real loopback socket** with the **real
codec**: the ui commit is gated into an ``astra.ui.surface``, the page answers with
``astra.ui.action``, and that action becomes an ordinary user turn whose transcript is
``[ui] <name> <compact-json context>`` (docs/02 §2.2 and the T-0028 spec delta (a) in docs/02 §4).
T-0024 criterion 7 ends at "the message is on the wire"; this is where it lands.

Seams assumed: ``tests/unit/_hud_rig.py``'s module docstring (``HudServer(session,
*, frame_sink, port=0)``, ``server.url``, the session's typed ``publish`` seam). Nothing here
looks at the implementation — the expected transcript is read off docs/02 §2.2, the expected
surface id off ``UiGate`` (``f"turn-{turn_id}"``, T-0010 A3/A7), and the expected turn count off
the machine's own rows.

Regime: no measurement. ``ScriptedClock`` drives the session, the socket runs on real time, and
every wait has a 2 s budget (docs/tasks/T-0028.md, "Regime").
"""

from __future__ import annotations

import json

import pytest

from local_astra.contracts.protocol import (
    AstraAssistState,
    AstraTurnState,
    AstraUiAction,
    AstraUiSurface,
)
from tests.unit._hud_rig import HudClient, answer, frame, served, ui

pytestmark = pytest.mark.contract

#: ``UiGate.surface_id_prefix`` is ``"turn"`` and the id is ``f"{prefix}-{turn_id}"`` (T-0010
#: A3/A7). The first turn is ``t1``, the turn the click opens is ``t2`` — the machine numbers
#: turns from ``turn_seq``, so this is arithmetic, not an observation of the session.
FIRST_SURFACE = "turn-t1"
SECOND_SURFACE = "turn-t2"

#: Ruling A13: virtual milliseconds of the session's clock, not wall seconds.
BUDGET_MS = 5_000

ACTION_NAME = "choose"
ACTION_CONTEXT = {"answer": "one half"}


def compact(context: dict[str, object]) -> str:
    """``<compact-json context>``: JSON with no whitespace between tokens (docs/02 §2.2)."""
    return json.dumps(context, separators=(",", ":"))


async def test_a_surface_goes_out_a_click_comes_back_and_becomes_the_next_turn(tmp_path):
    """Criterion 7: ui commit → ``astra.ui.surface`` → ``astra.ui.action`` → ``[ui] …`` turn.

    Hand-computed expectations:

    * turn 1 speaks ``"Here it is. "`` and commits one *valid* A2UI canvas, so exactly one
      surface id (``turn-t1``) reaches the page;
    * the click carries that id, the name ``choose`` and one context key, so the second turn's
      transcript is ``[ui] choose {"answer":"one half"}`` — name, then compact JSON, in that
      order, with a single space between the three parts;
    * the reply of turn 2 is another canvas, so a *second* surface id (``turn-t2``) reaches the
      page: the round trip is closed, not merely acknowledged;
    * two turns ran, so ``turns.jsonl`` has exactly two lines (T-0007 criterion 12 — one line per
      opened turn, written on the turn-ending row).
    """
    script = [answer("Here it is. "), ui(canvas_idx=1)]
    async with served(tmp_path, script=script, frames=[frame(0, "a")], ms_per_word=100) as box:
        rig = box.rig
        async with HudClient.connect(box.url) as client:
            await client.hello()
            await client.drain(
                until=lambda got: any(isinstance(m, AstraAssistState) for m in got),
                rig=rig,
                budget_ms=BUDGET_MS,
            )

            await rig.ask("show me the surface")
            await client.drain(
                until=lambda got: any(
                    isinstance(m, AstraUiSurface) and m.surface_id == FIRST_SURFACE for m in got
                ),
                rig=rig,
                budget_ms=BUDGET_MS,
            )
            await rig.speak_out()
            first = [m for m in client.of(AstraUiSurface) if m.surface_id == FIRST_SURFACE]
            assert first, (
                "criterion 7/9: the gated canvas must reach the page as astra.ui.surface; the "
                "session computes result.surface today and drops it (docs/tasks/T-0028.md, "
                f"Owned files). The client saw {[getattr(m, 'type', m) for m in client.received]}"
            )

            # the page answers the surface it was given
            rig.llm.script = [answer("Correct. "), ui(canvas_idx=1)]
            await client.send(
                AstraUiAction(
                    surface_id=first[-1].surface_id, name=ACTION_NAME, context=ACTION_CONTEXT
                )
            )
            await client.drain(
                until=lambda got: any(
                    isinstance(m, AstraUiSurface) and m.surface_id == SECOND_SURFACE for m in got
                ),
                rig=rig,
                budget_ms=BUDGET_MS,
            )
            await rig.speak_out()

        transcripts = rig.llm.transcripts()
        assert len(transcripts) == 2, (
            "criterion 7: the click opens one ordinary user turn, so the LLM saw exactly two "
            f"turn prompts; it saw {len(transcripts)}"
        )
        expected = f"[ui] {ACTION_NAME} {compact(ACTION_CONTEXT)}"
        assert expected in transcripts[1], (
            "criterion 7 / docs/02 §2.2: the second turn's transcript is "
            f"'[ui] <name> <compact-json context>' = {expected!r}; the prompt's last message was "
            f"{transcripts[1]!r}"
        )
        assert len(rig.lines()) == 2, (
            "two turns ran, so turns.jsonl has two lines (T-0007 criterion 12); it has "
            f"{len(rig.lines())}: {rig.lines()}"
        )
        assert rig.records()[1].turn_id == "t2"

        states = [m.state for m in client.of(AstraTurnState)]
        assert "THINKING" in states and "SPEAKING" in states, (
            "criterion 9: the ui turn runs the normal rows and every state change is on the "
            f"wire; the page saw {states}"
        )
        assert [m.surface_id for m in client.of(AstraUiSurface)][-1] == SECOND_SURFACE, (
            "the reply of the ui turn renders too: the last surface the page receives is the "
            f"second turn's ({SECOND_SURFACE})"
        )
