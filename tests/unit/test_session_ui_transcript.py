"""The click-opened turn's transcript on the wire (T-0028 ruling A19, carried to T-0020).

A19 is a *coverage* row, not a new behaviour: the merged ``session.py`` already publishes the
click's transcript at open (``_open_ui_transcript`` → ``_publish_transcript(final=True)``) and
would publish it again at the ENDPOINTING → THINKING row, where the ``_transcript_final`` fence
stops it. Nothing pinned either half, so "exactly one ``astra.transcript(final=true)`` carrying
the ``[ui] …`` text, before the first ``astra.spoken`` of the turn" could regress into a duplicate
(the page shows the question twice) or into a final published after the answer has started
(ruling A2's ordering) without a test going red. This module is that pin; T-0020 owns
``session.py`` now, so it lives with T-0020's other session rows rather than in T-0028's files.

Not ``pending``: the behaviour exists today and this row must stay green from the first run —
its job is to fail if the fence or the open-time publish is removed.

Regime: no measurement, ``ScriptedClock``, ``tests/fakes/``, one click, no microphone.
"""

from __future__ import annotations

import json

from local_astra.contracts import events as E
from local_astra.contracts.protocol import AstraSpoken, AstraTranscript
from local_astra.contracts.turn import TurnState
from tests.unit._hud_rig import Recorder, answer, frame, running

NAME = "choose"
CONTEXT = {"answer": "one half"}
#: docs/02 §2.2: ``[ui] <name> <compact-json context>`` — compact JSON, no space after ``:``.
EXPECTED = f"[ui] {NAME} " + json.dumps(CONTEXT, separators=(",", ":"))


def action(t_ms: int) -> E.UiAction:
    return E.UiAction(t_ms=t_ms, surface_id="turn-t1", name=NAME, context=dict(CONTEXT))


async def test_a_ui_opened_turn_publishes_exactly_one_final_transcript_before_it_speaks(tmp_path):
    """Ruling A19: one ``final=true``, with the ``[ui] …`` text, before the first ``astra.spoken``.

    Hand-computed from one click and a one-sentence answer: the page must see the question once,
    complete (a click's transcript is complete the moment it opens — ruling A17), and never walked
    back by a partial afterwards.
    """
    published = Recorder()
    async with running(
        tmp_path,
        script=[answer("Correct. ")],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=published,
    ) as rig:
        assert rig.state.state is TurnState.IDLE
        rig.submit(action(rig.clock.now_ms))
        await rig.until(
            lambda: rig.llm.gen_prompts != [], "the click never opened a turn", limit_ms=5_000
        )
        await rig.speak_out()

        transcripts = published.of(AstraTranscript)
        finals = [m for m in transcripts if m.final]
        assert len(finals) == 1, (
            "ruling A19: exactly one final transcript for a click-opened turn — the open-time "
            f"publish and the flush row must not both reach the wire; got {finals}"
        )
        assert finals[0].text == EXPECTED, (
            f"the final carries the click's own text; got {finals[0].text!r}"
        )
        assert finals[0].turn_id == "t1", finals[0].turn_id

        order = [
            index
            for index, message in enumerate(published.messages)
            if isinstance(message, AstraTranscript | AstraSpoken)
        ]
        first_spoken = next(
            (i for i in order if isinstance(published.messages[i], AstraSpoken)), None
        )
        final_at = next(
            i
            for i in order
            if isinstance(published.messages[i], AstraTranscript) and published.messages[i].final
        )
        assert first_spoken is not None, "the turn must actually speak, or the ordering is vacuous"
        assert final_at < first_spoken, (
            "ruling A2/A19: the question goes final *before* the first astra.spoken of the turn; "
            f"got transcript at {final_at}, spoken at {first_spoken}"
        )
        assert not [
            m for m in transcripts[transcripts.index(finals[0]) + 1 :] if m.turn_id == "t1"
        ], (
            "a transcript that has gone final is never walked back by a later partial for the "
            f"same turn; got {transcripts}"
        )
