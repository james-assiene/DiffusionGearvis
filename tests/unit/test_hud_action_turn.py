"""``astra.ui.action`` becomes a user turn, or is refused — never smuggled in (criteria 7, 8).

The seam T-0024 stops one inch short of. ``ui.action`` is **rowless in the machine by design**
(``turn_machine._dispatch``'s own comment), so the turn is opened by the *session* from its own
events, and the refusal when the machine is not IDLE is the session's policy too — docs/02 §4's
T-0028 spec delta (a), recorded ahead of this implementation.

No socket: the session's seam is ``events.UiAction`` (``contracts/events.py``, already on the
union). ``tests/contract/test_hud_action_round_trip.py`` is the same criterion with the real
socket and the real codec around it.

Regime: no measurement, ``ScriptedClock``, ``tests/fakes/``.
"""

from __future__ import annotations

import json

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.perception import AssistFlags
from local_astra.contracts.protocol import AstraAssistToggle, AstraError, AstraTranscript
from local_astra.contracts.turn import TurnState
from tests.unit._hud_rig import Recorder, answer, frame, running, ui

NAME = "choose"
CONTEXT = {"answer": "one half"}
#: docs/02 §2.2: ``[ui] <name> <compact-json context>``. Compact JSON has no space after ``:``
#: or ``,``; with one key this is unambiguous, so the expected string is exact.
EXPECTED = f"[ui] {NAME} " + json.dumps(CONTEXT, separators=(",", ":"))

#: What the microphone says in the race row. Every word is disjoint from ``EXPECTED`` on purpose
#: (ruling A18(2)): "one half" is in the *click's* own context, so a word-level splice check built
#: on it would be satisfied by the legitimate transcript and catch nothing.
MIC_WORDS = "tell me about penguins"

#: Events with an *audio* meaning. Ruling A5: the **only** one a click may synthesise is the
#: ``UserSpeechOnset`` the machine needs to open a turn, stamped at the click's own time; the
#: transcript is injected as the turn's word buffer content, not replayed as ``stt.word`` events.
#: No microphone produced any of the following, so none of them may appear.
SPEECH_SHAPED = (
    E.UserBackchannel,
    E.UserTakeover,
    E.UserPause,
    E.UserSpeechOffset,
    E.UserNotOwner,
    E.SttStep,
    E.SttWord,
    E.SttEndOfTurn,
    E.SttFlushed,
)
#: Fields that are audio timestamps (docs/02 §4: attribution is by audio time).
AUDIO_FIELDS = ("onset_ms", "start_ms", "end_ms", "t_eot_ms", "offset_ms")


def action(t_ms: int, surface_id: str = "turn-t1") -> E.UiAction:
    return E.UiAction(t_ms=t_ms, surface_id=surface_id, name=NAME, context=dict(CONTEXT))


async def test_an_action_when_idle_opens_a_ui_user_turn(tmp_path, monkeypatch):
    """Criterion 7: one ordinary user turn, transcript ``[ui] <name> <compact json>``.

    Hand-computed: one action → one prompt, one ``llm.start`` whose transcript is ``EXPECTED``,
    one ``turns.jsonl`` line, one speech gate (so the utterance ids are this turn's ``t1-u…``).
    The assists are frozen when the turn opens, so the box ticked *after* the click is not in the
    line's regime — the same rule an ordinary spoken turn obeys (T-0007 criterion 10).
    """
    published = Recorder()
    async with running(
        tmp_path,
        script=[answer("Correct. "), ui(canvas_idx=1)],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=published,
    ) as rig:
        submitted: list[E.Event] = []
        real_submit = rig.session.submit
        monkeypatch.setattr(
            rig.session, "submit", lambda event: (submitted.append(event), real_submit(event))[1]
        )
        assert rig.state.state is TurnState.IDLE
        at = rig.clock.now_ms
        rig.submit(action(at))
        await rig.until(
            lambda: rig.llm.gen_prompts != [], "the click never opened a turn", limit_ms=5_000
        )
        # a box ticked after the turn opened must not reach this turn's regime
        rig.session.assists.toggle(AstraAssistToggle(flags=AssistFlags(page_text=True)))
        await rig.speak_out()

        starts = rig.machine.commands(C.LlmStart)
        assert len(starts) == 1, (
            f"criterion 7: the action opens exactly one turn; the machine issued {starts}"
        )
        assert starts[0].transcript == EXPECTED, (
            "criterion 7 / docs/02 §2.2: the transcript is '[ui] <name> <compact-json context>'. "
            f"Expected {EXPECTED!r}, the machine was given {starts[0].transcript!r}"
        )
        assert len(rig.llm.gen_prompts) == 1, (
            f"one prompt, not one per synthesised event; got {len(rig.llm.gen_prompts)}"
        )
        assert len(rig.lines()) == 1, f"one TurnRecord for one turn; got {rig.lines()}"
        record = rig.records()[0]
        assert record.turn_id == "t1"
        assert record.regime.assist_flags == AssistFlags(), (
            "criterion 7: the assists are frozen when the turn opens, so the box ticked while the "
            f"answer was being spoken is not in this line's regime; it says {record.regime}"
        )
        speaks = rig.machine.commands(C.TtsSpeak)
        assert speaks and all(c.utterance_id.startswith("t1-u") for c in speaks), (
            "one gate and one utterance sequence per turn, keyed on the machine's turn id; got "
            f"{[c.utterance_id for c in speaks]}"
        )
        assert rig.session.speech_gate is None, "the gate is dropped when the turn ends"

        # --- negative: nothing speech-shaped, and no invented audio time -------------------
        opener = [e for e in submitted if not isinstance(e, E.Tick)]
        strays = [e for e in opener if isinstance(e, SPEECH_SHAPED)]
        assert not strays, (
            "criterion 7, negative (ruling A5): a click synthesises no speech-shaped event "
            "besides the onset — the transcript is injected as the word buffer, not replayed as "
            f"stt.word/stt.end_of_turn/stt.flushed; the session also invented {strays}"
        )
        onsets = [e for e in opener if isinstance(e, E.UserSpeechOnset)]
        assert len(onsets) == 1 and onsets[0].onset_ms == at, (
            "ruling A5: exactly one UserSpeechOnset, stamped at the click's own time "
            f"({at} ms); the session submitted {onsets}"
        )
        now = rig.clock.now_ms
        for event in opener:
            for field in AUDIO_FIELDS:
                value = getattr(event, field, None)
                if value is None:
                    continue
                assert at <= value <= now, (
                    "criterion 7, negative: nothing invents audio timestamps that no microphone "
                    f"produced — {type(event).__name__}.{field} = {value} is outside the window "
                    f"[{at}, {now}] in which the click existed"
                )
        assert not [m for m in published.messages if isinstance(m, AstraError)], (
            f"an accepted action is not an error: {[m for m in published.messages if isinstance(m, AstraError)]}"
        )


async def test_an_action_during_speaking_is_refused_and_leaves_the_turn_intact(tmp_path):
    """Criterion 8, adversarial: the click lands during SPEAKING, one hop before the drain.

    The two events reach the pump back to back, so the refusal has to be decided while the turn
    is at its most fragile: everything is committed, the audio is finishing, and the very next
    hop ends the turn. A refusal that instead opened a turn — or nudged the machine — would show
    up as a second ``llm.start``, a second record, or an assistant history entry that is not what
    was played.

    Hand-computed: one refusal (``astra.error(code="busy")``), the ``ui.action`` step produces no
    command and no state change (it is rowless in the machine, docs/02 §4), one turn record, one
    history entry whose assistant text is exactly the played prefix, and the *next* turn's prompt
    carries no residue of the refused click.
    """
    published = Recorder()
    async with running(
        tmp_path,
        script=[answer("A long first sentence. ")],
        frames=[frame(0, "a")],
        ms_per_word=300,
        publish=published,
    ) as rig:
        await rig.ask("tell me everything")
        await rig.until(
            lambda: rig.state.state is TurnState.SPEAKING and rig.state.llm_done,
            "the turn never reached SPEAKING with the generate finished",
            limit_ms=10_000,
        )
        before = rig.state
        played = before.spoken_text
        assert played, "the turn must really be speaking, or the hop below is not the hop"

        rig.submit(action(rig.clock.now_ms, surface_id="turn-t1"))
        rig.submit(E.PlaybackDrained(t_ms=rig.clock.now_ms, played_ms=100, played_text=played))
        await rig.pump()

        refusals = [m for m in published.messages if isinstance(m, AstraError)]
        assert [m.code for m in refusals] == ["busy"], (
            "criterion 8 / docs/02 §4 spec delta (a): an action while a turn is live is refused "
            f"with astra.error(code='busy'); the page was told {[m.code for m in refusals]}"
        )
        step = [entry for entry in rig.machine.log if isinstance(entry[0], E.UiAction)]
        assert len(step) == 1 and step[0][2] == (), (
            "criterion 8: ui.action is rowless in the machine — the refusal is the session's "
            f"policy and must not reach the machine as commands; it produced {step}"
        )
        assert step[0][1] is TurnState.SPEAKING, (
            f"…and the live turn's state is untouched; it became {step[0][1]}"
        )
        assert len(rig.machine.commands(C.LlmStart)) == 1, (
            "the refused action opened no turn: there is still exactly one llm.start"
        )

        await rig.speak_out()
        assert len(rig.lines()) == 1, (
            f"the live turn still ends with exactly one record; got {rig.lines()}"
        )
        assert len(rig.state.history) == 1 and rig.state.history[0].assistant_spoken == played, (
            "history holds the played text of the interrupted-by-nothing turn and nothing the "
            f"click added; it holds {rig.state.history}"
        )

        # --- no residue in the next turn --------------------------------------------------
        rig.llm.script = [answer("Second answer. ")]
        await rig.ask("and now something else")
        await rig.speak_out()
        second = rig.llm.transcripts()[-1]
        assert "[ui]" not in second and NAME not in second, (
            "criterion 8: the refused action leaves no residue in the next turn's prompt; it "
            f"reads {second!r}"
        )
        assert len(rig.lines()) == 2, f"two turns, two lines; got {rig.lines()}"


async def test_a_click_and_a_microphone_onset_in_the_same_hop_do_not_splice(tmp_path):
    """Ruling A14, adversarial: the click and the user start in the same pump hop.

    The user taps an answer on the surface and starts talking at the same moment — one hop apart
    in the queue, which on a 50 ms tick is the same instant. The failure this guards is a *splice*:
    a click held as "pending" that resolves against whichever ``user.speech_onset`` comes next
    puts ``[ui] choose {...}`` into the turn the microphone opened, so the page sees a
    ``final=true`` transcript in the middle of a turn and then partials again — criterion 9
    breached on the wire, and a prompt that claims the user said something they did not.

    Either outcome is correct and the test accepts both: the click is refused with ``busy``, or it
    gets a turn of its own. What is never correct is one turn carrying both.
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
        at = rig.clock.now_ms
        rig.submit(action(at))
        rig.submit(E.UserSpeechOnset(t_ms=at, onset_ms=at))
        await rig.pump()
        await rig.words(MIC_WORDS)
        await rig.end_of_turn()
        await rig.speak_out(limit_ms=10_000)

        starts = rig.machine.commands(C.LlmStart)
        assert starts, "the microphone turn must have run"
        ui_starts = [c for c in starts if c.transcript.startswith("[ui] ")]
        # Ruling A18(2): the splice that actually happened was one *word* long
        # (``[ui] choose {"answer":"one half"} tell``), so "does the whole spoken phrase appear"
        # missed it. The click's transcript is known exactly, so require exactly that — and the
        # microphone words are chosen disjoint from the action's own context so that the
        # word-level check below cannot be satisfied by the context's own text.
        for command in ui_starts:
            assert command.transcript == EXPECTED, (
                "ruling A14/A18: a click-opened turn's transcript is the click and nothing else. "
                f"Expected {EXPECTED!r}, the machine was given {command.transcript!r}"
            )
            strays = [word for word in MIC_WORDS.split() if word in command.transcript]
            assert not strays, f"…the microphone words {strays} were spliced into the click's turn"
        for command in starts:
            if command in ui_starts:
                continue
            assert "[ui]" not in command.transcript, (
                "…and the click was not spliced into the spoken turn either: "
                f"{command.transcript!r}"
            )
        refusals = [m.code for m in published.messages if isinstance(m, AstraError)]
        assert ui_starts or "busy" in refusals, (
            "ruling A14: the click is either refused with busy or gets its own turn; it did "
            f"neither (transcripts {[c.transcript for c in starts]}, errors {refusals})"
        )
        assert len(ui_starts) <= 1, (
            f"…and never more than one turn per click; got {[c.transcript for c in ui_starts]}"
        )

        # criterion 9, on the wire: a turn's final transcript is the last one it ever sends
        by_turn: dict[str, list[bool]] = {}
        for message in published.messages:
            if isinstance(message, AstraTranscript):
                by_turn.setdefault(message.turn_id, []).append(message.final)
        for turn_id, finals in by_turn.items():
            if True not in finals:
                continue
            assert finals.count(True) == 1 and finals[-1] is True, (
                f"ruling A14 / criterion 9: turn {turn_id}'s final=true transcript is not the "
                f"last thing it sent — partials followed it, so the page showed a finished "
                f"question and then went back to typing. The sequence of `final` flags was "
                f"{finals}"
            )
