"""A typed question becomes an ordinary user turn — or is refused (T-0039 criteria 5-9, 21).

docs/tasks/T-0039.md group A, the session half. The wire type is ``astra.text`` and the
transport turns it into one ``events.UserText`` (``tests/unit/test_hud_ingress.py``); what this
module is about is what the *session* does with it:

* ``IDLE``: open exactly one turn through the ordinary rows — one ``user.speech_onset`` stamped
  at the message's own time, then the turn's word buffer seeded with the typed text and
  ``t_eot_ms``/``flush_t_ms`` set, so the existing ``ENDPOINTING → THINKING`` tick row runs
  untouched. The transcript is the typed text **verbatim**: no ``[ui]`` prefix, no synthesised
  ``stt.*`` event (criterion 5).
* not ``IDLE``: ``astra.error(code="busy")`` and the live turn is left alone (criterion 7).
* a microphone onset that opens a turn first: the typed text is refused, never spliced
  (criterion 6 — T-0028 ruling A14 replayed for text).
* the turn's ``turns.jsonl`` line says ``input="text"`` (criterion 9), and the typed text never
  reaches TTS, ``astra.spoken`` or assistant history (criterion 21).

The seams this module calls are fixed by ``tests/unit/_hud_rig.py``'s docstring plus, for this
task, the public names ``events.UserText(type="user.text", t_ms, text)`` and
``TurnRecord.input: Literal["voice","text","ui"] = "voice"``. No socket, no clock: ``ScriptedClock``
and ``tests/fakes/``. No measurement is produced here.
"""

from __future__ import annotations

import json

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.protocol import (
    SERVER_TO_CLIENT,
    AstraError,
    AstraSpoken,
    AstraSpokenPlayed,
    AstraTranscript,
)
from local_astra.contracts.turn import TurnState
from tests.unit._hud_rig import Recorder, answer, frame, running, ui

#: The typed question. Every word of it is absent from the answers and from the microphone line
#: below, so "did any part of it leak" is a question the fixtures can actually answer
#: (T-0028 ruling A18(2): a splice check built on shared words catches nothing).
TYPED = "which zsh plugin owns this prompt"

#: What the microphone says in the race row — disjoint from ``TYPED`` word by word.
MIC_WORDS = "tell me about penguins"

#: docs/tasks/T-0039.md criterion 6/7: the same code string the ``[ui]`` path already uses.
BUSY = "busy"

#: The ``[ui]`` transcript prefix that must **not** appear on a typed turn (criterion 5).
UI_PREFIX = "[ui]"


def text_event(t_ms: int, text: str = TYPED):
    """One ``events.UserText``; the assertion names the contract when it does not exist yet."""
    cls = getattr(E, "UserText", None)
    assert cls is not None, (
        'T-0039 criterion 4: contracts/events.py gains UserText(type="user.text", t_ms, text) '
        "(docs/tasks/T-0039.md)"
    )
    return cls(t_ms=t_ms, text=text)


def record_input_field() -> None:
    """Criterion 9's field, asserted before a test builds a record that needs it."""
    from local_astra.telemetry.turns_log import TurnRecord

    assert "input" in TurnRecord.model_fields, (
        'T-0039 criterion 9: TurnRecord gains input: Literal["voice","text","ui"] = "voice"; '
        f"it declares {sorted(TurnRecord.model_fields)}"
    )


def errors(published: Recorder) -> list[str]:
    return [m.code for m in published.messages if isinstance(m, AstraError)]


def transcripts_by_turn(published: Recorder) -> dict[str, list[tuple[str, bool]]]:
    out: dict[str, list[tuple[str, bool]]] = {}
    for message in published.messages:
        if isinstance(message, AstraTranscript):
            out.setdefault(message.turn_id, []).append((message.text, message.final))
    return out


async def test_a_typed_question_when_idle_opens_one_ordinary_turn(tmp_path, monkeypatch):
    """Criteria 5 and 8: IDLE → one turn → THINKING → SPEAKING, transcript verbatim.

    Hand-computed: one typed question → one ``llm.start`` whose transcript is ``TYPED`` exactly,
    one ``turns.jsonl`` line, one ``astra.transcript(final=True)`` for it and no partial after
    that final, and the answer goes out on the message types a spoken turn already uses — no new
    server→client type (criterion 8).

    The negative half is criterion 5's "no synthesised ``stt.*``": the endpointing rows belong to
    a microphone. A session that replayed the typed words as ``stt.word``/``stt.end_of_turn``
    would make T-0032's endpointer observe a flush no STT produced.
    """
    published = Recorder()
    async with running(
        tmp_path,
        script=[answer("It is oh-my-zsh. ")],
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
        rig.submit(text_event(at))
        await rig.until(
            lambda: rig.llm.gen_prompts != [],
            "the typed question never opened a turn",
            limit_ms=5_000,
        )
        assert rig.state.state in (TurnState.THINKING, TurnState.SPEAKING), (
            f"criterion 5: the turn walks to THINKING; it is in {rig.state.state.value}"
        )
        await rig.speak_out()

        starts = rig.machine.commands(C.LlmStart)
        assert len(starts) == 1, (
            f"criterion 5: one typed question opens exactly one turn; the machine issued {starts}"
        )
        assert starts[0].transcript == TYPED, (
            "criterion 5: the transcript is the typed text verbatim — no [ui] prefix, no "
            f"decoration. Expected {TYPED!r}, the machine was given {starts[0].transcript!r}"
        )
        assert UI_PREFIX not in starts[0].transcript
        assert len(rig.lines()) == 1, f"one turn, one record; got {rig.lines()}"

        # --- criterion 5, negative: nothing speech-shaped but the one onset ------------------
        opener = [e for e in submitted if not isinstance(e, E.Tick)]
        strays = [
            e
            for e in opener
            if isinstance(
                e,
                E.SttWord
                | E.SttEndOfTurn
                | E.SttFlushed
                | E.SttStep
                | E.UserPause
                | E.UserSpeechOffset
                | E.UserTakeover
                | E.UserBackchannel
                | E.UserNotOwner,
            )
        ]
        assert not strays, (
            "criterion 5: a typed question synthesises no stt.* and nothing else speech-shaped — "
            f"the transcript is injected as the word buffer; the session also submitted {strays}"
        )
        onsets = [e for e in opener if isinstance(e, E.UserSpeechOnset)]
        assert len(onsets) == 1 and onsets[0].onset_ms == at, (
            "criterion 5: exactly one user.speech_onset, stamped at the message's own time "
            f"({at} ms); the session submitted {onsets}"
        )

        # --- criterion 8: the answer reaches the page the way a spoken turn's does -----------
        kinds = [m.type for m in published.messages]
        for required in ("astra.transcript", "astra.spoken", "astra.turn.state"):
            assert required in kinds, (
                f"criterion 8: a typed turn publishes {required} like a spoken one; the page saw "
                f"{sorted(set(kinds))}"
            )
        assert any(isinstance(m, AstraSpokenPlayed) for m in published.messages), (
            f"criterion 8: …including astra.spoken.played; the page saw {sorted(set(kinds))}"
        )
        assert set(kinds) <= SERVER_TO_CLIENT, (
            "criterion 8: no new server→client type is invented for text mode; the page saw "
            f"{sorted(set(kinds) - SERVER_TO_CLIENT)} on top of the protocol"
        )
        assert not errors(published), f"an accepted question is not an error: {errors(published)}"

        turns = transcripts_by_turn(published)
        assert len(turns) == 1, f"criterion 5: one turn's transcripts; got {turns}"
        [(turn_id, lines)] = turns.items()
        finals = [final for _, final in lines]
        assert finals.count(True) == 1 and finals[-1] is True, (
            f"criterion 5/8: turn {turn_id} publishes exactly one final transcript and nothing "
            f"after it; the sequence of `final` flags was {finals}"
        )
        assert [text for text, final in lines if final] == [TYPED], (
            "criterion 5: …and that final transcript is the typed text verbatim; it reads "
            f"{[text for text, final in lines if final]}"
        )


async def test_the_typed_question_is_never_spoken_and_never_assistant_history(tmp_path):
    """Criterion 21: the question is the *user's* text; nothing new enters the spoken channel.

    The cheap way to make the transcript pane show a typed question is to publish it as an
    ``astra.spoken`` or to hand it to TTS "so the turn looks normal". Either one voices text the
    LLM never committed (CLAUDE.md) — the assistant would read the user's own question aloud —
    and an entry in assistant history would make the next turn's prompt claim the assistant said
    it.
    """
    published = Recorder()
    async with running(
        tmp_path,
        script=[answer("It is oh-my-zsh. ")],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=published,
    ) as rig:
        rig.submit(text_event(rig.clock.now_ms))
        await rig.until(lambda: rig.llm.gen_prompts != [], "no turn opened", limit_ms=5_000)
        await rig.speak_out()

        spoken = [m.text for m in published.messages if isinstance(m, AstraSpoken)]
        assert spoken, "premise: the answer really was spoken, or this row proves nothing"
        for line in spoken:
            assert TYPED not in line, (
                f"criterion 21: the typed question was published as spoken: {line!r}"
            )
        for command in rig.machine.commands(C.TtsSpeak):
            assert TYPED not in command.text and TYPED not in command.source, (
                f"criterion 21: the typed question was handed to TTS: {command!r}"
            )
        assert rig.tts.texts(), "premise: TTS really ran"
        for text in rig.tts.texts():
            assert TYPED not in text, f"criterion 21: TTS was asked to say the question: {text!r}"
        assert len(rig.state.history) == 1, f"one turn, one history entry; got {rig.state.history}"
        entry = rig.state.history[0]
        assert entry.user_text == TYPED, (
            f"criterion 5: the question is the *user* side of history; it reads {entry.user_text!r}"
        )
        assert TYPED not in entry.assistant_spoken, (
            "criterion 21: …and never the assistant side; the assistant is recorded as having "
            f"said {entry.assistant_spoken!r}"
        )


BUSY_STATES = ["listening", "endpointing", "thinking", "speaking"]


async def drive_to(rig, where: str) -> None:
    """Park the rig in one non-IDLE state, using only the ordinary microphone rows."""
    if where == "listening":
        await rig.onset()
        return
    if where == "endpointing":
        await rig.onset()
        await rig.words(MIC_WORDS)
        now = rig.clock.now_ms
        rig.submit(E.SttEndOfTurn(t_ms=now, t_eot_ms=now - 20, p_end=0.9))
        rig.submit(E.SttFlushed(t_ms=now, t_eot_ms=now - 20))
        await rig.pump()
        return
    if where == "thinking":
        rig.llm.hold()
        await rig.ask(MIC_WORDS)
        await rig.until(
            lambda: rig.state.state is TurnState.THINKING, "never reached THINKING", limit_ms=10_000
        )
        return
    rig.tts.release()
    await rig.ask(MIC_WORDS)
    await rig.until(
        lambda: rig.state.state is TurnState.SPEAKING, "never reached SPEAKING", limit_ms=10_000
    )


@pytest.mark.parametrize("where", BUSY_STATES)
async def test_a_typed_question_while_a_turn_is_live_is_refused_and_changes_nothing(
    tmp_path, where
):
    """Criterion 7: ``astra.error(code="busy")`` and the live turn is byte-identical.

    Compared before and after on the three things a refusal could damage: the machine state, the
    turn's word buffer and the commands the machine issued. ``user.text`` is rowless (criterion 4),
    so the *machine* cannot change anything; what this row guards is the session doing it — an
    onset submitted "just in case", a word buffer seeded into somebody else's turn, an abort.
    """
    published = Recorder()
    async with running(
        tmp_path,
        script=[answer("A long first sentence. ")],
        frames=[frame(0, "a")],
        ms_per_word=300,
        publish=published,
    ) as rig:
        await drive_to(rig, where)
        before = rig.state
        assert before.state is not TurnState.IDLE, f"premise: {where} is not IDLE"
        commands_before = list(rig.machine.commands())
        errors_before = errors(published)
        log_len = len(rig.machine.log)

        rig.submit(text_event(rig.clock.now_ms))
        await rig.pump()

        assert errors(published) == [*errors_before, BUSY], (
            "criterion 7: a typed question while a turn is live is refused with "
            f"astra.error(code='busy'); the page was told {errors(published)}"
        )
        after = rig.state
        assert (after.state, after.turn_id, after.turn_seq, after.words) == (
            before.state,
            before.turn_id,
            before.turn_seq,
            before.words,
        ), (
            f"criterion 7: the live turn is untouched. It was {before.state.value}/"
            f"{before.turn_id} with words {[w.text for w in before.words]}; it is now "
            f"{after.state.value}/{after.turn_id} with {[w.text for w in after.words]}"
        )
        assert list(rig.machine.commands()) == commands_before, (
            "criterion 7: …and the refusal issues no command; the machine additionally emitted "
            f"{[c.type for c in rig.machine.commands()[len(commands_before) :]]}"
        )
        steps = [
            entry for entry in rig.machine.log[log_len:] if type(entry[0]).__name__ == "UserText"
        ]
        assert len(steps) == 1 and tuple(steps[0][2]) == (), (
            "criterion 4/7: the typed question reaches the machine as a rowless event and comes "
            f"back with nothing; it produced {steps}"
        )
        for message in published.messages:
            if isinstance(message, AstraTranscript | AstraSpoken):
                assert TYPED not in message.text, (
                    f"criterion 7: the refused question leaked onto the wire: {message!r}"
                )


async def test_a_microphone_onset_that_wins_the_race_refuses_the_typed_question(tmp_path):
    """Criterion 6, adversarial (T-0028 ruling A14 replayed for text).

    The user types Enter and starts talking in the same instant: the microphone's
    ``user.speech_onset`` is one hop ahead of the onset the session submits for the text. The
    session must notice that the turn now running is not the one *its own* onset opened —
    identity, not shape — and refuse the typed question with ``busy``.

    The failure this exists for is the splice: the typed question seeded into the microphone's
    turn. On the wire that shows up as a ``final=true`` transcript followed by partials (the page
    shows a finished question and then goes back to typing) and in the prompt as a question the
    user never asked in that turn.
    """
    published = Recorder()
    async with running(
        tmp_path,
        script=[answer("Penguins are birds. ")],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=published,
    ) as rig:
        assert rig.state.state is TurnState.IDLE
        at = rig.clock.now_ms
        rig.submit(text_event(at))
        rig.submit(E.UserSpeechOnset(t_ms=at, onset_ms=at))
        await rig.pump()
        await rig.words(MIC_WORDS)
        await rig.end_of_turn()
        await rig.speak_out(limit_ms=10_000)

        assert BUSY in errors(published), (
            "criterion 6: the typed question is refused as busy when a microphone onset opened "
            f"the turn first; the page was told {errors(published)}"
        )
        starts = rig.machine.commands(C.LlmStart)
        assert len(starts) == 1, (
            f"criterion 6: the refused question opened no turn of its own; got "
            f"{[c.transcript for c in starts]}"
        )
        assert starts[0].transcript.split() == MIC_WORDS.split(), (
            "criterion 6: the spoken turn's transcript is the microphone's words and nothing "
            f"else; the machine was given {starts[0].transcript!r}"
        )
        for word in TYPED.split():
            assert word not in starts[0].transcript, (
                f"criterion 6: the typed word {word!r} was spliced into the spoken turn: "
                f"{starts[0].transcript!r}"
            )
        for turn_id, lines in transcripts_by_turn(published).items():
            finals = [final for _, final in lines]
            if True not in finals:
                continue
            assert finals.count(True) == 1 and finals[-1] is True, (
                f"criterion 6 / ruling A14: turn {turn_id}'s final=true transcript is not the "
                f"last thing it sent — partials followed it, which is what a splice looks like "
                f"on the wire. The sequence of `final` flags was {finals}"
            )
            assert all(TYPED not in text for text, _ in lines), (
                f"criterion 6: the typed question appeared in turn {turn_id}'s transcript: {lines}"
            )


async def test_two_typed_questions_in_one_pump_batch_leave_the_second_refused(tmp_path):
    """Criterion 6/7, adversarial: a double Enter (or two windows) in the same batch.

    The first question is already a turn that has not opened yet when the second arrives. Two
    pending turns is the state in which the onset-identity match of ruling A14 has two candidates
    and a splice becomes possible; one question in, one refused, is the only honest outcome.
    """
    published = Recorder()
    second = "and what does the second line mean"
    async with running(
        tmp_path,
        script=[answer("It is oh-my-zsh. ")],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=published,
    ) as rig:
        at = rig.clock.now_ms
        rig.submit(text_event(at))
        rig.submit(text_event(at, second))
        await rig.until(lambda: rig.llm.gen_prompts != [], "no turn opened", limit_ms=5_000)
        await rig.speak_out(limit_ms=10_000)

        starts = rig.machine.commands(C.LlmStart)
        assert [c.transcript for c in starts] == [TYPED], (
            "criterion 6/7: the first typed question runs and the second is refused — never two "
            f"turns and never one turn carrying both; the machine was given "
            f"{[c.transcript for c in starts]}"
        )
        assert errors(published) == [BUSY], (
            f"…and the second is refused with busy; the page was told {errors(published)}"
        )
        assert len(rig.lines()) == 1, f"one turn, one record; got {rig.lines()}"


async def test_the_turn_line_says_how_the_question_arrived(tmp_path):
    """Criterion 9: ``input`` is ``"text"``, ``"ui"`` and ``"voice"`` on the three paths.

    Three turns in one session, because a session mixes them (§5, ambiguity 2) — a field derived
    once per *session* would label the whole file with whatever the first turn was, and a bench
    row would then compare a typed run against an acoustic one without noticing.
    """
    record_input_field()
    published = Recorder()
    async with running(
        tmp_path,
        script=[answer("It is oh-my-zsh. ")],
        frames=[frame(0, "a")],
        ms_per_word=100,
        publish=published,
    ) as rig:
        rig.submit(text_event(rig.clock.now_ms))
        await rig.until(lambda: rig.llm.gen_prompts != [], "no typed turn", limit_ms=5_000)
        await rig.speak_out(limit_ms=10_000)

        rig.llm.script = [answer("Correct. "), ui(canvas_idx=1)]
        rig.submit(
            E.UiAction(
                t_ms=rig.clock.now_ms, surface_id="turn-t1", name="choose", context={"answer": "a"}
            )
        )
        await rig.until(
            lambda: len(rig.lines()) == 2, "the click turn never ended", limit_ms=15_000
        )

        rig.llm.script = [answer("Second answer. ")]
        await rig.ask(MIC_WORDS)
        await rig.speak_out(limit_ms=15_000)

        records = rig.records()
        assert len(records) == 3, f"three turns, three lines; got {len(records)}"
        assert [r.input for r in records] == ["text", "ui", "voice"], (
            "criterion 9: the turns.jsonl line says how the question arrived — typed, clicked, "
            f"spoken; the lines say {[r.input for r in records]}"
        )
        for line, record in zip(rig.lines(), records, strict=True):
            assert json.loads(line)["input"] == record.input, (
                "criterion 9: …and it is on the wire, not only on the object"
            )
        assert {r.regime.audio_path for r in records} == {records[0].regime.audio_path}, (
            "criterion 9: the regime is otherwise unchanged — `input` is a record field, not a "
            "second copy of the audio path (§5, ambiguity 2)"
        )
