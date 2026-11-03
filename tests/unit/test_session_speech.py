"""What reaches TTS and what reaches history (T-0007 criteria 4, 5, 6).

The session builds the machine's ``segment`` callable from a per-turn ``SpeechGate`` using the
adapter shape pinned in ``tests/unit/test_turn_invariants.py``: the machine hands back its whole
unspoken tail, only the suffix beyond ``gate.pending`` is fed, the gate's output goes to the
machine unmodified, and ``close()`` happens only when the machine asks with ``final=True``
(T-0004 A5/A9). The API these tests call is fixed in ``tests/unit/test_session_loop.py``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.conductor.speech_gate import normalise_for_tts
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import TurnState
from tests.unit.test_session_loop import answer, frame, require_session, running, thought


async def test_one_gate_per_turn_and_a_takeover_discards_it(tmp_path):
    """Criterion 4, the session-level twin of ``test_turn_edges.py``'s A9 test.

    Turn 1 commits ``"It is a stack trace. The error is on"``: the gate speaks the first sentence
    and *holds* ``"The error is on"``, which is not a sentence yet. The user cuts in. The held
    half-clause belongs to a question the assistant never finished answering, and ``close()``
    speaks, so the gate is dropped — never closed — and turn 2 gets a fresh one. The failure this
    catches is the held clause surfacing inside the next turn's answer.

    Ruling A8: the generate must still be **open** when the takeover lands. A script that ends
    there gives the machine ``llm.done``, which closes the gate with ``final=True`` and voices the
    tail — correctly, and with no held clause left for the takeover to discard. ``hold_after(1)``
    is the mid-answer interruption the criterion is about; the second commit is the text the user
    cut off before the model finished it.
    """
    script = [answer("It is a stack trace. The error is on"), answer(" line four. ")]
    async with running(tmp_path, script=script, frames=[frame(0, "a")], ms_per_word=200) as rig:
        rig.llm.hold_after(1)
        await rig.ask("what is this")
        await rig.until(
            lambda: rig.state.state is TurnState.SPEAKING, "the first sentence was never spoken"
        )
        gate = rig.session.speech_gate
        assert gate is not None and gate.pending == "The error is on", (
            "criterion 4: the turn's gate holds the unfinished clause; "
            f"session.speech_gate.pending={None if gate is None else gate.pending!r}"
        )
        emitted_before = len(gate.emitted)

        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.until(
            lambda: rig.state.state is TurnState.LISTENING,
            "the takeover never reached LISTENING (INTERRUPTED → playback.stopped)",
        )

        assert gate.pending == "The error is on", (
            "A9: a turn cut off by a takeover discards its gate *without* close(); close() "
            f"speaks, and it emptied the held clause (pending={gate.pending!r})"
        )
        assert len(gate.emitted) == emitted_before, (
            f"close() was called on the interrupted turn's gate: it emitted "
            f"{len(gate.emitted) - emitted_before} more sentence(s)"
        )
        assert rig.session.speech_gate is not gate, "criterion 4: a new gate when a turn opens"

        rig.llm.release()
        await rig.pump()
        rig.llm.script = [answer("A different answer.")]
        await rig.ask("say something else", onset=False)
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the second turn never ended")

        spoken = [text for _, text, _ in rig.tts.requests]
        assert "A different answer." in spoken, f"the second turn must speak; got {spoken}"
        assert not any("The error is on" in text for text in spoken), (
            "the clause the interrupted turn was holding must never be voiced (T-0004 A9)"
        )
        assert not any("line four" in text for text in spoken), (
            "nor the commit the aborted generate was still holding when the user cut in"
        )
        assert not any("The error is on" in turn.assistant_spoken for turn in rig.state.history), (
            "nor stored: history holds what was played"
        )


async def test_an_utterance_synthesised_across_a_barge_in_is_never_played(tmp_path):
    """Criterion 4, adversarial (ruling A10): the speech-cancel latch.

    ``TtsBackend.speak`` is a round trip to another process; a barge-in lands *inside* it more
    often than between utterances. The machine cancels TTS at the takeover, but the reply to a
    request already in flight arrives afterwards, and an implementation that plays whatever comes
    back from ``speak`` voices the interrupted turn's sentence on top of the user — the one thing
    a barge-in exists to stop. The session must therefore latch the cancellation (an epoch, a
    generation counter, whatever) and check it *after* the await, not only before it.

    The mutation this is aimed at: delete the ``_speech_epoch += 1`` in ``cancel_speech``.
    """
    script = [answer("It is a stack trace. ")]
    async with running(tmp_path, script=script, frames=[frame(0, "a")], ms_per_word=200) as rig:
        rig.tts.hold()  # the synthesis is in flight and cannot come back yet
        await rig.ask("what is this")
        await rig.until(lambda: rig.tts.requests != [], "the utterance never reached TtsBackend")
        assert [t for _, t, _ in rig.tts.requests] == ["It is a stack trace."], rig.tts.texts()
        assert [source for _, source in rig.audio.plays if source] == [], (
            "the fixture is wrong: the utterance must still be inside speak() at the takeover"
        )

        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.until(
            lambda: rig.state.state in (TurnState.INTERRUPTED, TurnState.LISTENING),
            "the takeover never left SPEAKING",
        )
        assert rig.machine.commands(C.TtsCancel), "the machine cancels TTS on a takeover"

        rig.tts.release()  # the synthesis comes back, after the user took the floor
        await rig.advance(2_000, step=100)

        assert [source for _, source in rig.audio.plays if source] == [], (
            "criterion 4/14: an utterance whose synthesis was in flight at the barge-in is "
            f"discarded, never played; the device got {rig.audio.plays}"
        )
        assert all("stack trace" not in turn.assistant_spoken for turn in rig.state.history), (
            "and nothing it would have played may reach history"
        )


async def test_empty_tts_text_is_never_spoken(tmp_path):
    """Criterion 5: a latched ``TtsSpeak(text="")`` never reaches ``TtsBackend.speak``.

    The commit is one real sentence followed by one the gate's voice guard refuses (a ``{`` plus
    the quoted A2UI key ``"component"``, T-0052 criterion 9): ruling A20 keeps the refused bytes
    inside ``source`` (conservation) but empties ``text``. The machine still emits the utterance —
    it is the *session* that must not synthesise it, and must not let it into history.

    T-0124 R3: this row used to reach the empty speak through a tools-off ``<end_of_turn>`` (the
    gate's post-EOS latch, T-0004 A2). T-0124 cuts every gate literal before the machine sees it,
    so that path is gone; the voice guard is the gate-made empty speak that no session cleaner
    touches, tools off or on, and does not rest on the tools-on cleaner gap T-0126 closes.
    """
    script = [answer('Four. {"component":"Text"} is the card.')]
    async with running(tmp_path, script=script, frames=[frame(0, "a")]) as rig:
        await rig.ask("how many")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        speaks = rig.machine.commands(C.TtsSpeak)
        latched = [c for c in speaks if not c.text]
        assert latched, (
            "the fixture is wrong: the machine must have emitted at least one latched "
            f"TtsSpeak(text='') here; got {[(c.text, c.source) for c in speaks]}"
        )
        assert [t for _, t, _ in rig.tts.requests] == ["Four."], (
            f"criterion 5: only the non-empty rendering is synthesised; got {rig.tts.texts()}"
        )
        played_sources = [source for _, source in rig.audio.plays if source]
        assert played_sources == ["Four. "], (
            "criterion 5, negative: the skipped utterance contributes nothing to played_text "
            f"(played sources {played_sources})"
        )
        assert rig.state.history[-1].assistant_spoken == "Four. ", (
            f"history got {rig.state.history[-1].assistant_spoken!r}"
        )


async def test_a_thought_token_that_leaks_into_content_is_never_voiced_or_stored(tmp_path):
    """Criterion 6, adversarial: T-0005 M9 — ``<|think|>`` inside an *answer* segment.

    The reasoning parser is not perfect; when the token survives into the answer channel the
    speech gate latches everything after it to ``""`` (T-0004 A2, thinking is never voiced). The
    session must then drop that utterance entirely, which is the only reason the thought text
    also stays out of ``played_text``, out of history, out of the episode store and out of
    ``turns.jsonl``.
    """
    secret = "the user is probably wrong about the deadline"
    script = [answer(f"Here it is. <|think|>{secret}</think> Done.")]
    async with running(tmp_path, script=script, frames=[frame(0, "a")]) as rig:
        await rig.ask("what does it say")
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")

        assert [t for _, t, _ in rig.tts.requests] == ["Here it is."], rig.tts.texts()
        assert all(secret not in text for _, text, _ in rig.tts.requests)
        assert all(secret not in (source or "") for _, source in rig.audio.plays), (
            "the thought bytes travel inside TtsSpeak.source; that utterance is never played"
        )
        assert all(secret not in turn.assistant_spoken for turn in rig.state.history)
        assert all(secret not in line for line in rig.lines()), (
            "criterion 12, negative: no thinking text anywhere in the turns.jsonl line"
        )
        episodes = json.dumps(
            [e.model_dump(mode="json") for e in rig.memory.episodes.values()], default=str
        )
        assert secret not in episodes, "nor in the memory store (T-0009's answer-only rule)"


# --- criterion 6 as a property -----------------------------------------------------------------

#: Answer chunks. Two of them end mid-clause on purpose (the gate then *holds*), and one carries a
#: path so ``source`` (the committed bytes) and ``text`` (the TTS rendering) differ — that is how a
#: session that hands ``source`` to ``speak()`` or ``text`` to ``play(source=…)`` is caught.
ANSWER_CHUNKS = (
    "It is a stack trace. ",
    "The error is on line four. ",
    "Open /etc/hosts now. ",
    "Yes",
    ", and then no. ",
)
#: Sentinel that appears only in thought segments and in commits of a dead request.
SENTINEL = "ZORKMID"

#: ``max_size`` of ``_scripts`` below: the most commits one generated turn can carry.
MAX_CHUNKS = 5
#: The property drives its rig with a fast TTS. ``FakeUtterance`` is ``ms_per_word`` of audio per
#: word of the *normalised* text and the session plays every sentence to the end before the turn
#: can drain, so the wall-clock cost of the property is set here and the deadline follows from it.
PROPERTY_MS_PER_WORD = 50
#: Slack for the part of a turn that is not audio: the 120 ms endpoint settle, the drain mark, the
#: pump's rounds. Virtual time, not a latency claim — this task asserts no latency (criterion 20).
NON_AUDIO_SLACK_MS = 2_000

_scripts = st.lists(
    st.tuples(
        st.sampled_from(("answer", "thought", "ui")),
        st.sampled_from(ANSWER_CHUNKS),
        st.booleans(),
    ),
    min_size=1,
    max_size=MAX_CHUNKS,
)


def worst_case_turn_ms(ms_per_word: int = PROPERTY_MS_PER_WORD) -> int:
    """The longest turn ``_scripts`` can produce, derived from the strategy itself (ruling A14).

    ``MAX_CHUNKS`` commits, each at most the longest normalised chunk in ``ANSWER_CHUNKS``
    (``"Open /etc/hosts now. "`` → ``"Open hosts in the etc folder now."``, seven spoken words),
    each word ``ms_per_word`` of audio that must be played in full before ``playback.drained``.
    A magic number here is a deadline that its own strategy can walk past: at 300 ms/word the
    example the review found (6 + 3×7 = 27 words, 8100 ms) sits just above the 8000 ms that used
    to be written here, and the session drained on the last sample.

    ``normalise_for_tts`` sizes the audio; it is the spoken-length oracle, not the thing under
    test — no assertion in the property depends on it.
    """
    longest = max(len(normalise_for_tts(chunk).split()) for chunk in ANSWER_CHUNKS)
    return MAX_CHUNKS * longest * ms_per_word + NON_AUDIO_SLACK_MS


def test_the_property_deadline_covers_the_scripts_its_own_strategy_generates():
    """Ruling A14: the bound is above the worst script ``_scripts`` can draw, at either speed.

    The witness is the example that failed: ``The error is on line four. `` (6 spoken words) plus
    three ``Open /etc/hosts now. `` (7 each) = 27 words of audio, hand-counted from the normalised
    renderings.
    """
    example = ["The error is on line four. ", *["Open /etc/hosts now. "] * 3]
    words = sum(len(normalise_for_tts(chunk).split()) for chunk in example)
    assert words == 27, [normalise_for_tts(chunk) for chunk in example]
    assert worst_case_turn_ms(300) > words * 300, (
        "the 27-word example needs 8100 ms of playback; the old deadline was 8000"
    )
    assert worst_case_turn_ms(PROPERTY_MS_PER_WORD) > words * PROPERTY_MS_PER_WORD


@given(script=_scripts)
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_only_committed_answer_text_reaches_tts(script):
    """Criterion 6 (Hypothesis, ``--hypothesis-seed=0``, ``max_examples=25`` — each example drives
    a real event loop and a whole turn, so the budget is deliberately small).

    Two properties, with the script itself as the oracle:

    * the concatenated ``TtsSpeak.source`` of the turn equals the concatenation of exactly those
      commits that are ``segment="answer"`` with ``thought_open=False`` — nothing invented,
      nothing reordered, nothing dropped (docs/02 §4; T-0004 A5). An answer canvas that is still
      inside an unclosed thought is never voiced *and never reappears later*, so it is out of the
      oracle as well as out of the machine (ruling A8; CLAUDE.md: never voice a canvas that ends
      inside an unclosed thought);
    * ``HistoryTurn.assistant_spoken`` is the concatenation of the ``source`` strings the session
      actually handed to ``AudioIO.play`` — so history can only hold text that was played.

    The sentinel checks the negative: a token that appeared only in a ``thought`` segment reaches
    no ``speak`` call, no history entry and no ``turns.jsonl`` line.
    """
    require_session()

    async def scenario() -> None:
        commits = []
        for segment, text, open_ in script:
            if segment == "answer":
                commits.append(
                    answer(text, canvas_idx=len(commits)).model_copy(update={"thought_open": open_})
                )
            elif segment == "thought":
                commits.append(thought(f"{SENTINEL} {text}", canvas_idx=len(commits), open_=open_))
            else:
                commits.append(
                    answer(f"{SENTINEL} {text}", canvas_idx=len(commits)).model_copy(
                        update={"segment": "ui", "text": f'[{{"kind":"x","note":"{SENTINEL}"}}]'}
                    )
                )
        expected = "".join(
            text for segment, text, open_ in script if segment == "answer" and not open_
        )
        with tempfile.TemporaryDirectory() as tmp:
            async with running(
                Path(tmp),
                script=commits,
                frames=[frame(0, "a")],
                ms_per_word=PROPERTY_MS_PER_WORD,
            ) as rig:
                await rig.ask("tell me")
                await rig.until(
                    lambda: rig.state.state is TurnState.IDLE,
                    "the turn never ended",
                    limit_ms=worst_case_turn_ms(),
                )

                sources = "".join(c.source for c in rig.machine.commands(C.TtsSpeak))
                assert sources == expected, (
                    f"criterion 6: spoken sources {sources!r} != committed answer {expected!r}"
                )
                for _, text, _ in rig.tts.requests:
                    assert text, "criterion 5: an empty rendering is never sent to TTS"
                    assert SENTINEL not in text

                played = "".join(source for _, source in rig.audio.plays if source)
                spoken = rig.state.history[-1].assistant_spoken if rig.state.history else ""
                assert spoken == played, (
                    f"history {spoken!r} is not the concatenation of the played sources {played!r}"
                )
                assert SENTINEL not in spoken
                assert all(SENTINEL not in line for line in rig.lines()), (
                    "a thought token reached turns.jsonl"
                )

    asyncio.run(scenario())


async def test_a_commit_of_a_dead_request_never_reaches_the_gate(tmp_path):
    """Criterion 6: only commits of the **live** request id may be voiced.

    A late canvas from the previous turn's generate (the engine had not finished aborting) is the
    realistic case. The machine drops it (``ev.request_id != state.active_generate``); this test
    is here because the *session* must not shortcut the machine and feed its gate directly.
    """
    script = [answer("Only this. ")]
    async with running(tmp_path, script=script, frames=[frame(0, "a")]) as rig:
        await rig.ask("hello")
        await rig.until(lambda: rig.state.state is TurnState.SPEAKING, "nothing was spoken")
        now = rig.clock.now_ms
        rig.submit(
            E.LlmCommit(
                t_ms=now,
                request_id="t0-gen-from-a-dead-turn",
                canvas_idx=9,
                segment="answer",
                text=f"{SENTINEL} must not be voiced.",
            )
        )
        await rig.until(lambda: rig.state.state is TurnState.IDLE, "the turn never ended")
        assert all(SENTINEL not in text for _, text, _ in rig.tts.requests), rig.tts.texts()
        assert rig.session.speech_gate is None or SENTINEL not in rig.session.speech_gate.pending
