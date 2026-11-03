"""T-0030 group B: the session publishes what was heard, and stops leaving a partial dangling.

Criteria 8-15 of docs/tasks/T-0030.md, against the ``published`` recorder T-0028 introduced
(``tests/unit/_hud_rig.py``), the fake audio and the scripted clock. No socket, no wall clock, no
real device.

==================================================================================================
THE SEAM (docs/tasks/T-0030.md "Owned files"; the implementer owns the design behind it)
==================================================================================================

``local_astra/conductor/session.py``, narrow mandate — three regions and nothing else:

  (a) the run ledger: which ``(utterance_id, source)`` pairs this playback run published as
      ``astra.spoken`` (a clip, ``source=""``, is never published and is not in the ledger);
  (b) the terminal-playback publication point (``PlaybackStopped``/``PlaybackDrained``): exactly
      one ``AstraSpokenPlayed(turn_id, played_ms, utterances=split_played(ledger, played_text))``
      per run that published at least one utterance;
  (c) the transcript-final path: ``_publish_transcript``, its ``_transcript_final`` fence and the
      turn-close row in ``_rotate_turn`` (criterion 13's final-at-close).

The oracles here are deliberately *not* the session's own egress:

  * what was published as spoken  -> the ``astra.spoken`` messages the recorder already holds;
  * what the device was given     -> ``rig.audio.plays`` (the ``(utterance, source)`` log of the
                                     fake device);
  * what went into history        -> ``rig.session.last_truncation``, which the **executor** set
                                     from the machine's ``TruncateHistoryToPlayed`` command
                                     (criterion 10: the wire and the history are compared across
                                     two different paths, never a string against itself);
  * how many playback runs ended  -> the machine's own event log (``rig.machine.events``).

Regime: profile ``lean`` defaults, ``ScriptedClock``, ``tests/fakes/`` throughout, ms_per_word
200, n = 1 and deterministic per row. No measurement, no GPU, no sockets.
"""

from __future__ import annotations

from typing import Any

from local_astra.contracts import events as E
from local_astra.contracts.protocol import AstraSpoken, AstraTranscript
from local_astra.contracts.turn import TurnState
from tests.unit._hud_rig import Recorder, answer, frame, running

_MISSING: str | None = None
try:
    from local_astra.contracts.protocol import AstraSpokenPlayed
except ImportError as exc:  # the pre-implementation path (group A lands first)
    AstraSpokenPlayed = None  # type: ignore[assignment,misc]
    _MISSING = f"contracts/protocol.py: {exc}"

PLAYED_TYPE = "astra.spoken.played"

#: The two committed sentences every playback fixture below uses, and their hand-counted words.
S1 = "It is a stack trace. "
S2 = "The error is on line four. "
SCRIPT = [answer(S1), answer(S2)]
MS_PER_WORD = 200


def played_of(published: Recorder) -> list[Any]:
    """Every ``astra.spoken.played`` the session published, with the type checked both ways."""
    assert _MISSING is None, (
        f"T-0030 group A is not implemented ({_MISSING}); group B publishes "
        "contracts.protocol.AstraSpokenPlayed (docs/tasks/T-0030.md criteria 1 and 8)"
    )
    by_type = [m for m in published.messages if getattr(m, "type", None) == PLAYED_TYPE]
    typed = [m for m in published.messages if isinstance(m, AstraSpokenPlayed)]
    assert by_type == typed, (
        "criterion 8: the played message is published as the model, not as a look-alike; "
        f"{[type(m).__name__ for m in by_type]}"
    )
    return typed


def only_played(published: Recorder) -> Any:
    """The single ``astra.spoken.played`` of a one-run fixture.

    Not a tuple unpack: ``(message,) = played_of(...)`` raises ``ValueError`` when the list is
    empty, and ``tests/conftest.py`` only tolerates ``AssertionError``/``NotImplementedError``
    from a pending test — the missing message must read as a failed criterion, not as an error.
    """
    messages = played_of(published)
    assert len(messages) == 1, (
        f"criterion 8: exactly one astra.spoken.played for this run; got {messages}"
    )
    return messages[0]


def entries_of(message: Any) -> list[tuple[str, str, int, bool]]:
    return [(e.utterance_id, e.played_text, e.played_words, e.complete) for e in message.utterances]


def spoken_of(published: Recorder, turn_id: str | None = None) -> list[AstraSpoken]:
    return [
        m
        for m in published.messages
        if isinstance(m, AstraSpoken) and (turn_id is None or m.turn_id == turn_id)
    ]


def sources_of(rig: Any, turn_id: str) -> list[tuple[str, str]]:
    """``(utterance_id, source)`` as the *device* received them — clips (``source=""``) excluded."""
    return [
        (u.utterance_id, source)
        for u, source in rig.audio.plays
        if source and u.utterance_id.startswith(f"{turn_id}-")
    ]


def transcripts_of(published: Recorder) -> list[tuple[str, str, bool]]:
    return [
        (m.turn_id, m.text, m.final) for m in published.messages if isinstance(m, AstraTranscript)
    ]


async def barge_in(rig: Any, published: Recorder, *, after_ms: int = 400) -> None:
    """Speak until both utterances are on the device, let ``after_ms`` play, then cut in.

    Hand-computed: at 200 ms/word with the 100 ms guard band (docs/02 §4), 400 ms into the run the
    first two words of the first sentence have started (0 ms and 200 ms, the third starts at
    400 ms and 400 + 100 > 400), so ``played_text`` is ``"It is"``.
    """
    await rig.ask("what is this")
    await rig.until(
        lambda: len([s for _, s in rig.audio.plays if s]) >= 2,
        "both utterances never reached the device",
        limit_ms=10_000,
    )
    await rig.advance(after_ms, step=100)
    now = rig.clock.now_ms
    rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
    await rig.until(
        lambda: rig.state.state in (TurnState.LISTENING, TurnState.INTERRUPTED),
        "the takeover never left SPEAKING",
    )
    await rig.advance(200, step=100)


# --- criteria 8 and 9 --------------------------------------------------------------------------


async def test_one_message_per_playback_run_and_none_during_playback(tmp_path):
    """Criteria 8 and 9: one per run that ended, and nothing while audio is playing.

    Two runs in one session: turn 1 is cut short (``PlaybackStopped``), turn 2 drains
    (``PlaybackDrained``). The count oracle is the machine's own event log — every terminal
    playback event it saw belongs to a run that published utterances in this fixture, so "one per
    run" is checkable without re-deriving it from the session.

    The negative half is taken *during* playback, after both ``astra.spoken`` are on the wire and
    several ticks have fired: an implementation that publishes per audio block, per utterance
    start or on ``Tick`` is caught there rather than by a count at the end.
    """
    published = Recorder()
    async with running(
        tmp_path, script=SCRIPT, frames=[frame(0, "a")], ms_per_word=MS_PER_WORD, publish=published
    ) as rig:
        await rig.ask("what is this")
        await rig.until(
            lambda: len(spoken_of(published)) >= 2,
            "the two utterances were never published as astra.spoken",
            limit_ms=10_000,
        )
        await rig.advance(400, step=100)
        assert rig.state.state is TurnState.SPEAKING, "the fixture must still be playing here"
        assert played_of(published) == [], (
            "criterion 9: nothing is published during playback — not per audio block, not per "
            f"utterance start, not on Tick; got {[m.type for m in published.messages]}"
        )

        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.until(
            lambda: rig.state.state in (TurnState.LISTENING, TurnState.INTERRUPTED),
            "the takeover never left SPEAKING",
        )
        await rig.advance(200, step=100)
        only_played(published)  # criterion 8: the run that just ended, and only it

        # ... and the second turn, which runs to the end.
        await rig.words("and now what")
        await rig.end_of_turn()
        await rig.speak_out()

    terminal = [
        e for e in rig.machine.events() if isinstance(e, E.PlaybackStopped | E.PlaybackDrained)
    ]
    messages = played_of(published)
    assert len(terminal) == 2, f"the fixture must have two playback runs, it had {terminal}"
    assert len(messages) == len(terminal), (
        f"criterion 8: one message per playback run; {len(terminal)} runs ended and "
        f"{len(messages)} messages were published"
    )
    assert [m.turn_id for m in messages] == ["t1", "t2"], (
        f"criterion 8: each carries the turn whose run ended; got {[m.turn_id for m in messages]}"
    )
    for message in messages:
        spoken = spoken_of(published, message.turn_id)
        assert [e.utterance_id for e in message.utterances] == [m.utterance_id for m in spoken], (
            "criterion 8: one entry per utterance published as astra.spoken in that run, in "
            f"publication order; {entries_of(message)} against {[m.utterance_id for m in spoken]}"
        )


async def test_the_barge_in_message_equals_what_went_into_history(tmp_path):
    """Criterion 10: the pane and the history tell the same story, byte for byte.

    The two sides come from different paths on purpose (CLAUDE.md's tautology rule): the left is
    the concatenation of the *published entries*, the right is the ``played_text`` the machine put
    in ``TruncateHistoryToPlayed`` and the **executor** handed to ``Session.note_truncation``.
    """
    published = Recorder()
    async with running(
        tmp_path, script=SCRIPT, frames=[frame(0, "a")], ms_per_word=MS_PER_WORD, publish=published
    ) as rig:
        await barge_in(rig, published)

        played_ms, into_history = rig.session.last_truncation
        assert into_history == "It is", (
            "the fixture must cut two words into the first sentence (200 ms/word, 100 ms guard "
            f"band, cut at 400 ms); the executor recorded {into_history!r}"
        )
        message = only_played(published)
        assert "".join(e.played_text for e in message.utterances) == into_history, (
            "criterion 10: the entries rejoin to exactly the played_text that went into history; "
            f"the wire says {entries_of(message)}, the executor recorded {into_history!r}"
        )
        assert message.played_ms == played_ms, (
            f"criterion 8: and the same played_ms the event carried ({played_ms})"
        )
        assert entries_of(message) == [
            ("t1-u1", "It is", 2, False),
            ("t1-u2", "", 0, False),
        ], "criteria 4-6, hand-computed: a two-word prefix, then nothing at all"
        assert rig.state.history[-1].assistant_spoken.startswith("It is"), (
            "premise (T-0007, unchanged): history holds the played prefix plus [interrupted]"
        )


async def test_a_drained_run_marks_everything_complete(tmp_path):
    """Criterion 11: after a clean drain the page dims nothing.

    ``played_words`` is checked against the sources the *device* was given
    (``rig.audio.plays``), not against the message's own text.
    """
    published = Recorder()
    async with running(
        tmp_path, script=SCRIPT, frames=[frame(0, "a")], ms_per_word=MS_PER_WORD, publish=published
    ) as rig:
        await rig.ask("what is this")
        await rig.speak_out()

        expected = [
            (uid, source, len(source.split()), True) for uid, source in sources_of(rig, "t1")
        ]
        assert expected and [e[2] for e in expected] == [5, 6], (
            f"the fixture's two sentences are 5 and 6 words; the device got {sources_of(rig, 't1')}"
        )
        message = only_played(published)
        assert entries_of(message) == expected, (
            "criterion 11: every entry complete, played_words == len(source.split()); got "
            f"{entries_of(message)}"
        )


async def test_a_clip_only_run_publishes_nothing_and_a_clip_mid_run_only_under_reports(tmp_path):
    """Criterion 12, adversarial: an acknowledgement clip is not an utterance.

    A clip is played with ``source=""`` and is never published as ``astra.spoken`` (it is not
    assistant text and never reaches history), so it contributes no entry — and, critically, it
    must not shift the entry-to-source mapping of the utterances around it. The session's own
    filler policy already enqueues one ``t1-clipack`` ahead of the first sentence in this fixture,
    so the barge-in run below is the mixed case; the first half is the pure one.
    """
    published = Recorder()
    async with running(
        tmp_path, script=SCRIPT, frames=[frame(0, "a")], ms_per_word=MS_PER_WORD, publish=published
    ) as rig:
        await rig.session.play_clip("ack")
        await rig.pump()
        await rig.advance(200, step=100)
        await rig.audio.flush()  # the clip-only run ends
        await rig.advance(200, step=100)
        assert any(
            isinstance(e, E.PlaybackStopped | E.PlaybackDrained) for e in rig.machine.events()
        ), "the fixture must have ended a run here, or the negative below proves nothing"
        assert played_of(published) == [], (
            "criterion 12: a run that published no utterance publishes no message at all"
        )

        await barge_in(rig, published)
        clips = [u.utterance_id for u, source in rig.audio.plays if not source]
        assert clips, "the fixture must contain a clip inside the run (the filler policy's ack)"
        message = only_played(published)
        assert not ({e.utterance_id for e in message.utterances} & set(clips)), (
            f"criterion 12: no entry for a clip; the message holds {entries_of(message)} and the "
            f"device played clips {clips}"
        )
        assert entries_of(message) == [
            ("t1-u1", "It is", 2, False),
            ("t1-u2", "", 0, False),
        ], (
            "criterion 12: the clip must not shift the mapping — the prefix belongs to the first "
            f"*utterance*, not to whatever came after the clip; got {entries_of(message)}"
        )


async def test_a_turn_that_ended_without_a_terminal_playback_event_leaves_no_entries_for_the_next_turn(
    tmp_path,
):
    """Criterion 8, ruling A10: the run ledger belongs to the turn, not to the session.

    A turn can end while audio is still queued and never produce a ``PlaybackStopped`` or
    ``PlaybackDrained`` at all — the client goes away mid-sentence, the turn closes, and the run
    is simply abandoned. Its published utterances must go with it: the *next* turn's message
    reports what the next turn played and nothing else.

    The failure mode is a ledger that is only ever appended to. The next turn then drains, and its
    one message opens with the previous turn's utterances marked ``complete`` — a turn the user
    interrupted, shown as heard in full, under the wrong turn id. Deleting the per-turn clear in
    ``_rotate_turn`` is exactly that mutation.
    """
    published = Recorder()
    async with running(
        tmp_path, script=SCRIPT, frames=[frame(0, "a")], ms_per_word=MS_PER_WORD, publish=published
    ) as rig:
        await rig.ask("what is this")
        await rig.until(
            lambda: len([s for _, s in rig.audio.plays if s]) >= 1,
            "nothing ever reached the device",
            limit_ms=10_000,
        )
        await rig.advance(400, step=100)
        rig.submit(E.ClientDisconnected(t_ms=rig.clock.now_ms))
        await rig.pump()
        await rig.advance(500, step=100)

        assert played_of(published) == [], (
            "criterion 8: no run ended here — the turn was abandoned with audio still queued — "
            f"so there is nothing to report; got {[entries_of(m) for m in played_of(published)]}"
        )
        assert not [
            e for e in rig.machine.events() if isinstance(e, E.PlaybackStopped | E.PlaybackDrained)
        ], "the fixture is wrong: this path must produce no terminal playback event at all"
        assert spoken_of(published, "t1"), "premise: t1 did publish utterances before it died"

        await rig.ask("and now what")
        await rig.speak_out()

        message = only_played(published)
        assert message.turn_id == "t2"
        assert [e.utterance_id for e in message.utterances] == [
            m.utterance_id for m in spoken_of(published, "t2")
        ], (
            "ruling A10: the message reports the utterances *this* run published; it carries "
            f"{entries_of(message)} while t2 published "
            f"{[m.utterance_id for m in spoken_of(published, 't2')]}"
        )
        assert all(e.utterance_id.startswith("t2-") for e in message.utterances), (
            "ruling A10: an entry from the abandoned turn would tell the page that a sentence the "
            f"user cut off was heard in full, under t2's id; got {entries_of(message)}"
        )


# --- criteria 13-15: the transcript's loose end (T-0028 rulings A17/A19) -----------------------


async def test_an_aborted_turn_gets_exactly_one_final_transcript_with_its_own_words(tmp_path):
    """Criterion 13, with the adversarial ordering built in (landmine 3).

    The turn never reaches THINKING (the client goes away while the user is still speaking), so
    after T-0028 ruling A17 it publishes no final at all today and the page keeps typing forever
    under a dead turn id (ruling A19). The close row must publish one — and the next turn's onset
    and first word are submitted in the same batch, so an implementation that publishes
    ``self._state.words`` after ``_rotate_turn`` carries the *next* turn's id and words, or an
    empty string, instead of the closing turn's.
    """
    published = Recorder()
    async with running(
        tmp_path, script=SCRIPT, frames=[frame(0, "a")], ms_per_word=MS_PER_WORD, publish=published
    ) as rig:
        await rig.onset()
        await rig.words("tell me about")
        assert [t for t in transcripts_of(published) if t[2]] == [], (
            "premise (ruling A17): nothing is final before the flush row"
        )
        now = rig.clock.now_ms
        rig.submit(E.ClientDisconnected(t_ms=now))
        rig.submit(E.UserSpeechOnset(t_ms=now + 5, onset_ms=now + 5))
        rig.submit(E.SttWord(t_ms=now + 10, text="next", start_ms=now + 6, end_ms=now + 10))
        await rig.pump()
        await rig.advance(300, step=100)

        lines = transcripts_of(published)
        finals = [line for line in lines if line[2]]
        assert finals == [("t1", "tell me about", True)], (
            "criterion 13: exactly one astra.transcript(final=true), carrying the *closing* "
            f"turn's id and words; the wire carried {lines}"
        )
        index = published.messages.index(
            next(m for m in published.messages if isinstance(m, AstraTranscript) and m.final)
        )
        later = [m for m in published.messages[index:] if getattr(m, "turn_id", "t1") != "t1"]
        assert published.messages[:index] and all(
            getattr(m, "turn_id", "t1") == "t1" for m in published.messages[:index]
        ), (
            "criterion 13: the final is published before any message belonging to the next turn; "
            f"messages before it were {[(m.type, getattr(m, 'turn_id', None)) for m in published.messages[:index]]}"
        )
        assert later, "the fixture must actually produce a next turn, or the ordering is vacuous"
        assert not [
            line for line in lines[lines.index(finals[0]) :] if line[0] == "t1" and not line[2]
        ], "criterion 14: a final is never followed by a partial for the same turn id"


async def test_a_turn_that_went_final_at_the_flush_row_publishes_no_second_final(tmp_path):
    """Criterion 14 *(negative)*: no walk-back, no second final.

    Green today and it must stay green: criterion 13 adds a publication point at the close row,
    and the ``_transcript_final`` fence (T-0028 ruling A17) is what keeps it from firing twice for
    an ordinary turn. Unmarked on purpose — a ``pending`` marker on a row that already passes
    would fail the suite as XPASS(strict).
    """
    published = Recorder()
    async with running(
        tmp_path, script=SCRIPT, frames=[frame(0, "a")], ms_per_word=MS_PER_WORD, publish=published
    ) as rig:
        await rig.ask("what is this")
        await rig.speak_out()

        lines = transcripts_of(published)
        finals = [line for line in lines if line[2]]
        assert finals == [("t1", "what is this", True)], (
            f"criterion 14: exactly one final for a turn that flushed; the wire carried {lines}"
        )
        assert lines[-1][2], "criterion 14: and it is the last transcript line of the turn"


async def test_a_turn_with_no_words_publishes_no_transcript_at_all(tmp_path):
    """Criterion 14 *(negative)*: a turn that said nothing gets no line, final or otherwise.

    The most obvious wrong implementation of criterion 13 — "publish a final at every close" —
    puts an empty transcript line on the page for every aborted onset. Green today; it must stay
    green.
    """
    published = Recorder()
    async with running(
        tmp_path, script=SCRIPT, frames=[frame(0, "a")], ms_per_word=MS_PER_WORD, publish=published
    ) as rig:
        await rig.onset()
        now = rig.clock.now_ms
        rig.submit(E.ClientDisconnected(t_ms=now))
        await rig.pump()
        await rig.advance(300, step=100)

        assert rig.state.state is TurnState.IDLE, "the fixture must have closed the turn"
        assert transcripts_of(published) == [], (
            f"criterion 14: no words, no transcript; the wire carried {transcripts_of(published)}"
        )


async def test_a_ui_opened_turn_publishes_one_final_transcript_before_its_first_spoken(tmp_path):
    """Criterion 15 (T-0028 ruling A19's owed coverage, taken over from T-0020).

    A click-opened turn is complete the moment it opens, so its transcript is final at once and
    stays the only one — including across the new close row (criterion 13), which must not append
    a second copy. Green today; this row is the pin that keeps it that way.
    """
    published = Recorder()
    async with running(
        tmp_path, script=SCRIPT, frames=[frame(0, "a")], ms_per_word=MS_PER_WORD, publish=published
    ) as rig:
        now = rig.clock.now_ms
        rig.submit(E.UiAction(t_ms=now, surface_id="s1", name="next", context={"i": 1}))
        await rig.pump()
        await rig.speak_out()

        lines = transcripts_of(published)
        assert lines == [("t1", '[ui] next {"i":1}', True)], (
            "criterion 15: exactly one astra.transcript(final=true) carrying the [ui] prompt, and "
            f"no partial before or after it; the wire carried {lines}"
        )
        spoken = spoken_of(published)
        assert spoken, "the fixture must speak, or the ordering below is vacuous"
        final_at = published.messages.index(
            next(m for m in published.messages if isinstance(m, AstraTranscript))
        )
        assert final_at < published.messages.index(spoken[0]), (
            "criterion 15 (and T-0028 ruling A2): before the turn's first astra.spoken"
        )
