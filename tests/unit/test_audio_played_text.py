"""``played_text``: T-0006 criterion 6 (docs/02 §4 + the 2026-09-13 spec delta).

The rule under test, stated once: ``played_text`` is the in-order concatenation, with **no
separator inserted**, of the ``TtsSpeak.source`` strings of fully played utterances plus the
word-boundary prefix of the partially played one, where a word counts as played iff
``utterance_offset_ms + word.start_ms + 100 <= played_ms``; and if
``len(source.split()) != len(utterance.words())`` the partially played utterance contributes no
text at all. Every expected string below is written out by hand from that rule and the word
tables in this file — nothing is computed by asking the module.

Cuts are placed at least 100 ms away from a guard-band boundary so that criterion 2's tolerance
(over-report ≤ 20 ms, under-report ≤ 40 ms) can never flip an expectation.
"""

from __future__ import annotations

import pytest

from local_astra.contracts.events import PlaybackDrained, PlaybackStopped
from tests.conftest import VirtualClock
from tests.fakes.audio_device import (
    EventSink,
    FakeAudioDevice,
    PcmUtterance,
    call,
    make_audio_io,
    play,
    play_out,
    played_ms,
    tone,
    words_of,
)

#: "alpha bravo charlie delta": four words, the second one long, so the guard-band rule
#: ("started ≥ 100 ms before the cut") and a naive "the word finished" rule disagree at 600 ms.
ABCD_SOURCE = "alpha bravo charlie delta"
ABCD_WORDS = words_of(
    [("alpha", 0, 300), ("bravo", 400, 1100), ("charlie", 1200, 1500), ("delta", 1600, 1900)]
)
ABCD_MS = 2000.0

#: The normalisation mismatch of criterion 6: two source words, four rendered words.
APPLES_SOURCE = "3.5 apples"
APPLES_TTS_TEXT = "three point five apples"
APPLES_WORDS = words_of(
    [("three", 0, 200), ("point", 250, 400), ("five", 450, 650), ("apples", 700, 1000)]
)
APPLES_MS = 1100.0


def abcd(uid: str = "abcd") -> PcmUtterance:
    return PcmUtterance(uid, samples=tone(ABCD_MS), words=ABCD_WORDS, text=ABCD_SOURCE)


def apples(uid: str = "apples") -> PcmUtterance:
    return PcmUtterance(uid, samples=tone(APPLES_MS), words=APPLES_WORDS, text=APPLES_TTS_TEXT)


def hello(uid: str = "hello", text: str = "Hello, ") -> PcmUtterance:
    return PcmUtterance(
        uid, samples=tone(400.0), words=words_of([(text.strip(), 0, 300)]), text=text
    )


def rig() -> tuple[object, FakeAudioDevice, EventSink]:
    device = FakeAudioDevice(VirtualClock(), buffer_ms=200.0)
    io = make_audio_io(device)
    return io, device, EventSink(io)


async def cut_at(io: object, device: FakeAudioDevice, sink: EventSink, ms: float) -> str:
    """Play up to ``ms`` of virtual time, flush, and return the ``played_text`` of the run."""
    await play_out(device, ms)
    await call(io, "flush", "AudioIO protocol")
    await sink.sync()
    stopped = sink.of_type(PlaybackStopped)
    assert len(stopped) == 1, f"expected one PlaybackStopped, got {stopped}"
    return str(stopped[0].played_text)


@pytest.mark.parametrize(
    ("cut_ms", "expected"),
    [
        # bravo starts at 400, so it counts as played from played_ms = 500 on.
        (400.0, "alpha"),
        (600.0, "alpha bravo"),
        # charlie starts at 1200 -> boundary 1300; delta starts at 1600 -> boundary 1700.
        (1200.0, "alpha bravo"),
        (1400.0, "alpha bravo charlie"),
        (1800.0, "alpha bravo charlie delta"),
    ],
)
async def test_the_guard_band_decides_the_last_word(cut_ms: float, expected: str) -> None:
    """Criterion 6: a word counts as played iff it started ≥ 100 ms before the cut.

    At cut 600 the guard-band rule keeps "bravo" (started at 400) while a "the word finished"
    rule would drop it (it ends at 1100); at cut 400 both drop it. At cut 1200 a rule without
    the guard band would already include "charlie" (it starts exactly at 1200).
    """
    io, device, sink = rig()
    await play(io, abcd(), source=ABCD_SOURCE)
    assert await cut_at(io, device, sink, cut_ms) == expected
    await sink.close()


async def test_the_played_prefix_is_never_longer_than_what_was_heard() -> None:
    """docs/02 §4: the history never contains assistant text that was not played.

    Independent of the exact rule: whatever the module returns must be a prefix of the source
    (word-wise) and every word in it must have *started* no later than the reported cut.
    Under-reporting is allowed, over-reporting is the defect.
    """
    for cut_ms in (150.0, 350.0, 700.0, 950.0, 1450.0, 1950.0):
        io, device, sink = rig()
        await play(io, abcd(), source=ABCD_SOURCE)
        await play_out(device, cut_ms)
        reported = await played_ms(io)
        text = await cut_at(io, device, sink, 0.0)
        words = text.split()
        assert words == ABCD_SOURCE.split()[: len(words)], (
            f"cut {cut_ms} ms: {text!r} is not a word-prefix of {ABCD_SOURCE!r}"
        )
        for i, word in enumerate(words):
            start = ABCD_WORDS[i].start_ms
            assert start <= reported, (
                f"cut {cut_ms} ms: {word!r} starts at {start} ms but only {reported:.0f} ms "
                f"were played — history would hold text the user never heard"
            )
        await sink.close()


async def test_utterances_are_concatenated_with_no_separator() -> None:
    """Criterion 6: the sources are exact substrings of the committed answer (T-0003 c10)."""
    io, device, sink = rig()
    await play(io, hello("u1", "Hello, "), source="Hello, ")
    await play(io, hello("u2", "world."), source="world.")
    await call(io, "mark_drain", "criterion 7")
    await play_out(device, 1200.0)
    await sink.sync()
    drained = sink.of_type(PlaybackDrained)
    assert len(drained) == 1, f"expected one PlaybackDrained, got {drained}"
    assert drained[0].played_text == "Hello, world.", (
        "criterion 6 inserts no separator: the space is the one already in the first source"
    )
    await sink.close()


async def test_a_full_utterance_plus_a_partial_one() -> None:
    """Criterion 6: whole sources for what finished, a word-boundary prefix for what did not."""
    io, device, sink = rig()
    await play(io, hello("u1", "Hello, "), source="Hello, ")
    await play(io, abcd("u2"), source=ABCD_SOURCE)
    # 400 ms of "Hello, " then 600 ms of the second utterance: bravo started at 400 (+ 400 ms
    # of offset = 800 on the run clock, boundary 900) and the cut is at 1000.
    assert await cut_at(io, device, sink, 1000.0) == "Hello, alpha bravo"
    await sink.close()


async def test_a_normalisation_mismatch_contributes_no_text_when_partial() -> None:
    """Criterion 6, adversarial: "3.5 apples" rendered as four words, cut after two of them.

    The partial utterance contributes nothing — never "3.5 apples", which would put two words
    into history that the user did not hear, and never "3.5" or "three point", which are not
    substrings of the committed answer.
    """
    io, device, sink = rig()
    await play(io, apples(), source=APPLES_SOURCE)
    await play_out(device, 500.0)  # "three" and "point" played; "five" starts at 450 -> 550
    reported = await played_ms(io)
    text = await cut_at(io, device, sink, 0.0)
    assert text == "", (
        f"criterion 6: len({APPLES_SOURCE!r}.split()) == 2 but the rendering has 4 words, so a "
        f"partially played utterance contributes no text; got {text!r}"
    )
    assert reported >= 400.0, (
        f"the mismatched utterance must still contribute to played_ms, got {reported:.0f} ms"
    )
    await sink.close()


async def test_a_normalisation_mismatch_still_contributes_when_fully_played() -> None:
    """Criterion 6: the ruling is about the *partially* played utterance only."""
    io, device, sink = rig()
    await play(io, apples(), source=APPLES_SOURCE)
    await call(io, "mark_drain", "criterion 7")
    await play_out(device, 1400.0)
    await sink.sync()
    drained = sink.of_type(PlaybackDrained)
    assert len(drained) == 1, f"expected one PlaybackDrained, got {drained}"
    assert drained[0].played_text == APPLES_SOURCE, (
        f"a fully played utterance contributes its whole source, got {drained[0].played_text!r}"
    )
    await sink.close()


async def test_source_none_falls_back_to_the_tts_text() -> None:
    """Criterion 6: ``source=None`` -> the TTS text is used (and the span says so)."""
    io, device, sink = rig()
    await play(io, apples(), source=None)
    await call(io, "mark_drain", "criterion 7")
    await play_out(device, 1400.0)
    await sink.sync()
    drained = sink.of_type(PlaybackDrained)
    assert len(drained) == 1, f"expected one PlaybackDrained, got {drained}"
    assert drained[0].played_text == APPLES_TTS_TEXT, (
        f"with source=None the TTS text is the fallback, got {drained[0].played_text!r}"
    )
    await sink.close()


async def test_a_partial_fallback_is_cut_on_the_tts_word_boundaries() -> None:
    """Criterion 6: with ``source=None`` there is no mismatch — the words *are* the text."""
    io, device, sink = rig()
    await play(io, apples(), source=None)
    # cut at 500: "three" (0) and "point" (250) started ≥ 100 ms ago, "five" (450) did not.
    assert await cut_at(io, device, sink, 500.0) == "three point"
    await sink.close()
