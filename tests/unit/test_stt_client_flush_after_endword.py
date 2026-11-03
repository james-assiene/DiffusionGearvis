"""The flush still lands after every word it confirms — T-0038 criterion 5.

docs/02 §4 row 6: ``ENDPOINTING`` appends ``stt.word(ts <= t_eot)`` and leaves on the flush, so
``SttFlushedEvent(t_eot)`` promises that every word up to ``t_eot`` has already been delivered.
T-0038 makes a word wait for its ``EndWord``, which is exactly the change that could break that
promise: a word still open when the ``Marker`` echo arrives must be published **before** the
flush, with ``end_ms = start_ms`` (never dropped, never dated from the fed-audio clock), and a
late ``EndWord`` must not publish it a second time.

Times are the group B census's (``logs/t0007a-group-b-census-f114adf.log``, 2026-09-14, commit
f114adf, live 0.6.4 speech server on CPU, fixture ``pocket_tts_sentence.wav``, n=1): words at
0.72 1.44 2.24 3.04 3.76 s, ends at 1.44 2.24 3.04 3.76 s, converted to ms by hand with the
connection base at 0. Pure: no socket, no service.
"""

from __future__ import annotations

from tests.fakes.moshi_stt_double import end_word, types_of, word
from tests.unit.test_stt_client_endword import CENSUS_MS, block, connected


async def test_a_word_with_no_endword_is_published_at_the_marker_echo_before_the_flush():
    """Criterion 5: the open word is closed at its own start, not at the fed-audio clock."""
    client, double, drain = await connected()
    for i in range(1, 10):  # nine more 80 ms blocks: the fed-audio clock is at 800 ms
        await client.feed(block(), i * 80)
    double.push(word("tail", 0.5))
    await client.mark(1000)

    events = await drain.take(2)
    assert types_of(events) == ["word", "flushed"], (
        f"the flush must land after the words it confirms: {types_of(events)}"
    )
    assert (events[0].text, events[0].start_ms, events[0].end_ms) == ("tail", 500, 500), (
        "a word with no end takes its own start as its end — never the 800 ms audio clock"
    )
    assert events[1].t_eot_ms == 1000
    assert events[0].end_ms <= events[1].t_eot_ms
    assert client.stats().get("words_without_end") == 1
    assert client.stats()["unknown_messages"] == 0
    await client.close()


async def test_every_censused_word_is_delivered_before_the_flush_that_confirms_it():
    """Criterion 5 with the real sentence: four closed words, one still open, then the echo."""
    client, double, drain = await connected()
    pairs = [(0.72, 1.44), (1.44, 2.24), (2.24, 3.04), (3.04, 3.76)]
    for (start_s, stop_s), (text, _, _) in zip(pairs, CENSUS_MS, strict=False):
        double.push(word(text, start_s))
        double.push(end_word(stop_s))
    double.push(word("five.", 3.76))  # the census's fifth EndWord never arrived
    t_eot_ms = 4200
    await client.mark(t_eot_ms)

    events = await drain.take(6)
    assert types_of(events) == ["word"] * 5 + ["flushed"], types_of(events)
    assert [(e.text, e.start_ms, e.end_ms) for e in events[:5]] == [
        *CENSUS_MS[:4],
        ("five.", 3760, 3760),
    ]
    assert all(e.end_ms <= t_eot_ms for e in events[:5]), (
        "the flush may only confirm words that ended at or before t_eot"
    )
    assert events[5].t_eot_ms == t_eot_ms
    assert client.stats()["unknown_messages"] == 0
    assert client.stats().get("words_without_end") == 1
    await client.close()


async def test_an_endword_arriving_after_the_flush_does_not_publish_the_word_twice():
    """Adversarial: the word closed at the echo is final; the late end is a stray, not a repeat.

    A second word event for the same word would be appended to a turn that has already been
    endpointed (docs/02 §4 row 6), and a duplicated word in the transcript is the kind of bug
    that only shows up in a live session.
    """
    client, double, drain = await connected()
    double.push(word("five.", 3.76))
    await client.mark(4200)
    events = await drain.take(2)
    assert types_of(events) == ["word", "flushed"]

    double.push(end_word(4.04))  # the end the server was still holding when we flushed
    late = await drain.quiet(window_s=0.25)
    assert types_of(late) == [], f"the late EndWord produced {types_of(late)}"
    assert client.stats().get("stray_end_words") == 1
    assert client.stats()["unknown_messages"] == 0
    await client.close()
