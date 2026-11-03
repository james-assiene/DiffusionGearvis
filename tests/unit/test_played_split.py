"""T-0030 group A: ``conductor/played_split.py`` — who actually heard what, split per utterance.

Criteria 3-6 of docs/tasks/T-0030.md. The splitter takes the run's published utterances and the
run's ``played_text`` (T-0006's rule, docs/02 §4) and says, per utterance, how much of it reached
the device. It is pure: this module imports it and calls it, nothing else.

==================================================================================================
THE SEAM (the implementer owns the design behind it; these are the names the tests call)
==================================================================================================

``local_astra/contracts/protocol.py`` (criterion 1)

    class PlayedUtterance(Contract):
        utterance_id: str
        played_text: str                  # a *word prefix* of that utterance's source, or ""
        played_words: int = Field(ge=0)   # == len(played_text.split())
        complete: bool                    # the whole source was covered

    class AstraSpokenPlayed(Contract):
        type: Literal["astra.spoken.played"]
        turn_id: str
        played_ms: int = Field(ge=0)
        utterances: list[PlayedUtterance]

``local_astra/conductor/played_split.py`` (criterion 3)

    split_played(utterances: Sequence[tuple[str, str]], played_text: str) -> list[PlayedUtterance]

        ``utterances`` are ``(utterance_id, source)`` pairs in publication order — the sources are
        the committed strings ``played_text`` is built from (``TtsSpeak.source``, *not* the
        TTS-normalised ``AstraSpoken.text``; landmine 1). One entry out per pair in, always.

The oracle for what ``played_text`` looks like is ``tests/fakes/audio.py::played_text`` and
``local_astra/conductor/audio_io.py::_played_text``: whole sources concatenated **with no
separator**, plus a word-boundary prefix of the utterance the cut fell inside. Every expected
value below is hand-computed from that rule (300 ms/word, 100 ms guard band) or drawn from the
fake itself in the Hypothesis row.

Regime: pure functions, no clock, no I/O, n = 1 per row; Hypothesis derandomised so the property
row is reproducible under the plain fast-suite command.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.conftest import VirtualClock
from tests.fakes.audio import GUARD_BAND_MS, FakeAudioIO
from tests.fakes.tts import FakeUtterance

_MISSING: dict[str, str] = {}

try:
    from local_astra.conductor.played_split import split_played as _split_played
except ImportError as exc:  # the pre-implementation path
    _split_played = None  # type: ignore[assignment]
    _MISSING["played_split"] = f"local_astra/conductor/played_split.py: {exc}"

try:
    from local_astra.contracts.protocol import PlayedUtterance as _PlayedUtterance
except ImportError as exc:  # the pre-implementation path
    _PlayedUtterance = None  # type: ignore[assignment]
    _MISSING["PlayedUtterance"] = f"contracts.protocol.PlayedUtterance: {exc}"

FIELDS = ("utterance_id", "played_text", "played_words", "complete")


def split(pairs: Any, played_text: str) -> list[Any]:
    """``split_played`` behind an assertion, so the pending window fails loudly but tolerably.

    ``tests/conftest.py`` only tolerates ``NotImplementedError``/``AssertionError`` from a pending
    test; an ``ImportError`` at collection time would take the whole module red instead.
    """
    assert not _MISSING, (
        "T-0030 group A is not implemented: "
        + "; ".join(sorted(_MISSING.values()))
        + " (docs/tasks/T-0030.md criteria 1 and 3; this module's docstring fixes the seam)"
    )
    out = _split_played(list(pairs), played_text)
    assert isinstance(out, list), f"split_played returns a list of PlayedUtterance, got {out!r}"
    assert len(out) == len(list(pairs)), (
        "criterion 3: one entry per published utterance, in publication order; "
        f"{len(list(pairs))} pairs in, {len(out)} entries out"
    )
    for entry in out:
        assert isinstance(entry, _PlayedUtterance), (
            f"criterion 1: the entries are contracts.protocol.PlayedUtterance, got {entry!r}"
        )
        for name in FIELDS:
            assert hasattr(entry, name), f"criterion 1: PlayedUtterance needs {name!r}: {entry!r}"
    return out


def rows(entries: list[Any]) -> list[tuple[str, str, int, bool]]:
    return [(e.utterance_id, e.played_text, e.played_words, e.complete) for e in entries]


# --- the fixtures, and what T-0006's rule makes of them ----------------------------------------

#: Three committed sources, exactly as ``TtsSpeak.source`` carries them (trailing space included:
#: T-0003 criterion 10 makes the sources exact substrings of the committed answer).
S1 = "It is a stack trace. "
S2 = "The error is on line four. "
S3 = "Shall I fix it? "
THREE = (("t1-u1", S1), ("t1-u2", S2), ("t1-u3", S3))

#: Hand-counted word counts of the three sources: 5, 6, 4.
WORDS = (5, 6, 4)


def test_a_full_run_marks_every_utterance_complete():
    """Criterion 4: the whole concatenation → three complete entries, word counts per source."""
    played = S1 + S2 + S3
    assert (len(S1.split()), len(S2.split()), len(S3.split())) == WORDS, "the fixture drifted"
    assert rows(split(THREE, played)) == [
        ("t1-u1", S1, 5, True),
        ("t1-u2", S2, 6, True),
        ("t1-u3", S3, 4, True),
    ]


def test_a_cut_mid_utterance_splits_into_whole_then_prefix_then_nothing():
    """Criteria 4-6, hand-computed.

    The user cut in two words into the second sentence, so T-0006 emitted the whole first source
    followed by ``"The error"`` and stopped (``_played_text`` breaks at the first incomplete
    item). The page must dim the rest of sentence two and the whole of sentence three.
    """
    played = S1 + "The error"
    assert rows(split(THREE, played)) == [
        ("t1-u1", S1, 5, True),
        ("t1-u2", "The error", 2, False),
        ("t1-u3", "", 0, False),
    ]
    assert "".join(e.played_text for e in split(THREE, played)) == played, (
        "criterion 4: conservation — the entries' texts rejoin to exactly played_text"
    )


def test_entries_are_monotone():
    """Criterion 6: once an entry is not complete, no later entry has played text.

    Four inputs: the clean run (vacuously monotone), the mid-utterance cut, a *gap* — a
    played_text whose middle does not reconcile with the middle source while its tail looks
    exactly like the third source — and, ruling A6, the **clean skip**: a played_text in which the
    middle source is simply absent and the remainder *is* the third source, byte for byte.

    The clean skip is the discriminating one. In the gap case the remainder is corrupted
    (``"XXDelta. "``), so even a splitter with no monotonicity clamp fails to match the third
    source and the row stays green with the clamp deleted — it proves nothing. Here the remainder
    after the first entry is exactly ``"Delta. "``, so the only thing that can stop the third
    entry from reporting itself heard is the clamp: audio is rendered in order (docs/02 §4), a
    source that did not play cannot be skipped over, and "played, unplayed, played" is a bug.
    """
    gap_sources = (("g1", "Alpha beta. "), ("g2", "Gamma. "), ("g3", "Delta. "))
    gap_played = "Alpha beta. " + "XX" + "Delta. "
    skip_sources = (("g1", "Alpha. "), ("g2", "Gamma. "), ("g3", "Delta. "))
    skip_played = "Alpha. Delta. "
    assert skip_played == "Alpha. " + "Delta. ", (
        "the fixture is only discriminating if the remainder after g1 is *exactly* g3's source"
    )
    for pairs, played, what in (
        (THREE, S1 + S2 + S3, "the clean run"),
        (THREE, S1 + "The error", "the mid-utterance cut"),
        (gap_sources, gap_played, "the gap"),
        (skip_sources, skip_played, "the clean skip"),
    ):
        entries = split(pairs, played)
        seen_incomplete = False
        for entry in entries:
            if seen_incomplete:
                assert (entry.played_text, entry.played_words, entry.complete) == ("", 0, False), (
                    f"criterion 6 ({what}): {entry.utterance_id} reports played text after an "
                    f"incomplete entry — a played/unplayed/played pattern is a bug, not a state; "
                    f"got {rows(entries)}"
                )
            if not entry.complete:
                seen_incomplete = True
    assert rows(split(gap_sources, gap_played)) == [
        ("g1", "Alpha beta. ", 2, True),
        ("g2", "", 0, False),
        ("g3", "", 0, False),
    ], "criterion 5: from the unreconcilable entry on, nothing is claimed as heard"
    assert rows(split(skip_sources, skip_played)) == [
        ("g1", "Alpha. ", 1, True),
        ("g2", "", 0, False),
        ("g3", "", 0, False),
    ], (
        "criterion 6, ruling A6: the remainder after g1 is exactly g3's source, and g3 must "
        "*still* report nothing — a splitter that keeps matching after an entry failed tells the "
        "page a sentence was heard that the device never reached"
    )


def test_the_no_separator_boundary_merge_does_not_steal_a_word():
    """Criterion 5, adversarial — landmine 2, the most likely failure mode.

    ``audio_io._played_text`` and ``tests/fakes/audio.py::played_text`` both ``"".join(...)`` the
    whole sources, so two sentences with no trailing space meet inside one token:
    ``"Hello." + "World." == "Hello.World."``, whose ``.split()`` is the single token
    ``["Hello.World."]``. A token-wise splitter gives that token to the first utterance and
    reports the second as unheard — the page then strikes through a sentence the user did hear,
    which is the over-report docs/02 §4 forbids.
    """
    pairs = (("b1", "Hello."), ("b2", "World."))
    played = "Hello.World."
    assert played.split() == ["Hello.World."], "the fixture is only interesting if it merges"
    assert rows(split(pairs, played)) == [
        ("b1", "Hello.", 1, True),
        ("b2", "World.", 1, True),
    ]


@pytest.mark.parametrize(
    ("source", "played", "exact"),
    [
        # Ruling A2: this is the spelling both the real module and the fake now produce
        # (``audio_io._word_prefix`` = ``source[:index]``), so it must reconcile, not degrade.
        ("Hello   world\nagain", "Hello   world", True),
        # The re-joined spelling is what T-0006's rule looked like before ruling A2 and what
        # criterion 5 names; a splitter may reconcile it or refuse it, never mis-attribute it.
        ("Hello   world\nagain", "Hello world", False),
    ],
    ids=["literal-spacing", "normalised"],
)
def test_a_whitespace_normalised_partial_is_still_a_prefix_of_its_source(source, played, exact):
    """Criterion 5 and ruling A2: the partial is a *word* prefix, whatever the spacing.

    ``" ".join(words[:n])`` is not always a literal substring of the source, so a splitter that
    only does ``source.startswith(rest)`` must fall back to the conservative answer rather than
    invent one — for that spelling either outcome is allowed. For the literal spelling there is no
    such licence: it is what the playback module and the fake both emit, so an empty entry there
    would under-report a sentence the user heard half of.
    """
    (entry,) = split((("w1", source),), played)
    assert entry.played_words == len(entry.played_text.split())
    assert entry.played_text.split() == source.split()[: entry.played_words], (
        f"criterion 5: {entry.played_text!r} is not a word prefix of {source!r}"
    )
    if exact:
        assert rows([entry]) == [("w1", played, 2, False)], (
            "ruling A2: the literal-spacing partial is the contract's own output and must "
            f"reconcile exactly; got {rows([entry])}"
        )
    else:
        assert (entry.played_words, entry.complete) in ((2, False), (0, False)), (
            "criterion 5: either the partial reconciles to its two words, or the conservative "
            f"empty entry; got {rows([entry])}"
        )


def test_an_unreconcilable_played_text_reports_nothing_rather_than_guessing():
    """Criterion 5: under-report, never over-report — and never raise."""
    assert rows(split(THREE, "Something else entirely.")) == [
        ("t1-u1", "", 0, False),
        ("t1-u2", "", 0, False),
        ("t1-u3", "", 0, False),
    ]
    assert rows(split(THREE, "")) == [
        ("t1-u1", "", 0, False),
        ("t1-u2", "", 0, False),
        ("t1-u3", "", 0, False),
    ]
    assert split((), "") == [], "no utterances in, no entries out"
    (empty,) = split((("z1", ""),), "")
    assert (empty.played_text, empty.played_words) == ("", 0), (
        "an empty source has no words to claim either way (``complete`` is the implementer's "
        f"call here: there is nothing to dim); got {rows([empty])}"
    )


# --- the property: the splitter inverts the audio module's own rule ---------------------------

#: Single-spaced sources, so T-0006's partial (``" ".join(words[:n])``) is also a literal prefix
#: and conservation (criterion 4) is exact. The words are picked so that boundaries merge
#: ("Hello." + "World.") as often as they do not.
WORD = st.sampled_from(["Hello.", "World.", "a", "is", "3.5", "line", "four."])
SOURCE = st.lists(WORD, min_size=1, max_size=4).map(" ".join)

MS_PER_WORD = 300


def build_run(sources: list[str], cut_ms: int) -> str:
    """``played_text`` for this run and this cut, straight from the fake (the oracle)."""

    async def go() -> str:
        clock = VirtualClock()
        audio = FakeAudioIO(clock)
        for i, source in enumerate(sources):
            utterance = FakeUtterance(f"u{i}", source, ms_per_word=MS_PER_WORD)
            await audio.play(utterance, source=source)
        return audio.played_text(cut_ms)

    return asyncio.run(go())


@settings(
    deadline=None,
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    sources=st.lists(SOURCE, min_size=1, max_size=4), cut=st.integers(min_value=0, max_value=4000)
)
def test_split_played_inverts_the_fake_audio_oracle(sources, cut):
    """Criteria 4 and 5 as a property: the splitter must invert ``tests/fakes/audio.py``.

    The fake builds ``played_text`` from the sources and the word timings (it is the independent
    oracle: it knows the cut in milliseconds, the splitter only ever sees the string). This row
    fails if the two disagree about anything — the boundary merge, the guard band, the empty
    partial, the run that ended before its first word.

    ``complete`` is deliberately *not* compared against "the device finished this utterance": with
    ``GUARD_BAND_MS`` = 100 < one word, the last word of an utterance counts as played up to
    ``ms_per_word - guard`` before its audio ends, so the text of a not-quite-finished utterance
    can be the whole source. Text is all the splitter has; what is asserted is that ``complete``
    means *the text was fully covered*.
    """
    assert GUARD_BAND_MS < MS_PER_WORD, "the fixture assumes the guard band is under one word"
    pairs = [(f"u{i}", source) for i, source in enumerate(sources)]
    played = build_run(sources, cut)
    entries = split(pairs, played)

    assert "".join(e.played_text for e in entries) == played, (
        f"criterion 4 (conservation): {rows(entries)} does not rejoin to {played!r}"
    )
    seen_incomplete = False
    for entry, (_, source) in zip(entries, pairs, strict=True):
        assert entry.played_words == len(entry.played_text.split())
        assert entry.played_text.split() == source.split()[: entry.played_words], (
            f"criterion 5: {entry.played_text!r} is not a word prefix of {source!r}"
        )
        assert entry.complete == (entry.played_text == source), (
            f"criterion 4/5: complete means the entry covers the whole source; {rows(entries)} "
            f"against {pairs}"
        )
        if seen_incomplete:
            assert entry.played_text == "", f"criterion 6: {rows(entries)} for cut={cut}"
        seen_incomplete = seen_incomplete or not entry.complete


def test_split_played_is_pure():
    """Criterion 3: the same discipline as ``turn_machine.py`` — no clock, no I/O, no session."""
    assert not _MISSING, (
        "T-0030 criterion 3 adds local_astra/conductor/played_split.py: "
        + "; ".join(sorted(_MISSING.values()))
    )
    import local_astra.conductor.played_split as module

    source = pathlib.Path(module.__file__).read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    banned = {"asyncio", "time", "random", "logging", "os", "pathlib", "httpx", "threading"}
    assert not banned & {name.split(".")[0] for name in imported}, sorted(imported)
    forbidden = (
        "local_astra.conductor.session",
        "local_astra.contracts.events",
        "local_astra.conductor.audio_io",
    )
    assert not [n for n in imported if n.startswith(forbidden)], (
        "criterion 3: the splitter takes strings and returns models; importing the session, the "
        f"event set or the playback module would make it another seam. Got {sorted(imported)}"
    )
