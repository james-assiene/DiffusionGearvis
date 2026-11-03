"""Split a playback run's ``played_text`` back into one entry per published utterance.

T-0030 group A, criteria 3-6. The run's terminal event (``PlaybackStopped`` / ``PlaybackDrained``)
carries a single string built by ``audio_io._played_text``: the committed sources of the utterances
that finished, concatenated **with no separator**, plus — for the one the cut fell inside — a word
prefix of its source in the source's own spelling (``_word_prefix`` returns ``source[:index]``).
This module inverts that rule so the HUD can dim exactly the tail nobody heard.

Pure by the same discipline as ``turn_machine.py`` (docs/02 §4): strings in, models out. No clock,
no I/O, no asyncio, no import of the session, the event set or the playback module — the word-prefix
rule is small enough to state twice, and the conformance of the two statements is what
``tests/unit/test_played_split.py`` asserts against the fake audio oracle.

Two rules decide every ambiguous case, both from docs/02 §4's invariant that the history (and now
the pane) never claims text that was not played:

- **In order.** Audio is rendered in publication order and ``_played_text`` breaks at the first
  item that did not finish, so the match walks the string left to right and stops at the first
  entry that does not reconcile. A "played, unplayed, played" pattern is a bug, not a state.
- **Under-report, never over-report.** Where the string cannot be reconciled with the sources —
  a whitespace-normalised partial that is not a literal prefix, a truncated string, a mismatch —
  that entry and every entry after it report nothing heard rather than a guess.
"""

from __future__ import annotations

from collections.abc import Sequence

from local_astra.contracts.protocol import PlayedUtterance


def _word_prefix(source: str, words: int) -> str:
    """The first ``words`` whitespace-separated tokens of ``source``, spacing kept as written.

    The same rule as ``audio_io._word_prefix``: a prefix *of the committed source*, not a re-joined
    one, so that a double space or a newline survives.
    """
    if words <= 0:
        return ""
    seen, index, length = 0, 0, len(source)
    while index < length:
        while index < length and source[index].isspace():
            index += 1
        start = index
        while index < length and not source[index].isspace():
            index += 1
        if index > start:
            seen += 1
            if seen == words:
                return source[:index]
    return source


def _entry(utterance_id: str, played_text: str, *, complete: bool) -> PlayedUtterance:
    return PlayedUtterance(
        utterance_id=utterance_id,
        played_text=played_text,
        played_words=len(played_text.split()),
        complete=complete,
    )


def _nothing(utterance_id: str) -> PlayedUtterance:
    return PlayedUtterance(
        utterance_id=utterance_id, played_text="", played_words=0, complete=False
    )


def split_played(utterances: Sequence[tuple[str, str]], played_text: str) -> list[PlayedUtterance]:
    """One entry per ``(utterance_id, source)`` pair, in publication order.

    ``utterances`` are the run's published utterances with their *committed sources* (the strings
    ``played_text`` was built from, not the TTS-normalised rendering); ``played_text`` is the run's
    terminal-event text. Never raises: an unreconcilable string yields empty entries.
    """
    entries: list[PlayedUtterance] = []
    rest = played_text
    cut = False  # True once an entry did not cover its whole source: nothing later was heard
    for utterance_id, source in utterances:
        if cut:
            entries.append(_nothing(utterance_id))
            continue
        if rest.startswith(source):
            # The whole source is there: this utterance was heard out (subject to the guard band,
            # T-0030 ruling A3 — text is all we have, and it says the entry is covered).
            rest = rest[len(source) :]
            entries.append(_entry(utterance_id, source, complete=True))
            continue
        # The cut falls inside this utterance: what is left must be a word prefix of its source,
        # spelled exactly as the source spells it. Anything else is not reconcilable.
        candidate = _word_prefix(source, len(rest.split()))
        cut = True
        if rest and candidate == rest:
            entries.append(_entry(utterance_id, rest, complete=False))
            rest = ""
        else:
            entries.append(_nothing(utterance_id))
    return entries
