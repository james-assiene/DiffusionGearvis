"""T-0142: a pure detector for a thought that collapses into repetition.

The committed thought text of a request sometimes degenerates into a 1-3 token unit repeated back to back (``the``,
``the.``, ``the to``, ``, the``) and from there into runs of dots (T-0134 at e59aa69, T-0132 at 2ccff41: 10 of 1,986
requests with thought commits). ``degenerate`` names such a collapse; the session (``degeneracy_guard``) judges a
request's whole committed thought text so far with it, and on a trip aborts the request.

The rule: some unit of 1 to ``MAX_UNIT_TOKENS`` tokens occurs at least ``MIN_REPEATS`` times back to back, or one
non-whitespace character occurs at least ``MIN_CHAR_RUN`` times in a row. Tokens are
``re.findall(r"\\w+|[^\\w\\s]+", text.casefold())``. Over the 1,976 clean logged requests the longest back-to-back
repeat is 2 and the longest character run 8 (H-T142-2: ``MIN_REPEATS = 8``).

T-0142b: the session also judges a request's committed *answer* text, with ``min_repeats=MIN_REPEATS_ANSWER``. Over
the 1,665 clean logged answers the longest back-to-back repeat is 1; 6 is the largest value that stops the one observed
answer collapse (T-0135 arm B S2#5) before any of it is voiced (H-T142b-2). ``min_repeats`` lowers the n-gram rule
only; the character rule stays ``MIN_CHAR_RUN``.

Pure and stdlib-only: no I/O, no clock, deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

MAX_UNIT_TOKENS = 3
MIN_REPEATS = 8
MIN_CHAR_RUN = 64
#: T-0142b (H-T142b-2): the n-gram threshold for committed answer text
MIN_REPEATS_ANSWER = 6

_TOKEN = re.compile(r"\w+|[^\w\s]+")


@dataclass(frozen=True)
class Trip:
    """What tripped: the repeated ``unit`` (tokens joined by a space; for ``char_run`` the one character), its
    back-to-back count, and which rule."""

    unit: str
    repeats: int
    kind: Literal["ngram", "char_run"]


def _longest_ngram_run(tokens: list[str]) -> tuple[list[str], int]:
    """The 1-``MAX_UNIT_TOKENS`` token unit with the most back-to-back repeats, and that count (the shortest unit
    wins a tie, then the earliest run)."""
    best_unit: list[str] = []
    best = 0
    for n in range(1, MAX_UNIT_TOKENS + 1):
        # a maximal stretch where tokens[j] == tokens[j + n] for every j is periodic with period n
        start = 0
        while start + n <= len(tokens):
            end = start
            while end + n < len(tokens) and tokens[end] == tokens[end + n]:
                end += 1
            repeats = (end - start + n) // n
            if repeats > best:
                best, best_unit = repeats, tokens[start : start + n]
            start = end + 1
    return best_unit, best


def _longest_char_run(text: str) -> tuple[str, int]:
    """The non-whitespace character with the longest run in ``text``, and its length (the earliest on a tie)."""
    best_char, best = "", 0
    i = 0
    while i < len(text):
        j = i
        while j + 1 < len(text) and text[j + 1] == text[i]:
            j += 1
        if not text[i].isspace() and j - i + 1 > best:
            best_char, best = text[i], j - i + 1
        i = j + 1
    return best_char, best


def degenerate(text: str, *, min_repeats: int = MIN_REPEATS) -> Trip | None:
    """The collapse in ``text`` (a request's whole committed thought or answer text so far), or ``None``.
    ``min_repeats`` replaces ``MIN_REPEATS`` in the n-gram rule (T-0142b: ``MIN_REPEATS_ANSWER`` for answers)."""
    unit, repeats = _longest_ngram_run(_TOKEN.findall(text.casefold()))
    if repeats >= min_repeats:
        return Trip(unit=" ".join(unit), repeats=repeats, kind="ngram")
    char, run = _longest_char_run(text)
    if run >= MIN_CHAR_RUN:
        return Trip(unit=char, repeats=run, kind="char_run")
    return None
