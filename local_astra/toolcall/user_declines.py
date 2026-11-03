"""T-0107e: the user-declines backstop's predicate (pure, stdlib only, total, linear time).

``user_declines(transcript)`` is true iff the user's turn is a clear no (docs/tasks/T-0107e.md, Definition):

- **Normalise:** NFKC, then ``casefold()``; U+2018, U+2019 and U+02BC read as ``'``. A ``no`` followed by ``.``,
  optional whitespace and a digit means "number" and is never a head (``No. 10 Downing Street``).
- **Clauses and words:** clauses split at ``, . ! ? ; : … — – ( ) " “ ” « »`` and line breaks; clauses with no word
  are dropped. A word is a maximal run of letters or digits, joined by an inner ``'`` or ``-``.
- **Leading hesitation** (``uh um er erm euh hmm actually wait attends``) is skipped at the very start, across clause
  boundaries.
- **Heads:** interjections ``no nope non`` / ``no thanks``; verbs ``stop cancel annule``; negators ``don't dont`` /
  ``do not``.
- **True iff** (a) the first clause is one or more heads followed by only T words; or (verb head) one P word, then
  only T words; or (negator head) any word, one P word, then only T words; or (negator head, the first clause being
  the whole transcript) any word, one D word, any word, then only T words; and (b) no word after the first head, in
  any clause, is in R.

One pass builds the clauses; each check after it walks the words a bounded number of times, so the whole is linear.
"""

from __future__ import annotations

import unicodedata

#: the reason a call drafted in a declined turn is refused with (a constant: never text built from the call)
USER_DECLINED_REASON = "the user said no"

_APOSTROPHES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'"})
_CLAUSE_BREAKS = frozenset(',.!?;:…—–()"“”«»\n\r\v\f\x1c\x1d\x1e\x85  ')
_JOINERS = frozenset("'-")
#: a ``no`` read as "number" (``No. 10``); never a head, a hesitation or an R word
_NUMBER = "no."

_HESITATION = frozenset({"uh", "um", "er", "erm", "euh", "hmm", "actually", "wait", "attends"})
_INTERJECTIONS = frozenset({"no", "nope", "non"})
_VERBS = frozenset({"stop", "cancel", "annule"})
_NEGATORS = frozenset({"don't", "dont"})
_P = frozenset({"it", "that", "this", "them"})
_D = frozenset({"the", "my", "your", "that", "this", "these", "those", "any"})
_T = frozenset({"please", "yet", "now", "thanks", "merci"})
_R = frozenset(
    {"wait", "actually", "instead", "rather", "but", "mais", "plutôt", "plutot", "attends"}
)


def _is_word_char(ch: str) -> bool:
    return ch.isalpha() or ch.isdigit()


def _clauses(text: str) -> list[list[str]]:
    """The normalised text's clauses (each a list of words), clauses without a word dropped."""
    clauses: list[list[str]] = []
    current: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if _is_word_char(ch):
            start = i
            i += 1
            while i < n:
                if _is_word_char(text[i]):
                    i += 1
                elif text[i] in _JOINERS and i + 1 < n and _is_word_char(text[i + 1]):
                    i += 2
                else:
                    break
            word = text[start:i]
            if word == "no" and i < n and text[i] == ".":
                j = i + 1
                while j < n and text[j].isspace():
                    j += 1
                if j < n and text[j].isdigit():
                    word = _NUMBER
            current.append(word)
            continue
        if ch in _CLAUSE_BREAKS and current:
            clauses.append(current)
            current = []
        i += 1
    if current:
        clauses.append(current)
    return clauses


def _head_at(words: list[str], i: int) -> tuple[str, int] | None:
    """The head starting at ``words[i]`` as ``(kind, length)``, or ``None``."""
    w = words[i]
    nxt = words[i + 1] if i + 1 < len(words) else None
    if w == "no" and nxt == "thanks":
        return "interjection", 2
    if w == "do" and nxt == "not":
        return "negator", 2
    if w in _INTERJECTIONS:
        return "interjection", 1
    if w in _VERBS:
        return "verb", 1
    if w in _NEGATORS:
        return "negator", 1
    return None


def _only_t(words: list[str], i: int) -> bool:
    """``words[i:]`` holds only T words (``thank you`` is one)."""
    n = len(words)
    while i < n:
        if words[i] in _T:
            i += 1
        elif words[i] == "thank" and i + 1 < n and words[i + 1] == "you":
            i += 2
        else:
            return False
    return True


def _rest_fits(words: list[str], i: int, kind: str, whole: bool) -> bool:
    """Clause (a) after a run of heads ending at ``i`` whose last head is of ``kind``."""
    if _only_t(words, i):
        return True
    n = len(words)
    if kind == "verb":
        return i < n and words[i] in _P and _only_t(words, i + 1)
    if kind == "negator":
        if i + 1 < n and words[i + 1] in _P and _only_t(words, i + 2):
            return True
        return whole and i + 2 < n and words[i + 1] in _D and _only_t(words, i + 3)
    return False


def user_declines(transcript: str) -> bool:
    """True iff ``transcript`` is a clear no (see the module docstring); never raises."""
    text = unicodedata.normalize("NFKC", transcript).casefold().translate(_APOSTROPHES)
    clauses = _clauses(text)
    # leading hesitation, across clause boundaries
    c = 0
    while c < len(clauses):
        words = clauses[c]
        k = 0
        while k < len(words) and words[k] in _HESITATION:
            k += 1
        if k < len(words):
            break
        c += 1
    if c == len(clauses):
        return False
    first = clauses[c][k:]
    whole = c == len(clauses) - 1
    # (b): no R word after the first head, in any clause
    if any(w in _R for w in first[1:]) or any(w in _R for cl in clauses[c + 1 :] for w in cl):
        return False
    # (a): one or more heads, then one of the tails; every split of the head run is tried
    i = 0
    while i < len(first):
        head = _head_at(first, i)
        if head is None:
            return False
        kind, length = head
        i += length
        if _rest_fits(first, i, kind, whole):
            return True
    return False
