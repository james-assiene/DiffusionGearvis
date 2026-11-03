"""T-0107e criterion 1: ``user_declines`` and ``USER_DECLINED_REASON`` in the new pure module
``local_astra.toolcall.user_declines``.

Every expected value is written by hand from the task's **Definition** (docs/tasks/T-0107e.md, "Acceptance criteria"),
never from an implementation. The criterion-1 lists are encoded verbatim (``TASK_TRUE``, ``TASK_FALSE``,
``TASK_MISSES``); no row of them is flipped. Rows the test-author added are derived clause by clause from the
Definition, and each says which clause decides it.

The Definition, in short: NFKC, ``casefold()``, map U+2018/U+2019/U+02BC to ``'``; ``no.`` + optional whitespace + a
digit is never a head; clauses split at ``, . ! ? ; : … — – ( ) " “ ” « »`` and line breaks, empty ones dropped; a word
is a maximal run of letters or digits joined by an inner ``'`` or ``-``; leading hesitation (``uh um er erm euh hmm
actually wait attends``) is skipped across clause boundaries; heads are ``no nope non`` / ``no thanks`` (interjections),
``stop cancel annule`` (verbs), ``don't dont`` / ``do not`` (negators); P = ``it that this them``, D = ``the my your
that this these those any``, T = ``please yet now thanks`` / ``thank you`` / ``merci``, R = ``wait actually instead
rather but mais plutôt plutot attends``. True iff (a) the first clause is one or more heads followed by only T words, or
(verb head) one P word then only T words, or (negator head) any word, one P word, then only T words, or (negator head,
the first clause being the whole transcript) any word, one D word, any word, then only T words; and (b) no word after
the first head, in any clause, is in R.

The module does not exist at the base (9275442) and the test-author may not create it (owned by the implementer), so it
is imported by ``_mod()`` inside each row: its absence is a named ``AssertionError`` (strict xfail while pending), never a
collection error.

Readings fixed here (listed in the test-author's report):

- "the first clause is the whole transcript" is read after empty clauses are dropped, so ``Don't send the message.``
  (the task's true row) has one clause; ``Don't send the message, I'll do it`` therefore has two, and is false by the
  Definition (the D form needs the whole transcript) -- a consequence pinned in ``test_determiner_form_needs_whole_
  transcript``, not a guess;
- a leading hesitation clause (``Uh, ``) does not count towards "the whole transcript" for the D form, and a head may
  be the D form's "any word" (``Don't stop the timer``): both readings accepted in the task's Rulings and pinned true;
- the linear-time row is a guard against quadratic behaviour (200,000-character inputs, < 1.0 s each in a fresh child
  process killed at 30 s), not a benchmark.

Regime: pure function, CPU, no I/O; Hypothesis with the project's default profile; the timing row in a fresh
``sys.executable`` child, n = 1 per shape.
"""

from __future__ import annotations

import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

from hypothesis import example, given
from hypothesis import strategies as st

REPO = Path(__file__).resolve().parents[2]
MODULE = "local_astra.toolcall.user_declines"
SOURCE = REPO / "local_astra" / "toolcall" / "user_declines.py"
#: the task's Definition, verbatim (never imported from the module under test)
REASON = "the user said no"


def _mod() -> ModuleType:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"T-0107e owned files: the pure module {MODULE} (user_declines, USER_DECLINED_REASON) does not "
            f"exist: {exc}"
        ) from None


def declines(text: str) -> bool:
    mod = _mod()
    assert hasattr(mod, "user_declines"), f"T-0107e: {MODULE} has no user_declines"
    return mod.user_declines(text)  # type: ignore[no-any-return]


def _check(cases: list[tuple[str, bool]], what: str) -> None:
    got = [(text, want, declines(text)) for text, want in cases]
    wrong = [g for g in got if g[1] is not g[2]]
    assert not wrong, f"T-0107e criterion 1 ({what}): (text, expected, got) {wrong}"


# =====================================================================================================
# Criterion 1's lists, verbatim (never flipped)
# =====================================================================================================

TASK_TRUE = (
    "No, leave it.",
    "No thanks, I'll do it myself",
    "Stop!",
    "Don't send it",
    "Don’t send it",
    "Dont send it",
    "No",
    "nope",
    "NO!",
    "no.",
    "no thank you",
    "Non, laisse.",
    "Non merci",
    "Annule",
    "Cancel.",
    "stop it",
    "cancel that please",
    "Don't send it yet",
    "Don't send the message.",
    "Do not delete them",
    "no no no",
    "no don't",
    "Actually no, don't send it",
    "Wait, no, don't.",
    "Uh, no.",
    "Stop, stop.",
    "No, don't send the message, I'll do it",
    "No, the other one",  # an accepted over-refusal (H-T107e-4)
)

TASK_FALSE = (
    "no wait, send it to Bob instead",
    "No, but send it tomorrow",
    "No. Open the settings instead.",
    "nothing",
    "know",
    "I know, do it",
    "Nobody is there",
    "Notify Bob",
    "No problem, go ahead",
    "No worries, send it",
    "No need to ask, just send it",
    "Cancel my 3pm meeting",
    "Stop the timer",
    "Annule la commande",
    "Don't forget to save",
    "Don't close the tab, click Save",
    "Don't send it to Alice, send it to Bob",
    "Don't worry, send it",
    "Don't worry about it",
    "Do not disturb mode on",
    "Non-stop flights to Paris",
    "No-code tools",
    "No. 10 Downing Street opening hours",
    "Oh no, the page crashed, reload it",
    "Which mark do you have ever seen? No",
    "Yes, go ahead.",
    "",
    '[ui] __gate.clicked {"label": "No"}',
)

#: Known misses, pinned false (widening them is H-T107e-1)
TASK_MISSES = (
    "Okay, no, leave it",
    "I don't want that, leave it",
    "no leave it",
    "Leave it.",
    "Never mind",
    "Ne l'envoie pas",
)


def test_true_forms() -> None:
    """Criterion 1's true list verbatim, then rows derived from the Definition: every head alone and with T words,
    the verb + P and negator + word + P forms with T tails, the negator + word + D + word form on a one-clause
    transcript, repeated heads, and empty clauses (``!!``) dropped."""
    _check([(t, True) for t in TASK_TRUE], "the task's true list")
    _check(
        [
            # interjection heads alone, and followed only by T words
            ("Nope.", True),
            ("Non.", True),
            ("no thanks", True),
            ("No thanks.", True),
            ("no, thanks", True),  # first clause "no"; "thanks" after the head is not in R
            ("nope, thanks", True),
            ("no please", True),
            ("No, not yet", True),  # first clause "no"; "not", "yet" after the head are not in R
            ("No not yet", False),  # one clause: "not" is no T word, so (a) fails
            ("Do not", True),  # a negator head alone: one head, then zero T words
            ("Don't", True),
            ("no thanks now", True),
            ("Non merci.", True),
            ("no thank you, I'll do it", True),
            # verb heads alone, + P, + P + T
            ("Annule.", True),
            ("stop", True),
            ("Stop it now.", True),
            ("Cancel that, please.", True),
            ("stop this please", True),
            ("cancel them", True),
            ("Cancel it now please", True),
            ("Stop it, I'll do it", True),  # first clause "stop it"; (b): no R word later
            # negator heads + any word + P (+ T)
            ("Do not send it", True),
            ("do not touch that", True),
            ("don't do it now", True),
            ("Dont delete them please", True),
            ("Don't send this yet, thanks", True),
            # negator + any word + D + any word (+ T), the first clause being the whole transcript
            ("Don't delete those files", True),
            ("Don't open any links", True),
            ("don't send my email yet", True),
            ("Don't send your reply.", True),
            ("Do not move these windows", True),
            ("Don't send the message", True),
            ("Don't send the message!", True),  # the empty clause after "!" is dropped
            # one or more heads
            ("Stop! Stop!", True),
            ("No!!", True),
            ("no no", True),
            ("No, no, no.", True),
            ("no no thanks", True),
            ("nope stop", True),
            ("no, stop", True),
            ("Non, annule", True),
        ],
        "derived true forms",
    )


def test_false_forms() -> None:
    """Criterion 1's false list verbatim -- adversarial on the substring and first-word failure modes (``nothing``,
    ``know``, ``Nobody is there``: a head inside a longer word; ``Cancel my 3pm meeting``, ``Stop the timer``,
    ``Don't forget to save``: a head that begins a command) -- then rows derived from the Definition."""
    _check([(t, False) for t in TASK_FALSE], "the task's false list")
    _check(
        [
            # a head as a prefix or substring of a longer word is no head (mutation 1, ``"no" in text``)
            ("Nothing to add", False),
            ("Note that down", False),
            ("Nonsense, send it", False),
            ("Stopwatch on", False),
            ("Cancellation policy", False),
            ("Donte is here", False),
            ("Snow in Paris", False),
            ("Canon printer settings", False),
            ("annuler la commande", False),
            ("dont-care flag", False),
            # a verb head that begins a command: the next word is not P (mutation 2, first word only; mutation 3)
            ("Cancel my order", False),
            ("Stop the music please", False),
            ("Cancel the meeting", False),
            ("Stop it from crashing", False),  # verb + P, then "from" is no T word
            ("Cancel that order", False),  # verb + P ("that"), then "order" is no T word
            # an interjection followed by a non-T word in the first clause
            ("No problem", False),
            ("no idea", False),
            ("No way, that's great", False),
            # an interjection takes no P word (the P form is the verb heads'; reviewers' surviving mutant)
            ("no that", False),
            ("nope it", False),
            ("Nope, not that one instead", False),  # R ("instead")
            # a negator followed by a non-P word where P is required
            (
                "Don't forget the milk, thanks",
                False,
            ),  # two clauses, so the D form does not apply; "the" is not P
            ("Don't worry", False),
            (
                "Don't send the message to Bob",
                False,
            ),  # D form: "to bob" after "message" are no T words
            # empty or wordless
            ("   ", False),
            ("...", False),
            ("?!", False),
        ],
        "derived false forms",
    )


def test_known_misses_pinned() -> None:
    """Criterion 1's known misses, pinned false (H-T107e-1: widening the lexicon is the human's call) -- and further
    forms the H-T107e-1 examples name, each false by the Definition (no head first, or a non-P/non-T word after it).
    A too-wide implementation that guesses at the lexicon goes red here."""
    _check([(t, False) for t in TASK_MISSES], "the task's known misses")
    _check(
        [
            ("Forget it", False),
            ("Hold on", False),
            ("Not now", False),
            ("Okay no", False),
            ("Well, no.", False),  # "well" is no hesitation word
            ("Annule ça", False),  # "ça" is not in P
            ("Leave it alone", False),
            ("nah", False),
            ("Never mind, leave it", False),
            ("N'envoie pas", False),
        ],
        "further known misses (H-T107e-1)",
    )


def test_reversal_after_head_cancels() -> None:
    """Clause (b), adversarial for mutation 4 (drop the reversal check): each true form, with an R word added after the
    first head (in the first clause or a later one), is false; the same text without the R word is true. An R word that
    is only a substring of a longer word (``butter``, ``waiter``) is no R word."""
    pairs = [
        ("No, the other one", "No, the other one instead"),
        ("Stop it, I'll do it", "Stop it, but I'll do it"),
        ("Non, laisse.", "Non, mais laisse."),
        ("Cancel that, thanks", "Cancel that, rather the draft"),
        ("No, leave it.", "No, actually leave it."),
        ("No, leave it.", "No, wait, leave it."),
        ("Non, l'autre", "Non, plutôt l'autre"),
        ("Non, l'autre", "Non, plutot l'autre"),
        ("Non, laisse", "Non, attends, laisse"),
        ("Don't send it", "Don't send it, send it instead"),
        ("No. Do it later.", "No. Do it later instead."),
        ("Stop, stop.", "Stop, but stop."),
    ]
    rows: list[tuple[str, bool]] = []
    for plain, reversed_ in pairs:
        rows += [(plain, True), (reversed_, False)]
    rows += [
        ("NO, WAIT", False),  # casefolded R word
        ("No, Actually", False),
        ("Non, PLUTÔT", False),
        ("No, butter", True),  # "butter" is not "but"
        ("No, waiter", True),  # "waiter" is not "wait"
        (
            "No, the instead-of form",
            True,
        ),  # "instead-of" is one word (an inner hyphen), not "instead"
    ]
    _check(rows, "a reversal word after the head")


def test_hesitation_before_head_is_skipped() -> None:
    """Leading hesitation is skipped at the very start, across clause boundaries -- each of the nine hesitation words
    before a head -- and only there: an R word after the first head still reverses (``No, uh, wait``), and a non-
    hesitation opener (``Oh``, ``Well``, ``Okay``, ``Please``) is not skipped."""
    _check(
        [
            ("Uh, no.", True),
            ("um no", True),
            ("Er, stop.", True),
            ("Erm, cancel it", True),
            ("euh, non", True),
            ("Euh non merci", True),
            ("Hmm... no thanks", True),
            ("Hmm no", True),
            ("Actually, no.", True),
            ("Wait, stop!", True),
            ("Attends, annule", True),
            ("Actually, wait, no.", True),  # two hesitations, two clause boundaries
            ("uh uh no", True),
            ("Wait wait stop", True),
            ("Um, uh, nope", True),
            ("Uh, don't send it", True),
            ("Uh, no, um, leave it", True),  # "um" after the head is no R word
            # only at the very start
            ("No, uh, wait", False),  # "wait" after the head is R
            ("Stop, actually", False),
            ("Uh, no, actually send it", False),
            # not a hesitation word, or a hesitation before a non-head
            ("Oh, no", False),
            ("Well no", False),
            ("Okay, stop", False),
            ("Please stop", False),
            ("Uh, yes", False),
            ("Um, send it", False),
            ("Uh", False),
            ("Wait", False),
            ("Actually, send it now", False),
        ],
        "leading hesitation",
    )


def test_trailing_no_does_not_count() -> None:
    """Clause (a) reads the first clause: a head that comes later in the transcript (after a non-head first clause, or
    after other words of the first clause) never makes it a no -- adversarial for mutation 1 (substring) and for a
    "some clause is a no" reading."""
    _check(
        [
            ("Which mark do you have ever seen? No", False),
            ("Send it, no?", False),
            ("I said no", False),
            ("Yes, no", False),
            ("Is it done? No.", False),
            ("Click Cancel", False),
            ("Just stop", False),
            ("Send it now, don't wait", False),
            ("Oh no", False),
            ("Press the Stop button", False),
            ("Send it. No.", False),
            ("Open the menu\nno", False),
            ("tell me if the answer is no", False),
        ],
        "a trailing head",
    )


def test_number_abbreviation() -> None:
    """Normalise, third item: ``no.`` followed by optional whitespace and a digit means "number" and is never a head
    (NFKC first, so a fullwidth digit is a digit); followed by anything else it is the head ``no``."""
    _check(
        [
            ("No. 10 Downing Street opening hours", False),
            ("No.10", False),
            ("no. 5 please", False),
            ("No.  42", False),
            ("No.\t7 please", False),
            ("NO. 3", False),
            ("No. ５", False),  # NFKC: U+FF15 is "5"
            ("No.１０ Downing Street", False),
            # not followed by a digit: the head "no", the first clause "no"
            ("No.", True),
            ("No. Five.", True),
            ("No. Stop.", True),
            ("No..", True),
            ("No. thanks", True),
        ],
        "the number abbreviation",
    )


def test_apostrophe_and_case_variants() -> None:
    """Normalise, first two items, adversarial for mutation 9 (drop casefold or the apostrophe map): U+2019, U+2018 and
    U+02BC are read as ``'``; NFKC folds fullwidth forms (letters and U+FF07); every case pattern is one word. Other
    marks are not mapped: a backtick or an acute accent (NFKC: space + U+0301) splits ``don`` from ``t``, and ``don``
    is no head."""
    _check(
        [
            ("Don't send it", True),
            ("Don’t send it", True),
            ("Don‘t send it", True),
            ("Donʼt send it", True),
            ("Don＇t send it", True),
            ("DON'T SEND IT", True),
            ("DON’T SEND IT", True),
            ("dOn'T sEnD iT", True),
            ("ＤＯＮ'Ｔ ＳＥＮＤ ＩＴ", True),
            ("DO NOT DELETE THEM", True),
            ("NO THANKS", True),
            ("Ｎｏ", True),
            ("ＳＴＯＰ!", True),
            ("CANCEL THAT PLEASE", True),
            ("NON MERCI", True),
            ("NOPE", True),
            ("Don’t send the message.", True),
            # not mapped: not an apostrophe of the Definition
            ("Don`t send it", False),
            ("Don´t send it", False),
        ],
        "apostrophes and case",
    )


def test_determiner_form_needs_whole_transcript() -> None:
    """Clause (a), last form: negator + any word + D + any word holds only when the first clause is the whole
    transcript (after empty clauses are dropped). The same first clause followed by another clause is false -- unless
    an interjection head opens the transcript (the task's ``No, don't send the message, I'll do it``)."""
    _check(
        [
            ("Don't send the message.", True),
            ("Don't send the message", True),
            ("Don't send the message!!", True),
            ("Don't send the message, I'll do it", False),
            ("Don't close the tab, click Save", False),
            ("Don't delete those files. Thanks.", False),
            ("No, don't send the message, I'll do it", True),
            # accepted readings (T-0107e Rulings): a leading hesitation clause does not count against "the whole
            # transcript"; a head may be the "any word" of the negator form
            ("Uh, don't send the message", True),
            ("Don't stop the timer", True),
            # the P forms do not need the whole transcript
            ("Don't send it, I'll do it", True),
            ("Stop it, I'll do it", True),
        ],
        "the determiner form",
    )


# =====================================================================================================
# Total and deterministic (Hypothesis)
# =====================================================================================================

INTERJECTIONS = ("no", "nope", "non", "no thanks")
VERBS = ("stop", "cancel", "annule")
NEGATORS = ("don't", "dont", "do not", "Don’t")
P_WORDS = ("it", "that", "this", "them")
T_WORDS = ("please", "yet", "now", "thanks", "thank you", "merci")
R_WORDS = ("wait", "actually", "instead", "rather", "but", "mais", "plutôt", "plutot", "attends")

#: single characters every clause of the Definition distinguishes, plus whole words and the number form
_pieces = st.sampled_from(
    [
        *"noNOstpcadeil' -.,!?;:…—–()\"“”«»\n\t5０ ’‘ʼ",
        *("no", "stop", "don't", "uh", "No. 1", "but", "it", "the", "please"),
    ]
)
_texts = st.one_of(st.text(), st.lists(_pieces, max_size=30).map("".join))
_tails = st.lists(st.sampled_from(T_WORDS), max_size=3)


def _case(word: str, how: int) -> str:
    return (word, word.upper(), word.capitalize())[how]


@given(
    text=_texts,
    interj=st.sampled_from(INTERJECTIONS),
    verb=st.sampled_from(VERBS),
    neg=st.sampled_from(NEGATORS),
    pron=st.sampled_from(P_WORDS),
    rev=st.sampled_from(R_WORDS),
    tails=_tails,
    how=st.integers(0, 2),
)
@example(text="", interj="no", verb="stop", neg="don't", pron="it", rev="wait", tails=[], how=0)
def test_total_and_deterministic(
    text: str,
    interj: str,
    verb: str,
    neg: str,
    pron: str,
    rev: str,
    tails: list[str],
    how: int,
) -> None:
    """Hypothesis: on any string ``user_declines`` returns a ``bool`` without raising and the same one twice; and
    consequences of the Definition (a grammar enumeration, not a re-implementation): a transcript opened by ``Yes, ``
    is never a no (its first clause holds no head); ``No, `` + anything + `` <R word>`` is never a no (clause (b));
    and each form of clause (a) -- interjection + T words, verb + P + T words, negator + ``send`` + P + T words -- in
    any case pattern, is a no."""
    first = declines(text)
    assert type(first) is bool, f"user_declines({text!r}) returned {type(first).__name__}, not bool"
    assert declines(text) is first, f"user_declines({text!r}) is not deterministic"
    assert declines("Yes, " + text) is False, f"user_declines({'Yes, ' + text!r}) is true"
    reversed_ = f"No, {text} {_case(rev, how)}"
    assert declines(reversed_) is False, f"(b): user_declines({reversed_!r}) is true"
    tail = "".join(" " + t for t in tails)
    for form in (
        f"{_case(interj, how)}{tail}",
        f"{_case(verb, how)} {_case(pron, how)}{tail}",
        f"{_case(neg, how)} send {_case(pron, how)}{tail}",
    ):
        assert declines(form) is True, f"(a): user_declines({form!r}) is false"
        assert declines(form + ".") is True, f"(a): user_declines({form + '.'!r}) is false"


# =====================================================================================================
# Linear time
# =====================================================================================================

_TIMED = """
import json, sys, time
from local_astra.toolcall.user_declines import user_declines
shapes = json.loads(sys.argv[1])
out = []
for unit, n, tail in shapes:
    text = unit * n + tail
    t0 = time.perf_counter()
    got = user_declines(text)
    out.append({"len": len(text), "got": got, "s": time.perf_counter() - t0})
print(json.dumps(out))
"""

#: (unit, repeat, tail, expected by hand, why) -- each about 200,000 characters
SHAPES: tuple[tuple[str, int, str, bool, str], ...] = (
    ("no ", 66_667, "", True, "one first clause of 66,667 heads"),
    ("uh, ", 50_000, "no", True, "50,000 hesitation clauses skipped, then the head"),
    (",", 200_000, "stop", True, "200,000 empty clauses dropped, then the head"),
    ("no ", 66_666, "instead", False, "heads then an R word (and a non-T word)"),
    ("a'", 100_000, "", False, "one long apostrophe-joined word, no head"),
    ("no-", 66_667, "", False, "one long hyphen-joined word, no head"),
    ("it ", 66_666, "", False, "no head at all"),
    ("uh ", 66_666, "x", False, "66,666 hesitation words, then a non-head"),
    ("no. ", 50_000, "", True, "each 'no.' is followed by a space then 'n', not a digit: heads"),
    ("no. 1", 40_000, "", False, "the first 'no.' is followed by ' 1': a number, never a head"),
)


def test_linear_time() -> None:
    """Adversarial (the Definition: "linear"): ten 200,000-character shapes that a scan re-walking a run (of heads,
    hesitations, empty clauses, joined words, ``no.`` lookaheads) from each position makes quadratic. Each call runs in
    one fresh CPU child process (killed at 30 s) and must take < 1.0 s -- a generous guard, not a benchmark -- and
    return the answer computed by hand."""
    _mod()  # the module's absence is a named assertion, not a child traceback
    arg = json.dumps([[u, n, t] for u, n, t, _w, _y in SHAPES])
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _TIMED, arg],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=REPO,
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(
            "T-0107e (linear time): the ten 200,000-character shapes did not finish within 30 s (quadratic scan)"
        ) from None
    assert proc.returncode == 0, f"the timing child failed:\n{proc.stderr[-2000:]}"
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    for (unit, n, tail, want, why), got in zip(SHAPES, out, strict=True):
        name = f"{unit!r} * {n} + {tail!r}"
        assert got["len"] >= 199_998, f"premise: {name} is ~200,000 characters; got {got['len']}"
        assert got["got"] is want, (
            f"T-0107e criterion 1 ({why}): user_declines({name}) is {got['got']}"
        )
        assert got["s"] < 1.0, (
            f"T-0107e (linear time): user_declines({name}) took {got['s']:.3f} s (bound 1.0 s, CPU, n = 1)"
        )


# =====================================================================================================
# The module
# =====================================================================================================

#: Stdlib modules a pure, total predicate has no business importing (I/O, clocks, randomness, processes).
IMPURE = frozenset(
    {
        "asyncio",
        "datetime",
        "io",
        "logging",
        "os",
        "pathlib",
        "random",
        "shutil",
        "socket",
        "subprocess",
        "tempfile",
        "threading",
        "time",
        "urllib",
        "http",
    }
)


def test_module_is_pure() -> None:
    """The module exists, defines ``USER_DECLINED_REASON = "the user said no"`` (verbatim from the task) and
    ``user_declines``, and imports only the standard library, none of it I/O, clocks or randomness, and nothing from
    ``local_astra`` (``ast`` over the source; the source is read, never executed for this check)."""
    mod = _mod()
    assert getattr(mod, "USER_DECLINED_REASON", None) == REASON, (
        f"T-0107e: USER_DECLINED_REASON is {getattr(mod, 'USER_DECLINED_REASON', None)!r}"
    )
    assert callable(getattr(mod, "user_declines", None)), "T-0107e: user_declines is not callable"
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, (
                f"T-0107e: a relative import in a stdlib-only module: {ast.dump(node)}"
            )
            assert node.module is not None
            roots.add(node.module.split(".")[0])
    roots.discard("__future__")
    non_std = sorted(r for r in roots if r not in sys.stdlib_module_names)
    assert not non_std, f"T-0107e: {MODULE} imports only the stdlib; it imports {non_std}"
    impure = sorted(roots & IMPURE)
    assert not impure, f"T-0107e: {MODULE} is pure; it imports {impure}"
    calls = sorted(
        {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        & {"open", "print", "input", "exec", "eval", "__import__"}
    )
    assert not calls, f"T-0107e: {MODULE} is pure; it calls {calls}"
