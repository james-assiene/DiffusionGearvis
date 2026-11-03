"""T-0107a criterion 1: ``asks_user`` and ``ASKED_FIRST_REASON`` in the new pure module ``local_astra.toolcall.ask_first``.

The expected values are written by hand from the task's **Definition as replaced** by the orchestrator's ruling
"after reviewers A and B" of 2026-10-19 (docs/tasks/T-0107a.md, Rulings, the last line), never from an
implementation:

    ``asks_user(text)`` is true iff some ``?`` (U+003F or U+FF1F) satisfies both (a) the character right after it
    is not ``str.isalnum()``; and (b) after skipping every following character that is not alphanumeric, either the
    end of the text is reached, or the first alphanumeric character is not ``str.islower()`` (uppercase or a digit).
    The function scans in linear time.

The earlier (widened) definition is retired; its quote/bracket/guillemet/whitespace clauses are gone, so the rows
below no longer test them as such. Rows that change direction (``? × 3``, ``?.4``, ``Proceed?\\n1. Yes`` and
``? 4 euros`` become true; the safe direction) are moved; the listed false forms stay false (each is followed by a
lowercase word). The reviewers' five missed forms (NBSP before ``»``, ``**Shall I pay?**``, ``"Shall I pay?".``,
``Shall I pay? — Then``, ``?`` then an emoji) are pinned true.

**T-0107a2, named re-scope R1 (docs/tasks/T-0107a2.md, criterion 1), narrows clause (a) to (a′):** a ``?`` is a
non-question only if the character right after it is a lowercase letter or a digit (``ch.islower() or ch.isdigit()``):

    ``asks_user(text)`` is true iff some ``?`` (U+003F or U+FF1F) satisfies both (a′) the character right after it,
    if any, is neither lowercase nor a digit; and (b) after skipping every following non-alphanumeric character,
    either the end of the text is reached or the first alphanumeric is not lowercase.

So ``Shall I pay?Then`` and ``Shall I?Then`` move from the false rows to the true rows (``test_glued_uppercase_asks``;
the intended change), ``3?4`` and ``pass?word`` stay false, and ``_oracle`` reads (a′). Readings fixed for (a′):
"a digit" is ``str.isdigit()`` as the task writes it (so ``²`` and ``٣`` are digits, ``½`` is not and asks by (b));
a titlecase letter (``ǅ``) and a CJK ideograph glued to the mark ask (neither lowercase nor a digit). The rows
``test_glued_uppercase_asks`` and ``test_agrees_with_brute_force_oracle`` are red at the base (87cf028) and land
unmarked: ``tests/conftest.py`` accepts only ``T-NNNN[a-z]`` as a pending task id, so ``T-0107a2`` cannot be
expressed as a marker, and criterion 5 forbids touching conftest in this task.

Besides the hand rows, ``_oracle`` is an independent brute-force reading of the definition (per ``?``, slice the
rest of the text, look at its first character and at its first alphanumeric character; quadratic, for tiny
Hypothesis inputs only) -- the definition transcribed, not the linear scan the implementation must use.

The module does not exist at the base (d3765de) and the test-author may not create it (owned by the implementer), so
it is imported by ``_mod()`` inside each row: its absence is a named assertion, never a collection error.

Readings fixed here (listed in the test-author's report):

- "letter or digit" is ``str.isalnum()`` exactly as the ruling says, so ``_`` and combining marks are not
  alphanumeric and are skipped by (b);
- "not lowercase" is ``not ch.islower()``: a titlecase letter (``ǅ``), a CJK ideograph and a digit of any script
  count as not lowercase (true); ``ß`` and ``ω`` are lowercase (false);
- the linear-time row is a guard against quadratic behaviour (200,000 ``?`` then ``x``; < 0.5 s in a child process
  killed at 20 s), not a benchmark.

Regime: pure function, CPU, no I/O; Hypothesis with its default profile (n per the project settings); the timing row
in a fresh ``sys.executable`` child, n = 1.
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
MODULE = "local_astra.toolcall.ask_first"
SOURCE = REPO / "local_astra" / "toolcall" / "ask_first.py"

CLOSING = "\"'”’)»"
OPENING = "\"'“‘(«"
QMARKS = ("?", "？")


def _mod() -> ModuleType:
    try:
        return importlib.import_module(MODULE)
    except ModuleNotFoundError as exc:
        raise AssertionError(
            f"T-0107a owned files: the pure module {MODULE} (asks_user, ASKED_FIRST_REASON) does not exist: {exc}"
        ) from None


def asks(text: str) -> bool:
    mod = _mod()
    assert hasattr(mod, "asks_user"), f"T-0107a: {MODULE} has no asks_user"
    return mod.asks_user(text)  # type: ignore[no-any-return]


def _check(cases: list[tuple[str, bool]], what: str) -> None:
    wrong = [(text, want, asks(text)) for text, want in cases]
    wrong = [w for w in wrong if w[1] is not w[2]]
    assert not wrong, f"T-0107a criterion 1 ({what}): (text, expected, got) {wrong}"


def _oracle(text: str) -> bool:
    """The Definition with T-0107a2's clause (a′), transcribed clause by clause (brute force, quadratic; tiny inputs
    only)."""
    for i, ch in enumerate(text):
        if ch not in QMARKS:
            continue
        rest = text[i + 1 :]
        if rest and (
            rest[0].islower() or rest[0].isdigit()
        ):  # (a′) glued to lowercase or a digit: no ask
            continue
        first_alnum = [c for c in rest if c.isalnum()][:1]
        if not first_alnum or not first_alnum[0].islower():  # (b) the end, or uppercase / a digit
            return True
    return False


# =====================================================================================================
# The named forms
# =====================================================================================================


def test_true_forms() -> None:
    """Criterion 1's four true forms, ``Should I go ahead? (It can't be undone.)`` (the ``(`` is skipped, ``I`` is
    not lowercase), and the forms the earlier definitions pinned true, re-derived by hand under the replaced rule
    (each: the character after the mark is not alphanumeric, and the first alphanumeric after it is absent or not
    lowercase)."""
    _check(
        [
            # the task's four, verbatim
            ("Shall I place the order for 42.99 euros?", True),
            ("Send it to Omar? He is in your chats.", True),
            ("Je passe la commande ?", True),
            ('Delete the standup event?"', True),
            # the bracket form (orchestrator, 0939e9b ruling): asks
            ("Should I go ahead? (It can't be undone.)", True),
            # (b), the end reached: nothing alphanumeric follows
            ("Shall I?", True),
            ("Shall I?   ", True),
            ("Shall I?\n", True),
            ("Shall I?\t\n ", True),
            # U+FF1F, end and mid-text
            ("Shall I place the order？", True),
            ("Shall I？ Then I pay.", True),
            # (b): mid-text, the first alphanumeric is uppercase (Latin-1 uppercase too)
            ("Delete it?\tNext I close the tab.", True),
            ("Supprimer l’événement ? Élise est prévenue.", True),
            ("Is that right? À vous.", True),
            ("Pay now? Þat is the total.", True),
            # "some ``?``": a later asking mark is enough; an earlier label does not cancel it
            ("I see a Forgot password? link. Shall I click it?", True),
            ("Is it 3?4 or shall I pay?", True),
            # a run of ``!?.…`` after the mark is skipped
            ("Pay?!", True),
            ("Pay?.", True),
            ("Pay now?! Then I close the tab.", True),
            ("Shall I pay?? Then I stop.", True),
            ("Shall I pay?… ", True),
            ("Shall I pay?!…", True),
            ('He asked "shall I pay?!" Then he waited.', True),
            ("Shall I pay？! ", True),
            # guillemets and spaces are skipped
            ("« Je passe la commande ? »", True),
            ("« Je passe la commande ? » Puis j’attends.", True),
            ("Je supprime l’événement ? « Réunion » disparaîtra.", True),
            # any whitespace (NBSP, em space) is skipped
            ("Shall I pay? ", True),
            ("Shall I pay? Then I stop.", True),
            ("Shall I pay? Then I stop.", True),
            # not lowercase beyond A-Z and À-Þ
            ("Shall I pay? Ā is the account.", True),
            ("Shall I pay? Ω is the total.", True),
            ("Отправить? Да.", True),
        ],
        "true forms",
    )


def test_reviewer_missed_forms_ask() -> None:
    """Adversarial, the unsafe direction (reviewers A and B, 2026-10-19; the reason the definition was replaced):
    each of the five forms the widened definition missed is a real question and asks -- an NBSP (and a narrow NBSP)
    before ``»`` (correct French typography; the call executed in B's probe), Markdown bold around the question, a
    full stop after the closing quote, a dash after the mark then an uppercase word, and an emoji after the mark."""
    _check(
        [
            ("« Je passe la commande ? »", True),
            ("« Je passe la commande ? »", True),
            ("« Je passe la commande ? »", True),
            ("« Je passe la commande ? » Puis j’attends.", True),
            ("**Shall I pay?**", True),
            ("**Shall I pay?** ", True),
            ("**Shall I pay?** Then I stop.", True),
            ('"Shall I pay?".', True),
            ('"Shall I pay?". Then I stop.', True),
            ("Shall I pay? — Then I stop.", True),
            ("Shall I pay?—Then I stop.", True),
            ("Shall I pay? 🙂", True),
            ("Shall I pay?🙂", True),
            ("Shall I pay? 🙂 Then I stop.", True),
            ("Shall I pay？🛒", True),
        ],
        "the reviewers' missed forms",
    )


def test_now_true_over_refusals() -> None:
    """The ruling's consequences, the safe direction: rows the widened definition held false and the replacement
    makes true (each ``?`` is followed by non-alphanumerics and then a digit or the end): ``? × 3``, ``?.4``, a
    question followed by a numbered list, and a digit after the mark and a space; plus the new rule's other true
    readings (a titlecase letter, a CJK ideograph, ``_`` skipped)."""
    _check(
        [
            ("Shall I? × 3 it is.", True),
            ("Shall I?.4", True),
            ("Proceed?\n1. Yes", True),
            ("Proceed?\n\n1. Yes\n2. No", True),
            ("Shall I? 4 euros it is.", True),
            ("Shall I pay? (42 euros)", True),
            ("Shall I pay? ǅ is the account.", True),
            ("Shall I pay? 好的", True),
            ("Shall I pay?_Then", True),
        ],
        "now true (over-refusals)",
    )


def test_false_forms() -> None:
    """Criterion 1's six false forms -- adversarial: the two click-02 "Forgot password?" label forms, the ten trials
    a plain "contains ?" rule refused in the four GPU runs (the Goal) -- the ruling's listed false forms, which stay
    false under the replaced rule (each is followed by a lowercase word), and the near misses of (a) and (b)."""
    _check(
        [
            # the task's six, verbatim
            ("I will click the Forgot password? link.", False),
            ("I will click the 'Forgot password?' link.", False),
            ("", False),
            ("Done.", False),
            ("what? no idea.", False),
            ("Is it 3?4", False),
            # the label forms with the other quotes and U+FF1F
            ('I will click the "Forgot password?" link.', False),
            ("I will click the “Forgot password?” link.", False),
            ("I will click the (Forgot password?) link.", False),
            ("I will click the Forgot password？ link.", False),
            # (b) finds a lowercase letter first (``Shall I?Then`` moved to test_glued_uppercase_asks, T-0107a2 R1)
            ("Shall I? then I pay.", False),
            ("Shall I? (it can't be undone.)", False),
            ("Shall I? «oui»", False),
            ("Shall I? ßeta", False),
            # the ruling's listed false forms (each followed by a lowercase word), verbatim then in a sentence
            ("Forgot password? link", False),
            ("'Forgot password?' link", False),
            ("what? no idea.", False),
            ("?! then", False),
            ("? » puis", False),
            ("? « réunion", False),
            ("? ω", False),
            ("Shall I?! then I pay.", False),
            ("« Je passe la commande ? » puis j’attends.", False),
            ("Je supprime l’événement ? « réunion » disparaîtra.", False),
            ("Shall I pay? then I stop.", False),
            ("Shall I pay? ω is the total.", False),
            # (a′): glued to a lowercase letter or a digit (the ruling's two; ``Shall I pay?Then`` moved to
            # test_glued_uppercase_asks, T-0107a2 R1)
            ("3?4", False),
            ("pass?word", False),
            ("Type pass?word into the field.", False),
            # (b) through the reviewers' punctuation, then a lowercase word
            ("Shall I pay?** then", False),
            ("Shall I pay? — then I stop.", False),
            ("Shall I pay? 🙂 then", False),
            ('"Shall I pay?". then', False),
            ("Shall I pay? » puis", False),
            ("Shall I pay?_then", False),
            # no question mark at all
            ("I will place the order. Then I will tell you.", False),
            ("   ", False),
        ],
        "false forms",
    )


def test_glued_uppercase_asks() -> None:
    """T-0107a2 criterion 2 and named re-scope R1, the unsafe direction (the model's own glued shape,
    "…Mona Lisa?Leonardo da Vinci…", t0098c-r4-plain-04-C3): a mark glued to a character that is neither lowercase
    nor a digit satisfies (a′), and that character is then the first alphanumeric, which is not lowercase, so (b)
    holds too: each row asks. The two rows T-0107a held false (``Shall I?Then``, ``Shall I pay?Then``) are moved
    here. Adversarial for the reviewer's mutations: revert to ``isalnum`` (every row false), ``isalpha`` in place of
    ``islower or isdigit`` (every letter row false), the old clause kept for U+FF1F (the ``？`` rows false)."""
    _check(
        [
            # the task's four, verbatim
            ("Shall I pay?Then I close the tab.", True),
            ("Shall I pay?Ok", True),
            ("…Mona Lisa?Leonardo da Vinci painted it.", True),
            ("Pay？Then", True),
            # moved from test_false_forms (R1, the intended change)
            ("Shall I?Then", True),
            ("Shall I pay?Then", True),
            # U+FF1F glued, mid-text and at the end of a word
            ("Shall I pay？Ok", True),
            ("Je passe la commande？Oui.", True),
            # uppercase beyond A-Z, a titlecase letter, a CJK ideograph: none is lowercase or a digit
            ("Supprimer l’événement?Élise est prévenue.", True),
            ("Pay?Ω is the total.", True),
            ("Отправить?Да.", True),
            ("Shall I pay?ǅ is the account.", True),
            ("Shall I pay?好的", True),
            # ``½`` is alphanumeric but not ``isdigit()`` (the task's spelling): (a′) holds, and (b) finds ``½`` first
            ("Shall I pay?½ now", True),
            # "some ``?``": an earlier glued-digit mark does not ask, a later glued-uppercase one does
            ("Is it 3?4 or shall I pay?Then I stop.", True),
        ],
        "glued to an uppercase (not lowercase, not a digit) character: asks",
    )


def test_glued_lowercase_or_digit_does_not() -> None:
    """T-0107a2 criterion 3, the too-wide direction (a rule that drops (a), or reads only (b)): a mark glued to a
    lowercase letter or a digit fails (a′), so each row is false -- the task's seven, verbatim, then the same for
    U+FF1F, non-ASCII lowercase, and digits of other scripts (``²`` and ``٣`` are ``isdigit()``). Green at the base
    by design: criterion 3 is "still false", a preservation row (the same forms were false under (a))."""
    _check(
        [
            # the task's seven, verbatim
            ("3?4", False),
            ("pass?word", False),
            ("Is it 3?4", False),
            ("I will click the Forgot password? link.", False),
            ("I will click the 'Forgot password?' link.", False),
            ("what? no idea.", False),
            ("?x", False),
            # U+FF1F glued to lowercase / a digit (kills "drop (a) for U+FF1F")
            ("Pay？t", False),
            ("Pay？4", False),
            # non-ASCII lowercase and other digits, glued
            ("Shall I pay?ßeta", False),
            ("Shall I pay?ω is the total.", False),
            ("Shall I pay?élise", False),
            ("Shall I pay?²", False),
            ("Shall I pay?٣", False),
            # the model's shape with a lowercase word glued
            ("…Mona Lisa?leonardo da Vinci painted it.", False),
            ("Shall I pay?then I close the tab.", False),
        ],
        "glued to a lowercase letter or a digit: does not ask",
    )


def test_sentence_final_after_quote_or_bracket() -> None:
    """Quotes and brackets after the mark are skipped by (b) like any non-alphanumeric: each closing ``"'”’)»`` and
    each opening ``"'“‘(«`` on its own, then the end, an uppercase word or a lowercase word."""
    cases: list[tuple[str, bool]] = []
    for q in QMARKS:
        for c in CLOSING:
            cases += [
                (f"Delete the event{q}{c}", True),
                (f"Delete the event{q}{c}  ", True),
                (f"Delete the event{q}{c} Then I stop.", True),
                (f"Delete the event{q}{c} then I stop.", False),
                (f"Delete the event{q}{c}x", False),
            ]
        for o in OPENING:
            cases += [
                (f"Delete the event{q} {o}Yes{o}", True),
                (f"Delete the event{q} {o}yes{o}", False),
            ]
        # a run of closings, then an opening
        cases += [
            (f'He asked "delete it{q}") Then I stopped.', True),
            (f"Delete it{q}) (Then I stop.)", True),
            (f"Delete it{q}») then", False),
        ]
    _check(cases, "after a closing quote or bracket")


def test_french_spacing() -> None:
    """A space (or a narrow no-break space, French typography) before the mark is allowed: nothing before ``?``
    matters; what follows decides."""
    _check(
        [
            ("Je passe la commande ?", True),
            ("Je passe la commande ? ", True),
            ("Je passe la commande  ?", True),
            ("Je passe la commande ?", True),
            ("Je passe la commande ？", True),
            ("Je passe la commande ? Oui, je confirme.", True),
            ("Je clique sur Mot de passe oublié ? puis j’attends.", False),
            ("Je passe la commande ? oui", False),
        ],
        "French spacing",
    )


# =====================================================================================================
# Total and deterministic
# =====================================================================================================

_alphabet = st.sampled_from(
    list("aZé É?？.!,;:-0123456789\n\t _*×—ßωΩǅ好🙂")
    + list(CLOSING)
    + list(OPENING)
    # T-0107a2 (a′): a non-``isdigit`` numeric and two non-ASCII ``isdigit`` characters
    + ["½", "²", "٣"]
    + [" ", " ", "́"]
)
_texts = st.one_of(st.text(), st.text(alphabet=_alphabet, max_size=40))
_no_mark = st.text().filter(lambda s: "?" not in s and "？" not in s)


@given(
    text=_texts,
    prefix=_no_mark,
    tail=st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8),
)
def test_total_and_deterministic(text: str, prefix: str, tail: str) -> None:
    """Hypothesis: on any string ``asks_user`` returns a ``bool`` without raising and the same one twice; and three
    consequences of the definition (not a re-implementation): a text with no mark never asks; a text that ends in a
    mark (then optional whitespace) always asks; a lone mark directly followed by a lowercase letter or a digit
    never asks (clause (a) fails for the only mark)."""
    first = asks(text)
    assert type(first) is bool, f"asks_user({text!r}) returned {type(first).__name__}, not bool"
    assert asks(text) is first, f"asks_user({text!r}) is not deterministic"
    stripped = text.replace("?", "").replace("\uff1f", "")
    assert asks(stripped) is False, f"no question mark, yet asks_user({stripped!r}) is true"
    for q in QMARKS:
        assert asks(text + q) is True, f"a text ending in {q!r} asks: asks_user({text + q!r})"
        assert asks(text + q + " \n") is True, f"trailing whitespace after {q!r}"
        assert asks(f"{prefix}{q}{tail}") is False, f"asks_user({prefix + q + tail!r})"


@given(text=st.one_of(st.text(alphabet=_alphabet, max_size=24), st.text(max_size=12)))
@example(text="Shall I pay?Then")
@example(text="Pay？Then")
@example(text="Pay?½")
@example(text="3?4")
def test_agrees_with_brute_force_oracle(text: str) -> None:
    """Hypothesis, tiny inputs: ``asks_user`` equals ``_oracle``, the Definition with T-0107a2's clause (a′)
    transcribed per mark (the module docstring). The alphabet holds every class the rule distinguishes: marks,
    lowercase, uppercase, titlecase, digits (ASCII and not), a non-digit numeric, CJK, whitespace incl. NBSP, quotes,
    brackets, ``_``, ``*``, ``×``, a dash, an emoji, a combining mark. The explicit examples are the glued forms
    (a′) moves, so the row is red at the base deterministically, not by Hypothesis's luck."""
    assert asks(text) is _oracle(text), f"asks_user({text!r}) != the definition's {_oracle(text)}"


def test_oracle_matches_the_hand_rows() -> None:
    """Premise for the oracle row: ``_oracle`` agrees with a handful of the hand-written rows above in both
    directions (so a broken oracle cannot make the agreement row vacuous)."""
    rows = [
        ("Shall I pay?", True),
        ("Shall I? × 3 it is.", True),
        ("**Shall I pay?**", True),
        ("Proceed?\n1. Yes", True),
        ("Forgot password? link", False),
        ("3?4", False),
        ("pass?word", False),
        ("? ω", False),
        ("", False),
        # T-0107a2 (a′), by hand from the task's criteria 2 and 3
        ("Shall I pay?Then I close the tab.", True),
        ("Pay？Then", True),
        ("…Mona Lisa?Leonardo da Vinci painted it.", True),
        ("Shall I pay?½ now", True),
        ("Is it 3?4", False),
        ("?x", False),
        ("Pay？t", False),
        ("Shall I pay?٣", False),
    ]
    wrong = [(t, w) for t, w in rows if _oracle(t) is not w]
    assert not wrong, f"the oracle disagrees with the hand rows: {wrong}"


# =====================================================================================================
# Linear time (the ruling: "The function scans in linear time"; reviewer A's quadratic case)
# =====================================================================================================

_TIMED = """
import json, sys, time
from local_astra.toolcall.ask_first import asks_user
text = sys.argv[1] * int(sys.argv[2]) + sys.argv[3]
t0 = time.perf_counter()
got = asks_user(text)
print(json.dumps({"got": got, "s": time.perf_counter() - t0}))
"""


def test_linear_time_on_a_run_of_marks() -> None:
    """Adversarial (reviewer A's quadratic case): 200,000 ``?`` then ``x``. Every mark but the last is followed by
    a non-alphanumeric run that ends at the lowercase ``x`` (false by (b)); the last is glued to ``x`` (false by (a));
    so the answer is false, and a scan that re-walks the run from each mark does ~2 * 10^10 steps. The call runs in a
    fresh CPU child process (killed at 20 s) and must take < 0.5 s -- a generous guard against quadratic behaviour,
    not a benchmark; a linear scan of 200,001 characters takes milliseconds."""
    _mod()  # the module's absence is a named assertion, not a child traceback
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _TIMED, "?", "200000", "x"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=REPO,
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(
            "T-0107a (linear time): asks_user('?' * 200_000 + 'x') did not return within 20 s (quadratic scan)"
        ) from None
    assert proc.returncode == 0, f"the timing child failed:\n{proc.stderr[-2000:]}"
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["got"] is False, (
        f"premise (by hand): '?' * 200_000 + 'x' does not ask; got {out['got']}"
    )
    assert out["s"] < 0.5, (
        f"T-0107a (linear time): asks_user('?' * 200_000 + 'x') took {out['s']:.3f} s (bound 0.5 s, CPU, n = 1)"
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
    """The module exists, defines ``ASKED_FIRST_REASON = "asked the user first"`` (verbatim from the task) and
    ``asks_user``, and imports only the standard library, none of it I/O, clocks or randomness (``ast`` over the
    source; the source is read, never executed for this check)."""
    mod = _mod()
    assert getattr(mod, "ASKED_FIRST_REASON", None) == "asked the user first", (
        f"T-0107a: ASKED_FIRST_REASON is {getattr(mod, 'ASKED_FIRST_REASON', None)!r}"
    )
    assert callable(getattr(mod, "asks_user", None)), "T-0107a: asks_user is not callable"
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, (
                f"T-0107a: a relative import in a stdlib-only module: {ast.dump(node)}"
            )
            assert node.module is not None
            roots.add(node.module.split(".")[0])
    roots.discard("__future__")
    non_std = sorted(r for r in roots if r not in sys.stdlib_module_names)
    assert not non_std, f"T-0107a: {MODULE} imports only the stdlib; it imports {non_std}"
    impure = sorted(roots & IMPURE)
    assert not impure, f"T-0107a: {MODULE} is pure; it imports {impure}"
    calls = sorted(
        {
            n.func.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        & {"open", "print", "input", "exec", "eval", "__import__"}
    )
    assert not calls, f"T-0107a: {MODULE} is pure; it calls {calls}"
