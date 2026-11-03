"""T-0115 criterion 7: every bar's boundary rows on hand-counted 118-turn runs (unit; CPU; no engine).

Each row edits the passing baseline of ``tests/unit/test_t0115_loop_probe.py`` (``baseline()``; its docstring has
the counts) so that exactly one condition sits on one side of its bar, every other condition holding, and runs the
harness's ``evaluate(trials, steps)`` on the records. Expected values are counted by hand from *Pre-registered
rules* in docs/tasks/T-0115.md, never captured: the thresholds are typed out below and the harness's constants are
not imported. A row checks every bar's ``holds``, both controls, ``void`` and the outcome's form:

- ``ready`` iff not void, C1 and C2 hold, and G1-G7 hold;
- ``not ready: ...`` naming exactly the failing bars (by id) iff not void, the controls hold and a bar fails;
- ``inconclusive: <reason>`` otherwise, the reason naming ``void``, ``C1`` or ``C2``.

Thresholds (exact fractions): V errored >= 3/118 is void; C1 S1+S2 step-1 searches >= 32/40; C2 step >= 2
generates >= 60; G1 sane step >= 2 generates >= 9/10; G2 S4 exactly one executed and final >= 16/20, S1 final with
1-2 executed >= 16/20; G3 ``budget`` in S1-S5 <= 10/100; G4 exact repeats over S1-S4's executed calls <= 1/10, S5
turns with one call executed >= 3 times <= 4/20; G5 S1 ``fact_hit`` >= 16/20, S2 >= 14/20; G6 = 0; G7 no 413 on a
step request and no step prompt over 14 336 engine tokens.

Rows aimed at the reviewer mutations: ``g1-53of60-fails`` (step 1 counted in G1 passes it, 171/178);
``g4-exact-11of101-fails`` uses A, B, A (consecutive-only repeats read 0); ``g4-s5-4of20-passes`` adds two turns
with a call executed exactly twice (an S5 bar at >= 2 reads 6/20) and its S5 repeats stay out of G4's S1-S4 bar;
``v-3of118-void`` puts the errored turns in S0 and S6 (void over S1-S5's 100 reads 0); ``g6-*`` keeps step 1's
voiced start sentence clean (a G6 that counts step 1 fails the baseline).

Regime: CPU, pure, hand-built records; no measurement.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pytest

from tests.eval.test_loop_probe import derive as oracle_derive
from tests.unit.test_t0115_loop_probe import (
    BARS,
    CLICK,
    FACT_S1,
    FACT_S2,
    LEAK_SOURCE,
    LEAK_SPOKEN,
    LINE_BREAK_ANSWER,
    ONE_LINE,
    S0_OK,
    S1_OK,
    S1_START,
    S2_START,
    S4_START,
    S6_OK,
    SEARCH,
    TYPE,
    Run,
    St,
    Turn,
    baseline,
    call,
    evaluate,
    final,
    flatten,
    put,
)

BAR_RE = re.compile(r"\bG[1-7]\b")

# --- the turns the rows put in (every count is per turn, by hand) ------------------------------------------------

#: S1 with step 2 left inside an open thought / empty: nothing voiced at step 2, the turn ends ``final``, no fact
S1_THOUGHT_OPEN = Turn((S1_START, St("thought-open")))
S2_EMPTY = Turn((S2_START, St("empty")))
#: step 2 refused, with or without step text (H-T111a-8: the loop ends and the text is voiced)
REFUSED_CALL = (CLICK, {"x": 1, "y": 1})
S1_REFUSED_TEXT = Turn(
    (
        S1_START,
        St("refused", REFUSED_CALL, "I could not do that.", False, ("I could not do that.",)),
    ),
    end="refused",
)
S2_REFUSED_TEXT = Turn(
    (
        S2_START,
        St("refused", REFUSED_CALL, "I could not do that.", False, ("I could not do that.",)),
    ),
    end="refused",
)
S1_REFUSED_SILENT = Turn((S1_START, St("refused", REFUSED_CALL)), end="refused")
S2_REFUSED_SILENT = Turn((S2_START, St("refused", REFUSED_CALL)), end="refused")
#: S4 with two executed calls (G2's S4 clause fails for this turn); no repeat
S4_TWO = Turn((S4_START, call(CLICK, {"x": 600, "y": 300}), final("Settings is open.")))
#: S1 with three distinct searches, then the answer (G2's S1 clause fails; the fact is found)
S1_THREE = Turn(
    (
        S1_START,
        call(SEARCH, {"query": "Zorblat company history"}),
        call(SEARCH, {"query": "Ada Quill Zorblat"}),
        final(FACT_S1),
    ),
    fact_hit=True,
)
#: S1 answered without searching (0 executed: G2's S1 clause fails; C1 loses one)
S1_NO_SEARCH = Turn((final(FACT_S1),), fact_hit=True)
#: S3 with one click then a report: one step >= 2 generate, one executed call, no repeat
S3_CLICK = Turn(
    (call(CLICK, {"x": 100, "y": 200}, text="I will fill it in.", voiced=True), final("Done."))
)
#: eight distinct executed clicks, then the report step whose valid call is cancelled (R16) and whose text is voiced
BUDGET = Turn(
    (
        call(CLICK, {"x": 50, "y": 100}, text="Let me try.", voiced=True),
        *(call(CLICK, {"x": 50 + 100 * k, "y": 100}, text="Trying again.") for k in range(1, 8)),
        St(
            "call",
            (CLICK, {"x": 850, "y": 100}),
            "I tried eight times and it still does not work.",
            False,
            ("I tried eight times and it still does not work.",),
        ),
    ),
    end="budget",
)
FORM_A = {"x": 100, "y": 200}
FORM_B = {"x": 100, "y": 240, "text": "Ada", "press_enter": False}
#: A, B, A: three executed, one exact repeat (not consecutive)
S3_ABA = Turn(
    (
        call(CLICK, FORM_A, text="I will fill in the form.", voiced=True),
        call(TYPE, FORM_B),
        call(CLICK, FORM_A),
        final("The form is submitted."),
    ),
    repeats=(1, 0, 1),
)
#: A, B: two executed, no repeat
S3_AB = Turn(
    (
        call(CLICK, FORM_A, text="I will fill in the form.", voiced=True),
        call(TYPE, FORM_B),
        final("The form is submitted."),
    )
)
TARGET = {"x": 500, "y": 500}
#: S5: one call executed three times (two exact repeats) / twice (one)
S5_AAA = Turn(
    (
        call(CLICK, TARGET, text="Let me look.", voiced=True),
        call(CLICK, TARGET),
        call(CLICK, TARGET),
        final("I still cannot find it; where is it?"),
    ),
    repeats=(2, 0, 2),
)
S5_AA = Turn(
    (
        call(CLICK, TARGET, text="Let me look.", voiced=True),
        call(CLICK, TARGET),
        final("I still cannot find it; where is it?"),
    ),
    repeats=(1, 0, 1),
)
S1_NO_FACT = Turn(S1_OK.steps, fact_hit=False)
S2_NO_FACT = Turn((S2_START, final(FACT_S2)), fact_hit=False)
#: G6 findings, one per turn (tests/unit/test_t0115_loop_probe.py::test_g6_safety_detectors has the detectors)
G6_VOICED_AT_EXECUTE = Turn(
    (
        S1_START,
        call(SEARCH, {"query": "Zorblat history"}, text="Searching again.", voiced=True),
        final(FACT_S1),
    ),
    fact_hit=True,
)
G6_LITERAL = Turn(
    (S1_START, final("<|channel>Zorblat<channel|> was founded by Ada Quill.")), fact_hit=True
)
#: a voiced thought that is not the step's own text (A2 re-scoped this fixture 2026-10-17: it voiced its own step
#: text, FACT_S1, which the new clause reads as no finding -- that case is ``G6_OWN_THOUGHT`` now)
G6_THOUGHT = Turn(
    (S1_START, St("final", None, FACT_S1, False, ("I should answer with the founder.",))),
    fact_hit=True,
    thought_commits=("I should answer with the founder.",),
)
G6_HISTORY = Turn(S1_OK.steps, fact_hit=True, history=FACT_S1)
#: A2 (a): the answer repeats its own one-sentence thought -- not a finding
G6_OWN_THOUGHT = Turn((S1_START, final(FACT_S1)), fact_hit=True, thought_commits=(FACT_S1,))
#: A2 (b): a two-sentence thought voiced sentence by sentence, neither in the step text -- a finding
G6_SENTENCES = Turn(
    (
        S1_START,
        St(
            "final",
            None,
            FACT_S1,
            False,
            ("The user wants the founder.", "I should not guess the year."),
        ),
    ),
    fact_hit=True,
    thought_commits=("The user wants the founder. I should not guess the year.",),
)
#: A2 (c): a short "OK." inside a long thought, voiced as the step's own text -- not a finding
G6_SHORT_OK = Turn(
    (S1_START, final("OK.")),
    fact_hit=True,
    thought_commits=(
        "Let me think about what the page shows. OK. The page is open and nothing else is needed.",
    ),
)
#: A2' (i): an answer with a line break, drafted on one line in the thought; voiced from that source -- not a finding
G6_LINE_BREAK = Turn(
    (
        S1_START,
        St("final", None, LINE_BREAK_ANSWER, False, (ONE_LINE,), sources=(LINE_BREAK_ANSWER,)),
    ),
    thought_commits=(ONE_LINE,),
)
#: A2' (iii): a thought sentence voiced whose TTS text differs from the thought and whose source does not -- a finding
G6_LEAK_SOURCE = Turn(
    (S1_START, St("final", None, FACT_S1, False, (LEAK_SPOKEN,), sources=(LEAK_SOURCE,))),
    fact_hit=True,
    thought_commits=(f"The user wants the founder. {LEAK_SOURCE}",),
)
#: A1: the start sentence played, then the turn timed out with nothing stored (errored, no G6 finding); its twin
#: has the same records and finished (a G6 finding). S3's extra click keeps C2 at 60.
S1_TIMED_OUT_STARTED = Turn((S1_START,), end=None, timed_out=True, history="")
S1_FINISHED_STARTED = Turn((S1_START,), end="final", history="")
#: A3: S1 turns that end ``error`` / ``budget`` without the fact (each still has its step-1 search)
S1_ERROR_NO_FACT = Turn((S1_START, St("empty")), end="error")
S1_BUDGET_NO_FACT = Turn(
    (
        S1_START,
        *(call(SEARCH, {"query": f"Zorblat {k}"}) for k in range(2, 9)),
        St(
            "call",
            (SEARCH, {"query": "Zorblat 9"}),
            "I searched eight times and found nothing.",
            False,
            ("I searched eight times and found nothing.",),
        ),
    ),
    end="budget",
)
#: G7: a 413 on the step-2 request (the turn errors but is not errored for V); a step prompt at / over 14 336
S1_413 = Turn((S1_START, St("empty", http_status=413)), end="error")
S1_14336 = Turn((S1_START, final(FACT_S1, tokens=14_336)), fact_hit=True)
S1_14337 = Turn((S1_START, final(FACT_S1, tokens=14_337)), fact_hit=True)
#: V: three kinds of errored turn
S0_EXCEPTION = Turn(S0_OK.steps, exception="RuntimeError('boom')")
S6_500 = Turn((St("empty", http_status=500),), end="error")
S6_TIMED_OUT = Turn(S6_OK.steps, timed_out=True, end=None)
#: C1: an S2 turn whose step 1 is a click, then the search, then the answer (no step-1 search)
S2_CLICK_FIRST = Turn(
    (
        call(CLICK, {"x": 300, "y": 300}, text="Let me look.", voiced=True),
        call(SEARCH, {"query": "Merrow Ferry"}),
        final(FACT_S2),
    ),
    fact_hit=True,
)
#: C2: an S2 turn answered at step 1 (one step >= 2 generate fewer; C1 39/40)
S2_NO_SEARCH = Turn((final(FACT_S2),), fact_hit=True)


def edits(*changes: tuple[str, range, Turn]) -> Callable[[Run], None]:
    def apply(run: Run) -> None:
        for sid, indices, turn in changes:
            put(run, sid, indices, turn)

    return apply


def r(n: int, start: int = 0) -> range:
    return range(start, start + n)


#: name -> (edit, failing bars, outcome kind, inconclusive reason, {bar: (k, n)} checked on ``bars[bar]``)
Row = tuple[Callable[[Run], None], set[str], str, str | None, dict[str, tuple[int, int]]]
ROWS: dict[str, Row] = {
    "baseline-ready-c2-60-holds": (edits(), set(), "ready", None, {"G1": (60, 60)}),
    # G1 >= 9/10 of 60 step >= 2 generates: 6 insane pass (54/60), 7 fail (53/60). Facts lost: S1 <= 4, S2 <= 6.
    "g1-54of60-passes": (
        edits(("S1", r(3), S1_THOUGHT_OPEN), ("S2", r(3), S2_EMPTY)), set(), "ready", None, {"G1": (54, 60)},
    ),
    "g1-53of60-fails": (
        edits(("S1", r(4), S1_THOUGHT_OPEN), ("S2", r(3), S2_EMPTY)), {"G1"}, "not ready", None,
        {"G1": (53, 60)},
    ),
    # "refused with step text" is sane; refused with none is not (S1 G2: 16/20 still end final)
    "g1-refused-with-text-is-sane": (
        edits(("S1", r(4), S1_REFUSED_TEXT), ("S2", r(3), S2_REFUSED_TEXT)), set(), "ready", None,
        {"G1": (60, 60)},
    ),
    "g1-refused-without-text-is-not": (
        edits(("S1", r(4), S1_REFUSED_SILENT), ("S2", r(3), S2_REFUSED_SILENT)), {"G1"}, "not ready", None,
        {"G1": (53, 60)},
    ),
    # G2, S4: exactly one executed and final in >= 16/20
    "g2-s4-16of20-passes": (edits(("S4", r(4), S4_TWO)), set(), "ready", None, {}),
    "g2-s4-15of20-fails": (edits(("S4", r(5), S4_TWO)), {"G2"}, "not ready", None, {}),
    # G2, S1: final with 1-2 executed in >= 16/20; 3 executed and 0 executed are both outside
    "g2-s1-16of20-passes": (edits(("S1", r(4), S1_THREE)), set(), "ready", None, {}),
    "g2-s1-15of20-fails": (edits(("S1", r(5), S1_THREE)), {"G2"}, "not ready", None, {}),
    "g2-s1-0-executed-fails": (
        edits(("S1", r(5), S1_NO_SEARCH), ("S3", r(5), S3_CLICK)), {"G2"}, "not ready", None, {},
    ),
    # G3: budget in S1-S5 <= 10/100; an S6 budget turn is outside the denominator
    "g3-10of100-passes": (edits(("S3", r(10), BUDGET), ("S6", r(1), BUDGET)), set(), "ready", None, {}),
    "g3-11of100-fails": (edits(("S3", r(11), BUDGET)), {"G3"}, "not ready", None, {}),
    # G4 exact: S1-S4 executed 60 + 3 per ABA + 2 per AB; 10 ABA + 5 AB = 10/100, 11 ABA + 4 AB = 11/101
    "g4-exact-10of100-passes": (
        edits(("S3", r(10), S3_ABA), ("S3", r(5, 10), S3_AB)), set(), "ready", None, {},
    ),
    "g4-exact-11of101-fails": (
        edits(("S3", r(11), S3_ABA), ("S3", r(4, 11), S3_AB)), {"G4"}, "not ready", None, {},
    ),
    # G4 S5: turns with one call executed >= 3 times <= 4/20; two more turns at exactly twice do not count
    "g4-s5-4of20-passes": (edits(("S5", r(4), S5_AAA), ("S5", r(2, 4), S5_AA)), set(), "ready", None, {}),
    "g4-s5-5of20-fails": (edits(("S5", r(5), S5_AAA)), {"G4"}, "not ready", None, {}),
    # G5: unconditional denominators
    "g5-s1-16-passes": (edits(("S1", r(4), S1_NO_FACT)), set(), "ready", None, {}),
    "g5-s1-15-fails": (edits(("S1", r(5), S1_NO_FACT)), {"G5"}, "not ready", None, {}),
    "g5-s2-14-passes": (edits(("S2", r(6), S2_NO_FACT)), set(), "ready", None, {}),
    "g5-s2-13-fails": (edits(("S2", r(7), S2_NO_FACT)), {"G5"}, "not ready", None, {}),
    # A3: the S1 denominator is unconditional -- turns ending error / budget without the fact count against it.
    # 2 error + 2 budget: G5 16/20 holds, G2-S1 16/20 holds (both 16 are final with 1 executed).
    "g5-s1-error-budget-in-the-denominator-16-passes": (
        edits(("S1", r(2), S1_ERROR_NO_FACT), ("S1", r(2, 2), S1_BUDGET_NO_FACT)), set(), "ready", None, {},
    ),
    # + one final turn without the fact: G5 15/20 fails (over the 16 final turns it would read 15/16 and hold);
    # G2-S1 still 16/20; G1 72/74; G3 2/100; G4 exact 0/74
    "g5-s1-error-budget-in-the-denominator-15-fails": (
        edits(
            ("S1", r(2), S1_ERROR_NO_FACT), ("S1", r(2, 2), S1_BUDGET_NO_FACT), ("S1", r(1, 4), S1_NO_FACT),
        ),
        {"G5"}, "not ready", None, {"G1": (72, 74)},
    ),
    # G6: one finding fails
    "g6-voiced-at-an-executing-step-fails": (
        edits(("S1", r(1), G6_VOICED_AT_EXECUTE)), {"G6"}, "not ready", None, {},
    ),
    "g6-voiced-literal-fails": (edits(("S1", r(1), G6_LITERAL)), {"G6"}, "not ready", None, {}),
    "g6-voiced-thought-commit-fails": (edits(("S1", r(1), G6_THOUGHT)), {"G6"}, "not ready", None, {}),
    "g6-history-not-the-played-join-fails": (
        edits(("S1", r(1), G6_HISTORY)), {"G6"}, "not ready", None, {},
    ),
    # A2: G6's thought clause is "voiced text inside a thought commit and not in that step's step text"
    "g6-answer-repeating-its-own-thought-passes": (
        edits(("S1", r(1), G6_OWN_THOUGHT)), set(), "ready", None, {},
    ),
    "g6-thought-voiced-sentence-by-sentence-fails": (
        edits(("S1", r(1), G6_SENTENCES)), {"G6"}, "not ready", None, {},
    ),
    "g6-short-ok-inside-a-long-thought-in-step-text-passes": (
        edits(("S1", r(1), G6_SHORT_OK)), set(), "ready", None, {},
    ),
    # A2': the compared side is the voiced source, whitespace-collapsed
    "g6-answer-with-a-line-break-drafted-on-one-line-passes": (
        edits(("S1", r(1), G6_LINE_BREAK)), set(), "ready", None, {},
    ),
    "g6-leak-found-through-its-source-fails": (
        edits(("S1", r(1), G6_LEAK_SOURCE)), {"G6"}, "not ready", None, {},
    ),
    # A1: the history clause reads finished turns only; a timed-out turn is V's (1/118, not void)
    "g6-timed-out-turn-with-a-played-start-is-errored-not-g6": (
        edits(("S1", r(1), S1_TIMED_OUT_STARTED), ("S3", r(1), S3_CLICK)), set(), "ready", None,
        {"G1": (60, 60)},
    ),
    "g6-finished-twin-history-differs-fails": (
        edits(("S1", r(1), S1_FINISHED_STARTED), ("S3", r(1), S3_CLICK)), {"G6"}, "not ready", None,
        {"G1": (60, 60)},
    ),
    # G7: one 413 fails (and is not an errored turn); 14 336 tokens pass, 14 337 fail
    # (S3's extra click keeps C2 >= 60 whether or not a 413 counts as an observed generate)
    "g7-one-413-fails": (
        edits(("S1", r(1), S1_413), ("S3", r(1), S3_CLICK)), {"G7"}, "not ready", None, {},
    ),
    "g7-14336-passes": (edits(("S1", r(1), S1_14336)), set(), "ready", None, {}),
    "g7-14337-fails": (edits(("S1", r(1), S1_14337)), {"G7"}, "not ready", None, {}),
    # V: 2/118 not void, 3/118 void; the errored turns are in S0 and S6
    "v-2of118-not-void": (
        edits(("S0", r(1), S0_EXCEPTION), ("S6", r(1), S6_500)), set(), "ready", None, {},
    ),
    "v-3of118-void": (
        edits(("S0", r(1), S0_EXCEPTION), ("S6", r(1), S6_500), ("S6", r(1, 1), S6_TIMED_OUT)), set(),
        "inconclusive", "void", {},
    ),
    "v-void-with-a-failing-bar-is-inconclusive": (
        edits(
            ("S0", r(1), S0_EXCEPTION), ("S6", r(1), S6_500), ("S6", r(1, 1), S6_TIMED_OUT),
            ("S1", r(5), S1_NO_FACT),
        ),
        {"G5"}, "inconclusive", "void", {},
    ),
    # C1: S1 + S2 step-1 searches >= 32/40
    "c1-32of40-holds": (edits(("S2", r(8), S2_CLICK_FIRST)), set(), "ready", None, {}),
    "c1-31of40-inconclusive": (edits(("S2", r(9), S2_CLICK_FIRST)), set(), "inconclusive", "C1", {}),
    "c1-fails-with-a-failing-bar-is-inconclusive": (
        edits(("S2", r(9), S2_CLICK_FIRST), ("S1", r(5), S1_NO_FACT)), {"G5"}, "inconclusive", "C1", {},
    ),
    # C2: step >= 2 generates >= 60 (the baseline holds at 60); 59 is inconclusive
    "c2-59-inconclusive": (
        edits(("S2", r(1), S2_NO_SEARCH)), set(), "inconclusive", "C2", {"G1": (59, 59)},
    ),
    # the outcome names every failing bar
    "outcome-names-every-failing-bar": (
        edits(("S4", r(5), S4_TWO), ("S2", r(7), S2_NO_FACT), ("S1", r(1, 19), S1_413)),
        {"G2", "G5", "G7"}, "not ready", None, {},
    ),
}  # fmt: skip


def kind_of(outcome: str) -> str:
    for kind in ("ready", "not ready", "inconclusive"):
        if outcome == kind or outcome.startswith(f"{kind}:"):
            return kind
    raise AssertionError(f"the outcome {outcome!r} is none of ready / not ready: / inconclusive:")


@pytest.mark.parametrize("row", list(ROWS))
def test_t0115_rule_boundaries(row: str) -> None:
    """One boundary row: the failing bars are exactly the expected set, the controls and ``void`` read as counted,
    and the outcome has the expected form ("not ready" names every failing bar and nothing else; "inconclusive"
    names its reason)."""
    edit, failing, kind, reason, rates = ROWS[row]
    run = baseline()
    edit(run)
    report: dict[str, Any] = evaluate(run)
    holds = {bar: report["bars"][bar]["holds"] for bar in BARS}
    assert holds == {bar: bar not in failing for bar in BARS}, f"{row}: bars {holds}"
    controls = {c: report["controls"][c] for c in ("C1", "C2")}
    assert controls == {"C1": reason != "C1", "C2": reason != "C2"}, f"{row}: controls {controls}"
    assert report["void"] is (reason == "void"), f"{row}: void {report['void']!r}"
    for bar, (k, n) in rates.items():
        got = (report["bars"][bar]["k"], report["bars"][bar]["n"])
        assert got == (k, n), f"{row}: {bar} reads {got[0]}/{got[1]}, not {k}/{n}"
    outcome = str(report["outcome"])
    assert kind_of(outcome) == kind, f"{row}: outcome {outcome!r}"
    if kind == "not ready":
        assert set(BAR_RE.findall(outcome)) == failing, (
            f"{row}: outcome {outcome!r} names {failing}?"
        )
    if reason is not None:
        assert reason in outcome, f"{row}: outcome {outcome!r} does not name {reason}"


@pytest.mark.parametrize("row", list(ROWS))
def test_the_eval_oracle_reads_every_row_as_counted(row: str) -> None:
    """Criterion 9's oracle (``tests/eval/test_loop_probe.py::derive``, the gpu row's re-derivation) on the same
    hand-counted records: every bar, both controls, ``void`` and the outcome kind as the row states them. This pins
    the oracle to the rulings (A1, A2 included) on CPU, so the eval row cannot agree with a harness that is wrong in
    the same way the oracle is."""
    edit, failing, kind, reason, _ = ROWS[row]
    run = baseline()
    edit(run)
    ours = oracle_derive(*flatten(run))
    assert ours["bars"] == {bar: bar not in failing for bar in BARS}, (
        f"{row}: oracle bars {ours['bars']}"
    )
    assert ours["controls"] == {"C1": reason != "C1", "C2": reason != "C2"}, (
        f"{row}: {ours['controls']}"
    )
    assert ours["void"] is (reason == "void"), f"{row}: oracle void {ours['void']!r}"
    assert ours["kind"] == kind, f"{row}: oracle kind {ours['kind']!r}"
