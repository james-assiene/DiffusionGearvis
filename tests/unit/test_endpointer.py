"""The pure endpointer, ``local_astra/reflex/turn_predictor.py`` (T-0032 criterion 3).

**The API these tests call** (the task fixes the module, the decisions and the constants, and
leaves the shape to the test-author; this is that shape, stated so the implementer builds to it
and the reviewer can object to it in one place):

    from local_astra.reflex.turn_predictor import (
        Endpointer, EOT_P_THRESHOLD, EOT_SILENCE_MS, MIN_SPEECH_MS, PAUSE_HEAD_INDEX,
    )

    ep = Endpointer()                       # no arguments; one per session, reset by construction
    ep.observe(event) -> decision | None     # event is a contracts.speech.SttWordEvent or
                                             # SttStepEvent, in wire order, with *audio* times

    decision is ("onset", audio_ms)                      -- criterion 3(a)
             or ("end_of_turn", t_eot_ms, p_end)          -- criterion 3(b)/(c)

``observe`` returns at most one decision per event and ``None`` the rest of the time. Nothing
else is promised: the EMA, the state, the internals are the implementer's. ``TurnPredictor``
(the existing stub, smart-turn v3.2) is untouched by this task.

On the box, note (T-0007a census, T-0038): the server sends ``EndWord``, so until T-0038 merges
a real word's ``end_ms`` equals its ``start_ms`` -- rule 3(b)'s ``t_eot_ms`` is then the last
word's *start*, a documented under-estimate of the end of turn. Nothing here is weakened for it:
the words in these rows carry honest durations, and the rule under test is "the last word's
``end_ms``", whatever the server puts there.

Regime: pure arithmetic, no clock, no I/O, n = 1 deterministic per row (the Hypothesis row states
its own); every expected value below is computed by hand from the word times in the test, never
from the module. The pause-head values (0.9 in silence, 0.05 in speech) are the fake's chosen
values (``tests/fakes/stt.py``), not measurements -- the measurement is T-0032 criterion 12.
"""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.speech import SttEvent, SttStepEvent, SttWordEvent
from local_astra.contracts.turn import TurnState
from local_astra.reflex import turn_predictor as _tp

REPO = Path(__file__).resolve().parents[2]
MODULE = REPO / "local_astra" / "reflex" / "turn_predictor.py"
FIXTURES = REPO / "tests" / "fixtures"

STEP_MS = 80  # moshi steps every 80 ms of audio (docs/02 §2.2)
HIGH = 0.9  # the fake's silence head
LOW = 0.05  # the fake's speech head

_MISSING: list[str] = []
try:  # the module exists as a stub; the endpointer surface does not
    from local_astra.reflex.turn_predictor import (
        EOT_P_THRESHOLD,
        EOT_SILENCE_MS,
        MIN_SPEECH_MS,
        PAUSE_HEAD_INDEX,
        Endpointer,
    )
except ImportError as exc:  # pragma: no cover - the pre-implementation path
    Endpointer = None  # type: ignore[assignment,misc]
    # Placeholders so the helpers below can be *defined*; every test calls ``require()`` first,
    # so a pending run fails with the AssertionError it names and never with a NameError.
    EOT_P_THRESHOLD, EOT_SILENCE_MS, MIN_SPEECH_MS, PAUSE_HEAD_INDEX = 0.6, 800, 200, 0
    _MISSING.append(str(exc))


def require() -> None:
    assert not _MISSING, (
        "T-0032 criterion 3: local_astra/reflex/turn_predictor.py must export Endpointer, "
        "EOT_P_THRESHOLD, EOT_SILENCE_MS, MIN_SPEECH_MS and PAUSE_HEAD_INDEX "
        f"(docs/tasks/T-0032.md); import said: {'; '.join(_MISSING)}"
    )


def heads(p: float, slots: int = 8) -> tuple[float, ...]:
    """A uniform ``prs`` vector, so the test does not have to know ``PAUSE_HEAD_INDEX``."""
    return (p,) * max(slots, PAUSE_HEAD_INDEX + 1)


def spoken(text: str, start_ms: int, ms_per_word: int = 200) -> list[SttWordEvent]:
    return [
        SttWordEvent(
            text=word,
            start_ms=start_ms + i * ms_per_word,
            end_ms=start_ms + (i + 1) * ms_per_word,
        )
        for i, word in enumerate(text.split())
    ]


def arrival_of(word: SttWordEvent, *, delivery_lag_ms: int, from_ms: int = 0) -> int:
    """The step-clock time at which ``word`` is delivered, given a recogniser lag of ``L``.

    The recogniser has examined audio up to ``now - L``, so a word is published on the first step
    whose own ``t_ms`` is ``>= end_ms + L`` (T-0051 *Implementation notes*). Steps sit on the
    ``from_ms + 80 k`` grid, so the arrival is that grid point, never ``end_ms + L`` itself -- and
    for ``L = 0`` on a grid-aligned word it *is* ``end_ms``, which is why every pre-T-0051 row in
    this module keeps its expected value (T-0051 criterion 2, "the L = 0 case is the guard").
    """
    target = word.end_ms + delivery_lag_ms
    if target <= from_ms:
        return from_ms
    steps_needed = -(-(target - from_ms) // STEP_MS)  # ceil division onto the step grid
    return from_ms + steps_needed * STEP_MS


def stream(
    words: list[SttWordEvent],
    *,
    until_ms: int,
    head_of: object = None,
    from_ms: int = 0,
    delivery_lag_ms: int = 0,
) -> list[SttEvent]:
    """Words and 80 ms steps merged into one wire-ordered stream, in *step-clock* order.

    A step takes its place at its own ``t_ms``. A word takes its place at its **arrival**
    (:func:`arrival_of`), *after* the step of that time: on the wire the recogniser publishes a
    word having just consumed the audio frame it ends in, and the desk log of 2026-09-16 shows
    exactly that shape -- ``machine.step`` records 3069 → 3073, 3080 → 3081 and 3112 → 3116 each
    put the ``stt.step`` immediately before the ``stt.word`` carrying the same stamp.

    ``delivery_lag_ms`` (``L``) is the recogniser's emission delay, ≈ 510 ms on the GPU tonight
    and seconds on the CPU (T-0051 *Regime*). The word's ``start_ms``/``end_ms`` stay where they
    were *spoken*; only its place in the stream moves. ``L = 0`` reproduces the pre-T-0051
    streams byte for byte.

    ``head_of(t_ms) -> float`` decides the pause head; the default is the fake's rule -- low
    while a word is being spoken, high otherwise.
    """
    spans = [(w.start_ms, w.end_ms) for w in words]

    def default_head(t_ms: int) -> float:
        return LOW if any(a <= t_ms < b for a, b in spans) else HIGH

    head = head_of if head_of is not None else default_head
    timeline: list[tuple[int, int, SttEvent]] = [
        (arrival_of(w, delivery_lag_ms=delivery_lag_ms, from_ms=from_ms), 2, w) for w in words
    ]
    for t_ms in range(from_ms, until_ms + 1, STEP_MS):
        value = head(t_ms)
        prs = value if isinstance(value, tuple) else heads(float(value))
        timeline.append((t_ms, 1, SttStepEvent(pause_heads=prs, t_ms=t_ms)))
    timeline.sort(key=lambda row: (row[0], row[1]))
    return [event for _, _, event in timeline]


def drive(events: list[SttEvent]) -> list[tuple[SttEvent, tuple]]:
    """``(event, decision)`` for every event that produced one."""
    require()
    ep = Endpointer()
    out: list[tuple[SttEvent, tuple]] = []
    for event in events:
        decision = ep.observe(event)
        if decision is not None:
            out.append((event, decision))
    return out


def kinds(fired: list[tuple[SttEvent, tuple]]) -> list[str]:
    return [decision[0] for _, decision in fired]


# --- criterion 3(a), 3(b), 3(d) ----------------------------------------------------------------


def test_a_run_of_speech_then_silence_yields_one_onset_and_one_end_of_turn():
    """One onset at the first word's ``start_ms``, one end of turn at the last word's ``end_ms``.

    Hand-computed: five words of 200 ms from 1000 ms, so the run is [1000, 2000]; the steps after
    2000 ms carry the silence head, so the EMA crosses ``EOT_P_THRESHOLD`` somewhere after it.
    The decision's ``t_eot_ms`` is **2000** -- the last word's end -- and *not* the audio time of
    the step that triggered it, which is what a naive implementation reports and what would make
    ``freeze_after`` swallow the end of the sentence (docs/02 §4, ENDPOINTING rows).
    """
    words = spoken("what is on this screen", 1000)
    assert (words[0].start_ms, words[-1].end_ms) == (1000, 2000)
    fired = drive(stream(words, until_ms=4000))

    assert kinds(fired) == ["onset", "end_of_turn"], (
        f"criterion 3(a,b,d): exactly one onset then one end of turn; got {fired}"
    )
    assert fired[0][1] == ("onset", 1000), (
        f"the onset carries the first word's start_ms (1000); got {fired[0][1]}"
    )
    kind, t_eot_ms, p_end = fired[1][1]
    assert (kind, t_eot_ms) == ("end_of_turn", 2000), (
        f"criterion 3(b): t_eot_ms is the last word's end_ms (2000); got {fired[1][1]}"
    )
    assert p_end >= EOT_P_THRESHOLD, (
        f"the reported p_end is the EMA that crossed {EOT_P_THRESHOLD}; got {p_end}"
    )
    trigger = fired[1][0]
    assert isinstance(trigger, SttStepEvent) and trigger.t_ms > t_eot_ms, (
        "the decision is triggered by a *later* step than the time it reports; it fired on "
        f"{trigger}"
    )


# --- criterion 3(c): the backstop runs on the step clock ---------------------------------------


def test_the_silence_backstop_uses_the_step_clock_not_wall_time():
    """**Adversarial, and the reason this module is pure.**

    The step stream's audio time advances 80 ms per step while the test's own wall clock does not
    move at all (the whole row runs in well under a millisecond, asserted below). The pause heads
    are *low* throughout the silence, so rule 3(b) cannot fire and only the backstop can: an
    implementation that measured silence with ``time.monotonic()`` produces nothing here, and on
    the box would keep the user waiting for as long as the utterance was short.
    """
    started = time.perf_counter()
    words = spoken("hello there", 400)  # [400, 800]
    fired = drive(
        stream(words, until_ms=800 + 3 * EOT_SILENCE_MS, head_of=lambda t_ms: LOW, from_ms=0)
    )
    wall_ms = (time.perf_counter() - started) * 1000.0

    assert kinds(fired) == ["onset", "end_of_turn"], (
        f"criterion 3(c): the backstop must endpoint on the step clock alone; got {fired}"
    )
    kind, t_eot_ms, _ = fired[1][1]
    assert (kind, t_eot_ms) == ("end_of_turn", 800), fired[1][1]
    trigger_ms = fired[1][0].t_ms
    assert 800 + EOT_SILENCE_MS <= trigger_ms <= 800 + EOT_SILENCE_MS + 2 * STEP_MS, (
        f"the backstop fires {EOT_SILENCE_MS} ms of *step* time after the last word (800 ms), i.e."
        f" between {800 + EOT_SILENCE_MS} and {800 + EOT_SILENCE_MS + 2 * STEP_MS} ms of audio; it"
        f" fired at {trigger_ms}"
    )
    assert wall_ms < EOT_SILENCE_MS, (
        f"the wall clock moved {wall_ms:.1f} ms while {trigger_ms - 800} ms of audio went by: the "
        "two are not interchangeable, which is the point of this row"
    )


# --- criterion 3(e): silence alone is not a turn ----------------------------------------------


def test_silence_with_no_words_never_endpoints():
    """2.8 s of high pause heads and zero words produce no decision at all.

    The twin of T-0007a's 2.8 s no-marker row: room noise, a fan, a door -- the STT stream steps
    on regardless, and an endpointer that treats "the heads are high" as an end of turn opens a
    turn with an empty transcript and sends it to the LLM.
    """
    fired = drive(stream([], until_ms=2800, head_of=lambda t_ms: HIGH))
    assert fired == [], f"criterion 3(e): no words, no decisions; got {fired}"


# --- criterion 3(d) ----------------------------------------------------------------------------


def test_a_second_end_of_turn_needs_an_intervening_onset():
    """After one end of turn, more silence changes nothing until somebody speaks again.

    Two runs, hand-computed: five 200 ms words from 1000 ms is [1000, 2000]; two from 5000 ms is
    [5000, 5400]. The stretch between them is 3 s of silence -- more than three backstops -- and
    must yield exactly nothing.
    """
    first = spoken("what is on this screen", 1000)
    second = spoken("and now", 5000)
    assert (first[-1].end_ms, second[0].start_ms, second[-1].end_ms) == (2000, 5000, 5400)
    fired = drive(stream(first + second, until_ms=9000))

    assert [(decision[0], decision[1]) for _, decision in fired] == [
        ("onset", 1000),
        ("end_of_turn", 2000),
        ("onset", 5000),
        ("end_of_turn", 5400),
    ], f"criterion 3(d): one end of turn per run, each needing its own onset; got {fired}"


# --- criterion 3(c), adversarial: a server that sends no usable heads --------------------------


def test_empty_pause_heads_fall_through_to_the_backstop():
    """``prs=[]`` (or a vector shorter than ``PAUSE_HEAD_INDEX``) must not raise and must not cut.

    Kyutai's ``Step`` frame carries whatever the model was built with; T-0032 criterion 12 has not
    read the real vector yet, so the index is documented as unverified. Until it is, a missing
    head is a *missing* signal -- the backstop decides, at its own time -- and never an
    ``IndexError`` in the reflex lane, nor a 0.0 that silently means "still speaking" for ever.
    """
    words = spoken("hello there", 400)
    short = (HIGH,) * PAUSE_HEAD_INDEX  # one entry too few, by construction

    def head_of(t_ms: int) -> tuple[float, ...]:
        return () if (t_ms // STEP_MS) % 2 else short

    fired = drive(stream(words, until_ms=800 + 3 * EOT_SILENCE_MS, head_of=head_of))
    assert kinds(fired) == ["onset", "end_of_turn"], (
        f"an unusable pause-head vector must be survivable, and the backstop still fires; {fired}"
    )
    trigger_ms = fired[1][0].t_ms
    assert trigger_ms >= 800 + EOT_SILENCE_MS, (
        "…and it must be the *backstop* that fired, not an empty vector read as a high head: it "
        f"fired at {trigger_ms} ms of audio, before the last word's end (800) + {EOT_SILENCE_MS}"
    )


# --- criterion 3(b), adversarial: the false cut X2/Q5 measures ---------------------------------


def test_the_ema_needs_more_than_one_step_above_threshold():
    """One spurious 0.99 head between low ones must not cut the user off mid-sentence.

    The failure mode this task can create (docs/02 §10 X2/Q5: false-cut rate). The window is kept
    strictly shorter than ``EOT_SILENCE_MS`` so the backstop cannot rescue the row, and the same
    window with a *sustained* high head is asserted to fire -- otherwise an endpointer that never
    endpoints would pass this test.
    """
    require()
    budget = (EOT_SILENCE_MS // STEP_MS) - 1
    assert budget >= 4, (
        f"EOT_SILENCE_MS={EOT_SILENCE_MS} leaves only {budget} steps before the backstop; this "
        "row needs at least four to tell a spike from a run of high heads (say so to the reviewer "
        "rather than weakening it)"
    )
    words = spoken("hello there", 400)

    def spiky(t_ms: int) -> float:
        if t_ms < 800:
            return LOW
        return 0.99 if ((t_ms - 800) // STEP_MS) % 3 == 1 else 0.1

    spikes = stream(words, until_ms=800 + budget * STEP_MS, head_of=spiky)
    assert kinds(drive(spikes)) == ["onset"], (
        "criterion 3(b): a single 0.99 head between 0.1 heads is noise, not an end of turn; "
        f"decisions were {drive(spikes)}"
    )

    sustained = stream(
        words, until_ms=800 + budget * STEP_MS, head_of=lambda t_ms: LOW if t_ms < 800 else 0.99
    )
    assert kinds(drive(sustained)) == ["onset", "end_of_turn"], (
        "…while a sustained high head in the same window *does* end the turn (the control that "
        f"keeps the assertion above from passing vacuously); decisions were {drive(sustained)}"
    )


# --- the reflex lane is pure (.claude/rules/conductor.md) --------------------------------------


def test_module_imports_no_clock():
    """No ``asyncio``, no ``time``, no ``datetime`` in ``reflex/turn_predictor.py``.

    Time arrives in the event (docs/02 §4). A module that can read a clock will eventually read
    one, and the backstop row above would then pass in the fast suite and fail on the box, where
    the steps arrive in real time.
    """
    require()
    banned = {"asyncio", "time", "datetime", "threading"}
    tree = ast.parse(MODULE.read_text())
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name.split(".")[0] for a in node.names if a.name.split(".")[0] in banned]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in banned:
            found.append(node.module or "")
    assert not found, (
        f"the reflex lane is pure here (.claude/rules/conductor.md): {MODULE.name} imports {found}"
    )
    source = MODULE.read_text()
    for spelling in ("monotonic", "perf_counter", "now(", "get_event_loop"):
        assert spelling not in source, f"{MODULE.name} reaches for a clock: {spelling!r}"


# --- property: the machine's "one end of turn per run" invariant, over random streams -----------


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=1),  # 0 = word, 1 = step
            st.integers(min_value=1, max_value=12),  # gap, in 80 ms steps
            st.floats(min_value=0.0, max_value=1.0),  # pause head
        ),
        min_size=1,
        max_size=60,
    )
)
def test_end_of_turns_never_outnumber_onsets_on_any_stream(rows):
    """Criterion 3(d) as a property: over any interleaving, every end of turn follows an onset.

    n = 60 Hypothesis examples, seeded by Hypothesis's own default database; the stream is built
    with non-decreasing audio times only (the wire is ordered), which is the one assumption the
    endpointer may make.
    """
    require()
    ep = Endpointer()
    t_ms = 0
    onsets = eots = 0
    for kind, gap, head in rows:
        t_ms += gap * STEP_MS
        event: SttEvent = (
            SttWordEvent(text="w", start_ms=max(0, t_ms - 200), end_ms=t_ms)
            if kind == 0
            else SttStepEvent(pause_heads=heads(head), t_ms=t_ms)
        )
        decision = ep.observe(event)
        if decision is None:
            continue
        if decision[0] == "onset":
            onsets += 1
        else:
            eots += 1
        assert eots <= onsets, (
            f"criterion 3(d): an end of turn with no onset before it, at audio {t_ms} ms "
            f"({onsets} onsets, {eots} end-of-turns so far)"
        )


# --- ruling A7 (review A, F1): the decision must survive the seam to the machine ---------------

#: A deliberately large, constant offset between the session clock and the audio clock. Nothing
#: below reads it back; it exists so a delivery stamp can never be mistaken for an audio time.
CLOCK_OFFSET_MS = 1_000_000


def _through_the_machine(
    decisions: list[tuple], *, policy: TurnPolicy | None = None
) -> tuple[object, list[C.Command]]:
    """Submit endpointer decisions to the pure machine, exactly as the session does.

    ``("onset", ms)`` becomes ``E.UserSpeechOnset(t_ms=<clock>, onset_ms=ms)`` and
    ``("end_of_turn", ms, p)`` becomes ``E.SttEndOfTurn(t_ms=<clock>, t_eot_ms=ms, p_end=p)``
    (criterion 4). The ``p_end`` carried is the decision's own — that is the whole point of the
    row: the endpointer is authoritative and the machine's ``eot_p_threshold`` is a second gate
    only on the head path (ruling A7).
    """
    machine = TurnMachine(policy or TurnPolicy())
    state = machine.initial()
    seen: list[C.Command] = []
    for decision in decisions:
        if decision[0] == "onset":
            event: E.Event = E.UserSpeechOnset(
                t_ms=CLOCK_OFFSET_MS + decision[1], onset_ms=decision[1]
            )
        else:
            event = E.SttEndOfTurn(
                t_ms=CLOCK_OFFSET_MS + decision[1], t_eot_ms=decision[1], p_end=decision[2]
            )
        state, cmds = machine.step(state, event)
        seen.extend(cmds)
    return state, seen


@pytest.mark.parametrize(
    "label, prs",
    [
        ("prs=[] (the server sends no heads at all)", ()),
        ("prs shorter than PAUSE_HEAD_INDEX", "short"),
    ],
)
def test_a_backstop_decision_reaches_endpointing_through_the_machine(label, prs):
    """**The crossing test owed by ruling A7 (review A finding F1).**

    The endpointer and the machine are each correct on their own and the pair is inert: the
    machine drops ``stt.end_of_turn`` with ``p_end < policy.eot_p_threshold`` (0.6), and on the
    *silence backstop* the EMA has by construction never moved (an empty or too-short ``prs``
    vector is a missing signal, criterion 3(c)), so a decision carrying the EMA carries 0.0. The
    result on the box is no ``FreezeAfter``, no ``mark()``, and LISTENING for ever — group B's
    most likely failure, invisible to every test that stops at one side of the seam.

    Ruling A7: a **backstop** decision carries ``p_end = 1.0`` (the ``SttEndOfTurnEvent`` default
    exists for this); a **head** decision carries the EMA. Nothing here asserts how the endpointer
    spells that internally — only that the decision it hands the session, submitted verbatim,
    reaches ENDPOINTING with a ``FreezeAfter`` at the last word's ``end_ms``.

    Regime: pure; default ``TurnPolicy`` (``eot_p_threshold`` 0.6), no session, no clock. Words
    "what is on this screen", five 200 ms words from 1000 ms, run [1000, 2000] by hand.
    """
    require()
    assert TurnPolicy().eot_p_threshold == 0.6, (
        "this row is about the machine's 0.6 gate (docs/01 §8); the default moved"
    )
    vector = (HIGH,) * PAUSE_HEAD_INDEX if prs == "short" else ()
    assert len(vector) <= PAUSE_HEAD_INDEX, "the vector must be unusable, by construction"

    words = spoken("what is on this screen", 1000)
    fired = drive(stream(words, until_ms=2000 + 3 * EOT_SILENCE_MS, head_of=lambda t_ms: vector))
    assert kinds(fired) == ["onset", "end_of_turn"], (
        f"criterion 3(c) with {label}: the backstop still endpoints; got {fired}"
    )
    decision = fired[1][1]
    assert decision[1] == 2000, f"t_eot_ms is the last word's end_ms; got {decision}"
    trigger_ms = fired[1][0].t_ms
    assert trigger_ms >= 2000 + EOT_SILENCE_MS, (
        f"…and it is the backstop that fired, at {trigger_ms} ms of audio"
    )

    state, cmds = _through_the_machine([d for _, d in fired])
    assert state.state is TurnState.ENDPOINTING, (
        f"ruling A7 with {label}: the backstop decision (p_end={decision[2]}) was dropped by the "
        f"machine's eot_p_threshold={TurnPolicy().eot_p_threshold} gate, so the session stays in "
        f"{state.state.value} for ever — no FreezeAfter, no mark(), no turn. A backstop decision "
        "must carry p_end = 1.0."
    )
    freezes = [c for c in cmds if isinstance(c, C.FreezeAfter)]
    assert [c.t_eot_ms for c in freezes] == [2000], (
        f"exactly one FreezeAfter at the last word's end_ms (2000); got {freezes}"
    )


# --- ruling A8 (review A, F2): MIN_SPEECH_MS gates the head path only --------------------------


def test_a_short_utterance_still_ends_on_the_silence_backstop():
    """ "Yes." — one short word, then silence — must end the turn (ruling A8).

    ``MIN_SPEECH_MS`` exists so a cough recognised as one 40 ms word does not open a turn on the
    *head* path, where a burst of high pause heads can cross the threshold within four steps of a
    word arriving. Applied to the **silence backstop** as well it does something else entirely:
    it makes every utterance shorter than ``MIN_SPEECH_MS`` unendable, so a one-word answer hangs
    in LISTENING until the user says something longer. Criterion 3(c) as written is unconditional
    — ``EOT_SILENCE_MS`` of step-clock time after the last word ends the run, whatever else is
    true.

    Hand-computed: one word "yes" over [400, 480] (80 ms, deliberately under ``MIN_SPEECH_MS``);
    the heads are ``LOW`` throughout so rule 3(b) cannot fire and only the backstop can; the steps
    run to 480 + ``EOT_SILENCE_MS`` + 160 ms, i.e. two steps past the backstop's own time.

    Regime: pure, n = 1 deterministic, audio time only.
    """
    require()
    word = SttWordEvent(text="yes", start_ms=400, end_ms=480)
    assert word.end_ms - word.start_ms < MIN_SPEECH_MS, (
        f"the row needs a word shorter than MIN_SPEECH_MS={MIN_SPEECH_MS}; it is "
        f"{word.end_ms - word.start_ms} ms — pick a shorter one rather than weakening the row"
    )
    fired = drive(
        stream(
            [word],
            until_ms=480 + EOT_SILENCE_MS + 2 * STEP_MS,
            head_of=lambda t_ms: LOW,
        )
    )
    assert kinds(fired) == ["onset", "end_of_turn"], (
        f"ruling A8: MIN_SPEECH_MS={MIN_SPEECH_MS} must gate the head path only; the silence "
        f"backstop is unconditional, so a single 80 ms word followed by {EOT_SILENCE_MS} ms of "
        f"silence ends the turn. Decisions were {fired}"
    )
    kind, t_eot_ms, _ = fired[1][1]
    assert (kind, t_eot_ms) == ("end_of_turn", 480), (
        f"t_eot_ms is the last word's end_ms (480); got {fired[1][1]}"
    )
    trigger_ms = fired[1][0].t_ms
    assert 480 + EOT_SILENCE_MS <= trigger_ms <= 480 + EOT_SILENCE_MS + 2 * STEP_MS, (
        f"and it fires at the backstop's own time, not earlier; it fired at {trigger_ms}"
    )


# --- ruling A9 (review A, F3): the EMA reset at onset is load-bearing --------------------------


def test_the_ema_reset_at_onset_protects_the_second_run():
    """The EMA must be reset when a run opens, or the *second* run is cut off on its first step.

    Why this row exists (ruling A9): deleting ``self._ema = 0.0`` from the onset path leaves every
    other row in this module green — a single-run test never sees the carried value, and after an
    end of turn the run is closed, so the steps in between do not move the EMA. The carried value
    is precisely the one that *crossed*, so the next run starts one step away from ending.

    Hand-computed from ``EMA_HALF_LIFE_STEPS = 3`` ⇒ α = 1 − 2**(−1/3) = 0.206299…, and
    ``e_n = h·(1 − (1−α)**n)`` for a constant head ``h``:

      run 1 — "hello there" over [1000, 1400], then steps at 1480…1800 with head 0.95:
              e₄ = 0.95·(1 − 2**(−4/3)) = 0.5730 (below 0.6, no decision)
              e₅ = 0.95·(1 − 2**(−5/3)) = 0.6508 → crosses, end of turn at 1400, p_end 0.6508.
      run 2 — "and now" over [5000, 5400] (run length 400 ms > MIN_SPEECH_MS), then one step at
              5480 with head 0.55 and low heads after it, all before the backstop:
              **with** the reset: e = α·0.55 = 0.1135 — nothing fires;
              **without** it:     e = 0.6508 + α·(0.55 − 0.6508) = 0.6300 ≥ 0.6 — the user is cut
              off mid-sentence, on the first step of their second utterance.

    Regime: pure arithmetic, n = 1 deterministic, audio time only; the 0.6508/0.1135/0.6300
    figures above are computed by hand from α, never read back from the module.
    """
    require()
    run1 = [
        SttWordEvent(text="hello", start_ms=1000, end_ms=1200),
        SttWordEvent(text="there", start_ms=1200, end_ms=1400),
    ]
    run2 = [
        SttWordEvent(text="and", start_ms=5000, end_ms=5200),
        SttWordEvent(text="now", start_ms=5200, end_ms=5400),
    ]
    assert run1[-1].end_ms - run1[0].start_ms >= MIN_SPEECH_MS
    assert run2[-1].end_ms - run2[0].start_ms >= MIN_SPEECH_MS

    events: list[SttEvent] = [*run1]
    events += [SttStepEvent(pause_heads=heads(0.95), t_ms=1400 + 80 * i) for i in range(1, 6)]
    # The 3.6 s between the two runs is *not* a hole in the wire: moshi-server emits a step every
    # 80 ms of audio whether or not anybody is speaking (docs/02 §2.2), and the desk session of
    # 2026-09-16 has no gap in the step stream in 787 s. The steps are written out here (T-0051)
    # so this row constrains the EMA reset and nothing else; they carry LOW heads and land while
    # the run is closed, where every rule of the module ignores them, so both branches of the
    # hand computation in the docstring are untouched. What happens to the *backstop* when the
    # step stream really does stop is its own row,
    # ``test_a_hole_in_the_step_stream_still_cannot_cut_the_user_off_early``. The last step before
    # run 2's words is 5400 = their own ``end_ms``: on the wire a word is published *after* the
    # audio it ends in has been consumed, never before it (desk log, seq 3069 → 3073).
    events += [SttStepEvent(pause_heads=heads(LOW), t_ms=t) for t in range(1880, 5401, STEP_MS)]
    events += [*run2]
    events.append(SttStepEvent(pause_heads=heads(0.55), t_ms=5480))
    # …and low heads afterwards, stopping one step short of the backstop so nothing but the head
    # path can fire in run 2.
    last_step = 5400 + EOT_SILENCE_MS - STEP_MS
    events += [
        SttStepEvent(pause_heads=heads(LOW), t_ms=t) for t in range(5560, last_step + 1, STEP_MS)
    ]
    assert events[-1].t_ms - 5400 < EOT_SILENCE_MS, "the backstop must not rescue run 2"

    ep = Endpointer()
    fired = [(ev, d) for ev in events if (d := ep.observe(ev)) is not None]

    assert [(d[0], d[1]) for _, d in fired] == [
        ("onset", 1000),
        ("end_of_turn", 1400),
        ("onset", 5000),
    ], (
        "ruling A9: run 2 must survive its first moderate head. With the EMA reset at onset the "
        "EMA there is 0.1135; without it the run-1 crossing value 0.6508 is carried across and "
        "the 0.55 head lands at 0.6300, ending the turn on the first step of the second "
        f"utterance. Decisions were {fired}"
    )
    assert fired[1][1][2] == pytest.approx(0.6508, abs=5e-4), (
        "run 1's p_end is the crossing EMA, 0.95·(1 − 2**(−5/3)) = 0.6508 by hand from "
        f"EMA_HALF_LIFE_STEPS = 3; got {fired[1][1][2]}"
    )


# --- ruling A9 (review A, F4/F5): the constants are pinned to their bands ----------------------


def test_the_constants_sit_in_their_bands():
    """Each constant is pinned to the band its stated reason implies (ruling A9).

    Not a tautology: every bound below comes from a written source outside the module — the
    census (the real ``prs`` vector is a 4-vector), ruling A4 (the EMA needs four steps of slack),
    docs/01 §8 (the 0.6 threshold), and the B3 end-of-turn column (docs/03 §7). A future edit that
    moves one of these silently — to make a test pass, or to "tune" it without a bench run — is
    what this row catches (CLAUDE.md: a threshold is not raised without a docs/tasks entry).
    """
    require()
    assert 400 <= EOT_SILENCE_MS <= 1200, (
        f"EOT_SILENCE_MS={EOT_SILENCE_MS}: ruling A4 keeps it ≥ 400 ms so the EMA has at least "
        "four 80 ms steps of slack to tell a spike from a run of high heads, and ≤ 1200 ms "
        "because the backstop is the end-of-turn delay the user feels (docs/03 §7 B3's "
        "end-of-turn column); ≈ 800 is the chosen default"
    )
    assert 0 <= PAUSE_HEAD_INDEX < 4, (
        f"PAUSE_HEAD_INDEX={PAUSE_HEAD_INDEX}: the live server's prs is a **4-vector** (T-0007a "
        "census A12, e.g. [1.3e-05, 1.4e-04, 2.8e-04, 5.8e-04] at speech), so an index ≥ 4 is "
        "never usable on the box and the endpointer would silently run on the backstop alone; "
        "T-0032 criterion 12 reads the vector at silence and replaces the guess"
    )
    assert 0 < MIN_SPEECH_MS <= 500, (
        f"MIN_SPEECH_MS={MIN_SPEECH_MS}: it must exclude a cough recognised as one 40 ms word "
        "(so > 0) and must not exclude a real one-word answer on the head path (so ≤ 500 ms — "
        'a spoken "yes, please" is about that long); ruling A8 keeps it off the backstop entirely'
    )
    assert EOT_P_THRESHOLD == 0.6, (
        f"EOT_P_THRESHOLD={EOT_P_THRESHOLD}: 0.6 is docs/02 §2.2's Unmute-inherited rule and the "
        "machine's TurnPolicy.eot_p_threshold default (docs/01 §8); the two are one number and "
        "moving either without the other makes the head path's second gate meaningless"
    )
    assert TurnPolicy().eot_p_threshold == EOT_P_THRESHOLD, (
        "the endpointer's threshold and the machine's must be the same number (ruling A7: the "
        "machine's gate is a second gate on the head path, not a different rule)"
    )


# --- ruling A16 F3: MIN_SPEECH_MS pins the *head* path ----------------------------------------


def test_min_speech_ms_makes_the_head_path_wait_for_the_backstop():
    """A cough is not a turn: high heads right after one 80 ms word must not end it early.

    The twin of ``test_a_short_utterance_still_ends_on_the_silence_backstop`` and the row that
    pins the gate itself (ruling A16 F3). There the heads are ``LOW``, so only the backstop can
    fire and deleting ``last - first >= MIN_SPEECH_MS`` from the head path changes nothing. Here
    the heads are **0.95 from the moment the word lands**, so the EMA crosses long before the
    backstop, and the two implementations part company:

      with the gate    — nothing fires on the heads; the run ends at the backstop, at
                         480 + ``EOT_SILENCE_MS`` of step time, carrying ``p_end = 1.0``;
      without it       — the run ends on the heads at 800 ms of audio, carrying ``p_end ≈ 0.65``,
                         i.e. the machine opens a turn on a cough (X2/Q5's false-cut failure).

    Hand-computed from ``EMA_HALF_LIFE_STEPS = 3`` ⇒ α = 1 − 2**(−1/3) = 0.206299…, and
    ``e_n = h·(1 − (1−α)**n)`` for the constant head h = 0.95. The word "ahem" spans [400, 480]
    (80 ms, under ``MIN_SPEECH_MS``), so the first step of the run is the one at 480 ms and

      e₁ = 0.1862, e₂ = 0.3341, e₃ = 0.4515, e₄ = 0.5447, e₅ = 0.6187 ≥ 0.6

    — the fifth step, at 480 + 4·80 = **800 ms** of audio, 320 ms after the word and well inside
    the 480 + 800 = 1280 ms backstop (asserted below, so the row cannot go vacuous if a constant
    moves). The ``p_end`` assertion is the second half: the backstop reports 1.0 (ruling A7),
    the head path reports the crossing EMA, so "which path fired" is observable and not merely
    "when".

    Regime: pure arithmetic, audio time only, n = 1 deterministic; every number above computed by
    hand from α and the word times in this row, never read back from the module.
    """
    require()
    word = SttWordEvent(text="ahem", start_ms=400, end_ms=480)
    run_ms = word.end_ms - word.start_ms
    assert run_ms < MIN_SPEECH_MS, (
        f"this row needs a run shorter than MIN_SPEECH_MS={MIN_SPEECH_MS}; it is {run_ms} ms — "
        "shorten the word rather than weakening the row"
    )
    ema_crossing_ms = 800  # hand-computed above, from α and h = 0.95
    assert ema_crossing_ms < word.end_ms + EOT_SILENCE_MS, (
        f"the row is only adversarial while the EMA would cross ({ema_crossing_ms} ms) *before* "
        f"the backstop ({word.end_ms + EOT_SILENCE_MS} ms); recompute it if EMA_HALF_LIFE_STEPS "
        "or EOT_SILENCE_MS moved"
    )

    fired = drive(
        stream(
            [word],
            until_ms=word.end_ms + EOT_SILENCE_MS + 2 * STEP_MS,
            head_of=lambda t_ms: 0.95,
        )
    )

    assert kinds(fired) == ["onset", "end_of_turn"], f"one onset, one end of turn; got {fired}"
    trigger_ms = fired[1][0].t_ms
    assert trigger_ms >= word.end_ms + EOT_SILENCE_MS, (
        f"ruling A16 F3: MIN_SPEECH_MS={MIN_SPEECH_MS} gates the head path, so a {run_ms} ms run "
        f"cannot end on the heads; the EMA crosses 0.6 at {ema_crossing_ms} ms of audio and the "
        f"decision must wait for the backstop at {word.end_ms + EOT_SILENCE_MS} ms. It fired at "
        f"{trigger_ms} ms — a cough opened a turn"
    )
    kind, t_eot_ms, p_end = fired[1][1]
    assert (kind, t_eot_ms) == ("end_of_turn", word.end_ms), (
        f"t_eot_ms is the last word's end_ms ({word.end_ms}); got {fired[1][1]}"
    )
    assert p_end == 1.0, (
        "and it is the *backstop* that fired, which reports p_end = 1.0 (ruling A7); p_end = "
        f"{p_end} says the head path decided this turn after all"
    )


# ==================================================================================================
# T-0051 group A — the endpointer's clock (docs/tasks/T-0051.md, criteria 1–7)
#
# The defect, in one line: the backstop subtracts two clocks. ``SttStepEvent.t_ms`` is the
# *fed-audio* clock ("now"), a word's ``end_ms`` is *when it was spoken*, and their difference
# always contains the recogniser's emission delay -- ≈ 510 ms on the GPU on 2026-09-16, seconds on
# the CPU. So ``EOT_SILENCE_MS = 800`` ends the turn after ≈ 290 ms of audio the recogniser has
# actually examined, and a speaker who pauses mid-sentence is cut off (ruling A3).
# ==================================================================================================


def new_symbol(name: str) -> object:
    """A constant T-0051 adds, or a named assertion failure saying which criterion owes it.

    ``getattr`` rather than a module-level ``from ... import``: a pending row that dies with an
    ``ImportError`` is a bug in the test (``tests/conftest.py`` tolerates only
    ``AssertionError``/``NotImplementedError`` while the marker is on).
    """
    value = getattr(_tp, name, None)
    assert value is not None, (
        f"T-0051 criterion 5/6: local_astra/reflex/turn_predictor.py must export {name}, pinned "
        "to a band with a written source (docs/tasks/T-0051.md, 'Any new constant is pinned to a "
        "band with a written source in the constants row')"
    )
    return value


def decisions_of(fired: list[tuple[SttEvent, tuple]]) -> list[tuple]:
    return [decision for _, decision in fired]


def trigger_ms_of(event: SttEvent) -> int:
    """The step-clock time at which a decision fired, whatever event carried it."""
    return event.t_ms if isinstance(event, SttStepEvent) else event.end_ms


# --- criterion 1: tonight's failure, reproduced ------------------------------------------------


def test_a_mid_sentence_pause_does_not_close_the_turn_under_a_lagging_recogniser():
    """**The headline row (T-0051 criterion 1, adversarial).** What happened at the desk.

    The human asked "what do you see?" and the model answered "what do you": the backstop ended
    the turn 290 ms of *examined* audio after the last word, in the middle of a breath. Here the
    same stream with hand-chosen times:

      words   ``[1000,1200] [1200,1400] [1400,1600]``  — three 200 ms words
      pause   **600 ms** of silence, [1600, 2200]
      words   ``[2200,2360] [2360,2520]``              — two 160 ms words
      lag     ``L = 500`` ms, tonight's measured delivery delay (T-0051 *Regime*)
      heads   0.30 on every step, so the EMA converges to 0.30 and **never** reaches
              ``EOT_P_THRESHOLD`` = 0.6: only the backstop can fire, which is the point.

    Arrivals, hand-computed by :func:`arrival_of` (first step on the 80 ms grid at or after
    ``end_ms + L``): 1760, 1920, **2160**, **2880**, 3040. The silence the recogniser has examined
    between the third word and the fourth is therefore 2880 − 2160 = **720 ms**, one step inside
    ``EOT_SILENCE_MS`` = 800 — asserted below so the row cannot go vacuous if a constant moves.

    Note the arithmetic that band hides, and that the reviewer should see: measured from
    *arrival*, the silence budget is consumed by the pause **plus the next word's own duration**
    (a word is only published once it has ended). A 600 ms pause before a 200 ms word is exactly
    800 ms of arrival-silence — the boundary — which is why this row's fourth word is 160 ms. The
    fix buys the speaker ``L`` ms; it does not make the backstop indifferent to word length.

    Against today's module this stream fires ``("end_of_turn", 1600, 1.0)`` at step 2400, 800 ms
    of *audio* clock after a word the recogniser only delivered at 2160 — i.e. 240 ms of examined
    silence — and then re-opens a second run at the word "about". The assertion message quotes
    whatever fired.

    Regime: pure arithmetic, audio/step clock only, n = 1 deterministic, no clock read.
    """
    require()
    lag_ms = 500
    words = [
        *spoken("what is this", 1000, 200),
        SttWordEvent(text="about", start_ms=2200, end_ms=2360),
        SttWordEvent(text="here", start_ms=2360, end_ms=2520),
    ]
    assert words[2].end_ms == 1600 and words[3].start_ms == 2200, "the pause is 600 ms by hand"
    before, after = (
        arrival_of(words[2], delivery_lag_ms=lag_ms),
        arrival_of(words[3], delivery_lag_ms=lag_ms),
    )
    assert (before, after) == (2160, 2880), f"arrivals recomputed: {before}, {after}"
    assert after - before < EOT_SILENCE_MS, (
        f"the row is only meaningful while the pause as the *recogniser* sees it ({after - before}"
        f" ms) is inside EOT_SILENCE_MS={EOT_SILENCE_MS}; shorten the fourth word or say so to the"
        " reviewer rather than weakening the assertion"
    )
    last_arrival = arrival_of(words[-1], delivery_lag_ms=lag_ms)

    fired = drive(
        stream(
            words,
            until_ms=last_arrival + 3 * EOT_SILENCE_MS,
            head_of=lambda t_ms: 0.30,
            delivery_lag_ms=lag_ms,
        )
    )
    early = [
        (trigger_ms_of(ev), d)
        for ev, d in fired
        if d[0] == "end_of_turn" and trigger_ms_of(ev) < last_arrival
    ]
    assert not early, (
        f"criterion 1: the speaker paused {words[3].start_ms - words[2].end_ms} ms mid-sentence "
        f"and was cut off. The endpointer fired {early} before the last word of the sentence had "
        f"even arrived (arrival {last_arrival} ms of step clock). {EOT_SILENCE_MS} ms of silence "
        f"measured from the word's *audio* end is {EOT_SILENCE_MS - lag_ms} ms of audio the "
        "recogniser has examined; measured from its arrival it is the whole 800 (ruling A3). All "
        f"decisions: {decisions_of(fired)}"
    )
    onsets = [d for d in decisions_of(fired) if d[0] == "onset"]
    assert onsets == [("onset", 1000)], (
        "criterion 1: one sentence is one run — exactly one onset, at the first word's start_ms "
        f"(1000). Got {onsets}; a second onset means the pause split the sentence in two turns"
    )


def test_the_accepted_limit_a_600_ms_pause_before_a_200_ms_word_is_still_cut():
    """**Ruling A7's accepted limitation, as a test rather than as prose** (ruling A23).

    The headline row above buys the speaker the recogniser's delay, and its docstring says what
    that does *not* buy. Measured from arrival, the silence budget is spent on the pause **plus
    the following word's own duration**, because a word is published only once it has ended. So a
    600 ms pause followed by a 200 ms word is exactly ``EOT_SILENCE_MS`` = 800 ms of
    arrival-silence — the boundary — and the fix does not rescue it. Ruling A7 accepted that,
    with a reason: the backstop is a fallback, not the normal decider; on the desk stream the
    pause-head EMA crossed ≈ 994 ms after the last word while the corrected backstop would have
    fired at ≈ 1310, so on any server that sends usable heads the heads decide first, and the
    backstop exists for one that does not.

    Accepted-with-a-reason is not the same as unpinned. Written down only in a ruling, the
    boundary can be moved by a later "improvement" — a cap on the arrival stamp, a widened budget,
    a fabricated end for an unfinished word — with nothing going red and nobody noticing that the
    published ``EOT_SILENCE_MS`` now means something else. This row pins **both sides** of it, so
    a change in either direction has to be deliberate and carry a ``docs/tasks/`` entry
    (CLAUDE.md: thresholds are tests).

    Hand-computed, ``L`` = 500 and heads 0.30 throughout so the EMA settles at 0.30 and only the
    backstop can fire. Three 200 ms words over ``[1000,1600]``, the third arriving at
    ``ceil80(1600 + 500)`` = 2160, so the backstop is due at 2960:

      cut    a 200 ms fourth word over ``[2200, 2400]`` arrives at ``ceil80(2900)`` = **2960** —
             the same step. A step precedes a word of the same stamp on the wire (see
             :func:`stream`), so the decision fires first, at ``t_eot`` 1600: the sentence is cut.
      kept   a 160 ms fourth word over ``[2200, 2360]`` arrives at ``ceil80(2860)`` = **2880**,
             80 ms of grid inside the budget, and extends the run instead.

    The arithmetic is asserted, not assumed. Note the quantisation ruling A23 asks group C to
    write down: the arrival stamp is the *step* at or after ``end_ms + L``, so the measured
    arrival-gap is the pause plus the word's duration only up to one 80 ms frame — exactly 800 in
    the cut case, 720 rather than 760 in the kept one. The backstop may therefore fire up to one
    frame early, and this row is one frame from its boundary on purpose (ruling A15).

    Regime: pure arithmetic, step clock only, n = 1 deterministic per case, ``L`` = 500 ms
    (measured on the GPU, 2026-09-16).
    """
    require()
    lag_ms = 500
    words = spoken("what is this", 1000, 200)
    assert words[-1].end_ms == 1600, "three 200 ms words, by hand"
    third_arrival = arrival_of(words[-1], delivery_lag_ms=lag_ms)
    assert third_arrival == 2160, f"recompute: the third word arrives at {third_arrival}"

    cut = SttWordEvent(text="here", start_ms=2200, end_ms=2400)
    kept = SttWordEvent(text="here", start_ms=2200, end_ms=2360)
    cut_arrival = arrival_of(cut, delivery_lag_ms=lag_ms)
    kept_arrival = arrival_of(kept, delivery_lag_ms=lag_ms)
    assert (cut_arrival, kept_arrival) == (2960, 2880), (
        f"recompute the arrivals: {cut_arrival}, {kept_arrival}"
    )
    assert cut.start_ms - words[-1].end_ms == 600 and kept.start_ms - words[-1].end_ms == 600, (
        "both halves share the same 600 ms pause; only the following word's length differs"
    )
    assert cut_arrival - third_arrival == EOT_SILENCE_MS, (
        f"the premise of the 'cut' half: {cut_arrival - third_arrival} ms of arrival-silence is "
        f"exactly EOT_SILENCE_MS={EOT_SILENCE_MS}, the boundary ruling A7 accepted. If a constant "
        "moved, recompute the word lengths rather than relaxing the assertion"
    )
    assert kept_arrival - third_arrival < EOT_SILENCE_MS, (
        f"the premise of the 'kept' half: {kept_arrival - third_arrival} ms is inside "
        f"EOT_SILENCE_MS={EOT_SILENCE_MS}"
    )

    def ends_before_or_at(word: SttWordEvent, arrival_ms: int) -> list[tuple[int, tuple]]:
        fired = drive(
            stream(
                [*words, word],
                until_ms=arrival_ms + 3 * EOT_SILENCE_MS,
                head_of=lambda t_ms: 0.30,
                delivery_lag_ms=lag_ms,
            )
        )
        return [
            (trigger_ms_of(ev), d)
            for ev, d in fired
            if d[0] == "end_of_turn" and trigger_ms_of(ev) <= arrival_ms
        ]

    cut_ends = ends_before_or_at(cut, cut_arrival)
    # 1.0 is what a backstop decision reports (ruling A7), as criterion 2's row also pins.
    assert cut_ends == [(2960, ("end_of_turn", 1600, 1.0))], (
        "ruling A7's accepted limit: 600 ms of pause plus a 200 ms word is 800 ms of "
        "arrival-silence, so the backstop fires at step 2960 with t_eot 1600 and the speaker is "
        "cut off one step before their next word is delivered. This row asserts the limitation on "
        "purpose. If it fails because nothing fired, the budget has been widened — which may be "
        "right, but it is a threshold change and it owes a docs/tasks entry and a bench, not a "
        f"quiet edit (T-0051 criterion 6, ruling A7). Got {cut_ends}"
    )
    kept_ends = ends_before_or_at(kept, kept_arrival)
    assert kept_ends == [], (
        "…and the other side of the same boundary: with a 160 ms word the arrival-silence is "
        f"{kept_arrival - third_arrival} ms, inside the budget, so the run must still be open when"
        f" the word is delivered at {kept_arrival}. Got {kept_ends}"
    )


# --- criterion 2: the backstop measures examined silence, in one clock -------------------------


def test_the_backstop_measures_silence_the_recogniser_has_actually_examined():
    """Criterion 2, for ``L ∈ {0, 240, 500, 1200}`` ms of delivery lag, both sides of the bound.

    One run, "hello there" = two 200 ms words over ``[1000, 1400]``, heads ``LOW`` throughout so
    the head path cannot fire and only the backstop can, then silence for 4 s of step clock.

    The rule (T-0051 *Implementation notes*): when a word arrives at step-clock time ``a`` the
    recogniser has by definition examined audio up to ``a − L``, so the audio it has examined
    *after* that word at a later step ``t`` is exactly ``t − a``. The decision must therefore land
    no earlier than ``last_word_end + L + EOT_SILENCE_MS`` — the negative half, "there is no L for
    which a decision arrives earlier" — and no later than two steps after it, the slack the 80 ms
    grid needs when ``end_ms + L`` falls between two steps. Hand-computed windows:

      L =    0 → [2200, 2360]     L =  500 → [2700, 2860]
      L =  240 → [2440, 2600]     L = 1200 → [3400, 3560]

    ``t_eot_ms`` stays **1400**, the last word's audio ``end_ms``, for every L: the machine's
    ``FreezeAfter`` and the ``mark()`` that follows are statements about *audio*, and an arrival
    time there would freeze away the end of the sentence (T-0051 *Implementation notes*).

    The four cases are one row on purpose: ``L = 0`` is the guard that the fix changes nothing for
    a zero-lag stream — it passes on today's module — so as a separate parametrisation it would
    XPASS strict while its task is pending and hide the other three.

    Against today's module: L = 240 and L = 500 both fire at step 2240 (before their windows),
    and L = 1200 fires **twice**, at 2480 and 2720, because each word arrives already "stale".

    Regime: pure arithmetic, step clock only, n = 1 deterministic per L.
    """
    require()
    words = spoken("hello there", 1000, 200)
    last = words[-1]
    assert (words[0].start_ms, last.end_ms) == (1000, 1400)
    violations: list[str] = []
    for lag_ms, (low, high) in (
        (0, (2200, 2360)),
        (240, (2440, 2600)),
        (500, (2700, 2860)),
        (1200, (3400, 3560)),
    ):
        assert (low, high) == (
            last.end_ms + lag_ms + EOT_SILENCE_MS,
            last.end_ms + lag_ms + EOT_SILENCE_MS + 2 * STEP_MS,
        ), f"the hand-computed window for L={lag_ms} no longer follows from EOT_SILENCE_MS"
        fired = drive(
            stream(
                words,
                until_ms=last.end_ms + lag_ms + 4000,
                head_of=lambda t_ms: LOW,
                delivery_lag_ms=lag_ms,
            )
        )
        ends = [(trigger_ms_of(ev), d) for ev, d in fired if d[0] == "end_of_turn"]
        if len(ends) != 1:
            violations.append(f"L={lag_ms}: expected exactly one end of turn, got {ends}")
            continue
        trigger_ms, (_, t_eot_ms, p_end) = ends[0]
        if trigger_ms < low:
            violations.append(
                f"L={lag_ms}: fired at {trigger_ms}, i.e. {trigger_ms - (last.end_ms + lag_ms)} ms"
                f" of examined silence — the user was cut off {low - trigger_ms} ms early"
            )
        elif trigger_ms > high:
            violations.append(f"L={lag_ms}: fired at {trigger_ms}, later than {high}")
        if t_eot_ms != last.end_ms:
            violations.append(
                f"L={lag_ms}: t_eot_ms is the last word's audio end_ms ({last.end_ms}), never its "
                f"arrival; got {t_eot_ms}"
            )
        if p_end != 1.0:
            violations.append(
                f"L={lag_ms}: LOW heads cannot cross {EOT_P_THRESHOLD}, so this is the backstop "
                f"and ruling A7 makes it report p_end = 1.0; got {p_end}"
            )
    assert not violations, (
        "criterion 2: the backstop must measure silence the recogniser has *examined*, in one "
        "clock (ruling A3):\n  " + "\n  ".join(violations)
    )


def test_a_hole_in_the_step_stream_still_cannot_cut_the_user_off_early():
    """Criterion 2's floor, on the stream shape that breaks the recommended rule (adversarial).

    "The step clock at which the last word arrived" is the recommended implementation
    (T-0051 *Implementation notes*) and it is exact **while the steps keep coming** — the server
    emits one every 80 ms of audio whether or not anybody speaks (docs/02 §2.2), and the desk
    session of 2026-09-16 has no gap in 787 s. When the step stream *does* stop, the last step's
    ``t_ms`` is stale, and a word delivered against a stale reading is recorded as having arrived
    long ago: the backstop then fires ``EOT_SILENCE_MS`` after the **hole**, not after the word.

    Criterion 2's negative half rules that out without naming a mechanism: there is no delivery lag
    ``L >= 0`` for which a decision may arrive earlier than ``last_word_end + L + EOT_SILENCE_MS``,
    so ``end_ms + EOT_SILENCE_MS`` of step clock is a floor under every stream, holes included.

    Hand-computed: steps from 0 to 1800, then a 3.2 s hole, then a word over ``[5000, 5200]``,
    then steps again from 5480. The earliest admissible decision is 5200 + 0 + 800 = **6000**.

    This row passes on today's module — its rule, wrong as it is, is at least measured against the
    word — so it carries no ``pending`` marker: it is the guard that catches a T-0051
    implementation which reads the stale step clock and cuts the user off 520 ms early. The same
    shape is what T-0048 group A's feed gate would create on purpose, which is why that task must
    land after this one carrying a row that the backstop still fires with the gate on (ruling A4).

    Regime: pure arithmetic, step clock only, n = 1 deterministic.
    """
    require()
    word = SttWordEvent(text="and", start_ms=5000, end_ms=5200)
    events: list[SttEvent] = [
        SttStepEvent(pause_heads=heads(LOW), t_ms=t) for t in range(0, 1801, STEP_MS)
    ]
    events.append(word)
    events += [SttStepEvent(pause_heads=heads(LOW), t_ms=t) for t in range(5480, 9000, STEP_MS)]
    floor_ms = word.end_ms + EOT_SILENCE_MS
    assert floor_ms == 6000, "recompute the floor if EOT_SILENCE_MS moved"

    fired = drive(events)
    early = [
        (trigger_ms_of(ev), d)
        for ev, d in fired
        if d[0] == "end_of_turn" and trigger_ms_of(ev) < floor_ms
    ]
    assert not early, (
        f"criterion 2, negative half: the word ended at {word.end_ms} of audio, so no decision may"
        f" land before {floor_ms} whatever the delivery lag was; the step stream had a hole "
        f"between 1800 and 5480 and the decision landed at {early}. An arrival clock read from a "
        "stale step is not an arrival clock"
    )
    ends = [(trigger_ms_of(ev), d) for ev, d in fired if d[0] == "end_of_turn"]
    assert len(ends) == 1 and ends[0][1][1] == word.end_ms, (
        f"…and the turn must still end, once, at the word's end_ms ({word.end_ms}); got {ends}"
    )


# --- criterion 3: the desk stream of 2026-09-16 ------------------------------------------------

DESK_TURN1 = FIXTURES / "stt" / "desk-2026-09-16-turn1.json"


def desk_stream(offset_ms: int) -> tuple[list[SttEvent], dict]:
    """The recorded turn-1 stream, in the audio clock the endpointer saw.

    The fixture stores every step's *session* stamp, because that is what the session log records;
    the fed-audio clock runs behind it by a constant the session could only bound, 330–395 ms
    (T-0051 *Regime*). ``offset_ms`` is that constant. Nothing in the endpointer's rules can
    depend on it — it shifts every step by the same amount, and the words' audio times are not
    shifted at all — which is why this row replays the stream at both ends of the band and asserts
    the same decisions.
    """
    fixture = json.loads(DESK_TURN1.read_text())
    steps = [
        SttStepEvent(pause_heads=tuple(row["prs"]), t_ms=row["t_ms"] - offset_ms)
        for row in fixture["steps"]
    ]
    events: list[SttEvent] = []
    for index, step in enumerate(steps):
        events.append(step)
        for word in fixture["words"]:
            if word["arrival_after_step_index"] == index:
                events.append(
                    SttWordEvent(
                        text=word["text"], start_ms=word["start_ms"], end_ms=word["end_ms"]
                    )
                )
    return events, fixture


def test_the_desk_stream_of_2026_09_16_ends_the_turn_on_the_heads_not_the_backstop():
    """Criterion 3 — the row that pins tonight's evidence rather than a story about it.

    ``tests/fixtures/stt/desk-2026-09-16-turn1.json`` is the turn-1 slice of the human's own desk
    session: 29 real step vectors with their real four-entry ``prs``, the three word rows with
    their real times and the step after which each arrived, and (for the record only, asserted
    nowhere) the four decisions the session logged. The words are ``w1``/``w2``/``w3``: the
    evidence is the timings and the head vectors, not the sentence.

    What must come out, once the backstop measures examined silence:

    * **exactly one onset, at 242320** — ``w3`` (audio ``[242720, 242720]``, delivered 1673 ms
      later by the marker echo) is a late word of a closed run and opens nothing (criterion 4);
    * **exactly one end of turn**, where tonight there were two 560 ms apart, each with its own
      ``mark()`` and flush (~0.7 s of turn latency, T-0051 *Regime*);
    * its ``p_end`` is **strictly below 1.0**, i.e. ``BACKSTOP_P_END`` did not decide. The pause
      heads were working: index 2 rises 0.339 → 0.451 → 0.594 → 0.733 → 0.758 → 0.687 → 0.805 →
      0.938 over the steps after the onset, and the EMA (α = 1 − 2**(−1/3)) crosses 0.6 while the
      corrected backstop is still ≈ 340 ms away. That is the mitigation the *Implementation
      notes* promise for B3, measured on the real stream rather than assumed;
    * its trigger is **at or after** the arrival of the word whose ``end_ms`` it reports — the
      endpointer never decides an end of turn on audio the recogniser has not delivered yet.

    Today's module yields onset 242320, ``end_of_turn(242720, 1.0)``, onset 242720,
    ``end_of_turn(242720, 1.0)`` — the log's own four rows, seq 3070/3093/3113/3121.

    Regime: replay of one recorded session (n = 1, no seed, profile ``lean``, STT
    ``kyutai/stt-1b-en_fr`` on the GPU in moshi-server 0.6.4, ``asr_delay_in_tokens`` 6); pure
    arithmetic here, no clock, no service.
    """
    require()
    events, fixture = desk_stream(fixture_offset := fixture_offset_ms())
    band = fixture["clock"]["audio_clock_offset_band_ms"]
    assert band[0] <= fixture_offset <= band[1], (
        f"the fixture's audio-clock offset {fixture_offset} left its derived band {band}"
    )
    words = fixture["words"]
    assert [(w["start_ms"], w["end_ms"]) for w in words] == [
        (242320, 242560),
        (242560, 242720),
        (242720, 242720),
    ], f"the fixture's word times moved: {words}"

    fired = drive(events)
    onsets = [d for d in decisions_of(fired) if d[0] == "onset"]
    ends = [(trigger_ms_of(ev), d) for ev, d in fired if d[0] == "end_of_turn"]
    assert onsets == [("onset", 242320)], (
        "criterion 3: one sentence, one onset, at the first word's start_ms 242320. A second "
        f"onset is the oscillation of seq 3113 (LISTENING → ENDPOINTING → LISTENING). Got {onsets}"
    )
    assert len(ends) == 1, (
        "criterion 3: exactly one end of turn. Tonight there were two, at session 243914 and "
        f"244474, each with its own mark() and flush. Got {ends}"
    )
    trigger_ms, (_, t_eot_ms, p_end) = ends[0]
    assert t_eot_ms == 242720, f"t_eot_ms is the last word's audio end_ms (242720); got {t_eot_ms}"
    assert p_end < 1.0, (
        f"criterion 3: p_end = {p_end} is BACKSTOP_P_END — the backstop decided this turn on the "
        "real stream, which is the defect. The pause heads crossed on their own ≈ 340 ms before "
        "the corrected backstop could fire (T-0051 *Regime*, the EMA trace)"
    )
    assert p_end >= EOT_P_THRESHOLD, (
        f"a head decision reports the EMA that crossed {EOT_P_THRESHOLD} (ruling A7); got {p_end}"
    )
    covering = next(w for w in words if w["end_ms"] == t_eot_ms)
    arrival_ms = covering["arrival_t_ms"] - fixture_offset
    assert trigger_ms >= arrival_ms, (
        f"criterion 3: the decision reports audio up to {t_eot_ms} but fired at step {trigger_ms},"
        f" before the recogniser had delivered that word (arrival {arrival_ms} in the same clock)"
        " — an end of turn decided on audio nobody has looked at"
    )
    backstop_ms = arrival_ms + EOT_SILENCE_MS
    assert trigger_ms < backstop_ms, (
        f"…and it is the *head* path that decided: the corrected backstop for that word cannot "
        f"fire before {backstop_ms}, and this decision fired at {trigger_ms}"
    )

    other = [
        d
        for edge in fixture["clock"]["audio_clock_offset_band_ms"]
        for d in [decisions_of(drive(desk_stream(edge)[0]))]
    ]
    assert other[0] == other[1] == decisions_of(fired), (
        "the fixture's one free parameter — the 330–395 ms audio/session clock offset the session "
        "could only bound — must not change a single decision, since it shifts every step by the "
        f"same amount. At the band's edges the decisions were {other}"
    )


def fixture_offset_ms() -> int:
    return int(json.loads(DESK_TURN1.read_text())["clock"]["audio_clock_offset_ms"])


# --- criterion 4: a late word of a closed run never opens a new one ----------------------------


@pytest.mark.parametrize("late_start_delta", [0, -240], ids=["start == t_eot", "start < t_eot"])
def test_a_late_word_at_or_before_the_decided_t_eot_never_opens_a_run(late_start_delta):
    """Criterion 4 — the oscillation, as a rule: a word whose audio predates the decision is old.

    The live shape, twice in one session: ``end_of_turn 242720`` (seq 3093) then ``onset 242720``
    (seq 3113) 479 ms later, and ``end_of_turn 268880`` (seq 3889) then ``onset 268880`` (seq
    3909). Each re-opened run produced a second end of turn, a second ``mark()`` and a third
    ``stt.flushed`` for two real turns; the churn cost ≈ 700 ms of turn latency.

    Hand-computed: two 200 ms words over ``[1000, 1400]``, ``L = 500`` so the last one arrives at
    step 1920, ``LOW`` heads so the backstop decides at 1920 + 800 = **2720** with
    ``t_eot_ms = 1400``. Then a word ``[1400 + δ, 1400]`` arrives at step 2880 — δ = 0 is the
    log's own shape (a word whose whole span is at the decided end of turn), δ = −240 a word
    that started before it. Neither may produce **any** decision: not an onset, and not the second
    end of turn that follows one on today's module.

    Regime: pure arithmetic, step clock only, n = 1 deterministic per case.
    """
    require()
    lag_ms = 500
    words = spoken("hello there", 1000, 200)
    t_eot_expected = 1400
    late = SttWordEvent(
        text="late", start_ms=t_eot_expected + late_start_delta, end_ms=t_eot_expected
    )
    events = stream(
        words, until_ms=1400 + lag_ms + 3000, head_of=lambda t_ms: LOW, delivery_lag_ms=lag_ms
    )
    late_arrival = 2880
    at = next(
        i for i, e in enumerate(events) if isinstance(e, SttStepEvent) and e.t_ms == late_arrival
    )
    events.insert(at + 1, late)

    ep = Endpointer()
    # By *position* in the stream, not by the time a decision reports: an onset produced by the
    # late word carries the word's own (old) audio time, so a filter on times would not see it.
    fired = [(i, ev, d) for i, ev in enumerate(events) if (d := ep.observe(ev)) is not None]
    ends = [(trigger_ms_of(ev), d) for _, ev, d in fired if d[0] == "end_of_turn"]
    assert ends and ends[0][1][1] == t_eot_expected, (
        f"the row's premise: the run ends first, at t_eot {t_eot_expected}; got {ends}"
    )
    after = [(trigger_ms_of(ev), d) for i, ev, d in fired if i >= at + 1]
    assert not after, (
        f"criterion 4: a word ({late.start_ms} → {late.end_ms}) whose audio is at or before the "
        f"decided t_eot ({t_eot_expected}) arrived at step {late_arrival}, after the run closed. "
        f"It must produce no decision at all — it is the tail of the sentence the machine has "
        f"already frozen, not a new one. Got {after}; all decisions "
        f"{[d for _, _, d in fired]}"
    )
    assert len(ends) == 1, (
        f"…so the run ends exactly once: a second end of turn at the same t_eot is the log's seq "
        f"3093/3121 pair, and it costs a second mark() and a third flush. Got {ends}"
    )


# --- criterion 5: a stale or impossibly long word, and the negative twin ------------------------


def test_a_stale_or_impossibly_long_word_never_opens_a_run():
    """Criterion 5 — tonight's ``see?``: ``start_ms`` 247200, ``end_ms`` 268240, **21040 ms** long.

    That word was held inside ``MoshiSttClient`` waiting for an ``EndWord`` that never came and
    was finally closed by the *next* word's arrival 21 s later (ruling A2), so its start time is
    not when it was said and its duration is invented. It opened a turn (seq 3841/3844, log
    ``onset 247200``) 800 ms into the assistant's own reply.

    Two guards, one row: a word longer than ``MAX_WORD_MS`` opens nothing, and a word whose
    ``start_ms`` precedes the last decision's ``t_eot`` opens nothing. The second case reuses the
    log's second turn: ``end_of_turn`` at ``t_eot`` 268880 (seq 3889), then a word starting at
    268240 — earlier audio than the decision — must stay silent.

    Regime: pure arithmetic, step clock only, n = 1 deterministic per case; the two word rows are
    the log's own.
    """
    require()

    max_word_ms = new_symbol("MAX_WORD_MS")
    assert isinstance(max_word_ms, int)

    # (a) the impossible duration, from a cold endpointer: nothing but the word can be blamed.
    ep = Endpointer()
    long_word = SttWordEvent(text="see?", start_ms=247200, end_ms=268240)
    assert long_word.end_ms - long_word.start_ms == 21040, "tonight's word, verbatim"
    steps_before = [SttStepEvent(pause_heads=heads(LOW), t_ms=t) for t in (268000, 268080)]
    for step in steps_before:
        ep.observe(step)
    assert ep.observe(long_word) is None, (
        f"criterion 5: a word {long_word.end_ms - long_word.start_ms} ms long is not speech the "
        f"recogniser timed — it is a held word closed by the next one (ruling A2). With "
        f"MAX_WORD_MS = {max_word_ms} it must not open a run; on 2026-09-16 it opened one 800 ms "
        "into the assistant's reply"
    )
    quiet = [
        ep.observe(SttStepEvent(pause_heads=heads(HIGH), t_ms=268240 + STEP_MS * i))
        for i in range(1, 2 + 2 * EOT_SILENCE_MS // STEP_MS)
    ]
    assert all(d is None for d in quiet), (
        f"…and a run it never opened cannot end either (rule (e)); got {[d for d in quiet if d]}"
    )

    # (b) the stale start, after a real decision — the second turn of the log.
    words = spoken("hello there", 1000, 200)
    events = stream(words, until_ms=4400, head_of=lambda t_ms: LOW, delivery_lag_ms=500)
    ep2 = Endpointer()
    fired = [(ev, d) for ev in events if (d := ep2.observe(ev)) is not None]
    ends = [d for _, d in fired if d[0] == "end_of_turn"]
    assert len(ends) == 1, f"the row's premise: one decision first; got {decisions_of(fired)}"
    t_eot_ms = ends[0][1]
    stale = SttWordEvent(text="stale", start_ms=t_eot_ms - 640, end_ms=t_eot_ms - 400)
    assert stale.end_ms - stale.start_ms < max_word_ms, "this half is about staleness, not length"
    assert ep2.observe(stale) is None, (
        f"criterion 5: a word spanning [{stale.start_ms}, {stale.end_ms}] is audio from *before* "
        f"the decided end of turn at {t_eot_ms}; it belongs to the turn already frozen and must "
        "not open a new run"
    )


def test_a_normal_word_after_a_long_silence_still_opens_a_run():
    """Criterion 5's **negative twin**: the guards must not make the endpointer deaf.

    A guard against impossible words is one line away from a guard against speaking again. Thirty
    seconds after the previous ``t_eot`` — a human reading the screen, thinking, then asking the
    next question — a perfectly ordinary 240 ms word must open a run and its turn must end
    normally. This row passes on today's module **by construction** (there is no guard yet); it is
    the row that goes red if the implementer writes the guard too wide, which is why it carries no
    ``pending`` marker (a pending row that passes is an XPASS-strict failure, and would have to be
    deleted rather than kept as the guard it is).

    Hand-computed: run 1 is "hello there" over ``[1000, 1400]`` with ``L = 500`` and ``LOW``
    heads, so the backstop decides at step 2720 with ``t_eot_ms`` 1400. Run 2's word spans
    ``[31400, 31640]`` — 30 000 ms after that ``t_eot``, 240 ms long — and arrives at step 32160.
    Its onset is at 31400 and its own backstop at 32160 + 800 = 32960.

    Regime: pure arithmetic, step clock only, n = 1 deterministic.
    """
    require()
    lag_ms = 500
    words = spoken("hello there", 1000, 200)
    later = SttWordEvent(text="again", start_ms=31400, end_ms=31640)
    assert later.start_ms - 1400 == 30_000 and later.end_ms - later.start_ms == 240
    events = stream(
        [*words, later],
        until_ms=33600,
        head_of=lambda t_ms: LOW,
        delivery_lag_ms=lag_ms,
    )
    fired = drive(events)
    assert [(d[0], d[1]) for d in decisions_of(fired)] == [
        ("onset", 1000),
        ("end_of_turn", 1400),
        ("onset", 31400),
        ("end_of_turn", 31640),
    ], (
        "criterion 5, negative half: a normal 240 ms word 30 s after the previous end of turn is "
        "the next question, not a stale fragment. The guards of criterion 5 must exclude the "
        f"21 040 ms held word and the word that predates t_eot, and nothing else. Got {fired}"
    )


def test_a_short_resumption_delivered_after_the_decision_still_opens_a_run():
    """Criterion 5's negative half **on the shape that discriminates** (adversarial, ruling A19).

    The thirty-second twin above cannot fail: no plausible over-wide guard reaches 30 s. A
    reviewer built the one that matters — the staleness guard keyed on the **step clock at the
    decision** instead of on the word's **audio end** — and every row of this module stayed green.
    That build refuses each word starting within ``L + EOT_SILENCE_MS`` of the previous decision:
    ≈ 1.3 s with tonight's 500 ms GPU delivery lag, and ≈ 2.8 s on the CPU recogniser, whose lag
    is seconds. "Deaf for the next two and a half seconds after every end of turn" is precisely
    what this criterion exists to forbid, and it left no trace.

    The discriminating shape is the twin's opposite: a **short** resumption **delivered after**
    the decision. Hand-computed from the 80 ms grid and nothing else (``ceil80`` = the first step
    at or after a time, which is :func:`arrival_of`):

      run 1   "hello there" over ``[1000,1200] [1200,1400]``, heads ``LOW`` so the EMA cannot
              cross ``EOT_P_THRESHOLD`` and only the backstop can fire. The last word arrives at
              ``ceil80(1400 + L)``; the run closes ``EOT_SILENCE_MS`` later at ``t_eot`` = 1400.
      pause   600 ms of audio — the speaker draws breath and carries on.
      run 2   one 400 ms word over ``[2000, 2400]``, arriving at ``ceil80(2400 + L)``.

                       L = 500        L = 2000 (the CPU fallback)
        decision 1     step 2720      step 4240
        arrival 2      step 2960      step 4400
        margin          240 ms         160 ms

    The margin is asserted, so the row cannot go vacuous if a constant or the grid moves: the
    resumption really is delivered *after* the run it follows was closed, which is the whole
    point — a guard keyed on the decision's step clock has already swallowed it by then.

    Both lags must yield the **same four decisions in audio time**. The guard is keyed on audio,
    so how late the recogniser is may move *when* a decision fires but never *which* words are
    believed; the reviewer's build refuses the second onset at both lags (2000 ≤ 2720 and
    2000 ≤ 4240) and the speaker's second sentence is never heard at all.

    Regime: pure arithmetic, step clock only, n = 1 deterministic per lag, lags 500 (measured on
    the GPU on 2026-09-16) and 2000 (the CPU fallback's order of magnitude, ruling A21).
    """
    require()
    run_one = spoken("hello there", 1000, 200)
    resumption = SttWordEvent(text="wait", start_ms=2000, end_ms=2400)
    assert resumption.start_ms - run_one[-1].end_ms == 600, "a 600 ms pause, by hand"
    assert resumption.end_ms - resumption.start_ms == 400, "an ordinary word, not a fabrication"
    expected = [
        ("onset", 1000),
        ("end_of_turn", 1400),
        ("onset", 2000),
        ("end_of_turn", 2400),
    ]
    violations: list[str] = []
    for lag_ms, decision_ms, arrival_ms in ((500, 2720, 2960), (2000, 4240, 4400)):
        assert decision_ms == arrival_of(run_one[-1], delivery_lag_ms=lag_ms) + EOT_SILENCE_MS, (
            f"L={lag_ms}: recompute the table — run 1 closes EOT_SILENCE_MS after its last word "
            "arrived (criterion 2), and that is the step whose clock the reviewer's build keyed on"
        )
        assert arrival_ms == arrival_of(resumption, delivery_lag_ms=lag_ms), (
            f"L={lag_ms}: recompute the table — the resumption's arrival is on the 80 ms grid"
        )
        assert arrival_ms > decision_ms, (
            f"L={lag_ms}: the row's premise is that the resumption is delivered *after* the "
            f"decision ({arrival_ms} vs {decision_ms}); as written it arrives first and the row "
            "would pin nothing"
        )
        assert resumption.start_ms <= decision_ms, (
            f"L={lag_ms}: and that it starts *inside* the window a decision-clock guard would "
            f"refuse ({resumption.start_ms} ≤ {decision_ms}); otherwise the mutation this row "
            "exists for would pass it"
        )
        fired = drive(
            stream(
                [*run_one, resumption],
                until_ms=arrival_ms + 3 * EOT_SILENCE_MS,
                head_of=lambda t_ms: LOW,
                delivery_lag_ms=lag_ms,
            )
        )
        got = [(d[0], d[1]) for d in decisions_of(fired)]
        if got != expected:
            violations.append(f"L={lag_ms}: {got}")
    assert not violations, (
        "criterion 5, negative half (ruling A19): a 400 ms word starting 600 ms after the decided "
        f"end of turn is the speaker carrying on, and it must open a run. Expected {expected} in "
        "audio time at every delivery lag — the staleness guard is keyed on the word's audio end, "
        "never on the step clock at which the decision happened, which trails it by the "
        f"recogniser's emission delay plus EOT_SILENCE_MS={EOT_SILENCE_MS}. Got:\n  "
        + "\n  ".join(violations)
    )


# --- the property: the generalisation of criteria 4 and 5 --------------------------------------


@settings(max_examples=150, deadline=None, report_multiple_bugs=False)
@given(
    st.integers(min_value=0, max_value=12),  # the recogniser's own emission delay, in 80 ms steps
    st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=1),  # 0 = word, 1 = step only
            st.integers(min_value=1, max_value=6),  # gap to the next step, in 80 ms steps
            st.one_of(
                st.sampled_from([0.0, 0.05, 0.5, 0.95, 0.99, 1.0]),
                st.floats(min_value=0.0, max_value=1.0),
            ),  # pause head
            st.one_of(st.just(0), st.integers(min_value=0, max_value=24)),  # extra hold, in steps
            st.integers(min_value=1, max_value=6),  # word duration, in 80 ms steps
            # T-0051 criterion 12: one word in four is one the recogniser never closed, which is
            # what the hold of criterion 9 releases. Weighted rather than 50/50 so most streams
            # still exercise the ordinary path.
            st.sampled_from([False, False, False, True]),  # end_estimated
        ),
        min_size=1,
        max_size=30,
    ),
)
def test_end_of_turns_never_outnumber_the_words_that_could_have_started_a_run(base_lag, rows):
    """**The class, not the instance** (T-0051, *Tests to write*, the property row).

    Every end of turn closes a run, and a run is opened by a word that is *new*: one whose
    ``start_ms`` is later than the ``t_eot`` of the previous decision and whose claimed span is
    inside ``MAX_WORD_MS`` (criteria 4 and 5). Four consequences, checked at the moment of every
    decision, over any interleaving:

    1. ``#end_of_turn <= #words whose start_ms > the last decided t_eot`` — counted **when the
       decision fires**, not at the end of the stream: a later, legitimate word must not
       retroactively pay for an end of turn that had no word behind it;
    2. the decided ``t_eot`` values are **strictly increasing** — the log's literal symptom is the
       same time decided twice (242720 at seq 3093 and again at 3121; 268880 at 3889 and 3917),
       each one costing a ``mark()``, a flush and ~350 ms of turn latency;
    3. no ``onset`` is at or before the last decided ``t_eot`` — criterion 4 as a property;
    4. **the converse of (3)**: while no run is open, a word that criteria 4, 5 and **12** do
       *not* exclude must open one, at its own ``start_ms``.

    (4) is what ruling A19 found missing. Sub-invariants 1–3 are all **one-sided**: each fires
    when the guard is too permissive and is trivially satisfied when it is too strict, so the
    sensitivity figures below — all measured against a build with the guard *removed* — say
    nothing about a build with the guard *widened*. A guard keyed on the step clock at the
    decision rather than on the word's audio end refuses every word starting within
    ``lag + EOT_SILENCE_MS`` of the previous decision and passed 1–3 without a murmur. (4) makes
    the row two-sided: it is the only sub-invariant that can fire on a deaf endpointer.

    The oracle reads only the decisions the endpointer emitted and the word times this test
    generated. (4) restates criteria 4 and 5's admission rule — the same predicate (1) already
    counts ``qualifying`` words with, now read in the other direction — and nothing else: the
    backstop, the EMA, ``MIN_SPEECH_MS`` and the arrival rule stay the module's own, and the
    ``t_eot`` the predicate is keyed on is the one the endpointer itself reported.

    **Group B landed the third refusal** (ruling A26): a word with ``end_estimated=True`` never
    opens a run, because the hold of criterion 9 can release one in the middle of the assistant's
    reply and an onset there is a barge-in through the STT tap. The predicate of (1) and (4) is
    extended with it rather than left alone — it would otherwise be silently *wrong*, since every
    word generated before group B carried the field's default — and the generator now produces
    such words (one in four) so the extension is exercised instead of merely written down. (4)
    stays two-sided with it: an endpointer that refused every *measured* word as well would break
    (4) on three streams in four.

    The streams model a **lagging** recogniser, which is what the box does and what no earlier
    property covered: one emission delay per stream (0–960 ms, the constant the recogniser has)
    plus a per-word *hold* of up to 1.9 s, which is the shape of the word that sat inside the
    client on 2026-09-16 waiting for an ``EndWord`` that never came (ruling A2). A word published
    at step ``t`` carries audio ``[t − lag − duration, t − lag]``.

    n = 150 Hypothesis examples, Hypothesis's own database and default seeding. Measured
    sensitivity over 20 000 uniformly random streams of this shape (``random.Random(20260916)``,
    the strategy above sampled directly), against the reference build and two wrong ones:

      build                                (1)     (2)     (3)     (4)
      reference                            0 %     0 %     0 %     0 %      (0 of 20 000)
      ``_admits`` → ``return True``      2.1 %   5.9 %  34.0 %     0 %
      guard keyed on the decision's step   0 %     0 %     0 %  64.4 %

    The second line is ruling A10's measurement, replicated here (it reported 2 %, 7 %, 36 % off
    a different generator); the third is ruling A19's build, ``self._last_t_eot_ms`` set to
    ``event.t_ms`` instead of to the last word's audio ``end_ms``. **The two wrong builds break
    disjoint sub-invariants** — that is the whole case for (4), and the reason a suite of
    one-sided invariants read as covered for as long as it did.
    """
    require()
    require_end_estimated()
    from tests.fakes.stt import word_event

    max_word_ms = new_symbol("MAX_WORD_MS")
    assert isinstance(max_word_ms, int)
    ep = Endpointer()
    t_ms = 0
    qualifying = 0
    eots = 0
    last_t_eot: int | None = None
    run_open = False  # a run is open exactly while the last decision was an onset (rule d)
    trace: list[str] = []
    breach: str | None = None

    def take(decision: tuple | None, where: str) -> None:
        nonlocal eots, last_t_eot, breach, run_open
        if decision is None:
            return
        trace.append(f"{where} → {decision}")
        if decision[0] == "onset":
            run_open = True
            if last_t_eot is not None and decision[1] <= last_t_eot and breach is None:
                breach = (
                    f"onset at {decision[1]}, which is at or before the end of turn already "
                    f"decided at {last_t_eot}: a run re-opened on audio the endpointer had "
                    "declared finished (log seq 3093 → 3113)"
                )
            return
        eots += 1
        run_open = False
        if last_t_eot is not None and decision[1] <= last_t_eot and breach is None:
            breach = (
                f"end of turn at t_eot {decision[1]} after one at {last_t_eot}: the same audio "
                "ended the turn twice"
            )
        last_t_eot = decision[1]
        if eots > qualifying and breach is None:
            breach = (
                f"{eots} end-of-turns off {qualifying} words that could have opened a run, at "
                f"step {t_ms}"
            )

    for kind, gap, head, hold, duration, estimated in rows:
        t_ms += gap * STEP_MS
        take(ep.observe(SttStepEvent(pause_heads=heads(head), t_ms=t_ms)), f"step {t_ms}")
        if kind != 0:
            continue
        end_ms = max(0, t_ms - (base_lag + hold) * STEP_MS)
        word = word_event(
            "w",
            max(0, end_ms - duration * STEP_MS),
            end_ms,
            end_estimated=estimated,
        )
        # A word that could open a run: new audio (criterion 4), a span the recogniser could have
        # timed (criterion 5), and an end the recogniser actually measured (criterion 12). An
        # estimated end is *not* such a word, so it may never be what an end of turn is counted
        # against in (1) either.
        fresh = (last_t_eot is None or word.start_ms > last_t_eot) and not estimated
        if fresh:
            qualifying += 1
        # Sub-invariant (4), evaluated *before* the decision is folded in: this word is new audio
        # of a length the recogniser could plausibly have timed, its end was measured, and no run
        # is open — so criteria 4, 5 and 12 leave the endpointer no reason to refuse it.
        must_open = fresh and word.end_ms - word.start_ms <= max_word_ms and not run_open
        where = (
            f"word [{word.start_ms},{word.end_ms}] (end_estimated={estimated}) arriving at {t_ms}"
        )
        decision = ep.observe(word)
        if must_open and decision != ("onset", word.start_ms) and breach is None:
            breach = (
                f"{where} was refused (decision {decision}): it starts after the last decided "
                f"t_eot ({last_t_eot}), claims {word.end_ms - word.start_ms} ms of audio which is "
                f"inside MAX_WORD_MS={max_word_ms}, carries an end the recogniser measured, and "
                "no run was open — so it is the speaker starting again and it must open a run at "
                "its own start_ms. A guard keyed on "
                "anything later than the word's audio end (the step clock at the decision, say) "
                "makes the endpointer deaf for the recogniser's delay plus EOT_SILENCE_MS after "
                "every turn: ≈ 1.3 s on the GPU and seconds on the CPU (T-0051 criterion 5's "
                "negative half, ruling A19)"
            )
        take(decision, where)
    # One assertion, at one line: two `assert` sites would let Hypothesis report two distinct
    # failures and raise an ExceptionGroup, which the `pending` marker's
    # `xfail(raises=AssertionError)` would not recognise (tests/conftest.py).
    assert breach is None, f"{breach}. Trace:\n  " + "\n  ".join(trace)


# --- criteria 5 and 6: the constants ------------------------------------------------------------


def test_max_word_ms_sits_in_its_band():
    """Criterion 5's new constant, pinned to a band with a source outside the module.

    Lower bound **1000 ms**: the longest single word a human says at the desk is around a second
    ("characteristically", a drawn-out "riiiight"), and moshi-server labels one *word* per token
    run, so a guard below that starts discarding speech. Upper bound **5000 ms**: it must exclude
    tonight's 21 040 ms fabrication (``see?``, ``start_ms`` 247200 → ``end_ms`` 268240, log seq
    3844) by a wide margin and it must be shorter than the 21 s hold that produced it; 2000 ms is
    the value the task recommends, and anything in this band is defensible with a stated reason.

    Not a tautology: both bounds come from written sources (docs/tasks/T-0051.md criterion 5 and
    the session log), never from the value in the module.
    """
    require()
    max_word_ms = new_symbol("MAX_WORD_MS")
    assert isinstance(max_word_ms, int), f"MAX_WORD_MS is a whole number of ms; got {max_word_ms!r}"
    assert 1000 <= max_word_ms <= 5000, (
        f"MAX_WORD_MS={max_word_ms}: ≥ 1000 ms because a real spoken word can be that long and a "
        "guard below it discards speech; ≤ 5000 ms because it exists to refuse the 21 040 ms word "
        "of 2026-09-16 (log seq 3844) and anything of that family"
    )
    assert max_word_ms < 21040, (
        f"MAX_WORD_MS={max_word_ms} would admit tonight's 21 040 ms held word, which is the one "
        "case criterion 5 names"
    )


def test_the_constants_this_task_must_not_move():
    """Criterion 6: T-0051 fixes what the numbers are measured against, not the numbers.

    Every value below is pinned by a document, not by the module: ``EOT_P_THRESHOLD`` 0.6 is
    docs/02 §2.2 and ``TurnPolicy.eot_p_threshold``'s default (docs/01 §8); ``EMA_HALF_LIFE_STEPS``
    3 steps ≈ 240 ms and ``EMA_ALPHA`` is derived from it, never tuned separately;
    ``MIN_SPEECH_MS`` 200 and ``PAUSE_HEAD_INDEX`` 2 are T-0032's, and T-0051's ambiguity 2
    recommends keeping index 2 until X2/Q5 decides it on more than two turns; ``EOT_SILENCE_MS``
    800 is kept by ambiguity 3 ("keep 800 and let the heads decide, revisit with a bench, never
    with a tuning commit"), and criterion 2 changes the clock it is measured in, not its size.

    This row passes before the implementation exists — it is a *do not move* guard, so it carries
    no ``pending`` marker — and goes red the moment the endpointer is made to pass a test by
    editing a constant instead of the rule. CLAUDE.md: a threshold is not moved without a
    docs/tasks entry explaining why; the escalation triggers in T-0051's *Blast radius* name
    exactly these names.
    """
    require()
    assert _tp.EOT_P_THRESHOLD == 0.6, "docs/01 §8 / docs/02 §2.2; the machine's default is 0.6"
    assert _tp.EMA_HALF_LIFE_STEPS == 3.0, "3 steps ≈ 240 ms of audio (T-0032 ruling A9)"
    assert _tp.EMA_ALPHA == 1.0 - 0.5 ** (1.0 / 3.0), "α is derived from the half-life, not tuned"
    assert _tp.MIN_SPEECH_MS == 200, "T-0032 ruling A8/A16 F3; the head path's gate"
    assert _tp.PAUSE_HEAD_INDEX == 2, "T-0051 ambiguity 2: keep 2, decide with X2/Q5"
    assert _tp.EOT_SILENCE_MS == 800, (
        "T-0051 ambiguity 3: keep 800 and let the heads decide. Criterion 2 changes the clock the "
        "800 is measured in — from the word's audio end to the step at which it arrived — and a "
        "backstop that fires on time only because this number shrank has not been fixed"
    )


# ==================================================================================================
# Measured sensitivity of the three rows below plus the property's fourth sub-invariant, against a
# reference criterion-12 implementation built in an isolated copy (import origin proved from
# inside the run, `.claude/rules/tests.md`): all 29 rows green. Then, by direction —
#   the refusal **absent**:                         3 red (both rows below and the property)
#   the refusal in the wrong **place** (ruling A14's shape: moved out of the top of the word path
#     into the "no run open" branch, so a refused word still extends a run and stamps an arrival):
#                                                   1 red — the A14/A22 row below, and nothing
#                                                   else, which is why ruling A17 asked for it
#   the refusal **too wide** (every word refused):  23 red, including the property's (4)
# The three sets are what a two-sided guard looks like: absent and too-wide break different rows.
#
# T-0051 group B — criterion 12: a word whose end was never measured can never open a turn.
#
# Group B bounds the wait for an ``EndWord`` that never comes (criterion 9), which means a word
# recognised before the assistant started speaking can now be *released in the middle of its
# reply*. Read as an onset that is a barge-in through the STT tap: docs/02 §4's SPEAKING row takes
# the machine to PAUSED on ``user.speech_onset``, T-0032's design ruling says the endpointer must
# not do that and T-0034 (the real barge-in) is not built. So group B's own fix creates the defect
# unless the endpointer refuses such a word — which is why the criterion is not optional polish.
# ==================================================================================================


def estimated_word(text: str, start_ms: int, end_ms: int) -> SttWordEvent:
    """A word the recogniser never closed, as ``MoshiSttClient`` publishes it (criterion 8/9)."""
    from tests.fakes.stt import word_event

    return word_event(text, start_ms, end_ms, end_estimated=True)


def require_end_estimated() -> None:
    from tests.fakes.stt import require_end_estimated as _require

    _require()


def test_an_estimated_end_word_never_opens_a_run():
    """**Adversarial, T-0051 criterion 12** — the accidental barge-in, refused at the endpointer.

    What goes wrong without it, concretely: the hold of criterion 9 releases the word the server
    never closed. On 2026-09-16 that word (``see?``) was released 800 ms *into the assistant's own
    reply*. With the hold in place it would be released sooner — still inside SPEAKING — and an
    ``onset`` there becomes ``E.UserSpeechOnset``, which docs/02 §4's SPEAKING row turns into
    **PAUSED**: a barge-in through the STT tap, forbidden by T-0032's design ruling and not
    implemented (T-0034). The machine half of that sentence is not taken on trust below; it is
    executed against the real ``TurnMachine``, so the row cannot quietly stop being about anything.

    Three assertions, in the order that makes the refusal non-vacuous:

    1. the premise, from the machine itself: an onset **would** take SPEAKING to PAUSED;
    2. the estimated word produces **no decision**, and the silence after it produces none either
       (rule (e): a run it never opened cannot end);
    3. **the other direction** — the identical word with ``end_estimated=False`` opens a run at its
       own ``start_ms``. Without (3) an endpointer that refused every word would pass, and that is
       the deafness criterion 5's twin exists to forbid. The two words differ in one field.

    The word's other properties are deliberately impeccable: 200 ms long (well inside
    ``MAX_WORD_MS``), starting after every decision (there are none), delivered against a live
    step stream. Only the flag can refuse it.

    Regime: pure arithmetic, step clock only, n = 1 deterministic.
    """
    require()
    require_end_estimated()

    # (1) the premise, read off the real machine rather than described
    machine = TurnMachine(TurnPolicy())
    speaking = machine.initial().model_copy(update={"state": TurnState.SPEAKING})
    paused, _ = machine.step(speaking, E.UserSpeechOnset(t_ms=1_000_000, onset_ms=1200))
    assert paused.state is TurnState.PAUSED, (
        "the premise of this row: an onset during the assistant's turn is a barge-in — docs/02 §4 "
        f"SPEAKING + user.speech_onset → PAUSED. The machine went to {paused.state} instead, so "
        "either the machine changed or this row is no longer about what it says it is"
    )

    held = estimated_word("see?", start_ms=1200, end_ms=1400)
    honest = SttWordEvent(text="see?", start_ms=1200, end_ms=1400)
    assert (held.start_ms, held.end_ms) == (honest.start_ms, honest.end_ms), (
        "the two words differ in exactly one field, so nothing but the flag can explain a "
        "difference in the decision"
    )
    assert held.end_ms - held.start_ms < new_symbol("MAX_WORD_MS"), (
        "this word is short and plausible: criterion 5's length guard must not be what refuses it"
    )

    # (2) the refusal
    ep = Endpointer()
    for t_ms in (1000, 1080, 1160, 1240):
        assert ep.observe(SttStepEvent(pause_heads=heads(LOW), t_ms=t_ms)) is None
    assert ep.observe(held) is None, (
        "criterion 12: a word whose end nobody measured (end_estimated=True — the hold of "
        "criterion 9, the marker echo, close(), a disconnect, or the next word's arrival) must "
        "not open a run. Released during the assistant's reply it becomes the onset asserted "
        "above, i.e. SPEAKING → PAUSED through the STT tap, which T-0032's design ruling and "
        "T-0034 forbid"
    )
    quiet = [
        ep.observe(SttStepEvent(pause_heads=heads(HIGH), t_ms=1400 + STEP_MS * i))
        for i in range(1, 2 + 2 * EOT_SILENCE_MS // STEP_MS)
    ]
    assert all(decision is None for decision in quiet), (
        f"…and a run it never opened cannot end either (rule (e)); got {[d for d in quiet if d]}"
    )

    # (3) the other direction: the same word, measured, is the speaker starting to talk
    ep2 = Endpointer()
    for t_ms in (1000, 1080, 1160, 1240):
        ep2.observe(SttStepEvent(pause_heads=heads(LOW), t_ms=t_ms))
    assert ep2.observe(honest) == ("onset", 1200), (
        "criterion 12 refuses the *estimate*, not the word: the same span with a real EndWord "
        "behind it must open a run at 1200. An endpointer that refused both is deaf, which is a "
        "worse defect than the one this criterion fixes (the shape of ruling A19)"
    )


def test_a_refused_estimated_word_moves_neither_the_end_of_turn_nor_the_step_that_decides_it():
    """Ruling A14, as group B inherits it (rulings A17 and A26), asserted the way A22 demands.

    A14: *a refused word is refused entirely* — it does not open a run, does not extend an open
    one, and does not stamp an arrival. Nothing pinned that for criteria 4 and 5: moving the
    refusal out of the top of the word path and into the "no run open" branch left all 25 rows of
    this module green, and group B adds a third refusal whose implementer could write the same
    shape. A22 adds the sharper point: **"nothing was decided" is not the assertion**. Under that
    mutation a fabricated end pushes the arrival stamp into the future, so a 21-second word holds
    the turn open for 21 seconds and then reports an invented audio time straight into the
    ``FreezeAfter`` and the ``mark()``. One of the two mutation shapes produces no decision at all
    within any finite stream, and it is still wrong. So this row asserts the **reported
    end-of-turn time** and the **step that triggered it**.

    Hand-computed, ``L`` = 500 and heads 0.30 throughout so the EMA settles at 0.30 and only the
    backstop can fire:

      words        ``[1000,1200] [1200,1400] [1400,1600]``, the last arriving at ceil80(2100)=2160
      held word    ``[1700, 3400]``, ``end_estimated=True``, delivered at step **2240** — inside
                   the run, with an end 1160 ms in the *future* of the step clock, which is the
                   shape of the word that sat in the client on 2026-09-16 (its end was the next
                   word's start, 21 s later). 1700 ms of span is inside ``MAX_WORD_MS`` and 1700
                   is after every decided ``t_eot`` (there are none yet), so criteria 4 and 5 both
                   admit it: criterion 12 is the only thing that may refuse it.
      expected     one onset at 1000, one end of turn at ``t_eot`` **1600**, triggered at step
                   **2960** = 2160 + ``EOT_SILENCE_MS``.

    Under "refuse only at the onset" the held word extends the run to an end of 3400 and stamps an
    arrival of max(2240, 3400) = 3400, so the decision becomes ``t_eot`` 3400 at step 4200 — a
    freeze and a mark on audio that never happened, 1240 ms late. Both numbers move; this row
    reads both.

    Regime: pure arithmetic, step clock only, n = 1 deterministic.
    """
    require()
    require_end_estimated()
    max_word_ms = new_symbol("MAX_WORD_MS")
    lag_ms = 500
    words = spoken("what is this", 1000, 200)
    last_arrival = arrival_of(words[-1], delivery_lag_ms=lag_ms)
    assert (words[-1].end_ms, last_arrival) == (1600, 2160), "recompute the arrivals"
    expected_trigger = last_arrival + EOT_SILENCE_MS
    assert expected_trigger == 2960, f"recompute: the backstop is due at {expected_trigger}"

    insert_at_ms = 2240
    held = estimated_word("see?", start_ms=1700, end_ms=3400)
    assert held.end_ms - held.start_ms <= max_word_ms, (
        f"the premise: {held.end_ms - held.start_ms} ms is inside MAX_WORD_MS={max_word_ms}, so "
        "criterion 5's length guard is not what refuses this word"
    )
    assert held.end_ms > insert_at_ms, (
        "…and its end lies in the future of the step clock at delivery, which is what makes the "
        "arrival stamp move under the mutation this row exists for"
    )
    assert insert_at_ms < expected_trigger, "…and it is delivered while the run is still open"

    events = stream(words, until_ms=6000, head_of=lambda t_ms: 0.30, delivery_lag_ms=lag_ms)
    at = next(
        i for i, e in enumerate(events) if isinstance(e, SttStepEvent) and e.t_ms == insert_at_ms
    )
    events.insert(at + 1, held)

    ep = Endpointer()
    fired = [(ev, d) for ev in events if (d := ep.observe(ev)) is not None]
    got = [(trigger_ms_of(ev), d) for ev, d in fired]

    # A22's two halves, asserted separately: *what was reported* and *what triggered it*. Neither
    # is "nothing was decided" — one of the two mutation shapes decides nothing at all inside any
    # finite stream, and it is still wrong.
    assert decisions_of(fired) == [("onset", 1000), ("end_of_turn", 1600, 1.0)], (
        "ruling A14/A22: a word the endpointer refuses is refused **entirely** — it may not extend "
        "the run to its fabricated end. 1600 must stay 1600: that number becomes the FreezeAfter "
        f"and the mark(), and 3400 is audio nobody spoke. Got {got}"
    )
    assert [trigger_ms_of(ev) for ev, d in fired if d[0] == "end_of_turn"] == [expected_trigger], (
        "…and it may not stamp an arrival: the backstop is due EOT_SILENCE_MS after the last "
        f"*honest* word arrived, i.e. at {expected_trigger}. A refused word that still stamped one "
        f"would push it to {max(insert_at_ms, held.end_ms) + EOT_SILENCE_MS} — the turn held open "
        "until a moment that never happened, which on 2026-09-16 was 21 seconds. Got "
        f"{got}"
    )
