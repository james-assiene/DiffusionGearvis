"""Filler policy decision table (docs/03 §3, docs/11 §1.1). Task T-0013.

Activated at the start of ``/implement T-0013`` (the ``pending`` marker is gone): these tests
fail until ``local_astra/reflex/filler_policy.py::decide`` exists and are never weakened to get
green (CLAUDE.md). Every test names the acceptance criterion of ``docs/tasks/T-0013.md`` it comes
from; nothing here is derived from an implementation, because there is none.

Boundaries and readings fixed by this file, with their source (the criteria state some of them
and are silent on the rest; see the test-author's report):

* **ack threshold is strict.** Criterion 2 says "iff predicted first canvas > ``ack_threshold_ms``",
  and the turn machine's own ack row uses ``first_canvas_ms > policy.ack_first_canvas_ema_ms``
  (docs/02 §4 row 6, T-0003 criterion 7). So ``predicted == ack_threshold_ms`` → ``none``.
* **min gap is inclusive.** Criterion 3 forbids "two fillers within ``min_gap_ms``"; a filler
  emitted exactly ``min_gap_ms`` ago is not *within* the gap, so ``last_filler_age_ms ==
  min_gap_ms`` is allowed and ``min_gap_ms - 1`` is not.
* **interim update is inclusive**, as criterion 4 writes it: ``elapsed ≥ interim_update_ms``.
* **``last_filler_age_ms`` is the age of the previous filler's *emission***, per its docstring in
  ``FillerInputs`` ("None: no filler yet this turn"), so the previous filler was emitted at
  ``elapsed_ms - last_filler_age_ms`` on the same ``llm.start`` clock. That is the only
  information ``decide`` is given about the past of the turn, so "``clip:still_working`` … once"
  (criterion 4) is decided from it: an interim update has already been spoken iff the previous
  filler was emitted at ``elapsed ≥ interim_update_ms``.
* **only the THINKING state gets fillers**, as in the decision table shipped with the scaffold
  (rows for SPEAKING and LISTENING) and as docs/02 §4 places the ack (``decide_ack`` on the
  ENDPOINTING → THINKING row) and docs/11 §1.1 places the interim update (during the think).
* **the criteria license exactly two fillers per turn**: the ack at ``llm.start`` (criterion 1/2)
  and the interim update at ``elapsed ≥ interim_update_ms`` (criterion 4). Cells the criteria do
  not license are ``none`` — including "a second ack halfway through the think".
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from local_astra.reflex.filler_policy import FillerInputs, FillerPolicyConfig, decide

CFG = FillerPolicyConfig(
    ack_threshold_ms=300, min_gap_ms=2500, interim_update_ms=4000, max_fillers_per_turn=2
)
#: same budgets, a cap high enough that the "once" rule of criterion 4 is not masked by the cap
CFG_ROOMY = FillerPolicyConfig(
    ack_threshold_ms=300, min_gap_ms=2500, interim_update_ms=4000, max_fillers_per_turn=3
)
#: a talker-less profile with a single filler per turn (criterion 3, cap before the interim)
CFG_ONE = FillerPolicyConfig(
    ack_threshold_ms=300, min_gap_ms=2500, interim_update_ms=4000, max_fillers_per_turn=1
)

NON_THINKING = ("IDLE", "LISTENING", "ENDPOINTING", "SPEAKING", "PAUSED", "INTERRUPTED")
DECISIONS = ("none", "clip:ack", "clip:hmm", "clip:still_working", "generate_short")


def inputs(**kw) -> FillerInputs:
    base = dict(
        state="THINKING",
        elapsed_ms=0,
        predicted_first_canvas_ms=1500,
        thinking_enabled=True,
        last_filler_age_ms=None,
        first_canvas_committed=False,
        fillers_this_turn=0,
    )
    base.update(kw)
    return FillerInputs(**base)


# --- criteria 1–4: the table ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({}, "clip:ack"),  # thinking on, llm.start → always ack
        ({"thinking_enabled": False, "predicted_first_canvas_ms": 250}, "none"),  # fast enough
        ({"thinking_enabled": False, "predicted_first_canvas_ms": 350}, "clip:ack"),  # slow → ack
        ({"first_canvas_committed": True}, "none"),  # never after first canvas
        ({"elapsed_ms": 1200, "last_filler_age_ms": 1200, "fillers_this_turn": 1}, "none"),  # gap
        (
            {"elapsed_ms": 4000, "last_filler_age_ms": 4000, "fillers_this_turn": 1},
            "clip:still_working",
        ),
        ({"elapsed_ms": 4000, "last_filler_age_ms": 4000, "fillers_this_turn": 2}, "none"),  # cap
        ({"state": "SPEAKING", "elapsed_ms": 4000}, "none"),  # only while thinking
        ({"state": "LISTENING"}, "none"),
    ],
)
def test_decision_table(kw, expected):
    assert decide(inputs(**kw), CFG) == expected


@pytest.mark.parametrize("state", NON_THINKING)
@pytest.mark.parametrize(
    "kw",
    [
        {"elapsed_ms": 0},  # would be the ack (criterion 1)
        {"elapsed_ms": 4000, "last_filler_age_ms": 4000, "fillers_this_turn": 1},  # the interim
    ],
)
def test_fillers_only_while_thinking(state, kw):
    """Criteria 1 and 4 place both fillers inside the think (docs/02 §4 row 6, docs/11 §1.1).

    The same inputs fire in THINKING (asserted in ``test_decision_table``), so each row here
    isolates the state.
    """
    assert decide(inputs(state=state, **kw), CFG) == "none"


# --- criterion 1: thinking on always acks at llm.start ------------------------------------------


@pytest.mark.parametrize("predicted", [0, 1, 50, 299, 300])
def test_thinking_on_acks_at_llm_start_whatever_the_ema_predicts(predicted):
    """Criterion 1: an EMA predicting an instant (or already-due) canvas does not suppress the ack.

    docs/01 D11: thinking is always on for the assistant session, and the acknowledgement is what
    buys the 2–4 s that research says is acceptable — so the EMA is *not* consulted at all here.
    ``predicted == 300`` is the ack threshold: with thinking off that same input is ``none``
    (``test_ack_threshold_is_strict``), which is what makes this row about criterion 1 and not
    about criterion 2.
    """
    got = decide(
        inputs(elapsed_ms=0, thinking_enabled=True, predicted_first_canvas_ms=predicted), CFG
    )
    assert got == "clip:ack"


# --- criterion 2: thinking off acks iff the EMA exceeds the threshold ---------------------------


@pytest.mark.parametrize(
    ("predicted", "expected"),
    [
        (0, "none"),
        (299, "none"),
        (300, "none"),  # boundary: criterion 2 says "> ack_threshold_ms", strictly
        (301, "clip:ack"),
        (5000, "clip:ack"),
    ],
)
def test_ack_threshold_is_strict(predicted, expected):
    """Criterion 2 at and around ``ack_threshold_ms`` (300 ms, docs/02 §4 row 6)."""
    got = decide(
        inputs(elapsed_ms=0, thinking_enabled=False, predicted_first_canvas_ms=predicted), CFG
    )
    assert got == expected


# --- criterion 3: committed canvas, gap, cap ----------------------------------------------------


@pytest.mark.parametrize(
    ("kw", "would_be_without_the_guard"),
    [
        ({"elapsed_ms": 0}, "clip:ack"),
        (
            {"thinking_enabled": False, "elapsed_ms": 0, "predicted_first_canvas_ms": 5000},
            "clip:ack",
        ),
        (
            {"elapsed_ms": 4000, "last_filler_age_ms": 4000, "fillers_this_turn": 1},
            "clip:still_working",
        ),
        (
            {"elapsed_ms": 9000, "last_filler_age_ms": 9000, "fillers_this_turn": 1},
            "clip:still_working",
        ),
    ],
)
def test_no_filler_once_the_first_canvas_has_committed(kw, would_be_without_the_guard):
    """Criterion 3 (and criterion 4's "only when no canvas has committed").

    The adversarial row is the third: the canvas commits *between* the ack and the interim update
    (the ack was played at elapsed 0, the answer is already being spoken at 4 s), so the interim
    "still reading the second file…" must be suppressed — saying it over a committed answer is the
    dead-air fix causing an interruption. Each row is a cell that fires without the guard, which
    is asserted here so the row cannot rot into a vacuous ``none``.
    """
    assert decide(inputs(**kw), CFG) == would_be_without_the_guard
    assert decide(inputs(first_canvas_committed=True, **kw), CFG) == "none"


@pytest.mark.parametrize(
    ("elapsed", "age", "expected"),
    [
        (4000, 2499, "none"),  # previous filler at 1501 ms: inside the gap
        (4000, 2500, "clip:still_working"),  # exactly the gap: not "within" it
        (4000, 2501, "clip:still_working"),
        (4200, 500, "none"),  # interim due, but the ack went out 500 ms ago
        (6000, 1000, "none"),  # previous filler at 5000 ms, i.e. 1 s ago
        (6000, 6000, "clip:still_working"),  # same elapsed, filler back at llm.start
        (30000, 10, "none"),  # a long think does not buy a filler right after one
    ],
)
def test_min_gap_is_measured_from_the_previous_fillers_emission(elapsed, age, expected):
    """Criterion 3: "never two fillers within ``min_gap_ms``" (2500 ms).

    The pairs at ``elapsed = 6000`` are the adversarial ones: they hold ``elapsed`` fixed and vary
    only *when the last filler was emitted*, so an implementation that measures the gap from
    ``llm.start`` (or from the interim deadline) instead of from the previous emission fails.
    """
    got = decide(inputs(elapsed_ms=elapsed, last_filler_age_ms=age, fillers_this_turn=1), CFG)
    assert got == expected


@pytest.mark.parametrize("elapsed", [4000, 6000, 12000])
def test_cap_reached_before_the_interim_update(elapsed):
    """Criterion 3: ``max_fillers_per_turn`` wins over criterion 4's interim update.

    ``CFG_ONE`` spends its single filler on the ack at ``llm.start``; the gap is long satisfied
    and no canvas has committed, so only the cap can hold the interim update back. Under ``CFG``
    (cap 2) the very same inputs give the interim update, which is what makes this non-vacuous.
    """
    inp = inputs(elapsed_ms=elapsed, last_filler_age_ms=elapsed, fillers_this_turn=1)
    assert decide(inp, CFG) == "clip:still_working"
    assert decide(inp, CFG_ONE) == "none"


# --- criterion 4: the interim spoken update, once -----------------------------------------------


@pytest.mark.parametrize(
    ("elapsed", "expected"),
    [
        (3999, "none"),  # boundary: criterion 4 says "elapsed ≥ interim_update_ms"
        (4000, "clip:still_working"),
        (4001, "clip:still_working"),
    ],
)
def test_interim_update_boundary(elapsed, expected):
    """Criterion 4 at ``interim_update_ms`` (4 s, docs/01 §8 / docs/11 §1.1 tier B)."""
    got = decide(inputs(elapsed_ms=elapsed, last_filler_age_ms=elapsed, fillers_this_turn=1), CFG)
    assert got == expected


def test_interim_update_fires_once_per_turn():
    """Criterion 4: "``clip:still_working`` … once".

    The cap is deliberately out of the way (``CFG_ROOMY``, 3 fillers) so that only the "once" rule
    can suppress the second interim update. Timeline, tier B (docs/11 §1.1): ack at 0, interim at
    4000, and at 7000 the gap (2500) is satisfied and the cap is not reached — yet a second "still
    working" is not licensed by any criterion. ``decide`` knows the interim already went out
    because the previous filler was emitted at ``7000 - 3000 == 4000 ≥ interim_update_ms``.
    """
    assert decide(inputs(elapsed_ms=0), CFG_ROOMY) == "clip:ack"
    first = inputs(elapsed_ms=4000, last_filler_age_ms=4000, fillers_this_turn=1)
    assert decide(first, CFG_ROOMY) == "clip:still_working"
    second = inputs(elapsed_ms=7000, last_filler_age_ms=3000, fillers_this_turn=2)
    assert decide(second, CFG_ROOMY) == "none"
    later = inputs(elapsed_ms=20000, last_filler_age_ms=16000, fillers_this_turn=2)
    assert decide(later, CFG_ROOMY) == "none"


def test_defaults_match_the_documented_budgets():
    """``cfg=None`` uses the budgets of docs/02 §4 row 6 (300 ms) and docs/11 §1.1 tier B (~4 s).

    Exercised *through* ``decide`` rather than by reading the dataclass fields, so this is a
    statement about the policy and not about a default value.
    """
    assert decide(inputs(thinking_enabled=False, predicted_first_canvas_ms=300)) == "none"
    assert decide(inputs(thinking_enabled=False, predicted_first_canvas_ms=301)) == "clip:ack"
    assert decide(inputs(elapsed_ms=3999, last_filler_age_ms=3999, fillers_this_turn=1)) == "none"
    assert (
        decide(inputs(elapsed_ms=4000, last_filler_age_ms=4000, fillers_this_turn=1))
        == "clip:still_working"
    )


# --- whole turns: the caller's bookkeeping ------------------------------------------------------


def run_turn(ticks, cfg, *, thinking=True, committed_at_ms=None):
    """Drive ``decide`` over one turn the way the Conductor does and return the decisions.

    The loop is *caller* bookkeeping (count the fillers it emitted, remember when), never a
    re-implementation of the policy: it never decides anything, it only records what ``decide``
    returned.
    """
    decisions = []
    fillers = 0
    last_emission_ms = None
    for t in ticks:
        got = decide(
            FillerInputs(
                state="THINKING",
                elapsed_ms=t,
                predicted_first_canvas_ms=1500,
                thinking_enabled=thinking,
                last_filler_age_ms=None if last_emission_ms is None else t - last_emission_ms,
                first_canvas_committed=committed_at_ms is not None and t >= committed_at_ms,
                fillers_this_turn=fillers,
            ),
            cfg,
        )
        decisions.append(got)
        if got != "none":
            fillers += 1
            last_emission_ms = t
    return decisions


def test_tier_b_turn_timeline():
    """Criteria 1, 3 and 4 over one 8 s tier-B think (docs/11 §1.1), expectation written by hand.

    0 ms ack (criterion 1) · 1000 ms inside the gap · 2500 ms gap satisfied but no criterion
    licenses a second ack · 4000 ms the interim update (criterion 4) · 5000 ms inside the gap
    · 8000 ms the cap of 2 is spent (criterion 3).
    """
    assert run_turn([0, 1000, 2500, 4000, 5000, 8000], CFG) == [
        "clip:ack",
        "none",
        "none",
        "clip:still_working",
        "none",
        "none",
    ]


def test_turn_where_the_canvas_commits_before_the_interim_deadline():
    """Criteria 3 and 4: the ack goes out, the answer commits at 2 s, nothing else is spoken."""
    assert run_turn([0, 1000, 2500, 4000, 8000], CFG, committed_at_ms=2000) == [
        "clip:ack",
        "none",
        "none",
        "none",
        "none",
    ]


def test_a_second_turn_starts_from_a_clean_count():
    """Criterion 3's budgets are *per turn*: a fresh turn acks again (criterion 1).

    ``decide`` is a pure decision table, so the only thing that could carry a spent budget into
    the next turn is hidden state inside the module — which this pins down.
    """
    first = run_turn([0, 4000, 8000], CFG)
    assert first == ["clip:ack", "clip:still_working", "none"]
    second = run_turn([0, 4000, 8000], CFG)
    assert second == first


def test_decide_is_a_pure_function_of_its_inputs():
    """Same inputs, same decision — whatever was asked in between (docs/03 §2: pure core)."""
    probe = inputs(elapsed_ms=0)
    interim = inputs(elapsed_ms=4000, last_filler_age_ms=4000, fillers_this_turn=1)
    before = decide(probe, CFG)
    for _ in range(3):
        decide(interim, CFG)
        decide(inputs(first_canvas_committed=True), CFG)
    assert decide(probe, CFG) == before == "clip:ack"


# --- criterion 5 and the criterion-3 invariants, as properties ----------------------------------

any_inputs = st.builds(
    FillerInputs,
    state=st.sampled_from(("THINKING", *NON_THINKING)),
    elapsed_ms=st.integers(min_value=0, max_value=30_000),
    predicted_first_canvas_ms=st.integers(min_value=0, max_value=20_000),
    thinking_enabled=st.booleans(),
    last_filler_age_ms=st.none() | st.integers(min_value=0, max_value=30_000),
    first_canvas_committed=st.booleans(),
    fillers_this_turn=st.integers(min_value=0, max_value=4),
)


@settings(derandomize=True, max_examples=400)
@given(inp=any_inputs)
def test_generate_short_is_never_returned(inp):
    """Criterion 5: reserved for the talker profile (docs/11 §3), not emitted in v1.

    400 derandomised examples (fixed generation, no flaky seed) over every input field.
    """
    got = decide(inp, CFG)
    assert got in DECISIONS
    assert got != "generate_short"


@settings(derandomize=True, max_examples=400)
@given(inp=any_inputs)
def test_criterion_3_invariants_hold_for_every_input(inp):
    """Criterion 3 as three implications, over arbitrary inputs.

    (a) a committed first canvas silences the lane; (b) the per-turn cap is never exceeded;
    (c) two fillers are never closer than ``min_gap_ms``. Stated as implications, so the test says
    nothing about the cells where a filler *is* allowed (those are the table above).
    """
    got = decide(inp, CFG)
    if inp.first_canvas_committed:
        assert got == "none"
    if inp.fillers_this_turn >= CFG.max_fillers_per_turn:
        assert got == "none"
    if inp.last_filler_age_ms is not None and inp.last_filler_age_ms < CFG.min_gap_ms:
        assert got == "none"


@settings(derandomize=True, max_examples=200)
@given(predicted=st.integers(min_value=0, max_value=20_000))
def test_thinking_on_always_acknowledges_the_turn_at_llm_start(predicted):
    """Criterion 1 as a property: at ``elapsed == 0``, no EMA value silences the acknowledgement.

    The turn is fresh (no filler yet, nothing committed), which is exactly the state in which
    criterion 1 applies; criterion 3 legitimately overrides criterion 1 outside it.
    """
    inp = inputs(elapsed_ms=0, thinking_enabled=True, predicted_first_canvas_ms=predicted)
    assert decide(inp, CFG) == "clip:ack"


def test_generate_short_is_reserved_for_the_talker_profile():
    for elapsed in (0, 2000, 4000, 8000):
        assert (
            decide(
                inputs(
                    elapsed_ms=elapsed,
                    last_filler_age_ms=elapsed or None,
                    fillers_this_turn=1 if elapsed else 0,
                ),
                CFG,
            )
            != "generate_short"
        )
