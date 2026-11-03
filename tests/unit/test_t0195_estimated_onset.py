"""T-0195 group A, the pure half: an estimated word may open a run when the caller says so and none is open.

``docs/tasks/T-0195.md`` A1 (the rule), A2 (the default is BASE bit for bit) and A8's endpointer half ((f) and (g)
still win). The surface these rows call is A1's, verbatim::

    Endpointer().observe(event, *, may_open_on_estimate: bool = False) -> Decision | None

An ``SttWordEvent`` with ``end_estimated=True`` produces ``("onset", start_ms)`` iff the flag is ``True``, no run is
open, and rules (f) and (g) admit it; when it opens, the run is set exactly as for any first word. Reaching an *open*
run it is refused entirely (no extension, no arrival stamp). Everything else is BASE.

**The oracles.** Never the module under test:

- ``tests/unit/_frozen_turn_predictor_t0195.py``, BASE (341215b7) verbatim, pinned by sha256 below and re-checked
  against git when the commit is in the clone. A2 is a differential against it.
- *The rewritten stream* (A1 in one sentence): with the flag ``True`` on every call, the live endpointer on a stream
  must decide exactly what BASE decides on the same stream in which every estimated word that reaches **no open run**
  (BASE's own ``_open`` at that point, read off the frozen copy, which the tests own) is replaced by the same word
  with ``end_estimated=False``, and every other estimated word is **deleted**. "Opens like a first word" is the
  replacement; "refused entirely" is the deletion. (f) and (g) are BASE's own and apply to the replaced word as to any.
- Hand-computed numbers for the desk word (A5's shape, ``Hello,`` at 458480, the 2026-10-29 log's seq 7045).

Regime: pure arithmetic, step clock only, no session; n = 1 deterministic per hand row; the Hypothesis rows use the
default profile (the verify stage adds 10 fresh seeds, task *Regime*). Every row except the two named guards
(``test_frozen_copy_is_base_verbatim`` and ``test_the_base_row_of_t0051_is_kept_with_the_default``) is
was ``pending(task="T-0195")`` until ``/implement T-0195`` group A started (activated 2026-10-29).
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import pathlib
import subprocess
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.contracts.speech import SttEvent, SttStepEvent, SttWordEvent
from local_astra.reflex import turn_predictor as live_tp
from local_astra.reflex.turn_predictor import Endpointer

#: the frozen copy's body (from its marker line to its end marker, trailing newlines normalised to one) hashes to
#: ``git show 341215b7:local_astra/reflex/turn_predictor.py | tail -n +14 | sha256sum`` (computed 2026-10-29)
T0195_FROZEN_BODY_SHA256 = "472a1ed9cdfe552dee6428e80c7da9ff66e2831c964bb974bd454e00fc2c74d7"
T0195_BASE = "341215b7e08e2d895bb6180b2ca8eaee4fa66950"
FLAG = "may_open_on_estimate"

STEP_MS = 80
#: BASE's values, written out (``test_endpointer.test_the_constants_this_task_must_not_move`` pins them in the module)
EOT_SILENCE_MS = 800
MAX_WORD_MS = 2000
MIN_SPEECH_MS = 200


def _frozen() -> Any:
    return importlib.import_module("tests.unit._frozen_turn_predictor_t0195")


def require_flag() -> None:
    """A1's keyword, or a named assertion (never a ``TypeError`` from inside a row)."""
    params = inspect.signature(Endpointer.observe).parameters
    param = params.get(FLAG)
    assert param is not None, (
        f"T-0195 A1: Endpointer.observe(event, *, {FLAG}: bool = False) does not exist yet; "
        f"observe takes {list(params)}"
    )
    assert param.kind is inspect.Parameter.KEYWORD_ONLY and param.default is False, (
        f"T-0195 A1: {FLAG} is keyword-only with default False; got kind={param.kind}, default={param.default!r}"
    )


def heads(p: float) -> tuple[float, ...]:
    """A uniform ``prs`` vector (8 slots), so no row depends on ``PAUSE_HEAD_INDEX``."""
    return (p,) * 8


def step(t_ms: int, p: float = 0.9) -> SttStepEvent:
    return SttStepEvent(pause_heads=heads(p), t_ms=t_ms)


def word(text: str, start_ms: int, end_ms: int, *, estimated: bool = False) -> SttWordEvent:
    if estimated:
        return SttWordEvent(text=text, start_ms=start_ms, end_ms=end_ms, end_estimated=True)
    return SttWordEvent(text=text, start_ms=start_ms, end_ms=end_ms)


def run(ep: Any, events: list[SttEvent], **kw: Any) -> list[tuple[int, tuple[Any, ...]]]:
    """``(index of the triggering event, decision)`` for every event that produced one."""
    out: list[tuple[int, tuple[Any, ...]]] = []
    for i, event in enumerate(events):
        decision = ep.observe(event, **kw)
        if decision is not None:
            out.append((i, tuple(decision)))
    return out


# =====================================================================================================================
# Guards (BASE-green on purpose, no ``pending``)
# =====================================================================================================================


def test_frozen_copy_is_base_verbatim() -> None:
    """A2's oracle is only an oracle while nobody edits it: its body hashes to the pinned sha and, when the base commit
    is in the clone, equals ``turn_predictor.py`` there from its line 14 on. **BASE-green guard.**"""
    frozen = _frozen()
    text = pathlib.Path(frozen.__file__).read_text(encoding="utf-8")
    start, end = "dropped) ---\n", "# --- end of verbatim copy ---\n"
    body = text[text.index(start) + len(start) : text.index(end)].rstrip("\n") + "\n"
    assert hashlib.sha256(body.encode()).hexdigest() == T0195_FROZEN_BODY_SHA256
    assert frozen.BASE_SHA == T0195_BASE
    shown = subprocess.run(
        ["git", "show", f"{T0195_BASE}:local_astra/reflex/turn_predictor.py"],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).parent,
    )
    if shown.returncode != 0:
        pytest.skip(f"{T0195_BASE} is not in this clone; the sha256 pin above still holds")
    assert "".join(shown.stdout.splitlines(keepends=True)[13:]) == body


def test_the_base_row_of_t0051_is_kept_with_the_default() -> None:
    """A2/A9's other half, stated here so a mutation that flips the *default* is caught in this module too: with the
    keyword omitted an estimated word opens nothing, exactly as T-0051 criterion 12 says (``test_endpointer``'s
    ``test_an_estimated_end_word_never_opens_a_run`` stays unedited). **BASE-green guard.**"""
    ep = Endpointer()
    for t in (0, 80, 160):
        ep.observe(step(t))
    assert ep.observe(word("Hello,", 200, 200, estimated=True)) is None
    quiet = [ep.observe(step(240 + STEP_MS * i)) for i in range(30)]
    assert quiet == [None] * 30


# =====================================================================================================================
# A1: the rule, by hand
# =====================================================================================================================


def test_estimated_word_opens_a_run_when_allowed_and_none_is_open() -> None:
    """A1: flag ``True``, no run open, (f) and (g) admit -> ``("onset", start_ms)``. Three cases where no run is open:
    a fresh endpointer (IDLE at session start); after a completed run whose ``t_eot`` the word starts after; and the
    desk word itself (``start == end``, a 0 ms word). Each twin with the flag ``False`` opens nothing (BASE (h)), so the
    flag, and nothing else, is what opened it."""
    require_flag()

    # (1) fresh: steps, then the word
    for flag, expected in ((True, ("onset", 1200)), (False, None)):
        ep = Endpointer()
        for t in (1000, 1080, 1160, 1240):
            assert ep.observe(step(t, 0.05), **{FLAG: flag}) is None
        got = ep.observe(word("see?", 1200, 1400, estimated=True), **{FLAG: flag})
        assert (tuple(got) if got else None) == expected, (
            f"A1: fresh endpointer, flag={flag}: an estimated [1200,1400] word decides {expected}; got {got}"
        )

    # (2) after a completed run: honest "what is" [1000,1400], delivered after step 1400 (arrival 1400); heads 0.3 (the
    # EMA never reaches 0.6), so the backstop ends it at 1400 + 800 = 2200
    ep = Endpointer()
    events: list[SttEvent] = [step(t, 0.3) for t in range(1000, 1401, STEP_MS)]
    events += [word("what", 1000, 1200), word("is", 1200, 1400)]
    events += [step(t, 0.3) for t in range(1480, 2201, STEP_MS)]
    fired = [tuple(d) for e in events if (d := ep.observe(e, **{FLAG: True})) is not None]
    assert fired == [("onset", 1000), ("end_of_turn", 1400, 1.0)], (
        f"recompute the first run; got {fired}"
    )
    got = ep.observe(word("Hello,", 2500, 2500, estimated=True), **{FLAG: True})
    assert got is not None and tuple(got) == ("onset", 2500), (
        f"A1: after a decided run (t_eot 1400), an estimated word starting at 2500 > 1400 opens a new run; got {got}"
    )

    # (3) the desk word: 0 ms long, start == end == 458480 (logs/sessions/20261029-121653-2974784.jsonl seq 7045)
    ep = Endpointer()
    for t in range(458000, 459521, STEP_MS):
        ep.observe(step(t), **{FLAG: True})
    got = ep.observe(word("Hello,", 458480, 458480, estimated=True), **{FLAG: True})
    assert got is not None and tuple(got) == ("onset", 458480), (
        f"A1/A5: the desk's 'Hello,' (458480, 0 ms, end estimated) opens a run while none is open; got {got}"
    )


def test_estimated_word_never_extends_an_open_run_even_when_allowed() -> None:
    """**Adversarial** (A1, "an estimated word reaching an open run is still refused entirely"). An implementation that
    deletes (h) instead of narrowing it must fail, and so must one that keeps (h) only for the *onset*.

    Hand-computed, the T-0051 A14/A22 stream with the flag ``True`` on every call, ``L`` = 500, heads 0.30 (the EMA
    settles at 0.30, so only the backstop fires):

      words      ``what is this`` = [1000,1200] [1200,1400] [1400,1600]; the last arrives on the first step >= 2100,
                 i.e. 2160
      held word  [1700, 3400], ``end_estimated=True``, delivered right after step 2240 (inside the run), its end 1160 ms
                 in the future of the step clock; 1700 ms <= ``MAX_WORD_MS`` and no ``t_eot`` yet, so (f) and (g) admit
                 it: only (h) may refuse it
      expected   ``[("onset", 1000), ("end_of_turn", 1600, 1.0)]``, the end triggered at step 2160 + 800 = **2960**

    Deleting (h): the run extends to 3400 and the arrival to max(2240, 3400) = 3400, so ``t_eot`` 3400 at step 4200.
    Twin (premise): a second estimated word delivered *after* the decision, starting after 1600, opens a run with the
    flag (so the refusal above is about the open run, not about the word)."""
    require_flag()
    lag = 500
    spoken = [word("what", 1000, 1200), word("is", 1200, 1400), word("this", 1400, 1600)]
    arrival = {w.text: -(-(w.end_ms + lag) // STEP_MS) * STEP_MS for w in spoken}
    assert arrival == {"what": 1760, "is": 1920, "this": 2160}, f"recompute the arrivals: {arrival}"
    timeline: list[tuple[int, int, SttEvent]] = [(arrival[w.text], 2, w) for w in spoken]
    timeline += [
        (t, 1, SttStepEvent(pause_heads=heads(0.30), t_ms=t)) for t in range(0, 6001, STEP_MS)
    ]
    timeline.append((2240, 3, word("see?", 1700, 3400, estimated=True)))
    timeline.append((4400, 3, word("then", 4000, 4000, estimated=True)))
    timeline.sort(key=lambda r: (r[0], r[1]))
    events = [e for _, _, e in timeline]

    ep = Endpointer()
    fired = run(ep, events, **{FLAG: True})
    got = [(getattr(events[i], "t_ms", None), d) for i, d in fired]
    assert got[:2] == [(None, ("onset", 1000)), (2960, ("end_of_turn", 1600, 1.0))], (
        "A1: an estimated word reaching an OPEN run is refused entirely, flag or no flag: it may neither extend the "
        "run to its invented end (t_eot 1600 must stay 1600) nor stamp an arrival (the backstop stays due at "
        f"2160 + 800 = 2960). Got {got}"
    )
    assert got[2:] == [(None, ("onset", 4000)), (5200, ("end_of_turn", 4000, 1.0))], (
        "the twin, after the decision: an estimated word [4000,4000] delivered after step 4400 (> t_eot 1600, no run "
        "open) is an onset with the flag, and its run ends on the backstop at max(4400, 4000) + 800 = 5200; got "
        f"{got[2:]}"
    )


@pytest.mark.parametrize(
    ("label", "first_step", "release_after_step"),
    [
        # the desk shape: the hold publishes the word ~1 s after its audio; arrival = max(459520, 458480) = 459520
        ("released by the hold", 458000, 459520),
        # no step yet when the word is delivered: arrival = max(None -> end_ms) = 458480
        ("before any step", 458480, None),
    ],
)
def test_one_word_estimated_run_ends_on_the_backstop_with_p_end_one(
    label: str, first_step: int, release_after_step: int | None
) -> None:
    """A1/A5: the one-word run ends on the **backstop** at the first step with ``t - arrival >= EOT_SILENCE_MS``,
    deciding ``("end_of_turn", S, 1.0)``; the head path cannot decide it, even with heads at 1.0 on every step, because
    the run is 0 ms long (< ``MIN_SPEECH_MS``, ruling A8). Hand-computed, ``S`` = 458480:

      released by the hold  word after step 459520 -> arrival 459520 -> end at step **460320**
      before any step       word first, steps from 458480 -> arrival 458480 -> end at step **459280**

    An implementation that lets the head path decide a 0 ms run ends it ~4 steps after the onset with the EMA, not 1.0;
    one that stamps the arrival from the word's end alone ends the first case at 459280, 1040 ms early."""
    require_flag()
    s = 458480
    hello = word("Hello,", s, s, estimated=True)
    events: list[SttEvent] = []
    t = first_step
    if release_after_step is None:
        events.append(hello)
    while t <= 462000:
        events.append(step(t, 1.0))
        if t == release_after_step:
            events.append(hello)
        t += STEP_MS
    expected_trigger = {"released by the hold": 460320, "before any step": 459280}[label]
    ep = Endpointer()
    fired = run(ep, events, **{FLAG: True})
    got = [(getattr(events[i], "t_ms", "word"), d) for i, d in fired]
    assert got == [("word", ("onset", s)), (expected_trigger, ("end_of_turn", s, 1.0))], (
        f"A1/A5 ({label}): onset at {s}, then the backstop's end of turn ({s}, 1.0) at step {expected_trigger} and "
        f"nothing earlier (heads 1.0 throughout; a 0 ms run never ends on the heads). Got {got}"
    )


def test_an_estimated_open_resets_the_ema() -> None:
    """A1, "EMA reset" (review F1; the reviewer's stream). The previous run ends on the *heads*, so the EMA it leaves
    behind is above ``EOT_P_THRESHOLD``; an estimated word then opens the next run and must reset it, as any first word
    does (ruling A9). For constant heads ``h`` from 0 the EMA after ``n`` steps is ``h * (1 - 0.5 ** (n / 3))``
    (half-life 3 steps), so by hand:

      [0] word [0,300] (no step yet: arrival 300), steps 300..620 at 1.0: after 4 steps (540) the EMA is
          1 - 0.5 ** (4/3) = 0.6031 >= 0.6, the run is 300 ms >= MIN_SPEECH_MS -> ("end_of_turn", 300, 0.6031) at [4]
      [6] estimated [1000,1000], flag True: no run open, 1000 > t_eot 300 -> ("onset", 1000), EMA reset to 0
      [7] word [1000,1400] extends it (run 400 ms; arrival max(620, 1400) = 1400)
      steps from 1400 at 0.9: after 4 steps 0.9 * (1 - 0.5 ** (4/3)) = 0.5428 < 0.6; after 5 (step 1720, [12])
          0.9 * (1 - 0.5 ** (5/3)) = 0.6165 -> ("end_of_turn", 1400, 0.6165); the backstop (2200) is later

    Without the reset the carried 0.6031 moves to 0.6031 + α(0.9 - 0.6031) = 0.664 at the first step ([8]) and cuts
    the turn 320 ms early. The decisions also equal the rewritten-stream oracle's."""
    require_flag()
    events: list[SttEvent] = [word("what", 0, 300)]
    events += [step(t, 1.0) for t in range(300, 621, STEP_MS)]
    events += [word("Hello,", 1000, 1000, estimated=True), word("there", 1000, 1400)]
    events += [step(t, 0.9) for t in range(1400, 2401, STEP_MS)]
    got = run(Endpointer(), events, **{FLAG: True})
    first_eot = 1 - 0.5 ** (4 / 3)
    second_eot = 0.9 * (1 - 0.5 ** (5 / 3))
    assert 0.9 * (1 - 0.5 ** (4 / 3)) < 0.6 <= second_eot and first_eot >= 0.6, (
        "recompute the hand EMAs"
    )
    expected = [
        (0, ("onset", 0)),
        (4, ("end_of_turn", 300, pytest.approx(first_eot, abs=1e-9))),
        (6, ("onset", 1000)),
        (12, ("end_of_turn", 1400, pytest.approx(second_eot, abs=1e-9))),
    ]
    assert got == expected, (
        "A1: an estimated word that opens a run resets the EMA like any first word; without the reset the carried "
        f"{first_eot:.4f} ends the run at step index 8 with p_end ~0.664. Got {got}"
    )
    assert got == rewritten_oracle(events, [True] * len(events)), (
        "…and equals the rewritten-stream oracle"
    )


def test_flag_true_still_obeys_f_and_g() -> None:
    """A8 (endpointer half): with the flag ``True`` an estimated word with ``start_ms <= last t_eot`` opens nothing (the
    boundary ``==`` included), nor does one longer than ``MAX_WORD_MS``. The twins one millisecond either side open a
    run, so the refusals are (f) and (g), not a blanket refusal (and the row is red at BASE, where (h) refuses all)."""
    require_flag()

    def after_a_run() -> Any:
        ep = Endpointer()
        for e in [word("what", 1000, 1200), *[step(t) for t in range(1200, 2001, STEP_MS)]]:
            ep.observe(e, **{FLAG: True})
        return ep  # t_eot 1200, decided at step 2000 (arrival 1200 + 800)

    cases = [
        ("(f) start before t_eot", word("x", 1100, 1100, estimated=True), None),
        ("(f) start == t_eot", word("x", 1200, 1300, estimated=True), None),
        ("(f) twin: start = t_eot + 1", word("x", 1201, 1300, estimated=True), ("onset", 1201)),
        ("(g) one ms too long", word("x", 3000, 3000 + MAX_WORD_MS + 1, estimated=True), None),
        (
            "(g) twin: exactly MAX_WORD_MS",
            word("x", 3000, 3000 + MAX_WORD_MS, estimated=True),
            ("onset", 3000),
        ),
    ]
    for label, w, expected in cases:
        ep = after_a_run()
        got = ep.observe(w, **{FLAG: True})
        assert (tuple(got) if got else None) == expected, (
            f"A8 {label}: expected {expected}, got {got}"
        )


# =====================================================================================================================
# A2 and A1 as differentials (Hypothesis)
# =====================================================================================================================

HEADS = st.sampled_from([(), (0.9,), heads(0.05), heads(0.3), heads(0.9), heads(1.0)])


@st.composite
def streams(draw: Any, *, estimated: bool = True, per_call_flags: bool = False) -> Any:
    """A wire-ordered stream on one step clock: steps (with holes) and words whose spans lie at or before "now", some
    reaching back past earlier decisions (f), some longer than ``MAX_WORD_MS`` (g), some estimated (h)."""
    rows = draw(
        st.lists(
            st.one_of(
                st.tuples(st.just("step"), st.integers(1, 12), HEADS),
                st.tuples(
                    st.just("word"),
                    st.integers(0, 3000),  # how far back from now the word ends
                    st.sampled_from([0, 0, 80, 200, 640, MAX_WORD_MS, MAX_WORD_MS + 1, 5000]),
                    st.booleans() if estimated else st.just(False),
                ),
            ),
            max_size=60,
        )
    )
    t = draw(st.integers(0, 2000))
    events: list[SttEvent] = []
    for row in rows:
        if row[0] == "step":
            t += STEP_MS * row[1]
            events.append(SttStepEvent(pause_heads=row[2], t_ms=t))
        else:
            end = max(0, t - row[1])
            start = max(0, end - row[2])
            events.append(word("w", start, end, estimated=row[3]))
    flags = (
        draw(st.lists(st.booleans(), min_size=len(events), max_size=len(events)))
        if per_call_flags
        else None
    )
    return events, flags


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=streams())
def test_flag_false_is_base_on_every_stream(case: Any) -> None:
    """A2: with the keyword omitted, and with ``False`` passed on every call, ``observe`` equals BASE (the frozen
    copy) on every stream, estimated words included: the same decisions on the same events."""
    require_flag()
    events, _ = case
    base = run(_frozen().Endpointer(), events)
    assert run(Endpointer(), events) == base, "A2: the keyword omitted must be BASE bit for bit"
    assert run(Endpointer(), events, **{FLAG: False}) == base, (
        "A2: the flag False must be BASE bit for bit"
    )


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=streams(estimated=False))
def test_flag_true_is_base_on_streams_without_estimated_words(case: Any) -> None:
    """A2: with ``True`` on every call, a stream with no estimated word decides exactly what BASE decides."""
    require_flag()
    events, _ = case
    assert run(Endpointer(), events, **{FLAG: True}) == run(_frozen().Endpointer(), events)


def rewritten_oracle(
    events: list[SttEvent], flags: list[bool]
) -> list[tuple[int, tuple[Any, ...]]]:
    """A1 as a stream rewrite over BASE (see the module docstring): an estimated word whose call carries the flag and
    which reaches no open run becomes the same honest word; every other estimated word is deleted (refused entirely).
    Decisions keep the *original* event index."""
    frozen = _frozen().Endpointer()
    out: list[tuple[int, tuple[Any, ...]]] = []
    for i, (event, flag) in enumerate(zip(events, flags, strict=True)):
        if isinstance(event, SttWordEvent) and event.end_estimated:
            if not (flag and not frozen._open):
                continue  # refused entirely: BASE never sees it
            event = word(event.text, event.start_ms, event.end_ms)
        decision = frozen.observe(event)
        if decision is not None:
            out.append((i, tuple(decision)))
    return out


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=streams(per_call_flags=True))
def test_flag_true_equals_base_on_the_rewritten_stream(case: Any) -> None:
    """A1 whole: per-call flags drawn at random; the live decisions equal the rewritten-stream oracle's. This is where
    "opens exactly as any first word" (arrival via ``_arrival_of``, EMA reset, first/last) and "an estimated word
    reaching an open run is refused entirely" are checked on every stream, not just the hand rows."""
    require_flag()
    events, flags = case
    live = Endpointer()
    got = [
        (i, tuple(d))
        for i, (e, f) in enumerate(zip(events, flags, strict=True))
        if (d := live.observe(e, **{FLAG: f})) is not None
    ]
    assert got == rewritten_oracle(events, flags)


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(case=streams(per_call_flags=True))
def test_end_of_turns_never_outnumber_onsets_with_flag_true(case: Any) -> None:
    """Rule (d) and (f) survive the new opening path (the ``test_endpointer`` property, with the flag): decisions
    alternate onset, end, onset, ... starting with an onset; each end's ``t_eot`` is at or after its run's onset; and
    each onset starts strictly after the previous ``t_eot``. Run with the flag ``True`` everywhere and with random
    per-call flags; with ``True`` everywhere at least one estimated onset must be possible (checked by a fixed stream
    below, so the property is not vacuous on a no-op implementation)."""
    require_flag()
    events, flags = case
    for label, fl in (("all True", [True] * len(events)), ("random", flags)):
        ep = Endpointer()
        decisions = [
            tuple(d)
            for e, f in zip(events, fl, strict=True)
            if (d := ep.observe(e, **{FLAG: f})) is not None
        ]
        kinds = [d[0] for d in decisions]
        assert kinds == ["onset", "end_of_turn"] * (len(kinds) // 2) + ["onset"] * (
            len(kinds) % 2
        ), f"({label}) rule (d): decisions alternate starting with an onset; got {decisions}"
        last_eot: int | None = None
        onset: int | None = None
        for d in decisions:
            if d[0] == "onset":
                assert last_eot is None or d[1] > last_eot, (
                    f"({label}) rule (f): onset {d[1]} after t_eot {last_eot}"
                )
                onset = d[1]
            else:
                assert onset is not None and d[1] >= onset, (
                    f"({label}) t_eot {d[1]} before its onset {onset}"
                )
                last_eot = d[1]
    # anti-vacuity: a stream made of one estimated word opens with the flag
    ep = Endpointer()
    first = ep.observe(word("w", 10, 10, estimated=True), **{FLAG: True})
    assert first is not None and tuple(first) == ("onset", 10)


def test_constants_unchanged_by_this_task() -> None:
    """A9's escalation names, read off the live module (BASE-green guard; ``test_endpointer``'s own pin is the
    authoritative one and stays unedited)."""
    assert (live_tp.EOT_SILENCE_MS, live_tp.MAX_WORD_MS, live_tp.MIN_SPEECH_MS) == (
        EOT_SILENCE_MS,
        MAX_WORD_MS,
        MIN_SPEECH_MS,
    )
    assert (live_tp.EOT_P_THRESHOLD, live_tp.PAUSE_HEAD_INDEX, live_tp.EMA_HALF_LIFE_STEPS) == (
        0.6,
        2,
        3.0,
    )
