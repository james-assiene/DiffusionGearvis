"""The speech gate refuses to voice interface data. T-0052 group B (criteria 8-11, 15).

The second line of defence (design decision D5, ruling A2). ``SpeechGate`` is the single funnel
every committed answer string passes through on its way to Pocket TTS — ``session._segment`` feeds
it, the machine's ``TtsSpeak`` carries its ``source``, and T-0030's played-text ledger keys on that
same source — it is pure, and it already normalises rather than echoes. So the guarantee that JSON
is never spoken lives *here*, downstream of the classifier, and a future regression of the fence
carry (group A) becomes a missing surface and a counted refusal instead of a synthesiser reading
JSON aloud.

The negative oracle is the human's own session: ``tests/fixtures/llm/flight_canvases.json`` carries
the 810-character source the gate handed to TTS on 2026-09-16 and the 807 characters it said. No
row here may ever allow that string to be *voiced*.

**Refuse means conserve, not drop (ruling A20).** A refused sentence is still returned, with its
``source`` byte-exact and ``text=""`` — the idiom this module already uses for post-EOS bytes. The
turn machine polices ``"".join(sources) + remainder == text`` on every segmenter call
(``turn_machine.py`` §Segmenter) and raises ``ValueError`` — emitting *no* speech at all — when it
does not hold, and ``session.py`` swallows that exception without logging it. A gate that drops the
refused item therefore turns "the human hears 807 characters of JSON" into "the human hears nothing
and no log says why", which is strictly worse. ``test_a_refused_sentence_still_conserves_the_turn``
composes the real gate with the real machine over the real payload, which is the row that would
have caught it: the invariant row that already does this uses seven brace-free strings, so the
composed law was never exercised with anything the guard refuses.

**The guard must be narrow.** A false positive silences a real answer, which is worse than the bug
being fixed (ruling A2), so half the rows in this module assert that ordinary prose containing
braces, quotes, colons and the words *component*, *children* and *surface* is still spoken
byte-exact. Those rows are green today and their job is to stay green: they are not ``pending``,
because a guard that is too strict passes every row a "does it refuse JSON?" suite contains. Ruling
A21 added the multilingual half of that: English, French, Japanese and Korean answers that quote a
small JSON object are refused by the density rule as first implemented, and only *length* separates
that prose from a real surface — hence ``STRUCTURAL_MIN_CHARS`` at the observed size of a real A2UI
source (409 and 810 characters in the fixture), which the constants row pins from both sides.

**The density rule gets its own job (ruling A23).** Deleting it, or moving its fraction to 0.9,
0.21 or 0.05, or its floor to 0 or 100 000, all left the whole suite green: every refusal fired on
the key rule. ``test_a_long_surface_whose_keys_are_not_pinned_is_refused`` is the only thing in the
suite the density rule can catch and the key rule cannot, and
``test_a_long_quote_dense_sentence_is_still_spoken`` is the behavioural twin of the constants row's
one-sided bound on the fraction.

This module deliberately does **not** depend on group A: criterion 8 drives the *old*
``split_segments(None, text, False)`` call, standing in for any future regression of the carry.

Regime: pure CPU, n = 1 per row, no seed, no timing claimed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import pytest

from local_astra.conductor import speech_gate as speech_gate_module
from local_astra.conductor.llm_client import split_segments
from local_astra.conductor.speech_gate import Sentence, SpeechGate
from local_astra.conductor.turn_machine import TurnMachine
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from tests.fakes.llm import flight_canvases
from tests.unit.test_speech_gate import gate_segmenter

FIX = flight_canvases()
REC = FIX["recorded"]

#: The A2UI keys criterion 9 pins. ``version``/``v0.9.1`` is a key *and* a value, so the row
#: below checks the six bare keys and mentions the seventh rather than guessing its spelling.
PINNED_KEYS = frozenset(
    {"component", "surfaceId", "createSurface", "updateComponents", "catalogId", "children"}
)

#: A real A2UI fragment, 170 characters, balanced brackets so the gate's hold rule releases it as
#: one sentence. Measured structural density (``{}[]":,`` over length): 0.400. It is refused by the
#: *key* rule — it is shorter than ``STRUCTURAL_MIN_CHARS`` once that constant moves (ruling A21).
A2UI_FRAGMENT = (
    '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":'
    '[{"id":"root","component":"Column","children":["a","b"]},'
    '{"id":"a","component":"Text","text":"hi"}]}}'
)

#: Ruling A21's table: ordinary answers, in four languages, that quote a small JSON object. Every
#: one contains ``{``; none contains a pinned key; lengths 52-65; measured structural densities
#: 0.328 / 0.262 / 0.298 / 0.327, i.e. all four sit *inside* the real A2UI band (0.293-0.365) and
#: no fraction separates them from a surface. Only length does.
JSON_QUOTING_PROSE: tuple[str, ...] = (
    'The dict {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5} has five keys.',
    'Ecris {"nom": "Ada", "age": 36, "ville": "Bath"} dans le fichier.',
    'これは {"名前": "太郎", "年齢": 36, "町": "大阪"} という辞書です。次に実行してください。',
    '이것은 {"이름": "민수", "나이": 36, "도시": "부산"} 입니다. 확인해 주세요.',
)

#: A surface of the shape the gate sees, 986 characters, measured structural density 0.307 — inside
#: the band the real payloads measure (0.293-0.365) — and **no member name from the pinned set**:
#: ``widget``/``kids``/``slots``/``kind`` instead of ``component``/``children``. It is the only
#: input in the suite that the density rule can refuse and the key rule cannot (ruling A23).
NON_PINNED_SURFACE = (
    '{"id":"root","widget":"Column","kids":["row1","row2","row3"],'
    '"kind":"layout","label":"the trip planner panel"}'
) + "".join(
    '{"id":"row' + str(i) + '","widget":"TextField",'
    '"label":"departure airport or city name",'
    '"kids":["a' + str(i) + '"],"kind":"input","slots":{"lead":"a small aeroplane icon"}}'
    for i in range(1, 7)
)


def commit(text: str, idx: int = 0) -> Commit:
    return Commit(request_id="t6-gen", canvas_idx=idx, segment="answer", text=text)


def speak(gate: SpeechGate, *chunks: str) -> list[Sentence]:
    """Feed chunks and flush; every sentence the gate returned, refused ones included.

    Ruling A20: a refusal is *conserved*, not dropped — it comes back with its ``source`` intact
    and an empty ``text``. Use :func:`voiced` for the ones that actually reach the synthesiser.
    """
    out: list[Sentence] = []
    for i, chunk in enumerate(chunks):
        out.extend(gate.feed(commit(chunk, i)))
    out.extend(gate.close())
    return out


def voiced(sentences: Sequence[Sentence]) -> list[Sentence]:
    """The sentences TTS would actually say: an empty ``text`` voices nothing."""
    return [s for s in sentences if s.text != ""]


def assert_conserved(sentences: Sequence[Sentence], gate: SpeechGate, fed: str) -> None:
    """The law ``turn_machine.py`` checks on every segmenter call, asserted directly.

    ``"".join(sources) + remainder == text``. The machine raises ``ValueError`` and emits no
    ``tts.speak`` at all when this fails, and ``session.py`` swallows that exception without
    logging it (ruling A20), so a gate that drops a refused item silences the whole turn.
    """
    rebuilt = "".join(s.source for s in sentences) + gate.pending
    assert rebuilt == fed, (
        "ruling A20 / turn_machine Segmenter: the sources plus the remainder must rebuild the fed "
        f"text exactly. {len(fed) - len(rebuilt)} characters were dropped; the machine raises "
        "ValueError on this and the turn goes silent with nothing in the log"
    )


# --- discovery: the criteria pin behaviour, not spellings ----------------------------------------


def refusal_names(gate: SpeechGate) -> list[str]:
    """Names that mention a refusal. Nothing here is fetched, let alone called (ruling A15)."""
    return sorted(n for n in dir(gate) if "refus" in n.lower() and not n.startswith("__"))


def refusal_attributes(gate: SpeechGate) -> dict[str, Any]:
    """Everything the gate *exposes* about refusals, found by name rather than by guessing one.

    Criterion 10 asks for "a counter and a bounded list"; it does not name them, and the task says
    the criteria pin behaviour rather than names. Any public attribute whose name mentions a
    refusal counts, so the implementer picks the spelling and this module does not have to be
    edited when they do.

    **Ruling A15 and `.claude/rules/tests.md`: never invoke what you discovered.** An earlier
    version of this helper called every callable it found, so a private ``_refuse(self, source)``
    killed nine rows with a ``TypeError`` raised inside the helper instead of an assertion about
    the code — a test reaching into an implementation by name pattern silently dictating its shape.
    Private names are not fetched at all and callables are filtered out, so the implementer may
    name and shape its helpers however it likes.
    """
    found: dict[str, Any] = {}
    for name in refusal_names(gate):
        if name.startswith("_"):
            continue
        value = getattr(gate, name)
        if callable(value):
            continue
        found[name] = value
    return found


def refused_count(gate: SpeechGate) -> int:
    found = refusal_attributes(gate)
    counters = {n: v for n, v in found.items() if isinstance(v, int) and not isinstance(v, bool)}
    assert counters, (
        "criterion 10: the gate exposes no count of refused sentences. Nothing else can publish "
        f"the refusal (T-0054 depends on this). Names mentioning a refusal: {refusal_names(gate)}"
    )
    assert len(counters) == 1, f"more than one refusal counter is ambiguous: {counters}"
    return next(iter(counters.values()))


def refused_sources(gate: SpeechGate) -> list[str]:
    found = refusal_attributes(gate)
    lists = {
        n: list(v)
        for n, v in found.items()
        if isinstance(v, Sequence) and not isinstance(v, str) and all(isinstance(x, str) for x in v)
    }
    assert lists, (
        "criterion 10: the refused sources are not retrievable, so a refusal is indistinguishable "
        f"from text that was never committed. Names mentioning a refusal: {refusal_names(gate)}"
    )
    assert len(lists) == 1, f"more than one list of refused sources is ambiguous: {sorted(lists)}"
    return next(iter(lists.values()))


def module_fractions() -> dict[str, float]:
    return {
        name: value
        for name, value in vars(speech_gate_module).items()
        if isinstance(value, float) and 0.0 < value < 1.0
    }


def module_key_sets() -> dict[str, frozenset[str]]:
    """Module-level containers of strings, normalised so ``'"component"'`` matches ``component``."""
    out = {}
    for name, value in vars(speech_gate_module).items():
        if isinstance(value, frozenset | set | tuple | list) and value:
            members = [v for v in value if isinstance(v, str)]
            if len(members) == len(value):
                out[name] = frozenset(m.strip('":') for m in members)
    return out


def structural_min_chars(default: int | None = None) -> int | None:
    """``STRUCTURAL_MIN_CHARS``, the one constant criterion 9 names in so many words.

    Returns ``default`` when it does not exist yet, so the behavioural rows below assert what they
    are about rather than dying on a missing name; the constants row is where its absence fails.
    """
    value = getattr(speech_gate_module, "STRUCTURAL_MIN_CHARS", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


# --- criterion 8: the recorded utterance --------------------------------------------------------


def test_the_recorded_utterance_is_refused() -> None:
    """Criterion 8, the single most important row in group B.

    The path is the one that failed on 2026-09-16, reproduced exactly: the 902-character canvas 2
    through the **old** ``split_segments(None, text, False)`` call — one ``answer`` piece — into a
    fresh gate. The gate's own segmentation then makes the first sentence 810 characters of A2UI
    JSON with the model's first prose sentence glued to its tail, which is what Pocket TTS read
    out for 807 characters (``tts_utterance_text`` in the fixture, recorded at seq 2223).

    Driving the old call is deliberate: this guarantee must hold *without* group A, so a future
    regression of the fence carry is a missing surface, never a synthesiser reading JSON.

    **Ruling A20 rewrote what "refused" means here**: the sentence comes back, ``source`` byte-
    exact, ``text`` empty. Nothing is voiced and nothing is lost, which is the only shape that
    satisfies both the safety rule and the machine's conservation law.
    """
    text = REC["answer_piece_canvas2"]
    pieces = split_segments(None, text, False)
    answers = [t for segment, t, _ in pieces if segment == "answer"]
    assert answers == [text], (
        "setup: the old splitter call classified the whole canvas as one answer piece — that is "
        f"the regression this row stands in for. got {[(s, t[:40]) for s, t, _ in pieces]}"
    )

    gate = SpeechGate()
    returned = speak(gate, *answers)

    offenders = [s for s in voiced(returned) if '"component"' in s.source or '{"' in s.source]
    assert not offenders, "interface data reached the voice: " + repr(
        [(s.source[:70], s.text[:70]) for s in offenders]
    )
    assert refused_count(gate) == 1, (
        f"exactly one sentence of this canvas is interface data; the gate refused "
        f"{refused_count(gate)}"
    )
    assert refused_sources(gate) == [REC["tts_utterance_source"]], (
        "the refused source must be exactly what the gate handed to TTS on 2026-09-16 (810 "
        "characters, session log seq 2223); got " + repr([s[:60] for s in refused_sources(gate)])
    )

    # Ruling A20: conserved, not dropped. The refused item is still in the returned sequence.
    refused = [s for s in returned if s.source == REC["tts_utterance_source"]]
    assert len(refused) == 1, (
        "ruling A20: the refused sentence is still returned, with its source intact — the machine "
        "checks '\"\".join(sources) + remainder == text' and emits no speech at all when it fails, "
        "and session.py swallows that ValueError without logging it. Sources returned: "
        + repr([s.source[:40] for s in returned])
    )
    assert refused[0].text == "", (
        f"the refused sentence must voice nothing: text={refused[0].text[:80]!r}"
    )
    assert_conserved(returned, gate, text)
    for sentence in voiced(returned):
        assert '"children"' not in sentence.text and "surfaceId" not in sentence.text


def test_the_prose_in_the_same_canvas_is_still_spoken() -> None:
    """Criterion 8's second half: refusing must not silence the turn.

    Not ``pending``, and that is the point: it is green today (there is no guard yet) and its job
    is to stay green afterwards. This is criterion 8's *too wide* direction — the same canvas that
    must lose its JSON must keep its prose — and a guard that silenced this sentence would satisfy
    every "does it refuse JSON?" row in this module.

    The second sentence of canvas 2 — "You can select your departure and destination, pick dates,
    and search for available flights." — is 92 characters of ordinary prose and was utterance
    ``t6-u2`` in the log. It must still be spoken, ``source`` byte-exact, because ``source`` is
    what T-0030's played-text ledger and the §4 conservation check key on.

    *Known cost, quantified:* 56 of the 152 characters of prose in this canvas ("I've designed a
    simple flight booking interface for you.") are inside the refused sentence, because the model
    wrote them with no separator the gate could split on. Dropping them is D5's accepted price at
    sentence granularity and the reason criterion 10 makes the loss countable. Ruling A11's cost
    *model* is withdrawn by A21, but the 56-character measurement it reported stands and is why
    criterion 8's "the 152 characters are still emitted" cannot be literal on the old splitter
    path; see the test-author report.
    """
    text = REC["answer_piece_canvas2"]
    pieces = split_segments(None, text, False)
    gate = SpeechGate()
    returned = speak(gate, *[t for s, t, _ in pieces if s == "answer"])

    sources = [s.source for s in voiced(returned)]
    assert REC["tts_second_utterance_text"] in sources, (
        "the turn went silent: the ordinary prose sentence of this canvas must still be spoken "
        f"byte-exact. got {[s[:50] for s in sources]}"
    )
    said = next(s for s in returned if s.source == REC["tts_second_utterance_text"])
    assert said.text == REC["tts_second_utterance_text"], (
        f"the spoken rendering of plain prose is the prose: {said.text!r}"
    )


def test_a_refused_sentence_still_conserves_the_turn() -> None:
    """**Ruling A20, the row that would have caught the blocker.** Real gate, real turn machine.

    ``tests/unit/test_turn_invariants.py`` already drives the real gate through the real machine,
    but over seven **brace-free** commit strings, so the composed law is never exercised with
    anything the guard refuses. Measured end to end on a real ``Session`` over the fakes with the
    first implementation: ``history_spoken=['']``, ``failures=['ValueError']``, **zero** log lines
    mentioning the error — one probe left the machine stuck in SPEAKING for 20 000 ms of virtual
    time and never wrote a turn record at all.

    So this row commits the recorded canvas — JSON *and* the prose that follows it — through
    ``TurnMachine`` with the same adapter ``session._segment`` is, and asserts three things at
    once: the machine does not raise, nothing interface-shaped carries a non-empty ``text``, and
    the human still hears the answer. A gate that drops the refused item fails the first; a gate
    with no guard fails the second; a gate that refuses the whole buffer fails the third.
    """
    gate = SpeechGate()
    machine = TurnMachine(segment=gate_segmenter(gate))
    state = machine.initial()
    for event in (
        E.UserSpeechOnset(t_ms=1000, onset_ms=1000),
        E.SttWord(t_ms=1500, text="build", start_ms=1100, end_ms=1300),
        E.SttEndOfTurn(t_ms=2650, t_eot_ms=2600),
        E.SttFlushed(t_ms=2900, t_eot_ms=2600),
        E.Tick(t_ms=3100),
    ):
        state, _ = machine.step(state, event)
    rid = state.active_generate
    assert rid is not None, "setup: a generate is live once the machine is THINKING (T-0003)"

    committed = REC["answer_piece_canvas2"]
    spoken: list[Any] = []
    try:
        state, cmds = machine.step(
            state,
            E.LlmCommit(t_ms=3200, request_id=rid, canvas_idx=0, segment="answer", text=committed),
        )
        spoken.extend(c for c in cmds if c.type == "tts.speak")
        state, cmds = machine.step(state, E.LlmDone(t_ms=3400, request_id=rid))
        spoken.extend(c for c in cmds if c.type == "tts.speak")
    except ValueError as exc:  # pragma: no cover - the failure this row exists to name
        pytest.fail(
            "ruling A20: the machine refused to speak *anything* because the gate did not conserve "
            "its input. A refused sentence must come back with its source intact and an empty "
            "text; dropping it silences the whole turn and session.py swallows this exception "
            f"without logging it. {exc}"
        )

    assert spoken, "the turn produced no tts.speak at all"
    assert state.spoken_text == committed, (
        "docs/02 §4: every committed byte is accounted for in a spoken source. got "
        f"{len(state.spoken_text)} of {len(committed)} characters"
    )
    interface = [c for c in spoken if '"component"' in c.source]
    assert interface and all(c.text == "" for c in interface), (
        "the JSON of the recorded canvas must reach TTS with an empty text (or not at all as a "
        f"non-empty one): {[(c.source[:40], c.text[:40]) for c in interface]}"
    )
    said = [c.text for c in spoken if c.text]
    assert REC["tts_second_utterance_text"] in said, (
        f"the rest of the turn must still be spoken: {[t[:50] for t in said]}"
    )


# --- criteria 9, 10, 11 --------------------------------------------------------------------------


def test_the_refusal_predicate_is_stated_as_module_constants() -> None:
    """Criterion 9: the pinned key set, the density fraction and its floor are readable constants.

    A predicate buried in a regex cannot be reviewed, cannot be widened deliberately, and cannot be
    quoted in a report. The band on the fraction is one-sided and measured: quote-and-comma dense
    *dialogue* ("Yes," she said, "no," he said, …) has structural density 0.217 and must be
    spoken, so a fraction at or below that refuses real answers. The A2UI bodies from the session
    measure 0.293 (canvas 1), 0.338 (the refused sentence) and 0.365 (canvas 2's JSON tail).

    **Ruling A21 pins the floor from both sides.** Below it, the density rule must not reach the
    four ``JSON_QUOTING_PROSE`` answers (52-65 characters, densities 0.262-0.328, i.e. *inside* the
    A2UI band — only length separates them); above it, it must still reach a real surface, and the
    two the session recorded arrive at the gate as 409- and 810-character sources. A floor of 40
    sits inside ordinary prose; a floor above 409 exempts a real surface from the rule entirely.
    """
    key_sets = {n: v for n, v in module_key_sets().items() if v >= PINNED_KEYS}
    assert key_sets, (
        "no module-level constant in speech_gate holds the pinned A2UI key set "
        f"{sorted(PINNED_KEYS)} (criterion 9). String constants found: "
        + repr({n: sorted(v)[:4] for n, v in module_key_sets().items()})
    )

    fractions = module_fractions()
    assert fractions, (
        "criterion 9's structural-character fraction is not a module constant a test can read"
    )
    for name, value in fractions.items():
        assert value > 0.2, (
            f"speech_gate.{name} = {value}: a structural fraction this low refuses ordinary "
            "dialogue (measured 0.217 on the row below), i.e. it silences real answers"
        )

    floor = structural_min_chars()
    assert floor is not None, (
        "criterion 9 names STRUCTURAL_MIN_CHARS: the length below which the density rule does not "
        "apply is a constant a test can read, not a literal in the predicate"
    )
    longest_prose = max(len(s) for s in JSON_QUOTING_PROSE)
    assert floor > longest_prose, (
        f"speech_gate.STRUCTURAL_MIN_CHARS = {floor}: ruling A21 sets it at the observed size of a "
        f"real A2UI source, because the densest ordinary answers that quote JSON are "
        f"{longest_prose} characters or shorter and sit inside the A2UI density band. A floor "
        "below them silences an ordinary answer in four languages"
    )
    smallest_real_surface = len(REC["ui_piece_canvas1"])
    assert floor <= smallest_real_surface, (
        f"speech_gate.STRUCTURAL_MIN_CHARS = {floor}: the smallest A2UI source the session "
        f"recorded reaches the gate at {smallest_real_surface} characters (the other at "
        f"{len(REC['tts_utterance_source'])}). A floor above it exempts a real surface from the "
        "density rule, which is ruling A23's 'floor = 100 000' mutation with a smaller number"
    )


def test_the_refusal_is_counted_and_the_source_retrievable(
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """Criterion 10: countable, retrievable, and silent about it.

    The gate is a pure module: a refusal is not printed, not logged from inside it, and not
    raised — the caller decides what to do with it (publishing it is T-0054). Two refusals in one
    turn must both be counted, or "how often does this happen?" is unanswerable.
    """
    gate = SpeechGate()
    chunks = (A2UI_FRAGMENT + ". Here is the panel. ", A2UI_FRAGMENT + ". And that is it.")
    with caplog.at_level(logging.DEBUG):
        returned = speak(gate, *chunks)

    assert refused_count(gate) == 2, f"two refusals, counted: got {refused_count(gate)}"
    assert refused_sources(gate) == [A2UI_FRAGMENT + ". ", A2UI_FRAGMENT + ". "], (
        "the refused sources must be retrievable verbatim, in order: "
        + repr([s[:40] for s in refused_sources(gate)])
    )
    assert [s.source for s in voiced(returned)] == ["Here is the panel. ", "And that is it."], (
        f"everything else in the turn is spoken: {[s.source for s in voiced(returned)]}"
    )
    assert_conserved(returned, gate, "".join(chunks))
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == "", (
        f"a pure module must not print: {captured.out!r} {captured.err!r}"
    )
    assert caplog.records == [], (
        f"a refusal is not logged from inside the pure gate: {[r.message for r in caplog.records]}"
    )
    assert not hasattr(speech_gate_module, "logging"), (
        "speech_gate imported logging: the pure module reports through its return values and its "
        "counters, never through a logger (docs/02 §4)"
    )


def test_the_gate_continues_after_a_refusal() -> None:
    """Criterion 11: a refusal loses text; it must not lose the rest of the answer.

    The state after a refusal must be the state a spoken sentence would have left: the control run
    replaces the refused sentence with ordinary prose and the two runs must agree on the pending
    buffer, on every later sentence, and on what ``close()`` flushes. A gate that swallows the
    buffer, or latches, turns one refused sentence into a silent turn.

    Ruling A20: the two runs also agree on the *number* of sentences returned — the refused one
    comes back with an empty ``text`` — because the machine rebuilds its input from what the gate
    hands back and speaks nothing at all when a byte goes missing.
    """
    tail = ". Then the real answer. And a held clause"
    gate = SpeechGate()
    emitted = list(gate.feed(commit(A2UI_FRAGMENT + tail)))
    control = SpeechGate()
    control_emitted = list(control.feed(commit("All right" + tail)))

    assert not any(A2UI_FRAGMENT[:30] in s.source for s in voiced(emitted)), (
        f"the fragment was spoken: {[s.source[:40] for s in voiced(emitted)]}"
    )
    assert [s.source for s in emitted][1:] == [s.source for s in control_emitted][1:], (
        "after a refusal the following sentence is emitted exactly as it would have been: "
        f"{[s.source for s in emitted]} vs {[s.source for s in control_emitted]}"
    )
    assert [s.text for s in emitted][1:] == [s.text for s in control_emitted][1:], (
        "…and voiced exactly as it would have been: "
        f"{[s.text for s in emitted]} vs {[s.text for s in control_emitted]}"
    )
    assert len(emitted) == len(control_emitted) and emitted[0].text == "", (
        "ruling A20: the refused sentence is conserved with an empty text rather than dropped: "
        f"{[(s.source[:30], s.text[:30]) for s in emitted]}"
    )
    assert gate.pending == control.pending == "And a held clause", (
        f"the pending buffer must be untouched by the refusal: {gate.pending!r}"
    )
    assert [s.source for s in gate.close()] == [s.source for s in control.close()] != [], (
        "close() must still flush the held clause after a refusal"
    )


def test_a_refused_source_that_carries_the_end_of_text_token_still_latches() -> None:
    """**Ruling A24**: the refusal check runs *after* the sentence is built, and that is load-bearing.

    ``_sentence`` latches on ``<|endoftext|>``/``<eos>``/``<|think|>`` (T-0004 A6): once the model
    has stopped, every later sentence of the turn is conserved in ``source`` with an empty
    ``text``. If a refusal short-circuits before the sentence is built, a refused source carrying
    the token never latches and the prose after it is **voiced** — which CLAUDE.md forbids
    absolutely ("never voice text the LLM has not committed", and everything past EOS is not
    committed answer text). Swapping the two statements leaves the rest of the suite green, which
    is why this row exists.

    The refusal fires on the key rule, so the row is independent of the density constants.
    """
    gate = SpeechGate()
    text = A2UI_FRAGMENT + ".<|endoftext|> stray. The model kept going. And so did this."
    returned = speak(gate, text)

    assert refused_count(gate) == 1, (
        f"setup: the A2UI fragment carrying the token is one refused sentence; got "
        f"{refused_count(gate)}"
    )
    assert "<|endoftext|>" in refused_sources(gate)[0], (
        f"setup: the refused source is the one carrying the token: {refused_sources(gate)[0][-40:]!r}"
    )
    after = [s for s in returned if "kept going" in s.source or "so did this" in s.source]
    assert len(after) == 2, (
        f"setup: two sentences follow the refused one: {[s.source[:40] for s in returned]}"
    )
    assert [s.text for s in after] == ["", ""], (
        "ruling A24: the end-of-text latch must be set by the refused sentence exactly as if it "
        "had been spoken. It was not, so post-EOS prose reached the voice: "
        f"{[(s.source[:30], s.text[:40]) for s in after]}"
    )
    assert_conserved(returned, gate, text)


def test_a_long_a2ui_fragment_is_refused() -> None:
    """The boundary, refusal side: 170 characters of real A2UI in the middle of a turn.

    Paired with ``test_a_json_object_of_another_shape_in_prose_is_still_spoken`` below, which is
    the same shape without the A2UI keys and short. Between them they say where the line is: it is
    drawn around *interface data*, not around punctuation.
    """
    gate = SpeechGate()
    text = "Here it is. " + A2UI_FRAGMENT + ". Anything else?"
    returned = speak(gate, text)

    assert [s.source for s in voiced(returned)] == ["Here it is. ", "Anything else?"], (
        f"the A2UI fragment must not be spoken: {[s.source[:40] for s in voiced(returned)]}"
    )
    assert refused_count(gate) == 1
    assert_conserved(returned, gate, text)


def test_a_long_surface_whose_keys_are_not_pinned_is_refused() -> None:
    """**Ruling A23**: the density rule's own job — the one thing the key rule cannot do.

    Every other refusal in this module fires on the key rule, which is why deleting the density
    branch (or setting its fraction to 0.9, or its floor to 100 000) left the whole suite green in
    two independent mutation runs. This row is a 986-character surface of exactly the shape the
    gate saw on 2026-09-16 — ``updateComponents``-style nesting, id/label/kind members, measured
    structural density 0.307, inside the 0.293-0.365 band the real payloads measure — whose member
    names are deliberately ``widget``/``kids``/``slots`` instead of ``component``/``children``.
    The setup assertion proves the key rule cannot fire on it; the ``{`` conjunct (ruling A9) is
    satisfied; only the density rule is left.

    A model that invents its own component vocabulary, or a future A2UI version that renames a
    key, produces exactly this, and it must not be read aloud.
    """
    for key in PINNED_KEYS | {"version"}:
        assert f'"{key}"' not in NON_PINNED_SURFACE, (
            f"setup: this row exists to exercise the density rule alone, but {key!r} is in the "
            "surface, so the key rule would refuse it and the row would prove nothing"
        )
    for name, members in module_key_sets().items():
        if members >= PINNED_KEYS:
            assert not any(f'"{m}"' in NON_PINNED_SURFACE for m in members), (
                f"setup: speech_gate.{name} matches this surface, so the key rule would fire"
            )
    floor = structural_min_chars(default=0)
    assert floor is not None and len(NON_PINNED_SURFACE) > floor, (
        f"setup: the surface is {len(NON_PINNED_SURFACE)} characters and "
        f"STRUCTURAL_MIN_CHARS is {floor}: the density rule never applies to it, so the floor is "
        "above the size of a real surface (the session's arrive at 409 and 810 characters)"
    )

    gate = SpeechGate()
    text = "Here is the panel. " + NON_PINNED_SURFACE + ". Tell me what to change."
    returned = speak(gate, text)

    assert [s.source for s in voiced(returned)] == [
        "Here is the panel. ",
        "Tell me what to change.",
    ], (
        "a surface using unpinned member names was voiced: the density rule is the only thing "
        f"that can catch it. {[(s.source[:50], len(s.text)) for s in voiced(returned)]}"
    )
    assert refused_count(gate) == 1, (
        f"the surface must be counted as one refusal: got {refused_count(gate)}"
    )
    assert_conserved(returned, gate, text)


# --- criterion 15: the guard is narrow (green today, and must stay green) ------------------------


@pytest.mark.parametrize("prose", JSON_QUOTING_PROSE, ids=["en", "fr", "ja", "ko"])
def test_an_answer_that_quotes_a_small_json_object_is_spoken_in_full(prose: str) -> None:
    """**Ruling A21, the narrowness row the density rule owes.** Four ordinary answers, four
    languages, each quoting a small JSON object — the commonest thing a tutor says about data.

    **This row fails against the constants as first implemented** (fraction 0.25, floor 40) and
    passes once ``STRUCTURAL_MIN_CHARS`` moves to the observed size of a real A2UI source. That is
    the whole point: restricted to sources containing ``{`` — the only domain the rule applies to —
    the two populations *overlap*. Measured densities here are 0.328 / 0.262 / 0.298 / 0.327
    against a real A2UI band of 0.293-0.365: **no fraction separates them**, only length does
    (52-65 characters here; 409 and 810 for the surfaces the session recorded).

    The design is explicitly not English-only, and the cost of a false positive is not one
    sentence in Japanese and Korean: ``_split`` terminates on ASCII ``.!?`` only, so ``。`` is not
    a sentence end and the whole answer arrives as one source (ruling A22, queued separately).
    The row therefore asserts conservation and full voicing rather than a sentence count.
    """
    gate = SpeechGate()
    returned = speak(gate, prose)

    if refusal_attributes(gate):
        assert refused_count(gate) == 0, (
            "an ordinary answer that quotes a small JSON object was refused, so the density rule "
            f"reaches ordinary prose: {prose!r} ({len(prose)} characters, "
            f"STRUCTURAL_MIN_CHARS={structural_min_chars()})"
        )
    assert_conserved(returned, gate, prose)
    said = " ".join(s.text for s in returned).strip()
    assert said, f"the answer was silenced entirely: {prose!r}"
    for fragment in prose.split("}")[-1].split():
        assert fragment.strip(".,") in said, (
            f"the answer must be spoken in full; {fragment!r} is missing from {said!r}"
        )


def test_a_long_quote_dense_sentence_that_mentions_a_brace_is_still_spoken() -> None:
    """**Ruling A23**, the fraction's lower bound as behaviour rather than as an assertion about
    a constant. Setting the fraction to 0.21 or to 0.05 left the whole suite green; this row is
    what those mutations must turn red.

    One sentence, longer than ``STRUCTURAL_MIN_CHARS`` whatever the implementer sets it to (it is
    built from the constant), containing a literal ``{…}`` so the ``{`` conjunct is satisfied, and
    quote-and-comma dense in the way reported dialogue really is: measured structural density
    0.216 at its shortest and 0.236 asymptotically, i.e. just above ruling A9's 0.217 "densest
    ordinary prose" and comfortably below the 0.293-0.365 the real surfaces measure.

    Not ``pending``: green today, green after, red for any fraction at or below ordinary dialogue.
    """
    head = "Braces {see below} aside, the note reads:"
    clause = ' "yes," she said, "no," he said, "maybe," "later," "fine," we all agreed,'
    floor = structural_min_chars(default=0) or 0
    text = head + clause * 4
    while len(text) <= floor:
        text += clause
    text = text[:-1] + "."

    gate = SpeechGate()
    returned = speak(gate, text)

    assert [s.source for s in voiced(returned)] == [text], (
        "a long, quote-dense ordinary sentence was refused or re-cut: the density fraction is at "
        f"or below reported dialogue. {[(s.source[:40], len(s.text)) for s in voiced(returned)]}"
    )
    assert_conserved(returned, gate, text)


@pytest.mark.parametrize(
    "prose",
    [
        "The component you asked about is the timer.",
        "Use the set notation open brace x close brace to write it.",
        "The component tree uses { and } as delimiters, but the children stay on the surface.",
        'In JSON you would write it as {"a": 1}, but in speech I would just say a is one.',
        '"Yes," she said, "no," he said, "maybe," they said, "later," we said.',
        "The surface area is two hundred square metres: the children measured it, and the "
        "component parts add up.",
    ],
)
def test_prose_about_components_and_braces_is_still_spoken(prose: str) -> None:
    """Criterion 15, adversarial. **A false positive here silences a real answer**, which is a
    worse bug than the one being fixed (ruling A2) — and it is invisible: nobody files a bug
    saying "the assistant did not say the thing it never said".

    Six sentences a user could actually get back, each carrying something a lazy guard keys on:
    the word *component*, the words *children* and *surface*, literal ``{`` and ``}``, a small
    JSON object, and quote-and-comma dense dialogue (structural density 0.217, the highest of any
    ordinary prose measured for this task — the A2UI bodies from the session measure 0.293-0.365).

    Not ``pending``: it passes today and must still pass afterwards. A guard that is too strict
    satisfies every "does it refuse JSON?" row in this module, which is exactly why this row is
    written as an equality on ``source`` and ``text`` rather than as "something was spoken".
    """
    gate = SpeechGate()
    returned = speak(gate, prose)

    assert [s.source for s in returned] == [prose], (
        f"ordinary prose was refused or re-cut: {[s.source for s in returned]}"
    )
    assert [s.text for s in returned] == [prose], (
        f"the spoken rendering of ordinary prose is the prose: {[s.text for s in returned]}"
    )
    assert refusal_attributes(gate) == {} or refused_count(gate) == 0, (
        f"the guard refused ordinary prose: {prose!r}"
    )


def test_a_json_object_of_another_shape_in_prose_is_still_spoken() -> None:
    """The boundary, spoken side (criterion 9): a short object with no A2UI key is not interface
    data. Not pending — it is green today and the guard must not change it.
    """
    prose = 'The config is {"a": 1} and nothing more.'
    gate = SpeechGate()
    returned = speak(gate, prose)
    assert [s.source for s in voiced(returned)] == [prose], (
        f"a JSON object of another shape, in prose, is an answer: {[s.source for s in returned]}"
    )


@pytest.mark.parametrize(
    "text",
    [
        '",:[]",:[]",:[]",:[]",:[]",:[]",:[]",:[]",:[]",:[]. ok',
        "a,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,, b.",
        "((([[[:::,,,;;;'''\"\"\")))]]] and then some words follow here.",
    ],
)
def test_punctuation_dense_text_the_conservation_property_generates_is_still_conserved(
    text: str,
) -> None:
    """Adversarial, criterion 15 and the "too wide" direction of the guard.

    ``tests/unit/test_speech_gate.py`` line ~145 draws Hypothesis text from the alphabet
    ``abcdefgij .,!?;:'\\"()[]-/~_`\\n0123456789`` in up to five chunks of 40 characters and asserts
    byte-for-byte conservation; criterion 15 requires that module to keep passing **unedited**.
    Five of that alphabet's characters (``,``, ``:``, ``"``, ``[``, ``]``) are in criterion 9's
    structural set and **none** of them is ``{``, so a density rule applied on its own — without
    also requiring a JSON object opener — refuses inputs that property can generate (these three
    measure 0.93, 0.87 and 0.53) and turns a settled property test flaky. That is not a reason to
    weaken criterion 9; it is the shape the predicate has to have.

    Not pending: green today, and its whole value is going red if the guard is written too wide.
    """
    gate = SpeechGate()
    returned = speak(gate, text)
    assert "".join(s.source for s in returned) == text, (
        "conservation (docs/03 §Speech gate, T-0003 criterion 10): every byte fed must come back "
        f"in a source. got {[s.source for s in returned]}"
    )
    assert [s.text for s in returned] != [""] * len(returned), (
        "the guard silenced text drawn from the settled conservation property's own alphabet, "
        f"which contains no '{{': {[s.source[:40] for s in returned]}"
    )
