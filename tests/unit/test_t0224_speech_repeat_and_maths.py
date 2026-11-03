"""T-0224: speech never voices a repeated reply, and maths markup is spoken as words.

Found in the T-0223 GPU run, turn P4 r1 (session log ``logs/t0223-handwritten-9064532/sessions/
20261101-151122-375268.jsonl``): after its answer and card the model wrote self-talk in the answer
channel and then re-emitted its whole diagnosis, and the gate voiced it (``tts.utterance`` u9-u11);
several spoken sentences carried LaTeX (``$x squared$``, ``$2\\times2$`` ...) that nothing stripped.

Every expected value here comes from the task's acceptance criteria, the hand-written tables below,
the P4 r1 log (copied in as literals; this module never reads ``logs/``), or the frozen copy of the
gate at main a0b43c98 (``_frozen_speech_gate_t0224_base.py``) for criterion 3's "voices exactly
what it voices on main". Nothing is computed by re-running the new gate's own logic.

**Group A, the repeat stop (criteria 1-3).** The repeat rule as the task states it: within one
turn, an answer sentence whose normalised form (casefold, whitespace collapsed, trailing punctuation
dropped) equals that of a sentence already *voiced* in the turn, and at least 20 characters long,
is refused with reason ``"repeat"``, and so is every later answer sentence of the turn. Ruling R-3
widens "equals" to "equals, or contains word-bounded, an earlier voiced key of >= 20 characters";
R-2 names the reason list ``SpeechGate.silenced_reasons`` and R-1 logs it as ``"reasons"``. "Refused"
is the voice guard's shape (T-0052 ruling A20): the sentence comes back with ``source`` byte-exact
and ``text == ""``, it is counted on the gate's refusal counter and its source is retrievable, and
nothing enters history. The 20-character boundary itself is not pinned (the task does not say
whether it is measured before or after dropping the trailing punctuation); rows sit well clear of
it on both sides (<= 16 characters voiced, >= 30 refused).

**Group B, maths markup (criteria 4-6).** The rule for a ``$`` (this module's ruling, stated
because criterion 4 needs one and the task asks for it): a ``$…$`` (or ``$$…$$``) is a maths span
only when the opening ``$`` is followed by a non-space character and the closing ``$``, in the same
sentence, is preceded by a non-space character and not followed by a digit (Pandoc's
``tex_math_dollars`` rule). Any other ``$`` is a currency sign in plain prose and is left exactly as
it is today: "It costs $5." stays "It costs $5." (Pocket TTS reads it), and "$5 and $10" is not a
maths span because the second ``$`` follows a space. Inside a maths span the commands are spoken as
words per criterion 4; outside one, prose is today's prose (braces in prose are pinned byte-exact by
``test_speech_gate_voice_guard.py``'s criterion-15 rows, which criterion 3 keeps green).

Not pinned, because the task does not say (stated in the T-0224 test report): LaTeX commands,
``^``/``_`` *outside* a ``$`` span (``4.9t^2`` in bare prose); a lone unmatched backtick; the
content of a backtick span that is itself maths. (Criterion 5's history is ruled by R-5
(history keeps the committed source).)

Regime: pure CPU, n = 1 per row unless a Hypothesis row says otherwise, no timing claimed.
"""

from __future__ import annotations

import re
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, example, given, settings
from hypothesis import strategies as st

from local_astra.conductor.speech_gate import SpeechGate
from local_astra.contracts.llm import Commit
from tests.unit import _frozen_speech_gate_t0224_base as frozen
from tests.unit._hud_rig import answer, frame, running
from tests.unit._session_log_rig import SPEECH_REFUSED, make_log, of_kind, read_records
from tests.unit.test_speech_gate_voice_guard import refused_count, refused_sources

# --- helpers -------------------------------------------------------------------------------------


def commit(text: str, idx: int = 0) -> Commit:
    return Commit(request_id="t1-gen", canvas_idx=idx, segment="answer", text=text)


def speak(gate: Any, *chunks: str) -> list[Any]:
    """Feed the chunks as successive canvases, flush, return every sentence (refused included),
    asserting the turn machine's conservation law at every step (T-0004 A1, T-0052 A20)."""
    out: list[Any] = []
    consumed = ""
    for i, chunk in enumerate(chunks):
        out.extend(gate.feed(commit(chunk, i)))
        consumed += chunk
        rebuilt = "".join(s.source for s in out) + gate.pending
        assert rebuilt == consumed, (
            "conservation: sources plus pending must rebuild every byte fed; the turn machine "
            f"raises ValueError otherwise and the turn goes silent. lost {len(consumed) - len(rebuilt)}"
        )
    out.extend(gate.close())
    assert "".join(s.source for s in out) == consumed and gate.pending == ""
    return out


def voiced(sentences: list[Any]) -> list[Any]:
    return [s for s in sentences if s.text != ""]


def norm(sentence: str) -> str:
    """The task's normalised form: casefold, whitespace collapsed, trailing punctuation dropped."""
    collapsed = " ".join(sentence.casefold().split())
    return collapsed.rstrip(".!?,;:\"')]} ")


# --- criterion 1: the repeat stop ------------------------------------------------------------------

FORCE = "The force on the block is nine point eight newtons."  # 51 characters
DIVIDE = "Then you divide it by the total mass of both objects."  # 53
NEW = "This sentence is new and comes after the repeat."  # 48


def test_a_repeated_sentence_and_every_later_one_are_refused() -> None:
    """Criterion 1: the repeat and every later answer sentence are refused, conserved, counted."""
    gate = SpeechGate()
    text = f"{FORCE} {DIVIDE} {FORCE} {NEW}"
    out = speak(gate, text)

    assert [s.source for s in out] == [f"{FORCE} ", f"{DIVIDE} ", f"{FORCE} ", NEW], (
        f"setup: segmentation is unchanged by this task; got {[s.source for s in out]}"
    )
    assert [s.text for s in out] == [FORCE, DIVIDE, "", ""], (
        "criterion 1: the repeated sentence and every later answer sentence of the turn are "
        f"refused (text ''); got {[s.text for s in out]}"
    )
    assert refused_count(gate) == 2, (
        f"criterion 1: refused 'in the same way the voice guard refuses' - each refusal is "
        f"counted; got {refused_count(gate)}"
    )
    assert refused_sources(gate) == [f"{FORCE} ", NEW], (
        f"criterion 1: the refused sources are retrievable byte-exact; got {refused_sources(gate)}"
    )
    assert silenced_reasons(gate) == ["repeat", "repeat"], (
        "criterion 1 / R-2: each refusal's reason, index-matched to refused_sources"
    )


def silenced_reasons(gate: Any) -> list[str]:
    """R-2: ``SpeechGate.silenced_reasons``, index-matched to ``refused_sources``. Read, never
    called; a missing attribute is an assertion, not an ``AttributeError``."""
    reasons = getattr(gate, "silenced_reasons", None)
    assert isinstance(reasons, list), (
        f"R-2: the gate exposes no list silenced_reasons; got {reasons!r}"
    )
    return list(reasons)


def test_each_refusal_keeps_its_own_reason() -> None:
    """R-2: interface data refused by the T-0052 guard says ``interface_data``; the repeat after it
    says ``repeat``. The JSON refusal is not voiced, so it is not an earlier voiced sentence."""
    gate = SpeechGate()
    json_sentence = (
        '{"version":"v0.9.1","updateComponents":{"surfaceId":"s","components":'
        '[{"id":"root","component":"Column","children":["a"]}]}}. '
    )
    out = speak(gate, f"{FORCE} ", json_sentence, f"{FORCE} {NEW}")
    assert [s.text for s in voiced(out)] == [FORCE], f"got {[s.text for s in out]}"
    assert refused_sources(gate) == [json_sentence, f"{FORCE} ", NEW]
    assert silenced_reasons(gate) == ["interface_data", "repeat", "repeat"], (
        f"R-2: reasons index-matched to the refused sources; got {silenced_reasons(gate)}"
    )


def test_a_repeat_differing_only_in_case_whitespace_and_final_punctuation_is_refused() -> None:
    """Criterion 1's normalised form: casefold, whitespace collapsed, trailing punctuation dropped."""
    gate = SpeechGate()
    variant = "the FORCE on  the block\nis nine point eight Newtons!"
    out = speak(gate, f"{FORCE} {DIVIDE} ", f"{variant} ", NEW)
    assert [s.text for s in out] == [FORCE, DIVIDE, "", ""], (
        f"a repeat that differs only in case, whitespace and its final mark was voiced: "
        f"{[s.text for s in out]}"
    )


def test_a_repeat_split_across_canvases_is_refused() -> None:
    """Adversarial, the failure mode of P4 r1: the repeat arrives cut at arbitrary canvas edges,
    including mid-word, so a check on commit text rather than on gate sentences misses it."""
    gate = SpeechGate()
    out = speak(
        gate,
        f"{FORCE} {DIVIDE}",
        " The force on the bl",
        "ock is nine point ei",
        "ght newtons. And then",
        f" {NEW}",
    )
    assert [s.text for s in voiced(out)] == [FORCE, DIVIDE], (
        f"a repeat split across canvases was voiced: {[s.text for s in voiced(out)]}"
    )


def test_a_repeat_flushed_by_close_is_refused() -> None:
    """Criterion 1 holds on the held clause too: an unterminated repeat released by ``close()``."""
    gate = SpeechGate()
    out = speak(gate, f"{FORCE} {DIVIDE} The force on the block is nine point eight newtons")
    assert [s.text for s in voiced(out)] == [FORCE, DIVIDE], (
        f"the repeat released by close() was voiced: {[s.text for s in out]}"
    )


def test_short_repeated_sentences_are_still_voiced() -> None:
    """Criterion 1, the too-wide direction: "Yes." "OK." and a 16-character sentence repeat
    freely. Green today and must stay green (a too-wide rule silences real answers)."""
    gate = SpeechGate()
    text = "Yes. That is right. Yes. OK. Check line four. OK. Check line four. Good."
    out = speak(gate, text)
    expected = [
        "Yes.",
        "That is right.",
        "Yes.",
        "OK.",
        "Check line four.",
        "OK.",
        "Check line four.",
        "Good.",
    ]
    assert [s.text for s in out] == expected, f"a short repeat was refused: {[s.text for s in out]}"


def test_a_later_sentence_embedding_an_earlier_voiced_one_is_refused() -> None:
    """R-3: a word-bounded containment of an earlier voiced key of >= 20 characters is a repeat
    (P4 r1's u9 glued the repeat to the component listing). Accepted limit, per the ruling."""
    gate = SpeechGate()
    first = "You have a mistake in the formula."  # key 33 characters
    out = speak(gate, f"{first} {DIVIDE} All good.You have a mistake in the formula. {NEW}")
    assert [s.text for s in out] == [first, DIVIDE, "", ""], f"got {[s.text for s in out]}"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # the later sentence is *contained in* the earlier one, not the other way round (R-3)
        (
            "You have a mistake in the formula for acceleration.",
            "You have a mistake in the formula.",
        ),
        # the earlier key IS a substring of the later one, but its word boundary fails
        # ("costs five" + "fold"), review T1
        (
            "The ticket for the show costs five.",
            "The ticket for the show costs fivefold more now.",
        ),
        # R-7: a contained key that does not end the candidate is a correction, not a repeat
        ("Your answer is forty two.", "Check: your answer is forty two point five."),
        (
            "The derivative of x squared is 2x.",
            "So the derivative of x squared is 2x, and you add the constant.",
        ),
        # an earlier key under 20 characters embedded in a later sentence
        ("Check line four.", "Check line four and line five of the proof."),
        # an internal comma is not trailing punctuation
        ("The net force divided by the total mass.", "The net force, divided by the total mass."),
        # one word different
        (
            "The tension in the rope is thirty six newtons.",
            "The tension in the rope is forty one newtons.",
        ),
    ],
    ids=[
        "contained-in-earlier",
        "substring-not-word-bounded",
        "r7-correction",
        "r7-continues",
        "short-key",
        "internal-comma",
        "one-word",
    ],
)
def test_a_similar_but_unequal_sentence_is_voiced(first: str, second: str) -> None:
    """Criterion 1 / R-3 are equality or word-bounded containment of an earlier voiced key of
    >= 20 characters, not similarity: each of these is voiced. Green today, must stay green."""
    gate = SpeechGate()
    out = speak(gate, f"{first} {second} {NEW}")
    assert [s.text for s in out] == [first, second, NEW], (
        f"an unequal sentence was refused as a repeat: {[s.text for s in out]}"
    )


def test_r7_a_correction_and_the_sentence_after_it_are_voiced() -> None:
    """R-7: containment counts only when the contained key ends the candidate (or sits just before
    ``.!?``). A correction that extends an earlier sentence is voiced, and nothing latches."""
    text = "Your answer is forty two. Check: your answer is forty two point five. Then carry on."
    out = speak(SpeechGate(), text)
    assert [s.text for s in out] == [
        "Your answer is forty two.",
        "Check: your answer is forty two point five.",
        "Then carry on.",
    ], f"R-7: all three are voiced; got {[s.text for s in out]}"


def test_a_gate_reused_after_close_starts_a_new_turn() -> None:
    """Review T3: ``close()`` ends the turn (the module's contract: a gate reused for the next one
    starts clean), so the repeat state resets with it."""
    gate = SpeechGate()
    first = speak(gate, f"{FORCE} {DIVIDE}")
    second = speak(gate, f"{FORCE} {DIVIDE}")
    assert [s.text for s in first] == [s.text for s in second] == [FORCE, DIVIDE], (
        f"a reused gate refused the next turn's sentences: {[s.text for s in second]}"
    )


def test_a_maths_markup_repeat_is_refused_on_its_spoken_text() -> None:
    """Review T4: the repeat key is the cleaned (spoken) text, so one sentence said once with
    ``$x^2$`` and once as "x squared" is a repeat."""
    text = (
        "The slope is $x^2$ at the top of the hill. "
        f"The slope is x squared at the top of the hill. {NEW}"
    )
    out = speak(SpeechGate(), text)
    assert [s.text for s in out] == ["The slope is x squared at the top of the hill.", "", ""], (
        f"got {[s.text for s in out]}"
    )


def test_embedding_a_refused_interface_data_source_is_not_a_repeat() -> None:
    """Review T4: a refused interface-data source was never voiced, so a later sentence embedding it
    is refused as interface data (it is one), not as a repeat, and nothing latches after it."""
    card = '{"id":"root","component":"Column","children":["a"]}.'
    gate = SpeechGate()
    out = speak(gate, f"{card} The card {card[:-1]} is done. {NEW}")
    assert [s.text for s in voiced(out)] == [NEW], f"got {[(s.source, s.text) for s in out]}"
    assert silenced_reasons(gate) == ["interface_data", "interface_data"], silenced_reasons(gate)


@pytest.mark.parametrize(
    "prose",
    [
        r"Open C:\Users\james\Desktop\file.blend now.",
        r'File "C:\Program Files\Blender Foundation\Blender 4.2\4.2\scripts\startup\bl_ui'
        r'\space_view3d.py", line 12, in draw.',
        r"Type \n for a new line.",
        "Press ^C to stop it.",
        "That face ^_^ means happy.",
    ],
    ids=["windows-path", "blender-traceback", "backslash-n", "caret-c", "caret-face"],
)
def test_backslashes_and_carets_outside_a_span_are_mains_text(prose: str) -> None:
    """Review / R-6: outside a ``$`` or backtick span nothing is rewritten, so text with
    backslashes and carets is voiced exactly as the frozen main gate voices it."""
    new = speak(SpeechGate(), prose)
    old = speak(frozen.SpeechGate(), prose)
    assert [(s.source, s.text) for s in new] == [(s.source, s.text) for s in old]


def test_a_repeat_in_another_turn_is_voiced() -> None:
    """Criterion 1 says "within one turn": a fresh gate (one gate is one turn) voices a sentence
    another turn already voiced. Green today; catches state kept beyond the gate."""
    first = speak(SpeechGate(), f"{FORCE} {DIVIDE}")
    second = speak(SpeechGate(), f"{FORCE} {DIVIDE}")
    assert [s.text for s in first] == [s.text for s in second] == [FORCE, DIVIDE]


async def test_the_repeat_refusal_is_recorded_with_its_reason_and_kept_out_of_history(
    tmp_path: Path,
) -> None:
    """Criterion 1 end to end through a real ``Session`` over the fakes: the repeat is not handed
    to TTS, a ``speech.refused`` record carries the reason ``"repeat"``, and history holds the
    first occurrence only. The reason is
    ``"repeat"`` in the record's ``reasons`` (R-1)."""
    log = make_log(tmp_path / "logs")
    async with running(
        tmp_path / "data",
        script=[answer(f"{FORCE} {DIVIDE} ", canvas_idx=0), answer(f"{FORCE} {NEW}", canvas_idx=1)],
        frames=[frame(0, "a")],
        ms_per_word=100,
        session_log=log,
    ) as rig:
        await rig.ask("what is the force on the block")
        await rig.speak_out()
        spoken = list(rig.tts.texts())
        history = list(rig.state.history)

    assert spoken == [FORCE, DIVIDE], f"criterion 1: TTS was handed {spoken}"
    refusals = of_kind(read_records(Path(log.path)), SPEECH_REFUSED)
    assert refusals, "criterion 1: no speech.refused record for the repeat"
    data = refusals[-1]["data"]
    assert data.get("reasons") == ["repeat", "repeat"], (
        f"criterion 1 / R-1: the record carries the reasons; got {data}"
    )
    assert history, "setup: the turn left no history entry"
    said = history[-1].assistant_spoken
    assert said.count("nine point eight") == 1 and "comes after the repeat" not in said, (
        f"criterion 1: nothing refused enters history; history holds {said!r}"
    )


# --- criterion 2: P4 r1, replayed ---------------------------------------------------------------
#
# The five ``llm.commit`` records with ``segment == "answer"`` of turn t1 (seq 120, 145, 158, 174,
# 191), byte-exact, copied from the session log once. The ``tts.utterance`` sources u1-u11 below are
# the same log's; u5 and u8 were refused by the T-0052 voice guard (``speech.refused`` x2).

P4_COMMITS = (
    "You have a mistake in the formula for acceleration. You divided the net force by just the "
    "second mass, but you should divide it by the total mass of both objects. The correct formula "
    "is net force divided by m1 plus m2.\n\n",
    ' (Wait, I have an extra closing brace in the JSON. Let me fix that.)\n\nCorrected JSON:\n`{"vers'
    'ion":"v0.9.1","updateComponents":{"surfaceId":"s","components":[{"id":"root","component":"Colu'
    'mn","children":["h","c1"]},{"id":"',
    'h","component":"Text","text":"Correct Acceleration Calculation","variant":"h3"},{"id":"c1","co'
    'mponent":"Card","children":["f1","f2","f3"]},{"id":"f1","component":"Text","text":"Formula: a '
    '= (m2 - m1)g / (m1 + m2)"},{"id":"f2","component":"Text","text":"a = 19.6N / (3kg + 5kg)"},{"i'
    'd":"f3","component":"Text","text":"a = 19.6 / 8 = 2.45 m/s\\u00b2"}]}}`\n\nWait, the instructi'
    'ons say "write the whole surface once, in your reply, after you have finished thinking".\n\nLe'
    "t's re-verify:\n",
    '\n\nWait, I see one extra brace at the end of my previous block. Let me be careful.\n`{"versio'
    'n":"v0.9.1","updateComponents":{"surfaceId":"s","components":[{"id":"root","component":"Colum'
    'n","children":["h","c1"]},{"id":"h","component":"Text","text":"Correct Acceleration Calculati'
    'on","variant":"h3"},{"id":"c1","component":"Card","children":["f1","f2","f3"]},{"id":"f1","co'
    'mponent":"Text","text":"Formula:',
    ' a = (m2 - m1)g / (m1 + m2)"},{"id":"f2","component":"Text","text":"a = 19.6N / (3kg + 5kg)"},'
    '{"id":"f3","component":"Text","text":"a = 19.6 / 8 = 2.45 m/s\\u00b2"}]}}` -> This is correct.'
    "\n\nFinal check on components:\nroot (Column) -> h, c1\nh (Text)\nc1 (Card) -> f1, f2, f3\nf1 ("
    "Text)\nf2 (Text)\nf3 (Text)\nAll good.You have a mistake in the formula for acceleration. You d"
    "ivided the net force by just the second mass, but you should divide it by the total mass of b"
    "oth objects. The correct formula is net force divided by m1 plus m2.\n\n",
)

U1 = "You have a mistake in the formula for acceleration. "
U2 = (
    "You divided the net force by just the second mass, but you should divide it by the total "
    "mass of both objects. "
)
U3 = "The correct formula is net force divided by m1 plus m2.\n\n"
U4 = " (Wait, I have an extra closing brace in the JSON. Let me fix that.)\n\n"
U6 = "Let's re-verify:\n\n\nWait, I see one extra brace at the end of my previous block. "
U7 = "Let me be careful.\n"
U9 = (
    "Final check on components:\nroot (Column) -> h, c1\nh (Text)\nc1 (Card) -> f1, f2, f3\n"
    "f1 (Text)\nf2 (Text)\nf3 (Text)\nAll good.You have a mistake in the formula for acceleration. "
)
U10 = U2
U11 = U3
#: What the log says was voiced on main, in order (u5 and u8 were refused, so absent).
P4_VOICED_ON_MAIN = [U1, U2, U3, U4, U6, U7, U9, U10, U11]


def test_the_p4_fixture_reproduces_the_logged_utterances_on_main() -> None:
    """Setup for criterion 2: the frozen main gate, fed the five copied commits, voices exactly the
    nine sources the log recorded and refuses two (u5, u8). Green today and after: it proves the
    literals are faithful, so criterion 2's row is about the gate and not about a typo."""
    gate = frozen.SpeechGate()
    out = speak(gate, *P4_COMMITS)
    assert [s.source for s in voiced(out)] == P4_VOICED_ON_MAIN
    assert gate.refused_count == 2


def test_p4_r1_voices_the_diagnosis_once_and_not_its_repeat() -> None:
    """Criterion 2 with R-3. The first diagnosis (u1-u3) is voiced; the self-talk before the first
    repeat (u4, u6, u7) is still voiced (the task's accepted limit); the JSON stays refused; u9,
    whose source glues the component listing to the diagnosis's first sentence ("All good.You have
    a mistake ..."), is the first repeat by R-3's word-bounded containment, and it, u10 and u11 are
    refused.
    """
    gate = SpeechGate()
    out = speak(gate, *P4_COMMITS)
    said = [s.source for s in voiced(out)]
    assert said[:6] == [U1, U2, U3, U4, U6, U7], (
        f"criterion 2: the diagnosis and the self-talk before the first repeat are voiced; got {said}"
    )
    assert said == [U1, U2, U3, U4, U6, U7], (
        f"criterion 2 / R-3: none of u9-u11's repeat is voiced; voiced {said}"
    )
    assert all(not s.text for s in out if '"component"' in s.source), "the JSON stays refused"
    tail = out[-3:]
    assert [s.source for s in tail] == [U9, U10, U11] and [s.text for s in tail] == ["", "", ""], (
        f"criterion 2: u9-u11 come back refused (source kept, text ''); got {tail}"
    )
    assert silenced_reasons(gate) == [
        "interface_data",
        "interface_data",
        "repeat",
        "repeat",
        "repeat",
    ]


# --- criterion 3: no repeat, no change -----------------------------------------------------------

WORD = st.text(alphabet="abcdefghijklmnop", min_size=1, max_size=8)
LONG = st.builds(
    lambda words, end: " ".join(words).capitalize() + end,
    st.lists(WORD, min_size=1, max_size=8),
    st.sampled_from([".", "!", "?"]),
)
SHORT_POOL = ("Yes.", "OK.", "Right.", "Check line four.", "Good!", "Hmm?")
SENTENCE = st.one_of(LONG, st.sampled_from(SHORT_POOL))
SEP = st.sampled_from([" ", "  ", "\n", "\n\n"])


#: Canvas edges, as offsets clipped to the text: a turn cut into up to six commits.
CUTS = st.lists(st.integers(0, 400), max_size=5)


def chunked(cuts: list[int], text: str) -> list[str]:
    bounds = [0, *sorted({min(c, len(text)) for c in cuts}), len(text)]
    return [text[a:b] for a, b in pairwise(bounds)]


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
@given(st.lists(st.tuples(SENTENCE, SEP), min_size=1, max_size=8), CUTS)
# T-0232 R-6: 14 letters but 20 normalised characters, a repeat by the rule's own measure; the old
# 15-letter filter let it through and the row went red on main.
@example([("Yes.", " "), ("Aaaa a a a aa aaaa a.", " "), ("Aaaa a a a aa aaaa a.", " ")], [])
def test_a_turn_without_a_repeat_voices_exactly_what_main_voices(
    pairs: Any, cuts: list[int]
) -> None:
    """Criterion 3, the differential against the frozen main gate (a0b43c98). Turns built from
    plain prose (no maths markup, no braces, no backticks), cut at arbitrary canvas edges, where no
    sentence whose normalised form (``norm``) is 20 characters or more, spaces included, repeats
    or is embedded in a later one (R-3; the rule's own measure, T-0232 R-6): sources, texts and the refusal count are identical.
    Short sentences ("Yes.", "Check line four.") do repeat here, so a rule that is too wide fails
    this row. Green today; must stay green. n = 200 examples, Hypothesis default seed.
    """
    text = "".join(s + sep for s, sep in pairs)
    chunks = chunked(cuts, text)
    base_gate = frozen.SpeechGate()
    base = speak(base_gate, *chunks)
    seen: list[str] = []
    for s in voiced(base):
        key = norm(s.source)
        # R-3 measured as the rule measures it (T-0232 R-6): no earlier normalised key of 20+
        # characters, spaces included, equal to or anywhere inside this one (conservative: any
        # containment, not only R-7's sentence-final one)
        assume(not any(len(k) >= 20 and k in key for k in seen))
        seen.append(key)

    new_gate = SpeechGate()
    new = speak(new_gate, *chunks)
    assert [(s.source, s.text) for s in new] == [(s.source, s.text) for s in base]
    assert refused_count(new_gate) == base_gate.refused_count


#: Long sentences (all >= 35 characters) the repeat property draws from, with their variants.
LONG_POOL = (
    "The force on the block is nine point eight newtons.",
    "You divided by the second mass only, not the total.",
    "The tension in the rope is thirty six newtons.",
)


def _variant(sentence: str, case: str, end: str, spaced: bool) -> str:
    body = sentence[:-1]
    body = {"same": body, "upper": body.upper(), "lower": body.lower()}[case]
    if spaced:
        body = body.replace(" ", "  ", 1)
    return body + end


POOL_SENTENCE = st.one_of(
    st.builds(
        _variant,
        st.sampled_from(LONG_POOL),
        st.sampled_from(["same", "upper", "lower"]),
        st.sampled_from([".", "!", "?"]),
        st.booleans(),
    ),
    st.sampled_from(SHORT_POOL),
)


@settings(max_examples=200, deadline=None, report_multiple_bugs=False)
@given(st.lists(st.tuples(POOL_SENTENCE, SEP), min_size=1, max_size=8), CUTS)
@example([(LONG_POOL[0], " "), (LONG_POOL[0], " ")], [])  # the P4 shape
def test_everything_from_the_first_long_repeat_on_is_refused(pairs: Any, cuts: list[int]) -> None:
    """Criterion 1 as a property, with a hand oracle: the gate's sentences are exactly the drawn
    ones (no abbreviation, decimal or bracket in the pool), so the expected voiced prefix is every
    sentence before the first one whose normalised form equals, or word-bounded contains (R-3), an
    earlier sentence's of at least 30 characters - all pool sentences are either >= 35 or <= 16
    characters, so the 20-character boundary is never what decides. Sources stay byte-exact at
    every step (``speak``); a canvas edge inside a separator may move whitespace to the next
    source (main's behaviour), so the setup check compares stripped sources.
    n = 200 examples plus the forced P4-shaped example.
    """
    sentences = [s for s, _ in pairs]
    text = "".join(s + sep for s, sep in pairs)
    out = [s for s in speak(SpeechGate(), *chunked(cuts, text)) if s.source.strip()]
    assert [s.source.strip() for s in out] == sentences, "setup: segmentation"

    first_repeat = len(sentences)
    for i, s in enumerate(sentences):
        earlier = [norm(t) for t in sentences[:i] if len(norm(t)) >= 30]
        if any(k == norm(s) or re.search(rf"\b{re.escape(k)}\b", norm(s)) for k in earlier):
            first_repeat = i
            break
    # the voiced text is the sentence with its whitespace collapsed (normalise_for_tts, unchanged)
    expected = [" ".join(s.split()) if i < first_repeat else "" for i, s in enumerate(sentences)]
    assert [s.text for s in out] == expected, (
        f"criterion 1: voiced up to the first long repeat, refused from it on; got "
        f"{[s.text for s in out]}"
    )


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.text(alphabet="ab $\\^_{}`()=+2x.!? \n", min_size=0, max_size=40), min_size=1, max_size=5
    )
)
@example(["Both $x^2$ and ", "$y^2$ are. Both $x^2$ and $y^2$ are. Then $5."])
def test_conservation_holds_with_maths_markup_and_repeats(chunks: list[str]) -> None:
    """Criterion 3 / the §4 conservation law under the new rules: with maths markup, backticks,
    currency signs and repeats in the alphabet, sources plus pending rebuild every byte fed, at every
    step (asserted inside ``speak``). Green today; must stay green. n = 200 examples."""
    speak(SpeechGate(), *chunks)


# --- criteria 4-6: maths markup spoken as words ---------------------------------------------------

#: (committed answer sentence, the text TTS must be handed). Hand-written from criteria 4 and 6.
MATHS_ROWS: list[tuple[str, str]] = [
    # criterion 6, the five T-0223 examples
    ("So the derivative of $x squared$ is easy.", "So the derivative of x squared is easy."),
    ("The derivative is $2x \\cos(x squared)$.", "The derivative is 2x cos(x squared)."),
    ("That is $2\\times2$ in total.", "That is 2 times 2 in total."),
    ("The normal force is $mg \\cos \\theta$.", "The normal force is mg cos theta."),
    ("Solve $20 = 5t + 4.9t^2$ for t.", "Solve 20 = 5t + 4.9t squared for t."),
    # criterion 4
    ("We get $$a = b$$ at once.", "We get a = b at once."),
    ("Both $x^2$ and $y^2$ are positive.", "Both x squared and y squared are positive."),
    ("The value is $\\frac{\\sqrt{2}}{2}$ here.", "The value is square root of 2 over 2 here."),
    ("So a is $\\frac{19.6}{8}$ metres per second.", "So a is 19.6 over 8 metres per second."),
    (
        "So $a = \\frac{(m_2 - m_1)g}{m_1 + m_2}$ here.",
        "So a = (m 2 - m 1)g over m 1 + m 2 here.",
    ),
    ("Take $\\sqrt{x}$ first.", "Take square root of x first."),
    ("Then $x^3$ grows fast.", "Then x cubed grows fast."),
    ("Then $x^{n}$ grows.", "Then x to the power n grows."),
    ("Then $x^n$ grows.", "Then x to the power n grows."),
    ("Mass $m_1$ goes up.", "Mass m 1 goes up."),
    ("Mass $m_{1}$ goes up.", "Mass m 1 goes up."),
    ("Use $\\sin x + \\tan x$ now.", "Use sin x + tan x now."),
    ("Here $\\log n$ and $\\ln x$ appear.", "Here log n and ln x appear."),
    ("Then $a \\cdot b$ is the product.", "Then a times b is the product."),
    ("The vector $\\vec v$ points up.", "The vector vec v points up."),
    ("So ${x}$ is fine.", "So x is fine."),
    ("Type `ls` and press Enter.", "Type ls and press Enter."),
    ("Call `foo_bar` now.", "Call foo bar now."),
    # R-6: rewrites happen inside spans only, and an unwrapped backtick span is one
    ("Write `{x}` here.", "Write x here."),
    ("Write {x} and `{y}` here.", "Write {x} and y here."),
    *[
        (f"The angle $\\{g}$ is small.", f"The angle {g} is small.")
        for g in ("theta", "alpha", "beta", "lambda", "mu", "pi", "omega")
    ],
]


@pytest.mark.parametrize(("raw", "spoken"), MATHS_ROWS, ids=[r for r, _ in MATHS_ROWS])
def test_maths_markup_is_spoken_as_words(raw: str, spoken: str) -> None:
    """Criteria 4-6: the text handed to TTS; criterion 5: the committed ``source`` keeps the
    model's own text byte-exact (the session log, cards and the §4 invariant read that)."""
    out = speak(SpeechGate(), raw)
    assert [s.source for s in out] == [raw], f"criterion 5: source is the model's text; {out}"
    assert [s.text for s in out] == [spoken], (
        f"criterion 4/6: TTS was handed {[s.text for s in out]}"
    )


def test_capital_omega_is_spoken_as_its_name() -> None:
    """Criterion 4 names ``\\Omega``; the case of the spoken name is not pinned."""
    out = speak(SpeechGate(), "The unit is $\\Omega$ here.")
    assert [s.text.casefold() for s in out] == ["the unit is omega here."], (
        f"got {[s.text for s in out]}"
    )


@pytest.mark.parametrize(
    "prose",
    [
        "It costs $5.",
        "It costs $5-$10 a month.",
        "Between $5 and $10 a month.",
        "Pay $5, not $50, for it.",
        "The angle θ is small and x² is positive.",
        "√2 is irrational.",
        "Write the set as {1, 2, 3} on the board.",
    ],
    ids=[
        "currency",
        "price-range",
        "two-prices",
        "price-comma",
        "unicode-greek-sup",
        "unicode-root",
        "braces",
    ],
)
def test_currency_unicode_maths_and_prose_braces_are_left_as_they_are(prose: str) -> None:
    """Criterion 4's limits and this module's ``$`` rule: a currency sign that opens no maths span,
    Unicode maths already in plain text, and braces in prose outside a span are today's text.
    Green today; must stay green."""
    out = speak(SpeechGate(), prose)
    assert [s.text for s in out] == [prose], f"plain prose was rewritten: {[s.text for s in out]}"


MATHS_TOKENS = st.sampled_from(
    [
        "\\frac{a}{b}",
        "\\sqrt{x}",
        "\\theta",
        "\\cos",
        "\\times",
        "\\cdot",
        "\\foo",
        "x^2",
        "x^3",
        "x^{n}",
        "y^n",
        "m_1",
        "m_{2}",
        "{y}",
        "2",
        "+",
        "=",
        "(a)",
    ]
)


@settings(max_examples=200, deadline=None, report_multiple_bugs=False)
@given(st.lists(MATHS_TOKENS, min_size=1, max_size=6), st.sampled_from(["", " "]), CUTS)
@example(["x^2"], " ", [])
def test_no_maths_delimiter_or_command_reaches_tts(
    tokens: list[str], glue: str, cuts: list[int]
) -> None:
    """Criterion 4 as a property: a well-formed maths span (by this module's ``$`` rule) inside a
    sentence leaves no ``$``, backslash, brace, caret or underscore in the text handed to TTS,
    however the turn is cut into canvases. n = 200 examples."""
    span = glue.join(tokens)
    text = f"We have ${span}$ here. And `ls` too."
    out = speak(SpeechGate(), *chunked(cuts, text))
    for s in voiced(out):
        assert not set(s.text) & set("$\\{}^_`"), (
            f"markup reached TTS: {s.text!r} from {s.source!r}"
        )


async def test_tts_gets_the_spoken_words_and_history_keeps_the_committed_source(
    tmp_path: Path,
) -> None:
    """Criterion 5 as ruled by R-5, end to end through a real ``Session`` over the fakes: TTS is
    handed the cleaned text; the session log's committed text keeps the model's ``$x^2$``; history
    keeps the committed *source* of what was voiced (only the text handed to TTS is cleaned)."""
    raw = "The derivative is $x^2$ here."
    log = make_log(tmp_path / "logs")
    async with running(
        tmp_path / "data",
        script=[answer(raw, canvas_idx=0)],
        frames=[frame(0, "a")],
        ms_per_word=100,
        session_log=log,
    ) as rig:
        await rig.ask("what is the derivative")
        await rig.speak_out()
        spoken = list(rig.tts.texts())
        history = list(rig.state.history)

    assert spoken == ["The derivative is x squared here."], f"criterion 4/5: TTS got {spoken}"
    commits = [r["data"]["text"] for r in read_records(Path(log.path)) if r["kind"] == "llm.commit"]
    assert raw in "".join(commits), f"criterion 5: the log keeps the committed text; got {commits}"
    assert history and history[-1].assistant_spoken.strip() == raw, (
        "criterion 5 / R-5: history keeps the committed source of what was voiced; got "
        f"{history[-1].assistant_spoken if history else None!r}"
    )
