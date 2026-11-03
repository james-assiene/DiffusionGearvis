"""Turn machine: the edge policies — late words, PAUSED, errors, history (docs/02 §4). Task T-0003.

Companion of ``tests/unit/test_turn_invariants.py`` (the §4 table and the Hypothesis properties);
this module holds the rows that need long event sequences to be readable: word attribution by
audio time across a turn boundary, the PAUSED resume/interrupt policy, the error rows of
criterion 20 and the history bound of criterion 13.

Every test names the acceptance criterion of ``docs/tasks/T-0003.md`` (including its
``## Amendments``) it comes from. Nothing here is derived from an implementation — there is none.
Conventions, as in the sibling module: assertions name the *required* commands and their relative
order, never an exact command list, so an implementation stays free to add ``hud.state`` or timer
commands (criteria preamble, amendment A3). Fields and constructor parameters that T-0003
introduces are read through the helpers below, so a missing one fails as a named criterion instead
of an ``AttributeError`` half-way through a test.
"""

from __future__ import annotations

import inspect

import pytest

from local_astra.conductor.speech_gate import SpeechGate
from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.contracts.turn import TurnState

POLICY = TurnPolicy(thinking=True)

#: The three commands §4 calls idempotent per interruption (criteria 18 and 21).
INTERRUPT_COMMANDS = {"llm.abort", "tts.cancel", "playback.flush"}

#: Criterion 20(b) / amendment A5: the states in which a generate is live and an error ends the turn.
SPEECH_STATES = ("speaking", "paused", "interrupted")

#: What the STT client reports when the transcription socket dies (criterion 20(c)).
STT_DEAD = "transcription socket closed"

_MISSING = object()


def kinds(cmds) -> list[str]:
    return [c.type for c in cmds]


def run(machine: TurnMachine, events, state=None):
    """Apply events in order; return (state, all commands)."""
    state = state if state is not None else machine.initial()
    out = []
    for ev in events:
        state, cmds = machine.step(state, ev)
        out.extend(cmds)
    return state, out


def policy_field(name: str, expected):
    """Read a ``TurnPolicy`` field T-0003 introduces, with a readable message when it is absent."""
    value = getattr(POLICY, name, _MISSING)
    assert value is not _MISSING, f"T-0003 adds TurnPolicy.{name} (default {expected!r})"
    assert value == expected, f"TurnPolicy.{name} default should be {expected!r}, got {value!r}"
    return value


def next_words(state):
    """``MachineState.next_words`` (criterion 4; amendment A1 makes this name contract)."""
    buf = getattr(state, "next_words", _MISSING)
    assert buf is not _MISSING, "criterion 4 (A1) adds MachineState.next_words, and it is contract"
    return buf


def segment_parameter():
    """The keyword-only ``segment`` parameter of ``TurnMachine.__init__`` (criterion 10)."""
    param = inspect.signature(TurnMachine.__init__).parameters.get("segment")
    assert param is not None, "criterion 10: TurnMachine(policy, *, segment=...) injection point"
    assert param.kind is inspect.Parameter.KEYWORD_ONLY, "criterion 10: segment is keyword-only"
    return param


def commit(s, text, *, idx=0, segment="answer", open_=False, t=3200):
    return E.LlmCommit(
        t_ms=t,
        request_id=s.active_generate,
        canvas_idx=idx,
        segment=segment,
        text=text,
        thought_open=open_,
    )


def drive_to_thinking(
    machine, *, onset_ms=1000, t_eot_ms=2600, words=(("what", 1100), ("is", 1400)), state=None
):
    """IDLE -> THINKING (criteria 1, 3, 6). Returns ``(state, commands, llm_start_t_ms)``.

    With the defaults the transcript is ``"what is"``, the turn's audio window is 1000..2600 ms
    and ``llm.start`` is emitted at t = 3020 ms, so a commit at t = 3200 ms and every event this
    module sends afterwards are still in the future.
    """
    eot_t_ms = t_eot_ms + 50
    flush_t_ms = eot_t_ms + 250
    start_t_ms = flush_t_ms + machine.policy.eot_settle_ms
    events = [
        E.UserSpeechOnset(t_ms=onset_ms, onset_ms=onset_ms),
        *[
            E.SttWord(t_ms=start + 400, text=text, start_ms=start, end_ms=start + 200)
            for text, start in words
        ],
        E.SttEndOfTurn(t_ms=eot_t_ms, t_eot_ms=t_eot_ms),
        E.SttFlushed(t_ms=flush_t_ms, t_eot_ms=t_eot_ms),
        E.Tick(t_ms=start_t_ms),
    ]
    state, cmds = run(machine, events, state)
    return state, cmds, start_t_ms


def sentence_segmenter(text: str, *, final: bool):
    """Reference segmenter that *conserves* text: complete sentences end at ``.``, tail held.

    ``"".join(source) + remainder == text`` by construction, so criterion 10's conservation check
    passes and a test can watch where the held tail surfaces.
    """
    out: list[tuple[str, str]] = []
    rest = text
    while (i := rest.find(".")) != -1:
        cut = i + 1
        while cut < len(rest) and rest[cut] == " ":
            cut += 1
        out.append((rest[:cut], rest[:cut].strip()))
        rest = rest[cut:]
    if final and rest:
        out.append((rest, rest.strip()))
        rest = ""
    return out, rest


def honest_until_the_flush(text: str, *, final: bool):
    """Conserves text on every commit and lies only on the ``final=True`` flush.

    The commit path is already policed by the sibling module's lying-segmenter test; this one
    exists because ``llm.done`` calls the segmenter a second time (criterion 12) and that call is
    the remaining way for text the LLM never committed to reach ``tts.speak``.
    """
    if not final:
        return sentence_segmenter(text, final=False)
    return [("Call 1-800-EXAMPLE now", "Call 1-800-EXAMPLE now")], ""


def state_at(machine, where: str, *, answer="Zebra crossing."):
    """Drive ``machine`` into ``where``; each path is a §4 row the sibling module pins down."""
    if where == "idle":
        return machine.initial()
    if where == "listening":
        s, _ = machine.step(machine.initial(), E.UserSpeechOnset(t_ms=1000, onset_ms=1000))
        return s
    if where == "endpointing":
        s, _ = run(
            machine,
            [
                E.UserSpeechOnset(t_ms=1000, onset_ms=1000),
                E.SttWord(t_ms=1500, text="what", start_ms=1100, end_ms=1300),
                E.SttEndOfTurn(t_ms=2650, t_eot_ms=2600),
            ],
        )
        return s
    s, _, _ = drive_to_thinking(machine)
    if where == "thinking":
        return s
    s, _ = machine.step(s, commit(s, answer))
    if where == "speaking":
        return s
    s, _ = machine.step(s, E.UserSpeechOnset(t_ms=3400, onset_ms=3400))
    if where == "paused":
        return s
    if where == "interrupted":
        s, _ = machine.step(s, E.UserTakeover(t_ms=3900, onset_ms=3400))
        return s
    raise AssertionError(f"no path to {where!r}")


# --- criterion 4: word attribution is by audio time ---------------------------------------------


class TestLateWords:
    """Criterion 4 / amendment A1: ``next_words`` carries a word across the turn boundary."""

    def test_word_after_eot_seeds_the_next_turn_transcript(self):
        """A word whose audio starts after t_eot belongs to the *next* turn, not this one."""
        m = TurnMachine(POLICY)
        s, _, _ = drive_to_thinking(m)
        assert s.state is TurnState.THINKING
        assert [w.text for w in s.words] == ["what", "is"]

        # delivered while THINKING, audio time 2700 > t_eot 2600: not this turn's transcript
        s, cmds = m.step(s, E.SttWord(t_ms=3100, text="really", start_ms=2700, end_ms=2900))
        assert [w.text for w in next_words(s)] == ["really"]
        assert [w.text for w in s.words] == ["what", "is"], "audio after t_eot is not this turn"
        assert not any(isinstance(c, C.LlmStart) for c in cmds), "a late word starts nothing"

        s, _ = m.step(s, E.LlmDone(t_ms=3200, request_id=s.active_generate))
        assert s.state is TurnState.IDLE
        assert [w.text for w in next_words(s)] == ["really"], "the buffer survives the turn end"

        s, _ = m.step(s, E.UserSpeechOnset(t_ms=4000, onset_ms=4000))
        assert s.state is TurnState.LISTENING
        assert [w.text for w in s.words] == ["really"], "the buffer seeds the new turn"
        assert next_words(s) == (), "and empties"

        s, cmds = run(
            m,
            [
                E.SttWord(t_ms=4600, text="now", start_ms=4100, end_ms=4300),
                E.SttEndOfTurn(t_ms=4700, t_eot_ms=4650),
                E.SttFlushed(t_ms=4800, t_eot_ms=4650),
                E.Tick(t_ms=4800 + POLICY.eot_settle_ms),
            ],
            s,
        )
        assert s.state is TurnState.THINKING
        start = next(c for c in cmds if isinstance(c, C.LlmStart))
        assert start.transcript == "really now"

    def test_the_same_late_word_delivered_twice_is_not_duplicated(self):
        """Adversarial: a re-delivered word is one audio segment, so it is one word."""
        m = TurnMachine(POLICY)
        s, _, _ = drive_to_thinking(m)
        late = {"text": "really", "start_ms": 2700, "end_ms": 2900}
        s, _ = m.step(s, E.SttWord(t_ms=3100, **late))
        s, _ = m.step(s, E.SttWord(t_ms=3150, **late))  # same audio, delivered again
        s, _ = m.step(s, E.LlmDone(t_ms=3200, request_id=s.active_generate))
        s, _ = m.step(s, E.UserSpeechOnset(t_ms=4000, onset_ms=4000))
        s, cmds = run(
            m,
            [
                E.SttWord(t_ms=4600, text="now", start_ms=4100, end_ms=4300),
                E.SttEndOfTurn(t_ms=4700, t_eot_ms=4650),
                E.SttFlushed(t_ms=4800, t_eot_ms=4650),
                E.Tick(t_ms=4800 + POLICY.eot_settle_ms),
            ],
            s,
        )
        start = next(c for c in cmds if isinstance(c, C.LlmStart))
        assert start.transcript == "really now", "one audio segment, one word in the transcript"

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            (("is", 1400, 1850), ("what", 1100, 1900)),
            (("what", 1100, 1900), ("is", 1400, 1850)),
        ],
        ids=["delivered_in_reverse", "t_ms_descending_while_start_ms_ascends"],
    )
    def test_out_of_order_delivery_does_not_reorder_the_transcript(self, first, second):
        """Criterion 6: the transcript is ``" ".join(...)`` in ``start_ms`` order.

        The first case breaks an implementation that appends in delivery order, the second one
        breaks an implementation that sorts by delivery time ``t_ms``. Both must read "what is".
        """
        m = TurnMachine(POLICY)
        s, cmds = run(
            m,
            [
                E.UserSpeechOnset(t_ms=1000, onset_ms=1000),
                *[
                    E.SttWord(t_ms=t_ms, text=text, start_ms=start, end_ms=start + 200)
                    for text, start, t_ms in (first, second)
                ],
                E.SttEndOfTurn(t_ms=2650, t_eot_ms=2600),
                E.SttFlushed(t_ms=2900, t_eot_ms=2600),
                E.Tick(t_ms=2900 + POLICY.eot_settle_ms),
            ],
        )
        assert s.state is TurnState.THINKING
        start = next(c for c in cmds if isinstance(c, C.LlmStart))
        assert start.transcript == "what is"
        assert [w.text for w in s.words] == ["what", "is"]

    def test_word_buffers_are_capped_at_max_buffered_words(self):
        """Criteria 4 and 23: 5000 words, oldest dropped first, both buffers bounded."""
        cap = policy_field("max_buffered_words", 256)
        n = 5000

        listening = TurnMachine(POLICY)
        s, _ = listening.step(listening.initial(), E.UserSpeechOnset(t_ms=0, onset_ms=0))
        for i in range(n):
            s, _ = listening.step(
                s, E.SttWord(t_ms=i * 10 + 5, text=f"w{i}", start_ms=i * 10, end_ms=i * 10 + 5)
            )
            assert len(s.words) <= cap, f"state.words grew past the cap after {i + 1} words"
        assert [w.text for w in s.words] == [f"w{i}" for i in range(n - cap, n)]

        frozen = TurnMachine(POLICY)
        s2, _ = run(
            frozen, [E.UserSpeechOnset(t_ms=0, onset_ms=0), E.SttEndOfTurn(t_ms=100, t_eot_ms=50)]
        )
        for i in range(n):
            s2, _ = frozen.step(
                s2,
                E.SttWord(
                    t_ms=200 + i * 10, text=f"n{i}", start_ms=100 + i * 10, end_ms=105 + i * 10
                ),
            )
            assert len(next_words(s2)) <= cap, f"next_words grew past the cap after {i + 1} words"
        assert [w.text for w in next_words(s2)] == [f"n{i}" for i in range(n - cap, n)]
        assert s2.words == (), "every one of those words starts after t_eot"


# --- criteria 16-18: the PAUSED policy ----------------------------------------------------------


class TestPaused:
    def test_paused_holds_commits_and_speaks_them_on_resume_in_order(self):
        """Criteria 16 and 17: synthesis is not spent while a takeover could still cancel it."""
        segment_parameter()
        m = TurnMachine(POLICY, segment=sentence_segmenter)
        s, _, _ = drive_to_thinking(m)
        s, cmds = m.step(s, commit(s, "First sentence. "))
        assert s.state is TurnState.SPEAKING
        assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == ["First sentence. "]

        s, cmds = m.step(s, E.UserSpeechOnset(t_ms=3400, onset_ms=3400))
        assert s.state is TurnState.PAUSED
        assert kinds(cmds)[0] == "playback.pause", "nothing may precede the pause (B4 budget)"

        s, c2 = m.step(s, commit(s, "Second sentence. ", idx=1, t=3500))
        s, c3 = m.step(s, commit(s, "Third sentence. ", idx=2, t=3600))
        assert s.state is TurnState.PAUSED, "a commit does not end the pause"
        assert not [c for c in c2 + c3 if isinstance(c, C.TtsSpeak)], "no synthesis while paused"
        assert s.committed_answer == "First sentence. Second sentence. Third sentence. "
        assert s.spoken_text == "First sentence. ", "only the pre-pause sentence has been spoken"

        s, cmds = m.step(s, E.UserBackchannel(t_ms=3700, duration_ms=250))
        assert s.state is TurnState.SPEAKING
        speaks = [c for c in cmds if isinstance(c, C.TtsSpeak)]
        assert [c.source for c in speaks] == ["Second sentence. ", "Third sentence. "]
        k = kinds(cmds)
        assert k.index("cancel_timer") < k.index("playback.resume") < k.index("tts.speak")
        assert [c.name for c in cmds if isinstance(c, C.CancelTimer)] == ["takeover"]
        assert "takeover" not in [t.name for t in s.timers]
        assert s.spoken_text == s.committed_answer
        assert len({c.utterance_id for c in speaks}) == 2

    def test_paused_offset_longer_than_resume_max_interrupts_without_a_tick(self):
        """Criterion 18: ``user.speech_offset(speech_ms >= resume_max_ms)`` is enough on its own.

        ``takeover_ms`` is raised to 2000 ms so the only rule that can fire at t = 4050 ms is the
        offset one (the takeover deadline would be 5400 ms, and no ``Tick`` is delivered anyway).
        """
        policy = TurnPolicy(thinking=True, takeover_ms=2000)
        m = TurnMachine(policy)
        s, _, _ = drive_to_thinking(m)
        s, _ = m.step(s, commit(s, "Zebra crossing."))
        s, _ = m.step(s, E.UserSpeechOnset(t_ms=3400, onset_ms=3400))
        assert s.state is TurnState.PAUSED
        rid = s.active_generate

        long_enough = policy.resume_max_ms + 50
        s, cmds = m.step(s, E.UserSpeechOffset(t_ms=4050, offset_ms=4050, speech_ms=long_enough))
        assert s.state is TurnState.INTERRUPTED, "no Tick is needed: the offset carries the length"
        k = kinds(cmds)
        assert k.index("llm.abort") < k.index("tts.cancel") < k.index("playback.flush")
        assert next(c for c in cmds if isinstance(c, C.LlmAbort)).request_id == rid
        armed = [c for c in cmds if isinstance(c, C.ArmTimer) and c.name == "interrupt_timeout"]
        assert len(armed) == 1 and armed[0].at_ms == 4050 + policy.interrupt_timeout_ms
        assert ("interrupt_timeout", 4050 + policy.interrupt_timeout_ms) in [
            (t.name, t.at_ms) for t in s.timers
        ]

    def test_paused_short_offset_resumes_only_when_the_resume_window_elapses(self):
        """Criterion 17(b): a short offset arms the resume; the ``Tick`` performs it.

        ``takeover_ms`` is raised to 2000 ms so the resume tick at 4500 ms cannot be mistaken for
        criterion 18(c)'s takeover deadline.
        """
        policy = TurnPolicy(thinking=True, takeover_ms=2000)
        m = TurnMachine(policy)
        s, _, _ = drive_to_thinking(m)
        s, _ = m.step(s, commit(s, "Zebra crossing."))
        s, _ = m.step(s, E.UserSpeechOnset(t_ms=3400, onset_ms=3400))

        short = policy.resume_max_ms - 100
        s, cmds = m.step(s, E.UserSpeechOffset(t_ms=3900, offset_ms=3900, speech_ms=short))
        assert s.state is TurnState.PAUSED, "the offset alone does not resume"
        assert "playback.resume" not in kinds(cmds)
        assert not (INTERRUPT_COMMANDS & set(kinds(cmds)))

        s, cmds = m.step(s, E.Tick(t_ms=3900 + policy.resume_max_ms - 1))
        assert s.state is TurnState.PAUSED, "the resume window has not elapsed"
        assert "playback.resume" not in kinds(cmds)

        s, cmds = m.step(s, E.Tick(t_ms=3900 + policy.resume_max_ms))
        assert s.state is TurnState.SPEAKING
        assert "playback.resume" in kinds(cmds)
        assert not (INTERRUPT_COMMANDS & set(kinds(cmds)))

    def test_paused_word_without_backchannel_interrupts(self):
        """Criterion 18: words during the pause are real speech — unless their audio predates it."""
        m = TurnMachine(POLICY)
        s, _, _ = drive_to_thinking(m)
        s, _ = m.step(s, commit(s, "Zebra crossing."))
        s, _ = m.step(s, E.UserSpeechOnset(t_ms=3400, onset_ms=3400))
        assert s.state is TurnState.PAUSED
        rid = s.active_generate

        # 0.5 s STT lag: this word's audio is from before the pause, so it is not the barge-in
        s, cmds = m.step(s, E.SttWord(t_ms=3500, text="trace", start_ms=3100, end_ms=3300))
        assert s.state is TurnState.PAUSED, "a word whose audio predates the pause is not speech"
        assert not (INTERRUPT_COMMANDS & set(kinds(cmds)))

        s, cmds = m.step(s, E.SttWord(t_ms=3900, text="actually", start_ms=3500, end_ms=3700))
        assert s.state is TurnState.INTERRUPTED, "a word inside the pause, no backchannel seen"
        k = kinds(cmds)
        assert k.index("llm.abort") < k.index("tts.cancel") < k.index("playback.flush")
        assert next(c for c in cmds if isinstance(c, C.LlmAbort)).request_id == rid
        armed = [c for c in cmds if isinstance(c, C.ArmTimer) and c.name == "interrupt_timeout"]
        assert len(armed) == 1 and armed[0].at_ms == 3900 + POLICY.interrupt_timeout_ms

    def test_paused_backchannel_then_word_still_resumes(self):
        """Criteria 17 and 18: the backchannel ends the pause; a later word cannot re-interrupt."""
        m = TurnMachine(POLICY)
        s, _, _ = drive_to_thinking(m)
        s, _ = m.step(s, commit(s, "Zebra crossing."))
        s, _ = m.step(s, E.UserSpeechOnset(t_ms=3400, onset_ms=3400))

        s, cmds = m.step(s, E.UserBackchannel(t_ms=3600, duration_ms=200))
        assert s.state is TurnState.SPEAKING, "criterion 17(a): the backchannel resumes at once"
        assert "playback.resume" in kinds(cmds)

        s, cmds = m.step(s, E.SttWord(t_ms=3900, text="actually", start_ms=3500, end_ms=3700))
        assert s.state is TurnState.SPEAKING, "the pause is over; stt.word is not a SPEAKING row"
        assert not (INTERRUPT_COMMANDS & set(kinds(cmds)))
        assert [w.text for w in next_words(s)] == ["actually"], "criterion 4 holds in any state"

    def test_interrupt_commands_emitted_once_when_takeover_repeats(self):
        """Criterion 21: the abort/cancel/flush trio is emitted once per interruption."""
        m = TurnMachine(POLICY)
        s, _, _ = drive_to_thinking(m)
        s, _ = m.step(s, commit(s, "Zebra crossing."))
        s, _ = m.step(s, E.UserSpeechOnset(t_ms=3400, onset_ms=3400))

        s, first = m.step(s, E.UserTakeover(t_ms=3900, onset_ms=3400))
        assert s.state is TurnState.INTERRUPTED
        assert set(kinds(first)) >= INTERRUPT_COMMANDS

        s, second = m.step(s, E.UserTakeover(t_ms=3950, onset_ms=3400))
        assert s.state is TurnState.INTERRUPTED
        assert not (INTERRUPT_COMMANDS & set(kinds(second))), "the same cause, already handled"
        assert not any(isinstance(c, C.HudState) for c in second), "no state change, no hud.state"

        # still inside the interrupt_timeout window (armed at 3900 + 300 ms)
        s, third = m.step(s, E.Tick(t_ms=3900 + POLICY.interrupt_timeout_ms - 1))
        assert s.state is TurnState.INTERRUPTED

        seen = kinds(first) + kinds(second) + kinds(third)
        for name in sorted(INTERRUPT_COMMANDS):
            assert seen.count(name) == 1, f"{name} emitted {seen.count(name)} times for one cause"
        arms = [
            c
            for c in [*first, *second, *third]
            if isinstance(c, C.ArmTimer) and c.name == "interrupt_timeout"
        ]
        assert len(arms) == 1 and arms[0].at_ms == 3900 + POLICY.interrupt_timeout_ms

    @pytest.mark.parametrize(
        ("played_text", "expected"),
        [
            ("First sentence.", "First sentence. [interrupted]"),
            ("", "[interrupted]"),
        ],
        ids=["played_prefix", "nothing_played"],
    )
    def test_interrupted_history_marker_is_exactly_the_played_prefix(self, played_text, expected):
        """Criterion 19 and amendment A6: one separator, no trailing space, nothing invented."""
        m = TurnMachine(POLICY)
        s = state_at(m, "interrupted", answer="First sentence. Second sentence.")
        s, cmds = m.step(
            s,
            E.PlaybackStopped(
                t_ms=3950, played_ms=1200 if played_text else 0, played_text=played_text
            ),
        )
        assert s.state is TurnState.LISTENING
        trunc = next(c for c in cmds if isinstance(c, C.TruncateHistoryToPlayed))
        assert trunc.played_text == played_text
        h = s.history[-1]
        assert h.assistant_spoken == expected
        assert h.interrupted is True
        assert h.user_text == "what is"
        assert "Second sentence" not in h.assistant_spoken


# --- criterion 20(b)/(c) and amendment A5: the error rows ----------------------------------------


@pytest.mark.parametrize("where", ["idle", "listening", "endpointing", "thinking", *SPEECH_STATES])
def test_tts_error_degrades_everywhere_and_ends_only_the_speaking_states(where):
    """Criterion 20(b): the degrade is unconditional, ending the turn is not."""
    m = TurnMachine(POLICY)
    s = state_at(m, where)
    after, cmds = m.step(s, E.TtsError(t_ms=6000, message="the voice server is gone"))
    degrades = [c for c in cmds if isinstance(c, C.Degrade)]
    assert len(degrades) == 1 and degrades[0].mode == "text_only"
    assert after.degraded is True

    if where in SPEECH_STATES:
        assert after.state is TurnState.IDLE
        assert "commit_history" in kinds(cmds)
        assert after.active_generate is None and after.active_prefill is None
        for h in after.history:
            assert "Zebra" not in h.assistant_spoken, "history holds only what was played"
    else:
        assert after.state is s.state, "criterion 20(b): these states survive a tts failure"
        assert "commit_history" not in kinds(cmds), "no turn ended, no history entry"
        assert after.active_generate == s.active_generate
        assert after.active_prefill == s.active_prefill
        assert not any(isinstance(c, C.HudState) for c in cmds), "no state change, no hud.state"


@pytest.mark.parametrize("where", SPEECH_STATES)
def test_llm_error_ends_the_turn_from_every_state_with_a_live_generate(where):
    """Amendment A5: criterion 14's row is not THINKING-only; here it commits the played prefix."""
    m = TurnMachine(POLICY)
    s = state_at(m, where)
    after, cmds = m.step(s, E.LlmError(t_ms=6000, request_id=s.active_generate, message="boom"))
    assert after.state is TurnState.IDLE
    assert "hud.error" in kinds(cmds)
    commits = [c for c in cmds if isinstance(c, C.CommitHistory)]
    assert len(commits) == 1 and commits[0].user_only is False, "A5: the played prefix, not user"
    assert after.active_generate is None and after.active_prefill is None
    assert not any(isinstance(c, C.TtsSpeak) for c in cmds), "a dead generation speaks no more"
    for h in after.history:
        assert "Zebra" not in h.assistant_spoken, "history holds only what was played"


@pytest.mark.parametrize("where", ["listening", "endpointing"])
def test_stt_error_while_listening_commits_user_only(where):
    """Criterion 20(c): a transcript exists, so the user half of the turn is kept."""
    m = TurnMachine(POLICY)
    s = state_at(m, where)
    if where == "listening":  # state_at only adds a word on the endpointing path
        s, _ = m.step(s, E.SttWord(t_ms=1500, text="what", start_ms=1100, end_ms=1300))
    assert s.words, "the precondition of this row is a non-empty transcript"
    after, cmds = m.step(s, E.SttError(t_ms=2700, message=STT_DEAD))
    assert after.state is TurnState.IDLE
    assert "hud.error" in kinds(cmds)
    commits = [c for c in cmds if isinstance(c, C.CommitHistory)]
    assert len(commits) == 1 and commits[0].user_only is True
    assert after.active_prefill is None and after.active_generate is None


def test_stt_error_with_an_empty_transcript_leaves_no_history_entry():
    """Criterion 20(c): nothing was said, so nothing is stored."""
    m = TurnMachine(POLICY)
    s = state_at(m, "listening")
    assert s.words == ()
    after, cmds = m.step(s, E.SttError(t_ms=1400, message=STT_DEAD))
    assert after.state is TurnState.IDLE
    assert "hud.error" in kinds(cmds)
    assert "commit_history" not in kinds(cmds)
    assert len(after.history) == len(s.history)


@pytest.mark.parametrize("where", ["thinking", "speaking"])
def test_stt_error_outside_the_listening_states_only_reports_it(where):
    """Criterion 20(c): the user has stopped talking; a dead STT must not kill the answer."""
    m = TurnMachine(POLICY)
    s = state_at(m, where)
    after, cmds = m.step(s, E.SttError(t_ms=6000, message=STT_DEAD))
    assert after.state is s.state
    assert "hud.error" in kinds(cmds)
    assert "commit_history" not in kinds(cmds)
    assert "degrade" not in kinds(cmds), "criterion 20(b) degrades, 20(c) does not"
    assert not any(isinstance(c, C.LlmAbort) for c in cmds)
    assert after.active_generate == s.active_generate


# --- criterion 13: the history bound -------------------------------------------------------------


def test_history_is_truncated_to_max_history_turns():
    """Criterion 13: five completed turns leave the last ``policy.max_history_turns`` entries."""
    keep = POLICY.max_history_turns
    turns = keep + 2
    m = TurnMachine(POLICY)
    s = m.initial()
    for i in range(1, turns + 1):
        base = i * 10_000
        answer = f"Answer {i}."
        s, _ = run(
            m,
            [
                E.UserSpeechOnset(t_ms=base, onset_ms=base),
                E.SttWord(t_ms=base + 400, text=f"turn{i}", start_ms=base + 100, end_ms=base + 300),
                E.SttEndOfTurn(t_ms=base + 600, t_eot_ms=base + 500),
                E.SttFlushed(t_ms=base + 700, t_eot_ms=base + 500),
                E.Tick(t_ms=base + 700 + POLICY.eot_settle_ms),
            ],
            s,
        )
        assert s.state is TurnState.THINKING, f"turn {i} did not start generating"
        s, _ = m.step(s, commit(s, answer, t=base + 1000))
        s, _ = m.step(s, E.LlmDone(t_ms=base + 1100, request_id=s.active_generate))
        s, _ = m.step(s, E.PlaybackDrained(t_ms=base + 1500, played_ms=500, played_text=answer))
        assert s.state is TurnState.IDLE, f"turn {i} did not finish"
        assert len(s.history) <= keep

    kept = range(turns - keep + 1, turns + 1)
    assert len(s.history) == keep
    assert [h.user_text for h in s.history] == [f"turn{i}" for i in kept]
    assert [h.assistant_spoken for h in s.history] == [f"Answer {i}." for i in kept]
    assert all(h.interrupted is False for h in s.history)


# --- criterion 10, adversarial: the second segmenter call ----------------------------------------


def test_a_segmenter_that_lies_only_on_the_final_flush_is_caught():
    """Criteria 10 and 12: the ``final=True`` flush is policed like every other segmenter call.

    The sibling module covers a segmenter that lies on a commit. This one is honest there and
    lies exactly once, on the ``llm.done`` flush — the remaining path by which text the LLM never
    committed could reach ``tts.speak`` (CLAUDE.md: "never voice text the LLM has not committed").
    """
    segment_parameter()
    m = TurnMachine(POLICY, segment=honest_until_the_flush)
    s, _, _ = drive_to_thinking(m)
    rid = s.active_generate
    s, cmds = m.step(s, commit(s, "A full sentence. A dangling tail"))
    assert [c.source for c in cmds if isinstance(c, C.TtsSpeak)] == ["A full sentence. "]
    assert s.state is TurnState.SPEAKING

    before = s.model_copy(deep=True)
    with pytest.raises(ValueError):
        m.step(s, E.LlmDone(t_ms=3300, request_id=rid))
    assert s == before, "a rejected flush leaves the state untouched"


# --- amendment A16: a row that ends a turn must not orphan the generate --------------------------


@pytest.mark.parametrize("where", ["thinking", "speaking", "paused"])
def test_llm_error_aborts_the_live_generate(where):
    """A16: ending the turn on ``llm.error`` leaves the engine denoising unless the row says so.

    Criterion 14 and docs/02 §4 list only ``hud.error`` + ``commit_history`` on this row, so the
    machine can idle, clear ``active_generate`` and leave the request running with nobody left to
    cancel it. §4's "at most one active generate" does not catch it — an orphan is not a second
    active generate. It is budget-relevant: ``.claude/rules/conductor.md`` caps the session at one
    generate + one prefill + captioner against ``--max-num-seqs 3``, with ~2.5 GiB of unprofiled
    sampler transient per concurrently denoising request, so the next turn's ``llm.start`` would
    run beside the ghost.
    """
    m = TurnMachine(POLICY)
    s = state_at(m, where)
    rid = s.active_generate
    assert rid is not None, f"{where} is supposed to have a live generate"

    after, cmds = m.step(s, E.LlmError(t_ms=6000, request_id=rid, message="engine died"))
    assert after.state is TurnState.IDLE
    assert [c.request_id for c in cmds if isinstance(c, C.LlmAbort)] == [rid], kinds(cmds)
    assert after.active_generate is None


def test_llm_error_without_a_request_id_still_aborts_the_live_generate():
    """A16: the abort is owed to whatever *this session* started, not to what the error names.

    An ``llm.error`` with no ``request_id`` (a transport fault, not a per-request one) is the case
    where the machine is the only party that still knows the id. A16: emit it anyway — a redundant
    abort is harmless (``IDEMPOTENT_COMMANDS``), a missing one costs VRAM.
    """
    m = TurnMachine(POLICY)
    s = state_at(m, "thinking")
    rid = s.active_generate
    after, cmds = m.step(s, E.LlmError(t_ms=6000, request_id=None, message="connection reset"))
    assert after.state is TurnState.IDLE
    assert [c.request_id for c in cmds if isinstance(c, C.LlmAbort)] == [rid], kinds(cmds)


def test_llm_timeout_tick_aborts_the_live_generate():
    """A16, criterion 14: 20 s with no token is the case that most needs the abort.

    Nothing has failed on the server side — the request is still hot — so idling without an
    ``llm.abort`` is precisely how a denoising request is leaked.
    """
    m = TurnMachine(POLICY)
    s, _, start_t = drive_to_thinking(m)
    rid = s.active_generate
    after, cmds = m.step(s, E.Tick(t_ms=start_t + POLICY.llm_timeout_ms))
    assert after.state is TurnState.IDLE, "the deadline fires"
    assert [c.request_id for c in cmds if isinstance(c, C.LlmAbort)] == [rid], kinds(cmds)
    assert after.active_generate is None and after.timers == ()


def test_llm_error_after_an_interruption_does_not_abort_a_second_time():
    """A16 with criterion 21: the abort is owed once per request, and the interrupt already paid.

    Guards the shape of the fix: "always emit ``llm.abort`` on the error row" must mean "abort what
    is still live", not "abort the id the turn used to hold".
    """
    m = TurnMachine(POLICY)
    s = state_at(m, "interrupted")
    assert s.active_generate is None, "criterion 18 aborted the generate on the way in"
    after, cmds = m.step(s, E.LlmError(t_ms=6000, request_id=None, message="engine died"))
    assert after.state is TurnState.IDLE
    assert not any(isinstance(c, C.LlmAbort) for c in cmds), kinds(cmds)


# --- criterion 6: only the matching marker settles the endpoint ----------------------------------


def _to_endpointing(machine, *, t_eot_ms=2600):
    """IDLE -> ENDPOINTING with one word at 1100 ms (criteria 1 and 3)."""
    s, _ = run(
        machine,
        [
            E.UserSpeechOnset(t_ms=1000, onset_ms=1000),
            E.SttWord(t_ms=1500, text="what", start_ms=1100, end_ms=1300),
            E.SttEndOfTurn(t_ms=t_eot_ms + 50, t_eot_ms=t_eot_ms),
        ],
    )
    assert s.state is TurnState.ENDPOINTING and s.t_eot_ms == t_eot_ms
    return s


class TestEndpointSettling:
    """Criterion 6: ENDPOINTING -> THINKING needs *this* endpoint's marker, once."""

    def test_a_flush_for_a_different_t_eot_does_not_settle_the_endpoint(self):
        """A marker that does not echo ``state.t_eot_ms`` says nothing about words <= t_eot.

        The transcriber emits a ``Marker`` per flush request; one left over from an earlier
        endpoint (or from a flush the machine did not ask for) only proves delivery up to *its own*
        time. Accepting it settles the endpoint early and sends ``llm.start`` with a transcript
        missing the words still in flight — the truncation criterion 6 exists to prevent.
        """
        m = TurnMachine(POLICY)
        s = _to_endpointing(m)

        s, cmds = m.step(s, E.SttFlushed(t_ms=2700, t_eot_ms=2500))
        assert kinds(cmds) == [], "a non-matching marker is not this endpoint's flush"

        s, cmds = m.step(s, E.Tick(t_ms=2700 + POLICY.eot_settle_ms))
        assert s.state is TurnState.ENDPOINTING, "the settle window never started"
        assert not any(isinstance(c, C.LlmStart) for c in cmds), kinds(cmds)
        assert not any(isinstance(c, C.CheckOwner) for c in cmds), kinds(cmds)

        # the word the stale marker claimed was already delivered: audio at 2400 (<= t_eot), on the
        # wire 450 ms later. It belongs to this turn (criterion 4) and must reach the transcript.
        s, _ = m.step(s, E.SttWord(t_ms=2850, text="is", start_ms=2400, end_ms=2550))
        s, cmds = run(
            m,
            [
                E.SttFlushed(t_ms=2900, t_eot_ms=2600),
                E.Tick(t_ms=2900 + POLICY.eot_settle_ms),
            ],
            s,
        )
        assert s.state is TurnState.THINKING
        start = next(c for c in cmds if isinstance(c, C.LlmStart))
        assert start.transcript == "what is", "the stale marker would have truncated the turn"

    def test_a_duplicate_flush_does_not_restart_the_settle_window(self):
        """The window runs from the first marker; a re-delivered one must not push the deadline.

        The transcriber can repeat a ``Marker``; if each repeat re-armed ``eot_settle_ms`` a chatty
        transcriber would hold the turn in ENDPOINTING indefinitely and nothing would be answered.
        """
        m = TurnMachine(POLICY)
        s = _to_endpointing(m)

        s, _ = m.step(s, E.SttFlushed(t_ms=2900, t_eot_ms=2600))
        s, cmds = m.step(s, E.SttFlushed(t_ms=3000, t_eot_ms=2600))
        assert kinds(cmds) == [], "a repeated marker is not news"
        assert s.state is TurnState.ENDPOINTING

        s, cmds = m.step(s, E.Tick(t_ms=2900 + POLICY.eot_settle_ms))
        assert s.state is TurnState.THINKING, "the window runs from the first marker, not the last"
        assert any(isinstance(c, C.LlmStart) for c in cmds), kinds(cmds)


# --- criteria 4 and 18: the two boundaries the criteria state with <= and >= ---------------------


class TestAudioTimeBoundaries:
    """Both rules are inclusive in the criteria; a strict inequality changes who owns a word."""

    def test_a_word_starting_exactly_at_t_eot_belongs_to_this_turn(self):
        """Criterion 4: the split is ``start_ms <= t_eot_ms``, so the word *on* the cut stays.

        ``freeze_after(t_eot)`` tells STT to stop at that instant, so a word whose audio starts
        exactly there is the last word the user spoke in this turn; pushing it to ``next_words``
        would drop it from the question and prepend it to the next one.
        """
        m = TurnMachine(POLICY)
        endpointing = _to_endpointing(m)

        s, cmds = m.step(endpointing, E.SttWord(t_ms=2700, text="now", start_ms=2600, end_ms=2750))
        assert [w.text for w in s.words] == ["what", "now"]
        assert next_words(s) == (), "2600 == t_eot is inside the turn"
        assert kinds(cmds) == [], "a word does not settle anything"

        s, cmds = run(
            m,
            [
                E.SttFlushed(t_ms=2900, t_eot_ms=2600),
                E.Tick(t_ms=2900 + POLICY.eot_settle_ms),
            ],
            s,
        )
        assert next(c for c in cmds if isinstance(c, C.LlmStart)).transcript == "what now"

        # one millisecond later is the other side of the same boundary
        after, _ = m.step(endpointing, E.SttWord(t_ms=2700, text="now", start_ms=2601, end_ms=2750))
        assert [w.text for w in after.words] == ["what"]
        assert [w.text for w in next_words(after)] == ["now"]

    def test_a_word_starting_exactly_at_pause_onset_interrupts(self):
        """Criterion 18: the rule is ``start_ms >= pause_onset_ms`` (A13: the *audio* onset).

        The onset detector fires on the first speech sample, so the barge-in word's audio starts at
        the pause onset itself. A strict ``>`` makes the commonest barge-in of all — one word,
        spoken the instant the assistant is cut off — resume playback over the user instead of
        stopping.
        """
        m = TurnMachine(POLICY)
        paused = state_at(m, "paused")  # onset_ms == 3400
        assert paused.state is TurnState.PAUSED
        rid = paused.active_generate

        s, cmds = m.step(paused, E.SttWord(t_ms=3500, text="no", start_ms=3400, end_ms=3600))
        assert s.state is TurnState.INTERRUPTED
        assert set(kinds(cmds)) >= INTERRUPT_COMMANDS, kinds(cmds)
        assert next(c for c in cmds if isinstance(c, C.LlmAbort)).request_id == rid

        # one millisecond earlier is audio from before the pause: STT lag, not a barge-in
        s, cmds = m.step(paused, E.SttWord(t_ms=3500, text="no", start_ms=3399, end_ms=3600))
        assert s.state is TurnState.PAUSED
        assert not (INTERRUPT_COMMANDS & set(kinds(cmds))), kinds(cmds)


# --- the real speech gate behind the segmenter seam (T-0004 A5 and A9) ---------------------------
#
# T-0003 criterion 10 leaves the machine a segmenter seam and a default segmenter that never holds
# anything back; T-0004 fills the seam with ``SpeechGate``, which does. The postcondition T-0004 A5
# prescribes — the spoken sources of a turn are always a *prefix* of what that turn committed, and
# the whole of it once ``llm.done`` has been flushed — is only non-trivial with a gate that holds.
# The Hypothesis model in the sibling module asserts it on every trace; the three tests below pin
# the two boundaries that must not be left to luck: the flush at ``llm.done`` and the barge-in that
# throws the held clause away (A9: the gate is *discarded*, never ``close()``d, because closing it
# speaks).


class GateWiring:
    """One ``SpeechGate`` per turn behind T-0003's injected segmenter (T-0004 A9).

    Adapter shape as in ``tests/unit/test_speech_gate.py::gate_segmenter``: the machine hands back
    its whole unspoken tail (the gate's ``pending`` plus the new commit), so only the suffix is
    fed, and whatever the gate returns goes straight to the machine's conservation check — the
    wiring never repairs it. ``closes`` counts ``close()`` calls so a test can assert a turn ended
    without one; ``new_turn`` is what T-0007's wiring does at a turn boundary.
    """

    def __init__(self) -> None:
        self.gate = SpeechGate()
        self.closes = 0
        self.gates = 1

    def new_turn(self) -> None:
        self.gate = SpeechGate()
        self.gates += 1

    def __call__(self, text: str, *, final: bool):
        held = self.gate.pending
        assert text.startswith(held), (
            f"gate.pending {held!r} is not a prefix of the machine's unspoken tail {text!r}"
        )
        suffix = text[len(held) :]
        out = (
            list(
                self.gate.feed(Commit(request_id="r", canvas_idx=0, segment="answer", text=suffix))
            )
            if suffix
            else []
        )
        if final:
            out.extend(self.gate.close())
            self.closes += 1
        return [(s.source, s.text) for s in out], self.gate.pending


def spoken_sources(cmds) -> list[str]:
    return [c.source for c in cmds if isinstance(c, C.TtsSpeak)]


def test_llm_done_flushes_the_gate_so_the_spoken_sources_rebuild_the_committed_answer():
    """T-0004 A5, the equality half; T-0003 criterion 12 (``llm.done`` flushes with ``final=True``).

    Expected values are the input, computed by hand: the committed answer is the concatenation of
    the two commit chunks, and the sentence split is T-0004 A1 (the whitespace after a terminator
    rides on the preceding sentence, so ``pending`` starts at a word). The intermediate assertion
    is the point of the test: after the first commit the gate is *still holding* a clause, so the
    spoken text is a **strict** prefix — a machine that flushed eagerly, or a postcondition that
    only ever saw a gate with nothing pending, would not be distinguished from a correct one here.
    """
    wiring = GateWiring()
    m = TurnMachine(POLICY, segment=wiring)
    s, _, _ = drive_to_thinking(m)
    rid = s.active_generate

    chunks = ["It is a stack trace. The error is on", " line four. Try again"]
    committed = "It is a stack trace. The error is on line four. Try again"
    sources: list[str] = []
    for i, chunk in enumerate(chunks):
        s, cmds = m.step(s, commit(s, chunk, idx=i, t=3200 + i))
        sources += spoken_sources(cmds)
        assert committed.startswith("".join(sources)), (
            f"spoken {''.join(sources)!r} is not a prefix of the committed answer"
        )
        if i == 0:
            assert sources == ["It is a stack trace. "], "A1: the separator rides on the sentence"
            assert "".join(sources) != committed[: len("It is a stack trace. The error is on")], (
                "the gate must still be holding 'The error is on': without a held clause this "
                "test cannot tell a flush from an eager segmenter"
            )

    assert wiring.closes == 0, "nothing is final until llm.done"
    s, cmds = m.step(s, E.LlmDone(t_ms=3400, request_id=rid))
    sources += spoken_sources(cmds)
    assert wiring.closes == 1, "criterion 12: llm.done flushes the gate with final=True"
    assert sources == [
        "It is a stack trace. ",
        "The error is on line four. ",
        "Try again",
    ], "every sentence, in order, each an exact slice of the committed answer"
    assert "".join(sources) == committed, "A5: on llm.done the spoken text IS the committed answer"
    assert s.committed_answer == committed


def test_a_takeover_discards_the_gates_held_clause_and_never_voices_it_later():
    """T-0004 A5 (the barge-in half) and A9 (``close()`` speaks, so a takeover must not call it).

    The failure mode: the held half-sentence of a turn the user cut off surfaces during the *next*
    turn, i.e. the assistant answers a question that was interrupted and never finished. Nothing
    of turn 1's answer beyond what was already spoken may ever reach TTS, and the second turn's
    speech must rebuild the second turn's committed answer exactly — not more.
    """
    wiring = GateWiring()
    m = TurnMachine(POLICY, segment=wiring)
    s, _, _ = drive_to_thinking(m)
    rid = s.active_generate

    s, cmds = m.step(s, commit(s, "It is a stack trace. The error is on"))
    assert s.state is TurnState.SPEAKING
    first_turn = s.turn_id
    assert spoken_sources(cmds) == ["It is a stack trace. "]

    s, cmds = m.step(s, E.UserTakeover(t_ms=3400, onset_ms=3350))
    assert s.state is TurnState.INTERRUPTED
    assert set(kinds(cmds)) >= INTERRUPT_COMMANDS, kinds(cmds)
    assert next(c for c in cmds if isinstance(c, C.LlmAbort)).request_id == rid
    assert spoken_sources(cmds) == [], "a takeover speaks nothing, least of all the held clause"
    assert wiring.closes == 0, (
        "A9: the gate is discarded on a takeover, never closed — closing speaks"
    )

    s, cmds = m.step(
        s, E.PlaybackStopped(t_ms=3500, played_ms=900, played_text="It is a stack trace.")
    )
    assert s.state is TurnState.LISTENING and s.turn_id != first_turn
    assert spoken_sources(cmds) == []
    wiring.new_turn()  # what the executor does at a turn boundary (T-0007)

    s, cmds = run(
        m,
        [
            E.SttWord(t_ms=4000, text="again", start_ms=3600, end_ms=3800),
            E.SttEndOfTurn(t_ms=4050, t_eot_ms=3950),
            E.SttFlushed(t_ms=4100, t_eot_ms=3950),
            E.Tick(t_ms=4100 + POLICY.eot_settle_ms),
        ],
        s,
    )
    assert s.state is TurnState.THINKING
    second_rid = s.active_generate
    assert spoken_sources(cmds) == []

    s, cmds = m.step(s, commit(s, "A different answer. And a tail", t=4400))
    s, more = m.step(s, E.LlmDone(t_ms=4500, request_id=second_rid))
    second = spoken_sources(cmds) + spoken_sources(more)
    assert "".join(second) == "A different answer. And a tail", "A5, again, for the new turn"
    assert not any("The error is on" in source for source in second), (
        "the clause the interrupted turn was holding must never be voiced"
    )
    assert wiring.gates == 2, "one gate per turn (A9)"


def test_a_pause_holds_the_gates_text_and_the_resume_gives_all_of_it_back():
    """T-0004 A5 across criteria 16–17: the PAUSED rows defer the flush, they do not lose it.

    ``llm.done`` arriving while PAUSED may not speak (criterion 16 forbids synthesis a takeover
    would cancel), so the equality A5 asks for is owed at the resume instead. The dangerous
    failure is the mirror of the takeover one: text committed during the pause is dropped on the
    floor and the user hears half an answer with no sign that the other half existed.
    """
    wiring = GateWiring()
    m = TurnMachine(POLICY, segment=wiring)
    s, _, _ = drive_to_thinking(m)
    rid = s.active_generate

    s, cmds = m.step(s, commit(s, "It is a stack trace. The error is on"))
    sources = spoken_sources(cmds)
    assert sources == ["It is a stack trace. "]

    s, cmds = m.step(s, E.UserSpeechOnset(t_ms=3400, onset_ms=3400))
    assert s.state is TurnState.PAUSED and kinds(cmds)[0] == "playback.pause"

    s, cmds = m.step(s, commit(s, " line four. Try again", idx=1, t=3450))
    assert spoken_sources(cmds) == [], "criterion 16: nothing is synthesised while paused"
    s, cmds = m.step(s, E.LlmDone(t_ms=3500, request_id=rid))
    assert spoken_sources(cmds) == [], "still paused: the flush waits for the resume"
    assert wiring.closes == 0

    s, cmds = m.step(s, E.UserBackchannel(t_ms=3550, duration_ms=200))
    assert s.state is TurnState.SPEAKING
    k = kinds(cmds)
    assert k.index("playback.resume") < k.index("tts.speak"), "resume before the held text"
    sources += spoken_sources(cmds)
    assert "".join(sources) == "It is a stack trace. The error is on line four. Try again"
    assert wiring.closes == 1, "the resume flushes with final=True because llm.done already came"
