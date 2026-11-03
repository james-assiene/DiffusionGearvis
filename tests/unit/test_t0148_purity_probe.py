"""T-0148: the ``step_is_pure`` probe must not move the model's speech gate.

``TurnMachineModel.step_is_pure`` (``tests/unit/test_turn_invariants.py``) checks the §4 "the machine
is pure" invariant by calling ``step`` twice on the same state. The machine's segmenter seam is the
model's own stateful ``SpeechGate`` fixture, so at BASE ec0b123 the first call feeds the gate and the
second meets a gate that has moved ahead of the machine's held text (the ``_segment`` prefix
assertion fires inside the probe), or the gate silently drifts. Ruling H-T148-1 option A: each probe
call gets its own copy of the gate, the original is restored, and the probe asserts it unchanged.

Every row drives the model through its own ``_step`` with explicit ``t_ms`` values, so no row depends
on rule timing; the model's ``committed_answer`` ledger is kept exactly as the ``commit`` rule keeps
it. Expected texts are worked out by hand from T-0004 A1 (byte-exact conservation; the whitespace
after a terminator rides on the sentence before it) and A7 (the balance hold).

Rows 1-4 (criteria 1, 2, 3a, 3b) are red at BASE (pending until the activation commit, which also
changed the probe); the rest hold at BASE and at TIP.
"""

from __future__ import annotations

import copy
import io
import pathlib
import tokenize
from typing import ClassVar

import pytest

from local_astra.conductor.speech_gate import SpeechGate
from local_astra.conductor.turn_machine import default_segmenter
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import TurnState
from tests.unit import test_turn_invariants as tti

REPO = pathlib.Path(__file__).resolve().parents[2]
SESSION = REPO / "local_astra" / "conductor" / "session.py"

#: The probe time used by every row: ``step_is_pure`` probes ``Tick(model.t + 1)``.
PROBE_AT = 2000


# --- drivers -----------------------------------------------------------------------------------------


def _commit(model: tti.TurnMachineModel, text: str, t_ms: int) -> list:
    """One answer commit of the live generate, booked in the model's ledger as the rule books it."""
    rid = model.s.active_generate
    assert rid is not None, "premise: a commit needs a live generate"
    model.committed_answer += text
    cmds = model._step(
        E.LlmCommit(
            t_ms=t_ms,
            request_id=rid,
            canvas_idx=model.canvas,
            segment="answer",
            text=text,
            thought_open=False,
        )
    )
    model.canvas += 1
    model.t = t_ms
    return list(cmds)


def _to_thinking(model: tti.TurnMachineModel) -> None:
    """Onset 0, word "what" 400, end_of_turn 500 (t_eot 450), flushed 505, settle tick -> THINKING."""
    model._step(E.UserSpeechOnset(t_ms=0, onset_ms=0))
    model._step(E.SttWord(t_ms=400, text="what", start_ms=0, end_ms=200))
    model._step(E.SttEndOfTurn(t_ms=500, t_eot_ms=450))
    model._step(E.SttFlushed(t_ms=505, t_eot_ms=450))
    t = 505 + tti.POLICY.eot_settle_ms + 1
    model._step(E.Tick(t_ms=t))
    model.t = t
    assert model.s.state is TurnState.THINKING, f"premise: {model.s.state.value}, not thinking"
    assert model.s.active_generate is not None


def _pause(model: tti.TurnMachineModel) -> None:
    """Onset 1100 -> PAUSED; offset 1200 with speech_ms 300 arms the resume deadline."""
    model._step(E.UserSpeechOnset(t_ms=1100, onset_ms=1100))
    model._step(E.UserSpeechOffset(t_ms=1200, offset_ms=1200, speech_ms=300))
    model.t = 1200
    assert model.s.state is TurnState.PAUSED, f"premise: {model.s.state.value}, not paused"


def _resume_is_due(model: tti.TurnMachineModel) -> None:
    """Premise: the machine is PAUSED and the probe's tick is past an armed ``resume`` deadline."""
    assert model.s.state is TurnState.PAUSED, f"premise: {model.s.state.value}, not paused"
    resume = [t.at_ms for t in model.s.timers if t.name == "resume"]
    assert resume and resume[0] <= PROBE_AT + 1, (
        f"premise: no resume deadline before the probe tick {PROBE_AT + 1}: {model.s.timers}"
    )


def _criterion_1_model() -> tti.TurnMachineModel:
    """Reviewer B's shape: a second sentence completed by a commit made while PAUSED.

    By hand: ``" line four."`` at 1000 opens SPEAKING and the gate holds it (no whitespace after the
    terminator, so it is not yet a sentence end). The second ``" line four."`` arrives while PAUSED:
    nothing is spoken, and the turn's unspoken tail is ``" line four. line four."``.
    """
    model = tti.TurnMachineModel()
    _to_thinking(model)
    assert _speaks(_commit(model, " line four.", 1000)) == [], "the gate holds an unfinished end"
    assert model.s.state is TurnState.SPEAKING
    _pause(model)
    assert _speaks(_commit(model, " line four.", 2000)) == [], "nothing is voiced while paused"
    assert model.s.held_text == " line four. line four."
    _resume_is_due(model)
    model.t = PROBE_AT
    return model


def _speaks(cmds) -> list[str]:
    return [c.source for c in cmds if isinstance(c, C.TtsSpeak)]


def _assert_probe_leaves_gate_unchanged(model: tti.TurnMachineModel) -> None:
    gate = model.gate
    snapshot = copy.deepcopy(gate)
    model.step_is_pure()
    assert model.gate is gate, "H-T148-1 A: the probe must restore the model's original gate object"
    assert model.gate == snapshot, (
        f"the probe moved the model's speech gate: pending {snapshot.pending!r} -> "
        f"{model.gate.pending!r}, emitted {[s.source for s in snapshot.emitted]!r} -> "
        f"{[s.source for s in model.gate.emitted]!r}"
    )


# --- criteria 1-3: the probe neither trips nor moves the gate ------------------------------------------


def test_probe_survives_second_sentence_committed_while_paused():
    """Criterion 1 (adversarial: the exact shape reviewer B's database replays).

    At BASE the first probe call feeds ``" line four."`` to the shared gate, which then holds
    ``"line four."``; the second call's ``_segment`` finds that is not a prefix of the machine's tail
    and raises AssertionError.
    """
    model = _criterion_1_model()
    _assert_probe_leaves_gate_unchanged(model)


def test_probe_survives_balance_hold_released_while_paused():
    """Criterion 2 (the seed-160 shape): a balance hold inside text committed while PAUSED.

    By hand: ``"Yes. "`` is spoken at once (terminator + whitespace), so the gate holds nothing.
    While PAUSED, ``") is the one. "`` and ``"Consider (a. b"`` arrive; on resume the gate gives up
    ``") is the one. "`` and holds ``"Consider (a. b"`` (open parenthesis, A7), which is not a
    prefix of the tail the second probe call hands it.
    """
    model = tti.TurnMachineModel()
    _to_thinking(model)
    assert _speaks(_commit(model, "Yes. ", 1000)) == ["Yes. "]
    _pause(model)
    _commit(model, ") is the one. ", 1300)
    _commit(model, "Consider (a. b", 1400)
    assert model.s.held_text == ") is the one. Consider (a. b"
    _resume_is_due(model)
    model.t = PROBE_AT
    _assert_probe_leaves_gate_unchanged(model)


def test_probe_leaves_gate_untouched_when_resume_is_final():
    """Criterion 3a: ``llm.done`` while PAUSED makes the resume final, so the probe calls ``close()``.

    At BASE the assertion passes but the first call's ``close()`` empties the gate (pending
    ``" line four."`` -> ``""``) and the second call feeds the tail again: silent drift.
    """
    model = tti.TurnMachineModel()
    _to_thinking(model)
    _commit(model, " line four.", 1000)
    _pause(model)
    rid = model.s.active_generate
    assert _speaks(model._step(E.LlmDone(t_ms=1300, request_id=rid))) == []
    assert model.gate.pending == " line four.", "premise: the gate still holds the clause"
    _resume_is_due(model)
    model.t = PROBE_AT
    _assert_probe_leaves_gate_unchanged(model)


def test_probe_leaves_gate_untouched_when_nothing_completes():
    """Criterion 3b: a PAUSED commit that completes no sentence ("The error is on" + " line four.").

    At BASE the first call feeds ``" line four."`` and the gate's pending grows to the whole
    ``"The error is on line four."``; the second call feeds nothing. Outputs agree, the gate drifted.
    """
    model = tti.TurnMachineModel()
    _to_thinking(model)
    assert _speaks(_commit(model, "The error is on", 1000)) == []
    _pause(model)
    _commit(model, " line four.", 1300)
    assert model.gate.pending == "The error is on", "premise: the PAUSED commit is not fed yet"
    _resume_is_due(model)
    model.t = PROBE_AT
    _assert_probe_leaves_gate_unchanged(model)


# --- criterion 4: the machine itself voices each committed sentence once ------------------------------


def test_machine_voices_each_committed_sentence_once():
    """Criterion 4 (negative: the machine's output does not change), the criterion-1 shape, no probe.

    Hand-computed from T-0004 A1: the resume feeds ``" line four. line four."``; the first terminator
    is followed by a space, which rides on the first sentence, so the resume voices
    ``" line four. "`` and the gate holds ``"line four."`` until ``llm.done`` flushes it.
    """
    model = _criterion_1_model()
    resumed = model._step(E.Tick(t_ms=PROBE_AT + 1))
    assert _speaks(resumed) == [" line four. "]
    done = model._step(E.LlmDone(t_ms=2100, request_id=model.s.active_generate))
    assert _speaks(done) == ["line four."]
    assert "".join(_speaks(resumed) + _speaks(done)) == " line four. line four."
    assert model.turn_committed == " line four. line four."


# --- criteria 5 and 6: the probe is neither too narrow nor too wide -----------------------------------


class _MachineDouble:
    """Delegates every name to the real machine; ``step`` optionally appends a per-call abort."""

    def __init__(self, real, *, impure: bool):
        self._real = real
        self._impure = impure
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def step(self, state, event):
        self.calls += 1
        new, cmds = self._real.step(state, event)
        if self._impure:
            cmds = (*cmds, C.LlmAbort(request_id=f"impure-{self.calls}"))
        return new, cmds


@pytest.mark.parametrize("impure", [True, False], ids=["impure", "pure-control"])
def test_probe_still_rejects_an_impure_machine(impure):
    """Criterion 5 (too narrow): a ``step`` whose commands differ on a repeated identical call fails.

    The pure-control arm runs the same double without the counter, so the raise in the impure arm is
    the equality's and not something the double did.
    """
    model = tti.TurnMachineModel()
    model.t = 100
    double = _MachineDouble(model.m, impure=impure)
    model.m = double
    if impure:
        with pytest.raises(AssertionError):
            model.step_is_pure()
    else:
        model.step_is_pure()
    assert double.calls == 2, f"the probe must call the model's machine twice, not {double.calls}"


class _CountingGate(SpeechGate):
    """A real ``SpeechGate`` whose feeds are counted at class level, so copies count too."""

    feeds: ClassVar[int] = 0

    def feed(self, commit):
        type(self).feeds += 1
        return super().feed(commit)


def test_probe_feeds_the_real_gate_on_each_call(monkeypatch):
    """Criterion 6 (too wide): the probe must go through the real gate, once per call.

    Shape chosen so the probe holds at BASE as well as at TIP (see the report): ``"Yes. "`` is spoken,
    then ``"Yes. "`` and ``"Yes"`` are committed while PAUSED. Each probe call feeds the tail; the
    gate voices ``"Yes. "`` and holds ``"Yes"``, which is still a prefix of ``"Yes. Yes"``, so even a
    shared gate is fed by the second call. A probe on ``default_segmenter`` or without the segmenter
    feeds nothing.
    """
    monkeypatch.setattr(tti, "SpeechGate", _CountingGate)
    model = tti.TurnMachineModel()
    _to_thinking(model)
    assert _speaks(_commit(model, "Yes. ", 1000)) == ["Yes. "]
    _pause(model)
    _commit(model, "Yes. ", 1300)
    _commit(model, "Yes", 1400)
    assert model.s.held_text == "Yes. Yes"
    _resume_is_due(model)
    model.t = PROBE_AT
    assert isinstance(model.gate, _CountingGate), "premise: the model's turn gate is the counter"
    _CountingGate.feeds = 0
    model.step_is_pure()
    assert _CountingGate.feeds >= 2, (
        f"the probe fed the real gate {_CountingGate.feeds} time(s); it must feed it once per call"
    )
    # control: the same machine on the default segmenter feeds the gate nothing
    _CountingGate.feeds = 0
    tti.TurnMachine(tti.POLICY, segment=default_segmenter).step(model.s, E.Tick(t_ms=PROBE_AT + 1))
    assert _CountingGate.feeds == 0


# --- criterion 9: the product calls the machine once per event ----------------------------------------


def _code_tokens(path: pathlib.Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    skip = {tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE}
    skip |= {
        getattr(tokenize, n) for n in ("FSTRING_START", "FSTRING_MIDDLE") if hasattr(tokenize, n)
    }
    return [
        t.string
        for t in tokenize.generate_tokens(io.StringIO(src).readline)
        if t.type not in skip and t.string.strip()
    ]


def test_session_has_one_machine_step_call_site():
    """Criterion 9: ``session.py`` has exactly one ``.machine.step(`` in code (comments, strings out).

    A second call with the live session gate would re-feed the whole tail and the conservation check
    would raise, silencing the turn (the latent hazard in the task's triage).
    """
    toks = _code_tokens(SESSION)
    pattern = [".", "machine", ".", "step", "("]
    hits = [i for i in range(len(toks)) if toks[i : i + len(pattern)] == pattern]
    assert len(hits) == 1, (
        f"session.py has {len(hits)} `.machine.step(` call(s) in code; the session must call the "
        "machine exactly once per event (T-0148 criterion 9). Comments and strings are stripped "
        "before counting, but if prose inside code (an unusual f-string) tripped this, say so."
    )
