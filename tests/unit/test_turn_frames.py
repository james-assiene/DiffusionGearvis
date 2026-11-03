"""Turn machine: the frames one turn accumulates (docs/tasks/T-0059.md group A, criteria 1-7).

A turn used to show the model exactly one screen ("the last snapshot wins", docs/01 D9). After
T-0059 the machine records the *distinct* screens the user saw while speaking, in arrival order,
so that the prompt can carry them all before the transcript. This module is the machine half:
``MachineState.turn_phashes`` (the name is contract, criterion 1) and
``TurnPolicy.max_frames_per_turn`` (criterion 3).

Nothing here is derived from an implementation -- there is none when this is written. Every row
names the criterion of ``docs/tasks/T-0059.md`` it comes from and the design ruling behind it.
Conventions follow the sibling modules ``test_turn_invariants.py`` / ``test_turn_edges.py``: the
two new attributes are read through the helpers below, so a missing one fails as a *named
criterion* instead of an ``AttributeError`` (or a dataclass ``TypeError``) half-way through a row.

One piece of criterion wording had to be interpreted, and the interpretation is asserted rather
than assumed -- see ``test_each_scene_change_appends_a_phash_in_arrival_order``'s docstring and
``tuple_slots`` below: criterion 1 reads "three distinct changes give ``(a, b, c)``" while
criterion 3 caps the tuple at ``max_frames_per_turn - 1`` because the turn's first frame is the
onset snapshot, whose hash no event ever carries (D3). At the default cap of 3 those two cannot
both be true, so the arrival-order row is run at a cap that leaves room for three scene changes
and the cap arithmetic is pinned separately.
"""

from __future__ import annotations

import dataclasses
import inspect
import pathlib
import subprocess

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import HistoryTurn, MachineState, Timer, TurnState, Word

POLICY = TurnPolicy(thinking=True)

#: Four distinct perceptual hashes, in the shape ``SceneChanged.phash`` carries on the wire
#: (16 hex digits, as the existing §4 rows use: ``"ab" * 8``).
PH_A = "a1a1a1a1a1a1a1a1"
PH_B = "b2b2b2b2b2b2b2b2"
PH_C = "c3c3c3c3c3c3c3c3"
PH_D = "d4d4d4d4d4d4d4d4"
PHASHES = (PH_A, PH_B, PH_C, PH_D)

_MISSING = object()


# --- the two new attributes, read so that "absent" is an assertion ------------------------------


def turn_phashes(state: MachineState) -> tuple[str, ...]:
    """``MachineState.turn_phashes`` (criterion 1: "the name is contract")."""
    value = getattr(state, "turn_phashes", _MISSING)
    assert value is not _MISSING, (
        "criterion 1: T-0059 adds MachineState.turn_phashes (the name is contract), a tuple of "
        "the distinct scene-change hashes of the current turn in arrival order"
    )
    assert isinstance(value, tuple), (
        f"MachineState.turn_phashes is a tuple (implementation notes: the state is frozen and "
        f"hashable), got {type(value).__name__}"
    )
    return value


def frame_cap(policy: TurnPolicy = POLICY) -> int:
    """``TurnPolicy.max_frames_per_turn`` (criterion 3; default 3 per D2 and ruling A1)."""
    value = getattr(policy, "max_frames_per_turn", _MISSING)
    assert value is not _MISSING, (
        "criteria 3 and 5: T-0059 adds TurnPolicy.max_frames_per_turn (default 3, D2), the "
        "number of frames one turn may show the model"
    )
    assert isinstance(value, int) and not isinstance(value, bool), (
        f"TurnPolicy.max_frames_per_turn is an int, got {value!r}"
    )
    return value


def policy_with_cap(n: int) -> TurnPolicy:
    """``TurnPolicy(max_frames_per_turn=n)``, guarded so a missing field is a named failure.

    Without the guard the constructor raises ``TypeError: unexpected keyword argument`` before
    any assertion runs, and the row reports a crash where it means "criterion 3 is not built".
    """
    frame_cap()
    return TurnPolicy(thinking=True, max_frames_per_turn=n)


def tuple_slots(policy: TurnPolicy = POLICY) -> int:
    """How many entries ``turn_phashes`` may hold under ``policy`` (criteria 3 and 5).

    The turn's first frame is the onset snapshot and the machine never learns its hash (D3), so
    it occupies a frame slot the tuple cannot show: at ``max_frames_per_turn == 3`` the tuple
    holds at most **two** scene-change hashes (criterion 3, "the tuple never exceeds
    ``max_frames_per_turn - 1``"). At ``max_frames_per_turn == 1`` criterion 5 says the tuple
    holds "exactly the newest" hash -- the scene change *replaces* the onset frame there, which
    is docs/01 D9's "the last snapshot wins" -- so the floor is one slot, never zero.
    """
    cap = frame_cap(policy)
    return cap - 1 if cap > 1 else 1


# --- driving ------------------------------------------------------------------------------------


def kinds(cmds) -> list[str]:
    return [c.type for c in cmds]


def scene(t_ms: int, phash: str) -> E.SceneChanged:
    return E.SceneChanged(t_ms=t_ms, source="screen", phash=phash)


def listening(machine: TurnMachine, *, onset_ms: int = 1000) -> MachineState:
    """IDLE -> LISTENING with a live prefill: the state every scene-change row starts from."""
    state, _ = machine.step(machine.initial(), E.UserSpeechOnset(t_ms=onset_ms, onset_ms=onset_ms))
    assert state.state is TurnState.LISTENING and state.active_prefill is not None
    return state


def expected_commands(before: MachineState, t_ms: int) -> list[C.Command]:
    """Criterion 1: a distinct scene change emits exactly these two, in this order, as today.

    ``before`` is the state the event was delivered **to**, never the state the step returned
    (ruling A10). The prefill the row re-warms is the one the turn already opened, so reading the
    id back out of the machine's own output would make the assertion agree with whatever the
    machine did: a row that re-issued a bogus request id would pass every row in this module and
    be left to T-0003's ``at_most_one_generate_and_one_prefill`` invariant to catch, which is one
    stateful trace away from not being caught at all.
    """
    assert before.active_prefill is not None, "the row needs a live prefill to re-warm"
    return [C.SnapshotFrame(at_ms=t_ms), C.PrefillReissue(request_id=before.active_prefill)]


def with_two_frames(machine: TurnMachine, *, onset_ms: int = 1000) -> MachineState:
    """A LISTENING turn that has already seen screens A and B."""
    state = listening(machine, onset_ms=onset_ms)
    state, _ = machine.step(state, scene(onset_ms + 200, PH_A))
    state, _ = machine.step(state, scene(onset_ms + 400, PH_B))
    assert turn_phashes(state) == (PH_A, PH_B)
    return state


def to_thinking_with_two_frames(machine: TurnMachine) -> MachineState:
    """LISTENING (two screens seen) -> ENDPOINTING -> THINKING, the same turn throughout."""
    state = with_two_frames(machine)
    events = [
        E.SttWord(t_ms=1700, text="compare", start_ms=1500, end_ms=1650),
        E.SttEndOfTurn(t_ms=1800, t_eot_ms=1750),
        E.SttFlushed(t_ms=2050, t_eot_ms=1750),
        E.Tick(t_ms=2050 + POLICY.eot_settle_ms),
    ]
    for ev in events:
        state, _ = machine.step(state, ev)
    assert state.state is TurnState.THINKING
    assert turn_phashes(state) == (PH_A, PH_B), (
        "the frames belong to the turn, not to LISTENING: endpointing is the same turn"
    )
    return state


def to_interrupted_with_two_frames(machine: TurnMachine) -> MachineState:
    """... -> SPEAKING -> a barge-in -> INTERRUPTED, still the turn that saw the two screens."""
    state = to_thinking_with_two_frames(machine)
    commit = E.LlmCommit(
        t_ms=3000,
        request_id=state.active_generate,
        canvas_idx=0,
        segment="answer",
        text="The left one.",
        thought_open=False,
    )
    state, _ = machine.step(state, commit)
    assert state.state is TurnState.SPEAKING
    state, _ = machine.step(state, E.UserTakeover(t_ms=3400, onset_ms=3300))
    assert state.state is TurnState.INTERRUPTED
    assert turn_phashes(state) == (PH_A, PH_B), "the interrupted turn still owns its frames"
    return state


# --- criterion 1: a scene change appends --------------------------------------------------------


def test_each_scene_change_appends_a_phash_in_arrival_order():
    """Criterion 1: the hash lands at the *end*, and the two commands are today's, in order.

    Interpretation (recorded, see this module's docstring): criterion 1's "three distinct changes
    give ``turn_phashes == (a, b, c)``" and criterion 3's "the tuple never exceeds
    ``max_frames_per_turn - 1``" cannot both hold at the default cap of 3, because the onset
    frame -- whose hash the machine never sees (D3) -- is the turn's first frame. Arrival order
    is therefore asserted over three changes at a cap that leaves three slots, and over two
    changes at the shipped default; the cap arithmetic itself is
    ``test_at_the_cap_the_newest_frame_replaces_the_last_slot``'s.
    """
    machine = TurnMachine(POLICY)
    state = listening(machine)
    assert turn_phashes(state) == (), "criterion 4: a turn opens with no scene-change frames"

    before = state
    state, cmds = machine.step(state, scene(1200, PH_A))
    assert turn_phashes(state) == (PH_A,)
    assert list(cmds) == expected_commands(before, 1200), (
        "criterion 1: snapshot_frame(at_ms=ev.t_ms) then prefill_reissue(active_prefill), "
        f"unchanged from today; got {kinds(cmds)}"
    )
    assert state.state is TurnState.LISTENING, "a scene change does not move the machine"

    before = state
    state, cmds = machine.step(state, scene(1400, PH_B))
    assert turn_phashes(state) == (PH_A, PH_B), "appended at the end, not at the front"
    assert list(cmds) == expected_commands(before, 1400)

    wide = TurnMachine(policy_with_cap(4))
    state = listening(wide)
    for t_ms, phash in ((1200, PH_A), (1400, PH_B), (1600, PH_C)):
        before = state
        state, cmds = wide.step(state, scene(t_ms, phash))
        assert list(cmds) == expected_commands(before, t_ms)
    assert turn_phashes(state) == (PH_A, PH_B, PH_C), (
        "criterion 1: three distinct changes, in arrival order, when the cap leaves room"
    )


def test_a_scene_change_without_a_live_prefill_records_nothing():
    """Criterion 1's precondition, two-sided (the §4 row is LISTENING **with** a prefill).

    ``test_scene_change_outside_listening_is_ignored`` (T-0003 criterion 2) pins the states; this
    pins the other half of the guard on the row that *is* LISTENING: with no prefill to re-warm
    there is no request the frame could reach, so nothing is recorded either. A machine that
    appended here would carry a hash the prompt never sent.
    """
    turn_phashes(MachineState())  # criterion 1 must exist before this row means anything
    machine = TurnMachine(POLICY)
    no_prefill = listening(machine).model_copy(update={"active_prefill": None})
    after, cmds = machine.step(no_prefill, scene(1200, PH_A))
    assert list(cmds) == []
    assert after == no_prefill, "no prefill to re-warm: the event has no row"
    assert turn_phashes(after) == ()


def test_a_scene_change_is_ignored_in_endpointing_and_in_speaking():
    """Criterion 1's state guard, driven: ENDPOINTING (a live prefill) and SPEAKING (ruling A9).

    The §4 row for ``scene.changed`` is LISTENING and nothing else. T-0003's deterministic row
    (``test_scene_change_outside_listening_is_ignored``) covers IDLE and THINKING, and reviewer 1
    measured what that leaves open: a guard widened to ENDPOINTING survives the stateful run at
    seed 0 and only fails at seeds 1-4, so the §4 row nearest to LISTENING was pinned by trace
    luck. ENDPOINTING is the dangerous one because the prefill is still live there — the user has
    stopped speaking, the transcript is frozen, and a frame captured now would be recorded into a
    turn whose prompt is about to be built, one frame later than the words it goes with.

    Both states are reached by driving real events, and each half is checked for being vacuous
    before it asserts: ENDPOINTING genuinely holds a live prefill; SPEAKING has had it aborted by
    ``_start_thinking``, so the second half re-arms one by hand — a guard widened to SPEAKING
    would otherwise fail on the ``active_prefill is None`` clause and the row would prove nothing
    about the state clause it is here for.
    """
    machine = TurnMachine(POLICY)
    state = with_two_frames(machine)
    endpointing, _ = machine.step(state, E.SttEndOfTurn(t_ms=1800, t_eot_ms=1750))
    assert endpointing.state is TurnState.ENDPOINTING
    assert endpointing.active_prefill is not None, (
        "the row is only meaningful while the prefill is live: the guard under test is the state"
    )
    after, cmds = machine.step(endpointing, scene(1850, PH_C))
    assert list(cmds) == [], f"ENDPOINTING has no scene.changed row; got {kinds(cmds)}"
    assert after == endpointing, (
        f"the frozen turn recorded a screen: {turn_phashes(endpointing)} -> {turn_phashes(after)}"
    )

    machine = TurnMachine(POLICY)
    thinking = to_thinking_with_two_frames(machine)
    commit = E.LlmCommit(
        t_ms=3000,
        request_id=thinking.active_generate,
        canvas_idx=0,
        segment="answer",
        text="The left one.",
        thought_open=False,
    )
    driven, _ = machine.step(thinking, commit)
    assert driven.state is TurnState.SPEAKING
    assert driven.active_prefill is None, (
        "T-0003: the prefill is aborted on the way to THINKING, so the state clause is the only "
        "thing standing between a scene change and the answer being re-prefilled mid-sentence"
    )
    relit = driven.model_copy(update={"active_prefill": f"{driven.turn_id}-prefill"})
    after, cmds = machine.step(relit, scene(3100, PH_C))
    assert list(cmds) == [], f"SPEAKING has no scene.changed row; got {kinds(cmds)}"
    assert after == relit, (
        f"a screen was recorded while the assistant was speaking: {turn_phashes(after)}"
    )


OUTSIDE_LISTENING = [s for s in TurnState if s is not TurnState.LISTENING]


@pytest.mark.parametrize("state", OUTSIDE_LISTENING, ids=[s.value for s in OUTSIDE_LISTENING])
def test_no_state_but_listening_records_a_screen_however_live_the_prefill(state):
    """Criterion 1's state guard over the whole state space (ruling A9, the general case).

    The companion above drives the two states that matter in practice; this one closes the table,
    since the machine is a pure function of ``(state, event)`` and the §4 row exists in exactly
    one cell. Every state is given a live prefill and a turn that already holds a screen, so the
    row fails for the state clause and not for the prefill clause.
    """
    turn_phashes(MachineState())
    machine = TurnMachine(POLICY)
    before = MachineState(
        state=state,
        turn_id="t3",
        turn_seq=3,
        onset_ms=1000,
        active_prefill="t3-prefill",
        turn_phashes=(PH_A,),
    )
    frozen = before.model_copy(deep=True)
    after, cmds = machine.step(before, scene(2000, PH_B))
    assert list(cmds) == [], f"scene.changed in {state.value} emitted {kinds(cmds)}"
    assert after == frozen, f"scene.changed in {state.value} changed the state"
    assert before == frozen, "the machine is pure: the input state is not mutated in place"


# --- criterion 2: the same screen twice costs nothing -------------------------------------------


def test_the_same_screen_twice_emits_nothing():
    """Criterion 2, adversarial: ``A, B, A`` and ``A, A, A`` (D3, the cost criterion).

    The assertion is that the returned state is **equal to its input**, not merely that no
    snapshot was emitted: an implementation that recorded the repeat (moving it to the end, or
    growing the tuple) and only suppressed the command would still fail here, and that
    implementation would silently reorder the frames the prompt carries.
    """
    machine = TurnMachine(POLICY)
    state = listening(machine)
    state, first = machine.step(state, scene(1200, PH_A))
    state, second = machine.step(state, scene(1400, PH_B))
    frozen = state.model_copy(deep=True)

    repeat, third = machine.step(state, scene(1600, PH_A))
    assert list(third) == [], (
        f"criterion 2: a screen already in turn_phashes costs no snapshot and no re-issue; "
        f"got {kinds(third)}"
    )
    assert repeat == frozen, (
        "criterion 2: the returned state is equal to its input -- the repeat is not recorded, "
        f"not re-ordered, not counted. turn_phashes went {turn_phashes(frozen)} -> "
        f"{turn_phashes(repeat)}"
    )
    assert state == frozen, "the machine is pure: the input state is not mutated in place"
    assert turn_phashes(repeat) == (PH_A, PH_B)
    assert len([c for c in (*first, *second, *third) if isinstance(c, C.SnapshotFrame)]) == 2, (
        "criterion 2: A, B, A is two snapshots"
    )

    machine = TurnMachine(POLICY)
    state = listening(machine)
    all_cmds: list = []
    for t_ms in (1200, 1400, 1600):
        state, cmds = machine.step(state, scene(t_ms, PH_A))
        all_cmds.extend(cmds)
    assert turn_phashes(state) == (PH_A,), "criterion 2: A, A, A records one screen"
    assert kinds(all_cmds) == ["snapshot_frame", "prefill_reissue"], (
        f"criterion 2: A, A, A is one snapshot and one re-issue; got {kinds(all_cmds)}"
    )


# --- criterion 3: the cap holds and the last slot stays live ------------------------------------


def test_at_the_cap_the_newest_frame_replaces_the_last_slot():
    """Criterion 3 / D1: at the cap the newest screen replaces the last entry, and only it.

    Hand-computed at the shipped default (``max_frames_per_turn == 3``, D2 / ruling A1). The
    onset frame is the turn's first frame and carries no hash the machine can see (D3), so two
    scene changes fill the turn::

        A        -> (A,)
        A, B     -> (A, B)          the turn now holds 3 frames: onset, A, B
        A, B, C  -> (A, C)          C replaces the *last* slot, A is never touched
        ..., D   -> (A, D)

    The alternative the task rejects ("ignore everything past the cap", *Ambiguities* 2) freezes
    the tuple at ``(A, B)`` and stops re-issuing; a FIFO ("drop the oldest") gives ``(B, C)``.
    Both are red here, in the entry the row names.
    """
    cap = frame_cap()
    assert cap == 3, f"D2 and ruling A1: the machine's default cap is 3 frames per turn, not {cap}"
    slots = tuple_slots()

    machine = TurnMachine(POLICY)
    state = with_two_frames(machine)
    assert len(turn_phashes(state)) == slots == 2

    before = state
    state, cmds = machine.step(state, scene(1600, PH_C))
    assert list(cmds) == expected_commands(before, 1600), (
        "criterion 3: the replacing frame is still snapshotted and still re-issued"
    )
    assert turn_phashes(state) == (PH_A, PH_C), (
        f"criterion 3: the newest screen takes the last slot; got {turn_phashes(state)}"
    )

    before = state
    state, cmds = machine.step(state, scene(1800, PH_D))
    assert list(cmds) == expected_commands(before, 1800)
    assert turn_phashes(state) == (PH_A, PH_D), "the last slot is the live one, again"
    assert turn_phashes(state)[0] == PH_A, (
        "D1: a frame that has been sent in a prefill is never removed or reordered -- the "
        "prefix the engine has already cached must survive the whole turn"
    )
    assert len(turn_phashes(state)) == slots, "the tuple never grows past max_frames_per_turn - 1"


def test_the_cap_follows_the_policy_it_was_given():
    """Criterion 3 at a non-default cap: the arithmetic reads the policy, not a literal.

    An implementation that hard-coded 3 (or 2) instead of reading ``policy.max_frames_per_turn``
    passes every default-cap row above and fails here, which is also what makes criterion 5's
    rollback lever a lever rather than a third behaviour.
    """
    machine = TurnMachine(policy_with_cap(4))
    state = listening(machine)
    for i, phash in enumerate((PH_A, PH_B, PH_C)):
        state, _ = machine.step(state, scene(1200 + 100 * i, phash))
    assert turn_phashes(state) == (PH_A, PH_B, PH_C)
    before = state
    state, cmds = machine.step(state, scene(1600, PH_D))
    assert list(cmds) == expected_commands(before, 1600)
    assert turn_phashes(state) == (PH_A, PH_B, PH_D), (
        "at cap 4 the tuple holds three scene changes and the fourth replaces the last slot"
    )
    assert frame_cap(machine.policy) == 4, "the machine reads its cap from the policy it was given"


# --- criterion 4: a turn boundary clears it, continuing a turn does not -------------------------


def test_a_new_turn_starts_with_no_frames_and_a_continued_turn_keeps_them():
    """Criterion 4 over all four turn-opening rows plus the ENDPOINTING -> LISTENING row.

    The opening rows are T-0003 criteria 1, 15 and 19: onset from IDLE, ``user.takeover`` from
    THINKING, and both INTERRUPTED exits (``playback.stopped`` and the interrupt timeout). Each
    is driven from a state that *does* hold frames, so the assertion cannot be satisfied by the
    previous row having already cleared them.
    """
    turn_phashes(MachineState())  # the field is the criterion; fail on it, not on ValidationError

    # (a) onset from IDLE (T-0003 criterion 1). A synthetic IDLE state carrying a stale tuple: a
    # turn that ended has already cleared it, so only a state built by hand asks the question the
    # criterion asks -- that the *opening* row clears it.
    machine = TurnMachine(POLICY)
    stale = MachineState(state=TurnState.IDLE, turn_id="t7", turn_seq=7, turn_phashes=(PH_A, PH_B))
    opened, cmds = machine.step(stale, E.UserSpeechOnset(t_ms=5000, onset_ms=5000))
    assert opened.state is TurnState.LISTENING and "snapshot_frame" in kinds(cmds)
    assert turn_phashes(opened) == (), "criterion 4: onset from IDLE opens a turn with no frames"

    # (b) user.takeover from THINKING (T-0003 criterion 15).
    machine = TurnMachine(POLICY)
    thinking = to_thinking_with_two_frames(machine)
    taken, _ = machine.step(thinking, E.UserTakeover(t_ms=3400, onset_ms=3300))
    assert taken.turn_id != thinking.turn_id, "the takeover opened a new turn"
    assert turn_phashes(taken) == (), (
        "criterion 4: the new turn does not inherit the screens of the turn it interrupted"
    )

    # (c) playback.stopped from INTERRUPTED (T-0003 criterion 19).
    machine = TurnMachine(POLICY)
    interrupted = to_interrupted_with_two_frames(machine)
    relistened, _ = machine.step(
        interrupted,
        E.PlaybackStopped(t_ms=3600, played_ms=200, played_text="The left one."),
    )
    assert relistened.state is TurnState.LISTENING
    assert relistened.turn_id != interrupted.turn_id
    assert turn_phashes(relistened) == (), "criterion 4: the turn after a barge-in starts empty"

    # (d) the interrupt timeout from INTERRUPTED (T-0003 criterion 19, the timeout path).
    machine = TurnMachine(POLICY)
    interrupted = to_interrupted_with_two_frames(machine)
    deadline = next(t.at_ms for t in interrupted.timers if t.name == "interrupt_timeout")
    timed_out, _ = machine.step(interrupted, E.Tick(t_ms=deadline))
    assert timed_out.state is TurnState.LISTENING
    assert timed_out.turn_id != interrupted.turn_id
    assert turn_phashes(timed_out) == (), "criterion 4: the timeout path opens a turn too"

    # (e) ENDPOINTING -> LISTENING on a new onset (T-0003 criterion 5): the *same* turn, still
    # being spoken, so the frames stay.
    machine = TurnMachine(POLICY)
    state = with_two_frames(machine)
    state, _ = machine.step(state, E.SttEndOfTurn(t_ms=1800, t_eot_ms=1750))
    assert state.state is TurnState.ENDPOINTING
    assert turn_phashes(state) == (PH_A, PH_B), "endpointing is not a turn boundary"
    resumed, cmds = machine.step(state, E.UserSpeechOnset(t_ms=1900, onset_ms=1880))
    assert resumed.state is TurnState.LISTENING and "unfreeze" in kinds(cmds)
    assert resumed.turn_id == state.turn_id, "the same turn is still being spoken"
    assert turn_phashes(resumed) == (PH_A, PH_B), (
        "criterion 4: ENDPOINTING -> LISTENING keeps the frames -- the user never stopped talking"
    )


def test_a_finished_turn_leaves_no_frames_behind():
    """Criterion 4 / criterion 6's IDLE invariant: the end of a turn clears the tuple too.

    Driven to IDLE the ordinary way (done + tts.done + drained), because "in IDLE,
    ``turn_phashes == ()``" is a Hypothesis invariant of ``test_turn_invariants.py`` and a
    machine that only cleared on *opening* would carry a finished turn's screens through the
    whole idle period, with the next turn's opening row as the only thing between a stale screen
    and the prompt.
    """
    machine = TurnMachine(POLICY)
    state = to_thinking_with_two_frames(machine)
    request_id = state.active_generate
    commit = E.LlmCommit(
        t_ms=3000,
        request_id=request_id,
        canvas_idx=0,
        segment="answer",
        text="The left one.",
        thought_open=False,
    )
    state, _ = machine.step(state, commit)
    state, _ = machine.step(state, E.LlmDone(t_ms=3100, request_id=request_id))
    state, _ = machine.step(state, E.TtsDone(t_ms=3200, utterance_id=state.utterances[-1]))
    state, _ = machine.step(
        state, E.PlaybackDrained(t_ms=3500, played_ms=300, played_text="The left one.")
    )
    assert state.state is TurnState.IDLE
    assert turn_phashes(state) == (), "IDLE owns no frames (criterion 6's invariant)"


# --- criterion 5: the rollback lever ------------------------------------------------------------


def test_one_frame_per_turn_is_todays_last_snapshot_wins():
    """Criterion 5: ``max_frames_per_turn=1`` restores docs/01 D9 exactly -- the newest wins.

    This is the criterion that makes the setting a *rollback lever* rather than a third
    behaviour, and it is the row that rejects the naive rule "append while under the cap, ignore
    anything past it": that rule freezes the turn on the onset frame at cap 1 and stops
    re-issuing, so the screen the user is pointing at while they finish the sentence never
    reaches the engine. Hand-computed: each distinct change emits snapshot + re-issue, and the
    tuple holds exactly one entry -- the newest.
    """
    machine = TurnMachine(policy_with_cap(1))
    assert tuple_slots(machine.policy) == 1
    state = listening(machine)

    before = state
    state, cmds = machine.step(state, scene(1200, PH_A))
    assert list(cmds) == expected_commands(before, 1200)
    assert turn_phashes(state) == (PH_A,)

    before = state
    state, cmds = machine.step(state, scene(1400, PH_B))
    assert list(cmds) == expected_commands(before, 1400), (
        "criterion 5: at cap 1 every distinct scene change still snapshots and re-issues"
    )
    assert turn_phashes(state) == (PH_B,), (
        f"criterion 5: the last snapshot wins -- the tuple holds exactly the newest hash, got "
        f"{turn_phashes(state)}"
    )

    frozen = state.model_copy(deep=True)
    repeat, cmds = machine.step(state, scene(1600, PH_B))
    assert list(cmds) == [] and repeat == frozen, "criterion 2 still holds at cap 1"

    before = state
    state, cmds = machine.step(state, scene(1800, PH_C))
    assert list(cmds) == expected_commands(before, 1800)
    assert turn_phashes(state) == (PH_C,)
    assert len(turn_phashes(state)) == 1, "one frame per turn means one slot, forever"


# --- criterion 6 (this module's half): the accumulation as a property ---------------------------


@settings(max_examples=200, deadline=None)
@given(sequence=st.lists(st.sampled_from(PHASHES), min_size=1, max_size=12))
def test_the_frames_of_a_turn_are_ordered_deduplicated_and_bounded(sequence):
    """Criterion 6 over one turn: the facts D1 promises, checked at every step.

    Stated as independent properties rather than as an expected tuple, so the row is not the
    rule re-implemented: (i) no duplicate; (ii) at most ``max_frames_per_turn - 1`` entries;
    (iii) the *first* entry is the turn's first screen and never moves -- the prefill the engine
    has already cached stays a byte-prefix of the next one; (iv) the live screen is always
    represented, and when it is new it is in the last slot. Plus the step-local shape: entries
    other than the last never change, nothing is lost inside a turn, and the two commands are
    emitted exactly for the screens the tuple did not already hold (the cost criterion).
    """
    slots = tuple_slots()
    machine = TurnMachine(POLICY)
    state = listening(machine)
    for i, phash in enumerate(sequence):
        t_ms = 2000 + 10 * i
        was = state
        before = turn_phashes(state)
        state, cmds = machine.step(state, scene(t_ms, phash))
        after = turn_phashes(state)

        if phash in before:
            assert list(cmds) == [], f"{phash} was already recorded: {kinds(cmds)}"
            assert after == before, "a repeat leaves the tuple alone"
        else:
            assert list(cmds) == expected_commands(was, t_ms), (
                f"a screen not in {before} must be snapshotted and re-issued; got {kinds(cmds)}"
            )
            assert after[-1] == phash, (
                f"D1: the newest screen is the live one, in the last slot; got {after}"
            )

        assert len(set(after)) == len(after), f"a screen is recorded twice: {after}"
        assert len(after) <= slots, f"{after} exceeds max_frames_per_turn - 1 = {slots}"
        assert len(after) >= len(before), "a frame is lost only at a turn boundary"
        keep = len(after) - 1
        assert after[:keep] == before[:keep], (
            f"D1: only the last slot may change inside a turn; {before} -> {after}"
        )
        assert after[0] == sequence[0], (
            f"D1: the turn's first screen is never removed or reordered; got {after}"
        )
        assert phash in after, f"the live screen {phash} is not in {after}"


# --- criterion 7: nothing else about the machine moves ------------------------------------------


CONTRACT_FILES = ("local_astra/contracts/events.py", "local_astra/contracts/commands.py")

REPO = pathlib.Path(__file__).resolve().parents[2]


def _git(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, check=False, timeout=30
    )


#: T-0089 ruling T1: what T-0089 criterion 1 may add to each contract file, and nothing else.
T0089_NEW_CLASSES = {
    "local_astra/contracts/events.py": set(),
    "local_astra/contracts/commands.py": {"ToolRepairStart", "ToolExecute", "ToolCancel"},
}
#: (class, field) -> what changes: a new field, or a Literal that gains exactly one value.
T0089_FIELD_CHANGES = {
    ("LlmDone", "tool"): "new",
    ("ArmTimer", "name"): "tool_repair",
    ("CancelTimer", "name"): "tool_repair",
}
#: T-0112 (T-0111a §2, re-scoped in docs/tasks/T-0112.md "Re-scope after implementation"): the
#: loop adds exactly one event and one command, and the ``tool_result`` timer is one more value of
#: the same ``TimerName`` Literal. ``step_budget`` joins ``ToolCancel.reason``, a T-0089 class this
#: pin does not walk field by field. Nothing else may change.
T0112_NEW_CLASSES = {
    "local_astra/contracts/events.py": {"ToolResult"},
    "local_astra/contracts/commands.py": {"LlmContinue"},
}
T0112_LITERAL_GAINS = {
    ("ArmTimer", "name"): ("tool_result",),
    ("CancelTimer", "name"): ("tool_result",),
}
#: T-0162 A-2 (pre-authorised pin move, amendment 2026-10-27): the session reset adds exactly one
#: event and one command. ``"reset"`` joins ``ToolCancel.reason``, a T-0089 class this pin does not
#: walk field by field (as ``step_budget`` did for T-0112). Nothing else may change.
T0162_NEW_CLASSES = {
    "local_astra/contracts/events.py": {"SessionReset"},
    "local_astra/contracts/commands.py": {"ResetSession"},
}
#: T-0181 A0-1 (H-T181-2, James 2026-10-28): the mic mute adds exactly one event, ``MicMuted`` (``t_ms`` only), and
#: no command. Nothing else may change.
T0181_NEW_CLASSES = {
    "local_astra/contracts/events.py": {"MicMuted"},
    "local_astra/contracts/commands.py": set(),
}
#: T-0193 A0-1 (the HUD Stop button): exactly one event, ``UserStop`` (``t_ms``, ``turn_id``), and no command class.
#: ``"stopped"`` joins ``ToolCancel.reason``, a T-0089 class this pin does not walk field by field (as ``"reset"`` did
#: for T-0162). Nothing else may change.
T0193_NEW_CLASSES = {
    "local_astra/contracts/events.py": {"UserStop"},
    "local_astra/contracts/commands.py": set(),
}
#: T-0178 B6a re-pin (amendment B4a): ``ClipKind`` (``PlayClip.kind``) gains exactly the two ending clips, in this
#: order after BASE's values; no new class, no other field. Re-pinned after the implementation commit f8e26760.
#: T-0179 B re-pin (after ebaf4248; commands.py sha256 776033c5...): ``lost_thread`` is appended after T-0178's two,
#: and nothing else moves.
#: T-0192 B1 re-pin (before implementation): ``at_checkout`` is appended after T-0179's ``lost_thread``, and nothing
#: else moves (``ToolResult.hand_back`` is on a class T-0112 added, which this walk does not visit; its field set is
#: pinned by ``test_turn_tool_loop_rows.test_criterion_11_tool_result_carries_no_text``).
T0178_LITERAL_GAINS = {
    ("PlayClip", "kind"): ("out_of_steps", "not_finished", "lost_thread", "at_checkout"),
}


#: T-0089's base: ``main`` just before T-0089 branched (T-0090 branched from it too). A literal
#: SHA, never the moving ``main`` ref: once T-0089 merged, ``main`` already holds the additions
#: and a pin against it reports "adds []" (fixed on T-0089-frames-pin-fix).
T0089_BASE = "e0174f9"


def _load_main(relative: str):
    """Execute ``git show <T-0089 base>:<relative>`` as a throwaway module (today's imports)."""
    import sys
    import types

    shown = _git("show", f"{T0089_BASE}:{relative}")
    assert shown.returncode == 0, shown.stderr.decode()
    name = f"_main_{pathlib.Path(relative).stem}_t0089"
    module = types.ModuleType(name)
    sys.modules[name] = module
    try:
        exec(compile(shown.stdout, f"{T0089_BASE}:{relative}", "exec"), module.__dict__)
    finally:
        del sys.modules[name]
    return module


def _own_models(module) -> dict[str, type]:
    from pydantic import BaseModel

    return {
        k: v
        for k, v in vars(module).items()
        if isinstance(v, type) and issubclass(v, BaseModel) and v.__module__ == module.__name__
    }


def _field_shape(info) -> tuple:
    return (info.annotation, info.default, info.is_required(), info.description, info.metadata)


def _code_shape(func) -> object:
    """A function's behaviour as data: bytecode, names and constants (recursively), never its
    identity or line numbers -- main's class is exec'd from ``git show``, so ``is`` never holds."""
    f = getattr(func, "__func__", func)
    code = getattr(f, "__code__", None)
    if code is None:
        return repr(type(f))

    def shape(c) -> tuple:
        consts = tuple(shape(x) if hasattr(x, "co_code") else repr(x) for x in c.co_consts)
        return (c.co_name, c.co_code, c.co_names, c.co_varnames, consts)

    return shape(code)


def _decorator_shape(cls) -> dict:
    """T-0089 ruling T12: every validator, serializer and computed field pydantic collected."""
    infos = cls.__pydantic_decorators__
    kinds = (
        "validators",
        "field_validators",
        "root_validators",
        "field_serializers",
        "model_serializers",
        "model_validators",
        "computed_fields",
    )
    return {
        kind: {
            name: (repr(dec.info), _code_shape(dec.func))
            for name, dec in getattr(infos, kind).items()
        }
        for kind in kinds
    }


def _schema_without_t0089(cls_name: str, schema: dict) -> dict:
    """The JSON schema with exactly T-0089's additions taken back out (ruling T12).

    ``LlmDone`` loses its ``tool`` property and the ``ToolDraft`` definition it brought;
    ``ArmTimer``/``CancelTimer`` lose the ``tool_repair`` timer name. Anything else T-0089 might
    have changed in those schemas survives the subtraction and fails the comparison.
    """
    import copy

    out = copy.deepcopy(schema)
    for (name, field), change in T0089_FIELD_CHANGES.items():
        if name != cls_name:
            continue
        if change == "new":
            out["properties"].pop(field)
            if field in out.get("required", []):
                out["required"].remove(field)
            defs = out.get("$defs", {})
            defs.pop("ToolDraft", None)
            if "$defs" in out and not defs:
                del out["$defs"]
        else:
            enum = out["properties"][field]["enum"]
            gone = {change, *T0112_LITERAL_GAINS.get((name, field), ())}
            out["properties"][field]["enum"] = [v for v in enum if v not in gone]
    for (name, field), gains in T0178_LITERAL_GAINS.items():
        if name == cls_name:
            enum = out["properties"][field]["enum"]
            out["properties"][field]["enum"] = [v for v in enum if v not in gains]
    return out


@pytest.mark.parametrize("relative", CONTRACT_FILES)
# T-0181 A0-1: re-scoped before implementation (``T0181_NEW_CLASSES``: events.py gains exactly MicMuted)
# T-0178 B6a: re-pinned after the implementation commit f8e26760 (``T0178_LITERAL_GAINS``: ClipKind gains exactly
# "out_of_steps", "not_finished")
def test_the_event_and_command_contracts_are_byte_identical_to_main(relative):
    """Criterion 7: no new event, no new command, no new field on ``SnapshotFrame``.

    T-0003 criterion 24's row already pins the *type names*; this pins every field of every
    contract class, because a field carrying a frame list to the executor would be added to an
    existing command and would change no name at all.

    "main" here is ``T0089_BASE`` (e0174f9, main before T-0089 branched), not the moving ref.

    Widened by T-0089 ruling T1 from a byte comparison against ``git show main:<path>`` to a
    class-by-class one: every class on main keeps every field (annotation, default, required,
    description, constraints) and gains none, except exactly ``LlmDone.tool`` and the
    ``tool_repair`` value of ``TimerName`` on ``ArmTimer``/``CancelTimer.name``; the only new
    classes are the three tool commands.

    Made exact by ruling T12: each main class's ``model_json_schema()`` (with exactly those
    additions subtracted), its ``__pydantic_decorators__`` and its ``model_config`` must equal
    main's, so an alias or an added validator on a main class is caught too.
    """
    import typing

    if _git("cat-file", "-e", f"{T0089_BASE}^{{commit}}").returncode != 0:
        pytest.skip(
            f"T-0089 base commit {T0089_BASE} is not available in this checkout (shallow clone?): "
            "criterion 7 is checked against main as it was before T-0089"
        )
    main_module = _load_main(relative)
    old = _own_models(main_module)
    here_module = E if relative.endswith("events.py") else C
    here = _own_models(here_module)
    added = set(here) - set(old)
    assert not set(old) - set(here), f"criterion 7: {relative} lost {sorted(set(old) - set(here))}"
    allowed = (
        T0089_NEW_CLASSES[relative]
        | T0112_NEW_CLASSES[relative]
        | T0162_NEW_CLASSES[relative]
        | T0181_NEW_CLASSES[relative]
        | T0193_NEW_CLASSES[relative]
    )
    assert added == allowed, (
        f"criterion 7 / T-0089 T1 / T-0112: {relative} may add only {sorted(allowed)}; "
        f"it adds {sorted(added)}"
    )
    for cls_name, old_cls in old.items():
        new_cls = here[cls_name]
        old_fields, new_fields = old_cls.model_fields, new_cls.model_fields
        for field in sorted(set(old_fields) | set(new_fields)):
            change = T0089_FIELD_CHANGES.get((cls_name, field))
            if field not in old_fields:
                assert change == "new", f"criterion 7: {cls_name}.{field} is a new field"
                continue
            assert field in new_fields, f"criterion 7: {cls_name}.{field} was removed"
            was, now = _field_shape(old_fields[field]), _field_shape(new_fields[field])
            gains178 = T0178_LITERAL_GAINS.get((cls_name, field))
            if change is None and gains178 is not None:
                # T-0178 B6a: exactly the two ending clips appended, nothing else about the field moves
                assert typing.get_args(now[0]) == (*typing.get_args(was[0]), *gains178), (
                    f"T-0178 B4a: {cls_name}.{field} gains exactly {gains178!r}; {was[0]} -> {now[0]}"
                )
                assert now[1:] == was[1:], f"{cls_name}.{field} changed beyond its Literal"
            elif change is None:
                assert now == was, f"criterion 7: {cls_name}.{field} changed: {was} -> {now}"
            else:
                values = typing.get_args(now[0])
                old_values = typing.get_args(was[0])
                gains = (change, *T0112_LITERAL_GAINS.get((cls_name, field), ()))
                assert sorted(values) == sorted((*old_values, *gains)), (
                    f"T-0089/T-0112: {cls_name}.{field} gains exactly {gains!r}; got {values}"
                )
                assert now[1:] == was[1:], f"{cls_name}.{field} changed beyond its Literal"
        # ruling T12: the field walk above cannot see an alias, a serialization setting, a
        # validator or a config change; the schema, the decorators and the config can.
        assert _schema_without_t0089(cls_name, new_cls.model_json_schema()) == (
            old_cls.model_json_schema()
        ), f"T-0089 T12: {cls_name}'s JSON schema changed beyond T-0089's additions"
        assert _decorator_shape(new_cls) == _decorator_shape(old_cls), (
            f"T-0089 T12: {cls_name}'s validators/serializers changed: "
            f"{_decorator_shape(old_cls)} -> {_decorator_shape(new_cls)}"
        )
        assert dict(new_cls.model_config) == dict(old_cls.model_config), (
            f"T-0089 T12: {cls_name}.model_config changed: "
            f"{old_cls.model_config} -> {new_cls.model_config}"
        )
    for (cls_name, field), change in T0089_FIELD_CHANGES.items():
        if cls_name in old and change == "new":
            assert field in here[cls_name].model_fields, f"T-0089: {cls_name}.{field} is missing"
    # the module-level tables the byte pin used to cover
    if relative.endswith("events.py"):
        # T-0112: main's events, in main's order, plus exactly ToolResult; T-0162: plus SessionReset;
        # T-0181 A0-1: plus MicMuted
        gained = (
            "ToolResult",
            "SessionReset",
            "MicMuted",
            "UserStop",
        )  # T-0193 A0-1: plus UserStop
        new_types = [m.__name__ for m in E.EVENT_TYPES]
        assert [n for n in new_types if n not in gained] == [
            m.__name__ for m in main_module.EVENT_TYPES
        ], "criterion 7: EVENT_TYPES changed"
        for name in gained:
            assert new_types.count(name) == 1, f"T-0112/T-0162: EVENT_TYPES {new_types}"
        union_old = typing.get_args(typing.get_args(main_module.Event)[0])
        union_new = [m.__name__ for m in typing.get_args(typing.get_args(E.Event)[0])]
        assert [n for n in union_new if n not in gained] == [m.__name__ for m in union_old]
        for name in gained:
            assert union_new.count(name) == 1, f"T-0112/T-0162: the Event union {union_new}"
    else:
        union_old = [m.__name__ for m in typing.get_args(typing.get_args(main_module.Command)[0])]
        union_new = [m.__name__ for m in typing.get_args(typing.get_args(C.Command)[0])]
        assert sorted(union_new) == sorted([*union_old, *allowed]), (
            "T-0089 T1 / T-0112: the Command union is main's plus the three tool commands and "
            f"LlmContinue; got {union_new}"
        )
        assert C.IDEMPOTENT_COMMANDS == main_module.IDEMPOTENT_COMMANDS
        # T-0178 B6a re-pin: main's clip kinds plus exactly the two ending clips, appended
        assert typing.get_args(C.ClipKind) == (
            *typing.get_args(main_module.ClipKind),
            *T0178_LITERAL_GAINS[("PlayClip", "kind")],
        )


#: Every ``MachineState`` field on main, with the annotation and default it has there
#: (criterion 7: "every existing MachineState field keeps its name, type and default").
MACHINE_STATE_FIELDS: dict[str, tuple[object, object]] = {
    "state": (TurnState, TurnState.IDLE),
    "turn_id": (str, "t0"),
    "turn_seq": (int, 0),
    "onset_ms": (int | None, None),
    "t_eot_ms": (int | None, None),
    "words": (tuple[Word, ...], ()),
    "next_words": (tuple[Word, ...], ()),
    "active_generate": (str | None, None),
    "active_prefill": (str | None, None),
    "committed_answer": (str, ""),
    "spoken_text": (str, ""),
    "held_text": (str, ""),
    "utterances": (tuple[str, ...], ()),
    "utterance_seq": (int, 0),
    "timers": (tuple[Timer, ...], ()),
    "history": (tuple[HistoryTurn, ...], ()),
    "fillers_this_turn": (int, 0),
    "first_canvas_ms": (int | None, None),
    "first_canvas_measured": (bool, False),
    "llm_start_ms": (int | None, None),
    "flush_t_ms": (int | None, None),
    "pause_onset_ms": (int | None, None),
    "played_text": (str, ""),
    "llm_done": (bool, False),
    "tts_done": (bool, False),
    "playback_drained": (bool, False),
    "clip_playing": (bool, False),
    "degraded": (bool, False),
}

#: Every ``TurnPolicy`` field on main, with its type and default (criterion 7, same sentence).
TURN_POLICY_FIELDS: dict[str, tuple[type, object]] = {
    "eot_settle_ms": (int, 120),
    "eot_p_threshold": (float, 0.6),
    "owner_check_min_speech_ms": (int, 1500),
    "takeover_ms": (int, 600),
    "resume_max_ms": (int, 600),
    "pause_latency_budget_ms": (int, 150),
    "interrupt_timeout_ms": (int, 300),
    "llm_timeout_ms": (int, 20_000),
    "interim_update_ms": (int, 4_000),
    "played_word_margin_ms": (int, 100),
    "ack_first_canvas_ema_ms": (int, 300),
    "first_canvas_ema_alpha": (float, 0.3),
    "thinking": (bool, True),
    "max_history_turns": (int, 3),
    "max_buffered_words": (int, 256),
}


def test_every_existing_machine_state_field_keeps_its_name_type_and_default():
    """Criterion 7, plus the fence around the *shape* of the one addition.

    The tuple is the addition the implementation notes insist on ("hashes, not counters": an
    integer counting dropped scene changes grows without bound over a long LISTENING and would
    break T-0003 criterion 23). A second field here -- a counter, a "frames seen" int, a
    last-phash cache -- fails this row; "how many were dropped" belongs to group C, where the
    session sees every event it submitted.
    """
    fields = MachineState.model_fields
    missing = sorted(set(MACHINE_STATE_FIELDS) - set(fields))
    assert not missing, f"criterion 7: MachineState lost {missing}"
    for name, (annotation, default) in MACHINE_STATE_FIELDS.items():
        assert fields[name].annotation == annotation, (
            f"criterion 7: MachineState.{name} changed type to {fields[name].annotation}"
        )
        assert fields[name].default == default, (
            f"criterion 7: MachineState.{name} default changed to {fields[name].default!r}"
        )
    added = sorted(set(fields) - set(MACHINE_STATE_FIELDS))
    # T-0182 B re-scope (B-R5, "Implementation constraints": B4's state attribute is the only new MachineState
    # member). Its name is not fixed by the task: exactly one name beyond the list below, either a model field
    # declared ``exclude=True`` or a private attribute (MachineState has none at BASE); it is set aside here and
    # pinned by test_t0182_clip_policy.py's B4 rows.
    known = ["spoken_after_drain", "tool", "tool_repairs", "turn_phashes", "tool_step"]
    known += ["tool_steps_executed", "awaiting_result", "step_text", "played_prior"]
    # T-0193 A0-1 (census move): A0's per-turn stop field is the name beyond fdb9fc01's fields; it is set aside by that
    # name difference (at most one, declared exclude=True) and pinned by test_t0193_stop_row.stop_field
    from tests.unit.test_t0193_stop_row import BASE_STATE_FIELDS as T0193_BASE_FIELDS

    t0193 = sorted(set(fields) - set(T0193_BASE_FIELDS))
    assert len(t0193) == 1, (
        f"T-0193 A0-1: exactly one MachineState field beyond fdb9fc01's; {t0193}"
    )
    for name in t0193:
        assert fields[name].exclude is True, f"T-0193 A0-1: {name} is declared Field(exclude=True)"
    added = [n for n in added if n not in t0193]
    extra_fields = [n for n in added if n not in known]
    private = list(getattr(MachineState, "__private_attributes__", {}))
    assert len(extra_fields) + len(private) == 1, (
        "T-0182 B4/B-R5: exactly one new MachineState member (B4's attribute); "
        f"new fields {extra_fields}, private attributes {private}"
    )
    for name in extra_fields:
        assert fields[name].exclude is True, f"T-0182 B-R5: {name} is declared Field(exclude=True)"
    added = [n for n in added if n not in extra_fields]
    # T-0089 ruling T1: T-0089 criterion 1 adds exactly ``tool`` and ``tool_repairs`` beside it,
    # and ruling T15 exactly ``spoken_after_drain`` (a bool, default False).
    # T-0112 (T-0111a §2 MachineState): exactly the five per-turn loop fields, with their defaults.
    t0112 = {
        "tool_step": (int, 1),
        "tool_steps_executed": (int, 0),
        "awaiting_result": (str | None, None),
        "step_text": (str, ""),
        "played_prior": (str, ""),
    }
    assert added == sorted(
        ["spoken_after_drain", "tool", "tool_repairs", "turn_phashes", *t0112]
    ), (
        "T-0059 group A adds exactly one MachineState field, turn_phashes, T-0089 exactly "
        f"tool, tool_repairs and spoken_after_drain, and T-0112 exactly {sorted(t0112)}; "
        f"got {added}"
    )
    for name, (annotation, default) in t0112.items():
        assert (fields[name].annotation, fields[name].default) == (annotation, default), (
            f"T-0112: MachineState.{name} is {annotation} = {default!r}"
        )
    assert fields["spoken_after_drain"].annotation is bool, "T15: spoken_after_drain is a bool"
    assert fields["spoken_after_drain"].default is False, "T15: default False"
    assert fields["turn_phashes"].annotation == tuple[str, ...], (
        f"turn_phashes is a tuple of phashes, got {fields['turn_phashes'].annotation}"
    )
    assert MachineState().turn_phashes == (), "a fresh state has seen no scene change"


def test_every_existing_turn_policy_field_keeps_its_name_and_default():
    """Criterion 7 for the policy, plus the default the cap ships with (D2 / ruling A1)."""
    defaults = {f.name: f.default for f in dataclasses.fields(TurnPolicy)}
    missing = sorted(set(TURN_POLICY_FIELDS) - set(defaults))
    assert not missing, f"criterion 7: TurnPolicy lost {missing}"
    for name, (kind, default) in TURN_POLICY_FIELDS.items():
        assert defaults[name] == default, (
            f"criterion 7: TurnPolicy.{name} default changed to {defaults[name]!r}"
        )
        assert isinstance(getattr(POLICY, name), kind), f"TurnPolicy.{name} changed type"
    added = sorted(set(defaults) - set(TURN_POLICY_FIELDS))
    # T-0089 ruling T1: T-0089 criterion 1 adds exactly ``tool_repair_timeout_ms`` beside it.
    # T-0112 (T-0111a §2 Policy, H-T111a-1/-2): exactly three loop fields, with these defaults.
    t0112 = {"tool_loop": False, "max_tool_steps": 8, "tool_result_timeout_ms": 15_000}
    # T-0182 B re-scope (B1, B4, ruling B-R4: the product default lives in the CLI, so the contract defaults keep
    # BASE's behaviour): exactly ``ack_clip: bool = True`` and ``still_working_after_ms: int | None = None``
    t0182 = {"ack_clip": True, "still_working_after_ms": None}
    assert added == sorted(["max_frames_per_turn", "tool_repair_timeout_ms", *t0112, *t0182]), (
        "T-0059 group A adds exactly one TurnPolicy field, max_frames_per_turn, T-0089 "
        f"exactly tool_repair_timeout_ms, T-0112 exactly {sorted(t0112)}, and T-0182 B exactly "
        f"{sorted(t0182)}; got {added}"
    )
    assert {name: defaults[name] for name in t0112} == t0112, "T-0112: the loop defaults"
    got = {name: defaults[name] for name in t0182}
    assert got == t0182 and defaults["ack_clip"] is True, (
        f"T-0182 B-R4: the contract defaults; got {got}"
    )
    assert frame_cap() == 3, "D2: three current frames plus one history image is the engine's 4"


def test_turn_machine_init_keeps_policy_first_positional():
    """Criterion 7: ``session.py`` keeps constructing ``TurnMachine(policy)`` (T-0003 A24).

    Green today and green after: it exists so that "the machine needs its cap" cannot become a
    second constructor argument, or a keyword one, behind the session's back.
    """
    params = list(inspect.signature(TurnMachine.__init__).parameters)
    assert params[:2] == ["self", "policy"]
    assert inspect.signature(TurnMachine.__init__).parameters["policy"].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
