"""T-0179 group B, the pure machine and the clip registry: ``lost_thread`` joins T-0178's ending clips (docs/tasks/
T-0179.md B-6, B-7 as amended 2026-10-28 by T-0182 H-T182-3: at most one *ending* clip per turn, plus at most one
progress clip; B-8's invariants).

**What these rows can and cannot pin** (in the test-author's report): the machine's only input for a degenerate end
is ``LlmError(request_id, message)`` (``contracts/events.py`` is not owned by T-0179), and T-0178's ending-clip
emission lives in ``_finish_step`` (an ``LlmDone`` row). Whether the implementation emits ``PlayClip(kind=
"lost_thread")`` from the machine's error row or plays it from the session is therefore not pinned here; the session
rows (``test_t0179_degenerate_retry.py``) pin the observable clip. What this module pins for any implementation:

- the registry (pending, red at BASE): ``ClipKind`` gains exactly ``"lost_thread"``; ``CLIP_TEXT["lost_thread"]`` is
  H-T179-4's sentence verbatim; ``reflex.clips.registered()`` holds it and ``default_enabled`` enables it (T-0182 A1's
  rule: ending kinds play by default, so ``scripts/render_clips.py`` renders it);
- the Hypothesis property (active, green at BASE and after: a guard): T-0178's ``ToolLoopModel`` (imported from
  ``test_turn_invariants.py``, unedited) extended with ``LlmError`` events carrying the session's degenerate messages
  (every ``str`` constant of ``conductor.session`` whose value starts with ``"degenerate "``, read at collection, so a
  message the implementation adds is driven too). Over every trace: at most one ending clip
  (``out_of_steps`` | ``not_finished`` | ``lost_thread``) per turn, so never ``lost_thread`` together with
  ``out_of_steps``; a ``lost_thread`` only in the tuple of an ``LlmError`` whose message is a degenerate one (B-7: only on
  a ``degenerate`` end), after that tuple's ``HudError`` (B-6); no ``PlayClip`` ever carries a non-empty ``source``; no
  clip text ever enters history; and every property of ``ToolLoopModel`` (L1, P4, P6', L9, L10, ...) still holds.

Regime: CPU, pure, Hypothesis stateful (300 examples x 80 steps, the ``TestToolLoopInvariants`` settings), no
measurement.
"""

from __future__ import annotations

import typing
from typing import Any

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import invariant, rule

from local_astra.conductor import session as session_mod
from local_astra.conductor.turn_machine import TurnMachine
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import TurnState
from tests.unit import test_turn_invariants as ti
from tests.unit.test_turn_tool_rows import to_thinking

LOST_THREAD = "lost_thread"
LOST_THREAD_TEXT = "Sorry, I lost my train of thought. Could you ask again?"  # H-T179-4, verbatim
#: B-7 as amended: the ending kinds (T-0182's ``ENDING_KINDS``, written out by hand)
ENDING = ("out_of_steps", "not_finished", LOST_THREAD)
#: T-0178's kinds, as BASE (aeac453f) has them
BASE_KINDS = {"ack", "hmm", "still_working", "out_of_steps", "not_finished"}

#: amendment B-6a: the retried-end messages (``DEGENERATE_RETRIED_MESSAGE`` and any segment variant), by name
RETRIED_MESSAGES = tuple(
    sorted(
        {
            v
            for k, v in vars(session_mod).items()
            if k.startswith("DEGENERATE_RETRIED") and isinstance(v, str)
        }
    )
)
#: every degenerate message the session can hand the machine, read off the module (never typed here)
DEGENERATE_MESSAGES = tuple(
    sorted(
        {
            v
            for k, v in vars(session_mod).items()
            if k.isupper() and isinstance(v, str) and v.startswith("degenerate ")
        }
        | set(RETRIED_MESSAGES)
    )
)


def retried_message() -> str:
    value = getattr(session_mod, "DEGENERATE_RETRIED_MESSAGE", None)
    assert isinstance(value, str) and value.strip(), (
        "T-0179 B-6a: session.DEGENERATE_RETRIED_MESSAGE (a non-empty str module constant) does not exist yet"
    )
    return value


# --- the registry (B-6) -------------------------------------------------------------------------------------------


def test_clip_kind_gains_lost_thread() -> None:
    """B-6 (owned: ``contracts/commands.py``, ``ClipKind`` only): ``ClipKind`` is BASE's kinds plus ``lost_thread``,
    and ``PlayClip`` takes it. T-0192 B1 re-pin (before implementation): T-0192 then adds ``at_checkout``, and only
    that."""
    kinds = set(typing.get_args(C.ClipKind))
    assert kinds == BASE_KINDS | {LOST_THREAD, "at_checkout"}, (
        f"T-0179 B-6 / T-0192 B1: ClipKind gains lost_thread, then at_checkout, only; {sorted(kinds)}"
    )
    assert C.PlayClip(kind=LOST_THREAD).kind == LOST_THREAD  # type: ignore[arg-type]


def test_lost_thread_text_and_registry() -> None:
    """B-6 / H-T179-4 / B-7 amendment: the fixed text, registered (kind and text both present) and enabled by default
    as an ending kind, so the renderer renders it and doctor reports it."""
    from local_astra.reflex import clips

    assert session_mod.CLIP_TEXT.get(LOST_THREAD) == LOST_THREAD_TEXT, session_mod.CLIP_TEXT.get(
        LOST_THREAD
    )
    registered = clips.registered()
    assert registered.get(LOST_THREAD) == LOST_THREAD_TEXT, registered
    assert LOST_THREAD in clips.default_enabled(registered), clips.default_enabled(registered)


def test_degenerate_messages_premise() -> None:
    """Premise of the property: the session's two T-0142/T-0142b messages are driven (a vacuous strategy would make the
    property pass for any machine)."""
    assert session_mod.DEGENERATE_MESSAGE in DEGENERATE_MESSAGES
    assert session_mod.DEGENERATE_ANSWER_MESSAGE in DEGENERATE_MESSAGES


# --- B-6a: the machine's LlmError row -------------------------------------------------------------------------------


def _error_tuple(clock: Any, policy: Any, message: str) -> tuple[Any, list[Any]]:
    """A turn taken to THINKING (its step-1 generate live), then ``LlmError(t1-gen, message)``."""
    m = TurnMachine(policy)
    s = to_thinking(m, clock)
    s, cmds = m.step(s, E.LlmError(t_ms=clock.advance(500), request_id="t1-gen", message=message))
    return s, list(cmds)


POLICIES = {
    "loop-off": ti.POLICY,
    "loop-on": ti.loop_policy(tool_loop=True),
}


@pytest.mark.parametrize("policy", list(POLICIES))
def test_retried_message_plays_lost_thread_after_hud_error(clock: Any, policy: str) -> None:
    """B-6a: ``LlmError`` with ``DEGENERATE_RETRIED_MESSAGE`` for the live generate gives the error ending (one
    ``HudError`` carrying the message, the turn ends) and then exactly one ``PlayClip(kind="lost_thread")`` with no
    source, after the ``HudError``."""
    message = retried_message()
    s, cmds = _error_tuple(clock, POLICIES[policy], message)
    names = [type(c).__name__ for c in cmds]
    huds = [c for c in cmds if isinstance(c, C.HudError)]
    plays = [c for c in cmds if isinstance(c, C.PlayClip)]
    assert [h.message for h in huds] == [message], f"T-0179 B-6a: the error ending; {names}"
    assert [(c.kind, c.source) for c in plays] == [(LOST_THREAD, None)], (
        f"T-0179 B-6a: exactly one PlayClip(lost_thread), no source; {plays}"
    )
    assert names.index("HudError") < names.index("PlayClip"), (
        f"T-0179 B-6a: after hud.error; {names}"
    )
    assert s.state is TurnState.IDLE, s.state
    for h in s.history:
        assert LOST_THREAD_TEXT not in h.assistant_spoken, h


@pytest.mark.parametrize("policy", list(POLICIES))
@pytest.mark.parametrize("which", ["DEGENERATE_MESSAGE", "DEGENERATE_ANSWER_MESSAGE", "other"])
def test_unretried_messages_play_no_clip(clock: Any, policy: str, which: str) -> None:
    """B-6a / H-T179-7 (the guard too wide): ``LlmError`` with T-0142's or T-0142b's message (a trip that was not
    retried) or any other message gives the error ending and no clip at all. Active: holds at BASE."""
    message = getattr(session_mod, which) if which != "other" else "vLLM returned HTTP 500"
    s, cmds = _error_tuple(clock, POLICIES[policy], message)
    assert [c.message for c in cmds if isinstance(c, C.HudError)] == [message], cmds
    assert [c for c in cmds if isinstance(c, C.PlayClip)] == [], (
        f"T-0179 B-6a: no clip on {which}; {[type(c).__name__ for c in cmds]}"
    )
    assert s.state is TurnState.IDLE, s.state


def test_machine_lost_thread_message_tracks_the_session() -> None:
    """B-12 drift guard (as T-0178 B3a's): the pure machine keeps the retried-end message as a plain string copy,
    ``turn_machine._LOST_THREAD_MESSAGE``; it must equal ``session.DEGENERATE_RETRIED_MESSAGE``, or a reworded session
    message would silently stop the clip. It is neither of T-0142's two messages (they must never play it)."""
    from local_astra.conductor import turn_machine

    got = getattr(turn_machine, "_LOST_THREAD_MESSAGE", None)
    want = retried_message()
    assert isinstance(got, str), f"T-0179 B-12: turn_machine._LOST_THREAD_MESSAGE is a str; {got!r}"
    assert got == want, f"T-0179 B-12: the machine's {got!r} drifted from the session's {want!r}"
    assert got not in (session_mod.DEGENERATE_MESSAGE, session_mod.DEGENERATE_ANSWER_MESSAGE), got


# --- the property (B-6, B-7, B-8) ----------------------------------------------------------------------------------


class LostThreadModel(ti.ToolLoopModel):
    """``ToolLoopModel`` plus degenerate ``LlmError`` events and the B-6/B-7 checks on every tuple."""

    def __init__(self) -> None:
        super().__init__()
        self.ending_clips: dict[str, list[str]] = {}

    @rule(
        which=st.sampled_from(DEGENERATE_MESSAGES or ("degenerate thought: request aborted",)),
        active=st.booleans(),
    )
    def degenerate_error(self, which: str, active: bool) -> None:
        """The session's degenerate end: ``LlmError`` for the active generate (or with no id)."""
        t = self._advance(5)
        rid = self.s.active_generate if active else None
        self._step(E.LlmError(t_ms=t, request_id=rid, message=which))

    def _step(self, ev):  # type: ignore[no-untyped-def]
        turn = self.s.turn_id
        cmds = list(super()._step(ev))
        plays = [c for c in cmds if isinstance(c, C.PlayClip)]
        for c in plays:
            assert not c.source, f"T-0179 B-6: a clip is played with source=''; got {c}"
        kinds = [c.kind for c in plays if c.kind in ENDING]
        if kinds:
            self.ending_clips.setdefault(turn, []).extend(kinds)
            got = self.ending_clips[turn]
            assert len(got) <= 1, f"T-0179 B-7: at most one ending clip in {turn}; got {got}"
        if LOST_THREAD in kinds:
            assert isinstance(ev, E.LlmError) and ev.message in RETRIED_MESSAGES, (
                f"T-0179 B-6a/B-7: lost_thread only on the retried degenerate end; event {ev}"
            )
            names = [type(c).__name__ for c in cmds]
            assert "HudError" in names and names.index("HudError") < names.index("PlayClip"), (
                f"T-0179 B-6: the clip after hud.error; {names}"
            )
        return cmds

    @invariant()
    def clip_text_never_in_history(self) -> None:
        texts = [v for k, v in session_mod.CLIP_TEXT.items() if k in ENDING] + [LOST_THREAD_TEXT]
        for h in self.s.history:
            for text in texts:
                assert text not in h.assistant_spoken, f"T-0179 B-6: clip text in history: {h}"


class TestLostThreadInvariants(LostThreadModel.TestCase):  # type: ignore[name-defined,misc]
    """B-6/B-7/B-8 over the T-0178 clip rows (see the module docstring). Active: a guard, green at BASE."""

    def runTest(self) -> None:
        try:
            super().runTest()
        except BaseExceptionGroup as group:
            if all(isinstance(e, AssertionError) for e in group.exceptions):
                raise AssertionError(
                    f"{len(group.exceptions)} distinct property failures; first: {group.exceptions[0]}"
                ) from group
            raise


TestLostThreadInvariants.settings = settings(
    max_examples=300, stateful_step_count=80, deadline=None
)
