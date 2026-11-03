"""T-0192 group B, the pure machine: a ``tool.result`` that hands the turn back at a checkout ends the turn with the
``at_checkout`` clip (docs/tasks/T-0192.md B1, B2; docs/02 §4 spec delta).

Every expected tuple is written by hand from B2:

    tool.result(call_id = awaiting_result, hand_back = "checkout") in THINKING, a result awaited, loop on
      -> IDLE with: cancel_timer(tool_result); no llm.continue; play_clip(at_checkout);
         commit_history(user_only = false) over the joined played text; schedule_summary

in that order (T-0178 B2's ending tuple: timers, then the clip, then history, then the summary; ``hud.state`` last,
criterion 21). "T-0178 B2's ending tuple for any still_working clip playing" is read off the frozen BASE machine
(``_frozen_turn_machine_t0192``, da29ddbe): whether BASE's own budget ending leads with ``StopClip`` while a
still_working clip plays is the oracle for whether the hand-back's does (the row computes it, never assumes it).

What the rows fix where B2 is silent (in the test-author's report):

- the hand-back ends the turn whatever the result's ``status`` among the two the session sends (``executed`` for an
  arrival, ``refused`` for a refusal on a checkout page, B4); B2 names no status;
- the history entry is ``HistoryTurn(user_text=<transcript>, assistant_spoken=<joined played runs>, interrupted=False)``
  (``_commit_played``'s shape, B2 "over the joined played text"; implementation note "reuse ``_commit_played``").

Regime: CPU, pure ``step``, virtual time in the events; n = 1 per row, deterministic.
"""

from __future__ import annotations

import ast
import hashlib
import pathlib
import subprocess
import typing
from typing import Any

import pytest

from local_astra.conductor.turn_machine import TurnMachine, TurnPolicy
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.turn import HistoryTurn, MachineState, TurnState
from tests import _git_range as gr
from tests.unit import _frozen_turn_machine_t0192 as frozen
from tests.unit import test_turn_invariants as ti

CHECKOUT = "checkout"
AT_CHECKOUT = "at_checkout"
#: B4: the two statuses the session hands back with (an arrival is ``executed``, a refusal ``refused``)
HAND_BACK_STATUSES = ("executed", "refused")
#: the ToolResult fields at BASE (da29ddbe ``contracts/events.py``, by hand): B1 adds exactly ``hand_back``
BASE_TOOL_RESULT_FIELDS = {"t_ms", "type", "call_id", "status"}
#: ``ClipKind`` at BASE (da29ddbe ``contracts/commands.py`` line 100, by hand)
BASE_CLIP_KINDS = ("ack", "hmm", "still_working", "out_of_steps", "not_finished", "lost_thread")
USER = "open"  # the transcript ``start`` drives (one word)
PREFIX = "Opening it. "
STEP2 = "I will complete the purchase now. "
MIN_W = 5_000  # ``Profile.still_working_after_s`` minimum (T-0182 B4), in ms


def hand_back(t_ms: int, call_id: str, status: str = "executed") -> E.ToolResult:
    """``ToolResult(..., hand_back="checkout")``, or a named failure while B1's field does not exist."""
    assert "hand_back" in E.ToolResult.model_fields, (
        'T-0192 B1: events.ToolResult gains `hand_back: Literal["checkout"] | None = None`; its fields are '
        f"{sorted(E.ToolResult.model_fields)}"
    )
    return E.ToolResult(t_ms=t_ms, call_id=call_id, status=status, hand_back=CHECKOUT)  # type: ignore[call-arg]


def machine(*, budget: int = 8, w: int | None = None, loop: bool = True) -> TurnMachine:
    """A live-loop machine as ``cli`` builds it (ack off, T-0182 B-R4), ``max_tool_steps = budget``."""
    return TurnMachine(
        TurnPolicy(
            thinking=True,
            tool_loop=loop,
            max_tool_steps=budget,
            ack_clip=False,
            still_working_after_ms=w,
        )
    )


def frozen_machine(*, budget: int = 8, w: int | None = None) -> Any:
    return frozen.TurnMachine(
        frozen.TurnPolicy(
            thinking=True,
            tool_loop=True,
            max_tool_steps=budget,
            ack_clip=False,
            still_working_after_ms=w,
        )
    )


def click(call_id: str) -> Any:
    return ti.tool_draft(
        call_id=call_id, status="valid", name="click_at", arguments={"x": 500, "y": 250}
    )


def start(m: Any, *, at: int = 1000, state: MachineState | None = None) -> tuple[MachineState, int]:
    s, cmds, t0 = ti.drive_to_thinking(
        m, onset_ms=at, t_eot_ms=at + 1600, words=[(USER, at + 100)], state=state
    )
    assert s.state is TurnState.THINKING and any(isinstance(c, C.LlmStart) for c in cmds), "premise"
    return s, t0


def step1_executes(m: Any, *, prefix: bool, at: int = 1000) -> tuple[MachineState, int, int]:
    """Step 1 of a loop turn ends with a valid click ``c1`` that R11 executes. With ``prefix`` the step first speaks
    ``PREFIX`` and the execute waits for its drain (T-0089). Returns the awaiting state, ``t0`` (llm.start) and the
    time of the last event."""
    s, t0 = start(m, at=at)
    t = t0 + 200
    if prefix:
        s, _ = m.step(s, ti.commit(s, PREFIX, t=t))
        assert s.state is TurnState.SPEAKING, "premise: the prefix is voiced"
        t += 100
        s, _ = m.step(s, ti.llm_done(t, s.active_generate, click("c1")))
        t += 600
        s, cmds = m.step(s, E.PlaybackDrained(t_ms=t, played_ms=600, played_text=PREFIX))
    else:
        s, cmds = m.step(s, ti.llm_done(t, s.active_generate, click("c1")))
    assert [c.call_id for c in cmds if isinstance(c, C.ToolExecute)] == ["c1"], (
        f"premise: R11 {ti.kinds(cmds)}"
    )
    assert s.state is TurnState.THINKING and s.awaiting_result == "c1", "premise: c1 is awaited"
    return s, t0, t


def to_step(m: Any, s: MachineState, t: int, k: int) -> tuple[MachineState, int]:
    """From awaiting ``c{j}`` (step j), run ordinary results and step texts until ``c{k}`` is awaited (R12, R15)."""
    j = int(str(s.awaiting_result)[1:])
    while j < k:
        t += 50
        s, cmds = m.step(s, E.ToolResult(t_ms=t, call_id=f"c{j}", status="executed"))
        assert any(isinstance(c, C.LlmContinue) for c in cmds), f"premise: R12 {ti.kinds(cmds)}"
        t += 100
        s, _ = m.step(s, ti.commit(s, STEP2, t=t))
        t += 100
        s, cmds = m.step(s, ti.llm_done(t, s.active_generate, click(f"c{j + 1}")))
        assert s.awaiting_result == f"c{j + 1}", f"premise: step {j + 1} executed {ti.kinds(cmds)}"
        j += 1
    return s, t


def expected_tuple(turn_id: str = "t1", *, lead: tuple[Any, ...] = ()) -> tuple[Any, ...]:
    """B2's tuple, by hand."""
    return (
        *lead,
        C.CancelTimer(name="tool_result"),
        C.PlayClip(kind=AT_CHECKOUT),  # type: ignore[arg-type]
        C.CommitHistory(user_only=False),
        C.ScheduleSummary(),
        C.HudState(state=TurnState.IDLE.value, turn_id=turn_id),
    )


def plays(cmds: Any) -> list[str]:
    return [c.kind for c in cmds if isinstance(c, C.PlayClip)]


# =====================================================================================================================
# B1: the contract
# =====================================================================================================================


def test_b1_tool_result_gains_hand_back_only() -> None:
    """B1: ``ToolResult.hand_back`` is ``Literal["checkout"] | None``, default ``None``; no other field is added (the
    event still carries no text)."""
    fields = set(E.ToolResult.model_fields)
    assert fields == BASE_TOOL_RESULT_FIELDS | {"hand_back"}, (
        f"T-0192 B1: ToolResult's fields are BASE's plus hand_back; got {sorted(fields)}"
    )
    assert E.ToolResult(t_ms=1, call_id="c1", status="executed").hand_back is None  # type: ignore[attr-defined]
    assert hand_back(1, "c1").hand_back == CHECKOUT  # type: ignore[attr-defined]
    for bad in ("Checkout", "cart", "", "checkout ", "at_checkout"):
        with pytest.raises(ValueError):
            E.ToolResult(t_ms=1, call_id="c1", status="executed", hand_back=bad)  # type: ignore[call-arg]


def test_b1_clip_kind_gains_at_checkout() -> None:
    """B1: ``ClipKind`` is BASE's six kinds plus ``at_checkout``; ``PlayClip(kind="at_checkout")`` validates."""
    kinds = typing.get_args(C.ClipKind)
    assert (
        set(kinds) == {*BASE_CLIP_KINDS, AT_CHECKOUT} and len(kinds) == len(BASE_CLIP_KINDS) + 1
    ), kinds
    assert C.PlayClip(kind=AT_CHECKOUT).kind == AT_CHECKOUT  # type: ignore[arg-type]


# =====================================================================================================================
# B2: the row
# =====================================================================================================================


@pytest.mark.parametrize("status", HAND_BACK_STATUSES)
def test_b2_at_step_1_with_a_spoken_prefix(status: str) -> None:
    """B2 at step 1: the prefix was played, ``c1`` executed; the hand-back ends the turn with B2's exact tuple; the
    history holds what was played (the clip is never history)."""
    m = machine()
    s, _, t = step1_executes(m, prefix=True)
    new, cmds = m.step(s, hand_back(t + 50, "c1", status))
    assert tuple(cmds) == expected_tuple(), f"T-0192 B2: {ti.kinds(cmds)}"
    assert (
        new.state is TurnState.IDLE and new.awaiting_result is None and new.active_generate is None
    )
    assert new.timers == (), new.timers
    assert new.history[-1] == HistoryTurn(
        user_text=USER, assistant_spoken=PREFIX, interrupted=False
    ), new.history


def test_b2_at_step_1_silent() -> None:
    """B2 at step 1 with nothing played: ``commit_history(user_only=False)`` still, the user half stored with an
    empty assistant half (``_commit_played``)."""
    m = machine()
    s, _, t = step1_executes(m, prefix=False)
    new, cmds = m.step(s, hand_back(t + 50, "c1"))
    assert tuple(cmds) == expected_tuple(), f"T-0192 B2: {ti.kinds(cmds)}"
    assert new.history[-1] == HistoryTurn(user_text=USER, assistant_spoken="", interrupted=False), (
        new.history
    )


@pytest.mark.parametrize("k", [2, 3])
def test_b2_at_step_k_withholds_the_step_text(k: int) -> None:
    """B2 at step k: steps 2..k each committed ``STEP2`` (withheld by R15: each executed a call); the hand-back of
    ``c{k}`` ends the turn with the same tuple; nothing of the steps >= 2 was ever voiced or stored."""
    m = machine()
    s, _, t = step1_executes(m, prefix=True)
    s, t = to_step(m, s, t, k)
    new, cmds = m.step(s, hand_back(t + 50, f"c{k}"))
    assert tuple(cmds) == expected_tuple(), f"T-0192 B2 (step {k}): {ti.kinds(cmds)}"
    assert new.history[-1] == HistoryTurn(
        user_text=USER, assistant_spoken=PREFIX, interrupted=False
    )
    assert all(STEP2 not in h.assistant_spoken for h in new.history)


@pytest.mark.parametrize("budget", [1, 2])
def test_b2_after_the_last_allowed_execute_no_out_of_steps(budget: int) -> None:
    """B2 "same at any step, including after the last allowed execute (no out_of_steps instead)": with
    ``max_tool_steps = budget`` the hand-back of the budget's last call plays ``at_checkout`` only; an ordinary result
    at the same point would start the report step (premise: shows the hand-back is what ends it)."""
    m = machine(budget=budget)
    s, _, t = step1_executes(m, prefix=False)
    s, t = to_step(m, s, t, budget)
    assert s.tool_steps_executed == budget, "premise: the budget is spent"
    _, ordinary = m.step(s, E.ToolResult(t_ms=t + 50, call_id=f"c{budget}", status="executed"))
    assert any(isinstance(c, C.LlmContinue) for c in ordinary), (
        "premise: an ordinary result starts the report step"
    )
    new, cmds = m.step(s, hand_back(t + 50, f"c{budget}"))
    assert tuple(cmds) == expected_tuple(), f"T-0192 B2 (budget {budget}): {ti.kinds(cmds)}"
    assert plays(cmds) == [AT_CHECKOUT] and new.state is TurnState.IDLE


def test_b2_with_still_working_playing_mirrors_the_t0178_ending() -> None:
    """B2 with the still_working clip playing: the tuple leads with ``StopClip`` exactly when BASE's own ending tuple
    does (the frozen da29ddbe machine's T-0178 B2 budget ending after still_working, the oracle); the at_checkout
    clip plays once; later ticks play nothing."""
    # the oracle: BASE's budget ending (out_of_steps) after a playing still_working
    fm = frozen_machine(budget=1, w=MIN_W)
    fs, ft0, _ = step1_executes(fm, prefix=False)
    fs, cmds = fm.step(fs, E.Tick(t_ms=ft0 + MIN_W))
    assert plays(cmds) == ["still_working"], "premise: still_working plays on BASE"
    fs, _ = fm.step(fs, E.ToolResult(t_ms=ft0 + MIN_W + 100, call_id="c1", status="executed"))
    fs, base_end = fm.step(fs, ti.llm_done(ft0 + MIN_W + 200, fs.active_generate, click("c2")))
    assert "out_of_steps" in plays(base_end), f"premise: BASE's budget ending {ti.kinds(base_end)}"
    lead = (C.StopClip(),) if any(isinstance(c, C.StopClip) for c in base_end) else ()

    m = machine(w=MIN_W)
    s, t0, _ = step1_executes(m, prefix=False)
    s, cmds = m.step(s, E.Tick(t_ms=t0 + MIN_W))
    assert plays(cmds) == ["still_working"] and s.clip_playing, "premise: still_working plays"
    new, cmds = m.step(s, hand_back(t0 + MIN_W + 100, "c1"))
    assert tuple(cmds) == expected_tuple(lead=lead), (
        f"T-0192 B2 (still_working playing): {ti.kinds(cmds)}"
    )
    assert new.clip_playing is False and new.still_working_played is False, (
        "per-turn fields cleared at IDLE"
    )
    for t in (t0 + MIN_W + 300, t0 + 3 * MIN_W):
        new, cmds = m.step(new, E.Tick(t_ms=t))
        assert plays(cmds) == [], f"no clip after IDLE: {ti.kinds(cmds)}"


def test_b2_the_turn_is_over_late_events_and_a_second_hand_back_are_inert() -> None:
    """Adversarial (L1/L3, R13 after B2): after the hand-back, the same call's result again (with and without
    ``hand_back``), a late llm.done for step 1 and a tool_result tick change nothing; the next turn opens as t2 with
    t1's history entry kept."""
    m = machine()
    s, _, t = step1_executes(m, prefix=True)
    s, _ = m.step(s, hand_back(t + 50, "c1"))
    for ev in (
        hand_back(t + 60, "c1"),
        E.ToolResult(t_ms=t + 70, call_id="c1", status="executed"),
        E.LlmDone(t_ms=t + 80, request_id="t1-gen"),
        E.Tick(t_ms=t + 15_000 + 100),
    ):
        new, cmds = m.step(s, ev)
        assert (new, tuple(cmds)) == (s, ()), (
            f"T-0192 R13: {ev.type} after the hand-back: {ti.kinds(cmds)}"
        )
    s2, _ = start(m, at=t + 20_000, state=s)
    assert s2.turn_id == "t2" and s2.history == s.history


# =====================================================================================================================
# R13: every other hand_back is inert
# =====================================================================================================================


def _inert(m: Any, s: MachineState, ev: Any, what: str) -> None:
    new, cmds = m.step(s, ev)
    assert (new, tuple(cmds)) == (s, ()), f"T-0192 B2/R13 ({what}): {ti.kinds(cmds)}"


def test_r13_stale_unknown_and_unawaited_hand_backs_are_inert() -> None:
    """B2 "any other tool.result with hand_back is inert": an unknown id; a call not executed yet (step 2's pending
    draft id before its execute); the previous step's (answered) call; and one while step 2 is generating."""
    m = machine()
    s, _, t = step1_executes(m, prefix=False)
    _inert(m, s, hand_back(t + 10, "zz-9"), "unknown id while c1 is awaited")
    _inert(m, s, hand_back(t + 10, "c2"), "a call not executed yet")
    s, _ = m.step(s, E.ToolResult(t_ms=t + 20, call_id="c1", status="executed"))
    assert s.active_generate == "t1-gen-s2", "premise: step 2 generates"
    _inert(m, s, hand_back(t + 30, "c1"), "the answered call, while step 2 generates")
    _inert(m, s, hand_back(t + 30, "c2"), "step 2's call before its execute")


def test_r13_hand_back_outside_thinking_is_inert() -> None:
    """Inert in IDLE, LISTENING, THINKING before any execute and SPEAKING (step 1's prefix), whatever the id."""
    m = machine()
    idle = m.initial()
    _inert(m, idle, hand_back(10, "c1"), "IDLE")
    listening, _ = m.step(idle, E.UserSpeechOnset(t_ms=100, onset_ms=100))
    _inert(m, listening, hand_back(120, "c1"), "LISTENING")
    thinking, t0 = start(m, at=1000)
    _inert(m, thinking, hand_back(t0 + 10, "c1"), "THINKING, no call executed")
    speaking, _ = m.step(thinking, ti.commit(thinking, PREFIX, t=t0 + 100))
    speaking, _ = m.step(speaking, ti.llm_done(t0 + 200, speaking.active_generate, click("c1")))
    assert speaking.state is TurnState.SPEAKING and speaking.awaiting_result is None, "premise"
    _inert(m, speaking, hand_back(t0 + 300, "c1"), "SPEAKING, c1 pending its drain")


def test_r13_loop_off_hand_back_is_inert() -> None:
    """Loop off (L8): a tool turn whose valid call executed at completion is IDLE; a hand-back for that call, and one in
    THINKING, change nothing."""
    m = machine(loop=False)
    s, t0 = start(m)
    _inert(m, s, hand_back(t0 + 10, "c1"), "loop off, THINKING")
    s, cmds = m.step(s, ti.llm_done(t0 + 100, s.active_generate, click("c1")))
    assert [c.call_id for c in cmds if isinstance(c, C.ToolExecute)] == [
        "c1"
    ] and s.state is TurnState.IDLE
    _inert(m, s, hand_back(t0 + 200, "c1"), "loop off, after the execute")


def test_r13_a_cut_turns_call_handed_back_in_the_next_turn_is_inert() -> None:
    """A turn cut by R18 (tool timeout) after executing ``c1``; the next turn is generating; the late hand-back of
    ``c1`` (the cut turn's call) is inert."""
    m = machine()
    s, _, t = step1_executes(m, prefix=False)
    s, cmds = m.step(s, E.Tick(t_ms=t + 15_000))
    assert s.state is TurnState.IDLE and C.HudError(message="tool timeout") in cmds, "premise: R18"
    s, t1 = start(m, at=t + 16_000, state=s)
    assert s.turn_id == "t2" and s.state is TurnState.THINKING
    _inert(m, s, hand_back(t1 + 10, "c1"), "t1's call in t2")


# =====================================================================================================================
# The pin moves (Owned files: "Existing pins that move under B1/B2"): the machine and the two contract files change
# only by B1/B2's named additions
# =====================================================================================================================
#
# The byte pins of ``turn_machine.py``, ``contracts/events.py`` and ``contracts/commands.py`` (test_session_loop's
# ``TURN_MACHINE_SHA256``, test_session_tool_loop, test_session_engine_empty_retry, test_t0181_mic_states) cannot know
# the implementation's bytes before it exists. They move the T-0177 A-P2 way (``_t0177_protocol_drift`` in
# test_session_engine_empty_retry): each file is checked at T-0192 B's literal base against its earlier sha256 pin (the
# base end), and the working tree is held to "that blob plus exactly B1/B2's additions" by an AST comparison
# (``t0192_drift``). B-RV6, after the implementation was final: the census pins now hold the working tree to sha256
# literals of the merged blobs at 97409910 (``T0192_MOVED_SHA256``, tests.md "Pin both ends", as T-0181 A0-7 did at
# 9dbddf96); the drift rows below stay as the check of the B delta's shape.

#: T-0192 group B's literal base: main when group B branched (da29ddbe, "T-0192: group A merged"). Never a moving ref.
T0192_BASE = "da29ddbe644f7ed4af21e83968aa83a8a5aa47c3"
T0192_MACHINE = "local_astra/conductor/turn_machine.py"
T0192_EVENTS = "local_astra/contracts/events.py"
T0192_COMMANDS = "local_astra/contracts/commands.py"
#: sha256 of each file at T0192_BASE (``git show da29ddbe:<path> | sha256sum``, 2026-10-29). Each is the file's
#: earlier pin: the machine is T-0181 A0's 9dbddf96 content (``test_session_loop.TURN_MACHINE_SHA256``), events.py
#: T-0181 A0's (``MicMuted``), commands.py T-0179 B's ebaf4248 content (``lost_thread``).
T0192_BASE_SHA256 = {
    T0192_MACHINE: "634c9260161c14935f70965166ebf9e876a8974543b543df7dfc51735635dcc7",
    T0192_EVENTS: "76aabfefc8faad5b7ee772f8a3ca6e5024a735aaa4a1d45fb42386acb4996934",
    T0192_COMMANDS: "776033c56d60eb06a2cab6560d2a10c7ed5d8943835c23e92907cb5026220057",
}
#: B-RV6 (A5), the tree end: T-0192 group B's final implementation commit (97409910, "handed_back derived from the
#: closing at_checkout clip"), a literal sha, never a moving ref.
T0192_TIP = "97409910"
#: sha256 of each file's merged blob at T0192_TIP (``git show 97409910:<path> | sha256sum``, 2026-10-29). The census
#: pins (test_session_loop's ``TURN_MACHINE_SHA256``, test_session_tool_loop, test_session_engine_empty_retry,
#: test_t0181_mic_states) hold the working tree to these bytes, and the base ends to ``T0192_BASE_SHA256`` (both ends
#: pinned, as T-0181 A0-7 did at 9dbddf96). ``t0192_drift`` stays as the check of the B delta's shape.
T0192_MOVED_SHA256 = {
    T0192_MACHINE: "305060235e23ff46223688ddbf5e331c1c108e0390c984ec9d8f47c0dcd58b40",
    T0192_EVENTS: "02016d69192d601fa37b18c4f60559b85ab910a77caf21c061eaee672b1e2ddf",
    T0192_COMMANDS: "658cd09344999ac97d726719a87077e6930757a967bc2763feb27aba917e679a",
}
_REPO = pathlib.Path(__file__).resolve().parents[2]
#: The machine's base end needs no git: ``_frozen_turn_machine_t0192`` holds da29ddbe's machine from line 18 on
#: (its body sha256 is test_turn_invariants' frozen-copy pin), and lines 1-17 (the module docstring the copy drops)
#: hash to this literal (``git show da29ddbe:local_astra/conductor/turn_machine.py | head -17 | sha256sum``).
T0192_FROZEN_BODY_SHA256 = "84b20dc3b790d22c9fc53df3a05692b0f630596e892646418b5a7aa3e3831385"
T0192_MACHINE_HEAD_SHA256 = "3627faf45fe36a8cc943b4b8729a64230d28fac3b52ecd5ad6933e6956dd9e82"
T0192_MACHINE_HEAD_LINES = 17


def _frozen_body() -> str:
    text = pathlib.Path(frozen.__file__).read_text(encoding="utf-8")
    start, end = "dropped) ---\n", "# --- end of verbatim copy ---\n"
    body = text[text.index(start) + len(start) : text.index(end)].rstrip("\n") + "\n"
    assert hashlib.sha256(body.encode()).hexdigest() == T0192_FROZEN_BODY_SHA256, (
        "premise: _frozen_turn_machine_t0192 is da29ddbe's machine verbatim"
    )
    return body


def t0192_end_sha256(path: str, ref: str | None = None) -> str:
    """sha256 of ``path`` in the working tree (``ref`` None) or of its blob at the literal commit ``ref`` (read with
    git; skips, never passes, when the commit is absent). The census pins compare it with the literals above."""
    if ref is None:
        return hashlib.sha256((_REPO / path).read_bytes()).hexdigest()
    gr.require_commits(ref)
    blob = subprocess.run(
        ["git", "-C", str(_REPO), "show", f"{ref}:{path}"], capture_output=True, check=True
    ).stdout
    return hashlib.sha256(blob).hexdigest()


def _split_machine(src: str) -> tuple[str, str]:
    """(lines 1-17, the rest) of a ``turn_machine.py`` text."""
    lines = src.splitlines(keepends=True)
    return "".join(lines[:T0192_MACHINE_HEAD_LINES]), "".join(lines[T0192_MACHINE_HEAD_LINES:])


def t0192_base_blob(path: str) -> str:
    """``path`` at T0192_BASE, checked against its earlier pin; for the machine, its body from line 18 on, read from
    the frozen copy (no git). The contract files are read with git (skips, never passes, when the commit is absent)."""
    if path == T0192_MACHINE:
        return _frozen_body()
    gr.require_commits(T0192_BASE)
    blob = subprocess.run(
        ["git", "-C", str(_REPO), "show", f"{T0192_BASE}:{path}"], capture_output=True, check=True
    ).stdout
    assert hashlib.sha256(blob).hexdigest() == T0192_BASE_SHA256[path], (
        f"T-0192 B pin move: {path} at T-0192 B's base {T0192_BASE[:8]} is not its earlier pinned content"
    )
    return blob.decode("utf-8")


def _named(node: ast.stmt) -> str | None:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        return target.id if isinstance(target, ast.Name) else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _docstring_free(body: list[ast.stmt]) -> list[str]:
    first = body[0] if body else None
    first_value = first.value if isinstance(first, ast.Expr) else None
    if isinstance(first_value, ast.Constant) and isinstance(first_value.value, str):
        body = body[1:]
    return [ast.dump(n) for n in body]


def _shell(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """The statement without its body (name, bases, decorators, arguments, return annotation)."""
    import copy

    bare = copy.copy(node)
    bare.body = []
    return ast.dump(bare)


def _body_drift(
    base: list[ast.stmt],
    now: list[ast.stmt],
    where: str,
    *,
    moved: dict[str, Any],
    may_add: tuple[type, ...],
) -> list[str]:
    """Every way ``now`` differs from ``base`` beyond the names in ``moved`` (each checked by its own rule) and new
    named statements of the types in ``may_add``: BASE's named statements all present, in BASE's order, each with an
    identical AST; an import may only gain names; every unnamed statement (docstring, expression) identical."""
    drift: list[str] = []
    base_order = [k for k in map(_named, base) if k is not None]
    now_order = [k for k in map(_named, now) if k is not None]
    assert len(set(base_order)) == len(base_order), f"premise: {where} binds each BASE name once"
    if len(set(now_order)) != len(now_order):
        drift.append(f"{where}: a name is bound twice")
    base_named = {_named(n): n for n in base if _named(n) is not None}
    now_named = {_named(n): n for n in now if _named(n) is not None}
    missing = [k for k in base_order if k not in now_named]
    if missing:
        drift.append(f"{where}: BASE's {missing} removed")
    if [k for k in now_order if k in base_named] != [k for k in base_order if k in now_named]:
        drift.append(f"{where}: BASE's named statements reordered")
    for name, node in base_named.items():
        if name not in now_named:
            continue
        if name in moved:
            drift.extend(moved[name](node, now_named[name]))
        elif ast.dump(now_named[name]) != ast.dump(node):
            drift.append(f"{where}.{name}: changed")
    for name in now_order:
        if name not in base_named and not isinstance(now_named[name], may_add):
            drift.append(
                f"{where}.{name}: a new {type(now_named[name]).__name__} is not one of B1/B2's additions"
            )
    base_imports = [n for n in base if isinstance(n, ast.Import | ast.ImportFrom)]
    now_imports = [n for n in now if isinstance(n, ast.Import | ast.ImportFrom)]
    if len(now_imports) != len(base_imports):
        drift.append(f"{where}: the import statements changed in number")
    for b, n in zip(base_imports, now_imports, strict=False):
        if getattr(b, "module", None) != getattr(n, "module", None) or not {
            a.name for a in b.names
        } <= {a.name for a in n.names}:
            drift.append(f"{where}: an import lost names or moved: {ast.dump(b)}")
    rest = (ast.Import, ast.ImportFrom)
    base_rest = [ast.dump(n) for n in base if _named(n) is None and not isinstance(n, rest)]
    now_rest = [ast.dump(n) for n in now if _named(n) is None and not isinstance(n, rest)]
    if base_rest != now_rest:
        drift.append(f"{where}: an unnamed statement (docstring, expression, ...) changed")
    return drift


def _on_tool_result_gains_the_row(base: ast.stmt, now: ast.stmt) -> list[str]:
    """B2: ``_on_tool_result`` keeps its signature and gains the hand-back row (its code is not BASE's; its behaviour
    is the B2 rows' to pin). Its docstring may change."""
    assert isinstance(base, ast.FunctionDef)
    if not isinstance(now, ast.FunctionDef) or _shell(now) != _shell(base):
        return ["TurnMachine._on_tool_result: its signature changed"]
    if _docstring_free(now.body) == _docstring_free(base.body):
        return ["TurnMachine._on_tool_result: B2's hand-back row is missing (its code is BASE's)"]
    return []


def _turn_machine_class(base: ast.stmt, now: ast.stmt) -> list[str]:
    assert isinstance(base, ast.ClassDef)
    if not isinstance(now, ast.ClassDef) or _shell(now) != _shell(base):
        return ["TurnMachine: its class statement (bases, decorators) changed"]
    return _body_drift(
        base.body,
        now.body,
        "TurnMachine",
        moved={"_on_tool_result": _on_tool_result_gains_the_row},
        may_add=(ast.FunctionDef,),
    )


def _tool_result_gains_hand_back(base: ast.stmt, now: ast.stmt) -> list[str]:
    """B1: ``ToolResult`` is BASE's class plus exactly one annotated ``hand_back`` after BASE's fields (its type and
    default are pinned at run time by ``test_turn_tool_loop_rows``). Its docstring may change."""
    assert isinstance(base, ast.ClassDef)
    if not isinstance(now, ast.ClassDef) or _shell(now) != _shell(base):
        return ["ToolResult: its class statement changed"]
    was, got = _docstring_free(base.body), _docstring_free(now.body)
    added = now.body[len(now.body) - (len(got) - len(was)) :] if len(got) > len(was) else []
    if (
        got[: len(was)] != was
        or len(added) != 1
        or not isinstance(added[0], ast.AnnAssign)
        or not isinstance(added[0].target, ast.Name)
        or added[0].target.id != "hand_back"
    ):
        return [
            f"ToolResult: not BASE's fields plus exactly hand_back ({len(was)} -> {len(got)} statements)"
        ]
    return []


def _clip_kind_gains_at_checkout(base: ast.stmt, now: ast.stmt) -> list[str]:
    """B1: ``ClipKind`` is BASE's ``Literal[...]`` with exactly ``"at_checkout"`` appended."""
    assert isinstance(base, ast.Assign) and isinstance(base.value, ast.Subscript)
    if not isinstance(now, ast.Assign) or not isinstance(now.value, ast.Subscript):
        return ["ClipKind: no longer a Literal[...] assignment"]
    if ast.dump(now.value.value) != ast.dump(base.value.value):
        return ["ClipKind: no longer a Literal"]
    try:
        kinds = ast.literal_eval(now.value.slice)
    except ValueError:
        return ["ClipKind: its members are not literals"]
    want = (*ast.literal_eval(base.value.slice), AT_CHECKOUT)
    if kinds != want:
        return [f"ClipKind: {kinds!r} is not BASE's plus exactly at_checkout ({want!r})"]
    return []


#: per file: (the top-level names that move, with their rule; the new top-level statement types allowed)
_T0192_RULES: dict[str, tuple[dict[str, Any], tuple[type, ...]]] = {
    # B2: the machine gains the hand-back row in ``_on_tool_result``; a new private helper function or constant is
    # allowed, no new class, no other edit
    T0192_MACHINE: (
        {"TurnMachine": _turn_machine_class},
        (ast.FunctionDef, ast.Assign, ast.AnnAssign),
    ),
    # B1: ``ToolResult`` gains ``hand_back``; a new alias (e.g. a ``Literal`` for the field) is allowed, no new class
    T0192_EVENTS: ({"ToolResult": _tool_result_gains_hand_back}, (ast.Assign, ast.AnnAssign)),
    # B1: ``ClipKind`` gains ``at_checkout``; nothing new
    T0192_COMMANDS: ({"ClipKind": _clip_kind_gains_at_checkout}, ()),
}


def t0192_drift(path: str, now_src: str | None = None, base_src: str | None = None) -> list[str]:
    """Every way ``path`` (the working tree's, or ``now_src``) differs from its T0192_BASE blob (or ``base_src``)
    beyond B1/B2's named additions; empty when only those changed. At BASE itself it names the missing addition."""
    if base_src is None:
        base_src = t0192_base_blob(path)
    if now_src is None:
        now_src = (_REPO / path).read_text(encoding="utf-8")
        if path == T0192_MACHINE:
            # the machine's base is the frozen body (line 18 on); its module docstring is held to the literal
            head, now_src = _split_machine(now_src)
            if hashlib.sha256(head.encode()).hexdigest() != T0192_MACHINE_HEAD_SHA256:
                return [
                    f"{path}: lines 1-{T0192_MACHINE_HEAD_LINES} (the module docstring) are not BASE's"
                ]
    moved, may_add = _T0192_RULES[path]
    return _body_drift(
        ast.parse(base_src).body, ast.parse(now_src).body, path, moved=moved, may_add=may_add
    )


# --- the checker's power, on edits of the BASE blobs (read with git; skipped when T0192_BASE is absent) ------------


#: the hand-back row as a minimal edit of BASE's ``_on_tool_result`` (inserted after its R13 guard): what an
#: implementation of B2 could look like, for the checker's positive row only (B2's behaviour is the rows above)
_ROW = (
    "            return state, []  # R13\n"
    '        if ev.hand_back == "checkout":\n'
    '            return self._end_turn(state), [C.PlayClip(kind="at_checkout")]\n'
)


def _edit(src: str, old: str, new: str) -> str:
    assert src.count(old) == 1, f"premise: {old!r} occurs once in the BASE blob"
    return src.replace(old, new)


def test_pin_move_base_ends_are_the_base_blobs() -> None:
    """Provenance of the base ends: at da29ddbe each file hashes to its earlier pin (``T0192_BASE_SHA256``), and the
    machine's blob is the head literal's 17 lines followed by the frozen body (so the git-free machine check reads
    exactly da29ddbe's machine)."""
    gr.require_commits(T0192_BASE)
    for path in (T0192_EVENTS, T0192_COMMANDS):
        t0192_base_blob(path)  # asserts the sha256
    blob = subprocess.run(
        ["git", "-C", str(_REPO), "show", f"{T0192_BASE}:{T0192_MACHINE}"],
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    assert hashlib.sha256(blob.encode()).hexdigest() == T0192_BASE_SHA256[T0192_MACHINE]
    head, body = _split_machine(blob)
    assert hashlib.sha256(head.encode()).hexdigest() == T0192_MACHINE_HEAD_SHA256
    assert body.rstrip("\n") + "\n" == _frozen_body()


def test_pin_move_tree_ends_are_the_merged_blobs() -> None:
    """B-RV6 provenance of the tree ends: at 97409910 each file hashes to its ``T0192_MOVED_SHA256`` literal (a fixed
    blob against a fixed literal), each differs from its base end, and that merged blob is exactly the base blob plus
    B1/B2's additions (``t0192_drift`` of the two fixed blobs is empty)."""
    gr.require_commits(T0192_TIP, T0192_BASE)
    for path, sha in T0192_MOVED_SHA256.items():
        blob = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{T0192_TIP}:{path}"],
            capture_output=True,
            check=True,
        ).stdout
        assert hashlib.sha256(blob).hexdigest() == sha, (
            f"B-RV6: {path} at {T0192_TIP} is not its tree-end literal"
        )
        assert sha != T0192_BASE_SHA256[path], f"B-RV6: {path}'s tree end equals its base end"
        base = subprocess.run(
            ["git", "-C", str(_REPO), "show", f"{T0192_BASE}:{path}"],
            capture_output=True,
            check=True,
        ).stdout.decode("utf-8")
        drift = t0192_drift(path, blob.decode("utf-8"), base)
        assert drift == [], (
            f"B-RV6: {path} at {T0192_TIP} is not the base blob plus B1/B2's additions: {drift}"
        )


def test_pin_move_checker_names_the_missing_additions_at_base() -> None:
    """At BASE each file's drift names exactly the B1/B2 addition it lacks (this is why the moved pins are red until
    group B is implemented)."""
    for path, needle in (
        (T0192_MACHINE, "_on_tool_result: B2's hand-back row is missing"),
        (T0192_EVENTS, "ToolResult: not BASE's fields plus exactly hand_back"),
        (
            T0192_COMMANDS,
            "ClipKind: ('ack', 'hmm', 'still_working', 'out_of_steps', 'not_finished', 'lost_thread')",
        ),
    ):
        base = t0192_base_blob(path)
        drift = t0192_drift(path, base, base)
        assert len(drift) == 1 and needle in drift[0], (path, drift)


def test_pin_move_checker_accepts_exactly_the_named_additions() -> None:
    """Positive: BASE plus only the named addition (and a new private helper / alias where allowed) has no drift."""
    machine = t0192_base_blob(T0192_MACHINE)
    with_row = _edit(machine, "            return state, []  # R13\n", _ROW)
    assert t0192_drift(T0192_MACHINE, with_row, machine) == []
    helper = with_row + "\n\ndef _hand_back_clip() -> str:\n    return 'at_checkout'\n"
    assert t0192_drift(T0192_MACHINE, helper, machine) == []
    events = t0192_base_blob(T0192_EVENTS)
    field = "    status: ToolResultStatus\n"
    with_field = _edit(events, field, field + '    hand_back: Literal["checkout"] | None = None\n')
    assert t0192_drift(T0192_EVENTS, with_field, events) == []
    commands = t0192_base_blob(T0192_COMMANDS)
    kind = '"not_finished", "lost_thread"]'
    with_kind = _edit(commands, kind, '"not_finished", "lost_thread", "at_checkout"]')
    assert t0192_drift(T0192_COMMANDS, with_kind, commands) == []


def test_pin_move_checker_refuses_every_other_edit() -> None:
    """Negative: with the named addition in place, each further edit is named (an edit to another method, a new
    import, a new class, a removed name, a second or misnamed new field, an extra, misplaced or misspelt clip kind, a
    changed ``_on_tool_result`` signature)."""
    machine = t0192_base_blob(T0192_MACHINE)
    good = _edit(machine, "            return state, []  # R13\n", _ROW)
    for mutant, needle in (
        (
            _edit(good, 'new = new.model_copy(update={"step_text": ""})\n', "new = new\n"),
            "_finish_step: changed",
        ),
        (good + "\nimport time\n", "import statements changed in number"),
        (good + "\n\nclass Extra:\n    pass\n", "Extra: a new ClassDef"),
        (
            _edit(
                good,
                "self, state: MachineState, ev: E.ToolResult\n",
                "self, state: MachineState, ev: E.Event\n",
            ),
            "signature",
        ),
        (_edit(good, "_POLICY_REASONS = (", "_OTHER_REASONS = ("), "['_POLICY_REASONS'] removed"),
    ):
        drift = t0192_drift(T0192_MACHINE, mutant, machine)
        assert any(needle in d for d in drift), (needle, drift)
    events = t0192_base_blob(T0192_EVENTS)
    field = "    status: ToolResultStatus\n"
    for extra, needle in (
        (
            '    hand_back: Literal["checkout"] | None = None\n    note: str = ""\n',
            "exactly hand_back",
        ),
        ('    handed_back: Literal["checkout"] | None = None\n', "exactly hand_back"),
    ):
        drift = t0192_drift(T0192_EVENTS, _edit(events, field, field + extra), events)
        assert any(needle in d for d in drift), (needle, drift)
    good_events = _edit(events, field, field + '    hand_back: Literal["checkout"] | None = None\n')
    extra_class = good_events + "\n\nclass Checkout(_Event):\n    type: Literal['x'] = 'x'\n"
    assert any(
        "Checkout: a new ClassDef" in d for d in t0192_drift(T0192_EVENTS, extra_class, events)
    )
    commands = t0192_base_blob(T0192_COMMANDS)
    kind = '"not_finished", "lost_thread"]'
    for members in (
        '"not_finished", "lost_thread", "at_checkout", "at_payment"]',
        '"not_finished", "at_checkout", "lost_thread"]',
        '"not_finished", "lost_thread", "checkout"]',
    ):
        drift = t0192_drift(T0192_COMMANDS, _edit(commands, kind, members), commands)
        assert any("ClipKind" in d for d in drift), (members, drift)
