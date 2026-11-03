"""T-0162 group A, criteria A-3, A-4 and A-6: the session executes ``ResetSession``.

Expectations come from ``docs/tasks/T-0162.md`` (Goal, A-3, A-4, A-6, ruling H-T162-1) and are
observed only through what the session already exposes: the machine's command log
(``RecordingMachine``), the HUD publisher (``Recorder``), the session-log file, ``turns.jsonl``, the
fake LLM's prompt log, the executor double and the memory double. Two private names are read
because the task names them verbatim: ``Session._executions`` (A-3 "running tasks and deadlines are
cancelled") and the module tables ``RESET_CLEARS`` / ``RESET_KEEPS`` (A-4).

The reset is delivered the way ``hud_server.route`` delivers it (A-5): one ``E.SessionReset`` on the
session's one pump, ``t_ms`` the session clock's ``now_ms``.

Surface fixed here where the task is silent (listed in the test-author's report):

- the published echo is ``astra.session.cleared`` with ``resets`` / ``ended_turn_id`` /
  ``memory_forgotten`` / ``replay`` read as attributes;
- the session-log record kind is ``session.reset`` with exactly the four data keys A-3 lists;
- the turn line of a turn cut by a reset carries ``"ended_by": "reset"`` in its ``extras`` (ruling 2026-10-27 (b); read as raw JSON, since
  ``TurnRecord`` may or may not grow the field);
- a failed deletion is one session-log ``error`` record whose JSON mentions the raised message.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, tools off unless a row says
dry-run; n = 1 and deterministic per row. No measurement.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import inspect
import json
import textwrap
from typing import Any

import pytest

from local_astra.conductor import session as session_module
from local_astra.conductor.turn_machine import TurnPolicy
from local_astra.contracts import events as E
from local_astra.contracts.memory import Episode
from local_astra.contracts.perception import AssistFlags, Source
from local_astra.contracts.protocol import AstraAssistToggle
from local_astra.contracts.turn import TurnState
from tests.fakes import FakeMemory
from tests.unit import _tool_turn_support as S
from tests.unit._tool_turn_support import (
    CALL_ID,
    VALID_CLICK,
    WatchingExecutor,
    frame,
    tool,
    tool_rig,
)
from tests.unit.test_session_loop import answer

CLEARED = "astra.session.cleared"
#: A-6: the message types that belong to a turn or a request; none of a pre-reset one after the echo
STALE_PREFIXES = (
    "astra.canvas.",
    "astra.spoken",
    "astra.transcript",
    "astra.ui.",
    "astra.tool.call",
)
RESET_RECORD_KEYS = frozenset({"resets", "ended_turn_id", "state_before", "memory_forgotten"})
FIRST_Q = "what is on this screen"
FIRST_A = "It is a stack trace. "


# --- doubles ---------------------------------------------------------------------------------------


class CountingMemory(FakeMemory):
    """``FakeMemory`` (by inheritance, so every other method is the fake's) that records the calls
    the reset is judged by, and can be told to fail the deletion."""

    def __init__(self, *, raise_: BaseException | None = None) -> None:
        super().__init__()
        self.raise_ = raise_
        self.forgets: list[int] = []
        self.returned: list[int] = []
        self.ingests: list[Episode] = []

    async def ingest(self, episode: Episode) -> int:
        self.ingests.append(episode)
        return await super().ingest(episode)

    async def forget_since(self, t0_ms: int) -> int:
        self.forgets.append(t0_ms)
        if self.raise_ is not None:
            raise self.raise_
        n = await super().forget_since(t0_ms)
        self.returned.append(n)
        return n


class HangingExecutor(WatchingExecutor):
    """An executor whose call never finishes on its own: only a cancellation ends it."""

    def __init__(self) -> None:
        super().__init__()
        self.cancelled: list[str] = []

    async def execute(self, inv: Any) -> Any:
        self.invocations.append(inv)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.append(inv.call_id)
            raise
        raise AssertionError("unreachable")  # pragma: no cover


def episode(ts_ms: int, tag: str) -> Episode:
    return Episode(
        ts_ms=ts_ms, source=Source.SCREEN, thumbnail_path=f"/tmp/{tag}.jpg", phash=tag * 16
    )


def use_memory(monkeypatch: pytest.MonkeyPatch, memory: FakeMemory) -> None:
    """``tool_rig`` builds ``FakeMemory()``; hand it this instance instead."""
    monkeypatch.setattr(S, "FakeMemory", lambda: memory)


# --- driving ---------------------------------------------------------------------------------------


def reset_event(t_ms: int) -> Any:
    cls = getattr(E, "SessionReset", None)
    assert cls is not None, (
        "T-0162 A-2: contracts/events.py gains SessionReset (type session.reset)"
    )
    return cls(t_ms=t_ms)


async def do_reset(rig: Any) -> int:
    """What ``hud_server.route`` does with ``astra.session.reset`` (A-5), then let it run."""
    at = rig.clock.now_ms
    rig.session.submit(reset_event(at))
    await rig.pump()
    await rig.advance(200)
    return at


def cleared(rig: Any) -> list[Any]:
    return [m for m in rig.published.messages if getattr(m, "type", None) == CLEARED]


def one_cleared(rig: Any) -> Any:
    got = cleared(rig)
    assert len(got) == 1, (
        f"A-3: one astra.session.cleared per reset; published {len(got)} "
        f"(types {[getattr(m, 'type', '?') for m in rig.published.messages][-12:]})"
    )
    return got[0]


async def answered_turn(rig: Any, question: str = FIRST_Q) -> None:
    await rig.ask(question)
    await rig.turn_done(f"the turn asking {question!r} never ended")


def image_parts(prompt: Any) -> list[Any]:
    return [p for m in prompt.messages for p in m.content if getattr(p, "type", "") == "image_url"]


def carries_frame(prompt: Any, f: Any) -> bool:
    b64 = base64.b64encode(f.jpeg).decode()
    return any(p.phash == f.phash or b64 in p.url for p in image_parts(prompt))


def prompt_text(prompt: Any) -> str:
    return "\n".join(m.text() for m in prompt.messages)


def turn_lines(rig: Any) -> dict[str, dict[str, Any]]:
    return {d["turn_id"]: d for d in (json.loads(line) for line in rig.lines())}


# --- A-3: a reset mid tool loop ----------------------------------------------------------------------


async def test_a_reset_mid_tool_loop_cancels_the_execution_and_publishes_one_cleared(tmp_path):
    executor = HangingExecutor()
    async with tool_rig(
        tmp_path,
        script=[tool(VALID_CLICK)],
        executor=executor,
        policy=TurnPolicy(thinking=True, tool_loop=True),
    ) as rig:
        await rig.ask("click the sign in button")
        await rig.until(lambda: bool(executor.invocations), "the click never reached the executor")
        assert rig.state.state is not TurnState.IDLE, "premise: the turn waits on the call"
        await do_reset(rig)
        assert executor.cancelled == [CALL_ID], (
            f"A-3: the running execution is cancelled by the reset; cancelled {executor.cancelled}"
        )
        assert not rig.session._executions, "A-3: _executions is cleared (tasks and deadlines)"
        msg = one_cleared(rig)
        assert (msg.resets, msg.ended_turn_id, msg.replay) == (1, "t1", False), msg
        assert rig.state.state is TurnState.IDLE and rig.state.history == ()
        at = rig.published.messages.index(msg)
        late = [m.status for m in rig.tool_calls(CALL_ID) if rig.published.messages.index(m) > at]
        assert not late, f"A-6: astra.tool.call for {CALL_ID} after the echo: {late}"


# --- A-3: the next prompt starts from nothing --------------------------------------------------------


async def test_the_next_prompt_has_no_history_and_no_old_frame(tmp_path):
    old = frame(0, "a")
    async with tool_rig(tmp_path, kit=None, script=[answer(FIRST_A)], frames=[old]) as rig:
        await answered_turn(rig)
        assert rig.state.history, "premise: turn 1 is in history"
        assert carries_frame(rig.llm.gen_prompts[-1][1], old), "premise: turn 1 carried frame a"
        reset_at = await do_reset(rig)
        new = frame(reset_at + 1, "b")
        rig.perception.add(new)
        await answered_turn(rig, "and what about now")
        prompt = rig.llm.gen_prompts[-1][1]
        text = prompt_text(prompt)
        assert not [m for m in prompt.messages if m.role == "assistant"], (
            "A-3: the first prompt after a reset has no assistant history"
        )
        assert FIRST_Q not in text and FIRST_A.strip() not in text, (
            f"A-3: nothing of the pre-reset turn reaches the next prompt: {text!r}"
        )
        assert not carries_frame(prompt, old), "A-3: the pre-reset frame is not shown again"
        assert carries_frame(prompt, new), (
            "A-3 guard width: a frame newer than the reset is the turn's frame -- a floor that "
            "drops it too is a reset that blinds the next session"
        )


async def test_a_frame_stamped_at_the_reset_time_is_ignored(tmp_path):
    """A-3: "ignores frames whose ``ts_ms`` is at or below the session-clock time of the reset"."""
    async with tool_rig(
        tmp_path, kit=None, script=[answer(FIRST_A)], frames=[frame(0, "a")]
    ) as rig:
        await answered_turn(rig)
        reset_at = await do_reset(rig)
        at_floor = frame(reset_at, "c")
        rig.perception.add(at_floor)
        await answered_turn(rig, "and what about now")
        prompt = rig.llm.gen_prompts[-1][1]
        assert not image_parts(prompt), (
            f"A-3: frames at ts <= {reset_at} are ignored; the prompt carried "
            f"{[p.phash for p in image_parts(prompt)]}"
        )


# --- A-3 / H-T162-1: the memory wipe -----------------------------------------------------------------


async def test_memory_wiped(tmp_path, monkeypatch):
    memory = CountingMemory()
    for i, tag in enumerate("xyz"):
        await memory.ingest(episode(100 + i, tag))
    use_memory(monkeypatch, memory)
    async with tool_rig(tmp_path, kit=None, script=[answer(FIRST_A)]) as rig:
        await answered_turn(rig)
        rig.llm.hold()
        await rig.ask("now a second question")
        assert rig.state.state is TurnState.THINKING, "premise: turn 2 is cut mid-generation"
        held = len(memory.episodes)
        assert held >= 3, "premise: the seeded episodes are in the store"
        ingests_at_reset = len(memory.ingests)
        await do_reset(rig)
        rig.llm.release()
        await rig.advance(2000)
        assert memory.forgets == [0], f"H-T162-1: exactly one forget_since(0); got {memory.forgets}"
        assert memory.episodes == {}, (
            f"H-T162-1: the long-term memory is wiped, every row removed; {len(memory.episodes)} left"
        )
        assert len(memory.ingests) == ingests_at_reset, (
            "A-3: no memory.ingest for the cut turn (commit_history is not emitted on a reset); "
            f"{len(memory.ingests) - ingests_at_reset} ingest(s) after the reset"
        )
        msg = one_cleared(rig)
        assert memory.returned == [held] and msg.memory_forgotten == held, (
            f"A-1/A-3: memory_forgotten is the count forget_since returned ({memory.returned}); "
            f"published {msg.memory_forgotten}"
        )


async def test_memory_failure_logged_reset_completes(tmp_path, monkeypatch):
    memory = CountingMemory(raise_=RuntimeError("the memory disk went away"))
    use_memory(monkeypatch, memory)
    async with tool_rig(tmp_path, kit=None, script=[answer(FIRST_A)]) as rig:
        await answered_turn(rig)
        await do_reset(rig)
        assert memory.forgets == [0], memory.forgets
        msg = one_cleared(rig)
        assert msg.memory_forgotten == 0, "A-3: a failed deletion publishes memory_forgotten = 0"
        assert rig.state.state is TurnState.IDLE and rig.state.history == ()
        assert not rig.session.failures, f"the pump survived: {rig.session.failures}"
    errors = [r for r in rig.kind("error") if "the memory disk went away" in json.dumps(r)]
    assert len(errors) == 1, f"A-3: one `error` record for the failed deletion; got {errors}"
    resets = rig.kind("session.reset")
    assert [r.get("memory_forgotten") for r in resets] == [0], resets


# --- A-3: the log record and the turn line -----------------------------------------------------------


async def test_the_log_record_and_the_turn_line_ended_by_reset(tmp_path, monkeypatch):
    memory = CountingMemory()
    await memory.ingest(episode(100, "x"))
    use_memory(monkeypatch, memory)
    async with tool_rig(tmp_path, kit=None, script=[answer(FIRST_A)]) as rig:
        rig.llm.hold()
        await rig.ask()
        assert rig.state.state is TurnState.THINKING, "premise"
        await do_reset(rig)
        rig.llm.release()
        await rig.advance(500)
        await do_reset(rig)  # a second reset, from IDLE
        lines = turn_lines(rig)
        assert "t1" in lines, f"A-3: the cut turn still writes its line; lines {sorted(lines)}"
        assert lines["t1"].get("extras", {}).get("ended_by") == "reset", (
            f"A-3: a turn ended by a reset has ended_by: 'reset'; got {lines['t1'].get('ended_by')!r}"
        )
        assert [(m.resets, m.ended_turn_id) for m in cleared(rig)] == [(1, "t1"), (2, None)], (
            "A-1: resets counts the session's resets; ended_turn_id is the turn cut, None from IDLE"
        )
    records = rig.kind("session.reset")
    assert [set(r) for r in records] == [RESET_RECORD_KEYS, RESET_RECORD_KEYS], records
    assert [
        (r["resets"], r["ended_turn_id"], r["state_before"], r["memory_forgotten"]) for r in records
    ] == [(1, "t1", "THINKING", memory.returned[0]), (2, None, "IDLE", 0)], records


# --- A-3: what survives --------------------------------------------------------------------------------


async def test_assist_flags_and_the_log_toggle_survive(tmp_path):
    async with tool_rig(tmp_path, kit=None, script=[answer(FIRST_A)]) as rig:
        rig.session.apply_assist_toggle(AstraAssistToggle(flags=AssistFlags(page_text=True)))
        assists = rig.session.assists.state_message()
        log_state = rig.session.log_state_message()
        log = rig.session.session_log
        assert log is not None, "premise: the rig logs"
        await answered_turn(rig)
        await do_reset(rig)
        one_cleared(rig)
        assert rig.session.assists.state_message() == assists, "A-3: assist flags survive"
        assert rig.session.assists.state_message().flags.page_text is True
        # ruling 2026-10-27 (d): the toggle is `on` and `path`; the message's other fields (e.g. a
        # byte count) legitimately move as the session keeps logging
        after = rig.session.log_state_message()
        assert (after.on, after.path) == (log_state.on, log_state.path), (
            f"A-3: the log toggle survives: {(log_state.on, log_state.path)} -> {(after.on, after.path)}"
        )
        assert after.on is True, "premise: the rig's session is logging"
        assert rig.session.session_log is log, "A-3: the session log keeps writing the same file"


# --- A-4: every attribute is classified ---------------------------------------------------------------


def init_attributes() -> set[str]:
    """Every ``self.<name>`` that ``Session.__init__`` assigns, read from its source."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(session_module.Session.__init__)))
    names: set[str] = set()

    def collect(target: ast.AST) -> None:
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            if target.value.id == "self":
                names.add(target.attr)
        elif isinstance(target, ast.Tuple | ast.List):
            for elt in target.elts:
                collect(elt)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                collect(t)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            collect(node.target)
    return names


def test_every_session_attribute_is_classified():
    clears = getattr(session_module, "RESET_CLEARS", None)
    keeps = getattr(session_module, "RESET_KEEPS", None)
    assert clears is not None and keeps is not None, (
        "A-4: session.py declares module-level RESET_CLEARS and RESET_KEEPS"
    )
    clears, keeps = set(clears), set(keeps)
    assert all(isinstance(n, str) for n in clears | keeps), "the tables hold attribute names"
    attrs = init_attributes()
    assert len(attrs) > 60, f"premise: the AST walk found the attributes ({len(attrs)})"
    assert not clears & keeps, f"A-4: an attribute is in both tables: {sorted(clears & keeps)}"
    assert not attrs - clears - keeps, (
        f"A-4: Session.__init__ sets attributes in neither table: {sorted(attrs - clears - keeps)}"
    )
    assert not (clears | keeps) - attrs, (
        f"A-4: the tables name attributes __init__ never sets: {sorted((clears | keeps) - attrs)}"
    )
    must_clear = {
        "_frame",
        "_frames",
        "_frame_tags",
        "_calls",
        "_executions",
        "_gate",
        "_run_ledger",
        "last_truncation",
        "_searches_this_turn",  # T-0125
        "_search_facts",  # T-0125
    }
    assert must_clear <= clears, f"A-3 names these as cleared: {sorted(must_clear - clears)}"
    must_keep = {
        "_assist_flags",
        "assists",
        "session_log",
        "log_dir",
        "tool_kit",
        "llm",
        "memory",
        "clock",
        "config",
        "machine",
    }
    assert must_keep <= keeps, (
        f"A-3 keeps the assists, the log toggle and the warm services: {sorted(must_keep - keeps)}"
    )
    # T-0177 A-P2/B9: the vision choice survives a reset (T-0162 D-3); every other vision attribute the session
    # holds (A5's frozen copy, B7's published-frame slot and its last-published key, whatever their names) is
    # turn- or frame-scoped and is cleared by it. The behaviour is pinned in test_t0177_vision_frame_publish
    # (``test_a_reset_clears_the_frame_slot``); this row pins the classification.
    assert "vision_source" in keeps, "T-0177 B9: the vision choice survives a reset (RESET_KEEPS)"
    vision_state = {n for n in attrs if "vision" in n.lower()} - {"vision_source"}
    assert "_turn_vision" in vision_state, (
        f"premise: A5's frozen copy is found; {sorted(vision_state)}"
    )
    assert vision_state <= clears, (
        f"T-0177 B9: every vision attribute but the choice is cleared by a reset (the frame slot included); "
        f"in RESET_KEEPS: {sorted(vision_state - clears)}"
    )


# --- T-0181 A-R14: the mic switch survives a reset, the word fence and span ledger do not ---------------

#: T-0181 group A's BASE (main at 74ae8b25, where group A branched; a literal): the attributes ``Session.__init__``
#: set there are the ones T-0181 did not add
T0181_BASE_SHA = "74ae8b25229c65494530c18a26e00e503a932c19"


def _init_attributes_of(source: str) -> set[str]:
    """``init_attributes`` over a ``session.py`` source text (``Session.__init__``'s ``self.<name>`` targets)."""
    module = ast.parse(source)
    init = next(
        f
        for c in module.body
        if isinstance(c, ast.ClassDef) and c.name == "Session"
        for f in c.body
        if isinstance(f, ast.FunctionDef) and f.name == "__init__"
    )
    names: set[str] = set()
    for node in ast.walk(init):
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign | ast.AugAssign):
            targets = [node.target]
        stack = list(targets)
        while stack:
            t = stack.pop()
            if isinstance(t, ast.Tuple | ast.List):
                stack.extend(t.elts)
            elif (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                names.add(t.attr)
    return names


def _holds(value: Any, needle: Any, depth: int = 0) -> bool:
    """``needle`` (an int or a str) occurs in ``value``: the value itself, or an element, key, field or attribute of it,
    four levels deep. Read-only: nothing found is ever called."""
    if depth > 4:
        return False
    if isinstance(value, bool):
        return False
    if type(value) is type(needle) and value == needle:
        return True
    if isinstance(value, dict):
        return any(
            _holds(k, needle, depth + 1) or _holds(v, needle, depth + 1) for k, v in value.items()
        )
    if isinstance(value, list | tuple | set | frozenset):
        return any(_holds(v, needle, depth + 1) for v in value)
    fields = getattr(value, "__dict__", None)
    if isinstance(fields, dict) and not isinstance(value, type):
        return any(_holds(v, needle, depth + 1) for v in fields.values())
    slots = getattr(type(value), "__slots__", ())
    return any(_holds(getattr(value, s, None), needle, depth + 1) for s in slots)


async def test_t0181_reset_keeps_the_mic_switch_and_clears_the_fence_and_ledger(tmp_path):
    """T-0181 A-R14 (and A9): the mute flag, the gate state and ``mic_unavailable`` are in ``RESET_KEEPS``; the A0-S2
    word fence and the A6 span ledger are in ``RESET_CLEARS`` (the reset fence already drops pre-reset words, so the
    ledger and the fence are pre-reset conversational state).

    The attribute names are the implementer's, so they are found by *value*, never by name, among the attributes
    ``Session.__init__`` sets now and did not set at T-0181 A's BASE 74ae8b25 (both read by AST; values only read,
    never called):

    - the **reason**: a session built with ``mic_unavailable=<a reason no other code holds>``; every new attribute
      holding that exact string is in ``RESET_KEEPS``, and there is one;
    - the **mute flag**: a mic session before and after a mute; at least one new attribute that was ``False`` and is
      ``True`` is in ``RESET_KEEPS`` (the gate's behaviour across a reset, zeros after it, is pinned by
      ``test_t0181_mic_state_wire.test_the_mute_survives_a_reset``);
    - the **word fence**: a voice question whose first word starts at audio 1360 ms (a value nothing else new holds)
      is abandoned by the mute (A0-2); while muted, every new attribute holding 1360 (the fence's lower bound, the
      abandoned turn's ``onset_ms``) is in ``RESET_CLEARS``, and there is one;
    - the **span ledger**: right after the unmute, every new attribute holding the span's last zeroed frame's
      ``t_ms`` is in ``RESET_CLEARS``, and there is one (the gate state may hold the first frame of a state, never the
      last zeroed one).

    Finally the classification is the reset's truth: after a reset, every one of those KEEPS attributes still holds
    its value, and every one of those CLEARS attributes holds neither 1360 nor the span's last ``t_ms``."""
    from tests import _git_range as gr
    from tests.unit._t0181_rig import mic_rig, reactive_stt, zero_spans

    gr.require_commits(T0181_BASE_SHA)
    base_src = gr._git("show", f"{T0181_BASE_SHA}:local_astra/conductor/session.py")
    assert base_src.returncode == 0, base_src.stderr
    new = sorted(init_attributes() - _init_attributes_of(base_src.stdout))
    keeps = set(session_module.RESET_KEEPS)
    clears = set(session_module.RESET_CLEARS)
    reason = "t0181 a-r14: a reason held by no other attribute"
    onset = 1360

    def snap(session: Any) -> dict[str, Any]:
        return {n: getattr(session, n, None) for n in new}

    # the reason
    async with mic_rig(tmp_path / "reason", mic_unavailable=reason) as rig:
        holding_reason = [n for n, v in snap(rig.session).items() if _holds(v, reason)]
        await rig.reset()
        after_reason = snap(rig.session)
    assert holding_reason, (
        f"T-0181 A-R14: no attribute new since {T0181_BASE_SHA[:8]} holds mic_unavailable; new {new}"
    )
    assert set(holding_reason) <= keeps, (
        f"T-0181 A-R14: mic_unavailable survives a reset (RESET_KEEPS); in RESET_CLEARS: "
        f"{sorted(set(holding_reason) - keeps)}"
    )
    assert all(_holds(after_reason[n], reason) for n in holding_reason), (
        "T-0181 A-R14: the reason is still held after the reset"
    )

    # the flag, the fence and the ledger
    (tmp_path / "mic").mkdir()
    async with mic_rig(tmp_path / "mic", stt=reactive_stt(steps=True)) as rig:
        await rig.advance(400, step=40)
        before = snap(rig.session)
        rig.say("hello quokka narwhal zebra", onset)
        await rig.until(
            lambda: len(rig.state.words) >= 1,
            "premise: the machine never held the first word",
            limit_ms=4_000,
            step=40,
        )
        assert rig.state.words[0].start_ms == onset, rig.state.words
        assert [n for n, v in before.items() if _holds(v, onset)] == [], (
            "premise: no new attribute held the onset before the question"
        )
        await rig.engage()
        await rig.advance(1_200, step=40)
        assert rig.state.state is TurnState.IDLE, (
            f"premise: the mute abandoned the question; {rig.state.state}"
        )
        muted = snap(rig.session)
        flags = [n for n in new if before[n] is False and muted[n] is True]
        fence = [n for n, v in muted.items() if _holds(v, onset)]
        await rig.release()
        ((span_first, span_last),) = zero_spans(rig.rec_stt)
        unmuted = snap(rig.session)
        # the ledger holds the span (both bounds); a gate attribute may hold one frame time, so one bound is not enough
        ledger = [n for n, v in unmuted.items() if _holds(v, span_first) and _holds(v, span_last)]
        await rig.mute(True)
        await rig.reset()
        after = snap(rig.session)
        kept_flags = [n for n in flags if n in keeps]
    assert kept_flags, (
        f"T-0181 A-R14: the mute flag survives a reset: no new attribute that went False -> True on the mute is in "
        f"RESET_KEEPS; flips {flags}"
    )
    assert all(after[n] is True for n in kept_flags), (
        f"T-0181 A-R14/A9: re-muted before the reset, the kept flag is still True after it; "
        f"{ {n: after[n] for n in kept_flags} }"
    )
    assert fence, (
        f"T-0181 A-R14: no new attribute holds the abandoned turn's onset {onset} (the A0-S2 fence); {new}"
    )
    assert set(fence) <= clears, (
        f"T-0181 A-R14: the A0-S2 word fence is cleared by a reset; in RESET_KEEPS: {sorted(set(fence) - clears)}"
    )
    assert ledger, (
        f"T-0181 A-R14: no new attribute holds the span's last zeroed t_ms {span_last} (the ledger); {new}"
    )
    assert set(ledger) <= clears, (
        f"T-0181 A-R14: the A6 span ledger is cleared by a reset; in RESET_KEEPS: {sorted(set(ledger) - clears)}"
    )
    stale = [n for n in {*fence, *ledger} if _holds(after[n], onset) or _holds(after[n], span_last)]
    assert not stale, (
        f"T-0181 A-R14: still holding pre-reset fence/span bounds after the reset: {stale}"
    )


# --- A-6, adversarial: nothing stale after the echo -----------------------------------------------------


async def test_no_pre_reset_message_is_published_after_cleared(tmp_path):
    """R1: Restart mid-canvas, with the LLM and TTS still holding the cut turn's work.

    The generate is held after its first commit and synthesis is held, so the cut turn still has a
    canvas stream, queued utterances and playback in flight when the reset lands. Both are then let
    go. No new turn is opened, so *any* turn-scoped message after the echo is a pre-reset one.
    """
    script = [answer("One. ", 0), answer("Two. ", 1), answer("Three. ", 2), answer("Four.", 3)]
    async with tool_rig(tmp_path, kit=None, script=script) as rig:
        rig.llm.hold_after(1)
        await rig.ask()
        await rig.until(lambda: rig.state.state is TurnState.SPEAKING, "never reached SPEAKING")
        rig.tts.hold()
        await do_reset(rig)
        rig.llm.release()
        rig.tts.release()
        await rig.advance(3000)
        msg = one_cleared(rig)
        at = rig.published.messages.index(msg)
        after = [getattr(m, "type", "?") for m in rig.published.messages[at + 1 :]]
        stale = [t for t in after if t.startswith(STALE_PREFIXES)]
        assert not stale, f"A-6: pre-reset messages published after astra.session.cleared: {stale}"
        assert rig.state.state is TurnState.IDLE and rig.state.history == ()
        assert not [c for c in rig.machine.commands() if c.type == "commit_history"], (
            "A-2: the cut turn is never committed to history"
        )


# --- reviewer B rows (amendment 2026-10-27 (d)) -----------------------------------------------------------
#
# The A-6 filter, the floors and the browser reset were left unpinned: 21 mutants survived. Each row below
# *produces* the stale thing it is about (a message of a pre-reset turn, a pre-reset scene change, pre-reset assist
# text, the previous conversation's page) and has the companion that the too-wide direction would break.

TURN_KINDS = ("tool.call", "spoken", "played", "transcript", "surface", "fence")


def turn_message(kind: str, turn: str) -> Any:
    """One message of ``kind`` belonging to ``turn`` (``t<seq>``), built by the contract's own example."""
    from local_astra.contracts import protocol as P
    from tests.unit._examples import example_of

    if kind == "tool.call":
        return example_of(P.AstraToolCall).model_copy(update={"turn_id": turn})
    if kind == "spoken":
        return P.AstraSpoken(turn_id=turn, utterance_id=f"{turn}-u1", text="Stale words.")
    if kind == "played":
        return example_of(P.AstraSpokenPlayed).model_copy(update={"turn_id": turn})
    if kind == "transcript":
        return P.AstraTranscript(turn_id=turn, text="a stale question", final=True)
    if kind == "surface":
        return example_of(P.AstraUiSurface).model_copy(update={"surface_id": f"turn-{turn}"})
    if kind == "fence":
        return example_of(P.AstraUiFence).model_copy(update={"turn_id": turn})
    raise AssertionError(kind)  # pragma: no cover


def session_publish(rig: Any) -> Any:
    """The session's one HUD egress (A-6 filters there); the row produces the stale message through it."""
    publish = getattr(rig.session, "_publish", None)
    assert callable(publish), "A-6: Session._publish is the session's HUD egress"
    return publish


def delivered(rig: Any, publish: Any, message: Any) -> list[Any]:
    n = len(rig.published.messages)
    publish(message)
    return rig.published.messages[n:]


@pytest.mark.parametrize("turns_before", [1, 2])
@pytest.mark.parametrize("kind", TURN_KINDS)
async def test_a_message_of_a_pre_reset_turn_is_dropped_and_the_next_turns_is_not(
    tmp_path, kind, turns_before
):
    """A-6, one row per ``is_pre_reset`` branch: the floor is the turn_seq at the reset, inclusive.

    With n turns before the reset, ``t1``..``tn`` are dropped and ``t(n+1)`` -- the first turn of the new session --
    is delivered. n = 2 tells "the floor is turn_seq" apart from "the floor is 1"; ``tn`` dropped and ``t(n+1)``
    delivered is the ``<=`` boundary in both directions.
    """
    async with tool_rig(tmp_path, kit=None, script=[answer(FIRST_A)]) as rig:
        publish = session_publish(rig)
        for i in range(turns_before):
            await answered_turn(rig, f"question number {i}")
        floor = rig.state.turn_seq
        assert floor == turns_before, f"premise: {turns_before} turns opened, turn_seq {floor}"
        probe = turn_message(kind, f"t{floor}")
        assert delivered(rig, publish, probe) == [probe], (
            "premise: before a reset nothing is filtered"
        )
        await do_reset(rig)
        one_cleared(rig)
        for seq in range(1, floor + 1):
            stale = turn_message(kind, f"t{seq}")
            assert delivered(rig, publish, stale) == [], (
                f"A-6: {stale.type} of pre-reset turn t{seq} (floor t{floor}) is published after the echo"
            )
        fresh = turn_message(kind, f"t{floor + 1}")
        assert delivered(rig, publish, fresh) == [fresh], (
            f"A-6 too wide: {fresh.type} of t{floor + 1}, the first turn after the reset, is dropped"
        )


@pytest.mark.parametrize("turns_before", [1, 2])
@pytest.mark.parametrize(
    "model_name", ["AstraCanvasStep", "AstraCanvasCommit", "AstraCanvasSynthetic"]
)
async def test_a_canvas_frame_of_a_pre_reset_request_is_pre_reset_and_the_next_is_not(
    tmp_path, model_name, turns_before
):
    """Reviewer A: ``Session.is_pre_reset``'s canvas branch, whose request ids carry the session id and ``t<seq>``.

    At seq == floor the frame is pre-reset (dropped by ``_publish`` too); at floor + 1 it is not (delivered). Both
    directions of the ``<=`` boundary, with the floor at turn_seq (n = 2).
    """
    from local_astra.contracts import protocol as P
    from tests.unit._examples import example_of

    async with tool_rig(tmp_path, kit=None, script=[answer(FIRST_A)]) as rig:
        publish = session_publish(rig)
        for i in range(turns_before):
            await answered_turn(rig, f"question number {i}")
        floor = rig.state.turn_seq
        sid = rig.session.session_id

        def canvas(seq: int) -> Any:
            rid = f"{sid}-t{seq}-gen-1"
            return example_of(getattr(P, model_name)).model_copy(update={"request_id": rid})

        assert rig.session.is_pre_reset(canvas(floor)) is False, "premise: no reset, no filter"
        await do_reset(rig)
        one_cleared(rig)
        at_floor, after = canvas(floor), canvas(floor + 1)
        assert rig.session.is_pre_reset(at_floor) is True, (
            f"A-6: a canvas frame of t{floor} (the floor) is pre-reset"
        )
        assert delivered(rig, publish, at_floor) == [], "A-6: …and is not published"
        assert rig.session.is_pre_reset(after) is False, (
            f"too wide: a canvas frame of t{floor + 1}, after the reset, is not pre-reset"
        )
        assert delivered(rig, publish, after) == [after], "too wide: …and is published"


async def test_turns_after_a_reset_never_mention_the_old_frame(tmp_path, monkeypatch):
    """A-3 (reviewer's 'window a' probe): two turns after a reset, with no new frame, carry nothing of it."""
    memory = CountingMemory()
    use_memory(monkeypatch, memory)
    old = frame(0, "a")
    async with tool_rig(tmp_path, kit=None, script=[answer(FIRST_A)], frames=[old]) as rig:
        await answered_turn(rig)
        assert carries_frame(rig.llm.gen_prompts[-1][1], old), "premise: turn 1 carried frame a"
        prompts_at, ingests_at = len(rig.llm.gen_prompts), len(memory.ingests)
        await do_reset(rig)
        await answered_turn(rig, "the second question")
        await answered_turn(rig, "the third question")
        later = rig.llm.gen_prompts[prompts_at:]
        assert len(later) >= 2, f"premise: turns 2 and 3 each sent a prompt ({len(later)})"
        b64 = base64.b64encode(old.jpeg).decode()
        for rid, prompt in later:
            dump = prompt.model_dump_json()
            for needle in (old.title, old.phash, b64):
                assert needle not in dump, (
                    f"A-3: {rid} after the reset mentions the old frame ({needle!r})"
                )
        for ep in memory.ingests[ingests_at:]:
            assert ep.phash != old.phash and ep.window_title != old.title, (
                f"A-3: an episode after the reset is the old frame: {ep}"
            )


class QueuePerception(S.RecordingPerception):  # type: ignore[misc,name-defined]
    """The fake perception, with a scene-change stream the row feeds while the session runs."""

    def __init__(self, frames: Any = ()) -> None:
        super().__init__(frames)
        self.queue: asyncio.Queue[Any] = asyncio.Queue()

    async def events(self) -> Any:
        while True:
            yield await self.queue.get()


async def test_a_scene_change_stamped_at_or_before_the_reset_is_ignored(tmp_path, monkeypatch):
    """A-3's floor on the scene-change path: ``ts`` at or below the reset never reaches the machine; after it does."""
    monkeypatch.setattr(S, "RecordingPerception", QueuePerception)
    stale_hash, fresh_hash = "d" * 16, "e" * 16
    async with tool_rig(tmp_path, kit=None, script=[answer(FIRST_A)]) as rig:
        await answered_turn(rig)
        reset_at = await do_reset(rig)
        await rig.onset()
        assert rig.state.state is TurnState.LISTENING, "premise: a new turn is listening"
        rig.perception.queue.put_nowait(
            E.SceneChanged(t_ms=reset_at, source="screen", phash=stale_hash)
        )
        await rig.pump()
        rig.perception.queue.put_nowait(
            E.SceneChanged(t_ms=rig.clock.now_ms, source="screen", phash=fresh_hash)
        )
        await rig.pump()
        seen = [e.phash for e in rig.machine.events(E.SceneChanged)]
        assert stale_hash not in seen, (
            f"A-3: a scene change stamped at the reset ({reset_at}) reached the machine"
        )
        assert fresh_hash in seen, "A-3 too wide: a scene change after the reset was dropped"


def ASSIST_HORIZON_MS() -> int:
    from local_astra.perception import assists

    return int(assists.ASSIST_HORIZON_MS)


OLD_PAGE = "OLD-PAGE-CANARY-7f3a"
NEW_PAGE = "NEW-PAGE-CANARY-91c2"


async def test_pre_reset_assist_text_never_reaches_a_post_reset_prompt(tmp_path):
    """Amendment (b): the flags survive, the buffered text does not."""
    from local_astra.contracts.protocol import AstraFrameText

    def page_text(ts_ms: int, text: str) -> Any:
        return AstraFrameText(source=Source.SCREEN, ts_ms=ts_ms, kind="page_text", text=text)

    async with tool_rig(
        tmp_path, kit=None, script=[answer(FIRST_A)], frames=[frame(0, "a")]
    ) as rig:
        rig.session.apply_assist_toggle(AstraAssistToggle(flags=AssistFlags(page_text=True)))
        assert rig.session.accept_frame_text(page_text(0, OLD_PAGE)) is not None, (
            "premise: accepted"
        )
        await answered_turn(rig)
        assert OLD_PAGE in rig.llm.gen_prompts[-1][1].model_dump_json(), (
            "premise: turn 1 saw the page text"
        )
        reset_at = await do_reset(rig)
        rig.perception.add(frame(reset_at + 1, "b"))
        # Turn 2 gets no new page text: the buffer answers the newest entry per kind, so a newer text would mask
        # the old one and make this check vacuous. The old text is still inside the assist horizon.
        await answered_turn(rig, "and what about now")
        assert rig.clock.now_ms < ASSIST_HORIZON_MS(), "premise: the old text is inside the horizon"
        dump = rig.llm.gen_prompts[-1][1].model_dump_json()
        assert OLD_PAGE not in dump, (
            "amendment (b): pre-reset page text reached the first prompt after the reset"
        )
        # too wide: page text sent after the reset (Page text still on) reaches the next turn
        assert rig.session.accept_frame_text(page_text(rig.clock.now_ms, NEW_PAGE)) is not None
        await answered_turn(rig, "and the third question")
        assert NEW_PAGE in rig.llm.gen_prompts[-1][1].model_dump_json(), (
            "too wide: page text sent after the reset (Page text still on) was dropped"
        )
        assert rig.session.assists.state_message().flags.page_text is True, "the flag survives"


async def test_a_reset_sends_the_agent_browser_back_to_its_start_page(tmp_path):
    """Amendment (c): ``reset_page()`` is awaited exactly once, and the next turn's frame is taken after it."""
    from tests.unit._live_browser_support import FakeLiveBrowser, live_frame, live_rig

    start = live_frame("start-page")

    class ResettableBrowser(FakeLiveBrowser):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.page_resets = 0

        async def reset_page(self) -> None:
            self.calls.append(("reset_page",))
            self.page_resets += 1
            self.frames.appendleft(
                start
            )  # what the browser shows once it is back on its start page

    browser = ResettableBrowser([live_frame("previous-conversation")])
    async with live_rig(tmp_path, browser, script=[answer(FIRST_A)]) as rig:
        await answered_turn(rig)
        # live prompts carry the session's own capture hash, so frames are recognised by their bytes
        old = [base64.b64encode(f.jpeg).decode() for f in browser.frames_returned]
        assert old, "premise: turn 1 took an own frame"
        await do_reset(rig)
        assert browser.page_resets == 1, (
            f"amendment (c): reset_page() once per reset; {browser.page_resets}"
        )
        await answered_turn(rig, "what is on the page now")
        urls = [p.url for p in image_parts(rig.llm.gen_prompts[-1][1])]
        start_b64 = base64.b64encode(start.jpeg).decode()
        assert any(start_b64 in u for u in urls), (
            "amendment (c): the first turn after the reset shows the start page "
            f"({len(urls)} image(s), none the start page)"
        )
        assert not [b for b in old if any(b in u for u in urls)], (
            "the previous conversation's page is not shown again"
        )
        assert browser.page_resets == 1, "one reset, one reset_page"
