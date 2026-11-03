"""T-0179 group B through the session: one fresh retry after a degenerate abort, then the ``lost_thread`` clip
(docs/tasks/T-0179.md B-1 to B-8; amendment B-7: one *ending* clip per turn).

Every row runs T-0142's ``guard_rig`` (imported: the session over the fakes with ``degeneracy_guard=True``; ``live``
with the live kit over a ``FakeLiveBrowser``, ``dry_run``, or ``off``, the conversational rig), wrapped by
``retry_rig`` below, which adds one journal (``rig.order``): T-0142's ``("abort", engine id)`` and ``("submit", event,
request id)``, plus ``("hud.error", message)`` when the session publishes a turn error and ``("play", kind, source)``
for every utterance handed to the audio device. The fake LLM is ``RetryLlm`` (T-0142's ``LateLlm``) plus a record of
which engine requests were live when each generate started (P4 read at the engine, see below) and abort hooks.

**What the rows read** (from the task file, verbatim where it is verbatim; the readings are in the test-author's
report):

- the retry note: ``kind == "note"``, ``what == "degenerate request: retrying once"``, exactly the keys
  ``what, request_id, failed_attempt_id, attempt_id, phase, segment`` plus ``step`` on a ``tool_step`` (B-2's
  ``"step"?``); ``segment`` is the segment that tripped (``thought`` | ``answer``); ids, never text;
- one ``llm.request`` record per attempt; the retry's record equals the first except ``attempt_id``; the ``Prompt``,
  ``LlmOptions`` and plugin arguments handed to the backend are equal, and (tools on) the body the real
  ``VllmClient.request_body`` builds equals the first once the engine request id is taken out (T-0171's ``body_of``).
  That equality is what kills a seed or any added sampling option (reviewer mutation 6): ``LlmOptions`` cannot carry
  ``seed`` (``extra="forbid"``), so the only way to add one is ``xargs``, which changes the body. The fake is **not**
  taught to reject a seed: the real engine's behaviour for a ``seed`` under ``vllm_xargs`` is not known (H-T179-2 is
  about the top-level field), and a fake must not invent it;
- the turn line: ``extras["degenerate_retries"] == 1`` on a retried turn, absent otherwise; ``degeneracy_guard`` is
  ``{"on": true, "tripped": n}`` counting every tripped request (B-3); ``extras["system_clip"] == "lost_thread"`` on the
  B-4 end (B-6), absent on every other turn of this module;
- the clip: an utterance handed to the audio device with ``source=""`` whose ``kind`` is ``lost_thread`` and whose
  ``text`` is H-T179-4's sentence (as T-0178's rows read ``out_of_steps``). Whether the machine or the session emits
  the ``PlayClip`` is **not** pinned here (see the report: T-0178's emission row is ``_finish_step``, on ``LlmDone``,
  and a degenerate end reaches the machine as ``LlmError``); the order is: both aborts, then the one ``LlmError``, then
  the turn's ``hud.error``, then the clip's play;
- "ends as at BASE" / "the T-0142 way": no ``LlmDone`` for that id, ``tool_loop_end == "degenerate"`` when the loop is
  on. Amendment B-6a (2026-10-28): the B-4 end's ``LlmError`` message is ``session.DEGENERATE_RETRIED_MESSAGE`` (pinned
  by name; the retry rows here trip on thought), and a first trip that is not retried keeps ``DEGENERATE_MESSAGE`` /
  ``DEGENERATE_ANSWER_MESSAGE``;
- P4 (one live generate) is read at the engine (``RetryLlm.live_at_start``: the turn requests the fake still held live
  when each generate started, prefills and captions excluded), not from ``RecordingLlm.concurrency``: a session that
  returns out of ``async for`` leaves the fake's async generator to be finalised later, so the recording fake's
  bookkeeping would count an already aborted attempt as live.

**Pending and active rows.** Every row that needs the retry is ``pending(task="T-0179")`` and red at BASE on a named
assertion. The negative rows (no retry after an answer was submitted, abort before the resend, the out-of-scope paths)
hold at BASE already; they are active so they guard the implementation from its first commit, and each names the
too-wide retry it catches.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, loop on (8 steps unless a row
says otherwise) in ``live`` and ``dry_run``, 1280x800 live frames, n = 1 per row, deterministic.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit
from local_astra.contracts.turn import TurnState
from tests.fakes.llm import engine_empty_error, is_caption_prompt
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0142_session as t42
from tests.unit._hud_rig import split_thoughts
from tests.unit._tool_turn_support import (
    REPAIRABLE_CLICK,
    REPAIRED_CLICK,
    VALID_CLICK,
    answer,
    thought,
    tool,
)
from tests.unit.test_session_engine_empty_retry import body_of
from tests.unit.test_session_reground import click

NOTE = "degenerate request: retrying once"  # B-2, verbatim
NOTE_KEYS = frozenset({"what", "request_id", "failed_attempt_id", "attempt_id", "phase", "segment"})
EXTRA = "degenerate_retries"  # B-3, verbatim
CLIP_KEY = "system_clip"  # B-6, verbatim
LOST_THREAD = "lost_thread"
LOST_THREAD_TEXT = (
    "Sorry, I lost my train of thought. Could you ask again?"  # B-6 / H-T179-4, verbatim
)
EMPTY_NOTE = "the engine returned an empty retryable response; retrying once"  # T-0171 criterion 5
GUARD = "degeneracy_guard"

S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
#: a thought that trips the T-0142 rule (8 back-to-back) at its own commit, with a sentinel before the repeats
TRIP_THOUGHT = "ZQX-TRIP-THOUGHT I see the page and" + " the" * 12
TRIP_THOUGHT_2 = "ZQX-TRIP-AGAIN I see the form and" + " the" * 12
#: an answer commit that trips the T-0142b rule (6 back-to-back) at its own commit
TRIP_ANSWER = "ZQX-TRIP-ANSWER It is" + " the" * 8
LATE = "ZQX-LATE-T0179 your order is placed."
CLEAN_THOUGHT = "I click the search field now."
FINAL = "Done, the settings are open."
ANSWER = "It is a shopping cart page."
CLICK4 = click(640, 360)
CLICK4_ARGS = {"x": 640, "y": 360}
CLICK_X = click(300, 300)


def retried_message() -> str:
    """Amendment B-6a: ``session.DEGENERATE_RETRIED_MESSAGE``, a non-empty str distinct from T-0142's two messages."""
    s = t42.session_mod()
    value = getattr(s, "DEGENERATE_RETRIED_MESSAGE", None)
    assert isinstance(value, str) and value.strip(), (
        "T-0179 B-6a: session.DEGENERATE_RETRIED_MESSAGE (a non-empty str module constant) does not exist yet"
    )
    assert value not in (s.DEGENERATE_MESSAGE, s.DEGENERATE_ANSWER_MESSAGE), (
        "T-0179 B-6a: a distinct message"
    )
    return value


# --------------------------------------------------------------------------------------------- the rig


class RetryLlm(t42.LateLlm):
    """``LateLlm`` (it keeps serving an aborted stream with ``late=True``) plus: ``live_at_start`` (engine id, the turn
    requests the fake still held live when it started) and ``abort_hooks`` (awaited after each abort, with the id)."""

    def __init__(self, script: Any = (), *, late: bool = False) -> None:
        super().__init__(script, late=late)
        self.live_at_start: list[tuple[str, frozenset[str]]] = []
        self.abort_hooks: list[Callable[[str], Awaitable[None]]] = []

    async def abort(self, request_id: str) -> None:
        await super().abort(request_id)
        for hook in list(self.abort_hooks):
            await hook(request_id)

    async def generate(  # type: ignore[override]
        self,
        prompt: Any,
        request_id: str,
        options: Any = None,
        *,
        xargs: Any = None,
        cd: Any = None,
    ) -> Any:
        if not is_caption_prompt(prompt):
            prefills = {rid for rid, _ in self.prefills}
            others = frozenset(r for r in self.active if r not in prefills and "caption" not in r)
            self.live_at_start.append((request_id, others))
        async for c in super().generate(prompt, request_id, options, xargs=xargs, cd=cd):
            yield c


@asynccontextmanager
async def retry_rig(
    tmp_path: Path, *, mode: str = "live", script: Any = (), late: bool = False, llm: Any = None
) -> AsyncIterator[Any]:
    """``t42.guard_rig`` with the guard on, a ``RetryLlm`` and the journal of the module docstring."""
    llm = llm if llm is not None else RetryLlm(late=late)
    async with t42.guard_rig(tmp_path, mode=mode, script=list(script), llm=llm) as rig:
        journal = rig.order
        publish = rig.session.publish_turn_error

        def published(message: str) -> None:
            journal.append(("hud.error", message))
            publish(message)

        rig.session.publish_turn_error = published
        play = rig.audio.play

        async def played(utterance: Any, *, source: str | None = None) -> None:
            journal.append(("play", getattr(utterance, "kind", None), source))
            await play(utterance, source=source)

        rig.audio.play = played
        yield rig


async def run(rig: Any, *steps: Any, question: str = tsl.QUESTION, limit_ms: int = 40_000) -> None:
    rig.llm.script_steps(*steps)
    await rig.ask(question)
    await rig.turn_done("T-0179: the turn never ended", limit_ms=limit_ms)
    await rig.advance(300)


async def run_off(rig: Any, question: str = "what is on this screen") -> None:
    await rig.ask(question)
    await rig.turn_done("T-0179: the turn never ended", limit_ms=30_000)
    await rig.advance(300)


# --------------------------------------------------------------------------------------------- readers


def engine_ids(rig: Any, machine_id: str) -> list[str]:
    return [rid for rid, _ in tsl.generates_for(rig, machine_id)]


def prompts_of(rig: Any, machine_id: str) -> list[Any]:
    return [p for _, p in tsl.generates_for(rig, machine_id)]


def call_args(rig: Any, engine_id: str) -> tuple[Any, Any]:
    """(options, xargs) the backend was handed for ``engine_id``."""
    found = [
        (o, x)
        for (rid, _p, o), x in zip(rig.llm.generates, rig.llm.xargs_log, strict=True)
        if rid == engine_id
    ]
    assert len(found) == 1, (engine_id, found)
    return found[0]


def retry_notes(rig: Any) -> list[dict[str, Any]]:
    return [d for d in rig.kind("note") if d.get("what") == NOTE]


def trip_notes(rig: Any) -> list[dict[str, Any]]:
    return [
        d
        for d in rig.kind("note")
        if d.get("what") in (t42.NOTE, "degenerate answer: request aborted")
    ]


def requests_for(rig: Any, machine_id: str) -> list[dict[str, Any]]:
    return [d for d in rig.kind("llm.request") if d.get("request_id") == machine_id]


def llm_errors(rig: Any, machine_id: str) -> list[Any]:
    return [e for e in rig.machine.events(E.LlmError) if e.request_id == machine_id]


def clips(rig: Any, kind: str | None = None) -> list[Any]:
    """Utterances handed to the device with ``source=""`` (clips), optionally of one kind (the ``ack`` filler too)."""
    out = rig.audio.clip_plays()
    return [u for u in out if kind is None or getattr(u, "kind", None) == kind]


def ending_clip_kinds(rig: Any) -> list[str]:
    ending = ("out_of_steps", "not_finished", LOST_THREAD)
    return [str(getattr(u, "kind", None)) for u in clips(rig) if getattr(u, "kind", None) in ending]


def spoken(rig: Any) -> str:
    return rig.spoken_sources()


def executed(rig: Any) -> list[tuple[str, dict[str, Any]]]:
    return [(i.call_id, dict(i.arguments)) for i in rig.browser.invocations]


def messages(rig: Any) -> str:
    return json.dumps([str(e.message) for e in rig.machine.events(E.LlmError)])


def assert_one_retry_note(
    rig: Any, machine_id: str, failed: str, retry: str, phase: str, segment: str, step: int | None
) -> None:
    found = retry_notes(rig)
    assert len(found) == 1, f"T-0179 B-2: one retry note; got {found}"
    note = found[0]
    keys = NOTE_KEYS | ({"step"} if step is not None else set())
    assert set(note) == keys, f"T-0179 B-2: the note's keys are {sorted(keys)}; got {sorted(note)}"
    got = (note["request_id"], note["failed_attempt_id"], note["attempt_id"], note["phase"])
    assert got == (machine_id, failed, retry, phase), (
        f"T-0179 B-2: the note names the machine id, both attempts and the phase; got {note}"
    )
    assert note["segment"] == segment, f"T-0179 B-2: the tripped segment {segment!r}; got {note}"
    if step is not None:
        assert note["step"] == step, note
    dumped = json.dumps(note)
    assert "ZQX" not in dumped and tsl.QUESTION not in dumped, (
        f"T-0179 implementation notes: ids and counts in the note, never text: {note}"
    )


def assert_identical_resend(rig: Any, machine_id: str, *, tools_on: bool) -> tuple[str, str]:
    """B-2: exactly two attempts for ``machine_id``, fresh id, identical ``Prompt``/options/xargs (and body), the
    retry's ``llm.request`` record equal to the first except ``attempt_id``, written after the note and before the
    retry's first commit. Returns (failed, retry)."""
    ids = engine_ids(rig, machine_id)
    assert len(ids) == 2, f"T-0179 B-2: one resend of {machine_id} (2 engine requests); got {ids}"
    failed, retry = ids
    assert failed != retry, "T-0179 B-2: the resend carries a fresh attempt id"
    first, second = prompts_of(rig, machine_id)
    assert first == second, "T-0179 B-2: the identical Prompt (never rebuilt, never spliced)"
    assert call_args(rig, failed) == call_args(rig, retry), (
        "T-0179 B-2 / H-T179-2: the same options and plugin arguments (no seed, no sampling option)"
    )
    if tools_on:
        assert body_of(rig, failed) == body_of(rig, retry), (
            "T-0179 B-2: the body equals the first attempt's except for the attempt id"
        )
    recs = requests_for(rig, machine_id)
    assert [r.get("attempt_id") for r in recs] == [failed, retry], (
        f"T-0179 B-2: a new llm.request record for the resend; got {[r.get('attempt_id') for r in recs]}"
    )
    strip = [{k: v for k, v in r.items() if k != "attempt_id"} for r in recs]
    assert strip[0] == strip[1], (
        "T-0179 B-2: the retry's llm.request equals the first but for the id"
    )
    records = rig.log_records
    idx = [
        i
        for i, r in enumerate(records)
        if r["kind"] == "note" and (r.get("data") or {}).get("what") == NOTE
    ]
    req = [
        i
        for i, r in enumerate(records)
        if r["kind"] == "llm.request" and (r.get("data") or {}).get("attempt_id") == retry
    ]
    commits = [
        i
        for i, r in enumerate(records)
        if r["kind"] == "llm.commit" and (r.get("data") or {}).get("attempt_id") == retry
    ]
    assert idx and req and commits, (idx, req, commits)
    assert idx[0] < commits[0] and req[0] < commits[0], (
        "T-0179 B-2: the note and the retry's llm.request precede the retry's first commit"
    )
    return failed, retry


def assert_released_before_issue(rig: Any, failed: str, retry: str) -> None:
    """B-2 / P4: the failed attempt reached the engine's abort before the retry was sent, and no other turn request
    was live at the engine when the retry started."""
    order = list(rig.order)
    assert ("abort", failed) in order, f"premise: the tripped attempt was aborted; {order}"
    starts = [i for i, (rid, _) in enumerate(rig.llm.live_at_start) if rid == retry]
    assert len(starts) == 1, rig.llm.live_at_start
    live = rig.llm.live_at_start[starts[0]][1]
    assert live == frozenset(), (
        f"T-0179 B-2 / P4: one live generate; {sorted(live)} still live when the retry {retry} started"
    )
    assert failed in rig.llm.aborted, rig.llm.aborted


def assert_p4(rig: Any) -> None:
    worst = [(rid, sorted(live)) for rid, live in rig.llm.live_at_start if live]
    assert not worst, f"T-0179 B-8 / P4: a generate started beside a live one: {worst[:3]}"


def assert_no_clip_or_retry(rig: Any, extras: dict[str, Any]) -> None:
    assert retry_notes(rig) == [], f"no retry note; got {retry_notes(rig)}"
    assert clips(rig, LOST_THREAD) == [], "T-0179 B-6: no lost_thread clip on this end"
    assert EXTRA not in extras and CLIP_KEY not in extras, extras


def live_turn_steps() -> list[list[Commit]]:
    return [
        [thought("Step two plan.", 0), tool(tsl.CLICK2)],
        [thought("Step three plan.", 0), tool(tsl.CLICK3)],
    ]


# =====================================================================================================
# B-1 to B-3: a clean retry continues the turn
# =====================================================================================================


async def test_step_k_thought_trip_retried_clean(tmp_path: Path) -> None:
    """B-1/B-2/B-3, the t10 shape: the step-4 request's thought collapses; the session releases it, resends the
    identical step-4 request once under a fresh id, the retry's call executes and the turn ends ``final`` with
    ``degenerate_retries: 1`` and ``tripped: 1``; no clip, no error anywhere."""
    t42.degenerate_message()
    trip = [thought(TRIP_THOUGHT, 0, open_=True), answer(LATE, 1), tool(CLICK_X, canvas_idx=1)]
    clean = [thought(CLEAN_THOUGHT, 0), tool(CLICK4)]
    async with retry_rig(tmp_path, script=t42.STEP1) as rig:
        await run(rig, *live_turn_steps(), trip, clean, [answer(FINAL)])
        extras = dict(rig.record("t1").extras)
        done = executed(rig)
        speech = spoken(rig)
        hud_errors = rig.commands(C.HudError)
        errors = llm_errors(rig, "t1-gen-s4")
    failed, retry = assert_identical_resend(rig, "t1-gen-s4", tools_on=True)
    assert_one_retry_note(rig, "t1-gen-s4", failed, retry, "tool_step", "thought", 4)
    assert_released_before_issue(rig, failed, retry)
    assert done == [
        ("t1-call", tsl.CLICK_ARGS),
        ("t1-call-s2", tsl.CLICK2_ARGS),
        ("t1-call-s3", tsl.CLICK3_ARGS),
        ("t1-call-s4", CLICK4_ARGS),
    ], f"T-0179 B-3: the retry's call executes (and the tripped one never does); {done}"
    assert errors == [] and hud_errors == [], (errors, hud_errors)
    assert speech == S1 + FINAL, (
        f"T-0179 B-3: the turn continues as if the retry were first; {speech!r}"
    )
    assert extras.get("tool_loop_end") == "final", extras.get("tool_loop_end")
    assert extras.get(EXTRA) == 1, f"T-0179 B-3: extras[{EXTRA!r}] == 1; got {extras}"
    assert extras.get(GUARD) == {"on": True, "tripped": 1}, extras.get(GUARD)
    assert CLIP_KEY not in extras and ending_clip_kinds(rig) == [], (extras, ending_clip_kinds(rig))
    assert_p4(rig)


@pytest.mark.parametrize("mode", ["live", "off"])
async def test_step1_trip_retried_clean(tmp_path: Path, mode: str) -> None:
    """B-1/B-2/B-3 for the step-1 ``generate``: live (loop on, the retry's call executes) and tools off (the
    conversational rig: the retry's answer is voiced). A second, ordinary turn has no ``degenerate_retries`` key."""
    t42.degenerate_message()
    if mode == "live":
        first = [
            thought(TRIP_THOUGHT, 0, open_=True),
            answer(S1, 1),
            tool(VALID_CLICK, canvas_idx=1),
        ]
        second = [thought(CLEAN_THOUGHT, 0), answer(S1, 1), tool(VALID_CLICK, canvas_idx=1)]
    else:
        first = [thought(TRIP_THOUGHT, 0, open_=True), answer(LATE, 1)]
        second = [thought("Let me look at the page.", 0), answer(ANSWER, 1)]
    async with retry_rig(tmp_path, mode=mode, script=[answer("Second answer here.")]) as rig:
        rig.llm.script_generates(first, second)
        if mode == "live":
            await run(rig, [answer(S3)])
        else:
            await run_off(rig)
        extras = dict(rig.record("t1").extras)
        speech = spoken(rig)
        done = executed(rig) if mode == "live" else []
        rig.llm.script = [answer("Second answer here.")]
        await rig.ask("and now")
        await rig.until(lambda: len(rig.lines()) >= 2, "the second turn never ended")
        await rig.advance(300)
        extras2 = dict(rig.record("t2").extras)
    failed, retry = assert_identical_resend(rig, "t1-gen", tools_on=mode == "live")
    assert_one_retry_note(rig, "t1-gen", failed, retry, "generate", "thought", None)
    assert_released_before_issue(rig, failed, retry)
    if mode == "live":
        assert done == [("t1-call", tsl.CLICK_ARGS)], done
        assert speech.startswith(S1) and S3 in speech, speech
        assert extras.get("tool_loop_end") == "final", extras
    else:
        assert ANSWER in speech and "ZQX" not in speech, speech
    assert extras.get(EXTRA) == 1 and extras.get(GUARD) == {"on": True, "tripped": 1}, extras
    assert EXTRA not in extras2, f"T-0179 B-3: the key is absent when there was no retry; {extras2}"
    assert llm_errors(rig, "t1-gen") == [], llm_errors(rig, "t1-gen")
    assert_p4(rig)


@pytest.mark.parametrize("mode", ["live", "off"])
async def test_answer_trip_at_commit0_retried(tmp_path: Path, mode: str) -> None:
    """B-1 (T-0142b): the *answer* collapses at its first answer commit (nothing of the answer submitted, thought
    commits only) -> one retry, ``segment: "answer"``; live at step 2 (``tool_step``), tools off at step 1."""
    t42.degenerate_message()
    trip = [thought("Checking the page.", 0), answer(TRIP_ANSWER, 1)]
    if mode == "live":
        async with retry_rig(tmp_path, script=t42.STEP1) as rig:
            await run(rig, [*trip, tool(tsl.CLICK3, canvas_idx=1)], [answer(S3)])
            extras = dict(rig.record("t1").extras)
            speech = spoken(rig)
        mid, phase, step, want = "t1-gen-s2", "tool_step", 2, S1 + S3
    else:
        async with retry_rig(tmp_path, mode="off", script=[answer("unused")]) as rig:
            rig.llm.script_generates(trip, [thought("Checking the page.", 0), answer(ANSWER, 1)])
            await run_off(rig)
            extras = dict(rig.record("t1").extras)
            speech = spoken(rig)
        mid, phase, step, want = "t1-gen", "generate", None, ANSWER
    failed, retry = assert_identical_resend(rig, mid, tools_on=mode == "live")
    assert_one_retry_note(rig, mid, failed, retry, phase, "answer", step)
    assert want in speech and "ZQX" not in speech, speech
    assert extras.get(EXTRA) == 1 and extras.get(GUARD) == {"on": True, "tripped": 1}, extras


@pytest.mark.parametrize("mode", ["live", "off"])
async def test_no_retry_after_answer_submitted(tmp_path: Path, mode: str) -> None:
    """B-1 / B-6 / H-T179-7 (adversarial; reviewer mutations 1 and 4): an answer commit of the attempt reached the
    machine, then the answer collapses -> no resend, the BASE ending (``LlmError`` with the answer message, a
    ``hud.error``), no retry note, no clip, no ``degenerate_retries`` and no ``system_clip``. Active: holds at BASE."""
    said = "ZQX-SAID Opening the menu now. "
    if mode == "live":
        async with retry_rig(tmp_path, script=t42.STEP1) as rig:
            await run(
                rig, [answer(said, 0), answer(TRIP_ANSWER, 1), tool(tsl.CLICK3, canvas_idx=1)]
            )
            extras = dict(rig.record("t1").extras)
            mid = "t1-gen-s2"
    else:
        async with retry_rig(
            tmp_path, mode="off", script=[answer(said, 0), answer(TRIP_ANSWER, 1)]
        ) as rig:
            await run_off(rig)
            extras = dict(rig.record("t1").extras)
            mid = "t1-gen"
    submitted = [
        e
        for e in rig.machine.events(E.LlmCommit)
        if e.request_id == mid and e.segment == "answer" and "ZQX-SAID" in e.text
    ]
    assert submitted, f"premise: an answer commit of {mid} reached the machine before the trip"
    assert len(engine_ids(rig, mid)) == 1, f"T-0179 B-1: no resend; got {engine_ids(rig, mid)}"
    errors = llm_errors(rig, mid)
    # B-6a: a trip that is not retried keeps today's message (never the retried one)
    assert [e.message for e in errors] == [t42.session_mod().DEGENERATE_ANSWER_MESSAGE], errors
    assert rig.commands(C.HudError), "the BASE ending: hud.error"
    assert_no_clip_or_retry(rig, extras)
    assert extras.get(GUARD) == {"on": True, "tripped": 1}, extras.get(GUARD)


# =====================================================================================================
# B-4 / B-6 / B-7: the retry trips too
# =====================================================================================================


@pytest.mark.parametrize("phase", ["tool_step", "generate-live", "generate-off"])
async def test_second_trip_ends_with_clip(tmp_path: Path, phase: str) -> None:
    """B-4/B-6: the retry trips too -> exactly 2 engine requests, both aborted, then (in order) the one ``LlmError``
    with ``DEGENERATE_RETRIED_MESSAGE`` (B-6a), the turn's ``hud.error`` and one played ``lost_thread`` clip (``source=""``, H-T179-4's
    sentence, zero-length with no asset); ``tool_loop_end == "degenerate"`` with the loop on; ``tripped: 2``,
    ``degenerate_retries: 1``, ``system_clip: "lost_thread"``; the clip never in history or a ``tts.utterance``."""
    message = retried_message()
    trip1 = [thought(TRIP_THOUGHT, 0, open_=True), tool(CLICK_X)]
    trip2 = [thought(TRIP_THOUGHT_2, 0, open_=True), tool(CLICK4)]
    mode = "off" if phase == "generate-off" else "live"
    async with retry_rig(tmp_path, mode=mode, script=t42.STEP1) as rig:
        if phase == "tool_step":
            mid = "t1-gen-s2"
            await run(rig, trip1, trip2, [answer(S3)])
        else:
            mid = "t1-gen"
            rig.llm.script_generates(trip1, trip2, [answer("third request")])
            if mode == "live":
                await run(rig, [answer(S3)])
            else:
                await run_off(rig)
        extras = dict(rig.record("t1").extras)
        order = list(rig.order)
        lost = clips(rig, LOST_THREAD)
        history = " ".join(f"{h.user_text} || {h.assistant_spoken}" for h in rig.state.history)
        speaks = [c.source for c in rig.machine.commands(C.TtsSpeak)]
    ids = engine_ids(rig, mid)
    assert len(ids) == 2, f"T-0179 B-4: exactly 2 engine requests for {mid}; got {ids}"
    a1, a2 = ids
    errors = llm_errors(rig, mid)
    assert [e.message for e in errors] == [message], (
        f"T-0179 B-4/B-6a: one LlmError with DEGENERATE_RETRIED_MESSAGE; {errors}"
    )
    assert ("hud.error", message) in order, (
        f"T-0179 B-6a: the turn's hud.error carries the retried message; {order}"
    )
    assert rig.dones(mid) == [], "T-0179 B-4: no llm.done for the tripped id"
    if mode == "live":
        assert extras.get("tool_loop_end") == "degenerate", extras.get("tool_loop_end")
    assert len(lost) == 1, (
        f"T-0179 B-6: one lost_thread clip; got {[getattr(u, 'kind', None) for u in lost]}"
    )
    assert str(getattr(lost[0], "text", "")) == LOST_THREAD_TEXT, getattr(lost[0], "text", None)
    assert getattr(lost[0], "duration_ms", None) == 0, (
        "T-0179 B-6: no asset installed -> a zero-length clip (the clip honesty rule)"
    )
    assert ending_clip_kinds(rig) == [LOST_THREAD], (
        f"T-0179 B-7: one ending clip; {ending_clip_kinds(rig)}"
    )

    def at(entry: tuple[Any, ...]) -> int:
        assert entry in order, f"{entry} missing from the journal {order}"
        return order.index(entry)

    error_at = at(("submit", "LlmError", mid))
    hud_at = next(i for i, e in enumerate(order) if e[0] == "hud.error" and i > error_at)
    play_at = at(("play", LOST_THREAD, ""))
    assert at(("abort", a1)) < at(("abort", a2)) < error_at < hud_at < play_at, (
        f"T-0179 B-4/B-6: abort, abort, LlmError, hud.error, then the clip; {order}"
    )
    assert extras.get(EXTRA) == 1, extras
    assert extras.get(GUARD) == {"on": True, "tripped": 2}, (
        f"T-0179 B-3: both trips counted; {extras}"
    )
    assert extras.get(CLIP_KEY) == LOST_THREAD, (
        f"T-0179 B-6: the turn line records the clip; {extras}"
    )
    assert LOST_THREAD_TEXT not in history, "T-0179 B-6: the clip never enters history"
    assert not any(LOST_THREAD_TEXT in s for s in speaks), (
        "T-0179 B-6/B-8: no TtsSpeak for the clip"
    )
    assert not [r for r in rig.kind("tts.utterance") if LOST_THREAD_TEXT in json.dumps(r)], (
        "T-0179 B-6: the clip's text is never a tts.utterance (astra.spoken) record"
    )
    assert "ZQX" not in history and not any("ZQX" in s for s in speaks)
    assert_p4(rig)


async def test_second_trip_with_asset_installed(tmp_path: Path) -> None:
    """B-6 / B-7 amendment (``scripts/render_clips.py`` renders into the data dir): with
    ``<data_dir>/clips/lost_thread.s16le`` present the clip carries those bytes; the turn line records the clip
    either way."""
    # 100 ms of 24 kHz mono s16le, by hand: 2,400 samples x 2 bytes = 4,800 bytes (each 4-byte unit is 2 samples)
    pcm = b"\x10\x00\xf0\xff" * 1200
    async with retry_rig(tmp_path, script=t42.STEP1) as rig:
        path = Path(rig.session.config.data_dir) / "clips" / f"{LOST_THREAD}.s16le"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pcm)
        trip = [thought(TRIP_THOUGHT, 0, open_=True), tool(CLICK_X)]
        await run(rig, trip, [thought(TRIP_THOUGHT_2, 0, open_=True)], [answer(S3)])
        extras = dict(rig.record("t1").extras)
        lost = clips(rig, LOST_THREAD)
    ids = engine_ids(rig, "t1-gen-s2")
    assert len(ids) == 2, f"T-0179 B-4: exactly 2 engine requests for t1-gen-s2; got {ids}"
    assert len(lost) == 1, lost
    assert getattr(lost[0], "duration_ms", None) == 100, getattr(lost[0], "duration_ms", None)
    assert extras.get(CLIP_KEY) == LOST_THREAD, extras


async def test_report_step_trip_plays_lost_thread_only(tmp_path: Path, monkeypatch: Any) -> None:
    """B-7: at ``max_tool_steps=2`` the post-budget report request (step 3) trips, is retried once, and the retry trips
    too -> the turn ends ``degenerate`` with ``lost_thread`` and never ``out_of_steps``: one ending clip."""
    t42.degenerate_message()
    monkeypatch.setattr(tsl, "LOOP", tsl.loop_policy(2))
    async with retry_rig(tmp_path, script=t42.STEP1) as rig:
        await run(
            rig,
            [thought("Step two plan.", 0), tool(tsl.CLICK2)],
            [thought(TRIP_THOUGHT, 0, open_=True), answer("I will click the button.", 1)],
            [thought(TRIP_THOUGHT_2, 0, open_=True), answer("I will click the button.", 1)],
        )
        extras = dict(rig.record("t1").extras)
    reqs = requests_for(rig, "t1-gen-s3")
    assert reqs and all(r.get("report_fact") is True for r in reqs), (
        "premise: step 3 is the report step"
    )
    assert len(engine_ids(rig, "t1-gen-s3")) == 2, engine_ids(rig, "t1-gen-s3")
    assert ending_clip_kinds(rig) == [LOST_THREAD], (
        f"T-0179 B-7: lost_thread only, never with out_of_steps; {ending_clip_kinds(rig)}"
    )
    assert extras.get("tool_loop_end") == "degenerate", extras.get("tool_loop_end")


# =====================================================================================================
# B-1 / B-5: abort and barge-in win; the shared timer; the T-0171 resend
# =====================================================================================================


@pytest.mark.parametrize("phase", ["generate", "tool_step"])
async def test_abort_or_barge_in_wins_before_the_resend(tmp_path: Path, phase: str) -> None:
    """B-1 last bullet (adversarial): a barge-in lands and its abort has fully run (``Session.abort_request`` of the
    machine id) when the session aborts the tripped attempt -> nothing is sent: one engine request, one
    ``llm.request`` record, no retry note, no clip. Staged as T-0171 criterion 9 stages it. Active: holds at BASE."""
    mid = "t1-gen" if phase == "generate" else "t1-gen-s2"
    trip = [thought(TRIP_THOUGHT, 0, open_=True), tool(CLICK_X)]
    async with retry_rig(tmp_path, script=t42.STEP1) as rig:
        fired: list[str] = []

        async def barge(rid: str) -> None:
            if fired or not rid.endswith(f"-{mid}-{rid.rsplit('-', 1)[-1]}"):
                return
            fired.append(rid)
            now = rig.clock.now_ms
            rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
            await rig.session.abort_request(mid)

        rig.llm.abort_hooks.append(barge)
        if phase == "generate":
            rig.llm.script_generates(trip, [answer(S1), tool(VALID_CLICK)])
            rig.llm.script_steps([answer(S3)])
        else:
            rig.llm.script_steps(trip, [answer(S3)])
        await rig.ask(tsl.QUESTION)
        await rig.until(lambda: bool(fired), "premise: the tripped attempt was aborted")
        await rig.advance(3_000)
        extras = dict(rig.record("t1").extras) if rig.lines() else {}
    assert len(engine_ids(rig, mid)) == 1, (
        f"T-0179 B-1: no resend after the abort; {engine_ids(rig, mid)}"
    )
    assert len(requests_for(rig, mid)) == 1, requests_for(rig, mid)
    assert_no_clip_or_retry(rig, extras)


@pytest.mark.parametrize("phase", ["generate", "tool_step"])
async def test_abort_or_barge_in_wins_during_the_resend(tmp_path: Path, phase: str) -> None:
    """B-1 / B-5 (and reviewer mutation 7 in its ``release(machine id)``-after-issue form, which forgets the retry's
    id): a barge-in while the retry is live aborts the retry's own engine id; no third request, no ``llm.done``, no
    clip, nothing of the retry voiced."""
    mid = "t1-gen" if phase == "generate" else "t1-gen-s2"
    trip = [thought(TRIP_THOUGHT, 0, open_=True), tool(CLICK_X)]
    clean = [
        thought(CLEAN_THOUGHT, 0),
        answer("ZQX-RETRY spoken words.", 1),
        tool(CLICK4, canvas_idx=1),
    ]
    async with retry_rig(tmp_path, script=t42.STEP1) as rig:
        held: list[str] = []

        async def hold(rid: str) -> None:
            # only the machine id's own tripped attempt (the turn's prefill is aborted first, under its own id)
            if held or not rid.endswith(f"-{mid}-{rid.rsplit('-', 1)[-1]}"):
                return
            held.append(rid)
            if phase == "generate":
                rig.llm.hold()
            else:
                rig.llm.hold_steps()

        rig.llm.abort_hooks.append(hold)
        if phase == "generate":
            rig.llm.script_generates(trip, clean)
            rig.llm.script_steps([answer(S3)])
        else:
            rig.llm.script_steps(trip, clean, [answer(S3)])
        await rig.ask(tsl.QUESTION)
        await rig.until(
            lambda: len(engine_ids(rig, mid)) >= 2,
            f"T-0179 B-2: no resend after the trip; sent {engine_ids(rig, mid)}",
            limit_ms=5_000,
        )
        failed, retry = engine_ids(rig, mid)
        assert rig.state.state is TurnState.THINKING, "premise: the retry is live"
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.until(
            lambda: retry in rig.llm.aborted,
            f"T-0179 B-1: the abort targets the retry {retry}; aborted {rig.llm.aborted}",
            limit_ms=2_000,
        )
        rig.llm.release()
        rig.llm.release_steps()
        await rig.advance(3_000)
        ids = engine_ids(rig, mid)
        speech = spoken(rig)
    assert ids == [failed, retry], f"T-0179 B-5: no third request; got {ids}"
    assert rig.dones(mid) == [], rig.dones(mid)
    assert clips(rig, LOST_THREAD) == [], "T-0179 B-6: a barge-in end plays no clip"
    assert "ZQX-RETRY" not in speech, speech


def _arm_of(rig: Any, request_id: str) -> int:
    """The ``llm_timeout`` deadline the machine armed in the tuple that issued ``request_id``."""
    for _ev, _st, cmds in rig.machine.log:
        if any(
            getattr(c, "request_id", None) == request_id
            for c in cmds
            if isinstance(c, C.LlmContinue)
        ):
            arms = [c.at_ms for c in cmds if isinstance(c, C.ArmTimer) and c.name == "llm_timeout"]
            assert len(arms) == 1, cmds
            return int(arms[0])
    raise AssertionError(f"premise: no LlmContinue for {request_id}")


async def test_shared_llm_timeout(tmp_path: Path) -> None:
    """B-5 (adversarial; reviewer mutation 3, a re-armed or extended timer): the step-2 request trips 1 s before the
    machine's ``llm_timeout`` deadline and its retry hangs -> the turn ends through the existing timeout row
    (``hud.error("llm timeout")``) by that deadline (plus one tick), with no clip and no third request; the machine
    armed ``llm_timeout`` once for the step."""
    trip = [thought(TRIP_THOUGHT, 0, open_=True), tool(CLICK_X)]
    async with retry_rig(tmp_path, script=t42.STEP1) as rig:
        rig.llm.script_steps(trip, [answer(S3)])
        rig.llm.hold_steps()
        held: list[str] = []

        async def hold(rid: str) -> None:
            if not held:
                held.append(rid)
                rig.llm.hold_steps()

        await rig.ask(tsl.QUESTION)
        await rig.until(
            lambda: len(engine_ids(rig, "t1-gen-s2")) >= 1, "premise: step 2 was requested"
        )
        deadline = _arm_of(rig, "t1-gen-s2")
        assert deadline - rig.clock.now_ms > 1_500, "premise: room before the deadline"
        await rig.advance(deadline - 1_000 - rig.clock.now_ms)
        rig.llm.abort_hooks.append(hold)
        rig.llm.release_steps()
        await rig.until(
            lambda: len(engine_ids(rig, "t1-gen-s2")) >= 2,
            "T-0179 B-2: no resend after the trip",
            limit_ms=500,
        )
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE,
            "T-0179 B-5: the turn outlived the shared llm_timeout",
            limit_ms=1_200,
        )
        ended_at = rig.clock.now_ms
        await rig.advance(3_000)
        rig.llm.release_steps()
        await rig.advance(1_000)
        hud = [c.message for c in rig.commands(C.HudError)]
        arms = [
            c
            for _e, _s, cmds in rig.machine.log
            for c in cmds
            if isinstance(c, C.ArmTimer) and c.name == "llm_timeout"
        ]
    assert ended_at <= deadline + 200, f"T-0179 B-5: ended at {ended_at}, deadline {deadline}"
    assert hud == ["llm timeout"], f"T-0179 B-5: the existing timeout row; hud.error {hud}"
    assert len(engine_ids(rig, "t1-gen-s2")) == 2, engine_ids(rig, "t1-gen-s2")
    assert clips(rig, LOST_THREAD) == [], "T-0179 B-5: no clip from this task on a timeout end"
    assert len(arms) == 2, f"llm_timeout armed once at step 1 and once at step 2; got {arms}"


@pytest.mark.parametrize("phase", ["tool_step", "generate"])
@pytest.mark.parametrize("order", ["empty-then-trip", "trip-then-empty"])
async def test_with_t0171(tmp_path: Path, order: str, phase: str) -> None:
    """B-5 (fresh accumulators, B-2): the T-0171 resend and the degenerate retry are independent: an empty retryable
    response and a trip on the same request, in either order, give exactly 3 engine requests and a clean turn with
    ``llm_empty_retries: 1`` and ``degenerate_retries: 1``. ``trip-then-empty`` fails if the retry reuses the tripped
    attempt's ``delivered`` count (T-0171 resends only with nothing delivered): on a tool step, and (B-13b, mutants
    M11a/M8g) on the step-1 generate."""
    if phase == "tool_step":
        mid = "t1-gen-s2"
        trip = [thought(TRIP_THOUGHT, 0, open_=True), tool(CLICK_X)]
    else:
        mid = "t1-gen"
        trip = [
            thought(TRIP_THOUGHT, 0, open_=True),
            answer(S1, 1),
            tool(VALID_CLICK, canvas_idx=1),
        ]
        clean = [thought(CLEAN_THOUGHT, 0), answer(S1, 1), tool(VALID_CLICK, canvas_idx=1)]
    errors_in = (
        (engine_empty_error, None, None, None)
        if order == "empty-then-trip"
        else (None, engine_empty_error, None, None)
    )
    async with retry_rig(tmp_path, script=t42.STEP1) as rig:
        if phase == "tool_step":
            rig.llm.fail_steps(*errors_in)
            await run(rig, trip, [answer(S3)])
        else:
            rig.llm.fail_generates(*errors_in)
            rig.llm.script_generates(trip, clean)
            await run(rig, [answer(S3)])
        extras = dict(rig.record("t1").extras)
        speech = spoken(rig)
    ids = engine_ids(rig, mid)
    assert len(ids) == 3 and len(set(ids)) == 3, f"T-0179 B-5: 3 engine requests; got {ids}"
    assert extras.get("llm_empty_retries") == 1 and extras.get(EXTRA) == 1, extras
    assert extras.get("tool_loop_end") == "final" and speech == S1 + S3, (extras, speech)
    assert len([d for d in rig.kind("note") if d.get("what") == EMPTY_NOTE]) == 1
    assert len(retry_notes(rig)) == 1
    assert_p4(rig)


async def test_at_most_three_requests_per_step(tmp_path: Path) -> None:
    """B-5 bound: empty, then a trip, then the retry ends empty too -> the T-0171 resend is spent, the turn ends on
    the error path (an ``LlmError`` naming the third attempt), never a fourth request and no clip (not the B-4 end)."""
    trip = [thought(TRIP_THOUGHT, 0, open_=True), tool(CLICK_X)]
    async with retry_rig(tmp_path, script=t42.STEP1) as rig:
        rig.llm.fail_steps(engine_empty_error, None, engine_empty_error, None)
        await run(rig, trip, [answer(S3)])
        extras = dict(rig.record("t1").extras)
    ids = engine_ids(rig, "t1-gen-s2")
    assert len(ids) == 3, f"T-0179 B-5: at most 3 engine requests per step; got {ids}"
    errors = llm_errors(rig, "t1-gen-s2")
    assert len(errors) == 1 and ids[2] in errors[0].message, errors
    assert clips(rig, LOST_THREAD) == [] and CLIP_KEY not in extras, extras
    assert extras.get(EXTRA) == 1, extras


# =====================================================================================================
# B-1 out of scope (active: holds at BASE)
# =====================================================================================================


async def test_out_of_scope_pass2(tmp_path: Path) -> None:
    """B-1 out of scope: a ReGround pass-2 thought collapse is never retried (one pass-2 request, outcome
    ``pass2_timeout``, no retry note)."""
    collapse = [thought(t42.COLLAPSE, 0, open_=True), tool(click(900, 100), canvas_idx=1)]
    rig, extras = await t42._pass2_turn(tmp_path, True, collapse)
    reground = [d for d in rig.kind("llm.request") if d.get("phase") == "reground"]
    assert len(reground) == 1, reground
    assert [r["outcome"] for r in t42.regrounds(rig)] == ["pass2_timeout"]
    assert_no_clip_or_retry(rig, extras)


async def test_out_of_scope_repair(tmp_path: Path) -> None:
    """B-1 out of scope: a ``tool_repair`` whose thought collapses is never resent (one repair request)."""
    async with retry_rig(
        tmp_path, mode="dry_run", script=[answer(S1), tool(REPAIRABLE_CLICK)]
    ) as rig:
        rig.llm.script_continuation(thought(t42.COLLAPSE, 0, open_=True), tool(REPAIRED_CLICK))
        await run(rig, [answer(S3)])
        extras = dict(rig.record("t1").extras)
    repairs = [rid for rid, *_ in rig.llm.continuations() if "-t1-repair-" in rid]
    assert len(repairs) == 1, f"one repair request; got {repairs}"
    assert_no_clip_or_retry(rig, extras)


async def test_out_of_scope_reask(tmp_path: Path) -> None:
    """B-1 out of scope: the T-0172 ``announce_reask`` request keeps T-0172 B5 (a trip in it ends the turn, never
    retried): one re-ask, no retry note, no clip."""
    from tests.unit import test_t0172_session_reask as r72

    async with r72.reask_rig(tmp_path, script=r72.STEP1, extra={GUARD: True}) as rig:
        await r72.run(
            rig, [answer(r72.ANN_CLICK)], reasks=[[thought(t42.COLLAPSE), tool(tsl.CLICK2)]]
        )
        extras = dict(rig.record("t1").extras)
    assert len(rig.llm.reask_prompts) == 1, rig.llm.reask_prompts
    assert extras.get("tool_loop_end") == "degenerate", extras.get("tool_loop_end")
    assert_no_clip_or_retry(rig, extras)


async def test_out_of_scope_captioner(tmp_path: Path) -> None:
    """B-1 out of scope: a collapsing caption is never resent (one caption request) and nothing is stored."""
    rig, llm, stored = await t42._caption_run(
        tmp_path, True, [thought(t42.COLLAPSE, 0, open_=True)]
    )
    captions = [rid for rid, prompt, _ in llm.generates if is_caption_prompt(prompt)]
    assert len(captions) == 1 and stored == [], (captions, stored)
    assert retry_notes(rig) == []


# =====================================================================================================
# B-8: never voiced; the denoising view stays real
# =====================================================================================================


@pytest.mark.parametrize("mode", ["live", "off"])
async def test_never_voiced(tmp_path: Path, mode: str) -> None:
    """B-8: an engine that keeps serving the aborted stream (late answer, late call) -> sentinels of the tripped
    attempt (its tripping thought, its late answer) reach no ``TtsSpeak.source``, history, HUD message or
    ``machine.events()`` text, and its late call never executes; the retry's answer is voiced (red at BASE).

    Narrowed by T-0180 (B5, B-R5): the tripping thought commit may appear on the HUD only as the ``text`` of an
    ``astra.thought``; the late answer and the trip answer (``ZQX-LATE``, ``ZQX-TRIP-ANSWER``) stay absent
    everywhere, that text included. Anti-vacuity: the tripping thought **is** in a block that ends
    ``stopped_repeating``."""
    trip = [
        thought(TRIP_THOUGHT, 0, open_=True),
        answer(LATE, 1),
        tool(CLICK_X, canvas_idx=1),
    ]
    async with retry_rig(tmp_path, mode=mode, late=True, script=t42.STEP1) as rig:
        if mode == "live":
            await run(rig, trip, [answer(FINAL)])
            done = executed(rig)
        else:
            rig.llm.script_generates(
                [thought(TRIP_THOUGHT, 0, open_=True), answer(LATE, 1)], [answer(ANSWER)]
            )
            await run_off(rig)
            done = []
        speech = spoken(rig)
        hist = t42.history_text(rig)
        # T-0180 B-R5: the HUD searched is every message but astra.thought's ``text`` field
        split = split_thoughts(rig.published.messages)
        hud = split.elsewhere
        texts = json.dumps([getattr(e, "text", "") for e in rig.machine.events()])
    mid = "t1-gen-s2" if mode == "live" else "t1-gen"
    assert len(engine_ids(rig, mid)) == 2, f"T-0179 B-2: one resend; got {engine_ids(rig, mid)}"
    assert (FINAL if mode == "live" else ANSWER) in speech, (
        f"T-0179 B-3: the retry is voiced; {speech!r}"
    )
    for sentinel in ("ZQX-TRIP", "ZQX-LATE"):
        assert sentinel not in speech, f"T-0179 B-8: {sentinel} voiced: {speech!r}"
        assert sentinel not in hist, f"T-0179 B-8: {sentinel} in history"
        assert sentinel not in hud, f"T-0179 B-8: {sentinel} on the HUD"
    for text in split.texts():
        for sentinel in ("ZQX-LATE", "ZQX-TRIP-ANSWER"):
            assert sentinel not in text, (
                f"T-0179 B-8 (narrowed by T-0180): {sentinel} in astra.thought text"
            )
        assert "ZQX-TRIP" not in text.replace("ZQX-TRIP-THOUGHT", ""), (
            f"T-0179 B-8 (narrowed by T-0180): only the tripping thought may be shown; {text!r}"
        )
    tripped = [text for _, _, end, text in split.thoughts if end == "stopped_repeating"]
    assert any(TRIP_THOUGHT in text for text in tripped), (
        "T-0180 B5 anti-vacuity (B-R5): the tripping thought commit is in a block ending stopped_repeating; "
        f"published thoughts {split.thoughts}"
    )
    assert "ZQX-LATE" not in texts, "T-0179 B-8: the late answer reached the machine"
    assert "the the the the the the the the" not in texts, "the tripping commit reached the machine"
    if mode == "live":
        assert done == [("t1-call", tsl.CLICK_ARGS)], f"the tripped call never executes; {done}"


async def test_canvas_attempt_ids(tmp_path: Path) -> None:
    """B-8: every denoising-view canvas carries its real attempt id (the tripped attempt's own canvases are real and
    carry its id; the retry's carry the retry's), and no ``astra.canvas.synthetic`` is added. Through a real
    ``CanvasRelay`` fed by T-0171's ``TapLlm``."""
    from local_astra.conductor.canvas_relay import CanvasRelay
    from tests.fakes.canvas_tap import StubPieces, optional_piece_kwargs
    from tests.unit.test_session_engine_empty_retry import TapLlm

    llm = TapLlm()
    async with retry_rig(tmp_path, mode="off", llm=llm) as rig:
        llm.relay = CanvasRelay(
            publish=rig.published,
            clock=rig.clock,
            state=rig.session.turn_state_message,
            **optional_piece_kwargs(CanvasRelay, StubPieces()),
        )
        llm.script_generates(
            [thought(TRIP_THOUGHT, 0, open_=True), answer(LATE, 1)],
            [thought("Reading the page.", 0), answer(ANSWER, 1)],
        )
        await run_off(rig)
        canvas = [
            m
            for m in rig.published.messages
            if str(getattr(m, "type", "")).startswith("astra.canvas.")
        ]
    ids = engine_ids(rig, "t1-gen")
    assert len(ids) == 2, f"T-0179 B-2: one resend; got {ids}"
    failed, retry = ids
    assert [m for m in canvas if m.type == "astra.canvas.synthetic"] == [], (
        "B-8: no synthetic canvas"
    )
    got = {getattr(m, "request_id", None) for m in canvas}
    assert got == {failed, retry}, (
        f"B-8: canvases carry the real attempt ids {failed}, {retry}; got {got}"
    )
    failed_idx = {getattr(m, "canvas_idx", None) for m in canvas if m.request_id == failed}
    assert failed_idx <= {0}, (
        f"B-8: the tripped attempt's canvases are only those it committed; {failed_idx}"
    )


# =====================================================================================================
# B-13 (amendment 2026-10-28, after reviewer A): fresh accumulators per attempt, and no retry after a card
# =====================================================================================================


@pytest.mark.parametrize("phase", ["tool_step", "generate"])
async def test_tripped_attempts_call_never_executes(tmp_path: Path, phase: str) -> None:
    """B-13b / B-2 "fresh accumulators" (safety; mutant M9, the tripped attempt's kept ``tool`` commit survives into
    the retry), on both paths (reviewers A and B): the tripped attempt commits a call, *then* its thought collapses;
    the clean retry makes no call and answers. The tripped call is never judged or executed: no ``tool.call`` record
    names it, the retry's answer is voiced and the loop ends ``final``. Tool step: only step 1's click runs. Step-1
    generate: nothing runs."""
    trip = [tool(CLICK_X, canvas_idx=0), thought(TRIP_THOUGHT, 1, open_=True)]
    async with retry_rig(tmp_path, script=t42.STEP1) as rig:
        if phase == "tool_step":
            mid, call, want_done, want_said = (
                "t1-gen-s2",
                "t1-call-s2",
                [("t1-call", tsl.CLICK_ARGS)],
                S1 + FINAL,
            )
            await run(rig, trip, [answer(FINAL)], [answer(S3)])
        else:
            mid, call, want_done, want_said = "t1-gen", "t1-call", [], FINAL
            rig.llm.script_generates(trip, [answer(FINAL)])
            await run(rig, [answer(S3)])
        done = executed(rig)
        extras = dict(rig.record("t1").extras)
        speech = spoken(rig)
        records = [r for r in rig.tool_records() if r.get("call_id") == call]
    assert len(engine_ids(rig, mid)) == 2, engine_ids(rig, mid)
    assert extras.get(EXTRA) == 1, extras
    assert done == want_done, f"T-0179 B-13b (M9): the tripped attempt's call executed; {done}"
    assert records == [], f"T-0179 B-13b: the tripped attempt's call was judged; {records}"
    assert speech == want_said, speech
    assert extras.get("tool_loop_end") == "final", extras.get("tool_loop_end")


#: an answer commit that opens a thought block in the answer channel: it cleans to nothing (not submitted, so it does
#: not count as answer text reaching the machine), and a cleaner carried into the retry would withhold the retry's text
OPEN_THOUGHT_ANSWER = "<|channel>thought ZQX hidden"
#: an answer commit ending in a partial special-token literal: the literal cleaner holds it back (it could still become
#: ``<eos>``), so it is never submitted; a literal cleaner carried into the retry would prepend it to the retry's answer
#: (mutant M3c: the tools-off generate retry keeps the tripped attempt's ``literal = _LiteralCleaner()``)
LITERAL_TAIL = "<eos"


@pytest.mark.parametrize("mode", ["off", "live", "literal"], ids=["generate", "step", "literal"])
async def test_retry_answer_voiced_after_a_withheld_answer(tmp_path: Path, mode: str) -> None:
    """B-13b (mutants M11c, generate, and M11d, step: the tripped attempt's answer cleaner survives into the retry;
    M3c, literal: the tools-off generate retry keeps the tripped attempt's literal cleaner): the tripped attempt's
    first answer commit is withheld (an open thought block, or a held-back ``<eos`` tail; never submitted), then its
    answer collapses; the retry's clean answer is voiced, and nothing of the tripped attempt is."""
    trip = [answer(OPEN_THOUGHT_ANSWER, 0), answer(TRIP_ANSWER, 1)]
    marker = "ZQX"
    if mode == "literal":
        marker = LITERAL_TAIL
        async with retry_rig(tmp_path, mode="off", script=[answer("unused")]) as rig:
            rig.llm.script_generates(
                [answer(LITERAL_TAIL, 0), answer(TRIP_ANSWER, 1)], [answer(ANSWER, 0)]
            )
            await run_off(rig)
            extras = dict(rig.record("t1").extras)
            speech = spoken(rig)
        mid, want = "t1-gen", ANSWER
    elif mode == "off":
        async with retry_rig(tmp_path, mode="off", script=[answer("unused")]) as rig:
            rig.llm.script_generates(
                trip, [thought("Let me look at the page.", 0), answer(ANSWER, 1)]
            )
            await run_off(rig)
            extras = dict(rig.record("t1").extras)
            speech = spoken(rig)
        mid, want = "t1-gen", ANSWER
    else:
        async with retry_rig(tmp_path, script=t42.STEP1) as rig:
            await run(rig, trip, [thought(CLEAN_THOUGHT, 0), answer(FINAL, 1)])
            extras = dict(rig.record("t1").extras)
            speech = spoken(rig)
        mid, want = "t1-gen-s2", FINAL
    submitted = [
        e
        for e in rig.machine.events(E.LlmCommit)
        if e.request_id == mid and e.segment == "answer" and marker in e.text
    ]
    # both attempts share the machine id: this is the premise (the tripped attempt's commit was withheld) and, for the
    # retry's commits, the property (M11c/M11d/M3c: a carried cleaner submits the withheld text with the retry's)
    assert submitted == [], (
        f"T-0179 B-13b: the withheld answer text ({marker!r}) reached the machine; {submitted}"
    )
    assert len(engine_ids(rig, mid)) == 2, engine_ids(rig, mid)
    assert extras.get(EXTRA) == 1, extras
    assert want in speech, f"T-0179 B-13b (M11c/M11d): the retry's answer is voiced; {speech!r}"
    assert "ZQX" not in speech and "the the the" not in speech, speech
    if mode == "literal":
        assert speech == ANSWER, (
            f"T-0179 B-13b (M3c): the tripped attempt's held-back {LITERAL_TAIL!r} reached the retry; {speech!r}"
        )


@pytest.mark.parametrize("phase", ["generate", "tool_step"])
async def test_no_retry_after_a_published_card(tmp_path: Path, phase: str) -> None:
    """B-13a (new behaviour, amendment 2026-10-28), at both retry sites: a ``ui`` surface of the attempt was submitted
    (and gated and published to the HUD), then its thought collapses -> something reached the user, so no retry: one
    engine request, no retry note, today's ``DEGENERATE_MESSAGE``, no ``degenerate_retries`` key and no clip (the BASE
    ending). Generate: tools off, step 1. Tool step (mutant: the ``tool_step`` predicate counts submitted answers only):
    live, step 2; the loop ends ``degenerate`` and the tripped attempt's call never executes."""
    from tests.unit._hud_rig import ui

    if phase == "generate":
        mid = "t1-gen"
        async with retry_rig(tmp_path, mode="off", script=[answer("unused")]) as rig:
            rig.llm.script_generates(
                [ui(0), thought(TRIP_THOUGHT, 1, open_=True), answer(LATE, 2)],
                [answer("ZQX-RETRY a second answer.", 0)],
            )
            await run_off(rig)
            extras = dict(rig.record("t1").extras)
            published = [str(getattr(m, "type", "")) for m in rig.published.messages]
            speech = spoken(rig)
            done: list[tuple[str, dict[str, Any]]] = []
    else:
        mid = "t1-gen-s2"
        async with retry_rig(tmp_path, script=t42.STEP1) as rig:
            await run(
                rig,
                [ui(0), thought(TRIP_THOUGHT, 1, open_=True), tool(CLICK_X, canvas_idx=1)],
                [answer(FINAL)],
            )
            extras = dict(rig.record("t1").extras)
            published = [str(getattr(m, "type", "")) for m in rig.published.messages]
            speech = spoken(rig)
            done = executed(rig)
    surfaces = [t for t in published if t.startswith("astra.ui") and t != "astra.ui.fence"]
    assert surfaces, (
        f"premise: the attempt's card was published to the HUD; {sorted(set(published))}"
    )
    assert len(engine_ids(rig, mid)) == 1, (
        f"T-0179 B-13a: no retry once a card was submitted; {engine_ids(rig, mid)}"
    )
    assert [e.message for e in llm_errors(rig, mid)] == [t42.degenerate_message()]
    assert_no_clip_or_retry(rig, extras)
    assert "ZQX-RETRY" not in speech, speech
    if phase == "tool_step":
        assert extras.get("tool_loop_end") == "degenerate", extras.get("tool_loop_end")
        assert done == [("t1-call", tsl.CLICK_ARGS)], (
            f"T-0179 B-13a: only step 1's call executes (no retry, the tripped call never runs); {done}"
        )
        assert FINAL not in speech, f"the BASE ending: no step 3; {speech!r}"
