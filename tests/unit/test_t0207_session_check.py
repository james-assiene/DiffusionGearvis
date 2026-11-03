"""T-0207 group A2: the completion check in the session, behind ``Session(completion_check: bool = False)``
(docs/tasks/T-0207.md A2.1-A2.13).

When a live turn that executed at least one browser action ends a ``tool_step`` request with a non-empty answer and no
call, the step's commits are held; one fresh ``frame()`` of the agent browser is taken; the step is asked once more
with ``completion_check.check_prompt(<the step prompt as sent>, <step text>, <fresh frame>, report=...)``; only the
check's commits reach the machine. A failure (request or frame) falls back to the draft exactly as BASE; an abort
voices nothing.

**Rigs.** T-0134's live rig (``_live_browser_support.feedback_rig``: loop on, one ``FakeLiveBrowser`` as executor and
live browser) built with ``Session(..., completion_check=...)`` only when the row gives it (``ABSENT``: not passed),
T-0177's ``vision_rig`` for the shared screen, and ``_tool_turn_support.tool_rig`` for dry-run and tools-off. Every
frame is a real JPEG (``jpeg_frame``), so a session that applies T-0200's ``blank_frame`` to the fresh capture can
decode it; ``BLANK`` is a uniform white JPEG.

**The fake (named at activation).** ``tests/fakes/llm.py`` gained, additively, a completion-check route told apart
structurally by ``is_check_prompt`` (last message a user turn holding exactly ``(ImagePart, TextPart(<a check
instruction>))`` after an assistant message): ``script_checks``, ``fail_checks``, ``fail_checks_after_script``,
``hold_checks``/``release_checks``, ``check_prompts``, ``check_bodies`` (built by the real
``VllmClient.request_body``) and ``check_overruns``. Before it a check prompt was a "turn generate" to the fake: it
replayed ``script`` and could not be held in flight, which the barge-in and timeout rows need. No real-service
contract row: the route only chooses which scripted stream answers; the wire body is the real client's.

**Oracles.** The expected check prompt is ``check_prompt`` (A1, merged) over the step prompt the fake received, with
the fresh frame keyed by its capture identity typed by hand (``"capture:" + sha256(jpeg)[:32]``, T-0111g B7). The
"outcome follows the existing rows" oracle is the keyword-absent run of the same turn in which the *step itself*
committed the check's stream (``as_step``): the voiced text, the executed calls and the next step's request must
match it. BASE equality is a subprocess run over ``git archive`` of BASE, origin-proved (the T-0201 B1 pattern).

**Readings where the task is silent** (each in the test-author's report):

- R1 ``fresh_frame.phash`` (A2.4): the fresh frame's executor phash or its capture identity (either is accepted);
  ``same_as_last`` is pinned only where both readings agree (identical bytes and phash: ``True``; different bytes and
  phash: ``False``).
- R2 a frame that fails, times out, is closed or blank (A2.3/A2.6): the outcome note is ``no_frame``; whether a
  ``completion check failed`` note is written too is not pinned (A2.3 and A2.6 read differently). The ``asked`` note
  is written only once a frame was taken (its ``same_as_last`` needs one), so a ``no_frame`` turn has none.
- R3 the turn line with the keyword ``True`` and no check: ``{on: true, asked: 0, outcome: null}`` (T-0172 B7's
  precedent); ``asked`` is ``1`` once the check request was sent.
- R4 a T-0142 trip inside the check: outcome ``failed`` (T-0172 B-F4's precedent: a trip is a failed re-ask).
- R5 an abort during the frame: no check request is sent at all; an outcome note, if any, is ``aborted``.
- R6 A2.9 "a call carrying the step's older frame identity, when the fresh frame differs": the check's click is read
  off the fresh frame, and the browser's newest own frame then changes before it executes (``land``, a capture by
  someone else): the click is refused with T-0111g B9's reason and never dispatched. A check-made call is not exempt
  from the stale rule.
- R7 "over-budget prompt" (A2.6) is the engine refusing the check as over-long (HTTP 413, T-0172 R-c).
- R8 "unchanged except ``t_ms``" (A2.2): the armed-no-check rows compare every machine event with ``t_ms`` removed
  from ``LlmCommit`` events only, and every command list exactly.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word, tools ``live`` (1280x800
frames, ReGround off, no live feedback unless named), loop on (8 steps unless a row sets 2), n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import subprocess
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_astra.conductor import session as session_mod
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit, ImagePart, Prompt
from local_astra.contracts.perception import Frame
from local_astra.contracts.turn import TurnState
from local_astra.telemetry import session_log
from local_astra.toolcall import router
from local_astra.toolcall.completion_check import (
    CHECK_INSTRUCTION,
    CHECK_INSTRUCTION_REPORT,
    check_prompt,
)
from tests import _git_range as gr
from tests.fakes.llm import engine_empty_error, http_error, is_check_prompt
from tests.unit import test_session_tool_loop as tsl
from tests.unit import test_t0135_live_recovery as t0135
from tests.unit import test_t0142_session as t0142
from tests.unit._live_browser_support import (
    FRESHNESS,
    FakeLiveBrowser,
    feedback_rig,
    jpeg_frame,
    scene_frame,
)
from tests.unit._tool_turn_support import NAMELESS, VALID_CLICK, answer, thought, tool
from tests.unit.test_session_live import _wire
from tests.unit.test_session_loop import POLICY
from tests.unit.test_t0177_shared_frame import VisionBrowser, select, vision_rig

REPO = Path(__file__).resolve().parents[2]
#: BASE: main at 5a0602f2 (``T-0207 A1 merged``), where group A2 branched. A literal, never a moving ref (tests.md).
BASE_SHA = "5a0602f2454526bba06db59aa0d200c6ea811853"
#: The implementation tip: af988fcb (A2 impl 3281676c + R3 b0d2dfb1 + N1 af988fcb on T-0207-a2-session; BASE and
#: 3281676c are its ancestors). Re-pinned at merge to the merge's second parent if later implementation commits land.
TIP_IMPL: str | None = "af988fcbd3247d9513ca5d04e1513cace30c2606"

KW = "completion_check"
PHASE = "completion_check"
TURN_KEY = "completion_check"
NOTE_ASKED = "completion check asked"
NOTE_OUTCOME = "completion check outcome"
NOTE_FAILED = "completion check failed"
ASKED_KEYS = frozenset({"what", "request_id", "step", "chars", "same_as_last", "report"})
OUTCOME_KEYS = frozenset({"what", "request_id", "step", "outcome"})
OUTCOMES = frozenset({"call", "no_call", "refused", "failed", "aborted", "no_frame"})
REQUEST_KEYS = frozenset(
    {"phase", "step", "request_id", "attempt_id", "options", "prompt", "fresh_frame"}
)
RETRY_NOTE = "the engine returned an empty retryable response; retrying once"
ABSENT: Any = object()

Q = tsl.Q
S1, S2, S3 = tsl.S1, tsl.S2, tsl.S3
CLICK2, CLICK2_ARGS = tsl.CLICK2, tsl.CLICK2_ARGS
CLICK3 = tsl.CLICK3
CLICK_ARGS = tsl.CLICK_ARGS
SCROLL = f"<|tool_call>call:scroll{{direction:{Q}down{Q}}}<tool_call|>"
SEARCH_CALL = tsl.SEARCH_CALL
QUESTION = "go on Amazon and buy volume 7 of Vinland Saga"

#: the two false claims of James's session (docs/tasks/T-0207.md Evidence, seq 2532 and 5593), verbatim
CLAIM = "I have added volume 7 of Vinland Saga to your cart."
CLAIM_5538 = "I have purchased The Legend of Zelda: Tears of the Kingdom for you."
REPORT_DRAFT = "I opened the checkout page and placed your order."
DRAFT_THOUGHT = "ZQX-T0207-DRAFT-THOUGHT the add to cart click worked"
CHECK_THOUGHT = "ZQX-T0207-CHECK-THOUGHT the cart badge still shows zero"
#: a check's no-call answer, and its words before a call
CHECK_SAYS = "The cart still shows no items, so the book is not in your cart yet."
REPORT_SAYS = "This screen shows the basket with the book; I stopped before paying."
R_CALL = "Clicking it again."
ANN_CLICK = "I will click the Sign in button."
R_STOPPED = "The search limit was reached, so I could not search again."
REPORT = tsl.REPORT

F0 = jpeg_frame("t0207-onset")
F1 = jpeg_frame("t0207-after-1")
FR = jpeg_frame("t0207-fresh")
F2 = jpeg_frame("t0207-after-2")
F3 = jpeg_frame("t0207-after-3")
FX = jpeg_frame("t0207-landed")
#: T-0200's blank capture: one colour everywhere
BLANK = scene_frame("t0207-blank")
#: step 1 clicks (VALID_CLICK on F0, after-frame F1); step 2 claims, with no call
STEP1 = [answer(S1), tool(VALID_CLICK)]
CLAIM_STEP = [thought(DRAFT_THOUGHT), answer(CLAIM)]
NO_CALL_CHECK = [thought(CHECK_THOUGHT), answer(CHECK_SAYS)]
CALL_CHECK = [thought(CHECK_THOUGHT), answer(R_CALL), tool(CLICK2)]


# --------------------------------------------------------------------------------------------- the rig


def require_keyword() -> None:
    params = inspect.signature(session_mod.Session.__init__).parameters
    assert KW in params, (
        "T-0207 A2.1: Session(..., completion_check: bool = False) does not exist yet"
    )
    assert params[KW].default is False, (
        f"T-0207 A2.1: completion_check defaults to False; got {params[KW].default!r}"
    )


def identity(frame: Frame) -> str:
    """T-0111g B7's capture identity, typed by hand (never read off the session)."""
    return "capture:" + hashlib.sha256(frame.jpeg).hexdigest()[:32]


def held(frame: Frame) -> Frame:
    """The copy of an own frame the session carries in prompts: its ``phash`` is its capture identity."""
    return frame.model_copy(update={"phash": identity(frame)})


def browser_of(frames: Sequence[Frame] = (F0, FR), posts: Sequence[Frame] = (F1, F2, F3)) -> Any:
    return FakeLiveBrowser(list(frames), post_frames=list(posts))


@asynccontextmanager
async def live(
    tmp: Path,
    browser: Any = None,
    *,
    on: Any = True,
    script: Sequence[Commit] = STEP1,
    extra: dict[str, Any] | None = None,
    policy: Any = None,
    machine_cls: Any = None,
    kit_of: Callable[[Any], Any] | None = None,
) -> AsyncIterator[Any]:
    """The live session over ``browser`` with ``completion_check=on`` (``ABSENT``: not passed); ``policy``: the loop's
    ``TurnPolicy`` (default 8 steps); ``kit_of``: the kit built from the browser (default the live kit)."""
    kw: dict[str, Any] = {}
    if on is not ABSENT:
        require_keyword()
        kw[KW] = on
    kw.update(extra or {})
    saved = tsl.LOOP
    if policy is not None:
        tsl.LOOP = policy
    browser = browser if browser is not None else browser_of()
    try:
        async with feedback_rig(
            tmp,
            browser,
            script=list(script),
            session_kwargs=kw,
            machine_cls=machine_cls,
            kit=kit_of(browser) if kit_of is not None else None,
        ) as rig:
            yield rig
    finally:
        tsl.LOOP = saved


async def drive(
    rig: Any,
    *steps: Sequence[Commit],
    checks: Sequence[Sequence[Commit]] = (),
    reasks: Sequence[Sequence[Commit]] = (),
    question: str = QUESTION,
    limit_ms: int = 40_000,
) -> None:
    rig.llm.script_steps(*steps)
    rig.llm.script_checks(*checks)
    rig.llm.script_reasks(*reasks)
    await rig.ask(question)
    await rig.turn_done("T-0207: the turn never ended", limit_ms=limit_ms)
    await rig.advance(300)


async def claim_turn(
    tmp: Path,
    on: Any = True,
    *,
    steps: Sequence[Sequence[Commit]] = (CLAIM_STEP,),
    checks: Sequence[Sequence[Commit]] = (NO_CALL_CHECK,),
    reasks: Sequence[Sequence[Commit]] = (),
    browser: Any = None,
    script: Sequence[Commit] = STEP1,
    extra: dict[str, Any] | None = None,
    policy: Any = None,
    setup: Callable[[Any], None] | None = None,
    kit_of: Callable[[Any], Any] | None = None,
) -> Any:
    """One live turn: step 1 ``script`` (default: a click on F0), then ``steps``; checks and re-asks scripted."""
    tmp.mkdir(parents=True, exist_ok=True)
    async with live(
        tmp, browser, on=on, script=script, extra=extra, policy=policy, kit_of=kit_of
    ) as rig:
        if setup is not None:
            setup(rig)
        await drive(rig, *steps, checks=checks, reasks=reasks)
    return rig


# --------------------------------------------------------------------------------------------- reading the rig


def step_id(k: int, turn: str = "t1") -> str:
    return tsl.step_id(k, turn)


def call_id(k: int, turn: str = "t1") -> str:
    return tsl.call_id(k, turn)


def step_text(raw: str) -> str:
    """A2.1's step text: the answer channel with thoughts removed (``router.route(...).spoken_prefix``)."""
    return router.route(raw).spoken_prefix


def checks_of(rig: Any, machine_id: str) -> list[tuple[str, Prompt]]:
    pat = re.compile(rf"-{re.escape(machine_id)}-\d+$")
    return [(rid, p) for rid, p in rig.llm.check_prompts if pat.search(rid)]


def check_requests(rig: Any) -> list[dict[str, Any]]:
    return [d for d in rig.kind("llm.request") if d.get("phase") == PHASE]


def notes(rig: Any, what: str | None = None) -> list[dict[str, Any]]:
    return [d for d in rig.kind("note") if what is None or d.get("what") == what]


def outcomes(rig: Any) -> list[Any]:
    return [n.get("outcome") for n in notes(rig, NOTE_OUTCOME)]


def turn_key(rig: Any, turn: str = "t1") -> Any:
    return rig.record(turn).extras.get(TURN_KEY, ABSENT)


def invoked(rig: Any) -> list[str]:
    return [i.call_id for i in rig.browser.invocations]


def inv_of(rig: Any, cid: str) -> Any:
    found = [i for i in rig.browser.invocations if i.call_id == cid]
    assert len(found) == 1, (
        f"one invocation of {cid}; got {[i.call_id for i in rig.browser.invocations]}"
    )
    return found[0]


def results(rig: Any, cid: str) -> list[str]:
    return [e.status for e in rig.machine.events(E.ToolResult) if e.call_id == cid]


def frame_calls(rig: Any) -> int:
    return sum(1 for c in rig.browser.calls if c[0] == "frame")


def spoken_published(rig: Any) -> str:
    return json.dumps(
        [m.text for m in rig.published.messages if getattr(m, "type", None) == "astra.spoken"]
    )


def history_json(rig: Any) -> str:
    return json.dumps([h.model_dump(mode="json") for h in rig.state.history])


def memory_json(rig: Any) -> str:
    return json.dumps([e.model_dump(mode="json") for e in rig.memory.episodes.values()])


def record_json(rig: Any, turn: str = "t1") -> str:
    return json.dumps(rig.record(turn).model_dump(mode="json"))


def commits_for(rig: Any, machine_id: str, segment: str | None = None) -> list[E.LlmCommit]:
    return [
        e
        for e in rig.machine.events(E.LlmCommit)
        if e.request_id == machine_id and (segment is None or e.segment == segment)
    ]


def log_index(rig: Any, kind: str, pred: Callable[[dict[str, Any]], bool]) -> list[int]:
    return [
        i for i, r in enumerate(rig.log_records) if r["kind"] == kind and pred(r.get("data") or {})
    ]


def machine_seq(rig: Any) -> list[Any]:
    """R8: every machine event (``t_ms`` removed from ``LlmCommit`` only) and every command list, as JSON."""
    out: list[Any] = []
    for ev, state, cmds in rig.machine.log:
        dumped = ev.model_dump(mode="json")
        if isinstance(ev, E.LlmCommit):
            dumped.pop("t_ms", None)
        out.append([dumped, str(state), [c.model_dump(mode="json") for c in cmds]])
    sid = rig.session.session_id
    return json.loads(json.dumps(out).replace(sid, "<sid>"))


def assert_no_check(rig: Any, what: str) -> None:
    assert rig.llm.check_prompts == [], (
        f"T-0207 A2.1 ({what}): no completion check is sent; the fake saw {[r for r, _ in rig.llm.check_prompts]}"
    )
    assert check_requests(rig) == [], (
        f"T-0207 A2.4 ({what}): no llm.request with phase {PHASE!r}; got {check_requests(rig)}"
    )


def assert_one_check(rig: Any, machine_id: str, what: str) -> tuple[str, Prompt]:
    found = checks_of(rig, machine_id)
    assert len(found) == 1 and len(rig.llm.check_prompts) == 1, (
        f"T-0207 A2.1 ({what}): exactly one completion check, under {machine_id}; the fake saw "
        f"{[r for r, _ in rig.llm.check_prompts]} (llm.request phases "
        f"{[(d.get('phase'), d.get('request_id')) for d in rig.kind('llm.request')]})"
    )
    return found[0]


# =====================================================================================================
# Premises: the fake routes a check prompt on its own script
# =====================================================================================================


def test_fake_routes_check_prompts() -> None:
    """Premise of every row (the fake's route): ``check_prompt`` over a step prompt is a check to the fake and neither a
    step nor a re-ask, under both instructions; the step prompt it was built from is not a check; the scenario's frames
    are distinct and real JPEGs, and ``BLANK`` is blank by T-0200's predicate."""
    from local_astra.computer.executor import blank_frame
    from local_astra.toolcall import loop
    from tests.fakes.llm import engine_empty_prompt, is_reask_prompt, is_step_prompt

    step1 = engine_empty_prompt()
    step2 = loop.step_prompt(step1, [loop.LoopStep(S1, "click_at", CLICK_ARGS, tsl.RECORDED_CLICK)])
    for report in (False, True):
        c = check_prompt(step2, CLAIM, held(FR), report=report)
        assert is_check_prompt(c) and not is_step_prompt(c) and not is_reask_prompt(c)
    assert not is_check_prompt(step2) and not is_check_prompt(step1)
    jpegs = [f.jpeg for f in (F0, F1, FR, F2, F3, FX)]
    assert (
        len(set(jpegs)) == len(jpegs) and len({identity(f) for f in (F0, F1, FR, F2, F3, FX)}) == 6
    )
    assert not any(blank_frame(j) for j in jpegs) and blank_frame(BLANK.jpeg)


# =====================================================================================================
# A2.1: the keyword
# =====================================================================================================


@pytest.mark.parametrize("mode", ["dry_run", "off"])
async def test_keyword_refused_outside_live(tmp_path: Path, mode: str) -> None:
    """A2.1: ``completion_check=True`` with tools ``dry_run`` or off raises ``ValueError`` naming ``live``;
    ``False`` constructs and a turn runs (too-wide companion)."""
    from tests.unit import _tool_turn_support as tts

    require_keyword()
    real = tts.Session

    def wrapped(value: bool) -> Any:
        class WithKw(real):  # type: ignore[misc, valid-type]
            def __init__(self, *a: Any, **k: Any) -> None:
                k.setdefault(KW, value)
                super().__init__(*a, **k)

        return WithKw

    kit = "default" if mode == "dry_run" else None
    policy = tsl.LOOP if mode == "dry_run" else POLICY
    tts.Session = wrapped(True)  # type: ignore[misc]
    try:
        with pytest.raises(ValueError, match="live"):
            async with tts.tool_rig(tmp_path / "on", script=[answer(S3)], kit=kit, policy=policy):
                pass
    finally:
        tts.Session = real  # type: ignore[misc]
    tts.Session = wrapped(False)  # type: ignore[misc]
    try:
        async with tts.tool_rig(
            tmp_path / "off", script=[answer(S3)], kit=kit, policy=policy
        ) as rig:
            await rig.ask("hello")
            await rig.turn_done()
    finally:
        tts.Session = real  # type: ignore[misc]
    assert rig.spoken_sources() == S3


# =====================================================================================================
# A2.2: off is BASE, and an armed step that is not checked is BASE
# =====================================================================================================


def _norm(obj: Any, session_id: str, tmp: str) -> Any:
    text = json.dumps(obj, sort_keys=True, default=str)
    return json.loads(text.replace(session_id, "<sid>").replace(tmp, "<tmp>"))


def _sent_bodies(rig: Any) -> list[dict[str, Any]]:
    """Every generate's wire body as the real client builds it from what the session passed: the prompt, the engine
    request id, the ``LlmOptions`` and the plugin arguments (``xargs``), so an option change shows."""
    from local_astra.conductor.llm_client import VllmClient
    from tests.fakes.llm import _refusing_transport

    client = VllmClient(transport=_refusing_transport())
    out = []
    for (rid, prompt, options), xargs in zip(rig.llm.generates, rig.llm.xargs_log, strict=True):
        out.append(client.request_body(prompt, rid, options, dict(xargs or {})))
    return out


def _dump(rig: Any, tmp: Path) -> dict[str, Any]:
    sid = rig.session.session_id
    return _norm(
        {
            "prefills": [_wire(p) for _, p in rig.llm.prefills],
            "bodies": [_wire(p) for _, p, _ in rig.llm.generates],
            # review (tests): the body as sent, with the request's own options, request id and plugin arguments
            "sent": _sent_bodies(rig),
            "options": [o.model_dump(mode="json") for _, _, o in rig.llm.generates],
            "records": [
                {k: r.get(k) for k in ("seq", "kind", "turn_id", "session_ms", "data")}
                for r in rig.log_records
                if r["kind"] in ("llm.request", "llm.done", "note", "tool.call")
            ],
            "hud": [m.model_dump(mode="json") for m in rig.published.messages],
            "machine": [
                [ev.model_dump(mode="json"), str(st), [c.model_dump(mode="json") for c in cmds]]
                for ev, st, cmds in rig.machine.log
            ],
            "extras": [r.extras for r in rig.records()],
        },
        sid,
        str(tmp),
    )


def _off_kwargs(mode: str) -> dict[str, Any] | None:
    return {KW: False} if mode == "false" else None


async def _scenario(tmp: Path, mode: str) -> dict[str, Any]:
    """A2.2's BASE scenarios, with ``completion_check`` absent (``mode="absent"``) or ``False``: the claim after a click
    (the trigger shape), a call then a claim, a T-0172 re-ask, the report step, a search-only turn, a dry-run tool turn
    and a tools-off turn."""
    from tests.unit import _tool_turn_support as tts

    kw = _off_kwargs(mode)
    on = ABSENT if kw is None else False
    out: dict[str, Any] = {}

    async def one(name: str, **k: Any) -> None:
        rig = await claim_turn(tmp / name, on, **k)
        out[name] = _dump(rig, tmp / name)

    await one("claim")
    await one("call_then_claim", steps=([answer(S2), tool(CLICK2)], [answer(CLAIM_5538)]))
    await one(
        "reask",
        steps=([answer(ANN_CLICK)],),
        reasks=([answer(R_STOPPED)],),
        extra={"reask_announced_calls": True},
    )
    await one(
        "report",
        steps=([answer(S2), tool(CLICK2)], [answer(REPORT_DRAFT)]),
        policy=tsl.loop_policy(2),
    )
    await one("search_only", script=[answer(S1), tool(SEARCH_CALL)])
    real = tts.Session

    class WithKw(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            for key, value in (kw or {}).items():
                k.setdefault(key, value)
            super().__init__(*a, **k)

    tts.Session = WithKw  # type: ignore[misc]
    try:
        async with tts.tool_rig(tmp / "dry_run", script=STEP1, policy=tsl.LOOP) as rig:
            rig.llm.script_steps(CLAIM_STEP)
            await rig.ask(QUESTION)
            await rig.turn_done(limit_ms=40_000)
            await rig.advance(300)
        out["dry_run"] = _dump(rig, tmp / "dry_run")
        async with tts.tool_rig(
            tmp / "off", script=[answer(CLAIM)], kit=None, policy=POLICY
        ) as rig:
            await rig.ask(QUESTION)
            await rig.turn_done()
            await rig.advance(300)
        out["off"] = _dump(rig, tmp / "off")
    finally:
        tts.Session = real  # type: ignore[misc]
    return out


_BASE_RUNNER = """
import asyncio, json, pathlib, sys
import local_astra
from tests.unit import test_t0207_session_check as m
out = pathlib.Path(sys.argv[1])
data = asyncio.run(m._scenario(out.parent / "run", "absent"))
out.write_text(json.dumps({"origin": local_astra.__file__, "data": data}))
"""


def _base_capture(root: Path) -> dict[str, Any]:
    gr.require_commits(BASE_SHA)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(tsl, "BASE_SHA", BASE_SHA)
        base_dir = tsl._base_tree(root / "base_tree")
    out = root / "base_out" / "data.json"
    out.parent.mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, "-c", _BASE_RUNNER, str(out)],
        cwd=base_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, f"the base run failed:\n{proc.stderr[-3000:]}"
    got = json.loads(out.read_text())
    assert Path(got["origin"]).resolve().is_relative_to(base_dir.resolve()), (
        f"origin proof: the base run imported local_astra from {got['origin']}"
    )
    data: dict[str, Any] = got["data"]
    return data


async def test_off_is_byte_identical_to_base(tmp_path: Path) -> None:
    """A2.2/A2.13: with the keyword absent and with ``False``, seven scenarios' prefill and generate wire bodies,
    ``llm.request``/``llm.done``/``note``/``tool.call`` records, HUD messages, machine events with their command lists,
    and turn-line extras equal BASE's (5a0602f2, run in a subprocess over ``git archive`` of BASE, origin-proved).
    Premises: the claim turn voiced the claim at BASE, the re-ask was sent, the report step carried ``report_fact``."""
    base = await asyncio.to_thread(_base_capture, tmp_path)
    for mode in ("absent", "false"):
        if mode == "false":
            require_keyword()
        now = await _scenario(tmp_path / mode, mode)
        assert set(now) == set(base)
        for name in sorted(base):
            b, n = base[name], now[name]
            for part in sorted(b):
                assert n[part] == b[part], (
                    f"T-0207 A2.2 ({mode}, {name}): {part} differs from BASE "
                    f"({len(b[part])} vs {len(n[part])} items)"
                )
    assert CLAIM in json.dumps(base["claim"]["machine"]), "premise: BASE submits the claim"
    assert [r for r in base["reask"]["records"] if r["data"].get("phase") == "announce_reask"]
    assert any("report_fact" in r["data"] for r in base["report"]["records"])
    assert CHECK_INSTRUCTION not in json.dumps(base) and PHASE not in json.dumps(base["claim"])


def _act_then_break(fb: Any, inv: Any) -> Any:
    out = fb.perform(inv)
    fb.frame_mode = "raise"
    return out


ARMED: dict[str, dict[str, Any]] = {
    # a frame that fails after the step: no check, the held draft submitted (A2.3)
    "no_frame": dict(steps=(CLAIM_STEP,), handler=_act_then_break),
    # the step ends with a call (a draft), and the report step with a call: no check anywhere in the turn
    "call_drafted": dict(
        steps=([answer(S2), tool(CLICK2)], [answer(REPORT_DRAFT), tool(CLICK3)]),
        policy=tsl.loop_policy(2),
    ),
    # T-0172 re-asked this step (H-T207-5)
    "reasked": dict(
        steps=([answer(ANN_CLICK)],),
        reasks=([answer(R_STOPPED)],),
        extra={"reask_announced_calls": True},
    ),
    # an empty step text (thoughts only)
    "empty_text": dict(steps=([thought(CLAIM)],)),
}


async def _armed_turn(tmp: Path, case: str, on: Any) -> Any:
    c = ARMED[case]
    browser = browser_of()
    if "handler" in c:
        browser.handlers["t1-call"] = c["handler"]
    return await claim_turn(
        tmp,
        on,
        steps=c["steps"],
        reasks=c.get("reasks", ()),
        checks=(NO_CALL_CHECK,),
        browser=browser,
        extra=c.get("extra"),
        policy=c.get("policy"),
    )


@pytest.mark.parametrize("case", list(ARMED))
async def test_armed_no_check_machine_sequence_equals_base(tmp_path: Path, case: str) -> None:
    """A2.2 (R8): with the keyword on, a step that is armed (live, a click executed, a ``tool_step``) but ends without a
    check -- the fresh frame fails, a call was drafted, T-0172 re-asked it, or its text is empty -- reaches the machine
    exactly as with the keyword absent: the same events in the same order (``LlmCommit`` ``t_ms`` aside) and every
    command list equal; the same voiced text. No check request is sent."""
    on = await _armed_turn(tmp_path / "on", case, True)
    base = await _armed_turn(tmp_path / "absent", case, ABSENT)
    assert_no_check(on, case)
    a, b = machine_seq(on), machine_seq(base)
    first = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
    assert a == b, (
        f"T-0207 A2.2 ({case}): the machine sequence differs from the keyword-absent run at index {first} "
        f"(lengths {len(a)} vs {len(b)}): on {a[first] if first is not None and first < len(a) else None}, "
        f"base {b[first] if first is not None and first < len(b) else None}"
    )
    assert on.spoken_sources() == base.spoken_sources()
    if case == "no_frame":
        assert frame_calls(on) == 2 and frame_calls(base) == 1, (
            f"premise: the armed step tried one fresh frame; frame() calls on {frame_calls(on)}, "
            f"absent {frame_calls(base)}"
        )
        assert outcomes(on) == ["no_frame"], f"T-0207 A2.3/A2.11: outcome no_frame; {notes(on)}"
        assert CLAIM in on.spoken_sources(), "T-0207 A2.3: the draft is voiced as at BASE"


# =====================================================================================================
# A2.1/A2.4: when it fires, and the request it sends
# =====================================================================================================


async def test_fires_once_with_shape(tmp_path: Path) -> None:
    """A2.1/A2.3/A2.4 (seq 2493's shape): step 1 clicks, step 2 claims with no call. Exactly one fresh ``frame()``
    after the step and one check under ``t1-gen-s2``: a fresh engine id under the same machine id; the prompt is
    ``check_prompt(<the step-2 prompt as sent>, <step text>, <the fresh frame>, report=False)``; the ``llm.request``
    record has phase ``completion_check``, the step, both ids, the options, the prompt sent and ``fresh_frame:
    {phash, same_as_last: false}`` (R1); every body of the turn has the same system message, tools and template
    arguments; the record comes after the step's last commit and before its one ``llm.done``."""
    rig = await claim_turn(tmp_path)
    sid = step_id(2)
    engine, prompt = assert_one_check(rig, sid, "seq 2493 shape")
    step_rids = [rid for rid, _ in rig.llm.step_prompts if re.search(rf"-{sid}-\d+$", rid)]
    assert len(step_rids) == 1, step_rids
    sent = tsl.step_request(rig, 2)
    want = check_prompt(sent, step_text(CLAIM), held(FR), report=False)
    assert prompt == want, (
        "T-0207 A2.4: the check prompt is check_prompt(<the step prompt as sent>, <step text>, <fresh frame>)\n"
        f"  last messages want {[m.model_dump(mode='json') for m in want.messages[-2:]]}"[:2000]
        + f"\n  got {[m.model_dump(mode='json') for m in prompt.messages[-2:]]}"[:2000]
    )
    assert engine != step_rids[0] and int(engine.rsplit("-", 1)[1]) > int(
        step_rids[0].rsplit("-", 1)[1]
    ), (
        f"T-0207 A2.4: a fresh engine id under the same machine id; step {step_rids[0]}, check {engine}"
    )
    assert frame_calls(rig) == 2, (
        f"T-0207 A2.3: one fresh frame() for the check (onset + 1); calls {rig.browser.calls}"
    )
    (req,) = check_requests(rig)
    assert set(req) == REQUEST_KEYS, sorted(req)
    assert (req["request_id"], req["attempt_id"], req["step"]) == (sid, engine, 2), req
    assert req["prompt"] == session_log.redact(prompt.model_dump(mode="json"))
    ff = req["fresh_frame"]
    assert set(ff) == {"phash", "same_as_last"} and ff["phash"] in (FR.phash, identity(FR)), ff
    assert ff["same_as_last"] is False, ff
    bodies = [b for b in rig.llm.bodies]
    assert len({json.dumps(b["messages"][0], sort_keys=True) for b in bodies}) == 1
    assert len({json.dumps(b.get("tools"), sort_keys=True) for b in bodies}) == 1
    assert len({json.dumps(b.get("chat_template_kwargs"), sort_keys=True) for b in bodies}) == 1
    last_commit = log_index(rig, "llm.commit", lambda d: d.get("request_id") == sid)
    at = log_index(rig, "llm.request", lambda d: d.get("phase") == PHASE)
    done = log_index(rig, "llm.done", lambda d: d.get("request_id") == sid)
    step_commits = [
        i for i in last_commit if rig.log_records[i]["data"].get("attempt_id") == step_rids[0]
    ]
    assert step_commits and at and step_commits[-1] < at[0], "the check follows the step's stream"
    assert len(done) == 1 and done[0] > at[0], (
        f"T-0207 A2.5: one llm.done for {sid}, after the check; done {done}, check {at}"
    )
    assert len(rig.dones(sid)) == 1


# =====================================================================================================
# A2.1: when it does not fire (and the minimal pair that does)
# =====================================================================================================


def _blind_browser() -> Any:
    b = FakeLiveBrowser([], post_frames=[F1, F2])
    b.frame_mode = "raise"
    return b


NOT_WHEN: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "no_browser_action_search": (
        dict(script=[answer(S1), tool(SEARCH_CALL)]),
        dict(script=STEP1),
    ),
    "no_action_step1_answer": (
        dict(script=[answer(CLAIM)], steps=()),
        dict(script=STEP1),
    ),
    "call_present": (
        dict(steps=([answer(S2), tool(CLICK2)], [answer(REPORT_DRAFT), tool(CLICK3)]), policy=2),
        dict(steps=([answer(S2), tool(CLICK2)], [answer(REPORT_DRAFT)]), policy=2),
    ),
    "refused_draft": (
        dict(steps=([answer(CLAIM), tool(NAMELESS)],)),
        dict(steps=([answer(CLAIM)],)),
    ),
    "recovered_thought_call": (
        dict(steps=([answer(CLAIM), t0135.d5()],), extra={"recover_thought_calls": True}),
        dict(steps=([answer(CLAIM), t0135.d5()],), extra={"recover_thought_calls": False}),
    ),
    "reasked": (
        dict(
            steps=([answer(ANN_CLICK)],),
            reasks=([answer(R_STOPPED)],),
            extra={"reask_announced_calls": True},
        ),
        dict(steps=([answer(ANN_CLICK)],), extra={"reask_announced_calls": False}),
    ),
    "degenerate_trip": (
        dict(steps=([answer(CLAIM), thought(t0142.COLLAPSE)],), extra={"degeneracy_guard": True}),
        dict(steps=([answer(CLAIM), thought(t0142.COLLAPSE)],), extra={"degeneracy_guard": False}),
    ),
    "empty_text": (
        dict(steps=([thought(CLAIM)],)),
        dict(steps=([answer(CLAIM)],)),
    ),
    "engine_empty_twice": (
        dict(steps=([answer(CLAIM)],), step_errors=(engine_empty_error,) * 2),
        dict(steps=([answer(CLAIM)],), step_errors=(engine_empty_error,)),
    ),
    "blind": (
        dict(script=[answer(S1), tool(SCROLL)], browser=_blind_browser),
        dict(script=[answer(S1), tool(SCROLL)]),
    ),
    # Re-pinned by T-0208 A2.1 (H-T208-1, which supersedes H-T207-9 for shared-start turns): was "shared_source"
    # (a shared turn whose step-1 scroll swapped in a browser frame: no check). Under A2.1 that turn IS checked (the
    # power side below). The not-when row is now a shared *step image* (T-0208 A2.3): the scroll executes, but no
    # fresh browser frame can be taken, so step 2 still carries the shared image.
    "shared_step_image": (
        dict(script=[answer(S1), tool(SCROLL)], shared=True, stuck=True),
        dict(script=[answer(S1), tool(SCROLL)], shared=True),
    ),
}


def _stuck_vision_browser() -> Any:
    """T-0208 A2.3 (T-0177 B-R3's path): the scroll executes, its after-frame is not the browser's latest, and the
    one fresh ``frame()`` after it fails (it works afterwards), so step 2 keeps the shared image."""
    from tests.unit.test_t0177_step_screens import _OnceBlindBrowser

    def elsewhere(fb: Any, inv: Any) -> Any:
        out = fb.perform(inv)
        return out.model_copy(
            update={"resolved": {**out.resolved, "frame_after": "not-the-latest"}}
        )

    browser = _OnceBlindBrowser([FR], post_frames=[F1, F2])
    browser.handlers["t1-call"] = elsewhere
    return browser


async def _not_when_turn(tmp: Path, case: dict[str, Any]) -> Any:
    tmp.mkdir(parents=True, exist_ok=True)
    steps = case.get("steps", (CLAIM_STEP,))
    script = case.get("script", STEP1)
    policy = tsl.loop_policy(case["policy"]) if "policy" in case else None
    if "shared" in case:
        require_keyword()
        browser = (
            _stuck_vision_browser()
            if case.get("stuck")
            else VisionBrowser([F0, FR], post_frames=[F1, F2])
        )
        async with vision_rig(tmp, browser, script=list(script), session_kwargs={KW: True}) as rig:
            if case["shared"]:
                await select(rig, "shared")
            await drive(rig, *steps, checks=(NO_CALL_CHECK,))
        return rig
    make = case.get("browser")
    browser = make() if make is not None else browser_of()
    async with live(tmp, browser, script=script, extra=case.get("extra"), policy=policy) as rig:
        if "step_errors" in case:
            rig.llm.fail_steps(*case["step_errors"])
        await drive(rig, *steps, checks=(NO_CALL_CHECK,), reasks=case.get("reasks", ()))
    return rig


@pytest.mark.parametrize("name", list(NOT_WHEN))
async def test_not_when(tmp_path: Path, name: str) -> None:
    """A2.1 (too wide): no browser action this turn (a search only; a step-1 answer), a call drafted, a refused draft,
    a recovered thought call, a T-0172 re-ask, a T-0142 trip, an empty step text, a step the engine answered empty twice
    (T-0171), a blind turn and a shared step image (T-0208 A2.1/A2.3: re-pinned from "a shared vision source") send
    no check; the turn line says ``asked: 0`` (R3)."""
    blocked, _ = NOT_WHEN[name]
    rig = await _not_when_turn(tmp_path, blocked)
    if name in ("blind", "shared_step_image"):
        assert results(rig, call_id(1)) == ["executed"], (
            f"premise ({name}): the step-1 scroll executed, so only the {name} rule blocks the check"
        )
    if name == "shared_step_image":
        from tests.unit._live_browser_support import builder_url
        from tests.unit.test_t0177_shared_frame import SH
        from tests.unit.test_t0177_step_screens import current_images

        assert current_images(_wire(tsl.step_request(rig, 2))) == [builder_url(SH)], (
            "premise (shared_step_image): step 2 still carries the shared image"
        )
    assert_no_check(rig, name)
    assert turn_key(rig) == {"on": True, "asked": 0, "outcome": None}, (
        f"T-0207 A2.11 ({name}, R3): {turn_key(rig)}"
    )


@pytest.mark.parametrize("name", list(NOT_WHEN))
async def test_not_when_power(tmp_path: Path, name: str) -> None:
    """A2.1 (absent): each ``test_not_when`` case with its one blocking feature removed sends exactly one check, so
    the negative rows are not vacuous. ``shared_step_image`` (T-0208 A2.1, superseding H-T207-9 for this case): a
    shared turn whose executed scroll swapped in a browser frame at step 2 is checked."""
    _, power = NOT_WHEN[name]
    rig = await _not_when_turn(tmp_path, power)
    assert len(rig.llm.check_prompts) == 1 and len(check_requests(rig)) == 1, (
        f"T-0207 A2.1 ({name} power): one check; the fake saw {[r for r, _ in rig.llm.check_prompts]}, "
        f"records {check_requests(rig)}"
    )
    assert turn_key(rig).get("asked") == 1, turn_key(rig)


async def test_once_per_turn(tmp_path: Path) -> None:
    """A2.1 / H-T207-6 (adversarial): the check at step 2 makes a call; step 3 claims again with no call: no second
    check, step 3's answer is voiced."""
    rig = await claim_turn(
        tmp_path, steps=(CLAIM_STEP, [answer(CLAIM_5538)]), checks=(CALL_CHECK, NO_CALL_CHECK)
    )
    assert_one_check(rig, step_id(2), "once per turn")
    assert rig.spoken_sources().endswith(CLAIM_5538), rig.spoken_sources()


async def test_once_per_turn_not_per_session(tmp_path: Path) -> None:
    """A2.1: the limit is per turn: a second turn with its own click and claim is checked again."""
    async with live(tmp_path, browser_of((F0, FR, F2, FX), (F1, F3))) as rig:
        await drive(rig, CLAIM_STEP, checks=(NO_CALL_CHECK, NO_CALL_CHECK))
        rig.llm.script = list(STEP1)
        rig.llm.script_steps([answer(CLAIM_5538)])
        await rig.ask("and now buy the zelda game")
        await rig.until(
            lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) == 2, "t2 ended"
        )
        await rig.advance(300)
    assert [len(checks_of(rig, step_id(2, t))) for t in ("t1", "t2")] == [1, 1], [
        r for r, _ in rig.llm.check_prompts
    ]


# =====================================================================================================
# A2.5/A2.8: the draft never reaches anyone; the check is the step
# =====================================================================================================


@pytest.mark.parametrize("check", ["no_call", "call"])
async def test_draft_never_voiced_or_stored(tmp_path: Path, check: str) -> None:
    """A2.5/A2.8 (adversarial): when the check completes, nothing of the draft (its answer nor its thought) is ever
    submitted to the machine, voiced, published as ``astra.spoken``, stored in history, memory or the turn record, or
    sent in a later request; the machine's ``LlmCommit`` events for ``t1-gen-s2`` are exactly the check's commits (its
    answer text, its thought). The instruction is in no history, record or later request either."""
    script = NO_CALL_CHECK if check == "no_call" else CALL_CHECK
    steps = (CLAIM_STEP,) if check == "no_call" else (CLAIM_STEP, [answer(S3)])
    rig = await claim_turn(tmp_path, steps=steps, checks=(script,))
    sid = step_id(2)
    assert_one_check(rig, sid, check)
    words = CHECK_SAYS if check == "no_call" else R_CALL
    answers = "".join(e.text for e in commits_for(rig, sid, "answer"))
    thoughts = "".join(e.text for e in commits_for(rig, sid, "thought"))
    assert answers.strip() == words, (
        f"T-0207 A2.5: the step's answer commits are the check's; got {answers!r}"
    )
    assert CHECK_THOUGHT in thoughts and DRAFT_THOUGHT not in thoughts, thoughts
    later = [p for rid, p in rig.llm.step_prompts if re.search(rf"-{step_id(3)}-\d+$", rid)]
    assert later or check == "no_call", "premise: step 3 was sent after the check's call"
    places = {
        "spoken": rig.spoken_sources(),
        "astra.spoken": spoken_published(rig),
        "history": history_json(rig),
        "memory": memory_json(rig),
        "record": record_json(rig),
        "machine": json.dumps([e.model_dump(mode="json") for e in rig.machine.events()]),
        "later requests": json.dumps([p.model_dump(mode="json") for p in later]),
    }
    for where, text in places.items():
        for banned in (CLAIM, DRAFT_THOUGHT):
            assert banned not in text, f"T-0207 A2.5/A2.8: {banned!r} reached {where}"
    for where in ("history", "record", "later requests", "memory"):
        assert CHECK_INSTRUCTION not in places[where], f"T-0207 A2.8: the instruction is in {where}"
    if check == "no_call":
        assert rig.spoken_sources() in (S1 + CHECK_SAYS, S1 + " " + CHECK_SAYS), (
            rig.spoken_sources()
        )


async def test_check_refused_call_outcome(tmp_path: Path) -> None:
    """A2.11 (review, tests): a check whose call ``_judge_tool`` refuses (a tool not in the list) has outcome
    ``refused``, never ``call``; the turn line says so; nothing is executed beyond step 1; the voiced text equals the
    keyword-absent turn whose step 2 committed the same stream; the draft is never voiced."""
    check = [answer(R_CALL), tool(NAMELESS)]
    rig = await claim_turn(tmp_path / "on", checks=(check,))
    as_step = await claim_turn(tmp_path / "as_step", ABSENT, steps=(check,))
    assert_one_check(rig, step_id(2), "refused")
    assert outcomes(rig) == ["refused"], notes(rig)
    assert turn_key(rig) == {"on": True, "asked": 1, "outcome": "refused"}, turn_key(rig)
    assert invoked(rig) == invoked(as_step) == [call_id(1)]
    assert CLAIM not in rig.spoken_sources()
    assert rig.spoken_sources() == as_step.spoken_sources()


async def test_check_call_executes_withheld(tmp_path: Path) -> None:
    """A2.5 (R15/R11): a check that commits ``click_at`` executes it as step 2's call (``t1-call-s2``); nothing of step 2
    is voiced (neither the draft nor the check's words); the withheld note counts the check's text only; one
    ``LlmDone(t1-gen-s2)`` carrying the valid call, one ``llm.done`` record after the check; step 3 is voiced; outcome
    ``call``. The voiced text and executed calls equal the keyword-absent turn whose step 2 committed the check's
    stream itself."""
    rig = await claim_turn(tmp_path / "on", steps=(CLAIM_STEP, [answer(S3)]), checks=(CALL_CHECK,))
    as_step = await claim_turn(tmp_path / "as_step", ABSENT, steps=(CALL_CHECK, [answer(S3)]))
    sid = step_id(2)
    assert_one_check(rig, sid, "call")
    dones = rig.dones(sid)
    assert len(dones) == 1 and dones[0].tool is not None and dones[0].tool.status == "valid", dones
    assert results(rig, call_id(2)) == ["executed"], results(rig, call_id(2))
    assert invoked(rig) == invoked(as_step) == [call_id(1), call_id(2)]
    assert rig.spoken_sources() == as_step.spoken_sources(), (
        f"T-0207 A2.5: the outcome follows R15/R11 as if the step had committed the check's stream; "
        f"{rig.spoken_sources()!r} vs {as_step.spoken_sources()!r}"
    )
    for banned in (CLAIM, R_CALL):
        assert banned not in rig.spoken_sources()
    held_notes = [(n.get("step"), n.get("chars")) for n in notes(rig, "step text withheld")]
    assert held_notes == [(2, len(R_CALL))], f"T-0207 A2.5: the check's text only; {held_notes}"
    assert len(log_index(rig, "llm.done", lambda d: d.get("request_id") == sid)) == 1
    assert outcomes(rig) == ["call"], notes(rig)
    assert turn_key(rig) == {"on": True, "asked": 1, "outcome": "call"}, turn_key(rig)


@pytest.mark.parametrize("ending", ["no_call", "call"])
async def test_report_step_variant_and_r16(tmp_path: Path, ending: str) -> None:
    """A2.4/A2.5 (H-T207-4): at the report step (``max_tool_steps`` 2, step 3) the check uses
    ``CHECK_INSTRUCTION_REPORT`` (``report=True``); a no-call check answer is voiced and a check call follows R16's
    report row: whatever happens equals the keyword-absent turn whose report step committed the check's stream (voiced
    text, executed calls, ``ToolResult`` statuses); the draft is never voiced; the asked note says ``report: true``."""
    check = [answer(REPORT_SAYS)] if ending == "no_call" else [answer(R_CALL), tool(CLICK3)]
    rig = await claim_turn(
        tmp_path / "on",
        steps=([answer(S2), tool(CLICK2)], [answer(REPORT_DRAFT)]),
        checks=(check,),
        policy=tsl.loop_policy(2),
    )
    as_step = await claim_turn(
        tmp_path / "as_step",
        ABSENT,
        steps=([answer(S2), tool(CLICK2)], check),
        policy=tsl.loop_policy(2),
    )
    sid = step_id(3)
    _, prompt = assert_one_check(rig, sid, "report step")
    sent = tsl.step_request(rig, 3)
    # the fresh frame is FR: step 1's click captured F1, step 2's F2, and FR is the next frame() result
    assert prompt == check_prompt(sent, step_text(REPORT_DRAFT), held(FR), report=True), (
        "T-0207 A2.4: the report step's check prompt uses report=True over the step prompt as sent"
    )
    assert prompt.messages[-1].text() == CHECK_INSTRUCTION_REPORT
    assert REPORT_DRAFT not in rig.spoken_sources()
    assert rig.spoken_sources() == as_step.spoken_sources(), (
        rig.spoken_sources(),
        as_step.spoken_sources(),
    )
    assert invoked(rig) == invoked(as_step)
    st = lambda r: [(e.call_id, e.status) for e in r.machine.events(E.ToolResult)]  # noqa: E731
    assert st(rig) == st(as_step), (st(rig), st(as_step))
    (asked,) = notes(rig, NOTE_ASKED)
    assert asked["report"] is True and asked["step"] == 3, asked


async def test_report_flag_false_before_the_report_step(tmp_path: Path) -> None:
    """A2.4 (too-wide companion): a step-2 check with budget left uses ``CHECK_INSTRUCTION`` and notes ``report:
    false``."""
    rig = await claim_turn(tmp_path)
    _, prompt = assert_one_check(rig, step_id(2), "step 2")
    assert prompt.messages[-1].text() == CHECK_INSTRUCTION
    (asked,) = notes(rig, NOTE_ASKED)
    assert asked["report"] is False, asked


# =====================================================================================================
# Review A (F1-F3, N1): armed steps that end abnormally, what counts as an action, the held tail, a blank raise
# =====================================================================================================

#: a check answer whose tail the answer cleaner holds back until the end of the response (``<|`` may still grow
#: into a special literal); the step's own end must release it (``_flush_answer``)
HELD_TAIL_SAYS = "The basket on this screen is still empty <|"
#: a clean thought before the trip, so a held commit exists when the stream trips
OK_THOUGHT = "Looking at the basket now."


def _search_kit(browser: Any) -> Any:
    from tests.fakes.search import FakeSearch
    from tests.unit._live_browser_support import live_kit
    from tests.unit._tool_turn_support import with_search

    return with_search(live_kit(browser), FakeSearch())


ARMED_ABNORMAL: dict[str, dict[str, Any]] = {
    # F1a: the step's stream delivers commits, then the transport fails (an llm.error at BASE)
    "transport_mid_stream": dict(
        steps=(CLAIM_STEP,), setup=lambda r: r.llm.fail_steps_after_script(httpx.ReadError("cut"))
    ),
    # F1b: the engine's empty response after a delivery (T-0171 never resends then)
    "engine_empty_after_delivery": dict(
        steps=(CLAIM_STEP,), setup=lambda r: r.llm.fail_steps_after_script(engine_empty_error)
    ),
    # F1c: a T-0142 trip after a held thought, retried (T-0179) and tripping again
    "trip_retried": dict(
        steps=(
            [thought(OK_THOUGHT), thought(t0142.COLLAPSE)],
            [thought(OK_THOUGHT), thought(t0142.COLLAPSE)],
        ),
        extra={"degeneracy_guard": True},
    ),
    # F1c: a T-0142 trip after a held ui card: the card reached the user at BASE, so no retry (T-0179 B-13a)
    "trip_after_card": dict(
        steps="card",
        extra={"degeneracy_guard": True},
    ),
}


async def _abnormal_turn(tmp: Path, case: str, on: Any) -> Any:
    from tests.unit._hud_rig import ui

    c = ARMED_ABNORMAL[case]
    steps = c["steps"]
    if steps == "card":
        steps = (
            [ui(0), thought(t0142.COLLAPSE, 1, open_=True), tool(CLICK2, canvas_idx=1)],
            [answer(CLAIM_5538)],
        )
    return await claim_turn(
        tmp, on, steps=steps, checks=(NO_CALL_CHECK,), extra=c.get("extra"), setup=c.get("setup")
    )


@pytest.mark.parametrize("case", list(ARMED_ABNORMAL))
async def test_armed_abnormal_end_equals_base(tmp_path: Path, case: str) -> None:
    """A2.2 (review F1): an armed step that ends abnormally -- the transport fails after commits, the engine answers
    empty after a delivery, a T-0142 trip after a held thought (retried by T-0179, then tripping again), a trip after a
    held ``ui`` card (no retry: the card reached the user) -- reaches the machine exactly as with the keyword absent
    (R8), sends no check, and the HUD gets the same messages. Kills: not releasing the held commits on any of these
    paths, and releasing a card without counting it as submitted (which lets T-0179 retry after the card was shown)."""
    on = await _abnormal_turn(tmp_path / "on", case, True)
    base = await _abnormal_turn(tmp_path / "absent", case, ABSENT)
    assert_no_check(on, case)
    a, b = machine_seq(on), machine_seq(base)
    first = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
    assert a == b, (
        f"T-0207 A2.2 ({case}): the machine sequence differs from the keyword-absent run at index {first} "
        f"(lengths {len(a)} vs {len(b)})"
    )
    gens = lambda r: [rid for rid, _, _ in r.llm.generates]  # noqa: E731
    assert len(gens(on)) == len(gens(base)), (case, gens(on), gens(base))
    hud = lambda r: [  # noqa: E731
        getattr(m, "type", None)
        for m in r.published.messages
        if getattr(m, "type", None) != "astra.vision.frame"
    ]
    # the same HUD messages; their order may differ (A2.2 pins machine order only): a held card is published when it
    # is released, after the attempt's thought, where BASE published it as it streamed
    assert sorted(map(str, hud(on))) == sorted(map(str, hud(base))), (
        f"T-0207 A2.2 ({case}): the HUD messages differ"
    )
    if case == "trip_retried":
        assert len([g for g in gens(base) if "-t1-gen-s2-" in g]) == 2, "premise: T-0179 retried"
    if case == "trip_after_card":
        assert len([g for g in gens(base) if "-t1-gen-s2-" in g]) == 1, (
            "premise: no retry after a card"
        )
        assert any(str(t).startswith("astra.ui") and t != "astra.ui.fence" for t in hud(base)), (
            "premise: the card was published"
        )
    if case in ("transport_mid_stream", "engine_empty_after_delivery"):
        assert any(isinstance(e, E.LlmCommit) and e.text == CLAIM for e in base.machine.events()), (
            "premise: BASE submitted the claim before the error"
        )


#: review F2: actions that are not executed browser actions never arm the check
NO_ACTION: dict[str, dict[str, Any]] = {
    "executed_search": dict(script=[answer(S1), tool(SEARCH_CALL)], kit_of=_search_kit),
    "refused_click": dict(outcome="refused"),
    "failed_click": dict(outcome="failed"),
}


async def _no_action_turn(tmp: Path, case: str, *, power: bool) -> Any:
    from tests.unit._live_browser_support import failed, refused

    c = NO_ACTION[case]
    browser = browser_of()
    if not power and "outcome" in c:
        make = refused if c["outcome"] == "refused" else failed
        browser.outcome("t1-call", lambda inv: make(inv, "bad coordinates"))
    script = STEP1 if power else c.get("script", STEP1)
    return await claim_turn(tmp, browser=browser, script=script, kit_of=c.get("kit_of"))


@pytest.mark.parametrize("case", list(NO_ACTION))
async def test_not_when_no_executed_browser_action(tmp_path: Path, case: str) -> None:
    """A2.1 (review F2): an executed ``google_search`` (through the search backend), a refused click or a failed click,
    and nothing else this turn, is not an executed browser action: no check, the claim voiced as at BASE. Power: the
    same turn with the click executed is checked once."""
    rig = await _no_action_turn(tmp_path / "blocked", case, power=False)
    want = {"executed_search": "executed", "refused_click": "refused", "failed_click": "failed"}[
        case
    ]
    assert results(rig, call_id(1)) == [want], f"premise ({case}): {results(rig, call_id(1))}"
    assert_no_check(rig, case)
    assert CLAIM in rig.spoken_sources(), rig.spoken_sources()
    power = await _no_action_turn(tmp_path / "power", case, power=True)
    assert len(power.llm.check_prompts) == 1, f"power ({case}): {power.llm.check_prompts}"


async def test_check_held_tail_is_voiced(tmp_path: Path) -> None:
    """A2.5 (review F3): the check's answer ends with text the answer cleaner holds back (``<|``); the check's end
    releases it, so the machine receives exactly the answer commits the keyword-absent turn submits when its step 2
    commits the same stream, and the same text is voiced."""
    check = [answer(HELD_TAIL_SAYS)]
    rig = await claim_turn(tmp_path / "on", checks=(check,))
    as_step = await claim_turn(tmp_path / "as_step", ABSENT, steps=(check,))
    assert_one_check(rig, step_id(2), "held tail")
    got = "".join(e.text for e in commits_for(rig, step_id(2), "answer"))
    want = "".join(e.text for e in commits_for(as_step, step_id(2), "answer"))
    assert want.endswith("<|"), f"premise: BASE releases the held tail at the end; {want!r}"
    assert got == want, f"T-0207 A2.5: the check's held tail is released; {got!r} vs {want!r}"
    assert rig.spoken_sources() == as_step.spoken_sources()


async def test_blank_predicate_raise_is_no_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A2.3 (ruling N1): a raise from T-0200's ``blank_frame`` on the fresh frame is no frame: no check request, outcome
    ``no_frame``, no asked note, and the step ends exactly as the keyword-absent turn (the draft voiced)."""
    from local_astra.computer import executor

    def boom(jpeg: bytes) -> bool:
        raise OSError("cannot identify image file")

    monkeypatch.setattr(executor, "blank_frame", boom)
    on = await claim_turn(tmp_path / "on")
    base = await claim_turn(tmp_path / "absent", ABSENT)
    assert_no_check(on, "blank_frame raised")
    assert outcomes(on) == ["no_frame"], notes(on)
    assert notes(on, NOTE_ASKED) == []
    assert [e for e in on.machine.events(E.LlmError)] == []
    assert machine_seq(on) == machine_seq(base)
    assert CLAIM in on.spoken_sources()


# =====================================================================================================
# A2.6: a failure falls back to BASE
# =====================================================================================================

REQUEST_FAILURES: dict[str, Callable[[], Any]] = {
    "transport": lambda: httpx.ConnectError("connection refused"),
    "engine_empty": lambda: engine_empty_error,
    "over_long_413": lambda: http_error(413),
}


@pytest.mark.parametrize("name", list(REQUEST_FAILURES))
async def test_failure_falls_back_to_base(tmp_path: Path, name: str) -> None:
    """A2.6 (R7): a check that fails (transport, the engine's empty response, an over-long prompt) writes one
    ``completion check failed`` note naming the error type and no text; is never retried (one check request, no T-0171
    retry note for it); raises no ``llm.error``/``HudError``; the draft is voiced and the step ends exactly as the
    keyword-absent turn (machine sequence and voiced text); outcome ``failed``."""
    on = await claim_turn(
        tmp_path / "on", setup=lambda r: r.llm.fail_checks(REQUEST_FAILURES[name]())
    )
    base = await claim_turn(tmp_path / "absent", ABSENT)
    sid = step_id(2)
    assert_one_check(on, sid, name)
    failed = notes(on, NOTE_FAILED)
    assert len(failed) == 1, f"T-0207 A2.6 ({name}): one failure note; got {failed}"
    want_type = {
        "transport": "ConnectError",
        "engine_empty": type(engine_empty_error("x")).__name__,
        "over_long_413": "HTTPStatusError",
    }[name]
    assert want_type in json.dumps(failed[0]), failed[0]
    for text in (CLAIM, CHECK_INSTRUCTION, "connection refused"):
        assert text not in json.dumps(failed[0]), f"the failure note carries no text: {failed[0]}"
    assert [n for n in notes(on, RETRY_NOTE) if n.get("request_id") == sid] == []
    assert [e for e in on.machine.events(E.LlmError)] == [] and on.commands(C.HudError) == []
    assert on.spoken_sources() == base.spoken_sources() and CLAIM in on.spoken_sources()
    assert machine_seq(on) == machine_seq(base), "T-0207 A2.6: the step ends exactly as BASE"
    assert outcomes(on) == ["failed"], notes(on)
    assert turn_key(on) == {"on": True, "asked": 1, "outcome": "failed"}, turn_key(on)


async def test_failed_check_call_never_executes(tmp_path: Path) -> None:
    """A2.6 (adversarial): a check that commits a call and then fails (the stream raises after its script) executes no
    call and voices nothing of itself; the draft is voiced as at BASE."""
    rig = await claim_turn(
        tmp_path,
        checks=(CALL_CHECK,),
        setup=lambda r: r.llm.fail_checks_after_script(httpx.ReadError("stream cut")),
    )
    assert invoked(rig) == [call_id(1)], invoked(rig)
    assert R_CALL not in rig.spoken_sources() and CLAIM in rig.spoken_sources(), (
        rig.spoken_sources()
    )
    assert outcomes(rig) == ["failed"], notes(rig)


def _frame_fails(how: str) -> Callable[[Any, Any], Any]:
    def act(fb: Any, inv: Any) -> Any:
        out = fb.perform(inv)
        if how == "raise":
            fb.frame_mode = "raise"
        elif how == "hang":
            fb.frame_mode = "hang"
        elif how == "closed":
            fb.close_window()
        elif how == "blank":
            fb.frames.clear()
            fb.frames.append(BLANK)
        return out

    return act


@pytest.mark.parametrize("how", ["raise", "hang", "closed", "blank"])
async def test_frame_failure_voices_draft(tmp_path: Path, how: str) -> None:
    """A2.3/A2.6 (R2): a fresh frame that fails, times out (``LIVE_FRAME_TIMEOUT_MS``), finds the browser closed or is
    blank (T-0200's predicate) means no check: no check request, outcome ``no_frame``, the draft voiced as at BASE."""
    browser = browser_of()
    browser.handlers["t1-call"] = _frame_fails(how)
    rig = await claim_turn(tmp_path, browser=browser)
    assert_no_check(rig, f"frame {how}")
    assert outcomes(rig) == ["no_frame"], f"T-0207 A2.3 ({how}): {notes(rig)}"
    assert CLAIM in rig.spoken_sources(), rig.spoken_sources()
    assert notes(rig, NOTE_ASKED) == [], "R2: no asked note without a frame"


async def test_trip_in_check_ends_turn(tmp_path: Path) -> None:
    """A2.6 (R4): a T-0142 trip inside the check ends the turn the T-0142 way: the trip note names the check's attempt,
    one ``LlmError(t1-gen-s2, <degenerate message>)``, no ``LlmDone`` for the step, the draft dropped (never voiced),
    no call executed; outcome ``failed``."""
    rig = await claim_turn(
        tmp_path,
        checks=([thought(t0142.COLLAPSE), tool(CLICK2)],),
        extra={"degeneracy_guard": True},
    )
    sid = step_id(2)
    engine, _ = assert_one_check(rig, sid, "trip")
    assert [n.get("attempt_id") for n in t0142.notes(rig)] == [engine], t0142.notes(rig)
    errors = [e.message for e in rig.machine.events(E.LlmError) if e.request_id == sid]
    assert errors == [t0142.degenerate_message()] and rig.dones(sid) == [], (errors, rig.dones(sid))
    assert CLAIM not in rig.spoken_sources() and invoked(rig) == [call_id(1)]
    assert outcomes(rig) == ["failed"], notes(rig)


# =====================================================================================================
# A2.7: an abort wins (adversarial)
# =====================================================================================================


def _interrupt(rig: Any, how: str) -> None:
    now = rig.clock.now_ms
    if how == "disconnect":
        rig.submit(E.ClientDisconnected(t_ms=now))
    elif how == "user_stop":
        rig.submit(E.UserStop(t_ms=now, turn_id="t1"))
    else:
        rig.submit(E.UserSpeechOnset(t_ms=now, onset_ms=now))
        rig.submit(E.UserTakeover(t_ms=now + 1, onset_ms=now))


@pytest.mark.parametrize(
    ("phase", "how"),
    [
        ("frame", "takeover"),
        ("frame", "user_stop"),
        ("check", "takeover"),
        ("check", "user_stop"),
        ("check", "disconnect"),
    ],
)
async def test_barge_in_during_frame_and_check(tmp_path: Path, phase: str, how: str) -> None:
    """A2.7 (adversarial, R5): a takeover, a ``UserStop`` or a disconnect while the fresh frame is being taken or while
    the check is in flight ends the turn: the check (if sent) is aborted; when the frame or the engine would still
    answer (released afterwards) no call from it executes and no further request is sent for ``t1``; nothing of the
    draft or of the check is voiced or in history; outcome ``aborted`` when noted."""
    gate = asyncio.Event()

    def act(fb: Any, inv: Any) -> Any:
        out = fb.perform(inv)
        if phase == "frame":
            fb.frame_gate = gate
        return out

    browser = browser_of()
    browser.handlers["t1-call"] = act
    async with live(tmp_path, browser) as rig:
        rig.llm.script_steps(CLAIM_STEP, [answer(S3)])
        rig.llm.script_checks([answer(CHECK_SAYS), tool(CLICK2)])
        if phase == "check":
            rig.llm.hold_checks()
        await rig.ask(QUESTION)
        if phase == "frame":
            await rig.until(
                lambda: browser.frame_active == 1,
                "T-0207 A2.3: the fresh frame was never requested after the claim",
                limit_ms=10_000,
                step=10,
            )
        else:
            await rig.until(
                lambda: bool(rig.llm.check_prompts),
                "T-0207 A2.4: the check was never sent",
                limit_ms=10_000,
            )
        before = len(rig.llm.generates)
        _interrupt(rig, how)
        await rig.until(
            lambda: rig.state.turn_id != "t1" or rig.state.state is TurnState.IDLE,
            "T-0207 A2.7: the turn never closed",
        )
        await rig.advance(200)
        gate.set()
        rig.llm.release_checks()
        await rig.advance(3_000)
        t1_after = [
            rid for rid, _, _ in rig.llm.generates[before:] if re.search(r"-t1-(gen|repair)", rid)
        ]
    if phase == "check":
        engine = rig.llm.check_prompts[0][0]
        assert engine in rig.llm.aborted, f"the check {engine} is aborted; {rig.llm.aborted}"
    else:
        assert rig.llm.check_prompts == [], "R5: no check is sent after an abort during the frame"
    assert call_id(2) not in invoked(rig), invoked(rig)
    assert t1_after == [], f"T-0207 A2.7: no further request for t1; {t1_after}"
    for banned in (CLAIM, CHECK_SAYS):
        assert banned not in rig.spoken_sources(), f"{banned!r} was voiced"
        assert banned not in history_json(rig), f"{banned!r} is in history"
    assert set(outcomes(rig)) <= {"aborted"}, notes(rig)
    if phase == "check":
        assert outcomes(rig) == ["aborted"], notes(rig)


async def test_timeout_shared_not_rearmed(tmp_path: Path) -> None:
    """A2.4/A2.7 (adversarial): the check shares the step's ``llm_timeout`` and never re-arms it. Step 2 is held 15 s
    of session time before its claim arrives; the check is then held: the turn fails with the llm timeout at the step's
    own deadline (its ``ToolResult`` + timeout), not a timeout after the check started; the check is aborted, its call
    never executes, nothing of the draft is voiced, no step-3 request."""
    timeout = POLICY.llm_timeout_ms
    async with live(tmp_path) as rig:
        rig.llm.script_steps(CLAIM_STEP, [answer(S3)])
        rig.llm.script_checks([answer(R_CALL), tool(CLICK2)])
        rig.llm.hold_steps()
        rig.llm.hold_checks()
        await rig.ask(QUESTION)
        await tsl.until_steps(rig, 1)
        armed = rig.machine.events(E.ToolResult)[-1].t_ms
        await rig.advance(15_000, step=500)
        rig.llm.release_steps()
        await rig.until(
            lambda: bool(rig.llm.check_prompts), "the check was never sent", limit_ms=10_000
        )
        started = rig.clock.now_ms
        assert started - armed >= 15_000, "premise: the check starts 15 s into the step"
        await rig.until(
            lambda: bool(rig.machine.events(E.LlmError)) or rig.state.state is TurnState.IDLE,
            "T-0207 A2.4: the shared llm_timeout never fired",
            limit_ms=30_000,
            step=100,
        )
        engine = rig.llm.check_prompts[0][0]
        rig.llm.release_checks()
        await rig.advance(3_000)
    deadline = armed + timeout
    tick_end = next(
        (ev.t_ms for ev, st, cmds in rig.machine.log
         if any(isinstance(c, C.HudError) for c in cmds)),
        None,
    )  # fmt: skip
    assert tick_end is not None and deadline <= tick_end <= deadline + 200, (
        f"T-0207 A2.4: llm timeout at the step's deadline {deadline} (armed {armed}), never re-armed by the check "
        f"(started {started}); the turn failed at {tick_end}"
    )
    assert engine in rig.llm.aborted, rig.llm.aborted
    assert call_id(2) not in invoked(rig)
    assert tsl.generates_for(rig, step_id(3)) == []
    assert CLAIM not in rig.spoken_sources() and R_CALL not in rig.spoken_sources()
    assert outcomes(rig) == ["aborted"], notes(rig)


# =====================================================================================================
# A2.9: grounding (adversarial)
# =====================================================================================================


async def test_check_click_grounded_on_fresh_frame(tmp_path: Path) -> None:
    """A2.9: the check's ``click_at`` carries the fresh frame's geometry (the executor's phash of FR, 1280x800), is not
    refused as stale (the browser's newest own frame is FR) and executes; with the step's frame F1 it would be refused
    (F1 is no longer the newest after the fresh capture)."""
    rig = await claim_turn(tmp_path, steps=(CLAIM_STEP, [answer(S3)]), checks=(CALL_CHECK,))
    inv = inv_of(rig, call_id(2))
    assert inv.frame is not None and inv.frame.phash == FR.phash, (
        f"T-0207 A2.9: the check's click maps through the fresh frame; geometry {inv.frame}"
    )
    assert (inv.frame.width, inv.frame.height) == (FR.width, FR.height)
    assert results(rig, call_id(2)) == ["executed"], results(rig, call_id(2))


async def test_old_frame_click_still_stale(tmp_path: Path) -> None:
    """A2.9 (R6): a check-made click is not exempt from T-0111g B9: the check is held, the browser's newest own frame
    then changes (``land``), the check commits a click read off its fresh frame: the click is refused with B9's reason
    before ``execute`` and never dispatched."""
    async with live(tmp_path) as rig:
        rig.llm.script_steps(CLAIM_STEP, [answer(S3)])
        rig.llm.script_checks(CALL_CHECK)
        rig.llm.hold_checks()
        await rig.ask(QUESTION)
        await rig.until(
            lambda: bool(rig.llm.check_prompts), "the check was never sent", limit_ms=10_000
        )
        rig.browser.land(FX)
        rig.llm.release_checks()
        await rig.turn_done(limit_ms=40_000)
        await rig.advance(300)
    assert call_id(2) not in invoked(rig), f"never dispatched; {invoked(rig)}"
    assert results(rig, call_id(2)) == ["refused"], results(rig, call_id(2))
    step3 = tsl.step_request(rig, 3)
    assert f"reason:{Q}{FRESHNESS}{Q}" in step3.messages[-1].text(), step3.messages[-1].text()[
        -300:
    ]


# =====================================================================================================
# A2.10: the HUD
# =====================================================================================================


async def test_vision_publish_is_fresh_frame(tmp_path: Path) -> None:
    """A2.10: the check's ``astra.vision.frame`` is the fresh frame (one image whose identity is FR's, source
    ``browser``, not blind, step 2), published after step 2's own; the check's thoughts are published as
    ``astra.thought`` with kind ``reask`` under the check's attempt id."""
    rig = await claim_turn(tmp_path)
    engine, _ = assert_one_check(rig, step_id(2), "publish")
    frames = [m for m in rig.published.messages if getattr(m, "type", None) == "astra.vision.frame"]
    fresh = [m for m in frames if [i.identity for i in m.images] == [identity(FR)]]
    assert len(fresh) == 1, (
        f"T-0207 A2.10: one vision frame of the fresh capture; got {[[i.identity for i in m.images] for m in frames]}"
    )
    m = fresh[0]
    assert (m.source, m.blind, m.step) == ("browser", False, 2), m
    assert frames.index(m) == len(frames) - 1, "the fresh frame is the newest published"
    thoughts = [
        t for t in rig.published.messages
        if getattr(t, "type", None) == "astra.thought" and t.attempt_id == engine
    ]  # fmt: skip
    assert thoughts and {t.kind for t in thoughts} == {"reask"}, (
        f"T-0207 A2.10 (H-T207-10): the check's thoughts are kind reask; {[(t.kind, t.text) for t in thoughts]}"
    )
    assert any(CHECK_THOUGHT in t.text for t in thoughts)


# =====================================================================================================
# A2.11: logging
# =====================================================================================================


@pytest.mark.parametrize("same", [False, True])
async def test_notes_counts_only(tmp_path: Path, same: bool) -> None:
    """A2.11: one ``completion check asked`` note with exactly ``{what, request_id, step, chars, same_as_last,
    report}`` (``chars`` = the step text's length, ``same_as_last`` true only when the fresh frame is the step's image,
    R1) and one outcome note with exactly ``{what, request_id, step, outcome}``; no note carries the draft, the
    instruction or the check's words; no new log kind beside the keyword-absent run."""
    fresh = F1.model_copy() if same else FR
    rig = await claim_turn(tmp_path / "on", browser=browser_of((F0, fresh)))
    base = await claim_turn(tmp_path / "absent", ABSENT)
    asked, outcome = notes(rig, NOTE_ASKED), notes(rig, NOTE_OUTCOME)
    assert len(asked) == 1 and set(asked[0]) == ASKED_KEYS, asked
    assert len(outcome) == 1 and set(outcome[0]) == OUTCOME_KEYS, outcome
    a = asked[0]
    assert (a["request_id"], a["step"], a["chars"], a["same_as_last"], a["report"]) == (
        step_id(2),
        2,
        len(step_text(CLAIM)),
        same,
        False,
    ), a
    assert outcome[0]["outcome"] == "no_call" and outcome[0]["outcome"] in OUTCOMES
    (req,) = check_requests(rig)
    assert req["fresh_frame"]["same_as_last"] is same, req["fresh_frame"]
    for note in notes(rig):
        text = json.dumps(note)
        for banned in (
            CLAIM,
            CHECK_INSTRUCTION,
            CHECK_SAYS,
            DRAFT_THOUGHT,
            CHECK_THOUGHT,
            "Vinland",
        ):
            assert banned not in text, f"T-0207 A2.11: a note carries {banned!r}: {note}"
    kinds = {r["kind"] for r in rig.log_records}
    assert kinds <= {r["kind"] for r in base.log_records} | {"note", "llm.request"}, sorted(kinds)


@pytest.mark.parametrize("on", [True, False, ABSENT], ids=["true", "false", "absent"])
async def test_turn_line_key(tmp_path: Path, on: Any) -> None:
    """A2.11 (R3): with ``True`` the live turn line's extras carry ``completion_check: {on: true, asked: 1, outcome:
    "no_call"}``; a later turn with no action ``{on: true, asked: 0, outcome: null}``; with ``False`` or absent nothing
    is added; the key lives in extras only."""
    require_keyword()  # every arm: the absent arm compares against a session that has the keyword
    async with live(tmp_path, on=on) as rig:
        await drive(rig, CLAIM_STEP, checks=(NO_CALL_CHECK,))
        rig.llm.script = [answer("It is a small shop.")]
        await rig.ask("what is on the page")
        await rig.until(lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) == 2, "t2")
        lines = [json.loads(x) for x in rig.lines()]
    keys = [turn_key(rig, t) for t in ("t1", "t2")]
    if on is True:
        assert keys == [
            {"on": True, "asked": 1, "outcome": "no_call"},
            {"on": True, "asked": 0, "outcome": None},
        ], keys
    else:
        assert keys == [ABSENT, ABSENT], f"T-0207 A2.11: nothing is added with {on!r}; {keys}"
    for line in lines:
        rest = {k: v for k, v in line.items() if k != "extras"}
        assert TURN_KEY not in json.dumps(rest), sorted(rest)


# =====================================================================================================
# A2.8: the step after a check-made call
# =====================================================================================================


async def test_step_after_check_call_has_no_instruction(tmp_path: Path) -> None:
    """A2.8: after a check-made call, step 3's request is byte-identical to the keyword-absent turn whose step 2
    committed the check's stream itself: ``step_prompt(base, steps)`` with ``LoopStep.text`` holding the check's text
    only and the image that action's after-frame; no instruction, draft, draft thought or check image in it."""
    rig = await claim_turn(tmp_path / "on", steps=(CLAIM_STEP, [answer(S3)]), checks=(CALL_CHECK,))
    as_step = await claim_turn(
        tmp_path / "as_step", ABSENT, steps=(CALL_CHECK, [answer(S3)]), browser=browser_of((F0,))
    )
    got, want = tsl.step_request(rig, 3), tsl.step_request(as_step, 3)
    assert got == want, (
        "T-0207 A2.8: step 3 is step_prompt(base, steps) with the check's text and call\n"
        f"  got tail ...{tsl.tail(got)[-400:]!r}\n  want tail ...{tsl.tail(want)[-400:]!r}"
    )
    text = json.dumps(got.model_dump(mode="json"))
    for banned in (CHECK_INSTRUCTION, CLAIM, DRAFT_THOUGHT, CHECK_THOUGHT, identity(FR)):
        assert banned not in text, f"T-0207 A2.8: step 3 carries {banned[:40]!r}"
    images = [p.phash for m in got.messages for p in m.content if isinstance(p, ImagePart)]
    assert images[-1] == identity(F2), f"the image is the check action's after-frame; {images}"


# =====================================================================================================
# A2.13: frozen paths over the closed range
# =====================================================================================================

FROZEN = (
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/prompt_builder.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/contracts",
    "local_astra/toolcall/loop.py",
    "local_astra/toolcall/live.py",
    "local_astra/toolcall/announced_call.py",
    "local_astra/computer",
    "tests/unit/test_turn_invariants.py",
)


def _doc02_at(rev: str) -> str:
    return gr._git("show", f"{rev}:docs/02-architecture.md").stdout


def test_frozen_paths_over_range() -> None:
    """A2.13: the frozen paths are byte-identical over ``BASE_SHA..TIP_IMPL`` (two literals, never the working tree),
    none added or removed, and docs/02 §4 (the invariants) is unedited. Red until the test-author sets ``TIP_IMPL``
    after the implementation commit. Power: the same predicate sees T-0172's ``announced_call.py`` change
    (``3e4d45b7`` to this BASE)."""
    assert TIP_IMPL is not None, (
        "T-0207 A2.13: TIP_IMPL is set by the test-author after the implementation commit"
    )
    gr.require_commits(BASE_SHA, TIP_IMPL, "3e4d45b7")
    assert gr.is_ancestor(BASE_SHA, TIP_IMPL), (
        f"{BASE_SHA[:8]} is not an ancestor of {TIP_IMPL[:8]}"
    )
    listed = gr._git("ls-tree", "-r", "--name-only", BASE_SHA, "--", *FROZEN).stdout.split()
    for p in FROZEN:
        assert any(x == p or x.startswith(p + "/") for x in listed), f"premise: {p} exists at BASE"
    changed = gr.changed(BASE_SHA, TIP_IMPL, *FROZEN)
    assert changed == [], (
        f"T-0207 A2.13: frozen paths differ in {BASE_SHA[:8]}..{TIP_IMPL[:8]}: {changed}"
    )
    from tests.unit import test_t0107a_docs as t7d

    s4_base = t7d.section(_doc02_at(BASE_SHA), "## 4. ")
    s4_now = t7d.section(_doc02_at(TIP_IMPL), "## 4. ")
    assert s4_base and s4_now == s4_base, "T-0207 A2.13: the §4 invariants are unedited"
    assert "local_astra/toolcall/announced_call.py" in gr.changed("3e4d45b7", BASE_SHA, *FROZEN), (
        "power"
    )


def test_frozen_paths_power_premise() -> None:
    """A2.13 power (green at BASE): the predicate sees a change a known task made to a frozen path, so the guard above
    cannot pass by seeing nothing."""
    gr.require_commits(BASE_SHA, "3e4d45b7")
    assert "local_astra/toolcall/announced_call.py" in gr.changed("3e4d45b7", BASE_SHA, *FROZEN)


# =====================================================================================================
# A2-R3: replay readers never read a completion_check request as a step
# =====================================================================================================


def test_replay_thought_calls_skips_check_requests(tmp_path: Path) -> None:
    """A2-R3 (``scripts/replay_thought_calls.py``): a ``phase: "completion_check"`` request whose commits have the
    firing shape (an open thought ending in a complete call, no ``tool`` commit) is neither scanned nor a firing; the
    step request beside it, with the same shape and the same machine id, is (power: the shape fires as a step). Built
    with T-0135's run-directory builder (``tests/unit/test_t0135_replay.py``, read only)."""
    from tests.unit import test_t0135_replay as t135

    shape = [("thought", "I will click the button.\n" + t135.CLICK_A, True)]
    step = t135._req("20991231-000207-1.jsonl", "C1", shape)
    check = {
        **t135._req("20991231-000207-1.jsonl", "C1", shape, phase="completion_check"),
        "seq": 9,
        "request_id": "t1-gen-s2",
        "attempt_id": "20991231-t1-gen-s2-9",
        "step": 2,
    }
    doc = t135.run_json(t135.build_run(tmp_path, [step, check]))
    assert doc["scanned"] == 1, f"A2-R3: only the step is scanned; {doc['scanned']}"
    fired = [(f["request_id"], f["step"]) for f in doc["firings"]]
    assert fired == [("t1-gen-s2", 2)], f"A2-R3: one firing, the step's; {doc['firings']}"
