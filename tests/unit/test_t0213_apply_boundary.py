"""T-0213 group A2-apply: changing a setting from the panel applies only at a turn boundary (docs/tasks/T-0213.md
A2a-1..A2a-4, A2a-6, A2a-7; "Tests to write -> Unit (A2-apply) -> test_t0213_apply_boundary.py").

**The surface these rows assume** (the task's names, and the implementer's as the orchestrator relayed them):

- A2a-1: ``contracts/protocol.py`` defines ``AstraSettingsSet`` (``type = "astra.settings.set"``, client to server,
  in ``CLIENT_TO_SERVER`` and not in ``SERVER_TO_CLIENT``) with ``rev: int``, ``changes: dict[str, value]`` and
  ``stop_turn: bool = False``; an unknown field decodes ``Malformed``. The rows send it **over the wire**, as raw
  JSON into a greeted page of a socketless ``HudServer`` (``tests/unit/_t0181_rig.mic_rig``), so before A2a-1 lands
  the codec answers ``Unknown`` and every row is red on its own named "does not exist yet" assertion.
- A2a-2: every routed set is answered, to **every** page, by one ``astra.settings.state`` (A2r-1) carrying the rev in
  force; a refused set's state carries ``refused`` (``[{key: "*", reason: "stale"}]`` for a stale rev; ``validate``'s
  ``Refusal(key, reason)`` pairs for an invalid one) and changes nothing.
- A2a-3: a set that arrives while a turn is in flight (a prefill, a generate, a tool execution, a step, a completion
  check) is applied only once the machine is IDLE; at IDLE it is applied at once; ``stop_turn=true`` submits the
  ``astra.turn.stop`` stop (T-0193 ``UserStop``) first.
- A2a-4: applying sets ``visual_tokens`` on the main client **by attribute** (``VllmClient.visual_tokens``, a plain
  attribute read per request, ``llm_client.py:212,266``) and rebuilds the system message. **Reading AP-1:** the
  fake LLM here (``BudgetLlm``) carries that attribute and builds every request's body with the real
  ``VllmClient(visual_tokens=<its attribute at request time>).request_body``, so ``mm_processor_kwargs`` is what the
  real client would send; a session that switched the budget only on objects of type ``VllmClient`` would never
  reach a fake and fail here.
- **Reading AP-2:** the rig's session is built without ``cli`` (``Session(..., visual_tokens=1120)``, live kit over
  ``FakeLiveBrowser``, loop on): its snapshot is ``Session._own_settings()`` (A2-read), rev 0.
- **Reading AP-3 (ReGround, A2a-4 "switches ReGround's pass-2 client"):** the pass-2 client is observed at the
  ``cli`` level (``tests/unit/test_cli_t0132.build``, every ``VllmClient`` replaced by a recorded ``FakeLlm``): a
  live session built with ReGround off, given ``{reground: True, overview_frame: False}`` (R-4) at IDLE, ends with
  ``session.reground`` true, ``session.reground_visual_tokens == 280`` (T-0133 A5: the crop at the 1120 main
  budget) and ``session.reground_llm`` a client other than ``session.llm`` whose budget is 280 (its
  ``visual_tokens`` attribute, or the ``visual_tokens`` keyword ``cli`` built it with). The implementation notes
  put the pre-built client in ``cli``, which A2-apply does not own; the row does not care who builds it.

Expected values are typed from the criteria, the catalog and the config constants, or come from an independent oracle:
the real ``VllmClient.request_body``, ``live_settings.validate`` (A1, merged and tested on its own) for refusal
reasons, ``prompt_builder.USER_INSTRUCTIONS_HEADER`` for the block. Nothing is read off the apply path.

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, tools ``live`` (loop on, 8 steps, ReGround off,
1280x800 JPEG frames), ``RecordingTts`` 200 ms a word, ``FakeAudioIO``, n = 1, deterministic per row. No measurement.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from local_astra.cli import guidance_table_of
from local_astra.conductor.llm_client import VllmClient
from local_astra.conductor.prompt_builder import USER_INSTRUCTIONS_HEADER
from local_astra.config import action_outcome_without_feedback_refusal
from local_astra.contracts import protocol as P
from local_astra.contracts.codec import Malformed, decode
from local_astra.contracts.llm import THINKING_TEMPLATE_KWARG, LlmOptions, Prompt
from local_astra.contracts.toolcall import ToolInvocation, ToolOutcome
from local_astra.contracts.turn import TurnState
from tests.fakes.llm import is_caption_prompt
from tests.unit import test_session_tool_loop as tsl
from tests.unit._live_browser_support import FakeLiveBrowser, jpeg_frame, live_kit
from tests.unit._t0181_rig import MicRig, mic_rig
from tests.unit._tool_turn_support import VALID_CLICK, ToolLlm, answer, tool

SET = "astra.settings.set"
STATE = "astra.settings.state"
#: the live main budget the rig's session is built with (``LIVE_DEFAULT_VISUAL_TOKENS``, T-0132 A1)
OLD_BUDGET = 1120
NEW_BUDGET = 560
#: T-0133 A5: ReGround's crop budget at the 1120 main budget
CROP = 280
#: the user's instructions the rows apply: a canary no other text in the session holds
CANARY = "ZQX-T0213-CANARY answer in British English and keep it short."
QUESTION_1 = "open the settings menu"
QUESTION_2 = "and what does it say now"
VOICE = "please open the settings menu"
ANSWER_1 = "The menu is open now. "
ANSWER_2 = "It says welcome back. "
CID = tsl.call_id(1)
#: T-0207's no-call check answer (the check confirms the claim)
CHECK_SAYS = "The page shows the settings, so it is open."

_TURN_OF = re.compile(r"-(t\d+)-")


# =====================================================================================================
# the fake LLM that records what the real client would send
# =====================================================================================================


@dataclass(frozen=True)
class Sent:
    """One request as the real ``VllmClient`` would put it on the wire (reading AP-1)."""

    kind: str  # "prefill" | "generate" | "step" | "check"
    engine_id: str
    turn: str | None
    #: the system message of the body, JSON with sorted keys (byte for byte what the client sends)
    system_json: str
    #: the rendered system text (``Message.text()``)
    system_text: str
    mm: Any
    template_kwargs: dict[str, Any]


class BudgetLlm(ToolLlm):
    """``ToolLlm`` with the client's ``visual_tokens`` attribute; every non-caption prefill and generate is recorded
    in ``sent`` as the real client's body built with the attribute's value **at request time**."""

    def __init__(self, script: Sequence[Any] = (), *, visual_tokens: int | None = OLD_BUDGET):
        super().__init__(list(script))
        self.visual_tokens = visual_tokens
        self.sent: list[Sent] = []

    def _see(self, kind: str, prompt: Prompt, engine_id: str, options: Any) -> None:
        if is_caption_prompt(prompt):
            return
        body = VllmClient(visual_tokens=self.visual_tokens).request_body(prompt, engine_id, options)
        systems = [m for m in body["messages"] if m.get("role") == "system"]
        texts = [m.text() for m in prompt.messages if m.role == "system"]
        assert len(systems) == 1 and len(texts) == 1, (
            f"premise: one system message per request; got {len(systems)} in {engine_id}"
        )
        if kind == "generate":
            from tests.fakes.llm import is_check_prompt, is_step_prompt

            kind = (
                "step" if is_step_prompt(prompt) else "check" if is_check_prompt(prompt) else kind
            )
        found = _TURN_OF.search(engine_id)
        self.sent.append(
            Sent(
                kind=kind,
                engine_id=engine_id,
                turn=found.group(1) if found else None,
                system_json=json.dumps(systems[0], sort_keys=True),
                system_text=texts[0],
                mm=body.get("mm_processor_kwargs"),
                template_kwargs=dict(body["chat_template_kwargs"]),
            )
        )

    async def prefill(self, prefix: Prompt, request_id: str) -> Any:
        self._see("prefill", prefix, request_id, LlmOptions(max_tokens=1))
        return await super().prefill(prefix, request_id)

    async def generate(  # type: ignore[override]
        self,
        prompt: Prompt,
        request_id: str,
        options: Any = None,
        *,
        xargs: Any = None,
        cd: Any = None,
    ) -> AsyncIterator[Any]:
        self._see("generate", prompt, request_id, options)
        async for commit in super().generate(prompt, request_id, options, xargs=xargs, cd=cd):
            yield commit

    def of_turn(self, turn: str) -> list[Sent]:
        return [s for s in self.sent if s.turn == turn]


# =====================================================================================================
# the rig
# =====================================================================================================


def require_set() -> Any:
    cls = getattr(P, "AstraSettingsSet", None)
    assert cls is not None, (
        "T-0213 A2a-1: contracts/protocol.py AstraSettingsSet (astra.settings.set {rev, changes, stop_turn}) does "
        "not exist yet"
    )
    return cls


def browser_of() -> FakeLiveBrowser:
    return FakeLiveBrowser(
        [jpeg_frame("t0213-onset"), jpeg_frame("t0213-fresh")],
        post_frames=[jpeg_frame(f"t0213-after-{k}") for k in (1, 2, 3, 4)],
    )


@asynccontextmanager
async def settings_rig(
    tmp_path: Path,
    *,
    script: Sequence[Any] = (answer(ANSWER_1),),
    browser: FakeLiveBrowser | None = None,
    session_kwargs: dict[str, Any] | None = None,
    guidance_table: dict[str, tuple[str | None, bool]] | None = None,
) -> AsyncIterator[tuple[MicRig, BudgetLlm, FakeLiveBrowser]]:
    """A live session (loop on) over ``FakeLiveBrowser``, ``Session(..., visual_tokens=1120)``, the ``BudgetLlm``,
    a socketless ``HudServer`` with one greeted page (``rig.page``).

    R-18: ``guidance_table`` is handed to ``init_settings`` as ``cli`` hands it in live (the session's own snapshot,
    no sources, no other rows), so a rig that changes ``live_guidance`` can apply it; without it such a change is
    refused."""
    browser = browser if browser is not None else browser_of()
    llm = BudgetLlm(list(script))
    async with mic_rig(
        tmp_path,
        kit=live_kit(browser),
        policy=tsl.LOOP,
        llm=llm,
        session_kwargs={"visual_tokens": OLD_BUDGET, **(session_kwargs or {})},
    ) as rig:
        browser.now_ms = lambda: rig.clock.now_ms
        if guidance_table is not None:
            rig.session.init_settings(rig.session.settings, {}, {}, guidance_table=guidance_table)
        await rig.advance(400, step=40)
        yield rig, llm, browser


def send_set(
    rig: MicRig,
    changes: dict[str, Any],
    *,
    rev: int = 0,
    stop_turn: bool | None = None,
    page: Any = None,
) -> int:
    """The page's ``astra.settings.set`` as raw JSON (``Unknown`` before A2a-1, never a ``ValidationError`` here).
    Returns the published-message index the reply is read from."""
    mark = len(rig.published.messages)
    body: dict[str, Any] = {"type": SET, "rev": rev, "changes": changes}
    if stop_turn is not None:
        body["stop_turn"] = stop_turn
    (page or rig.page).receive(json.dumps(body))
    return mark


async def set_and_pump(rig: MicRig, changes: dict[str, Any], **kw: Any) -> int:
    mark = send_set(rig, changes, **kw)
    await rig.pump()
    await rig.advance(40, step=20)
    return mark


def states(messages: Sequence[Any]) -> list[Any]:
    return [m for m in messages if getattr(m, "type", None) == STATE]


def last_state(rig: MicRig, since: int = 0, what: str = "") -> Any:
    found = states(rig.published.messages[since:])
    assert found, (
        f"T-0213 A2a-2: the session answers an astra.settings.set with astra.settings.state{what}; none was "
        f"published (astra.settings.set does not exist yet, or is not routed)"
    )
    return found[-1]


def item(state: Any, key: str) -> Any:
    got = [i for i in state.items if i.key == key]
    assert len(got) == 1, f"one item for {key}; got {[i.key for i in state.items]}"
    return got[0]


def refused_pairs(state: Any) -> list[tuple[str, str]]:
    out = []
    for r in state.refused:
        key = r["key"] if isinstance(r, dict) else r.key
        reason = r["reason"] if isinstance(r, dict) else r.reason
        out.append((key, reason))
    return out


def block(text: str) -> str:
    """A1-4: the user's block, the last part of the system message."""
    return USER_INSTRUCTIONS_HEADER + "\n" + text


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def mm(budget: int) -> dict[str, int]:
    return {"max_soft_tokens": budget}


async def typed_turn(rig: MicRig, question: str, reply: str, n_lines: int) -> None:
    rig.llm.script = [answer(reply)]
    await rig.type_question(question)
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= n_lines,
        f"premise: the typed question {question!r} was not answered",
        limit_ms=20_000,
        step=40,
    )
    await rig.advance(300, step=40)


def assert_one_regime(sent: Sequence[Sent], *, system_json: str, budget: int, what: str) -> None:
    """Every request of a turn carries one settings revision: byte-identical system messages and one budget."""
    assert sent, f"premise: {what} sent requests"
    systems = {s.system_json for s in sent}
    assert systems == {system_json}, (
        f"T-0213 A2a-3: every request of {what} carries the same system message, byte for byte; "
        f"{[(s.kind, s.engine_id, sha12(s.system_json)) for s in sent]} vs {sha12(system_json)}"
    )
    budgets = [(s.kind, s.mm) for s in sent]
    assert all(s.mm == mm(budget) for s in sent), (
        f"T-0213 A2a-3: every request of {what} carries max_soft_tokens {budget}; got {budgets}"
    )


def assert_thinking_on(sent: Sequence[Sent]) -> None:
    bad = [
        (s.engine_id, s.template_kwargs)
        for s in sent
        if s.template_kwargs != {THINKING_TEMPLATE_KWARG: True}
    ]
    assert not bad, (
        f"T-0213 A2a-6: no setting changes thinking (the template kwarg stays on); got {bad}"
    )


# =====================================================================================================
# A2a-1: the contract
# =====================================================================================================


def test_set_contract_round_trip_and_direction():
    """A2a-1: ``astra.settings.set`` decodes to ``AstraSettingsSet`` with ``rev``, ``changes`` and ``stop_turn``
    (default False); it is client to server only."""
    cls = require_set()
    got = decode(json.dumps({"type": SET, "rev": 3, "changes": {"visual_tokens": 560}}))
    assert isinstance(got, cls), f"T-0213 A2a-1: decodes to AstraSettingsSet; got {got!r}"
    assert (got.rev, dict(got.changes), got.stop_turn) == (3, {"visual_tokens": 560}, False), got
    stop = decode(json.dumps({"type": SET, "rev": 0, "changes": {}, "stop_turn": True}))
    assert isinstance(stop, cls) and stop.stop_turn is True, stop
    assert SET in P.CLIENT_TO_SERVER and SET not in P.SERVER_TO_CLIENT, (
        "T-0213 A2a-1: astra.settings.set joins CLIENT_TO_SERVER only"
    )


@pytest.mark.parametrize(
    "raw",
    [
        {"type": SET, "rev": 0, "changes": {}, "extra": 1},
        {"type": SET, "changes": {}},
        {"type": SET, "rev": 0},
        {"type": SET, "rev": "0", "changes": {}},
        {"type": SET, "rev": 0, "changes": [], "stop_turn": False},
    ],
    ids=["unknown-field", "no-rev", "no-changes", "rev-string", "changes-list"],
)
def test_set_malformed(raw):
    """A2a-1: an unknown field is malformed (and so is a set missing ``rev`` or ``changes``, or of the wrong type)."""
    require_set()
    assert isinstance(decode(json.dumps(raw)), Malformed), f"T-0213 A2a-1: {raw} must be Malformed"


# =====================================================================================================
# A2a-3: at IDLE a set applies at once
# =====================================================================================================


async def test_idle_set_applies_at_once(tmp_path):
    """A2a-3: at IDLE (no turn yet) a valid set is applied at once: every page gets a state with rev 1, the changed
    items read their new value with source ``ui`` and nothing pending; the next turn's every request carries the new
    budget and a system message ending with the user's block."""
    require_set()
    async with settings_rig(tmp_path) as (rig, llm, _browser):
        other = rig.greet()
        other_mark = len(other.sent())
        mark = await set_and_pump(rig, {"visual_tokens": NEW_BUDGET, "user_instructions": CANARY})
        state = last_state(rig, mark)
        assert state.rev == 1 and not state.refused, (state.rev, state.refused)
        vt, ui = (
            item(state, "visual_tokens"),
            item(state, "user_instructions")
            if any(i.key == "user_instructions" for i in state.items)
            else None,
        )
        assert (vt.value, vt.source, vt.pending) == (NEW_BUDGET, "ui", None), vt
        assert state.instructions == CANARY, state.instructions
        if ui is not None:
            assert ui.source == "ui", ui
        others = states(other.sent()[other_mark:])
        assert others and others[-1].rev == 1, "T-0213 A2a-2: every page receives the updated state"
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        t1 = llm.of_turn("t1")
    assert t1, "premise: t1 sent requests"
    for s in t1:
        assert s.mm == mm(NEW_BUDGET), f"T-0213 A2a-4: {s.kind} {s.engine_id} carries {s.mm}"
        assert s.system_text.endswith(block(CANARY)), (
            f"T-0213 A2a-4/A1-4: the system message ends with the user's block; {s.system_text[-200:]!r}"
        )


# =====================================================================================================
# A2a-3: the core invariant (adversarial): a set mid-turn waits for the boundary
# =====================================================================================================

WINDOWS = ("prefill", "between_steps", "completion_check")


async def _open_window(
    rig: MicRig, llm: BudgetLlm, browser: FakeLiveBrowser, window: str, gate: asyncio.Event
) -> None:
    """Drive ``t1`` to ``window`` and stop there with the turn in flight."""
    if window == "prefill":
        # a spoken question as the recogniser reports it (onset, then words); its end of turn comes after the set
        await rig.onset()
        await rig.words(VOICE)
        await rig.until(
            lambda: (
                rig.state.state is TurnState.LISTENING
                and rig.state.active_prefill is not None
                and [s for s in llm.of_turn("t1") if s.kind == "prefill"]
            ),
            "premise: the voice onset never opened t1 with a live prefill",
            limit_ms=4_000,
            step=20,
        )
        assert rig.state.active_generate is None and not [
            s for s in llm.of_turn("t1") if s.kind != "prefill"
        ], "premise: the set arrives after the prefill started and before the generate"
    elif window == "between_steps":
        await rig.type_question(QUESTION_1)
        await rig.until(
            lambda: ("execute", CID) in browser.calls,
            "premise: the click never reached the browser",
            limit_ms=10_000,
            step=20,
        )
        await rig.advance(100, step=20)
        assert rig.state.state is TurnState.THINKING and not [
            s for s in llm.of_turn("t1") if s.kind == "step"
        ], "premise: the set arrives while step 1's click executes, before step 2"
    else:
        llm.hold_checks()
        await rig.type_question(QUESTION_1)
        await rig.until(
            lambda: bool(llm.check_prompts),
            "premise: the completion check was never sent",
            limit_ms=15_000,
            step=20,
        )
        await rig.advance(100, step=20)
        assert rig.state.state is TurnState.THINKING, (
            f"premise: the check is in flight; {rig.state.state}"
        )


async def _close_window(rig: MicRig, llm: BudgetLlm, window: str, gate: asyncio.Event) -> None:
    gate.set()
    if window == "prefill":
        await rig.end_of_turn()
    if window == "completion_check":
        llm.release_checks()
    await rig.until(
        lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= 1,
        f"premise: t1 never ended ({window})",
        limit_ms=40_000,
        step=40,
    )
    await rig.advance(300, step=40)


def _window_rig(tmp_path: Path, window: str, gate: asyncio.Event) -> Any:
    """``prefill``: a voice turn answering with no call. ``between_steps``: step 1 clicks (blocked on ``gate``), step 2
    answers. ``completion_check``: step 1 clicks, step 2 claims with no call, the check (held) confirms it."""
    browser = browser_of()
    if window == "prefill":
        return settings_rig(tmp_path, script=[answer(ANSWER_1)], browser=browser)

    async def blocked(fb: FakeLiveBrowser, inv: ToolInvocation) -> ToolOutcome:
        await gate.wait()
        return fb.perform(inv)

    if window == "between_steps":
        browser.handlers[CID] = blocked
    kw = {"completion_check": True} if window == "completion_check" else {}
    return settings_rig(
        tmp_path, script=[answer(tsl.S1), tool(VALID_CLICK)], browser=browser, session_kwargs=kw
    )


@pytest.mark.parametrize("window", WINDOWS)
async def test_mid_turn_set_leaves_the_turn_on_one_revision(tmp_path, window):
    """A2a-3 (adversarial, the task's most likely failure): a set of ``user_instructions`` and ``visual_tokens``
    arriving after t1's prefill started and before its generate, between its tool steps 1 and 2, or while its
    completion check is in flight leaves **every** request of t1 with t1's first system message (byte for byte, as
    the real client renders it) and ``max_soft_tokens`` 1120; it is answered at once with a state that still has rev
    0 in force and the change pending; t2's every request, its prefill first, carries the new system message (ending
    with the block) and 560. The set is applied after the turn (rev 1, source ``ui``)."""
    require_set()
    gate = asyncio.Event()
    async with _window_rig(tmp_path, window, gate) as (rig, llm, browser):
        if window != "prefill":
            llm.script_steps([answer(tsl.S3)])
            if window == "completion_check":
                llm.script_checks([answer(CHECK_SAYS)])
        await _open_window(rig, llm, browser, window, gate)
        mark = await set_and_pump(rig, {"user_instructions": CANARY, "visual_tokens": NEW_BUDGET})
        reply = last_state(rig, mark, " (mid-turn)")
        assert not reply.refused, (
            f"T-0213 A2a-3: a valid set mid-turn is not refused; {reply.refused}"
        )
        vt = item(reply, "visual_tokens")
        assert vt.value == OLD_BUDGET and vt.pending == NEW_BUDGET, (
            f"T-0213 A2a-3 / A2r-1: mid-turn the change is pending, not in force; visual_tokens {vt}"
        )
        await _close_window(rig, llm, window, gate)
        after = last_state(rig, mark, " (after the turn)")
        assert after.rev == 1 and item(after, "visual_tokens").value == NEW_BUDGET, (
            f"T-0213 A2a-3: applied at the boundary; rev {after.rev}, {item(after, 'visual_tokens')}"
        )
        t1 = llm.of_turn("t1")
        assert t1, "premise: t1 sent requests"
        base_json = t1[0].system_json
        kinds = {s.kind for s in t1}
        want_kinds = {
            "prefill": {"prefill"},
            "between_steps": {"step"},
            "completion_check": {"check"},
        }[window]
        assert want_kinds <= kinds, f"premise: t1's {window} request was recorded; kinds {kinds}"
        assert_one_regime(t1, system_json=base_json, budget=OLD_BUDGET, what=f"t1 ({window})")
        assert CANARY not in t1[0].system_text, "premise: t1 starts on the old system message"
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        t2 = llm.of_turn("t2")
    assert t2 and t2[0].kind == "prefill", (
        f"premise: t2 opens with its prefill; {[s.kind for s in t2]}"
    )
    assert t2[0].system_text.endswith(block(CANARY)), (
        f"T-0213 A2a-3: the next turn's first request has the new system message; {t2[0].system_text[-200:]!r}"
    )
    assert t2[0].system_text.startswith(t1[0].system_text), (
        "T-0213 A1-4: the block is appended after everything else, which stays as it was"
    )
    assert_one_regime(t2, system_json=t2[0].system_json, budget=NEW_BUDGET, what="t2")
    assert_thinking_on(t1 + t2)


async def test_visual_tokens_change_moves_the_budget_on_the_next_turn_only(tmp_path):
    """A2a-3/A2a-4: ``visual_tokens`` 1120 -> 560 sent while t1's click executes: every t1 request carries 1120, every
    t2 request 560, and the system message is byte-identical across both turns (the budget is not prompt text)."""
    require_set()
    gate = asyncio.Event()
    async with _window_rig(tmp_path, "between_steps", gate) as (rig, llm, browser):
        llm.script_steps([answer(tsl.S3)])
        await _open_window(rig, llm, browser, "between_steps", gate)
        await set_and_pump(rig, {"visual_tokens": NEW_BUDGET})
        await _close_window(rig, llm, "between_steps", gate)
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        t1, t2 = llm.of_turn("t1"), llm.of_turn("t2")
    assert_one_regime(t1, system_json=t1[0].system_json, budget=OLD_BUDGET, what="t1")
    assert_one_regime(t2, system_json=t1[0].system_json, budget=NEW_BUDGET, what="t2")


# =====================================================================================================
# A2a-3: stop_turn
# =====================================================================================================


async def test_stop_turn_stops_then_applies(tmp_path):
    """A2a-3: with ``stop_turn=true`` while t1's click executes, the session submits T-0193's stop (one ``UserStop``
    for t1 stepped, the turn ends IDLE with ``ended_by: "stopped"``), t1 sends nothing more, and the change is then
    applied (rev 1); t2 carries the new budget."""
    require_set()
    gate = asyncio.Event()
    async with _window_rig(tmp_path, "between_steps", gate) as (rig, llm, browser):
        llm.script_steps([answer(tsl.S3)])
        await _open_window(rig, llm, browser, "between_steps", gate)
        before = len(llm.of_turn("t1"))
        since = len(rig.machine.log)
        mark = await set_and_pump(rig, {"visual_tokens": NEW_BUDGET}, stop_turn=True)
        await rig.advance(500, step=40)
        stops = rig.stepped("user.stop", since)
        assert [getattr(e, "turn_id", None) for e in stops] == ["t1"], (
            f"T-0213 A2a-3: stop_turn submits the astra.turn.stop stop (UserStop for t1); stepped {stops}"
        )
        assert rig.state.state is TurnState.IDLE, (
            f"T-0213 A2a-3: the stop ended t1; {rig.state.state}"
        )
        gate.set()
        await rig.advance(1_000, step=40)
        assert len(llm.of_turn("t1")) == before, "T-0213 A2a-3: the stopped turn sent nothing more"
        after = last_state(rig, mark)
        assert after.rev == 1 and item(after, "visual_tokens").value == NEW_BUDGET, (
            f"T-0213 A2a-3: applied after the stop; rev {after.rev}"
        )
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        t2 = llm.of_turn("t2")
        lines = rig.raw_lines()
    assert all(s.mm == mm(NEW_BUDGET) for s in t2), [s.mm for s in t2]
    t1_line = [x for x in lines if x["turn_id"] == "t1"]
    assert t1_line and t1_line[0]["extras"].get("ended_by") == "stopped", t1_line


async def test_r23_stop_and_apply_resending_the_pending_change_stops_the_turn_once(tmp_path):
    """R-23 (the page's "Stop and apply" resends the pending changes): a set waits mid-turn (t1's click executing);
    a second set with the same changes and ``stop_turn=true`` is a no-op on top of the pending snapshot, yet it
    submits exactly one ``UserStop`` for t1 and the change applies at IDLE with ``applied_at == "after_stop"``; a
    third identical ``stop_turn`` set while that stop is already submitted submits no second stop. Afterwards, at IDLE
    with nothing pending, a no-op ``stop_turn`` set stops nothing."""
    require_set()
    gate = asyncio.Event()
    async with _window_rig(tmp_path, "between_steps", gate) as (rig, llm, browser):
        llm.script_steps([answer(tsl.S3)])
        await _open_window(rig, llm, browser, "between_steps", gate)
        since = len(rig.machine.log)
        changes = {"visual_tokens": NEW_BUDGET}
        mark = await set_and_pump(rig, changes)
        pending = last_state(rig, mark, " (the first set)")
        assert item(pending, "visual_tokens").pending == NEW_BUDGET, "premise: the change waits"
        assert rig.stepped("user.stop", since) == [], "premise: a plain set stops nothing"
        await set_and_pump(rig, changes, stop_turn=True)
        await set_and_pump(rig, changes, stop_turn=True)
        await rig.advance(500, step=40)
        stops = rig.stepped("user.stop", since)
        assert [getattr(e, "turn_id", None) for e in stops] == ["t1"], (
            f"T-0213 R-23: one UserStop for t1 across two Stop and apply sets; stepped {stops}"
        )
        assert rig.state.state is TurnState.IDLE, f"R-23: the stop ended t1; {rig.state.state}"
        gate.set()
        await rig.advance(1_000, step=40)
        after = last_state(rig, mark)
        assert after.rev == 1 and item(after, "visual_tokens").value == NEW_BUDGET, after.rev
        idle_since = len(rig.machine.log)
        await set_and_pump(rig, changes, rev=1, stop_turn=True)
        await rig.advance(300, step=40)
        idle_stops = rig.stepped("user.stop", idle_since)
        idle_moved = [
            (b, a)
            for b, ev, a in rig.machine.transitions()[idle_since:]
            if ev.type == "user.stop" and b != a
        ]
    changes_logged = rig.kind("settings.change")
    assert [d.get("applied_at") for d in changes_logged][:1] == ["after_stop"], changes_logged
    assert idle_stops == [] or idle_moved == [], (
        f"R-23: a no-op stop_turn set with nothing pending stops nothing; {idle_stops}"
    )
    assert idle_moved == [], idle_moved


@pytest.mark.parametrize(
    "changes",
    [{}, {"visual_tokens": OLD_BUDGET}],
    ids=["empty", "same-value"],
)
async def test_r26c_a_noop_stop_turn_set_mid_turn_with_nothing_pending_stops_nothing(
    tmp_path, changes
):
    """R-26 (c), R-23's guard on the pending list: t1's click executing (a turn in flight), **nothing pending**, a
    no-op set (``{}``, or ``visual_tokens`` at the value in force, 1120) with ``stop_turn=true`` stops nothing (no
    ``UserStop`` is stepped, the machine stays THINKING), raises nothing, and is still answered with a state at rev
    0 that has nothing pending and nothing refused. The turn then runs to its own end: not ``ended_by: "stopped"``,
    and its step 2 is sent."""
    require_set()
    gate = asyncio.Event()
    async with _window_rig(tmp_path, "between_steps", gate) as (rig, llm, browser):
        llm.script_steps([answer(tsl.S3)])
        await _open_window(rig, llm, browser, "between_steps", gate)
        since = len(rig.machine.log)
        mark = send_set(rig, changes, stop_turn=True)  # an exception here is the failure
        await rig.pump()
        await rig.advance(300, step=40)
        stops = rig.stepped("user.stop", since)
        in_flight = rig.state.state
        replies = states(rig.published.messages[mark:])
        assert replies, (
            "R-23: a no-op stop_turn set mid-turn with nothing pending is still answered with a state; none was "
            "published (the handler raised before its echo, e.g. on an empty pending list)"
        )
        reply = replies[-1]
        await _close_window(rig, llm, "between_steps", gate)
        t1_steps = [s for s in llm.of_turn("t1") if s.kind == "step"]
        lines = rig.raw_lines()
    assert stops == [], (
        f"R-23: a no-op stop_turn set with nothing pending stops nothing; stepped {stops}"
    )
    assert in_flight is TurnState.THINKING, (
        f"R-23: t1 is still in flight after the set; {in_flight}"
    )
    assert reply.rev == 0 and not reply.refused, (reply.rev, reply.refused)
    assert item(reply, "visual_tokens").pending is None, item(reply, "visual_tokens")
    assert t1_steps, "R-23: the unstopped t1 went on to step 2"
    t1_line = [x for x in lines if x["turn_id"] == "t1"]
    assert t1_line and t1_line[0]["extras"].get("ended_by") != "stopped", t1_line


async def test_r26d_two_stop_and_apply_presses_before_the_stop_is_stepped_submit_one_stop(
    tmp_path,
):
    """R-26 (d), R-23's "a second Stop and apply while the stop is already submitted submits nothing more": a set
    waits mid-turn (t1's click executing); then two identical ``stop_turn`` sets arrive **before a pump**, so the
    first's ``UserStop`` has not yet brought the machine to IDLE when the second is handled. Exactly one
    ``UserStop`` (for t1) is stepped across both; the change then applies (rev 1, 560)."""
    require_set()
    gate = asyncio.Event()
    async with _window_rig(tmp_path, "between_steps", gate) as (rig, llm, browser):
        llm.script_steps([answer(tsl.S3)])
        await _open_window(rig, llm, browser, "between_steps", gate)
        since = len(rig.machine.log)
        changes = {"visual_tokens": NEW_BUDGET}
        mark = await set_and_pump(rig, changes)
        assert item(last_state(rig, mark), "visual_tokens").pending == NEW_BUDGET, (
            "premise: the change waits"
        )
        assert rig.state.state is TurnState.THINKING, "premise: t1 in flight"
        send_set(rig, changes, stop_turn=True)
        between = rig.state.state
        send_set(rig, changes, stop_turn=True)
        await rig.pump()
        await rig.advance(500, step=40)
        stops = rig.stepped("user.stop", since)
        gate.set()
        await rig.advance(1_000, step=40)
        after = last_state(rig, mark)
    assert between is not TurnState.IDLE, (
        f"premise: the second press is handled before the first stop reached IDLE; {between}"
    )
    assert [getattr(e, "turn_id", None) for e in stops] == ["t1"], (
        f"T-0213 R-23: two presses before the stop is stepped submit one UserStop; stepped {stops}"
    )
    assert after.rev == 1 and item(after, "visual_tokens").value == NEW_BUDGET, after.rev


async def test_stop_turn_at_idle_applies_at_once_and_stops_nothing(tmp_path):
    """A2a-3 (the guard's other side): ``stop_turn=true`` at IDLE applies at once and steps no stop that acts (the
    machine stays IDLE; nothing is cut)."""
    require_set()
    async with settings_rig(tmp_path) as (rig, _llm, _browser):
        since = len(rig.machine.log)
        mark = await set_and_pump(rig, {"visual_tokens": NEW_BUDGET}, stop_turn=True)
        state = last_state(rig, mark)
        moved = [
            (b, a)
            for b, ev, a in rig.machine.transitions()[since:]
            if ev.type == "user.stop" and b != a
        ]
    assert state.rev == 1 and item(state, "visual_tokens").value == NEW_BUDGET, state.rev
    assert moved == [], f"T-0213 A2a-3: a stop at IDLE moves nothing; {moved}"


# =====================================================================================================
# A2a-2: refusals change nothing
# =====================================================================================================


@pytest.mark.parametrize("rev", [1, 7, -1], ids=["ahead", "far-ahead", "negative"])
async def test_stale_rev_refused(tmp_path, rev):
    """A2a-2: a set whose ``rev`` is not the current one (0) is refused with ``refused=[{key: "*", reason:
    "stale"}]``, the rev stays 0, and the next turn's requests are exactly as before (1120, no block)."""
    require_set()
    async with settings_rig(tmp_path) as (rig, llm, _browser):
        mark = await set_and_pump(
            rig, {"visual_tokens": NEW_BUDGET, "user_instructions": CANARY}, rev=rev
        )
        state = last_state(rig, mark)
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        t1 = llm.of_turn("t1")
        settings = rig.session.settings
    assert refused_pairs(state) == [("*", "stale")], refused_pairs(state)
    assert state.rev == 0 and settings.rev == 0, (state.rev, settings.rev)
    assert (
        item(state, "visual_tokens").value == OLD_BUDGET
        and item(state, "visual_tokens").pending is None
    )
    assert all(s.mm == mm(OLD_BUDGET) and CANARY not in s.system_text for s in t1), [
        s.mm for s in t1
    ]


async def test_old_rev_after_a_change_is_stale(tmp_path):
    """A2a-2: after one set moved the rev to 1, a second set still naming rev 0 is stale and changes nothing; the
    same set naming rev 1 is accepted (rev 2). Both sides of the guard."""
    require_set()
    async with settings_rig(tmp_path) as (rig, _llm, _browser):
        await set_and_pump(rig, {"visual_tokens": NEW_BUDGET})
        mark = await set_and_pump(rig, {"visual_tokens": 280}, rev=0)
        stale = last_state(rig, mark)
        mark = await set_and_pump(rig, {"visual_tokens": 280}, rev=1)
        fresh = last_state(rig, mark)
    assert refused_pairs(stale) == [("*", "stale")] and stale.rev == 1, (
        refused_pairs(stale),
        stale.rev,
    )
    assert item(stale, "visual_tokens").value == NEW_BUDGET
    assert not fresh.refused and fresh.rev == 2 and item(fresh, "visual_tokens").value == 280, (
        fresh.rev
    )


def _validate_oracle(rig: MicRig, changes: dict[str, Any]) -> list[tuple[str, str]]:
    from local_astra.conductor.live_settings import validate

    web = bool(getattr(rig.kit, "web", False))
    snap, refusals = validate(rig.session.settings, changes, mode="live", web=web)
    assert snap is None, f"premise: {changes} is invalid"
    return [(r.key, r.reason) for r in refusals]


INVALID = {
    "outcome-without-feedback": {"action_outcome": True, "live_feedback": "off"},
    "special-token": {"user_instructions": "be brief <|think|> now"},
    "unknown-key": {"colour_scheme": "dark"},
    "relaunch-key": {"browser_web": False},
    "thinking": {"thinking": False},
    "strict-int": {"visual_tokens": 560.0},
    "one-bad-one-good": {"visual_tokens": NEW_BUDGET, "repeat_guard": "yes"},
}


@pytest.mark.parametrize("case", list(INVALID))
async def test_invalid_set_refused_with_validate_reasons(tmp_path, case):
    """A2a-2 / A1-3: an invalid set is refused with ``validate``'s own ``(key, reason)`` pairs, atomically (in the
    mixed case the valid half is not applied either); the rev stays 0 and the next turn is as before."""
    require_set()
    changes = INVALID[case]
    async with settings_rig(tmp_path) as (rig, llm, _browser):
        want = _validate_oracle(rig, changes)
        mark = await set_and_pump(rig, changes)
        state = last_state(rig, mark)
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        t1 = llm.of_turn("t1")
        rev = rig.session.settings.rev
    if case == "unknown-key":
        # R-22 (re-pin): under R-19 (d) a refusal's key on the wire is "*" for a key no catalog row names, so the
        # page's own key never comes back; validate's reason is carried as it is
        assert [k for k, _ in want] == ["colour_scheme"], f"oracle premise: {want}"
        want = [("*", r) for _, r in want]
        assert not any(
            "colour_scheme" in k or "colour_scheme" in r for k, r in refused_pairs(state)
        )
    assert sorted(refused_pairs(state)) == sorted(want), (refused_pairs(state), want)
    if case == "outcome-without-feedback":
        assert any(action_outcome_without_feedback_refusal() in r for _, r in want), (
            "oracle premise"
        )
    assert state.rev == 0 and rev == 0, (state.rev, rev)
    assert all(s.mm == mm(OLD_BUDGET) for s in t1) and all(
        "<|think|> now" not in s.system_text for s in t1
    )
    assert_thinking_on(t1)


# =====================================================================================================
# A2a-4 / A2a-7: what survives, what never happens
# =====================================================================================================


async def _after_one_turn_then_set(tmp_path: Path, changes: dict[str, Any]) -> dict[str, Any]:
    """t1 (typed, answered), then the set at IDLE, then 2 s; then t2. What the rows read."""
    async with settings_rig(tmp_path) as (rig, llm, browser):
        forgets: list[int] = []
        real_forget = rig.memory.forget_since

        async def spy(t0_ms: int) -> int:
            forgets.append(t0_ms)
            return int(await real_forget(t0_ms))

        rig.memory.forget_since = spy  # type: ignore[method-assign]
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        history = [(h.user_text, h.assistant_spoken) for h in rig.state.history]
        frame = rig.session._frame
        episodes = dict(rig.memory.episodes)
        calls, generation = len(browser.calls), browser.generation
        log_mark, plays = len(rig.machine.log), len(rig.audio.plays)
        pub_mark = len(rig.published.messages)
        mark = await set_and_pump(rig, changes)
        state = last_state(rig, mark)
        await rig.advance(2_000, step=40)
        out = {
            "state": state,
            "history_before": history,
            "history_after": [(h.user_text, h.assistant_spoken) for h in rig.state.history],
            "frame_kept": rig.session._frame is frame,
            "episodes_kept": dict(rig.memory.episodes) == episodes,
            "browser_calls": browser.calls[calls:],
            "generation_kept": browser.generation == generation and not browser.closed,
            "speaks": rig.commands_of("tts.speak", log_mark),
            "plays": rig.audio.plays[plays:],
            "published": [getattr(m, "type", None) for m in rig.published.messages[pub_mark:]],
        }
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        out["t2"] = llm.of_turn("t2")
        out["forgets"] = list(forgets)
        out["history_final"] = [(h.user_text, h.assistant_spoken) for h in rig.state.history]
    return out


async def test_a_change_keeps_history_memory_frames_and_the_browser(tmp_path):
    """A2a-4 / H-T213-1 ("keep everything"): a set applied at IDLE after t1 keeps the history (t1's entry, and t2's
    request still carries t1's question and answer), the memory (``forget_since`` called 0 times; the episodes as they
    were), the session's frame, and the agent browser (not opened, closed or driven; same generation). Nothing is
    published as a reset."""
    require_set()
    got = await _after_one_turn_then_set(
        tmp_path, {"visual_tokens": NEW_BUDGET, "user_instructions": CANARY}
    )
    assert got["state"].rev == 1, "premise: the set was applied"
    assert got["history_after"] == got["history_before"] and got["history_before"], got[
        "history_after"
    ]
    assert got["forgets"] == [], (
        f"T-0213 A2a-7: memory is untouched (forget_since never called); {got['forgets']}"
    )
    assert got["episodes_kept"], "T-0213 A2a-4: the memory's episodes are as they were"
    assert got["frame_kept"], "T-0213 A2a-4: the session's frame is kept"
    assert not [c for c in got["browser_calls"] if c[0] in ("open", "close", "execute")], got[
        "browser_calls"
    ]
    assert got["generation_kept"], "T-0213 A2a-4: the browser and its page are kept"
    assert "astra.session.cleared" not in got["published"], got["published"]
    t2_texts = " ".join(s.system_text for s in got["t2"])
    assert t2_texts, "premise: t2 sent requests"
    assert got["history_final"][0] == got["history_before"][0], got["history_final"]


async def test_a_change_speaks_nothing_and_adds_no_history(tmp_path):
    """A2a-7: applying a set emits no ``TtsSpeak``, plays no clip or utterance, and adds no history entry."""
    require_set()
    got = await _after_one_turn_then_set(
        tmp_path, {"visual_tokens": NEW_BUDGET, "user_instructions": CANARY}
    )
    assert got["state"].rev == 1, "premise: the set was applied"
    assert got["speaks"] == [], (
        f"T-0213 A2a-7: no TtsSpeak is caused by a settings change; {got['speaks']}"
    )
    assert got["plays"] == [], f"T-0213 A2a-7: nothing is played (no clip); {got['plays']}"
    assert len(got["history_after"]) == len(got["history_before"]), got["history_after"]
    assert len(got["history_final"]) == 2, f"t1 and t2 only; {got['history_final']}"


async def test_instructions_reset_gives_base_bytes(tmp_path):
    """A2a-4 / A1-4: ``user_instructions`` applied, then reset to ``""``: t2's system message ends with the block, and
    t3's is byte-identical to t1's (BASE)."""
    require_set()
    async with settings_rig(tmp_path) as (rig, llm, _browser):
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        await set_and_pump(rig, {"user_instructions": CANARY})
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        mark = await set_and_pump(rig, {"user_instructions": ""}, rev=1)
        reset = last_state(rig, mark)
        await typed_turn(rig, "and now", "Still the same. ", 3)
        t1, t2, t3 = llm.of_turn("t1"), llm.of_turn("t2"), llm.of_turn("t3")
    assert reset.rev == 2 and reset.instructions == "", (reset.rev, reset.instructions)
    assert t2 and all(s.system_text.endswith(block(CANARY)) for s in t2), (
        "the block is in force in t2"
    )
    assert_one_regime(
        t3, system_json=t1[0].system_json, budget=OLD_BUDGET, what="t3 (after the reset)"
    )


async def test_thinking_stays_on_and_first_across_a_change(tmp_path):
    """A2a-6: a set that changes the system message keeps ``thinking`` on (the template kwarg, which renders
    ``<|think|>`` as the first system token) on every request before and after it, and the system message never
    carries the ``<|think|>`` literal itself."""
    require_set()
    # R-18: the guidance change is applied only with a guidance table, built as cli builds it (the rig's live
    # session has no web mode and no wait tool)
    table = guidance_table_of(False, False)
    async with settings_rig(tmp_path, guidance_table=table) as (rig, llm, _browser):
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        mark = await set_and_pump(rig, {"user_instructions": CANARY, "live_guidance": "v2"})
        applied = last_state(rig, mark)
        assert applied.rev == 1 and not applied.refused, (
            f"premise: the set was applied; rev {applied.rev}, refused {refused_pairs(applied)}"
        )
        assert item(applied, "live_guidance").value == "v2", item(applied, "live_guidance")
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        sent = llm.of_turn("t1") + llm.of_turn("t2")
    assert_thinking_on(sent)
    assert not [s.engine_id for s in sent if "<|think|>" in s.system_text], (
        "A2a-6: <|think|> rides on the kwarg"
    )


async def test_a_guidance_change_without_a_guidance_table_is_refused_with_a_reason(tmp_path):
    """R-18: a session given no guidance table cannot carry out a ``live_guidance`` change, so the set is refused with
    a reason naming ``live_guidance`` (rev stays 0, nothing pending), never silently accepted; the next turn's system
    message is t1's byte for byte."""
    require_set()
    async with settings_rig(tmp_path) as (rig, llm, _browser):
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        mark = await set_and_pump(rig, {"live_guidance": "v2"})
        state = last_state(rig, mark, " (refused)")
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        t1, t2 = llm.of_turn("t1"), llm.of_turn("t2")
    pairs = refused_pairs(state)
    assert [k for k, _ in pairs] == ["live_guidance"] and all(r.strip() for _, r in pairs), (
        f"R-18: the guidance change is refused with a reason; refused {pairs}"
    )
    lg = item(state, "live_guidance")
    assert state.rev == 0 and lg.pending is None and lg.value != "v2", (state.rev, lg)
    assert_one_regime(t1 + t2, system_json=t1[0].system_json, budget=OLD_BUDGET, what="t1 and t2")


async def test_without_a_set_nothing_moves(tmp_path):
    """A2a-7 (negative; green before and after A2-apply): with no ``astra.settings.set``, a session publishes no
    settings state outside the hello, and two turns carry one system message and 1120 on every request."""
    async with settings_rig(tmp_path) as (rig, llm, _browser):
        mark = len(rig.published.messages)
        await typed_turn(rig, QUESTION_1, ANSWER_1, 1)
        await typed_turn(rig, QUESTION_2, ANSWER_2, 2)
        published = states(rig.published.messages[mark:])
        t1, t2 = llm.of_turn("t1"), llm.of_turn("t2")
    assert published == [], f"A2a-7: no settings state without a set; {published}"
    assert_one_regime(t1 + t2, system_json=t1[0].system_json, budget=OLD_BUDGET, what="t1 and t2")
    assert_thinking_on(t1 + t2)


# =====================================================================================================
# A2a-4: ReGround off -> on (reading AP-3)
# =====================================================================================================


def _budget_of(client: Any, made: Sequence[tuple[Any, dict[str, Any], Any]]) -> Any:
    own = getattr(client, "visual_tokens", None)
    if own is not None:
        return own
    for _, kwargs, built in made:
        if built is client:
            return kwargs.get("visual_tokens")
    return None


@pytest.mark.parametrize(
    "changes",
    [
        {"reground": True, "overview_frame": False},
        {"reground": True, "overview_frame": False, "reground_visual_tokens": CROP},
    ],
    ids=["panel-pair", "explicit-crop"],
)
def test_reground_off_to_on_switches_to_a_pass2_client_at_crop_280(
    tmp_path, capsys, monkeypatch, changes
):
    """A2a-4 (reading AP-3): a ``cli``-built live session with ReGround off, given ReGround on with the overview frame
    off (R-4) at IDLE, ends with ReGround on, the crop budget 280 (T-0133 A5 at the 1120 main budget) and a pass-2
    client other than the main one whose budget is 280. The main client's budget stays 1120.

    The row is synchronous: ``cli`` runs its own ``asyncio.run`` (``build`` returns once the graph is recorded), so the
    session is started afterwards in a loop of the row's own. The rig is proven (the session starts, the page is
    greeted and holds rev 0 with ReGround off) before the contract is asked for, so a setup error is never read as
    "the set does not exist yet"."""
    from local_astra.contracts.codec import encode
    from local_astra.contracts.protocol import AstraHello
    from tests.unit.test_cli_t0132 import build
    from tests.unit.test_t0151_cli import LIVE_PROFILE

    calls, built = build(tmp_path, capsys, monkeypatch, body=LIVE_PROFILE)
    graph = calls.graphs[-1]
    session = graph.session
    assert session.reground is False, "premise: ReGround is off at start"

    async def body() -> None:
        await session.start()
        try:
            page = graph.server.open()
            page.receive(encode(AstraHello()))
            hello = states(page.sent())
            assert hello and hello[-1].rev == 0 and item(hello[-1], "reground").value is False, (
                f"premise: the greeted page holds rev 0 with ReGround off; {[(s.rev,) for s in hello]}"
            )
            require_set()
            page.receive(json.dumps({"type": SET, "rev": 0, "changes": changes}))
            for _ in range(
                200
            ):  # the cli session's pump runs on the real clock: up to 2 s for the echo
                if [s for s in states(page.sent()) if s.rev == 1 or s.refused]:
                    break
                await asyncio.sleep(0.01)
            found = states(page.sent())
            assert found and found[-1].rev == 1, (
                f"T-0213 A2a-3: applied at IDLE; states {[(s.rev, refused_pairs(s)) for s in found]}"
            )
            assert session.reground is True, "T-0213 A2a-4: ReGround is on"
            assert session.reground_visual_tokens == CROP, session.reground_visual_tokens
            second = session.reground_llm
            assert second is not None and second is not session.llm, (
                "T-0213 A2a-4: pass 2 has its own client (the crop budget differs from the main one)"
            )
            assert _budget_of(second, built.made) == CROP, (
                f"T-0213 A2a-4: the pass-2 client sends 280; got {_budget_of(second, built.made)}"
            )
            assert _budget_of(session.llm, built.made) == OLD_BUDGET, "the main budget is unchanged"
        finally:
            await session.aclose()

    asyncio.run(body())


# =====================================================================================================
# R-20 / R-18: a change this session cannot carry out is refused; one it can is carried out
# =====================================================================================================


async def test_reground_on_without_a_pass2_factory_is_refused(tmp_path):
    """R-18: the rig's session has no pass-2 client and no factory, so ReGround on (whose crop is 280 at the 1120 main
    budget, T-0133 A5, so it needs a second client) is refused with a reason, never accepted and silently run
    through the main client: rev 0, nothing pending, ReGround off, no pass-2 client."""
    require_set()
    async with settings_rig(tmp_path) as (rig, _llm, _browser):
        mark = await set_and_pump(rig, {"reground": True, "overview_frame": False})
        state = last_state(rig, mark, " (refused)")
        session = rig.session
        got = (session.reground, session.reground_llm, session.settings.rev)
    pairs = refused_pairs(state)
    assert pairs and all(r.strip() for _, r in pairs), (
        f"T-0213 R-18: ReGround on with no pass-2 client is refused with a reason; refused {pairs}"
    )
    assert state.rev == 0 and item(state, "reground").pending is None, (
        state.rev,
        item(state, "reground"),
    )
    assert got == (False, None, 0), (
        f"T-0213 R-18: nothing was applied; (reground, reground_llm, rev) {got}"
    )


def _page_said(prompt: Prompt) -> bool:
    return any("new page loaded" in m.text() for m in prompt.messages)


async def test_a_guidance_change_with_a_table_moves_the_text_and_the_page_fact(tmp_path):
    """R-18 / A2a-4 (the applied side of the refusal row above): with ``cli``'s guidance table (no web, no wait
    tool), ``live_guidance`` -> ``v2`` applied after t1 makes t2's system message carry v2's guidance text (which
    t1's did not) and switches on v2's page fact (T-0174 B2-1): t2's step after a click that navigated says "new page
    loaded", t1's step after the same click (the rig's ``page_fact`` off) did not."""
    require_set()
    table = guidance_table_of(False, False)
    v2_text, v2_fact = table["v2"]
    assert v2_text and v2_fact is True, "premise: cli's v2 has guidance text and the page fact on"
    browser = browser_of()
    browser.resolved_extra[tsl.call_id(1, "t1")] = {"navigated": True}
    browser.resolved_extra[tsl.call_id(1, "t2")] = {"navigated": True}
    script = [answer(tsl.S1), tool(VALID_CLICK)]
    async with settings_rig(tmp_path, script=script, browser=browser, guidance_table=table) as (
        rig,
        llm,
        _browser,
    ):

        async def click_turn(question: str, n_lines: int) -> None:
            rig.llm.script = list(script)
            llm.script_steps([answer(tsl.S3)])
            await rig.type_question(question)
            await rig.until(
                lambda: rig.state.state is TurnState.IDLE and len(rig.lines()) >= n_lines,
                f"premise: the click turn {question!r} never ended",
                limit_ms=20_000,
                step=40,
            )
            await rig.advance(300, step=40)

        await click_turn(QUESTION_1, 1)
        t1_steps = [p for rid, p in llm.step_prompts if "-t1-" in rid]
        mark = await set_and_pump(rig, {"live_guidance": "v2"})
        applied = last_state(rig, mark)
        assert applied.rev == 1 and not applied.refused, (applied.rev, refused_pairs(applied))
        await click_turn(QUESTION_2, 2)
        t2_steps = [p for rid, p in llm.step_prompts if "-t2-" in rid]
        t1, t2 = llm.of_turn("t1"), llm.of_turn("t2")
    assert t1_steps and t2_steps, "premise: both turns sent a step after the click"
    assert v2_text not in t1[0].system_text, "premise: t1 is not on v2's guidance"
    assert all(v2_text in s.system_text for s in t2), (
        "T-0213 A2a-4: the next turn's every request carries the new guidance text"
    )
    assert not any(_page_said(p) for p in t1_steps), "premise: t1 ran with the page fact off"
    assert any(_page_said(p) for p in t2_steps), (
        "T-0213 A2a-4 / R-18: the guidance version's page fact is on in the next turn"
    )


# =====================================================================================================
# R-20 / R-13a: the pass-2 client's lifecycle
# =====================================================================================================


class ClosableLlm(BudgetLlm):
    """A pass-2 client that counts ``aclose`` calls."""

    def __init__(self, *, visual_tokens: int | None = CROP) -> None:
        super().__init__([], visual_tokens=visual_tokens)
        self.closes = 0

    async def aclose(self) -> None:
        self.closes += 1


@asynccontextmanager
async def factory_rig(tmp_path: Path) -> AsyncIterator[tuple[MicRig, list[ClosableLlm]]]:
    """The settings rig with a pass-2 factory handed in as ``cli`` hands it (R-13a); ``made`` lists what it built."""
    made: list[ClosableLlm] = []

    def factory() -> ClosableLlm:
        made.append(ClosableLlm())
        return made[-1]

    async with settings_rig(tmp_path) as (rig, _llm, _browser):
        rig.session.init_settings(rig.session.settings, {}, {}, pass2_factory=factory)
        yield rig, made


async def test_aclose_closes_the_lazily_built_pass2_client(tmp_path):
    """R-13a: a pass-2 client the factory builds at the boundary of a ReGround on change is the session's to close:
    ``aclose`` closes it exactly once. The other side: a session given the factory and never changed builds nothing
    (BASE's clients only)."""
    require_set()
    async with factory_rig(tmp_path / "idle") as (_rig, unused):
        pass
    assert unused == [], "T-0213 R-13a: a session nobody changes never calls the factory"
    async with factory_rig(tmp_path / "on") as (rig, made):
        mark = await set_and_pump(rig, {"reground": True, "overview_frame": False})
        assert last_state(rig, mark).rev == 1, refused_pairs(last_state(rig, mark))
        assert len(made) == 1 and rig.session.reground_llm is made[0], (
            "premise: the lazily built client is ReGround's pass-2 client"
        )
        assert made[0].closes == 0, "premise: not closed while the session runs"
    assert made[0].closes == 1, (
        f"T-0213 R-13a: aclose closes the lazily built pass-2 client once; closes {made[0].closes}"
    )


async def test_reground_on_then_off_clears_the_pass2_client(tmp_path):
    """A2a-4: ReGround on (pass 2 through the factory's client at 280), then off: ``reground`` false and
    ``reground_llm`` cleared (``None``), so no later turn can send a pass 2; the main client and its budget are
    unchanged throughout."""
    require_set()
    async with factory_rig(tmp_path) as (rig, made):
        session = rig.session
        main = session.llm
        await set_and_pump(rig, {"reground": True, "overview_frame": False})
        on = (session.reground, session.reground_llm, session.reground_visual_tokens)
        # M5 (A1): ReGround off clears its crop in the same set (``{reground: False}`` alone is refused)
        mark = await set_and_pump(rig, {"reground": False, "reground_visual_tokens": None}, rev=1)
        off_state = last_state(rig, mark)
        off = (session.reground, session.reground_llm)
        main_kept = session.llm is main and getattr(main, "visual_tokens", None) == OLD_BUDGET
    assert made and on == (True, made[0], CROP), (
        f"premise: ReGround on through the pass-2 client; {on}"
    )
    assert off_state.rev == 2 and not off_state.refused, (off_state.rev, refused_pairs(off_state))
    assert off == (False, None), (
        f"T-0213 A2a-4: ReGround off clears its pass-2 client; (reground, reground_llm) {off}"
    )
    assert main_kept, "the main client and its budget are unchanged"


#: R-21's reason text, as the ruling words it (docs/tasks/T-0213.md, Status, R-21)
PASS2_FAILED = "ReGround could not start its second pass"


class Pass2FactoryFailed(RuntimeError):
    """What the rig's pass-2 factory raises (a client that cannot be built: a bad URL, a refused port)."""


@pytest.mark.parametrize("when", ["idle", "mid_turn"])
async def test_a_pass2_factory_that_fails_at_the_boundary_refuses_the_set_once(tmp_path, when):
    """R-21 (adversarial: the failure mode is a set retried at every hop, or applied half-way): ReGround on with a
    pass-2 factory that raises when the boundary calls it. ``idle``: the set reaches a boundary at once.
    ``mid_turn``: the set waits while t1's step-1 click executes (answered as pending), and t1's end is the boundary.
    Either way the waiting set is refused with a reason containing "ReGround could not start its second pass", in
    exactly one published state that carries the refusal and nothing pending (rev 0), and in exactly one note; the
    factory is called once however many hops follow; nothing is applied (R-19 f: the settings snapshot, ReGround
    and its pass-2 client are as before); a later valid set still applies (rev 1)."""
    require_set()
    calls: list[int] = []

    def factory() -> Any:
        calls.append(1)
        raise Pass2FactoryFailed("the pass-2 client cannot be built")

    gate = asyncio.Event()
    rig_cm = (
        _window_rig(tmp_path, "between_steps", gate)
        if when == "mid_turn"
        else settings_rig(tmp_path)
    )
    async with rig_cm as (rig, llm, browser):
        session = rig.session
        session.init_settings(session.settings, {}, {}, pass2_factory=factory)
        before = (session.settings, session.reground, session.reground_llm)
        assert before[1] is False and before[2] is None, "premise: ReGround off, no pass-2 client"
        if when == "mid_turn":
            llm.script_steps([answer(tsl.S3)])
            gate.clear()
            await _open_window(rig, llm, browser, "between_steps", gate)
        mark = await set_and_pump(rig, {"reground": True, "overview_frame": False})
        if when == "mid_turn":
            waiting = last_state(rig, mark, " (mid-turn)")
            assert not waiting.refused and item(waiting, "reground").pending is True, (
                f"premise: mid-turn the set waits; refused {refused_pairs(waiting)}, "
                f"{item(waiting, 'reground')}"
            )
            assert calls == [], "premise: the factory is not called before the boundary"
            await _close_window(rig, llm, "between_steps", gate)
        # many more hops: a set left waiting would be retried at each one
        await rig.advance(2_000, step=20)
        answered = states(rig.published.messages[mark:])
        after = (session.settings, session.reground, session.reground_llm)
        calls_after_hops = len(calls)
        mark2 = await set_and_pump(rig, {"visual_tokens": NEW_BUDGET})
        later_found = states(rig.published.messages[mark2:])
        later_got = (session.settings.rev, session.settings.visual_tokens, len(calls))
    with_reason = [s for s in answered if any(PASS2_FAILED in r for _, r in refused_pairs(s))]
    assert len(with_reason) == 1, (
        f"T-0213 R-21: the waiting set is refused with {PASS2_FAILED!r} in one published state; "
        f"states since the set: {[(s.rev, refused_pairs(s)) for s in answered]}"
    )
    refused = with_reason[0]
    assert refused.rev == 0 and all(i.pending is None for i in refused.items), (
        f"T-0213 R-21: the refusal state has nothing pending and rev 0; rev {refused.rev}, "
        f"pending {[(i.key, i.pending) for i in refused.items if i.pending is not None]}"
    )
    assert answered[-1] is refused, (
        "T-0213 R-21: nothing after the refusal re-publishes the set (no retry at a later hop)"
    )
    assert calls_after_hops == 1, (
        f"T-0213 R-21: the failed factory is called once, not at every hop; called {calls_after_hops} times"
    )
    assert after == before, (
        f"T-0213 R-21 / R-19 (f): nothing is applied; (settings, reground, reground_llm) {before} -> {after}"
    )
    assert later_found, (
        "T-0213 R-21 / A2a-2: the later valid set is answered with a state; none was published"
    )
    later = later_found[-1]
    assert later.rev == 1 and not later.refused, (
        f"T-0213 R-21: a later valid set still applies; rev {later.rev}, refused {refused_pairs(later)}"
    )
    assert later_got == (1, NEW_BUDGET, 1), (
        f"T-0213 R-21: (rev, visual_tokens, factory calls) after the later set {later_got}"
    )
    notes = [
        r["data"]
        for r in rig.log_records
        if r.get("kind") == "note" and PASS2_FAILED in json.dumps(r.get("data"))
    ]
    assert len(notes) == 1, (
        f"T-0213 R-21: the refusal is noted once; notes carrying the reason {notes}"
    )
    assert notes[0].get("what") == "settings refused", (
        f"docs/02 A2-apply: a refusal writes note {{what: 'settings refused', refusals}}; {notes[0]}"
    )


# =====================================================================================================
# R-20 / R-6: the session passes its fixed values to validate
# =====================================================================================================


def test_a_cli_session_with_the_page_overview_refuses_the_overview_frame(
    tmp_path, capsys, monkeypatch
):
    """R-6 / A2a-2: a ``cli``-built live session with ``--page-overview`` (so the overview frame is off, T-0152) is
    given ``{overview_frame: true}``: the session hands its fixed ``page_overview`` to ``validate``, which refuses the
    pair (T-0151/T-0152: the overview frame is not used beside the page-overview tool); rev stays 0 and the overview
    frame stays off. The reason is ``validate``'s own (the oracle, called with ``fixed``)."""
    from local_astra.conductor.live_settings import validate
    from local_astra.contracts.codec import encode
    from local_astra.contracts.protocol import AstraHello
    from tests.unit.test_cli_t0132 import build
    from tests.unit.test_t0151_cli import LIVE_PROFILE

    calls, _built = build(tmp_path, capsys, monkeypatch, "--page-overview", body=LIVE_PROFILE)
    session = calls.graphs[-1].session
    graph = calls.graphs[-1]
    changes = {"overview_frame": True}
    assert session.settings.overview_frame is False, (
        f"premise (T-0152): the overview frame is off beside --page-overview; {session.settings.overview_frame}"
    )
    snap, want = validate(
        session.settings, changes, mode="live", web=False, fixed={"page_overview": True}
    )
    assert snap is None and want, (
        "oracle premise: validate refuses the pair given the fixed page overview"
    )
    loose, _ = validate(session.settings, changes, mode="live", web=False)
    assert loose is not None, (
        "oracle premise: without fixed the same set is accepted (so the row discriminates)"
    )

    async def body() -> list[Any]:
        await session.start()
        try:
            page = graph.server.open()
            page.receive(encode(AstraHello()))
            mark = len(page.sent())
            require_set()
            page.receive(json.dumps({"type": SET, "rev": 0, "changes": changes}))
            for _ in range(200):
                if states(page.sent()[mark:]):
                    break
                await asyncio.sleep(0.01)
            return states(page.sent()[mark:])
        finally:
            await session.aclose()

    found = asyncio.run(body())
    assert found, "T-0213 A2a-2: the set is answered"
    assert sorted(refused_pairs(found[-1])) == sorted((r.key, r.reason) for r in want), (
        f"T-0213 R-6: the session passes its fixed page overview; refused {refused_pairs(found[-1])}"
    )
    assert found[-1].rev == 0 and session.overview_frame is False, (
        found[-1].rev,
        session.overview_frame,
    )


# =====================================================================================================
# R-20: each sub-check of the boundary holds a set pending
# =====================================================================================================

BOUNDARY_HOLDS = ("execution", "generate", "prefill", "live_snapshot", "closing")


@pytest.mark.parametrize("hold", BOUNDARY_HOLDS)
async def test_each_boundary_check_holds_a_set(tmp_path, hold):
    """A2a-3 (one row per sub-check): with the machine IDLE and exactly one of the boundary's other conditions false
    (a tool execution's task running, a generate task not done, a prefill task not done, the live snapshot still
    taking its frame, or a turn line still closing), a valid set stays pending (its echo has rev 0 and the change
    pending, and the budget is not moved); once that one thing is over, the next pump hop applies it (rev 1).

    The held thing is planted in the session's own bookkeeping (each is a private container the boundary reads);
    the machine is IDLE throughout, so only that sub-check stands between the set and the boundary."""
    from types import SimpleNamespace

    require_set()
    async with settings_rig(tmp_path) as (rig, llm, _browser):
        session = rig.session
        assert rig.state.state is TurnState.IDLE, "premise: IDLE"
        release = asyncio.Event()
        task = asyncio.ensure_future(release.wait())
        if hold == "execution":
            planted = SimpleNamespace(
                task=task,
                call=SimpleNamespace(turn_id="t0-held", call_id="held"),
                deadline=None,
                settled=True,
            )
            session._executions["held"] = planted
        elif hold == "generate":
            session._generate["held"] = task
        elif hold == "prefill":
            session._prefills["held"] = task
        elif hold == "live_snapshot":
            session._live_snap = task
        else:
            session._closing = SimpleNamespace(turn_id="t0-held", extras={})
        try:
            mark = await set_and_pump(rig, {"visual_tokens": NEW_BUDGET})
            held = last_state(rig, mark, f" ({hold} held)")
            budget_held = llm.visual_tokens
        finally:
            release.set()
            await task
            if hold == "execution":
                session._executions.pop("held", None)
            elif hold == "generate":
                session._generate.pop("held", None)
            elif hold == "prefill":
                session._prefills.pop("held", None)
            elif hold == "closing":
                session._closing = None
        vt = item(held, "visual_tokens")
        assert (held.rev, vt.value, vt.pending) == (0, OLD_BUDGET, NEW_BUDGET), (
            f"T-0213 A2a-3: with the {hold} still running the set waits; rev {held.rev}, {vt}"
        )
        assert budget_held == OLD_BUDGET, f"T-0213 A2a-3: the budget moved while {hold} ran"
        # the next pump hop: a planted task ends without an event of its own, so the row sends one (an empty set,
        # whose hop settles what is pending before it is handled)
        await set_and_pump(rig, {})
        after = session.settings
    assert after.rev == 1 and after.visual_tokens == NEW_BUDGET, (
        f"premise: once the {hold} is over the set applies; rev {after.rev}"
    )
