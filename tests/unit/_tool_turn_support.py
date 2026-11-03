"""Shared rig for the T-0092c rows: the tool turn over the fakes (detection, repair, dry run, HUD, telemetry).

Nothing here imports a T-0092c symbol at module level. The rows observe the session only through what it already
exposes -- the machine's event/command log (``RecordingMachine``), the HUD publisher (``Recorder``), the session-log
file, ``turns.jsonl``, the fake LLM's request log and the recording executor -- so while T-0092c is unimplemented
every row is red on its own named assertion, never on an import error at collection.

**The surface the rows assume.** docs/tasks/T-0092c.md names the behaviour; where it is silent the choice is fixed
here, in one place, and listed in the test-author's report:

- the call id is ``f"{turn_id}-call"`` (criterion 1, verbatim): ``t1-call`` for the first turn;
- ``astra.tool.call`` is published through the session's injected ``publish`` (the HUD seam), one message per step;
- the session-log kind is ``tool.call`` (criterion 7, verbatim) with the data keys of ``TOOL_CALL_KEYS``;
- ``TurnRecord.tool`` is a dict carrying at least ``call_id`` and ``status`` (the final state; criterion 7);
- the span names are ``tool.repair`` and ``tool.execute`` (criterion 7, verbatim);
- the repair is told apart on the wire by ``prompt.continue_final_message`` (the fake never hashes a ``Prompt``).

The fake LLM is ``ToolLlm``: ``test_session_loop.RecordingLlm`` (call log, concurrency log, holds) plus the ``cd``
keyword, delegated to ``tests/fakes/llm.FakeLlm.generate`` (the continuation script, the continuation hold and
errors, the deterministic ``tokenize`` and the recorded ``cd`` live there).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time), thinking on, 200 ms per word unless a row says otherwise,
n = 1 and deterministic per row. No measurement.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit, LlmOptions, Prompt
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.turn import TurnState
from local_astra.toolcall.special_tokens import SPECIAL_TOKENS
from tests.fakes import FakeMemory, FakeStt
from tests.fakes.llm import FakeLlm, fake_tokenize, is_caption_prompt
from tests.fakes.tool_executor import RecordingToolExecutor
from tests.unit import _tools_switch_support as T
from tests.unit._hud_rig import Recorder
from tests.unit._session_log_rig import make_log, of_kind, read_records
from tests.unit.test_session_loop import (
    POLICY,
    RecordingAudio,
    RecordingLlm,
    RecordingMachine,
    RecordingPerception,
    RecordingTts,
    Rig,
    ScriptedClock,
    Session,
    SessionConfig,
    answer,
    require_session,
    thought,
)

#: T-0092c's base: main with T-0092a and T-0092b merged (``4f8d4e0``). A literal SHA, never a moving ref (tests.md).
BASE_SHA = "4f8d4e0"

CALL_ID = "t1-call"
REPAIR_ID = "t1-repair"
PASS1_ID = "t1-gen"
TOOL_CALL_TYPE = "astra.tool.call"
TOOL_CALL_KIND = "tool.call"
#: Criterion 7: the data keys of one ``tool.call`` record, besides ``status`` (written out from the criterion).
TOOL_CALL_KEYS = frozenset(
    {
        "status",
        "call_id",
        "source",
        "name",
        "arguments",
        "parse",
        "schema_valid",
        "query_problem",
        "repair_ms",
        "action",
        "reason",
        "resolved",
        "spoken_prefix",
    }
)
FINAL = ("dry_run", "refused", "cancelled")
INTERMEDIATE = ("proposed", "repairing")

MS_PER_WORD = 200
Q = '<|"|>'

# --- the calls the rows script (verdicts written by hand from T-0091's rules, confirmed by live.classify) ------

#: valid: click_at with in-range ints (schema 0-999)
VALID_CLICK = "<|tool_call>call:click_at{x:500,y:250}<tool_call|>"
#: repairable (schema invalid: an extra key ``button``, coordinates in range; the name is a v1 tool). Ends in "\n" so
#: the rstrip is observable. T-0100 moved it off ``x:1200``: an out-of-range coordinate is refused at detection (H-T99e).
REPAIRABLE_CLICK = f"<|tool_call>call:click_at{{x:12,y:5,button:{Q}left{Q}}}<tool_call|>\n"
#: route(REPAIRABLE_CLICK).draft_call_text, written out: everything after the tag
REPAIRABLE_DRAFT = f"call:click_at{{x:12,y:5,button:{Q}left{Q}}}<tool_call|>\n"
#: the repair's own valid call (the continuation's ``tool`` commit starts with the tag, T-0088 E4)
REPAIRED_CLICK = "<|tool_call>call:click_at{x:12,y:5}<tool_call|>"
#: the repair's output, still schema-invalid (an extra key, in range: ``repair_verdict`` says "schema invalid")
STILL_INVALID = f"<|tool_call>call:click_at{{x:12,y:5,button:{Q}left{Q}}}<tool_call|>"
#: T-0100 (H-T99d): a well-formed press_key whose combo is not in the list -- the a794a98 kout-02 pass-1 text verbatim
OUT_OF_LIST_KEY = f"<|tool_call>call:press_key{{combo:{Q}Ctrl+S{Q}}}<tool_call|>"
#: T-0100 (H-T99e): a click outside 0-999 -- the a794a98 cpix-03 pass-1 text verbatim
OUT_OF_RANGE_CLICK = "<|tool_call>call:click_at{x:1088,y:748}<tool_call|>"
#: T-0100 (Q1): an alias of a list entry (Control -> Ctrl): still repairable, the repair writes the exact string
ALIASED_KEY = f"<|tool_call>call:press_key{{combo:{Q}Control+C{Q}}}<tool_call|>"
#: the repair of ``ALIASED_KEY``
REPAIRED_KEY = f"<|tool_call>call:press_key{{combo:{Q}Ctrl+C{Q}}}<tool_call|>"
#: refused: parses, names no v1 tool (``no tool name``)
NAMELESS = f"<|tool_call>call:navigate{{url:{Q}https://example.org{Q}}}<tool_call|>"
#: refused: a ``<cc>`` tool commit (H6)
CC_CALL = '<cc>{"name":"click_at","arguments":{"x":1,"y":2}}</cc>'
#: valid google_search whose query narrates (``score.query_problem`` -> "narration"; log-only, H7)
NARRATED_QUERY = "I will search for the weather in Lisbon"
SEARCH_NARRATED = f"<|tool_call>call:google_search{{query:{Q}{NARRATED_QUERY}{Q}}}<tool_call|>"

SPOKEN = "I will click the Sign in button. "


def tool(text: str, *, stop_id: int | None = 50, canvas_idx: int = 1) -> Commit:
    return FakeLlm.tool_commit(text, stop_id=stop_id, canvas_idx=canvas_idx)


def seed_ids(draft: str, stop_id: int | None) -> list[int]:
    """Criterion 2 step 3 / T-0091 criterion 5: ``(tokenize(draft.rstrip()) + [stop_id])[:256]`` -- the fake's
    deterministic map is the oracle for the ids (each special literal one id, every other byte its ``<0xNN>`` id)."""
    ids = fake_tokenize(draft.rstrip())
    return [*ids, *([stop_id] if stop_id is not None else [])][:256]


# --- the fake LLM with the cd keyword --------------------------------------------------------------------


class ToolLlm(RecordingLlm):
    """``RecordingLlm`` whose ``generate`` takes ``cd`` (T-0088 criterion 3) and forwards it to ``FakeLlm``.

    The body is ``RecordingLlm.generate``'s bookkeeping (call log, concurrency log, the holds) verbatim; only the
    delegation passes ``cd`` through, which ``RecordingLlm`` cannot do (it predates the keyword)."""

    async def generate(  # type: ignore[override]
        self,
        prompt: Prompt,
        request_id: str,
        options: LlmOptions | None = None,
        *,
        xargs: Mapping[str, str] | None = None,
        cd: Any = None,
    ) -> AsyncIterator[Commit]:
        caption = is_caption_prompt(prompt)
        if not caption:
            self.gen_prompts.append((request_id, prompt))
        if self.fail_generate and not caption:
            self._note("generate", request_id)
            raise RuntimeError("vLLM returned HTTP 400: prompt too long")
        (self.live_captions if caption else self.live_generates).add(request_id)
        self._note("caption" if caption else "generate", request_id)
        delivered = 0
        try:
            async for commit in FakeLlm.generate(
                self, prompt, request_id, options, xargs=xargs, cd=cd
            ):
                if not caption and self._hold_after is not None and delivered >= self._hold_after:
                    self._flow.clear()
                await self._flow.wait()
                if request_id not in self.active:
                    return
                yield commit
                delivered += 1
        finally:
            self.live_generates.discard(request_id)
            self.live_captions.discard(request_id)
            self.concurrency.append(
                (len(self.live_prefills), len(self.live_generates), len(self.live_captions))
            )

    # -- what the rows read --------------------------------------------------------------------------

    def continuations(self) -> list[tuple[str, Prompt, LlmOptions, Any]]:
        """``(engine id, prompt, options, cd)`` of every continuation (repair) request, in order."""
        out = []
        for (rid, prompt, options), cd in zip(self.generates, self.cd_log, strict=True):
            if getattr(prompt, "continue_final_message", False):
                out.append((rid, prompt, options, cd))
        return out

    def continuation_bodies(self) -> list[dict[str, Any]]:
        return [b for b in self.bodies if b.get("continue_final_message") is True]

    def pass1_bodies(self) -> list[dict[str, Any]]:
        return [
            b
            for b in self.bodies
            if b.get("continue_final_message") is not True and b.get("max_tokens") != 1
        ]


# --- the executor double that also says when it ran ---------------------------------------------------------


class WatchingExecutor(RecordingToolExecutor):
    """The recording fake, plus what the device had played when each call ran (the G8 re-check).

    ``inner`` (optional) answers instead of the scripted default, e.g. the real ``DryRunToolExecutor``."""

    def __init__(self, *, inner: Any = None, raise_: BaseException | None = None) -> None:
        super().__init__()
        self.inner = inner
        self.raise_ = raise_
        self.audio: Any = None
        self.clock: Any = None
        self.plays_at_call: list[int] = []
        self.times: list[int] = []

    async def execute(self, inv: Any) -> Any:
        self.invocations.append(inv)
        if self.audio is not None:
            self.plays_at_call.append(len(self.audio.plays))
        if self.clock is not None:
            self.times.append(self.clock.now_ms)
        if self.raise_ is not None:
            raise self.raise_
        if self.inner is not None:
            return await self.inner.execute(inv)
        from local_astra.contracts.toolcall import ToolOutcome

        return ToolOutcome(
            call_id=inv.call_id,
            status="dry_run",
            action=f"recorded {inv.name}",
            reason=None,
            resolved={},
        )


# --- frames -----------------------------------------------------------------------------------------------


def frame(ts_ms: int, tag: str, *, width: int = 1280, height: int = 720) -> Frame:
    return Frame(
        source=Source.SCREEN,
        ts_ms=ts_ms,
        width=width,
        height=height,
        jpeg=b"\xff\xd8" + tag.encode() * 8,
        phash=f"{tag * 4}{ts_ms:012d}",
        title=f"window {tag}",
        app="firefox",
    )


# --- the rig ------------------------------------------------------------------------------------------------


@dataclass
class ToolRig(Rig):
    published: Recorder = field(default_factory=Recorder)
    log: Any = None
    executor: Any = None
    kit: Any = None
    log_records: list[dict[str, Any]] = field(default_factory=list)

    # -- the HUD -----------------------------------------------------------------------------------------
    def tool_calls(self, call_id: str | None = None) -> list[Any]:
        return [
            m
            for m in self.published.messages
            if getattr(m, "type", None) == TOOL_CALL_TYPE
            and (call_id is None or m.call_id == call_id)
        ]

    def statuses(self, call_id: str = CALL_ID) -> list[str]:
        return [m.status for m in self.tool_calls(call_id)]

    # -- the machine -------------------------------------------------------------------------------------
    def dones(self, request_id: str | None = None) -> list[Any]:
        return [
            e
            for e in self.machine.events(E.LlmDone)
            if request_id is None or e.request_id == request_id
        ]

    def commands(self, kind: type) -> list[Any]:
        return self.machine.commands(kind)

    def speaks(self) -> list[Any]:
        return self.machine.commands(C.TtsSpeak)

    def spoken_sources(self) -> str:
        return "".join(c.source for c in self.speaks())

    # -- the log (filled after aclose) ---------------------------------------------------------------------
    def kind(self, kind: str) -> list[dict[str, Any]]:
        return [r["data"] for r in of_kind(self.log_records, kind)]

    def tool_records(self) -> list[dict[str, Any]]:
        return self.kind(TOOL_CALL_KIND)

    def record(self, turn_id: str = "t1") -> Any:
        found = [r for r in self.records() if r.turn_id == turn_id]
        assert len(found) == 1, (
            f"one turns.jsonl line for {turn_id}; got {[r.turn_id for r in self.records()]}"
        )
        return found[0]

    # -- driving -------------------------------------------------------------------------------------------
    async def turn_done(
        self, what: str = "the turn never ended", *, limit_ms: int = 30_000
    ) -> None:
        await self.until(
            lambda: self.state.state is TurnState.IDLE and bool(self.lines()),
            what,
            limit_ms=limit_ms,
        )

    async def until_status(
        self, status: str, call_id: str = CALL_ID, *, limit_ms: int = 10_000
    ) -> None:
        await self.until(
            lambda: status in self.statuses(call_id),
            f"no astra.tool.call {status!r} for {call_id} was published; statuses {self.statuses(call_id)}",
            limit_ms=limit_ms,
        )

    async def until_repair_live(self, *, limit_ms: int = 5_000) -> None:
        await self.until(
            lambda: bool(self.llm.continuations()),
            "T-0092c criterion 2: no repair (continuation) generate was sent; statuses "
            f"{self.statuses()}, ToolRepairStart {self.commands(C.ToolRepairStart)}",
            limit_ms=limit_ms,
        )


def with_search(kit: Any, search: Any) -> Any:
    """T-0114: ``kit`` with ``search=search`` (the ``ToolKit`` gains ``search: SearchBackend | None = None``).

    A kit without the field is a named failure, never a ``TypeError`` inside the rig."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(kit)}
    assert "search" in names, (
        f"T-0114 (owned files): ToolKit gains `search: SearchBackend | None = None`; its fields are {sorted(names)}"
    )
    return dataclasses.replace(kit, search=search)


@asynccontextmanager
async def tool_rig(
    tmp_path: Path,
    *,
    script: Sequence[Commit] = (),
    continuation: Sequence[Commit] | None = None,
    kit: Any = "default",
    executor: Any = None,
    frames: Sequence[Frame] | None = None,
    ms_per_word: int = MS_PER_WORD,
    llm: Any = None,
    audio_cls: type = RecordingAudio,
    tts: Any = None,
    machine_cls: type = RecordingMachine,
    policy: Any = POLICY,
    search: Any = None,
) -> AsyncIterator[ToolRig]:
    """The T-0007 session over the fakes, tools ``dry_run`` (``kit="default"``) or off (``kit=None``).

    The HUD publisher is a ``Recorder``; the session log is a real ``SessionLog`` read back after ``aclose``
    (``rig.log_records``), because its writer is asynchronous and a mid-row read races it.

    T-0114: ``policy`` is the machine's ``TurnPolicy`` (default ``POLICY``, so every earlier row is unchanged) and
    ``search`` a ``SearchBackend`` put on the default kit (``None``: the kit as before).
    """
    require_session()
    clock = ScriptedClock()
    machine = machine_cls(policy)
    if llm is None:
        llm = ToolLlm(list(script))
    llm.clock = clock
    if continuation is not None:
        llm.script_continuation(*continuation)
    tts = tts if tts is not None else RecordingTts(ms_per_word=ms_per_word)
    audio = audio_cls(clock)
    perception = RecordingPerception(list(frames) if frames is not None else [frame(0, "a")])
    published = Recorder()
    log = make_log(tmp_path / "logs")
    if executor is None:
        executor = WatchingExecutor()
    if isinstance(executor, WatchingExecutor):
        executor.audio, executor.clock = audio, clock
    if kit == "default":
        kit = T.make_kit(executor=executor)
        if search is not None:
            kit = with_search(kit, search)
    turns_path = tmp_path / "turns.jsonl"
    extra: dict[str, Any] = {"publish": published, "session_log": log}
    if kit is not None:
        extra[T.KIT_KWARG] = kit
    session = Session(
        machine,
        llm,
        FakeStt(),
        tts,
        audio,
        perception,
        FakeMemory(),
        clock=clock,
        config=SessionConfig(data_dir=tmp_path, turns_path=turns_path),
        **extra,
    )
    rig = ToolRig(
        clock=clock,
        machine=machine,
        llm=llm,
        stt=session.stt,
        tts=tts,
        audio=audio,
        perception=perception,
        memory=session.memory,
        session=session,
        turns_path=turns_path,
        published=published,
        log=log,
        executor=executor,
        kit=kit,
    )
    await session.start()
    try:
        yield rig
    finally:
        await session.aclose()
        rig.log_records = read_records(log.path)


# --- shared checks ----------------------------------------------------------------------------------------


def literal_in(text: str) -> list[str]:
    return [lit for lit in SPECIAL_TOKENS if lit in text]


def assert_never_voiced(rig: ToolRig, *, arguments: Mapping[str, Any] = (), what: str = "") -> None:
    """Criterion 6 "Never voiced": no ``TtsSpeak.text``, and no ``source`` of a ``TtsSpeak`` with non-empty
    ``text``, contains a special literal, ``call:`` or an argument value of the call."""
    values = [str(v) for v in dict(arguments).values()]
    for c in rig.speaks():
        checked = [c.text] + ([c.source] if c.text else [])
        for s in checked:
            bad = (
                literal_in(s)
                + (["call:"] if "call:" in s else [])
                + [v for v in values if v and v in s]
            )
            assert not bad, (
                f"T-0092c criterion 6{what}: a TtsSpeak voices {bad} (text={c.text!r}, source={c.source!r})"
            )


def assert_history_clean(rig: ToolRig, *, arguments: Mapping[str, Any] = ()) -> None:
    values = [str(v) for v in dict(arguments).values()]
    for h in rig.state.history:
        s = h.assistant_spoken
        bad = (
            literal_in(s)
            + (["call:"] if "call:" in s else [])
            + [v for v in values if v and v in s]
        )
        assert not bad, f"T-0092c criterion 6: HistoryTurn.assistant_spoken holds {bad}: {s!r}"


def assert_one_final(rig: ToolRig, call_id: str = CALL_ID) -> str:
    """Criterion 4: exactly one final message per call id, and nothing after it."""
    statuses = rig.statuses(call_id)
    finals = [s for s in statuses if s in FINAL]
    assert len(finals) == 1, (
        f"T-0092c criterion 4: {call_id} gets exactly one final message (dry_run, refused or cancelled); "
        f"published {statuses}"
    )
    assert statuses[-1] in FINAL, (
        f"T-0092c criterion 4: the final message is the last one; {statuses}"
    )
    return finals[0]


def final_message(rig: ToolRig, call_id: str = CALL_ID) -> Any:
    assert_one_final(rig, call_id)
    return rig.tool_calls(call_id)[-1]


def assert_concurrency(rig: ToolRig) -> None:
    """Criterion 8 (the I8 shape over fakes): at most one live generate plus one prefill at every step."""
    worst = [(p, g) for p, g, _ in rig.llm.concurrency if g > 1 or p > 1]
    assert not worst, (
        "T-0092c criterion 8: at most one live generate (the repair counted) and one prefill at every step; "
        f"saw (prefills, generates) {worst[:5]}"
    )


def patch_bound(monkeypatch: Any, module: Any, name: str, replacement: Callable[..., Any]) -> None:
    """Replace ``module.name`` and every binding of the same object in a loaded ``local_astra.conductor`` module.

    A row that forces a branch (a guard refusal, a verdict) must not dictate how the session imports the function:
    ``from local_astra.toolcall import live`` and ``from local_astra.toolcall.live import repair_prompt`` are both
    reached."""
    original = getattr(module, name)
    monkeypatch.setattr(module, name, replacement)
    for modname, mod in list(sys.modules.items()):
        if not modname.startswith("local_astra.conductor") or mod is None:
            continue
        for attr, value in list(vars(mod).items()):
            if value is original:
                monkeypatch.setattr(mod, attr, replacement)


async def drain_settle(rig: ToolRig, rounds: int = 6) -> None:
    for _ in range(rounds):
        await rig.pump()
        await asyncio.sleep(0)


__all__ = [
    "BASE_SHA",
    "CALL_ID",
    "answer",
    "thought",
]
