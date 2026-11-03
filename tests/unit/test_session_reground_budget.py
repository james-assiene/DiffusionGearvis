"""T-0132 B2-B6 (session side): the pass-2 (crop) budget through a second client, only for single-image pass-2
prompts; the abort on the issuing client; the ``tool.reground`` record's new keys (docs/tasks/T-0132.md).

Every row runs T-0131's ``reground_rig`` (live session, loop on, real-JPEG frames, the pass-2-aware main LLM) with
``Session(..., reground=True, visual_tokens=1120, reground_llm=<second>, reground_visual_tokens=<N>)``. The second
client is ``SecondLlm`` below: a ``FakeLlm`` (so it has the whole ``LlmBackend`` surface) whose ``generate`` records
every request and serves a *timed* script on the session clock, and whose ``aclose`` is counted.

**The interfaces these rows pin** (the task's; readings in the test-author report):

- ``Session(..., reground_llm: LlmBackend | None = None, reground_visual_tokens: int | None = None)``: both or
  neither (``ValueError``), and both refused without ``reground`` (``ValueError``).
- Pass 2 goes through ``reground_llm`` only when its prompt carries exactly one ``ImagePart`` and the two budgets
  differ; otherwise through ``self.llm`` (B3, R-e).
- ``tool.reground`` gains ``budget`` (the budget the issuing client sends: ``reground_visual_tokens`` through the
  second client, ``visual_tokens`` through the main one, ``None`` when unset), ``images`` (the pass-2 prompt's image
  count), ``first_ms`` and ``tool_ms`` (session-clock ms from the request start to pass 2's first commit and to its
  ``tool`` commit; ``None`` when absent).
- **Reading:** "closing the session closes a second client that cli built" is ``Session.aclose`` awaiting
  ``reground_llm.aclose()`` when ``reground_llm`` is not ``self.llm`` (``LlmBackend`` has no ``aclose``, so it is
  looked up on the object).

Group B, activated 2026-10-22 (until then ``pending(task="T-0132")``); at BASE a2d9ac6 the rig refuses the unknown keywords with a named
``AssertionError`` (``reground_rig``'s pinned-seam check).

Regime: CPU over fakes, ``ScriptedClock`` (virtual time, 50 ms steps), thinking on, 200 ms per word, loop on (8
steps), 1280x800 frames, n = 1, deterministic.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts.llm import Commit, ImagePart
from tests.fakes import FakeLlm
from tests.unit import test_session_tool_loop as tsl
from tests.unit._hud_rig import split_thoughts
from tests.unit._live_browser_support import (
    HANG,
    FakeLiveBrowser,
    jpeg_frame,
    reground_rig,
)
from tests.unit._tool_turn_support import VALID_CLICK, answer, thought, tool

SENTINEL = "SECOND-CLIENT-SENTINEL pass two voiced this."
SECRET = "SECOND-CLIENT-THOUGHT never shown."
B5 = ("budget", "images", "first_ms", "tool_ms")


def click(x: int, y: int) -> str:
    return f"<|tool_call>call:click_at{{x:{x},y:{y}}}<tool_call|>"


class SecondLlm(FakeLlm):
    """The pass-2 client. ``timed``: one entry per request, a list of ``(offset_ms, Commit)`` served at the request
    start + offset on the session clock, or ``HANG`` (held until aborted). Records ``calls`` (engine id, prompt) and
    ``closes``. ``bomb``: any request is recorded and then raises (the row asserts ``calls == []``)."""

    def __init__(self, *, bomb: bool = False) -> None:
        super().__init__()
        self.bomb = bomb
        self.clock: Any = None
        self.timed: list[Any] = []
        self.calls: list[tuple[str, Any]] = []
        self.closes = 0
        self._hang: dict[str, asyncio.Event] = {}

    async def prefill(self, prefix: Any, request_id: str) -> Any:
        self.calls.append((request_id, prefix))
        raise AssertionError("T-0132 B2: the second client is used for pass 2 only")

    async def generate(  # type: ignore[override]
        self,
        prompt: Any,
        request_id: str,
        options: Any = None,
        *,
        xargs: Any = None,
        cd: Any = None,
    ) -> Any:
        self.calls.append((request_id, prompt))
        if self.bomb:
            raise AssertionError("T-0132 B3: the second client was used")
        entry = self.timed.pop(0) if self.timed else []
        if entry == HANG:
            ev = self._hang.setdefault(request_id, asyncio.Event())
            await ev.wait()
            return
        start = self.clock.now_ms
        for offset, commit in entry:
            if self.clock.now_ms < start + offset:
                fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                self.clock.call_at(start + offset, lambda f=fut: f.done() or f.set_result(None))
                await fut
            yield commit.model_copy(update={"request_id": request_id})

    async def abort(self, request_id: str) -> None:
        await super().abort(request_id)
        ev = self._hang.get(request_id)
        if ev is not None:
            ev.set()

    async def aclose(self) -> None:
        self.closes += 1


def frames() -> FakeLiveBrowser:
    return FakeLiveBrowser(
        [jpeg_frame("onset"), jpeg_frame("second-turn")],
        post_frames=[jpeg_frame("after-1"), jpeg_frame("after-2"), jpeg_frame("after-3")],
    )


def regrounds(rig: Any) -> list[dict[str, Any]]:
    return [r["data"] for r in rig.log_records if r["kind"] == "tool.reground"]


def images(prompt: Any) -> int:
    return sum(1 for m in prompt.messages for p in m.content if isinstance(p, ImagePart))


async def click_turn(
    tmp_path: Path,
    second: SecondLlm | None,
    *,
    crop: int | None = 280,
    main_pass2: Any = (),
    first_turn: bool = False,
    extra: dict[str, Any] | None = None,
    browser: FakeLiveBrowser | None = None,
) -> tuple[Any, dict[str, Any]]:
    """A live click turn (``S1`` + ``VALID_CLICK``, then ``S3``) on a session with ``visual_tokens=1120``, ReGround
    on and, when ``second`` is given, ``reground_llm=second, reground_visual_tokens=crop``. ``first_turn``: a toolless
    live turn first, so the click turn's prompt carries a history image beside its own frame."""
    kwargs: dict[str, Any] = {"visual_tokens": 1120, **(extra or {})}
    if second is not None:
        kwargs |= {"reground_llm": second, "reground_visual_tokens": crop}
    browser = browser if browser is not None else frames()
    script = [answer("It is a small shop. ")] if first_turn else [answer(tsl.S1), tool(VALID_CLICK)]
    async with reground_rig(tmp_path, browser, script=script, session_kwargs=kwargs) as rig:
        if second is not None:
            second.clock = rig.clock
        rig.llm.script_pass2(*main_pass2)
        if first_turn:
            await rig.ask("what is on the page")
            await rig.turn_done()
            await rig.advance(300)
            rig.llm.script = [answer(tsl.S1), tool(VALID_CLICK)]
        rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        turn = "t2" if first_turn else "t1"
        await rig.until(
            lambda: rig.state.state.name == "IDLE" and len(rig.lines()) >= (2 if first_turn else 1),
            "the click turn never ended",
            limit_ms=40_000,
        )
        extras = dict(rig.record(turn).extras)
    return rig, extras


def ok_pass2() -> list[tuple[int, Commit]]:
    return [(0, answer(SENTINEL)), (0, tool(click(900, 100)))]


# =====================================================================================================
# B3: which client
# =====================================================================================================


async def test_single_image_uses_reground_llm(tmp_path: Path) -> None:
    """B3/B5: a fresh live session's pass 2 carries one image (the crop) and goes to ``reground_llm``; the main
    client gets no pass 2; the record has ``budget == 280`` and ``images == 1``."""
    second = SecondLlm()
    second.timed = [ok_pass2()]
    rig, _ = await click_turn(tmp_path, second)
    assert len(second.calls) == 1, (
        f"T-0132 B3: pass 2 went to reground_llm; {len(second.calls)} calls"
    )
    assert images(second.calls[0][1]) == 1, "premise: the pass-2 prompt carries one image"
    assert rig.llm.pass2_ids == [], f"T-0132 B3: the main client got a pass 2: {rig.llm.pass2_ids}"
    (rec,) = regrounds(rig)
    assert rec.get("budget") == 280 and rec.get("images") == 1, rec
    assert rec["outcome"] == "ok", rec


async def test_multi_image_uses_main_llm(tmp_path: Path) -> None:
    """B3 (adversarial; reviewer mutation 4, always using ``reground_llm``): on the second live turn the step prompt
    carries a history image plus the current frame, so pass 2 carries two images and goes to ``self.llm`` at the main
    budget; ``reground_llm`` is a bomb and is never called; the record has ``budget == 1120`` and ``images == 2``."""
    bomb = SecondLlm(bomb=True)
    rig, _ = await click_turn(
        tmp_path,
        bomb,
        first_turn=True,
        main_pass2=[[answer(SENTINEL), tool(click(900, 100))]],
    )
    assert bomb.calls == [], f"T-0132 B3: reground_llm was used with several images: {bomb.calls}"
    assert len(rig.llm.pass2_prompts) == 1, f"pass 2 went to the main client; {rig.llm.pass2_ids}"
    assert images(rig.llm.pass2_prompts[0][1]) == 2, (
        "premise: the second turn's pass-2 prompt carries a history image and the crop"
    )
    (rec,) = regrounds(rig)
    assert rec.get("budget") == 1120 and rec.get("images") == 2, rec


async def test_abort_on_issuing_client(tmp_path: Path) -> None:
    """B4 (reviewer mutation 5): a barge-in while pass 2 hangs on ``reground_llm`` aborts that engine id on
    ``reground_llm``, never on the main client; nothing is dispatched."""
    second = SecondLlm()
    second.timed = [HANG]
    browser = frames()
    kwargs = {
        "visual_tokens": 1120,
        "reground_llm": second,
        "reground_visual_tokens": 280,
    }
    async with reground_rig(
        tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)], session_kwargs=kwargs
    ) as rig:
        second.clock = rig.clock
        await rig.ask(tsl.QUESTION)
        await rig.until(lambda: bool(second.calls), "premise: pass 2 started", limit_ms=10_000)
        rid = second.calls[0][0]
        now = rig.clock.now_ms
        rig.submit(E.UserTakeover(t_ms=now, onset_ms=now))
        await rig.pump()
        await rig.advance(500)
        assert rid in second.aborted, f"T-0132 B4: the abort goes to reground_llm; {second.aborted}"
        assert rid not in rig.llm.aborted, "T-0132 B4: the pass-2 abort was sent to self.llm"
        assert browser.executes() == 0, browser.calls


# =====================================================================================================
# B5: the record's times
# =====================================================================================================


@pytest.mark.parametrize(
    ("script", "first", "tool_ms"),
    [
        ([(300, answer(SENTINEL)), (700, tool(click(900, 100)))], 300, 700),
        ([(300, answer(SENTINEL))], 300, None),
        ([], None, None),
    ],
    ids=["answer-then-tool", "no-tool", "empty"],
)
async def test_first_and_tool_ms(
    tmp_path: Path, script: list[Any], first: int | None, tool_ms: int | None
) -> None:
    """B5: commits scripted at the request start + 300 ms and + 700 ms on the session clock give ``first_ms == 300``
    and ``tool_ms == 700``; no tool commit gives ``tool_ms is None``; no commit gives both ``None``. Every T-0131 key
    is kept."""
    second = SecondLlm()
    second.timed = [script]
    rig, _ = await click_turn(tmp_path, second)
    (rec,) = regrounds(rig)
    assert (rec.get("first_ms"), rec.get("tool_ms")) == (first, tool_ms), rec
    assert {"call_id", "step", "name", "p1", "p2", "crop", "ms", "tool_text", "outcome"} <= set(rec)


async def test_budget_null_when_unset(tmp_path: Path) -> None:
    """B5: a ReGround session with no budget at all (``visual_tokens`` unset, no second client) records
    ``budget is None`` and ``images == 1``."""
    browser = frames()
    async with reground_rig(tmp_path, browser, script=[answer(tsl.S1), tool(VALID_CLICK)]) as rig:
        rig.llm.script_pass2([answer(SENTINEL), tool(click(900, 100))])
        rig.llm.script_steps([answer(tsl.S3)])
        await rig.ask(tsl.QUESTION)
        await rig.turn_done(limit_ms=40_000)
    (rec,) = regrounds(rig)
    assert "budget" in rec and rec["budget"] is None and rec.get("images") == 1, rec


# =====================================================================================================
# B6: equal budgets change nothing
# =====================================================================================================


def _normalised(rig: Any, where: Path) -> dict[str, Any]:
    """The compared sections with exactly four things masked, and nothing else:

    - the run's own log path, the exact string ``str(rig.log.path)``, becomes ``<log>`` (T-0144). It is replaced
      first, before the run directory. ``SessionLog`` names its file ``YYYYMMDD-HHMMSS-<pid>.jsonl`` from the wall
      clock, so two runs ~50 ms apart that fall on either side of a wall-clock second name different files, and the
      turn line's ``session_log`` carries that name. No regex, the ``session_log`` key is kept, and any other path
      (a turn line naming a log that is not this run's own) stays compared;
    - the session id becomes ``<session>``;
    - the run directory ``where`` becomes ``<run>`` (the two runs live in different directories; implementer's
      finding at bbba10c);
    - ``regime.git_commit`` is removed from each turn line.

    Nothing else is dropped: B5's keys and the capture counters (``stt_frames_fed``) stay, so with equal budgets
    they must be equal too (O-2)."""
    sid = rig.session.session_id
    text = json.dumps(
        {
            "log": [
                {"kind": r["kind"], "data": r["data"]}
                for r in rig.log_records
                if r["kind"] not in ("log.started", "log.stopped")
            ],
            "records": [json.loads(line) for line in rig.lines()],
            "bodies": list(rig.llm.bodies),
            "pass2": [p.model_dump(mode="json") for _, p in rig.llm.pass2_prompts],
        },
        sort_keys=True,
    )
    text = text.replace(str(rig.log.path), "<log>").replace(sid, "<session>")
    text = text.replace(str(where), "<run>")
    out: dict[str, Any] = json.loads(text)
    for rec in out["records"]:
        rec.get("regime", {}).pop("git_commit", None)
    return out


def _occupy_seconds(logs: Path, *, before_s: int = 2, after_s: int = 120) -> None:
    """Pre-create, in ``logs``, the plain log file names (``YYYYMMDD-HHMMSS-<pid>.jsonl``, local wall clock, as
    ``SessionLog`` names them) for every second from ``before_s`` ago to ``after_s`` ahead, so a ``SessionLog`` built
    in that directory within the window takes the ``-2`` suffix. Through the file system only: no ``SessionLog``
    internal is touched. A run built outside the window is caught by the basename premise of the row that uses it."""
    logs.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    for k in range(-before_s, after_s + 1):
        stamp = (now + timedelta(seconds=k)).strftime("%Y%m%d-%H%M%S")
        (logs / f"{stamp}-{os.getpid()}.jsonl").touch()


def _own_log_premises(rig: Any, normed: dict[str, Any], run: str) -> None:
    """T-0144 criterion 2's premises: the run's turn line names the run's own log file (so the masked string is that
    path and nothing else), and the ``<log>`` token occurs in the normalised ``records`` and ``log`` (so the mask
    is not vacuous)."""
    first = json.loads(rig.lines()[0])
    assert first["session_log"] == str(rig.log.path), (
        f"premise ({run}): the turn line names the run's own log; {first['session_log']!r} vs {rig.log.path}"
    )
    for section in ("records", "log"):
        assert "<log>" in json.dumps(normed[section]), (
            f"premise ({run}): the own-log token occurs in the normalised {section!r}"
        )


async def _equal_budget_runs(
    tmp_path: Path, pass2: list[Commit], *, other_second: bool = False
) -> tuple[Any, dict[str, Any], Any, dict[str, Any]]:
    """B6's two runs, ``(plain_rig, plain, same_rig, same)``: ``plain`` is the group A session (no new keywords,
    under ``tmp_path / "plain"``), ``same`` passes the main client itself as ``reground_llm`` with
    ``reground_visual_tokens == visual_tokens`` (under ``tmp_path / "same"``); both sections are ``_normalised``.
    ``other_second``: before the ``same`` run, its ``logs/`` directory is filled by ``_occupy_seconds``, so its log
    file name differs from ``plain``'s as if the two runs had straddled a wall-clock second (T-0144)."""
    rigs: dict[str, Any] = {}

    async def run(where: Path, mode: str) -> dict[str, Any]:
        browser = frames()
        kwargs: dict[str, Any] = {"visual_tokens": 1120}
        holder: dict[str, Any] = {}
        async with reground_rig(
            where, browser, script=[answer(tsl.S1), tool(VALID_CLICK)], session_kwargs=kwargs
        ) as rig:
            holder["rig"] = rig
            rig.llm.script_pass2(pass2)
            rig.llm.script_steps([answer(tsl.S3)])
            if mode == "same":
                assert rig.session.llm is rig.llm
            await rig.ask(tsl.QUESTION)
            await rig.turn_done(limit_ms=40_000)
            await rig.advance(300)
        rigs[mode] = holder["rig"]
        return _normalised(holder["rig"], where)

    plain = await run(tmp_path / "plain", "plain")
    browser = frames()
    holder: dict[str, Any] = {}

    def same_client_kwargs(llm: Any) -> dict[str, Any]:
        return {"visual_tokens": 1120, "reground_llm": llm, "reground_visual_tokens": 1120}

    from tests.unit._live_browser_support import make_reground_llm

    main = make_reground_llm(browser, [answer(tsl.S1), tool(VALID_CLICK)])
    from tests.unit import _tool_turn_support as tts
    from tests.unit._live_browser_support import hud_frame, live_kit

    real = tts.Session
    import inspect

    params = inspect.signature(real.__init__).parameters
    assert "reground_llm" in params and "reground_visual_tokens" in params, (
        "T-0132 B2: Session(..., reground_llm=..., reground_visual_tokens=...) does not exist yet"
    )

    class Same(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            for key, value in {"reground": True, **same_client_kwargs(main)}.items():
                k.setdefault(key, value)
            super().__init__(*a, **k)

    if other_second:
        _occupy_seconds(tmp_path / "same" / "logs")
    tts.Session = Same  # type: ignore[misc]
    try:
        async with tts.tool_rig(
            tmp_path / "same",
            kit=live_kit(browser),
            policy=tsl.LOOP,
            llm=main,
            frames=[hud_frame(0)],
        ) as rig:
            browser.now_ms = lambda: rig.clock.now_ms
            rig.browser = browser  # type: ignore[attr-defined]
            holder["rig"] = rig
            rig.llm.script_pass2(pass2)
            rig.llm.script_steps([answer(tsl.S3)])
            await rig.ask(tsl.QUESTION)
            await rig.turn_done(limit_ms=40_000)
            await rig.advance(300)
    finally:
        tts.Session = real  # type: ignore[misc]
    same = _normalised(holder["rig"], tmp_path / "same")
    return rigs["plain"], plain, holder["rig"], same


async def test_equal_budget_identical_to_group_a(tmp_path: Path) -> None:
    """B6 (reviewer mutation 6's session half): ``reground_llm`` the main client itself with
    ``reground_visual_tokens == visual_tokens`` gives the group A state's bodies, records and turn lines (a session
    without the new keywords), B5's keys included (both runs carry them, and with equal budgets they are equal,
    O-2); and a *distinct* second client with an equal budget is never used (B3: "both budgets equal, nothing
    changes"). ``_normalised`` masks exactly the session id, the run directory, each run's own log path (named from
    the wall-clock second, T-0144) and ``git_commit``; nothing else is dropped. Premises: each run's turn line names
    its own log path, and the ``<log>`` token occurs in both normalised ``records`` and ``log``."""
    pass2 = [answer(SENTINEL), tool(click(900, 100))]
    plain_rig, plain, same_rig, same = await _equal_budget_runs(tmp_path, pass2)
    _own_log_premises(plain_rig, plain, "plain")
    _own_log_premises(same_rig, same, "same")
    rg = [r["data"] for r in same["log"] if r["kind"] == "tool.reground"]
    assert len(rg) == 1 and rg[0].get("budget") == 1120 and rg[0].get("images") == 1, (
        f"premise: B5's keys are present and compared (O-2); {rg}"
    )
    assert len(plain["pass2"]) == 1, "premise: pass 2 ran"
    for section in ("log", "records", "bodies", "pass2"):
        assert same[section] == plain[section], f"T-0132 B6: equal budgets changed {section!r}"
    distinct = SecondLlm(bomb=True)
    rig, _ = await click_turn(tmp_path / "distinct", distinct, crop=1120, main_pass2=[pass2])
    assert distinct.calls == [], "T-0132 B3: a second client with an equal budget was used"
    assert len(rig.llm.pass2_ids) == 1, "pass 2 went through self.llm"


async def test_b6_survives_a_wall_clock_second(tmp_path: Path) -> None:
    """T-0144 criterion 1 (adversarial, the flake's actual cause): with the ``same`` run's log file name forced to
    differ from ``plain``'s (``_occupy_seconds``: it takes the ``-2`` suffix, as two runs straddling a wall-clock
    second take different ``HHMMSS`` stamps), B6's four sections are still equal. Before T-0144, ``_normalised``
    masked only the run directory, so the turn line's ``session_log`` basename differed and ``records`` and ``log``
    were red 5/5. Premise: the two basenames differ, so a forcing that did nothing fails here, not silently."""
    pass2 = [answer(SENTINEL), tool(click(900, 100))]
    plain_rig, plain, same_rig, same = await _equal_budget_runs(tmp_path, pass2, other_second=True)
    assert Path(plain_rig.log.path).name != Path(same_rig.log.path).name, (
        f"premise: the forcing made the log basenames differ; {plain_rig.log.path} vs {same_rig.log.path}"
    )
    _own_log_premises(plain_rig, plain, "plain")
    _own_log_premises(same_rig, same, "same")
    assert len(plain["pass2"]) == 1, "premise: pass 2 ran"
    for section in ("log", "records", "bodies", "pass2"):
        assert same[section] == plain[section], (
            f"T-0144: the run's wall-clock second changed {section!r}"
        )


class _Prompt:
    """A pass-2 prompt stub: ``model_dump(mode="json")`` returns ``body``."""

    def __init__(self, body: dict[str, Any]) -> None:
        self.body = body

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        return dict(self.body)


def _stub_rig(
    where: Path,
    sid: str,
    stamp: str,
    *,
    named: str | None = None,
    fed: int = 43,
    tokens: int = 1120,
) -> Any:
    """A rig-shaped stub for ``_normalised``: its own log is ``where/logs/<stamp>-77.jsonl``; its turn line names
    ``named`` (default: its own log path), carries ``stt_frames_fed == fed`` and a ``git_commit``; one request body
    with ``max_visual_tokens == tokens``."""
    own = where / "logs" / f"{stamp}-77.jsonl"
    session_log = str(own) if named is None else named
    line = {
        "turn_id": "t1",
        "session_id": sid,
        "session_log": session_log,
        "regime": {"git_commit": "abc123", "visual_tokens": 1120},
        "extras": {"stt_frames_fed": fed},
    }
    return SimpleNamespace(
        session=SimpleNamespace(session_id=sid),
        log=SimpleNamespace(path=own),
        log_records=[
            {"kind": "log.started", "data": {"path": str(own)}},
            {"kind": "turn.record", "data": {"session_log": session_log, "session": sid}},
            {"kind": "log.stopped", "data": {}},
        ],
        lines=lambda: [json.dumps(line)],
        llm=SimpleNamespace(
            bodies=[{"max_visual_tokens": tokens, "image": str(where / "frame.jpg")}],
            pass2_prompts=[("r1", _Prompt({"messages": ["crop"]}))],
        ),
    )


_WA, _WB = Path("/w/plain"), Path("/w/same")
_STRANGER = str(_WB / "logs" / "20261023-120002-77.jsonl")


@pytest.mark.parametrize(
    ("other", "equal"),
    [
        ({}, True),
        ({"named": _STRANGER}, False),
        ({"fed": 44}, False),
        ({"tokens": 1121}, False),
    ],
    ids=["only-the-stamp", "not-its-own-path", "stt-frames-fed", "request-body"],
)
def test_normalised_masks_only_the_own_log_path(other: dict[str, Any], equal: bool) -> None:
    """T-0144 criterion 3 (the too-wide guard; pure, stub rigs, no session; equal / not equal decided by hand).
    Stub ``a`` is run ``/w/plain`` with log stamp 12:00:00; stub ``b`` is run ``/w/same`` with stamp 12:00:01 and a
    different session id, plus one change:

    - ``only-the-stamp`` (none): equal;
    - ``not-its-own-path``: ``b``'s turn line names ``/w/same/logs/20261023-120002-77.jsonl``, not its own
      ``log.path``: not equal (removing ``session_log``, M4, or masking by a timestamp regex makes it equal);
    - ``stt-frames-fed``: 44 instead of 43: not equal (the capture counters stay compared);
    - ``request-body``: ``max_visual_tokens`` 1121 instead of 1120: not equal."""
    a = _stub_rig(_WA, "sid-aaaa", "20261023-120000")
    b = _stub_rig(_WB, "sid-bbbb", "20261023-120001", **other)
    assert (_normalised(b, _WB) == _normalised(a, _WA)) is equal, (
        f"T-0144 criterion 3: {other or 'only the wall-clock stamp'} -> expected "
        f"{'equal' if equal else 'not equal'}"
    )


def test_normalised_tree_by_hand() -> None:
    """T-0144 criterion 2: on a stub rig, ``_normalised`` gives exactly the hand-written tree: the own log path ->
    ``<log>`` (the ``session_log`` key kept), session id -> ``<session>``, run directory -> ``<run>``,
    ``regime.git_commit`` removed, ``log.started`` / ``log.stopped`` left out; every other value as it was."""
    a = _stub_rig(_WA, "sid-aaaa", "20261023-120000")
    expected = {
        "log": [{"kind": "turn.record", "data": {"session_log": "<log>", "session": "<session>"}}],
        "records": [
            {
                "turn_id": "t1",
                "session_id": "<session>",
                "session_log": "<log>",
                "regime": {"visual_tokens": 1120},
                "extras": {"stt_frames_fed": 43},
            }
        ],
        "bodies": [{"max_visual_tokens": 1120, "image": "<run>/frame.jpg"}],
        "pass2": [{"messages": ["crop"]}],
    }
    assert _normalised(a, _WA) == expected, (
        "T-0144 criterion 2: the own log, session id and run directory masked, git_commit removed; nothing else"
    )


# =====================================================================================================
# B6: never voiced
# =====================================================================================================


async def test_pass2_never_voiced_via_reground_llm(tmp_path: Path) -> None:
    """B6 (T-0131 B2 isolation through the second client): its answer sentence and thought appear in no
    ``TtsSpeak``, history entry, HUD message, ``Episode`` or machine event, and no machine event carries its engine
    id.

    Narrowed by T-0180 (B1, R-1, B-R5): pass 2's thought may reach the HUD only as the ``text`` of an
    ``astra.thought`` with ``kind: "reground"``. Every other message, and every other field of ``astra.thought``, is
    searched as before; pass 2's answer is in no ``astra.thought`` text. Anti-vacuity: the thought **is** in a
    ``reground`` block's text."""
    second = SecondLlm()
    second.timed = [[(0, thought(SECRET)), (0, answer(SENTINEL)), (50, tool(click(900, 100)))]]
    rig, _ = await click_turn(tmp_path, second)
    assert second.calls, "premise: pass 2 ran on the second client"
    ids = {rid for rid, _ in second.calls}
    events = list(rig.machine.events())
    assert not [ev for ev in events if getattr(ev, "request_id", None) in ids], "T-0132 B6"
    speaks = [c.source for c in rig.machine.commands(C.TtsSpeak)]
    # T-0180 B-R5: the HUD searched is every message but astra.thought's ``text`` field
    split = split_thoughts(rig.published.messages)
    hud = split.elsewhere
    history = json.dumps([h.model_dump(mode="json") for h in rig.state.history])
    episodes = repr(getattr(rig.session.memory, "episodes", ""))
    machine = repr(events)
    for secret in (SENTINEL, SECRET, "SECOND-CLIENT-"):
        assert not any(secret in s for s in speaks), "voiced"
        for where, text in (
            ("HUD", hud),
            ("history", history),
            ("episode", episodes),
            ("machine", machine),
        ):
            assert secret not in text, f"T-0132 B6: the second client's pass 2 reached the {where}"
    for step, kind, _, text in split.thoughts:
        assert SENTINEL not in text, (
            f"T-0132 B6 (narrowed by T-0180 B2): pass 2's answer reached astra.thought text ({step}, {kind})"
        )
        assert "SECOND-CLIENT-" not in text.replace(SECRET, ""), (
            f"T-0132 B6 (narrowed by T-0180): only pass 2's thought may be in astra.thought text; {text!r}"
        )
        if kind != "reground":
            assert SECRET not in text, (
                f"T-0180 B6: pass 2's thought is shown under kind {kind!r}, not 'reground'"
            )
    assert any(SECRET in text for text in split.texts(kind="reground")), (
        "T-0180 B1 anti-vacuity (B-R5): the second client's committed pass-2 thought is in a kind='reground' "
        f"astra.thought text; published thoughts {split.thoughts}"
    )


# =====================================================================================================
# B2: keywords and closing
# =====================================================================================================


@pytest.mark.parametrize(
    ("reground", "keys"),
    [
        (True, ("reground_llm",)),
        (True, ("reground_visual_tokens",)),
        (False, ("reground_llm", "reground_visual_tokens")),
        (None, ("reground_llm", "reground_visual_tokens")),
    ],
    ids=["llm-only", "budget-only", "reground-false", "reground-absent"],
)
async def test_keywords_given_together(
    tmp_path: Path, reground: bool | None, keys: tuple[str, ...]
) -> None:
    """B2: ``reground_llm`` and ``reground_visual_tokens`` are given together or not at all, and only with
    ``reground``; each other combination is a ``ValueError`` at construction. Too-wide companion: both with
    ``reground=True`` construct."""
    second = SecondLlm()
    values = {"reground_llm": second, "reground_visual_tokens": 280}
    kwargs = {"visual_tokens": 1120, **{k: values[k] for k in keys}}
    with pytest.raises(ValueError):
        async with reground_rig(
            tmp_path / "bad", frames(), reground=reground, session_kwargs=kwargs
        ):
            pass
    async with reground_rig(
        tmp_path / "ok", frames(), session_kwargs={"visual_tokens": 1120, **values}
    ):
        pass


async def test_second_client_closed(tmp_path: Path) -> None:
    """B2 (reading: the session closes it): closing a session whose ``reground_llm`` is a second client awaits that
    client's ``aclose`` once."""
    second = SecondLlm()
    async with reground_rig(
        tmp_path,
        frames(),
        session_kwargs={
            "visual_tokens": 1120,
            "reground_llm": second,
            "reground_visual_tokens": 280,
        },
    ) as rig:
        assert second.closes == 0
        await rig.session.aclose()
    assert second.closes == 1, (
        f"T-0132 B2: the second client is closed with the session; {second.closes}"
    )


async def test_main_client_not_closed_with_equal_budget(tmp_path: Path) -> None:
    """B2 (review A, optional; mutant: the ``reground_llm is not self.llm`` guard dropped from ``aclose``): with
    equal budgets ``cli`` passes the main client itself as ``reground_llm``; closing the session leaves the main
    client's lifetime unchanged, so its ``aclose`` is never awaited by the session (a close counter on the main
    fake stays 0)."""
    import inspect

    from tests.unit import _tool_turn_support as tts
    from tests.unit._live_browser_support import hud_frame, live_kit, make_reground_llm

    browser = frames()
    main = make_reground_llm(browser, [])
    closes: list[int] = []

    async def aclose() -> None:
        closes.append(1)

    main.aclose = aclose
    real = tts.Session
    params = inspect.signature(real.__init__).parameters
    assert "reground_llm" in params and "reground_visual_tokens" in params, (
        "T-0132 B2: Session(..., reground_llm=..., reground_visual_tokens=...) does not exist yet"
    )

    class Same(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            k.setdefault("reground", True)
            k.setdefault("visual_tokens", 1120)
            k.setdefault("reground_llm", main)
            k.setdefault("reground_visual_tokens", 1120)
            super().__init__(*a, **k)

    tts.Session = Same  # type: ignore[misc]
    try:
        async with tts.tool_rig(
            tmp_path, kit=live_kit(browser), policy=tsl.LOOP, llm=main, frames=[hud_frame(0)]
        ) as rig:
            assert rig.session.llm is main and rig.session.reground_llm is main, "premise"
            await rig.session.aclose()
    finally:
        tts.Session = real  # type: ignore[misc]
    assert closes == [], f"T-0132 B2: the session closed its main client ({len(closes)} time(s))"
