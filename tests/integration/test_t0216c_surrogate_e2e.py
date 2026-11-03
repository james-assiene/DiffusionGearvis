"""T-0216c criteria 2 and 3 end to end: a typed ``astra.text`` carrying a lone surrogate over a real loopback socket.

The page sends ``{"type":"astra.text","text":"a\\ud800 hi"}`` (the JSON escape a browser's ``JSON.stringify``
emits for broken text). Before T-0216c the codec hands ``"a\\ud800 hi"`` to the session; the echo
``astra.transcript`` then fails ``codec.encode`` in ``HudServer.publish``, the ``hud-writer`` task dies, the page
receives nothing more, and a reloaded page stalls on the journal's replayed transcript (the T-0216b reviewer's
probes ``rv216b_probe/p_wire.py`` and ``p_reload.py``).

**The oracle is a control run** (never a list written into the test): the same scenario, same script, same
clock, with the page typing ``"a\\ufffd hi"`` instead -- criterion 1 says decode returns exactly what it returns
for the text with U+FFFD in place, so everything downstream of the codec must be what the control produced.

The rig is T-0028's ``served`` + ``HudClient`` (``tests/unit/_hud_rig.py``): ``HudServer`` on ``127.0.0.1:0``,
the production codec on both ends, ``ScriptedClock``, fakes from ``tests/fakes/``. The memory client is the
rig's ``FakeMemory`` behind a spy that *records* every string the session hands it that has no UTF-8 encoding
(it does not raise: the real ``SqliteMemory.search`` tokenises its query and was measured not to raise on
``"a\ud800 hi"``, 2026-10-31, while ``sqlite3`` itself refuses such a bound parameter). The row asserts the goal's
"nothing downstream ever sees one", which holds for any store.

Regime: no measurement; n = 1 scenario per arm (bad, control), two turns each, no GPU, no browser.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.codec import Malformed, Unknown
from local_astra.contracts.protocol import AstraError, AstraSpoken, AstraTranscript
from local_astra.contracts.turn import TurnState
from tests.unit import _t0181_pins as t0181
from tests.unit._hud_rig import HudClient, answer, served
from tests.unit._session_log_rig import make_log, of_kind, read_records

pytestmark = [pytest.mark.integration]

#: what the page types, as JSON text on the wire (a literal backslash-u escape, not a Python surrogate)
BAD_FRAME = '{"type":"astra.text","text":"a\\ud800 hi"}'
#: the control: the same text with U+FFFD where the lone surrogate was
CONTROL_FRAME = '{"type":"astra.text","text":"a\\ufffd hi"}'
REPLACED = "a� hi"
SECOND_FRAME = '{"type":"astra.text","text":"plain second"}'
FIRST_ANSWER = "Okay then. "
SECOND_ANSWER = "Second. "
BUDGET_MS = 6_000
SETTLE_MS = 500
TAIL_MS = 300
#: T-0043 criterion 18 + T-0181 A4: the hello reply's live head (``tests/contract/test_hud_hello_replay_wire.py``)
LIVE_HEAD = tuple(t0181.hello_head(("astra.assist.state", "astra.turn.state", "astra.log.state")))


def _strings(obj: Any) -> list[str]:
    """Every ``str`` inside ``obj`` (pydantic models dumped, mappings' keys included)."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        out: list[str] = []
        for k, v in obj.items():
            out += _strings(k) + _strings(v)
        return out
    if isinstance(obj, list | tuple | set | frozenset):
        return [s for v in obj for s in _strings(v)]
    return []


def _encodable(s: str) -> bool:
    try:
        s.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


@dataclass
class MemorySpy:
    """Every call the session makes on the memory client, and every string in it with no UTF-8 encoding."""

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)

    def install(self, memory: Any) -> None:
        for name in [n for n in dir(memory) if not n.startswith("_")]:
            orig = getattr(memory, name)
            if not asyncio.iscoroutinefunction(orig):
                continue
            setattr(memory, name, self._wrap(name, orig))

    def _wrap(self, name: str, orig: Any) -> Any:
        async def call(*args: Any, **kwargs: Any) -> Any:
            self.calls.append((name, args, kwargs))
            bad = [s for s in _strings([args, kwargs]) if not _encodable(s)]
            if bad:
                self.refused.append((name, repr(bad[0])[:80]))
            return await orig(*args, **kwargs)

        return call

    def strings(self) -> list[str]:
        return _strings([[a, k] for _, a, k in self.calls])


@dataclass
class Outcome:
    page: list[Any] = field(default_factory=list)
    first_turn_end: int = 0
    reload: list[Any] = field(default_factory=list)
    prompts: list[Any] = field(default_factory=list)
    memory: MemorySpy = field(default_factory=MemorySpy)
    writer_errors: list[str] = field(default_factory=list)
    #: what the session swallowed: ``session.failures`` reprs plus the session log's ``error`` rows
    failures: list[str] = field(default_factory=list)
    #: an exception that escaped the scenario (e.g. out of ``Session.aclose``), or None
    crashed: str | None = None


def _types(messages: list[Any]) -> list[str]:
    return [getattr(m, "type", type(m).__name__) for m in messages]


async def _read_tail(client: HudClient) -> None:
    """Empty the socket before closing (the T-0054 A10 bounded-queue trap); pure reading, clock still."""
    with contextlib.suppress(AssertionError):
        await client.drain(until=lambda got: False, rig=None, budget_ms=TAIL_MS)


def _writer_errors() -> list[str]:
    out = []
    for task in asyncio.all_tasks():
        done = task.get_name() == "hud-writer" and task.done() and not task.cancelled()
        if done and task.exception() is not None:
            out.append(repr(task.exception())[:160])
    return out


async def run_scenario(tmp: Path, first_frame: str) -> Outcome:
    """Page one types ``first_frame``, the turn runs, a reloaded page greets, then page one types a second turn.

    Total: an exception escaping the rig (the session log's writer dying on the surrogate surfaces from
    ``Session.aclose`` today) is recorded in ``crashed`` so the rows fail on an assertion, not a fixture error.
    """
    out = Outcome()
    log = make_log(tmp / "logs")
    try:
        async with served(
            tmp / "data", session_log=log, script=[answer(FIRST_ANSWER), answer(SECOND_ANSWER)]
        ) as box:
            out.memory.install(box.rig.memory)

            def idle_after(n: int, text: str) -> Any:
                def done(got: list[Any]) -> bool:
                    return box.rig.state.state is TurnState.IDLE and any(
                        isinstance(m, AstraSpoken) and m.text.strip() == text.strip()
                        for m in got[n:]
                    )

                return done

            try:
                async with HudClient.connect(box.url) as one:
                    out.page = one.received
                    await one.hello()
                    await one.drain(until=None, rig=box.rig, budget_ms=SETTLE_MS)
                    n0 = len(one.received)
                    await one.send(first_frame)
                    # a shortfall is what the rows assert on; the control run says what was due
                    with contextlib.suppress(AssertionError):
                        await one.drain(
                            until=idle_after(n0, FIRST_ANSWER), rig=box.rig, budget_ms=BUDGET_MS
                        )
                    await _read_tail(one)
                    out.first_turn_end = len(one.received)

                    # criterion 2: a reloaded page greets while the journal holds the first turn
                    async with HudClient.connect(box.url) as two:
                        out.reload = two.received
                        await two.hello()
                        await _read_tail(two)
                        await _read_tail(two)

                    n1 = len(one.received)
                    await one.send(SECOND_FRAME)
                    # a shortfall is what the rows assert on; the control run says what was due
                    with contextlib.suppress(AssertionError):
                        await one.drain(
                            until=idle_after(n1, SECOND_ANSWER), rig=box.rig, budget_ms=BUDGET_MS
                        )
                    await _read_tail(one)
                    out.writer_errors = _writer_errors()
            finally:
                out.prompts = [p for _, p in box.rig.llm.gen_prompts]
                out.failures = [repr(e)[:160] for e in box.rig.session.failures]
    except Exception as exc:  # recorded; every row asserts it is None
        out.crashed = f"{type(exc).__name__}: {str(exc)[:160]}"
    if log.path.exists():
        out.failures += [str(r.get("data"))[:160] for r in of_kind(read_records(log.path), "error")]
    return out


def _no_crash(arm: str, o: Outcome) -> None:
    assert o.crashed is None, (
        f"criteria 2-3 ({arm} run): an exception escaped the session: {o.crashed}"
    )


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Outcome]:
    """Both arms once per module (each is a few seconds of loopback traffic)."""
    out: dict[str, Outcome] = {}
    for arm, frame in (("control", CONTROL_FRAME), ("bad", BAD_FRAME)):
        out[arm] = asyncio.run(run_scenario(tmp_path_factory.mktemp(f"t0216c_{arm}"), frame))
    return out


def _premise_control(c: Outcome) -> None:
    transcripts = [m.text for m in c.page if isinstance(m, AstraTranscript) and not m.replay]
    assert REPLACED in transcripts and "plain second" in transcripts, (
        f"premise: the control run (U+FFFD typed) echoes both turns; it echoed {transcripts}"
    )


def _spoken(messages: list[Any]) -> list[str]:
    """The distinct live utterances, in order (one utterance can be published more than once as it plays)."""
    seen = dict.fromkeys(
        (m.utterance_id, m.text.strip())
        for m in messages
        if isinstance(m, AstraSpoken) and not m.replay
    )
    return [text for _, text in seen]


def test_the_echo_transcript_carries_u_fffd(runs: dict[str, Outcome]) -> None:
    """Criterion 2: the typed text's ``astra.transcript`` echo arrives, with U+FFFD in place of the surrogate."""
    _no_crash("control", runs["control"])
    _premise_control(runs["control"])
    bad = runs["bad"]
    first = bad.page[: bad.first_turn_end]
    echoes = [m.text for m in first if isinstance(m, AstraTranscript) and not m.replay]
    assert echoes == [REPLACED], (
        f"criterion 2: the page gets the echo {REPLACED!r} for a typed 'a\\ud800 hi'; it got {echoes!r} "
        f"(page received {_types(first)})"
    )
    assert not [m for m in bad.page if isinstance(m, Malformed | Unknown | AstraError)], (
        f"criterion 2: nothing the server sent is malformed or an error; {bad.page}"
    )
    _no_crash("bad", runs["bad"])


def test_the_first_turn_runs_and_the_second_turns_messages_still_arrive(
    runs: dict[str, Outcome],
) -> None:
    """Criterion 2: the turn runs (its answer is spoken to the page) and a second turn's messages still arrive."""
    _no_crash("control", runs["control"])
    control, bad = runs["control"], runs["bad"]
    _premise_control(control)
    halves = (control.page[: control.first_turn_end], control.page[control.first_turn_end :])
    assert all(_spoken(h) for h in halves), (
        f"premise: the control page hears an answer in each turn; it heard {[_spoken(h) for h in halves]}"
    )
    assert [_spoken(bad.page[: bad.first_turn_end]), _spoken(bad.page[bad.first_turn_end :])] == [
        _spoken(h) for h in halves
    ], (
        "criterion 2: after a typed lone surrogate the page hears the first turn's answer and the second turn's, as "
        f"the U+FFFD control does ({_spoken(control.page)}); it heard {_spoken(bad.page)}"
    )
    second = [
        m.text
        for m in bad.page[bad.first_turn_end :]
        if isinstance(m, AstraTranscript) and not m.replay
    ]
    assert second == ["plain second"], (
        f"criterion 2: the second turn's echo reaches the same page; got {second}"
    )
    assert bad.writer_errors == [], f"criterion 2: no hud-writer task died: {bad.writer_errors}"
    _no_crash("bad", runs["bad"])


def test_a_reloaded_page_gets_its_full_hello_reply(runs: dict[str, Outcome]) -> None:
    """Criterion 2: a page that greets after the turn gets the whole hello reply, journal (with the echo) included."""
    _no_crash("control", runs["control"])
    control, bad = runs["control"], runs["bad"]
    assert _types(control.reload)[: len(LIVE_HEAD)] == list(LIVE_HEAD), (
        f"premise: the control reload opens with the live head {LIVE_HEAD}; got {_types(control.reload)}"
    )
    replayed = [m.text for m in control.reload if isinstance(m, AstraTranscript)]
    assert replayed == [REPLACED], (
        f"premise: the control journal replays the transcript; got {replayed}"
    )
    assert _types(bad.reload) == _types(control.reload), (
        "criterion 2: the reloaded page's hello reply is the control's, message for message; control "
        f"{_types(control.reload)}, got {_types(bad.reload)}"
    )
    assert [m.text for m in bad.reload if isinstance(m, AstraTranscript)] == [REPLACED], (
        f"criterion 2: the replayed transcript carries U+FFFD; got {bad.reload}"
    )
    _no_crash("bad", runs["bad"])


def test_the_llm_request_carries_the_replaced_text(runs: dict[str, Outcome]) -> None:
    """Criterion 3: the fake LLM's request for that turn carries ``a\\ufffd hi`` and is UTF-8 encodable throughout."""
    _no_crash("control", runs["control"])
    control, bad = runs["control"], runs["bad"]
    assert len(control.prompts) == 2, (
        f"premise: the control run makes two requests; {len(control.prompts)}"
    )
    assert REPLACED in control.prompts[0].messages[-1].text(), (
        "premise: the control request carries the text"
    )
    assert len(bad.prompts) == 2, (
        f"criterion 2/3: both turns reach the LLM; {len(bad.prompts)} requests"
    )
    first = bad.prompts[0]
    assert REPLACED in first.messages[-1].text(), (
        f"criterion 3: the first request's user text carries {REPLACED!r}; it ends {first.messages[-1].text()[-80:]!r}"
    )
    assert first.model_dump(mode="json") == control.prompts[0].model_dump(mode="json"), (
        "criterion 3: the request is exactly the control's (the codec replaced the surrogate, nothing else)"
    )
    for p in bad.prompts:
        body = json.dumps(p.model_dump(mode="json"), ensure_ascii=False)
        assert _encodable(body), (
            "criterion 3: the request body has a UTF-8 encoding (httpx would raise otherwise)"
        )
    _no_crash("bad", runs["bad"])


def test_memory_sees_only_encodable_text_and_nothing_raises(runs: dict[str, Outcome]) -> None:
    """Criterion 3: every memory call carries text sqlite3 can store, the typed text arrives replaced, nothing raises."""
    _no_crash("control", runs["control"])
    control, bad = runs["control"], runs["bad"]
    assert control.memory.calls, "premise: the session talks to the memory client during a turn"
    assert bad.memory.refused == [], (
        f"criterion 3 / Goal: a memory call carried a lone surrogate (a bound sqlite3 parameter raises on one): {bad.memory.refused}"
    )
    if any(REPLACED in s for s in control.memory.strings()):
        assert any(REPLACED in s for s in bad.memory.strings()), (
            f"criterion 3: the memory query carries {REPLACED!r}, as the control's does"
        )
    assert control.failures == [], (
        f"premise: the control run swallows no failure: {control.failures}"
    )
    assert bad.failures == [], (
        f"criterion 3: nothing raised (swallowed failures / error rows): {bad.failures}"
    )
    _no_crash("bad", runs["bad"])
