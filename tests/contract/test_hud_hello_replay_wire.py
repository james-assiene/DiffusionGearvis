"""T-0037 group A over a real socket: what a reconnecting page is actually sent.

``tests/unit/test_hud_hello_replay.py`` asserts the journal through ``HudServer.open()``; this
module asserts the same thing through ``ws://127.0.0.1:0/v1/realtime`` with the *production*
codec doing the decoding, because the field this task adds (``replay``) is a wire change on both
sides: ``contracts/base.Contract`` forbids unknown fields and ``hud/src/protocol.js`` mirrors that
literally, so a flag added to Python alone turns every replayed message into ``kind: "malformed"``
on the page — counted, ignored, invisible (docs/tasks/T-0037.md, implementation notes).

Fixtures are T-0028's rig (``served``/``HudClient``/``drain`` from ``tests/unit/_hud_rig.py``,
the same ones ``tests/contract/test_hud_endpoint.py`` uses): loopback socket, virtual session
clock, budgets in virtual milliseconds (ruling A13).

Regime: no measurement; n = 1 turn per row, ``ScriptedClock``, no GPU.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_astra.contracts.codec import Malformed, Unknown, encode
from local_astra.contracts.protocol import (
    SERVER_TO_CLIENT,
    AstraAssistState,
    AstraSpoken,
    AstraSpokenPlayed,
    AstraTranscript,
    AstraTurnState,
    AstraUiSurface,
)
from local_astra.contracts.turn import TurnState
from tests.unit import _t0181_pins as t0181
from tests.unit._hud_rig import HudClient, answer, served, ui

pytestmark = [pytest.mark.contract]

BUDGET_MS = 5_000
SETTLE_MS = 500
#: Rounds of *pure reading* (no rig, so the session clock does not move) spent emptying the
#: socket before a client is closed. See ``read_the_turns_tail`` below: this is rig hygiene,
#: not a wait for anything the rows assert, so it is small and never grows a budget.
TAIL_MS = 200
QUESTION = "what is on this screen"

#: The five models criterion 2 adds ``replay`` to. Nothing else on the wire grows a field.
REPLAY_MODELS = (AstraTranscript, AstraSpoken, AstraUiSurface, AstraSpokenPlayed, AstraTurnState)


def turn_content_is_in(got: list) -> bool:
    """The first page has heard the whole of ``SCRIPT``: both spoken lines and the ui surface.

    Two rows below used to stop at the second spoken line, leaving the rest of the turn unread on
    the socket. That is a trap with a ten-second bill (T-0054 ruling A10): ``websockets`` gives a
    client a bounded receive queue (``max_queue``, 32 by default), and a client that stops reading
    while the server keeps publishing fills it, stops reading the socket at all, never sees the
    server's close frame, and pays the default ten-second ``close_timeout`` on the way out of its
    ``async with``. One extra message per ui commit was enough to cross it, and two 0.5 s rows
    became 10.4 s rows. **Drain depth is rig hygiene; it is not what these rows assert** — the
    replay assertions below are untouched, and reading the turn out is only how the socket is left
    in a state where closing it is free.
    """
    return len([m for m in got if isinstance(m, AstraSpoken)]) >= 2 and any(
        isinstance(m, AstraUiSurface) for m in got
    )


async def read_the_turns_tail(client: HudClient) -> None:
    """Empty whatever is still on the socket, without advancing the session's clock.

    ``rig=None`` means pure reads: the ``ScriptedClock`` only moves when a drain with a rig moves
    it, so the turn stays exactly where the caller left it and "mid-turn" stays a fact about the
    session. This is the half of the fix that survives the turn growing another message: the
    queue bound that tipped once will tip again on the next task that adds one.
    """
    await client.drain(until=None, rig=None, budget_ms=TAIL_MS)


SCRIPT = [answer("Here is a quick check. "), answer("Try the five questions. "), ui(canvas_idx=1)]


def require_the_field() -> None:
    """Criterion 2, asserted before anything constructs a replayed message.

    Without this the rows below would die inside a pydantic constructor with
    ``extra_forbidden``, which ``tests/conftest.py`` does not accept as a pending outcome — and
    rightly: the missing field is the acceptance criterion, so it must read as one.
    """
    missing = [m.__name__ for m in REPLAY_MODELS if "replay" not in m.model_fields]
    assert not missing, (
        "criterion 2: AstraTranscript, AstraSpoken, AstraUiSurface, AstraSpokenPlayed and "
        f"AstraTurnState each gain `replay: bool = False` (local_astra/contracts/protocol.py); "
        f"{missing} do not carry it"
    )


#: Criterion 18 of docs/tasks/T-0043.md: the live head of a hello reply is **three** messages —
#: the assist state, the turn state and then ``astra.log.state``, "before the journal replay".
#: T-0037 pinned it at two here; this is the same pin moved by the one message that joined it, not
#: a slice loosened. It is asserted (``journalled`` below) rather than assumed, because a reply
#: that lost its log state would otherwise just start the journal one message earlier and every
#: ``replay is True`` assertion below would still hold over a shorter list.
#:
#: The log state is *not* part of the journal and carries no ``replay`` field at all (the contract
#: row ``test_the_log_pair_is_one_direction_each_...`` in tests/contract/test_protocol_codec.py
#: pins its absence): what a reconnecting page is owed is the *current* state of the recorder, and
#: a replayed one would be a stale claim about whether the session is being recorded.
#: T-0181 A4 pin move (A-R8): ``astra.mic.state`` joins the head directly after the log state once the tree has it
#: (like the log state, it is not journalled and carries no ``replay`` field).
LIVE_HEAD = tuple(t0181.hello_head(("astra.assist.state", "astra.turn.state", "astra.log.state")))


def journalled(received: list) -> list:
    """The replayed part of a hello reply: everything after the three live messages."""
    head = [getattr(m, "type", type(m).__name__) for m in received[: len(LIVE_HEAD)]]
    assert head == list(LIVE_HEAD), (
        "T-0043 criterion 18 (over T-0037 criterion 1): a hello reply opens with the assist "
        "state, the turn state and the log state, in that order, and the journal starts after "
        f"them. This reply opened with {head}"
    )
    return [m for m in received[len(LIVE_HEAD) :] if not isinstance(m, AstraAssistState)]


def keys_by_type(messages: list) -> dict[str, set[str]]:
    """The wire shape of each message kind: what the real codec puts in the JSON object."""
    out: dict[str, set[str]] = {}
    for message in messages:
        out.setdefault(message.type, set()).update(json.loads(encode(message)))
    return out


async def _turn_in_progress(box, client: HudClient) -> list[AstraSpoken]:
    """Drive the rig until the page has heard the turn's first spoken line, and stop there.

    The clock is virtual and only moves when this loop moves it, so the turn stays exactly here
    while the second page connects: "mid-turn" is a fact about the session, not a race.
    """
    await box.rig.onset()
    await box.rig.words(QUESTION)
    await box.rig.end_of_turn()
    await client.drain(
        until=lambda got: any(isinstance(m, AstraSpoken) for m in got),
        rig=box.rig,
        budget_ms=BUDGET_MS,
    )
    assert box.rig.state.state.name in ("SPEAKING", "THINKING"), (
        f"premise: the turn must still be open when the second page greets; it is "
        f"{box.rig.state.state}"
    )
    return [m for m in client.received if isinstance(m, AstraSpoken)]


async def test_a_second_client_greeting_mid_turn_receives_the_turns_spoken_lines(tmp_path):
    """Criteria 1 and 9 over the wire: the catch-up is the lines the first page already has.

    The oracle is the *first* connection's traffic — what this server really published for this
    turn — not a list written into the test: criterion 3 is "nothing is invented", and a test that
    named the utterances itself could not tell an invented line from a journalled one.
    """
    require_the_field()
    async with served(tmp_path, script=SCRIPT, ms_per_word=100) as box:
        async with HudClient.connect(box.server.url) as one:
            await one.hello()
            await one.drain(until=None, rig=box.rig, budget_ms=SETTLE_MS)
            live = await _turn_in_progress(box, one)
            assert live, "premise: the first page must have heard at least one utterance"

            async with HudClient.connect(box.server.url) as two:
                await two.hello()
                received = await two.drain(
                    until=lambda got: (
                        len([m for m in got if isinstance(m, AstraSpoken)]) >= len(live)
                    ),
                    rig=None,
                    budget_ms=SETTLE_MS,
                )

        caught_up = [m for m in received if isinstance(m, AstraSpoken)]
        assert [m.utterance_id for m in caught_up] == [m.utterance_id for m in live], (
            "criterion 1: a page that greets mid-turn is caught up on the turn's astra.spoken "
            "lines, in publication order — today it is sent a live SPEAKING chip over an empty "
            f"pane. The first page had {[m.utterance_id for m in live]}; the fresh one got "
            f"{[m.utterance_id for m in caught_up]}"
        )
        assert [m.text for m in caught_up] == [m.text for m in live], (
            "criterion 3: field for field the lines the session published, not a summary"
        )
        transcripts = [m for m in received if isinstance(m, AstraTranscript)]
        assert [m.text for m in transcripts] == [QUESTION], (
            "criterion 1: and the turn's last transcript, so the question is not blank under the "
            f"answer; the page got {[m.text for m in transcripts]}"
        )
        assert received.index(transcripts[0]) < received.index(caught_up[0]), (
            f"criterion 1: transcript before the spoken lines; the reply was "
            f"{[m.type for m in received]}"
        )


async def test_every_replayed_message_decodes_and_is_server_to_client_with_replay_true(tmp_path):
    """Criterion 2 on the wire, and T-0028 criterion 9's rule applied to the replay.

    ``HudClient.drain`` decodes with ``local_astra/contracts/codec.py``; a ``Malformed`` here is
    what the page's mirror would also produce, which is the whole reason group B is not optional.
    """
    require_the_field()
    async with served(tmp_path, script=SCRIPT, ms_per_word=100) as box:
        async with HudClient.connect(box.server.url) as one:
            await one.hello()
            await one.drain(until=None, rig=box.rig, budget_ms=SETTLE_MS)
            await box.rig.ask(QUESTION)
            await box.rig.speak_out()
            # ``drain`` reads at most one message per round, so the budget must cover the turn —
            # all of it, tail included (``turn_content_is_in``).
            await one.drain(until=turn_content_is_in, rig=box.rig, budget_ms=BUDGET_MS)
            await read_the_turns_tail(one)
            live = [m for m in one.received if isinstance(m, AstraSpoken | AstraUiSurface)]
            assert live, f"premise: the turn published something; it sent {one.received}"
            assert all(m.replay is False for m in live), (
                "criterion 2, negative: a message published live always carries replay=False; "
                f"the wire carried {[(m.type, m.replay) for m in live]}"
            )

            async with HudClient.connect(box.server.url) as two:
                await two.hello()
                received = await two.drain(
                    until=lambda got: (
                        any(isinstance(m, AstraSpokenPlayed) for m in got)
                        or len([m for m in got if isinstance(m, AstraSpoken)]) >= 2
                    ),
                    rig=None,
                    budget_ms=BUDGET_MS,
                )

        assert not [m for m in received if isinstance(m, Malformed | Unknown)], (
            "criterion 2: every replayed message decodes under the production codec — an unknown "
            f"field would make it Malformed on the page too; got {received}"
        )
        replayed = journalled(received)
        assert replayed, f"criterion 1: nothing was replayed; the reply was {received}"
        for message in replayed:
            assert message.type in SERVER_TO_CLIENT, (
                f"T-0028 criterion 9: {message.type!r} is not a server→client type"
            )
        marked = [m for m in replayed if not isinstance(m, AstraTurnState)]
        assert marked and all(m.replay is True for m in marked), (
            "criterion 2: every replayed message carries replay=True on the wire; got "
            f"{[(m.type, m.replay) for m in marked]}"
        )
        assert all("replay" in json.loads(encode(m)) for m in marked), (
            "…and the flag is a real field of the JSON object, not a Python-side attribute"
        )


async def test_the_fake_and_the_real_endpoint_agree_on_the_replayed_shape(tmp_path):
    """Criterion 9: ``tests/fakes/hud_server.py`` speaks exactly what the endpoint speaks.

    The page-side rows (``tests/integration/test_hud_headless.py``) drive against ``HudStub``; if
    its replayed messages carried a field the real endpoint does not send — or missed one it does
    — the page would be tested against a wire nobody speaks (CLAUDE.md: changing a wire format
    means changing the fake in the same PR). The comparison is per type, over the JSON keys the
    production codec puts on the wire.
    """
    require_the_field()
    from tests.fakes.hud_server import replay_script

    fake = keys_by_type(list(replay_script()))
    async with (
        served(tmp_path, script=SCRIPT, ms_per_word=100) as box,
        HudClient.connect(box.server.url) as one,
    ):
        await one.hello()
        await one.drain(until=None, rig=box.rig, budget_ms=SETTLE_MS)
        await box.rig.ask(QUESTION)
        await box.rig.speak_out()
        await one.drain(until=turn_content_is_in, rig=box.rig, budget_ms=BUDGET_MS)
        await read_the_turns_tail(one)
        async with HudClient.connect(box.server.url) as two:
            await two.hello()
            received = await two.drain(
                until=lambda got: (
                    any(isinstance(m, AstraSpokenPlayed) for m in got)
                    or len([m for m in got if isinstance(m, AstraSpoken)]) >= 2
                ),
                rig=None,
                budget_ms=BUDGET_MS,
            )

    real = keys_by_type(journalled(received))
    shared = sorted(set(fake) & set(real))
    assert shared, (
        "criterion 9: the stub's replay script and the endpoint's replay must have message types "
        f"in common; the stub speaks {sorted(fake)} and the endpoint sent {sorted(real)}"
    )
    for kind in shared:
        assert fake[kind] == real[kind], (
            f"criterion 9: {kind} has fields {sorted(fake[kind])} in tests/fakes/hud_server.py "
            f"and {sorted(real[kind])} on the real endpoint — the page would be driven against a "
            "shape the conductor does not speak"
        )
    assert "astra.spoken" in shared and "replay" in real["astra.spoken"], (
        f"…and the field this task adds is in both; the endpoint sent {sorted(real)}"
    )


# --- T-0090 criterion 3: the tool card is journalled ----------------------------------------------
#
# docs/tasks/T-0090.md criterion 3: the journal "holds the latest message per call_id of the open
# turn and replays it on astra.hello after the spoken lines and before the played reports. It is
# cleared on a new turn. It is never dropped on outbox overflow (only astra.turn.state may be
# dropped)." The publisher of astra.tool.call is T-0092's session wiring, which does not exist yet,
# so these rows publish through ``HudServer.publish`` — the session's ``HudPublisher``, the one
# door every journalled message goes through (T-0037 criterion 3) — onto a real socket, and read
# the reply with the production codec.
#
# The wire order is the order the server really publishes in: the turn's spoken lines and its
# played report first (the session publishes them while speaking), then the action's lifecycle
# (T-0080d: execute only when the turn completes, after the spoken prefix has drained). Publication
# order is therefore *not* replay order — the played report was published first and must be
# replayed last — which is what makes the position assertion discriminating.

TOOL_CALL_TYPE = "astra.tool.call"


def tool_call_model():
    import local_astra.contracts.protocol as protocol

    model = getattr(protocol, "AstraToolCall", None)
    assert model is not None, (
        "T-0090 criterion 1: contracts/protocol.py gains AstraToolCall(type='astra.tool.call', "
        "turn_id, call_id, name, arguments, status, source, action, reason, query_problem, "
        "replay=False) in SERVER_TO_CLIENT"
    )
    return model


def lifecycle(turn_id: str) -> list:
    """Two calls of one turn, each through its real lifecycle, in wire order (hand-written)."""
    model = tool_call_model()

    def call(call_id: str, **kw: object):
        base: dict[str, object] = {
            "turn_id": turn_id,
            "call_id": call_id,
            "name": "click_at",
            "arguments": {"x": 1200, "y": 5},
            "source": "draft",
        }
        base.update(kw)
        return model(**base)

    return [
        call(f"{turn_id}-c1", status="proposed"),
        call(f"{turn_id}-c1", status="repairing"),
        call(
            f"{turn_id}-c1",
            status="dry_run",
            source="repair",
            arguments={"x": 120, "y": 5},
            action="would click at (153, 4) on screen",
        ),
        call(
            f"{turn_id}-c2",
            name="google_search",
            arguments={"query": "the"},
            status="proposed",
            query_problem="query is a single stopword",
        ),
        call(
            f"{turn_id}-c2",
            name="google_search",
            arguments={"query": "the"},
            status="refused",
            reason="no frame",
            query_problem="query is a single stopword",
        ),
    ]


def reply_is_complete(got: list) -> bool:
    """The hello reply ends with the replayed repeat of the current turn state (T-0037 A3)."""
    return any(isinstance(m, AstraTurnState) and m.replay for m in got)


async def _spoken_and_played_turn(box, client: HudClient) -> str:
    """Run one turn to IDLE with its played report on the wire; return its turn id."""
    seen = len(client.received)
    await box.rig.ask(QUESTION)
    await box.rig.speak_out()
    await client.drain(
        until=lambda got: (
            turn_content_is_in(got[seen:])
            and any(isinstance(m, AstraSpokenPlayed) for m in got[seen:])
        ),
        rig=box.rig,
        budget_ms=BUDGET_MS,
    )
    await read_the_turns_tail(client)
    spoken = [m for m in client.received[seen:] if isinstance(m, AstraSpoken)]
    return str(spoken[-1].turn_id)


async def test_the_latest_tool_call_per_call_id_is_replayed_between_spoken_and_played(tmp_path):
    """Criterion 3: latest-per-call_id, marked replay=True, after the spoken lines, before played."""
    tool_call_model()
    async with served(tmp_path, script=SCRIPT, ms_per_word=100) as box:  # noqa: SIM117 - the second page greets while the first is open
        async with HudClient.connect(box.server.url) as one:
            await one.hello()
            await one.drain(until=None, rig=box.rig, budget_ms=SETTLE_MS)
            turn = await _spoken_and_played_turn(box, one)
            published = lifecycle(turn)
            for message in published:
                box.server.publish(message)
            live = await one.drain(
                until=lambda got: (
                    len([m for m in got if getattr(m, "type", "") == TOOL_CALL_TYPE])
                    >= len(published)
                ),
                rig=None,
                budget_ms=SETTLE_MS,
            )
            live_calls = [m for m in live if getattr(m, "type", "") == TOOL_CALL_TYPE]
            assert [(m.call_id, m.status) for m in live_calls] == [
                (m.call_id, m.status) for m in published
            ], "premise: the live page is sent every lifecycle message, in publication order"
            assert all(m.replay is False for m in live_calls), (
                "criterion 1: a tool call published live carries replay=False"
            )

            async with HudClient.connect(box.server.url) as two:
                await two.hello()
                received = await two.drain(until=reply_is_complete, rig=None, budget_ms=BUDGET_MS)

    assert not [m for m in received if isinstance(m, Malformed | Unknown)], (
        f"criterion 1/3: every replayed message decodes under the production codec; got {received}"
    )
    replayed = journalled(received)
    calls = [m for m in replayed if getattr(m, "type", "") == TOOL_CALL_TYPE]
    by_id = {m.call_id: m for m in calls}
    assert len(calls) == 2 and set(by_id) == {f"{turn}-c1", f"{turn}-c2"}, (
        "criterion 3: the journal holds the *latest* message per call_id — one per call, not the "
        f"history; the reply carried {[(m.call_id, m.status) for m in calls]}"
    )
    c1, c2 = by_id[f"{turn}-c1"], by_id[f"{turn}-c2"]
    assert (c1.status, c1.source, c1.arguments) == ("dry_run", "repair", {"x": 120, "y": 5}), (
        f"criterion 3: c1's latest is the repaired dry run; got {c1!r}"
    )
    assert (c2.status, c2.reason, c2.query_problem) == (
        "refused",
        "no frame",
        "query is a single stopword",
    ), f"criterion 3: c2's latest is the refusal, warning kept; got {c2!r}"
    assert all(m.replay is True for m in calls), (
        f"criterion 3 / T-0037: a replayed card is marked; got {[(m.call_id, m.replay) for m in calls]}"
    )
    types = [getattr(m, "type", "") for m in replayed]
    last_spoken = max(i for i, t in enumerate(types) if t == "astra.spoken")
    first_played = min(i for i, t in enumerate(types) if t == "astra.spoken.played")
    positions = [i for i, t in enumerate(types) if t == TOOL_CALL_TYPE]
    assert all(last_spoken < i < first_played for i in positions), (
        "criterion 3: the cards are replayed after the spoken lines and before the played "
        f"reports — although they were published after the played report. The reply was {types}"
    )


async def test_a_new_turn_clears_the_tool_calls_and_keeps_the_new_turns_own(tmp_path):
    """Criterion 3, both directions: turn 1's cards are gone after turn 2 opens; turn 2's stay."""
    tool_call_model()
    async with served(tmp_path, script=SCRIPT, ms_per_word=100) as box:  # noqa: SIM117 - the second page greets while the first is open
        async with HudClient.connect(box.server.url) as one:
            await one.hello()
            await one.drain(until=None, rig=box.rig, budget_ms=SETTLE_MS)
            first = await _spoken_and_played_turn(box, one)
            for message in lifecycle(first):
                box.server.publish(message)
            second = await _spoken_and_played_turn(box, one)
            assert second != first, f"premise: the second turn has its own id; both are {first!r}"

            async with HudClient.connect(box.server.url) as two:
                await two.hello()
                after_new_turn = await two.drain(
                    until=reply_is_complete, rig=None, budget_ms=BUDGET_MS
                )
            assert not [m for m in after_new_turn if getattr(m, "type", "") == TOOL_CALL_TYPE], (
                "criterion 3: the journal is cleared on a new turn — turn 1's cards must not be "
                "replayed over turn 2's lines; the reply carried "
                f"{[(m.type, getattr(m, 'call_id', None)) for m in after_new_turn]}"
            )

            own = lifecycle(second)[:1]  # turn 2's own proposal
            for message in own:
                box.server.publish(message)
            await one.drain(until=None, rig=None, budget_ms=TAIL_MS)
            async with HudClient.connect(box.server.url) as three:
                await three.hello()
                later = await three.drain(until=reply_is_complete, rig=None, budget_ms=BUDGET_MS)
    calls = [m for m in journalled(later) if getattr(m, "type", "") == TOOL_CALL_TYPE]
    assert [(m.call_id, m.status) for m in calls] == [(f"{second}-c1", "proposed")], (
        "criterion 3, the other direction: a guard that clears too much is as wrong as one that "
        "clears nothing — the open turn's own card is replayed; got "
        f"{[(m.call_id, m.status) for m in calls]}"
    )


def test_a_tool_call_is_never_dropped_by_the_outbox():
    """Criterion 3: only ``astra.turn.state`` may be dropped; a tool call is never silently lost.

    Hand-computed over ``Outbox`` (T-0028's bound, unchanged by this task): a full queue refuses a
    tool call (``False`` — the caller closes the connection, ruling A12) rather than accepting and
    discarding it, and a turn state arriving at a full queue never evicts a tool call.
    """
    from local_astra.conductor.hud_server import Outbox
    from local_astra.contracts.turn import TurnState

    proposed = lifecycle("t1")[0]

    full = Outbox(maxsize=2)
    assert full.offer(AstraSpoken(turn_id="t1", utterance_id="u1", text="one")) is True
    assert full.offer(AstraSpoken(turn_id="t1", utterance_id="u2", text="two")) is True
    assert full.offer(proposed) is False, (
        "criterion 3: a tool call offered to a full outbox is refused, so the connection closes "
        "with a reason — never accepted and dropped"
    )
    assert full.dropped == 0, "…and it is not counted as a droppable drop"

    mixed = Outbox(maxsize=2)
    assert mixed.offer(AstraTurnState(state=TurnState.SPEAKING, turn_id="t1")) is True
    assert mixed.offer(proposed) is True
    assert mixed.offer(AstraTurnState(state=TurnState.IDLE, turn_id="t1")) is True
    held = mixed.pending()
    assert [getattr(m, "type", "") for m in held].count(TOOL_CALL_TYPE) == 1, (
        "criterion 3: the turn state supersedes the turn state — the tool call is never the "
        f"message evicted; the queue holds {[m.type for m in held]}"
    )

    only = Outbox(maxsize=1)
    assert only.offer(proposed) is True
    assert only.offer(AstraTurnState(state=TurnState.IDLE, turn_id="t1")) is True
    assert only.pending() == [proposed], (
        "criterion 3: with no state to supersede, the arriving state is the one dropped"
    )

    replayed = Outbox(maxsize=1)
    assert replayed.offer(AstraSpoken(turn_id="t1", utterance_id="u1", text="one")) is True
    assert replayed.offer(proposed.model_copy(update={"replay": True})) is False, (
        "criterion 3: a *replayed* tool call is no more droppable than a live one"
    )


async def test_the_fake_and_the_real_endpoint_agree_on_the_replayed_tool_call(tmp_path):
    """CLAUDE.md: changing a wire format means changing the fake in the same PR.

    ``tests/fakes/hud_server.replay_script(with_tool_call=True)`` must put a tool call where the
    real journal puts it (after the spoken lines, before the played report) with the keys the real
    endpoint sends for the same kind of message — a dry-run search with an action and no problem.
    """
    model = tool_call_model()
    from tests.fakes.hud_server import replay_script

    fake_script = list(replay_script(with_tool_call=True))
    fake_types = [m.type for m in fake_script]
    assert TOOL_CALL_TYPE in fake_types, f"the fake replays a tool call; it sends {fake_types}"
    i = fake_types.index(TOOL_CALL_TYPE)
    assert (
        max(j for j, t in enumerate(fake_types) if t == "astra.spoken")
        < i
        < fake_types.index("astra.spoken.played")
    ), f"the fake's order matches criterion 3: {fake_types}"
    fake = keys_by_type(fake_script)[TOOL_CALL_TYPE]

    async with served(tmp_path, script=SCRIPT, ms_per_word=100) as box:  # noqa: SIM117 - the second page greets while the first is open
        async with HudClient.connect(box.server.url) as one:
            await one.hello()
            await one.drain(until=None, rig=box.rig, budget_ms=SETTLE_MS)
            turn = await _spoken_and_played_turn(box, one)
            box.server.publish(
                model(
                    turn_id=turn,
                    call_id=f"{turn}-c1",
                    name="google_search",
                    arguments={"query": "weather in Paris tomorrow"},
                    status="dry_run",
                    source="draft",
                    action="would search for: weather in Paris tomorrow",
                )
            )
            await one.drain(until=None, rig=None, budget_ms=TAIL_MS)
            async with HudClient.connect(box.server.url) as two:
                await two.hello()
                received = await two.drain(until=reply_is_complete, rig=None, budget_ms=BUDGET_MS)
    real = keys_by_type(journalled(received)).get(TOOL_CALL_TYPE)
    assert real is not None, (
        f"criterion 3: the endpoint replayed no tool call; it sent {[m.type for m in received]}"
    )
    assert fake == real, (
        f"the fake replays a tool call with keys {sorted(fake)}, the endpoint with {sorted(real)}"
    )


# --- T-0090 review rows: the 16-call bound and ruling U7 (late tool calls never reopen a turn) ------
#
# docs/tasks/T-0090.md "Rulings after review". The bound is asserted on ``TurnJournal`` directly,
# where 17 call ids are cheap; U7 is asserted both on the journal and over a real socket, in the
# reviewer's exact order.


def _journal():
    from local_astra.conductor import hud_server

    return hud_server.TurnJournal()


def _tc(turn: str, call_id: str, status: str = "proposed"):
    model = tool_call_model()
    return model(
        turn_id=turn,
        call_id=call_id,
        name="scroll",
        arguments={"direction": "down"},
        status=status,
        source="draft",
    )


def test_the_journal_keeps_sixteen_calls_and_evicts_the_oldest():
    """Review row: with 17 distinct call_ids, 16 are kept, the oldest is gone, truncated is set."""
    from local_astra.conductor import hud_server

    assert hud_server.REPLAY_MAX_TOOL_CALLS == 16, "ruling: the bound is 16 calls"
    journal = _journal()
    journal.record(AstraTranscript(turn_id="t1", text="q", final=True))
    for i in range(17):
        journal.record(_tc("t1", f"c{i}"))
    kept = [m.call_id for m in journal.messages() if getattr(m, "type", "") == TOOL_CALL_TYPE]
    assert kept == [f"c{i}" for i in range(1, 17)], (
        f"review row: 16 kept, c0 (the oldest) evicted, order kept; got {kept}"
    )
    assert journal.truncated is True, "review row: dropping a card is declared (truncated)"


def test_a_repeated_call_id_at_the_cap_evicts_nothing():
    journal = _journal()
    journal.record(AstraTranscript(turn_id="t1", text="q", final=True))
    for i in range(16):
        journal.record(_tc("t1", f"c{i}"))
    journal.record(_tc("t1", "c0", status="dry_run"))
    calls = [m for m in journal.messages() if getattr(m, "type", "") == TOOL_CALL_TYPE]
    assert [m.call_id for m in calls] == [f"c{i}" for i in range(16)], (
        f"review row: an update for a held call_id at the cap evicts nothing; got {[m.call_id for m in calls]}"
    )
    assert calls[0].status == "dry_run", "…and it is the latest message for that call"
    assert journal.truncated is False, "nothing was dropped, so nothing is declared"


def test_a_late_tool_call_for_a_superseded_turn_does_not_wipe_the_open_turn():
    """U7 on the journal, the reviewer's sequence: t2's transcript and lines survive."""
    journal = _journal()
    journal.record(_tc("t1", "t1-c1"))
    journal.record(AstraTurnState(state=TurnState.INTERRUPTED, turn_id="t1"))
    journal.record(AstraTranscript(turn_id="t2", text="second question", final=True))
    journal.record(AstraSpoken(turn_id="t2", utterance_id="t2-u1", text="First line."))
    journal.record(_tc("t1", "t1-c1", status="cancelled"))
    journal.record(AstraSpoken(turn_id="t2", utterance_id="t2-u2", text="Second line."))
    out = journal.messages()
    assert [(m.type, getattr(m, "turn_id", None)) for m in out] == [
        ("astra.transcript", "t2"),
        ("astra.spoken", "t2"),
        ("astra.spoken", "t2"),
    ], (
        "U7: a late tool call of a superseded turn neither records nor clears — the open turn's "
        f"transcript and both lines are replayed; got {[(m.type, getattr(m, 'turn_id', None)) for m in out]}"
    )
    assert [m.utterance_id for m in out if isinstance(m, AstraSpoken)] == ["t2-u1", "t2-u2"]
    assert journal.truncated is False, "U7: and the journal does not claim it lost anything"


def test_a_tool_call_for_a_never_seen_turn_still_opens_it():
    """U7, the other direction: a guard that ignores every turn change would be as wrong."""
    journal = _journal()
    journal.record(AstraTranscript(turn_id="t2", text="second question", final=True))
    journal.record(AstraSpoken(turn_id="t2", utterance_id="t2-u1", text="First line."))
    journal.record(_tc("t3", "t3-c1"))
    out = journal.messages()
    assert [(m.type, getattr(m, "turn_id", None)) for m in out] == [(TOOL_CALL_TYPE, "t3")], (
        "U7: a tool call whose turn_id was never seen opens that turn, clearing t2; got "
        f"{[(m.type, getattr(m, 'turn_id', None)) for m in out]}"
    )


def test_the_last_eight_superseded_turns_are_remembered():
    """U7: "the journal remembers the last 8 superseded turn ids" — pinned only that far: with
    exactly eight superseded turns, a late tool call for any of them is ignored."""
    journal = _journal()
    for n in range(1, 10):  # t1..t8 superseded, t9 open
        journal.record(AstraTranscript(turn_id=f"t{n}", text=f"q{n}", final=True))
    journal.record(AstraSpoken(turn_id="t9", utterance_id="t9-u1", text="Ninth."))
    for n in range(1, 9):
        journal.record(_tc(f"t{n}", f"t{n}-c1", status="cancelled"))
    out = journal.messages()
    assert [(m.type, getattr(m, "turn_id", None)) for m in out] == [
        ("astra.transcript", "t9"),
        ("astra.spoken", "t9"),
    ], f"U7: none of the eight superseded turns reopens; got {[(m.type, m.turn_id) for m in out]}"


async def test_u7_over_the_wire_live_and_on_reconnect(tmp_path):
    """U7 through ``HudServer.publish`` and a real socket: the late t1 call is still published live
    (the page decides what to do with it), and a reconnecting page gets t2 whole, not truncated."""
    tool_call_model()
    from local_astra.conductor.hud_server import REPLAY_TRUNCATED

    sequence = [
        _tc("t1", "t1-c1"),
        AstraTurnState(state=TurnState.INTERRUPTED, turn_id="t1"),
        AstraTranscript(turn_id="t2", text="second question", final=True),
        AstraSpoken(turn_id="t2", utterance_id="t2-u1", text="First line."),
        _tc("t1", "t1-c1", status="cancelled"),
        AstraSpoken(turn_id="t2", utterance_id="t2-u2", text="Second line."),
    ]
    async with served(tmp_path, script=SCRIPT, ms_per_word=100) as box:  # noqa: SIM117 - the second page greets while the first is open
        async with HudClient.connect(box.server.url) as one:
            await one.hello()
            await one.drain(until=None, rig=None, budget_ms=SETTLE_MS)
            for message in sequence:
                box.server.publish(message)
            live = await one.drain(
                until=lambda got: any(getattr(m, "utterance_id", "") == "t2-u2" for m in got),
                rig=None,
                budget_ms=BUDGET_MS,
            )
            late = [
                m
                for m in live
                if getattr(m, "type", "") == TOOL_CALL_TYPE and m.status == "cancelled"
            ]
            assert [m.turn_id for m in late] == ["t1"], (
                "U7: a tool call for a superseded turn is still published live; the page got "
                f"{[(getattr(m, 'type', m), getattr(m, 'status', None)) for m in live]}"
            )
            async with HudClient.connect(box.server.url) as two:
                await two.hello()
                received = await two.drain(until=reply_is_complete, rig=None, budget_ms=BUDGET_MS)
    replayed = journalled(received)
    errors = [m for m in replayed if getattr(m, "code", None) == REPLAY_TRUNCATED]
    assert not errors, "U7: truncated stays false — nothing of t2 was lost"
    body = [
        (m.type, getattr(m, "turn_id", None)) for m in replayed if not isinstance(m, AstraTurnState)
    ]
    assert body == [("astra.transcript", "t2"), ("astra.spoken", "t2"), ("astra.spoken", "t2")], (
        f"U7: the reconnecting page is caught up on t2's transcript and both lines; got {body}"
    )


# --- T-0090 ruling U8: a departure clear does not supersede ---------------------------------------
#
# docs/tasks/T-0090.md U8: "superseded" means superseded by a newer turn. A departure clear (T-0037
# criterion 4: the last page gone, after RECONNECT_GRACE_MS) forgets the turn but remembers it as
# the turn that was left; the next turn to open supersedes both. Driven through HudServer itself
# (``open()``/``close()`` and the session's virtual clock, the T-0037 rig), because the departure is
# the server's ``_on_last_gone``, not something a journal-level row can stage honestly.


def _depart(rig: Any, server: Any) -> None:
    """One page greets, leaves, and the grace runs out: the departure clear (T-0037 criterion 4)."""
    from local_astra.conductor.hud_server import RECONNECT_GRACE_MS
    from tests.unit.test_hud_hello_replay import greet

    page = greet(server)
    page.close()
    rig.clock.advance(RECONNECT_GRACE_MS + 1)


def _replay_after_hello(server: Any) -> list:
    from tests.unit.test_hud_hello_replay import after_the_live_messages, greet, head_of

    page = greet(server)
    reply = page.sent()
    page.close()  # a page that stayed would make every later "departure" a no-op
    head_of(reply)
    return after_the_live_messages(reply)


async def test_u8_departure_then_the_same_turn_records_and_replays_its_tool_call(tmp_path):
    """U8 row 1, the reviewer's sequence: t5 spoken + proposed -> departure -> cancelled t5 ->
    turn state IDLE t5. A reconnecting page must see the ``cancelled``."""
    tool_call_model()
    from tests.unit.test_hud_hello_replay import make_server

    rig, server = make_server(tmp_path)
    server.publish(AstraSpoken(turn_id="t5", utterance_id="t5-u1", text="Opening it now."))
    server.publish(_tc("t5", "t5-c1"))
    premise = _replay_after_hello(server)
    assert [m.type for m in premise if m.type == TOOL_CALL_TYPE] == [TOOL_CALL_TYPE], (
        f"premise: the proposal is journalled before the departure; got {[m.type for m in premise]}"
    )
    _depart(rig, server)

    server.publish(_tc("t5", "t5-c1", status="cancelled"))
    server.publish(AstraTurnState(state=TurnState.IDLE, turn_id="t5"))
    replay = _replay_after_hello(server)
    calls = [m for m in replay if getattr(m, "type", "") == TOOL_CALL_TYPE]
    assert [(m.turn_id, m.call_id, m.status) for m in calls] == [("t5", "t5-c1", "cancelled")], (
        "U8: a departure clear does not supersede the turn it forgets — a tool call for that same "
        "turn afterwards is recorded and replayed, so the reconnecting page sees the cancellation; "
        f"the reply carried {[(m.type, getattr(m, 'turn_id', None)) for m in replay]}"
    )
    assert all(m.replay is True for m in calls)


async def test_u8_departure_then_the_next_turn_ignores_a_late_old_turn_call(tmp_path):
    """U8 row 2: departure during t1, then t2 opens, then a late t1 tool call: t2 stays whole.

    This is the row that tells U8 from "only a new-turn clear supersedes": after a departure the
    open turn is None, so a rule that records the superseded id only when a new turn replaces an
    open one never learns t1, and the late t1 call reopens t1 and wipes t2.
    """
    tool_call_model()
    from local_astra.conductor.hud_server import REPLAY_TRUNCATED
    from tests.unit.test_hud_hello_replay import make_server

    rig, server = make_server(tmp_path)
    server.publish(AstraSpoken(turn_id="t1", utterance_id="t1-u1", text="Looking."))
    server.publish(_tc("t1", "t1-c1"))
    _depart(rig, server)

    server.publish(AstraTranscript(turn_id="t2", text="second question", final=True))
    server.publish(AstraSpoken(turn_id="t2", utterance_id="t2-u1", text="First line."))
    server.publish(_tc("t1", "t1-c1", status="cancelled"))
    server.publish(AstraSpoken(turn_id="t2", utterance_id="t2-u2", text="Second line."))
    replay = _replay_after_hello(server)
    assert not [m for m in replay if getattr(m, "code", None) == REPLAY_TRUNCATED], (
        "U8: truncated stays false — nothing of t2 was lost"
    )
    body = [
        (m.type, getattr(m, "turn_id", None)) for m in replay if not isinstance(m, AstraTurnState)
    ]
    assert body == [("astra.transcript", "t2"), ("astra.spoken", "t2"), ("astra.spoken", "t2")], (
        "U8: the next turn supersedes the turn that was left, so the late t1 call is ignored and t2 "
        f"is replayed whole; got {body}"
    )


# --- T-0090 ruling U9: a message for a superseded turn supersedes nothing --------------------------


async def test_u9_a_stale_played_report_does_not_supersede_the_open_turn(tmp_path):
    """U9, the reviewer's sequence X: t1 opens; t2 opens (spoken t2); a late astra.spoken.played
    for t1 arrives; a t2 tool call (dry_run) is the first t2 message after it; turn state IDLE t2.

    The t2 card must be journalled and replayed, and t2 must not be in ``superseded``. Whether the
    stale played report reopens t1 is the deferred spoken.played follow-up, so it is not asserted.
    """
    tool_call_model()
    from local_astra.contracts.protocol import PlayedUtterance
    from tests.unit.test_hud_hello_replay import make_server

    _rig, server = make_server(tmp_path)
    server.publish(AstraTranscript(turn_id="t1", text="first question", final=True))
    server.publish(AstraSpoken(turn_id="t1", utterance_id="t1-u1", text="Starting to answer."))
    server.publish(AstraSpoken(turn_id="t2", utterance_id="t2-u1", text="Second answer."))
    server.publish(
        AstraSpokenPlayed(
            turn_id="t1",
            played_ms=400,
            utterances=[
                PlayedUtterance(
                    utterance_id="t1-u1", played_text="Starting", played_words=1, complete=False
                )
            ],
        )
    )
    card = _tc("t2", "t2-c1", status="dry_run")
    server.publish(card)
    server.publish(AstraTurnState(state=TurnState.IDLE, turn_id="t2"))

    replay = _replay_after_hello(server)
    calls = [m for m in replay if getattr(m, "type", "") == TOOL_CALL_TYPE]
    assert [(m.turn_id, m.call_id, m.status) for m in calls] == [("t2", "t2-c1", "dry_run")], (
        "U9: t2's card is journalled and replayed — before T-0090 this call was kept, and a stale "
        "report for t1 must not change that; the reply carried "
        f"{[(m.type, getattr(m, 'turn_id', None)) for m in replay]}"
    )

    # U10 restatement of "t2 must not be in `superseded`" (it read ``server._journal.superseded``):
    # a turn that is not superseded keeps journalling its tool calls, so a later update of the card
    # and a second card for t2 are both replayed, the update as the latest for its call_id. (Under
    # U10 the current turn is t2, the newest first-seen id; the stale t1 report changes no order.)
    server.publish(_tc("t2", "t2-c1", status="cancelled"))
    server.publish(_tc("t2", "t2-c2", status="proposed"))
    replay = _replay_after_hello(server)
    calls = [m for m in replay if getattr(m, "type", "") == TOOL_CALL_TYPE]
    assert [(m.turn_id, m.call_id, m.status) for m in calls] == [
        ("t2", "t2-c1", "cancelled"),
        ("t2", "t2-c2", "proposed"),
    ], (
        "U9/U10: the stale t1 report must not leave t2 superseded — t2's later calls are still "
        "recorded and replayed, latest per call_id in first-seen order; the reply carried "
        f"{[(m.type, getattr(m, 'turn_id', None), getattr(m, 'status', None)) for m in replay]}"
    )


# --- T-0090 ruling U10: one rule, by first-seen order -----------------------------------------------
#
# docs/tasks/T-0090.md U10. The reviewer's sequences A and B at 90dcef9: a stale t1 message, then t3
# opens, then a late t2 call wiped t3, because the stale message had kept t2 out of the superseded
# set. Under U10 t2 is superseded as soon as t3 is first seen, whatever came between. The property
# over random interleavings is tests/unit/test_hud_journal_property.py.


def _played(turn: str, ms: int = 400) -> AstraSpokenPlayed:
    return AstraSpokenPlayed(turn_id=turn, played_ms=ms, utterances=[])


def _assert_t3_whole(replay: list, row: str) -> None:
    from local_astra.conductor.hud_server import REPLAY_TRUNCATED

    assert not [m for m in replay if getattr(m, "code", None) == REPLAY_TRUNCATED], (
        f"U10 {row}: truncated stays false — nothing of t3 was lost"
    )
    body = [
        (m.type, getattr(m, "turn_id", None), getattr(m, "utterance_id", None))
        for m in replay
        if not isinstance(m, AstraTurnState)
    ]
    assert body == [
        ("astra.transcript", "t3", None),
        ("astra.spoken", "t3", "t3-v1"),
        ("astra.spoken", "t3", "t3-v2"),
    ], (
        f"U10 {row}: t2 is superseded once t3 is first seen, so the late cancelled t2 call is "
        f"neither recorded nor allowed to clear t3 — t3's transcript and both lines replay; got {body}"
    )


async def test_u10_sequence_a_a_stale_message_then_a_new_turn_then_a_late_call(tmp_path):
    """U10 sequence A: t1 opens, t2 opens, a stale ``spoken.played`` for t1, t3 opens, spoken t3, a
    late cancelled t2 tool call, spoken t3. At 90dcef9 the late t2 call wiped t3."""
    tool_call_model()
    from tests.unit.test_hud_hello_replay import make_server

    _rig, server = make_server(tmp_path)
    server.publish(AstraTranscript(turn_id="t1", text="first question", final=True))
    server.publish(AstraTranscript(turn_id="t2", text="second question", final=True))
    server.publish(AstraSpoken(turn_id="t2", utterance_id="t2-u1", text="Second answer."))
    server.publish(_played("t1"))
    server.publish(AstraTranscript(turn_id="t3", text="third question", final=True))
    server.publish(AstraSpoken(turn_id="t3", utterance_id="t3-v1", text="First line."))
    server.publish(_tc("t2", "t2-c1", status="cancelled"))
    server.publish(AstraSpoken(turn_id="t3", utterance_id="t3-v2", text="Second line."))
    _assert_t3_whole(_replay_after_hello(server), "sequence A")


async def test_u10_sequence_b_a_departure_during_t2_then_sequence_a(tmp_path):
    """U10 sequence B: as A, with a departure clear (T-0037 criterion 4, the real server path)
    during t2, before the stale t1 report. Departures never change the first-seen order."""
    tool_call_model()
    from tests.unit.test_hud_hello_replay import make_server

    rig, server = make_server(tmp_path)
    server.publish(AstraTranscript(turn_id="t1", text="first question", final=True))
    server.publish(AstraTranscript(turn_id="t2", text="second question", final=True))
    server.publish(AstraSpoken(turn_id="t2", utterance_id="t2-u1", text="Second answer."))
    _depart(rig, server)
    server.publish(_played("t1"))
    server.publish(AstraTranscript(turn_id="t3", text="third question", final=True))
    server.publish(AstraSpoken(turn_id="t3", utterance_id="t3-v1", text="First line."))
    server.publish(_tc("t2", "t2-c1", status="cancelled"))
    server.publish(AstraSpoken(turn_id="t3", utterance_id="t3-v2", text="Second line."))
    _assert_t3_whole(_replay_after_hello(server), "sequence B")


async def test_u10_a_late_call_after_a_stale_message_neither_records_nor_shows(tmp_path):
    """U10 negative half: after A's stale t1 report and t3's opening, the late t2 call is still
    published live (the page decides), and a t1 call is as superseded as a t2 one."""
    tool_call_model()
    from tests.unit.test_hud_hello_replay import greet, make_server

    _rig, server = make_server(tmp_path)
    page = greet(server)
    server.publish(AstraTranscript(turn_id="t1", text="first question", final=True))
    server.publish(AstraTranscript(turn_id="t2", text="second question", final=True))
    server.publish(_played("t1"))
    server.publish(AstraTranscript(turn_id="t3", text="third question", final=True))
    server.publish(AstraSpoken(turn_id="t3", utterance_id="t3-v1", text="First line."))
    server.publish(_tc("t2", "t2-c1", status="cancelled"))
    server.publish(_tc("t1", "t1-c1", status="cancelled"))
    server.publish(AstraSpoken(turn_id="t3", utterance_id="t3-v2", text="Second line."))
    live = [m for m in page.sent() if getattr(m, "type", "") == TOOL_CALL_TYPE]
    page.close()
    assert [(m.turn_id, m.status) for m in live] == [("t2", "cancelled"), ("t1", "cancelled")], (
        f"U10: a superseded turn's call is still published live; the page got {live}"
    )
    _assert_t3_whole(_replay_after_hello(server), "late t2 and t1 calls")


async def test_u10_sequence_e_a_stale_message_then_a_departure_then_a_new_turn(tmp_path):
    """U10, found by the property against the reviewer's prototype: B with the departure and the
    stale report swapped. t1 opens, t2 opens, a stale ``spoken.played`` for t1, a departure clear,
    t3 opens, spoken t3, a late cancelled t2 tool call, spoken t3.

    At 90dcef9 and under the prototype the departure forgets the stale-reopened t1 as "the turn
    that was left", so t3's opening never supersedes t2 and the late t2 call wipes t3. Under U10
    neither the stale message nor the departure changes the first-seen order: t3 is current, t2
    is superseded."""
    tool_call_model()
    from tests.unit.test_hud_hello_replay import make_server

    rig, server = make_server(tmp_path)
    server.publish(AstraTranscript(turn_id="t1", text="first question", final=True))
    server.publish(AstraTranscript(turn_id="t2", text="second question", final=True))
    server.publish(AstraSpoken(turn_id="t2", utterance_id="t2-u1", text="Second answer."))
    server.publish(_played("t1"))
    _depart(rig, server)
    server.publish(AstraTranscript(turn_id="t3", text="third question", final=True))
    server.publish(AstraSpoken(turn_id="t3", utterance_id="t3-v1", text="First line."))
    server.publish(_tc("t2", "t2-c1", status="cancelled"))
    server.publish(AstraSpoken(turn_id="t3", utterance_id="t3-v2", text="Second line."))
    _assert_t3_whole(_replay_after_hello(server), "sequence E (stale, then departure)")
