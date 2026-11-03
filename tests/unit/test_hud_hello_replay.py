"""T-0037 group A: the hello reply catches a (re)connecting page up on the open turn.

The defect (2026-09-14 rehearsal, D3): ``HudServer.hello_reply()`` answers a page with
``astra.assist.state`` + ``astra.turn.state`` and nothing else, so a page that reloads mid-turn
shows a live SPEAKING chip over an empty pane while TTS keeps playing. This module is the
acceptance suite of docs/tasks/T-0037.md criteria 1-8.

**The oracle is what the server was asked to publish.** Every row here puts messages through
``HudServer.publish`` — the session's own egress seam — and then asserts about the reply a fresh
connection gets. Nothing is compared against the journal's internals, and nothing is derived from
``Session``: criterion 3 is precisely "no replayed message is synthesised from session state,
from history, from turns.jsonl or from the machine", so a test that read any of those would be
asserting the bug.

THE SEAMS THIS TASK ADDS (names are the contract with the implementer; ``hud_server.py`` and
``contracts/protocol.py`` are group A's owned files):

    contracts/protocol.py
        AstraTranscript / AstraSpoken / AstraUiSurface / AstraSpokenPlayed / AstraTurnState
        each gain ``replay: bool = False``. No other field moves; ASTRA_MESSAGE_TYPES,
        SERVER_TO_CLIENT and CLIENT_TO_SERVER are unchanged (criterion 2).

    conductor/hud_server.py
        REPLAY_MAX_SPOKEN: int    how many astra.spoken lines of the open turn are kept
        REPLAY_MAX_PLAYED: int    how many astra.spoken.played messages are kept
        REPLAY_MAX_MESSAGES: int  the total cap, strictly < DEFAULT_OUTBOX (criterion 5)
        the error code "replay_truncated" (criterion 6)

        ``publish()`` journals; ``hello_reply()`` emits, after the live messages,
        [astra.error(replay_truncated)] · the turn's last transcript · its spoken lines in
        publication order · its most recent surface · [the INTERRUPTED state] · its
        astra.spoken.played messages · [the current turn state again] — every journalled message
        marked ``replay=True`` (criterion 1).

**AMENDED BY T-0043 criterion 18** (the only change to this module since T-0037): the live head
is three messages, not two — ``astra.log.state`` is yielded *third*, "after the assist state and
the turn state, before the journal replay", so a (re)connecting page starts from the server's
truth about whether the session is being recorded. Nothing else about this reply moves: the
journal's contents, its order, its bound and its ``replay`` marking are T-0037's, unchanged and
unweakened. The head is declared once as ``LIVE_HEAD`` and asserted by ``head_of`` in every row
that has to talk about "the part after the live messages", so a reply that regressed to two
messages reddens those rows instead of quietly shifting their slices by one.

Regime: no measurement, no socket, ``ScriptedClock``; the bounded row publishes n = 10 000 + 10 000
messages in one turn and asserts a count, never a time. The Hypothesis row is derandomised, which
is what ``--hypothesis-seed=0`` pins from the command line.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from hypothesis import HealthCheck, given, settings

from local_astra.contracts.codec import decode, encode
from local_astra.contracts.protocol import (
    SERVER_TO_CLIENT,
    AstraAssistState,
    AstraCanvasCommit,
    AstraCanvasStep,
    AstraError,
    AstraHello,
    AstraLogState,
    AstraSpoken,
    AstraSpokenPlayed,
    AstraTranscript,
    AstraTurnState,
    AstraUiSurface,
    PlayedUtterance,
)
from local_astra.contracts.turn import TurnState
from tests.unit import _t0181_pins as t0181
from tests.unit import test_t0213_settings_wire as t0213  # T-0213 R-5 named re-pin
from tests.unit._hud_rig import (
    UI_CANVAS,
    Rig,
    frame,
    make_rig,
    require_symbol,
)
from tests.unit._hud_rig import answer as answer_commit
from tests.unit._hud_rig import thought as thought_commit
from tests.unit.test_hud_egress import QUESTION, SENTINEL, commit_scripts

TURN = "t1"
OTHER = "t2"


class _NullSink:
    """The server's ``frame_sink``; no row here posts a frame."""

    def ingest(self, message: Any) -> None:
        return None


def make_server(tmp_path, **kwargs: Any) -> tuple[Rig, Any]:
    """A rig and a ``HudServer`` over it, with no socket: ``start()`` is never called."""
    rig = make_rig(tmp_path, **kwargs)
    server = require_symbol("HudServer")(rig.session, frame_sink=_NullSink(), port=0)
    return rig, server


def greet(server: Any) -> Any:
    """One fresh connection that says hello, exactly as a reloading page does."""
    connection = server.open()
    connection.receive(encode(AstraHello()))
    return connection


def replayed(message: Any) -> Any:
    """A copy of ``message`` marked ``replay=True``.

    The field's existence is *asserted* rather than assumed: criterion 2 adds it, and a missing
    field must surface as this task's failure (an assertion naming the criterion) rather than as
    a pydantic ``extra_forbidden`` error from inside a constructor.
    """
    assert "replay" in type(message).model_fields, (
        f"criterion 2: {type(message).__name__} gains `replay: bool = False` "
        "(local_astra/contracts/protocol.py, docs/tasks/T-0037.md)"
    )
    return message.model_copy(update={"replay": True})


def marker_of(message: Any) -> Any:
    """``message.replay``, as an assertion rather than an ``AttributeError``."""
    assert "replay" in type(message).model_fields, (
        f"criterion 2: {type(message).__name__} gains `replay: bool = False` "
        "(local_astra/contracts/protocol.py, docs/tasks/T-0037.md)"
    )
    return message.replay


def types_of(messages: list[Any]) -> list[str]:
    return [m.type for m in messages]


#: The **live** head of a hello reply, in order. T-0037 criterion 1 fixed it at two — the assist
#: state and the turn state — and T-0043 criterion 18 makes it three by putting ``astra.log.state``
#: third, "after the assist state and the turn state, before the journal replay". The journal and
#: its bound are unchanged; what moved is where the journal starts. Declared once and used by every
#: row below that has to say "after the live messages", so a reply that regressed to two reddens
#: each of them at ``head_of`` rather than silently shifting a slice.
#: T-0181 A4 pin move (A-R8): ``astra.mic.state`` joins the head directly after the log state once the tree has it.
LIVE_HEAD: tuple[type, ...] = tuple(
    t0181.hello_head((AstraAssistState, AstraTurnState, AstraLogState))
)


def head_of(reply: list[Any]) -> list[Any]:
    """The live head, **asserted** rather than assumed, and returned.

    A slice with a hard-coded index is a pin that drifts: if the reply lost its log state the
    journal would simply start one message earlier and every ``replay=True`` row below would still
    pass over a shorter list. So the shape is checked here, once, by every caller.
    """
    got = [type(m) for m in reply[: len(LIVE_HEAD)]]
    assert got == list(LIVE_HEAD), (
        "T-0043 criterion 18 (over T-0037 criterion 1): a hello reply opens with the assist "
        "state, the turn state and then the log state — three live messages, before the journal "
        f"replay. This reply opened with {types_of(reply)[: len(LIVE_HEAD)]}"
    )
    return reply[: len(LIVE_HEAD)]


def after_the_live_messages(reply: list[Any]) -> list[Any]:
    """Everything the journal contributed: the reply past its (asserted) live head.

    T-0213 R-5 named re-pin: and before the one ``astra.settings.state``, asserted to be the reply's last message.
    """
    head_of(reply)
    return t0213.hello_without_settings_state(reply)[len(LIVE_HEAD) :]


# --- the messages a session publishes during a turn ---------------------------------------------


def transcript(text: str, *, final: bool = False, turn: str = TURN) -> AstraTranscript:
    return AstraTranscript(turn_id=turn, text=text, final=final)


def spoken(utterance: str, text: str, *, turn: str = TURN) -> AstraSpoken:
    return AstraSpoken(turn_id=turn, utterance_id=utterance, text=text)


def surface(idx: int = 0, *, turn: str = TURN, final: bool = False) -> AstraUiSurface:
    import json

    return AstraUiSurface(
        surface_id=f"turn-{turn}",
        canvas_idx=idx,
        final=final,
        messages=tuple(json.loads(line) for line in UI_CANVAS.splitlines()),
    )


def played(*utterances: str, turn: str = TURN, ms: int = 1400) -> AstraSpokenPlayed:
    return AstraSpokenPlayed(
        turn_id=turn,
        played_ms=ms,
        utterances=[
            PlayedUtterance(utterance_id=u, played_text="one two", played_words=2, complete=False)
            for u in utterances
        ],
    )


def publish_a_turn(server: Any, *, turn: str = TURN) -> list[Any]:
    """The rehearsal's shape: a transcript, two spoken lines, a surface and a played report."""
    script = [
        transcript("what is on this screen", final=True, turn=turn),
        spoken(f"{turn}-u1", "Here is a quick check.", turn=turn),
        spoken(f"{turn}-u2", "Try the five questions.", turn=turn),
        surface(1, turn=turn, final=True),
        played(f"{turn}-u1", f"{turn}-u2", turn=turn),
    ]
    for message in script:
        server.publish(message)
    return script


# --- criterion 1: the reply catches the page up, in order --------------------------------------


async def test_a_hello_mid_turn_replays_the_open_turns_lines_in_order(tmp_path):
    """Criterion 1: assist state, turn state, then the turn's journal, played messages last.

    The order is not cosmetic. T-0030's ``withUtterance`` drops an utterance's played state when a
    new ``astra.spoken`` for the same ``utterance_id`` arrives, so a reply that put the played
    report before the spoken lines would lose the strike on every reload — the exact case the
    rehearsal complained about (docs/tasks/T-0037.md, implementation notes, "ordering matters").
    """
    _, server = make_server(tmp_path)
    script = publish_a_turn(server)
    connection = greet(server)
    reply = connection.sent()

    head = list(LIVE_HEAD)
    body = [AstraTranscript, AstraSpoken, AstraSpoken, AstraUiSurface, AstraSpokenPlayed]
    got = [type(m) for m in reply]
    assert got[: len(head)] == head, (
        "criterion 1 / T-0028 criterion 4 / T-0043 criterion 18: the live messages the page gets "
        "come first and byte-identically — the assist state, the turn state and (since T-0043) "
        f"the log state; the reply began {types_of(reply)[: len(head)]}"
    )
    assert got[len(head) : len(head) + len(body)] == body, (
        "criterion 1: the hello reply then replays the open turn — its last transcript, its "
        "spoken lines in publication order, its most recent surface and its astra.spoken.played "
        f"last. The page was sent {types_of(reply)} (docs/tasks/T-0037.md criterion 1); today "
        "hello_reply() yields two messages and the reloading page is left with a live chip over "
        "an empty pane"
    )
    # T-0213 R-5 named re-pin: the one settings state is the reply's last message, after everything below
    assert t0213.hello_without_settings_state(got)[len(head) + len(body) :] in (
        [],
        [AstraTurnState],
    ), (
        "…and nothing after the played report but, at most, the repeat of the current turn "
        f"state; the reply ended {types_of(reply)[len(head) + len(body) :]}"
    )

    caught_up = after_the_live_messages(reply)
    assert [m.utterance_id for m in caught_up if isinstance(m, AstraSpoken)] == [
        f"{TURN}-u1",
        f"{TURN}-u2",
    ], f"criterion 1: publication order, not set order; got {types_of(caught_up)}"
    last_journalled = max(
        i for i, m in enumerate(caught_up) if not isinstance(m, AstraTurnState | AstraAssistState)
    )
    assert isinstance(caught_up[last_journalled], AstraSpokenPlayed), (
        "criterion 1: every astra.spoken.played comes after every astra.spoken, so the strike "
        f"survives the reload; the last replayed line was {caught_up[last_journalled]!r}"
    )

    for message in reply:
        raw = encode(message)
        assert message.type in SERVER_TO_CLIENT, (
            f"T-0028 criterion 9: {message.type!r} is not a server→client type"
        )
        assert decode(raw) == message, f"every replayed message re-decodes; {raw!r} did not"

    # criterion 3: every replayed message is *equal, field for field except ``replay``*, to one
    # this server actually put through publish(). Nothing is summarised, re-ordered or re-worded.
    wanted = [replayed(m) for m in script]
    assert [m for m in reply if m in wanted] == wanted, (
        "criterion 3: the replay is the messages this server really published, marked and "
        f"otherwise untouched. Published {[m.type for m in script]}; the reply carried "
        f"{types_of(reply)} and the bodies did not match"
    )


async def test_a_hello_with_no_open_turn_is_only_the_live_messages(tmp_path):
    """Criterion 3's negative: nothing published, nothing replayed — no synthesis from state.

    A journal fed from ``Session`` (the machine's transcript, the history, ``turns.jsonl``) would
    have something to say here; one fed from ``publish()`` has nothing. This is the mutation the
    reviewer aims at.

    T-0043 criterion 18 moves the live head from two messages to three (the log state joins it);
    the property this row is about — an empty journal adds *nothing* — is unchanged, so the
    equality is against ``LIVE_HEAD`` and not against a count written out by hand.
    """
    _, server = make_server(tmp_path)
    connection = greet(server)
    reply = connection.sent()

    # T-0213 R-5 named re-pin: then the one settings state, last
    assert [type(m) for m in reply] == t0213.hello_with_settings_last(LIVE_HEAD), (
        "criterion 3: a session that published nothing owes the page exactly the live messages "
        "(assist state, turn state, log state — T-0043 criterion 18) and nothing else; it sent "
        f"{types_of(reply)}"
    )
    state = reply[1]
    assert marker_of(state) is False, (
        "criterion 2: the live current state is not a replay — ``replay`` is False on every "
        f"message that was not taken from the journal; got {state!r}"
    )


async def test_replayed_messages_are_marked_and_live_ones_are_not(tmp_path):
    """Criterion 2, both directions: the journal stores what was published, the reply marks copies."""
    _, server = make_server(tmp_path)
    live = server.open()
    live.receive(encode(AstraHello()))
    published = publish_a_turn(server)

    for message in published:
        assert marker_of(message) is False, (
            f"criterion 2: a message the session hands to publish() carries replay=False; "
            f"{message.type} arrived as {message!r}"
        )
    head_of(live.sent())
    for message in live.sent()[len(LIVE_HEAD) :]:
        assert marker_of(message) is False, (
            "criterion 2, negative: a message published live to an already-greeted connection is "
            f"not a replay; that connection was sent {message!r}"
        )

    reply = greet(server).sent()
    journalled = [m for m in after_the_live_messages(reply) if not isinstance(m, AstraTurnState)]
    assert journalled, f"criterion 1: nothing was replayed at all; the reply was {types_of(reply)}"
    assert all(marker_of(m) is True for m in journalled), (
        "criterion 2: every replayed message is marked, so the page can tell a catch-up from a "
        f"live line; got {[(m.type, m.replay) for m in journalled]}"
    )
    assert all("replay" in encode(m) for m in journalled), (
        "criterion 2: and the marker is on the wire, not only in the Python object"
    )


# --- criterion 4: one turn, never the thought ---------------------------------------------------


async def test_a_new_turn_clears_the_journal(tmp_path):
    """Criterion 4 (**mutation target**): a new ``turn_id`` on publish() drops the old turn.

    Removing the clear must make this red, so the assertion names both halves: turn 2's lines are
    replayed and no message of turn 1 survives — checked by ``turn_id`` *and* by a string that
    exists only in turn 1's text, which catches a journal that keeps the text but relabels it.
    """
    _, server = make_server(tmp_path)
    server.publish(transcript("the first question", final=True, turn=TURN))
    server.publish(spoken(f"{TURN}-u1", "ZQXFIRSTTURN answer one.", turn=TURN))
    publish_a_turn(server, turn=OTHER)

    reply = greet(server).sent()
    journalled = [m for m in after_the_live_messages(reply) if not isinstance(m, AstraTurnState)]
    assert [m.type for m in journalled].count("astra.spoken") == 2, (
        f"criterion 1: the newest turn's two spoken lines are replayed; got {types_of(reply)}"
    )
    turn_ids = {getattr(m, "turn_id", OTHER) for m in journalled}
    assert turn_ids == {OTHER}, (
        "criterion 4: the journal holds messages of one turn_id only — the newest turn seen on "
        f"publish(); the reply carried turns {sorted(turn_ids)}"
    )
    assert not [raw for raw in map(encode, reply) if "ZQXFIRSTTURN" in raw], (
        "criterion 4, negative: after a second turn opens, no message of the first turn is ever "
        "replayed — not its text either (a journal that kept the line and relabelled it would "
        "pass the turn_id check above and still show the user the wrong turn)"
    )


async def test_a_client_disconnected_clears_the_journal(tmp_path):
    """Criterion 4, privacy: nothing survives the session being told its last page is gone.

    The premise is asserted first (the journal *was* replayable), or the row would pass for a
    server that never journals anything.
    """
    grace = require_symbol("RECONNECT_GRACE_MS")
    rig, server = make_server(tmp_path)
    publish_a_turn(server)

    first = greet(server)
    assert [m for m in first.sent() if isinstance(m, AstraSpoken)], (
        "premise (criterion 1): the open turn must be replayable before the disconnect, or this "
        f"row proves nothing; the reply was {types_of(first.sent())}"
    )
    first.close()
    rig.clock.advance(grace + 1)

    reply = greet(server).sent()
    # T-0213 R-5 named re-pin: then the one settings state, last
    assert [type(m) for m in reply] == t0213.hello_with_settings_last(LIVE_HEAD), (
        "criterion 4: events.ClientDisconnected (the last page really gone, after the grace) "
        "clears the journal, so the next connection gets the live messages and nothing else "
        f"(T-0043 criterion 18: three of them); it was sent {types_of(reply)}"
    )


async def test_aclose_clears_the_journal(tmp_path):
    """Criterion 4, privacy: the session being taken down leaves nothing behind either."""
    _, server = make_server(tmp_path)
    publish_a_turn(server)
    assert [m for m in greet(server).sent() if isinstance(m, AstraSpoken)], (
        "premise: the turn is replayable before aclose()"
    )

    await server.aclose()
    reply = greet(server).sent()
    # T-0213 R-5 named re-pin: then the one settings state, last
    assert [type(m) for m in reply] == t0213.hello_with_settings_last(LIVE_HEAD), (
        "criterion 4: aclose() clears the journal, leaving only the live messages (T-0043 "
        f"criterion 18: three of them); a connection made afterwards was sent {types_of(reply)}"
    )


async def _hello_reply_after_a_turn(tmp_path, script, kept: dict | None = None) -> list[Any]:
    """Run one whole turn through a real server, then greet it and return what that page gets.

    ``kept`` (T-0180 B9): if given, receives the history and the ``turns.jsonl`` text after the turn.
    """
    rig, server = make_server(tmp_path, script=script, frames=[frame(0, "a")], ms_per_word=100)
    await rig.session.start()
    try:
        await rig.ask(QUESTION)
        await rig.speak_out()
        if kept is not None:
            kept["history"] = json.dumps([h.model_dump(mode="json") for h in rig.state.history])
            kept["turns"] = "\n".join(rig.lines())
        return greet(server).sent()
    finally:
        await rig.session.aclose()


THOUGHT_TYPE = "astra.thought"


def _sentinel_outside_thought_text(raw: str, sentinel: str = SENTINEL) -> bool:
    """T-0180 B9: ``sentinel`` is in the encoded message ``raw`` somewhere other than an
    ``astra.thought``'s ``text`` (another type, or another field of an ``astra.thought``)."""
    if sentinel not in raw:
        return False
    message = json.loads(raw)
    if message.get("type") != THOUGHT_TYPE:
        return True
    rest = {k: v for k, v in message.items() if k != "text"}
    return sentinel in json.dumps(rest)


@settings(
    max_examples=12,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(script=commit_scripts())
def test_no_thought_token_reaches_a_replayed_message(tmp_path, script):
    """Criterion 4, negative, **narrowed by T-0180** (B9, ruling R-1): the wire-side property of
    T-0028 criterion 11, over the hello reply.

    ``commit_scripts`` (``tests/unit/test_hud_egress.py``) mints thought text whose token appears
    nowhere else — not in an answer, not in the canvas, not in the question, not in any id the
    session mints — so finding it in the reply is proof it came from the thought channel. Since
    T-0180 the journal keeps the turn's latest ``astra.thought`` per attempt (A4), so the token may
    appear in the reply in exactly one place: the ``text`` of a replayed ``astra.thought``. It is
    absent from every other message type and from every other field of ``astra.thought``.

    Anti-vacuity (B9): the token **does** appear there whenever the script had a thought commit —
    every committed thought of the turn's one attempt is in the replayed cumulative ``text`` — and
    every replayed ``astra.thought`` carries ``replay=True``. It never reaches history or
    ``turns.jsonl``. The DOM half ("only inside ``[data-model-thoughts]``") is group C's and is not
    asserted here (this row has no page).
    """
    kept: dict = {}
    reply = asyncio.run(_hello_reply_after_a_turn(tmp_path, script, kept))
    offenders = [raw for raw in map(encode, reply) if _sentinel_outside_thought_text(raw)]
    assert not offenders, (
        "criterion 4 (narrowed by T-0180): thinking reached a reconnecting page outside "
        "astra.thought's text. The token is only ever in a thought segment of the script; it "
        f"turned up in {offenders[:2]}"
    )
    replayed_thoughts = [m for m in reply if m.type == THOUGHT_TYPE]
    assert all(marker_of(m) is True for m in replayed_thoughts), (
        f"T-0180 A4/B9: a replayed astra.thought is marked replay=True; {replayed_thoughts}"
    )
    shown = " ".join(m.text for m in replayed_thoughts)
    tokens = [f"{SENTINEL}{c.canvas_idx} " for c in script if c.segment == "thought"]
    missing = [t for t in tokens if t not in shown]
    assert not missing, (
        f"T-0180 B9 anti-vacuity: the committed thought tokens {missing} are in no replayed "
        f"astra.thought text (the reply was {types_of(reply)})"
    )
    for name in ("history", "turns"):
        assert SENTINEL not in kept[name], f"T-0180 B9: a thought token reached {name}"
    assert not [m for m in reply if m.type.startswith("astra.canvas.")], (
        f"criterion 4: no astra.canvas.* is journalled or replayed; got {types_of(reply)}"
    )
    if any(c.segment == "answer" for c in script):
        assert [m for m in reply if isinstance(m, AstraSpoken)], (
            "the property must not pass by replaying nothing: this script has an answer commit, "
            "so the turn published astra.spoken and a page greeting afterwards must be caught up "
            f"on it (criterion 1). The reply was {types_of(reply)}"
        )


#: T-0180 B9 replay rows: two thought commits of one attempt, then an answer. The thought texts
#: are hand-written and occur nowhere else, so the expected replayed ``text`` is their
#: concatenation, written out below by hand.
REPLAY_THOUGHT_1 = "zqxreplayone the chart has four bars. "
REPLAY_THOUGHT_2 = "zqxreplaytwo the tallest is March. "
REPLAY_ANSWER = "March is the tallest bar. "


async def test_a_hello_after_a_thinking_turn_replays_its_thought_once_after_the_lines(tmp_path):
    """T-0180 B9 (the replay-of-thoughts row of ``test_hud_hello_replay.py``), with A4 driven
    by the real session rather than by hand-fed ``publish()`` calls.

    One turn: two thought commits of the one step-1 attempt, then an answer. A page greeting the
    server afterwards gets, after the live head, exactly one ``astra.thought``: ``replay=True``,
    the turn's ``turn_id`` (the one its replayed ``astra.spoken`` carries), ``step`` 1,
    ``kind`` ``"step"``, ``end`` ``"done"``, ``omitted_chars`` 0, and ``text`` the two committed
    thoughts concatenated in order (B1; hand-written below). It sits after every ``astra.spoken``
    and before every ``astra.spoken.played`` (A4). The thought text is in no other replayed
    message, nor in history or ``turns.jsonl``. The DOM half is group C's.
    """
    script = [
        thought_commit(REPLAY_THOUGHT_1, canvas_idx=0),
        thought_commit(REPLAY_THOUGHT_2, canvas_idx=1),
        answer_commit(REPLAY_ANSWER, canvas_idx=2),
    ]
    kept: dict = {}
    reply = await _hello_reply_after_a_turn(tmp_path, script, kept)
    head_of(reply)
    body = after_the_live_messages(reply)
    thoughts = [m for m in body if m.type == THOUGHT_TYPE]
    assert len(thoughts) == 1, (
        "T-0180 A4/B1: one attempt with thought commits is one replayed astra.thought (latest "
        f"wins per attempt); the reply after the live head was {types_of(body)}"
    )
    got = thoughts[0]
    spoken_lines = [m for m in body if isinstance(m, AstraSpoken)]
    assert spoken_lines, f"the turn spoke, so its line is replayed too; got {types_of(body)}"
    assert (got.replay, got.turn_id, got.step, got.kind, got.end, got.omitted_chars) == (
        True,
        spoken_lines[0].turn_id,
        1,
        "step",
        "done",
        0,
    ), f"T-0180 A4/B3/B6: the replayed thought of a finished step-1 attempt; got {got}"
    assert (
        got.text == "zqxreplayone the chart has four bars. zqxreplaytwo the tallest is March. "
    ), (
        f"T-0180 B1: text is the attempt's committed thoughts concatenated in order; got {got.text!r}"
    )
    at = body.index(got)
    last_spoken = max(body.index(m) for m in spoken_lines)
    played_at = [i for i, m in enumerate(body) if isinstance(m, AstraSpokenPlayed)]
    assert last_spoken < at, f"T-0180 A4: thoughts come after the spoken lines; {types_of(body)}"
    assert all(at < i for i in played_at), (
        f"T-0180 A4: thoughts come before the played reports (played stays last); {types_of(body)}"
    )
    for marker in ("zqxreplayone", "zqxreplaytwo"):
        outside = [raw for raw in map(encode, reply) if _sentinel_outside_thought_text(raw, marker)]
        assert not outside, f"T-0180 B9: {marker} outside astra.thought's text: {outside[:2]}"
        for name in ("history", "turns"):
            assert marker not in kept[name], f"T-0180 B9: {marker} reached {name}"


async def test_a_hello_after_a_second_turn_replays_none_of_the_first_turns_thoughts(tmp_path):
    """T-0180 B9 / A4, adversarial: the journal holds the *current* turn's thoughts only.

    Turn 1 thinks (``zqxreplayone``) and answers; turn 2 answers without thinking. A page that
    greets after turn 2 must get no ``astra.thought`` at all and the turn-1 thought text
    nowhere, while turn 2's answer is replayed (anti-vacuity: the reply is not simply empty).
    The live publications of turn 1 did carry the thought (otherwise this row proves nothing).
    """
    published: list[Any] = []
    rig, server = make_server(
        tmp_path,
        script=[thought_commit(REPLAY_THOUGHT_1, canvas_idx=0), answer_commit(REPLAY_ANSWER, 1)],
        frames=[frame(0, "a")],
        ms_per_word=100,
    )
    live_publish = server.publish

    def tapped(message: Any) -> None:
        published.append(message)
        live_publish(message)

    rig.session.set_publisher(tapped)
    await rig.session.start()
    try:
        await rig.ask(QUESTION)
        await rig.speak_out()
        rig.llm.script = [answer_commit("Yes. ", canvas_idx=0)]
        await rig.ask("and the shortest one")
        await rig.speak_out()
        reply = greet(server).sent()
    finally:
        await rig.session.aclose()
    first = [m for m in published if m.type == THOUGHT_TYPE]
    assert first and "zqxreplayone" in first[-1].text, (
        "anti-vacuity: turn 1 published its thought live (T-0180 B1); published "
        f"{types_of(published)}"
    )
    body = after_the_live_messages(reply)
    assert [m for m in body if m.type == THOUGHT_TYPE] == [], (
        f"T-0180 A4: turn 2 had no thought, so none is replayed; got {types_of(body)}"
    )
    assert not [raw for raw in map(encode, reply) if "zqxreplayone" in raw], (
        "T-0180 A4: the first turn's thought must not reach a page greeting after turn 2"
    )
    assert any(isinstance(m, AstraSpoken) and "Yes." in m.text for m in body), (
        f"turn 2's answer is replayed (criterion 1); got {types_of(body)}"
    )


CANVAS_SENTINEL = "ZQXCANVASSECRET"


async def test_canvas_messages_are_never_journalled_or_replayed(tmp_path):
    """Criterion 4, negative: no ``astra.canvas.*`` is journalled or replayed (ruling A9).

    The denoising view is a live wire: a canvas step is one of up to 30 per second and a canvas
    commit carries the *raw segment text* — including ``segment="thought"``, the one thing that
    must never reach a page's transcript. The sibling property row
    (``test_no_thought_token_reaches_a_replayed_message``) drives a whole turn through the rig,
    but the rig's LLM fake never puts a canvas message through ``publish()``, so it cannot see a
    journal that accepts one: allow-listing ``AstraCanvasStep``/``AstraCanvasCommit`` in
    ``TurnJournal.record`` leaves it green. This row hands both straight to ``publish()`` — the
    shape T-0011's tap will have — and is therefore the fence criterion 4 asks for.

    The sentinel lives only in the canvas bodies, so finding it in the reply proves the journal
    took it from there; the spoken line is asserted too, so the row cannot pass by replaying
    nothing.
    """
    _, server = make_server(tmp_path)
    server.publish(transcript("what is on this screen", final=True))
    server.publish(spoken(f"{TURN}-u1", "Here is a quick check."))
    server.publish(
        AstraCanvasStep(
            request_id=f"{TURN}-r1",
            canvas_idx=1,
            step=3,
            max_steps=8,
            canvas_length=4,
            # T-0011 R10: a denoise frame carries all three of canvas/argmax/stable
            canvas=(21, 12, 23, 14),
            argmax=(11, 12, 13, 14),
            stable="50",
            accepted="c0",
            entropy=(0.5, 0.25, 0.12, 0.0),
            segment="thought",
            text_preview=f"{CANVAS_SENTINEL} preview",
        )
    )
    server.publish(
        AstraCanvasCommit(
            request_id=f"{TURN}-r1",
            canvas_idx=1,
            segment="thought",
            text=f"the user probably means {CANVAS_SENTINEL}",
            steps_used=8,
        )
    )

    reply = greet(server).sent()
    assert not [m for m in reply if m.type.startswith("astra.canvas.")], (
        "criterion 4, negative: no astra.canvas.* is journalled or replayed — the denoising view "
        "is a live tap, and a canvas commit carries the raw segment text of a thought. The hello "
        f"reply carried {types_of(reply)}"
    )
    offenders = [raw for raw in map(encode, reply) if CANVAS_SENTINEL in raw]
    assert not offenders, (
        "criterion 4, negative: the sentinel exists only inside the canvas step's preview and the "
        "canvas commit's text, so a replayed message carrying it was journalled from the canvas "
        f"channel (whatever its type says); found {offenders[:2]}"
    )
    assert [m.utterance_id for m in reply if isinstance(m, AstraSpoken)] == [f"{TURN}-u1"], (
        "…and the row is not vacuous: the turn's spoken line is still replayed (criterion 1); the "
        f"reply was {types_of(reply)}"
    )


# --- criterion 5: bounded, and harmless to a fresh connection -----------------------------------


async def test_the_journal_is_bounded_under_ten_thousand_publications(tmp_path):
    """Criterion 5 (**adversarial**: the likely failure mode is an unbounded per-turn list).

    Regime: n = 10 000 partial transcripts + 10 000 spoken lines in one turn, one process, virtual
    clock; the assertion is a retained *count* against the module constants and quotes no time.
    Ruling A12's "a connection that lives for an hour must not cost one message per message
    spoken", applied to the journal: a turn is minutes of speech, one transcript per word.
    """
    cap = require_symbol("REPLAY_MAX_MESSAGES")
    max_spoken = require_symbol("REPLAY_MAX_SPOKEN")
    max_played = require_symbol("REPLAY_MAX_PLAYED")
    assert cap >= 1 and max_spoken >= 1 and max_played >= 1, (cap, max_spoken, max_played)

    _, server = make_server(tmp_path)
    for i in range(10_000):
        server.publish(transcript(f"word {i}"))
        server.publish(spoken(f"{TURN}-u{i}", f"sentence {i}."))

    reply = greet(server).sent()
    journalled = [
        m for m in after_the_live_messages(reply) if not isinstance(m, AstraError | AstraTurnState)
    ]
    assert len(journalled) <= cap, (
        f"criterion 5: the journal retains at most REPLAY_MAX_MESSAGES={cap} messages; after "
        f"20 000 publications in one turn it replayed {len(journalled)}"
    )
    assert len(reply) <= cap + len(LIVE_HEAD) + 1, (
        "criterion 5: the whole reply is the live messages (T-0043 criterion 18: three of them), "
        "at most one truncation error and the journal — the same bound T-0037 pinned, moved by "
        f"the one message criterion 18 adds; it was {len(reply)} messages long"
    )
    assert len([m for m in journalled if isinstance(m, AstraTranscript)]) <= 1, (
        "criterion 5: at most one transcript message is kept (latest-wins); got "
        f"{len([m for m in journalled if isinstance(m, AstraTranscript)])}"
    )
    assert len([m for m in journalled if isinstance(m, AstraSpoken)]) <= max_spoken, (
        f"criterion 5: at most REPLAY_MAX_SPOKEN={max_spoken} spoken lines are kept"
    )
    assert journalled, "…and the bound is not zero: the page still gets the tail of the turn"


async def test_a_full_journal_never_overflows_a_fresh_outbox(tmp_path):
    """Criterion 5, negative: a reconnect can never close itself with ``outbox overflow``.

    ``astra.spoken`` may not be dropped (T-0028 criterion 12), so a journal as large as the outbox
    would close the very connection it is catching up — the page would reconnect, be greeted,
    and be disconnected again, forever.
    """
    cap = require_symbol("REPLAY_MAX_MESSAGES")
    outbox = require_symbol("DEFAULT_OUTBOX")
    assert cap < outbox, (
        f"criterion 5: REPLAY_MAX_MESSAGES ({cap}) must be strictly less than DEFAULT_OUTBOX "
        f"({outbox}), or a full journal closes the connection it is replayed to"
    )

    _, server = make_server(tmp_path)
    for i in range(cap + 50):
        server.publish(spoken(f"{TURN}-u{i}", f"sentence {i}."))
        server.publish(played(f"{TURN}-u{i}"))

    connection = greet(server)
    assert connection.close_reason is None, (
        "criterion 5: the hello reply of a full journal must not overflow a fresh connection's "
        f"outbox; it was closed with {connection.close_reason!r}"
    )
    assert not connection.closed, "…and the page is still connected after being caught up"
    assert len(connection.sent()) <= outbox, (
        f"the whole reply fits in one outbox ({outbox}); it was {len(connection.sent())} messages"
    )


# --- criterion 6: a truncated replay is declared ------------------------------------------------


async def test_a_truncated_replay_declares_itself_once(tmp_path):
    """Criterion 6: one ``astra.error(code="replay_truncated")`` before the replayed lines.

    A partial transcript the page believes is complete is worse than a banner: the user reads a
    sentence the assistant never finished saying. The negative twin is in the same row, because
    "the code is not reused for anything else" is only observable next to a complete replay.
    """
    max_spoken = require_symbol("REPLAY_MAX_SPOKEN")

    _, server = make_server(tmp_path)
    for i in range(max_spoken + 10):
        server.publish(spoken(f"{TURN}-u{i}", f"sentence {i}."))
    reply = greet(server).sent()
    errors = [m for m in reply if isinstance(m, AstraError)]
    assert [m.code for m in errors] == ["replay_truncated"], (
        "criterion 6: a replay that dropped lines says so, exactly once, rather than showing a "
        f"partial transcript as if it were whole; the reply carried {[m.code for m in errors]}"
    )
    assert errors[0].message.strip(), "…and the message names what is missing"
    first_replayed = min(
        i for i, m in enumerate(reply) if isinstance(m, AstraSpoken | AstraTranscript)
    )
    assert reply.index(errors[0]) < first_replayed, (
        "criterion 6: the notice comes *before* the replayed lines, so the error strip is up "
        f"before the pane fills; the reply was {types_of(reply)}"
    )

    _, complete = make_server(tmp_path)
    publish_a_turn(complete)
    clean = greet(complete).sent()
    assert not [m for m in clean if isinstance(m, AstraError)], (
        "criterion 6, negative: a complete replay sends no truncation notice; it sent "
        f"{[m for m in clean if isinstance(m, AstraError)]}"
    )


# --- criterion 1's INTERRUPTED clause -----------------------------------------------------------


async def test_an_interrupted_state_is_replayed_before_the_current_one(tmp_path):
    """Criterion 1: the barge-in chip is restated, then corrected — never left reading INTERRUPTED.

    A page that reloads after a barge-in must see the turn as it was (INTERRUPTED explains the
    struck tail) without the chip getting stuck on a state the server has already left.
    """
    rig, server = make_server(tmp_path)
    server.publish(transcript("what is on this screen", final=True))
    server.publish(spoken(f"{TURN}-u1", "Here is a quick check."))
    server.publish(AstraTurnState(state=TurnState.INTERRUPTED, turn_id=TURN))
    server.publish(played(f"{TURN}-u1"))

    reply = greet(server).sent()
    states = [m for m in reply if isinstance(m, AstraTurnState)]
    interrupted = [m for m in states if m.state == TurnState.INTERRUPTED]
    assert interrupted, (
        "criterion 1: an INTERRUPTED turn state published for this turn is part of the journal; "
        f"the reply carried {[m.state for m in states]}"
    )
    at = {id(m): i for i, m in enumerate(reply)}  # by identity: two turn states can be equal
    played_messages = [m for m in reply if isinstance(m, AstraSpokenPlayed)]
    assert played_messages and at[id(interrupted[0])] < at[id(played_messages[0])], (
        "criterion 1: the INTERRUPTED state comes before the played messages; the reply was "
        f"{types_of(reply)}"
    )
    current = rig.session.turn_state_message()
    assert states[-1].state == current.state, (
        "criterion 1: …followed by one repeat of the current astra.turn.state, so the chip is "
        f"not left reading INTERRUPTED. The last state in the reply was {states[-1].state} and "
        f"the server is in {current.state}"
    )
    assert at[id(states[-1])] > at[id(interrupted[0])], (
        "criterion 1: the repeat of the current state comes after the replayed INTERRUPTED one, "
        f"or the chip is left reading INTERRUPTED; the reply was {types_of(reply)}"
    )


# --- criterion 7: the pump and the outbox are unchanged ------------------------------------------


def test_turn_state_is_still_the_only_droppable_kind_including_replayed():
    """Criterion 7, negative, over ``Outbox`` itself: ``replay=True`` changes no drop rule.

    Hand-computed, exactly as T-0028's row: with ``maxsize=2`` three replayed turn states leave
    the newest queued and count a drop, and a third replayed ``astra.spoken`` is refused — a
    replayed spoken line is still a line the transcript pane would silently lose.
    """
    outbox_cls = require_symbol("Outbox")

    states = outbox_cls(maxsize=2)
    for i in range(3):
        state = replayed(AstraTurnState(state=TurnState.SPEAKING, turn_id=f"t{i}"))
        assert states.offer(state) is True, (
            "criterion 7: a replayed turn.state is still latest-wins and droppable"
        )
    assert states.dropped >= 1, f"a dropped state is counted; dropped={states.dropped}"
    assert [m.turn_id for m in states.drain()][-1] == "t2", "the newest state survives"

    lines = outbox_cls(maxsize=2)
    assert lines.offer(replayed(spoken("t1-u1", "one"))) is True
    assert lines.offer(replayed(spoken("t1-u2", "two"))) is True
    assert lines.offer(replayed(spoken("t1-u3", "three"))) is False, (
        "criterion 7: a replayed astra.spoken is not droppable either — the overflow closes that "
        "connection instead of silently desyncing the pane"
    )
