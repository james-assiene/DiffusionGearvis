"""Ingress: total decoding, direction, hello-first, frames, assists (T-0028 criteria 3-6).

No socket. ``server.open()`` hands back a connection with no socket behind it (the seam is fixed
in ``tests/unit/_hud_rig.py``'s module docstring), so what the transport *decides* is testable
without what the transport *transports*:

    conn = server.open(); conn.receive(raw); conn.sent(); conn.close_reason; server.counts

The five decode outcomes of criterion 3 are the real ``contracts/codec.decode``'s own branches —
this module never re-implements the codec, it sends bytes and asserts what came back. The assist
oracle is ``AssistState`` itself (T-0012 criterion 8: the one authority), never a badge string
built here.

Regime: no measurement, ``ScriptedClock``, ``tests/fakes/``.
"""

from __future__ import annotations

import base64
import json
from itertools import zip_longest
from pathlib import Path

from local_astra.contracts import events as E
from local_astra.contracts.codec import encode
from local_astra.contracts.perception import AssistFlags, Source
from local_astra.contracts.protocol import (
    PROTOCOL_VERSION,
    AstraAssistState,
    AstraAssistToggle,
    AstraError,
    AstraFrameText,
    AstraSpoken,
    AstraTurnState,
)
from local_astra.contracts.turn import TurnState
from tests.fakes.hud_server import oversized_message
from tests.unit import _t0181_pins as t0181
from tests.unit import test_t0213_settings_wire as t0213  # T-0213 R-5 named re-pin
from tests.unit._hud_rig import (
    Recorder,
    ScriptedClock,
    answer,
    astra_frame,
    require_symbol,
    running,
)
from tests.unit._session_log_rig import failing_writes, kinds, make_log, read_records


class RecordingSink:
    """A frame sink that records what the server handed it, and changes nothing (criterion 5)."""

    def __init__(self) -> None:
        self.ingested: list[object] = []

    def ingest(self, message):
        self.ingested.append(message)
        return None


def serve(rig, sink=None):
    """A ``HudServer`` over ``rig``'s session with **no socket**: nothing is bound or started."""
    hud_server = require_symbol("HudServer")
    return hud_server(rig.session, frame_sink=sink if sink is not None else RecordingSink(), port=0)


def hello_first(server):
    """A connection that has said a valid hello, with the hello's own reply cleared away."""
    from local_astra.contracts.codec import encode
    from local_astra.contracts.protocol import AstraHello

    conn = server.open()
    conn.receive(encode(AstraHello()))
    return conn


def errors(conn) -> list[AstraError]:
    return [m for m in conn.sent() if isinstance(m, AstraError)]


def codes(conn) -> list[str]:
    return [m.code for m in errors(conn)]


async def test_only_client_to_server_types_are_routed(tmp_path, monkeypatch):
    """Criterion 3: all five decode outcomes, each counted, none of them fatal.

    The five branches of ``contracts/codec.decode`` are, in the order sent below: a
    ``CLIENT_TO_SERVER`` type (routed), a server→client type (``wrong_direction``), an unknown
    ``astra.*`` type (``codec.Unknown`` — no *known* type is unhandled today, both frozensets
    together cover all fourteen), a non-``astra.`` ``RealtimeEvent`` (v1 relays nothing) and a
    ``Malformed``. After all five the connection is still usable.
    """
    async with running(tmp_path, publish=Recorder()) as rig:
        server = serve(rig)
        conn = hello_first(server)
        submitted: list[E.Event] = []
        real_submit = rig.session.submit
        monkeypatch.setattr(
            rig.session, "submit", lambda event: (submitted.append(event), real_submit(event))[1]
        )
        before = dict(server.counts)

        conn.receive('{"type":"astra.assist.toggle","flags":{"page_text":true}}')
        conn.receive('{"type":"astra.spoken","turn_id":"t1","utterance_id":"u","text":"hi"}')
        conn.receive('{"type":"astra.not.a.real.type","x":1}')
        conn.receive('{"type":"session.update","session":{}}')
        conn.receive("{this is not json")

        counts = {k: server.counts[k] - before.get(k, 0) for k in server.counts}
        assert counts.get("routed") == 1, (
            f"criterion 3: exactly the one client→server message was routed; counts={counts}"
        )
        assert counts.get("wrong_direction") == 1 and counts.get("malformed") == 1, (
            f"a server→client type and bad JSON are each refused and counted; counts={counts}"
        )
        assert counts.get("unhandled") == 1 and counts.get("relay_ignored") == 1, (
            "an unknown astra.* type and a plain RealtimeEvent are ignored and counted (v1 "
            f"relays nothing); counts={counts}"
        )
        assert codes(conn) == ["wrong_direction", "malformed"], (
            "criterion 3: the wrong-direction message and the malformed one each get their own "
            f"astra.error, and the ignored ones get nothing; the client saw {codes(conn)}"
        )
        assert rig.session.assists.flags == AssistFlags(page_text=True), (
            "the one routed message took effect (otherwise the counters above prove nothing)"
        )
        assert not [e for e in submitted if isinstance(e, E.UiAction | E.SceneChanged)], (
            f"none of the four refused messages reached the session as an event: {submitted}"
        )
        assert conn.close_reason is None, (
            f"in all five outcomes the socket stays open; it was closed: {conn.close_reason!r}"
        )

        conn.receive('{"type":"astra.assist.toggle","flags":{"cells":true}}')
        assert rig.session.assists.flags == AssistFlags(cells=True), (
            "…and the pump is unaffected: the next legitimate message still routes"
        )


async def test_nothing_is_routed_before_hello_and_a_bad_version_is_fail_closed(tmp_path):
    """Criterion 4: hello first, fail-closed, and the hello's reply is the server's truth.

    Three connections, one each: no hello at all, a good hello, a hello with the wrong protocol
    version. T-0025's panel and T-0008's chip start from ``astra.assist.state`` +
    ``astra.turn.state``, which is why the reply carries both.
    """
    from local_astra.contracts.codec import encode
    from local_astra.contracts.protocol import AstraHello

    async with running(tmp_path, publish=Recorder()) as rig:
        server = serve(rig)

        silent = server.open()
        silent.receive('{"type":"astra.assist.toggle","flags":{"page_text":true}}')
        assert codes(silent) == ["no_hello"], (
            f"criterion 4: nothing is routed before a valid hello; got {codes(silent)}"
        )
        assert server.counts.get("no_hello") == 1, server.counts
        assert rig.session.assists.flags == AssistFlags(), (
            "…and fail-closed means the assist flags did not move"
        )

        greeted = server.open()
        greeted.receive(encode(AstraHello()))
        kinds = [type(m) for m in greeted.sent()]
        assert AstraAssistState in kinds and AstraTurnState in kinds, (
            "criterion 4: a valid hello is answered with the current assist state and turn "
            f"state, so the page starts from the server's truth; it got {greeted.sent()}"
        )
        state = next(m for m in greeted.sent() if isinstance(m, AstraTurnState))
        assert state.state == rig.state.state.value and state.turn_id == rig.state.turn_id, (
            f"the turn state in the hello reply is the machine's current one; got {state}"
        )

        wrong = server.open()
        wrong.receive(encode(AstraHello(protocol_version=PROTOCOL_VERSION + 1)))
        assert codes(wrong) == ["protocol_version"], (
            f"criterion 4: a hello with the wrong protocol version is named as such; {codes(wrong)}"
        )
        assert wrong.close_reason is None, (
            "…and the socket stays open so the page can display the message; it was closed: "
            f"{wrong.close_reason!r}"
        )
        wrong.receive('{"type":"astra.assist.toggle","flags":{"cells":true}}')
        assert codes(wrong)[-1] == "no_hello" and rig.session.assists.flags == AssistFlags(), (
            "…and that connection stays fail-closed for ingress; it routed a toggle: "
            f"flags={rig.session.assists.flags}"
        )

        # negative: a second hello does not reset the assist flags
        greeted.receive(encode(AstraAssistToggle(flags=AssistFlags(page_text=True))))
        greeted.receive(encode(AstraHello()))
        assert rig.session.assists.flags == AssistFlags(page_text=True), (
            "criterion 4, negative: a reconnect's second hello re-states the server's truth, it "
            f"does not clear it; flags are {rig.session.assists.flags}"
        )
        echo = [m for m in greeted.sent() if isinstance(m, AstraAssistState)][-1]
        assert echo.flags == AssistFlags(page_text=True), (
            f"…and the reply carries the flags that are actually in force; it said {echo}"
        )


async def test_a_frame_reaches_the_frame_sink_untouched(tmp_path):
    """Criterion 5: ``astra.frame`` is handed to the sink verbatim, and nothing else happens.

    The sink is the **real** ``BrowserFrameSource`` (T-0008's, consumed unchanged), used — as
    ``astra session`` will use it — both as the session's ``PerceptionClient`` and as the
    server's frame sink. Asserted: the bytes in the ring are exactly the base64 the page sent,
    the phash is the one it sent, and the frame the turn snapshots is that frame (it reaches
    memory as the turn's episode). The *negative* is the second frame: a phash the source
    refuses must be refused by the source (``rejected``), not filtered out earlier — a second
    validation in the server is a second rule to drift (``.claude/rules/perception-memory.md``).
    """
    from local_astra.contracts.codec import encode
    from local_astra.perception.browser_source import BrowserFrameSource

    clock = ScriptedClock()
    source = BrowserFrameSource(now_ms=lambda: clock.now_ms)
    async with running(
        tmp_path,
        script=[answer("It is a stack trace. ")],
        ms_per_word=100,
        publish=Recorder(),
        clock=clock,
        perception=source,
    ) as rig:
        server = serve(rig, sink=source)
        conn = hello_first(server)

        posted = astra_frame("b")
        conn.receive(encode(posted))
        stored = await source.get_frame(Source.SCREEN, at_ms=rig.clock.now_ms)
        assert stored is not None, (
            "criterion 5: the frame must reach the injected sink; the ring is empty"
        )
        assert stored.jpeg == base64.b64decode(posted.jpeg_b64), (
            "the server does not decode, resize or re-encode the JPEG: the bytes in the ring are "
            "exactly what the page sent"
        )
        assert stored.phash == posted.phash, (
            f"nor does it recompute the phash; sent {posted.phash!r}, stored {stored.phash!r}"
        )
        assert (stored.width, stored.height) == (posted.width, posted.height)

        rejected_before = source.rejected
        bad = posted.model_copy(update={"phash": "not-a-hash"})
        conn.receive(encode(bad))
        assert source.rejected == rejected_before + 1, (
            "criterion 5, negative: validation is BrowserFrameSource's alone (T-0008 criteria "
            "14-18). A frame with an impossible phash must reach it and be counted there, not be "
            "filtered out by a second rule in the server"
        )
        assert conn.close_reason is None

        await rig.ask("what is on this screen")
        await rig.speak_out()
        phashes = [ep.phash for ep in rig.memory.episodes.values()]
        assert posted.phash in phashes, (
            "criterion 5: the session's SnapshotFrame(at_ms=onset) resolves to the posted frame, "
            f"so that is the frame the turn stores; memory holds {phashes}"
        )


async def test_a_toggle_echoes_the_servers_state_and_frame_text_is_fail_closed(tmp_path):
    """Criterion 6: the server echoes what ``AssistState`` returns and derives nothing.

    The oracle is ``session.assists.state_message()`` — the one authority (T-0012 criterion 8,
    T-0025 criterion 4). The badge in the echo must be *that* object's, not a string this task
    built: ``AssistFlags.badge()`` stays the single derivation.
    """
    from local_astra.contracts.codec import encode

    async with running(tmp_path, publish=Recorder()) as rig:
        server = serve(rig)
        one, two = hello_first(server), hello_first(server)

        flags = AssistFlags(page_text=True, cells=True)
        one.receive(encode(AstraAssistToggle(flags=flags)))
        authority = rig.session.assists.state_message()
        for name, conn in (("the sender", one), ("the other page", two)):
            echo = [m for m in conn.sent() if isinstance(m, AstraAssistState)][-1]
            assert echo == authority, (
                f"criterion 6: {name} receives the AssistState's own reply verbatim — flags and "
                f"badge. AssistState said {authority}; the wire carried {echo}"
            )
        assert authority.badge == "pixels+dom+cells", (
            "sanity on the oracle itself (AssistFlags.badge, T-0012 criterion 8): two assists on "
            f"read 'pixels+dom+cells'; it read {authority.badge!r}"
        )

        dropped_before = dict(rig.session.assists.dropped)
        off_kind = AstraFrameText(
            source=Source.SCREEN, ts_ms=0, kind="pdf_text", text="secret pdf text"
        )
        one.receive(encode(off_kind))
        assert rig.session.assists.dropped["pdf_text"] == dropped_before["pdf_text"] + 1, (
            "criterion 6: astra.frame.text for a kind that is off is dropped by AssistState "
            "(T-0012's fail-closed ingress) and counted there"
        )
        assert server.counts.get("assist_dropped") == 1, (
            f"…and the server counts the drop rather than rescuing it; counts={server.counts}"
        )
        assert rig.session.assists.texts_for(0) == (), (
            "…and nothing was buffered: an assist that is off produces no FrameText"
        )
        assert not [m for m in one.sent()[-1:] if isinstance(m, AstraError)], (
            "a fail-closed drop is not an error the page must show; it is a counter"
        )


async def test_a_server_direction_message_and_an_oversize_message_keep_the_socket_open(tmp_path):
    """Criterion 3, adversarial: ``astra.spoken`` from the client, then 8 MB, back to back.

    Both are things a confused or hostile page can send at any time. ``astra.spoken`` decodes
    *fine* — it is a real model — and is refused only because of its direction; the oversize
    message is refused by ``codec.MAX_MESSAGE_BYTES`` (T-0027's byte-counted limit, inherited
    as-is — this task adds no second size rule). Neither may produce an event, close the socket
    or raise out of the connection handler. Mirrors T-0008 criterion 2 on the server side.
    """
    from local_astra.contracts.codec import encode

    async with running(tmp_path, publish=Recorder()) as rig:
        server = serve(rig)
        conn = hello_first(server)
        events_before = len(rig.machine.log)

        conn.receive(encode(AstraSpoken(turn_id="t1", utterance_id="t1-u1", text="I said this")))
        huge = oversized_message()
        assert len(huge.encode()) > 8 * 1024 * 1024, "the adversarial message must really be 8 MB+"
        conn.receive(huge)
        await rig.pump()

        assert codes(conn) == ["wrong_direction", "malformed"], (
            "criterion 3: astra.spoken from a client is refused as wrong_direction and the "
            f"oversize message as malformed; the page was told {codes(conn)}"
        )
        assert conn.close_reason is None, (
            f"neither closes the socket (T-0008 criterion 2 expects the error, not a bare "
            f"close); it was closed with {conn.close_reason!r}"
        )
        assert server.counts.get("wrong_direction") == 1 and server.counts.get("malformed") == 1, (
            f"both are counted; counts={server.counts}"
        )
        assert len(rig.machine.log) == events_before, (
            "neither produced an event: the machine saw "
            f"{[type(e).__name__ for e, _, _ in rig.machine.log[events_before:]]}"
        )
        assert not rig.session.failures, (
            f"nothing raised out of the connection handler: {rig.session.failures}"
        )

        conn.receive(encode(AstraAssistToggle(flags=AssistFlags(at_spi=True))))
        assert rig.session.assists.flags == AssistFlags(at_spi=True), (
            "…and the connection still works afterwards"
        )


# --- T-0039 group A: astra.text ingress ---------------------------------------------------------
#
# Criterion 3 of docs/tasks/T-0039.md. The transport stays dumb: it decodes with the production
# codec, applies the hello fence every other client message obeys, builds **one**
# ``events.UserText`` and hands it to ``session.submit``. It does not strip, cap, refuse-as-busy
# or look at the machine's state — every refusal that depends on state is the session's, because
# the session is the only thing that knows what state it is in (§5, "the transport stays dumb").

TEXT_TYPE = "astra.text"

#: docs/tasks/T-0039.md criterion 2's cap, in code points.
TEXT_MAX_CODE_POINTS = 2000


def typed(text: str) -> str:
    """One ``astra.text`` frame exactly as the page would put it on the socket."""
    return json.dumps({"type": TEXT_TYPE, "text": text}, ensure_ascii=False)


def user_text_class():
    cls = getattr(E, "UserText", None)
    assert cls is not None, (
        'T-0039 criterion 4: contracts/events.py gains UserText(type="user.text", t_ms, text) '
        "and puts it in Event/EVENT_TYPES (docs/tasks/T-0039.md)"
    )
    return cls


def record_submissions(rig, monkeypatch) -> list:
    """Every event the session is handed, in order — the seam criterion 3 is about."""
    submitted: list = []
    real_submit = rig.session.submit
    monkeypatch.setattr(
        rig.session, "submit", lambda event: (submitted.append(event), real_submit(event))[1]
    )
    return submitted


async def test_a_typed_question_before_hello_is_refused_and_counted(tmp_path, monkeypatch):
    """Criterion 3: the hello fence covers ``astra.text`` like every other client message.

    The fence is what keeps a page that has not identified itself from opening a turn. Routing
    the new type before the greeting — the shortest way to "get text working" — would also skip
    the protocol-version check, so an old page's question would be answered by a newer endpoint.
    """
    async with running(tmp_path, publish=Recorder()) as rig:
        server = serve(rig)
        submitted = record_submissions(rig, monkeypatch)
        conn = server.open()  # no hello
        conn.receive(typed("what is on my screen"))
        await rig.pump()

        assert codes(conn) == ["no_hello"], (
            "criterion 3: an astra.text before astra.hello is answered "
            f"astra.error(code='no_hello'); the page was told {codes(conn)}"
        )
        assert server.counts.get("no_hello") == 1, f"…and counted; counts={server.counts}"
        assert server.counts.get("routed") == 0, "…and nothing was routed"
        assert not [e for e in submitted if not isinstance(e, E.Tick)], (
            f"…and no event reached the session: {submitted}"
        )
        assert conn.close_reason is None, "the socket stays open"


async def test_a_typed_question_after_hello_becomes_exactly_one_user_text_event(
    tmp_path, monkeypatch
):
    """Criterion 3: one routed message, one ``events.UserText``, on the one pump.

    The text carried is the *model's* text (criterion 2 strips it), which is how this row also
    shows the transport did not normalise anything itself: it never saw the raw string.

    "One event, not a method call" is the load-bearing half (§5, landmine): ordering with frames,
    ticks and microphone onsets on the single pump is the reason ``ui.action`` is an event, and
    losing that ordering is exactly how criterion 6's foreign-onset guard fails.
    """
    text_cls = user_text_class()
    async with running(tmp_path, publish=Recorder()) as rig:
        server = serve(rig)
        conn = hello_first(server)
        routed_before = server.counts.get("routed", 0)
        submitted = record_submissions(rig, monkeypatch)
        at = rig.clock.now_ms

        conn.receive(typed("  what is on my screen\n"))
        await rig.pump()

        texts = [e for e in submitted if isinstance(e, text_cls)]
        assert len(texts) == 1, (
            "criterion 3: one astra.text submits exactly one events.UserText; the session was "
            f"handed {[type(e).__name__ for e in submitted if not isinstance(e, E.Tick)]}"
        )
        assert texts[0].text == "what is on my screen", (
            "criterion 2/3: the text is the model's stripped text, verbatim — the transport does "
            f"not touch it; the event carries {texts[0].text!r}"
        )
        assert texts[0].type == "user.text", (
            f"criterion 4: the event tag is 'user.text'; it is {texts[0].type!r}"
        )
        assert at <= texts[0].t_ms <= rig.clock.now_ms, (
            "criterion 3/5: stamped on the session's own clock, at the message's own time; "
            f"t_ms={texts[0].t_ms} is outside [{at}, {rig.clock.now_ms}]"
        )
        assert server.counts.get("routed", 0) == routed_before + 1, (
            f"criterion 3: a routed astra.text increments counts['routed']; counts={server.counts}"
        )
        assert server.counts.get("malformed") == 0 and server.counts.get("unhandled") == 0, (
            f"…and nothing else; counts={server.counts}"
        )
        assert codes(conn) == [], f"an accepted question is not an error: {codes(conn)}"
        assert conn.close_reason is None


async def test_an_over_long_typed_question_is_malformed_and_keeps_the_socket_open(
    tmp_path, monkeypatch
):
    """Criteria 2 and 3, adversarial: the refusal the user can trigger by pasting a document.

    The page caps at 2000 code points too (criterion 12), so this arrives only from a page that
    is out of date, patched, or not the HUD at all. It must land on T-0028's existing
    ``malformed`` answer — not a new error code, not a truncation, and above all not a closed
    socket: a page whose socket dies on a long paste loses the turn *and* the connection, and the
    user sees an empty pane with no reason. The next question must still work.
    """
    text_cls = user_text_class()
    async with running(tmp_path, publish=Recorder()) as rig:
        server = serve(rig)
        conn = hello_first(server)
        submitted = record_submissions(rig, monkeypatch)
        long_question = "a" * (TEXT_MAX_CODE_POINTS + 1)
        assert len(long_question.encode()) < 8 * 1024 * 1024, (
            "premise: far under T-0027's byte limit, so this is criterion 2's cap failing, not "
            "the inherited one"
        )

        conn.receive(typed(long_question))
        await rig.pump()

        assert codes(conn) == ["malformed"], (
            "criterion 2: an over-long astra.text is answered astra.error(code='malformed') — "
            f"the existing code, not a new one; the page was told {codes(conn)}"
        )
        assert server.counts.get("malformed") == 1 and server.counts.get("routed", 0) == 1, (
            "criterion 2/3: counted as malformed and not routed (the 1 routed is the hello); "
            f"counts={server.counts}"
        )
        assert conn.close_reason is None and not conn.closed, (
            f"criterion 2: the socket stays open; it was closed with {conn.close_reason!r}"
        )
        assert not [e for e in submitted if isinstance(e, text_cls)], (
            "…and no turn was opened by a question the model refused"
        )

        conn.receive(typed("and now a short one"))
        await rig.pump()
        assert [e.text for e in submitted if isinstance(e, text_cls)] == ["and now a short one"], (
            "…and the connection still works afterwards"
        )


async def test_the_transport_routes_a_typed_question_even_when_the_session_is_busy(
    tmp_path, monkeypatch
):
    """Criterion 3, the negative twin: the transport makes **no turn decision of its own**.

    A ``busy`` refusal decided in ``route`` would read a state the message has not reached yet:
    between the socket and the pump sit the frames, ticks and onsets already queued, so the state
    the transport sees is not the state the question arrives in. The refusal below is therefore
    the *session's* (criterion 7) and the message is routed and counted all the same.
    """
    text_cls = user_text_class()
    async with running(tmp_path, publish=Recorder()) as rig:
        server = serve(rig)
        conn = hello_first(server)
        await rig.onset()  # the microphone has the floor: the machine is LISTENING
        assert rig.state.state is not TurnState.IDLE, "premise: a turn is live"
        submitted = record_submissions(rig, monkeypatch)
        routed_before = server.counts.get("routed", 0)

        conn.receive(typed("what about this one"))
        await rig.pump()

        assert server.counts.get("routed", 0) == routed_before + 1, (
            f"criterion 3: the message is routed whatever the state is; counts={server.counts}"
        )
        assert [e.text for e in submitted if isinstance(e, text_cls)] == ["what about this one"], (
            "criterion 3: …and it reaches the pump as an ordinary event; the session was handed "
            f"{[type(e).__name__ for e in submitted if not isinstance(e, E.Tick)]}"
        )
        assert codes(conn) == ["busy"], (
            "criterion 7: the refusal is the session's, with the code the [ui] path already "
            f"uses; the page was told {codes(conn)}"
        )


def long_typed_question() -> str:
    """2000 code points in which every position is distinguishable from every other.

    ``"a" * 2000`` would let a truncation to 1999 hide behind a length check nobody wrote and a
    reordering hide entirely; a numbered token per word means any cut, any splice and any
    normalisation shows up in the failure message at the exact word where it happened.
    """
    body = " ".join(f"w{i:03d}" for i in range(400))
    return body + "?"


async def test_a_long_typed_question_is_routed_byte_identical(tmp_path, monkeypatch):
    """Criterion 3, owed by ruling A14: the transport does not shorten the user's question.

    A question at the cap is the one the transport is most tempted to clip — a truncation to some
    "reasonable" prefix is invisible in every short-question row, and it is the user, not the
    page, who then gets an answer to half a question with no indication that half of it was
    dropped. Criterion 2 puts the only length rule on the model (over the cap → ``malformed``);
    what survives that rule reaches the pump unchanged, to the code point.
    """
    text_cls = user_text_class()
    question = long_typed_question()
    assert len(question) == TEXT_MAX_CODE_POINTS, (
        f"premise: exactly at the cap, so the model accepts it; it is {len(question)}"
    )
    assert question == question.strip(), "premise: the strip is a no-op, so this row is the cap's"
    assert len(set(question.split())) == 400, "premise: no two words are alike"

    async with running(tmp_path, publish=Recorder()) as rig:
        server = serve(rig)
        conn = hello_first(server)
        submitted = record_submissions(rig, monkeypatch)

        conn.receive(typed(question))
        await rig.pump()

        texts = [e for e in submitted if isinstance(e, text_cls)]
        assert len(texts) == 1, (
            "criterion 3: one astra.text at the cap submits exactly one events.UserText; the "
            f"session was handed {[type(e).__name__ for e in submitted if not isinstance(e, E.Tick)]}"
        )
        got = texts[0].text
        assert len(got) == TEXT_MAX_CODE_POINTS, (
            "criterion 3: the transport carries the question whole — no prefix, no clamp; the "
            f"event carries {len(got)} of {TEXT_MAX_CODE_POINTS} code points, ending {got[-40:]!r}"
        )
        assert got == question, (
            "criterion 3: …and byte-identical; the first difference is at word "
            f"{next(i for i, (a, b) in enumerate(zip_longest(got.split(), question.split())) if a != b)}"
        )
        assert codes(conn) == [], f"a question at the cap is not an error: {codes(conn)}"
        assert server.counts.get("malformed") == 0, f"…nor malformed; counts={server.counts}"


# --- T-0043 group B: the session-log toggle on the wire (criteria 17 and 18) ---------------------
#
# The message is built as raw JSON rather than from the model, so these rows exercise the same
# path a page does and do not need ``AstraLogToggle`` to exist to be *collected*: before group B
# lands, ``codec.decode`` answers ``Unknown`` and the assertions below fail on what the server
# did, which is the failure this file is for.
#
# The session is driven through the transport only — ``conn.receive(...)`` in, ``conn.sent()`` and
# the files in the log directory out — because criterion 17 is a statement about the wire and the
# file, not about whichever method name the session grows for it.

LOG_TOGGLE_TYPE = "astra.log.toggle"
LOG_STATE_TYPE = "astra.log.state"


def log_toggle(on: bool) -> str:
    """One ``astra.log.toggle`` exactly as the panel would put it on the socket."""
    return json.dumps({"type": LOG_TOGGLE_TYPE, "on": on})


def log_states(conn) -> list:
    """Every ``astra.log.state`` this connection would put on the wire, oldest first.

    Matched by wire type rather than by class, so a missing model is an assertion about what the
    page was sent and not an ``ImportError`` at collection time.
    """
    return [m for m in conn.sent() if getattr(m, "type", None) == LOG_STATE_TYPE]


def state_fields(message) -> dict:
    return {
        name: getattr(message, name, "<missing>")
        for name in ("on", "path", "records", "dropped", "bytes", "error")
    }


def log_files(directory) -> list:
    """The log files in a ``--log-dir``, oldest first by name (the name carries the timestamp)."""
    return sorted(p for p in Path(directory).glob("*.jsonl"))


async def settle_toggle(rig, predicate, what: str, rounds: int = 100) -> None:
    """Let the session finish what a routed toggle started (closing a file is a coroutine).

    Bounded and clock-free: the rig's own pump, not a sleep on the wall. A predicate that never
    comes true is an ``AssertionError`` naming what was waited for.
    """
    for _ in range(rounds):
        if predicate():
            return
        await rig.pump(rounds=1)
    raise AssertionError(
        f"T-0043 criterion 17: {what} did not happen within {rounds} pump rounds after the "
        "astra.log.toggle was routed"
    )


async def logging_rig(tmp_path, **kwargs):
    """A rig whose session is logging into ``tmp_path/logs`` — the state ``astra session`` starts in."""
    log_dir = tmp_path / "logs"
    log = make_log(log_dir)
    return log_dir, running(tmp_path, publish=Recorder(), session_log=log, **kwargs)


async def test_a_log_toggle_before_hello_is_refused_and_changes_nothing(tmp_path):
    """Criterion 17: the hello fence comes first, exactly as it does for every client message.

    The fence is not a formality here: an unauthenticated frame that could close the session's
    log would let anything that can open a socket switch the recording off, and the file would
    end with a ``log.stopped`` nobody asked for.
    """
    log_dir, rig_ctx = await logging_rig(tmp_path)
    async with rig_ctx as rig:
        server = serve(rig)
        conn = server.open()  # no hello
        before = dict(server.counts)

        conn.receive(log_toggle(False))
        await rig.pump()

        counts = {k: server.counts[k] - before.get(k, 0) for k in server.counts}
        assert counts.get("no_hello") == 1, (
            f"criterion 17: astra.log.toggle sits behind the hello fence like every other client "
            f"message; counts={counts}"
        )
        assert counts.get("routed", 0) == 0, f"…and is not routed; counts={counts}"
        assert codes(conn) == ["no_hello"], (
            f"criterion 17: the page is told why, on the path T-0028 built; got {codes(conn)}"
        )
        assert log_states(conn) == [], "a refused toggle publishes no state"
        assert kinds(read_records(rig.session_log.path)).count("log.stopped") == 0, (
            "criterion 17: the file the session is still writing has not been closed by an "
            "ungreeted socket"
        )
        assert len(log_files(log_dir)) == 1, (
            f"…and no second file was opened; {[p.name for p in log_files(log_dir)]}"
        )


async def test_turning_logging_off_closes_the_file_with_its_reason_and_counters(tmp_path):
    """Criterion 17: off writes ``log.stopped`` (reason + counters), drains and closes.

    The counters are checked against the file itself — the number of lines before the last one
    and the bytes on disk — so a ``log.stopped`` carrying zeros, or the counters of some other
    log, fails here. A log that stops without saying how much it holds cannot be told apart from
    one that was never written to.
    """
    log_dir, rig_ctx = await logging_rig(tmp_path, script=[answer("It is a stack trace. ")])
    async with rig_ctx as rig:
        server = serve(rig)
        conn = hello_first(server)
        path = Path(rig.session_log.path)
        before = dict(server.counts)

        conn.receive(log_toggle(False))
        await settle_toggle(
            rig,
            lambda: "log.stopped" in kinds(read_records(path)),
            "the log file was closed with a log.stopped record",
        )

        counts = {k: server.counts[k] - before.get(k, 0) for k in server.counts}
        assert counts.get("routed") == 1, (
            f"criterion 17: a toggle after the hello counts as routed, like every other client "
            f"message; counts={counts}"
        )
        records = read_records(path)
        assert kinds(records)[-1] == "log.stopped", (
            f"criterion 17: log.stopped is the last line of the file; {kinds(records)[-4:]}"
        )
        stopped = records[-1]["data"]
        assert stopped.get("reason") == "toggled off", (
            "criterion 17: the final record says *why* the log ended — 'toggled off' is not the "
            f"same event as a session that closed; got {stopped!r}"
        )
        assert int(stopped["records"]) == len(records) - 1, (
            f"criterion 17: log.stopped counts the records before it ({len(records) - 1} lines); "
            f"it says {stopped['records']}"
        )
        assert int(stopped["dropped"]) == 0, f"nothing overflowed in this session: {stopped!r}"
        size = path.stat().st_size
        assert abs(int(stopped["bytes"]) - size) <= len(path.read_text().splitlines()[-1]) + 1, (
            f"criterion 17: the byte count is the file it is the last line of ({size} bytes on "
            f"disk); it says {stopped['bytes']}"
        )

        # ... and the file is *closed*: a turn taken afterwards appends nothing to it.
        frozen = path.read_bytes()
        conn.receive(encode(astra_frame("z")))
        await rig.ask("does this still write anything")
        await rig.pump()
        assert path.read_bytes() == frozen, (
            "criterion 17: turning logging off closes the file — a session that keeps appending "
            f"to it wrote {len(path.read_bytes()) - len(frozen)} more bytes after the toggle"
        )
        assert len(log_files(log_dir)) == 1, (
            f"…and opened no replacement; {[p.name for p in log_files(log_dir)]}"
        )


async def test_the_log_state_reaches_every_connection(tmp_path):
    """Criterion 17: one ``astra.log.state`` per routed toggle, to **every** connection.

    Two pages are open on this session — the desk and the laptop — and the switch is the
    session's, not the socket's. A state sent only to the socket that clicked leaves the other
    page showing a checkbox that contradicts the file on disk, which is the criterion-19 failure
    one connection removed.
    """
    _log_dir, rig_ctx = await logging_rig(tmp_path)
    async with rig_ctx as rig:
        server = serve(rig)
        clicker, watcher = hello_first(server), hello_first(server)
        path = Path(rig.session_log.path)
        seen = {id(clicker): len(log_states(clicker)), id(watcher): len(log_states(watcher))}

        clicker.receive(log_toggle(False))
        await settle_toggle(
            rig,
            lambda: "log.stopped" in kinds(read_records(path)),
            "the log file was closed",
        )

        for conn, who in ((clicker, "the page that clicked"), (watcher, "the other page")):
            fresh = log_states(conn)[seen[id(conn)] :]
            assert len(fresh) == 1, (
                f"criterion 17: exactly one astra.log.state per routed toggle reaches {who}; it "
                f"got {len(fresh)}: {[state_fields(m) for m in fresh]}"
            )
            fields = state_fields(fresh[0])
            assert fields["on"] is False, (
                f"criterion 17: the echo says what the session did, and it stopped logging; "
                f"{who} was told {fields}"
            )
            assert fields["path"] is None, (
                "ruling B8: an off state names no file — `path` is the file this session is "
                "writing, and it is writing none. The panel renders the string verbatim beside "
                f"the checkbox, so the closed {path.name} here would read as the live recording; "
                f"{who} was told {fields}"
            )
            assert fields["error"] is None, f"nothing failed here; {who} was told {fields}"


async def test_turning_logging_back_on_opens_a_new_file(tmp_path):
    """Criterion 17: on opens a **new** file and writes ``log.started``; the old one is untouched.

    "A new file with a new timestamp" is asserted as *a different path whose first record is
    ``log.started``* — two toggles inside one second legitimately share a timestamp, and the
    property that matters is that the second session's records never land in the file the first
    one already closed with its counters.
    """
    log_dir, rig_ctx = await logging_rig(tmp_path, script=[answer("It is a stack trace. ")])
    async with rig_ctx as rig:
        server = serve(rig)
        conn = hello_first(server)
        first = Path(rig.session_log.path)

        conn.receive(log_toggle(False))
        await settle_toggle(
            rig, lambda: "log.stopped" in kinds(read_records(first)), "the first file was closed"
        )
        closed = first.read_bytes()

        conn.receive(log_toggle(True))
        await settle_toggle(
            rig, lambda: len(log_files(log_dir)) == 2, "a second log file was opened"
        )

        files = log_files(log_dir)
        assert len(files) == 2, (
            f"criterion 17: turning logging on opens a new file; the directory holds "
            f"{[p.name for p in files]}"
        )
        second = next(p for p in files if p != first)
        assert first.read_bytes() == closed, (
            "criterion 17: the closed file is closed — the new session's records must not be "
            "appended to the log whose log.stopped already reported its size"
        )
        started = read_records(second)
        assert kinds(started)[0] == "log.started", (
            f"criterion 17: the new file opens with log.started; it opens with {kinds(started)[:3]}"
        )
        assert started[0]["seq"] == 0, (
            f"criterion 2: a new file starts a new sequence; seq={started[0]['seq']}"
        )

        fields = state_fields(log_states(conn)[-1])
        assert fields["on"] is True and fields["path"] == str(second), (
            f"criterion 17/19: the echo names the file now being written, which is what the panel "
            f"renders verbatim; it says {fields}"
        )

        # The turn that follows is recorded in the new file, which is the whole point of re-arming.
        await rig.ask("is this turn in the new file")
        await rig.speak_out()
        await settle_toggle(
            rig,
            lambda: "turn.record" in kinds(read_records(second)),
            "the turn taken after the toggle reached the new file",
        )
        assert first.read_bytes() == closed, "…and still nothing reached the closed one"


async def test_a_second_on_while_already_on_opens_no_second_file(tmp_path):
    """Criterion 17: toggling is idempotent — a second "on" changes nothing.

    The failure this pins is a session that opens a file per click: every extra file splits the
    evidence in two, and the one the ``turns.jsonl`` line names is no longer the one holding the
    turn. The echo is still sent (the page's checkbox clears its "waiting for the server" on the
    echo and on nothing else, criterion 19), and it names the same path as before.
    """
    log_dir, rig_ctx = await logging_rig(tmp_path)
    async with rig_ctx as rig:
        server = serve(rig)
        conn = hello_first(server)
        path = Path(rig.session_log.path)
        before = path.read_bytes()

        conn.receive(log_toggle(True))
        conn.receive(log_toggle(True))
        await rig.pump()
        await settle_toggle(
            rig, lambda: len(log_states(conn)) >= 2, "the two toggles were each echoed"
        )

        files = log_files(log_dir)
        assert len(files) == 1, (
            f"criterion 17: a second 'on' while already on opens no second file; the directory "
            f"holds {[p.name for p in files]}"
        )
        records = read_records(path)
        assert kinds(records).count("log.started") == 1, (
            f"criterion 17: …and writes no second log.started; the file holds "
            f"{kinds(records).count('log.started')}"
        )
        assert kinds(records).count("log.stopped") == 0, (
            "criterion 17: …and does not stop and restart the log the page never asked to stop"
        )
        assert path.read_bytes().startswith(before), (
            "criterion 17: the file already open is the file still open — its existing records "
            "were not rewritten"
        )
        fields = state_fields(log_states(conn)[-1])
        assert fields["on"] is True and fields["path"] == str(path), (
            f"criterion 17: the echo confirms the state the page asked for, naming the same "
            f"file; it says {fields}"
        )


async def test_a_log_toggle_is_not_an_assist_toggle(tmp_path):
    """Criterion 16, where it can actually be violated: the runtime, not the field list.

    "A turn's regime says what the model could see, not whether we were debugging." A session
    that routed the log flag through ``AssistState`` would publish an ``astra.assist.state``, move
    the badge, and write a different ``assists`` on the next ``turns.jsonl`` line — a bench reader
    would then read two runs of the same script as two different regimes.
    """
    log_dir, rig_ctx = await logging_rig(tmp_path, script=[answer("It is a stack trace. ")])
    async with rig_ctx as rig:
        server = serve(rig)
        conn = hello_first(server)
        badge_before = rig.session.assists.flags.badge()
        flags_before = rig.session.assists.flags.model_dump()
        assists_before = len([m for m in conn.sent() if isinstance(m, AstraAssistState)])

        conn.receive(log_toggle(False))
        await settle_toggle(
            rig,
            lambda: "log.stopped" in kinds(read_records(Path(rig.session_log.path))),
            "the log file was closed",
        )
        conn.receive(log_toggle(True))
        await settle_toggle(rig, lambda: len(log_files(log_dir)) == 2, "the log was re-opened")

        assert rig.session.assists.flags.model_dump() == flags_before, (
            "criterion 16: the log flag is not an assist — toggling it changed the assist flags "
            f"from {flags_before} to {rig.session.assists.flags.model_dump()}"
        )
        assert rig.session.assists.flags.badge() == badge_before, (
            f"criterion 16: …and not the badge ({badge_before!r} -> "
            f"{rig.session.assists.flags.badge()!r})"
        )
        assert len([m for m in conn.sent() if isinstance(m, AstraAssistState)]) == assists_before, (
            "criterion 16: …and publishes no astra.assist.state; the page's assist rows must not "
            "move because someone switched the recorder on"
        )

        await rig.ask("what regime did this turn run in")
        await rig.speak_out()
        records = rig.records()
        assert records, "premise: the turn wrote its turns.jsonl line"
        regime = records[-1].regime
        assert (
            regime.assists == badge_before and regime.assist_flags.model_dump() == flags_before
        ), (
            f"criterion 16: the turn's regime is unchanged by the log switch; it reads "
            f"assists={regime.assists!r} flags={regime.assist_flags.model_dump()}"
        )


async def test_the_hello_reply_yields_the_log_state_third(tmp_path):
    """Criterion 18: assist state, turn state, **log state**, then the journal — in that order.

    A page that reconnects starts from the server's truth, and the order is the criterion: the
    log state must arrive before the replay, so the checkbox is settled before the page is busy
    catching up on a turn. The journal's own contents and bound are T-0037's and are asserted
    unchanged here by comparing the tail against what the journal holds.
    """
    _log_dir, rig_ctx = await logging_rig(tmp_path, script=[answer("It is a stack trace. ")])
    async with rig_ctx as rig:
        server = serve(rig)
        await rig.ask("say something so there is a journal to replay")
        await rig.speak_out()

        reply = list(server.hello_reply())
        types = [getattr(m, "type", type(m).__name__) for m in reply]
        # T-0181 A4 pin move (A-R8): the live head gains astra.mic.state right after the log state
        head = t0181.hello_head(["astra.assist.state", "astra.turn.state", LOG_STATE_TYPE])
        assert types[: len(head)] == head, (
            "criterion 18: the log state is the *third* message of the reply, after the assist "
            f"state and the turn state and before the journal replay; the reply is {types}"
        )
        assert types.count(LOG_STATE_TYPE) == 1, (
            f"criterion 18: exactly one log state per reply; the reply is {types}"
        )
        assert len(types) > len(head), (
            "premise: this session published a turn, so the reply carries a replay after the "
            f"live messages; it is {types}"
        )
        # T-0213 R-5 named re-pin: the one live settings state is the reply's last message, after the replay
        replayed = t0213.hello_without_settings_state(reply)[len(head) :]
        assert all(getattr(m, "replay", False) for m in replayed), (
            "T-0037 criterion 1, unchanged: everything after the live messages is marked "
            f"replay; the reply is {types}"
        )
        fields = state_fields(reply[2])
        assert fields["on"] is True and fields["path"] == str(rig.session_log.path), (
            f"criterion 18: …and it is the server's truth — the file it is writing right now; the "
            f"reply says {fields}"
        )


async def test_a_page_that_reconnects_after_a_toggle_is_told_the_state_the_server_is_in(tmp_path):
    """Criterion 18/19 adversarial: the reply is the server's state, not the session's start state.

    The failure mode: a page reloads after someone switched logging off, is handed the state the
    session *started* in, and renders a ticked box over a session that is writing nothing. The
    user then speaks freely believing there is a record. This is the same lie criterion 19 exists
    to prevent, arriving through the reconnect path instead of through the click.
    """
    _log_dir, rig_ctx = await logging_rig(tmp_path)
    async with rig_ctx as rig:
        server = serve(rig)
        first = hello_first(server)
        path = Path(rig.session_log.path)

        first.receive(log_toggle(False))
        await settle_toggle(
            rig, lambda: "log.stopped" in kinds(read_records(path)), "the log file was closed"
        )
        first.close()

        reconnected = hello_first(server)
        states = log_states(reconnected)
        assert len(states) == 1, (
            f"criterion 18: a reconnecting page is handed exactly one log state; it got "
            f"{[state_fields(m) for m in states]}"
        )
        fields = state_fields(states[0])
        assert fields["on"] is False, (
            "criterion 18: …and it says what is true *now* — logging is off, because the last "
            f"thing that happened on this session turned it off; the page was told {fields}"
        )


async def settle_until(rig, predicate, what: str, rounds: int = 200) -> None:
    """Pump until ``predicate`` holds. Bounded, clock-free, and loud when it never does."""
    for _ in range(rounds):
        if predicate():
            return
        await rig.pump(rounds=1)
    raise AssertionError(f"T-0043: {what} did not happen within {rounds} pump rounds")


async def test_a_log_that_dies_on_a_write_tells_every_page_it_is_off(tmp_path, monkeypatch):
    """Criterion 19's safety property on the server side (rulings B8, B14, B15).

    The page-side half is pinned in ``tests/hud/log_panel.test.mjs``: the checkbox comes from the
    server's echo and never from the click. That is worth nothing if the server's echo can lie,
    and the way it lies is this one: the writer hits an ``OSError`` mid-session, disables itself
    (criterion 4), and the page that **stays** connected — the human at the desk, the normal case —
    keeps a ticked box naming a file nothing is being written to, for the rest of the session. It
    is worse than the page-side variant, because here the server knows and says nothing.

    The failure is injected on **write**, not on open: a log that cannot be opened never had a
    path and never told anyone it was recording, which is a different branch and already has its
    row (``test_session_log_writer.py``). Only a log that *was* healthy can leave a stale claim
    behind.

    Three things are asserted, and they fail independently:

    * a page connecting after the failure is told the truth (the reply is the server's state);
    * the disabling publishes **one** state, unprompted, to the page already attached — this is
      the line ruling A1 deferred to group B ("group B adds the ``astra.log.state`` publish and
      pins it") and B14 found missing;
    * ticking the box afterwards cannot leave the page showing "on": the tick may re-arm and open
      a file, but when that file dies on its first write too, the last word the page has is off.

    The counters are not asserted here (``test_turning_logging_off_closes_the_file_with_its
    _counters`` owns them); what is asserted is the claim the checkbox renders.
    """
    log_dir, rig_ctx = await logging_rig(tmp_path, script=[answer("It is a stack trace. ")])
    switch = failing_writes(monkeypatch, log_dir)
    async with rig_ctx as rig:
        server = serve(rig)
        conn = hello_first(server)
        path = Path(rig.session_log.path)

        await rig.ask("record this while the disk still has room")
        await rig.speak_out()
        healthy = read_records(path)
        assert "log.started" in kinds(healthy) and len(healthy) > 1, (
            f"premise: the log opened and was writing — this row is about the *write* branch, so "
            f"the open must have worked; the file holds {kinds(healthy)[:4]}"
        )
        assert rig.session_log.stats()["error"] is None, (
            f"premise: …and it is healthy up to here; stats()={dict(rig.session_log.stats())!r}"
        )
        seen = len(log_states(conn))

        switch.arm()  # the disk fills, mid-session, with a page attached and a turn under way
        conn.receive(encode(astra_frame("z")))
        await rig.ask("and now there is nowhere to put it")
        await rig.speak_out()
        await settle_until(
            rig,
            lambda: rig.session_log.stats()["error"] is not None,
            "the writer's OSError disabled the log (criterion 4)",
        )
        assert switch.refused, (
            "premise: the injection bit — no write was refused, so the log disabled itself for "
            "some other reason and this row proves nothing"
        )

        # Criterion 4, on the write branch: the session loses its log and nothing else.
        records = rig.records()
        assert records and records[-1].session_log is None, (
            "criterion 4/14: the turn under way still completes and still writes its line, with "
            f"session_log null because logging was off for it; the line says "
            f"{records[-1].session_log if records else '<no line at all>'!r}"
        )

        # 1. The reconnect path — the server's truth, for a page that arrives after the failure.
        arriving = log_states(hello_first(server))
        assert len(arriving) == 1, (
            f"criterion 18: one log state per reply; a page arriving after the failure got "
            f"{[state_fields(m) for m in arriving]}"
        )
        fields = state_fields(arriving[0])
        assert fields["on"] is False, (
            "ruling B8: `on` means 'this session is writing a file', and this one is not — its "
            f"writer failed. The page was told {fields}"
        )
        assert fields["path"] is None, (
            "ruling B8: …and an off state names no file. The panel renders `path` verbatim beside "
            f"the checkbox, so a path here is a file the user is told holds their session; {fields}"
        )
        assert isinstance(fields["error"], str) and fields["error"], (
            f"criterion 16/ruling B8: …off *with the reason*; the page was told {fields}"
        )
        reason = fields["error"].lower()
        assert "no space left" in reason or "errno 28" in reason, (
            "ruling B8: the reason is why the log stopped, not that it stopped: the OSError this "
            f"row injected was ENOSPC/'No space left on device' and the page was told {fields}"
        )

        # 2. The page that never went away: it is told, unprompted. This is B14's blocker.
        unprompted = log_states(conn)[seen:]
        assert unprompted, (
            "rulings A1/B14/B15: a log that disables itself publishes one astra.log.state to "
            "every connection. The page that stayed attached was told nothing, so its checkbox "
            f"still says on={state_fields(log_states(conn)[-1]) if log_states(conn) else 'nothing'}"
            " over a session that has been recording nothing since the write failed. Nobody sent "
            "a toggle: the publish is owed by the disabling itself"
        )
        assert len(unprompted) == 1, (
            f"ruling B2's shape, on this path too: one state per event, not a stream — a panel "
            f"that repaints per failed write is its own defect; got "
            f"{[state_fields(m) for m in unprompted]}"
        )
        told = state_fields(unprompted[0])
        assert (told["on"], told["path"]) == (False, None) and told["error"], (
            f"…and it says the same thing the reconnect reply says: off, no file, with the "
            f"reason; the attached page was told {told}"
        )

        # 3. Ticking the box cannot talk the page back into "on" over a disk that is still full.
        before_tick = len(log_states(conn))
        conn.receive(log_toggle(True))
        await settle_until(
            rig,
            lambda: len(log_states(conn)) > before_tick,
            "the routed toggle was echoed (ruling B2: every toggle gets an answer)",
        )
        await rig.pump(rounds=60)  # let whatever the tick opened try to write, and be told
        last = state_fields(log_states(conn)[-1])
        assert last["on"] is False and last["path"] is None, (
            "ruling B8: the disk is still full, so there is still no file — a tick that leaves "
            "the checkbox ticked is exactly the lie criterion 19 exists to prevent, arriving "
            f"through the click instead of through the failure. The last word the page has is "
            f"{last}, over {[p.name for p in log_files(log_dir)]}"
        )
        assert last["error"], f"…and the page is still told why; it was told {last}"
