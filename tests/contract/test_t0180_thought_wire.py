"""T-0180 group A: ``astra.thought`` on the wire — the contract, the codec, the real endpoint, the fake.

docs/tasks/T-0180.md A1 (the strict model), A2 (the Python half of the shared corpus; the page half is
``tests/hud/protocol.test.mjs``), A3 (``tests/fakes/hud_server.thought_script``), A6 (size) and A7
(outbox). The journal (A4), the compact ``wire.out`` row (A5) and the BASE byte identity (A8) are in
``tests/unit/test_t0180_journal.py``.

THE SEAM THIS TASK ADDS (names are the contract with the implementer):

    contracts/protocol.py
        THOUGHT_MAX_CHARS = 16000
        AstraThought(type="astra.thought", turn_id, request_id, attempt_id, step, kind, text,
                     omitted_chars, end, replay=False)
        in AstraMessage, ASTRA_MESSAGE_TYPES and SERVER_TO_CLIENT only.

Reading recorded here (the task lists ``replay: bool = False`` with a default and every other field
without one): every field but ``replay`` is required. Expected values are the task's literals, written
out by hand; nothing is read back from the model under test to build an expectation.

Regime: CPU, no socket (``HudServer.open()``), ``ScriptedClock``, no measurement, n = 1 per row.
"""

from __future__ import annotations

import json
import typing
from typing import Any

import pytest
from pydantic import ValidationError

from local_astra.contracts import protocol
from local_astra.contracts.codec import MAX_MESSAGE_BYTES, Malformed, decode, encode
from local_astra.contracts.protocol import (
    AstraCanvasStep,
    AstraError,
    AstraHello,
    AstraSpoken,
    AstraTurnState,
)
from local_astra.contracts.turn import TurnState
from tests.fakes.hud_server import thought_script
from tests.unit._hud_rig import make_rig, require_symbol
from tests.unit.test_hud_constants_parity import corpus_cases, raw_of

pytestmark = [pytest.mark.contract]

TASK = "T-0180"
CAP = 16000  # H-T180-5, the task's literal
THOUGHT = "astra.thought"
KINDS = {"step", "reask", "retry", "repair", "reground"}
ENDS = {"open", "done", "stopped_repeating", "interrupted", "error"}
FIELDS = {
    "type",
    "turn_id",
    "request_id",
    "attempt_id",
    "step",
    "kind",
    "text",
    "omitted_chars",
    "end",
    "replay",
}
REQUIRED = FIELDS - {"type", "replay"}
ASTRAL = "\U0001f600"  # one code point, two UTF-16 units, four UTF-8 bytes

OK: dict[str, Any] = {
    "type": THOUGHT,
    "turn_id": "t4",
    "request_id": "chatcmpl-s1-t4-r1",
    "attempt_id": "t4-a1",
    "step": 1,
    "kind": "step",
    "text": "The page shows a search box.",
    "omitted_chars": 0,
    "end": "open",
}


def thought_model() -> Any:
    model = getattr(protocol, "AstraThought", None)
    assert model is not None, (
        "T-0180 A1: contracts/protocol.py gains AstraThought(type='astra.thought', turn_id, "
        "request_id, attempt_id, step, kind, text, omitted_chars, end, replay=False)"
    )
    return model


def thought(**over: Any) -> Any:
    fields = {k: v for k, v in {**OK, **over}.items() if k != "type"}
    return thought_model()(**fields)


def raw(**over: Any) -> str:
    return json.dumps({**OK, **over}, ensure_ascii=False)


class _NullSink:
    def ingest(self, message: Any) -> None:
        return None


def make_server(tmp_path: Any, **kwargs: Any) -> Any:
    rig = make_rig(tmp_path)
    return require_symbol("HudServer")(rig.session, frame_sink=_NullSink(), port=0, **kwargs)


def greet(server: Any) -> Any:
    connection = server.open()
    connection.receive(encode(AstraHello()))
    return connection


# --- A1: the model ------------------------------------------------------------------------------


def test_a1_the_thought_is_a_server_to_client_type_in_every_inventory():
    model = thought_model()
    assert model in protocol.ASTRA_MESSAGE_TYPES, "A1: ASTRA_MESSAGE_TYPES gains AstraThought"
    assert THOUGHT in protocol.SERVER_TO_CLIENT, "A1: server->client"
    assert THOUGHT not in protocol.CLIENT_TO_SERVER, "A1: and only that direction"
    got = decode(raw())
    assert isinstance(got, model), f"A1: AstraMessage decodes it as AstraThought; got {got!r}"
    assert got.replay is False, "A1: replay defaults to False (a live message)"
    assert getattr(protocol, "THOUGHT_MAX_CHARS", None) == CAP, "A6: THOUGHT_MAX_CHARS = 16000"


def test_a1_the_field_set_its_required_fields_and_its_two_literals():
    model = thought_model()
    assert set(model.model_fields) == FIELDS, sorted(model.model_fields)
    required = {n for n, f in model.model_fields.items() if f.is_required()}
    assert required == REQUIRED, f"A1: every field but replay is required; got {sorted(required)}"
    assert model.model_fields["replay"].default is False

    def literals(name: str) -> set[str]:
        annotation = model.model_fields[name].annotation
        assert typing.get_origin(annotation) is typing.Literal, f"{name}: {annotation!r}"
        return {str(v) for v in typing.get_args(annotation)}

    assert literals("kind") == KINDS
    assert literals("end") == ENDS


REFUSED: list[tuple[str, dict[str, Any] | None, str | None]] = [
    ("empty turn_id", {"turn_id": ""}, None),
    ("empty request_id", {"request_id": ""}, None),
    ("empty attempt_id", {"attempt_id": ""}, None),
    ("step 0", {"step": 0}, None),
    ("step -1", {"step": -1}, None),
    ("unknown kind", {"kind": "pass2"}, None),
    ("kind as a label", {"kind": "Step"}, None),
    ("unknown end", {"end": "aborted"}, None),
    ("end done-ish", {"end": "stopped"}, None),
    ("negative omitted_chars", {"omitted_chars": -1}, None),
    ("one over the cap", {"text": "x" * (CAP + 1)}, None),
    ("one astral over the cap", {"text": ASTRAL * (CAP + 1)}, None),
    ("omitted with text one below the cap", {"text": "x" * (CAP - 1), "omitted_chars": 1}, None),
    ("omitted with short text", {"omitted_chars": 5}, None),
    ("an extra field", {"segment": "thought"}, None),
    ("a text_preview field", {"text_preview": "draft"}, None),
    *[(f"missing {name}", None, name) for name in sorted(REQUIRED)],
]


@pytest.mark.parametrize(("why", "over", "missing"), REFUSED, ids=[r[0] for r in REFUSED])
def test_a1_refused(why, over, missing):
    """A1: each of these is ``Malformed`` through the real codec and a ``ValidationError`` from the
    model itself. Each row is one rule; the accepted twin of each boundary is the next test."""
    model = thought_model()
    payload = {**OK, **(over or {})}
    if missing is not None:
        payload.pop(missing)
    got = decode(json.dumps(payload, ensure_ascii=False))
    assert isinstance(got, Malformed) and got.type == THOUGHT, f"A1 {why}: decode said {got!r}"
    with pytest.raises(ValidationError):
        model.model_validate({k: v for k, v in payload.items() if k != "type"})


ACCEPTED: list[tuple[str, dict[str, Any]]] = [
    ("step 1", {"step": 1}),
    ("step 12", {"step": 12}),
    ("text exactly at the cap", {"text": "x" * CAP}),
    ("at the cap with omitted_chars", {"text": "x" * CAP, "omitted_chars": 4000}),
    ("astral at the cap with omitted_chars", {"text": ASTRAL * CAP, "omitted_chars": 7}),
    ("8001 astral code points (16002 UTF-16 units)", {"text": ASTRAL * 8001}),
    ("replay true", {"replay": True}),
    ("markup and a channel token are data", {"text": "<|channel>thought\nx</script><b>y"}),
    *[(f"kind {k}", {"kind": k}) for k in sorted(KINDS)],
    *[(f"end {e}", {"end": e}) for e in sorted(ENDS)],
]


@pytest.mark.parametrize(("why", "over"), ACCEPTED, ids=[a[0] for a in ACCEPTED])
def test_a1_accepted_and_the_codec_round_trips_it(why, over):
    """A1 boundaries (the too-strict direction) and the codec round trip: ``decode(encode(m)) == m``,
    from ``str`` and from ``bytes``, with the text unchanged code point for code point."""
    model = thought_model()
    got = decode(raw(**over))
    assert isinstance(got, model), f"A1 {why}: decode said {got!r}"
    assert got.text == {**OK, **over}["text"], f"A1 {why}: the text is carried unchanged"
    wire = encode(got)
    assert decode(wire) == got and decode(wire.encode("utf-8")) == got, f"round trip: {why}"


# --- A2: the shared corpus, Python half -----------------------------------------------------------


def _thought_rows() -> list[dict[str, Any]]:
    return [c for c in corpus_cases() if str(c["name"]).startswith("thought_")]


def test_a2_the_thought_corpus_rows_are_the_tasks_list_and_carry_the_cap():
    """A2 fixture premise (green before the implementation): the rows both halves read, and the build
    rows sit on 16000 code points. ``test_hud_constants_parity`` checks each verdict with the codec."""
    rows = {c["name"]: c for c in _thought_rows()}
    assert sorted(rows) == [
        "thought_at_the_cap_in_astral_code_points",
        "thought_at_the_cap_with_omitted_chars",
        "thought_done_replayed",
        "thought_empty_attempt_id_is_malformed",
        "thought_extra_field_is_malformed",
        "thought_from_the_client_is_the_wrong_direction",
        "thought_markup_and_channel_tokens_stay_data",
        "thought_omitted_without_the_cap_is_malformed",
        "thought_open",
        "thought_over_the_cap_is_malformed",
        "thought_step_zero_is_malformed",
        "thought_unknown_end_is_malformed",
        "thought_unknown_kind_is_malformed",
    ]
    built = {
        n: (c["build"]["char"], c["build"]["count"], c["expect"])
        for n, c in rows.items()
        if "build" in c
    }
    assert built == {
        "thought_at_the_cap_with_omitted_chars": ("x", 16000, "astra"),
        "thought_over_the_cap_is_malformed": ("x", 16001, "malformed"),
        "thought_at_the_cap_in_astral_code_points": (ASTRAL, 16000, "astra"),
    }
    assert (
        json.loads(raw_of(rows["thought_at_the_cap_with_omitted_chars"]))["omitted_chars"] == 4000
    )
    assert all(c.get("type") == THOUGHT for c in rows.values())


def test_a1_a_thought_sent_by_a_client_is_refused_as_wrong_direction_by_the_real_route(tmp_path):
    """A1/A2: the corpus row ``thought_from_the_client_is_the_wrong_direction`` sent by a greeted page
    to the real ``HudServer``: counted ``wrong_direction``, answered ``astra.error(wrong_direction)``,
    never routed."""
    server = make_server(tmp_path)
    connection = greet(server)
    row = next(c for c in _thought_rows() if c.get("direction_verdict"))
    assert row["direction_verdict"] == "wrong_direction"
    before = dict(server.counts)
    connection.receive(raw_of(row))
    assert server.counts["wrong_direction"] == before["wrong_direction"] + 1, server.counts
    assert server.counts["routed"] == before["routed"], "A1: never routed"
    assert server.counts["unhandled"] == before["unhandled"], "A1: a known type, not an unknown one"
    errors = [m for m in connection.sent() if isinstance(m, AstraError)]
    assert [e.code for e in errors] == ["wrong_direction"], errors


# --- A6: size ------------------------------------------------------------------------------------


def test_a6_the_worst_escaping_thought_at_the_cap_encodes_under_the_message_limit():
    """A6: 16000 x U+0001 is the most bytes a carried text can cost (``\\u0001``, six bytes each)."""
    message = thought(text="\x01" * CAP, omitted_chars=123, end="done")
    wire = encode(message)
    assert wire.count("\\u0001") == CAP, "premise: every character is escaped on the wire"
    size = len(wire.encode("utf-8"))
    assert 6 * CAP < size < MAX_MESSAGE_BYTES, size
    assert decode(wire) == message


def test_a6_sixteen_worst_case_thoughts_fit_in_one_hello_reply(tmp_path):
    """A6: 16 attempts, each at the cap in U+0001, published through the real server: a greeting page
    gets all 16, in order, ``replay=True``, texts intact, no truncation notice, and its connection is
    not closed by the reply."""
    model = thought_model()
    server = make_server(tmp_path)
    published = [
        thought(
            attempt_id=f"t4-a{i}",
            request_id=f"r{i}",
            step=i,
            text="\x01" * CAP,
            omitted_chars=9,
            end="done",
        )
        for i in range(1, 17)
    ]
    for message in published:
        server.publish(message)
    connection = greet(server)
    reply = connection.sent()
    got = [m for m in reply if isinstance(m, model)]
    assert [m.attempt_id for m in got] == [f"t4-a{i}" for i in range(1, 17)]
    assert all(m.replay is True for m in got)
    assert [m.model_copy(update={"replay": False}) for m in got] == published
    assert not [m for m in reply if isinstance(m, AstraError)], "A6: nothing truncated"
    assert not connection.closed, connection.close_reason
    assert all(len(encode(m).encode("utf-8")) < MAX_MESSAGE_BYTES for m in reply)


# --- A7: the outbox -----------------------------------------------------------------------------


def _canvas() -> AstraCanvasStep:
    return AstraCanvasStep(
        request_id="chatcmpl-s1-t4-r1",
        canvas_idx=0,
        step=1,
        max_steps=4,
        canvas_length=1,
        canvas=(1,),
        argmax=(1,),
        stable="00",
    )


def _spoken(n: int) -> AstraSpoken:
    return AstraSpoken(turn_id="t4", utterance_id=f"u{n}", text=f"Line {n}.")


def test_a7_a_thought_overflowing_a_full_outbox_closes_the_connection_not_the_thought(tmp_path):
    """A7: non-droppable. An outbox of two spoken lines refuses the thought and the connection is
    closed with a reason naming it, rather than the thought silently dropped."""
    thought_model()
    outbox = require_symbol("Outbox")(2)
    assert outbox.offer(_spoken(1)) and outbox.offer(_spoken(2))
    assert outbox.offer(thought()) is False, "A7: a full outbox refuses a thought (never drops it)"
    server = make_server(tmp_path, outbox_size=2)
    connection = greet(server)
    connection.take()  # the hello reply went out
    server.publish(_spoken(1))
    server.publish(_spoken(2))
    server.publish(thought())
    assert connection.closed and THOUGHT in (connection.close_reason or ""), connection.close_reason


def test_a7_a_thought_evicts_a_canvas_frame_and_a_turn_state_never_evicts_a_thought():
    """A7, both directions: the thought is not a canvas message (a queued denoise frame goes for it),
    and the droppable turn state is what goes when the queue is all thoughts."""
    model = thought_model()
    outbox = require_symbol("Outbox")(2)
    assert outbox.offer(_canvas()) and outbox.offer(_spoken(1))
    assert outbox.offer(thought()) is True
    assert [m.type for m in outbox.pending()] == ["astra.spoken", THOUGHT]
    full = require_symbol("Outbox")(2)
    assert full.offer(thought(attempt_id="a1")) and full.offer(thought(attempt_id="a2"))
    assert full.offer(AstraTurnState(state=TurnState.THINKING, turn_id="t4")) is True
    assert [type(m) for m in full.pending()] == [model, model], "A7: no thought is evicted"
    assert full.dropped == 1, "the turn state was the message that went"


# --- A3: the fake and the real endpoint agree ----------------------------------------------------


def test_a3_the_fake_builds_what_the_real_endpoint_replays(tmp_path):
    """A3: publish ``thought_script(progressive=True)`` (open, then final, per attempt) through the
    real server; the thoughts of a greeting page's reply equal ``thought_script(replay=True)`` — the
    latest per attempt, in first-seen order, marked as replays."""
    model = thought_model()
    server = make_server(tmp_path)
    live = thought_script("t4", progressive=True)
    assert [m.end for m in live] == ["open", "done", "open", "done", "open"], "premise: the builder"
    for message in live:
        server.publish(message)
    got = [m for m in greet(server).sent() if isinstance(m, model)]
    assert got == thought_script("t4", replay=True), [(m.attempt_id, m.end) for m in got]
