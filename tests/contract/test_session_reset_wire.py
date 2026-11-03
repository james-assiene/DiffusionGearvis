"""T-0162 group A, criterion A-1: the two wire types of the session reset, through the real codec.

- ``AstraSessionReset``: ``astra.session.reset``, client->server, no fields;
- ``AstraSessionCleared``: ``astra.session.cleared``, server->client, ``resets: int >= 1``,
  ``ended_turn_id: str | None``, ``memory_forgotten: int >= 0``, ``replay: bool = False``.

Both join ``ASTRA_MESSAGE_TYPES`` and their direction set; unknown fields are malformed (Contract is
``extra="forbid"``). Every verdict below is the real ``codec.decode``'s, on JSON written out here by
hand, never a model this module constructed and then compared with itself.

``ended_turn_id`` must default to ``None``: ``codec.encode`` drops ``None`` fields, so a reset from
IDLE goes on the wire without the key and must still decode (the task names no default; this is the
only reading under which the codec round-trips its own output -- recorded in the test-author's
report). No fake or real service is involved: this is the wire contract only, so there is no
``backend`` parametrisation (as in ``test_protocol_codec.py``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_astra.contracts import protocol
from local_astra.contracts.codec import Malformed, decode, encode

RESET = "astra.session.reset"
CLEARED = "astra.session.cleared"


def model(name: str) -> Any:
    cls = getattr(protocol, name, None)
    assert cls is not None, f"T-0162 A-1: contracts/protocol.py gains {name}"
    return cls


def wire(**fields: Any) -> str:
    return json.dumps(fields)


def test_both_types_are_declared_with_their_direction():
    reset, cleared = model("AstraSessionReset"), model("AstraSessionCleared")
    assert reset.model_fields["type"].default == RESET
    assert cleared.model_fields["type"].default == CLEARED
    assert set(reset.model_fields) == {"type"}, "A-1: astra.session.reset has no fields"
    assert set(cleared.model_fields) == {
        "type",
        "resets",
        "ended_turn_id",
        "memory_forgotten",
        "replay",
    }, sorted(cleared.model_fields)
    assert reset in protocol.ASTRA_MESSAGE_TYPES and cleared in protocol.ASTRA_MESSAGE_TYPES
    assert RESET in protocol.CLIENT_TO_SERVER and RESET not in protocol.SERVER_TO_CLIENT
    assert CLEARED in protocol.SERVER_TO_CLIENT and CLEARED not in protocol.CLIENT_TO_SERVER


def test_the_reset_decodes_and_refuses_any_field():
    got = decode(wire(type=RESET))
    assert isinstance(got, model("AstraSessionReset")), got
    assert decode(encode(got)) == got
    for extra in ({"resets": 1}, {"memory": "keep"}, {"replay": False}):
        assert isinstance(decode(wire(type=RESET, **extra)), Malformed), (
            f"A-1: astra.session.reset has no fields, so {extra} is malformed"
        )


def test_the_cleared_message_round_trips_with_every_field():
    cleared = model("AstraSessionCleared")
    raw = wire(type=CLEARED, resets=2, ended_turn_id="t7", memory_forgotten=13, replay=True)
    got = decode(raw)
    assert isinstance(got, cleared), got
    assert (got.resets, got.ended_turn_id, got.memory_forgotten, got.replay) == (2, "t7", 13, True)
    assert decode(encode(got)) == got
    assert json.loads(encode(got)) == json.loads(raw), "every field goes back on the wire as sent"


def test_a_reset_from_idle_goes_on_the_wire_without_ended_turn_id():
    got = decode(wire(type=CLEARED, resets=1, memory_forgotten=0))
    assert isinstance(got, model("AstraSessionCleared")), (
        f"A-1: ended_turn_id is None when no turn was open, and encode drops None; got {got}"
    )
    assert got.ended_turn_id is None and got.replay is False, "replay defaults to False"
    assert "ended_turn_id" not in json.loads(encode(got))
    assert decode(encode(got)) == got
    explicit = decode(wire(type=CLEARED, resets=1, ended_turn_id=None, memory_forgotten=0))
    assert explicit == got, "an explicit null is the same message"


@pytest.mark.parametrize(
    ("field", "value", "why"),
    [
        ("resets", 0, "resets >= 1: a cleared message reports at least the reset it echoes"),
        ("resets", -1, "a negative count"),
        ("resets", 1.5, "a fractional count"),
        ("resets", None, "a null count"),
        ("memory_forgotten", -1, "memory_forgotten >= 0"),
        ("memory_forgotten", 2.5, "a fractional episode count"),
        ("ended_turn_id", 7, "a numeric turn id"),
        ("ended_turn_id", ["t1"], "a list turn id"),
    ],
)
def test_the_cleared_message_refuses(field, value, why):
    model("AstraSessionCleared")
    good = {"type": CLEARED, "resets": 1, "ended_turn_id": "t1", "memory_forgotten": 3}
    got = decode(json.dumps({**good, field: value}))
    assert isinstance(got, Malformed), f"A-1: {why}: {field}={value!r} decoded as {got!r}"


def test_the_cleared_message_refuses_unknown_fields_and_needs_its_counts():
    model("AstraSessionCleared")
    good = {"type": CLEARED, "resets": 1, "memory_forgotten": 3}
    for extra in ({"history": []}, {"session_id": "abc"}, {"final": True}):
        assert isinstance(decode(json.dumps({**good, **extra})), Malformed), (
            f"A-1: unknown fields are malformed: {extra}"
        )
    for key in ("resets", "memory_forgotten"):
        partial = {k: v for k, v in good.items() if k != key}
        assert isinstance(decode(json.dumps(partial)), Malformed), (
            f"A-1 names no default for {key}: a cleared message without it is malformed"
        )
