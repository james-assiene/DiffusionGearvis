"""HUD protocol codec: every ``astra.*`` message round-trips; decode is total (docs/03 §3)."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from local_astra.contracts import codec
from local_astra.contracts.codec import (
    MAX_MESSAGE_BYTES,
    Malformed,
    Unknown,
    decode,
    encode,
)
from local_astra.contracts.protocol import (
    ASTRA_MESSAGE_TYPES,
    CLIENT_TO_SERVER,
    SERVER_TO_CLIENT,
    AstraFrame,
    AstraTranscript,
    RealtimeEvent,
)
from tests.unit._examples import example_of

# The payloads of T-0027 criteria 1, 2 and 4 are built from the *shared* corpus
# (tests/fixtures/hud/decode_corpus.jsonl), by the same builder tests/hud/protocol.test.mjs uses,
# so the page and the codec are held to one fixture and not to two copies of a construction.
from tests.unit.test_hud_constants_parity import PREFIX, SUFFIX, corpus_raw


@pytest.mark.parametrize("model", ASTRA_MESSAGE_TYPES, ids=lambda m: m.__name__)
def test_every_astra_message_round_trips(model):
    ex = example_of(model)
    wire = encode(ex)
    back = decode(wire)
    assert back == ex
    assert decode(wire.encode()) == ex  # bytes input too


def test_every_astra_message_has_a_direction():
    tags = {m.model_fields["type"].default for m in ASTRA_MESSAGE_TYPES}
    assert tags == CLIENT_TO_SERVER | SERVER_TO_CLIENT
    assert not (CLIENT_TO_SERVER & SERVER_TO_CLIENT)


def test_realtime_events_pass_through_unchanged():
    raw = json.dumps({"type": "session.update", "session": {"voice": "x"}, "event_id": "e1"})
    ev = decode(raw)
    assert isinstance(ev, RealtimeEvent)
    assert ev.type == "session.update"
    assert ev.payload == {"session": {"voice": "x"}, "event_id": "e1"}
    assert json.loads(encode(ev)) == json.loads(raw)


def test_unknown_astra_type_is_ignored_not_an_error():
    assert decode('{"type": "astra.future.thing", "x": 1}') == Unknown(type="astra.future.thing")


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ("", "invalid json"),
        ("{", "invalid json"),
        ("[1,2]", "object"),
        ('{"no_type": 1}', "type"),
        ('{"type": 5}', "type"),
        ('{"type": "astra.frame"}', "source"),  # required fields missing → path in error
        (
            '{"type": "astra.frame", "source": "screen", "ts_ms": -1, "width": 1, "height": 1, "jpeg_b64": "a"}',
            "ts_ms",
        ),
        (
            '{"type": "astra.canvas.step", "request_id": "r", "canvas_idx": 0, "step": 0, "max_steps": 1, "canvas_length": 1, "argmax": [1], "accepted": "zz", "entropy": [0.0]}',
            "accepted",
        ),
        ('{"type": "Session.Update"}', "type"),  # Realtime types are lower-case
    ],
)
def test_malformed_inputs_produce_malformed_with_a_reason(raw, fragment):
    out = decode(raw)
    assert isinstance(out, Malformed), out
    assert fragment in out.error


def test_oversize_message_is_rejected_before_parsing():
    big = b'{"type": "astra.hello", "pad": "' + b"a" * MAX_MESSAGE_BYTES + b'"}'
    out = decode(big)
    assert isinstance(out, Malformed) and "exceeds" in out.error


def test_invalid_utf8_bytes_are_malformed():
    out = decode(b'{"type": "astra.hello"\xff}')
    assert isinstance(out, Malformed)


@settings(max_examples=300, deadline=None)
@given(st.binary(max_size=512))
def test_decode_never_raises_on_arbitrary_bytes(data):
    decode(data)


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=512))
def test_decode_never_raises_on_arbitrary_text(text):
    decode(text)


@settings(max_examples=200, deadline=None)
@given(
    st.dictionaries(
        st.text(min_size=1, max_size=8),
        st.recursive(
            st.none() | st.booleans() | st.integers() | st.text(max_size=8),
            lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=4), c, max_size=3),
            max_leaves=6,
        ),
        max_size=6,
    )
)
def test_decode_never_raises_on_arbitrary_json_objects(obj):
    decode(json.dumps(obj))


def test_frame_message_shape_matches_v1_perception_source():
    """docs/12 §3: JPEG at ≤ 1 fps over the WebSocket, phash optional (computed server-side)."""
    m = AstraFrame(source="screen", ts_ms=10, width=1280, height=800, jpeg_b64="/9j/")
    d = json.loads(encode(m))
    assert set(d) == {"type", "source", "ts_ms", "width", "height", "jpeg_b64"}
    assert d["source"] == "screen"


# The two canvas-step tests that used to live here were written against the pre-T-0011 shape of
# AstraCanvasStep (`accepted` and `entropy` required, no `canvas`/`stable`/`phase`). T-0011
# criterion 6 amends that contract -- v0.29.0 exposes neither an acceptance mask nor a
# per-position entropy -- so they are replaced, not edited, by
# tests/contract/test_canvas_step_wire.py::test_amended_canvas_step_carries_the_real_fields and
# ::test_canvas_step_rejects_an_inconsistent_frame, which cover the same validator plus the new
# fields. Deleted here by the T-0011 test-author rather than left to fail in an implementer's
# hands, since the implementer may not edit tests/ (CLAUDE.md).


# --- T-0027: the inbound limit counts UTF-8 bytes ------------------------------------------------
#
# ``hud/src/protocol.js::overTheLimit`` counts bytes (T-0008 ruling R11); ``codec.decode`` counted
# code points, so the two disagreed about the same 8 MiB envelope. Every payload below comes from
# the shared corpus, and every one asserts its *premise* (how many code points, how many bytes)
# separately from its outcome — "malformed because I built it malformed" would be vacuous
# (.claude/rules/tests.md).


def test_a_multibyte_message_over_the_byte_limit_is_malformed():
    """Criterion 1: 3,000,051 code points but 9,000,051 UTF-8 bytes — over the limit the socket
    really carried, and the page already refuses it (T-0008 R11)."""
    raw = corpus_raw("multibyte_oversized_euro")
    assert len(raw) == 3_000_051, "premise: 3,000,000 euro signs plus a 51-byte ASCII envelope"
    assert len(raw.encode("utf-8")) == 9_000_051, "premise: 3 UTF-8 bytes per euro sign"
    assert len(raw) < MAX_MESSAGE_BYTES < len(raw.encode("utf-8")), (
        "premise: the two ways of counting disagree about this message, which is the whole point"
    )
    out = decode(raw)
    assert isinstance(out, Malformed), (
        "criterion 1: the limit is bytes, so a message of 9,000,051 bytes is Malformed; "
        f"decode returned {type(out).__name__}"
    )
    assert "exceeds" in out.error


def test_an_astral_payload_is_measured_in_bytes_not_in_four_byte_guesses():
    """Criterion 2 (ruling R2): the page may skip the encode when ``length * 3 <= MAX`` because
    its unit is a UTF-16 unit; Python's unit is a code point, where the bound is 4 bytes. A port
    of the page's constant passes criterion 1 and fails here."""
    raw = corpus_raw("astral_oversized_defeats_a_times_three_shortcut")
    assert len(raw) == 2_500_051, "premise: 2,500,000 astral characters plus the 51-byte envelope"
    assert len(raw.encode("utf-8")) == 10_000_051, "premise: 4 UTF-8 bytes per astral character"
    assert len(raw) * 3 == 7_500_153 <= MAX_MESSAGE_BYTES, (
        "premise (R2): a `len(raw) * 3 <= MAX -> accept` short-circuit would wave this through"
    )
    out = decode(raw)
    assert isinstance(out, Malformed), (
        "criterion 2: 10,000,051 bytes is over the limit however cheaply you try to guess it; "
        f"decode returned {type(out).__name__}"
    )
    assert "exceeds" in out.error


def test_the_size_check_runs_before_json_loads(monkeypatch):
    """Criterion 3: an oversized message is refused, not parsed — the page's already-tested
    ordering ("the size check comes before JSON.parse", tests/hud/protocol.test.mjs).

    ``monkeypatch`` restores ``json.loads`` whatever the test does; the recorder measures the
    *bytes* of each parsed argument, because bytes is the quantity the limit is about.
    """
    payloads = [
        corpus_raw(name)  # built before the spy goes in: reading the corpus parses JSON itself
        for name in (
            "multibyte_oversized_euro",
            "astral_oversized_defeats_a_times_three_shortcut",
            "ascii_one_byte_over_the_limit",
        )
    ]
    seen: list[int] = []
    real_loads = json.loads

    def recorder(s, *args, **kwargs):
        seen.append(len(s.encode("utf-8", "surrogatepass")) if isinstance(s, str) else len(s))
        return real_loads(s, *args, **kwargs)

    monkeypatch.setattr(codec.json, "loads", recorder)
    try:
        # Outcomes are criteria 1, 2 and 4b's business; this test asserts only the *ordering*.
        for raw in payloads:
            decode(raw)
    finally:
        monkeypatch.undo()
    assert seen == [] or max(seen) <= MAX_MESSAGE_BYTES, (
        f"criterion 3: json.loads was handed {max(seen)} bytes, over the "
        f"{MAX_MESSAGE_BYTES}-byte limit: the size check must come first"
    )


@pytest.mark.parametrize("as_bytes", [False, True], ids=["str", "bytes"])
def test_the_byte_limit_boundary_is_exclusive(as_bytes):
    """Criterion 4 (ruling R6): exactly ``MAX_MESSAGE_BYTES`` is accepted, one byte more is not.

    Both forms of the same payload, because the ``bytes`` branch already had the right rule and
    a rewrite of the ``str`` branch must not move it. 49-byte prefix + 8,388,557 "x" + 2-byte
    suffix = 8,388,608 bytes exactly (hand arithmetic).
    """
    at_limit = corpus_raw("ascii_exactly_at_the_limit")
    over = corpus_raw("ascii_one_byte_over_the_limit")
    assert len(at_limit.encode("utf-8")) == MAX_MESSAGE_BYTES, "premise: exactly at the limit"
    assert len(over.encode("utf-8")) == MAX_MESSAGE_BYTES + 1, "premise: one byte over"

    ok = decode(at_limit.encode("utf-8") if as_bytes else at_limit)
    assert isinstance(ok, AstraTranscript), (
        f"criterion 4a: a message of exactly {MAX_MESSAGE_BYTES} bytes is accepted (the "
        f"comparison is `>`, not `>=`); decode returned {type(ok).__name__}"
    )
    assert len(ok.text) == 8_388_557 and ok.turn_id == "t"

    bad = decode(over.encode("utf-8") if as_bytes else over)
    assert isinstance(bad, Malformed), "criterion 4b: one byte over the limit is Malformed"
    assert "exceeds" in bad.error


def test_a_string_that_cannot_be_encoded_as_utf8_is_malformed_not_an_exception():
    """Criterion 5 (ruling R3): a ``str`` of lone surrogates has no UTF-8 encoding, so a byte
    guard can raise where a character count could not. ``decode`` is total (module docstring,
    docs/03 §3): the outcome is ``Malformed`` with the bytes branch's wording."""
    raw = PREFIX + "\ud800" * 3_000_000 + SUFFIX
    assert len(raw) == 3_000_051, "premise: the same code-point count as the euro payload"
    with pytest.raises(UnicodeEncodeError):
        raw.encode("utf-8")  # premise: this string has no UTF-8 encoding at all
    try:
        out = decode(raw)
    except Exception as e:  # a raise is the failure this test exists for
        raise AssertionError(
            f"criterion 5: decode must never raise, whatever arrives on the socket; got {e!r}"
        ) from e
    assert isinstance(out, Malformed), (
        f"criterion 5: an unencodable str is Malformed, not accepted; got {type(out).__name__}"
    )
    assert "not utf-8" in out.error, out.error


def test_a_short_unencodable_string_keeps_todays_outcome():
    """Criterion 6, the negative twin: a *short* transcript carrying a lone surrogate is nowhere
    near the limit on any counting, so the limit must not touch it.

    ``AstraTranscript.text`` is a ``str`` and the contract asks nothing of its encodability, so
    this message is a valid ``astra.transcript`` — as it is today. The obvious fix
    (``len(raw.encode()) > MAX``) turns every message into an encode and this one into
    ``Malformed``, changing behaviour for traffic that has nothing to do with the limit.
    """
    raw = PREFIX + "\ud800" + SUFFIX
    assert len(raw) * 4 <= MAX_MESSAGE_BYTES, "premise: under the limit however you count it"
    out = decode(raw)
    assert isinstance(out, AstraTranscript), (
        "criterion 6: a message far under the limit decodes by its fields, not by its "
        f"encodability; got {type(out).__name__}"
    )
    assert out.text == "\ud800" and out.turn_id == "t"


# --- T-0030 group A: astra.spoken.played ------------------------------------------------------
#
# Criteria 1 and 2 of docs/tasks/T-0030.md. The parametrised round-trip above picks the new model
# up automatically once it is in ``ASTRA_MESSAGE_TYPES``, but ``_examples.example_of`` builds the
# *smallest* valid instance, so the explicit row below is what exercises a non-empty nested
# ``utterances`` list — a model whose list happens to default to empty would round-trip vacuously.
#
#   AstraSpokenPlayed(type="astra.spoken.played", turn_id: str, played_ms: int (ge=0),
#                     utterances: list[PlayedUtterance])
#   PlayedUtterance(utterance_id: str, played_text: str, played_words: int (ge=0), complete: bool)

_T0030_MISSING: str | None = None
try:
    from local_astra.contracts.protocol import AstraSpokenPlayed, PlayedUtterance
except ImportError as _exc:  # the pre-implementation path
    AstraSpokenPlayed = PlayedUtterance = None  # type: ignore[assignment,misc]
    _T0030_MISSING = f"contracts/protocol.py: {_exc}"

PLAYED_TYPE = "astra.spoken.played"


def played_message():
    """The message a barge-in produces, with two entries — one heard whole, one cut."""
    assert _T0030_MISSING is None, (
        f"T-0030 criterion 1 adds AstraSpokenPlayed and PlayedUtterance ({_T0030_MISSING}); "
        "docs/tasks/T-0030.md, and tests/unit/test_played_split.py's docstring for the fields"
    )
    return AstraSpokenPlayed(
        turn_id="t1",
        played_ms=1_400,
        utterances=[
            PlayedUtterance(
                utterance_id="t1-u1",
                played_text="It is a stack trace. ",
                played_words=5,
                complete=True,
            ),
            PlayedUtterance(
                utterance_id="t1-u2", played_text="The error", played_words=2, complete=False
            ),
        ],
    )


def test_the_played_message_round_trips_with_a_non_empty_utterance_list():
    """Criterion 1: through the real codec, nested models and all."""
    message = played_message()
    wire = encode(message)
    assert json.loads(wire)["type"] == PLAYED_TYPE
    assert decode(wire) == message
    assert decode(wire.encode()) == message
    entries = json.loads(wire)["utterances"]
    assert entries == [
        {
            "utterance_id": "t1-u1",
            "played_text": "It is a stack trace. ",
            "played_words": 5,
            "complete": True,
        },
        {
            "utterance_id": "t1-u2",
            "played_text": "The error",
            "played_words": 2,
            "complete": False,
        },
    ], "criterion 1: the entries go on the wire as objects with exactly those four keys"


def test_the_new_type_is_server_to_client_only():
    """Criterion 1: a page that *sends* it is refused as ``wrong_direction`` (T-0028 criterion 3),
    which is what membership in exactly one of the two sets buys."""
    assert PLAYED_TYPE in SERVER_TO_CLIENT
    assert PLAYED_TYPE not in CLIENT_TO_SERVER
    assert not (CLIENT_TO_SERVER & SERVER_TO_CLIENT)
    assert PLAYED_TYPE in {str(m.model_fields["type"].default) for m in ASTRA_MESSAGE_TYPES}, (
        "criterion 1: and it is in ASTRA_MESSAGE_TYPES, so the parametrised round-trip covers it"
    )


def test_the_played_message_refuses_a_negative_word_count():
    """Criterion 1: ``played_words`` is ``ge=0``; a negative cut index would make the page slice
    from the end of the string (JS ``slice(-2)``) and dim the wrong words."""
    bad = json.dumps(
        {
            "type": PLAYED_TYPE,
            "turn_id": "t1",
            "played_ms": 10,
            "utterances": [
                {
                    "utterance_id": "u1",
                    "played_text": "",
                    "played_words": -1,
                    "complete": False,
                }
            ],
        }
    )
    assert isinstance(decode(bad), Malformed)
    negative_ms = json.loads(encode(played_message()))
    negative_ms["played_ms"] = -1
    assert isinstance(decode(json.dumps(negative_ms)), Malformed), "played_ms is ge=0 too"


def test_astra_spoken_is_unchanged():
    """T-0030 criterion 2 *(negative)*, and it must stay green through the whole task.

    ``AstraSpoken.text`` is the TTS-normalised text as handed to TTS (T-0028 ruling A1). The
    played half arrives as its own message; nobody may "just add played_text here" — the two
    strings are built from different sources (landmine 1) and a field that silently carries one
    where the other is expected is how the pane ends up striking text the user heard.
    """
    from local_astra.contracts.protocol import PROTOCOL_VERSION, AstraSpoken

    # T-0037 amends this row by exactly one field (criterion 2 adds ``replay: bool = False`` to
    # five server→client models). The intent is unchanged and still mutation-sensitive: the
    # *payload* fields are these three, ``played_text`` is not among them, and nothing else may
    # be smuggled in — ``replay`` is named here so that "one more field" stays a spec change and
    # not a habit. The row above (round-trip) covers its encoding.
    assert [n for n in AstraSpoken.model_fields if n not in ("type", "replay")] == [
        "turn_id",
        "utterance_id",
        "text",
    ]
    assert {
        n: f.annotation for n, f in AstraSpoken.model_fields.items() if n not in ("type", "replay")
    } == {
        "turn_id": str,
        "utterance_id": str,
        "text": str,
    }
    assert "played_text" not in AstraSpoken.model_fields, (
        "T-0030 criterion 2: the played prefix travels as astra.spoken.played; the two strings "
        "are built from different sources (landmine 1)"
    )
    assert PROTOCOL_VERSION == 1, (
        "criterion 2: adding a message type is backward compatible — an old page ignores an "
        "unknown astra.* through the default branch of hud/src/state.js — so the version stays 1"
    )


# --- T-0037: the replay marker ------------------------------------------------------------------

REPLAY_MODELS = (
    "AstraTranscript",
    "AstraSpoken",
    "AstraUiSurface",
    "AstraSpokenPlayed",
    "AstraTurnState",
)


def replay_models():
    import local_astra.contracts.protocol as protocol

    return [getattr(protocol, name) for name in REPLAY_MODELS]


@pytest.mark.parametrize("name", REPLAY_MODELS)
def test_the_replay_marker_defaults_to_false_and_round_trips(name):
    """T-0037 criterion 2: five models gain ``replay: bool = False`` and nothing else changes.

    The default is the load-bearing half. Every one of these messages is published *live* far more
    often than it is replayed, and the session constructs them without naming the field; a default
    of ``True`` — or a required field — would mark the whole wire as a catch-up and the page would
    render a permanent replay.
    """
    import local_astra.contracts.protocol as protocol

    model = getattr(protocol, name)
    assert "replay" in model.model_fields, (
        f"criterion 2: {name} gains `replay: bool = False` (docs/tasks/T-0037.md)"
    )
    field = model.model_fields["replay"]
    assert field.annotation is bool and field.default is False, (
        f"criterion 2: replay is a bool defaulting to False; {name} declares {field!r}"
    )

    live = example_of(model)
    assert live.replay is False, "example_of builds a live message: the default is False"
    body = json.loads(encode(live))
    assert body["replay"] is False, f"…and the default is on the wire, not implied: {body!r}"

    marked = live.model_copy(update={"replay": True})
    assert decode(encode(marked)) == marked, "a replayed message round-trips through the codec"


def test_the_replay_marker_does_not_open_the_models_up():
    """T-0037 criterion 2, negative: ``extra="forbid"`` still refuses everything else.

    **Green before and after T-0037 by design** (so it carries no ``pending`` marker): it is the
    fence around the cheap way to make a new field travel — relaxing ``contracts/base.Contract``
    to ``extra="allow"`` — which would let a typo (``replayy``) cross the wire to a page that
    still refuses it, and the reconnecting page would show nothing at all, silently.
    """
    from local_astra.contracts.protocol import AstraSpoken

    raw = json.loads(encode(example_of(AstraSpoken)))
    for key in ("replayy", "Replay", "replay_"):
        assert isinstance(decode(json.dumps({**raw, key: True})), Malformed), (
            f"criterion 2: {key!r} is not a field of AstraSpoken and contracts/base.Contract "
            "forbids unknown fields"
        )
    # (``{"replay": "yes"}`` is *not* asserted: pydantic's lax mode coerces it to True, exactly as
    # it does for ``final`` on the same message. That is the contract's behaviour, not this task's.)


def test_no_message_type_is_added_or_moved_by_the_replay():
    """T-0037 criterion 2, negative: the replay is a field, not a new message.

    ``ASTRA_MESSAGE_TYPES``, ``SERVER_TO_CLIENT`` and ``CLIENT_TO_SERVER`` are the page's dispatch
    table (mirrored in ``hud/src/protocol.js``); adding a type for the catch-up would move both
    sides and the version. The five marked models are exactly the ones the journal holds.
    """
    assert {str(m.model_fields["type"].default) for m in ASTRA_MESSAGE_TYPES} == (
        CLIENT_TO_SERVER | SERVER_TO_CLIENT
    ), "criterion 2: no new message type"
    assert not (CLIENT_TO_SERVER & SERVER_TO_CLIENT)
    marked = {m.__name__ for m in ASTRA_MESSAGE_TYPES if "replay" in m.model_fields}
    # T-0090 criteria 1 and 3 move this pin by one: astra.tool.call is journalled (the latest
    # message per call_id of the open turn) and so carries `replay: bool = False` too.
    # T-0162 A-1/A-5 (pre-authorised pin move): AstraSessionCleared carries `replay: bool = False`,
    # since the hello reply after a Restart begins with it replayed. It is the one addition.
    # T-0177 A1 (pin move A-P2): AstraVisionState and AstraVisionFrame carry `replay: bool = False`
    # (the live hello's state, A4, and B9's replayed latest frame); both are server->client, which
    # the loop below still checks. Neither is held by the T-0037 journal (implementation notes).
    # T-0180 A1/A4 (activation pin move): AstraThought carries `replay: bool = False`; the journal
    # holds the latest per attempt_id of the current turn. It is server->client only.
    # T-0213 A2r-1 (R-5 named re-pin, the new type): AstraSettingsState carries `replay: bool = False` (A2r-1 names
    # the field); it is server->client only and not held by the T-0037 journal (the hello builds it fresh, last).
    assert marked == set(REPLAY_MODELS) | {
        "AstraToolCall",
        "AstraSessionCleared",
        "AstraVisionState",
        "AstraVisionFrame",
        "AstraThought",
        "AstraSettingsState",
    }, (
        "criterion 2: exactly the models the journal holds carry the marker — T-0037's five plus "
        "T-0090's AstraToolCall; not astra.error, not the canvas relay, not a client→server "
        f"message; got {sorted(marked)}"
    )
    for model in replay_models():
        assert str(model.model_fields["type"].default) in SERVER_TO_CLIENT, (
            f"{model.__name__} is replayed, so it is a server→client message"
        )


# --- T-0039 group A: astra.text, the typed question ---------------------------------------------
#
# Criteria 1 and 2 of docs/tasks/T-0039.md. The wire type the page uses to ask a question without
# a microphone:
#
#   AstraText(type="astra.text", text: str)
#
# ``text`` is validated **on the model** (criterion 2), so a violation is a ``codec.decode`` →
# ``Malformed`` and the page is answered ``astra.error(code="malformed")`` on the path T-0028
# already built — never a truncation and never a silent drop. The cap is **2000 code points**;
# T-0027's 8 MiB *byte* limit is a different rule and is untouched, which is why the emoji row
# below must be *accepted*.
#
# Every row goes through the real ``decode``/``encode``: this module asserts what the codec did
# with a string, never what a model constructed here remembers (.claude/rules/tests.md).

TEXT_TYPE = "astra.text"

#: Criterion 2's chosen bound (ambiguity 1 of §5: chosen, not measured).
TEXT_MAX_CODE_POINTS = 2000


def astra_text_model():
    import local_astra.contracts.protocol as protocol

    model = getattr(protocol, "AstraText", None)
    assert model is not None, (
        "T-0039 criterion 1: contracts/protocol.py gains "
        'AstraText(type="astra.text", text: str) and puts it in AstraMessage, '
        "ASTRA_MESSAGE_TYPES and CLIENT_TO_SERVER (docs/tasks/T-0039.md)"
    )
    return model


def text_wire(text: str) -> str:
    """One ``astra.text`` exactly as the page would put it on the socket."""
    return json.dumps({"type": TEXT_TYPE, "text": text}, ensure_ascii=False)


def test_the_typed_text_message_round_trips_through_the_real_codec():
    """Criterion 1: it decodes to the model, encodes back, and carries exactly one payload field.

    The field census is the load-bearing half: a ``turn_id``, a ``surface_id`` or a ``replay``
    smuggled in here would make the transport's job a decision (criterion 3: it validates nothing
    of its own), and ``replay`` specifically is a *server→client* catch-up marker (T-0037) that
    has no meaning on a message the page sends.
    """
    model = astra_text_model()
    message = decode(text_wire("what is on my screen"))
    assert isinstance(message, model), (
        f"criterion 1: an astra.text decodes to AstraText; decode returned {type(message).__name__}"
    )
    assert message.text == "what is on my screen"
    assert decode(encode(message)) == message
    assert decode(encode(message).encode()) == message, "bytes input too"
    assert json.loads(encode(message)) == {"type": TEXT_TYPE, "text": "what is on my screen"}, (
        "criterion 1: the message is its type and its text, nothing else"
    )
    assert "replay" not in model.model_fields, (
        "T-0037's replay marker is a server→client catch-up field; a client→server message "
        "cannot be replayed to anybody (docs/tasks/T-0039.md §4, the replay census)"
    )


def test_the_typed_text_message_is_client_to_server_only():
    """Criterion 1 / 3: one direction, so a server that *sends* one is impossible by construction
    and a page that sends anything else on this path is still refused as ``wrong_direction``."""
    model = astra_text_model()
    assert TEXT_TYPE in CLIENT_TO_SERVER, (
        f"criterion 1: astra.text is a client→server message; CLIENT_TO_SERVER is "
        f"{sorted(CLIENT_TO_SERVER)}"
    )
    assert TEXT_TYPE not in SERVER_TO_CLIENT, (
        "criterion 3: an astra.text in the server→client set would let the endpoint send one"
    )
    assert not (CLIENT_TO_SERVER & SERVER_TO_CLIENT), "the two sets stay disjoint"
    tags = {str(m.model_fields["type"].default) for m in ASTRA_MESSAGE_TYPES}
    assert TEXT_TYPE in tags, (
        "criterion 1: astra.text is in ASTRA_MESSAGE_TYPES, so the parametrised round-trip at the "
        "top of this module covers it too"
    )
    assert tags == CLIENT_TO_SERVER | SERVER_TO_CLIENT, (
        "criterion 1: the partition still covers every model — a type in the union but in neither "
        "set is routed by nothing and answered by nobody"
    )
    assert model in ASTRA_MESSAGE_TYPES


@pytest.mark.parametrize(
    ("label", "text"),
    [
        ("empty", ""),
        ("whitespace_only", " \t\n  "),
        ("one_code_point_over_the_cap", "a" * (TEXT_MAX_CODE_POINTS + 1)),
    ],
    ids=["empty", "whitespace_only", "2001_code_points"],
)
def test_a_question_the_model_refuses_is_malformed_not_truncated(label, text):
    """Criterion 2: the three refusals, each through ``decode``, each landing on ``Malformed``.

    ``Malformed`` is the whole point: T-0028's ingress answers it with
    ``astra.error(code="malformed")`` and keeps the socket open, so the page is *told*. A model
    that clamped the string instead would send the LLM a question the user did not ask, and one
    that dropped the message would leave the page waiting for an answer that never comes.
    """
    astra_text_model()
    if label == "one_code_point_over_the_cap":
        assert len(text) == 2001, "premise: one code point over the 2000-code-point cap"
        assert len(text.encode("utf-8")) < MAX_MESSAGE_BYTES, (
            "premise: far under T-0027's byte limit, so the refusal is this task's cap and not "
            "the inherited one"
        )
    else:
        assert text.strip() == "", "premise: nothing survives the strip"
    out = decode(text_wire(text))
    assert isinstance(out, Malformed), (
        f"criterion 2: an astra.text whose text is {label} is Malformed (the page gets "
        f"astra.error(code='malformed')); decode returned {type(out).__name__} — "
        f"{getattr(out, 'text', None)!r}"
    )


def test_two_thousand_four_byte_characters_are_accepted_because_the_cap_counts_code_points():
    """Criterion 2, the row that separates the two limits: 2000 astral characters = 8000 bytes.

    The cap is **code points**; T-0027's is **bytes**. An implementation that caps
    ``len(text.encode())`` at 2000 refuses this perfectly legal question, and one that caps
    UTF-16 units (the page's natural unit) would count each of these twice and refuse it too.
    """
    model = astra_text_model()
    text = "\U0001f600" * TEXT_MAX_CODE_POINTS
    assert len(text) == 2000, "premise: exactly 2000 code points"
    assert len(text.encode("utf-8")) == 8000, "premise: 4 UTF-8 bytes each"
    assert len(text.encode("utf-16-le")) // 2 == 4000, "premise: 2 UTF-16 units each"
    out = decode(text_wire(text))
    assert isinstance(out, model), (
        "criterion 2: 2000 code points is at the cap, whatever they encode to; decode returned "
        f"{type(out).__name__}"
    )
    assert out.text == text and len(out.text) == 2000, "…and nothing was truncated on the way in"


def test_surrounding_whitespace_is_stripped_by_the_model_not_by_the_transport():
    """Criterion 2: the strip happens on the model, so every entry point agrees.

    A page that sends ``"  what is this?\\n"`` (the shape a textarea produces) must reach the LLM
    as ``"what is this?"`` and appear in the transcript pane that way — the transport is dumb
    (criterion 3), so if the strip lived there the fake, the rung and any later client would each
    need their own copy of it.
    """
    model = astra_text_model()
    out = decode(text_wire("  \t what is this?\n "))
    assert isinstance(out, model), f"decode returned {type(out).__name__}"
    assert out.text == "what is this?", (
        f"criterion 2: surrounding whitespace is stripped by the model; got {out.text!r}"
    )
    inner = decode(text_wire("  what   is  this?  "))
    assert isinstance(inner, model) and inner.text == "what   is  this?", (
        "…and only the *surrounding* whitespace: the inside of the question is the user's, not "
        f"the codec's, to normalise; got {getattr(inner, 'text', inner)!r}"
    )


def test_the_cap_is_counted_after_stripping():
    """Ruling A1, owed by A14: strip first, **then** count — the order is the whole rule.

    A textarea hands the page a trailing newline, and a user who pastes a question often brings a
    leading space with it. Under cap-before-strip a question of exactly 2000 code points with
    that ordinary surrounding whitespace is 2002 on the wire and is refused as ``malformed`` —
    the user sees a rejection for characters they cannot see and did not type. Under
    strip-before-cap it is accepted and is 2000 code points when the session gets it.

    The negative half pins the other direction: stripping does not *buy* room. 2001 code points
    of question is over the cap no matter how much whitespace surrounds it, because the strip
    removes only what is outside.
    """
    model = astra_text_model()
    at_the_cap = "a" * TEXT_MAX_CODE_POINTS
    padded = " " + at_the_cap + "\n"
    assert len(padded) == TEXT_MAX_CODE_POINTS + 2, "premise: 2002 code points on the wire"

    out = decode(text_wire(padded))
    assert isinstance(out, model), (
        "ruling A1: the 2000-code-point cap is counted after the strip, so a question at the cap "
        "with a leading space and a trailing newline is accepted; decode returned "
        f"{type(out).__name__} — {getattr(out, 'text', None)!r}"
    )
    assert len(out.text) == TEXT_MAX_CODE_POINTS, (
        f"…carrying exactly {TEXT_MAX_CODE_POINTS} code points; it carries {len(out.text)}"
    )
    assert out.text == at_the_cap, "…and the question itself, unchanged"

    over = " " + "a" * (TEXT_MAX_CODE_POINTS + 1) + "\n"
    assert isinstance(decode(text_wire(over)), Malformed), (
        "ruling A1: the strip removes surrounding whitespace, it does not raise the cap — 2001 "
        "code points of question is Malformed with or without the padding"
    )


# --- T-0043 group B: the session-log pair -------------------------------------------------------
#
# Criterion 16 of docs/tasks/T-0043.md. A **new pair**, deliberately not a field on the assist
# toggle:
#
#   AstraLogToggle(type="astra.log.toggle", on: bool)                    CLIENT_TO_SERVER only
#   AstraLogState(type="astra.log.state", on: bool, path: str | None,
#                 records: int, dropped: int, bytes: int, error: str | None)   SERVER_TO_CLIENT
#
# The separation is the criterion's own reasoning and is pinned below: the log flag is not an
# assist, so it may not enter ``AssistFlags``, the assist badge or ``Regime`` — a turn's regime
# says what the model could *see*, not whether we were debugging. A `log_*` field on any of those
# three would silently change what every bench line claims about the turns it describes.
#
# Every row goes through the real ``decode``/``encode``; nothing here asserts what a model built
# in this file remembers (.claude/rules/tests.md).

LOG_TOGGLE_TYPE = "astra.log.toggle"
LOG_STATE_TYPE = "astra.log.state"

#: Criterion 16, transcribed from the task file: the exact payload fields of each model, in the
#: order the criterion names them. This tuple is the spec; the models are compared against it.
LOG_TOGGLE_FIELDS = ("on",)
LOG_STATE_FIELDS = ("on", "path", "records", "dropped", "bytes", "error")


def log_model(name: str):
    """One of the two new models, or an ``AssertionError`` naming what group B has not added."""
    import local_astra.contracts.protocol as protocol

    model = getattr(protocol, name, None)
    assert model is not None, (
        f"T-0043 criterion 16: contracts/protocol.py gains {name} and puts it in AstraMessage, "
        "ASTRA_MESSAGE_TYPES and exactly one of CLIENT_TO_SERVER / SERVER_TO_CLIENT "
        "(docs/tasks/T-0043.md)"
    )
    return model


def log_state_wire(**overrides) -> str:
    """One ``astra.log.state`` exactly as the server would put it on the socket."""
    payload = {
        "type": LOG_STATE_TYPE,
        "on": True,
        "path": "/home/u/astra/logs/sessions/20260916-041500-12345.jsonl",
        "records": 412,
        "dropped": 0,
        "bytes": 190_233,
        "error": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_the_log_toggle_round_trips_and_carries_one_payload_field():
    """Criterion 16: the page's half of the pair is a bare boolean request.

    The field census is the load-bearing half. A ``path`` or a ``log_dir`` on the *client's*
    message would let the page choose where the session writes its file, which is the CLI's
    decision (criterion 15) and not the page's; a ``replay`` would be T-0037's server→client
    catch-up marker on a message the page sends.
    """
    model = log_model("AstraLogToggle")
    message = decode(json.dumps({"type": LOG_TOGGLE_TYPE, "on": True}))
    assert isinstance(message, model), (
        f"criterion 16: an astra.log.toggle decodes to AstraLogToggle; decode returned "
        f"{type(message).__name__}"
    )
    assert message.on is True
    assert decode(encode(message)) == message
    assert decode(encode(message).encode()) == message, "bytes input too"
    assert json.loads(encode(message)) == {"type": LOG_TOGGLE_TYPE, "on": True}, (
        "criterion 16: the message is its type and its flag, and nothing else"
    )
    assert tuple(n for n in model.model_fields if n != "type") == LOG_TOGGLE_FIELDS, (
        f"criterion 16: AstraLogToggle carries {LOG_TOGGLE_FIELDS}; it carries "
        f"{tuple(n for n in model.model_fields if n != 'type')}"
    )
    assert isinstance(decode(json.dumps({"type": LOG_TOGGLE_TYPE})), Malformed), (
        "criterion 16: `on` is required — a toggle that does not say which way is not a toggle"
    )
    for bad in ("maybe", [], {}, 2):
        assert isinstance(decode(json.dumps({"type": LOG_TOGGLE_TYPE, "on": bad})), Malformed), (
            f"criterion 16: `on` is a bool; {bad!r} is not one, and a page that could put it on "
            "the wire would leave the server guessing which way the switch went"
        )


def test_the_log_state_round_trips_with_the_writers_counters():
    """Criterion 16/17: the server's echo carries what the panel renders — path and counters.

    ``path`` and ``error`` are nullable on purpose: ``path`` is ``None`` while logging is off and
    ``error`` is the reason an ``OSError`` disabled it (criterion 4, ruling A1). The panel renders
    the string the server sent, verbatim, so both must survive the round trip untouched.
    """
    model = log_model("AstraLogState")
    message = decode(log_state_wire())
    assert isinstance(message, model), (
        f"criterion 16: an astra.log.state decodes to AstraLogState; got {type(message).__name__}"
    )
    assert tuple(n for n in model.model_fields if n != "type") == LOG_STATE_FIELDS, (
        f"criterion 16: AstraLogState carries {LOG_STATE_FIELDS} in that order; it carries "
        f"{tuple(n for n in model.model_fields if n != 'type')}"
    )
    assert message.path == "/home/u/astra/logs/sessions/20260916-041500-12345.jsonl"
    assert (message.records, message.dropped, message.bytes) == (412, 0, 190_233)
    assert message.error is None
    assert decode(encode(message)) == message, (
        "criterion 16: it survives its own encoding. ``codec.encode`` is ``exclude_none=True``, "
        "so a *required* nullable field would not survive this round trip at all: `path` and "
        "`error` carry ``= None`` defaults, which is what lets 'logging is off, no file' be "
        "expressible on the wire"
    )
    on_the_wire = {k: v for k, v in json.loads(log_state_wire()).items() if v is not None}
    assert json.loads(encode(message)) == on_the_wire, (
        "criterion 16: the encode is the message the page decodes — no field added and none "
        "renamed. ``codec.encode`` omits null fields for every message type (exclude_none), so "
        "the page's field table must not require `path` or `error`"
    )

    off = decode(log_state_wire(on=False, path=None, records=0, dropped=0, bytes=0))
    assert isinstance(off, model) and off.on is False and off.path is None, (
        "criterion 17: logging off is a state the server publishes, with no path; decode gave "
        f"{off!r}"
    )
    failed = decode(log_state_wire(on=False, path=None, error="cannot open /x: is a directory"))
    assert isinstance(failed, model) and failed.error == "cannot open /x: is a directory", (
        "criterion 4/16: the reason logging disabled itself reaches the page verbatim"
    )
    assert isinstance(decode(log_state_wire(on=None)), Malformed), (
        "criterion 19: `on` is not nullable — the page's three-state rendering distinguishes "
        "'no state has arrived' (nothing decoded at all) from a state that says off, and a "
        "nullable `on` would make 'the server said nothing yet' indistinguishable from "
        "'the server said null'"
    )


def test_the_log_pair_is_one_direction_each_and_the_partition_still_covers_every_model():
    """Criterion 16: two disjoint sets, jointly covering every model, with the pair split.

    A toggle in the server set would let the endpoint send one; a state in the client set would
    let a page claim its own session is being logged. The transport's ``wrong_direction`` refusal
    (T-0028 criterion 3) is derived from these two sets and from nothing else.
    """
    toggle, state = log_model("AstraLogToggle"), log_model("AstraLogState")
    assert LOG_TOGGLE_TYPE in CLIENT_TO_SERVER, f"CLIENT_TO_SERVER is {sorted(CLIENT_TO_SERVER)}"
    assert LOG_TOGGLE_TYPE not in SERVER_TO_CLIENT, (
        "criterion 16: the toggle is the page's egress only — a server that sends one is "
        "impossible by construction"
    )
    assert LOG_STATE_TYPE in SERVER_TO_CLIENT, f"SERVER_TO_CLIENT is {sorted(SERVER_TO_CLIENT)}"
    assert LOG_STATE_TYPE not in CLIENT_TO_SERVER, (
        "criterion 16: a page that sends astra.log.state is refused as wrong_direction"
    )
    assert not (CLIENT_TO_SERVER & SERVER_TO_CLIENT), "the two sets stay disjoint"
    assert toggle in ASTRA_MESSAGE_TYPES and state in ASTRA_MESSAGE_TYPES, (
        "criterion 16: both are in ASTRA_MESSAGE_TYPES, so the parametrised round-trip at the top "
        "of this module covers them unchanged in shape"
    )
    tags = {str(m.model_fields["type"].default) for m in ASTRA_MESSAGE_TYPES}
    assert tags == CLIENT_TO_SERVER | SERVER_TO_CLIENT, (
        "criterion 16: the partition still covers every model — a type in the union but in "
        "neither set is routed by nothing and answered by nobody"
    )
    for model in (toggle, state):
        assert "replay" not in model.model_fields, (
            "T-0037's replay census: the log pair is not replayed. The state a reconnecting page "
            "is owed is the *current* one, sent live as the third message of hello_reply "
            "(criterion 18), not a marked copy of an old one"
        )


def test_the_log_flag_is_not_an_assist_and_never_enters_a_turns_regime():
    """Criterion 16, the separation, as a field census over the three places it must not reach.

    "A turn's regime states what the model could see, not whether we were debugging." A
    ``log`` field on ``AssistFlags`` would put the debug switch in the assist badge, in the HUD's
    assist rows and — through ``Regime.assist_flags`` — on every ``turns.jsonl`` line, where a
    bench reader comparing two runs would read it as a difference in what the model was shown.
    """
    from local_astra.contracts.perception import AssistFlags
    from local_astra.contracts.protocol import AstraAssistToggle
    from local_astra.telemetry.turns_log import Regime

    log_model("AstraLogToggle")  # the pair exists, so this census is about a real alternative

    assert tuple(AssistFlags.model_fields) == ("page_text", "pdf_text", "cells", "at_spi"), (
        "criterion 16: AssistFlags is the four non-pixel assists and gains no log flag; it has "
        f"{tuple(AssistFlags.model_fields)}"
    )
    assert tuple(AstraAssistToggle.model_fields) == ("type", "flags"), (
        "criterion 16: the log toggle is a new message, not a field on the assist toggle; "
        f"AstraAssistToggle has {tuple(AstraAssistToggle.model_fields)}"
    )
    offenders = [name for name in Regime.model_fields if "log" in name.lower()]
    assert offenders == [], (
        "criterion 16: a turn's Regime says what the model could see; it does not say whether we "
        f"were debugging. Regime gained {offenders} (TurnRecord.session_log, criterion 14, is a "
        "top-level field of the record and not part of the regime)"
    )
    flags = AssistFlags()
    assert flags.badge() == "pixels", (
        "criterion 16: the assist badge is unchanged by this task — a badge that grew a `+log` "
        f"would relabel every turn taken while the log was on; it reads {flags.badge()!r}"
    )
    assert AssistFlags(page_text=True).badge() == "pixels+dom", "…and so is every other spelling"


# --- T-0090: astra.tool.call, the lifecycle of a proposed action -----------------------------------
#
# Criterion 1 of docs/tasks/T-0090.md, transcribed:
#
#   AstraToolCall(type="astra.tool.call", turn_id, call_id, name, arguments: dict,
#                 status: proposed | repairing | dry_run | executed | refused | cancelled,
#                 source: draft | repair,
#                 action: str | None, reason: str | None, query_problem: str | None,
#                 replay: bool = False)
#
# in SERVER_TO_CLIENT only, decoded totally. Every expected value below is the criterion's text or
# a hand-written wire object; nothing is read back from the model to build its own expectation.

TOOL_CALL_TYPE = "astra.tool.call"
TOOL_CALL_FIELDS = {
    "type",
    "turn_id",
    "call_id",
    "name",
    "arguments",
    "status",
    "source",
    "action",
    "reason",
    "query_problem",
    "replay",
}
TOOL_CALL_STATUSES = ("proposed", "repairing", "dry_run", "executed", "refused", "cancelled")
TOOL_CALL_SOURCES = ("draft", "repair")


def tool_call_model():
    """``AstraToolCall``, or an ``AssertionError`` naming what criterion 1 adds."""
    import local_astra.contracts.protocol as protocol

    model = getattr(protocol, "AstraToolCall", None)
    assert model is not None, (
        "T-0090 criterion 1: contracts/protocol.py gains AstraToolCall(type='astra.tool.call', "
        "turn_id, call_id, name, arguments, status, source, action, reason, query_problem, "
        "replay=False) and puts it in AstraMessage, ASTRA_MESSAGE_TYPES and SERVER_TO_CLIENT"
    )
    return model


def tool_call_wire(**overrides: object) -> dict[str, object]:
    """One ``astra.tool.call`` as the server would put it on the socket (a dry-run search)."""
    payload: dict[str, object] = {
        "type": TOOL_CALL_TYPE,
        "turn_id": "t7",
        "call_id": "t7-c1",
        "name": "google_search",
        "arguments": {"query": "weather in Paris tomorrow"},
        "status": "dry_run",
        "source": "draft",
        "action": "would search for: weather in Paris tomorrow",
        "reason": None,
        "query_problem": None,
    }
    payload.update(overrides)
    return payload


def test_the_tool_call_carries_exactly_the_criterion_one_fields():
    model = tool_call_model()
    # T-0162 C-1 (pre-authorised pin move, amendment 2026-10-27): exactly step and result are added
    from tests.unit._t0162c_pins import TOOL_CALL_ADDED

    assert set(model.model_fields) == TOOL_CALL_FIELDS | TOOL_CALL_ADDED, (
        "T-0090 criterion 1: AstraToolCall carries exactly "
        f"{sorted(TOOL_CALL_FIELDS)}; it carries {sorted(model.model_fields)}. A field the page's "
        "table does not mirror turns every live astra.tool.call into kind:'malformed' on the page"
    )
    replay = model.model_fields["replay"]
    assert replay.annotation is bool and replay.default is False, (
        f"criterion 1: replay is `bool = False` — a live message is not a catch-up; got {replay!r}"
    )


def test_the_tool_call_round_trips_through_the_real_codec():
    model = tool_call_model()
    raw = json.dumps(tool_call_wire())
    message = decode(raw)
    assert isinstance(message, model), (
        f"criterion 1: a well-formed astra.tool.call decodes to AstraToolCall; got {message!r}"
    )
    assert message.arguments == {"query": "weather in Paris tomorrow"}
    assert (message.status, message.source) == ("dry_run", "draft")
    assert message.action == "would search for: weather in Paris tomorrow"
    assert message.replay is False, "criterion 1: replay defaults to False"
    assert decode(encode(message)) == message
    assert decode(encode(message).encode()) == message, "bytes input too"
    on_the_wire = {k: v for k, v in tool_call_wire().items() if v is not None}
    on_the_wire["replay"] = False
    assert json.loads(encode(message)) == on_the_wire, (
        "criterion 1: the encode is the message the page decodes — nothing added, nothing renamed "
        "(codec.encode omits null fields, so action/reason/query_problem must carry `= None` "
        "defaults or a message without them would not survive its own round trip)"
    )


def test_a_proposed_call_with_no_action_reason_or_problem_round_trips():
    """``action``, ``reason`` and ``query_problem`` are ``str | None``; a proposal has none of them.

    ``codec.encode`` is ``exclude_none``, so a *required* nullable field would vanish on encode and
    the decode would then refuse the server's own message. Hand-written: the minimal proposal.
    """
    model = tool_call_model()
    minimal = {
        "type": TOOL_CALL_TYPE,
        "turn_id": "t7",
        "call_id": "t7-c1",
        "name": "click_at",
        "arguments": {"x": 120, "y": 845},
        "status": "proposed",
        "source": "draft",
    }
    message = decode(json.dumps(minimal))
    assert isinstance(message, model), f"criterion 1: the minimal proposal decodes; got {message!r}"
    assert (message.action, message.reason, message.query_problem) == (None, None, None)
    assert decode(encode(message)) == message


@pytest.mark.parametrize("status", TOOL_CALL_STATUSES)
def test_every_criterion_one_status_is_accepted(status):
    model = tool_call_model()
    message = decode(json.dumps(tool_call_wire(status=status)))
    assert isinstance(message, model) and message.status == status, (
        f"criterion 1: status {status!r} is one of the six; decode gave {message!r}"
    )


@pytest.mark.parametrize("source", TOOL_CALL_SOURCES)
def test_both_sources_are_accepted(source):
    model = tool_call_model()
    message = decode(json.dumps(tool_call_wire(source=source)))
    assert isinstance(message, model) and message.source == source, (
        f"criterion 1: source {source!r} is draft|repair; decode gave {message!r}"
    )


@pytest.mark.parametrize(
    ("field", "value", "why"),
    [
        ("status", "executing", "a status outside the six literals"),
        ("status", "done", "a status outside the six literals"),
        ("status", "", "an empty status"),
        ("status", None, "a null status"),
        ("source", "model", "a source outside draft|repair"),
        ("source", None, "a null source"),
        ("arguments", ["query", "x"], "arguments that are a list, not an object"),
        ("arguments", "query=x", "arguments that are a string"),
        ("arguments", None, "null arguments"),
        ("call_id", 7, "a call id that is not a string"),
        ("turn_id", None, "a null turn id"),
        ("name", None, "a null tool name"),
        ("action", 3, "an action that is not a string"),
        ("query_problem", ["too short"], "a query_problem that is not a string"),
    ],
)
def test_a_malformed_tool_call_is_malformed_with_its_type(field, value, why):
    tool_call_model()
    got = decode(json.dumps(tool_call_wire(**{field: value})))
    assert isinstance(got, Malformed), (
        f"criterion 1: {why} ({field}={value!r}) is refused, not coerced; decode gave {got!r}"
    )
    assert got.type == TOOL_CALL_TYPE, "…and the refusal says which type it was"


@pytest.mark.parametrize("missing", ["turn_id", "call_id", "name", "arguments", "status", "source"])
def test_every_required_field_is_required(missing):
    tool_call_model()
    payload = tool_call_wire()
    del payload[missing]
    got = decode(json.dumps(payload))
    assert isinstance(got, Malformed), (
        f"criterion 1: {missing} carries no default in the criterion; a message without it is "
        f"Malformed, got {got!r}"
    )


def test_an_unknown_field_on_the_tool_call_is_refused():
    tool_call_model()
    for extra in ({"url": "https://www.google.com/search?q=x"}, {"href": "x"}, {"resolved": {}}):
        got = decode(json.dumps({**tool_call_wire(), **extra}))
        assert isinstance(got, Malformed), (
            f"criterion 1 (contracts/base.Contract is extra='forbid'): {extra} is not a field of "
            f"astra.tool.call — least of all a URL (H2: the query is shown as text, no link); got "
            f"{got!r}"
        )


def test_the_tool_call_is_server_to_client_only():
    model = tool_call_model()
    assert TOOL_CALL_TYPE in SERVER_TO_CLIENT, (
        f"criterion 1: astra.tool.call is in SERVER_TO_CLIENT; it holds {sorted(SERVER_TO_CLIENT)}"
    )
    assert TOOL_CALL_TYPE not in CLIENT_TO_SERVER, (
        "criterion 1: …and only there — a page that sends one is refused as wrong_direction "
        "(the card is display-only and sends nothing back)"
    )
    assert model in ASTRA_MESSAGE_TYPES, (
        "criterion 1: in ASTRA_MESSAGE_TYPES, so the parametrised round trip at the top of this "
        "module covers it"
    )
    tags = {str(m.model_fields["type"].default) for m in ASTRA_MESSAGE_TYPES}
    assert tags == CLIENT_TO_SERVER | SERVER_TO_CLIENT
    assert not (CLIENT_TO_SERVER & SERVER_TO_CLIENT)


@settings(max_examples=300, deadline=None)
@given(
    st.dictionaries(
        st.sampled_from([*sorted(TOOL_CALL_FIELDS - {"type"}), "extra", "url"]),
        st.recursive(
            st.none()
            | st.booleans()
            | st.integers()
            | st.text(max_size=8)
            | st.sampled_from(TOOL_CALL_STATUSES + TOOL_CALL_SOURCES),
            lambda c: st.lists(c, max_size=3) | st.dictionaries(st.text(max_size=4), c, max_size=3),
            max_leaves=6,
        ),
        max_size=11,
    )
)
def test_tool_call_decode_is_total(fields):
    """Criterion 1: "the codec decodes it totally" — an AstraToolCall or a Malformed, never a raise."""
    model = tool_call_model()
    got = decode(json.dumps({**fields, "type": TOOL_CALL_TYPE}))
    assert isinstance(got, model | Malformed), (
        f"criterion 1: decode of a {TOOL_CALL_TYPE} is the model or Malformed; got {got!r}"
    )
