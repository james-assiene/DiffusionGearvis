"""Every contract model is strict, frozen and round-trips through JSON (docs/02 §5)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from local_astra.contracts.commands import IDEMPOTENT_COMMANDS, Command, CommandAdapter
from local_astra.contracts.events import EVENT_TYPES, EventAdapter, LlmCommit, Tick
from local_astra.contracts.perception import AssistFlags
from local_astra.contracts.speech import (
    SttEndOfTurnEvent,
    SttErrorEvent,
    SttEvent,
    SttEventAdapter,
    SttStepEvent,
    SttWordEvent,
)
from local_astra.contracts.turn import TurnState, TutorState
from tests.unit._examples import example_of

COMMAND_TYPES = [
    a
    for a in Command.__args__[0].__args__  # type: ignore[attr-defined]
]


@pytest.mark.parametrize("model", EVENT_TYPES, ids=lambda m: m.__name__)
def test_event_round_trips_through_discriminated_union(model):
    ex = example_of(model)
    back = EventAdapter.validate_json(ex.model_dump_json())
    assert back == ex
    assert type(back) is model


@pytest.mark.parametrize("model", COMMAND_TYPES, ids=lambda m: m.__name__)
def test_command_round_trips_through_discriminated_union(model):
    ex = example_of(model)
    back = CommandAdapter.validate_json(ex.model_dump_json())
    assert back == ex
    assert type(back) is model


@pytest.mark.parametrize("model", [*EVENT_TYPES, *COMMAND_TYPES], ids=lambda m: m.__name__)
def test_models_are_frozen_and_reject_unknown_fields(model):
    ex = example_of(model)
    with pytest.raises(ValidationError):
        ex.__class__(**{**ex.model_dump(), "unexpected_field": 1})
    with pytest.raises(ValidationError):
        ex.type = "something.else"  # frozen


def test_discriminator_rejects_wrong_type_tag():
    with pytest.raises(ValidationError):
        EventAdapter.validate_python({"type": "no.such.event", "t_ms": 0})
    with pytest.raises(ValidationError):
        # right tag, wrong payload: canvas_idx must be >= 0
        EventAdapter.validate_python(
            {
                "type": "llm.commit",
                "t_ms": 0,
                "request_id": "r",
                "canvas_idx": -1,
                "segment": "answer",
                "text": "",
            }
        )


def test_event_type_tags_are_unique_and_match_class_defaults():
    tags = [m.model_fields["type"].default for m in EVENT_TYPES]
    assert len(tags) == len(set(tags))
    assert all((isinstance(t, str) and "." in t) or t == "tick" for t in tags)


def test_idempotent_commands_exist():
    tags = {m.model_fields["type"].default for m in COMMAND_TYPES}
    assert tags >= IDEMPOTENT_COMMANDS


def test_llm_commit_segment_is_closed_enum():
    ok = LlmCommit(t_ms=0, request_id="r", canvas_idx=0, segment="ui", text="{}")
    assert ok.segment == "ui"
    with pytest.raises(ValidationError):
        LlmCommit(t_ms=0, request_id="r", canvas_idx=0, segment="reasoning", text="")  # type: ignore[arg-type]


def test_tick_carries_only_time():
    t = Tick(t_ms=1234)
    assert json.loads(t.model_dump_json()) == {"type": "tick", "t_ms": 1234}


def test_turn_state_names_are_exactly_the_architecture_states():
    assert [s.value for s in TurnState] == [
        "IDLE",
        "LISTENING",
        "ENDPOINTING",
        "THINKING",
        "SPEAKING",
        "PAUSED",
        "INTERRUPTED",
    ]


@pytest.mark.parametrize(
    ("flags", "badge"),
    [
        (AssistFlags(), "pixels"),
        (AssistFlags(page_text=True), "pixels+dom"),
        (AssistFlags(pdf_text=True, at_spi=True), "pixels+pdf+a11y"),
        (
            AssistFlags(page_text=True, pdf_text=True, cells=True, at_spi=True),
            "pixels+dom+pdf+cells+a11y",
        ),
    ],
)
def test_assist_badge_is_pixels_plus_enabled_assists_in_fixed_order(flags, badge):
    assert flags.badge() == badge
    assert flags.any_on() == (badge != "pixels")


def test_assist_flags_default_off():
    """Pixel-first (docs/12 §3): nothing is on unless the user ticks it."""
    assert not AssistFlags().any_on()


def test_tutor_state_prompt_block_hides_solution_unless_present_and_is_empty_without_problem():
    assert TutorState().as_prompt_block() == ""
    block = TutorState(
        problem="2x+3=11", solution="x=4", current_step=1, total_steps=3, ladder_level=2
    ).as_prompt_block()
    assert block.splitlines()[0] == "[tutor state]"
    assert "problem: 2x+3=11" in block
    assert "step: 1/3" in block
    assert "x=4" in block and "do not reveal" in block


# --- the STT event union (T-0007a criterion 18) ---------------------------------------------------

#: The four members that exist today. They must come through T-0007a byte-identical: the task
#: adds one member and changes nothing else ("Owned files": ``contracts/speech.py`` *narrowly*).
STT_EXAMPLES = {
    "word": SttWordEvent(text="five", start_ms=1000, end_ms=1250),
    "step": SttStepEvent(pause_heads=(0.1, 0.2, 0.3, 0.4), t_ms=880),
    "end_of_turn": SttEndOfTurnEvent(t_eot_ms=1750, p_end=0.9),
    "error": SttErrorEvent(message="unknown message type 'Frobnicate'"),
}


@pytest.mark.parametrize("tag", sorted(STT_EXAMPLES), ids=sorted(STT_EXAMPLES))
def test_stt_event_round_trips_through_its_discriminated_union(tag):
    example = STT_EXAMPLES[tag]
    back = SttEventAdapter.validate_json(example.model_dump_json())
    assert back == example and type(back) is type(example)
    assert json.loads(example.model_dump_json())["type"] == tag


def test_the_four_existing_stt_members_keep_their_fields():
    """Criterion 18's field snapshot, **amended by T-0051 criterion 8** — one field, one member.

    T-0007a's version of this row said the four members were byte-identical. T-0051 criterion 8
    adds ``end_estimated`` to ``word`` and to nothing else ("no other field of any contract
    changes"), so the snapshot moves here rather than being deleted: this is the row that says
    the addition was *one* field on *one* member and that the other three were not touched.

    (Named so its red is not read as collateral damage: ``docs/tasks/T-0051.md``'s "pins that
    legitimately move" lists ``tests/unit/test_stt_client*.py`` and the STT fake but not this
    module — the test-author's note to the orchestrator says so.)
    """
    assert {tag: sorted(type(ex).model_fields) for tag, ex in STT_EXAMPLES.items()} == {
        "word": ["end_estimated", "end_ms", "start_ms", "text", "type"],
        "step": ["pause_heads", "t_ms", "type"],
        "end_of_turn": ["p_end", "t_eot_ms", "type"],
        "error": ["message", "type"],
    }, (
        "T-0051 criterion 8: SttWordEvent gains `end_estimated: bool = False` and no other "
        "contract member changes a field"
    )


def test_a_words_end_is_a_measurement_unless_it_says_otherwise():
    """T-0051 criterion 8: the flag is **serialised**, defaults to ``False``, and is a bool.

    Default ``False`` matters beyond tidiness: every producer that is not the STT client — the
    fakes, a replayed log, a future recogniser — keeps meaning "this end was measured", so
    criterion 12's refusal cannot start swallowing ordinary speech because somebody forgot a
    keyword. Serialised matters because the flag has to survive the discriminated-union round trip
    on the stream between the STT client and the endpointer, which is the reader that acts on it
    (criterion 12). **It deliberately stops there**: criterion 8's negative keeps the field off
    ``contracts/events.py``'s ``SttWord``, so the *session log* still cannot say that ``see?``'s
    21 040 ms duration was invented — that plumbing is queued, not done (ruling B39), and this
    docstring used to claim otherwise.
    """
    from local_astra.contracts.speech import SttWordEvent as W

    assert "end_estimated" in W.model_fields, (
        "T-0051 criterion 8: contracts/speech.py must add `end_estimated: bool = False` to "
        f"SttWordEvent; it has {sorted(W.model_fields)}"
    )
    measured = W(text="five", start_ms=1000, end_ms=1250)
    assert measured.end_estimated is False, (
        "the default is False — an end is a measurement unless the producer says it is not"
    )
    assert json.loads(measured.model_dump_json()) == {
        "type": "word",
        "text": "five",
        "start_ms": 1000,
        "end_ms": 1250,
        "end_estimated": False,
    }, "the flag is on the wire, not a Python-side attribute"

    estimated = W(text="see?", start_ms=247200, end_ms=248200, end_estimated=True)
    back = SttEventAdapter.validate_json(estimated.model_dump_json())
    assert back == estimated and back.end_estimated is True, (
        "…and it survives the discriminated union round trip, which is how the endpointer "
        f"receives it from the STT client; got {back!r}"
    )
    with pytest.raises(ValidationError):
        W(text="x", start_ms=0, end_ms=0, end_estimated="maybe")  # a bool, not a free-form value


def test_stt_flushed_event_round_trips_and_is_the_only_new_member():
    """Criterion 18: the ``Marker`` echo travels on the same ordered stream as the words."""
    from local_astra.contracts import speech

    flushed_cls = getattr(speech, "SttFlushedEvent", None)
    assert flushed_cls is not None, (
        "T-0007a criterion 18: contracts/speech.py must add SttFlushedEvent "
        '(type: Literal["flushed"], t_eot_ms: int >= 0) as a member of SttEvent'
    )
    example = flushed_cls(t_eot_ms=1750)
    back = SttEventAdapter.validate_json(example.model_dump_json())
    assert back == example and type(back) is flushed_cls
    assert json.loads(example.model_dump_json()) == {"type": "flushed", "t_eot_ms": 1750}

    members = set(SttEvent.__args__[0].__args__)
    assert members == {
        SttWordEvent,
        SttStepEvent,
        SttEndOfTurnEvent,
        SttErrorEvent,
        flushed_cls,
    }, "the union gains SttFlushedEvent and nothing else"

    with pytest.raises(ValidationError):
        flushed_cls(t_eot_ms=-1)  # t_eot_ms >= 0
    with pytest.raises(ValidationError):
        flushed_cls(t_eot_ms=1, unexpected_field=2)  # strict, like every Contract
    with pytest.raises(ValidationError):
        example.t_eot_ms = 2  # frozen


# --- T-0039 group A: the typed-question event ---------------------------------------------------
#
# Criterion 4 of docs/tasks/T-0039.md:
#
#   UserText(type="user.text", t_ms: int >= 0, text: str)
#
# It is an *event* rather than a session method call so that a typed question is ordered against
# the frames, ticks and microphone onsets already on the one pump (§5, landmine). The pinned list
# of event tags lives in ``tests/unit/test_turn_invariants.py``; this module covers the shape.


def test_the_typed_text_event_round_trips_and_carries_only_a_time_and_a_text():
    """Criterion 4: on the union, with the two fields and no third.

    A ``turn_id`` or an ``onset_ms`` here would make the transport (which builds this event)
    decide something about the turn; the session is what knows.
    """
    from local_astra.contracts import events

    cls = getattr(events, "UserText", None)
    assert cls is not None, (
        'T-0039 criterion 4: contracts/events.py gains UserText(type="user.text", t_ms, text)'
    )
    example = cls(t_ms=1234, text="what is on my screen")
    back = EventAdapter.validate_json(example.model_dump_json())
    assert back == example and type(back) is cls, "criterion 4: it is a member of Event"
    assert json.loads(example.model_dump_json()) == {
        "type": "user.text",
        "t_ms": 1234,
        "text": "what is on my screen",
    }, "criterion 4: the event is its tag, its delivery time and the text"
    assert cls in EVENT_TYPES, (
        "criterion 4: and it is in EVENT_TYPES, so the round-trip and frozen/strict rows at the "
        "top of this module cover it too"
    )
    with pytest.raises(ValidationError):
        cls(t_ms=-1, text="x")  # t_ms >= 0, like every other event
