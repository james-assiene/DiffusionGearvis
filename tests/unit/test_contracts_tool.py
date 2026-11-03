"""T-0089 criterion 1: the tool-turn contracts (``ToolDraft``, ``LlmDone.tool``, three commands,
the ``tool_repair`` timer, the two state fields and the policy default).

Every expectation is read off ``docs/tasks/T-0089.md`` criterion 1; nothing here is captured from
an implementation. Names the task introduces are looked up with ``need`` so that their absence is
a named failed criterion, not an ``AttributeError`` or a collection error.
"""

from __future__ import annotations

import typing

import pytest
from pydantic import ValidationError

from local_astra.conductor import turn_machine as TM
from local_astra.contracts import commands as C
from local_astra.contracts import events as E
from local_astra.contracts import turn as T

_MISSING = object()


def need(module, name: str, what: str):
    value = getattr(module, name, _MISSING)
    assert value is not _MISSING, f"T-0089 criterion 1: {module.__name__}.{name} ({what})"
    return value


def tool_draft():
    return need(T, "ToolDraft", "call_id, status, name, arguments, reason, source")


ARGS = {"x": 412, "y": 87}


def draft(**over):
    fields = {
        "call_id": "c1",
        "status": "valid",
        "name": "click_at",
        "arguments": ARGS,
        "reason": None,
        "source": "draft",
        **over,
    }
    return tool_draft()(**fields)


# --- ToolDraft validators ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"status": "valid", "source": "draft"},
        {"status": "valid", "source": "repair"},
        {"status": "repairable", "arguments": None, "reason": "schema", "source": "draft"},
        {"status": "refused", "name": None, "arguments": None, "reason": "nameless"},
        {"status": "refused", "name": None, "arguments": None, "reason": "x", "source": "repair"},
    ],
    ids=["valid-draft", "valid-repair", "repairable", "refused-nameless", "refused-repair"],
)
def test_tool_draft_accepts_the_legal_shapes(fields):
    d = draft(**fields)
    for key, value in fields.items():
        assert getattr(d, key) == value


@pytest.mark.parametrize(
    "fields, rule",
    [
        ({"status": "valid", "name": None}, "valid => name is set"),
        ({"status": "valid", "arguments": None}, "valid => arguments are set"),
        ({"status": "repairable", "name": None}, "repairable => name is set"),
        ({"status": "repairable", "source": "repair"}, "repairable => source is draft"),
        ({"status": "maybe"}, "status is valid | repairable | refused"),
        ({"source": "pass2"}, "source is draft | repair"),
    ],
    ids=[
        "valid-no-name",
        "valid-no-args",
        "repairable-no-name",
        "repairable-repair",
        "status",
        "source",
    ],
)
def test_tool_draft_validators_refuse(fields, rule):
    tool_draft()
    with pytest.raises(ValidationError):
        draft(**fields)
        pytest.fail(f"criterion 1: ToolDraft accepted {fields}, but {rule}")


def test_tool_draft_has_exactly_the_criterion_fields():
    assert set(tool_draft().model_fields) == {
        "call_id",
        "status",
        "name",
        "arguments",
        "reason",
        "source",
    }


# --- LlmDone.tool ----------------------------------------------------------------------------------


def test_llm_done_gains_an_optional_tool_defaulting_to_none():
    assert "tool" in E.LlmDone.model_fields, "criterion 1: events.LlmDone gains tool"
    assert E.LlmDone(t_ms=1, request_id="t1-gen").tool is None


@pytest.mark.parametrize("status", ["valid", "repairable", "refused"])
def test_llm_done_with_a_tool_round_trips_through_the_event_adapter(status):
    assert "tool" in E.LlmDone.model_fields, "criterion 1: events.LlmDone gains tool"
    over = {"status": status}
    if status == "refused":
        over |= {"name": None, "arguments": None, "reason": "nameless"}
    ev = E.LlmDone(t_ms=7, request_id="t1-gen", tool=draft(**over))
    back = E.EventAdapter.validate_python(ev.model_dump())
    assert back == ev and isinstance(back, E.LlmDone)
    assert E.EventAdapter.validate_json(E.EventAdapter.dump_json(ev)) == ev
    assert back.tool.call_id == "c1" and back.tool.status == status


# --- commands ---------------------------------------------------------------------------------------


def command_members() -> set[type]:
    return set(typing.get_args(typing.get_args(C.Command)[0]))


def samples():
    return [
        need(C, "ToolRepairStart", "request_id, draft_request_id, call_id")(
            request_id="t1-repair", draft_request_id="t1-gen", call_id="c1"
        ),
        need(C, "ToolExecute", "call_id, turn_id, name, arguments, source")(
            call_id="c1", turn_id="t1", name="click_at", arguments=ARGS, source="repair"
        ),
        need(C, "ToolCancel", "call_id, reason")(call_id="c1", reason="interrupted"),
    ]


def test_the_three_commands_are_members_of_the_command_union():
    for cmd in samples():
        assert type(cmd) in command_members(), f"{type(cmd).__name__} is not in Command"


def test_the_three_commands_round_trip_through_the_command_adapter():
    for cmd in samples():
        assert C.CommandAdapter.validate_python(cmd.model_dump()) == cmd
        assert C.CommandAdapter.validate_json(C.CommandAdapter.dump_json(cmd)) == cmd


def test_the_three_commands_carry_exactly_the_criterion_fields():
    want = {
        "ToolRepairStart": {"request_id", "draft_request_id", "call_id"},
        "ToolExecute": {"call_id", "turn_id", "name", "arguments", "source"},
        "ToolCancel": {"call_id", "reason"},
    }
    for cmd in samples():
        assert set(type(cmd).model_fields) - {"type"} == want[type(cmd).__name__]


REASONS = (
    "interrupted",
    "takeover",
    "error",
    "disconnected",
    "refused",
    "repair_failed",
    "repair_timeout",
)


@pytest.mark.parametrize("reason", REASONS)
def test_tool_cancel_accepts_each_named_reason(reason):
    cls = need(C, "ToolCancel", "call_id, reason")
    assert cls(call_id="c1", reason=reason).reason == reason


@pytest.mark.parametrize("reason", ["timeout", "barge_in", ""])
def test_tool_cancel_refuses_any_other_reason(reason):
    cls = need(C, "ToolCancel", "call_id, reason")
    with pytest.raises(ValidationError):
        cls(call_id="c1", reason=reason)


def test_tool_execute_source_is_draft_or_repair_only():
    cls = need(C, "ToolExecute", "call_id, turn_id, name, arguments, source")
    for source in ("draft", "repair"):
        assert cls(call_id="c", turn_id="t1", name="n", arguments={}, source=source)
    with pytest.raises(ValidationError):
        cls(call_id="c", turn_id="t1", name="n", arguments={}, source="pass2")


def test_tool_repair_is_a_timer_name():
    assert "tool_repair" in typing.get_args(C.TimerName)
    assert C.ArmTimer(name="tool_repair", at_ms=3000).name == "tool_repair"


# --- state and policy --------------------------------------------------------------------------------


def test_machine_state_gains_tool_and_tool_repairs():
    s = T.MachineState()
    assert "tool" in T.MachineState.model_fields, "criterion 1: MachineState.tool"
    assert "tool_repairs" in T.MachineState.model_fields, "criterion 1: MachineState.tool_repairs"
    assert s.tool is None
    assert s.tool_repairs == 0


def test_both_new_fields_are_per_turn_and_reset_to_their_defaults():
    reset = TM._PER_TURN_RESET
    assert "tool" in reset and reset["tool"] is None
    assert "tool_repairs" in reset and reset["tool_repairs"] == 0


def test_policy_repair_timeout_default_is_3000_ms():
    """H12 of T-0080d: 3000 ms."""
    value = getattr(TM.TurnPolicy(), "tool_repair_timeout_ms", _MISSING)
    assert value is not _MISSING, "criterion 1: TurnPolicy.tool_repair_timeout_ms"
    assert value == 3000


def test_the_wire_tags_are_the_ruled_dotted_names():
    """Ruling T3: ``tool.repair_start``, ``tool.execute``, ``tool.cancel``."""
    tags = {type(c).__name__: c.type for c in samples()}
    assert tags == {
        "ToolRepairStart": "tool.repair_start",
        "ToolExecute": "tool.execute",
        "ToolCancel": "tool.cancel",
    }
