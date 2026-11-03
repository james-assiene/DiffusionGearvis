"""T-0162 group C, criterion C-1 (and C-4's wire half): ``AstraToolCall`` gains ``step`` and ``result``.

From docs/tasks/T-0162.md C-1:

- ``step: int >= 1 | None``;
- ``result: ToolCallResult | None``, set only on a final message, with ``outcome: Literal["executed", "dry_run",
  "refused", "failed", "cancelled", "timeout"]``, ``hosts: list[str]`` (at most 3 on the wire, host only),
  ``hosts_more: int >= 0`` and ``results: int | None``;
- ``ToolCallStatus`` is **not** widened.

Every verdict is the real ``codec.decode``'s on JSON written out here by hand, or pydantic's on a constructor call;
nothing is compared with a model this module built and then re-read.

Readings fixed here where the task is silent (listed in the test-author's report):

- "set only on a final message" is validated by the contract: a ``result`` on ``proposed`` or ``repairing`` is
  malformed (the four final statuses are ``dry_run``/``executed``/``refused``/``cancelled``, T-0092c);
- "host only" rejects a scheme, a path, a query, a port, userinfo, whitespace, markup and the empty string; it
  accepts dotted lower-case names with digits and hyphens (including punycode). Upper case and IP literals are not
  pinned either way;
- ``results`` and ``step`` default to ``None`` (``codec.encode`` is ``exclude_none``, so this is the only reading under
  which the codec decodes its own output); ``hosts`` and ``hosts_more`` are not pinned as required or defaulted;
- ``ToolCallResult``'s field set is exactly the four C-1 names: no ``engine`` (T-0125 gap 3, not adopted by the
  task) and no ``query``.

No fake or real service is involved (wire contract only), so there is no ``backend`` parametrisation, as in
``test_protocol_codec.py`` and ``test_session_reset_wire.py``. Written pending on ``T-0162`` (group C) and
activated with its implementation; the C-4 rows held on the base and must keep holding.
"""

from __future__ import annotations

import json
import typing
from pathlib import Path
from typing import Any

import pydantic
import pytest

from local_astra.contracts import protocol
from local_astra.contracts.codec import Malformed, decode, encode

REPO = Path(__file__).resolve().parents[2]

OUTCOMES = {"executed", "dry_run", "refused", "failed", "cancelled", "timeout"}
#: reviewer A T2: a dotted name of exactly 253 characters (the DNS cap) and one of 254; every label <= 63
HOST_253 = ".".join(["a" * 63, "b" * 63, "c" * 63, "d" * 61])
HOST_254 = ".".join(["a" * 63, "b" * 63, "c" * 63, "d" * 62])
assert (len(HOST_253), len(HOST_254)) == (253, 254)
STATUSES = {"proposed", "repairing", "dry_run", "executed", "refused", "cancelled"}
FINAL = ("dry_run", "executed", "refused", "cancelled")

BASE: dict[str, Any] = {
    "type": "astra.tool.call",
    "turn_id": "t3",
    "call_id": "t3-call",
    "name": "click_at",
    "arguments": {"x": 500, "y": 250},
    "status": "executed",
    "source": "draft",
    "action": "clicked at (640, 180)",
}
RESULT: dict[str, Any] = {
    "outcome": "executed",
    "hosts": ["shop.example", "cdn.shop.example"],
    "hosts_more": 0,
    "results": None,
}


def result_model() -> Any:
    cls = getattr(protocol, "ToolCallResult", None)
    assert cls is not None, "T-0162 C-1: contracts/protocol.py gains ToolCallResult"
    return cls


def tool_fields() -> set[str]:
    names = set(protocol.AstraToolCall.model_fields)
    assert {"step", "result"} <= names, (
        f"T-0162 C-1: AstraToolCall gains step and result; its fields are {sorted(names)}"
    )
    return names


def wire(msg: dict[str, Any], **over: Any) -> str:
    return json.dumps({**msg, **over})


def decoded(raw: str) -> Any:
    tool_fields()
    return decode(raw)


# =====================================================================================================
# The declaration
# =====================================================================================================


def test_the_two_fields_and_the_result_model_are_declared() -> None:
    names = tool_fields()
    assert names == {
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
        "step",
        "result",
    }, f"T-0162 C-1: exactly two fields are added; {sorted(names)}"
    cls = result_model()
    assert issubclass(cls, protocol.Contract), (
        "C-1: a strict Contract (unknown fields are malformed)"
    )
    assert set(cls.model_fields) == {"outcome", "hosts", "hosts_more", "results"}, (
        f"T-0162 C-1: ToolCallResult has exactly outcome, hosts, hosts_more, results (no engine, no query); "
        f"{sorted(cls.model_fields)}"
    )
    assert protocol.AstraToolCall.model_fields["step"].default is None
    assert protocol.AstraToolCall.model_fields["result"].default is None


def test_the_outcome_literal_is_the_six() -> None:
    cls = result_model()
    for outcome in OUTCOMES:
        assert cls(**{**RESULT, "outcome": outcome}).outcome == outcome
    for bad in ("proposed", "repairing", "error", "Failed", "timed_out", ""):
        with pytest.raises(pydantic.ValidationError):
            cls(**{**RESULT, "outcome": bad})


def test_tool_call_status_is_not_widened() -> None:
    """C-1: ``ToolCallStatus`` stays the six wire statuses; ``failed``/``timeout`` live only in ``result``. Holds on
    the base and must keep holding (not pending)."""
    assert set(typing.get_args(protocol.ToolCallStatus)) == STATUSES
    for bad in ("failed", "timeout"):
        got = decode(wire(BASE, status=bad))
        assert isinstance(got, Malformed), f"C-1: status {bad!r} stays malformed; {got!r}"


# =====================================================================================================
# Decoding hand-written JSON
# =====================================================================================================


@pytest.mark.parametrize("status", FINAL)
def test_a_final_message_with_a_result_decodes(status: str) -> None:
    got = decoded(wire(BASE, status=status, step=2, result=RESULT))
    assert isinstance(got, protocol.AstraToolCall), f"C-1: {got!r}"
    assert got.step == 2
    assert got.result is not None
    assert (
        got.result.outcome,
        list(got.result.hosts),
        got.result.hosts_more,
        got.result.results,
    ) == (
        "executed",
        ["shop.example", "cdn.shop.example"],
        0,
        None,
    )


def test_the_codec_round_trips_a_result_without_a_count() -> None:
    """``codec.encode`` is ``exclude_none``: ``results: None`` leaves the wire, and the message still decodes equal."""
    msg = decoded(wire(BASE, step=1, result=RESULT))
    assert isinstance(msg, protocol.AstraToolCall), msg
    raw = encode(msg)
    assert '"results"' not in raw, raw
    assert decode(raw) == msg


def test_a_search_result_decodes_with_its_count() -> None:
    search = {
        **BASE,
        "name": "google_search",
        "arguments": {"query": "weather in Paris"},
        "status": "cancelled",
        "reason": "timeout",
    }
    got = decoded(wire(search, step=3, result={"outcome": "timeout", "hosts": [], "hosts_more": 0}))
    assert isinstance(got, protocol.AstraToolCall), got
    assert got.result is not None and got.result.outcome == "timeout" and got.result.results is None
    got = decoded(
        wire(
            {**search, "status": "executed"},
            result={"outcome": "executed", "hosts": ["a.example"], "hosts_more": 4, "results": 8},
        )
    )
    assert isinstance(got, protocol.AstraToolCall), got
    assert got.result is not None and (got.result.results, got.result.hosts_more) == (8, 4)


@pytest.mark.parametrize("status", ["proposed", "repairing"])
def test_a_result_on_a_non_final_message_is_malformed(status: str) -> None:
    """C-1: ``result`` is set only on a final message."""
    got = decoded(wire(BASE, status=status, result=RESULT))
    assert isinstance(got, Malformed), f"C-1: a result on {status!r}; decode said {got!r}"
    ok = decoded(wire(BASE, status=status, step=1))
    assert isinstance(ok, protocol.AstraToolCall), (
        f"C-1: {status!r} with a step and no result; {ok!r}"
    )


@pytest.mark.parametrize(
    ("over", "why"),
    [
        ({"step": 0}, "step 0 (>= 1)"),
        ({"step": -1}, "a negative step"),
        ({"result": {**RESULT, "outcome": "failure"}}, "an outcome outside the six"),
        ({"result": {**RESULT, "hosts_more": -1}}, "a negative hosts_more"),
        ({"result": {**RESULT, "hosts": "shop.example"}}, "hosts that are a string"),
        ({"result": {**RESULT, "results": [2]}}, "a list count"),
        ({"result": {**RESULT, "results": -1}}, "a negative count (ruling: results >= 0)"),
        ({"result": {**RESULT, "engine": "brave"}}, "an engine field (gap 3 not adopted)"),
        ({"result": {**RESULT, "query": "weather in Paris"}}, "a query field (no query text)"),
        ({"result": {**RESULT, "thought": "x"}}, "any undeclared field"),
        ({"result": "executed"}, "a result that is a string"),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_bad_fields_are_malformed(over: dict[str, Any], why: str) -> None:
    # on the base every row is malformed because the field is unknown; the named check makes it fail for its reason
    got = decoded(wire(BASE, **over))
    assert isinstance(got, Malformed), f"C-1: {why}; decode said {got!r}"


@pytest.mark.parametrize(
    "hosts",
    [
        ["a.example", "b.example", "c.example", "d.example"],
        ["a.example", "b.example", "c.example", "d.example", "e.example", "f.example"],
    ],
    ids=["four", "six"],
)
def test_more_than_three_hosts_is_malformed(hosts: list[str]) -> None:
    """Adversarial (more than 3 hosts): at most 3 on the wire."""
    got = decoded(wire(BASE, result={**RESULT, "hosts": hosts}))
    assert isinstance(got, Malformed), f"C-1: {len(hosts)} hosts on the wire; decode said {got!r}"
    with pytest.raises(pydantic.ValidationError):
        result_model()(**{**RESULT, "hosts": hosts})
    ok = decoded(wire(BASE, result={**RESULT, "hosts": hosts[:3]}))
    assert isinstance(ok, protocol.AstraToolCall), f"C-1: exactly 3 hosts is legal; {ok!r}"


@pytest.mark.parametrize(
    "host",
    [
        "<img src=x onerror=alert(1)>",
        "evil.example<script>",
        'x.example"',
        "https://evil.example",
        "evil.example/path",
        "evil.example?q=1",
        "evil.example#frag",
        "evil.example:443",
        "user@evil.example",
        "evil example",
        "evil.example\n",
        "",
        HOST_254,  # reviewer A T2: over the 253-character cap
    ],
)
def test_a_host_that_is_not_host_only_is_malformed(host: str) -> None:
    """Adversarial (markup in the hosts; C-1 "host only")."""
    got = decoded(wire(BASE, result={**RESULT, "hosts": ["ok.example", host]}))
    assert isinstance(got, Malformed), f"C-1: host only; {host!r} decoded as {got!r}"


@pytest.mark.parametrize(
    "host",
    ["example.com", "news.bbc.co.uk", "xn--bcher-kva.example", "shop-1.example", "a.b", HOST_253],
)
def test_plain_hosts_are_accepted(host: str) -> None:
    """The too-wide direction of the host guard: ordinary names pass."""
    got = decoded(wire(BASE, result={**RESULT, "hosts": [host]}))
    assert isinstance(got, protocol.AstraToolCall), f"C-1: {host!r} is a host; {got!r}"


# =====================================================================================================
# C-4: a message without the new fields is byte-identical (holds on the base; not pending)
# =====================================================================================================


def test_a_message_without_step_or_result_encodes_as_before() -> None:
    """C-4: ``codec.encode`` of a tool call with neither field set is the base's bytes, written out by hand."""
    msg = protocol.AstraToolCall(
        turn_id="t3",
        call_id="t3-call",
        name="scroll",
        arguments={"direction": "down"},
        status="proposed",
        source="draft",
    )
    assert encode(msg) == (
        '{"type":"astra.tool.call","turn_id":"t3","call_id":"t3-call","name":"scroll",'
        '"arguments":{"direction":"down"},"status":"proposed","source":"draft","replay":false}'
    )


def test_docs_02_section_5_names_the_new_fields() -> None:
    """C-1: docs/02 §5 is updated with the two fields (a prose check: the words must be there)."""
    tool_fields()
    text = (REPO / "docs" / "02-architecture.md").read_text(encoding="utf-8")
    for word in ("ToolCallResult", "hosts_more"):
        assert word in text, f"T-0162 C-1: docs/02-architecture.md §5 mentions {word}"
