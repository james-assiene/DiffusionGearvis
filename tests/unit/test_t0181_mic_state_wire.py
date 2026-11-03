"""T-0181 group A, A2, A4 and A9: the session owns the mic switch, the hello carries it, and it survives a reset.

The rig is ``tests/unit/_t0181_rig.mic_rig``: the session over the fakes (tools off unless a row says otherwise), a
socketless ``HudServer`` and one greeted page ``rig.page``. Mutes go over the wire as raw JSON, so before A1 lands the
codec answers ``Unknown`` and the server counts the message ``unhandled``: every row fails on a named assertion, never
at collection. A session keyword not yet accepted is a named assertion in the rig (A-R1).

**Oracles.** The hello's BASE shape is written here by hand from ``hud_server.hello_reply`` at main 74ae8b25 and was
confirmed by running this rig on that tree (``_t0181_rig.BASE_FRESH_HELLO``; ``BASE_AFTER_TURN`` and
``BASE_AFTER_RESET`` below). Whether every *other* message is byte-identical to BASE is
``test_t0181_differential.py``'s row; this module pins the order and the count.

**Readings** (listed in the test-author's report):

- A4's "fourth message" counts from ``astra.assist.state``: with a replayed reset it sits fifth on the wire (A4: "the
  T-0162 replayed reset still comes first"), always right after ``astra.log.state``;
- A2's "redundant ones included": a mute equal to the current state still publishes one state to every connection
  and writes no record (A2's own words); the record is a ``note`` with data exactly ``{what: "mic.mute", from, to}``
  (A-R2, as T-0177's ``vision.select``), ``from``/``to`` the booleans;
- A2's "with no mic, the toggle changes nothing": in a no-mic session both values (``true`` and ``false``) answer the
  sender with ``astra.error(code="mic_unavailable")``, publish no state, write no ``mic.mute`` note, and the frames the
  STT is fed (when a capture seam exists but the session was told it has no mic) stay the captured frames;
- A9's "the mute state survives ``SessionReset``" is checked three ways: the hello says muted, the frames fed to the
  STT after the reset are still all-zero (the gate is the state's truth, "Silent self-disable rule"), and an unmute
  after the reset is a *change* (one ``mic.mute`` note ``True -> False``), so a reset that rebuilt the switch as
  unmuted while replaying a stale state would fail.

Regime: CPU over fakes, ``ScriptedClock``, thinking on, ``FakeAudioIO`` capture (80 ms STT frames), reactive
``FakeStt`` with no script, tools off, n = 1, deterministic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.unit import test_t0213_settings_wire as t0213  # T-0213 R-5 named re-pin
from tests.unit._t0181_rig import (
    BASE_FRESH_HELLO,
    MUTE_NOTE,
    NO_SEAM,
    REASON_MIC_NONE,
    REASON_NO_MIC,
    REASON_SILENT,
    STATE,
    UNAVAILABLE,
    mic_rig,
    mic_states,
    told,
    triple,
    types,
)
from tests.unit._tool_turn_support import answer

#: BASE (main 74ae8b25) hello reply of a page greeted after one answered typed turn, tools off (this rig, run on
#: that tree): the three-message head, the turn's journal (transcript, spoken, played), the repeated turn state,
#: then T-0177 B9's replayed vision frame
BASE_AFTER_TURN = [
    "astra.assist.state",
    "astra.turn.state",
    "astra.log.state",
    "astra.transcript",
    "astra.spoken",
    "astra.spoken.played",
    "astra.turn.state",
    "astra.vision.frame",
]
#: BASE hello reply after a session reset with no turn since: T-0162's replayed reset, then the head
BASE_AFTER_RESET = ["astra.session.cleared", *BASE_FRESH_HELLO]

LIVE = (False, True, None)
MUTED = (True, True, None)


def without_mic(messages: list[Any]) -> list[str]:
    return [t for t in types(messages) if t != STATE]


async def settle(rig: Any, ms: int = 400) -> None:
    await rig.advance(ms, step=40)


# =====================================================================================================
# A2: the fence, the route, the echo, the record
# =====================================================================================================


async def test_a_mute_before_hello_is_fenced(tmp_path: Path) -> None:
    """A2 "behind the hello fence": a mute on an ungreeted page is counted ``no_hello`` (not ``unhandled``, not
    routed), its sender is told ``no_hello``, nothing is published, and the switch is unchanged (that page's hello
    then reports it live)."""
    async with mic_rig(tmp_path) as rig:
        page = rig.server.open()
        before = dict(rig.server.counts)
        published = len(rig.published.messages)
        rig.send_mute(True, page)
        await rig.pump()
        after = dict(rig.server.counts)
        assert (after["no_hello"], after["unhandled"], after["routed"]) == (
            before["no_hello"] + 1,
            before["unhandled"],
            before["routed"],
        ), f"A2: the mute is fenced by the hello (no_hello +1, not unhandled, not routed); {after}"
        assert told(page) == [("astra.error", "no_hello")], told(page)
        assert mic_states(rig.published.messages[published:]) == [], "A2: nothing published"
        page.receive('{"type":"astra.hello","protocol_version":1,"client":"hud"}')
        assert [triple(m) for m in mic_states(page.sent())] == [LIVE], (
            f"A2/A4: the fenced mute changed nothing; the hello says live; {types(page.sent())}"
        )


async def test_route_hands_the_mute_to_the_session(tmp_path: Path) -> None:
    """A2: ``hud_server.route`` hands ``astra.mic.mute`` to ``session.apply_mic_mute(message)``; a spy that delegates
    sees one call per routed mute, carrying the page's ``muted``."""
    async with mic_rig(tmp_path) as rig:
        real = getattr(rig.session, "apply_mic_mute", None)
        assert callable(real), "T-0181 A2: Session.apply_mic_mute does not exist yet"
        seen: list[Any] = []

        def spy(message: Any) -> Any:
            seen.append(message)
            return real(message)

        rig.session.apply_mic_mute = spy
        await rig.mute(True)
        await rig.mute(False)
    assert [getattr(m, "muted", None) for m in seen] == [True, False], seen
    assert all(getattr(m, "type", None) == "astra.mic.mute" for m in seen), seen


async def test_every_routed_mute_is_echoed_to_every_connection_redundant_ones_included(
    tmp_path: Path,
) -> None:
    """A2 (T-0043 echo): with a mic, every routed mute publishes exactly one ``astra.mic.state`` to every greeted
    connection, the sender included and the redundant ones included, carrying the switch after the mute.

    Page A sends true, true (redundant), false, false (redundant); page B then sends true. Each page receives five
    states, in that order; every one is available with no reason. The publisher saw five, one per mute."""
    async with mic_rig(tmp_path) as rig:
        other = rig.greet()
        marks = (len(rig.page.sent()), len(other.sent()))
        published = len(rig.published.messages)
        for muted in (True, True, False, False):
            await rig.mute(muted)
        await rig.mute(True, page=other)
        got = (
            mic_states(rig.page.sent()[marks[0] :]),
            mic_states(other.sent()[marks[1] :]),
        )
        out = mic_states(rig.published.messages[published:])
    want = [MUTED, MUTED, LIVE, LIVE, MUTED]
    assert [triple(m) for m in got[0]] == want, f"A2: the sender's echoes; {got[0]}"
    assert [triple(m) for m in got[1]] == want, f"A2: every connection gets each echo; {got[1]}"
    assert len(out) == 5, f"A2: one published state per routed mute; {len(out)}"


async def test_a_record_only_when_the_switch_changes(tmp_path: Path) -> None:
    """A2 as read by A-R2: a change writes one ``note`` with data exactly ``{what: "mic.mute", from, to}``; a
    redundant toggle writes none. Over false (redundant at start), true, true, false, false that is two notes. No
    record of another kind carries ``what: "mic.mute"`` and the log refused no unknown kind."""
    async with mic_rig(tmp_path) as rig:
        for muted in (False, True, True, False, False):
            await rig.mute(muted)
    notes = rig.notes(MUTE_NOTE)
    assert [(d.get("from"), d.get("to")) for d in notes] == [(False, True), (True, False)], notes
    assert all(set(d) == {"what", "from", "to"} for d in notes), notes
    others = [
        r
        for r in rig.log_records
        if r.get("kind") != "note" and (r.get("data") or {}).get("what") == MUTE_NOTE
    ]
    assert others == [], f"A-R2: only note records carry what=mic.mute; {others}"
    assert rig.kind(MUTE_NOTE) == [], "A-R2: no record of kind mic.mute"


NO_MIC = [
    pytest.param({"mic": False}, NO_SEAM, id="no-capture-seam"),
    pytest.param({"mic": True, "mic_unavailable": REASON_NO_MIC}, REASON_NO_MIC, id="no-mic"),
    pytest.param({"mic": True, "mic_unavailable": REASON_MIC_NONE}, REASON_MIC_NONE, id="mic-none"),
    pytest.param({"mic": False, "mic_unavailable": REASON_SILENT}, REASON_SILENT, id="silent"),
]


@pytest.mark.parametrize(("kwargs", "reason"), NO_MIC)
async def test_without_a_mic_a_mute_answers_the_sender_only(
    tmp_path: Path, kwargs: dict[str, Any], reason: str
) -> None:
    """A2/A3: with no mic (no capture seam, or the A-R1 keyword's reason), a mute of either value is routed (not
    ``unhandled``), answers its sender with exactly one ``astra.error(code="mic_unavailable")``, tells the other page
    nothing, publishes no state and writes no ``mic.mute`` note; a later hello reports
    ``(muted=false, available=false, reason)`` with the exact reason. Where capture exists, every frame fed to the STT
    after the toggles is still the captured frame (non-zero: ``FakeAudioIO``'s pattern is never silence)."""
    async with mic_rig(tmp_path, **kwargs) as rig:
        other = rig.greet()
        await settle(rig)
        marks = (len(rig.page.sent()), len(other.sent()))
        published = len(rig.published.messages)
        before = dict(rig.server.counts)
        fed_before = len(rig.rec_stt.fed)
        for muted in (True, False):
            rig.send_mute(muted)
            await rig.pump()
        await settle(rig)
        assert rig.server.counts["unhandled"] == before["unhandled"], "A2: the mute is routed"
        assert told(rig.page, marks[0]) == [("astra.error", UNAVAILABLE)] * 2, told(
            rig.page, marks[0]
        )
        assert other.sent()[marks[1] :] == [], "A2: the error goes to the sender only"
        assert mic_states(rig.published.messages[published:]) == [], "A2: no state is published"
        third = rig.greet()
        (state,) = mic_states(third.sent())
        assert triple(state) == (False, False, reason), triple(state)
        if kwargs["mic"]:
            after = rig.rec_stt.zero()[fed_before:]
            assert after and not any(after), (
                f"A2: with no mic the toggle changes nothing; {sum(after)} of {len(after)} fed frames were zeroed"
            )
        else:
            assert rig.rec_stt.fed == [], "premise: a session with no capture seam feeds nothing"
    assert rig.notes(MUTE_NOTE) == [], "A2: no record without a mic"


@pytest.mark.parametrize(("kwargs", "reason"), NO_MIC)
async def test_without_a_mic_the_senders_error_message_is_the_reason(
    tmp_path: Path, kwargs: dict[str, Any], reason: str
) -> None:
    """docs/02 §5 (T-0181 spec delta; A-RV3): with no microphone only the sender gets
    ``astra.error(code="mic_unavailable")``, "whose message is the reason": for each toggle value the error's
    ``message`` is exactly the session's unavailable reason (A3's verbatim strings, or the no-seam reason), never the
    bare code."""
    async with mic_rig(tmp_path, **kwargs) as rig:
        await settle(rig)
        mark = len(rig.page.sent())
        for muted in (True, False):
            rig.send_mute(muted)
            await rig.pump()
        await settle(rig)
        errors = [m for m in rig.page.sent()[mark:] if getattr(m, "type", None) == "astra.error"]
    assert [(m.code, m.message) for m in errors] == [(UNAVAILABLE, reason)] * 2, (
        f"docs/02 §5: the sender's mic_unavailable error carries the reason {reason!r} as its message; got "
        f"{[(m.code, m.message) for m in errors]}"
    )


# =====================================================================================================
# A4: the hello
# =====================================================================================================


@pytest.mark.parametrize(
    ("kwargs", "want"),
    [
        pytest.param({"mic": True}, LIVE, id="mic"),
        pytest.param({"mic": False}, (False, False, NO_SEAM), id="no-mic"),
    ],
)
async def test_the_hello_carries_the_state_fourth_right_after_log_state(
    tmp_path: Path, kwargs: dict[str, Any], want: tuple[Any, Any, Any]
) -> None:
    """A4: the rig's own first page and a later one are each greeted with BASE's three messages and then
    ``astra.mic.state``, exactly four messages before any turn, in mic and no-mic sessions alike."""
    async with mic_rig(tmp_path, **kwargs) as rig:
        # T-0213 R-5 named re-pin: the one settings state, asserted last, set aside
        first = t0213.hello_without_settings_state(rig.page.sent())
        later = t0213.hello_without_settings_state(rig.greet().sent())
    for reply in (first, later):
        assert types(reply) == [*BASE_FRESH_HELLO, STATE], types(reply)
        assert triple(reply[3]) == want, triple(reply[3])


async def test_the_hello_reports_the_current_switch(tmp_path: Path) -> None:
    """A4: a page greeted after a mute is told ``muted: true``; after the unmute, ``muted: false``."""
    async with mic_rig(tmp_path) as rig:
        await rig.mute(True)
        muted = mic_states(rig.greet().sent())
        await rig.mute(False)
        live = mic_states(rig.greet().sent())
    assert [triple(m) for m in muted] == [MUTED], muted
    assert [triple(m) for m in live] == [LIVE], live


async def test_after_a_turn_the_state_still_sits_after_log_state_and_nothing_else_moves(
    tmp_path: Path,
) -> None:
    """A4: after one answered typed turn, the hello is BASE's reply with exactly one ``astra.mic.state`` inserted right
    after ``astra.log.state`` (so the journal and T-0177's replayed vision frame stay after it, the frame last)."""
    async with mic_rig(tmp_path, script=[answer("A shop. ")]) as rig:
        await rig.type_question("what is on my screen")
        await rig.turn_done()
        await settle(rig, 300)
        # T-0213 R-5 named re-pin: the one settings state, asserted last, set aside
        reply = t0213.hello_without_settings_state(rig.greet().sent())
    got = types(reply)
    assert got.count(STATE) == 1, got
    assert got[2:4] == ["astra.log.state", STATE], got
    assert without_mic(reply) == BASE_AFTER_TURN, got
    assert got[-1] == "astra.vision.frame", "A4: T-0177's messages still come last"


async def test_in_live_the_vision_messages_still_come_last(tmp_path: Path) -> None:
    """A4: in a live session (T-0177's ``vision_rig``: live kit, ``FakeAudioIO`` capture, so a mic is available)
    the hello carries ``astra.mic.state`` right after ``astra.log.state`` and still ends with ``astra.vision.state``."""
    from tests.unit.test_t0177_shared_frame import vision_rig

    async with vision_rig(tmp_path) as rig:
        # T-0213 R-5 named re-pin: the one settings state, asserted last, set aside
        reply = t0213.hello_without_settings_state(rig.page.sent())
    got = types(reply)
    assert got.count(STATE) == 1, f"A4: the live hello carries one astra.mic.state; {got}"
    i = got.index("astra.log.state")
    assert got[i + 1] == STATE, got
    assert got[-1] == "astra.vision.state", f"A4: T-0177's vision state is still last; {got}"
    assert triple(reply[i + 1]) == LIVE, triple(reply[i + 1])


# =====================================================================================================
# A9: reset and reconnect
# =====================================================================================================


async def test_the_mute_survives_a_reset(tmp_path: Path) -> None:
    """A9 (H-T181-3): muted, then a ``SessionReset``. A page greeted afterwards gets BASE's post-reset reply with
    ``astra.mic.state(muted=true)`` right after ``astra.log.state`` (the replayed reset still first); the frames fed to
    the STT after the reset are still all zeros; and an unmute after the reset is a change (one ``True -> False``
    note after the pre-reset ``False -> True``)."""
    async with mic_rig(tmp_path) as rig:
        await rig.mute(True)
        await settle(rig)
        await rig.reset()
        fed_mark = len(rig.rec_stt.fed)
        await settle(rig, 800)
        # T-0213 R-5 named re-pin: the one settings state, asserted last, set aside
        reply = t0213.hello_without_settings_state(rig.greet().sent())
        after_reset = rig.rec_stt.zero()[fed_mark:]
        await rig.mute(False)
    got = types(reply)
    assert got[0] == "astra.session.cleared", "premise: T-0162's replayed reset leads"
    assert without_mic(reply) == BASE_AFTER_RESET, got
    assert got[4:5] == [STATE], f"A4/A9: the state is fourth after the reset; {got}"
    assert triple(reply[4]) == MUTED, triple(reply[4])
    assert after_reset and all(after_reset), (
        f"A9: the gate survives the reset; {after_reset.count(False)} of {len(after_reset)} frames fed live"
    )
    notes = rig.notes(MUTE_NOTE)
    assert [(d.get("from"), d.get("to")) for d in notes] == [(False, True), (True, False)], notes


async def test_the_state_survives_a_reconnect(tmp_path: Path) -> None:
    """A2/A4 "across a reconnect": the muting page goes away; a page that connects and greets afterwards is told
    ``muted: true``, and its own unmute is echoed to it."""
    async with mic_rig(tmp_path) as rig:
        await rig.mute(True)
        rig.page.close()
        await rig.pump()
        page = rig.greet()
        hello = mic_states(page.sent())
        mark = len(page.sent())
        await rig.mute(False, page=page)
        echo = mic_states(page.sent()[mark:])
    assert [triple(m) for m in hello] == [MUTED], hello
    assert [triple(m) for m in echo] == [LIVE], echo
