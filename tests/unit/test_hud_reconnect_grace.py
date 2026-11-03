"""T-0166 G1-G3: the reconnect grace is 4000 ms and counts only greeted pages.

``docs/tasks/T-0166.md`` (ruling Q1: 4000 ms, strictly above the 3000 ms mount ceiling of T-0165
C5). ``events.ClientDisconnected`` means "the last page went away" (T-0028 c13): it takes the
machine to IDLE, aborts the live turn and cancels the tool loop, so firing it for a reload is the
defect, and never firing it for a page that really left is the opposite one. Both directions are
pinned with 3999/4001 rows.

"Greeted" means the connection sent a hello with the right ``protocol_version`` (G2). A socket that
never greets neither keeps a page alive nor ends one: its ``open()`` does not cancel the timer, its
close neither starts, restarts nor delays it, and a session whose only socket never greeted is
never told a page left (G2f, ruling Q6).

**What is observed.** ``Session.submit`` is wrapped by a spy that records each event and delegates
to the real method (the server's only route to the session for this event), so the count and the
``t_ms`` of ``ClientDisconnected`` are read exactly, on the session's own ``ScriptedClock``
(T-0028 c13: the transport owns no clock).

Existing grace rows live, unedited, in ``test_hud_hello_replay.py``, ``test_hud_endpoint.py`` and
``test_hud_server_reset.py`` (N3); this module adds only the new rows.

Regime: no socket, ``ScriptedClock``, ``tests/fakes/``, no wall time, no seed, n = 1 per row.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from local_astra.conductor.hud_server import RECONNECT_GRACE_MS
from local_astra.contracts import events as E
from local_astra.contracts.codec import encode
from local_astra.contracts.protocol import PROTOCOL_VERSION, AstraHello, AstraSpoken
from tests.unit import test_t0213_settings_wire as t0213  # T-0213 R-5 named re-pin
from tests.unit.test_hud_hello_replay import LIVE_HEAD, greet, make_server, publish_a_turn

REPO = Path(__file__).resolve().parents[2]
PANELS_JS = REPO / "hud" / "src" / "panels.js"

#: Ruling Q1 (orchestrator, 2026-10-27): the grace, written down by hand rather than read back
GRACE_MS = 4000
#: t0 is not zero, so a deadline computed from the wrong origin cannot pass by accident
T0 = 700


class Departures:
    """``session.submit`` spied: every event recorded, then handed to the real method."""

    def __init__(self, rig: Any) -> None:
        self.events: list[Any] = []
        real = rig.session.submit

        def submit(event: Any) -> None:
            self.events.append(event)
            real(event)

        rig.session.submit = submit

    def times(self) -> list[int]:
        return [e.t_ms for e in self.events if isinstance(e, E.ClientDisconnected)]


def setup(tmp_path: Path) -> tuple[Any, Any, Departures]:
    rig, server = make_server(tmp_path)
    departures = Departures(rig)
    rig.clock.advance(T0)
    return rig, server, departures


def advance_to(rig: Any, at_ms: int) -> None:
    assert at_ms >= rig.clock.now_ms, f"the row's own script runs backwards to {at_ms}"
    rig.clock.advance(at_ms - rig.clock.now_ms)


def assert_none_until_then_one(rig: Any, departures: Departures, what: str) -> None:
    """None at t0 + 3999, exactly one by t0 + 4001 (and still one long after)."""
    advance_to(rig, T0 + GRACE_MS - 1)
    assert departures.times() == [], (
        f"{what}: ClientDisconnected was submitted before the 4000 ms grace ran out "
        f"(at {[t - T0 for t in departures.times()]} ms after t0)"
    )
    advance_to(rig, T0 + GRACE_MS + 1)
    assert len(departures.times()) == 1, (
        f"{what}: exactly one ClientDisconnected by t0 + 4001; got "
        f"{[t - T0 for t in departures.times()]} ms after t0"
    )
    advance_to(rig, T0 + 5 * GRACE_MS)
    assert len(departures.times()) == 1, (
        f"{what}: and only one; got {[t - T0 for t in departures.times()]} ms after t0"
    )


def hello(connection: Any, protocol_version: int = PROTOCOL_VERSION) -> None:
    connection.receive(encode(AstraHello(protocol_version=protocol_version)))


# --- G1 ----------------------------------------------------------------------------------------


def test_the_grace_is_four_seconds_and_covers_the_mount_ceiling():
    """G1 (ruling Q1): 4000 ms, strictly above ``MOUNT_CEILING_MS`` parsed from the page (T-0165
    C5), so a reload that falls back to the mount ceiling still finds its journal."""
    source = PANELS_JS.read_text()
    found = re.findall(r"export\s+const\s+MOUNT_CEILING_MS\s*=\s*(\d+)\s*;", source)
    assert len(found) == 1, f"hud/src/panels.js must declare MOUNT_CEILING_MS once; found {found}"
    ceiling = int(found[0])
    assert RECONNECT_GRACE_MS == GRACE_MS, (
        f"G1 / ruling Q1: RECONNECT_GRACE_MS is 4000 ms; it is {RECONNECT_GRACE_MS}"
    )
    assert ceiling < RECONNECT_GRACE_MS, (
        f"G1: the grace ({RECONNECT_GRACE_MS} ms) must exceed the page's mount ceiling "
        f"({ceiling} ms, hud/src/panels.js), or a slow reload loses its turn"
    )


async def test_disconnect_fires_once_after_the_grace(tmp_path):
    """G1: one greeted page closes at t0: none at t0 + 3999, exactly one by t0 + 4001, its
    ``t_ms`` on the session clock inside that window (T-0028 c13)."""
    rig, server, departures = setup(tmp_path)
    page = greet(server)
    page.close()
    assert_none_until_then_one(rig, departures, "G1")
    (at,) = departures.times()
    assert T0 + GRACE_MS - 1 < at <= T0 + GRACE_MS + 1, (
        f"G1: the event's t_ms is on the session clock at the deadline; it was t0 + {at - T0}"
    )


# --- G2 ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("ungreeted", ["stays_open", "closes_at_2000"])
async def test_a_socket_that_never_says_hello_does_not_extend_the_grace(tmp_path, ungreeted):
    """G2d, adversarial: a greeted page closes at t0; a socket that never says hello opens at
    t0 + 1000 and either stays open or closes at t0 + 2000. Either way the grace fires at its
    original deadline: none before t0 + 3999, exactly one by t0 + 4001.

    The two named mutations: ``open()`` cancelling the timer (stays_open never fires) and an
    ungreeted close restarting it (closes_at_2000 fires at t0 + 6000)."""
    rig, server, departures = setup(tmp_path)
    greet(server).close()
    advance_to(rig, T0 + 1000)
    silent = server.open()
    if ungreeted == "closes_at_2000":
        advance_to(rig, T0 + 2000)
        silent.close()
    assert_none_until_then_one(rig, departures, f"G2d ({ungreeted})")


async def test_an_ungreeted_socket_does_not_hide_the_last_greeted_departure(tmp_path):
    """G2c: the last greeted page closes while an ungreeted socket stays open; the timer starts
    anyway and fires at its deadline."""
    rig, server, departures = setup(tmp_path)
    silent = server.open()
    page = greet(server)
    page.close()
    assert_none_until_then_one(rig, departures, "G2c")
    assert not silent.closed, "G2c: the ungreeted socket itself is left alone"


async def test_open_alone_does_not_cancel_the_grace_but_hello_does(tmp_path):
    """G2a, both directions. (1) An ``open()`` at t0 + 1000 followed by a valid hello at
    t0 + 3900 is a reconnect: no event, ever. (2) An ``open()`` at t0 + 1000 with no hello is not:
    one event by t0 + 4001."""
    rig, server, departures = setup(tmp_path)
    greet(server).close()
    advance_to(rig, T0 + 1000)
    back = server.open()
    advance_to(rig, T0 + 3900)
    hello(back)
    advance_to(rig, T0 + 5 * GRACE_MS)
    assert departures.times() == [], (
        "G2a, too wide: a valid hello inside the grace is a reconnect and cancels the timer; "
        f"ClientDisconnected was submitted at t0 + {[t - T0 for t in departures.times()]}"
    )

    rig2, server2, departures2 = setup(tmp_path / "second")
    greet(server2).close()
    advance_to(rig2, T0 + 1000)
    server2.open()
    assert_none_until_then_one(rig2, departures2, "G2a (open alone)")


async def test_a_never_greeted_socket_closing_reports_nothing(tmp_path):
    """G2f (ruling Q6, a named behaviour change): a session whose only socket never greeted
    submits no ``ClientDisconnected`` when that socket closes, because no page ever left."""
    rig, server, departures = setup(tmp_path)
    server.open().close()
    advance_to(rig, T0 + 5 * GRACE_MS)
    assert departures.times() == [], (
        "G2f: no page ever greeted, so none left; ClientDisconnected was submitted at "
        f"t0 + {[t - T0 for t in departures.times()]}"
    )


async def test_a_refused_hello_does_not_cancel_the_grace(tmp_path):
    """G2e: a hello refused for ``protocol_version`` leaves the connection ungreeted, so it does
    not cancel the timer; the valid-hello companion is G2a (1)."""
    rig, server, departures = setup(tmp_path)
    greet(server).close()
    advance_to(rig, T0 + 1000)
    wrong = server.open()
    advance_to(rig, T0 + 1500)
    hello(wrong, PROTOCOL_VERSION + 1)
    assert server.counts["protocol_version"] == 1, "premise: the hello was refused"
    assert_none_until_then_one(rig, departures, "G2e")


# --- G3: departure semantics are otherwise unchanged ---------------------------------------------


async def test_a_greeted_reconnect_inside_the_grace_replays_and_a_later_departure_clears(tmp_path):
    """G3: a greeted reconnect at t0 + 3500 still replays the open turn and no event follows; when
    that page then leaves for good, the journal is cleared as the grace fires (T-0037 c4), so the
    next hello gets the live head only."""
    rig, server, departures = setup(tmp_path)
    publish_a_turn(server)
    greet(server).close()
    advance_to(rig, T0 + 3500)
    back = greet(server)
    replayed = [m for m in back.sent() if isinstance(m, AstraSpoken) and m.replay]
    assert len(replayed) == 2, (
        "G3: a greeted reconnect inside the 4000 ms grace replays the open turn's two lines; it "
        f"got {[m.type for m in back.sent()]}"
    )
    advance_to(rig, T0 + 5 * GRACE_MS)
    assert departures.times() == [], "G3: a reconnect inside the grace is not a departure"

    t1 = rig.clock.now_ms
    back.close()
    advance_to(rig, t1 + GRACE_MS - 1)
    assert departures.times() == [], "G3: none before the grace ran out"
    advance_to(rig, t1 + GRACE_MS + 1)
    assert len(departures.times()) == 1, f"G3: one departure; got {departures.times()}"
    reply = greet(server).sent()
    # T-0213 R-5 named re-pin: then the one settings state, last
    assert [type(m) for m in reply] == t0213.hello_with_settings_last(LIVE_HEAD), (
        f"G3 / T-0037 c4: the journal is cleared when the grace fires; got {[m.type for m in reply]}"
    )
