"""T-0128c A6: canvas frames are droppable in the HUD transport and never cost a connection.

Finding 2 of the task: ``Outbox.offer`` could drop only ``astra.turn.state``, so a canvas flood
overflowed the 512-message outbox and closed the page's connection mid-turn. After A6:

* an ``astra.canvas.*`` message never closes a connection;
* at most ``8`` canvas messages are queued per connection;
* a newer denoise frame of the same request replaces a queued one (a commit is never replaced);
* overflow is counted in ``Outbox.dropped``;
* a non-canvas message is never refused because canvas frames filled the queue;
* spoken lines are still never dropped: a spoken overflow still closes (adversarial, unmarked --
  it holds today and must keep holding);
* ``TurnJournal`` still ignores ``astra.canvas.*`` (unmarked, holds today).

Rows that hold before the change are regression rows, not pending ones (a pending row that passes
is XPASS(strict)). Expected counts are by hand. Regime: no socket (``server.open()``), the T-0028
rig, ``ScriptedClock``.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_astra.conductor.hud_server import DEFAULT_OUTBOX, HudServer
from local_astra.contracts.protocol import (
    AstraCanvasStep,
    AstraCanvasSynthetic,
    AstraSpoken,
    AstraTurnState,
)
from local_astra.contracts.turn import TurnState
from tests.fakes.canvas_tap import commit, denoise
from tests.unit._hud_rig import Recorder, running

#: A6: at most this many canvas messages queued per connection.
CANVAS_CAP = 8


class _Sink:
    def ingest(self, message: Any) -> None:
        return None


def step(rid: str, canvas_idx: int = 0, s: int = 1) -> AstraCanvasStep:
    return AstraCanvasStep.model_validate(denoise(rid, canvas_idx, s))


def committed(rid: str, canvas_idx: int = 0, s: int = 5) -> AstraCanvasStep:
    return AstraCanvasStep.model_validate(commit(rid, canvas_idx, s))


def spoken(i: int) -> AstraSpoken:
    return AstraSpoken(turn_id="t1", utterance_id=f"u{i}", text=f"line {i}")


def canvas_in(conn: Any) -> list[Any]:
    return [m for m in conn.outbox.pending() if m.type.startswith("astra.canvas.")]


def spoken_in(conn: Any) -> list[str]:
    return [m.utterance_id for m in conn.outbox.pending() if isinstance(m, AstraSpoken)]


async def test_512_canvas_frames_then_a_spoken_line_keep_the_connection_open(tmp_path):
    """A6: 512 denoise frames of 512 requests (no replacement possible), then a spoken line.

    By hand: 8 canvas queued, 504 dropped, the spoken line queued, the connection open.
    """
    assert DEFAULT_OUTBOX == 512
    async with running(tmp_path, publish=Recorder()) as rig:
        server = HudServer(rig.session, frame_sink=_Sink(), port=0)
        conn = server.open()
        for i in range(512):
            server.publish(step(f"r{i}"))
        server.publish(spoken(1))
        assert not conn.closed, f"A6: a canvas flood closed the connection ({conn.close_reason})"
        assert spoken_in(conn) == ["u1"], "A6: the spoken line after the flood is queued"
        assert len(canvas_in(conn)) == CANVAS_CAP, len(canvas_in(conn))
        assert conn.outbox.dropped == 512 - CANVAS_CAP, conn.outbox.dropped


async def test_at_most_eight_canvas_messages_are_ever_queued(tmp_path):
    """A6: canvas frames of 20 requests interleaved with spoken lines; after every offer the queue
    holds <= 8 canvas messages and every spoken line."""
    async with running(tmp_path, publish=Recorder()) as rig:
        server = HudServer(rig.session, frame_sink=_Sink(), port=0)
        conn = server.open()
        for i in range(20):
            server.publish(step(f"r{i}"))
            server.publish(committed(f"q{i}"))
            server.publish(spoken(i))
            assert len(canvas_in(conn)) <= CANVAS_CAP, (i, len(canvas_in(conn)))
        assert spoken_in(conn) == [f"u{i}" for i in range(20)]
        assert not conn.closed
        assert conn.outbox.dropped >= 40 - CANVAS_CAP, conn.outbox.dropped


async def test_a_newer_denoise_frame_of_the_same_request_replaces_the_queued_one(tmp_path):
    """A6: A s1, B s1, A s2 -> A's queued frame is s2 alone; B's s1 stays."""
    async with running(tmp_path, publish=Recorder()) as rig:
        server = HudServer(rig.session, frame_sink=_Sink(), port=0)
        conn = server.open()
        server.publish(step("A", 0, 1))
        server.publish(step("B", 0, 1))
        server.publish(step("A", 0, 2))
        queued = [(m.request_id, m.step) for m in canvas_in(conn)]
        assert sorted(queued) == [("A", 2), ("B", 1)], f"A6: same-request replacement: {queued}"


async def test_a_queued_commit_is_never_replaced_by_a_later_denoise_frame(tmp_path):
    """A6 + A3: commit A/0 queued, then A/1 s1 and A/1 s2 -> [commit A/0, denoise A/1 s2], in that
    order: the commit stays and the denoise frames replace each other only."""
    async with running(tmp_path, publish=Recorder()) as rig:
        server = HudServer(rig.session, frame_sink=_Sink(), port=0)
        conn = server.open()
        server.publish(committed("A", 0, 5))
        server.publish(step("A", 1, 1))
        server.publish(step("A", 1, 2))
        queued = [(m.phase, m.canvas_idx, m.step) for m in canvas_in(conn)]
        assert queued == [("commit", 0, 5), ("denoise", 1, 2)], queued


async def test_a_spoken_line_is_never_refused_because_canvas_frames_filled_the_queue(tmp_path):
    """A6 adversarial: 504 spoken + 8 canvas = a full 512 queue; spoken 505 is accepted (a canvas
    frame goes, counted), the connection stays open."""
    async with running(tmp_path, publish=Recorder()) as rig:
        server = HudServer(rig.session, frame_sink=_Sink(), port=0)
        conn = server.open()
        for i in range(504):
            server.publish(spoken(i))
        for i in range(CANVAS_CAP):
            server.publish(step(f"r{i}"))
        assert len(conn.outbox) == 512 and conn.outbox.dropped == 0
        server.publish(spoken(504))
        assert not conn.closed, f"A6: {conn.close_reason}"
        assert spoken_in(conn) == [f"u{i}" for i in range(505)]
        assert len(canvas_in(conn)) == CANVAS_CAP - 1
        assert conn.outbox.dropped == 1


async def test_a_spoken_line_is_accepted_in_a_small_outbox_full_of_canvas(tmp_path):
    """A6: a 4-message outbox holding 4 canvas frames still takes a spoken line."""
    async with running(tmp_path, publish=Recorder()) as rig:
        server = HudServer(rig.session, frame_sink=_Sink(), port=0, outbox_size=4)
        conn = server.open()
        for i in range(4):
            server.publish(step(f"r{i}"))
        server.publish(spoken(1))
        assert not conn.closed, conn.close_reason
        assert spoken_in(conn) == ["u1"]


@pytest.mark.parametrize("kind", ["step", "synthetic"])
async def test_a_canvas_message_into_a_queue_full_of_spoken_lines_is_dropped_not_fatal(
    tmp_path, kind
):
    """A6: 512 spoken lines, then an ``astra.canvas.*`` -> dropped (counted), connection open, all
    512 lines still queued."""
    message = step("A") if kind == "step" else AstraCanvasSynthetic(request_id="A", reason="x")
    async with running(tmp_path, publish=Recorder()) as rig:
        server = HudServer(rig.session, frame_sink=_Sink(), port=0)
        conn = server.open()
        for i in range(512):
            server.publish(spoken(i))
        server.publish(message)
        assert not conn.closed, f"A6: an astra.canvas.* closed the connection: {conn.close_reason}"
        assert conn.outbox.dropped == 1
        assert len(spoken_in(conn)) == 512


async def test_an_outbox_full_of_spoken_lines_still_closes_on_a_spoken_overflow(tmp_path):
    """A6 adversarial (regression, holds today): the canvas rule must not make spoken lines
    droppable -- spoken 513 into 512 spoken lines closes the connection."""
    async with running(tmp_path, publish=Recorder()) as rig:
        server = HudServer(rig.session, frame_sink=_Sink(), port=0)
        conn = server.open()
        for i in range(512):
            server.publish(spoken(i))
        assert not conn.closed
        server.publish(spoken(512))
        assert conn.closed, "a spoken line is never dropped: overflow closes the connection"
        assert "u512" not in spoken_in(conn)


async def test_spoken_overflow_still_closes_with_canvas_frames_also_queued(tmp_path):
    """A6 adversarial (regression): 8 canvas + 504 spoken, then 8 more spoken -- evicting the 8
    canvas frames makes room for 8; the ninth extra spoken line still closes."""
    async with running(tmp_path, publish=Recorder()) as rig:
        server = HudServer(rig.session, frame_sink=_Sink(), port=0)
        conn = server.open()
        for i in range(CANVAS_CAP):
            server.publish(step(f"r{i}"))
        for i in range(504 + CANVAS_CAP + 1):
            server.publish(spoken(i))
        assert conn.closed, "A6: spoken lines are never dropped, so 513 of them close the page"


async def test_the_turn_journal_still_ignores_canvas_messages(tmp_path):
    """A6 (regression, holds today): a page that connects after canvas frames is not replayed any."""
    async with running(tmp_path, publish=Recorder()) as rig:
        server = HudServer(rig.session, frame_sink=_Sink(), port=0)
        server.publish(AstraTurnState(state=TurnState.THINKING, turn_id="t1"))
        server.publish(step("A"))
        server.publish(committed("A"))
        server.publish(spoken(1))
        reply = list(server.hello_reply())
        assert not [m for m in reply if m.type.startswith("astra.canvas.")], [m.type for m in reply]
        assert [m.utterance_id for m in reply if isinstance(m, AstraSpoken)] == ["u1"]


# --- O-A3 (group A review round 2) --------------------------------------------------------------


def _kinds(outbox: Any) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for m in outbox.pending():
        if isinstance(m, AstraCanvasStep):
            out.append((m.phase, (m.request_id, m.canvas_idx, m.step)))
        elif isinstance(m, AstraSpoken):
            out.append(("spoken", m.utterance_id))
        else:
            out.append((m.type, None))
    return out


def test_a_full_outbox_drops_the_incoming_denoise_before_a_queued_commit() -> None:
    """O-A3 (3): an outbox of 4 holding [commit A/0, spoken x3] is full; a denoise frame of B
    arrives. Denoise-first (as the canvas-cap path chooses): the denoise goes, the commit stays.

    By hand: pending = [commit A/0, u0, u1, u2]; dropped == 1; the offer returns True (a canvas
    message never costs the connection)."""
    from local_astra.conductor.hud_server import Outbox

    box = Outbox(4)
    assert box.offer(committed("A", 0, 5))
    for i in range(3):
        assert box.offer(spoken(i))
    assert box.offer(step("B", 0, 1)) is True
    assert _kinds(box) == [
        ("commit", ("A", 0, 5)),
        ("spoken", "u0"),
        ("spoken", "u1"),
        ("spoken", "u2"),
    ], f"O-A3 (3): the commit was evicted for a denoise frame: {_kinds(box)}"
    assert box.dropped == 1, box.dropped


def test_a_full_outbox_still_evicts_a_queued_denoise_for_a_commit() -> None:
    """O-A3 (3), the other side of the choice: [denoise A/0, spoken x3] full, a commit of B arrives
    -> the queued denoise goes and the commit is queued (a commit is not dropped while a denoise
    could go instead). By hand: [u0, u1, u2, commit B/0]; dropped == 1."""
    from local_astra.conductor.hud_server import Outbox

    box = Outbox(4)
    assert box.offer(step("A", 0, 1))
    for i in range(3):
        assert box.offer(spoken(i))
    assert box.offer(committed("B", 0, 5)) is True
    assert _kinds(box) == [
        ("spoken", "u0"),
        ("spoken", "u1"),
        ("spoken", "u2"),
        ("commit", ("B", 0, 5)),
    ], _kinds(box)
    assert box.dropped == 1, box.dropped


def test_a_newer_denoise_of_the_next_canvas_goes_behind_the_queued_commit() -> None:
    """Reviewer 2 T1 (kills in-place replacement, M12): denoise A/0 s3, commit A/0, denoise A/1 s1.
    The page must get canvas 0's commit before canvas 1's denoise: the newer denoise replaces the
    queued one by removal + append, never in the older frame's slot. By hand: [commit A/0,
    denoise A/1 s1]."""
    from local_astra.conductor.hud_server import Outbox

    box = Outbox(512)
    box.offer(step("A", 0, 3))
    box.offer(committed("A", 0, 5))
    box.offer(step("A", 1, 1))
    assert _kinds(box) == [("commit", ("A", 0, 5)), ("denoise", ("A", 1, 1))], (
        f"T1: canvas 1's denoise overtook canvas 0's commit: {_kinds(box)}"
    )


def test_a_same_request_replacement_counts_in_dropped() -> None:
    """Reviewer 2 T4: A s1 then A s2 (same request, both denoise) -> one queued (s2) and
    ``dropped == 1``: the replaced frame never reaches the page, so it is a loss and is counted."""
    from local_astra.conductor.hud_server import Outbox

    box = Outbox(512)
    box.offer(step("A", 0, 1))
    box.offer(step("A", 0, 2))
    assert _kinds(box) == [("denoise", ("A", 0, 2))], _kinds(box)
    assert box.dropped == 1, f"T4: a replacement is a drop: dropped == {box.dropped}"


# --- reviewer 3 (at 1eef8fd) --------------------------------------------------------------------


def test_a_spoken_line_into_a_full_outbox_evicts_the_denoise_not_the_older_commit() -> None:
    """Reviewer 3 (2): Outbox(4) = [commit A/0, denoise B, u0, u1], then spoken u2. A canvas frame
    goes so the line can stay, and it is the denoise frame although the commit is older
    (denoise-first). By hand: [commit A/0, u0, u1, u2]; dropped == 1."""
    from local_astra.conductor.hud_server import Outbox

    box = Outbox(4)
    box.offer(committed("A", 0, 5))
    box.offer(step("B", 0, 1))
    box.offer(spoken(0))
    box.offer(spoken(1))
    assert box.offer(spoken(2)) is True
    assert _kinds(box) == [
        ("commit", ("A", 0, 5)),
        ("spoken", "u0"),
        ("spoken", "u1"),
        ("spoken", "u2"),
    ], f"reviewer 3 (2): the older commit went instead of the denoise: {_kinds(box)}"
    assert box.dropped == 1, box.dropped


def test_at_the_canvas_cap_an_incoming_denoise_never_evicts_a_commit() -> None:
    """Reviewer 3 (3): a roomy Outbox(64) holding 8 commits (the canvas cap, A6), then a denoise
    frame arrives. The cap is reached and no denoise is queued: the incoming denoise is the one that
    goes. By hand: the 8 commits stay, in order, and dropped == 1 (the frame's ``wire.out`` says
    dropped: the companion row in test_hud_wire_tap.py)."""
    from local_astra.conductor.hud_server import Outbox

    box = Outbox(64)
    for c in range(CANVAS_CAP):
        box.offer(committed("A", c, 5))
    assert box.offer(step("B", 0, 1)) is True
    assert _kinds(box) == [("commit", ("A", c, 5)) for c in range(CANVAS_CAP)], (
        f"reviewer 3 (3): at the cap a denoise evicted a commit: {_kinds(box)}"
    )
    assert box.dropped == 1, box.dropped
