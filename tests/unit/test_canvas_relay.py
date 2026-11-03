"""T-0128c group A: the conductor's relay of the plugin's per-step tap (criteria A1-A5, A8, A10).

Every row is derived from ``docs/tasks/T-0128c.md`` and ``AstraCanvasStep`` (``local_astra/contracts/
protocol.py``), never from an implementation (none exists when these are written). The socket half
(SUB connects, HWM, LINGER, reconnect, silent port) is ``tests/contract/test_canvas_relay_wire.py``;
this module is pure: no socket is opened, the clock is ``ScriptedClock``.

==================================================================================================
THE SEAMS THESE ROWS CALL (``local_astra/conductor/canvas_relay.py``). The implementer owns the design
behind them; anything wrong here goes to the reviewer (the implementer never edits ``tests/``).
==================================================================================================

    DEFAULT_ENDPOINT = "tcp://127.0.0.1:8002"      (= services/versions.lock [plugin] tap_endpoint)
    ENV_ENDPOINT = "ASTRA_TAP_ENDPOINT"

    check_endpoint(endpoint: str) -> str
        Returns the endpoint when it is ``tcp://127.0.0.1:<port>`` or ``ipc://<path>``; anything
        else raises ``ValueError`` (A1). Pure.
    endpoint_from_env(environ: Mapping[str, str] | None = None) -> str
        ``environ[ENV_ENDPOINT]`` checked, or ``DEFAULT_ENDPOINT`` when unset (``None``: os.environ).

    CanvasRelay(*, publish, clock, state, session_log=None, endpoint=DEFAULT_ENDPOINT,
                zmq_module=None)
        ``publish``: the ``HudPublisher`` (``HudServer.publish`` in production), called on the loop
        thread only. ``clock``: the session clock (``now_ms``, ``call_at``). ``state``: a zero-argument
        callable returning ``AstraTurnState`` (``Session.turn_state_message``). ``session_log``: a
        ``SessionLog`` or ``None``. ``zmq_module``: the ``zmq`` module to use (``None``: import pyzmq
        lazily in ``start``). The constructor checks the endpoint (``ValueError`` before any
        context or socket exists) and opens nothing.

        receive(payload: bytes) -> None
            One payload part exactly as it came off the SUB (the msgpack after the topic frame).
            Decodes, validates as ``AstraCanvasStep``, applies admission (A4) and queues it. Invalid
            payloads are counted (``invalid``) and dropped. Never raises.
        flush() -> None
            Publishes what is due at ``clock.now_ms`` (A3: denoise latest-wins per request, at most
            30 per request and 60 in total in any 1000 ms of the session clock; commits never lost).
            It also reads ``state()``: a change from a non-IDLE state to IDLE closes that turn (one
            ``note`` in the session log, A7; the ``denoise_no_tap`` check, A8). ``start`` runs it
            periodically on ``clock``; these rows call it directly.
        counts -> Mapping[str, int]
            At least ``received`` (every payload handed to ``receive``, valid or not),
            ``published``, ``coalesced`` (valid admitted denoise frames never published because a
            newer frame or a commit superseded them), ``refused_idle`` and ``invalid``.
        endpoint -> str
        await start() -> None / await aclose() -> None      (the contract module covers these)

    The session-log ``note`` of a closed turn is ``record("note", {"denoise": {"turn_id",
    "request_ids", "received", "published", "coalesced", "refused_idle", "invalid"}}, ...)``;
    ``request_ids`` lists the admitted request ids that had frames in that turn.

Regime: no measurement, no socket, ``ScriptedClock`` (virtual ms), hand-built CL 8 frames and the
recorded 806da12 trajectory (``tests/fixtures/canvas/MANIFEST.md``).

==================================================================================================
T-0159 group A (docs/tasks/T-0159.md A3-A5, A7, A8) amends this module
==================================================================================================

The relay adds ``argmax_pieces`` (denoise) / ``committed_pieces`` (commit) to each frame it
publishes, from a piece source; the seam (``piece_source=``, ``piece_loader=``,
``counts["undecoded"]``) is fixed in ``tests/fakes/canvas_tap.py`` (T-0159 section). The T-0128c
rigs below hand the relay a ``StubPieces`` when it declares ``piece_source`` (so no row loads the
real tokenizer), and the T-0128c "unchanged" rows compare frames **except the T-0159 fields**,
which, when present, must be the stub's pieces of the frame's own ids (A2 as amended by T-0159
A3). Those rows stay green on the BASE relay, which adds nothing. The T-0159 rows at the end are
``pending(task="T-0159")``. The A7 note may also carry ``undecoded`` (T-0159 A3).

The real-tokenizer rows (A4 loop lag) skip, naming the reason, when the pinned ``tokenizer.json``
is not in the local HF cache. Regime of those rows: CPU, the session's event loop, pinned sha
cc8d3a0c (rev 2ea8372), trajectory 806da12 (CL 256, 48 steps, 51 frames) at 40 ms per frame, n = 1
load + 1 replay; the row prints its measured lags and the load average.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from local_astra.contracts.protocol import AstraCanvasStep, AstraError, AstraSpoken
from local_astra.contracts.turn import TurnState
from tests.fakes.canvas_tap import (
    INVALID_PAYLOADS,
    NO_PIECES_CODE,
    PIECES_OFF_PREFIX,
    GatedLoader,
    LogSpy,
    Published,
    QueueZmq,
    SpyZmq,
    StateBox,
    StubPieces,
    assert_relayed,
    commit,
    denoise,
    failing_loader,
    load_trajectory,
    optional_piece_kwargs,
    pack,
    piece_kwargs,
    pieces_of,
    relay_attr,
    stub_piece,
)
from tests.unit._hud_rig import ScriptedClock

REPO = Path(__file__).resolve().parents[2]

#: An endpoint the unit rows hand the constructor; nothing here ever starts a relay on it.
UNIT_ENDPOINT = "ipc:///nonexistent/t0128c-unit.sock"
NOTE_KEYS = {
    "turn_id",
    "request_ids",
    "received",
    "published",
    "coalesced",
    "refused_idle",
    "invalid",
}
NON_IDLE = [s for s in TurnState if s is not TurnState.IDLE]


class Rig:
    """One relay over a ``ScriptedClock``, a ``StateBox``, a stamped publisher and a log spy."""

    def __init__(
        self,
        state: TurnState = TurnState.THINKING,
        turn_id: str = "t1",
        *,
        relay_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.clock = ScriptedClock()
        self.state = StateBox(state, turn_id)
        self.published = Published(clock=self.clock)
        self.log = LogSpy()
        cls = relay_attr("CanvasRelay")
        #: T-0159: the stub the relay decodes with (when it declares the seam); ``relay_kwargs``
        #: (a T-0159 row, built with ``piece_kwargs``) replaces it
        self.pieces = StubPieces()
        extra = (
            relay_kwargs if relay_kwargs is not None else optional_piece_kwargs(cls, self.pieces)
        )
        self.relay = cls(
            publish=self.published,
            clock=self.clock,
            state=self.state,
            session_log=self.log,
            endpoint=UNIT_ENDPOINT,
            **extra,
        )
        #: (now_ms, frame dict) of every valid frame handed to ``receive``
        self.received: list[tuple[int, dict[str, Any]]] = []
        self.published.mark = lambda: len(self.received)

    def send(self, frame: dict[str, Any], *, flush: bool = True) -> None:
        self.received.append((self.clock.now_ms, frame))
        self.relay.receive(pack(frame))
        if flush:
            self.relay.flush()

    def send_raw(self, payload: bytes) -> None:
        self.relay.receive(payload)
        self.relay.flush()

    def run(self, ms: int, *, every: int = 1) -> None:
        """Advance the clock ``ms`` in steps of ``every``, flushing after each."""
        for _ in range(ms // every):
            self.clock.advance(every)
            self.relay.flush()

    def go(self, state: TurnState, turn_id: str | None = None) -> None:
        self.state.go(state, turn_id)
        self.relay.flush()

    # --- what came out -----------------------------------------------------------------------
    def steps(self) -> list[AstraCanvasStep]:
        return [m for m in self.published.messages if getattr(m, "type", "") == "astra.canvas.step"]

    def keys(self, rid: str | None = None) -> list[tuple[str, int, int, str]]:
        return [
            (m.request_id, m.canvas_idx, m.step, m.phase)
            for m in self.steps()
            if rid is None or m.request_id == rid
        ]


def key(frame: dict[str, Any]) -> tuple[str, int, int, str]:
    return (frame["request_id"], frame["canvas_idx"], frame["step"], frame["phase"])


def validated(frame: dict[str, Any]) -> dict[str, Any]:
    return AstraCanvasStep.model_validate(frame).model_dump()


def is_subsequence(sub: list[Any], seq: list[Any]) -> bool:
    it = iter(seq)
    return all(any(x == y for y in it) for x in sub)


def max_in_window(times: list[int], window_ms: int = 1000) -> int:
    """Brute force: the most timestamps in any half-open window ``[t, t + window_ms)``."""
    best = 0
    for t in times:
        best = max(best, sum(1 for u in times if t <= u < t + window_ms))
    return best


# =============================================================================================
# A1 -- loopback only, checked before any socket exists
# =============================================================================================

REFUSED_ENDPOINTS = [
    "tcp://0.0.0.0:8002",
    "tcp://*:8002",
    "tcp://localhost.evil:8002",
    "tcp://127.0.0.1.nip.io:8002",
    "tcp://192.168.1.10:8002",
    "tcp://10.0.0.2:8002",
    "tcp://8.8.8.8:8002",
    "epgm://127.0.0.1:8002",
    "udp://127.0.0.1:8002",
    "",
    "ipc://",
    "tcp://127.0.0.1",
    "tcp://127.0.0.1:",
    "tcp://127.0.0.1:8002/x",
    " tcp://127.0.0.1:8002",
    # ``re.match(r"...$")`` accepts a trailing newline: the anchored check must not
    "tcp://127.0.0.1:8002\n",
    # the criterion's literal rule is 127.0.0.1 only (ambiguity noted in the report)
    "tcp://localhost:8002",
    "tcp://[::1]:8002",
]
ACCEPTED_ENDPOINTS = ["tcp://127.0.0.1:8002", "tcp://127.0.0.1:5555", "ipc:///tmp/t0128c/tap.sock"]


@pytest.mark.parametrize("endpoint", REFUSED_ENDPOINTS)
def test_a_non_loopback_endpoint_raises_before_any_socket_exists(endpoint: str) -> None:
    """A1: ``ValueError`` from the check and from the constructor; no context, no socket."""
    check = relay_attr("check_endpoint")
    with pytest.raises(ValueError):
        check(endpoint)
    spy = SpyZmq()
    cls = relay_attr("CanvasRelay")
    with pytest.raises(ValueError):
        cls(
            publish=Published(),
            clock=ScriptedClock(),
            state=StateBox(),
            endpoint=endpoint,
            zmq_module=spy,
        )
    assert not spy.ledger.contexts and not spy.ledger.sockets, (
        f"A1: {endpoint!r} is refused before any socket exists; the relay made "
        f"{len(spy.ledger.contexts)} context(s) and {len(spy.ledger.sockets)} socket(s)"
    )


@pytest.mark.parametrize("endpoint", ACCEPTED_ENDPOINTS)
def test_a_loopback_endpoint_is_accepted_and_construction_opens_nothing(endpoint: str) -> None:
    """A1, the too-strict side: the two allowed shapes construct; constructing opens nothing."""
    assert relay_attr("check_endpoint")(endpoint) == endpoint
    spy = SpyZmq()
    relay = relay_attr("CanvasRelay")(
        publish=Published(), clock=ScriptedClock(), state=StateBox(), endpoint=endpoint,
        zmq_module=spy,
    )  # fmt: skip
    assert relay.endpoint == endpoint
    assert not spy.ledger.contexts and not spy.ledger.sockets, (
        "the constructor opens nothing; start() does"
    )


def test_the_default_endpoint_is_the_locked_tap_endpoint_and_the_env_overrides_it() -> None:
    """A1: default = ``services/versions.lock`` ``[plugin] tap_endpoint``; ``ASTRA_TAP_ENDPOINT``
    overrides it and is checked like any other endpoint."""
    lock = tomllib.loads((REPO / "services" / "versions.lock").read_text())
    locked = lock["plugin"]["tap_endpoint"]
    assert locked == "tcp://127.0.0.1:8002", "the lock moved; re-read the task before this row"
    assert relay_attr("DEFAULT_ENDPOINT") == locked
    assert relay_attr("ENV_ENDPOINT") == "ASTRA_TAP_ENDPOINT"
    from_env = relay_attr("endpoint_from_env")
    assert from_env({}) == locked
    assert (
        from_env({"ASTRA_TAP_ENDPOINT": "ipc:///tmp/t0128c/x.sock"}) == "ipc:///tmp/t0128c/x.sock"
    )
    with pytest.raises(ValueError):
        from_env({"ASTRA_TAP_ENDPOINT": "tcp://0.0.0.0:8002"})


# =============================================================================================
# A2 -- real frames only, unchanged
# =============================================================================================


def test_the_recorded_trajectory_passes_through_unchanged_at_its_recorded_rate() -> None:
    """A2 + A3 liveness: the 51 real frames at ~25 Hz (40 ms apart, below the 30 Hz cap), one
    request, a turn open: every frame is published once, in order, equal field for field.

    Nothing is superseded before its slot at 25 Hz, so latest-wins coalescing has nothing to
    discard; a relay that publishes less is dropping real frames.
    """
    rig = Rig()
    frames = load_trajectory()["frames"]
    assert len(frames) == 51
    for frame in frames:
        rig.send(frame)
        rig.run(40)
    rig.run(1000)
    out = rig.steps()
    assert [key(f) for f in frames] == [
        (m.request_id, m.canvas_idx, m.step, m.phase) for m in out
    ], f"A2/A3: published {len(out)} of 51 frames: {rig.keys()}"
    for frame, message in zip(frames, out, strict=True):
        assert isinstance(message, AstraCanvasStep)
        # A2 as amended by T-0159 A3: field for field except the piece fields, which (when the
        # relay adds them) are the stub's pieces of this frame's own ids
        assert_relayed(message, frame)


@pytest.mark.parametrize("why", sorted(INVALID_PAYLOADS))
def test_an_invalid_payload_is_counted_and_never_published(why: str) -> None:
    """A2: each payload is invalid by hand (``AstraCanvasStep``/R10); counted, never published --
    including an ``astra.canvas.synthetic`` dict under the step topic."""
    rig = Rig()
    rig.send_raw(INVALID_PAYLOADS[why])
    rig.run(200)
    assert not rig.published.canvas(), f"A2: {why!r} was published: {rig.published.messages}"
    assert rig.relay.counts["invalid"] == 1, f"A2: counted once; counts={dict(rig.relay.counts)}"
    assert rig.relay.counts["received"] == 1


def test_a_coalescer_tick_with_no_new_frame_republishes_nothing() -> None:
    """A2/A3 adversarial: one frame, then a second of ticks with nothing new -> one publish."""
    rig = Rig()
    rig.send(denoise("A", 0, 1))
    rig.run(1000)
    rig.run(1000, every=50)
    assert rig.keys() == [("A", 0, 1, "denoise")], (
        f"A2: the relay creates no frame -- a tick re-sent the held one: {rig.keys()}"
    )


# The interleaving property: 1-3 requests, valid and invalid frames, arbitrary clock steps.

_op = st.one_of(
    st.tuples(st.just("denoise"), st.sampled_from("ABC")),
    st.tuples(st.just("commit"), st.sampled_from("ABC")),
    st.tuples(st.just("invalid"), st.sampled_from(sorted(INVALID_PAYLOADS))),
    st.tuples(st.just("advance"), st.integers(min_value=0, max_value=60)),
)


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(_op, min_size=1, max_size=60))
def test_published_frames_are_a_subset_of_received_ones_in_order(
    ops: list[tuple[str, Any]],
) -> None:
    """A2/A3 property (Hypothesis): over any interleaving, with a turn open the whole time --

    * every published step equals one received valid payload, field for field, and none twice;
    * no request id, canvas or step the tap did not send; no ``astra.canvas.synthetic``;
    * per request, published order is a subsequence of received order;
    * every commit is published (never coalesced away);
    * after commit k of a request, no denoise frame of canvas <= k of it is published;
    * the last denoise frame of a request's open canvas is published (latest-wins is live);
    * ``invalid`` and ``received`` count what was handed in.
    """
    rig = Rig()
    canvas = {r: 0 for r in "ABC"}
    step = {r: 0 for r in "ABC"}
    invalid = 0
    for kind, arg in ops:
        if kind == "advance":
            rig.run(arg) if arg else rig.relay.flush()
        elif kind == "invalid":
            invalid += 1
            rig.send_raw(INVALID_PAYLOADS[arg])
        elif kind == "denoise":
            if step[arg] < 48:
                step[arg] += 1
                rig.send(denoise(arg, canvas[arg], step[arg]))
        else:
            rig.send(commit(arg, canvas[arg], step[arg]))
            canvas[arg] += 1
            step[arg] = 0
    rig.run(3000, every=10)

    sent = [f for _, f in rig.received]
    by_key = {key(f): f for f in sent}
    out = rig.steps()
    assert all(m.type != "astra.canvas.synthetic" for m in rig.published.messages)
    keys = [(m.request_id, m.canvas_idx, m.step, m.phase) for m in out]
    assert len(keys) == len(set(keys)), f"A2: a frame was published twice: {keys}"
    for m in out:
        k = (m.request_id, m.canvas_idx, m.step, m.phase)
        assert k in by_key, f"A2: published {k}, which the tap never sent"
        assert_relayed(m, by_key[k])  # A2 as amended by T-0159 A3
    for rid in "ABC":
        mine_sent = [key(f) for f in sent if f["request_id"] == rid]
        mine_out = [k for k in keys if k[0] == rid]
        assert is_subsequence(mine_out, mine_sent), f"A3: {rid} reordered: {mine_out}"
        commits = [k for k in mine_sent if k[3] == "commit"]
        assert [k for k in mine_out if k[3] == "commit"] == commits, (
            f"A3: every commit of {rid} is published, once, in order"
        )
        for i, k in enumerate(mine_out):
            if k[3] == "commit":
                late = [j for j in mine_out[i + 1 :] if j[3] == "denoise" and j[1] <= k[1]]
                assert not late, f"A3: denoise of canvas <= {k[1]} after commit {k[1]}: {late}"
        tail = [k for k in mine_sent if k[3] == "denoise" and k[1] == canvas[rid]]
        if tail:
            assert tail[-1] in mine_out, (
                f"A3 latest-wins: the newest frame of {rid}'s open canvas {tail[-1]} never "
                f"went out; published {mine_out}"
            )
    assert rig.relay.counts["invalid"] == invalid
    assert rig.relay.counts["received"] == len(sent) + invalid


# =============================================================================================
# A3 -- order and coalescing
# =============================================================================================


def test_denoise_frames_are_latest_wins_and_at_most_thirty_per_second_per_request() -> None:
    """A3: one request at 200 Hz for 2 s, flushed every virtual ms.

    By hand: 400 frames in; any 1000 ms window of publish times holds <= 30 (brute force over the
    windows); every published frame was that request's newest at the moment it went out; the
    final frame goes out (latest-wins is not "first wins").
    """
    rig = Rig()
    latest_at: list[tuple[int, int]] = []  # (ms, step) newest received
    for i in range(400):
        s = i % 48 + 1
        c = i // 48
        rig.send(denoise("A", c, s), flush=False)
        latest_at.append((rig.clock.now_ms, s))
        rig.relay.flush()
        rig.run(5)
    rig.run(2000)
    times = [t for m, t in zip(rig.published.messages, rig.published.at_ms, strict=True)
             if getattr(m, "phase", None) == "denoise"]  # fmt: skip
    assert times, "A3: a 2 s stream published nothing"
    assert max_in_window(times) <= 30, (
        f"A3: {max_in_window(times)} denoise frames of one request in one second (cap 30)"
    )
    last = (rig.received[-1][1]["canvas_idx"], rig.received[-1][1]["step"])
    assert (rig.steps()[-1].canvas_idx, rig.steps()[-1].step) == last, (
        f"A3: the newest frame {last} is the last published; got {rig.keys()[-3:]}"
    )
    # "newest" = the last frame handed to ``receive`` before this publish, in sequence order: the
    # rig's flush at ms t runs before the frame sent at the same ms t, so ``t <= at`` would credit
    # a frame the relay had not yet seen (fixed 2026-10-26 after the implementer's report)
    for message, at, seen in zip(
        rig.published.messages, rig.published.at_ms, rig.published.marks, strict=True
    ):
        assert seen >= 1, f"published at {at} ms before any frame was received"
        newest = rig.received[seen - 1][1]
        assert (message.canvas_idx, message.step) == (newest["canvas_idx"], newest["step"]), (
            f"A3 latest-wins: at {at} ms the relay published {message.canvas_idx}/"
            f"{message.step}, not the newest {newest['canvas_idx']}/{newest['step']}"
        )


def test_the_aggregate_rate_is_at_most_sixty_per_second() -> None:
    """A3: three admitted requests at 200 Hz each for 2 s: <= 60 publishes in any 1000 ms."""
    rig = Rig()
    for i in range(400):
        for rid in "ABC":
            rig.send(denoise(rid, i // 48, i % 48 + 1), flush=False)
        rig.relay.flush()
        rig.run(5)
    rig.run(2000)
    times = rig.published.at_ms
    assert {m.request_id for m in rig.steps()} == set("ABC"), "every request is shown sometimes"
    assert max_in_window(times) <= 60, f"A3: {max_in_window(times)} messages in one second (cap 60)"


def test_a_commit_discards_the_pending_denoise_frames_of_its_canvas_and_is_never_lost() -> None:
    """A3: 31 denoise frames of canvas 0 one ms apart, then commit 0, then canvas 1.

    By hand: under any <= 30 Hz limiter at least the 31st frame (step 31) is still pending when the
    commit arrives (30 publishes in 31 ms is the most a sliding window allows; a min-gap limiter
    allows one). It must never be published, nor any canvas-0 denoise after the commit; the commit
    and canvas 1's frame are.
    """
    rig = Rig()
    for s in range(1, 32):
        rig.send(denoise("A", 0, s))
        rig.run(1)
    rig.send(commit("A", 0, 31))
    rig.send(denoise("A", 1, 1))
    rig.run(2000)
    keys = rig.keys()
    assert ("A", 0, 31, "commit") in keys, f"A3: the commit was lost: {keys}"
    assert ("A", 0, 31, "denoise") not in keys, f"A3: the pending frame outlived the commit: {keys}"
    after = keys[keys.index(("A", 0, 31, "commit")) + 1 :]
    assert [k for k in after if k[1] == 0] == [], f"A3: canvas 0 after its commit: {after}"
    assert ("A", 1, 1, "denoise") in after, f"A3: the next canvas still shows: {keys}"


def test_commits_are_never_coalesced_even_back_to_back() -> None:
    """A3: five commits in five ms (the plugin can publish a commit per call) -> five published."""
    rig = Rig()
    for c in range(5):
        rig.send(commit("A", c, 16 + c))
        rig.run(1)
    rig.run(2000)
    assert rig.keys() == [("A", c, 16 + c, "commit") for c in range(5)], rig.keys()


def test_received_order_is_kept_per_request_when_requests_interleave() -> None:
    """A3: A and B alternate below the cap; each request's published order is its sent order."""
    rig = Rig()
    for s in range(1, 11):
        rig.send(denoise("A", 0, s))
        rig.run(20)
        rig.send(denoise("B", 0, s))
        rig.run(20)
    rig.send(commit("B", 0, 10))
    rig.send(commit("A", 0, 10))
    rig.run(1000)
    for rid in "AB":
        sent = [key(f) for _, f in rig.received if f["request_id"] == rid]
        out = rig.keys(rid)
        assert out and is_subsequence(out, sent), f"A3: {rid} out of order: {out}"
        assert out[-1] == (rid, 0, 10, "commit")


# =============================================================================================
# A4 -- admission by turn state (both sides of the guard)
# =============================================================================================


def test_a_request_first_seen_while_idle_stays_refused_through_the_next_turn() -> None:
    """A4 adversarial: the idle captioner's frames continue into LISTENING and THINKING -- refused
    for its lifetime, its commit included; counted in ``refused_idle``."""
    rig = Rig(TurnState.IDLE, "t0")
    rig.send(denoise("cap", 0, 1))
    rig.run(50)
    rig.go(TurnState.LISTENING, "t1")
    rig.send(denoise("cap", 0, 2))
    rig.run(50)
    rig.go(TurnState.THINKING)
    rig.send(denoise("cap", 0, 3))
    rig.send(commit("cap", 0, 3))
    rig.run(500)
    assert rig.keys("cap") == [], f"A4: the captioner reached the page: {rig.keys('cap')}"
    assert rig.relay.counts["refused_idle"] == 4, dict(rig.relay.counts)


@pytest.mark.parametrize("state", NON_IDLE, ids=lambda s: s.value)
def test_a_request_first_seen_in_any_non_idle_state_is_admitted(state: TurnState) -> None:
    """A4, the too-wide side: "any other state is admitted" -- not THINKING only."""
    rig = Rig(state, "t1")
    rig.send(denoise("gen", 0, 1))
    rig.run(100)
    assert rig.keys("gen") == [("gen", 0, 1, "denoise")], (
        f"A4: a request first seen in {state.value} is admitted; published {rig.keys()}"
    )


def test_a_generate_requests_final_commit_fifty_ms_after_idle_is_published() -> None:
    """A4 adversarial, the other side: admitted in THINKING, its last commit lands 50 ms after the
    turn went IDLE -- still published (admission is per request lifetime, not per frame)."""
    rig = Rig(TurnState.THINKING, "t1")
    rig.send(denoise("gen", 0, 1))
    rig.run(40)
    rig.go(TurnState.SPEAKING)
    rig.send(denoise("gen", 0, 2))
    rig.run(40)
    rig.go(TurnState.IDLE)
    rig.run(50)
    rig.send(commit("gen", 0, 2))
    rig.run(500)
    assert ("gen", 0, 2, "commit") in rig.keys("gen"), (
        f"A4: the post-IDLE commit of an admitted request is published; got {rig.keys()}"
    )
    assert rig.relay.counts["refused_idle"] == 0


def test_the_refused_table_is_bounded_at_sixty_four() -> None:
    """A4 LRU <= 64: after 64 newer refused ids, the first is forgotten -- its next frame, while a
    turn is open, is a first sighting and is admitted. With 63 newer it is still remembered.

    (The not-too-tight half reads "LRU <= 64" as capacity 64; see the report.)
    """
    for newer, admitted in ((63, False), (64, True)):
        rig = Rig(TurnState.IDLE, "t0")
        rig.send(denoise("old", 0, 1))
        for i in range(newer):
            rig.send(denoise(f"cap{i}", 0, 1))
        rig.go(TurnState.THINKING, "t1")
        rig.run(100)
        rig.send(denoise("old", 0, 2))
        rig.run(100)
        got = ("old", 0, 2, "denoise") in rig.keys("old")
        assert got is admitted, (
            f"A4: with {newer} newer refused ids the first is "
            f"{'forgotten' if admitted else 'still refused'}; published {rig.keys('old')}"
        )


def test_the_admitted_table_is_bounded_at_sixty_four() -> None:
    """A4 LRU <= 64, the admitted table: 64 newer admitted ids evict the first, whose post-IDLE
    commit is then a first sighting in IDLE (refused); with 63 newer it is still admitted."""
    for newer, published in ((63, True), (64, False)):
        rig = Rig(TurnState.THINKING, "t1")
        rig.send(denoise("first", 0, 1))
        for i in range(newer):
            rig.send(denoise(f"gen{i}", 0, 1))
        rig.go(TurnState.IDLE)
        rig.run(100)
        rig.send(commit("first", 0, 1))
        rig.run(500)
        got = ("first", 0, 1, "commit") in rig.keys("first")
        assert got is published, f"A4 LRU with {newer} newer admitted ids: {rig.keys('first')}"


# =============================================================================================
# A5 -- the relay never changes the turn
# =============================================================================================


async def _one_turn(
    tmp_path: Path,
    feed: Callable[[Any, Any], None] | None,
    *,
    relay_kwargs: dict[str, Any] | None = None,
    extras: dict[str, Any] | None = None,
) -> tuple[list[str], list[dict[str, Any]], int]:
    """One scripted turn through the T-0028 rig; ``feed(rig, relay)`` runs at every clock step.

    T-0159: ``relay_kwargs`` go to the relay (a piece source); ``extras``, when given, receives
    the TTS requests, the history's repr and the published canvas messages (A8)."""
    from tests.unit._hud_rig import Recorder, answer, running

    published = Recorder()

    class FeedingClock(ScriptedClock):
        hook: Callable[[], None] | None = None

        def advance(self, ms: int) -> int:
            now = super().advance(ms)
            if self.hook is not None:
                self.hook()
            return now

    clock = FeedingClock()
    script = [answer("Here it is. "), answer("And the second sentence. ")]
    async with running(tmp_path, script=script, ms_per_word=100, publish=published, clock=clock) as rig:  # fmt: skip
        relay = None
        if feed is not None:
            relay = relay_attr("CanvasRelay")(
                publish=published,
                clock=clock,
                state=rig.session.turn_state_message,
                endpoint=UNIT_ENDPOINT,
                **(relay_kwargs or {}),
            )
            clock.hook = lambda: feed(rig, relay)
        await rig.ask("tell me two things")
        await rig.speak_out()
        clock.hook = None
        lines = rig.lines()
        if extras is not None:
            extras["tts"] = list(rig.tts.requests)
            extras["history"] = repr(rig.state.history)
            extras["canvas"] = [m for m in published.messages if m.type.startswith("astra.canvas.")]
    spoken = [
        m.model_dump_json() for m in published.messages if not m.type.startswith("astra.canvas.")
    ]
    records = [json.loads(line) for line in lines]
    canvas = sum(1 for m in published.messages if m.type.startswith("astra.canvas."))
    return spoken, records, canvas


def _volatile(a: Any, b: Any, path: str = "") -> set[str]:
    """Paths whose values differ between two runs of the *same* turn without a relay."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: set[str] = set()
        for k in set(a) | set(b):
            out |= _volatile(a.get(k), b.get(k), f"{path}/{k}")
        return out
    return set() if a == b else {path}


def _strip(obj: Any, paths: set[str], path: str = "") -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip(v, paths, f"{path}/{k}") for k, v in obj.items() if f"{path}/{k}" not in paths
        }
    return obj


async def test_a_turn_with_the_relay_publishing_is_identical_to_the_turn_without_it(tmp_path):
    """A5: the same scripted turn three times -- twice without a relay (which fields differ between
    identical runs is the oracle for what is volatile) and once with a relay that is fed a real
    trajectory frame at every clock step and flushed. The non-canvas messages are byte-identical
    and the turn record is identical outside the volatile fields; the relay really did publish."""
    frames = iter(load_trajectory()["frames"] * 20)
    opened = {"yes": False}

    def feed(rig: Any, relay: Any) -> None:
        # The tap's request starts once the turn is open (the engine request follows the onset).
        # A frame fed while the session is still IDLE would be the idle captioner's by A4 and its
        # one request id refused for life -- the relay would publish nothing and the row would be
        # vacuous (fixed 2026-10-26 after the implementer's report). Once open, feed on every step.
        if not opened["yes"]:
            if rig.session.turn_state_message().state is TurnState.IDLE:
                relay.flush()
                return
            opened["yes"] = True
        frame = next(frames, None)
        if frame is not None:
            relay.receive(pack(frame))
        relay.flush()

    base1 = await _one_turn(tmp_path / "a", None)
    base2 = await _one_turn(tmp_path / "b", None)
    assert base1[0] == base2[0], "the rig itself is not deterministic; this row cannot judge"
    assert len(base1[1]) == len(base2[1]) == 1, "the scripted turn writes one turn record"
    assert any('"astra.spoken"' in m for m in base1[0]), "the turn spoke"
    with_relay = await _one_turn(tmp_path / "c", feed)
    assert with_relay[2] > 0, "the relay published nothing -- this row would be vacuous"
    assert with_relay[0] == base1[0], (
        "A5: with the relay attached the turn's published (non-canvas) messages changed"
    )
    assert len(with_relay[1]) == len(base1[1]) == 1, (with_relay[1], base1[1])
    volatile = _volatile(base1[1][0], base2[1][0])
    assert _strip(with_relay[1][0], volatile) == _strip(base1[1][0], volatile), (
        f"A5: the turn record differs with the relay attached (volatile fields excluded: {volatile})"
    )


# =============================================================================================
# A7 (the note) and A8 (denoise_no_tap)
# =============================================================================================


def test_each_closed_turn_writes_one_denoise_note() -> None:
    """A7: one ``note`` per closed turn with exactly the seven keys; the numbers by hand from what
    was sent and what the publisher saw (the published/coalesced split is the relay's to choose
    under the rate cap, so it is read off the publisher, then conserved)."""
    rig = Rig(TurnState.IDLE, "t0")
    rig.send(denoise("cap", 0, 1))  # refused: first seen in IDLE
    rig.go(TurnState.THINKING, "t1")
    rig.send(denoise("cap", 0, 2))  # refused
    rig.send(denoise("A", 0, 1))
    rig.run(1)
    rig.send(denoise("A", 0, 2))
    rig.run(1)
    rig.send(denoise("A", 0, 3))
    rig.send_raw(INVALID_PAYLOADS["commit with rows"])
    rig.run(1000)
    rig.go(TurnState.IDLE)
    rig.run(10)
    notes = [n for n in rig.log.notes() if isinstance(n, dict) and "denoise" in n]
    assert len(notes) == 1, f"A7: one denoise note per closed turn; got {rig.log.records}"
    note = notes[0]["denoise"]
    # T-0159 A3 amends A7: the note may also carry ``undecoded`` (0 here: every id decodes)
    assert set(note) - {"undecoded"} == NOTE_KEYS, (
        f"A7: the note's keys are exactly {sorted(NOTE_KEYS)} (+ T-0159's undecoded): {note}"
    )
    assert note.get("undecoded", 0) == 0, note
    assert note["turn_id"] == "t1"
    assert list(note["request_ids"]) == ["A"], f"A7: admitted ids only: {note['request_ids']}"
    published_a = len(rig.keys("A"))
    assert 1 <= published_a <= 3
    assert note["published"] == published_a, (note, rig.keys())
    assert note["coalesced"] == 3 - published_a, note
    assert note["invalid"] == 1, note
    assert note["refused_idle"] >= 1, note
    assert note["received"] == (
        note["published"] + note["coalesced"] + note["refused_idle"] + note["invalid"]
    ), f"A7: every received payload is published, coalesced, refused or invalid: {note}"
    # a second turn writes a second note, of its own id
    rig.go(TurnState.LISTENING, "t2")
    rig.go(TurnState.IDLE)
    ids = [
        n["denoise"]["turn_id"] for n in rig.log.notes() if isinstance(n, dict) and "denoise" in n
    ]
    assert ids == ["t1", "t2"], ids


def test_a_turn_that_closes_with_no_tap_frame_raises_denoise_no_tap_once_per_session() -> None:
    """A8: zero frames since the session started and a turn closes -> one ``astra.error`` with code
    ``denoise_no_tap``; a second empty turn adds none."""
    rig = Rig(TurnState.IDLE, "t0")
    rig.go(TurnState.THINKING, "t1")
    rig.run(100)
    assert not rig.published.errors(), "no error while the turn is open"
    rig.go(TurnState.IDLE)
    rig.run(10)
    errors = rig.published.errors("denoise_no_tap")
    assert len(errors) == 1 and isinstance(errors[0], AstraError), rig.published.messages
    assert errors[0].message, "the error says something"
    rig.go(TurnState.THINKING, "t2")
    rig.go(TurnState.IDLE)
    rig.run(10)
    assert len(rig.published.errors("denoise_no_tap")) == 1, "A8: once per session"


@pytest.mark.parametrize("when", ["before the turn (refused, idle)", "during the turn"])
def test_denoise_no_tap_is_never_raised_once_a_frame_was_received(when: str) -> None:
    """A8, the too-wide side: "zero tap frames received" -- a received frame, even a refused
    captioner frame, proves the tap is up. Then turns that close with no frames say nothing."""
    rig = Rig(TurnState.IDLE, "t0")
    if when.startswith("before"):
        rig.send(denoise("cap", 0, 1))
    rig.go(TurnState.THINKING, "t1")
    if when == "during the turn":
        rig.send(denoise("A", 0, 1))
    rig.run(100)
    rig.go(TurnState.IDLE)
    rig.go(TurnState.THINKING, "t2")
    rig.go(TurnState.IDLE)
    rig.run(10)
    assert not rig.published.errors("denoise_no_tap"), rig.published.messages


def test_no_turn_closing_means_no_denoise_no_tap() -> None:
    """A8, the absent side of the trigger: a session idle for a minute with no frames says nothing
    (the error is about a turn that closed, not about silence)."""
    rig = Rig(TurnState.IDLE, "t0")
    rig.run(60_000, every=100)
    assert not rig.published.messages, rig.published.messages


# =============================================================================================
# A10 -- with no tap frames nothing canvas-shaped is published
# =============================================================================================


def test_with_no_tap_frames_the_relay_publishes_no_canvas_message() -> None:
    """A10: turns open and close, no frame arrives -> zero ``astra.canvas.*`` (only the A8 error)."""
    rig = Rig(TurnState.IDLE, "t0")
    for t in ("t1", "t2", "t3"):
        rig.go(TurnState.LISTENING, t)
        rig.go(TurnState.THINKING)
        rig.run(200)
        rig.go(TurnState.SPEAKING)
        rig.go(TurnState.IDLE)
    assert not rig.published.canvas(), rig.published.canvas()
    assert all(isinstance(m, AstraError) for m in rig.published.messages)
    assert not any(isinstance(m, AstraSpoken) for m in rig.published.messages)


# =============================================================================================
# O-A2 (group A reviewer 1): the relay's own cadence, per-turn notes, survival, cancellation, LRU
# =============================================================================================
#
# ``LiveRig`` starts the relay for real (``start``/``aclose``) on ``QueueZmq`` (tests/fakes): the
# relay's reader awaits a queue the row fills and its timer is scheduled on ``ScriptedClock``, so
# the relay's *own* flushes are the only flushes -- no row-driven ``flush()`` -- and no wall time
# or socket is involved.


async def _settle(n: int = 8) -> None:
    import asyncio

    for _ in range(n):
        await asyncio.sleep(0)


class LiveRig:
    def __init__(
        self,
        state: TurnState = TurnState.THINKING,
        *,
        publish: Callable[[Any], None] | None = None,
        state_fn: Callable[[], Any] | None = None,
        relay_kwargs: dict[str, Any] | None = None,
    ) -> None:

        self.clock = ScriptedClock()
        self.box = StateBox(state, "t1")
        self.published = Published(clock=self.clock)
        self.log = LogSpy()
        self.zmq = QueueZmq()
        #: exceptions a timer callback raised (the event loop logs and drops them in production)
        self.timer_errors: list[BaseException] = []
        cls = relay_attr("CanvasRelay")
        # T-0159: a stub piece source when the relay declares the seam, so ``start`` never loads
        # the real tokenizer here; a T-0159 row passes its own ``relay_kwargs``
        self.pieces = StubPieces()
        extra = (
            relay_kwargs if relay_kwargs is not None else optional_piece_kwargs(cls, self.pieces)
        )
        self.relay = cls(
            publish=publish if publish is not None else self.published,
            clock=self.clock,
            state=state_fn if state_fn is not None else self.box,
            session_log=self.log,
            endpoint=UNIT_ENDPOINT,
            zmq_module=self.zmq,
            **extra,
        )

    async def start(self) -> LiveRig:
        await self.relay.start()
        await _settle()
        return self

    def push(self, frame: dict[str, Any]) -> None:
        self.zmq.sub().push_frame(frame)

    async def advance(self, ms: int) -> None:
        """Move the virtual clock 1 ms at a time, letting the reader run between steps; a timer
        callback that raises is recorded, as the event loop would log it, and time goes on."""
        for _ in range(ms):
            await self.advance_to(self.clock.now_ms + 1)

    async def advance_to(self, at_ms: int) -> None:
        while self.clock.now_ms < at_ms:
            try:
                self.clock.advance(at_ms - self.clock.now_ms)
            except Exception as exc:  # a raising _tick: asyncio would log it and keep running
                self.timer_errors.append(exc)
            await _settle()

    def denoise_out(self) -> list[Any]:
        return [m for m in self.published.messages if getattr(m, "phase", None) == "denoise"]


async def test_a_thirty_hz_stream_is_published_with_only_the_relays_own_flushes() -> None:
    """O-A2 (1): the plugin's cap is 30 Hz, a period of 33.34 ms. 300 frames of one request at that
    period, flushed only by the relay's reader and its own timer (no row flush).

    By hand: 30 frames at 33.34 ms span 966.9 ms, so the 30-per-1000-ms cap allows every one of
    them; a relay that rounds its gap up to 34 ms falls behind by 0.66 ms a frame and may lose one
    frame in ~50 (2 %). So >= 294 of 300 must be published. A reader flush that leaves the next
    slot to the 50 ms state poll loses a third (reviewer: 201/300)."""
    rig = await LiveRig().start()
    n = 300
    for i in range(n):
        await rig.advance_to(round(i * 33.34))
        rig.push(denoise("A", i // 48, i % 48 + 1))
        await _settle()
    await rig.advance(2000)
    await rig.relay.aclose()
    out = len(rig.denoise_out())
    assert out >= 294, f"O-A2 (1): {out}/{n} frames of a 30 Hz stream were published (>= 294)"


async def test_a_commit_is_published_within_one_ms_by_the_relays_own_flushes() -> None:
    """O-A2 M16: "a commit is published at the next flush" -- the next flush is the reader's, not
    the 50 ms state poll's. A commit pushed with no other traffic is out within 1 virtual ms."""
    rig = await LiveRig().start()
    await rig.advance(200)  # the timer is now on its idle poll
    rig.push(commit("A", 0, 16))
    await _settle()
    await rig.advance(1)
    keys = [(m.request_id, m.phase) for m in rig.published.messages if hasattr(m, "phase")]
    await rig.relay.aclose()
    assert ("A", "commit") in keys, f"M16: the commit waited for the state poll: {keys}"


def test_a_turn_id_change_without_idle_closes_the_turn_and_resets_its_counts() -> None:
    """O-A2 (2) / M25: THINKING t1 (A's frames) -> SPEAKING -> LISTENING t2 (no IDLE: a takeover
    opens the next turn straight into LISTENING) -> THINKING (B's frames) -> IDLE.

    By hand: two notes, t1 then t2. t1: request_ids [A], received 3, published 3 (frames 50 ms
    apart, under the cap). t2: request_ids [B], received 2, published 2 -- not 5: counts reset."""
    rig = Rig(TurnState.THINKING, "t1")
    for s in (1, 2, 3):
        rig.send(denoise("A", 0, s))
        rig.run(50)
    rig.go(TurnState.SPEAKING)
    rig.run(50)
    rig.go(TurnState.LISTENING, "t2")
    rig.run(50)
    rig.go(TurnState.THINKING)
    for s in (1, 2):
        rig.send(denoise("B", 0, s))
        rig.run(50)
    rig.go(TurnState.IDLE)
    rig.run(50)
    notes = [n["denoise"] for n in rig.log.notes() if isinstance(n, dict) and "denoise" in n]
    assert [n["turn_id"] for n in notes] == ["t1", "t2"], f"O-A2 (2): one note per turn: {notes}"
    t1, t2 = notes
    assert list(t1["request_ids"]) == ["A"] and list(t2["request_ids"]) == ["B"], notes
    assert (t1["received"], t1["published"], t1["coalesced"]) == (3, 3, 0), t1
    assert (t2["received"], t2["published"], t2["coalesced"]) == (2, 2, 0), (
        f"M25: t2's counts are its own, reset at t1's close: {t2}"
    )
    assert t1["refused_idle"] == t2["refused_idle"] == 0 and t1["invalid"] == t2["invalid"] == 0


async def test_publish_raising_once_does_not_end_the_relay() -> None:
    """O-A2 (3): the HUD publisher raises on its first call (frame 1). The relay keeps going: a
    frame of the same request 100 ms later is published."""
    calls = {"n": 0}
    out: list[Any] = []

    def flaky(message: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("t0128c: publish failed once")
        out.append(message)

    rig = await LiveRig(publish=flaky).start()
    rig.push(denoise("A", 0, 1))
    await _settle()
    await rig.advance(100)
    rig.push(denoise("A", 0, 2))
    await _settle()
    await rig.advance(100)
    await rig.relay.aclose()
    assert [(m.request_id, m.step) for m in out] == [("A", 2)], (
        f"O-A2 (3): after one failed publish the relay still publishes; got {out} "
        f"(publish called {calls['n']} times)"
    )


@pytest.mark.parametrize("where", ["reader", "timer"])
async def test_state_raising_once_does_not_end_the_relay(where: str) -> None:
    """O-A2 (3): ``state()`` raises once -- on the reader's path (a first frame's admission) or on
    the timer's (the idle poll). Afterwards frames still publish and a turn close is still seen
    (one A7 note): neither the reader nor the timer chain has died."""
    box = StateBox(TurnState.THINKING, "t1")
    armed = {"on": False}

    def state() -> Any:
        if armed["on"]:
            armed["on"] = False
            raise RuntimeError("t0128c: state() failed once")
        return box()

    rig = await LiveRig(state_fn=state).start()
    await rig.advance(10)
    armed["on"] = True
    if where == "reader":
        rig.push(denoise("A", 0, 1))
        await _settle()
    await rig.advance(200)  # the timer's poll fires at least once
    assert not armed["on"], "premise: state() was called (and raised) once"
    # (a) no frame after the failure: only the timer can see this turn close
    box.go(TurnState.IDLE)
    await rig.advance(200)
    notes = [n for n in rig.log.notes() if isinstance(n, dict) and "denoise" in n]
    # (b) a later turn's frame still reaches the page: the reader is alive
    box.go(TurnState.THINKING, "t2")
    await rig.advance(100)
    rig.push(denoise("B", 0, 1))
    await _settle()
    await rig.advance(100)
    await rig.relay.aclose()
    assert len(notes) == 1, (
        f"O-A2 (3): after state() raised on the {where}, the timer still closes the turn "
        f"(one note, no frame to wake the reader): {notes}"
    )
    assert ("B", 1) in [(m.request_id, m.step) for m in rig.denoise_out()], (
        f"O-A2 (3): after state() raised on the {where}, frames still publish: "
        f"{rig.published.messages}"
    )


async def test_a_relay_whose_socket_fails_for_good_stops_and_says_so(capsys) -> None:
    """O-A2 (3) / A8: the SUB's receive fails permanently (every call raises ``ZMQError``). The
    relay must neither spin on it (<= 20 receive calls over 100 loop turns) nor stop silently: one
    session-log note and one stderr line naming the error.

    Reading: a permanent socket error is the case where "a relay that does stop" applies; an
    exception in publish/state is survived per iteration (rows above). Ambiguity reported."""
    import zmq

    rig = await LiveRig().start()
    sub = rig.zmq.sub()
    why = "t0128c-socket-gone-for-good"
    for _ in range(200):
        sub.push_error(zmq.ZMQError(msg=why))
    await _settle(100)
    await rig.advance(100)
    calls = sub.recv_calls
    await rig.relay.aclose()
    err = capsys.readouterr().err
    assert calls <= 20, f"O-A2 (3): the relay spins on a dead socket ({calls} receive calls)"
    lines = [line for line in err.splitlines() if why in line]
    assert len(lines) == 1, f"O-A2 (3)/A8: one stderr line names {why!r}; stderr:\n{err}"
    notes = [n for n in rig.log.notes() if why in str(n)]
    assert len(notes) == 1, f"O-A2 (3)/A8: one note names {why!r}: {rig.log.records}"


async def test_cancelling_aclose_propagates_and_cleanup_still_completes() -> None:
    """O-A2 (4): the task running ``aclose`` is cancelled while ``aclose`` waits for the reader.
    The CancelledError reaches the caller (only the reader's own cancellation is suppressed),
    and the socket is still closed and the context still terminated."""
    import asyncio

    rig = await LiveRig().start()
    task = asyncio.ensure_future(rig.relay.aclose())
    await asyncio.sleep(0)  # aclose runs up to its first await
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await _settle()
    sub = rig.zmq.sub()
    assert sub.closed, "O-A2 (4): the SUB is closed even though aclose was cancelled"
    assert all(c.termed for c in rig.zmq.contexts), "O-A2 (4): the context is still terminated"


def test_an_admitted_id_touched_recently_survives_sixty_four_newer_ones() -> None:
    """O-A2 M21: the admitted table is LRU by use, not by insertion. ``first`` is admitted, 63
    newer follow (table full), ``first`` gets a frame (most recent), one more newer arrives: the
    least recently used (``gen0``) is evicted, not ``first``. After IDLE, ``first``'s commit is
    published; ``gen0``'s, a first sighting in IDLE, is refused (capacity still exactly 64)."""
    rig = Rig(TurnState.THINKING, "t1")
    rig.send(denoise("first", 0, 1))
    for i in range(63):
        rig.send(denoise(f"gen{i}", 0, 1))
    rig.send(denoise("first", 0, 2))
    rig.send(denoise("gen63", 0, 1))
    rig.go(TurnState.IDLE)
    rig.run(100)
    rig.send(commit("first", 0, 2))
    rig.send(commit("gen0", 0, 1))
    rig.run(500)
    assert ("first", 0, 2, "commit") in rig.keys("first"), (
        f"M21: a recently used admitted id was evicted by insertion order: {rig.keys('first')}"
    )
    assert ("gen0", 0, 1, "commit") not in rig.keys("gen0"), (
        f"the least recently used id is the one evicted (capacity 64): {rig.keys('gen0')}"
    )


def test_a_refused_id_touched_recently_stays_refused_past_sixty_four_newer_ones() -> None:
    """O-A2 M22: the refused table is LRU by use. ``old`` is refused in IDLE, 63 newer refused ids
    fill the table, ``old`` sends again (still IDLE: refused, most recent), one more newer: ``cap0``
    is forgotten, ``old`` is not. In THINKING, ``old``'s next frame is still refused and ``cap0``'s
    (a first sighting now) is admitted."""
    rig = Rig(TurnState.IDLE, "t0")
    rig.send(denoise("old", 0, 1))
    for i in range(63):
        rig.send(denoise(f"cap{i}", 0, 1))
    rig.send(denoise("old", 0, 2))
    rig.send(denoise("cap63", 0, 1))
    rig.go(TurnState.THINKING, "t1")
    rig.run(100)
    rig.send(denoise("old", 0, 3))
    rig.run(100)
    rig.send(denoise("cap0", 0, 2))
    rig.run(100)
    assert rig.keys("old") == [], f"M22: a recently refused id was forgotten: {rig.keys('old')}"
    assert rig.keys("cap0") == [("cap0", 0, 2, "denoise")], (
        f"the least recently refused id is the one forgotten (capacity 64): {rig.keys('cap0')}"
    )


# =============================================================================================
# O-A3 (group A review round 2): a failed publish duplicates nothing; a persistent fault is quiet
# =============================================================================================


def test_a_publish_that_raises_while_two_commits_are_queued_never_duplicates_a_commit() -> None:
    """O-A3 (1): commits c0 and c1 of one request are queued (no flush between them); the HUD
    publisher delivers c0 and raises on c1. Later flushes must never hand c0 to the publisher a
    second time (pop each commit before emitting it). Every publish *call* is recorded, the raising
    one included, so a retried commit shows up as a second call."""
    calls: list[tuple[int, str]] = []

    def flaky(message: Any) -> None:
        calls.append((message.canvas_idx, message.phase))
        if len(calls) == 2:
            raise RuntimeError("t0128c: publish failed on the second commit")

    clock = ScriptedClock()
    relay = relay_attr("CanvasRelay")(
        publish=flaky,
        clock=clock,
        state=StateBox(TurnState.THINKING, "t1"),
        endpoint=UNIT_ENDPOINT,
    )
    relay.receive(pack(commit("A", 0, 16)))
    relay.receive(pack(commit("A", 1, 18)))
    import contextlib

    for _ in range(5):
        # the relay may let it out of flush; what matters is what is called next
        with contextlib.suppress(RuntimeError):
            relay.flush()
        clock.advance(50)
    assert calls.count((0, "commit")) == 1, (
        f"O-A3 (1): commit c0 was handed to the publisher {calls.count((0, 'commit'))} times: "
        f"{calls}"
    )
    assert calls.count((1, "commit")) <= 1, f"O-A3 (1): commit c1 retried: {calls}"


async def test_a_persistent_publish_fault_logs_once_not_once_per_frame(caplog) -> None:
    """O-A3 (2): the publisher raises on every call. 50 frames, 50 virtual ms apart, through the
    relay's own reader: the relay keeps receiving (received == 50) and the log carries at most 2
    warning-or-worse records from ``local_astra`` (one, or a rate-limited few -- not 50 tracebacks)."""
    import logging

    def broken(message: Any) -> None:
        raise RuntimeError("t0128c: the HUD publisher is broken for good")

    caplog.set_level(logging.WARNING)
    rig = await LiveRig(publish=broken).start()
    for i in range(50):
        rig.push(denoise("A", i // 48, i % 48 + 1))
        await _settle()
        await rig.advance(50)
    received = rig.relay.counts["received"]
    await rig.relay.aclose()
    noisy = [
        r for r in caplog.records if r.name.startswith("local_astra") and r.levelno >= logging.WARNING
    ]  # fmt: skip
    assert received == 50, f"premise: the relay kept reading ({received} of 50)"
    assert len(noisy) <= 2, (
        f"O-A3 (2): a persistent publish fault logged {len(noisy)} records for 50 frames "
        f"(at most 2): {[r.getMessage() for r in noisy[:3]]}"
    )


#: The literal base T-0128c group A branched from (main at the task file's commit). Scope guard:
#: whoever merges group A converts this to the closed BASE..TIP range of its merge commit
#: (``.claude/rules/tests.md``; ``tests/_git_range.py``).
BASE = "9b8f5c1"
UNTOUCHED = (
    "plugin",
    "local_astra/conductor/turn_machine.py",
    "local_astra/conductor/session.py",
    "local_astra/conductor/speech_gate.py",
    "local_astra/conductor/llm_client.py",
    "local_astra/contracts",
    "services",
)


#: Group A's implementation tip (the verifier passed it, 2026-10-26): the closed range BASE..TIP is
#: what A10 is about. Never compared with the working tree, which later tasks legitimately move
#: (``.claude/rules/tests.md``, scope pins; ``tests/unit/test_guards_use_fixed_refs.py``).
TIP = "1eef8fd"


def test_the_files_group_a_may_not_touch_are_byte_identical_to_the_base() -> None:
    """A10: plugin/**, the turn machine, session, speech gate, LLM client, contracts/** and
    services/** are byte-identical between group A's base and its implementation tip
    (BASE..TIP, both literals; skipped when either commit is absent)."""
    from tests import _git_range as gr

    gr.require_commits(BASE, TIP)
    assert gr.is_ancestor(BASE, TIP), f"{BASE} is not an ancestor of {TIP}"
    changed = gr.changed(BASE, TIP, *UNTOUCHED)
    assert not changed, f"A10: group A changed {changed} between {BASE} and {TIP}"


# =============================================================================================
# T-0159 group A: the relay adds the pieces (docs/tasks/T-0159.md A3-A5, A7, A8; ruling O-A1)
# =============================================================================================
#
# Every row below hands the relay a piece source through the seam frozen in
# ``tests/fakes/canvas_tap.py`` (``piece_source=`` / ``piece_loader=``; ``piece_kwargs`` turns a
# relay without the seam into an assertion naming T-0159, never a ``TypeError``). Expected pieces
# are the stub's rule (``#<id>``) over the frame's own ids -- never the relay's output.
#
# O-A1 (2026-10-26): ``start()`` awaits the loader before it reads the tap, so the "frames before
# the load completes go out without pieces" row of the task's list does not apply to the default
# path; no row here pins that path (the ruling lets the implementation drop it).


def _piece_rig(source: Any = None, state: TurnState = TurnState.THINKING, turn_id: str = "t1") -> Rig:  # fmt: skip
    """A ``Rig`` whose relay decodes with ``source`` (default: a fresh counting ``StubPieces``)."""
    cls = relay_attr("CanvasRelay")
    src = source if source is not None else StubPieces()
    rig = Rig(state, turn_id, relay_kwargs=piece_kwargs(cls, piece_source=src))
    rig.pieces = src
    return rig


def _own_ids(message: Any) -> tuple[int, ...]:
    """The ids a published frame's pieces are of: committed ids (commit) or argmax (denoise)."""
    if message.phase == "commit":
        return tuple(message.committed_tokens)
    return tuple(message.argmax)


def _step_with_pieces(frame: dict[str, Any]) -> AstraCanvasStep:
    """A frame carrying the T-0159 fields, or an assertion naming A1 while the contract lacks them."""
    missing = [f for f in ("argmax_pieces", "committed_pieces") if f not in AstraCanvasStep.model_fields]  # fmt: skip
    assert not missing, f"T-0159 A1: AstraCanvasStep has no field(s) {missing} yet"
    return AstraCanvasStep.model_validate(frame)


def _denoise_notes(log: LogSpy) -> list[dict[str, Any]]:
    return [n["denoise"] for n in log.notes() if isinstance(n, dict) and "denoise" in n]


def _pieces_off_notes(log: LogSpy) -> list[Any]:
    """The A5 notes: ``{"denoise_relay": {"pieces_off": <reason>}}``."""
    return [
        n["denoise_relay"]["pieces_off"]
        for n in log.notes()
        if isinstance(n, dict)
        and isinstance(n.get("denoise_relay"), dict)
        and "pieces_off" in n["denoise_relay"]
    ]


async def _within(awaitable: Any, seconds: float, what: str) -> None:
    """``await`` it within ``seconds``, or fail with an assertion naming ``what`` (a pending row
    tolerates only ``AssertionError``, so a bare ``TimeoutError`` would read as a test bug)."""
    import asyncio

    try:
        await asyncio.wait_for(awaitable, timeout=seconds)
    except TimeoutError:
        raise AssertionError(f"T-0159 O-A2: {what} did not finish within {seconds} s") from None


async def _until_real(pred: Callable[[], bool], what: str, limit_s: float = 10.0) -> None:
    """Wait in real time (a deadline, never a measurement) for work a worker thread finishes."""
    import asyncio
    import time

    deadline = time.monotonic() + limit_s
    while not pred():
        assert time.monotonic() < deadline, f"T-0159: within {limit_s} s, {what}"
        await asyncio.sleep(0.01)


def test_every_published_frame_carries_the_pieces_of_its_own_ids() -> None:
    """A3: the recorded trajectory (51 frames, 40 ms apart, nothing coalesced) through a relay with
    the stub: every published step carries ``argmax_pieces`` (denoise) or ``committed_pieces``
    (commit) equal to ``#<id>`` over its own ids, every other field equals the received frame,
    and the published message re-validates as ``AstraCanvasStep`` (``model_copy`` skips
    validators; the implementation notes ask for proof of validity)."""
    rig = _piece_rig()
    frames = load_trajectory()["frames"]
    for frame in frames:
        rig.send(frame)
        rig.run(40)
    rig.run(1000)
    out = rig.steps()
    assert len(out) == 51, f"premise (T-0128c A2/A3): all 51 frames published: {rig.keys()}"
    for frame, message in zip(frames, out, strict=True):
        assert isinstance(message, AstraCanvasStep)
        assert_relayed(message, frame)
        argmax_p, committed_p = pieces_of(message)
        if frame["phase"] == "commit":
            assert committed_p is not None, f"A3: commit {key(frame)} published without pieces"
            assert tuple(committed_p) == tuple(stub_piece(i) for i in frame["committed_tokens"])
        else:
            assert argmax_p is not None, f"A3: denoise {key(frame)} published without pieces"
            assert tuple(argmax_p) == tuple(stub_piece(i) for i in frame["argmax"])
        again = AstraCanvasStep.model_validate(message.model_dump())
        assert pieces_of(again) == pieces_of(message), "A1: the published frame is a valid one"
    assert rig.relay.counts["undecoded"] == 0, dict(rig.relay.counts)


def test_a_frame_dropped_by_coalescing_is_never_decoded() -> None:
    """A3 adversarial (the failure mode: decoding in ``receive``): six denoise frames of one
    request in one virtual ms, then three more and a commit in another. By hand: the first flush
    publishes step 6 only (latest-wins), so steps 1-5 are coalesced, and the commit discards the
    pending step 9 or earlier ones. The counting stub saw exactly the ids of the published frames,
    in publish order -- flattened, so one call per frame or one per id are both allowed."""
    rig = _piece_rig()
    for s in range(1, 7):
        rig.send(denoise("A", 0, s), flush=False)
    rig.relay.flush()
    rig.run(100)
    for s in range(7, 10):
        rig.send(denoise("A", 0, s), flush=False)
    rig.send(commit("A", 0, 9), flush=False)
    rig.relay.flush()
    rig.run(500)
    out = rig.steps()
    assert rig.relay.counts["coalesced"] >= 5, f"premise: frames were coalesced: {rig.relay.counts}"
    shown = rig.keys()
    assert ("A", 0, 6, "denoise") in shown and ("A", 0, 9, "commit") in shown, shown
    decoded = [i for call in rig.pieces.calls for i in call]
    published = [i for m in out for i in _own_ids(m)]
    assert decoded == published, (
        f"A3: pieces are computed only for published frames; the stub decoded {len(decoded)} ids "
        f"for {len(out)} published frames ({rig.keys()})"
    )


def test_an_undecodable_frame_is_published_without_pieces_and_counted_undecoded() -> None:
    """A3: an argmax holding id 999 (the stub raises ``UnknownTokenId``) and a commit holding it
    are each published without pieces -- never dropped -- and counted ``undecoded``; the good frame
    between them carries its pieces. A decode failure is not a load failure: no
    ``denoise_no_pieces`` (A5 is about the load). The turn's note carries ``undecoded`` = 2, and
    the next turn's note counts its own (0)."""
    rig = _piece_rig(StubPieces(undecodable={999}))
    bad = denoise("A", 0, 1, argmax=[1, 2, 3, 999, 5, 6, 7, 8])
    good = denoise("A", 0, 2)
    bad_commit = commit("A", 0, 2, tokens=(11, 999))
    for frame in (bad, good, bad_commit):
        rig.send(frame)
        rig.run(50)
    out = rig.steps()
    assert [key(f) for f in (bad, good, bad_commit)] == rig.keys(), (
        f"A3: an undecodable frame is published, never dropped: {rig.keys()}"
    )
    for frame, message in zip((bad, good, bad_commit), out, strict=True):
        assert_relayed(message, frame)
    assert pieces_of(out[0]) == (None, None), f"A3: {pieces_of(out[0])}"
    assert pieces_of(out[2]) == (None, None), f"A3: {pieces_of(out[2])}"
    assert pieces_of(out[1])[0] is not None, "A3: a good frame after a bad one has its pieces"
    assert rig.relay.counts["undecoded"] == 2, dict(rig.relay.counts)
    assert rig.relay.counts["published"] == 3, dict(rig.relay.counts)
    assert not rig.published.errors(NO_PIECES_CODE), "A5 is the load's failure, not a frame's"
    rig.go(TurnState.IDLE)
    rig.go(TurnState.THINKING, "t2")
    rig.send(denoise("B", 0, 1))
    rig.run(50)
    rig.go(TurnState.IDLE)
    notes = _denoise_notes(rig.log)
    assert [n["turn_id"] for n in notes] == ["t1", "t2"], notes
    assert set(notes[0]) == NOTE_KEYS | {"undecoded"}, f"A3 amends A7: the note shape: {notes[0]}"
    assert notes[0]["undecoded"] == 2 and notes[1]["undecoded"] == 0, notes


async def test_a_failed_load_says_so_once_on_stderr_in_the_log_and_on_the_page(capsys) -> None:
    """A5: the loader raises. ``start`` does not; the relay still publishes (without pieces).
    Exactly one stderr line ``astra session: denoising text off: <reason>``, one session-log note
    ``{denoise_relay: {pieces_off: <reason>}}``, and one ``astra.error(denoise_no_pieces)`` for the
    session -- none before a frame needs it, and it comes with the first frame published without
    pieces (adjacent to it), not again in a second turn."""
    reason = "t0159-no-tokenizer-file-here"
    cls = relay_attr("CanvasRelay")
    rig = LiveRig(relay_kwargs=piece_kwargs(cls, piece_loader=failing_loader(reason)))
    await rig.start()
    # O-A2: the load runs in the background after ``start``; wait (real time, a deadline only)
    # until it has failed and said so, so the frames below are read after the failure
    await _until_real(lambda: bool(_pieces_off_notes(rig.log)), "the failed load was noted")
    await rig.advance(100)
    assert not rig.published.errors(NO_PIECES_CODE), "nothing on the page before a frame"
    for frame in (denoise("A", 0, 1), denoise("A", 0, 2), commit("A", 0, 2)):
        rig.push(frame)
        await _settle()
        await rig.advance(50)
    rig.box.go(TurnState.IDLE)
    await rig.advance(100)
    rig.box.go(TurnState.THINKING, "t2")
    await rig.advance(100)
    rig.push(denoise("B", 0, 1))
    await _settle()
    await rig.advance(100)
    await rig.relay.aclose()
    err = capsys.readouterr().err
    steps = [m for m in rig.published.messages if getattr(m, "type", "") == "astra.canvas.step"]
    assert [(m.request_id, m.step, m.phase) for m in steps] == [
        ("A", 1, "denoise"), ("A", 2, "denoise"), ("A", 2, "commit"), ("B", 1, "denoise")
    ], "A5: the relay still runs and publishes every frame"  # fmt: skip
    assert all(pieces_of(m) == (None, None) for m in steps), "A5: no source, no pieces"
    lines = [line for line in err.splitlines() if reason in line]
    assert len(lines) == 1, f"A5: one stderr line names the reason; stderr:\n{err}"
    assert lines[0].startswith(PIECES_OFF_PREFIX), f"A5: the line's wording: {lines[0]!r}"
    notes = _pieces_off_notes(rig.log)
    assert len(notes) == 1 and reason in str(notes[0]), f"A5: one note: {rig.log.records}"
    # O-A2 (5), reviewer T2: no decode was attempted after a failed load, so nothing is undecoded
    assert rig.relay.counts["undecoded"] == 0, dict(rig.relay.counts)
    turn_notes = _denoise_notes(rig.log)
    assert turn_notes and all(n.get("undecoded") == 0 for n in turn_notes), turn_notes
    errors = rig.published.errors(NO_PIECES_CODE)
    assert len(errors) == 1 and isinstance(errors[0], AstraError), rig.published.messages
    assert errors[0].message, "the error says something"
    at = rig.published.messages.index(errors[0])
    first = rig.published.messages.index(steps[0])
    assert abs(at - first) == 1, (
        f"A5: denoise_no_pieces goes out with the first frame without pieces (message {first}); "
        f"it is message {at}"
    )


async def test_start_returns_at_once_and_the_reader_waits_for_the_load() -> None:
    """O-A2 (1, 2), superseding O-A1: with a ``GatedLoader`` not yet released, ``start()`` returns
    within 2 s (it never awaits the load). The loader is called once. A frame pushed while the gate
    is closed is not published (no frame is read before the source is ready), however much
    virtual time passes; once the gate opens it is published, with the loaded stub's pieces."""

    cls = relay_attr("CanvasRelay")
    source = StubPieces()
    loader = GatedLoader(source)
    rig = LiveRig(relay_kwargs=piece_kwargs(cls, piece_loader=loader))
    frame = denoise("A", 0, 1)
    try:
        await _within(rig.relay.start(), 2.0, "start() with the load pending")
        await _until_real(lambda: loader.calls == 1, "the reader started the load")
        rig.push(frame)
        await _settle()
        await rig.advance(200)
        before_release = list(rig.published.canvas())
        loader.release()
        await _until_real(lambda: bool(rig.published.canvas()), "the frame was published")
    finally:
        loader.release()
        await rig.relay.aclose()
    assert before_release == [], f"O-A2 (2): a frame was read before the load: {before_release}"
    assert loader.calls == 1, f"the loader runs once ({loader.calls})"
    steps = rig.published.canvas()
    assert len(steps) == 1, steps
    assert_relayed(steps[0], frame)
    assert pieces_of(steps[0])[0] is not None, "O-A2: the first frame read has its pieces"
    assert source.calls, "the loaded source is the one the relay decodes with"


async def test_aclose_cancels_a_pending_load_and_closes_everything(capsys) -> None:
    """O-A2 (3), reviewer T1: a ``GatedLoader`` that is never released while the relay runs.
    ``aclose()`` returns within 2 s with the gate still closed (the pending load is cancelled, not
    awaited); the SUB is closed and the context terminated; a cancellation is not a failed load (no
    ``pieces_off`` note, no "denoising text off" line: a load wrapper that catches
    ``BaseException`` takes that path and keeps the reader alive); and nothing is read after
    ``aclose`` once the gate does open."""
    import asyncio

    cls = relay_attr("CanvasRelay")
    loader = GatedLoader(StubPieces())
    rig = LiveRig(relay_kwargs=piece_kwargs(cls, piece_loader=loader))
    try:
        await _within(rig.relay.start(), 2.0, "start() with the load pending")
        await _until_real(lambda: loader.calls == 1, "the reader started the load")
        await _within(rig.relay.aclose(), 2.0, "aclose() with the load pending")
        gated_at_close = not loader.done.is_set()
    finally:
        loader.release()
    await _until_real(lambda: loader.done.is_set(), "the released loader returned")
    rig.push(denoise("A", 0, 1))
    await asyncio.sleep(0.05)
    await _settle()
    err = capsys.readouterr().err
    assert gated_at_close, "premise: aclose returned while the load was still pending"
    sub = rig.zmq.sub()
    assert sub.closed, "O-A2 (3): the SUB is closed"
    assert rig.zmq.contexts and all(c.termed for c in rig.zmq.contexts), "the context is termed"
    assert not _pieces_off_notes(rig.log), f"O-A2 (3): a cancel took the A5 path: {rig.log.records}"
    assert PIECES_OFF_PREFIX not in err, f"O-A2 (3): a cancel was reported as a failed load:\n{err}"
    assert not rig.published.messages, (
        f"nothing is published after aclose: {rig.published.messages}"
    )


class _ShortPieces:
    """A piece source returning one piece too few (reviewer T3)."""

    def pieces(self, ids: Any) -> tuple[str, ...]:
        return tuple(stub_piece(int(i)) for i in ids)[:-1]


class _NonStringPieces:
    """A piece source whose last item is not a string (reviewer T3)."""

    def pieces(self, ids: Any) -> tuple[Any, ...]:
        return (*tuple(stub_piece(int(i)) for i in ids)[:-1], 5)


class _BrokenPieces:
    """A piece source that raises something other than ``UnknownTokenId`` (reading I2)."""

    def pieces(self, ids: Any) -> tuple[str, ...]:
        raise RuntimeError("t0159: the piece source is broken")


@pytest.mark.parametrize(
    "source",
    [_ShortPieces, _NonStringPieces, _BrokenPieces],
    ids=["one-short", "non-string", "raises"],
)
def test_a_source_that_returns_a_bad_row_publishes_the_frame_without_pieces(source: Any) -> None:
    """Reviewer T3 and O-A2 (5): the source's row is re-validated with the frame (``model_copy``
    would let a short row or a non-string through to the page). A denoise frame and a commit frame
    are each published -- never dropped -- without pieces, re-validate, and count ``undecoded``."""
    rig = _piece_rig(source())
    sent = [denoise("A", 0, 1), commit("A", 0, 1)]
    for frame in sent:
        rig.send(frame)
        rig.run(50)
    out = rig.steps()
    assert [key(f) for f in sent] == rig.keys(), f"published, never dropped: {rig.keys()}"
    for frame, message in zip(sent, out, strict=True):
        assert_relayed(message, frame)
        assert pieces_of(message) == (None, None), f"T3: a bad row reached the page: {message}"
        AstraCanvasStep.model_validate(message.model_dump())
    assert rig.relay.counts["undecoded"] == 2, dict(rig.relay.counts)


def test_a_source_raising_an_unexpected_error_goes_through_the_rate_limited_fault_log(
    caplog: Any,
) -> None:
    """O-A2 (5), reading I2: a ``RuntimeError`` from the source is not a decode verdict. Over 20
    frames 50 virtual ms apart, the relay logs at least one and at most two warning-or-worse
    records from ``local_astra`` (the O-A3 (2) rate limit), and every frame is still published."""
    import logging

    caplog.set_level(logging.WARNING)
    rig = _piece_rig(_BrokenPieces())
    for i in range(20):
        rig.send(denoise("A", 0, i + 1))
        rig.run(50)
    noisy = [r for r in caplog.records if r.name.startswith("local_astra") and r.levelno >= logging.WARNING]  # fmt: skip
    assert len(rig.steps()) == 20, rig.keys()
    assert 1 <= len(noisy) <= 2, (
        f"I2: an unexpected source error goes through the fault log, rate-limited: "
        f"{[r.getMessage() for r in noisy[:3]]}"
    )


def _lag_probe(stop: Any, cpu_gaps: list[float], wall_gaps: list[float]) -> Any:
    """A task waking every 1 ms: per wake, the loop thread's CPU time since the last wake (the
    gate: a callback that computes for 20 ms shows as a 20 ms gap whatever the box's load) and the
    wall time (printed, never gated -- ``.claude/rules/tests.md``)."""
    import asyncio
    import time

    async def run() -> None:
        cpu, wall = time.thread_time(), time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.001)
            c, w = time.thread_time(), time.perf_counter()
            cpu_gaps.append(c - cpu)
            wall_gaps.append(w - wall - 0.001)
            cpu, wall = c, w

    return asyncio.ensure_future(run())


async def test_frame_handling_lags_the_loop_under_twenty_ms_during_a_forty_ms_replay() -> None:
    """A4 as amended by O-A1: after ``start()`` returned, the recorded trajectory (51 frames) is
    pushed at 40 ms per frame through the relay's own reader on the session clock, with the stub
    source. The loop thread's CPU time between two 1 ms probe wakes stays < 20 ms, and the frames
    went out with their pieces (the discriminator: the BASE relay adds none).

    Regime: CPU, ``SessionClock`` on the test's loop, ``QueueZmq`` (no socket), trajectory 806da12
    (CL 256, 48 steps, 51 frames) at 40 ms per frame, n = 1 replay; prints max CPU and wall gaps."""
    import asyncio
    import os

    from local_astra.conductor.session import SessionClock

    cls = relay_attr("CanvasRelay")
    clock = SessionClock()
    published = Published(clock=clock)
    zmq = QueueZmq()
    relay = cls(
        publish=published,
        clock=clock,
        state=StateBox(TurnState.THINKING, "t1"),
        session_log=LogSpy(),
        endpoint=UNIT_ENDPOINT,
        zmq_module=zmq,
        **piece_kwargs(cls, piece_source=StubPieces()),
    )
    await relay.start()
    stop = asyncio.Event()
    cpu_gaps: list[float] = []
    wall_gaps: list[float] = []
    probe = _lag_probe(stop, cpu_gaps, wall_gaps)
    try:
        for frame in load_trajectory()["frames"]:
            zmq.sub().push_frame(frame)
            await asyncio.sleep(0.04)
        await asyncio.sleep(0.2)
    finally:
        stop.set()
        await probe
        await relay.aclose()
    steps = published.canvas()
    print(
        f"T-0159 A4 replay lag: max cpu {1000 * max(cpu_gaps):.2f} ms, max wall "
        f"{1000 * max(wall_gaps):.2f} ms, wakes {len(cpu_gaps)}, published {len(steps)}, "
        f"load {os.getloadavg()[0]:.2f}"
    )
    assert len(steps) >= 25, f"premise: the replay was published ({len(steps)} of 51)"
    assert all(any(p is not None for p in pieces_of(m)) for m in steps), (
        "A3: the replayed frames carry their pieces"
    )
    assert max(cpu_gaps) < 0.020, (
        f"A4: the loop computed {1000 * max(cpu_gaps):.1f} ms between two probe wakes (< 20 ms)"
    )


async def test_the_default_start_loads_the_pinned_detokeniser_before_reading() -> None:
    """O-A1 / A4, the real load: a relay given neither ``piece_source`` nor ``piece_loader`` loads
    ``HostDetokenizer`` from the pinned ``tokenizer.json`` in ``start()``, before it reads; the
    recorded trajectory pushed afterwards comes out with real pieces from the first frame on:
    every special position is its literal (``<|channel>`` and ``<channel|>``, 18 positions in the
    fixture, literals read off the pinned file by ``_detok_support``), each row as long as its ids,
    and handling any one frame costs the loop thread < 20 ms of CPU.

    Skips, naming the path, when the pinned file is not in the local HF cache. Regime: CPU, pinned
    sha cc8d3a0c (rev 2ea8372), ``ScriptedClock``, n = 1 load; prints the load time."""
    import time

    from local_astra.toolcall import detok
    from tests.unit._detok_support import SPECIALS_BY_ID

    path = detok.pinned_tokenizer_path()
    if not path.is_file():
        pytest.skip(f"T-0159 A4: the pinned tokenizer.json is not in the local HF cache ({path})")
    cls = relay_attr("CanvasRelay")
    piece_kwargs(cls, piece_source=None, piece_loader=None)  # the seam exists; defaults used below
    rig = LiveRig(relay_kwargs={})
    t0 = time.perf_counter()
    await rig.start()
    start_s = time.perf_counter() - t0
    # O-A2: the load runs in the reader after ``start`` returned; a warm-up frame of its own
    # request comes out once the reader has loaded (with pieces: O-A1 and O-A2 agree on that)
    rig.push(denoise("warm", 0, 1, canvas=[2, 3, 4, 5, 6, 7, 8, 9]))
    await _until_real(lambda: bool(rig.published.canvas()), "the reader loaded and read a frame")
    load_s = time.perf_counter() - t0
    warm = rig.published.canvas()[0]
    assert pieces_of(warm)[0] is not None, "the first frame read after the load has its pieces"
    await rig.advance(1000)
    specials = dict(SPECIALS_BY_ID)
    frames = load_trajectory()["frames"]
    cpu: list[float] = []
    for frame in frames:
        c0 = time.thread_time()
        rig.push(frame)
        await _settle()
        await rig.advance(40)
        cpu.append(time.thread_time() - c0)
    await rig.advance(1000)
    await rig.relay.aclose()
    print(
        f"T-0159 real load: start() {start_s:.3f} s, loaded and read by {load_s:.3f} s; "
        f"max cpu per frame {1000 * max(cpu):.2f} ms"
    )
    steps = [m for m in rig.published.canvas() if m.request_id != "warm"]
    assert len(steps) == 51, f"premise: the trajectory was published: {len(steps)}"
    checked = 0
    for frame, message in zip(frames, steps, strict=True):
        ids = frame["committed_tokens"] if frame["phase"] == "commit" else frame["argmax"]
        argmax_p, committed_p = pieces_of(message)
        got = committed_p if frame["phase"] == "commit" else argmax_p
        assert got is not None, f"O-A1: {key(frame)} went out without pieces after start()"
        assert len(got) == len(ids), f"A1: one piece per id in {key(frame)}"
        for token, piece in zip(ids, got, strict=True):
            if token in specials:
                assert piece == specials[token], f"A2: special {token} is {piece!r}"
                checked += 1
    assert checked == 18, f"premise: the fixture's 18 special positions were checked ({checked})"
    assert rig.relay.counts["undecoded"] == 0, dict(rig.relay.counts)
    assert max(cpu) < 0.020, f"A4: one frame cost the loop {1000 * max(cpu):.1f} ms of CPU"


async def test_a_thought_canvas_with_pieces_changes_nothing_spoken_stored_or_recorded(tmp_path):
    """A8 (and the failure mode named in the task's risks: thought text reaching another pane):
    the T-0128c A5 turn again, the relay fed the recorded trajectory -- thinking on, so thought
    canvases with ``<|channel>``/``<channel|>`` -- and decoding with a stub whose pieces are marked
    text (``secret<id>``). Against two runs with no relay: the non-canvas published messages
    (``astra.spoken`` included), the TTS requests and the history are byte-identical, the turn
    record is identical outside the volatile fields, no ``secret`` reaches any of them, and the
    relay did publish frames with pieces (else the row is vacuous)."""

    def rule(token_id: int) -> str:
        return "<|channel>" if token_id == 100 else f" secret{token_id}"

    frames = iter(load_trajectory()["frames"] * 20)
    opened = {"yes": False}

    def feed(rig: Any, relay: Any) -> None:
        if not opened["yes"]:
            if rig.session.turn_state_message().state is TurnState.IDLE:
                relay.flush()
                return
            opened["yes"] = True
        frame = next(frames, None)
        if frame is not None:
            relay.receive(pack(frame))
        relay.flush()

    cls = relay_attr("CanvasRelay")
    kwargs = piece_kwargs(cls, piece_source=StubPieces(rule=rule))
    base1_x: dict[str, Any] = {}
    base2_x: dict[str, Any] = {}
    with_x: dict[str, Any] = {}
    base1 = await _one_turn(tmp_path / "a", None, extras=base1_x)
    base2 = await _one_turn(tmp_path / "b", None, extras=base2_x)
    assert base1[0] == base2[0] and base1_x["tts"] == base2_x["tts"], "the rig is deterministic"
    assert base1_x["history"] == base2_x["history"], "the rig's history is deterministic"
    with_relay = await _one_turn(tmp_path / "c", feed, relay_kwargs=kwargs, extras=with_x)
    with_pieces = [m for m in with_x["canvas"] if any(p is not None for p in pieces_of(m))]
    assert with_pieces, "the relay published no frame with pieces -- this row would be vacuous"
    assert any("<|channel>" in (pieces_of(m)[0] or ()) for m in with_pieces), (
        "premise: a thought canvas (its <|channel> position) went out with pieces"
    )
    assert with_relay[0] == base1[0], "A8: the published non-canvas messages changed"
    assert with_x["tts"] == base1_x["tts"], "A8: the TTS requests changed"
    assert with_x["history"] == base1_x["history"], "A8: the history changed"
    assert len(with_relay[1]) == 1
    volatile = _volatile(base1[1][0], base2[1][0])
    assert _strip(with_relay[1][0], volatile) == _strip(base1[1][0], volatile), "A8: turn record"
    for text in (*with_relay[0], repr(with_x["tts"]), with_x["history"], json.dumps(with_relay[1])):
        assert "secret" not in text, f"A8: canvas text left the denoising view: {text[:200]}"


async def test_the_wire_out_row_of_a_frame_with_pieces_has_no_text(tmp_path):
    """A8: the session-log ``wire.out`` of a canvas step with pieces has the same keys, and the
    same ``msg``, as the same frame without them (BASE's compact row), and no piece text anywhere
    in the record (``n_bytes`` is the encoded size, which the pieces legitimately grow)."""
    from tests.unit.test_hud_wire_tap import _wire_out_of

    frame = denoise("A", 0, 3)
    plain = AstraCanvasStep.model_validate(frame)
    rich = _step_with_pieces({**frame, "argmax_pieces": [f"PIECE{i}<b>" for i in frame["argmax"]]})
    done = commit("A", 0, 3)
    plain_c = AstraCanvasStep.model_validate(done)
    rich_c = _step_with_pieces({**done, "committed_pieces": [f"PIECE{i}" for i in done["committed_tokens"]]})  # fmt: skip
    for name, a, b in (("denoise", plain, rich), ("commit", plain_c, rich_c)):
        base = await _wire_out_of(tmp_path / f"{name}-plain", a)
        got = await _wire_out_of(tmp_path / f"{name}-rich", b)
        assert set(got) == set(base), (sorted(got), sorted(base))
        assert got["msg"] == base["msg"], f"A8: the {name} wire.out row changed: {got['msg']}"
        assert "PIECE" not in json.dumps(got), f"A8: piece text in the {name} wire.out: {got}"


def test_the_turn_journal_ignores_a_canvas_frame_with_pieces() -> None:
    """A8: ``TurnJournal`` still ignores ``astra.canvas.*`` -- a step with pieces is not journalled
    (a reconnecting page is never replayed thought text) and opens no turn."""
    from local_astra.conductor.hud_server import TurnJournal

    frame = denoise("A", 0, 3)
    journal = TurnJournal()
    journal.record(_step_with_pieces({**frame, "argmax_pieces": [f"#{i}" for i in frame["argmax"]]}))  # fmt: skip
    journal.record(_step_with_pieces({**commit("A", 0, 3), "committed_pieces": ["#11", "#12", "#13"]}))  # fmt: skip
    assert journal.messages() == [] and journal.turn_id is None, journal.messages()


def _longest_pinned_piece() -> str:
    """The pinned vocab's piece with the longest UTF-8 JSON encoding, under the A2 display rule
    (``▁`` -> space; added tokens verbatim), read with ``json`` -- not through the code under test."""
    from local_astra.toolcall import detok

    path = detok.pinned_tokenizer_path()
    if not path.is_file():
        pytest.skip(f"T-0159 A7: the pinned tokenizer.json is not in the local HF cache ({path})")
    data = json.loads(path.read_bytes())
    candidates = [p.replace("▁", " ") for p in data["model"]["vocab"]]
    candidates += [a["content"] for a in data["added_tokens"]]
    return max(candidates, key=lambda p: len(json.dumps(p, ensure_ascii=False).encode("utf-8")))


@pytest.mark.parametrize("piece", ["pinned", "pessimistic"])
def test_a_worst_case_frame_fits_the_codec_and_the_outbox(piece: str) -> None:
    """A7: a CL 256 denoise frame relayed with every piece the longest one -- the pinned vocab's
    (skipped when the file is absent) or a pessimistic 1024 x NUL (6 JSON bytes each, ~1.5 MiB a
    row, far above any vocab piece) -- encodes under ``codec.MAX_MESSAGE_BYTES`` and decodes back;
    ``Outbox.offer`` takes it as a droppable canvas message (nine of them hold at most
    ``CANVAS_OUTBOX`` = 8; a full outbox of them still takes a non-canvas message)."""
    from local_astra.conductor import hud_server
    from local_astra.contracts import codec

    longest = _longest_pinned_piece() if piece == "pinned" else "\x00" * 1024
    rig = _piece_rig(StubPieces(rule=lambda _i: longest))
    rig.send(denoise("A", 0, 1, canvas=list(range(256)), stable="0" * 64))
    rig.run(100)
    (message,) = rig.steps()
    assert pieces_of(message)[0] == (longest,) * 256, "A3: the relayed frame carries the pieces"
    raw = codec.encode(message)
    size = len(raw.encode("utf-8"))
    print(f"T-0159 A7 [{piece}]: piece {len(longest)} chars, frame {size} bytes")
    assert size < codec.MAX_MESSAGE_BYTES, f"A7: {size} bytes >= {codec.MAX_MESSAGE_BYTES}"
    back = codec.decode(raw)
    assert isinstance(back, AstraCanvasStep) and pieces_of(back) == pieces_of(message), type(back)
    assert hud_server.CANVAS_OUTBOX == 8, "A7: the Outbox cap stays 8 canvas messages"
    outbox = hud_server.Outbox()
    for n in range(9):
        assert outbox.offer(message.model_copy(update={"request_id": f"r{n}"})), "droppable"
    assert len(outbox) <= 8 and outbox.dropped >= 1, (len(outbox), outbox.dropped)
    full = hud_server.Outbox(maxsize=8)
    for n in range(8):
        full.offer(message.model_copy(update={"request_id": f"r{n}"}))
    line = AstraError(code="t0159", message="a line that must not be refused")
    assert full.offer(line), "A7: a non-canvas message evicts a canvas frame, never refused"
