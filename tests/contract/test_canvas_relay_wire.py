"""T-0128c group A, the socket half of the conductor's tap relay (criteria A1, A2, A3, A5).

``tests/unit/test_canvas_relay.py`` pins the relay's logic with no socket; this module runs the relay
(``local_astra/conductor/canvas_relay.py``, the API fixed in that module's docstring and by ruling O-A)
against a real pyzmq PUB:

* ``fake`` -- ``tests/fakes/canvas_tap.FakeTap``: the plugin's own ``StepPublisher`` (topic + msgpack, a
  PUB that binds) on ``ipc://`` in a short pytest temp directory, replaying the recorded 806da12
  trajectory (``tests/fixtures/canvas/MANIFEST.md``). Never port 8002 (task, implementation notes).
* ``real`` (``gpu``, ``/launch`` only) -- the engine's tap on ``tcp://127.0.0.1:8002`` plus one short chat
  request to ``http://127.0.0.1:8001``. What the tap actually sent is observed by an independent raw SUB
  (its own context), which is the oracle the relay's output is compared with. The row prints the
  ``request_id`` it saw and whether it has the ``chatcmpl-<16 hex>-<8 hex>`` shape the recording has
  (reported, not gated: the shape is unverified on the live engine, task finding 5).

The relay is given ``SpyZmq`` (``tests/fakes/canvas_tap.py``) as ``zmq_module``: real pyzmq, with every
context and socket recorded, ``bind`` and ``Context.instance()`` refused by a named assertion.

Regime: fake -- no GPU, ``SessionClock`` over the test's event loop, loopback ipc, the 806da12 recording
(lean, CL 256, 48 steps, 51 frames, 5 commits) replayed at one frame per 5 ms. Real -- see the regime line
the row prints (lean, CL 256, 48 steps, thinking on, ``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev 2ea8372,
vLLM v0.29.0, n = 1 request, unseeded). Wall time appears only as a deadline for waiting; the cost bound
of the silent row is process CPU time and a count of wake-ups (``.claude/rules/tests.md``).
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import msgpack
import pytest

from local_astra.conductor.session import SessionClock
from local_astra.contracts.protocol import AstraCanvasStep
from local_astra.contracts.turn import TurnState
from tests.fakes.canvas_tap import (
    TOPIC,
    FakeTap,
    Published,
    SpyZmq,
    StateBox,
    StubPieces,
    assert_relayed,
    denoise,
    load_trajectory,
    optional_piece_kwargs,
    pack,
    piece_kwargs,
    pieces_of,
    relay_attr,
    without_pieces,
)

pytestmark = pytest.mark.contract

REPO = Path(__file__).resolve().parents[2]
ENGINE = "http://127.0.0.1:8001"
REAL_TAP = "tcp://127.0.0.1:8002"
#: the shape of the recorded id (``tests/fixtures/canvas/MANIFEST.md``); reported, never asserted
RECORDED_ID_SHAPE = re.compile(r"^chatcmpl-[0-9a-f]{16}-[0-9a-f]{8}$")
#: A5 / implementation notes: the receive high-water mark is bounded, and 0 means unbounded in ZeroMQ
MAX_RCVHWM = 256


# --------------------------------------------------------------------------- helpers


def key(frame: dict[str, Any]) -> tuple[str, int, int, str]:
    return (frame["request_id"], frame["canvas_idx"], frame["step"], frame["phase"])


def validated(frame: dict[str, Any]) -> dict[str, Any]:
    return AstraCanvasStep.model_validate(frame).model_dump()


def steps(published: Published) -> list[AstraCanvasStep]:
    return [m for m in published.messages if getattr(m, "type", "") == "astra.canvas.step"]


async def until(pred: Callable[[], bool], what: str, *, limit_s: float = 5.0) -> None:
    """Wait (a deadline, never a measurement) for ``pred``; fail naming ``what``."""
    deadline = time.monotonic() + limit_s
    while not pred():
        if time.monotonic() > deadline:
            raise AssertionError(f"T-0128c: within {limit_s} s, {what}")
        await asyncio.sleep(0.01)


class Observer:
    """An independent raw SUB on the tap (its own context): what the tap really sent."""

    def __init__(self, endpoint: str) -> None:
        import zmq

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.SUBSCRIBE, TOPIC)
        self._sock.connect(endpoint)
        self.frames: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._sock.poll(timeout=20):
                _topic, payload = self._sock.recv_multipart()
                self.frames.append(msgpack.unpackb(payload, raw=False))

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2.0)
        self._sock.close(linger=0)
        self._ctx.term()


def short_ipc(tmp_path_factory: pytest.TempPathFactory, name: str = "tap") -> str:
    """``ipc://`` under a short temp dir: a socket path over ~107 bytes is refused by the kernel."""
    return f"ipc://{tmp_path_factory.mktemp('t0128c') / f'{name}.sock'}"


def require_engine() -> str:
    """The real backend's health check: the served model id, or a skip naming what is missing."""
    import httpx

    try:
        reply = httpx.get(f"{ENGINE}/v1/models", timeout=5.0)
        return str(reply.json()["data"][0]["id"])
    except Exception as exc:  # the row runs under /launch with the engine up; otherwise skip
        pytest.skip(f"no engine on {ENGINE} ({type(exc).__name__}); real rows run under /launch")


def regime_line(row: str, **extra: Any) -> None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    print(
        f"T-0128c contract regime [{row}]: head={git('rev-parse', '--short', 'HEAD')} "
        f"dirty={bool(git('status', '--porcelain'))} lean CL256 steps48 thinking=on "
        f"vLLM0.29.0 rev2ea8372 n=1 seed=none {json.dumps(extra, sort_keys=True)}"
    )


class Run:
    """One relay on the injected spy, started on the test's loop with the session's clock."""

    def __init__(self, endpoint: str, state: TurnState = TurnState.THINKING) -> None:
        self.spy = SpyZmq()
        self.clock = SessionClock()
        self.published = Published(clock=self.clock)
        self.state_calls = 0
        box = StateBox(state, "t1")

        def state_fn() -> Any:
            self.state_calls += 1
            return box()

        self.box = box
        self.endpoint = endpoint
        cls = relay_attr("CanvasRelay")
        #: T-0159: a stub piece source when the relay declares the seam (``#<id>``), so ``start``
        #: never loads the real tokenizer here; BASE (no seam) gets nothing and adds no pieces
        self.pieces = StubPieces()
        self.relay = cls(
            publish=self.published,
            clock=self.clock,
            state=state_fn,
            endpoint=endpoint,
            zmq_module=self.spy,
            **optional_piece_kwargs(cls, self.pieces),
        )

    async def __aenter__(self) -> Run:
        await self.relay.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.relay.aclose()

    def subs(self) -> list[Any]:
        import zmq

        return [s for s in self.spy.ledger.sockets if s.type == zmq.SUB]


async def warm(tap: FakeTap, run: Run, rid: str = "warm") -> None:
    """Send frames of a throwaway request until the relay has received one (slow joiner)."""
    step = 0
    deadline = time.monotonic() + 5.0
    while run.relay.counts["received"] == 0:
        assert time.monotonic() < deadline, (
            "A1/A5: no frame of the fake tap reached the relay's SUB in 5 s "
            f"(connects: {[a for _, a in run.spy.ledger.connects]})"
        )
        step = step % 48 + 1
        tap.send(denoise(rid, 0, step))
        await asyncio.sleep(0.02)


def assert_published_is_a_subset_of_sent(published: Published, sent: list[dict[str, Any]]) -> None:
    """A2: every published step equals one frame the tap sent, field for field, none twice.

    T-0159 A3 amends A2 to "unchanged except the T-0159 fields": those, when present, are the stub
    source's pieces (``#<id>``) of the frame's own ids (``assert_relayed``)."""
    by_key = {key(f): f for f in sent}
    out = steps(published)
    assert out, "A2: the relay published no frame of what the tap sent"
    keys = [(m.request_id, m.canvas_idx, m.step, m.phase) for m in out]
    assert len(keys) == len(set(keys)), f"A2: a frame was published twice: {keys}"
    for m in out:
        k = (m.request_id, m.canvas_idx, m.step, m.phase)
        assert k in by_key, f"A2: published {k}, which the tap never sent"
        assert_relayed(m, by_key[k])  # A2 as amended by T-0159 A3
    assert not [m for m in published.messages if getattr(m, "type", "") == "astra.canvas.synthetic"]


# --------------------------------------------------------------------------- A2/A3 over the wire


async def test_tap_frames_reach_the_relay_and_are_published_unchanged(
    backend: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A2/A3 over a real socket, a turn open (THINKING) the whole time.

    fake: the 806da12 recording replayed through the plugin's own publisher. real: the engine's tap
    while one short chat request runs. Both: >= 1 frame published; every published step equals a
    frame the tap sent (``FakeTap.sent`` / the independent observer); every commit the tap sent is
    published, in order (commits are never coalesced away); nothing was counted invalid.
    """
    if backend == "fake":
        endpoint = short_ipc(tmp_path_factory)
        tap = FakeTap(endpoint).bind()
        try:
            async with Run(endpoint) as run:
                await warm(tap, run)
                frames = load_trajectory()["frames"]
                for frame in frames:
                    tap.send(frame)
                    await asyncio.sleep(0.005)
                commits = [key(f) for f in tap.sent if f["phase"] == "commit"]
                assert len(commits) == 5, f"the recording's 5 commits left the fake tap: {commits}"
                await until(
                    lambda: all(c in {key(m.model_dump()) for m in steps(run.published)} for c in commits),
                    "every commit the tap sent was published (A3: commits never lost)",
                )  # fmt: skip
                sent = list(tap.sent)
                counts = dict(run.relay.counts)
        finally:
            tap.close()
    else:
        import httpx

        model = require_engine()
        endpoint = relay_attr("DEFAULT_ENDPOINT")
        assert endpoint == REAL_TAP
        observer = Observer(endpoint)
        try:
            async with Run(endpoint) as run:
                await asyncio.sleep(0.5)  # both SUBs joined before the request starts
                async with httpx.AsyncClient(timeout=120.0) as client:
                    reply = await client.post(
                        f"{ENGINE}/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "What is 17 times 23?"}],
                            "max_tokens": 64,
                            "chat_template_kwargs": {"enable_thinking": True},
                        },
                    )
                assert reply.status_code == 200, reply.text
                await asyncio.sleep(1.0)  # the final commit and the next flush
                sent = list(observer.frames)
                counts = dict(run.relay.counts)
                commits = [key(f) for f in sent if f.get("phase") == "commit"]
        finally:
            observer.close()
        ids = sorted({str(f.get("request_id")) for f in sent})
        regime_line(
            "frames",
            request_ids=ids,
            recorded_shape=[bool(RECORDED_ID_SHAPE.match(i)) for i in ids],
            observed=len(sent),
            relay_counts=counts,
        )
        assert sent, "the independent observer saw no tap frame: is the tap on (xargs astra_tap)?"

    assert_published_is_a_subset_of_sent(run.published, sent)
    out_commits = [
        (m.request_id, m.canvas_idx, m.step, m.phase) for m in steps(run.published) if m.phase == "commit"
    ]  # fmt: skip
    assert [c for c in out_commits if c in commits] == commits, (
        f"A3: every commit the tap sent is published once, in order: sent {commits}, "
        f"published {out_commits}"
    )
    assert counts["invalid"] == 0, f"A2: real tap frames all validate: {counts}"


# --------------------------------------------------------------------------- A1/A5: the socket itself


async def test_the_relay_owns_one_sub_that_connects_bounded_and_closed_before_term(
    backend: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A1 + implementation notes: one SUB, which ``connect``s to the endpoint and never ``bind``s
    (``SpyZmq`` refuses ``bind`` with a named assertion); ``RCVHWM`` in 1..256 (0 is unbounded);
    ``LINGER`` 0; its own context (``Context.instance()`` refused by the spy); on ``aclose`` every
    socket of each context is closed before that context's ``term()``, and every context is termed."""
    if backend == "real":
        require_engine()
        endpoint = relay_attr("DEFAULT_ENDPOINT")
    else:
        endpoint = short_ipc(tmp_path_factory)
    run = Run(endpoint)
    import zmq

    async with run:
        subs = run.subs()
        assert len(subs) == 1, (
            f"A1: the relay makes exactly one SUB on the injected zmq_module; it made "
            f"{[s.type for s in run.spy.ledger.sockets]} on {len(run.spy.ledger.contexts)} "
            "context(s)"
        )
        sub = subs[0]
        assert [(s, a) for s, a in run.spy.ledger.connects] == [(sub, endpoint)], (
            f"A1: the SUB connects to {endpoint!r} once; connects: "
            f"{[a for _, a in run.spy.ledger.connects]}"
        )
        hwm = sub.getsockopt(zmq.RCVHWM)
        assert 1 <= hwm <= MAX_RCVHWM, f"A5: RCVHWM is {hwm} (0 is unbounded); bound it at <= 256"
        assert sub.getsockopt(zmq.LINGER) == 0, "LINGER 0: closing never waits on a dead peer"
        assert run.spy.ledger.contexts, "the relay owns a context of its own"
    assert sub.closed, "A5: aclose closes the SUB"
    termed = [ctx for ctx, _ in run.spy.ledger.terms]
    assert all(ctx in termed for ctx in run.spy.ledger.contexts), (
        f"aclose terms every context it made: made {len(run.spy.ledger.contexts)}, termed {len(termed)}"
    )
    for ctx, closed in run.spy.ledger.terms:
        assert all(closed), f"a socket of {ctx!r} was still open when term() ran: {closed}"


async def test_a_silent_endpoint_publishes_nothing_and_costs_almost_nothing(
    backend: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A5 "a silent port publishes nothing and costs nothing": the relay runs 0.6 s against an
    endpoint where nothing publishes (fake: an ipc path nobody binds; real: the engine's tap with no
    request running), a turn open the whole time.

    By hand: nothing received, nothing published. Cost, counted, not timed against the wall: process
    CPU over the window < 0.2 s (a busy poll costs ~0.6 s; a timer flush at <= 250 Hz costs well under
    0.05 s), and the relay's flushes (each reads ``state()``, ruling O-A) <= 150 in the window
    (<= 250/s, an upper bound only)."""
    if backend == "real":
        require_engine()
        endpoint = relay_attr("DEFAULT_ENDPOINT")
    else:
        endpoint = short_ipc(tmp_path_factory, "silent")
    run = Run(endpoint)
    async with run:
        await asyncio.sleep(0.05)  # past start-up work
        calls0, cpu0 = run.state_calls, time.process_time()
        await asyncio.sleep(0.6)
        cpu = time.process_time() - cpu0
        calls = run.state_calls - calls0
        counts = dict(run.relay.counts)
    assert counts["received"] == 0, f"nothing publishes on {endpoint}; the relay received {counts}"
    assert not run.published.messages, (
        f"A5: a silent port publishes nothing: {run.published.messages}"
    )
    assert cpu < 0.2, f"A5: a silent port costs nothing; the process spent {cpu:.3f} s CPU in 0.6 s"
    assert calls <= 150, f"A5: {calls} flushes in 0.6 s of silence (bound 150)"


async def test_the_relay_reconnects_when_the_tap_restarts(
    backend: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A5 "engine restarts are survived by ZeroMQ's reconnect": a frame gets through, the tap closes
    and a new tap binds the same endpoint (an engine restart), and a frame of a new request sent
    after the restart is published -- with no call into the relay in between.

    real: skipped; an engine restart is not part of the C2 ``/launch`` smoke (reported)."""
    if backend == "real":
        pytest.skip("an engine restart is not driven by the /launch smoke; fake tap only (T-0128c)")
    endpoint = short_ipc(tmp_path_factory, "restart")
    tap = FakeTap(endpoint).bind()
    async with Run(endpoint) as run:
        try:
            await warm(tap, run, "before")
        finally:
            tap.close()
        await asyncio.sleep(0.2)
        tap = FakeTap(endpoint).bind()
        try:
            step = 0
            deadline = time.monotonic() + 5.0
            while not [m for m in steps(run.published) if m.request_id == "after"]:
                assert time.monotonic() < deadline, (
                    "A5: after the tap restarted no frame of the new request was published in 5 s "
                    f"(counts {dict(run.relay.counts)})"
                )
                step = step % 48 + 1
                tap.send(denoise("after", 0, step))
                await asyncio.sleep(0.02)
        finally:
            tap.close()
        sent = [f for f in tap.sent]
        published = [m for m in steps(run.published) if m.request_id == "after"]
        for m in published:
            k = (m.request_id, m.canvas_idx, m.step, m.phase)
            assert without_pieces(m.model_dump()) in [
                without_pieces(validated(f)) for f in sent if key(f) == k
            ], f"A2 (amended by T-0159 A3): {k} after the restart is one the new tap sent"
            assert_relayed(m, next(f for f in sent if key(f) == k))


# --------------------------------------------------------------------------- O-A2 (reviewer 1)


async def _until_published(run: Run, tap: FakeTap, rid: str) -> None:
    """Send one sentinel frame of ``rid`` and wait for it: ZeroMQ keeps order on one pipe, so
    everything sent before it has been handled once it is out."""
    tap.send(denoise(rid, 0, 1))
    await until(
        lambda: any(m.request_id == rid for m in steps(run.published)),
        f"the sentinel {rid!r} was published",
    )


async def test_only_the_step_topic_is_subscribed_and_a_malformed_multipart_is_invalid(
    backend: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """O-A2 M23 and M24, through the plugin's own PUB.

    * M23: the SUB subscribes to ``astra.canvas.step`` and nothing else -- every ``SUBSCRIBE``
      option the relay set is that topic, and a valid frame published under ``astra.other`` never
      reaches the relay (not received, not counted, not published).
    * M24: under the step topic, a one-part ``[topic]`` and a three-part ``[topic, payload, x]``
      message are each counted ``received`` and ``invalid`` and never published.

    By hand, after the warm-up: received +3 (two malformed, one sentinel), invalid +2.
    real: skipped -- the engine's PUB cannot be made to send these (fake tap only, reported)."""
    if backend == "real":
        pytest.skip("the engine's tap cannot be made to send other topics or malformed parts")
    import zmq

    endpoint = short_ipc(tmp_path_factory, "topic")
    tap = FakeTap(endpoint).bind()
    try:
        async with Run(endpoint) as run:
            await warm(tap, run)
            await _until_published(run, tap, "warm-done")
            before = dict(run.relay.counts)
            raw = tap.publisher._socket()
            raw.send_multipart([b"astra.other", pack(denoise("other", 0, 1))])
            raw.send_multipart([TOPIC])
            raw.send_multipart([TOPIC, pack(denoise("three", 0, 1)), b"x"])
            await _until_published(run, tap, "sentinel")
            after = dict(run.relay.counts)
            subscribed = [
                v for s, opt, v in run.spy.ledger.options if opt == zmq.SUBSCRIBE and s in run.subs()
            ]  # fmt: skip
    finally:
        tap.close()
    assert subscribed and all(bytes(v) == TOPIC for v in subscribed), (
        f"M23: the SUB subscribes to {TOPIC!r} only; SUBSCRIBE options set: {subscribed}"
    )
    ids = {m.request_id for m in steps(run.published)}
    assert "other" not in ids and "three" not in ids, f"M23/M24: published {sorted(ids)}"
    assert after["received"] - before["received"] == 3, (
        f"M23/M24: two malformed messages and the sentinel are received, the other topic is "
        f"not: before {before}, after {after}"
    )
    assert after["invalid"] - before["invalid"] == 2, (
        f"M24: a 1-part and a 3-part message under the topic are invalid: {before} -> {after}"
    )


async def test_a_start_that_fails_at_connect_closes_its_socket_and_aclose_returns(
    backend: str,
) -> None:
    """O-A2 M20: ``ipc://`` + a 200-character path passes ``check_endpoint`` but ZeroMQ refuses it
    at ``connect`` (``File name too long``). The failed start leaves nothing open: every socket it
    made is closed before its context's ``term()`` (``SpyZmq`` fails, rather than hangs, on an open
    one), every context is terminated, and a later ``aclose`` returns within 2 s.

    real: skipped -- no engine is involved in a refused connect (fake only, reported)."""
    if backend == "real":
        pytest.skip("a refused connect involves no engine")
    endpoint = "ipc:///tmp/" + "x" * 200
    assert relay_attr("check_endpoint")(endpoint) == endpoint, "premise: A1 accepts the shape"
    run = Run(endpoint)
    raised: BaseException | None = None
    try:
        await run.relay.start()
    except Exception as exc:  # the CLI reports it (A8); here only what it left behind matters
        raised = exc
    assert not isinstance(raised, AssertionError), f"M20: {raised}"
    await asyncio.wait_for(run.relay.aclose(), timeout=2.0)
    assert run.spy.ledger.sockets, "premise: the relay got as far as making its SUB"
    assert all(s.closed for s in run.spy.ledger.sockets), "M20: a socket was left open"
    termed = [ctx for ctx, _ in run.spy.ledger.terms]
    assert all(ctx in termed for ctx in run.spy.ledger.contexts), "M20: a context was not termed"
    for _ctx, closed in run.spy.ledger.terms:
        assert all(closed), f"M20: term() ran with a socket still open: {closed}"


# --------------------------------------------------------------------------- T-0159 A3 over the wire


async def test_frames_relayed_over_a_real_socket_carry_the_stub_pieces_of_their_own_ids(
    backend: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """T-0159 A3 over real pyzmq, a stub piece source (``#<id>``) through the frozen seam, a turn
    open: every published step carries ``argmax_pieces`` (denoise) or ``committed_pieces``
    (commit), equal to the stub over that frame's own ids, and is otherwise the frame the tap sent
    (A2 as amended). fake: the 806da12 recording through the plugin's own publisher. real: the
    engine's tap during one short request, observed independently (as the A2 row above)."""
    cls = relay_attr("CanvasRelay")
    piece_kwargs(cls, piece_source=None)  # the seam exists, or an assertion naming T-0159
    if backend == "fake":
        endpoint = short_ipc(tmp_path_factory, "pieces")
        tap = FakeTap(endpoint).bind()
        try:
            async with Run(endpoint) as run:
                await warm(tap, run)
                for frame in load_trajectory()["frames"]:
                    tap.send(frame)
                    await asyncio.sleep(0.005)
                await _until_published(run, tap, "pieces-sentinel")
                sent = list(tap.sent)
        finally:
            tap.close()
    else:
        import httpx

        model = require_engine()
        observer = Observer(REAL_TAP)
        try:
            async with Run(REAL_TAP) as run:
                await asyncio.sleep(0.5)
                async with httpx.AsyncClient(timeout=120.0) as client:
                    reply = await client.post(
                        f"{ENGINE}/v1/chat/completions",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "What is 17 times 23?"}],
                            "max_tokens": 64,
                            "chat_template_kwargs": {"enable_thinking": True},
                        },
                    )
                assert reply.status_code == 200, reply.text
                await asyncio.sleep(1.0)
                sent = list(observer.frames)
        finally:
            observer.close()
        regime_line("pieces", observed=len(sent))
    assert_published_is_a_subset_of_sent(run.published, sent)
    for m in steps(run.published):
        argmax_p, committed_p = pieces_of(m)
        own = committed_p if m.phase == "commit" else argmax_p
        assert own is not None, f"T-0159 A3: {m.request_id}/{m.canvas_idx}/{m.step} has no pieces"
    assert run.pieces.calls, "the relay decoded with the injected stub"
