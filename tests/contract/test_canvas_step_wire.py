"""T-0011 criteria 8 and 6: the `astra.canvas.step` wire format and its ZeroMQ framing.

Two halves:

* **the contract** — the amended ``AstraCanvasStep`` (``canvas``, ``argmax``, ``stable``,
  ``phase``, ``committed_tokens`` real; ``accepted`` and ``entropy`` optional and ``None`` on the
  plugin path). These replace the two canvas-step tests that used to live in
  ``tests/contract/test_protocol_codec.py`` against the pre-amendment shape.
* **the framing** — one PUB message per step, two frames (``b"astra.canvas.step"`` then a msgpack
  payload), a bounded ``SNDHWM``, ``LINGER 0`` and ``NOBLOCK`` so that a slow subscriber drops
  frames instead of stalling the decode loop, and a constructor that refuses any endpoint that is
  not loopback (docs/02 §9, CLAUDE.md).

``StepPublisher`` is the seam this file fixes:

    StepPublisher(endpoint=ASTRA_TAP_ENDPOINT, *, hwm=64, zmq_module=None)
        TOPIC = b"astra.canvas.step"
        send(msg: dict) -> bool      # False when the frame was dropped at the HWM
        published / dropped counters, close(), classmethod from_env()

``zmq_module`` exists so the framing is checkable on a host that has no pyzmq (this one does
not — see the report for T-0011); the real module is imported lazily when it is ``None``. The
fake's constants are checked against the installed pyzmq by
``test_fake_zmq_constants_match_the_installed_pyzmq`` whenever pyzmq is available, as
`.claude/rules/tests.md` requires of a fake.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import msgpack
import pytest

from local_astra.contracts.codec import MAX_MESSAGE_BYTES, decode, encode
from local_astra.contracts.protocol import AstraCanvasStep
from tests.fakes.diffusion_sampler import (
    FakeZmq,
    attr,
    build_canvas_step,
    call_new_api,
    load_plugin_module,
    validate_canvas_step,
)

REPO = Path(__file__).resolve().parents[2]
HAVE_PYZMQ = importlib.util.find_spec("zmq") is not None
LOOPBACK_ENDPOINT = "tcp://127.0.0.1:8002"


@pytest.fixture(params=["fake"])
def backend(request: pytest.FixtureRequest) -> str:
    """Only the fake subscriber exists until T-0011b adds the Conductor's relay (docs/tasks)."""
    return str(request.param)


def _valid_step(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": "astra.canvas.step",
        "request_id": "cmpl-1",
        "canvas_idx": 0,
        "step": 3,
        "max_steps": 48,
        "canvas_length": 4,
        "canvas": (1, 2, 3, 4),
        "argmax": (1, 2, 3, 9),
        "stable": "a0",
        "phase": "denoise",
    }
    base.update(overrides)
    return base


def _publisher(**kwargs: Any) -> tuple[Any, FakeZmq]:
    tapmod = load_plugin_module("tap", "8")
    zmq = FakeZmq()
    cls = attr(tapmod, "StepPublisher", "8")
    pub = call_new_api(cls, "8", endpoint=LOOPBACK_ENDPOINT, zmq_module=zmq, **kwargs)
    return pub, zmq


# --------------------------------------------------------------------------- the contract


_ROWS = ("canvas", "argmax", "stable")


def _commit_step(**overrides: Any) -> dict[str, Any]:
    """R10's commit shape: committed ids, none of the three rows."""
    base = {k: v for k, v in _valid_step().items() if k not in _ROWS}
    base.update(phase="commit", committed_tokens=(7, 8))
    base.update(overrides)
    return base


def test_amended_canvas_step_carries_the_real_fields() -> None:
    """Criterion 6: canvas, argmax, stable, phase and committed_tokens are present and real -- the
    three rows on a denoise frame, the committed ids on a commit frame (R10: one shape per phase)."""
    m = build_canvas_step(**_valid_step())
    assert isinstance(m, AstraCanvasStep)
    assert tuple(m.canvas) == (1, 2, 3, 4)
    assert tuple(m.argmax) == (1, 2, 3, 9)
    assert m.stable == "a0"
    assert m.phase == "denoise"
    c = build_canvas_step(**_commit_step())
    assert c.phase == "commit"
    assert tuple(c.committed_tokens) == (7, 8)
    assert c.canvas is None and c.argmax is None and c.stable is None


@pytest.mark.parametrize(
    "missing",
    [("canvas",), ("argmax",), ("stable",), ("canvas", "argmax"), ("argmax", "stable"), _ROWS],
    ids=lambda m: "-".join(m),
)
def test_a_denoise_frame_requires_all_three_rows(missing: tuple[str, ...]) -> None:
    """R10: a ``denoise`` frame carries all of canvas/argmax/stable; a partial or empty set of rows
    is refused (the HUD would otherwise draw a canvas with no mask, or a mask with no canvas)."""
    frame = {k: v for k, v in _valid_step().items() if k not in missing}
    with pytest.raises(ValueError):
        AstraCanvasStep(**frame)


def test_a_denoise_frame_refuses_committed_tokens() -> None:
    """R10: committed ids belong to a commit frame only."""
    with pytest.raises(ValueError):
        AstraCanvasStep(**_valid_step(committed_tokens=(7, 8)))


@pytest.mark.parametrize(
    "present",
    [("canvas",), ("argmax",), ("stable",), ("canvas", "argmax"), _ROWS],
    ids=lambda p: "-".join(p),
)
def test_a_commit_frame_refuses_any_of_the_three_rows(present: tuple[str, ...]) -> None:
    """R10: a ``commit`` frame carries **none** of canvas/argmax/stable, whether published on its
    own call or lagged (the HUD draws the closed canvas from ``committed_tokens``)."""
    build_canvas_step(**_commit_step())  # the rowless commit is valid, or this proves nothing
    rows = {k: v for k, v in _valid_step().items() if k in present}
    with pytest.raises(ValueError):
        AstraCanvasStep(**_commit_step(**rows))


@pytest.mark.parametrize("committed", ["absent", "empty"])
def test_a_commit_frame_requires_committed_tokens(committed: str) -> None:
    """R10: a ``commit`` frame carries ``committed_tokens``. A commit is ``num_sampled > 0``, so an
    empty tuple is as wrong as a missing field."""
    frame = _commit_step()
    if committed == "absent":
        del frame["committed_tokens"]
    else:
        frame["committed_tokens"] = ()
    with pytest.raises(ValueError):
        AstraCanvasStep(**frame)


def test_accepted_and_entropy_are_optional_and_default_to_none() -> None:
    """Criterion 6: v0.29.0 exposes neither, so the contract must be able to say so."""
    m = build_canvas_step(**_valid_step())
    assert m.accepted is None
    assert m.entropy is None
    as_json = json.loads(encode(m))
    assert "accepted" not in as_json and "entropy" not in as_json, (
        "a None field must not go on the wire as null; the HUD reads absence as 'not available'"
    )


def test_canvas_step_round_trips_through_the_json_codec() -> None:
    """The Conductor relays these to the HUD over the WebSocket codec (T-0011b)."""
    m = build_canvas_step(**_valid_step())
    assert decode(encode(m)) == m


@pytest.mark.parametrize(
    "bad",
    [
        {"canvas": (1, 2)},  # length ≠ canvas_length
        {"argmax": (1, 2)},
        {"stable": ""},  # mask shorter than canvas_length bits
        {"stable": "abc"},  # not whole bytes
        {"stable": "zz"},  # not hex
        {"step": 49},  # a denoise step > max_steps (R15: == max_steps is the cap's frame)
        {"phase": "settled"},  # not a phase of this sampler
        {"accepted": "not-hex"},
    ],
)
def test_canvas_step_rejects_an_inconsistent_frame(bad: dict[str, Any]) -> None:
    """Criterion 6/7: the validator is what stops a wrong frame reaching the HUD."""
    build_canvas_step(**_valid_step())  # the base must be valid, or this test proves nothing
    with pytest.raises(ValueError):
        AstraCanvasStep(**_valid_step(**bad))


def test_a_256_position_frame_is_kilobytes_not_megabytes() -> None:
    """Criterion 8, oversize handling: the shipped canvas is 256 (services/vllm.sh).

    ``canvas_length`` is engine-global, so the frame size is bounded at boot rather than per
    request: two id arrays plus a 64-char mask. The codec's ceiling is 8 MiB; if a frame ever
    approached it the relay would answer ``Malformed`` and the HUD would show nothing.
    """
    m = build_canvas_step(
        **_valid_step(
            canvas_length=256,
            canvas=tuple(range(200_000, 200_256)),
            argmax=tuple(range(100_000, 100_256)),
            stable="f" * 64,
        )
    )
    payload = msgpack.packb(m.model_dump(exclude_none=True), use_bin_type=True)
    assert len(payload) < 8192, f"a 256-position frame packed to {len(payload)} bytes"
    assert len(payload) < MAX_MESSAGE_BYTES


# --------------------------------------------------------------------------- the framing


def test_one_step_is_one_pub_message_of_two_frames(backend: str) -> None:
    """Criterion 8: topic frame then msgpack payload; the payload validates as the contract."""
    assert backend == "fake"
    pub, zmq = _publisher()
    assert attr(type(pub), "TOPIC", "8") == b"astra.canvas.step"
    msg = _valid_step()
    pub.send(msg)
    sockets = [s for c in zmq.contexts for s in c.sockets]
    assert len(sockets) == 1, "one PUB socket per publisher"
    sock = sockets[0]
    assert sock.kind == FakeZmq.PUB
    assert sock.bound == [LOOPBACK_ENDPOINT]
    assert len(sock.sent) == 1, "one PUB message per step"
    parts, flags = sock.sent[0]
    assert len(parts) == 2, f"expected a two-frame message, got {len(parts)}"
    assert parts[0] == b"astra.canvas.step", "frame 0 is the SUB topic, byte for byte"
    payload = msgpack.unpackb(parts[1], raw=False)
    assert payload["type"] == "astra.canvas.step"
    validate_canvas_step(payload)
    assert flags & FakeZmq.NOBLOCK, "the sampler thread must never block on send"


def test_the_socket_is_bounded_and_lingers_for_nothing(backend: str) -> None:
    """Criterion 8: bounded SNDHWM and LINGER 0 — decode never waits on a subscriber."""
    pub, zmq = _publisher(hwm=32)
    pub.send(_valid_step())
    sock = next(s for c in zmq.contexts for s in c.sockets)
    hwm = sock.option(FakeZmq.SNDHWM)
    assert hwm == 32, f"SNDHWM is {hwm!r}; it must be bounded and configurable"
    assert sock.option(FakeZmq.LINGER) == 0, "LINGER must be 0 or shutdown blocks on a dead SUB"


def test_a_slow_subscriber_drops_frames_and_never_stalls_decode(backend: str) -> None:
    """Criterion 8/5c: at the high-water mark the frame is dropped and counted, never retried."""
    pub, zmq = _publisher()
    pub.send(_valid_step())
    sock = next(s for c in zmq.contexts for s in c.sockets)
    sock.hwm_full = True
    sent_before = len(sock.sent)
    result = pub.send(_valid_step(step=4))
    assert result is False, "send() reports the drop rather than raising into the sampler"
    assert len(sock.sent) == sent_before
    assert pub.dropped == 1
    assert pub.published == 1


def test_publisher_refuses_a_non_loopback_endpoint() -> None:
    """CLAUDE.md: every socket in this project binds to loopback; a test asserts it."""
    tapmod = load_plugin_module("tap", "8")
    cls = attr(tapmod, "StepPublisher", "8")
    assert attr(cls, "TOPIC", "8") == b"astra.canvas.step"
    for endpoint in (
        "tcp://0.0.0.0:8002",
        "tcp://192.168.1.10:8002",
        "tcp://*:8002",
        "tcp://[::]:8002",
        "udp://127.0.0.1:8002",
    ):
        with pytest.raises(ValueError):
            cls(endpoint)
    cls(LOOPBACK_ENDPOINT)  # no socket is opened until the first send
    cls("ipc:///tmp/astra-tap-contract-test")


def test_the_endpoint_comes_from_astra_tap_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Criterion 8: services/vllm.sh exports it and versions.lock pins the same value."""
    tapmod = load_plugin_module("tap", "8")
    cls = attr(tapmod, "StepPublisher", "8")
    assert attr(cls, "TOPIC", "8") == b"astra.canvas.step"
    monkeypatch.setenv("ASTRA_TAP_ENDPOINT", "ipc:///tmp/astra-tap-from-env")
    assert cls.from_env().endpoint == "ipc:///tmp/astra-tap-from-env"
    monkeypatch.delenv("ASTRA_TAP_ENDPOINT")
    assert cls.from_env().endpoint == LOOPBACK_ENDPOINT, "the default is the shipped endpoint"

    vllm_sh = (REPO / "services" / "vllm.sh").read_text()
    assert f"ASTRA_TAP_ENDPOINT={LOOPBACK_ENDPOINT}" in vllm_sh
    lock = (REPO / "services" / "versions.lock").read_text()
    assert LOOPBACK_ENDPOINT in lock, "versions.lock [plugin] tap_endpoint must match"


@pytest.mark.skipif(not HAVE_PYZMQ, reason="pyzmq is not a host dependency of local-astra")
def test_fake_zmq_constants_match_the_installed_pyzmq() -> None:
    """`.claude/rules/tests.md`: a fake's behaviour is checked against the real thing."""
    import zmq

    assert (FakeZmq.PUB, FakeZmq.SUB) == (zmq.PUB, zmq.SUB)
    assert FakeZmq.SNDHWM == zmq.SNDHWM
    assert FakeZmq.RCVHWM == zmq.RCVHWM
    assert FakeZmq.LINGER == zmq.LINGER
    assert FakeZmq.NOBLOCK == zmq.NOBLOCK
    assert FakeZmq.SUBSCRIBE == zmq.SUBSCRIBE


@pytest.mark.skipif(not HAVE_PYZMQ, reason="pyzmq is not a host dependency of local-astra")
def test_pub_sub_round_trip_over_ipc(tmp_path: Path) -> None:
    """Criterion 8 end to end on a real socket: two frames arrive, and validate."""
    import zmq

    tapmod = load_plugin_module("tap", "8")
    cls = attr(tapmod, "StepPublisher", "8")
    endpoint = f"ipc://{tmp_path / 'tap.sock'}"
    pub = cls(endpoint)
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"astra.canvas.step")
    sub.connect(endpoint)
    try:
        deadline = 2.0
        got: list[bytes] | None = None
        while deadline > 0 and got is None:
            pub.send(_valid_step())
            if sub.poll(timeout=100):
                got = sub.recv_multipart()
            deadline -= 0.1
        assert got is not None, "no frame reached the subscriber within 2 s"
        assert len(got) == 2
        assert got[0] == b"astra.canvas.step"
        validate_canvas_step(msgpack.unpackb(got[1], raw=False))
    finally:
        sub.close(linger=0)
        pub.close()


# --------------------------------------------------------------------------- R14/R15: step range per phase


@pytest.mark.parametrize(
    ("step", "ok"), [(0, True), (47, True), (48, True), (49, False), (-1, False)]
)
def test_a_denoise_frames_step_range_includes_the_cap(step: int, ok: bool) -> None:
    """R15: a denoise frame carries the post-call step, and the call that converges at the step cap
    leaves ``step == max_steps`` (stock: ``step = cur_step + 1``, converged at ``>= max_steps``).
    So ``0 <= step <= max_steps`` (max_steps 48 here): 48 is accepted, 49 and -1 refused."""
    if ok:
        build_canvas_step(**_valid_step(step=step))
    else:
        with pytest.raises(ValueError):
            AstraCanvasStep(**_valid_step(step=step))


@pytest.mark.parametrize(
    ("step", "ok"), [(0, True), (1, True), (48, True), (49, False), (-1, False)]
)
def test_a_commit_frames_step_range_is_zero_to_the_cap(step: int, ok: bool) -> None:
    """R14 as amended by T-0069 ruling O6: a commit frame carries the pre-reset step, and a canvas closed
    by the cap commits at ``max_steps``; a commit may also carry ``step == 0`` (a slot CD has ended loudly
    is forced to its EOS canvas before any denoise step of that canvas). So ``0 <= step <= max_steps``
    (48 here): 0 and 48 accepted, -1 and 49 refused. (Was ``1..max_steps`` with ``0`` refused, T-0011.)"""
    if ok:
        build_canvas_step(**_commit_step(step=step))
    else:
        with pytest.raises(ValueError):
            AstraCanvasStep(**_commit_step(step=step))


# --------------------------------------------------------------------------- T-0159 A1: the pieces
#
# docs/tasks/T-0159.md A1: two optional fields, ``argmax_pieces`` (denoise only, canvas_length
# strings) and ``committed_pieces`` (commit only, len(committed_tokens) strings). Every refusal
# row first builds the valid frame it differs from (so it fails while the fields do not exist:
# BASE refuses *every* frame with an extra key, which would make a bare refusal row vacuous).

#: base denoise frame: CL 4, argmax (1, 2, 3, 9); base commit: committed (7, 8). Pieces by hand.
ARGMAX_PIECES = ("<|channel>", "\n", "</script><b>", " the")
COMMITTED_PIECES = (" Hello", "<channel|>")


def _with_pieces(frame: dict[str, Any]) -> AstraCanvasStep:
    """The valid frame, or an assertion naming T-0159 A1 while the contract lacks the fields."""
    missing = [
        f for f in ("argmax_pieces", "committed_pieces") if f not in AstraCanvasStep.model_fields
    ]
    assert not missing, f"T-0159 A1: AstraCanvasStep has no field(s) {missing} yet"
    try:
        return AstraCanvasStep(**frame)
    except ValueError as exc:
        raise AssertionError(f"T-0159 A1: a valid frame with pieces was refused: {exc}") from None


def test_a_denoise_frame_may_carry_argmax_pieces_of_canvas_length() -> None:
    """A1: accepted, kept as given (markers, a newline and markup are text), default ``None``."""
    m = _with_pieces(_valid_step(argmax_pieces=ARGMAX_PIECES))
    assert tuple(m.argmax_pieces) == ARGMAX_PIECES
    assert m.committed_pieces is None
    assert AstraCanvasStep(**_valid_step()).argmax_pieces is None, "optional, default None"
    assert m.text_preview is None, "A1: text_preview is unchanged (nobody fills it)"


def test_a_commit_frame_may_carry_committed_pieces_of_its_id_count() -> None:
    """A1: two committed ids (CL 4) -> two pieces, not four."""
    c = _with_pieces(_commit_step(committed_pieces=COMMITTED_PIECES))
    assert tuple(c.committed_pieces) == COMMITTED_PIECES
    assert c.argmax_pieces is None
    assert AstraCanvasStep(**_commit_step()).committed_pieces is None


def test_explicit_null_pieces_mean_no_pieces_in_both_phases() -> None:
    """A1: ``None`` is "no pieces" (as ``text_preview``), on either phase, for either field."""
    _with_pieces(_valid_step(argmax_pieces=None, committed_pieces=None))
    _with_pieces(_commit_step(argmax_pieces=None, committed_pieces=None))


def test_pieces_round_trip_through_the_json_codec_and_absent_pieces_stay_off_the_wire() -> None:
    """A1 / docs/02 §5: the relay sends these to the HUD over the codec; ``None`` is not sent."""
    m = _with_pieces(_valid_step(argmax_pieces=ARGMAX_PIECES))
    assert decode(encode(m)) == m
    wire = json.loads(encode(m))
    assert wire["argmax_pieces"] == list(ARGMAX_PIECES)
    assert "committed_pieces" not in wire, "a None field is absent on the wire"
    c = _with_pieces(_commit_step(committed_pieces=COMMITTED_PIECES))
    assert decode(encode(c)) == c and "argmax_pieces" not in json.loads(encode(c))


@pytest.mark.parametrize("n", [0, 3, 5], ids=["empty", "one-short", "one-long"])
def test_argmax_pieces_must_have_canvas_length_entries(n: int) -> None:
    """A1: ``len(argmax_pieces) == canvas_length`` (4 here), both sides of the bound."""
    _with_pieces(_valid_step(argmax_pieces=ARGMAX_PIECES))
    with pytest.raises(ValueError):
        AstraCanvasStep(**_valid_step(argmax_pieces=("x",) * n))


@pytest.mark.parametrize("n", [1, 3, 4], ids=["one-short", "one-long", "canvas-length"])
def test_committed_pieces_must_have_one_entry_per_committed_id(n: int) -> None:
    """A1: ``len(committed_pieces) == len(committed_tokens)`` (2), not ``canvas_length`` (4)."""
    _with_pieces(_commit_step(committed_pieces=COMMITTED_PIECES))
    with pytest.raises(ValueError):
        AstraCanvasStep(**_commit_step(committed_pieces=("x",) * n))


@pytest.mark.parametrize("n", [0, 2, 4], ids=["empty", "two", "canvas-length"])
def test_a_denoise_frame_never_carries_committed_pieces(n: int) -> None:
    """A1: never, whatever its length (4 = canvas_length is the tempting one; an empty tuple is
    still a carried value -- ``None`` is the absence)."""
    _with_pieces(_valid_step(argmax_pieces=ARGMAX_PIECES))
    with pytest.raises(ValueError):
        AstraCanvasStep(**_valid_step(argmax_pieces=ARGMAX_PIECES, committed_pieces=("x",) * n))
    with pytest.raises(ValueError):
        AstraCanvasStep(**_valid_step(committed_pieces=("x",) * n))


@pytest.mark.parametrize("n", [0, 2, 4], ids=["empty", "id-count", "canvas-length"])
def test_a_commit_frame_never_carries_argmax_pieces(n: int) -> None:
    """A1: never, whether its length is the id count (2) or canvas_length (4) -- and an empty tuple
    (n = 0) is carried too, so it is refused (reviewer 1 T1: a truthiness test lets ``()`` in)."""
    _with_pieces(_commit_step(committed_pieces=COMMITTED_PIECES))
    with pytest.raises(ValueError):
        AstraCanvasStep(**_commit_step(committed_pieces=COMMITTED_PIECES, argmax_pieces=("x",) * n))


@pytest.mark.parametrize(
    "item", [5, None, 1.5, b"x", ["x"]], ids=["int", "null", "float", "bytes", "list"]
)
def test_a_piece_is_a_string(item: Any) -> None:
    """A1: ``tuple[str, ...]``; a non-string item is refused in either field (no coercion)."""
    _with_pieces(_valid_step(argmax_pieces=ARGMAX_PIECES))
    with pytest.raises(ValueError):
        AstraCanvasStep(**_valid_step(argmax_pieces=(*ARGMAX_PIECES[:3], item)))
    with pytest.raises(ValueError):
        AstraCanvasStep(**_commit_step(committed_pieces=(COMMITTED_PIECES[0], item)))


def test_a_frame_with_pieces_still_obeys_every_base_rule() -> None:
    """A1: the pieces add rules, they waive none: a denoise frame with valid pieces but no
    ``stable``, or past the step cap, and a commit with valid pieces but rows, are refused."""
    _with_pieces(_valid_step(argmax_pieces=ARGMAX_PIECES))
    no_stable = {k: v for k, v in _valid_step(argmax_pieces=ARGMAX_PIECES).items() if k != "stable"}
    for bad in (no_stable, _valid_step(argmax_pieces=ARGMAX_PIECES, step=49)):
        with pytest.raises(ValueError):
            AstraCanvasStep(**bad)
    with pytest.raises(ValueError):
        AstraCanvasStep(**_commit_step(committed_pieces=COMMITTED_PIECES, canvas=(1, 2, 3, 4)))
