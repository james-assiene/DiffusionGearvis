"""``BrowserFrameSource`` — the Python half of the HUD perception path (T-0008, criteria 14-18).

Derived from ``docs/tasks/T-0008.md`` and the contracts (``contracts/protocols.PerceptionClient``,
``contracts/perception.Frame``, ``perception/policy.py``), never from an implementation: when
these were written ``local_astra/perception/browser_source.py`` was the scaffold stub that raises
``NotImplementedError``.

Two things the task leaves to the implementer are pinned here so the tests can call them; both
are flagged in the test-author's report:

* **the injected receive clock** (criterion 14, 18) is either a zero-argument callable passed to
  ``BrowserFrameSource(...)`` under one of ``CLOCK_KW``, or a millisecond argument to ``ingest``
  under one of ``INGEST_TIME_KW``. ``harness()`` finds whichever exists and fails with a named
  criterion if neither does. It is never ``time.monotonic`` inside the module (criterion 18).
* **the rejection counter** (criterion 14, "rejects — counted, no exception escaping") is an int
  attribute named ``rejected`` (``rejected_frames`` / ``n_rejected`` also accepted).

``ingest`` may be sync or async; ``harness().ingest`` awaits it if it is awaitable.
"""

from __future__ import annotations

import ast
import base64
import inspect
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.events import SceneChanged
from local_astra.contracts.perception import Frame, Source
from local_astra.contracts.protocol import AstraFrame
from local_astra.contracts.protocols import PerceptionClient
from local_astra.perception import browser_source as bs_module
from local_astra.perception import policy
from local_astra.perception.browser_source import BrowserFrameSource

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "hud"
JPEG: bytes = (FIXTURES / "solid_16x16_q85.jpg").read_bytes()
JPEG_B64: str = base64.b64encode(JPEG).decode("ascii")
GRADIENT: bytes = (FIXTURES / "gradient_32x24_q85.jpg").read_bytes()
GRADIENT_B64: str = base64.b64encode(GRADIENT).decode("ascii")

CLOCK_KW = ("now_ms", "clock", "recv_clock", "receive_clock", "now")
INGEST_TIME_KW = ("recv_ms", "received_ms", "receive_ms", "now_ms", "t_ms", "at_ms")
COUNTER_NAMES = ("rejected", "rejected_frames", "n_rejected")

#: 16 lowercase hex characters — the dHash the page sends (criterion 5) and the only shape
#: ``policy.scene_changed`` may ever see (criterion 14).
PHASH_LEN = 16

_UNSET = object()


# --- harness -------------------------------------------------------------------------------------


@dataclass
class Harness:
    """Drives ``BrowserFrameSource`` with an injected receive clock (criterion 14)."""

    src: Any
    ingest_time_kw: str | None
    now: list[int] = field(default_factory=lambda: [0])

    async def ingest(self, msg: AstraFrame, recv_ms: int) -> Frame | None:
        self.now[0] = recv_ms
        out = (
            self.src.ingest(msg, **{self.ingest_time_kw: recv_ms})
            if self.ingest_time_kw
            else self.src.ingest(msg)
        )
        if inspect.isawaitable(out):
            out = await out
        return out

    async def drain(self, timeout_s: float = 0.2) -> list[object]:
        """Every event available now. Bounded so a never-ending ``events()`` fails, not hangs."""
        import asyncio

        it = self.src.events()
        if inspect.isawaitable(it):
            it = await it
        out: list[object] = []
        agen = it.__aiter__()
        while True:
            try:
                out.append(await asyncio.wait_for(agen.__anext__(), timeout_s))
            except (StopAsyncIteration, TimeoutError):
                return out


def harness(**kwargs: Any) -> Harness:
    now = [0]
    params = inspect.signature(BrowserFrameSource.__init__).parameters
    clock_kw = next((n for n in CLOCK_KW if n in params), None)
    ingest_params = inspect.signature(BrowserFrameSource.ingest).parameters
    ingest_kw = next((n for n in INGEST_TIME_KW if n in ingest_params), None)
    assert clock_kw or ingest_kw, (
        "criterion 14/18: Frame.ts_ms is stamped from an *injected* receive clock and the module "
        f"reads no clock of its own. Expected either BrowserFrameSource(<{'|'.join(CLOCK_KW)}>=...) "
        f"taking a zero-argument callable returning ms, or ingest(msg, <{'|'.join(INGEST_TIME_KW)}>=ms). "
        f"__init__ has {sorted(params)}, ingest has {sorted(ingest_params)}."
    )
    if clock_kw:
        kwargs[clock_kw] = lambda: now[0]
    src = BrowserFrameSource(**kwargs)
    h = Harness(src=src, ingest_time_kw=ingest_kw)
    h.now = now
    return h


def frame_msg(
    *,
    ts_ms: int,
    source: Source = Source.SCREEN,
    idx: int = 0,
    phash: Any = _UNSET,
    jpeg_b64: str = JPEG_B64,
    width: int = 16,
    height: int = 16,
) -> AstraFrame:
    """An ``astra.frame`` as the page sends it. ``idx`` labels the frame inside its phash."""
    return AstraFrame(
        source=source,
        ts_ms=ts_ms,
        width=width,
        height=height,
        jpeg_b64=jpeg_b64,
        phash=f"{idx:016x}" if phash is _UNSET else phash,
    )


def rejected_count(src: Any) -> int:
    for name in COUNTER_NAMES:
        value = getattr(src, name, None)
        if isinstance(value, int):
            return value
    raise AssertionError(
        "criterion 14: a rejected frame is *counted* (no exception escapes), so the count must be "
        f"readable; expected an int attribute named {COUNTER_NAMES[0]!r} "
        f"(or {COUNTER_NAMES[1:]}), found none on {type(src).__name__}"
    )


async def ingest_series(
    h: Harness, recv_times: list[int], *, source: Source = Source.SCREEN, first_idx: int = 0
) -> list[Frame | None]:
    return [
        await h.ingest(frame_msg(ts_ms=1_700_000_000_000 + t, source=source, idx=first_idx + i), t)
        for i, t in enumerate(recv_times)
    ]


def idx_of(frame: Frame | None) -> int | None:
    """The label ``frame_msg`` wrote into the phash, so assertions can name a frame by hand."""
    return None if frame is None else int(frame.phash, 16)


# --- criterion 14: conformance and ingest validation ----------------------------------------------


def test_browser_frame_source_satisfies_the_perception_client_protocol():
    """Criterion 14, the ``tests/contract/test_fakes_conform.py`` pattern."""
    assert isinstance(harness().src, PerceptionClient), (
        "T-0008 criterion 14: BrowserFrameSource must satisfy the runtime_checkable "
        "PerceptionClient protocol (get_frame, clip, events)"
    )


async def test_ingest_base64_decodes_the_jpeg_into_the_frame_unchanged():
    """Criterion 14: ``jpeg_b64`` → ``Frame.jpeg``, a real JPEG (ffd8 … ffd9)."""
    h = harness()
    out = await h.ingest(frame_msg(ts_ms=1_700_000_000_000, jpeg_b64=GRADIENT_B64), 4_200)
    assert isinstance(out, Frame), "criterion 14: ingest(AstraFrame) -> Frame | None"
    assert out.jpeg == GRADIENT, "the exact bytes the page encoded must survive the round trip"
    assert out.jpeg[:2].hex() == "ffd8" and out.jpeg[-2:].hex() == "ffd9"
    assert (out.source, out.width, out.height) == (Source.SCREEN, 16, 16)


async def test_frame_ts_ms_is_the_injected_receive_clock_not_the_client_clock():
    """Criterion 14: the page's ``Date.now()`` is not the audio clock (implementation notes)."""
    h = harness()
    client_ts = 1_999_999_999_999
    out = await h.ingest(frame_msg(ts_ms=client_ts), 4_200)
    assert out is not None
    assert out.ts_ms == 4_200, (
        "criterion 14: Frame.ts_ms comes from the injected receive clock; the client's ts_ms is "
        f"kept only as a jitter figure (got {out.ts_ms}, client sent {client_ts})"
    )


@pytest.mark.parametrize(
    ("phash", "why"),
    [
        ("0" * 8, "too short"),
        ("0" * 17, "too long"),
        ("", "empty"),
        (None, "absent — Frame.phash is required (min_length=1)"),
        ("z" * PHASH_LEN, "right length, not hex — int(x, 16) raises too"),
    ],
)
async def test_a_bad_phash_is_rejected_and_never_reaches_policy_scene_changed(
    monkeypatch, phash, why
):
    """Criterion 14: ``policy.scene_changed`` raises on differing lengths — unreachable from input."""
    assert isinstance(policy.scene_changed("0" * 16, "0" * 16), bool)
    with pytest.raises(ValueError):  # the failure mode this criterion exists to prevent
        policy.scene_changed("0" * 8, "0" * 16)

    calls: list[tuple[str, str]] = []
    real = policy.scene_changed

    def spy(a: str, b: str, **kw: Any) -> bool:
        calls.append((a, b))
        return real(a, b, **kw)

    monkeypatch.setattr(policy, "scene_changed", spy)
    monkeypatch.setattr(bs_module, "scene_changed", spy, raising=False)

    h = harness()
    good = await h.ingest(frame_msg(ts_ms=1_700_000_000_000, idx=1), 1_000)
    assert good is not None
    before = rejected_count(h.src)
    out = await h.ingest(frame_msg(ts_ms=1_700_000_001_000, phash=phash), 2_000)

    assert out is None, f"criterion 14: a frame whose phash is {why} must be rejected, not stored"
    assert rejected_count(h.src) == before + 1, "criterion 14: rejections are counted"
    assert all(len(a) == len(b) for a, b in calls), (
        f"criterion 14: policy.scene_changed was called with mismatched hash lengths {calls!r}; "
        "it raises ValueError on that and client input must never reach it unchecked"
    )
    latest = await h.src.get_frame(Source.SCREEN)
    assert idx_of(latest) == 1, "the rejected frame must not enter the ring"


@pytest.mark.parametrize(
    ("jpeg_b64", "why"),
    [
        ("!!!!", "not base64 at all"),
        ("/9j/\n/9j/", "a newline inside the payload — b64decode ignores it without validate=True"),
        ("/9j/ /9j/", "a space inside the payload, likewise ignored without validate=True"),
        ("====", "leading padding"),
        ("abc", "incorrect padding"),
    ],
)
async def test_a_jpeg_that_is_not_valid_base64_is_rejected_and_counted(jpeg_b64, why):
    """Criterion 14 (ruling R12): decode with ``validate=True``, or junk enters the ring.

    ``base64.b64decode`` without ``validate=True`` silently *discards* every character outside the
    alphabet, so a truncated or corrupted payload becomes a shorter "valid" blob that is stored,
    sent to the model as an image and only ever fails somewhere far away from here.
    """
    h = harness()
    await h.ingest(frame_msg(ts_ms=1, idx=3), 1_000)
    before = rejected_count(h.src)
    out = await h.ingest(frame_msg(ts_ms=2, idx=9, jpeg_b64=jpeg_b64), 2_000)
    assert out is None, f"criterion 14: jpeg_b64 with {why} must be rejected, not stored"
    assert rejected_count(h.src) == before + 1, "criterion 14: rejections are counted"
    assert idx_of(await h.src.get_frame(Source.SCREEN)) == 3, "the ring is untouched"


async def test_a_rejected_frame_raises_nothing_and_leaves_the_source_usable():
    """Criterion 14: "no exception escaping" — the next good frame still ingests."""
    h = harness()
    await h.ingest(frame_msg(ts_ms=1, idx=7), 1_000)
    assert await h.ingest(frame_msg(ts_ms=2, phash="nope"), 2_000) is None
    out = await h.ingest(frame_msg(ts_ms=3, idx=8), 3_000)
    assert idx_of(out) == 8
    assert idx_of(await h.src.get_frame(Source.SCREEN)) == 8


# --- criterion 15: ring and onset selection -------------------------------------------------------


@pytest.mark.parametrize(
    ("at_ms", "expected_idx"),
    [
        (999, None),  # before every frame
        (1_000, 0),  # exactly on the first onset -> "at or before"
        (1_001, 0),
        (2_999, 1),
        (3_000, 2),
        (3_500, 2),  # THE case: a later frame exists (4000, 5000) and must not be returned
        (4_999, 3),
        (5_000, 4),
        (9_999, 4),
        (None, 4),  # at_ms=None -> latest
    ],
)
async def test_get_frame_returns_the_frame_at_or_just_before_the_onset(at_ms, expected_idx):
    """Criterion 15, hand-built 5-frame ring; expected index written by hand.

    Ring (receive clock ms -> frame index): 1000->0, 2000->1, 3000->2, 4000->3, 5000->4.
    docs/03 §3: "the frame at speech onset, not the latest at end-of-turn".
    """
    h = harness()
    await ingest_series(h, [1_000, 2_000, 3_000, 4_000, 5_000])
    got = (
        await h.src.get_frame(Source.SCREEN, at_ms)
        if at_ms is not None
        else await h.src.get_frame(Source.SCREEN)
    )
    assert idx_of(got) == expected_idx, (
        f"criterion 15: get_frame(at_ms={at_ms}) must return the frame at or just before it "
        f"(index {expected_idx}), never a later one; got index {idx_of(got)}"
    )


async def test_get_frame_on_an_empty_ring_is_none():
    """Criterion 15."""
    h = harness()
    assert await h.src.get_frame(Source.SCREEN) is None
    assert await h.src.get_frame(Source.SCREEN, 1_000) is None


async def test_rings_are_independent_across_screen_window_and_camera():
    """Criterion 15: one 60 s ring *per source*."""
    h = harness()
    await ingest_series(h, [1_000, 3_000], source=Source.SCREEN, first_idx=0)
    await ingest_series(h, [2_000], source=Source.WINDOW, first_idx=10)
    await ingest_series(h, [4_000], source=Source.CAMERA, first_idx=20)
    assert idx_of(await h.src.get_frame(Source.SCREEN, 4_500)) == 1
    assert idx_of(await h.src.get_frame(Source.WINDOW, 4_500)) == 10
    assert idx_of(await h.src.get_frame(Source.CAMERA, 4_500)) == 20
    assert await h.src.get_frame(Source.CAMERA, 3_999) is None


async def test_the_ring_drops_frames_older_than_the_policy_window():
    """Criterion 15: one 60 s ring (``policy.RING_BUFFER_S``)."""
    ring_ms = policy.RING_BUFFER_S * 1_000
    assert ring_ms == 60_000
    h = harness()
    await h.ingest(frame_msg(ts_ms=1, idx=0), 1_000)
    await h.ingest(frame_msg(ts_ms=2, idx=1), 2_000)
    await h.ingest(frame_msg(ts_ms=3, idx=2), 1_000 + ring_ms + 1_000)  # 62 s later
    assert idx_of(await h.src.get_frame(Source.SCREEN, 1_500)) is None, (
        "criterion 15: a frame older than RING_BUFFER_S is evicted"
    )
    kept = await h.src.clip(Source.SCREEN, 0, 200_000, fps=1000.0)
    assert [idx_of(f) for f in kept] == [2]


async def test_a_frame_exactly_the_window_old_is_outside_the_window():
    """Ruling R8: the boundary is closed — at exactly ``RING_BUFFER_S`` the frame is gone.

    The window holds the *last* 60 s, so a frame whose age is exactly 60 000 ms is no longer in
    it. Written as its own case because the count-cap test used to hide this boundary inside a
    "1 fps keeps 61" claim, which is what the review caught.
    """
    ring_ms = policy.RING_BUFFER_S * 1_000
    h = harness()
    await h.ingest(frame_msg(ts_ms=1, idx=0), 1_000)
    await h.ingest(frame_msg(ts_ms=2, idx=1), 1_000 + ring_ms - 1)  # 59.999 s later: still in
    kept = await h.src.clip(Source.SCREEN, 0, 10_000_000, fps=1000.0)
    assert [idx_of(f) for f in kept] == [0, 1]
    await h.ingest(frame_msg(ts_ms=3, idx=2), 1_000 + ring_ms)  # exactly 60 s later: out
    kept = await h.src.clip(Source.SCREEN, 0, 10_000_000, fps=1000.0)
    assert [idx_of(f) for f in kept] == [1, 2], (
        "ruling R8: a frame exactly RING_BUFFER_S old is evicted, not kept"
    )


async def test_a_producer_that_stopped_an_hour_ago_yields_no_frame():
    """Ruling R9: the ring is pruned on *read* against the injected clock, not only on ingest.

    The failure the task's Blast-radius section names: the page stops posting (tab closed, mute,
    socket dropped) and ``get_frame`` keeps answering with an hour-old screenshot, which degrades
    every answer while nothing errors and no log line appears.
    """
    h = harness()
    await h.ingest(frame_msg(ts_ms=1, idx=0), 1_000)
    assert idx_of(await h.src.get_frame(Source.SCREEN)) == 0
    h.now[0] = 1_000 + 3_600_000  # an hour of silence: no further ingest to trigger eviction
    assert await h.src.get_frame(Source.SCREEN) is None, (
        "ruling R9: get_frame() evicts against the receive clock before selecting, so a producer "
        "that stopped an hour ago yields nothing rather than an hour-old frame"
    )
    assert await h.src.get_frame(Source.SCREEN, at_ms=h.now[0]) is None
    assert await h.src.get_frame(Source.SCREEN, at_ms=1_500) is None


async def test_the_ring_is_capped_by_frame_count_when_a_client_ignores_the_fps_rule():
    """Criterion 15: "additionally capped by frame count so a client cannot grow it without bound"."""
    h = harness()
    n = 3_000
    for i in range(n):  # 3000 frames in 3 s: 3000x the 1 fps rule, all inside the 60 s window
        await h.ingest(frame_msg(ts_ms=i, idx=i), 1_000 + i)
    kept = await h.src.clip(Source.SCREEN, 0, 1_000_000, fps=1000.0)
    assert len(kept) < n, (
        f"criterion 15: the ring must be capped by frame count; it kept all {n} frames a "
        "misbehaving client posted inside the 60 s window"
    )
    assert len(kept) <= 1_000, f"the cap is O(RING_BUFFER_S * MAX_FPS), not {len(kept)}"
    assert idx_of(kept[-1]) == n - 1, "eviction drops the *oldest* frames, never the newest"

    # Ruling R8: the count cap must never be the binding constraint for a 1 fps client. Sixty
    # frames one second apart (receive 1000 … 60 000) are all *inside* the 60 s window at the last
    # ingest (cutoff 0), so anything missing here was dropped for being too numerous, not too old.
    compliant = harness()
    for i in range(60):
        await compliant.ingest(frame_msg(ts_ms=i, idx=i), 1_000 + i * 1_000)
    kept_compliant = await compliant.src.clip(Source.SCREEN, 0, 1_000_000, fps=1.0)
    assert [idx_of(f) for f in kept_compliant] == list(range(60)), (
        "criterion 15 (ruling R8): the frame count cap must be >= 61 so it is never what limits a "
        f"client obeying policy.MAX_FPS; {60 - len(kept_compliant)} of 60 frames were lost"
    )


# --- criterion 15, adversarial: the browser clock steps backwards ---------------------------------


async def test_a_client_clock_that_jumps_backwards_changes_no_get_frame_answer():
    """Criterion 14/15 adversarial: ``ts_ms`` is the *browser's* clock and may step.

    Receive clock (the only ordering that counts):  1000 2000 3000 4000 5000 6000
    Client ``ts_ms`` (10 s backwards step at #3): 100k 101k 102k  93k  94k  95k
    A ring sorted or windowed by client time hands the model the wrong screenshot and nothing
    ever errors — which is why this is a test and not a comment.
    """
    h = harness()
    client = [100_000, 101_000, 102_000, 93_000, 94_000, 95_000]  # -9 s step at index 3
    recv = [1_000, 2_000, 3_000, 4_000, 5_000, 6_000]
    for i, (c, r) in enumerate(zip(client, recv, strict=True)):
        out = await h.ingest(frame_msg(ts_ms=c, idx=i), r)
        assert out is not None and out.ts_ms == r, "criterion 14: ts_ms is the receive clock"

    kept = await h.src.clip(Source.SCREEN, 0, 100_000, fps=1000.0)
    assert [idx_of(f) for f in kept] == [0, 1, 2, 3, 4, 5], (
        "criterion 15: the ring keeps receive order and evicts nothing when the client clock steps"
    )
    for at_ms, expected in [(4_500, 3), (3_999, 2), (6_000, 5), (999, None), (None, 5)]:
        got = (
            await h.src.get_frame(Source.SCREEN)
            if at_ms is None
            else await h.src.get_frame(Source.SCREEN, at_ms)
        )
        assert idx_of(got) == expected, (
            f"criterion 15: get_frame(at_ms={at_ms}) must answer by receive time (expected frame "
            f"{expected}), not by the client's ts_ms; got {idx_of(got)}"
        )


# --- criterion 16: clip ---------------------------------------------------------------------------


async def test_clip_returns_frames_inside_the_window_ascending_and_spaced():
    """Criterion 16: ascending, inside [t0, t1], gaps >= 1000/fps. Hand-computed expectation.

    Client posts at 2 Hz for 20 s (41 frames at 0, 500, ... 20000 on the receive clock);
    ``clip(..., fps=1.0)`` must thin that to the 21 frames 0, 1000, ... 20000.
    """
    h = harness()
    times = [i * 500 for i in range(41)]
    await ingest_series(h, times)
    got = await h.src.clip(Source.SCREEN, 0, 20_000, fps=1.0)
    ts = [f.ts_ms for f in got]
    assert ts == sorted(ts), "criterion 16: ascending by time"
    assert all(b - a >= 1_000 for a, b in pairwise(ts)), (
        f"criterion 16: spacing >= 1000/fps ms, got gaps {[b - a for a, b in pairwise(ts)]}"
    )
    assert ts[0] <= 1_000 and ts[-1] <= 20_000
    assert len(got) == 21, (
        f"criterion 16: 21 frames spaced 1 s across a 20 s window, got {len(got)}"
    )


async def test_clip_of_a_sixty_second_window_never_returns_more_than_61_frames():
    """Criterion 16 (docs/03 §4): <= 61 frames for a 60 s window, whatever the client sent."""
    h = harness()
    for i in range(121):  # 2 Hz for 60 s
        await h.ingest(frame_msg(ts_ms=i, idx=i), i * 500)
    got = await h.src.clip(Source.SCREEN, 0, 60_000, fps=1.0)
    assert len(got) <= 61, f"criterion 16: <= 61 frames for a 60 s window, got {len(got)}"
    ts = [f.ts_ms for f in got]
    assert all(b - a >= 1_000 for a, b in pairwise(ts))


async def test_clip_excludes_other_sources_and_anything_outside_the_window():
    """Criterion 16."""
    h = harness()
    await ingest_series(h, [1_000, 2_000, 3_000, 4_000, 5_000])
    await ingest_series(h, [2_500], source=Source.CAMERA, first_idx=99)
    got = await h.src.clip(Source.SCREEN, 2_000, 4_000, fps=1.0)
    assert [idx_of(f) for f in got] == [1, 2, 3]
    assert all(f.source == Source.SCREEN for f in got)
    assert await h.src.clip(Source.WINDOW, 0, 10_000, fps=1.0) == []


# --- criterion 17: scene events with hysteresis ---------------------------------------------------

PH_A = "0" * 16
PH_B = "ffff" + "0" * 12  # 16 bits apart from A: over policy.scene_changed's 10-bit threshold
PH_C = "0" * 8 + "f" * 8  # 32 bits from A


def test_the_scene_memory_is_bounded_and_never_outlives_the_ring_it_describes():
    """Criterion 17 (ruling R12): pin the hysteresis constants, they are policy not decoration.

    ``SCENE_MEMORY_MS`` is how long a phash counts as "recently seen" and therefore *not* a new
    scene. Longer than the ring itself and a user who comes back to an earlier window after a
    minute gets no ``SceneChanged`` at all, while nothing in the 60 s ring can corroborate the
    memory that suppressed it — 600 000 ms is precisely the value this test exists to reject.
    The lower end is pinned behaviourally by ``test_alternating_scenes_do_not_emit_an_event_per_frame``
    (A/B/A/B one second apart must still be one scene).
    """
    memory_ms = getattr(bs_module, "SCENE_MEMORY_MS", _UNSET)
    memory_keys = getattr(bs_module, "SCENE_MEMORY_KEYS", _UNSET)
    assert memory_ms is not _UNSET and memory_keys is not _UNSET, (
        "criterion 17 (R12): the hysteresis window and its key count are named module constants "
        "(SCENE_MEMORY_MS, SCENE_MEMORY_KEYS), not numbers buried in a method"
    )
    assert 1_000 <= memory_ms <= policy.RING_BUFFER_S * 1_000, (
        f"criterion 17 (R12): SCENE_MEMORY_MS is {memory_ms}; it must not outlive the "
        f"{policy.RING_BUFFER_S} s ring it describes"
    )
    assert 2 <= memory_keys <= 64, (
        f"criterion 17 (R12): SCENE_MEMORY_KEYS is {memory_keys}; hysteresis needs more than the "
        "last frame and less than an unbounded history"
    )


async def test_scene_changed_is_emitted_for_a_real_change():
    """Criterion 17: only ``SceneChanged``, carrying source and phash."""
    assert policy.scene_changed(PH_A, PH_C) and not policy.scene_changed(PH_A, PH_A)
    h = harness()
    await h.ingest(frame_msg(ts_ms=1, phash=PH_A), 1_000)
    await h.drain()
    await h.ingest(frame_msg(ts_ms=2, phash=PH_C), 2_000)
    events = await h.drain()
    assert len(events) == 1, f"criterion 17: one SceneChanged for one change, got {events!r}"
    assert isinstance(events[0], SceneChanged), "criterion 17: events() yields only SceneChanged"
    assert events[0].source == Source.SCREEN.value and events[0].phash == PH_C


async def test_alternating_scenes_do_not_emit_an_event_per_frame():
    """Criterion 17 adversarial: A/B/A/B/A/B is a flickering scene, not six scene changes."""
    assert policy.scene_changed(PH_A, PH_B), "premise: A and B differ by more than the threshold"
    h = harness()
    await h.ingest(frame_msg(ts_ms=0, phash=PH_A), 1_000)
    await h.drain()  # whatever the first frame does is not part of the alternation
    for i, ph in enumerate([PH_B, PH_A, PH_B, PH_A, PH_B]):
        await h.ingest(frame_msg(ts_ms=i + 1, phash=ph), 2_000 + i * 1_000)
    events = await h.drain()
    assert all(isinstance(e, SceneChanged) for e in events)
    assert len(events) <= 1, (
        f"criterion 17: hysteresis — an alternating A/B/A/B sequence must not emit an event per "
        f"frame; got {len(events)}"
    )


async def test_a_different_screen_after_a_long_silence_is_still_a_scene_change():
    """Criterion 17 (ruling R14): the comparison is against the last *accepted* frame, always.

    The user shares a tab, walks away for longer than ``SCENE_MEMORY_MS``, comes back on a
    completely different screen. The memory window exists to suppress A/B/A/B chatter, not to
    forget what is on screen: an empty window must not mean "no scene change", or the model is
    told nothing happened precisely when everything did.
    """
    memory_ms = getattr(bs_module, "SCENE_MEMORY_MS", 10_000)
    assert policy.scene_changed(PH_A, PH_C), "premise: A and C are a real change"
    h = harness()
    await h.ingest(frame_msg(ts_ms=0, phash=PH_A), 1_000)
    await h.drain()
    quiet_until = 1_000 + memory_ms + 5_000  # long enough that no memory of A can survive by age
    got = await h.ingest(frame_msg(ts_ms=1, phash=PH_C), quiet_until)
    assert got is not None, "premise: the frame itself is accepted"
    events = await h.drain()
    assert [type(e) for e in events] == [SceneChanged], (
        f"criterion 17 (R14): a genuinely different screen after {memory_ms + 5_000} ms of silence "
        f"is a scene change compared against the last accepted frame; got {events!r}"
    )
    assert events[0].phash == PH_C and events[0].source == Source.SCREEN.value


async def test_the_same_screen_after_a_long_silence_is_not_a_scene_change():
    """Criterion 17 (ruling R14), the other half: silence alone changes nothing.

    Guards the cheap fix — "emit whenever the memory window is empty" — which would turn every
    pause in the capture into a scene change and wake the Conductor for nothing.
    """
    memory_ms = getattr(bs_module, "SCENE_MEMORY_MS", 10_000)
    h = harness()
    await h.ingest(frame_msg(ts_ms=0, phash=PH_A), 1_000)
    await h.drain()
    await h.ingest(frame_msg(ts_ms=1, phash=PH_A), 1_000 + memory_ms + 5_000)
    assert await h.drain() == [], (
        "criterion 17 (R14): the same phash after a long silence is the same scene; only a real "
        "perceptual difference from the last accepted frame is a SceneChanged"
    )


# --- criterion 18: no I/O in the module -----------------------------------------------------------

FORBIDDEN_IMPORTS = {
    "time",
    "datetime",
    "socket",
    "os",
    "io",
    "pathlib",
    "httpx",
    "websockets",
    "urllib",
    "subprocess",
    "requests",
    "aiohttp",
    "sqlite3",
}
FORBIDDEN_CALLS = {"open", "print", "input"}


def test_browser_source_module_opens_no_socket_reads_no_clock_and_touches_no_file():
    """Criterion 18: time and identity arrive as arguments (the turn machine's discipline)."""
    path = Path(bs_module.__file__)
    tree = ast.parse(path.read_text())
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
    assert not (imported & FORBIDDEN_IMPORTS), (
        f"criterion 18: {path.name} imports {sorted(imported & FORBIDDEN_IMPORTS)}; it opens no "
        "socket, reads no clock and touches no file"
    )
    assert not (called & FORBIDDEN_CALLS), (
        f"criterion 18: {path.name} calls {sorted(called & FORBIDDEN_CALLS)}"
    )
