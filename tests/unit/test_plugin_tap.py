"""T-0011 criteria 6 and 7: what the tap publishes, and the honesty of what it leaves out.

Rewrite of the scaffold's file. The scaffold tested ``entropy_from_logits`` / ``entropy_torch``
and an ``accepted`` bitmask; v0.29.0 exposes **neither** an acceptance mask nor a per-position
entropy (docs/tasks/T-0011.md, verified on the box 2026-09-13: both are locals inside the
``@torch.compile``d ``_compiled_sample_step``, never written to ``DiffusionGemmaRequestStates``).
Those tests are deleted rather than kept green against unreachable code, and replaced by:

* criterion 7 — ``stable``, derived from ``accepted_canvas_history[ST, CL]``, against a
  hand-computed fixture (``tests/fixtures/canvas/stable_history.json``);
* criterion 6 — the honesty tests: ``accepted`` and ``entropy`` are ``None`` on every frame, and
  the tempting proxy (``canvas == argmax_canvas``, an acceptance *lower bound*) is never
  published in their place.

The wire builder under test is ``astra_vllm_plugin.tap.encode_step``:

    encode_step(*, request_id, canvas_idx, step, max_steps, canvas, argmax, stable, phase,
                committed_tokens=(), segment=None) -> dict

with ``stable`` a sequence of booleans that it renders as the MSB-first hex mask, and
``canvas_length`` derived from ``len(canvas)``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_astra.contracts.protocol import AstraCanvasStep
from tests.fakes.diffusion_sampler import (
    ManualClock,
    TapHarness,
    attr,
    call_new_api,
    load_plugin_module,
    validate_canvas_step,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "canvas"


def _tap_module() -> Any:
    return load_plugin_module("tap", "6")


def _cases() -> list[dict[str, Any]]:
    return json.loads((FIXTURES / "stable_history.json").read_text())["cases"]


# --------------------------------------------------------------------------- criterion 7


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_stable_mask_matches_a_hand_built_history(case: dict[str, Any]) -> None:
    """Criterion 7: bit p is set iff every row of the history agrees with row 0 at p.

    The expected hex in the fixture is computed by hand (see
    ``tests/fixtures/canvas/MANIFEST.md``), not by re-running the function under test.
    """
    tapmod = _tap_module()
    bits = attr(tapmod, "stable_from_history", "7")(case["history"], case["history_len"])
    assert list(bits) == case["expected_bits"], (
        f"{case['name']}: stable bits {list(bits)} != hand-computed {case['expected_bits']}"
    )
    assert attr(tapmod, "stable_mask_hex", "7")(bits) == case["expected_hex"]


def test_stable_mask_is_zero_until_the_history_is_full() -> None:
    """Criterion 7: ``history_len >= ST`` is part of the definition, not an optimisation.

    The history buffer is a ring that starts zero-filled, so a short history looks perfectly
    'stable' to a naive comparison; upstream guards it with the length and so must we
    (``_compiled_sample_step`` l.606-612).
    """
    tapmod = _tap_module()
    history = [[7, 7, 7, 7]] * 4  # every row identical: a naive reduce says "all stable"
    for length in range(4):
        bits = attr(tapmod, "stable_from_history", "7")(history, length)
        assert list(bits) == [False] * 4, f"history_len={length} of 4 must be all-unstable"
    assert list(attr(tapmod, "stable_from_history", "7")(history, 4)) == [True] * 4


_SETTLED_LEFT = [[3, 4, 5, 6, 7, 8, 9, 10], [3, 4, 5, 6, 0, 0, 0, 0]] * 2  # equal at 0-3 only


@pytest.mark.parametrize(
    ("history", "history_len", "want"),
    [
        ([[5] * 8] * 4, 3, "00"),  # ST - 1: identical rows, but the history is not full yet
        ([[5] * 8] * 4, 4, "ff"),  # ST: full and identical
        ([[5] * 8] * 4, 5, "ff"),  # > ST (a ring that wrapped): still full
        (_SETTLED_LEFT, 3, "00"),
        (_SETTLED_LEFT, 4, "f0"),  # 1111 0000: positions 0-3 agree in every row
    ],
    ids=["identical-ST-1", "identical-ST", "identical-ST+1", "mixed-ST-1", "mixed-ST"],
)
def test_stable_through_the_publish_path_needs_a_full_history(
    history: list[list[int]], history_len: int, want: str
) -> None:
    """Criterion 7 through the real publish path (device reduction, D2H bundle, consumer decode),
    not the host helper: CL 8, ST 4, one publishing call over the given ``accepted_canvas_history``
    and ``accepted_canvas_history_len``. Hand-computed masks. A tap that tests ``len >= ST - 1``, or
    drops the length term, publishes "ff" at ST - 1 over identical rows."""
    h = TapHarness(hz=1000.0, canvas_length=8, history_steps=4, criterion="7")
    h.bind(req_id="cmpl-st")
    h.states.set_slot(0, history=history, history_len=history_len)
    h.step(step=5)
    h.tap.drain()
    assert len(h.publisher.sent) == 1
    assert h.publisher.sent[0]["stable"] == want, (
        f"history_len {history_len} of ST 4: stable {h.publisher.sent[0]['stable']!r}, want {want!r}"
    )


@pytest.mark.parametrize(
    ("bits", "hexstr"),
    [
        ([], ""),
        ([True], "80"),
        ([False] * 8, "00"),
        ([True] * 8, "ff"),
        ([True, False, False, False, False, False, False, False, True], "8080"),  # 9 bits → 2 bytes
        ([False, True, False, True, False, True, False, True], "55"),
    ],
)
def test_mask_is_msb_first_and_padded_to_whole_bytes(bits: list[bool], hexstr: str) -> None:
    """Criterion 7: MSB first, zero-padded to whole bytes (kept from the scaffold)."""
    assert attr(_tap_module(), "stable_mask_hex", "7")(bits) == hexstr


def test_mask_of_256_positions_is_64_hex_chars() -> None:
    """Criterion 7: the shipped canvas length is 256 (services/vllm.sh --diffusion-config)."""
    assert len(attr(_tap_module(), "stable_mask_hex", "7")([False] * 256)) == 64


# --------------------------------------------------------------------------- criterion 6


def _encode(**overrides: Any) -> dict[str, Any]:
    tapmod = _tap_module()
    kwargs: dict[str, Any] = {
        "request_id": "cmpl-7",
        "canvas_idx": 2,
        "step": 5,
        "max_steps": 48,
        "canvas": [11, 12, 13, 14, 15, 16, 17, 18],
        "argmax": [11, 12, 13, 14, 15, 16, 17, 99],
        "stable": [True, True, False, False, True, False, False, False],
        "phase": "denoise",
    }
    kwargs.update(overrides)
    return call_new_api(attr(tapmod, "encode_step", "6"), "6", **kwargs)


def test_encoded_step_validates_as_astra_canvas_step() -> None:
    """Criterion 6/8: the dict the publisher sends is exactly what the Conductor validates."""
    msg = _encode()
    parsed = validate_canvas_step(msg)
    assert isinstance(parsed, AstraCanvasStep)
    assert parsed.request_id == "cmpl-7"
    assert parsed.canvas_length == 8
    assert tuple(parsed.canvas) == (11, 12, 13, 14, 15, 16, 17, 18)
    assert tuple(parsed.argmax) == (11, 12, 13, 14, 15, 16, 17, 99)
    # 1100 1000 → 0xc8, hand-computed from the `stable` list above
    assert parsed.stable == "c8"
    assert parsed.phase == "denoise"
    assert parsed.text_preview is None, "the sampler process has ids, not text (T-0011b joins them)"


def test_commit_frames_carry_the_committed_ids() -> None:
    """Criterion 6 + R10: ``phase`` and ``committed_tokens`` are real, from ``SamplerOutput``; a
    commit frame carries none of canvas/argmax/stable, so its length comes from ``canvas_length``
    (R10 replaced this row's pre-amendment shape, which sent the rows with the commit)."""
    msg = _encode(
        phase="commit",
        committed_tokens=[5, 9, 13],
        canvas=None,
        argmax=None,
        stable=None,
        canvas_length=8,
    )
    parsed = validate_canvas_step(msg)
    assert parsed.phase == "commit"
    assert tuple(parsed.committed_tokens) == (5, 9, 13)
    assert parsed.canvas_length == 8
    assert parsed.canvas is None and parsed.argmax is None and parsed.stable is None


def test_a_commit_with_rows_is_never_put_on_the_wire() -> None:
    """R10, negative: one wire shape per phase -- ``encode_step`` refuses a commit that carries the
    rows (``ValueError``), or produces a dict the contract refuses. Either way it is not sent."""
    try:
        msg = _encode(phase="commit", committed_tokens=[5, 9, 13])
    except ValueError:
        return
    with pytest.raises(AssertionError, match="does not validate"):
        validate_canvas_step(msg)


def test_accepted_and_entropy_are_none_on_the_plugin_path() -> None:
    """Criterion 6, the honesty test: what v0.29.0 does not expose is not invented.

    The entropy-bound acceptance mask and the per-position entropy are locals inside the
    compiled step; recovering them costs a second ``log_softmax`` over ``[canvas_length,
    262144]``. CLAUDE.md: "never a fake trajectory presented as real".
    """
    msg = _encode()
    assert msg.get("accepted") is None, f"accepted was filled with {msg.get('accepted')!r}"
    assert msg.get("entropy") is None, f"entropy was filled with {msg.get('entropy')!r}"
    parsed = validate_canvas_step(msg)
    assert parsed.accepted is None
    assert parsed.entropy is None


def test_every_frame_the_tap_emits_has_accepted_and_entropy_none() -> None:
    """Criterion 6: not just the encoder — every frame that leaves the publisher."""
    h = TapHarness(hz=1000.0, queue_size=64, criterion="6")
    h.bind(req_id="cmpl-9")
    for i in range(6):
        h.step(step=i, commit=(i == 5))
        h.clock.advance(0.01)
    h.tap.drain()
    assert len(h.publisher.sent) == 6
    for msg in h.publisher.sent:
        assert msg.get("accepted") is None and msg.get("entropy") is None
        parsed = validate_canvas_step(msg)
        assert parsed.accepted is None and parsed.entropy is None


def test_canvas_equal_to_argmax_is_never_published_as_accepted_or_stable() -> None:
    """Criterion 6, negative: the tempting proxy is an acceptance *lower bound*, not the mask.

    Here every position of the canvas equals the argmax, while the history says only positions
    0-3 have settled. An implementation that reached for ``canvas == argmax_canvas`` would emit
    an all-ones mask; the hand-computed answer is 0xf0.
    """
    h = TapHarness(hz=1000.0, canvas_length=8, history_steps=4, criterion="6")
    h.bind(req_id="cmpl-proxy")
    canvas = [3, 4, 5, 6, 7, 8, 9, 10]
    history = [
        [3, 4, 5, 6, 7, 8, 9, 10],
        [3, 4, 5, 6, 0, 8, 9, 10],
        [3, 4, 5, 6, 7, 0, 9, 10],
        [3, 4, 5, 6, 7, 8, 0, 0],
    ]
    h.states.set_slot(0, canvas=canvas, argmax=list(canvas), history=history, history_len=4)
    h.step(step=1)
    h.tap.drain()
    assert len(h.publisher.sent) == 1
    msg = h.publisher.sent[0]
    assert msg.get("accepted") is None, "canvas == argmax must not be published as `accepted`"
    assert msg["stable"] == "f0", (
        f"stable was {msg['stable']!r}; 'ff' means the canvas/argmax proxy was used instead of "
        "accepted_canvas_history"
    )
    assert tuple(msg["canvas"]) == tuple(canvas)


def test_the_tap_publishes_the_post_renoise_canvas_and_the_argmax_separately() -> None:
    """Criterion 6: ``canvas`` is the flicker (docs/12 §2), ``argmax`` the settled reading."""
    h = TapHarness(hz=1000.0, canvas_length=4, history_steps=2, criterion="6")
    h.bind(req_id="cmpl-two")
    h.states.set_slot(
        0,
        canvas=[1, 2, 3, 4],
        argmax=[1, 2, 3, 9],
        history=[[1, 2, 3, 4], [1, 2, 3, 4]],
        history_len=2,
    )
    h.step(step=0)
    h.tap.drain()
    msg = h.publisher.sent[0]
    assert tuple(msg["canvas"]) == (1, 2, 3, 4)
    assert tuple(msg["argmax"]) == (1, 2, 3, 9)
    assert msg["canvas"] != msg["argmax"], "the two fields must not be the same array"


def test_max_steps_comes_from_the_states_not_a_hardcoded_48() -> None:
    """Implementation notes: ``max_steps`` is ``diffusion_states.max_denoising_steps``.

    The scaffold hardcoded 48; ``--diffusion-config`` is engine-global and may say anything.
    """
    h = TapHarness(hz=1000.0, max_denoising_steps=12, criterion="6")
    h.bind(req_id="cmpl-ms")
    h.step(step=11)
    h.tap.drain()
    msg = h.publisher.sent[0]
    assert msg["max_steps"] == 12
    assert msg["step"] == 11
    validate_canvas_step(msg)  # step < max_steps


# --------------------------------------------------------------------------- rate limiter


def test_rate_limiter_caps_at_hz() -> None:
    """Criterion 4: the pure limiter, on the injected monotonic clock (never a wall clock).

    The limiter reads the same clock the tap does — ``RateLimiter(hz, clock)`` with
    ``allow(key, *, force=False)`` — so that a caller cannot hand it a wall-clock reading the
    rest of the tap never saw. The scaffold's ``allow(key, now)`` is that hazard.
    """
    clock = ManualClock(100.0)
    rl = call_new_api(attr(_tap_module(), "RateLimiter", "4"), "4", hz=30.0, clock=clock)
    assert rl.allow("k") is True
    clock.advance(0.010)
    assert rl.allow("k") is False, "10 ms later is inside the 33.3 ms window"
    clock.advance(0.024)  # 34 ms after the first pass
    assert rl.allow("k") is True
    assert rl.allow("other") is True, "the budget is per key, i.e. per request"


def test_rate_limiter_forget_gives_the_key_a_fresh_budget() -> None:
    """Criterion 4: the budget is per request; a released request's key is forgotten.

    Rewritten with R5: the old row forced the commit step through the limiter
    (``allow(key, force=True)``), which R5 rules impossible without a host sync -- a commit is now
    published *lagged* (``test_plugin_sampler.py``). What survives is the per-key budget.
    """
    clock = ManualClock(100.0)
    rl = call_new_api(attr(_tap_module(), "RateLimiter", "4"), "4", hz=30.0, clock=clock)
    assert rl.allow("k") is True
    clock.advance(0.001)
    assert rl.allow("k") is False
    rl.forget("k")
    assert rl.allow("k") is True, "a forgotten key starts with a fresh budget"
    clock.advance(0.001)
    assert rl.allow("k") is False, "forgetting must not turn the limiter off"


# --------------------------------------------------------------------------- encoder guards


def test_encode_step_rejects_a_length_mismatch() -> None:
    """A malformed frame must die in the plugin, not as a Malformed on the HUD socket."""
    with pytest.raises(ValueError):
        _encode(canvas=[1, 2], argmax=[1, 2, 3], stable=[True, False])
    with pytest.raises(ValueError):
        _encode(canvas=[1, 2], argmax=[1, 2], stable=[True, False, True])


def test_encode_step_rejects_an_unknown_phase() -> None:
    """``phase`` is ``denoise`` | ``commit``; nothing else is a real phase of this sampler."""
    with pytest.raises(ValueError):
        _encode(phase="settled")
