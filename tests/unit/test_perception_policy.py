"""Resolution buckets and scene change (docs/03 §3 "Perception policies"). Expected values by hand."""

from __future__ import annotations

import inspect
from fractions import Fraction
from typing import Any

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st

from local_astra.contracts.perception import Frame, Source
from local_astra.perception import policy as policy_module
from local_astra.perception.policy import (
    CAMERA_BUCKET,
    CROP_BUCKET,
    MAX_SCREEN_VISUAL_TOKENS,
    SCREEN_BUCKETS,
    bucket_for,
    scene_changed,
    target_size,
)


def test_screen_buckets_never_exceed_the_per_image_token_ceiling():
    assert all(b.est_visual_tokens <= MAX_SCREEN_VISUAL_TOKENS for b in SCREEN_BUCKETS)
    assert SCREEN_BUCKETS[0].est_visual_tokens == MAX_SCREEN_VISUAL_TOKENS  # pixel-first default
    assert all(not b.measured for b in SCREEN_BUCKETS), "flip to measured only with Q2 numbers"


@pytest.mark.parametrize(("w", "h"), [(3840, 2160), (1920, 1080), (1280, 800), (800, 600)])
def test_screen_frames_default_to_the_maximum_budget_bucket(w, h):
    """docs/12 §3: pixels first — the default spends the full visual-token budget on the screen."""
    assert bucket_for(Source.SCREEN, w, h) is SCREEN_BUCKETS[0]


@pytest.mark.parametrize(
    ("w", "h", "expected"),
    [
        (3840, 2160, "screen_max"),
        (1920, 1080, "screen_max"),
        (1280, 800, "screen_md"),  # fits md exactly
        (1024, 640, "screen_sm"),
        (800, 600, "screen_sm"),  # small window: smallest bucket, never upscaled
    ],
)
def test_lean_mode_picks_the_smallest_bucket_that_fits(w, h, expected):
    assert bucket_for(Source.SCREEN, w, h, lean=True).name == expected


def test_camera_and_crop_have_fixed_buckets():
    assert bucket_for(Source.CAMERA, 1920, 1080) is CAMERA_BUCKET
    assert bucket_for(Source.SCREEN, 1920, 1080, crop=True) is CROP_BUCKET


@pytest.mark.parametrize(
    ("w", "h", "bucket", "expected"),
    [
        (3840, 2160, SCREEN_BUCKETS[0], (1568, 882)),  # 1568/3840 = 0.40833 → 2160*0.40833 = 882
        (1920, 1080, SCREEN_BUCKETS[0], (1568, 882)),
        (
            1080,
            1920,
            SCREEN_BUCKETS[0],
            (882, 1568),
        ),  # portrait: long side binds (1568/1920 < 896/1080)
        (1280, 800, SCREEN_BUCKETS[1], (1280, 800)),  # exact fit, untouched
        (640, 480, SCREEN_BUCKETS[2], (640, 480)),  # smaller than bucket: never upscaled
        (1000, 1000, CAMERA_BUCKET, (896, 896)),
    ],
)
def test_target_size_hand_computed(w, h, bucket, expected):
    assert target_size(w, h, bucket) == expected


@given(w=st.integers(1, 8192), h=st.integers(1, 8192), i=st.integers(0, len(SCREEN_BUCKETS) - 1))
def test_target_size_properties(w, h, i):
    b = SCREEN_BUCKETS[i]
    tw, th = target_size(w, h, b)
    assert tw <= max(w, 2) and th <= max(h, 2), "never upscale"
    assert max(tw, th) <= max(b.max_long_side, 2) and min(tw, th) <= max(b.max_short_side, 2)
    assert tw % 2 == 0 and th % 2 == 0
    # aspect ratio preserved within the rounding of two even-ing steps (independent oracle: Fraction)
    if min(w, h) >= 64:
        want = Fraction(w, h)
        got = Fraction(tw, th)
        assert abs(float(got) - float(want)) <= float(want) * (2 / min(tw, th)) + 2 / min(tw, th)


def test_target_size_rejects_degenerate_dimensions():
    with pytest.raises(ValueError):
        target_size(0, 100, SCREEN_BUCKETS[0])


@pytest.mark.parametrize(
    ("a", "b", "changed"),
    [
        ("0" * 16, "0" * 16, False),  # identical
        ("0" * 16, "f" * 16, True),  # 64 bits differ
        ("0" * 16, "00000000000003ff", True),  # 10 bits differ → threshold reached
        ("0" * 16, "00000000000001ff", False),  # 9 bits differ → below threshold
    ],
)
def test_scene_changed_is_hamming_distance_at_threshold_10(a, b, changed):
    assert scene_changed(a, b) is changed


def test_scene_changed_rejects_mismatched_hash_lengths():
    with pytest.raises(ValueError):
        scene_changed("00", "0000")


# --- T-0012: the pure focus-crop policy (criteria 9, 10, 14) -------------------------------------
#
# docs/12 §3: "the whole screen plus a high-resolution crop of the active window or the region
# around the pointer, sent as two images". T-0012 adds the geometry and the magnification rule;
# capturing or transmitting crop pixels is explicitly out of scope. The symbols are fetched with
# ``policy_symbol`` so that a not-yet-written function reads as a named criterion, not as an
# ImportError that takes the whole (already passing) module down at collection time.

_MISSING = object()


def policy_symbol(name: str) -> Any:
    value = getattr(policy_module, name, _MISSING)
    assert value is not _MISSING, f"T-0012 adds perception.policy.{name}"
    return value


def crop_rect(frame_w: int, frame_h: int, **kwargs):
    return policy_symbol("focus_crop_rect")(frame_w, frame_h, **kwargs)


def magnification(frame_w: int, frame_h: int, rect) -> float:
    """Pixel density the crop buys, relative to the density the full frame already gets.

    Independent oracle for criterion 10, expressed only in terms of ``target_size`` and the two
    buckets: the full frame is served at ``target_size(frame, SCREEN_BUCKETS[0])`` and the crop at
    ``target_size(rect, CROP_BUCKET)``, so the ratio of output-pixels-per-input-pixel is what a
    second image is worth.
    """
    full_w, _ = target_size(frame_w, frame_h, SCREEN_BUCKETS[0])
    _, _, w, h = rect
    crop_w, _ = target_size(w, h, CROP_BUCKET)
    return (crop_w / w) / (full_w / frame_w)


def assert_inside(rect, frame_w: int, frame_h: int) -> None:
    x, y, w, h = rect
    assert w >= 1 and h >= 1, f"degenerate crop {rect}"
    assert x >= 0 and y >= 0, f"crop starts outside the frame: {rect}"
    assert x + w <= frame_w and y + h <= frame_h, (
        f"crop leaves the frame: {rect} in {frame_w}x{frame_h}"
    )


def assert_crop_aspect(rect) -> None:
    """The crop keeps ``CROP_BUCKET``'s aspect ratio (4:3), up to integer rounding."""
    _, _, w, h = rect
    long_side, short_side = max(w, h), min(w, h)
    num, den = CROP_BUCKET.max_long_side, CROP_BUCKET.max_short_side
    ratio = Fraction(num, den)
    err = abs(long_side * ratio.denominator - short_side * ratio.numerator)
    assert err <= 2 * (ratio.numerator + ratio.denominator), (
        f"crop {rect} has aspect {long_side}:{short_side}, bucket wants {ratio}"
    )


def test_focus_crop_prefers_the_window_rect_over_the_pointer():
    """Criterion 9: when both are known the active window wins; the pointer is the fallback.

    4K frame, a 1200x900 window in the bottom-right quadrant and the pointer in the opposite
    corner — a crop that followed the pointer would show the model the wrong half of the screen.
    """
    window = (2400, 1200, 1200, 900)
    pointer = (50, 50)
    both = crop_rect(3840, 2160, window=window, pointer=pointer)
    window_only = crop_rect(3840, 2160, window=window)
    pointer_only = crop_rect(3840, 2160, pointer=pointer)
    assert both is not None, "a 1200x900 window on a 4K screen magnifies 2.09x (see criterion 10)"
    assert both == window_only
    assert pointer_only is not None
    assert both != pointer_only

    x, y, w, h = both
    cx, cy = window[0] + window[2] // 2, window[1] + window[3] // 2
    assert x <= cx <= x + w and y <= cy <= y + h, "the crop covers the window it was asked for"
    assert_inside(both, 3840, 2160)
    assert_crop_aspect(both)

    px, py = pointer_only[0], pointer_only[1]
    assert px <= pointer[0] <= pointer_only[0] + pointer_only[2]
    assert py <= pointer[1] <= pointer_only[1] + pointer_only[3]


@pytest.mark.parametrize(
    ("kwargs", "expect"),
    [
        # Pointer 10 px from the top-left corner: no crop containing it can start above or left
        # of the frame, so the rect is pinned to the origin under any clamping rule.
        ({"pointer": (10, 10)}, {"x": 0, "y": 0}),
        # ... and symmetrically at the bottom-right corner it ends on the frame edge.
        ({"pointer": (3830, 2150)}, {"right": 3840, "bottom": 2160}),
        # A window hanging off the right and bottom edges: whether the implementation shifts the
        # rect inwards or shrinks it, the crop ends exactly on those two edges.
        ({"window": (3600, 1900, 900, 700)}, {"right": 3840, "bottom": 2160}),
    ],
    ids=["pointer-top-left", "pointer-bottom-right", "window-overhangs"],
)
def test_crop_rect_is_clamped_and_keeps_the_crop_bucket_aspect(kwargs, expect):
    """Criterion 9, hand-derived corner cases on a 3840x2160 frame."""
    rect = crop_rect(3840, 2160, **kwargs)
    assert rect is not None, f"a corner crop on a 4K frame magnifies; got None for {kwargs}"
    x, y, w, h = rect
    assert_inside(rect, 3840, 2160)
    assert_crop_aspect(rect)
    got = {"x": x, "y": y, "right": x + w, "bottom": y + h}
    assert {k: got[k] for k in expect} == expect


def test_a_region_taller_than_the_frame_is_shrunk_to_fit_it():
    """A4 (ruling), hand case: clamping shifts inward *and* shrinks when the region overflows.

    An ultrawide 3440x1474 frame and a tall 1122x1255 window. Growing that window to the crop
    bucket's 4:3 makes it 1122x1496 (1122 * 1024/768 = 1496), which is 22 px taller than the
    frame; no amount of shifting can put it inside, so the region has to shrink by 1474/1496 to
    1105x1474 (1122 * 1474/1496 = 1105.5). An implementation that only shifts returns a rect
    whose bottom edge is 1496 — 22 px of a screen that does not exist, cropped from a frame that
    ends at 1474.
    """
    frame_w, frame_h = 3440, 1474
    rect = crop_rect(frame_w, frame_h, window=(2221, 484, 1122, 1255))
    assert rect is not None, "1105x1474 of an ultrawide frame magnifies ~1.52x (criterion 10)"
    assert_inside(rect, frame_w, frame_h)
    assert_crop_aspect(rect)

    x, _y, w, h = rect
    assert h == frame_h, "the vertical overflow is what binds: the crop is exactly frame-tall"
    assert abs(w - 1122 * 1474 / 1496) <= 1, f"shrunk keeping aspect, want ~1105.5, got {w}"
    cx = 2221 + 1122 // 2
    assert x <= cx <= x + w, "and it still covers the window it was asked for"
    assert magnification(frame_w, frame_h, rect) >= policy_symbol("MIN_CROP_MAGNIFICATION")


@settings(deadline=None)
@seed(20260913)  # fixed seed: this task's regime says deterministic, no clock, no network
@given(
    frame_w=st.integers(640, 5120),
    ratio=st.sampled_from([Fraction(16, 9), Fraction(16, 10), Fraction(4, 3), Fraction(21, 9)]),
    px=st.floats(0.0, 1.0),
    py=st.floats(0.0, 1.0),
    wx=st.floats(0.0, 1.0),
    wy=st.floats(0.0, 1.0),
    ww=st.floats(0.02, 1.4),
    wh=st.floats(0.02, 1.4),
    use_window=st.booleans(),
)
def test_focus_crop_rect_is_always_inside_the_frame_and_worth_sending(
    frame_w, ratio, px, py, wx, wy, ww, wh, use_window
):
    """Criteria 9 + 10 as properties: inside the frame, non-degenerate, 4:3, and magnifying."""
    frame_h = max(64, int(frame_w / float(ratio)))
    pointer = (int(px * (frame_w - 1)), int(py * (frame_h - 1)))
    kwargs = {"pointer": pointer}
    if use_window:
        kwargs["window"] = (
            int(wx * frame_w),
            int(wy * frame_h),
            max(1, int(ww * frame_w)),
            max(1, int(wh * frame_h)),
        )
    rect = crop_rect(frame_w, frame_h, **kwargs)
    if rect is None:
        return
    assert_inside(rect, frame_w, frame_h)
    assert_crop_aspect(rect)
    mag = magnification(frame_w, frame_h, rect)
    floor = policy_symbol("MIN_CROP_MAGNIFICATION")
    assert mag >= floor - 1e-9, f"criterion 10: {rect} magnifies {mag:.3f}x on {frame_w}x{frame_h}"


def test_min_crop_magnification_is_the_policy_constant_criterion_10_names():
    assert policy_symbol("MIN_CROP_MAGNIFICATION") == 1.5


@pytest.mark.parametrize(
    ("frame", "kwargs", "why"),
    [
        # 1920x1080: the full frame is already served at 1568/1920 = 0.8167 px per px. A crop of
        # the whole screen lands on 1024/1440 = 0.711 (4:3 region) or 1024/1920 = 0.533 (the raw
        # region) — 0.87x or 0.65x, both below 1.5. A duplicate at 560 visual tokens is pure cost.
        ((1920, 1080), {"window": (0, 0, 1920, 1080)}, "whole screen as a window"),
        # 1024x768: target_size never upscales, so the full frame goes in unscaled (1.0 px per px)
        # and CROP_BUCKET is exactly 1024x768 — no region of that screen can beat 1.0x.
        ((1024, 768), {"window": (212, 168, 600, 450)}, "screen already sent unscaled"),
        ((1024, 768), {"pointer": (512, 384)}, "screen already sent unscaled, pointer"),
        # Criterion 10's second clause.
        ((3840, 2160), {}, "neither window nor pointer"),
    ],
    ids=["whole-screen-window", "unscaled-screen-window", "unscaled-screen-pointer", "no-hint"],
)
def test_no_crop_when_it_would_not_magnify(frame, kwargs, why):
    """Criterion 10, adversarial: the crops that cost tokens and buy nothing.

    The expectations are computed from ``target_size`` and the two buckets below, never copied
    from the implementation.
    """
    frame_w, frame_h = frame
    full_w, _ = target_size(frame_w, frame_h, SCREEN_BUCKETS[0])
    if kwargs:
        region = kwargs.get("window") or (0, 0, frame_w, frame_h)
        assert magnification(frame_w, frame_h, region) < 1.5, (
            f"oracle check: {why} magnifies {magnification(frame_w, frame_h, region):.3f}x"
        )
    assert full_w <= frame_w
    assert crop_rect(frame_w, frame_h, **kwargs) is None, f"criterion 10: {why}"


def _frame_for_crop(phash: str, *, w: int, h: int, crop_of: str | None = None) -> Frame:
    return Frame(
        source=Source.SCREEN,
        ts_ms=1000,
        width=w,
        height=h,
        jpeg=b"\xff\xd8\xff",
        phash=phash,
        crop_of=crop_of,
    )


def test_images_for_turn_never_returns_the_crop_alone_and_rejects_a_foreign_crop():
    """Criterion 9: "the whole screen *plus* a crop, as two images" — never the crop instead."""
    images_for_turn = policy_symbol("images_for_turn")
    full = _frame_for_crop("a" * 16, w=3840, h=2160)
    crop = _frame_for_crop("b" * 16, w=1200, h=900, crop_of="a" * 16)
    assert tuple(images_for_turn(full, None)) == (full,)
    assert tuple(images_for_turn(full, crop)) == (full, crop), "full frame first, crop second"

    # A6 (ruling): a foreign crop *raises*. Dropping it and sending the full frame alone would
    # hide the wiring bug in whichever task sent it — the pipeline would keep working, one image
    # short, and the crop that was supposed to make a dense document readable would silently
    # never arrive.
    for stranger, why in (
        (_frame_for_crop("c" * 16, w=1200, h=900, crop_of="d" * 16), "crop of another frame"),
        (_frame_for_crop("e" * 16, w=1200, h=900), "no crop_of: not identifiable as a crop"),
    ):
        with pytest.raises(ValueError):
            images_for_turn(full, stranger)
        assert tuple(images_for_turn(full, None)) == (full,), f"full frame still works ({why})"


def test_screen_default_bucket_is_unchanged_by_the_crop_work():
    """Criterion 14 regression guard: the crop is a *second* image, not a cheaper first one.

    Changing any of these requires re-running B2 (.claude/rules/perception-memory.md), which this
    task explicitly does not do.
    """
    assert MAX_SCREEN_VISUAL_TOKENS == 1120
    assert [
        (b.name, b.max_long_side, b.max_short_side, b.est_visual_tokens) for b in SCREEN_BUCKETS
    ] == [
        ("screen_max", 1568, 1176, 1120),
        ("screen_md", 1280, 960, 840),
        ("screen_sm", 1024, 768, 560),
    ]
    assert (
        CROP_BUCKET.max_long_side,
        CROP_BUCKET.max_short_side,
        CROP_BUCKET.est_visual_tokens,
    ) == (
        1024,
        768,
        560,
    )
    assert bucket_for(Source.SCREEN, 1920, 1080) is SCREEN_BUCKETS[0]
    assert target_size(1920, 1080, SCREEN_BUCKETS[0]) == (1568, 882)


# --- T-0027: bucket_for and the wire string its enum is made of ----------------------------------
#
# ``Source`` is a ``StrEnum``, so the value that arrives on the socket is ``"camera"`` and
# ``"camera" is Source.CAMERA`` is False. ``bucket_for`` compared with ``is``, so a raw camera
# frame was bucketed at 1120 visual tokens (the screen budget) instead of the camera bucket's 560.
# Ruling R4: the comparison becomes value-based; ``bucket_for`` does not become a validator.


def _agreement_cases():
    """(wire string, enum, kwargs, expected bucket) for every source x call-shape combination.

    Expected buckets by hand from the policy, not from the function: ``crop`` wins over
    everything (a crop is always CROP_BUCKET); a camera frame is always CAMERA_BUCKET, lean or
    not; a screen or window frame at 1920x1080 is SCREEN_BUCKETS[0] both by default (pixel-first)
    and under ``lean`` (1920 > every bucket's max_long_side, so the lean loop falls through).
    """
    sources = [
        ("camera", Source.CAMERA, CAMERA_BUCKET),
        ("screen", Source.SCREEN, SCREEN_BUCKETS[0]),
        ("window", Source.WINDOW, SCREEN_BUCKETS[0]),
    ]
    shapes: list[tuple[str, dict[str, Any]]] = [
        ("plain", {}),
        ("lean", {"lean": True}),
        ("crop", {"crop": True}),
    ]
    cases = []
    for wire, enum, bucket in sources:
        for shape, kwargs in shapes:
            expected = CROP_BUCKET if kwargs.get("crop") else bucket
            cases.append(pytest.param(wire, enum, kwargs, expected, id=f"{wire}-{shape}"))
    return cases


@pytest.mark.parametrize(("wire", "enum", "kwargs", "expected"), _agreement_cases())
def test_bucket_for_agrees_between_the_enum_and_its_wire_string(wire, enum, kwargs, expected):
    """Criterion 8: the same frame gets the same bucket whether the source arrives as the enum
    member or as the string that enum is made of (the value the HUD puts on the wire)."""
    assert str(enum) == wire, "premise: Source is a StrEnum whose value is the wire string"
    assert enum is not wire, "premise: ... and identity with that string is still False"
    from_enum = bucket_for(enum, 1920, 1080, **kwargs)
    from_wire = bucket_for(wire, 1920, 1080, **kwargs)
    assert from_enum is expected, f"the enum form already disagrees with the policy: {from_enum}"
    assert from_wire is expected, (
        f"criterion 8: bucket_for({wire!r}, 1920, 1080, **{kwargs}) gave {from_wire.name!r} "
        f"({from_wire.est_visual_tokens} visual tokens) where the policy says "
        f"{expected.name!r} ({expected.est_visual_tokens})"
    )
    assert from_wire is from_enum


@pytest.mark.parametrize("source", ["retina", "Camera", "CAMERA", "", "screen "])
def test_an_unrecognised_source_string_still_gets_the_screen_bucket(source):
    """Criterion 9 (ruling R4): the fix matches the enum's *values*; it does not normalise case,
    guess, or start validating. ``Source(source)`` would make criterion 8 green and give every
    caller on the perception ingress path a new ``ValueError`` for a one-branch bug — validation
    is ``contracts/perception.py``'s job, and ``bucket_for`` is a policy function."""
    try:
        got = bucket_for(source, 1920, 1080)
    except Exception as e:  # raising at all is the failure this test exists for
        raise AssertionError(
            f"ruling R4: bucket_for({source!r}, ...) must not raise; it is a policy function, "
            f"not a validator. Got {e!r}"
        ) from e
    assert got is SCREEN_BUCKETS[0], (
        f"criterion 9: an unrecognised source keeps today's outcome (the screen bucket); "
        f"got {got.name!r}"
    )


def test_bucket_for_still_declares_a_source_enum():
    """Criterion 10 (ruling R5): the runtime tolerance for wire-derived values is documented and
    tested, not advertised in the signature — internal callers pass the enum and ``mypy --strict``
    keeps saying so. ``policy.py`` has ``from __future__ import annotations``, so the annotation
    is the string ``"Source"``."""
    params = inspect.signature(bucket_for).parameters
    assert params["source"].annotation == "Source", (
        "criterion 10: bucket_for's source parameter stays annotated Source; widening it to "
        f"`Source | str` is out of scope, got {params['source'].annotation!r}"
    )
    assert [p.name for p in params.values()] == ["source", "width", "height", "crop", "lean"]
