"""Resolution and sampling policy for perception (docs/03 §3 "Perception policies").

Pixel-first (docs/12 §3): screen frames use the largest resolution bucket and a dense document is
handled by a *focus crop*, not by DOM text. How many visual tokens an image actually costs is set
per request by the profile's ``visual_tokens`` budget (T-0131): unset, nothing is sent and the
engine uses its default of 280 per image (``ENGINE_DEFAULT_SOFT_TOKENS``); 560 and 1120 are the
other supported budgets. ``expected_soft_tokens`` mirrors the processor's resize rule. The bucket
estimates below are placeholders until Phase 0 Q2 measures them; ``measured=False`` says so, and B2
re-runs when a bucket changes.
"""

from __future__ import annotations

import math

from local_astra.contracts.perception import Frame, ResolutionBucket, Source

#: The engine's per-image soft-token budget when a request names none (``processor_config.json``
#: ``max_soft_tokens: 280`` for this checkpoint; T-0131 A4).
ENGINE_DEFAULT_SOFT_TOKENS = 280

#: The budgets ``Profile.visual_tokens`` may name (T-0131 A1); the engine also supports 70 and 140.
SUPPORTED_SOFT_TOKENS: tuple[int, ...] = (280, 560, 1120)

#: Gemma 4's resize unit: patch 16 px x pooling 3 (T-0131 A4).
_SOFT_TOKEN_UNIT = 16 * 3

#: The screen bucket's target per-image budget (docs/12 §3; rules/perception-memory.md). The engine
#: reaches it only when the request sets ``visual_tokens = 1120``; unset, it serves 280 per image.
MAX_SCREEN_VISUAL_TOKENS = 1120

#: Largest first. Screen frames always use the first (maximum-budget) bucket — pixel-first means
#: we spend tokens on pixels; the smaller buckets exist for the B2 sweep and the `lean-tokens`
#: option, never as the default. ``target_size`` never upscales, so a small window stays small.
SCREEN_BUCKETS: tuple[ResolutionBucket, ...] = (
    ResolutionBucket(
        name="screen_max", max_long_side=1568, max_short_side=1176, est_visual_tokens=1120
    ),
    ResolutionBucket(
        name="screen_md", max_long_side=1280, max_short_side=960, est_visual_tokens=840
    ),
    ResolutionBucket(
        name="screen_sm", max_long_side=1024, max_short_side=768, est_visual_tokens=560
    ),
)
CAMERA_BUCKET = ResolutionBucket(
    name="camera", max_long_side=896, max_short_side=896, est_visual_tokens=560
)
#: Focus crop (active window / pointer region) sent as a second image for dense documents.
CROP_BUCKET = ResolutionBucket(
    name="focus_crop", max_long_side=1024, max_short_side=768, est_visual_tokens=560
)

#: A crop is a *second* image at 560 visual tokens, so it must buy pixel density to be worth
#: sending: it has to reach at least this many times the density the full frame already gets after
#: ``target_size(..., SCREEN_BUCKETS[0])``. A policy constant (docs/12 §3), not a measurement.
MIN_CROP_MAGNIFICATION = 1.5

JPEG_QUALITY = 85
MAX_FPS = 1.0
RING_BUFFER_S = 60


def bucket_for(
    source: Source, width: int, height: int, *, crop: bool = False, lean: bool = False
) -> ResolutionBucket:
    """Bucket for a frame. ``lean`` picks the smallest screen bucket that fits (B2 comparison)."""
    if crop:
        return CROP_BUCKET
    if source == Source.CAMERA:
        return CAMERA_BUCKET
    if not lean:
        return SCREEN_BUCKETS[0]
    long_side = max(width, height)
    for b in reversed(SCREEN_BUCKETS):
        if long_side <= b.max_long_side:
            return b
    return SCREEN_BUCKETS[0]


def target_size(width: int, height: int, bucket: ResolutionBucket) -> tuple[int, int]:
    """Downscale (never upscale) to fit the bucket, preserving aspect ratio; even dimensions."""
    if width < 1 or height < 1:
        raise ValueError("dimensions must be positive")
    long_side, short_side = max(width, height), min(width, height)
    scale = min(1.0, bucket.max_long_side / long_side, bucket.max_short_side / short_side)
    w = max(2, int(width * scale) // 2 * 2)
    h = max(2, int(height * scale) // 2 * 2)
    return w, h


def expected_soft_tokens(width: int, height: int, budget: int) -> int:
    """Soft tokens the Gemma 4 processor spends on a ``width`` x ``height`` image at ``budget`` (T-0131 A4).

    Mirrors the processor's resize rule: the unit is patch 16 x pooling 3 = 48 px; the image is scaled
    (up or down) by ``sqrt(budget * 9 / patches)``, with ``patches = (W/16) * (H/16)``; each side is
    floored to a multiple of 48; the count is the resized patches / 9, at most ``budget``.
    1280x800 gives 273 / 522 / 1092 at 280 / 560 / 1120, and so does a 512x320 crop.
    Unlike vLLM, it does not clamp each side to at least one unit, so at aspect ratios of 280:1 or more it
    returns 0 where the engine returns up to ``budget`` (T-0131 review A, note 2).
    """
    if width < 1 or height < 1:
        raise ValueError("dimensions must be positive")
    patches = (width / 16) * (height / 16)
    scale = math.sqrt(budget * 9 / patches)
    resized_w = math.floor(width * scale / _SOFT_TOKEN_UNIT) * _SOFT_TOKEN_UNIT
    resized_h = math.floor(height * scale / _SOFT_TOKEN_UNIT) * _SOFT_TOKEN_UNIT
    return min((resized_w // 16) * (resized_h // 16) // 9, budget)


def scene_changed(phash_a: str, phash_b: str, *, threshold_bits: int = 10) -> bool:
    """Perceptual-hash Hamming distance with a fixed threshold (hysteresis lives in the ring buffer)."""
    if len(phash_a) != len(phash_b):
        raise ValueError("hash lengths differ")
    a, b = int(phash_a, 16), int(phash_b, 16)
    return bin(a ^ b).count("1") >= threshold_bits


def _crop_aspect(w: int, h: int) -> tuple[int, int]:
    """Grow the deficient side until the region has ``CROP_BUCKET``'s aspect, keeping orientation.

    Growing (never shrinking) means the crop still covers the whole region it was asked for.
    """
    long_n, short_n = CROP_BUCKET.max_long_side, CROP_BUCKET.max_short_side
    if w >= h:  # landscape: w : h == long : short
        if w * short_n < h * long_n:
            return max(1, round(h * long_n / short_n)), h
        return w, max(1, round(w * short_n / long_n))
    if h * short_n < w * long_n:  # portrait: h : w == long : short
        return w, max(1, round(w * long_n / short_n))
    return max(1, round(h * short_n / long_n)), h


def _fit_in_frame(w: int, h: int, frame_w: int, frame_h: int) -> tuple[int, int]:
    """Shrink (keeping aspect) only if the region is larger than the frame."""
    if w <= frame_w and h <= frame_h:
        return w, h
    scale = min(frame_w / w, frame_h / h)
    return max(1, int(w * scale)), max(1, int(h * scale))


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def crop_magnification(frame_w: int, frame_h: int, rect: tuple[int, int, int, int]) -> float:
    """Output pixels per input pixel a crop buys, relative to the full frame's own density.

    The full frame is served at ``target_size(frame, SCREEN_BUCKETS[0])`` and the crop at
    ``target_size(rect, CROP_BUCKET)``; the ratio of the two scales is what the second image is
    worth. ``target_size`` never upscales, so a crop of an already-unscaled screen reads 1.0.
    """
    full_w, _ = target_size(frame_w, frame_h, SCREEN_BUCKETS[0])
    _, _, w, h = rect
    crop_w, _ = target_size(w, h, CROP_BUCKET)
    return (crop_w / w) / (full_w / frame_w)


def focus_crop_rect(
    frame_w: int,
    frame_h: int,
    *,
    window: tuple[int, int, int, int] | None = None,
    pointer: tuple[int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    """The focus crop for a frame as ``(x, y, w, h)``, or ``None`` when a crop is not worth sending.

    Pure geometry (docs/12 §3, the pixel-only mitigation for dense documents): the active window
    rect wins over the pointer, the region takes ``CROP_BUCKET``'s aspect, is shrunk only if it is
    larger than the frame and is then shifted inwards until it lies inside the frame. The pointer
    region is ``CROP_BUCKET`` at native size centred on the pointer.

    ``None`` means "send the full frame alone": either no hint was given, or the crop would reach
    less than ``MIN_CROP_MAGNIFICATION`` times the density the full frame already gets — a
    duplicate of the whole screen at 560 visual tokens is a pure cost.
    """
    if frame_w < 1 or frame_h < 1:
        raise ValueError("dimensions must be positive")
    if window is not None:
        x, y, w, h = window
        if w < 1 or h < 1:
            raise ValueError("window must have positive dimensions")
        cx, cy = x + w / 2, y + h / 2
    elif pointer is not None:
        w, h = CROP_BUCKET.max_long_side, CROP_BUCKET.max_short_side
        cx, cy = float(pointer[0]), float(pointer[1])
    else:
        return None

    w, h = _fit_in_frame(*_crop_aspect(w, h), frame_w, frame_h)
    rect = (
        _clamp(round(cx - w / 2), 0, frame_w - w),
        _clamp(round(cy - h / 2), 0, frame_h - h),
        w,
        h,
    )
    if crop_magnification(frame_w, frame_h, rect) < MIN_CROP_MAGNIFICATION:
        return None
    return rect


def images_for_turn(full: Frame, crop: Frame | None) -> tuple[Frame, ...]:
    """The whole screen, plus its focus crop as a second image — never the crop alone.

    A crop whose ``crop_of`` is not this frame's ``phash`` is a wiring bug in whoever sent it;
    raising says so instead of silently sending the model a region of some other screen.
    """
    if crop is None:
        return (full,)
    if crop.crop_of != full.phash:
        raise ValueError(f"crop_of {crop.crop_of!r} is not the full frame {full.phash!r}")
    return (full, crop)
