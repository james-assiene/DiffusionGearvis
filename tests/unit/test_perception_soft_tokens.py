"""T-0131 A4: ``perception/policy.py`` reconciles the per-image budget with the engine (docs/tasks/T-0131.md).

The criterion's rule, which mirrors the Gemma 4 processor's resize: unit = patch 16 x pooling 3 = 48 px;
``patches`` = (W/16)·(H/16) of the input; ``scale = sqrt(budget·9 / patches)``; each scaled side is floored to a
multiple of 48; ``tokens = min(patches' / 9, budget)`` over the resized image's patches.

Every expected value below is worked by hand (the arithmetic is in each docstring) and none is computed by the
test. The sizes are chosen so that both sides are multiples of 16 (no reading of a fractional patch count is
needed) and no scaled side lands within 0.02 of a multiple of 48 (floating-point rounding cannot move a floor).

Every row is ``pending(task="T-0131")``: at BASE (f30365d) the names do not exist, so each fails on a named
``AssertionError``. ``MAX_SCREEN_VISUAL_TOKENS`` staying 1120 is also pinned, green, by ``test_perception_policy.py``;
here it rides in a row that needs the new names.

Regime: CPU, pure; no measurement.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

import pytest

from local_astra.perception import policy


def _need(name: str) -> Any:
    assert hasattr(policy, name), f"T-0131 A4: perception.policy.{name} does not exist yet"
    return getattr(policy, name)


def expected(width: int, height: int, budget: int) -> int:
    fn = _need("expected_soft_tokens")
    got = fn(width, height, budget)
    assert type(got) is int, f"T-0131 A4: expected_soft_tokens returns an int; got {got!r}"
    return int(got)


def test_constants() -> None:
    """A4: ``ENGINE_DEFAULT_SOFT_TOKENS = 280`` (``processor_config.json`` ``max_soft_tokens``),
    ``SUPPORTED_SOFT_TOKENS = (280, 560, 1120)``; ``MAX_SCREEN_VISUAL_TOKENS`` stays 1120 (the bucket's target)."""
    assert _need("ENGINE_DEFAULT_SOFT_TOKENS") == 280
    assert tuple(_need("SUPPORTED_SOFT_TOKENS")) == (280, 560, 1120)
    assert policy.MAX_SCREEN_VISUAL_TOKENS == 1120
    assert policy.SCREEN_BUCKETS[0].est_visual_tokens == 1120


@pytest.mark.parametrize(
    ("budget", "want"),
    [(280, 273), (560, 522), (1120, 1092)],
)
def test_expected_tokens_hand_values(budget: int, want: int) -> None:
    """A4: 1280x800 -> 273 / 522 / 1092, and the 512x320 crop gives the same (the processor scales it up).

    1280x800 is 80x50 = 4000 patches; 512x320 is 32x20 = 640 and scales by exactly 2.5x more, so both reach the
    same resized size:

    - 280: scale = sqrt(2520/4000) = 0.79373; 1015.97 -> 1008 (21·48), 634.98 -> 624 (13·48); 63·39 = 2457
      patches; 2457/9 = 273.
    - 560: scale = sqrt(5040/4000) = 1.12250; 1436.80 -> 1392 (29·48), 898.00 -> 864 (18·48); 87·54 = 4698;
      4698/9 = 522.
    - 1120: scale = sqrt(10080/4000) = 1.58745; 2031.94 -> 2016 (42·48), 1269.96 -> 1248 (26·48); 126·78 = 9828;
      9828/9 = 1092.

    The same aspect at 2560x1600 and the portrait 800x1280 give the same counts (the rule is scale-free and
    symmetric in the two sides)."""
    assert expected(1280, 800, budget) == want
    assert expected(512, 320, budget) == want, (
        "the 512x320 ReGround crop is scaled up to the same budget"
    )
    assert expected(2560, 1600, budget) == want
    assert expected(800, 1280, budget) == want


@pytest.mark.parametrize(
    ("budget", "want"),
    [(280, 264), (560, 527), (1120, 1100)],
)
def test_expected_tokens_odd_size(budget: int, want: int) -> None:
    """A4 (adversarial: a rule that floors to 16, skips the floor, or rounds instead of flooring gives other
    numbers): 1600x896, 100x56 = 5600 patches.

    - 280: scale = sqrt(2520/5600) = sqrt(0.45) = 0.670820; 1073.31 -> 1056 (22·48), 601.06 -> 576 (12·48);
      66·36 = 2376; 2376/9 = 264.
    - 560: scale = sqrt(0.9) = 0.948683; 1517.89 -> 1488 (31·48), 850.02 -> 816 (17·48); 93·51 = 4743;
      4743/9 = 527.
    - 1120: scale = sqrt(1.8) = 1.341641; 2146.63 -> 2112 (44·48), 1202.11 -> 1200 (25·48); 132·75 = 9900;
      9900/9 = 1100.

    Not ``budget`` itself and not the unfloored 280/560/1120: the floor to 48 is what the rows discriminate."""
    assert expected(1600, 896, budget) == want


@pytest.mark.parametrize(("budget", "want"), [(280, 256), (560, 529), (1120, 1089)])
def test_expected_tokens_tiny_image_scaled_up(budget: int, want: int) -> None:
    """A4: a 48x48 image (3x3 = 9 patches) is scaled up to the budget, never served at 1 token.

    - 280: scale = sqrt(2520/9) = 16.733; 803.2 -> 768 (16·48) per side; 48·48 = 2304; 2304/9 = 256.
    - 560: scale = sqrt(5040/9) = 23.664; 1135.9 -> 1104 (23·48); 69·69 = 4761; 4761/9 = 529.
    - 1120: scale = sqrt(10080/9) = 33.466; 1606.4 -> 1584 (33·48); 99·99 = 9801; 9801/9 = 1089."""
    assert expected(48, 48, budget) == want


def test_expected_tokens_never_exceed_the_budget() -> None:
    """A4: ``tokens = min(..., budget)``, over a grid of sizes (a bound, not a re-derivation)."""
    for budget in (280, 560, 1120):
        for w in (48, 320, 512, 777, 1280, 1366, 1920, 3840):
            for h in (48, 200, 320, 768, 800, 1080, 2160):
                got = expected(w, h, budget)
                assert 0 < got <= budget, (w, h, budget, got)


def test_expected_soft_tokens_is_pure() -> None:
    """A4: a pure function: same answer twice, no clock, no I/O (its source names no ``open``/``time``/``os``)."""
    fn = _need("expected_soft_tokens")
    assert fn(1280, 800, 560) == fn(1280, 800, 560)
    tree = ast.parse(inspect.getsource(fn).lstrip())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not names & {"open", "time", "os", "subprocess", "Path"}, names


def test_docstrings_stop_claiming_the_engine_uses_1120() -> None:
    """A4: the module docstring no longer says screen frames go in at 1120 per image; it names the engine default
    280 and the ``visual_tokens`` setting. The ``#:`` comment of ``MAX_SCREEN_VISUAL_TOKENS`` says 1120 is reached
    only with ``visual_tokens = 1120``. Keywords only (the reviewer judges the wording)."""
    _need("ENGINE_DEFAULT_SOFT_TOKENS")
    doc = policy.__doc__ or ""
    assert "maximum* visual-token budget (1120 per image)" not in doc, (
        "T-0131 A4: the module docstring still claims frames go in at 1120 per image"
    )
    assert "280" in doc and "visual_tokens" in doc, doc
    src = inspect.getsource(policy).splitlines()
    at = next(i for i, line in enumerate(src) if line.startswith("MAX_SCREEN_VISUAL_TOKENS"))
    comment: list[str] = []
    for line in reversed(src[:at]):
        if not line.startswith("#:"):
            break
        comment.insert(0, line)
    text = " ".join(comment)
    assert "visual_tokens" in text and "1120" in text, (
        f"T-0131 A4: the constant's comment says 1120 is reached only with visual_tokens = 1120; it reads {text!r}"
    )
