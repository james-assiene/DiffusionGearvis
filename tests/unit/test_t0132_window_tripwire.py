"""T-0132 A6: the window tripwire (docs/tasks/T-0132.md).

Step k > 1 of a live turn has no token check; at 1120 the worst case (``Budget.max_images`` images at about 1092 soft
tokens each, plus the live system message, plus a ``MAX_CONTINUATION`` continuation) is what group C's P4 measures.
This row pins the *inputs* of that arithmetic, so a change to any of them is a red row that says to re-measure,
never a silent shift of the worst case. It computes no window and makes no characters-per-token estimate (B10).

Expected values are A6's literals; ``policy.expected_soft_tokens(1280, 800, 1120) == 1092`` was checked by hand:
1280x800 is 80 x 50 = 4000 patches; scale = sqrt(1120 * 9 / 4000) = 1.5875; 2032 -> 2016 and 1270 -> 1248 (floors to
multiples of 48); (2016 / 16) * (1248 / 16) / 9 = 126 * 78 / 9 = 1092.

Group A, activated 2026-10-22 (was ``pending(task="T-0132")``): at BASE a2d9ac6 ``config.LIVE_DEFAULT_VISUAL_TOKENS`` does not exist.

Regime: CPU, pure; reads ``services/vllm.sh`` as text; no measurement.
"""

from __future__ import annotations

import re
from pathlib import Path

from local_astra import config
from local_astra.conductor import session
from local_astra.conductor.prompt_builder import Budget
from local_astra.perception import policy
from local_astra.toolcall import loop

REPO = Path(__file__).resolve().parents[2]
WHY = "re-measure the worst-case window (T-0132 P4)"


def test_window_inputs_pinned() -> None:
    """A6: ``Budget.max_images == 4``, ``Budget.max_frames_per_turn == 3``, ``MAX_CONTINUATION == 14_000``,
    ``COMPACT_KEEP_LAST == 2``, ``services/vllm.sh`` carries ``--max-model-len 16384`` and ``"image": 4``,
    ``LIVE_DEFAULT_VISUAL_TOKENS == 1120`` and ``expected_soft_tokens(1280, 800, 1120) == 1092``."""
    budget = Budget()
    vllm_sh = (REPO / "services" / "vllm.sh").read_text()
    got = {
        "Budget.max_images": budget.max_images,
        "Budget.max_frames_per_turn": budget.max_frames_per_turn,
        "session.MAX_CONTINUATION": session.MAX_CONTINUATION,
        "loop.COMPACT_KEEP_LAST": loop.COMPACT_KEEP_LAST,
        "vllm.sh --max-model-len": (
            m.group(1) if (m := re.search(r"--max-model-len\s+(\d+)", vllm_sh)) else None
        ),
        "vllm.sh image limit": (
            m.group(1) if (m := re.search(r'\\?"image\\?"\s*:\s*(\d+)', vllm_sh)) else None
        ),
        "config.LIVE_DEFAULT_VISUAL_TOKENS": getattr(config, "LIVE_DEFAULT_VISUAL_TOKENS", None),
        "expected_soft_tokens(1280, 800, 1120)": policy.expected_soft_tokens(1280, 800, 1120),
    }
    want = {
        "Budget.max_images": 4,
        "Budget.max_frames_per_turn": 3,
        "session.MAX_CONTINUATION": 14_000,
        "loop.COMPACT_KEEP_LAST": 2,
        "vllm.sh --max-model-len": "16384",
        "vllm.sh image limit": "4",
        "config.LIVE_DEFAULT_VISUAL_TOKENS": 1120,
        "expected_soft_tokens(1280, 800, 1120)": 1092,
    }
    moved = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    assert not moved, f"T-0132 A6: a window input moved (got, pinned): {moved}; {WHY}"
