"""T-0057 criterion 4, measured: the four audio modules against a before taken in the same session.

Marked ``bench`` and kept out of ``tests/unit`` and ``tests/contract`` on purpose: this row runs
the four audio modules twice in child processes and costs about a minute, and the fast suite is a
per-turn tax on every agent on the box (T-0031). Run it explicitly:

    uv run pytest tests/bench/test_audio_switch_budget.py -q -m bench -s

and paste the ``T-0057 switch fence:`` line, which carries both switch counts, both load averages
and the ratio.

**This is the fence, not the gate.** The gate is zero leaked ``astra-`` threads, established row by
row in the fast suite by ``tests/conftest.py``'s guard, and it is load-independent. This number is
not: it is the poll rate times the total writer-alive time, so the same fix on the same tree
measured 0.74, 0.62 and 0.65 at load averages 6 to 13 (ruling A12). That is why the before arm is
re-measured here, beside the after, rather than read from a constant: the before arm is this same
tree with the teardown that calls ``close()`` switched off, which is exactly the state the 48
leaking rows were in.
"""

from __future__ import annotations

import pytest

from tests.bench.audio_switch_budget import (
    AUDIO_MODULES,
    SUBSET_MAX_RATIO,
    max_ratio_for,
    measure_pair,
    report_line,
    verdict,
)

pytestmark = pytest.mark.bench


def test_the_four_audio_modules_stay_under_the_switch_fence() -> None:
    """The fence is the ratio of two same-session arms. The wall is printed beside it and asserted
    about nowhere."""
    before, after = measure_pair(AUDIO_MODULES)
    fence = max_ratio_for(AUDIO_MODULES)
    line = report_line(after, before, max_ratio=fence)
    print(line)

    assert after.rows >= 60, (
        f"only {after.rows} rows ran; the census was taken over 66 rows in these four modules and "
        f"a switch count over a different selection is not comparable\n{line}"
    )
    ok, why = verdict(after, before, max_ratio=fence)
    assert ok, (
        f"{why}. The fence is {SUBSET_MAX_RATIO:.2f} x a before measured in the same session "
        f"(ruling A12), not an absolute count: 85 % of the blocking in these modules is one "
        f"thread that outlives its row by 2.00 s at a 0.2 ms poll\n{line}"
    )
