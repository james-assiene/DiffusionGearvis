"""T-0071a criterion 6: the two B13 rows in ``local_astra/bench/thresholds.py``, and nothing else changes.

Source of the rows: T-0071 criterion 6 (``Threshold("B13", "cd_step_overhead_pct_b64", 10.0, "%")`` and
``Threshold("B13", "cd_step_overhead_pct_b128", 15.0, "%")``) = T-0062 decision 8 (≤ 10 % at bucket ≤ 64, ≤ 15 % at
bucket ≤ 128). The negative half ("no pre-existing row changes"; on FAIL the limit is never raised, D-H1) is checked
against a literal copy of every row as it stood at 6c82d9e, typed out here (not read from the module), so editing a
limit, a unit, a note or the order of an old row fails, and so does a third B13 row or any other new row.

The ``note`` of the two new rows is free (the criterion does not fix it).
"""

from __future__ import annotations

import pytest

from local_astra.bench.thresholds import THRESHOLDS, Threshold

#: every row of THRESHOLDS at 6c82d9e, verbatim, in order (bench, metric, limit, unit, note)
PRE_EXISTING: tuple[tuple[str, str, float, str, str], ...] = (
    ("B3", "last_word_to_first_audio_p50_ms", 1500, "ms", "G1 p50; regime lean, canvas 256"),
    ("B3", "last_word_to_first_audio_p95_ms", 2200, "ms", "G1 p95"),
    ("B4", "barge_in_pause_p95_ms", 300, "ms", "speech onset → playback paused"),
    ("B7", "vram_total_gib", 29.0, "GiB", "lean profile incl. sampler transient head-room"),
    ("B11", "prefix_cache_hit_ratio", 0.9, "ratio", "prefill-then-generate; ≥"),
    ("B16", "tap_overhead_pct", 3.0, "%", "plugin per-step tap vs no plugin"),
)

#: criterion 6, exactly: (metric, limit, unit)
B13_ROWS = {
    "cd_step_overhead_pct_b64": (10.0, "%"),
    "cd_step_overhead_pct_b128": (15.0, "%"),
}


def _as_tuple(t: Threshold) -> tuple[str, str, float, str, str]:
    return (t.bench, t.metric, t.limit, t.unit, t.note)


def _b13() -> list[Threshold]:
    return [t for t in THRESHOLDS if t.bench == "B13"]


@pytest.mark.parametrize("metric", sorted(B13_ROWS))
def test_the_b13_row_exists_with_its_limit_and_unit(metric: str) -> None:
    rows = [t for t in _b13() if t.metric == metric]
    assert len(rows) == 1, (
        f"T-0071a criterion 6: want exactly one Threshold('B13', {metric!r}, ...), found {rows}"
    )
    limit, unit = B13_ROWS[metric]
    assert rows[0].limit == limit, (
        f"{metric}: limit {rows[0].limit}, want {limit} (T-0062 decision 8)"
    )
    assert rows[0].unit == unit, f"{metric}: unit {rows[0].unit!r}, want {unit!r}"


def test_b13_has_exactly_the_two_rows() -> None:
    got = sorted(t.metric for t in _b13())
    assert got == sorted(B13_ROWS), f"B13 rows are {got}, want exactly {sorted(B13_ROWS)}"


def test_the_bucket_64_limit_is_the_tighter_one() -> None:
    """Decision 8: 10 % at ≤ 64 is stricter than 15 % at ≤ 128; swapped limits would fail here and above."""
    by = {t.metric: t.limit for t in _b13()}
    assert by.get("cd_step_overhead_pct_b64", float("inf")) < by.get(
        "cd_step_overhead_pct_b128", float("-inf")
    ), by


def test_no_pre_existing_row_changes_and_no_other_row_is_added() -> None:
    others = [_as_tuple(t) for t in THRESHOLDS if t.bench != "B13"]
    assert others == list(PRE_EXISTING), (
        "criterion 6 (negative): every pre-existing threshold row must stay byte-identical, in order, and no "
        f"row other than the two B13 rows may be added\n got: {others}\nwant: {list(PRE_EXISTING)}"
    )
