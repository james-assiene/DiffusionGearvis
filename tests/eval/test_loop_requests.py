"""T-0115a criterion 6: the outcomes of the loop-request rows, re-derived from ``requests.jsonl`` alone and checked
against ``report.json`` (``gpu``, ``eval``).

Run through ``/launch`` on the 5090 box (``scripts/locked/t0115_loop_probe.sh`` step 4, after the harness)::

    T0115A_RUN_DIR=logs/t0115a-loop-requests-<sha7> uv run pytest tests/eval/test_loop_requests.py -q -m gpu -rA

Without ``T0115A_RUN_DIR`` every row skips with that reason. The rows spend no GPU time -- they read the run
directory -- but they are ``gpu`` because the data exists only after a ``/launch``-reviewed run (T-0046). They need
no harness, so they carry no ``pending`` marker (as ``tests/eval/test_ui_fields_ab.py``).

**Gate-sha drift.** Before anything else the manifest's ``gate_shas`` for ``local_astra/toolcall/loop.py`` and
``local_astra/conductor/llm_client.py`` are compared with the working tree; any difference skips with the reason
"re-derives against the live loop.py and llm_client.py; reproduces only at the run commit".

**The oracle is this module, not the harness.** From each E line's raw ``commits`` and the history's accepted
strings (``manifest.json`` ``e_histories``, pre-registered in ``docs/cd/toolcall-loop-requests.md``):

- a line that errored (an ``error`` recorded, or a status other than 200) -> errored, whatever its commits hold
  (ruling R4: never correct);
- a ``tool`` commit outside an open thought decides it: ``google_search`` -> re-search, any other call -> other;
- else a request whose **last** commit is thought-open -> other (ruling R1: the final state; the client marks every
  commit of a canvas that ends in a thought, so an earlier canvas's flag says nothing about the end);
- else the answer text (special literals dropped, stripped): empty -> other; containing an accepted string ->
  correct, matched case-insensitively with number/word boundaries (ruling R2: ``16:45`` does not hold ``6:45``);
  otherwise unsupported. The accepted strings are the manifest's list, read as the complete pre-registered list.

then the tally per (placement, arm), ``errored`` counted in the cell, and the bars of criterion 5 in
``fractions.Fraction`` (H-T115a-1, typed from docs/tasks/T-0115a.md): EC FULL correct >= 28/40 at both placements;
E1 kept COMPACT correct >= kept FULL correct - 4/40; E2 old COMPACT (correct + re-search) >= 32/40; E3 old COMPACT
unsupported <= 8/40; "inconclusive" if errored E requests >= 3/160 (ruling R4) or EC fails; else "elision viable"
iff all, "elision hurts: <failing bars>". The cost medians (report-only) are re-derived from the same lines: COMPACT
``prefix_query_tokens - prefix_hit_tokens`` and the paired COMPACT - FULL done ms (the pooled sanity medians), and
per placement (ruling R3) the first COMPACT per cell after its warm and each cell's first pair.

Regime (state it beside every number this prints): T-0115a *Regime* -- lean; vLLM v0.29.0; NVFP4 rev ``2ea8372``;
CL 256, 48 steps; thinking on; prefix caching on; RTX 5090, driver 595.84, gpu-mem 0.86; tools ``bbfc53ccdd80``; the
same boot as T-0115; 223 requests plus 40 warm; concurrency 1; unseeded.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall.special_tokens import drop_special_literals

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV = "T0115A_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]
GATE_FILES = ("local_astra/toolcall/loop.py", "local_astra/conductor/llm_client.py")
SKIP_REASON = (
    "re-derives against the live loop.py and llm_client.py; reproduces only at the run commit"
)

PLACEMENTS = ("old", "kept")
ARMS = ("FULL", "COMPACT")
OUTCOMES = ("correct", "re_search", "unsupported", "other", "errored")
#: ruling R4: this many errored E arm requests (of 160) make E inconclusive
ERRORED_MAX_EXCLUSIVE = 3
BARS = ("EC", "E1", "E2", "E3")
CALL_TAG = "<|tool_call>call:"
BAR_RE = re.compile(r"\b(?:EC|E[123])\b")

#: criterion 5 (H-T115a-1), typed out
EC_MIN = Fraction(28, 40)
E1_MARGIN = Fraction(4, 40)
E2_MIN = Fraction(32, 40)
E3_MAX = Fraction(8, 40)

#: criteria 2-4, typed out
N_Q4_PER_ARM = 30
N_PER_CELL = 40
N_WARM = 40
N_Q5 = 3
Q5_READINGS = frozenset({"accepted", "rejected", "engine-down"})


# =====================================================================================================
# The oracle
# =====================================================================================================


def call_name(text: str) -> str:
    body = text[text.index(CALL_TAG) + len(CALL_TAG) :] if CALL_TAG in text else text
    return body.split("{", 1)[0].strip()


def holds(answer: str, accepted_string: str) -> bool:
    """R2: ``accepted_string`` occurs in ``answer``, case-folded, not inside a longer number or word."""
    if not accepted_string:
        return False
    pattern = r"(?<![0-9a-z])" + re.escape(accepted_string.casefold()) + r"(?![0-9a-z])"
    return re.search(pattern, answer.casefold()) is not None


def errored(line: Mapping[str, Any]) -> bool:
    return bool(line.get("error")) or line.get("status") != 200


def outcome_of(commits: Sequence[Mapping[str, Any]], accepted: Iterable[str]) -> str:
    tools = [c for c in commits if c.get("segment") == "tool" and not c.get("thought_open")]
    if tools:
        return (
            "re_search" if call_name(str(tools[0].get("text", ""))) == "google_search" else "other"
        )
    if commits and commits[-1].get("thought_open"):
        return "other"
    answer = drop_special_literals(
        "".join(str(c.get("text", "")) for c in commits if c.get("segment") == "answer")
    ).strip()
    if not answer:
        return "other"
    return "correct" if any(holds(answer, a) for a in accepted) else "unsupported"


def derive(
    lines: Sequence[Mapping[str, Any]], accepted: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    tally = {p: {a: dict.fromkeys(OUTCOMES, 0) for a in ARMS} for p in PLACEMENTS}
    per_line: list[tuple[str, str]] = []
    for x in lines:
        if x.get("row") != "E" or x.get("arm") not in ARMS:
            continue
        got = (
            "errored" if errored(x) else outcome_of(x.get("commits") or [], accepted[x["history"]])
        )
        tally[x["placement"]][x["arm"]][got] += 1
        per_line.append((str(x["request_id"]), got))

    def rate(p: str, a: str, *keys: str) -> Fraction:
        cell = tally[p][a]
        n = sum(cell.values())
        assert n > 0, f"no {p}/{a} lines"
        return Fraction(sum(cell[k] for k in keys), n)

    bars = {
        "EC": all(rate(p, "FULL", "correct") >= EC_MIN for p in PLACEMENTS),
        "E1": rate("kept", "COMPACT", "correct") >= rate("kept", "FULL", "correct") - E1_MARGIN,
        "E2": rate("old", "COMPACT", "correct", "re_search") >= E2_MIN,
        "E3": rate("old", "COMPACT", "unsupported") <= E3_MAX,
    }
    n_errored = sum(tally[p][a]["errored"] for p in PLACEMENTS for a in ARMS)
    if not bars["EC"] or n_errored >= ERRORED_MAX_EXCLUSIVE:
        kind, failing = "inconclusive", None
    elif all(bars.values()):
        kind, failing = "elision viable", frozenset()
    else:
        kind, failing = "elision hurts", frozenset(b for b in BARS if not bars[b])
    return {"tally": tally, "bars": bars, "kind": kind, "failing": failing, "per_line": per_line}


def cost(lines: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    e = [x for x in lines if x.get("row") == "E"]
    misses = [
        int(x["prefix_query_tokens"]) - int(x["prefix_hit_tokens"])
        for x in e
        if x["arm"] == "COMPACT"
    ]
    by_key: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for x in e:
        if x["arm"] in ARMS:
            by_key.setdefault((x["history"], x["placement"], x["repeat"]), {})[x["arm"]] = x
    done = [
        float(v["COMPACT"]["done_ms"]) - float(v["FULL"]["done_ms"])
        for v in by_key.values()
        if set(v) == set(ARMS)
    ]
    return {
        "compact_miss_tokens_median": statistics.median(misses),
        "done_ms_diff_median": statistics.median(done),
    }


# =====================================================================================================
# Fixtures
# =====================================================================================================


@pytest.fixture(scope="module")
def run_dir() -> Path:
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        pytest.skip(f"{ENV} is not set: no T-0115a run directory to read")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO / path
    assert path.is_dir(), f"{ENV}={raw!r} is not a directory"
    return path


@pytest.fixture(scope="module")
def manifest(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return out


@pytest.fixture(scope="module", autouse=True)
def same_gate(manifest: dict[str, Any]) -> None:
    recorded = manifest.get("gate_shas") or {}
    missing = [p for p in GATE_FILES if p not in recorded]
    assert not missing, f"criterion 6: the manifest records no gate sha for {missing}"
    changed = [
        p for p in GATE_FILES if recorded[p] != hashlib.sha256((REPO / p).read_bytes()).hexdigest()
    ]
    if changed:
        pytest.skip(f"{SKIP_REASON} (changed since the run: {', '.join(changed)})")


@pytest.fixture(scope="module")
def lines(run_dir: Path) -> list[dict[str, Any]]:
    text = (run_dir / "requests.jsonl").read_text(encoding="utf-8")
    return [json.loads(x) for x in text.splitlines() if x.strip()]


@pytest.fixture(scope="module")
def stored(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    return out


@pytest.fixture(scope="module")
def accepted(manifest: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    hist = manifest["e_histories"]
    assert len(hist) == 4, f"criterion 4: {len(hist)} histories"
    return {k: tuple(v["accepted"]) for k, v in hist.items()}


# =====================================================================================================
# Rows
# =====================================================================================================


def test_the_design_ran_whole(lines: list[dict[str, Any]]) -> None:
    """Criteria 2-4: Q4 30 G and 30 A; E 40 per (placement, arm) and 40 warm; Q5 at most 3 and only after every
    other row; an ``engine-down`` Q5 reading is the last line (the harness stopped there). Request ids unique."""
    rows = [x["row"] for x in lines]
    q4 = Counter(x["arm"] for x in lines if x["row"] == "Q4")
    assert q4 == Counter({"G": N_Q4_PER_ARM, "A": N_Q4_PER_ARM}), q4
    cells = Counter((x["placement"], x["arm"]) for x in lines if x["row"] == "E")
    want = Counter({(p, a): N_PER_CELL for p in PLACEMENTS for a in ARMS})
    assert {k: v for k, v in cells.items() if k[1] in ARMS} == dict(want), cells
    assert sum(v for k, v in cells.items() if k[1] == "warm") == N_WARM, cells
    n_q5 = rows.count("Q5")
    assert 1 <= n_q5 <= N_Q5, f"{n_q5} Q5 lines"
    assert rows[-n_q5:] == ["Q5"] * n_q5, "criterion 3: Q5 did not run last"
    readings = [x["reading"] for x in lines if x["row"] == "Q5"]
    assert set(readings) <= Q5_READINGS, readings
    if "engine-down" in readings:
        assert readings.index("engine-down") == len(readings) - 1, (
            "the harness went on after engine-down"
        )
    else:
        assert n_q5 == N_Q5, f"{n_q5} Q5 lines without an engine-down"
    assert len({x["request_id"] for x in lines}) == len(lines), "a request id repeats"


def test_report_agrees_with_the_raw_requests(
    lines: list[dict[str, Any]], accepted: dict[str, tuple[str, ...]], stored: dict[str, Any]
) -> None:
    """Criteria 4-6: every E arm line's recorded outcome is the oracle's; the tally, the four bars and the outcome
    (its form, and the bars an "elision hurts" names) in ``report.json`` are the oracle's."""
    got = derive(lines, accepted)
    recorded = {str(x["request_id"]): x.get("outcome") for x in lines if x.get("row") == "E"}
    wrong = [(rid, recorded[rid], o) for rid, o in got["per_line"] if recorded[rid] != o]
    assert wrong == [], (
        f"{len(wrong)} lines classified otherwise (id, recorded, oracle): {wrong[:5]}"
    )
    rep = stored["E"]
    assert rep["tally"] == got["tally"], (rep["tally"], got["tally"])
    if got["bars"]["EC"]:
        assert {b: rep["bars"][b] for b in BARS} == got["bars"], (rep["bars"], got["bars"])
    else:
        assert rep["bars"]["EC"] is False, rep["bars"]
    outcome = str(rep["outcome"])
    assert outcome.startswith(got["kind"]), (outcome, got["kind"])
    if got["kind"] == "elision hurts":
        assert set(BAR_RE.findall(outcome)) == set(got["failing"]), outcome


def test_q5_readings_agree(lines: list[dict[str, Any]], stored: dict[str, Any]) -> None:
    """Criterion 3: the report's Q5 readings are the lines', in order."""
    assert list(stored["Q5"]["readings"]) == [x["reading"] for x in lines if x["row"] == "Q5"]


def test_cost_medians_agree(lines: list[dict[str, Any]], stored: dict[str, Any]) -> None:
    """Criterion 5, report-only cost: the median COMPACT miss (query - hit, COMPACT lines only) and the paired median
    COMPACT - FULL done ms, re-derived, equal the report's."""
    got = cost(lines)
    rep = stored["E"]["cost"]
    for k, v in got.items():
        assert float(rep[k]) == pytest.approx(v), (k, rep[k], v)


def first_compact_per_cell(lines: Sequence[Mapping[str, Any]]) -> list[int]:
    """Per (history, placement), in send order: the miss (query - hit) of the first COMPACT line after the cell's
    first warm line. A cell whose first COMPACT precedes every warm contributes nothing (the miss would not be a
    one-compaction cost)."""
    warmed: set[tuple[Any, Any]] = set()
    done: set[tuple[Any, Any]] = set()
    out: list[int] = []
    for x in lines:
        if x.get("row") != "E":
            continue
        key = (x["history"], x["placement"])
        if x["arm"] == "warm":
            warmed.add(key)
        elif x["arm"] == "COMPACT" and key in warmed and key not in done:
            done.add(key)
            out.append(int(x["prefix_query_tokens"]) - int(x["prefix_hit_tokens"]))
    return out


def test_first_compact_per_cell_agrees(
    lines: list[dict[str, Any]], manifest: dict[str, Any], stored: dict[str, Any]
) -> None:
    """Criterion 5's cost reading, as ruled 2026-10-17 (docs/tasks/T-0115a.md, last *Rulings* entry): the median miss
    of the first COMPACT per (history, placement) after its warm, n = 8 (4 histories x 2 placements), re-derived from
    ``requests.jsonl`` alone, equals the report's ``compact_miss_tokens_first_per_cell_median``; the predicted median
    (manifest ``predicted_miss_tokens``, measured cells only) equals ``predicted_miss_tokens_median``. Every cell's
    first COMPACT comes after a warm (the schedule); the reading is printed beside the all-COMPACT sanity median."""
    firsts = first_compact_per_cell(lines)
    cells = {(x["history"], x["placement"]) for x in lines if x.get("row") == "E"}
    assert len(firsts) == len(cells) == 8, (
        f"{len(firsts)} first COMPACTs after a warm over {len(cells)} cells"
    )
    rep = stored["E"]["cost"]
    want = statistics.median(firsts)
    assert int(rep["compact_miss_tokens_first_per_cell_n"]) == 8, rep
    assert float(rep["compact_miss_tokens_first_per_cell_median"]) == pytest.approx(want), (
        rep["compact_miss_tokens_first_per_cell_median"],
        want,
        firsts,
    )
    predicted = [
        v
        for v in (manifest.get("predicted_miss_tokens") or {}).values()
        if isinstance(v, int | float) and not isinstance(v, bool)
    ]
    if predicted:
        assert float(rep["predicted_miss_tokens_median"]) == pytest.approx(
            statistics.median(predicted)
        ), (rep["predicted_miss_tokens_median"], predicted)
    else:
        assert rep["predicted_miss_tokens_median"] is None, rep
    print(
        f"T-0115a cost (regime: module docstring): first COMPACT per cell median {want} (n {len(firsts)}, "
        f"{sorted(firsts)}); predicted {rep['predicted_miss_tokens_median']}; "
        f"all-COMPACT sanity median {rep['compact_miss_tokens_median']}"
    )


def first_pairs_by_placement(
    lines: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    """Per placement, over its cells, in send order (ruling R3): ``miss`` the first COMPACT after the cell's first
    warm; ``fc`` / ``done`` the COMPACT - FULL difference of the cell's first pair after that warm (the lowest repeat
    with both arms)."""
    out: dict[str, dict[str, list[float]]] = {
        p: {"miss": [], "fc": [], "done": []} for p in PLACEMENTS
    }
    cells: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
    for x in lines:
        if x.get("row") == "E":
            cells.setdefault((x["history"], x["placement"]), []).append(x)
    for (_, placement), mine in cells.items():
        warm_at = next((i for i, x in enumerate(mine) if x["arm"] == "warm"), None)
        if warm_at is None:
            continue
        after = mine[warm_at + 1 :]
        first_compact = next((x for x in after if x["arm"] == "COMPACT"), None)
        if first_compact is not None:
            out[placement]["miss"].append(
                int(first_compact["prefix_query_tokens"]) - int(first_compact["prefix_hit_tokens"])
            )
        by_rep: dict[Any, dict[str, Mapping[str, Any]]] = {}
        for x in after:
            if x["arm"] in ARMS:
                by_rep.setdefault(x["repeat"], {}).setdefault(x["arm"], x)
        both = sorted(r for r, v in by_rep.items() if set(v) == set(ARMS))
        if both:
            pair = by_rep[both[0]]
            for key, col in (("fc", "first_commit_ms"), ("done", "done_ms")):
                c, f = pair["COMPACT"].get(col), pair["FULL"].get(col)
                if c is not None and f is not None:
                    out[placement][key].append(float(c) - float(f))
    return out


def test_cost_by_placement_agrees(
    lines: list[dict[str, Any]], manifest: dict[str, Any], stored: dict[str, Any]
) -> None:
    """Ruling R3: per placement, the first COMPACT per cell after its warm (n = 4 each), the predicted miss of that
    placement's cells (manifest keys ``<history>/<placement>``), the first-pair paired first-commit and done ms, and
    ms per miss token (the done median over the miss median), re-derived from ``requests.jsonl`` alone, equal the
    report's ``cost["by_placement"]``. The old placement's reading is the one-compaction cost (R9)."""
    got = first_pairs_by_placement(lines)
    split = stored["E"]["cost"]["by_placement"]
    for p in PLACEMENTS:
        rep = split[p]
        miss = got[p]["miss"]
        assert len(miss) == 4, f"{p}: {len(miss)} first COMPACTs after a warm"
        assert int(rep["compact_miss_tokens_first_per_cell_n"]) == 4, rep
        assert float(rep["compact_miss_tokens_first_per_cell_median"]) == pytest.approx(
            statistics.median(miss)
        ), (p, rep, miss)
        predicted = [
            v
            for k, v in (manifest.get("predicted_miss_tokens") or {}).items()
            if str(k).endswith(f"/{p}") and isinstance(v, int | float) and not isinstance(v, bool)
        ]
        if predicted:
            assert float(rep["predicted_miss_tokens_median"]) == pytest.approx(
                statistics.median(predicted)
            ), (p, rep, predicted)
        for key, col in (("fc", "first_commit_ms_diff_median"), ("done", "done_ms_diff_median")):
            assert got[p][key], f"{p}: no first pairs for {col}"
            assert float(rep[col]) == pytest.approx(statistics.median(got[p][key])), (p, col, rep)
        miss_median = statistics.median(miss)
        if miss_median:
            assert float(rep["ms_per_miss_token"]) == pytest.approx(
                statistics.median(got[p]["done"]) / miss_median
            ), (p, rep)
        print(
            f"T-0115a cost by placement (regime: module docstring): {p}: first COMPACT miss median "
            f"{miss_median} (n {len(miss)}, {sorted(miss)}); first-pair done diff median "
            f"{statistics.median(got[p]['done'])} ms; ms per miss token {rep['ms_per_miss_token']}"
        )
