"""T-0120c criterion 10: the compaction re-check re-derived from ``requests.jsonl`` alone (``gpu`` rows), and the
oracle the CPU calibration row and the mock-engine run share.

**The gpu rows** run through ``/launch`` on the 5090 box, after the harness (``scripts/locked/t0120c_compaction_recheck.sh``)::

    T0120C_RUN_DIR=logs/t0120c-compaction-recheck-<sha7> uv run pytest tests/eval/test_compaction_recheck.py -q -m gpu -rA

Without ``T0120C_RUN_DIR`` they skip with that reason. They spend no GPU time -- they read the run directory -- but
they are ``gpu`` because the data exists only after a ``/launch``-reviewed run (T-0046). They need no harness, so they
carry no ``pending`` marker. Before anything else each compares the manifest's ``gate_shas`` for the four gate files
(``local_astra/toolcall/loop.py``, ``local_astra/conductor/llm_client.py``, ``scripts/gpu/t0115a_loop_requests.py``,
``scripts/gpu/t0120c_compaction_recheck.py``) with the working tree and skips on drift.

**Where the calibration row lives (a reading; flagged in the test-author report).** Criterion 10 puts
``test_calibration_c398fdf`` (criterion 7, CPU, not ``gpu``) in this module, but T-0046's invariant
(``tests/unit/test_pytest_config.py::test_a_path_only_run_of_the_eval_directory_selects_nothing``) requires every row
under ``tests/eval`` to be deselected by a path-only run, so a non-``gpu`` row here turns a fast-suite row red; marking
it ``gpu`` instead would put a CPU pre-launch check behind a ``/launch`` token. It therefore lives in
``tests/integration/test_t0120c_harness_smoke.py`` and uses this module's oracle; every row here is ``gpu``.

**The oracle is this module, not the harness.** Per E arm line, from its raw ``commits``:

- a line with an ``error`` recorded or a status other than 200 -> ``errored`` (R4), whatever its commits hold;
- a ``tool`` commit outside an open thought decides it: ``google_search`` -> ``re_search``, another call -> ``other``;
- else a **last** commit inside a thought -> ``other`` (R1, the final state);
- else the answer (the ``answer`` commits joined, special literals dropped, stripped): empty -> ``other``; holding an
  accepted string -> ``correct``; else ``unsupported``. Criterion 4's match: casefold both, every ``-``/``–`` a space,
  whitespace collapsed, no letter or digit next to the match (the list from the manifest's ``accepted``). T-0115a's
  match (``outcome_t0115a``): casefold only, the same boundaries (the manifest's ``accepted_t0115a``);
- ``invented`` (criterion 6, old COMPACT only): not errored, the last commit not inside a thought, and either the
  outcome is ``correct`` or the answer, normalised, matches one of the history's ``invented_patterns`` (manifest;
  ``re.search``).

then the tallies of (old, FULL), (old, COMPACT), (kept, COMPACT) under both lists, the bars of criterion 8 in exact
counts of each cell's n (H-T120c-1, typed from docs/tasks/T-0120c.md): **C** old FULL correct >= 36/40; **R** old
COMPACT re-search >= 24/40; **I** old COMPACT invented = 0; **K** kept COMPACT correct >= 38/40; and criterion 9's
outcome: "inconclusive: <reasons>" if C fails, 3 or more of the 120 arm requests errored, or the run is incomplete;
else "stub re-searches" iff R, I and K hold; else "stub falls short: <every failing bar>".

Regime (state it beside every number this prints): T-0120c *Regime* -- as c398fdf: lean, vLLM v0.29.0, NVFP4
``nvidia/diffusiongemma-26B-A4B-it-NVFP4`` rev ``2ea8372``, chat template ``36e3a42e``, tokenizer ``cc8d3a0c``, CL 256,
48 steps, thinking on, prefix caching on, ``max_num_seqs 3``, RTX 5090, driver 595.84, gpu-mem 0.86, tools
``native:bbfc53ccdd80``, repair on, dev split, screen ``scroll-01``, ``E_BUDGET`` 7000, concurrency 1, unseeded, no
audio; n 160 requests (120 arm + 40 warm), one run; differences: the commit, ``loop.py`` (the stub only), a fresh boot
with E first, no Q4/Q5, no kept FULL.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from local_astra.toolcall.special_tokens import drop_special_literals
from tests.unit.test_t0120c_compaction_recheck import (
    ACCEPTED_C4,
    ACCEPTED_T0115A,
    C398FDF_E_TEXTS_SHA256,
    COMPACT_KEYS,
    HISTORY_IDS,
    WARM_FULL_KEYS,
)

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

ENV_RUN = "T0120C_RUN_DIR"
REPO = Path(__file__).resolve().parents[2]
GATE_FILES = (
    "local_astra/toolcall/loop.py",
    "local_astra/conductor/llm_client.py",
    "scripts/gpu/t0115a_loop_requests.py",
    "scripts/gpu/t0120c_compaction_recheck.py",
)
SKIP_REASON = "re-derives against the live gate files; reproduces only at the run commit"

CELLS = (("old", "FULL"), ("old", "COMPACT"), ("kept", "COMPACT"))
OUTCOMES = ("correct", "re_search", "unsupported", "other", "errored")
BARS = ("C", "R", "I", "K")
#: criterion 8 (H-T120c-1), exact fractions of 40
C_MIN = Fraction(36, 40)
R_MIN = Fraction(24, 40)
K_MIN = Fraction(38, 40)
#: criterion 9 (H-T120c-5)
ERRORED_INCONCLUSIVE = 3
N_PER_CELL, N_WARM, N_ARM = 40, 40, 120
CALL_TAG = "<|tool_call>call:"
QUERY_RE = re.compile(r'query:<\|"\|>(.*?)<\|"\|>', re.DOTALL)
BAR_LETTER = re.compile(r"\b[CRIK]\b")


# =====================================================================================================
# The oracle
# =====================================================================================================


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold().replace("-", " ").replace("–", " ")).strip()


def _bounded(needle: str, hay: str) -> bool:
    if not needle:
        return False
    return re.search(r"(?<![0-9a-z])" + re.escape(needle) + r"(?![0-9a-z])", hay) is not None


def holds_c4(answer: str, accepted: str) -> bool:
    """Criterion 4's match."""
    return _bounded(normalise(accepted), normalise(answer))


def holds_t0115a(answer: str, accepted: str) -> bool:
    """T-0115a's R2 match, unchanged: casefold only."""
    return _bounded(accepted.casefold(), answer.casefold())


def answer_text(commits: Sequence[Mapping[str, Any]]) -> str:
    return drop_special_literals(
        "".join(str(c.get("text", "")) for c in commits if c.get("segment") == "answer")
    ).strip()


def _tools(commits: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [c for c in commits if c.get("segment") == "tool" and not c.get("thought_open")]


def call_name(text: str) -> str:
    body = text[text.index(CALL_TAG) + len(CALL_TAG) :] if CALL_TAG in text else text
    return body.split("{", 1)[0].strip()


def errored(line: Mapping[str, Any]) -> bool:
    return bool(line.get("error")) or line.get("status") != 200


def ends_in_thought(commits: Sequence[Mapping[str, Any]]) -> bool:
    return bool(commits) and bool(commits[-1].get("thought_open"))


def classify(line: Mapping[str, Any], accepted: Sequence[str], holds: Any) -> str:
    if errored(line):
        return "errored"
    commits = line.get("commits") or []
    tools = _tools(commits)
    if tools:
        return (
            "re_search" if call_name(str(tools[0].get("text", ""))) == "google_search" else "other"
        )
    if ends_in_thought(commits):
        return "other"
    answer = answer_text(commits)
    if not answer:
        return "other"
    return "correct" if any(holds(answer, a) for a in accepted) else "unsupported"


def query_of(line: Mapping[str, Any]) -> str | None:
    tools = _tools(line.get("commits") or [])
    if not tools:
        return None
    found = QUERY_RE.search(str(tools[0].get("text", "")))
    return found.group(1) if found else None


def is_invented(line: Mapping[str, Any], outcome: str, patterns: Sequence[str]) -> bool:
    """Criterion 6, on an old-COMPACT line (False on every other line)."""
    if (line.get("placement"), line.get("arm")) != ("old", "COMPACT"):
        return False
    if outcome == "errored" or ends_in_thought(line.get("commits") or []):
        return False
    if outcome == "correct":
        return True
    text = normalise(answer_text(line.get("commits") or []))
    return any(re.search(p, text) is not None for p in patterns)


def _empty() -> dict[str, dict[str, dict[str, int]]]:
    out: dict[str, dict[str, dict[str, int]]] = {}
    for p, a in CELLS:
        out.setdefault(p, {})[a] = dict.fromkeys(OUTCOMES, 0)
    return out


def derive(lines: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Every tally, ``invented``, the bars and the outcome kind, from the lines and the manifest's lists and patterns."""
    acc = {k: tuple(v) for k, v in manifest["accepted"].items()}
    acc15 = {k: tuple(v) for k, v in manifest["accepted_t0115a"].items()}
    patterns = {k: tuple(v) for k, v in manifest["invented_patterns"].items()}
    tally, tally15 = _empty(), _empty()
    per_line: dict[str, dict[str, Any]] = {}
    flagged: list[tuple[str, str]] = []
    arm_lines = [x for x in lines if x.get("row") == "E" and x.get("arm") in ("FULL", "COMPACT")]
    for x in arm_lines:
        hid, p, a = str(x["history"]), str(x["placement"]), str(x["arm"])
        o = classify(x, acc[hid], holds_c4)
        o15 = classify(x, acc15[hid], holds_t0115a)
        inv = is_invented(x, o, patterns[hid])
        if (p, a) in CELLS:
            tally[p][a][o] += 1
            tally15[p][a][o15] += 1
        if inv:
            flagged.append((str(x["request_id"]), answer_text(x.get("commits") or [])))
        per_line[str(x["request_id"])] = {
            "outcome": o,
            "outcome_t0115a": o15,
            "invented": inv,
            "query": query_of(x) if o == "re_search" else None,
        }

    def rate(p: str, a: str, k: str) -> Fraction:
        n = sum(tally[p][a].values())
        return Fraction(tally[p][a][k], n) if n else Fraction(0)

    n_invented = len(flagged)
    bars = {
        "C": rate("old", "FULL", "correct") >= C_MIN,
        "R": rate("old", "COMPACT", "re_search") >= R_MIN,
        "I": n_invented == 0,
        "K": rate("kept", "COMPACT", "correct") >= K_MIN,
    }
    n_errored = sum(tally[p][a]["errored"] for p, a in CELLS)
    warms = sum(1 for x in lines if x.get("row") == "E" and x.get("arm") == "warm")
    complete = len(arm_lines) == N_ARM and warms == N_WARM and not manifest.get("stopped")
    reasons = set()
    if not bars["C"]:
        reasons.add("C")
    if n_errored >= ERRORED_INCONCLUSIVE:
        reasons.add("errored")
    if not complete:
        reasons.add("incomplete")
    if reasons:
        kind, failing = "inconclusive", frozenset()
    elif bars["R"] and bars["I"] and bars["K"]:
        kind, failing = "stub re-searches", frozenset()
    else:
        kind, failing = "stub falls short", frozenset(b for b in ("R", "I", "K") if not bars[b])
    return {
        "tally": tally,
        "tally_t0115a": tally15,
        "per_line": per_line,
        "flagged": flagged,
        "invented": {"k": n_invented, "n": sum(tally["old"]["COMPACT"].values())},
        "errored": n_errored,
        "complete": complete,
        "bars": bars,
        "kind": kind,
        "failing": failing,
        "reasons": frozenset(reasons),
    }


def check_report(stored: Mapping[str, Any], got: Mapping[str, Any]) -> None:
    """``report.json`` against the oracle: both tallies, invented k/n and the flagged answers, the bars, the outcome's
    form, its failing bars or its reasons."""
    for key in ("tally", "tally_t0115a"):
        for p, a in CELLS:
            cell = stored[key][p][a]
            assert {o: cell.get(o, 0) for o in OUTCOMES} == got[key][p][a], (
                key,
                p,
                a,
                cell,
                got[key][p][a],
            )
    assert dict(stored["invented"]) == got["invented"], (stored["invented"], got["invented"])
    listed = [(str(x["request_id"]), str(x["answer"])) for x in stored["invented_answers"]]
    assert listed == got["flagged"], (
        f"criterion 6: the flagged answers are not listed verbatim: {listed}"
    )
    assert {b: bool(stored["bars"][b]) for b in BARS} == got["bars"], (stored["bars"], got["bars"])
    outcome = str(stored["outcome"])
    if got["kind"] == "inconclusive":
        assert outcome.startswith("inconclusive:"), outcome
        reasons = outcome.split(":", 1)[1]
        assert ("C" in got["reasons"]) is bool(re.search(r"\bC\b", reasons)), outcome
        assert ("errored" in got["reasons"]) is ("errored" in reasons.lower()), outcome
        assert ("incomplete" in got["reasons"]) is ("incomplete" in reasons.lower()), outcome
    elif got["kind"] == "stub re-searches":
        assert outcome == "stub re-searches", outcome
    else:
        assert outcome.startswith("stub falls short:"), outcome
        assert set(BAR_LETTER.findall(outcome.split(":", 1)[1])) == set(got["failing"]), outcome


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dir_from(env: str) -> Path | None:
    raw = os.environ.get(env, "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


# =====================================================================================================
# The gpu rows
# =====================================================================================================


@pytest.fixture(scope="module")
def run() -> tuple[Path, dict[str, Any]]:
    path = _dir_from(ENV_RUN)
    if path is None:
        pytest.skip(f"{ENV_RUN} is not set: no T-0120c run directory to read")
    assert path.is_dir(), f"{ENV_RUN}={path} is not a directory"
    manifest: dict[str, Any] = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest.get("gate_shas") or {}
    missing = [p for p in GATE_FILES if p not in recorded]
    assert not missing, f"criterion 10: the manifest records no gate sha for {missing}"
    changed = [p for p in GATE_FILES if recorded[p] != _sha_file(REPO / p)]
    if changed:
        pytest.skip(f"{SKIP_REASON} (changed since the run: {', '.join(changed)})")
    return path, manifest


@pytest.mark.gpu
def test_the_design_ran_whole(run: tuple[Path, dict[str, Any]]) -> None:
    """Criterion 2: 160 lines, all E (no Q4/Q5): 40 warm, old FULL 40, old COMPACT 40, kept COMPACT 40, no kept FULL;
    the ids are the schedule's (``t0120c-e-<history>-<placement>-<arm>-r<repeat>``), unique, and every cell's warm
    comes before its first arm line."""
    path, _ = run
    lines = read_lines(path / "requests.jsonl")
    assert {x.get("row") for x in lines} == {"E"}, {x.get("row") for x in lines}
    want: set[str] = set()
    for hid in HISTORY_IDS:
        for r in range(10):
            want |= {
                f"t0120c-e-{hid}-old-FULL-r{r}",
                f"t0120c-e-{hid}-old-COMPACT-r{r}",
                f"t0120c-e-{hid}-kept-COMPACT-r{r}",
            }
            if r % 2 == 0:
                want |= {f"t0120c-e-{hid}-old-warm-r{r}", f"t0120c-e-{hid}-kept-warm-r{r}"}
    ids = [str(x["request_id"]) for x in lines]
    assert len(ids) == len(set(ids)) == len(want) == 160, (len(ids), len(set(ids)))
    assert set(ids) == want, sorted(set(ids) ^ want)[:6]
    for hid in HISTORY_IDS:
        for p in ("old", "kept"):
            mine = [x for x in lines if x["history"] == hid and x["placement"] == p]
            assert mine[0]["arm"] == "warm", f"{hid}/{p}: the first line is not the warm"


@pytest.mark.gpu
def test_regime_continuity(run: tuple[Path, dict[str, Any]]) -> None:
    """Criterion 3: the 16 warm/FULL texts sent have c398fdf's sha256 (pinned by hand), the 8 COMPACT texts do not
    (the invite stub), and the elided set is {0, 1, 2} in all 8 cells. The prompt-token counts are report-only: any
    warm/FULL count that differs from c398fdf's is printed."""
    _, manifest = run
    shas = manifest["e_texts_sha256"]
    drifted = [k for k in WARM_FULL_KEYS if shas.get(k) != C398FDF_E_TEXTS_SHA256[k]]
    assert drifted == [], f"criterion 3: warm/FULL texts that are not c398fdf's: {drifted}"
    same = [k for k in COMPACT_KEYS if shas.get(k) == C398FDF_E_TEXTS_SHA256[k]]
    assert same == [], f"criterion 3: COMPACT texts identical to c398fdf's (no invite stub): {same}"
    elided = {k: list(v) for k, v in manifest["elided"].items()}
    assert elided == {f"{h}/{p}": [0, 1, 2] for h in HISTORY_IDS for p in ("old", "kept")}, elided
    print(f"T-0120c prompt_tokens (regime: module docstring): {manifest.get('prompt_tokens')}")


@pytest.mark.gpu
def test_manifest_carries_the_preregistered_lists(run: tuple[Path, dict[str, Any]]) -> None:
    """Criteria 4 and 6: the manifest's ``accepted`` is criterion 4's list and ``accepted_t0115a`` T-0115a's, both
    typed by hand in the unit module; ``invented_patterns`` holds, per history, regex strings that compile."""
    _, manifest = run
    assert {k: sorted(v) for k, v in manifest["accepted"].items()} == {
        k: sorted(v) for k, v in ACCEPTED_C4.items()
    }
    assert {k: sorted(v) for k, v in manifest["accepted_t0115a"].items()} == {
        k: sorted(v) for k, v in ACCEPTED_T0115A.items()
    }
    patterns = manifest["invented_patterns"]
    assert set(patterns) == set(HISTORY_IDS), sorted(patterns)
    for hid, ps in patterns.items():
        assert ps and all(isinstance(p, str) for p in ps), (hid, ps)
        for p in ps:
            re.compile(p)


@pytest.mark.gpu
def test_report_agrees_with_the_raw_requests(run: tuple[Path, dict[str, Any]]) -> None:
    """Criteria 5, 6, 8, 9 and 10: every arm line's recorded ``outcome``, ``outcome_t0115a``, ``invented`` and (on a
    re-search) ``query`` are the oracle's; ``report.json``'s tallies under both lists, invented k/n and its verbatim
    list, the bars and the outcome are the oracle's. Prints the readings with the regime."""
    path, manifest = run
    lines = read_lines(path / "requests.jsonl")
    got = derive(lines, manifest)
    wrong: list[tuple[str, str, Any, Any]] = []
    by_id = {str(x["request_id"]): x for x in lines}
    for rid, want in got["per_line"].items():
        line = by_id[rid]
        for key in ("outcome", "outcome_t0115a"):
            if line.get(key) != want[key]:
                wrong.append((rid, key, line.get(key), want[key]))
        if (
            line["placement"] == "old"
            and line["arm"] == "COMPACT"
            and bool(line.get("invented")) != want["invented"]
        ):
            wrong.append((rid, "invented", line.get("invented"), want["invented"]))
        if want["outcome"] == "re_search" and line.get("query") != want["query"]:
            wrong.append((rid, "query", line.get("query"), want["query"]))
    assert wrong == [], (
        f"{len(wrong)} fields recorded otherwise (id, field, recorded, oracle): {wrong[:6]}"
    )
    stored = json.loads((path / "report.json").read_text(encoding="utf-8"))
    check_report(stored, got)
    print(
        f"T-0120c (regime: module docstring; n 40 per cell): tally {got['tally']}; under T-0115a's list "
        f"{got['tally_t0115a']}; invented {got['invented']}; bars {got['bars']}; outcome {stored['outcome']}"
    )
