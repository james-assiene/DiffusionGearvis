"""T-0136 A2b-6 / A2b-8 on the full corpus: the re-tuned detector against the blind hand labels, scored per
independent group on the **test** split (docs/tasks/T-0136.md, amendment A1c/A2b: A2b-2, A2b-3, A2b-6, A2b-8;
"Tests to write (A2b)", eval).

**What is scored (A2b-6).** Pairs of ``tests/fixtures/t0136/labels.jsonl`` with ``frame_kind == "live"`` (group B
never measures overview-frame steps, H-T136-3) are grouped by ``group`` (``"<before cluster>:<after cluster>"``,
clusters by the O-A1 (7) jitter rule: same size and max full-resolution gray |delta| <= 8, transitive). Gated test
groups are ``split == "test"``, not ``trivial``, not ``stress``. A group is correct when every labelled pair in it
gives its (kind, region) through ``outcome_of_jpeg(before, after, point, pointers)`` with B-2's pointers (the previous
executed pointer point, if any; then the click). A mixed-label group is scored once per label, on its pairs.

- exact agreement over gated test groups: >= 0.90;
- true ``none`` kept (every pair ``none``): >= 0.95, n >= 20 groups;
- changes read as ``none`` (any pair ``none``): <= 0.05, with >= 20 ``near`` groups among them;
- ``page`` recall (every pair ``page``): >= 0.95, n >= 8 groups.

Each bar is printed with its n and a 95 % Wilson interval. Report-only (printed, never gated): pair-weighted numbers
(continuity with A2), per source (replay, page a, page b), the stress kinds per kind, the trivial groups, the dev
split, test groups sharing one frame with a dev group (A2b-2's stated cost), and T-0134's byte-identity ``updated``.

**Re-derived, not trusted:** the frame clusters (``test_frame_groups_rederived``, mutation 21: sha-only groups) and
``seen`` against the 20104cc corpus and the A2 labels at ``LABELS_A2`` (``test_seen_rederived_against_20104cc``,
mutation 22: a seen group in test).

**A2b-8.** ``outcome_of_jpeg`` on every live corpus pair, three timed passes after one warm-up: p95 <= 20 ms, printed
with the 1-minute load average and n.

**The corpus** is ``logs/t0136-outcome-corpus-610f36f/`` (gitignored; the A1c-5 rebuild, manifest sha256 in the
task file's Rulings (A1c/A2b)), looked up under this checkout's ``logs/`` and then the main checkout's. Every row
skips, naming the paths tried, when it is absent. The rows are ``gpu``-marked only to keep them out of a path-only
run of ``tests/eval`` (T-0046); they spend no GPU time (H-T136-A2b-5: the reviewer and verifier run them on CPU).

Run (CPU)::

    uv run pytest tests/eval/test_t0136_calibration.py -q -m gpu -rA -s

Regime: corpus 610f36f (Playwright 1.64.0, headless Chromium 156, 1280x800, DSF 1; t0151 + t0134 replays, page a and
page b probes), labels by the A2/A2b test-authors (H-T136-1, A2b-1), n per split as printed; CPU, single process.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from functools import cache
from math import ceil, sqrt
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.eval]

try:
    _AO: ModuleType | None = importlib.import_module("local_astra.conductor.action_outcome")
except ModuleNotFoundError as _exc:  # only the module itself may be missing
    if _exc.name != "local_astra.conductor.action_outcome":
        raise
    _AO = None


def detector() -> ModuleType:
    assert _AO is not None, (
        "T-0136 A2-1: local_astra/conductor/action_outcome.py does not exist yet"
    )
    return _AO


REPO = Path(__file__).resolve().parents[2]
LABELS = REPO / "tests" / "fixtures" / "t0136" / "labels.jsonl"
CORPUS = "t0136-outcome-corpus-610f36f"
MANIFEST_SHA256 = "3c8c1c6ff5502b4d71c3e64b25799c7d64239f4c7df5a0c83478433fda48fbe0"
#: The A2 corpus and labels, for ``seen`` (A2b-2): the A2 tuner saw the 20104cc dev split.
CORPUS_A2 = "t0136-outcome-corpus-20104cc"
LABELS_A2 = "ad8268d"
JITTER_MAX = 8  # O-A1 (7)

EXACT_MIN = 0.90
NONE_KEPT_MIN = 0.95
CHANGE_LOST_MAX = 0.05  # A2b-6 as specified; measured 3/51 = 0.059 at c94f529
#: James, 2026-10-25 ~14:20, "accept despite the miss": change-lost recorded as met at <= 3 of >= 51 groups, for this
#: measurement only (not a general bar).
CHANGE_LOST_ACCEPTED = (3, 51)
PAGE_RECALL_MIN = 0.95
#: A2b-3 / A2b-6 minima, in independent gated test groups. Never lowered (Q-A2b-2).
NONE_N_MIN = 20
NEAR_N_MIN = 20
ELSEWHERE_N_MIN = 10
PAGE_N_MIN = 8
P95_MS_MAX = 20.0


def _candidates(name: str) -> list[Path]:
    out = [REPO / "logs" / name]
    proc = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        out.append(Path(proc.stdout.strip()).parent / "logs" / name)
    return out


def find_corpus(name: str) -> Path:
    tried = _candidates(name)
    for p in tried:
        if (p / "pairs.jsonl").is_file() and (p / "frames").is_dir():
            return p
    pytest.skip(f"T-0136 A2b eval: {name} is absent (tried {[str(p) for p in tried]})")


def corpus_dir() -> Path:
    return find_corpus(CORPUS)


def labels() -> list[dict[str, Any]]:
    return [json.loads(line) for line in LABELS.read_text().splitlines() if line.strip()]


def pairs(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for line in (root / "pairs.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["pair_id"]] = rec
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def pointers_of(rec: dict[str, Any]) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    point = (int(rec["point"][0]), int(rec["point"][1]))
    prev = rec["prev_point"]
    return point, (((int(prev[0]), int(prev[1])), point) if prev else (point,))


# =====================================================================================================
# The jitter rule, independently (full-resolution gray through PIL; never the detector's gray())
# =====================================================================================================


@cache
def full_gray(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.int16)


def jitter_equal(a: Path, b: Path) -> bool:
    ga, gb = full_gray(str(a)), full_gray(str(b))
    return ga.shape == gb.shape and int(np.abs(ga - gb).max()) <= JITTER_MAX


def clusters(nodes: list[tuple[str, Path]]) -> dict[str, str]:
    """Transitive closure of the jitter relation over ``(key, frame path)`` nodes; a cluster's id is its smallest
    key. Brute force over all pairs (a few hundred frames)."""
    keys = sorted(k for k, _ in nodes)
    path = dict(nodes)
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            if find(a) != find(b) and jitter_equal(path[a], path[b]):
                ra, rb = find(a), find(b)
                parent[max(ra, rb)] = min(ra, rb)
    out: dict[str, str] = {}
    members: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        members[find(k)].append(k)
    for ms in members.values():
        for k in ms:
            out[k] = min(ms)
    return out


# =====================================================================================================
# Fixture-level rows (no detector)
# =====================================================================================================


def test_corpus_is_the_recorded_run_and_matches_the_labels() -> None:
    """The corpus is the A1c-5 rebuild (manifest sha256), and every label names one of its pairs with the same point,
    previous point and frame kind."""
    root = corpus_dir()
    assert hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() == MANIFEST_SHA256
    corpus = pairs(root)
    labs = labels()
    assert {r["pair_id"] for r in labs} == set(corpus)
    for lab in labs:
        rec = corpus[lab["pair_id"]]
        assert lab["point"] == rec["point"] and lab["prev_point"] == rec["prev_point"], lab[
            "pair_id"
        ]
        assert lab["frame_kind"] == ("overview" if rec["frame_kind"] == "overview" else "live"), (
            lab["pair_id"]
        )


def test_frame_groups_rederived() -> None:
    """A1c-4 / A2b-1 (mutation 21, sha-only groups): clustering every corpus frame here, by the jitter rule, gives
    exactly ``frame_groups.json`` and every label's ``group``; the corpus holds real jitter merges (fewer clusters
    than frames), so sha-only groups differ."""
    root = corpus_dir()
    corpus = pairs(root)
    shas = sorted({rec[k] for rec in corpus.values() for k in ("before", "after")})
    cl = clusters([(s, root / "frames" / f"{s}.jpg") for s in shas])
    assert len(set(cl.values())) < len(shas), "premise: at least one jitter merge"
    assert cl == json.loads((root / "frame_groups.json").read_text())
    for lab in labels():
        rec = corpus[lab["pair_id"]]
        assert lab["group"] == f"{cl[rec['before']]}:{cl[rec['after']]}", lab["pair_id"]
        assert lab["trivial"] is (cl[rec["before"]] == cl[rec["after"]]), lab["pair_id"]
    print(f"\nT-0136 A2b frame groups: {len(shas)} frames -> {len(set(cl.values()))} clusters")


def test_seen_rederived_against_20104cc() -> None:
    """A2b-2 (mutation 22): a group is seen exactly when its (before, after) frames are jitter-equivalent (one
    closure over both corpora's frames) to those of a 20104cc pair the A2 labels at ``LABELS_A2`` put in dev, or it
    holds a committed-subset pair. Skips when 20104cc or the commit is absent."""
    root = corpus_dir()
    old_root = find_corpus(CORPUS_A2)
    proc = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{LABELS_A2}:tests/fixtures/t0136/labels.jsonl"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        pytest.skip(f"T-0136 A2b eval: commit {LABELS_A2} is absent ({proc.stderr.strip()})")
    old_labels = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    corpus, old_corpus = pairs(root), pairs(old_root)
    nodes = [
        (f"N{rec[k]}", root / "frames" / f"{rec[k]}.jpg")
        for rec in corpus.values()
        for k in ("before", "after")
    ]
    nodes += [(f"O{rec[k]}", old_root / "frames" / f"{rec[k]}.jpg") for rec in old_corpus.values() for k in ("before", "after")]  # fmt: skip
    cl = clusters(list(dict(nodes).items()))
    dev_old = {
        (cl[f"O{old_corpus[r['pair_id']]['before']}"], cl[f"O{old_corpus[r['pair_id']]['after']}"])
        for r in old_labels
        if r["split"] == "dev" and r["pair_id"] in old_corpus
    }
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for lab in labels():
        by_group[lab["group"]].append(lab)
    wrong = []
    for gid, rs in by_group.items():
        unions = {
            (cl[f"N{corpus[r['pair_id']]['before']}"], cl[f"N{corpus[r['pair_id']]['after']}"])
            for r in rs
        }
        want = bool(unions & dev_old) or any(r["subset"] for r in rs)
        if rs[0]["seen"] is not want:
            wrong.append((gid[:20], rs[0]["seen"], want))
    assert not wrong, wrong[:10]
    print(
        f"\nT-0136 A2b seen: {sum(rs[0]['seen'] for rs in by_group.values())}/{len(by_group)} groups"
    )


# =====================================================================================================
# Scoring (the detector)
# =====================================================================================================


_CACHE: dict[str, list[dict[str, Any]]] = {}


def source_of(lab: dict[str, Any]) -> str:
    return {None: "replay", "outcomes.html": "page a", "outcomes_b.html": "page b"}[
        lab["probe_page"]
    ]


def scored() -> list[dict[str, Any]]:
    """Every live labelled pair with the detector's (kind, region); computed once."""
    if "all" in _CACHE:
        return _CACHE["all"]
    root = corpus_dir()
    corpus = pairs(root)
    out = []
    for lab in labels():
        if lab["frame_kind"] != "live":
            continue
        rec = corpus[lab["pair_id"]]
        point, ptrs = pointers_of(rec)
        before = (root / "frames" / f"{rec['before']}.jpg").read_bytes()
        after = (root / "frames" / f"{rec['after']}.jpg").read_bytes()
        o = detector().outcome_of_jpeg(before, after, point, ptrs)
        out.append(
            {
                **lab,
                "want": (lab["label"], lab["region"]),
                "got": (o.kind, o.region),
                "bytes_differ": before != after,
                "before_sha": rec["before"],
                "after_sha": rec["after"],
            }
        )
    _CACHE["all"] = out
    return out


def group_units(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One unit per (group, label): a mixed-label group is scored once per label, on the pairs that carry it."""
    units: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        units[(r["group"], r["label"])].append(r)
    return [
        {
            "group": g,
            "label": label,
            "n_pairs": len(rs),
            "correct": all(r["got"] == r["want"] for r in rs),
            "all_none": all(r["got"][0] == "none" for r in rs),
            "any_none": any(r["got"][0] == "none" for r in rs),
            "all_page": all(r["got"][0] == "page" for r in rs),
            "misses": [
                (r["pair_id"], r["want"], r["got"], r["note"]) for r in rs if r["got"] != r["want"]
            ],
        }
        for (g, label), rs in sorted(units.items())
    ]


def gated(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [r for r in rows if r["split"] == split and not r["trivial"] and not r["stress"]]


def metrics(units: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    none = [u for u in units if u["label"] == "none"]
    change = [u for u in units if u["label"] != "none"]
    page = [u for u in units if u["label"] == "page"]
    return {
        "exact": (sum(u["correct"] for u in units), len(units)),
        "none_kept": (sum(u["all_none"] for u in none), len(none)),
        "change_lost": (sum(u["any_none"] for u in change), len(change)),
        "page_recall": (sum(u["all_page"] for u in page), len(page)),
    }


def show(title: str, m: dict[str, tuple[int, int]]) -> None:
    print(f"\nT-0136 A2b-6 {title}:")
    for name, (k, n) in m.items():
        lo, hi = wilson(k, n)
        rate = k / n if n else float("nan")
        print(f"  {name}: {k}/{n} = {rate:.3f}  Wilson 95% [{lo:.3f}, {hi:.3f}]")


def gated_test_units() -> list[dict[str, Any]]:
    units = group_units(gated(scored(), "test"))
    show(f"test split, gated groups (n={len(units)})", metrics(units))
    counts = Counter(u["label"] for u in units)
    print(f"  groups per class: {dict(sorted(counts.items()))}")
    for u in units:
        for miss in u["misses"]:
            print(
                f"  miss {u['group'][:12]}.. {miss[0]}: label {miss[1]} got {miss[2]} ({miss[3]})"
            )
    return units


def test_test_split_minimum_groups() -> None:
    """A2b-3 on the corpus the eval scores (labels only, no detector): the gated test groups reach the minima."""
    rows = gated([r for r in labels() if r["frame_kind"] == "live"], "test")
    # one unit per (group, label), counted straight from the labels (no detector fields; reviewer F2)
    counts = Counter(label for _group, label in {(r["group"], r["label"]) for r in rows})
    want = {
        "none": NONE_N_MIN,
        "near": NEAR_N_MIN,
        "elsewhere": ELSEWHERE_N_MIN,
        "page": PAGE_N_MIN,
    }
    short = {k: (counts[k], m) for k, m in want.items() if counts[k] < m}
    assert not short, f"T-0136 A2b-3 / Q-A2b-2: short {short}; stop for an A1d, never lower n"


def test_exact_agreement_on_the_test_split() -> None:
    k, n = metrics(gated_test_units())["exact"]
    assert n > 0 and k / n >= EXACT_MIN, (
        f"T-0136 A2b-6: exact agreement {k}/{n} groups < {EXACT_MIN}"
    )


def test_true_none_kept_on_the_test_split() -> None:
    k, n = metrics(gated_test_units())["none_kept"]
    assert n >= NONE_N_MIN, f"T-0136 A2b-6: {n} gated none groups on test (< {NONE_N_MIN})"
    assert k / n >= NONE_KEPT_MIN, f"T-0136 A2b-6: true none kept {k}/{n} groups < {NONE_KEPT_MIN}"


def test_true_changes_read_as_none_on_the_test_split() -> None:
    """A2b-6's change-lost bar (<= 0.05) measured 3/51 at c94f529, one group over. James, 2026-10-25 ~14:20
    ("accept despite the miss", the task file's Rulings (A1c/A2b)): the bar is recorded as met at <= 3/51 for this
    measurement only. The row pins that measured result -- at most 3 lost of at least 51 change groups -- and is
    not a general 0.06 bar; any further lost group, or fewer groups, is red."""
    units = gated_test_units()
    k, n = metrics(units)["change_lost"]
    lo, hi = wilson(k, n)
    print(
        f"\nT-0136 A2b-6 change_lost: {k}/{n}  Wilson 95% [{lo:.3f}, {hi:.3f}] (accepted: <= 3/51)"
    )
    near_n = sum(u["label"] == "near" for u in units)
    assert near_n >= NEAR_N_MIN, (
        f"T-0136 A2b-6: {near_n} gated near groups on test (< {NEAR_N_MIN})"
    )
    k_max, n_min = CHANGE_LOST_ACCEPTED
    assert n >= n_min and k <= k_max, (
        f"T-0136 A2b-6: changes read as none {k}/{n} groups; accepted only at <= {k_max} of >= {n_min} "
        "(James, 2026-10-25 ~14:20, 'accept despite the miss': this measurement only, not a general bar)"
    )


def test_page_recall_on_the_test_split() -> None:
    k, n = metrics(gated_test_units())["page_recall"]
    assert n >= PAGE_N_MIN, f"T-0136 A2b-6: {n} gated page groups on test (< {PAGE_N_MIN})"
    assert k / n >= PAGE_RECALL_MIN, f"T-0136 A2b-6: page recall {k}/{n} groups < {PAGE_RECALL_MIN}"


# =====================================================================================================
# Report-only
# =====================================================================================================


def pair_metrics(rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    none = [r for r in rows if r["label"] == "none"]
    change = [r for r in rows if r["label"] != "none"]
    page = [r for r in rows if r["label"] == "page"]
    return {
        "exact": (sum(r["got"] == r["want"] for r in rows), len(rows)),
        "none_kept": (sum(r["got"][0] == "none" for r in none), len(none)),
        "change_lost": (sum(r["got"][0] == "none" for r in change), len(change)),
        "page_recall": (sum(r["got"][0] == "page" for r in page), len(page)),
    }


def test_pair_weighted_and_per_source_report() -> None:
    """Report-only: pair-weighted numbers on the gated test pairs (continuity with A2), and per-group numbers per
    source (replay, page a, page b)."""
    rows = gated(scored(), "test")
    show(f"test split, pair-weighted (n={len(rows)} pairs)", pair_metrics(rows))
    for src in ("replay", "page a", "page b"):
        sub = [r for r in rows if source_of(r) == src]
        show(f"test split, {src}, per group", metrics(group_units(sub)))
    assert rows, "no gated test pairs to report on"


def test_stress_report() -> None:
    """Report-only (H-T136-A2b-2): the stress kinds, per kind, both splits (known blind spots of rule 3')."""
    rows = [r for r in scored() if r["stress"]]
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    print("\nT-0136 A2b-6 stress kinds (report-only):")
    for kind, rs in sorted(by_kind.items()):
        ok = sum(r["got"] == r["want"] for r in rs)
        print(f"  {kind}: {ok}/{len(rs)} pairs as labelled; got {Counter(r['got'][0] for r in rs)}")
    assert rows, "no stress pairs"


def test_trivial_report() -> None:
    """Report-only: the trivial groups (before and after in one cluster), both splits."""
    units = group_units([r for r in scored() if r["trivial"]])
    k = sum(u["correct"] for u in units)
    print(f"\nT-0136 A2b-6 trivial groups (report-only): {k}/{len(units)} correct")
    for u in units:
        for miss in u["misses"]:
            print(f"  miss {miss[0]}: label {miss[1]} got {miss[2]} ({miss[3]})")
    assert units, "no trivial groups"


def test_dev_split_report() -> None:
    """Report-only: the dev split the implementer tunes on, per group; and the stated cost of A2b-2 -- gated test
    groups that share one frame (never both) with a dev group."""
    dev = gated(scored(), "dev")
    show(f"dev split, gated groups (n={len(group_units(dev))})", metrics(group_units(dev)))
    dev_frames = {
        r[k] for r in scored() if r["split"] == "dev" for k in ("before_sha", "after_sha")
    }
    test_groups = {r["group"]: r for r in gated(scored(), "test")}
    sharing = sum(
        bool({r["before_sha"], r["after_sha"]} & dev_frames) for r in test_groups.values()
    )
    print(f"  gated test groups sharing a frame with a dev pair: {sharing}/{len(test_groups)}")
    assert dev, "no gated dev groups"


def test_byte_identity_report() -> None:
    """Report-only (A2-5): T-0134's byte-identity ``updated`` on the same pairs -- how many true no-ops it would call
    changed (the cua lesson: exact hashing mislabelled 51 % of true no-ops), per pair and per group."""
    rows = scored()
    none = [r for r in rows if r["label"] == "none"]
    wrong = sum(r["bytes_differ"] for r in none)
    lo, hi = wilson(wrong, len(none))
    g_wrong = {r["group"] for r in none if r["bytes_differ"]}
    g_all = {r["group"] for r in none}
    print(
        f"\nT-0136 A2b-6 byte identity: {wrong}/{len(none)} true-none pairs would read 'updated' "
        f"(Wilson 95% [{lo:.3f}, {hi:.3f}]); {len(g_wrong)}/{len(g_all)} true-none groups"
    )
    assert none, "no true-none pairs to report on"


def test_cost_p95() -> None:
    """A2b-8: ``outcome_of_jpeg`` p95 <= 20 ms over every live corpus pair (1 warm-up + 3 timed passes)."""
    root = corpus_dir()
    corpus = pairs(root)
    work = []
    for lab in labels():
        if lab["frame_kind"] != "live":
            continue
        rec = corpus[lab["pair_id"]]
        point, ptrs = pointers_of(rec)
        work.append(
            (
                (root / "frames" / f"{rec['before']}.jpg").read_bytes(),
                (root / "frames" / f"{rec['after']}.jpg").read_bytes(),
                point,
                ptrs,
            )
        )
    ao = detector()
    for before, after, point, ptrs in work:
        ao.outcome_of_jpeg(before, after, point, ptrs)
    times = []
    for _ in range(3):
        for before, after, point, ptrs in work:
            t0 = time.perf_counter()
            ao.outcome_of_jpeg(before, after, point, ptrs)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p95 = times[ceil(0.95 * len(times)) - 1]
    load1 = os.getloadavg()[0]
    print(
        f"\nT-0136 A2b-8: outcome_of_jpeg p95 {p95:.2f} ms, median {times[len(times) // 2]:.2f} ms, "
        f"n={len(times)} ({len(work)} pairs x 3), 1-min load {load1:.2f}, "
        f"DRAFT_SCALE={getattr(ao, 'DRAFT_SCALE', '?')}, UNDERLINE_ROWS={getattr(ao, 'UNDERLINE_ROWS', '?')}"
    )
    assert p95 <= P95_MS_MAX, (
        f"T-0136 A2b-8: p95 {p95:.2f} ms > {P95_MS_MAX} ms (1-min load {load1:.2f})"
    )
