#!/usr/bin/env python3
"""T-0136 A2b-5: the dev-only re-tune of the outcome detector's constants (docs/tasks/T-0136.md, amendment A1c/A2b).

    uv run python scripts/t0136_outcome_sweep.py [--corpus DIR] [--set NAME=VALUE ...] [--cost]

**Dev only.** The script reads ``tests/fixtures/t0136/labels.jsonl`` and keeps the records with ``split == "dev"``
as it reads them; test records are dropped at the line they are parsed and never held. There is no option to read
another split, and the scorer (``score``) raises ``TestRecordReached`` if any record that is not dev reaches it.
The corpus is ``logs/t0136-outcome-corpus-610f36f/`` (this checkout's ``logs/`` first, then the main checkout's,
or ``--corpus``); only the frames of dev pairs are read.

**What is scored** (A2b-6's metrics, on dev): live pairs, grouped by ``group``, neither ``trivial`` nor ``stress``;
one unit per (group, label); a unit is correct when every pair in it gives its (kind, region) through the detector
with B-2's pointers (the previous point, if any, then the click). The four bars:

- exact agreement >= 0.90;
- true ``none`` kept (every pair ``none``) >= 0.95;
- changes read as ``none`` (any pair ``none``) <= 0.05;
- ``page`` recall (every pair ``page``) >= 0.95.

Each is printed with its n and a 95 % Wilson interval. Stress and trivial dev groups are printed, report-only.

**Constants.** The centre is the module's committed constants, overridable with ``--set NAME=VALUE`` for exploration.
The robustness table (A2b-5, F4) varies one constant at a time over its sweep:

- continuous constants (``NOISE_DELTA``, ``HOVER_DELTA``, ``HOVER_HALF_W``, ``HOVER_HALF_H``, ``PAGE_FRACTION``):
  the dev band, and whether the bars hold over the centre +-15 % or +-1 sweep step, whichever is larger;
- discrete constants (``DRAFT_SCALE``, ``CELL_PX``, ``MIN_CELL_PIXELS``, ``UNDERLINE_ROWS``): the dev band, and the
  dev groups that flip at each bound;
- a band that reaches both ends of its sweep is reported "unconstrained by dev".

``--cost`` times ``outcome_of_jpeg`` on the dev corpus pairs (one warm-up and three timed passes; A2b-8) and prints
the p95 with the 1-minute load.

The detector is used as committed: constants are set on the module for a call and restored after it; the rule
shape is the module's own. CPU only, single process, no network.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from math import ceil, floor, sqrt
from pathlib import Path
from typing import Any

import numpy as np

from local_astra.conductor import action_outcome as ao

REPO = Path(__file__).resolve().parents[1]
LABELS = REPO / "tests" / "fixtures" / "t0136" / "labels.jsonl"
CORPUS = "t0136-outcome-corpus-610f36f"
#: The only split this script reads (A2b-5). Deliberately a constant, not an option.
DEV = "dev"

EXACT_MIN = 0.90
NONE_KEPT_MIN = 0.95
CHANGE_LOST_MAX = 0.05
PAGE_RECALL_MIN = 0.95

CONTINUOUS = ("NOISE_DELTA", "HOVER_DELTA", "HOVER_HALF_W", "HOVER_HALF_H", "PAGE_FRACTION")
DISCRETE = ("DRAFT_SCALE", "CELL_PX", "MIN_CELL_PIXELS", "UNDERLINE_ROWS")
SWEEP: dict[str, list[float]] = {
    "NOISE_DELTA": list(range(8, 81, 4)),
    "HOVER_DELTA": list(range(16, 177, 8)),
    "HOVER_HALF_W": list(range(16, 417, 16)),
    "HOVER_HALF_H": list(range(8, 97, 4)),
    "PAGE_FRACTION": [round(0.01 * k, 2) for k in range(2, 31)],
    "DRAFT_SCALE": [1, 2],
    "CELL_PX": [4, 6, 8, 10, 12, 16],
    "MIN_CELL_PIXELS": list(range(1, 25)),
    "UNDERLINE_ROWS": [1, 2, 3],
}
NAMES = CONTINUOUS + DISCRETE
#: Constants whose sweep is their whole domain (A2-3: DRAFT_SCALE in {1, 2}; A2b-4: UNDERLINE_ROWS in {1, 2, 3}).
WHOLE_DOMAIN = frozenset({"DRAFT_SCALE", "UNDERLINE_ROWS"})


class TestRecordReached(RuntimeError):
    """A record outside the dev split reached the scorer (A2b-5: the script must fail)."""


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------


def load_dev_labels(path: Path = LABELS) -> list[dict[str, Any]]:
    """The dev records of the labels file; every other record is discarded as it is parsed."""
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("split") != DEV:
            continue
        out.append(rec)
    return out


def find_corpus(explicit: str | None) -> Path:
    tried = [Path(explicit)] if explicit else [REPO / "logs" / CORPUS]
    if not explicit:
        proc = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            tried.append(Path(proc.stdout.strip()).parent / "logs" / CORPUS)
    for p in tried:
        if (p / "pairs.jsonl").is_file() and (p / "frames").is_dir():
            return p
    raise SystemExit(f"corpus not found (tried {[str(p) for p in tried]})")


def corpus_pairs(root: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for line in (root / "pairs.jsonl").read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec["pair_id"] in wanted:
                out[rec["pair_id"]] = rec
    missing = wanted - set(out)
    if missing:
        raise SystemExit(
            f"{len(missing)} dev pairs are not in the corpus, e.g. {sorted(missing)[:3]}"
        )
    return out


class Frames:
    """Frame bytes, declared sizes and gray decodes (per DRAFT_SCALE), each computed once."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._bytes: dict[str, bytes] = {}
        self._gray: dict[tuple[str, int], np.ndarray] = {}

    def raw(self, sha: str) -> bytes:
        if sha not in self._bytes:
            self._bytes[sha] = (self.root / "frames" / f"{sha}.jpg").read_bytes()
        return self._bytes[sha]

    def gray(self, sha: str) -> np.ndarray:
        key = (sha, int(ao.DRAFT_SCALE))
        if key not in self._gray:
            self._gray[key] = ao.gray(self.raw(sha))
        return self._gray[key]


# ---------------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------------


@contextmanager
def constants(values: dict[str, float]) -> Iterator[None]:
    saved = {k: getattr(ao, k) for k in values}
    try:
        for k, v in values.items():
            setattr(ao, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(ao, k, v)


def pointers_of(rec: dict[str, Any]) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    point = (int(rec["point"][0]), int(rec["point"][1]))
    prev = rec["prev_point"]
    return point, (((int(prev[0]), int(prev[1])), point) if prev else (point,))


def detect(frames: Frames, pair: dict[str, Any]) -> tuple[str, str | None]:
    """``outcome_of_jpeg`` on the pair, with the gray decode cached: the declared-size test is the module's own
    (``_declared_size``), then ``outcome_of`` on the cached arrays."""
    point, ptrs = pointers_of(pair)
    b, a = frames.raw(pair["before"]), frames.raw(pair["after"])
    db, da = ao._declared_size(b), ao._declared_size(a)
    if db is not None and da is not None and db != da:
        o = ao.outcome_of_jpeg(b, a, point, ptrs)
    else:
        o = ao.outcome_of(frames.gray(pair["before"]), frames.gray(pair["after"]), point, ptrs)
    return (o.kind, o.region)


def score(
    recs: list[dict[str, Any]], corpus: dict[str, dict[str, Any]], frames: Frames
) -> list[dict[str, Any]]:
    """Every record's detector label; refuses any record that is not dev."""
    out = []
    for r in recs:
        if r.get("split") != DEV:
            raise TestRecordReached(
                f"T-0136 A2b-5: record {r.get('pair_id')} (split {r.get('split')!r})"
            )
        out.append(
            {**r, "want": (r["label"], r["region"]), "got": detect(frames, corpus[r["pair_id"]])}
        )
    return out


def units_of(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[(r["group"], r["label"])].append(r)
    return [
        {
            "group": g,
            "label": label,
            "correct": all(r["got"] == r["want"] for r in rs),
            "all_none": all(r["got"][0] == "none" for r in rs),
            "any_none": any(r["got"][0] == "none" for r in rs),
            "all_page": all(r["got"][0] == "page" for r in rs),
            "misses": [
                (r["pair_id"], r["kind"], r["want"], r["got"], r["note"])
                for r in rs
                if r["got"] != r["want"]
            ],
        }
        for (g, label), rs in sorted(by.items())
    ]


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


def failing_bars(m: dict[str, tuple[int, int]]) -> list[str]:
    def rate(name: str) -> float:
        k, n = m[name]
        return k / n if n else float("nan")

    bad = []
    if not rate("exact") >= EXACT_MIN:
        bad.append("exact")
    if not rate("none_kept") >= NONE_KEPT_MIN:
        bad.append("none_kept")
    if not rate("change_lost") <= CHANGE_LOST_MAX:
        bad.append("change_lost")
    if not rate("page_recall") >= PAGE_RECALL_MIN:
        bad.append("page_recall")
    return bad


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


class Scorer:
    """Gated dev units under a set of constants, cached per constant set."""

    def __init__(
        self, recs: list[dict[str, Any]], corpus: dict[str, dict[str, Any]], frames: Frames
    ) -> None:
        self.gated = [
            r for r in recs if r["frame_kind"] == "live" and not r["trivial"] and not r["stress"]
        ]
        self.corpus = corpus
        self.frames = frames
        self._cache: dict[tuple[tuple[str, float], ...], list[dict[str, Any]]] = {}

    def units(self, values: dict[str, float]) -> list[dict[str, Any]]:
        key = tuple(sorted(values.items()))
        if key not in self._cache:
            with constants(values):
                self._cache[key] = units_of(score(self.gated, self.corpus, self.frames))
        return self._cache[key]


def legal(values: dict[str, float]) -> bool:
    """The module's own constraints (A2-3, A2b-4): noise below hover below 255, a cell wider than an underline."""
    return (
        0 <= values["NOISE_DELTA"] < values["HOVER_DELTA"] < 255
        and values["CELL_PX"] > values["UNDERLINE_ROWS"]
        and values["MIN_CELL_PIXELS"] <= values["CELL_PX"] ** 2
    )


# ---------------------------------------------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------------------------------------------


def show(title: str, m: dict[str, tuple[int, int]]) -> None:
    print(f"{title}")
    for name, (k, n) in m.items():
        lo, hi = wilson(k, n)
        rate = k / n if n else float("nan")
        print(f"  {name}: {k}/{n} = {rate:.3f}  Wilson 95% [{lo:.3f}, {hi:.3f}]")


def flips(base: list[dict[str, Any]], other: list[dict[str, Any]]) -> list[str]:
    b = {(u["group"], u["label"]): u["correct"] for u in base}
    out = []
    for u in other:
        key = (u["group"], u["label"])
        if b.get(key) is not None and b[key] != u["correct"]:
            miss = u["misses"][0] if u["misses"] else None
            what = f"{miss[1]} {miss[0]} want {miss[2]} got {miss[3]}" if miss else "now correct"
            out.append(f"{u['label']} {u['group'][:8]}..{u['group'][65:73]} ({what})")
    return out


def fmt(name: str, v: float) -> str:
    return f"{v:.2f}" if name == "PAGE_FRACTION" else str(int(v))


def robustness(scorer: Scorer, centre: dict[str, float]) -> bool:
    base = scorer.units(centre)
    ok_all = True
    print("\nRobustness (one constant at a time, others at the centre; dev gated groups):")
    for name in NAMES:
        grid = sorted(set(SWEEP[name]) | {centre[name]})
        passing: dict[float, bool] = {}
        for v in grid:
            vals = {**centre, name: v}
            passing[v] = legal(vals) and not failing_bars(metrics(scorer.units(vals)))
        i = grid.index(centre[name])
        if not passing[grid[i]]:
            print(f"  {name} = {fmt(name, centre[name])}: the centre itself fails the dev bars")
            ok_all = False
            continue
        lo_i, hi_i = i, i
        while lo_i > 0 and passing[grid[lo_i - 1]]:
            lo_i -= 1
        while hi_i < len(grid) - 1 and passing[grid[hi_i + 1]]:
            hi_i += 1
        band = f"[{fmt(name, grid[lo_i])}, {fmt(name, grid[hi_i])}]"
        sides = []
        for side, j in (("lower", lo_i - 1), ("upper", hi_i + 1)):
            if 0 <= j < len(grid):
                vals = {**centre, name: grid[j]}
                if not legal(vals):
                    sides.append(
                        f"{side} bound: {fmt(name, grid[j])} is illegal (module constraint)"
                    )
                    continue
                us = scorer.units(vals)
                bars = failing_bars(metrics(us))
                fl = flips(base, us)
                sides.append(f"{side} bound: at {fmt(name, grid[j])} fails {bars}; flips {fl[:3]}")
            elif name in WHOLE_DOMAIN:
                sides.append(
                    f"{side} bound: the end of the domain {SWEEP[name]} (no value beyond it)"
                )
            else:
                sides.append(f"{side} bound: unbounded over the sweep")
        unconstrained = lo_i == 0 and hi_i == len(grid) - 1
        line = f"  {name} = {fmt(name, centre[name])}: dev band {band}"
        if unconstrained:
            line += "  -- unconstrained by dev"
        print(line)
        for s in sides:
            print(f"      {s}")
        if name in CONTINUOUS:
            c = centre[name]
            pos = SWEEP[name].index(c) if c in SWEEP[name] else None
            step = (
                (SWEEP[name][1] - SWEEP[name][0])
                if pos is None or pos == 0
                else (c - SWEEP[name][pos - 1])
            )
            lo, hi = min(c * 0.85, c - step), max(c * 1.15, c + step)
            if name != "PAGE_FRACTION":
                lo, hi = floor(lo), ceil(hi)  # integer constants: round the range outwards
            probe = [lo, hi] + [v for v in grid if lo <= v <= hi]
            bad = [
                v
                for v in probe
                if not (
                    legal({**centre, name: v})
                    and not failing_bars(metrics(scorer.units({**centre, name: v})))
                )
            ]
            verdict = "holds" if not bad else f"FAILS at {[fmt(name, v) for v in sorted(set(bad))]}"
            shown = f"[{lo:.4f}, {hi:.4f}]" if name == "PAGE_FRACTION" else f"[{lo}, {hi}]"
            print(f"      +-15 % / +-1 step range {shown}: {verdict}")
            ok_all = ok_all and not bad
        elif lo_i == hi_i:
            # A2b-5: a one-value band is allowed only when both bounds are named dev groups; a side that is the end
            # of the constant's whole domain has no value to name and is printed as such.
            if all("fails" in s for s in sides):
                print("      a one-value band; both bounds are named dev groups")
            elif all("fails" in s or "end of the domain" in s for s in sides):
                print(
                    "      a one-value band; one bound is a named dev group, the other the end of the domain"
                )
            else:
                print("      a one-value band whose bounds are not both named dev groups")
                ok_all = False
    return ok_all


def cost(recs: list[dict[str, Any]], corpus: dict[str, dict[str, Any]], frames: Frames) -> None:
    work = []
    for r in recs:
        if r["split"] != DEV:
            raise TestRecordReached(r["pair_id"])
        if r["frame_kind"] != "live":
            continue
        pair = corpus[r["pair_id"]]
        point, ptrs = pointers_of(pair)
        work.append((frames.raw(pair["before"]), frames.raw(pair["after"]), point, ptrs))
    for b, a, p, q in work:
        ao.outcome_of_jpeg(b, a, p, q)
    times = []
    for _ in range(3):
        for b, a, p, q in work:
            t0 = time.perf_counter()
            ao.outcome_of_jpeg(b, a, p, q)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    p95 = times[ceil(0.95 * len(times)) - 1]
    print(
        f"\nA2b-8 cost: outcome_of_jpeg p95 {p95:.2f} ms, median {times[len(times) // 2]:.2f} ms, "
        f"n={len(times)} ({len(work)} dev live pairs x 3), 1-min load {os.getloadavg()[0]:.2f}"
    )


def parse_set(items: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or name not in NAMES:
            raise SystemExit(f"--set NAME=VALUE with NAME in {NAMES}; got {item!r}")
        out[name] = float(value) if name == "PAGE_FRACTION" else int(value)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    ap.add_argument(
        "--corpus", help=f"corpus dir (default: logs/{CORPUS} here, then in the main checkout)"
    )
    ap.add_argument(
        "--set", action="append", default=[], metavar="NAME=VALUE", help="override a constant"
    )
    ap.add_argument(
        "--cost", action="store_true", help="time outcome_of_jpeg on the dev pairs (A2b-8)"
    )
    ap.add_argument("--no-robustness", action="store_true", help="score the centre only")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    root = find_corpus(args.corpus)
    recs = load_dev_labels()
    corpus = corpus_pairs(root, {r["pair_id"] for r in recs})
    frames = Frames(root)
    scorer = Scorer(recs, corpus, frames)
    centre: dict[str, float] = {k: getattr(ao, k) for k in NAMES}
    centre.update(parse_set(args.set))

    print(
        f"T-0136 A2b-5 dev sweep: corpus {root}, labels {LABELS.relative_to(REPO)}, split {DEV!r} only"
    )
    print(
        f"  {len(recs)} dev records; {len(scorer.gated)} gated live pairs (non-trivial, non-stress)"
    )
    print("  centre: " + ", ".join(f"{k}={fmt(k, centre[k])}" for k in NAMES))
    if not legal(centre):
        raise SystemExit("the centre violates the module's constraints")
    units = scorer.units(centre)
    print(
        f"  gated dev groups per class: {dict(sorted(Counter(u['label'] for u in units).items()))}"
    )
    m = metrics(units)
    show(f"\nDev split, gated groups (n={len(units)}):", m)
    bad = failing_bars(m)
    print(f"  bars: {'all met' if not bad else 'FAIL ' + ', '.join(bad)}")
    for u in units:
        for miss in u["misses"]:
            print(
                f"  miss {u['group'][:8]}.. {miss[0]} [{miss[1]}]: want {miss[2]} got {miss[3]} ({miss[4]})"
            )

    with constants(centre):
        for title, pick in (
            ("stress", lambda r: r["stress"]),
            ("trivial", lambda r: r["trivial"] and not r["stress"]),
        ):
            rows = score([r for r in recs if r["frame_kind"] == "live" and pick(r)], corpus, frames)
            ok = sum(r["got"] == r["want"] for r in rows)
            print(f"\nReport-only, dev {title} pairs: {ok}/{len(rows)} as labelled")
            by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for r in rows:
                by_kind[r["kind"]].append(r)
            for kind, rs in sorted(by_kind.items()):
                print(
                    f"  {kind}: {sum(r['got'] == r['want'] for r in rs)}/{len(rs)}; got {dict(Counter(r['got'][0] for r in rs))}"
                )

    robust = True
    if not args.no_robustness:
        robust = robustness(scorer, centre)
        print(f"\nRobustness: {'all bands meet A2b-5' if robust else 'NOT met (see above)'}")
    if args.cost:
        with constants(centre):
            cost(recs, corpus, frames)
    print(f"\nwall {time.perf_counter() - t_start:.1f} s, 1-min load {os.getloadavg()[0]:.2f}")
    return 0 if not bad and robust else 1


if __name__ == "__main__":
    sys.exit(main())
