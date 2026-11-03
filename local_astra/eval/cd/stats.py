"""Statistics for the CD evaluation (T-0070 criterion 7).

Wilson 95 % intervals and the exact McNemar test are diffgemma-fa's (``_vendor/stats.py``, unmodified).
``paired`` pairs two arms' per-record outcomes **by record id** (a record missing from either arm is an
error, never dropped), counts the discordant pairs, and reports the "both parsed" restriction beside
the full table. Bonferroni is over the pre-registered family: 5 pairs x 2 tasks = 10.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from local_astra.eval.cd._vendor import stats as _stats

__all__ = [
    "N_COMPARISONS",
    "PREREGISTERED_PAIRS",
    "Paired",
    "bonferroni",
    "fmt_rate",
    "mcnemar_exact",
    "paired",
    "wilson",
]

#: (A, B) pairs compared on BFCL exact call and Sudoku solved.
PREREGISTERED_PAIRS: tuple[tuple[str, str], ...] = (
    ("perstep_mar", "end_projection"),
    ("perstep_mar", "unconstrained"),
    ("end_projection", "unconstrained"),
    ("perstep_mar", "perstep_mf"),
    ("perstep_mar_sc", "perstep_mar"),
    # T-0076 Q2, fixed before T-0071c touches a test cut; the SC twin gets its Wilson CI only
    ("perstep_mar_qargmax", "perstep_mar"),
)
N_TASKS = 2
N_COMPARISONS = len(PREREGISTERED_PAIRS) * N_TASKS


def wilson(k: int, n: int, *, z: float = 1.96) -> tuple[float, float]:
    lo, hi = _stats.wilson(k, n, z=z)
    return float(lo), float(hi)


def mcnemar_exact(b: int, c: int) -> float:
    return float(_stats.mcnemar_exact(b, c))


def bonferroni(p: float, n_comparisons: int = N_COMPARISONS) -> float:
    return float(_stats.bonferroni(p, n_comparisons))


def fmt_rate(k: int, n: int) -> str:
    """``0.923 [0.865, 0.958]``: a rate is never printed without its interval."""
    return str(_stats.fmt_rate(k, n))


@dataclasses.dataclass(frozen=True)
class Paired:
    """``b``: A succeeded and B failed; ``c``: B succeeded and A failed."""

    n: int
    b: int
    c: int
    p: float
    p_bonferroni: float
    both_parsed: Paired | None = None


def _table(
    a: Mapping[str, Mapping[str, Any]], bb: Mapping[str, Mapping[str, Any]], ids: list[str]
) -> Paired:
    b = sum(1 for i in ids if bool(a[i]["ok"]) and not bool(bb[i]["ok"]))
    c = sum(1 for i in ids if bool(bb[i]["ok"]) and not bool(a[i]["ok"]))
    p = mcnemar_exact(b, c)
    return Paired(n=len(ids), b=b, c=c, p=p, p_bonferroni=bonferroni(p, N_COMPARISONS))


def paired(a: Mapping[str, Mapping[str, Any]], b: Mapping[str, Mapping[str, Any]]) -> Paired:
    """Exact McNemar on per-record ``{"ok", "parsed"}`` outcomes of arms A and B, keyed by record id."""
    if set(a) != set(b):
        only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
        raise ValueError(
            f"unpaired records: {len(only_a)} only in A {only_a[:5]}, {len(only_b)} only in B {only_b[:5]}"
        )
    ids = sorted(a)
    full = _table(a, b, ids)
    both = [i for i in ids if bool(a[i]["parsed"]) and bool(b[i]["parsed"])]
    return dataclasses.replace(full, both_parsed=_table(a, b, both))
