"""Brute-force oracles over strings (SPEC §2.2–2.4, §6.1). Written from the spec; see MANIFEST.md.

``Z = Σ_{x ∈ V^Leff} N(x) · Π_i p_i(x_i)`` where ``N(x)`` is the number of edge paths that read ``x``
from a start state in ``A`` and end in a state of ``B`` (the terminal factor's support). ``N`` is an
integer DP per string, so NFA parallel/overlapping edges carry their multiplicity exactly as SPEC
§2.2 defines. Positions ``≥ Leff`` are identity (unscored) and are not enumerated.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Iterable, Sequence
from fractions import Fraction

NEG_INF = float("-inf")


@dataclasses.dataclass(frozen=True)
class Toy:
    """An automaton in class-table form (SPEC §4.4 layers 1–2).

    ``classes[c] = (stored, is_neg)``: the true member set is ``stored`` when ``is_neg`` is False and
    ``V \\ stored`` (the complement ``N_c`` is what is stored) when it is True.
    ``edges[e] = (src, dst, class_id)``; parallel edges are allowed and meaningful.
    """

    n_states: int
    vocab_size: int
    classes: tuple[tuple[frozenset[int], bool], ...]
    edges: tuple[tuple[int, int, int], ...]
    finals: frozenset[int]

    def members(self, c: int) -> frozenset[int]:
        stored, neg = self.classes[c]
        return frozenset(range(self.vocab_size)) - stored if neg else stored

    def arcs_for_token(self) -> list[list[tuple[int, int]]]:
        """``arcs[v]`` = every (src, dst) edge whose label admits ``v``, with multiplicity."""
        mem = [self.members(c) for c in range(len(self.classes))]
        return [[(s, d) for (s, d, c) in self.edges if v in mem[c]] for v in range(self.vocab_size)]

    def is_dfa(self) -> bool:
        """Deterministic and unit multiplicity (the ``is_dfa`` fast-path gate)."""
        seen: dict[tuple[int, int], int] = {}
        for s, d, c in self.edges:
            for v in self.members(c):
                if (s, v) in seen:
                    return False
                seen[(s, v)] = d
        return True


def path_count(
    x: Sequence[int],
    arcs: list[list[tuple[int, int]]],
    n: int,
    start: Iterable[int],
    end: Iterable[int],
) -> int:
    cnt = [0] * n
    for s in start:
        cnt[s] = 1
    for v in x:
        nxt = [0] * n
        for s, d in arcs[v]:
            if cnt[s]:
                nxt[d] += cnt[s]
        cnt = nxt
    return sum(cnt[s] for s in end)


def _strings(vocab: int, length: int) -> Iterable[tuple[int, ...]]:
    return itertools.product(range(vocab), repeat=length)


def enumerate_fraction(
    p: Sequence[Sequence[Fraction]], toy: Toy, start: Iterable[int], end: Iterable[int]
) -> tuple[Fraction, list[list[Fraction]]]:
    """Exact ``(Z, q)`` for rational ``p`` (``len(p)`` = Leff positions)."""
    start, end = list(start), list(end)
    arcs = toy.arcs_for_token()
    L, V = len(p), toy.vocab_size
    Z = Fraction(0)
    acc = [[Fraction(0)] * V for _ in range(L)]
    for x in _strings(V, L):
        n = path_count(x, arcs, toy.n_states, start, end)
        if not n:
            continue
        w = Fraction(n)
        for i, v in enumerate(x):
            w *= p[i][v]
            if not w:
                break
        if not w:
            continue
        Z += w
        for i, v in enumerate(x):
            acc[i][v] += w
    q = [[a / Z for a in row] for row in acc] if Z else acc
    return Z, q


def _lse(xs: Sequence[float]) -> float:
    m = max(xs, default=NEG_INF)
    if m == NEG_INF:
        return NEG_INF
    return m + math.log(math.fsum(math.exp(x - m) for x in xs))


@dataclasses.dataclass(frozen=True)
class LogOracle:
    log_z: float
    q: list[list[float]]  # [Leff][V]
    entropy: list[float]  # [Leff], nats
    support: set[tuple[int, ...]]  # strings with N(x)·w(x) > 0


def log_softmax64(logits: Sequence[Sequence[float]]) -> list[list[float]]:
    out = []
    for row in logits:
        row = [float(v) for v in row]
        z = _lse(row)
        out.append([v - z if v != NEG_INF else NEG_INF for v in row])
    return out


def enumerate_log(
    logp: Sequence[Sequence[float]],
    toy: Toy,
    start: Iterable[int],
    end: Iterable[int],
    l_eff: int | None = None,
) -> LogOracle:
    """float64 log-space enumeration; exact to rounding at any dynamic range (two-band, 850 nats)."""
    start, end = list(start), list(end)
    arcs = toy.arcs_for_token()
    L = len(logp) if l_eff is None else l_eff
    V = toy.vocab_size
    items: list[tuple[tuple[int, ...], float]] = []
    for x in _strings(V, L):
        n = path_count(x, arcs, toy.n_states, start, end)
        if not n:
            continue
        lw = math.log(n)
        for i, v in enumerate(x):
            lw += logp[i][v]
        if lw == NEG_INF:
            continue
        items.append((x, lw))
    log_z = _lse([lw for _, lw in items])
    q = [[0.0] * V for _ in range(L)]
    if items:
        m = max(lw for _, lw in items)
        per: list[list[list[float]]] = [[[] for _ in range(V)] for _ in range(L)]
        for x, lw in items:
            w = math.exp(lw - m)
            for i, v in enumerate(x):
                per[i][v].append(w)
        tot = math.fsum(math.exp(lw - m) for _, lw in items)
        q = [[math.fsum(per[i][v]) / tot for v in range(V)] for i in range(L)]
    ent = [-math.fsum(qv * math.log(qv) for qv in row if qv > 0.0) for row in q]
    return LogOracle(log_z=log_z, q=q, entropy=ent, support={x for x, _ in items})


def distance_to_final(toy: Toy) -> list[float]:
    """``d(s)`` = min tokens from ``s`` to F by reverse BFS over edges with a non-empty label."""
    d = [math.inf] * toy.n_states
    rev: list[list[int]] = [[] for _ in range(toy.n_states)]
    for s, t, c in toy.edges:
        if toy.members(c):
            rev[t].append(s)
    frontier = sorted(toy.finals)
    for f in frontier:
        d[f] = 0
    depth = 0
    while frontier:
        depth += 1
        nxt = []
        for t in frontier:
            for s in rev[t]:
                if d[s] == math.inf:
                    d[s] = depth
                    nxt.append(s)
        frontier = nxt
    return d


def advance(toy: Toy, states: Iterable[int], ids: Sequence[int]) -> set[int]:
    """δ*(A, x): the set reachable by reading every id of ``x`` (no stop handling)."""
    arcs = toy.arcs_for_token()
    cur = set(states)
    for v in ids:
        cur = {d for s, d in arcs[v] if s in cur}
    return cur


def log_matmul_bruteforce(
    a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]
) -> list[list[float]]:
    """``C[i][k] = log Σ_j exp(A[i][j] + B[j][k])`` entry by entry, anchored per entry."""
    n, m, k = len(a), len(b), len(b[0])
    return [[_lse([a[i][j] + b[j][kk] for j in range(m)]) for kk in range(k)] for i in range(n)]


def log_fraction(x: Fraction) -> float:
    """``log x`` for a positive Fraction of any size (big-int ``math.log``)."""
    return math.log(x.numerator) - math.log(x.denominator)


# ---------------------------------------------------------------------------- T-0066: draw and MAP


def support_scores(
    logp: Sequence[Sequence[float]],
    toy: Toy,
    start: Iterable[int],
    end: Iterable[int],
    l_eff: int | None = None,
) -> dict[tuple[int, ...], tuple[int, float]]:
    """``x -> (N(x), Σ_i logp[i][x_i])`` for every ``x ∈ V^Leff`` with ``N(x) > 0`` and a finite score.

    The joint posterior is ``P(x) ∝ N(x)·exp(score)`` (SPEC §2.2, §2.6 eq. 8: path-multiplicity weighted);
    the constrained MAP is ``argmax score`` over the keys (SPEC §2.7: multiplicity cannot change a max).
    """
    start, end = list(start), list(end)
    arcs = toy.arcs_for_token()
    L = len(logp) if l_eff is None else l_eff
    out: dict[tuple[int, ...], tuple[int, float]] = {}
    for x in _strings(toy.vocab_size, L):
        n = path_count(x, arcs, toy.n_states, start, end)
        if not n:
            continue
        score = math.fsum(logp[i][v] for i, v in enumerate(x))
        if score == NEG_INF or math.isnan(score):
            continue
        out[x] = (n, score)
    return out


def string_posterior(
    logp: Sequence[Sequence[float]],
    toy: Toy,
    start: Iterable[int],
    end: Iterable[int],
    l_eff: int | None = None,
) -> dict[tuple[int, ...], float]:
    """Exact (float64) joint posterior over constrained prefixes of length ``Leff``."""
    sc = support_scores(logp, toy, start, end, l_eff)
    if not sc:
        return {}
    lw = {x: math.log(n) + s for x, (n, s) in sc.items()}
    m = max(lw.values())
    w = {x: math.exp(v - m) for x, v in lw.items()}
    tot = math.fsum(w.values())
    return {x: v / tot for x, v in w.items()}


def accepts(toy: Toy, start: Iterable[int], end: Iterable[int], ids: Sequence[int]) -> bool:
    """Host replay: some path from ``start`` reading ``ids`` ends in ``end``."""
    return bool(advance(toy, start, ids) & set(end))
