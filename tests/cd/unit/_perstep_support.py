"""Shared helpers for the T-0068 rows (per-step arms ``perstep_mf``, ``perstep_mar``, ``perstep_mar`` + constrained SC).

API assumed by these rows (test-author, 2026-09-30; becomes ruling P1-P8 when recorded in docs/tasks/T-0068.md)
  P1. The arms are selected through T-0064's wire keys only: ``astra_cd_mode`` in {``perstep_mf``,
      ``perstep_mar``} and ``astra_cd_self_conditioning`` in {``model``, ``constrained``}. A per-step request
      is **not** refused (T-0067's ``unsupported_option`` for per-step modes and for ``entropy_bound`` /
      ``confidence_threshold`` overrides is lifted for the per-step arms; ``end_projection`` keeps refusing
      the overrides). Everything else goes through T-0067's public entry points (``make_cd_model_state_cls``,
      the wrapped sampler); the rows never call ``cd/steps.py`` directly.
  P2. ``bound`` / ``threshold`` = ``astra_cd_entropy_bound`` / ``astra_cd_confidence_threshold`` when the
      request sets them, else the inner stock sampler's ``entropy_bound`` / ``confidence_threshold`` (in
      production those are generation_config's 0.1 / 0.005). The stand-in's ``Cfg`` sets the inner values.
  P3. The per-step distributions are the ones stock samples from at that step: ``p_i = softmax(logits_i / T)``
      with stock's temperature schedule ``T = t_min + (t_max - t_min) * max(steps - step, 1) / steps`` at the
      slot's step on entry. H(p), H(q), the joint draw and ``q @ E`` are all taken at that T; the MAP is
      T-invariant. Most rows set ``t_min = t_max = 1`` so T = 1 exactly; one row pins P3 at T = 0.5.
  P4. Per-position entropies for the accept mask and ``confident``: ``h_i = H(q_i)`` (Mar) or ``H(p_i)`` (Mf)
      for ``i < L_eff``; ``h_i = H(p_i)`` for ``L_eff <= i < CL`` (stock). The rows use ``L_eff = CL`` except
      where stated.
  P5. The stand-in's ``accepted_canvas_history`` rows, ``argmax_canvas`` and the committed canvas carry the
      constrained MAP on ``[0, L_eff)``; convergence = stable MAP history over ST and ``confident``, or the
      step cap; at convergence ``canvas <- MAP`` (so ``draft_tokens[slot, :CL]`` = the MAP).
  P6. Randomness (the joint draw and the uniform renoise ids) comes from a CD-owned ``torch.Generator``:
      the global RNG state after every call equals the stock run's, and the renoise ids are not stock's
      ``randint`` ids. The rows never seed or read that generator.
  P7. ``CdCanvasRecord.mean_accepted_per_step`` (per-step arms) = the mean, over the canvas's denoise steps, of
      the number of accepted positions in ``[0, L_eff)``; ``steps`` = the number of denoise steps of that
      canvas (as T-0067 fills it). ``shadow_argmax_ids``/``argmax_in_language``/``map_equals_argmax`` are
      null for the per-step arms (schema).
  P8. Stop ids (1, 50, 106), EOS = 1, as T-0067 S1.

Oracles: hand-built product languages (position ``i`` admits exactly ``S_i``), for which ``q_i`` is ``p_i``
restricted to ``S_i`` and renormalised -- exact in fp64 without any T-0065 code; enumeration over the live ids for
the non-product XOR language; ``_sampler_support.brute_map`` for the MAP.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
import torch

from tests.cd import _wire_support as ws
from tests.cd.fakes.diffusion_sampler import STOP_IDS, V, cd_module
from tests.cd.unit._sampler_support import LIVE, Grammar

#: the three per-step arms as (mode, self_conditioning)
ARMS: dict[str, tuple[str, str]] = {
    "mf": ("perstep_mf", "model"),
    "mar": ("perstep_mar", "model"),
    "mar_sc": ("perstep_mar", "constrained"),
}
#: T-0076's arms (kept apart from ``ARMS`` so the T-0068 modules' parametrisations do not change)
QA_ARMS: dict[str, tuple[str, str]] = {
    "qa": ("perstep_mar_qargmax", "model"),
    "qa_sc": ("perstep_mar_qargmax", "constrained"),
}
#: every per-step arm, T-0068's and T-0076's
ALL_ARMS: dict[str, tuple[str, str]] = {**ARMS, **QA_ARMS}
FLOOR = -30.0
#: a bound that accepts every position (Σ sorted − cummax ≤ 1e6 always) and a threshold nothing reaches
ACCEPT_ALL = 1e6
NEVER_CONFIDENT = 1e-30
ALWAYS_CONFIDENT = 1e9
TINY_BOUND = 1e-9


def ps_xargs(g: Grammar, tag: str, arm: str, **extra: Any) -> dict[str, Any]:
    """``astra_cd_*`` keys for a per-step arm (``extra``: ``entropy_bound``, ``confidence_threshold``, ...)."""
    mode, sc = ALL_ARMS[arm]
    return ws.raw_xargs(g.payload, mode=mode, self_conditioning=sc, record_tag=tag, **extra)


def ep_xargs(g: Grammar, tag: str) -> dict[str, Any]:
    return ws.raw_xargs(g.payload, mode="end_projection", self_conditioning="model", record_tag=tag)


def steps_module() -> Any:
    """``cd.steps`` (new in T-0068): absent reads as "T-0068 not implemented"."""
    import importlib
    import sys

    from tests.cd.fakes.diffusion_sampler import PLUGIN_SRC

    if str(PLUGIN_SRC) not in sys.path:
        sys.path.insert(0, str(PLUGIN_SRC))
    try:
        return importlib.import_module("astra_vllm_plugin.cd.steps")
    except ModuleNotFoundError as exc:
        if (exc.name or "").startswith("astra_vllm_plugin.cd"):
            pytest.fail(
                f"T-0068 not implemented: `astra_vllm_plugin.cd.steps` does not exist ({exc})"
            )
        raise


# --------------------------------------------------------------------------- languages


def product_grammar(
    sets: Sequence[Sequence[int]], name: str, *, loop_final: Sequence[int] = ()
) -> Grammar:
    """Position ``i`` admits exactly ``sets[i]``: states ``0..L``, edges ``i -t-> i+1``, F = {L}.

    ``loop_final``: ids with a self-loop on the final state (lets later canvases continue)."""
    L = len(sets)
    edges = [(i, t, i + 1) for i, s in enumerate(sets) for t in s]
    edges += [(L, t, L) for t in loop_final]
    return Grammar((L + 1, edges, {L}), name)


def loop_grammar(ids: Sequence[int], name: str) -> Grammar:
    """One final state with a self-loop on ``ids``: every string over ``ids`` (and any stop) is viable."""
    return Grammar((1, [(0, t, 0) for t in ids], {0}), name)


def full_vocab_grammar(name: str) -> Grammar:
    """Every per-position support is the full vocabulary: a final state looping on every non-stop id (stops
    lead to ACC, which reads anything). Constrained = unconstrained, so q = p exactly."""
    ids = [t for t in range(V) if t not in STOP_IDS]
    return Grammar((1, [(0, t, 0) for t in ids], {0}), name)


def xor_grammar(a: int, b: int, name: str) -> Grammar:
    """Length-2 language {ab, ba} (then stops): a product-of-marginals sampler produces aa/bb."""
    return Grammar((4, [(0, a, 1), (0, b, 2), (1, b, 3), (2, a, 3)], {3}), name)


# --------------------------------------------------------------------------- logits and fp64 oracles


def logits_rows(rows: Sequence[dict[int, float]], *, floor: float = FLOOR) -> torch.Tensor:
    """``[len(rows), V]`` at ``floor`` except ``rows[i][t]`` at position ``i``."""
    x = torch.full((len(rows), V), floor, dtype=torch.float32)
    for i, r in enumerate(rows):
        for t, v in r.items():
            x[i, t] = v
    return x


def flat(ids: Sequence[int], value: float = 0.0) -> dict[int, float]:
    return {t: value for t in ids}


def p64(row: torch.Tensor, T: float = 1.0) -> np.ndarray:
    """fp64 softmax of one ``[V]`` logits row at temperature ``T`` (float32 row / T as stock divides)."""
    x = (row.to(torch.float32) / torch.tensor(T, dtype=torch.float32)).to(torch.float64).numpy()
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()


def entropy64(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def q_product64(row: torch.Tensor, support: Sequence[int], T: float = 1.0) -> np.ndarray:
    """The exact constrained marginal at a product-language position: ``p`` restricted to ``support``."""
    p = p64(row, T)
    q = np.zeros_like(p)
    idx = np.asarray(sorted(support))
    q[idx] = p[idx]
    return q / q.sum()


def stock_rule(h: Sequence[float], bound: float) -> list[bool]:
    """The EntropyBound rule written from its definition: position ``i`` is accepted iff the entropies ranked
    strictly below it sum to ≤ ``bound`` (Σ sorted − cummax). Used only to check fixture premises: every row
    also states its expected mask as a literal and asserts a margin so ties and rounding cannot decide it."""
    order = sorted(range(len(h)), key=lambda i: h[i])
    out = [False] * len(h)
    acc = 0.0
    for i in order:
        out[i] = acc <= bound
        acc += h[i]
    return out


def margin(h: Sequence[float], bound: float) -> float:
    """Smallest |partial sum − bound| over the ranking (a fixture must not sit on the bound)."""
    s = sorted(h)
    acc, best = 0.0, math.inf
    for v in s:
        best = min(best, abs(acc - bound))
        acc += v
    return best


def in_set(t: int, ids: Sequence[int]) -> bool:
    return int(t) in set(int(x) for x in ids)


LIVE_OR_STOP = tuple(LIVE) + tuple(STOP_IDS)


def renoised(t: int) -> bool:
    """A position holds a uniform renoise id (not a live or stop id): P(false negative) = 11 / 262 144."""
    return int(t) not in LIVE_OR_STOP


def record_count_field(rec: dict[str, Any]) -> float | None:
    v = rec.get("mean_accepted_per_step")
    return None if v is None else float(v)


def cd_record_module() -> Any:
    return cd_module("record")


# --------------------------------------------------------------------------- T-0076: q by enumeration


def brute_q(
    g: Grammar,
    logits: torch.Tensor,
    start: Sequence[Any],
    l_eff: int,
    r_after: int,
    *,
    T: float = 1.0,
) -> list[dict[int, float]]:
    """fp64 constrained marginals ``q_i(v)`` over the first ``l_eff`` positions by enumeration over
    ``LIVE ∪ STOP_IDS`` (DFA: multiplicity 1): ``q_i(v) ∝ Σ_{x ∈ C, x_i = v} Π_j p_j(x_j)``. Exact for the
    hand-built grammars here, whose edges use only live ids and whose dead ids sit at the floor (they can only
    matter inside ACC, which these rows never reach within ``l_eff``)."""
    import itertools

    lp = [np.log(p64(logits[i], T)) for i in range(l_eff)]
    alphabet = sorted(set(LIVE) | set(STOP_IDS))
    ends = g.ends(r_after)
    mass: list[dict[int, float]] = [dict() for _ in range(l_eff)]
    for seq in itertools.product(alphabet, repeat=l_eff):
        if not (g.run(start, seq) & ends):
            continue
        w = math.exp(float(sum(lp[i][t] for i, t in enumerate(seq))))
        for i, t in enumerate(seq):
            mass[i][t] = mass[i].get(t, 0.0) + w
    out = []
    for m in mass:
        z = sum(m.values())
        assert z > 0, "oracle: the language is empty on this canvas"
        out.append({t: v / z for t, v in m.items()})
    return out


def argmax_q(q: Sequence[dict[int, float]], *, min_gap: float = 1e-6) -> list[int]:
    """Per-position argmax of ``brute_q``'s marginals, first id on exact ties; asserts every non-tied winner
    leads its runner-up by ``min_gap`` (so rounding cannot decide the row)."""
    row = []
    for i, qi in enumerate(q):
        ranked = sorted(qi.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(ranked) > 1 and ranked[0][1] != ranked[1][1]:
            assert ranked[0][1] - ranked[1][1] > min_gap, (
                f"oracle: near-tie at position {i}: {ranked[:2]}"
            )
        row.append(int(ranked[0][0]))
    return row
