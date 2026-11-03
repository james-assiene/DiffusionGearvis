"""Shared helpers for T-0077's precision-neutrality rows (N3/N4/N5 against the frozen pre-change path).

The frozen path is ``tests/cd/oracle/core_frozen_t0077/`` (byte-identical copies of ``cd/{semiring,tree,emission,
marginals,mapdecode,draw}.py`` at e73cfa7; ``MANIFEST.md``). ``frozen(name)`` imports one of them under the private
package ``tests.cd.oracle.core_frozen_t0077`` (their relative imports resolve there), side by side with the live
``astra_vllm_plugin.cd.*`` modules (``live(name)``, which honours ``ASTRA_CD_PLUGIN_SRC`` like every T-0065 row).

The pipeline here is the per-step composition ``cd/steps.py`` runs per ``L_eff`` candidate (T-0072 pass-through
keywords included): ``class_mass`` → ``class_log_weights_from`` → ``class_max_of`` → ``up_sweep(log_transfer(...))``
→ ``joint_draw`` → ``constrained_map`` → ``constrained_marginals``. Each path builds its own tables with its own
``build_tables``, so the frozen path never sees a live object.

API assumed (umbrella criterion 2, public signatures unchanged): every call below is the e73cfa7 signature; the live
path may add keyword-only parameters with defaults, which these rows never pass.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from tests.cd.oracle.enumerate import Toy
from tests.cd.unit._core_support import core, dfa_toy, random_toy

FROZEN_PKG = "tests.cd.oracle.core_frozen_t0077"
FROZEN_DIR = Path(__file__).resolve().parents[1] / "oracle" / "core_frozen_t0077"
FROZEN_COMMIT = "e73cfa7"
FROZEN_MODULES = ("semiring", "tree", "emission", "marginals", "mapdecode", "draw")
#: sha256 of each blob at e73cfa7 (MANIFEST.md); the frozen files must still hash to these.
FROZEN_SHA256 = {
    "semiring": "0b4ae05f7de105c3ea0642acd89876ad1bd14f725494342aa3a6d1d48577d00c",
    "tree": "00495db70edae3717f334db3297729c6228645c8768f6b647fea5065ccb44ae2",
    "emission": "1175409ba665fa415f31e5d547a5bbf29d44d1d0485f47c195315349f8716cb6",
    "marginals": "c6a48bd0530715be4a2284cc25d6635d543f305661466c46c4927abbe9b85566",
    "mapdecode": "09f6b7d204f698f8f4e9ba4c41f69421521c3b97412d50d9406f9a6c91af01f5",
    "draw": "74154c8ee28385ecfb0262b3b52638c8b9d6d63b61a060bd916d424791a8f7ba",
}
SENTINEL_MAX = -1e29


def frozen(name: str) -> ModuleType:
    return importlib.import_module(f"{FROZEN_PKG}.{name}")


def live(name: str) -> ModuleType:
    return core(name)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- instances


@dataclasses.dataclass(frozen=True)
class Inst:
    """One core instance: an automaton, its bucket, fp32 logits ``[L, V]``, start/end sets and ``L_eff``."""

    name: str
    toy: Toy
    bucket: int
    logits: torch.Tensor
    start: tuple[int, ...]
    end: tuple[int, ...]
    l_eff: int
    draw_seed: int


def _logits(seed: int, L: int, V: int, span: float) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(L, V, generator=g, dtype=torch.float64) * -span).to(torch.float32)


def _toy_with_last_state_reachable(toy: Toy, rng: random.Random) -> Toy:
    """Make state ``S-1`` matter: an edge into it and out of it with live labels, and it is final."""
    S, V = toy.n_states, toy.vocab_size
    if S < 2:
        return toy
    classes = list(toy.classes)
    edges = list(toy.edges)
    for src, dst in ((rng.randrange(S - 1), S - 1), (S - 1, rng.randrange(S))):
        classes.append((frozenset(rng.sample(range(V), max(1, V // 3))), False))
        edges.append((src, dst, len(classes) - 1))
    return Toy(S, V, tuple(classes), tuple(edges), frozenset(toy.finals | {S - 1}))


def single_state_toy(V: int) -> Toy:
    """S = 1: one final state looping on a positive class and a negated one (parallel edges)."""
    classes = ((frozenset(range(V // 2)), False), (frozenset(range(1, V, 3)), True))
    return Toy(1, V, classes, ((0, 0, 0), (0, 0, 1)), frozenset({0}))


def chain_toy(S: int, V: int) -> Toy:
    """``0 → 1 → … → S-1`` on class {0..V/2}, F = {S-1}: empty at any length < S-1 (the empty-language rows)."""
    classes = ((frozenset(range(V // 2)), False),)
    return Toy(S, V, classes, tuple((s, s + 1, 0) for s in range(S - 1)), frozenset({S - 1}))


def last_state_toy(S: int, V: int) -> tuple[Toy, tuple[int, ...]]:
    """The only accepting path uses state ``S-1``: ``0 -{1,2}-> S-1``, ``S-1 -{3,4}-> S-1``, F = {S-1}; every
    other state ``1..S-2`` is a distractor reachable from 0 on the high-logit ids ``5..V-1`` that never reaches
    F. Returns (toy, the tokens the distractor-free MAP must pick: 2 then 4 … under ``last_state_logits``)."""
    assert S >= 3
    classes: list[tuple[frozenset[int], bool]] = [
        (frozenset({1, 2}), False),
        (frozenset({3, 4}), False),
    ]
    edges = [(0, S - 1, 0), (S - 1, S - 1, 1)]
    rng = random.Random(S)
    for k in range(1, S - 1):
        classes.append((frozenset(rng.sample(range(5, V), 4)), False))
        edges.append((0, k, len(classes) - 1))
        classes.append((frozenset(rng.sample(range(5, V), 6)), False))
        edges.append((k, 1 + rng.randrange(S - 2), len(classes) - 1))
    return Toy(S, V, tuple(classes), tuple(edges), frozenset({S - 1})), (2, 4)


def last_state_logits(L: int, V: int, seed: int) -> torch.Tensor:
    """Distractor ids 5..V-1 in [-1, 0]; the language's ids far below: 1 at -9, 2 at -8, 3 at -7, 4 at -6."""
    x = _logits(seed, L, V, 1.0)
    x[:, 0] = -12.0
    x[:, 1], x[:, 2], x[:, 3], x[:, 4] = -9.0, -8.0, -7.0, -6.0
    return x


def build_inst(name: str) -> Inst:
    """The core instances of T-0077a's matrix (``INSTANCES``): S ∈ {1, 2, 31, 32 = bucket, 91/128, 205/256}."""
    spec = INSTANCES[name]
    S, B, L, V, kind, l_short = (
        spec["S"],
        spec["B"],
        spec["L"],
        spec["V"],
        spec["kind"],
        spec.get("l_short", 0),
    )
    seed = spec["seed"]
    rng = random.Random(seed)
    l_eff = L - l_short
    if kind == "single":
        toy = single_state_toy(V)
        start: tuple[int, ...] = (0,)
    elif kind == "chain_empty":
        toy = chain_toy(S, V)
        start = (0,)
    elif kind == "last_state":
        toy, _ = last_state_toy(S, V)
        x = last_state_logits(L, V, seed)
        return Inst(name, toy, B, x, (0,), (S - 1,), l_eff, seed + 1)
    elif kind == "dfa":
        toy = dfa_toy(rng, V=V, S=S)  # kept a DFA (the is_dfa fast path)
        start = tuple(sorted({0, *rng.sample(range(S), min(S, 2))}))
    else:
        toy = _toy_with_last_state_reachable(random_toy(rng, V=V, S=S, nfa=(kind == "nfa")), rng)
        start = tuple(sorted({0, *rng.sample(range(S), min(S, 2))}))
    end = tuple(sorted(toy.finals))
    x = _logits(seed, L, V, float(spec.get("span", 12)))
    return Inst(name, toy, B, x, start, end, l_eff, seed + 1)


#: name -> spec. ``span``: logits uniform in [-span, 0] (fp32). L is odd-carry-rich (11 → 6 → 3 → 2 → 1).
INSTANCES: dict[str, dict[str, Any]] = {
    "s1_b64": {"S": 1, "B": 64, "L": 11, "V": 40, "kind": "single", "seed": 101, "span": 12},
    "s2_b64": {"S": 2, "B": 64, "L": 11, "V": 40, "kind": "nfa", "seed": 102, "span": 12},
    "s31_b64_dfa": {"S": 31, "B": 64, "L": 11, "V": 48, "kind": "dfa", "seed": 108, "span": 30},
    "s31_b64_nfa_leff": {
        "S": 31,
        "B": 64,
        "L": 11,
        "V": 48,
        "kind": "nfa",
        "seed": 104,
        "span": 30,
        "l_short": 4,
    },
    "s32_b32": {"S": 32, "B": 32, "L": 11, "V": 48, "kind": "nfa", "seed": 105, "span": 30},
    "s91_b128_nfa": {"S": 91, "B": 128, "L": 11, "V": 48, "kind": "nfa", "seed": 106, "span": 30},
    "s91_b128_dfa_leff": {
        "S": 91,
        "B": 128,
        "L": 11,
        "V": 48,
        "kind": "dfa",
        "seed": 101,
        "span": 12,
        "l_short": 3,
    },
    "s205_b256": {"S": 205, "B": 256, "L": 9, "V": 48, "kind": "nfa", "seed": 108, "span": 30},
    # adversarial
    "last_state_s31_b64": {"S": 31, "B": 64, "L": 6, "V": 40, "kind": "last_state", "seed": 109},
    "last_state_s91_b128": {
        "S": 91,
        "B": 128,
        "L": 7,
        "V": 40,
        "kind": "last_state",
        "seed": 110,
        "l_short": 2,
    },
    "empty_s31_b64": {"S": 31, "B": 64, "L": 11, "V": 40, "kind": "chain_empty", "seed": 111},
}


# --------------------------------------------------------------------------- the per-step composition


def tables_with(em: ModuleType, inst: Inst) -> Any:
    toy = inst.toy
    return em.build_tables(
        n_states=toy.n_states,
        vocab_size=toy.vocab_size,
        class_members=[sorted(stored) for stored, _ in toy.classes],
        class_is_neg=[neg for _, neg in toy.classes],
        edge_src=[e[0] for e in toy.edges],
        edge_dst=[e[1] for e in toy.edges],
        edge_class=[e[2] for e in toy.edges],
        finals=sorted(toy.finals),
        bucket=inst.bucket,
        device="cpu",
    )


def bool_mask(ids: Sequence[int], size: int) -> torch.Tensor:
    m = torch.zeros(size, dtype=torch.bool)
    for s in ids:
        m[s] = True
    return m


@dataclasses.dataclass
class Run:
    tables: Any
    cmax: tuple[torch.Tensor, torch.Tensor]
    log_m: torch.Tensor
    tree: Any
    draw: Any
    gen_state: torch.Tensor
    map: Any
    marg: Any


def mods(which: str) -> dict[str, ModuleType]:
    get = frozen if which == "frozen" else live
    return {n: get(n) for n in ("emission", "tree", "draw", "mapdecode", "marginals")}


def run(which: str, inst: Inst) -> Run:
    """``cd/steps.py``'s per-candidate composition on the ``which`` ("live" | "frozen") modules."""
    m = mods(which)
    em = m["emission"]
    tables = tables_with(em, inst)
    B = tables.bucket
    x = inst.logits.clone()
    a = bool_mask(inst.start, B)
    b = bool_mask(inst.end, B)
    cm = em.class_mass(x, tables)
    cw = em.class_log_weights_from(cm)
    cmax = m["mapdecode"].class_max_of(x, tables)
    log_m = em.log_transfer(x, tables, inst.l_eff, class_mass=cm)
    tree = m["tree"].up_sweep(log_m)
    gen = torch.Generator().manual_seed(inst.draw_seed)
    d = m["draw"].joint_draw(x, tables, tree, a, b, inst.l_eff, generator=gen, class_log_weights=cw)
    gen_state = gen.get_state().clone()
    mp = m["mapdecode"].constrained_map(x, tables, a, b, inst.l_eff, class_max=cmax)
    mg = m["marginals"].constrained_marginals(x, tables, tree, a, b, inst.l_eff, class_mass=cm)
    return Run(tables, cmax, log_m, tree, d, gen_state, mp, mg)


# --------------------------------------------------------------------------- the frozen path in the stock world


#: (module under ``astra_vllm_plugin.cd``, attribute, frozen module, frozen attribute): every binding of a
#: T-0077a-owned function in the not-owned callers, as of e73cfa7 (``steps``, ``sampler``, ``requests``).
CALL_SITES: tuple[tuple[str, str, str, str], ...] = (
    ("steps", "joint_draw", "draw", "joint_draw"),
    ("steps", "class_log_weights_from", "emission", "class_log_weights_from"),
    ("steps", "class_mass", "emission", "class_mass"),
    ("steps", "log_transfer", "emission", "log_transfer"),
    ("steps", "class_max_of", "mapdecode", "class_max_of"),
    ("steps", "constrained_map", "mapdecode", "constrained_map"),
    ("steps", "constrained_marginals", "marginals", "constrained_marginals"),
    ("steps", "up_sweep", "tree", "up_sweep"),
    ("sampler", "class_max_of", "mapdecode", "class_max_of"),
    ("sampler", "constrained_map", "mapdecode", "constrained_map"),
    ("requests", "build_tables", "emission", "build_tables"),
)


@contextlib.contextmanager
def frozen_world() -> Iterator[None]:
    """Route every call the not-owned callers make into the T-0077a-owned modules to the frozen copies."""
    saved = []
    try:
        for mod, attr, fmod, fattr in CALL_SITES:
            target = live(mod)
            assert hasattr(target, attr), (
                f"call site moved: astra_vllm_plugin.cd.{mod}.{attr} is gone (update CALL_SITES)"
            )
            saved.append((target, attr, getattr(target, attr)))
            setattr(target, attr, getattr(frozen(fmod), fattr))
        yield
    finally:
        for target, attr, val in reversed(saved):
            setattr(target, attr, val)
