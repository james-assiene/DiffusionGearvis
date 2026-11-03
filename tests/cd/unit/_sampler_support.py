"""Shared helpers for the T-0067 rows (the ``ConstrainedDiffusionSampler``), and the API they fix.

API assumed by these rows (test-author, 2026-09-30; becomes the ruling when recorded in docs/tasks/T-0067.md)
  ``astra_vllm_plugin.cd.model_state.make_cd_model_state_cls(base) -> type``: a subclass of ``base``.
    ``custom_sampler(s)`` calls ``super()`` and returns ``(ConstrainedDiffusionSampler(inner), second)``;
    ``add_request(req_index, new_req_data)`` / ``remove_request(req_id)`` call ``super()`` and keep the CD
    registry (the reset happens in ``remove_request``).
  ``astra_vllm_plugin.cd.sampler.ConstrainedDiffusionSampler``: wraps ``inner``; its own state is private
    (``_``-prefixed). ``add_request(req_idx, prompt_len, sampling_params)`` reads
    ``sampling_params.extra_args`` with T-0064's ``parse_cd_xargs`` and ``sampling_params.max_tokens``,
    ``.logprobs``, ``.top_k``, ``.top_p``; ``__call__(logits, input_batch, draft_logits=None)``.
  ``astra_vllm_plugin.cd.record``: records are emitted as one line ``ASTRA_CD <json>`` through Python
    ``logging`` (any logger) **or** stdout/stderr; the rows accept either. ``error_count() -> int`` is the
    per-process counter of error records (criterion 7).
  Budget headroom (T-0065 R6): the stand-in has no ``max_model_len`` limit, so ``headroom`` never binds:
    ``R_after = max_tokens − emitted − L_eff``.
  Stop ids are ``(1, 50, 106)`` (EOS = 1), T-0063's augmentation.
  Empty δ* is injected by replacing ``delta_star`` (T-0065) in every loaded ``astra_vllm_plugin.cd.*``
    namespace; the row asserts the replacement was actually called, so an implementation that computes
    δ* some other way reads "cannot inject", not a pass.

Grammars are hand-built through T-0064's wire path (``tests/cd/_wire_support.py``): real payloads at
V = 262 144. Logits put every id at the softcap floor (−30) except a handful of live ids, so a brute-force
oracle over the live ids (plus EOS) is exact: grammar edges only use live ids, and dead ids are only ever
admissible inside ``ACC``, where a live id always beats them.
"""

from __future__ import annotations

import itertools
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pytest
import torch

from tests.cd import _wire_support as ws
from tests.cd.fakes.diffusion_sampler import EOS, STOP_IDS, cd_module

ACC = "ACC"
LIVE = tuple(range(1000, 1008))

#: G1 (DFA over 1000..1005): 0 -{1000,1001}-> 1 -1002-> 1 -1003-> 2 -1004-> 2 -1005-> 1; F = {2}.
G1 = (3, [(0, 1000, 1), (0, 1001, 1), (1, 1002, 1), (1, 1003, 2), (2, 1004, 2), (2, 1005, 1)], {2})
#: G2 (a different language on the same ids): 0 -1006-> 1 -1007-> 0; F = {1}.
G2 = (2, [(0, 1006, 1), (1, 1007, 0)], {1})
#: CHAIN6: 0 -1000-> 1 -1001-> ... -1005-> 6; F = {6}: d(start) = 7 with the stop.
CHAIN6 = (7, [(i, 1000 + i, i + 1) for i in range(6)], {6})


class Grammar:
    """A hand-written DFA, its wire payload, and an independent host replay with the stop augmentation."""

    def __init__(self, spec: tuple[int, list[tuple[int, int, int]], set[int]], name: str) -> None:
        self.n, self.edges, self.finals = spec
        self.name = name
        self.compiled = ws.hand_built(self.n, list(self.edges), set(self.finals), name=name)
        self.payload = ws.encode(self.compiled)
        self.bucket = int(self.compiled.n_states_bucket)
        self.delta: dict[tuple[Any, int], set[Any]] = {}
        for s, t, d in self.edges:
            self.delta.setdefault((s, t), set()).add(d)
        self.dist = self._distances()

    def step(self, states: Iterable[Any], tok: int) -> set[Any]:
        out: set[Any] = set()
        for s in states:
            if s == ACC:
                out.add(ACC)
                continue
            out |= self.delta.get((s, tok), set())
            if s in self.finals and tok in STOP_IDS:
                out.add(ACC)
        return out

    def run(self, states: Iterable[Any], ids: Sequence[int]) -> set[Any]:
        cur = set(states)
        for t in ids:
            cur = self.step(cur, t)
        return cur

    def _distances(self) -> dict[Any, float]:
        d: dict[Any, float] = {ACC: 0}
        for f in self.finals:
            d[f] = 1
        changed = True
        while changed:
            changed = False
            for s, _t, dst in self.edges:
                if dst in d and d.get(s, math.inf) > d[dst] + 1:
                    d[s] = d[dst] + 1
                    changed = True
        return d

    def ends(self, r_after: int) -> set[Any]:
        return {s for s, v in self.dist.items() if v <= r_after}

    def accepts(self, start: Iterable[Any], ids: Sequence[int], r_after: int) -> bool:
        return bool(self.run(start, ids) & self.ends(r_after))


def cd_xargs(g: Grammar, tag: str, **extra: Any) -> dict[str, Any]:
    return ws.raw_xargs(
        g.payload, mode="end_projection", self_conditioning="model", record_tag=tag, **extra
    )


def log_softmax64(rows: torch.Tensor) -> np.ndarray:
    x = rows.to(torch.float64).numpy()
    m = x.max(axis=1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))


def brute_map(
    g: Grammar, logits: torch.Tensor, start: Iterable[Any], l_eff: int, r_after: int
) -> tuple[list[int], float, float]:
    """Exact constrained MAP over the first ``l_eff`` positions by enumeration over ``LIVE ∪ {EOS}``.

    Returns (tokens, best score, gap to the runner-up); the rows assert the gap so the case is tie-free.
    """
    lp = log_softmax64(logits[:l_eff])
    alphabet = sorted(set(LIVE) | {EOS})
    ends = g.ends(r_after)
    best, second, arg = -math.inf, -math.inf, None
    for seq in itertools.product(alphabet, repeat=l_eff):
        if not (g.run(start, seq) & ends):
            continue
        sc = float(sum(lp[i, t] for i, t in enumerate(seq)))
        if sc > best:
            best, second, arg = sc, best, list(seq)
        elif sc > second:
            second = sc
    assert arg is not None, "oracle: the language is empty on this canvas"
    return arg, best, best - second


def argmax_ids(logits: torch.Tensor) -> list[int]:
    return [int(x) for x in logits.argmax(dim=-1)]


# --------------------------------------------------------------------------- records


RECORD_RE = re.compile(r"ASTRA_CD (\{.*\})\s*$")


class Records:
    """Every ``ASTRA_CD <json>`` line from logging and from stdout/stderr, in order of capture."""

    def __init__(
        self, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
    ) -> None:
        caplog.set_level(logging.DEBUG)
        self.caplog = caplog
        self.capsys = capsys
        self._stream: list[str] = []

    def all(self) -> list[dict[str, Any]]:
        cap = self.capsys.readouterr()
        self._stream += (cap.out + cap.err).splitlines()
        lines = [r.getMessage() for r in self.caplog.records] + self._stream
        out = []
        for line in lines:
            for part in line.splitlines():
                m = RECORD_RE.search(part)
                if m:
                    out.append(json.loads(m.group(1)))
        return out

    def tagged(self, tag: str | None) -> list[dict[str, Any]]:
        return [r for r in self.all() if r.get("record_tag") == tag]


def validate(record: Mapping[str, Any]) -> Any:
    return ws.contracts_cd().CdCanvasRecord.model_validate(dict(record))


def error_count() -> int:
    fn = getattr(cd_module("record"), "error_count", None)
    if fn is None:
        pytest.fail(
            "T-0067 not implemented: cd.record.error_count() (the per-process counter) is missing"
        )
    return int(fn())


# --------------------------------------------------------------------------- instruments


class SyncSpy(torch.overrides.TorchFunctionMode):
    """Records every host-synchronising call made while active (criterion 9; T-0065 criterion 8's list)."""

    FORBIDDEN = {
        torch.Tensor.item: "item",
        torch.Tensor.tolist: "tolist",
        torch.Tensor.numpy: "numpy",
        torch.Tensor.__bool__: "__bool__",
        torch.Tensor.__int__: "__int__",
        torch.Tensor.__float__: "__float__",
        torch.Tensor.__index__: "__index__",
        torch.Tensor.nonzero: "nonzero",
        torch.nonzero: "nonzero",
        torch.Tensor.any: None,  # allowed (stays on device); listed so the table is explicit
    }

    def __init__(self) -> None:
        super().__init__()
        self.hits: list[str] = []

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        name = self.FORBIDDEN.get(func)
        if name:
            self.hits.append(name)
        # ruling S8: a move counts only when it is device -> host and blocking; same-device `.to` is a no-op
        src = args[0] if args and isinstance(args[0], torch.Tensor) else None
        if src is not None and src.device.type != "cpu":
            if func is torch.Tensor.cpu:
                self.hits.append("cpu")
            elif func is torch.Tensor.to and not kwargs.get("non_blocking", False):
                dev = kwargs.get("device")
                for a in args[1:]:
                    if isinstance(a, (str, torch.device)):
                        dev = a
                    elif isinstance(a, torch.Tensor):
                        dev = a.device
                if dev is not None and torch.device(dev).type == "cpu":
                    self.hits.append("to(cpu, blocking)")
        if func is torch.Tensor.copy_ and len(args) > 1 and isinstance(args[1], torch.Tensor):
            dst, src2 = args[0], args[1]
            if (
                src2.device.type != "cpu"
                and dst.device.type == "cpu"
                and not kwargs.get("non_blocking", False)
            ):
                self.hits.append("copy_(D2H, blocking)")
        return func(*args, **kwargs)


def patch_everywhere(monkeypatch: pytest.MonkeyPatch, name: str, replacement: Any) -> int:
    """Replace ``name`` in every loaded ``astra_vllm_plugin.cd.*`` module namespace; returns how many."""
    import sys

    n = 0
    for modname, mod in list(sys.modules.items()):
        if modname.startswith("astra_vllm_plugin.cd") and mod is not None and hasattr(mod, name):
            monkeypatch.setattr(mod, name, replacement)
            n += 1
    return n
