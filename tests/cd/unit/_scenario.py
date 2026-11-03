"""Paired runs for T-0067 criteria 4, 6, 7 and 8: the CD world against the stock world on the same batch.

``run_pair`` drives two ``World`` s with identical calls and logits: ``a`` is CD-wrapped with the constrained
request's ``astra_cd_*`` keys, ``b`` is the stock sampler with those keys removed (criterion 4's reference).
Logits for canvas ``k`` of a request are fixed within the canvas (seeded by ``call // period``) so the stock
trajectory converges on stability every ``period`` calls; the tests read the actual convergence calls from
the snapshots rather than assuming them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from tests.cd.fakes.diffusion_sampler import Cfg, Snap, World, random_live_logits
from tests.cd.unit._sampler_support import LIVE

LogitsFn = Callable[[int, str, int], torch.Tensor]


def canvas_logits(seed_of: Mapping[str, int], period: int = 3) -> LogitsFn:
    def fn(call: int, req: str, n: int) -> torch.Tensor:
        return random_live_logits(1000 * seed_of.get(req, 7) + call // period, LIVE, n)

    return fn


@dataclasses.dataclass
class Spec:
    req_id: str
    xargs: Mapping[str, Any] | None = None
    params: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    params_b: Mapping[str, Any] | None = None  # override for the stock run (e.g. logprobs dropped)


def admit(w: World, specs: Sequence[Spec], *, cd_run: bool) -> None:
    for s in specs:
        params = dict(s.params if cd_run or s.params_b is None else s.params_b)
        w.add(s.req_id, xargs=s.xargs if cd_run else None, **params)


def run_pair(
    cfg: Cfg,
    specs: Sequence[Spec],
    logits_fn: LogitsFn,
    n_calls: int,
    *,
    seed: int = 0,
    stock_first: bool = True,
    end_on_eos: Sequence[str] = (),
    cd_ctx: Callable[[World], Callable[[], Any]] | None = None,
) -> tuple[list[Snap], list[Snap]]:
    """Returns (cd snaps, stock snaps) of the decode calls. The stock run goes first so a fixture
    precondition failure reads as such before the CD import can fail. Requests in ``end_on_eos`` are
    removed from the CD run right after a commit whose first id is EOS (the engine's stop check)."""

    def drive(cd: bool) -> list[Snap]:
        w = World(cfg, cd=cd, seed=seed)
        if cd and cd_ctx is not None:
            w.call_ctx = cd_ctx(w)
        admit(w, specs, cd_run=cd)
        w.prefill()
        out = []
        for _ in range(n_calls):
            snap = w.decode(logits_fn)
            out.append(snap)
            if cd:
                for r in end_on_eos:
                    ids = snap.committed(r)
                    if r in w.reqs and ids is not None and ids[0] == 1:
                        w.remove(r)
        return out

    if stock_first:
        b = drive(False)
        a = drive(True)
    else:
        a = drive(True)
        b = drive(False)
    return a, b


def commit_calls(snaps: Sequence[Snap], req: str) -> list[int]:
    return [i for i, s in enumerate(snaps) if s.committed(req) is not None]


def convergence_calls(snaps: Sequence[Snap], req: str) -> list[int]:
    """Calls after which ``req`` enters its commit step (is_encoder_phase set by that call)."""
    return [i - 1 for i, s in enumerate(snaps) if i > 0 and s.entered_commit.get(req)]
