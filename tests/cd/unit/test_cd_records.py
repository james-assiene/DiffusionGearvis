"""T-0067 criterion 8: one ``ASTRA_CD <json>`` record per committed canvas per constrained request.

Validated by T-0064's ``CdCanvasRecord`` (emitted here, schema unchanged). Field values are checked against the
stand-in's own observations and the independent host replay: ``committed_ids`` = what the sampler output
committed (valid length), ``canvas_idx`` counts from 0, ``steps`` = the stock step counter at convergence,
``bucket`` = the payload's bucket, ``stop_pos`` = first stop id or null, ``state_set_after``/``in_final``
agree with the host replay, and the end-projection shadow fields describe what stock would have committed.
A record may lag its commit by ≤ 2 sampler calls. Unconstrained requests emit none.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.cd.fakes.diffusion_sampler import STOP_IDS, Cfg, World
from tests.cd.unit._sampler_support import ACC, G1, G2, Grammar, Records, cd_xargs, validate
from tests.cd.unit._scenario import canvas_logits

BIG = 10**6


@pytest.fixture(scope="module")
def grammars() -> dict[str, Grammar]:
    return {"c1": Grammar(G1, "t0067-rec-g1"), "c2": Grammar(G2, "t0067-rec-g2")}


def test_records_one_per_commit_with_lag_at_most_two(
    grammars: dict[str, Grammar], caplog: Any, capsys: Any
) -> None:
    records = Records(caplog, capsys)
    cfg = Cfg(CL=4, stability_threshold=1)
    fn = canvas_logits({"c1": 71, "c2": 72, "u": 73}, period=3)
    # the stock run supplies the shadow argmax (what stock would have committed) per canvas
    stock = World(cfg, cd=False)
    for r in ("c1", "c2", "u"):
        stock.add(r)
    stock.prefill()
    shadow: dict[str, list[list[int]]] = {"c1": [], "c2": []}
    for _ in range(9):
        s = stock.decode(fn)
        for r in shadow:
            if s.committed(r) is not None:
                shadow[r].append(s.committed(r))

    w = World(cfg, cd=True)
    w.add("c1", xargs=cd_xargs(grammars["c1"], "rec-c1"))
    w.add("c2", xargs=cd_xargs(grammars["c2"], "rec-c2"))
    w.add("u")
    w.prefill()
    commits: dict[str, list[tuple[int, list[int], int]]] = {"c1": [], "c2": []}
    seen_by_call: list[int] = []
    for _ in range(11):
        pre = {r: int(w.ms.diffusion_states.step[w.reqs[r].slot]) for r in commits}
        s = w.decode(fn)
        for r in commits:
            if s.committed(r) is not None:
                commits[r].append((s.call, s.committed(r), pre[r]))
        seen_by_call.append(len(records.all()))
    recs = records.all()
    assert all(r.get("record_tag") in ("rec-c1", "rec-c2") for r in recs), (
        "unconstrained requests emit no record"
    )
    for r, g in (("c1", grammars["c1"]), ("c2", grammars["c2"])):
        mine = [validate(x) for x in records.tagged(f"rec-{r}")]
        n = len(commits[r])
        assert n >= 3, f"fixture: {r} committed {n} canvases"
        flushed = [c for c in commits[r] if c[0] + 2 <= w.calls - 1]
        assert len(mine) >= len(flushed) and len(mine) <= n, (r, len(mine), n)
        states: set[Any] = {0}
        for k, rec in enumerate(mine):
            call, ids, steps = commits[r][k]
            assert rec.canvas_idx == k and rec.arm == "end_projection" and rec.error is None
            assert list(rec.committed_ids) == ids
            assert rec.bucket == g.bucket
            stops = [i for i, t in enumerate(ids) if t in STOP_IDS]
            assert rec.stop_pos == (stops[0] if stops else None)
            assert rec.steps == steps, (
                f"{r} canvas {k}: steps {rec.steps}, stock step counter {steps}"
            )
            states = g.run(states, ids)
            assert len(rec.state_set_after) == len(states), (rec.state_set_after, states)
            assert rec.in_final == (ACC in states)
            assert rec.mean_accepted_per_step is None
            sh = shadow[r][k]
            assert list(rec.shadow_argmax_ids or []) == sh
            before = g.run({0}, []) if k == 0 else None  # canvas 0 starts at A_0
            if before is not None:
                assert rec.argmax_in_language == g.accepts(before, sh, BIG)
            assert rec.map_equals_argmax == (ids == sh)
            idx = call + 2 if call + 2 < len(seen_by_call) else None
            if idx is not None:
                n_by = sum(
                    1
                    for x in records.all()[: seen_by_call[idx]]
                    if x.get("record_tag") == f"rec-{r}"
                )
                assert n_by >= k + 1, f"{r} canvas {k}: record not out 2 calls after its commit"
