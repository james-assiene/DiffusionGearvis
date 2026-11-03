"""T-0067 criterion 9: with constrained slots in the batch the sampler thread makes no synchronising call.

200 decode calls (denoise, converging and commit steps) over {2 constrained, 1 unconstrained}, every sampler
call under ``SyncSpy`` (``.item``, ``.tolist``, ``.numpy``, ``bool()``, ``int()``, ``float()``, ``.nonzero``,
``.cpu``, a blocking ``.to('cpu')``). Records still reach the log (via buffers read ≥ 1 call later).
"""

from __future__ import annotations

from typing import Any

from tests.cd.fakes.diffusion_sampler import Cfg, World
from tests.cd.unit._sampler_support import G1, G2, Grammar, Records, SyncSpy, cd_xargs
from tests.cd.unit._scenario import canvas_logits


def test_two_hundred_steps_without_a_host_sync(caplog: Any, capsys: Any) -> None:
    records = Records(caplog, capsys)
    g1, g2 = Grammar(G1, "t0067-sync-g1"), Grammar(G2, "t0067-sync-g2")
    w = World(Cfg(CL=4, stability_threshold=1), cd=True)
    w.add("c1", xargs=cd_xargs(g1, "sync-c1"))
    w.add("c2", xargs=cd_xargs(g2, "sync-c2"))
    w.add("u")
    fn = canvas_logits({"c1": 81, "c2": 82, "u": 83}, period=3)
    spy = SyncSpy()
    w.call_ctx = lambda: spy
    w.prefill()
    commits = 0
    hits: list[tuple[int, list[str]]] = []
    for i in range(200):
        n0 = len(spy.hits)
        s = w.decode(fn)  # the spy wraps only the sampler call (World.call_ctx)
        if len(spy.hits) > n0:
            hits.append((i, spy.hits[n0:]))
        commits += sum(1 for r in ("c1", "c2") if s.committed(r) is not None)
    assert hits == [], f"host syncs on the sampler thread (call, ops): {hits[:5]}"
    assert commits >= 100, f"fixture: only {commits} constrained commits in 200 calls"
    n_rec = len([r for r in records.all() if r.get("record_tag") in ("sync-c1", "sync-c2")])
    assert n_rec >= commits - 4, (
        f"{n_rec} records for {commits} commits (lag ≤ 2 calls per request)"
    )
