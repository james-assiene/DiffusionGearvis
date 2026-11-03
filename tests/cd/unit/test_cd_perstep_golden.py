"""T-0076 criterion 9: the pre-existing arms are bit-identical to golden traces recorded before T-0076.

``tests/cd/fixtures/perstep_golden_8416e39.pt`` was written by ``_golden_support.write_golden`` at a tree whose code
equals 8416e39, **before any T-0076 implementation** (provenance in ``tests/cd/fixtures/MANIFEST.md``). Each row
re-runs one (arm, scenario) through the T-0067 CPU stand-in and compares, call by call: ``canvas``, the history
rows and length, ``argmax_canvas``, ``sc_embeds``, phase, ``confident``, ``step``, ``draft_tokens``, committed ids,
the device report bytes, the CD generator's state, the global RNG state, the sequence of torch functions the
sampler call issued (criterion 9's "no extra device op") and every ``ASTRA_CD`` record line.

Arms: ``end_projection``, ``perstep_mf``, ``perstep_mar``, ``perstep_mar`` + constrained SC. Scenarios: ``budget``
(CL 8, four canvases, the 4th with ``L_eff`` = 5 < CL from the budget) and ``pad`` (valid length 5 < CL 8 on every
call, seven canvases). Regime: CPU fp32, vendored v0.29.0 stock step, ``max_denoising_steps`` 6, ST 2,
T 0.4-0.8, bound 1.0, threshold 1e9, seeds 11 / 23.

Mutants this kills: any change of an existing arm's history row (e.g. argmax-of-q for ``perstep_mar``), an extra
``q.argmax`` computed for every Mar slot (op digest), an extra generator draw (generator state), a changed report
layout (report bytes), a ``marginal=`` flag flipped for Mf (ops and canvas).

**Re-record (T-0077a ruling L3, 2026-10-04).** T-0077a's live-state slicing changed only the op sequence: at
9c9c508 all 2 392 traced values and 44 record lines of the 8 keys were bit-identical to the 8416e39 traces and only
the op digests (184 of 184 calls) differed. The pin with the op digest is now ``perstep_golden_9c9c508.pt``
(``gs.FIXTURE``); the original ``perstep_golden_8416e39.pt`` is kept unchanged (``gs.FIXTURE_8416E39``) and
``test_existing_arm_values_unchanged_since_8416e39`` still compares every value and record line with it, so the
T-0076 guarantee (existing arms unchanged by later work) is anchored to the pre-T-0076 recording, not to a re-record.

**Re-record (T-0077b rulings M3/K8, 2026-10-05).** The T-0077b kernel dispatch check (``cd/kernels/backend.use_kernels``
reads ``q_raw.is_cuda``) adds exactly one ``getset_descriptor.__get__`` per ``constrained_marginals`` call and nothing
else: at 5321565 all 2 392 values and 44 record lines equal the 9c9c508 (and 8416e39) traces, 92 of 184 op digests moved
(the 92 Mar / Mar+SC calls, one ``constrained_marginals`` each), the 92 Mf and end-projection digests did not. The
op-digest pin is now ``perstep_golden_5321565.pt`` (``gs.FIXTURE``); ``perstep_golden_9c9c508.pt``
(``gs.FIXTURE_9C9C508``) is kept, and ``test_rerecord_5321565_moves_only_the_dispatch_read`` pins the delta between
the two recordings so the re-record cannot have absorbed any other change.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest
import torch

from tests.cd.fakes.diffusion_sampler import bits
from tests.cd.unit import _golden_support as gs

KEYS = [f"{arm}/{scen}" for arm in gs.GOLDEN_ARMS for scen in gs.SCENARIOS]


@pytest.fixture(scope="module")
def golden() -> dict[str, dict[str, Any]]:
    assert gs.FIXTURE.exists(), f"golden fixture missing: {gs.FIXTURE}"
    data = torch.load(gs.FIXTURE, weights_only=True)
    assert sorted(data) == sorted(KEYS), f"fixture keys {sorted(data)}"
    return data  # type: ignore[no-any-return]


def test_fixture_premises(golden: dict[str, dict[str, Any]]) -> None:
    """The fixture covers what criterion 9 asks for: ≥ 3 canvases per arm, an L_eff < CL case (budget and pad),
    per-step arms that actually consume the CD generator, and no error record."""
    for key, tr in golden.items():
        arm, scen = key.split("/")
        assert gs.canvases(tr) >= 3, f"{key}: fewer than 3 canvases"
        for line in tr["records"]:
            assert '"error":null' in line, f"{key}: error record in the golden trace: {line}"
        gens = [c["gen_state"] for c in tr["calls"]]
        if arm == "end_projection":
            assert all(g.numel() == 0 for g in gens), f"{key}: end_projection has no CD generator"
        else:
            assert all(g.numel() > 0 for g in gens), (
                f"{key}: per-step arm without a generator state"
            )
            changed = sum(not torch.equal(a, b) for a, b in itertools.pairwise(gens))
            assert changed >= 3, f"{key}: the CD generator barely advanced ({changed} changes)"
        lens = [int(c["committed"].numel()) for c in tr["calls"] if c["committed"].numel()]
        if scen == "budget":
            # max_tokens 29 and CL 8: the 4th canvas starts at emitted 24 -> L_eff = 5 < CL
            assert len(lens) >= 4 and sum(lens[:3]) == 24, f"{key}: budget scenario lens {lens}"
        else:
            assert set(lens) == {5}, f"{key}: pad scenario must commit valid length 5, got {lens}"
    # the arms differ from each other (the fixture is not four copies of one trajectory)
    for scen in gs.SCENARIOS:
        firsts = {
            arm: tuple(golden[f"{arm}/{scen}"]["calls"][0]["canvas"].tolist())
            for arm in gs.GOLDEN_ARMS
        }
        assert len(set(firsts.values())) >= 3, f"{scen}: arms indistinguishable {firsts}"


def _diff(i: int, want: dict[str, Any], got: dict[str, Any]) -> list[str]:
    bad = []
    for name, w in want.items():
        g = got.get(name)
        if name == "ops":
            if g != w:
                extra = {
                    k: (w["counts"].get(k, 0), g["counts"].get(k, 0))
                    for k in set(w["counts"]) | set(g["counts"])
                    if w["counts"].get(k, 0) != g["counts"].get(k, 0)
                }
                bad.append(
                    f"call {i}: ops differ (n {w['n']} -> {g['n']}; counts golden->now {extra})"
                )
            continue
        if not isinstance(g, torch.Tensor) or g.shape != w.shape or g.dtype != w.dtype:
            bad.append(f"call {i}: {name} shape/dtype {getattr(g, 'shape', None)} vs {w.shape}")
        elif not torch.equal(bits(g), bits(w)):
            bad.append(f"call {i}: {name} differs")
    return bad


@pytest.mark.parametrize("key", KEYS)
def test_existing_arm_bit_identical_to_golden(key: str, golden: dict[str, dict[str, Any]]) -> None:
    arm, scen = key.split("/")
    want = golden[key]
    got = gs.trace(arm, scen)
    assert len(got["calls"]) == len(want["calls"]), (
        f"{key}: {len(got['calls'])} calls now, {len(want['calls'])} in the golden trace"
    )
    bad: list[str] = []
    for i, (w, g) in enumerate(zip(want["calls"], got["calls"], strict=True)):
        bad += _diff(i, w, g)
    assert not bad, (
        f"{key}: not bit-identical to the golden trace {gs.FIXTURE.name}:\n" + "\n".join(bad[:20])
    )
    assert got["records"] == want["records"], f"{key}: ASTRA_CD record lines differ"


@pytest.fixture(scope="module")
def golden_8416e39() -> dict[str, dict[str, Any]]:
    assert gs.FIXTURE_8416E39.exists(), f"original golden fixture missing: {gs.FIXTURE_8416E39}"
    data = torch.load(gs.FIXTURE_8416E39, weights_only=True)
    assert sorted(data) == sorted(KEYS), f"fixture keys {sorted(data)}"
    return data  # type: ignore[no-any-return]


@pytest.mark.parametrize("key", KEYS)
def test_existing_arm_values_unchanged_since_8416e39(
    key: str, golden_8416e39: dict[str, dict[str, Any]]
) -> None:
    """Every traced value and record line still equals the pre-T-0076 recording (op digest excluded: it moved
    under T-0077a ruling L3). A value change in an existing arm reddens this row whatever the newer pin says."""
    arm, scen = key.split("/")
    want = golden_8416e39[key]
    got = gs.trace(arm, scen)
    assert len(got["calls"]) == len(want["calls"]), (
        f"{key}: {len(got['calls'])} calls now, {len(want['calls'])} at 8416e39"
    )
    bad: list[str] = []
    for i, (w, g) in enumerate(zip(want["calls"], got["calls"], strict=True)):
        bad += _diff(i, {k: v for k, v in w.items() if k != "ops"}, g)
    assert not bad, f"{key}: a value differs from the 8416e39 trace:\n" + "\n".join(bad[:20])
    assert got["records"] == want["records"], f"{key}: ASTRA_CD record lines differ from 8416e39"


#: T-0077b ruling M3: the only op name the kernel dispatch check may add (the ``is_cuda`` read in ``use_kernels``)
DISPATCH_READ = "getset_descriptor.__get__"
#: arms that call ``constrained_marginals`` once per sampler call (end projection and Mf never call it)
MAR_ARMS = ("perstep_mar", "perstep_mar_sc")


def test_rerecord_5321565_moves_only_the_dispatch_read(golden: dict[str, dict[str, Any]]) -> None:
    """Fixture against fixture, no sampler run: the 5321565 pin equals the 9c9c508 pin in every value and record
    line; its op digest differs only for the Mar arms, by exactly one ``getset_descriptor.__get__`` per call (each
    such call runs ``constrained_marginals`` once), and the end-projection and Mf digests are unchanged."""
    assert gs.FIXTURE.name == "perstep_golden_5321565.pt", gs.FIXTURE
    assert gs.FIXTURE_9C9C508.exists(), f"previous pin missing: {gs.FIXTURE_9C9C508}"
    prev = torch.load(gs.FIXTURE_9C9C508, weights_only=True)
    assert sorted(prev) == sorted(KEYS)
    moved = 0
    for key in KEYS:
        arm = key.split("/")[0]
        new, old = golden[key], prev[key]
        assert new["records"] == old["records"], f"{key}: record lines differ between the pins"
        assert len(new["calls"]) == len(old["calls"]), key
        for i, (cn, co) in enumerate(zip(new["calls"], old["calls"], strict=True)):
            assert set(cn) == set(co), f"{key} call {i}: fields {sorted(set(cn) ^ set(co))}"
            bad = _diff(i, {k: v for k, v in co.items() if k != "ops"}, cn)
            assert not bad, f"{key}: a value differs between the pins:\n" + "\n".join(bad)
            wc, nc = co["ops"]["counts"], cn["ops"]["counts"]
            delta = {k: nc.get(k, 0) - wc.get(k, 0) for k in set(wc) | set(nc)}
            delta = {k: v for k, v in delta.items() if v}
            if arm in MAR_ARMS:
                assert delta == {DISPATCH_READ: 1}, f"{key} call {i}: op delta {delta}"
                assert cn["ops"]["n"] == co["ops"]["n"] + 1, f"{key} call {i}"
                assert cn["ops"]["sha256"] != co["ops"]["sha256"], f"{key} call {i}"
                moved += 1
            else:
                assert cn["ops"] == co["ops"], f"{key} call {i}: digest moved for {arm} ({delta})"
    assert moved == 92, f"{moved} op digests moved, 92 expected (16 + 30 calls x 2 Mar arms)"
