"""T-0081 criteria 2, 4 and 5: what the seed writes, when, and how the steps are counted.

Rulings SD4 (semantics) and SD9 (step accounting) in ``tests/cd/unit/_seed_support.py``. Expectations are hand-built:
which positions keep the draft (``round(noise * len)`` noised: 0.5 x 8 = 4, 0.999 x 8 = 7.992 -> 8, 0.9 x 8 = 7.2 -> 7,
0.2 x 8 = 1.6 -> 2, 0.1 x 4 = 0.4 -> 0; no .5 cases, so half-even and half-up agree), the step row, the stock reference
run on the same global seed (for "the uniform ids are not stock's" and "the second canvas is stock's re-init"), and the
fp64 entropy oracle of ``_perstep_support`` for the temperature on entry. The generator is reached only as the task
names it (``ConstrainedDiffusionSampler._generator(device)``) and only to set or read its state, never to replay a draw.

Adversarial cases (task file): prefill completing in a mixed batch (decode slots present); chunked prefill where only
the completing chunk seeds; the prefill-only path; two seeded requests completing in one call; LIFO slot reuse by an
unseeded request (after a completed seed, and after a seeded request released mid-chunk).

Regime: CPU fp32, vendored v0.29.0 stock step, V = 262 144, CL 8, ``max_denoising_steps`` 6 unless stated, ST 2,
t 0.4-0.8 unless stated, CD generator seed ``steps.GENERATOR_SEED``, global seeds as stated per row.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import pytest
import torch

from tests.cd import _tap_cd_world as tw
from tests.cd import _wire_support as ws
from tests.cd.fakes.diffusion_sampler import Cfg, FakeModelStateBase, World
from tests.cd.unit import _perstep_support as ps
from tests.cd.unit import _seed_support as sd
from tests.cd.unit._sampler_support import LIVE, Records
from tests.fakes.diffusion_sampler import validate_canvas_step

CL = 8
MAX = 6
MODES = ("end_projection", "perstep_mf", "perstep_mar", "perstep_mar_qargmax")
LOGITS = sd.fixed_logits([{t: 3.0 + 0.1 * k for k, t in enumerate(LIVE)}])


def cfg(**kw: Any) -> Cfg:
    base: dict[str, Any] = dict(CL=CL, max_denoising_steps=MAX, stability_threshold=1)
    base.update(kw)
    return Cfg(**base)


def xargs(tag: str, mode: str = "end_projection", grammar: Any = None) -> dict[str, Any]:
    g = grammar or ps.loop_grammar(LIVE, "t0081-apply-loop")
    return ws.raw_xargs(g.payload, mode=mode, self_conditioning="model", record_tag=tag)


def seeded(tag: str, ids: Any, mode: str = "end_projection", **kw: Any) -> dict[str, Any]:
    return sd.with_seed(xargs(tag, mode), ids, **kw)


def diff_positions(canvas: list[int], draft: Any) -> list[int]:
    return [i for i, t in enumerate(draft) if canvas[i] != t]


# --------------------------------------------------------------------------- criterion 2: the first canvas


@pytest.mark.parametrize("mode", MODES)
def test_noise_0_full_length_first_canvas_is_the_draft(mode: str) -> None:
    w = World(cfg(), cd=True, seed=5)
    w.add("s", xargs=seeded("t0081-a0", sd.DRAFT8, mode))
    s = w.prefill()
    assert sd.row(s, "canvas", "s") == list(sd.DRAFT8)
    assert sd.draft_row(s, "s", CL) == list(sd.DRAFT8), (
        "draft_tokens[slot, :CL] != the seeded canvas"
    )
    assert sd.row(s, "step", "s") == [0]
    assert sd.row(s, "is_encoder_phase", "s") == [0]
    assert sd.row(s, "accepted_canvas_history_len", "s") == [0]
    assert not s.row("self_conditioning_embeds", "s").any(), "SC embeddings must stay zero (stock)"


def test_noise_0_short_draft_tail_is_uniform_from_the_cd_generator_not_stock() -> None:
    ref = World(cfg(), cd=False, seed=5)
    ref.add("s")
    r = ref.prefill()
    w = World(cfg(), cd=True, seed=5)
    w.add("s", xargs=seeded("t0081-a1", sd.DRAFT5))
    s = w.prefill()
    can = sd.row(s, "canvas", "s")
    assert can[:5] == list(sd.DRAFT5)
    tail, stock_tail = can[5:], sd.row(r, "canvas", "s")[5:]
    assert all(sd.renoised(t) for t in tail), tail
    same = [i for i in range(3) if tail[i] == stock_tail[i]]
    assert same == [], (
        f"tail ids equal stock init_canvas's global-RNG ids at {same} (SD4: from the CD generator)"
    )
    assert sd.draft_row(s, "s", CL) == can


@pytest.mark.parametrize(
    ("noise", "n", "k"),
    [(0.5, 8, 4), (0.999, 8, 8), (0.9, 8, 7), (0.2, 8, 2), (0.1, 4, 0), (0.25, 8, 2)],
)
def test_exactly_round_noise_times_len_positions_are_noised(noise: float, n: int, k: int) -> None:
    draft = sd.DRAFT8[:n]
    ref = World(cfg(), cd=False, seed=1)
    ref.add("s")
    stock_can = sd.row(ref.prefill(), "canvas", "s")
    w = World(cfg(), cd=True, seed=1)
    w.add("s", xargs=seeded("t0081-a2", draft, noise=noise))
    s = w.prefill()
    can = sd.row(s, "canvas", "s")
    moved = diff_positions(can, draft)
    assert len(moved) == k, (
        f"noise {noise} x {n}: {len(moved)} positions noised, expected {k} ({can})"
    )
    assert all(sd.renoised(can[i]) for i in moved), [can[i] for i in moved]
    assert all(sd.renoised(t) for t in can[n:])
    same = [i for i in range(CL) if can[i] == stock_can[i]]
    assert same == [], (
        f"uniform ids equal stock init_canvas's at {same} (SD4: from the CD generator)"
    )
    assert sd.draft_row(s, "s", CL) == can


def test_noised_positions_are_uniform_without_replacement() -> None:
    """n = 160 seeded requests (noise 0.5, len 8) in one world, slot reused: every set has exactly 4 positions; each
    position is noised with frequency 1/2 (|z| < 4, Bonferroni over 8 positions at two-sided alpha ~ 5e-4); >= 30 of
    the C(8, 4) = 70 sets occur; the uniform ids spread over V."""
    w = World(cfg(max_num_reqs=2), cd=True, seed=2)
    counts: Counter[int] = Counter()
    sets: set[tuple[int, ...]] = set()
    ids: list[int] = []
    n = 160
    for r in range(n):
        name = f"s{r}"
        w.add(name, xargs=seeded(f"t0081-u{r}", sd.DRAFT8, noise=0.5))
        s = w.prefill()
        can = sd.row(s, "canvas", name)
        moved = diff_positions(can, sd.DRAFT8)
        assert len(moved) == 4, (r, moved)
        counts.update(moved)
        sets.add(tuple(moved))
        ids += [can[i] for i in moved]
        w.remove(name)
    for pos in range(8):
        z = (counts[pos] / n - 0.5) / math.sqrt(0.25 / n)
        assert abs(z) < 4.0, f"position {pos} noised {counts[pos]}/{n} times (z = {z:.2f})"
    assert len(sets) >= 30, f"only {len(sets)} distinct noised sets"
    assert len(set(ids)) >= 0.95 * len(ids) and max(ids) - min(ids) > 262_144 // 2


def test_the_seed_ignores_the_global_rng_and_follows_the_cd_generator() -> None:
    def run(global_seed: int, gen_seed: int | None) -> list[int]:
        w = World(cfg(), cd=True, seed=global_seed)
        w.add("s", xargs=seeded("t0081-g", sd.DRAFT5, noise=0.5))
        if gen_seed is not None:
            sd.cd_generator(w).manual_seed(gen_seed)
        return sd.row(w.prefill(), "canvas", "s")

    a, b = run(0, None), run(7, None)
    assert a == b, "the seeded canvas depends on the global RNG (SD4: CD generator only)"
    c, d, e = run(0, 11), run(9, 11), run(0, 12)
    assert c == d, "same CD generator state, different global seed: canvases differ"
    assert c != e, (
        "a different CD generator state left the noised set and the uniform ids unchanged"
    )


@pytest.mark.parametrize("k0", [0, 3, MAX - 1])
def test_step_row_is_the_start_step(k0: int) -> None:
    w = World(cfg(), cd=True, seed=1)
    w.add("s", xargs=seeded("t0081-k", sd.DRAFT8, start=k0))
    s = w.prefill()
    assert sd.row(s, "canvas", "s") == list(sd.DRAFT8), "premise: the request was seeded"
    assert sd.row(s, "step", "s") == [k0]
    assert sd.row(s, "accepted_canvas_history_len", "s") == [0]


# --------------------------------------------------------------------------- adversarial batch shapes


def test_prefill_completing_in_a_mixed_batch_is_seeded_and_the_decoders_are_stock() -> None:
    def run(cd_world: bool) -> list[Any]:
        w = World(cfg(), cd=cd_world, seed=4)
        w.add("u")
        w.add("c", xargs=xargs("t0081-mix-c") if cd_world else None)
        w.prefill()
        w.decode(LOGITS)
        w.add("s", xargs=seeded("t0081-mix-s", sd.DRAFT8, start=2) if cd_world else None)
        rows = [sd.Row("u"), sd.Row("s", prefill=True), sd.Row("c")]
        return [w, sd.call(w, rows, LOGITS)]

    _, ref = run(False)
    _w, s = run(True)
    assert s.kind == "decode"
    assert sd.row(s, "canvas", "s") == list(sd.DRAFT8)
    assert sd.draft_row(s, "s", CL) == list(sd.DRAFT8)
    assert sd.row(s, "step", "s") == [2]
    from tests.cd.fakes.diffusion_sampler import rows_differ

    assert rows_differ(s, ref, "u") == [], "the unconstrained decode slot differs from stock"
    assert torch.equal(s.rng_after, ref.rng_after), "global RNG differs from stock"


@pytest.mark.parametrize("path", ["prefill_only", "mixed"])
def test_chunked_prefill_only_the_completing_chunk_seeds(path: str) -> None:
    w = World(cfg(), cd=True, seed=6)
    w.add("u")
    w.prefill()
    w.add("s", xargs=seeded("t0081-chunk", sd.DRAFT8, start=3))
    before = w.ms.diffusion_states.canvas[w.reqs["s"].slot].clone().tolist()
    lead = [] if path == "prefill_only" else [sd.Row("u")]
    s1 = sd.call(w, [*lead, sd.Row("s", prefill=True, computed=0, scheduled=3)], LOGITS)
    assert sd.row(s1, "canvas", "s") == before, "a mid-chunk call touched the canvas"
    assert sd.row(s1, "step", "s") == [0] and sd.row(s1, "is_encoder_phase", "s") == [1]
    s2 = sd.call(w, [*lead, sd.Row("s", prefill=True, computed=3, scheduled=2)], LOGITS)
    assert sd.row(s2, "canvas", "s") == list(sd.DRAFT8)
    assert sd.draft_row(s2, "s", CL) == list(sd.DRAFT8)
    assert sd.row(s2, "step", "s") == [3] and sd.row(s2, "is_encoder_phase", "s") == [0]


def test_two_seeded_requests_completing_in_one_call_each_get_their_own_seed() -> None:
    def run(seed_them: bool) -> Any:
        w = World(cfg(), cd=True, seed=8)
        w.add("s1", xargs=seeded("t0081-two1", sd.DRAFT8) if seed_them else xargs("t0081-two1"))
        w.add("p", xargs=xargs("t0081-two-p"))
        w.add(
            "s2",
            xargs=seeded("t0081-two2", sd.DRAFT5, start=4) if seed_them else xargs("t0081-two2"),
        )
        return w.prefill()

    s, ref = run(True), run(False)
    assert sd.row(s, "canvas", "s1") == list(sd.DRAFT8) and sd.row(s, "step", "s1") == [0]
    assert sd.row(s, "canvas", "s2")[:5] == list(sd.DRAFT5) and sd.row(s, "step", "s2") == [4]
    assert sd.row(s, "canvas", "p") == sd.row(ref, "canvas", "p"), (
        "the unseeded request was touched"
    )
    for r in ("s1", "s2", "p"):
        assert sd.draft_row(s, r, CL) == sd.row(s, "canvas", r)


@pytest.mark.parametrize("released", ["after_seed", "mid_chunk"])
@pytest.mark.parametrize("next_kind", ["constrained", "unconstrained"])
def test_lifo_slot_reuse_the_next_occupant_is_not_seeded(released: str, next_kind: str) -> None:
    def run(seed_a: bool) -> tuple[World, Any]:
        w = World(cfg(), cd=True, seed=9)
        w.add("a", xargs=seeded("t0081-lifo-a", sd.DRAFT8) if seed_a else xargs("t0081-lifo-a"))
        slot_a = w.reqs["a"].slot
        if released == "after_seed":
            first = w.prefill()
            if seed_a:
                assert sd.row(first, "canvas", "a") == list(sd.DRAFT8), "premise: A was seeded"
        else:
            sd.call(w, [sd.Row("a", prefill=True, computed=0, scheduled=2)])
        w.remove("a")
        w.add("b", xargs=xargs("t0081-lifo-b") if next_kind == "constrained" else None)
        assert w.reqs["b"].slot == slot_a, "fixture: B must take A's slot (LIFO)"
        return w, w.prefill()

    w, s = run(True)
    _, ref = run(False)
    assert sd.row(s, "canvas", "b") == sd.row(ref, "canvas", "b"), "B inherited A's seed"
    assert sd.row(s, "step", "b") == [0]
    assert sd.row(s, "canvas", "b") != list(sd.DRAFT8)
    # liveness in the same engine: a seeded request admitted after B is still seeded
    w.add("c", xargs=seeded("t0081-lifo-c", sd.DRAFT5, start=1))
    after = w.prefill()
    assert sd.row(after, "canvas", "c")[:5] == list(sd.DRAFT5) and sd.row(after, "step", "c") == [1]


# --------------------------------------------------------------------------- criterion 4: first canvas only


@pytest.mark.parametrize("mode", ["end_projection", "perstep_mar"])
def test_the_second_canvas_starts_from_stock_reinit(mode: str) -> None:
    """Fixed confident logits (independent of the canvas), so the seeded run and the reference converge on the same
    calls. end_projection: the canvas after the commit call equals the stock run's (stock's ``random_tokens`` from the
    same global RNG). Both arms: step 0, history length 0, every position renoised (not the draft).

    The worlds share the process's single global torch RNG (``World.__init__`` seeds it once), so each run goes to
    its commit **before** the next world is built: interleaving their calls would hand each world the other's
    draws. A stock-vs-stock control proves the sequential reference is reproducible."""
    c = cfg(max_denoising_steps=16)

    def stock_until_commit() -> list[Any]:
        ref = World(c, cd=False, seed=12)
        ref.add("s")
        out = [ref.prefill()]
        for _ in range(12):
            out.append(ref.decode(LOGITS))
            if out[-1].committed("s") is not None:
                return out
        raise AssertionError("fixture: the stock run never committed")

    ref_snaps = stock_until_commit()
    again = stock_until_commit()
    assert [sd.row(s, "canvas", "s") for s in again] == [
        sd.row(s, "canvas", "s") for s in ref_snaps
    ], "fixture: two sequential stock runs on seed 12 differ"
    w = World(c, cd=True, seed=12)
    w.add("s", xargs=seeded("t0081-c4", sd.DRAFT8, mode, start=2))
    assert sd.row(w.prefill(), "canvas", "s") == list(sd.DRAFT8), "premise: the request was seeded"
    commit = None
    for k in range(1, 13):
        a = w.decode(LOGITS)
        if a.committed("s") is not None:
            commit = (a, ref_snaps[k] if k < len(ref_snaps) else None)
            break
    assert commit is not None, "the seeded request never committed"
    a, b = commit
    can = sd.row(a, "canvas", "s")
    assert sd.row(a, "step", "s") == [0] and sd.row(a, "accepted_canvas_history_len", "s") == [0]
    assert all(sd.renoised(t) for t in can), f"second canvas holds non-uniform ids: {can}"
    assert sd.draft_row(a, "s", CL) == can
    if mode == "end_projection":
        assert b is not None and b.committed("s") is not None, (
            "fixture: the stock run must commit on the same call"
        )
        assert can == sd.row(b, "canvas", "s"), "the second canvas is not stock's re-init"
    nxt = w.decode(LOGITS)
    assert sd.row(nxt, "step", "s") == [1], "the second canvas did not start at step 0"


# --------------------------------------------------------------------------- criterion 5: step accounting


@pytest.mark.parametrize("mode", ["end_projection", "perstep_mar", "perstep_mar_qargmax"])
@pytest.mark.parametrize("k0", [0, 4, 5])
def test_record_steps_is_k0_plus_executed_and_the_cap_binds_at_max(
    mode: str, k0: int, caplog: Any, capsys: Any
) -> None:
    """Never confident (threshold 0): convergence only by the cap, so exactly MAX - k0 denoise calls run and the
    record's ``steps`` is MAX (= k0 + executed)."""
    records = Records(caplog, capsys)
    tag = f"t0081-steps-{mode}-{k0}"[:64]
    w = World(cfg(confidence_threshold=0.0), cd=True, seed=3)
    w.add("s", xargs=seeded(tag, sd.DRAFT8, mode, start=k0))
    pre = w.prefill()
    commit_call = None
    for _ in range(MAX + 3):
        snap = w.decode(LOGITS)
        if snap.committed("s") is not None:
            commit_call = snap.call
            break
    assert commit_call is not None
    executed = commit_call - pre.call - 1
    assert executed == MAX - k0, f"{executed} denoise calls before the commit, expected {MAX - k0}"
    w.remove("s")
    recs = [r for r in records.tagged(tag) if r.get("canvas_idx") == 0]
    assert len(recs) == 1 and recs[0]["error"] is None, recs
    assert recs[0]["steps"] == MAX, (
        f"record steps {recs[0]['steps']}, expected k0 + executed = {MAX}"
    )


@pytest.mark.parametrize("mode", ["end_projection", "perstep_mar"])
def test_temperature_on_entry_is_stock_schedule_at_k0(mode: str) -> None:
    """t 0.2-2.0, max 10, bound 1.0, every position {a: 2.0, b: 0.0} (floor -100). The first denoise step accepts
    the positions ``stock_rule`` picks from the fp64 entropies at T(k0) = t_min + (t_max - t_min) * max(10 - k0, 1)/10:
    T(0) = 2.0 -> h = 0.582, 2 accepted; T(8) = 0.56 -> h = 0.125, 8 accepted. Accepted positions hold a or b;
    the others are renoised. ``perstep_mar`` uses the full-vocabulary language, where q = p."""
    a, b = LIVE[0], LIVE[1]
    logits = sd.fixed_logits([{a: 2.0, b: 0.0}], floor=-100.0)
    c = cfg(
        t_min=0.2, t_max=2.0, max_denoising_steps=10, entropy_bound=1.0, confidence_threshold=1e9
    )
    grammar = ps.full_vocab_grammar("t0081-temp-full") if mode != "end_projection" else None
    for k0, want in ((0, 2), (8, 8)):
        T = 0.2 + 1.8 * max(10 - k0, 1) / 10
        h = ps.entropy64(ps.p64(logits(0, "s", 1)[0], T))
        assert ps.margin([h] * CL, 1.0) > 0.05, (k0, h)
        assert sum(ps.stock_rule([h] * CL, 1.0)) == want, (k0, h)
        w = World(c, cd=True, seed=21)
        x = sd.with_seed(xargs(f"t0081-temp-{k0}", mode, grammar), sd.DRAFT8, start=k0)
        w.add("s", xargs=x)
        w.prefill()
        s = w.decode(logits)
        got = sum(1 for t in sd.row(s, "canvas", "s") if t in (a, b))
        assert got == want, (
            f"{mode} k0={k0}: {got} positions accepted, stock's schedule at T={T:.2f} gives {want}"
        )


def test_tap_frames_keep_the_r10_r14_shape_with_a_late_start() -> None:
    """Tap ∘ CD ∘ stock, k0 = 4 of 6, never confident: denoise frames carry steps 5 and 6, the commit frame step 6
    (in 1..max_steps), every frame validates."""
    make = tw.plugin_attr("model_state", "make_model_state_cls", "T-0011")
    make_cd = tw.plugin_attr("cd.model_state", "make_cd_model_state_cls", "T-0067")
    factory = tw.TapFactory()
    c = cfg(confidence_threshold=0.0)
    w = tw.World(c, make(make_cd(FakeModelStateBase), tap=factory), factory=factory, seed=0)
    w.add("s", xargs=seeded("t0081-tap", sd.DRAFT8, start=4))
    pre = w.prefill()
    assert sd.row(pre, "step", "s") == [4], "premise: the seed set the step"
    steps: list[tuple[str, int]] = []
    for _ in range(4):
        snap = w.decode(LOGITS)
        for f in snap.frames:  # type: ignore[attr-defined]
            validate_canvas_step(f)
            if f.get("request_id") == "s":
                steps.append((f["phase"], f["step"]))
                assert 1 <= f["step"] <= MAX, f
        if snap.committed("s") is not None:
            break
    assert steps == [("denoise", 5), ("denoise", 6), ("commit", 6)], steps
